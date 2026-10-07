"""One reduction driver for lifted functions and the built-in reductions.

A reduction splits a batch into lanes: the reduced axes, walked in row-major
order, for each position of the kept axes. A kernel folds a run of one lane
into a partial result, combines partial results in order and finishes the
last one; the driver handles axes, empty lanes, parallel lanes and, for
associative kernels, chunks of one long lane folded on separate threads and
combined in order.
"""

from std.memory import ArcPointer
from ._layout import _FlatLayout, _axis
from ._parallel import _map_tensor
from ._tensor import _Tensor
from ._values import _Values, _borrow_values
from .value import Batch

# An associative reduction with fewer outputs than this splits each one into
# chunks of at least _CHUNK elements, about _TASKS chunks in all.
comptime _FEW_LANES = 64
comptime _CHUNK = 64
comptime _TASKS = 256


struct _ReduceAxes(Copyable, Movable):
    """The axes of a reduction: `axis=1` or `axis=[0, 2]`."""

    var axes: List[Int]

    @implicit
    def __init__(out self, axis: Int):
        self.axes = [axis]

    @implicit
    def __init__(out self, axes: List[Int]):
        self.axes = axes.copy()

    def __init__(out self, var *axes: Int, __list_literal__: NoneType):
        self.axes = List[Int]()
        for axis in axes:
            self.axes.append(axis)


def _layout_over(layout: _FlatLayout, axes: List[Int], offset: Int) raises -> _FlatLayout:
    """The source layout restricted to `axes`, walked in row-major order from `offset`."""
    var result = _FlatLayout(List[Int]())
    result.offset = offset
    for axis in axes:
        result.shape.append(layout.shape[axis])
        result.strides.append(layout.strides[axis])
    result.size = 1
    for extent in result.shape:
        result.size *= extent
    result.contiguous = result._is_contiguous()
    return result^


def _start(layout: _FlatLayout) -> Int:
    return 0 if layout.scalar else layout.offset


def _unravel(index: Int, shape: List[Int]) -> List[Int]:
    var coordinates = List[Int](length=len(shape), fill=0)
    var rest = index
    for j in range(len(shape)):
        var axis = len(shape) - 1 - j
        if shape[axis]:
            coordinates[axis] = rest % shape[axis]
            rest //= shape[axis]
    return coordinates^


def _failure(operation: StaticString, coordinates: List[Int], error: Error) -> Error:
    var text = String(operation, " failed at index [")
    for i in range(len(coordinates)):
        if i:
            text += ", "
        text += String(coordinates[i])
    return Error(String(text, "]: ", error))


struct _Lanes[T: ImplicitlyCopyable & Deinitable](Copyable, Movable):
    """A batch read as lanes: element k of lane j, for a set of reduced axes."""

    var values: ArcPointer[_Values[Self.T]]
    # Lane starts (kept axes, from the source offset) and in-lane offsets.
    var lanes: _FlatLayout
    var lane: _FlatLayout
    # In-lane offsets form one arithmetic run with this step, when `linear`.
    var step: Int
    var linear: Bool
    var kept: List[Int]
    var reduced: List[Int]

    def __init__(out self, values: Batch[Self.T], var kept: List[Int], var reduced: List[Int]) raises:
        ref layout = values._layout[]
        self.values = values._owner
        self.lanes = _layout_over(layout, kept, _start(layout))
        self.lane = _layout_over(layout, reduced, 0)
        var step = self.lane.progression()
        self.linear = Bool(step)
        self.step = step.value() if step else 0
        self.kept = kept^
        self.reduced = reduced^

    @staticmethod
    def _virtual(length: Int) raises -> Self:
        """One lane of `length` positions with no stored values, for kernels
        that read their own sources by position (dot products)."""
        var empty = Batch[Self.T](List[Self.T]())
        var lanes = Self(empty, List[Int](), [0])
        lanes.lane = _FlatLayout([length])
        lanes.linear = True
        lanes.step = 1
        return lanes^

    def count(self) -> Int:
        return self.lanes.size

    def length(self) -> Int:
        return self.lane.size

    @always_inline
    def base(self, j: Int) -> Int:
        return self.lanes.position(j)

    @always_inline
    def at(self, base: Int, k: Int) -> ref[self] Self.T:
        # Internal traversal only: 0 <= k < length(), so the position is in range.
        var position = base + (k * self.step if self.linear else self.lane.position(k))
        return _borrow_values[origin_of(self)](self.values[].list).unsafe_get(position)

    @always_inline
    def run(self, base: Int) -> Pointer[Self.T, origin_of(self)]:
        """Element 0 of a linear lane; element k is `step * k` further on."""
        return _borrow_values[origin_of(self)](self.values[].list).unsafe_ptr().unsafe_offset(base)

    def coordinates(self, j: Int, k: Int) -> List[Int]:
        """Source coordinates of element k of lane j."""
        var coordinates = List[Int](length=len(self.kept) + len(self.reduced), fill=0)
        var outer = _unravel(j, self.lanes.shape)
        var local = _unravel(k, self.lane.shape)
        for i in range(len(self.kept)):
            coordinates[self.kept[i]] = outer[i]
        for i in range(len(self.reduced)):
            coordinates[self.reduced[i]] = local[i]
        return coordinates^


trait _Fold(Copyable, Movable, Deinitable):
    """A reduction: fold runs of a lane, combine partial results in order, finish."""

    comptime Element: ImplicitlyCopyable & Deinitable
    comptime Partial: ImplicitlyCopyable & Deinitable
    comptime Result: ImplicitlyCopyable & Deinitable

    def fold(
        self, lanes: _Lanes[Self.Element], j: Int, start: Int, end: Int, opening: Bool,
    ) raises -> Self.Partial:
        """Fold elements [start, end) of lane j in order. `opening` marks the
        run that starts the lane; an empty lane is one empty opening run."""
        ...

    def combine(
        self, left: Self.Partial, right: Self.Partial, lanes: _Lanes[Self.Element], j: Int, k: Int,
    ) raises -> Self.Partial:
        """Combine consecutive partial results; `right` starts at element k."""
        ...

    def finish(self, var partial: Self.Partial) raises -> Self.Result:
        ...


@fieldwise_init
struct _FoldJob[K: _Fold](Copyable, Movable):
    var kernel: Self.K
    var lanes: _Lanes[Self.K.Element]
    var chunk: Int
    var chunks: Int


def _fold_lane[K: _Fold](job: _FoldJob[K], j: Int) raises -> K.Result:
    return job.kernel.finish(job.kernel.fold(job.lanes, j, 0, job.lanes.length(), True))


def _fold_chunk[K: _Fold](job: _FoldJob[K], task: Int) raises -> K.Partial:
    var j = task // job.chunks
    var c = task % job.chunks
    var start = c * job.chunk
    return job.kernel.fold(job.lanes, j, start, min(start + job.chunk, job.lanes.length()), c == 0)


def _reduce_lanes[K: _Fold](
    kernel: K, values: Batch[K.Element], axes: List[Int], every: Bool, keepdims: Bool,
    associative: Bool,
) raises -> Batch[K.Result]:
    """Reduce `axes` (or every axis) of `values` with `kernel`.

    Each result folds its elements in row-major order. Results run in
    parallel; with `associative`, a few long lanes also split into chunks
    combined in order.
    """
    ref layout = values._layout[]
    var rank = len(layout.shape)
    var reduced = List[Bool](length=rank, fill=every)
    if not every:
        for axis in axes:
            var a = _axis(axis, rank)
            if reduced[a]:
                raise Error("Cannot reduce an axis twice; list each axis once.")
            reduced[a] = True
    var kept = List[Int]()
    var summed = List[Int]()
    var shape = List[Int]()
    for i in range(rank):
        if reduced[i]:
            summed.append(i)
            if keepdims:
                shape.append(1)
        else:
            kept.append(i)
            shape.append(layout.shape[i])
    var lanes = _Lanes[K.Element](values, kept^, summed^)
    return Batch[K.Result](_tensor=_run_lanes(kernel, lanes^, associative)).reshape(shape)


def _reduce_positions[K: _Fold](kernel: K, length: Int, associative: Bool) raises -> K.Result:
    """Reduce one virtual lane of `length` positions that the kernel reads itself."""
    return _run_lanes(kernel, _Lanes[K.Element]._virtual(length), associative)._read(0)


def _run_lanes[K: _Fold](kernel: K, var lanes: _Lanes[K.Element], associative: Bool) raises -> _Tensor[K.Result]:
    """Every lane's result, in order."""
    var count = lanes.count()
    var length = lanes.length()
    var chunks = 1
    if associative and count < _FEW_LANES and length >= 2 * _CHUNK:
        chunks = min((length + _CHUNK - 1) // _CHUNK, (_TASKS + count - 1) // count)
    var chunk = max(1, (length + chunks - 1) // chunks)
    chunks = max(1, (length + chunk - 1) // chunk)
    var job = _FoldJob[K](kernel.copy(), lanes^, chunk, chunks)
    if chunks == 1:
        return _map_tensor[_fold_lane[K]](job, count)
    var parts = _map_tensor[_fold_chunk[K]](job, count * chunks)
    var results = List[K.Result](capacity=count)
    for j in range(count):
        var partial = parts._read(j * chunks)
        for c in range(1, chunks):
            partial = kernel.combine(partial, parts._read(j * chunks + c), job.lanes, j, c * chunk)
        results.append(kernel.finish(partial^))
    return _Tensor[K.Result](results^, [count])
