"""Checked layouts: _Layout is a 1-D strided run; _FlatLayout carries any rank.

Rank is runtime data, so layout code compiles once rather than once per rank.
"""

from std.builtin.builtin_slice import StridedSlice
from std.collections import Array
from std.memory import ArcPointer
from std.sys import size_of
from ._slices import _Selection


def _axis(axis: Int, rank: Int) raises -> Int:
    if axis < -rank or axis >= rank:
        raise Error("Batch axis is out of range; choose an axis within the batch rank.")
    return axis + rank if axis < 0 else axis


def _index(index: Int, length: Int) raises -> Int:
    if index < -length or index >= length:
        raise Error("Batch index is out of range; choose an index within this axis.")
    return index + length if index < 0 else index


comptime _Ints = Span[Int, ImmutAnyOrigin]
comptime _MutInts = Span[Int, MutAnyOrigin]


# Rank-erased views over fixed or growable integer lists for the helpers below.
@always_inline
def _ints[n: Int](ref values: Array[Int, n]) -> _Ints:
    return _Ints(unsafe_ptr=values.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=n)


@always_inline
def _mut_ints[n: Int](mut values: Array[Int, n]) -> _MutInts:
    return _MutInts(unsafe_ptr=values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), length=n)


@always_inline
def _list_ints(ref values: List[Int]) -> _Ints:
    return _Ints(unsafe_ptr=values.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=len(values))


@always_inline
def _list_mut_ints(mut values: List[Int]) -> _MutInts:
    return _MutInts(unsafe_ptr=values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), length=len(values))


def _row_major(
    shape: _Ints, strides: _MutInts, *,
    negative: StaticString = "Batch dimensions must be nonnegative; use zero for an empty axis.",
    overflow: StaticString = "Batch shape exceeds the addressable range; use smaller dimensions.",
) raises -> Int:
    """Validate a shape and return its size; fill strides when a span is supplied."""
    var empty = False
    for i in range(len(shape)):
        if shape[i] < 0:
            raise Error(String(negative))
        empty |= shape[i] == 0
    # Empty layouts have no addressable elements, even with huge other axes.
    if empty:
        return 0
    var size = 1
    for j in range(len(shape)):
        var i = len(shape) - j - 1
        if shape[i] > Int.MAX // size:
            raise Error(String(overflow))
        if len(strides):
            strides[i] = size
        size *= shape[i]
    return size


def _is_contiguous(shape: _Ints, strides: _Ints, size: Int) -> Bool:
    if size == 0:
        return True
    var stride = 1
    for j in range(len(shape)):
        var i = len(shape) - j - 1
        if shape[i] > 1 and strides[i] != stride:
            return False
        stride *= shape[i]
    return True


def _linear_index(shape: _Ints, indices: _Ints, size: Int) raises -> Int:
    if not size:
        raise Error("Cannot index an empty batch; select coordinates in a nonempty batch.")
    var result = 0
    for axis in range(len(shape)):
        result = result * shape[axis] + _index(indices[axis], shape[axis])
    return result




def _transpose_into(
    shape: _Ints, strides: _Ints, axes: _Ints,
    result_shape: _MutInts, result_strides: _MutInts, *,
    repeated: StaticString = "Cannot repeat a transpose axis; supply a permutation of all axes.",
) raises:
    for i in range(len(axes)):
        var a = _axis(axes[i], len(axes))
        for k in range(i):
            if _axis(axes[k], len(axes)) == a:
                raise Error(String(repeated))
        result_shape[i] = shape[a]
        result_strides[i] = strides[a]


@fieldwise_init
struct _AxisSelection(Movable):
    var axis: Int
    var index: Int
    var shape: List[Int]
    var size: Int

    def take_shape(deinit self) -> List[Int]:
        return self.shape^


def _select_axis(shape: _Ints, axis: Int, index: Int) raises -> _AxisSelection:
    """Drop one axis from a validated shape; an empty axis cannot be indexed."""
    var selected = _axis(axis, len(shape))
    var position = _index(index, shape[selected])
    var result = List[Int](capacity=len(shape) - 1)
    var empty = False
    for i in range(len(shape)):
        if i != selected:
            result.append(shape[i])
            empty |= shape[i] == 0
    var size = 0 if empty else 1
    if not empty:
        for dim in result:
            size *= dim
    return _AxisSelection(selected, position, result^, size)


def _broadcast_strides(shape: _Ints, strides: _Ints, target: _Ints, result: _MutInts) raises:
    """Fill zeroed result strides for broadcasting shape onto target."""
    for i in range(len(shape)):
        var dest = len(target) - len(shape) + i
        if shape[i] != target[dest] and shape[i] != 1:
            raise Error("Incompatible batch shapes; align trailing axes with equal or singleton dimensions.")
        if shape[i] == target[dest]:
            result[dest] = strides[i]


def _broadcast_dims(left: _Ints, right: _Ints, result: _MutInts) raises:
    var rank = len(result)
    for i in range(rank):
        var a = left[i - (rank - len(left))] if i >= rank - len(left) else 1
        var b = right[i - (rank - len(right))] if i >= rank - len(right) else 1
        if a == b or b == 1:
            result[i] = a
        elif a == 1:
            result[i] = b
        else:
            raise Error("Incompatible batch shapes; align trailing axes with equal or singleton dimensions.")


@always_inline
def _span_position(shape: _Ints, strides: _Ints, offset: Int, var logical: Int) -> Int:
    # Internal traversal only: 0 <= logical < size, so no empty-axis divide.
    var result = offset
    for j in range(len(shape)):
        var i = len(shape) - j - 1
        result += (logical % shape[i]) * strides[i]
        logical //= shape[i]
    return result


struct _Layout(ImplicitlyCopyable):
    """The 1-D strided run behind every batch owner; shapes live in _FlatLayout."""
    var shape: Array[Int, 1]
    var strides: Array[Int, 1]
    var offset: Int
    var size: Int

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self.size = copy.size

    def __init__(out self, shape: Array[Int, 1]) raises:
        self.shape = shape.copy()
        self.strides = Array[Int, 1](fill=0)
        self.offset = 0
        self.size = _row_major(_ints(self.shape), _mut_ints(self.strides))

    def __init__(out self, *, _size: Int, _offset: Int, _step: Int):
        self.shape = Array[Int, 1](fill=_size)
        self.strides = Array[Int, 1](fill=_step)
        self.offset = _offset
        self.size = _size

    @staticmethod
    def run(size: Int, offset: Int, step: Int) -> Self:
        """A validated run: size values from offset, step apart."""
        # A singleton's stride is unobservable and may otherwise overflow.
        return Self(_size=size, _offset=offset if size else 0, _step=step if size > 1 else 0)

    def contiguous(self) -> Bool:
        return self.size <= 1 or self.strides[0] == 1

    @always_inline
    def position(self, logical: Int) -> Int:
        # Internal traversal only: 0 <= logical < size.
        return self.offset + logical * self.strides[0]

    def sliced(self, axis: Int, selection: StridedSlice) raises -> Self:
        _ = _axis(axis, 1)
        var selected = _Selection(self.shape[0], selection)
        var result = Self([selected.count])
        if result.size == 0:
            return result
        result.offset = self.offset + selected.start * self.strides[0]
        # A singleton's stride is unobservable and may otherwise overflow.
        result.strides[0] = self.strides[0] * selected.step if selected.count > 1 else 0
        return result


struct _FlatLayout(ImplicitlyCopyable):
    """A rank-erased layout; a scalar always reads its only value.

    Traversal code shares one specialization per element type, not per rank.
    """
    var shape: List[Int]
    var strides: List[Int]
    var offset: Int
    var size: Int
    var contiguous: Bool
    var scalar: Bool

    def __init__(out self):
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.offset = 0
        self.size = 1
        self.contiguous = False
        self.scalar = True

    def __init__(out self, var shape: List[Int]) raises:
        """A fresh row-major layout, validated exactly like _Layout(shape)."""
        self = Self()
        self.scalar = False
        self.strides = List[Int](length=len(shape), fill=0)
        self.shape = shape^
        self.contiguous = True
        self.size = _row_major(_list_ints(self.shape), _list_mut_ints(self.strides))

    def permuted(self, axes: List[Int]) -> Self:
        """Reorder axes; callers pass a valid permutation."""
        var result = Self()
        result.scalar = False
        result.offset = self.offset
        result.size = self.size
        for axis in axes:
            result.shape.append(self.shape[axis])
            result.strides.append(self.strides[axis])
        result.contiguous = result._is_contiguous()
        return result^

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self.size = copy.size
        self.contiguous = copy.contiguous
        self.scalar = copy.scalar

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.shape = move.shape^
        self.strides = move.strides^
        self.offset = move.offset
        self.size = move.size
        self.contiguous = move.contiguous
        self.scalar = move.scalar

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    @always_inline
    def position(self, index: Int) -> Int:
        # Internal traversal only: 0 <= index < size, so no empty-axis divide.
        if self.scalar:
            return 0
        if self.contiguous:
            return self.offset + index
        if len(self.shape) == 1:
            return self.offset + index * self.strides[0]
        return _span_position(_list_ints(self.shape), _list_ints(self.strides), self.offset, index)

    def is_run(self, size: Int, offset: Int, step: Int) -> Bool:
        """Whether this is exactly the layout run(size, offset, step) builds."""
        return (
            not self.scalar and len(self.shape) == 1 and self.shape[0] == size
            and self.size == size and self.offset == (offset if size else 0)
            and self.strides[0] == (step if size > 1 else 0)
            and self.contiguous == (size <= 1 or step == 1)
        )

    @staticmethod
    def run(size: Int, offset: Int, step: Int) -> Self:
        """A validated 1-D run: size values from offset, step apart."""
        var result = Self()
        result.scalar = False
        result.shape.append(size)
        result.strides.append(step if size > 1 else 0)
        result.size = size
        result.offset = offset if size else 0
        result.contiguous = size <= 1 or step == 1
        return result^

    def at(self, axis: Int, index: Int) raises -> Self:
        """Select one index of an axis, dropping that axis."""
        var selection = _select_axis(_list_ints(self.shape), axis, index)
        var result = Self()
        result.scalar = False
        result.size = selection.size
        if result.size == 0:
            for _ in range(len(selection.shape)):
                result.strides.append(0)
        else:
            result.offset = self.offset + selection.index * self.strides[selection.axis]
            for i in range(len(self.shape)):
                if i != selection.axis:
                    result.strides.append(self.strides[i])
        result.shape = selection^.take_shape()
        result.contiguous = result._is_contiguous()
        return result^

    def _is_contiguous(self) -> Bool:
        return _is_contiguous(_list_ints(self.shape), _list_ints(self.strides), self.size)

    def ndim(self) -> Int:
        return len(self.shape)

    def reshaped(self, var shape: List[Int]) raises -> Self:
        """Reinterpret a contiguous layout; callers gather strided data first."""
        var result = Self(shape^)
        if result.size != self.size:
            raise Error("Cannot reshape to a different size; preserve the total number of elements.")
        result.offset = self.offset
        return result^

    def transposed(self, axes: List[Int]) raises -> Self:
        if len(axes) != len(self.shape):
            raise Error("Supply one transpose axis per batch axis; use a permutation of all axes.")
        var result = self
        _transpose_into(
            _list_ints(self.shape), _list_ints(self.strides), _list_ints(axes),
            _list_mut_ints(result.shape), _list_mut_ints(result.strides),
        )
        result.contiguous = result._is_contiguous()
        return result^

    def reversed_axes(self) -> Self:
        var axes = List[Int](capacity=len(self.shape))
        for i in range(len(self.shape)):
            axes.append(len(self.shape) - 1 - i)
        return self.permuted(axes)

    def sliced(self, axis: Int, selection: StridedSlice) raises -> Self:
        var a = _axis(axis, len(self.shape))
        var selected = _Selection(self.shape[a], selection)
        var shape = self.shape.copy()
        shape[a] = selected.count
        var result = Self(shape^)
        if result.size == 0:
            return result^
        result.strides = self.strides.copy()
        result.offset = self.offset + selected.start * self.strides[a]
        # A singleton's stride is unobservable and may otherwise overflow.
        result.strides[a] = self.strides[a] * selected.step if selected.count > 1 else 0
        result.contiguous = result._is_contiguous()
        return result^

    def broadcast_to(self, shape: List[Int]) raises -> Self:
        """View this layout with NumPy trailing-axis broadcasting."""
        if len(shape) < len(self.shape):
            raise Error("Incompatible batch shapes; align trailing axes with equal or singleton dimensions.")
        var result = Self(shape.copy())
        result.strides = List[Int](length=len(shape), fill=0)
        result.offset = self.offset
        _broadcast_strides(_list_ints(self.shape), _list_ints(self.strides), _list_ints(shape), _list_mut_ints(result.strides))
        result.contiguous = result._is_contiguous()
        return result^

    def progression(self) -> Optional[Int]:
        """The step if row-major order is one arithmetic run from offset.

        Such layouts can be stored as a 1-D strided vector without copying.
        """
        if self.scalar:
            return None
        if self.contiguous:
            return 1
        var step = Optional[Int]()
        var inner = 1
        for j in range(len(self.shape)):
            var i = len(self.shape) - j - 1
            if self.shape[i] == 1:
                continue
            if not step:
                step = self.strides[i]
            elif self.strides[i] != step.value() * inner:
                return None
            inner *= self.shape[i]
        if not step:
            return 1
        return step

    def cursor(self) -> _Cursor:
        return _Cursor(self)



def _broadcast_shapes(left: List[Int], right: List[Int]) raises -> List[Int]:
    var result = List[Int](length=max(len(left), len(right)), fill=1)
    _broadcast_dims(_list_ints(left), _list_ints(right), _list_mut_ints(result))
    return result^


struct _Cursor(Movable):
    """Visit a layout in row-major order, updating offsets incrementally.

    Stepping costs a few additions per element, with no division.
    """
    var shape: List[Int]
    var strides: List[Int]
    var index: List[Int]
    var position: Int
    var contiguous: Bool

    def __init__(out self, layout: _FlatLayout):
        self.shape = layout.shape.copy()
        self.strides = layout.strides.copy()
        self.index = List[Int](length=len(layout.shape), fill=0)
        self.position = 0 if layout.scalar else layout.offset
        self.contiguous = layout.contiguous and not layout.scalar

    @always_inline
    def advance(mut self):
        if self.contiguous:
            self.position += 1
            return
        var axis = len(self.shape) - 1
        while axis >= 0:
            self.index[axis] += 1
            self.position += self.strides[axis]
            if self.index[axis] < self.shape[axis]:
                return
            self.position -= self.strides[axis] * self.shape[axis]
            self.index[axis] = 0
            axis -= 1


def _layout_record_bytes() -> Int:
    """Bytes of a vector's shared layout record: its header, shape and strides."""
    return size_of[ArcPointer[_FlatLayout]._inner_type]() + 2 * size_of[Int]()
