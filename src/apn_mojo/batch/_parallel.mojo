"""Elementwise batch work on the worker pool, with sequential results and errors.

A run writes each result into its own preallocated slot, so the output is the
same as a sequential loop whatever the thread schedule. When elements fail,
the run reports the lowest failing index; chunks skip indices above a known
failure but finish every index below it, so that index is exact. Kernels are
pure, so the caller recomputes that one element on its own thread to raise
the error the sequential path would have raised.
"""

from std.atomic import Atomic, Ordering
from std.memory import ArcPointer, dealloc, Layout, ThinAllocation
from std.sys import size_of
from std.time import perf_counter_ns
from ..common._threads import _Context, _parallel_for, _parallel_workers
from ..common._sizes import _checked_count
from ._tensor import _Tensor
from ._values import _Ownership, _Values
from .mask import Mask
from ..integer._tiles import _Mask, _WIDTH

# Below this much estimated work a cold wake-up (30-40 us) eats most of the gain.
comptime _PARALLEL_NS = 100_000
# Elements timed on the calling thread before deciding.
comptime _PROBE = 16


struct _Slots[R: ImplicitlyCopyable & Deinitable]:
    """Output slots of one run, shared by every chunk."""

    var values: Pointer[Self.R, MutUntrackedOrigin]
    # Keep the counter outside the mutable slot record: with an inline
    # Atomic, Mojo 1.1 workers could lose a recorded failure.
    var failed: ArcPointer[Atomic[Int64]]

    def __init__(out self, values: Pointer[Self.R, MutUntrackedOrigin]):
        self.values = values
        self.failed = ArcPointer(Atomic(Int64.MAX))

    @always_inline
    def stopped(mut self, index: Int) -> Bool:
        """Whether a lower index already failed, so index needn't run."""
        return Int64(index) > self.failed[].load[ordering=Ordering.RELAXED]()

    @always_inline
    def put(self, index: Int, var value: Self.R):
        # Kernels write each range from its start and report how far they got,
        # so the written values are always a prefix of the range.
        self.values.unsafe_offset(index).unsafe_write(value^)

    def fail(mut self, index: Int):
        var current = self.failed[].load[ordering=Ordering.RELAXED]()
        while Int64(index) < current:
            if self.failed[].compare_exchange[
                success_ordering=Ordering.RELAXED, failure_ordering=Ordering.RELAXED
            ](current, Int64(index)):
                return


struct _ChunkLog(Movable):
    """Which participant ran each pool chunk and how much it wrote.

    Not generic, with out-of-line methods: runs are instantiated once per
    kernel, and this bookkeeping would otherwise be compiled into each.
    """

    # Pool chunks count from zero; tasks count from here. Task i writes
    # values [i * width, (i + 1) * width).
    var offset: Int
    var chunk: Int
    var width: Int
    # The participant that computed each pool chunk, for release by its owner.
    var owners: List[UInt16]
    # How many values each pool chunk wrote, from its start.
    var done: List[Int]

    def __init__(out self, width: Int):
        self.offset = 0
        self.chunk = 1
        self.width = width
        self.owners = List[UInt16]()
        self.done = List[Int]()

    @no_inline
    def start(mut self, offset: Int, chunk: Int, rest: Int):
        var chunks = (rest + chunk - 1) // chunk
        self.offset = offset
        self.chunk = chunk
        self.owners = List[UInt16](length=chunks, fill=0)
        self.done = List[Int](length=chunks, fill=0)

    @no_inline
    def record(mut self, begin: Int, worker: Int, written: Int):
        # Chunks are disjoint, so threads write disjoint entries.
        var index = begin // self.chunk
        self.owners[index] = UInt16(worker)
        self.done[index] = written

    @no_inline
    def ownership(mut self) -> Optional[_Ownership]:
        if not len(self.owners):
            return None
        var owners = List[UInt16]()
        swap(owners, self.owners)
        return _Ownership(self.offset * self.width, self.chunk * self.width, owners^)


struct _Run[R: ImplicitlyCopyable & Deinitable, S: Movable & Deinitable]:
    var state: Self.S
    var slots: _Slots[Self.R]
    var log: _ChunkLog

    def __init__(out self, var state: Self.S, var slots: _Slots[Self.R], width: Int):
        self.state = state^
        self.slots = slots^
        self.log = _ChunkLog(width)


def _run_chunk[
    R: ImplicitlyCopyable & Deinitable, S: Movable & Deinitable, //,
    kernel: def(S, Int, Int, mut _Slots[R]) thin -> Int,
](context: _Context, begin: Int, end: Int, worker: Int):
    ref run = context.value().unsafe_bitcast[_Run[R, S]]()[]
    var offset = run.log.offset
    run.log.record(begin, worker, kernel(run.state, offset + begin, offset + end, run.slots))


struct _ParallelResult[R: ImplicitlyCopyable & Deinitable](Movable):
    """Every result in order, or the lowest failing index with nothing kept."""

    var values: _Values[Self.R]
    var failed: Int

    def __init__(out self, var values: _Values[Self.R], failed: Int):
        self.values = values^
        self.failed = failed

    def take(deinit self) -> _Values[Self.R]:
        return self.values^


def _parallel_map[
    R: ImplicitlyCopyable & Deinitable, S: Movable & Deinitable, //,
    kernel: def(S, Int, Int, mut _Slots[R]) thin -> Int,
](var state: S, count: Int, width: Int = 1) -> _ParallelResult[R]:
    """Run kernel over tasks [0, count): the first here, the rest in parallel
    when they are estimated to take long enough to pay for the wake-up.

    Task i writes values [i * width, (i + 1) * width) in order. A kernel given
    tasks [begin, end) returns how many values it wrote from begin * width.
    """
    var values = List[R](unsafe_uninit_length=count * width)
    var run = _Run[R, S](state^, _Slots[R](values.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()), width)
    var context = _Context(Pointer(to=run).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]())
    var probe = min(_PROBE, count)
    var start = perf_counter_ns()
    var probed = kernel(run.state, 0, probe, run.slots)
    var elapsed = Int(perf_counter_ns() - start)
    # Values written by this thread after the probe, when the rest ran here.
    var tail = 0
    if probe < count and probed == probe * width:
        var rest = count - probe
        var estimate = elapsed * rest // max(1, probe)
        var workers = _parallel_workers() if estimate >= _PARALLEL_NS else 1
        if workers > 1:
            # About eight chunks per thread balance uneven elements.
            var chunk = max(1, rest // (workers * 8))
            run.log.start(probe, chunk, rest)
            _parallel_for(_run_chunk[kernel], context, rest, chunk)
        else:
            tail = kernel(run.state, probe, count, run.slots)
    var failed = Int(run.slots.failed[].load[ordering=Ordering.ACQUIRE]())
    if failed == Int(Int64.MAX):
        var ownership: Optional[_Ownership] = None
        # Values without a destructor have nothing for their owners to free.
        comptime if not R.__del__is_trivial:
            ownership = run.log.ownership()
        _ = run^
        return _ParallelResult(_Values[R](values^, ownership^), -1)
    _discard_written(values^, probed, probe * width, tail, run.log)
    _ = run^
    return _ParallelResult(_Values[R](List[R]()), failed)


@no_inline
def _discard_written[R: ImplicitlyCopyable & Deinitable](
    var values: List[R], probed: Int, rest: Int, tail: Int, log: _ChunkLog,
):
    """Destroy exactly the values a failed run wrote, then free the buffer.

    They are a prefix of the probe, of the tail (from value `rest`) and of
    every pool chunk.
    """
    for i in range(probed):
        values.unsafe_ptr().unsafe_offset(i).unsafe_deinit_pointee()
    for i in range(rest, rest + tail):
        values.unsafe_ptr().unsafe_offset(i).unsafe_deinit_pointee()
    for c in range(len(log.done)):
        var begin = (log.offset + c * log.chunk) * log.width
        for i in range(begin, begin + log.done[c]):
            values.unsafe_ptr().unsafe_offset(i).unsafe_deinit_pointee()
    var capacity = values.capacity()
    dealloc(ThinAllocation(unsafe_owned_ptr=values.unsafe_take_allocation().unsafe_leak()).unsafe_with_layout(Layout[R](count=capacity)))


trait _ElementKernel(Movable, Deinitable):
    """One chunk's private readers and scalar computation.

    A kernel starts at `begin` and receives consecutive indices. The job is
    borrowed and outlives the run; reader state must never be shared between
    chunks. Constructing a fresh kernel at an index must reproduce that call.
    """

    comptime Element: ImplicitlyCopyable & Deinitable
    comptime Job: Movable & Deinitable

    def __init__(out self, job: Self.Job, begin: Int) raises: ...

    def apply(mut self, job: Self.Job, index: Int) raises -> Self.Element: ...


trait _ElementErrors(Movable, Deinitable):
    """The frontend's coordinates and diagnostics, outside the hot loop."""

    def failure(self, index: Int, var error: Error) -> Error: ...

    def repeated(self, index: Int) -> Error: ...


@fieldwise_init
struct _MapErrors(_ElementErrors):
    var diagnostic: String
    var index_prefix: StaticString

    def failure(self, index: Int, var error: Error) -> Error:
        if not self.diagnostic and self.index_prefix == "":
            return error^
        return Error(self.diagnostic + self.index_prefix + String(index) + ": " + String(error))

    def repeated(self, index: Int) -> Error:
        return Error(String(
            "Batch element ", index, " failed on a worker thread but"
            " not when repeated; mapped functions must be pure.",
        ))


struct _IndexedKernel[
    R: ImplicitlyCopyable & Deinitable, S: Copyable & Deinitable, //,
    element: def(S, Int) raises thin -> R,
](_ElementKernel):
    comptime Element = Self.R
    comptime Job = Self.S

    def __init__(out self, job: Self.Job, begin: Int):
        pass

    @always_inline
    def apply(mut self, job: Self.Job, index: Int) raises -> Self.R:
        return Self.element(job, index)


def _kernel_chunk[K: _ElementKernel, origin: ImmOrigin](
    job: Pointer[K.Job, origin], begin: Int, end: Int, mut slots: _Slots[K.Element],
) -> Int:
    var index = begin
    try:
        var kernel = K(job[], begin)
        while index < end:
            if slots.stopped(index):
                break
            slots.put(index, kernel.apply(job[], index))
            index += 1
    except:
        slots.fail(index)
    return index - begin


def _execute_values[K: _ElementKernel, E: _ElementErrors](
    job: K.Job, count: Int, errors: E, *, probe: Bool = False,
) raises -> _Values[K.Element]:
    """Stage a complete elementwise result, or release it and raise.

    `probe` preserves callers that always time an initial chunk. Otherwise
    small runs use a plain loop. Both paths use the same kernel and report
    the first failing logical index, including reader initialization errors.
    """
    _ = _checked_count(count, max(1, size_of[K.Element]()))
    if probe or _parallel_ready(count):
        # The pool joins before returning. Lend the job, including its plan,
        # instead of copying it; the pointer retains the borrow's origin.
        var result = _parallel_map[_kernel_chunk[K, origin_of(job)]](Pointer(to=job), count)
        var failed = result.failed
        var values = result^.take()
        if failed >= 0:
            try:
                var kernel = K(job, failed)
                _ = kernel.apply(job, failed)
            except error:
                raise errors.failure(failed, error^)
            raise errors.repeated(failed)
        return values^
    var values = List[K.Element](capacity=count)
    var index = 0
    try:
        var kernel = K(job, 0)
        while index < count:
            values.append(kernel.apply(job, index))
            index += 1
    except error:
        raise errors.failure(index, error^)
    return _Values[K.Element](values^)



@always_inline
def _map_tensor[
    R: ImplicitlyCopyable & Deinitable, S: Copyable & Deinitable, //,
    element: def(S, Int) raises thin -> R,
](
    state: S, count: Int, *, diagnostic: String = "", index_prefix: StaticString = "",
) raises -> _Tensor[R]:
    """Elementwise results in order: on the pool when the run is long enough,
    else a plain loop. Errors read as the sequential map's."""
    return _Tensor[R](_map_values[element](state, count, diagnostic=diagnostic, index_prefix=index_prefix), [count])


@always_inline
def _map_values[
    R: ImplicitlyCopyable & Deinitable, S: Copyable & Deinitable, //,
    element: def(S, Int) raises thin -> R,
](
    state: S, count: Int, *, diagnostic: String = "", index_prefix: StaticString = "",
) raises -> _Values[R]:
    """As _map_tensor, keeping the values and who computed them."""
    return _execute_values[_IndexedKernel[element]](state, count, _MapErrors(diagnostic, index_prefix))

# Runs shorter than this stay on the sequential path: they cannot take long
# enough to pay for a wake-up.
comptime _PARALLEL_MIN = 64


def _parallel_ready(count: Int) -> Bool:
    """Whether a run is long enough to consider, and a pool exists to run it.

    Without a pool (one usable CPU, or an unsupported platform) callers keep
    their own sequential loop and pay nothing for the parallel machinery.
    """
    return count >= _PARALLEL_MIN and _parallel_workers() > 1


@fieldwise_init
struct _MaskJob[S: Copyable & Deinitable](Copyable, Movable):
    var state: Self.S
    var count: Int


def _mask_tile[
    S: Copyable & Deinitable, //,
    element: def(S, Int) raises thin -> Bool,
](job: _MaskJob[S], tile: Int) raises -> _Mask:
    """One mask tile, its elements in order; the first failure raises."""
    var result = _Mask(fill=False)
    var start = tile * _WIDTH
    for lane in range(min(_WIDTH, job.count - start)):
        result[lane] = element(job.state, start + lane)
    return result


struct _MaskErrors(_ElementErrors):
    def __init__(out self):
        pass

    def failure(self, index: Int, var error: Error) -> Error:
        return error^

    def repeated(self, index: Int) -> Error:
        return _MapErrors("", "").repeated(index * _WIDTH)


def _parallel_mask[
    S: Copyable & Deinitable, //,
    element: def(S, Int) raises thin -> Bool,
](state: S, count: Int) raises -> Mask:
    """Elementwise truth values as a Mask, in parallel when the run is long.

    Work is split by whole tiles, the mask's own storage, so no thread shares
    a tile and nothing is packed afterwards. Errors are the sequential ones.
    """
    var tiles = (count + _WIDTH - 1) // _WIDTH
    var job = _MaskJob(state.copy(), count)
    var values = _execute_values[_IndexedKernel[_mask_tile[element]]](job, tiles, _MaskErrors())
    return Mask(length=count, _tiles=values^.take_list())
