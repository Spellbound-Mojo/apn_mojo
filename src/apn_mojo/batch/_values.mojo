"""Batch value storage: the values and, for parallel results, who computed them.

A heap-backed result is cheapest to free on the thread that allocated it: the
allocator keeps its memory in per-thread caches and arenas, and freeing another
thread's block takes a lock and moves cache lines between cores. A parallel run
therefore records which pool participant computed each chunk, and when the last
reference to a large run goes away, every participant frees exactly the
elements it computed, in parallel.
"""

from std.memory import dealloc, Layout, ThinAllocation
from ..common._threads import _Context, _parallel_each

# Shorter runs release on the dropping thread: a pool wake-up costs more.
comptime _RELEASE_MIN = 1024


@always_inline
def _borrow_values[
    T: ImplicitlyCopyable & Deinitable, //, origin: ImmOrigin,
](values: List[T]) -> Span[T, origin]:
    """Read-only access to storage the caller retains for `origin`.

    The caller must own this list, directly or through a retained _Values,
    and keep its storage and elements unchanged until the borrow ends. Mojo
    1.1 cannot connect the list's element origin to that enclosing owner, so
    the lifetime cast lives here. It never grants mutable access.

    Span indexing checks bounds; layout walkers may use unsafe_get only for
    validated positions. This view neither copies elements nor retains an
    additional owner. A raw pointer, when needed, keeps the same origin.
    """
    return Span(unsafe_ptr=values.unsafe_ptr().as_imm().unsafe_origin_cast[origin](), length=len(values))


@fieldwise_init
struct _Ownership(Copyable, Movable):
    """Who computed which elements of a parallel run.

    Elements [0, offset) were computed by participant 0 (the calling thread);
    element i >= offset by participant owners[(i - offset) // chunk].
    """

    var offset: Int
    var chunk: Int
    var owners: List[UInt16]


struct _Values[T: ImplicitlyCopyable & Deinitable](Movable, Sized):
    """A batch's value list, released by the threads that computed it."""

    var list: List[Self.T]
    var ownership: Optional[_Ownership]

    def __init__(out self, var list: List[Self.T]):
        self.list = list^
        self.ownership = None

    def __init__(out self, var list: List[Self.T], var ownership: Optional[_Ownership]):
        self.list = list^
        self.ownership = ownership^

    def __len__(self) -> Int:
        return len(self.list)

    def append(mut self, var value: Self.T):
        self.list.append(value^)

    def take_list(deinit self) -> List[Self.T]:
        """The values alone, for a caller that releases them itself."""
        return self.list^

    @always_inline
    def __getitem__(self, index: Int) -> ref[self] Self.T:
        return _borrow_values[origin_of(self)](self.list)[index]

    def __deinit__(deinit self):
        if not self.ownership or len(self.list) < _RELEASE_MIN:
            return
        var list = self.list^
        var job = _Release[Self.T](list.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), len(list), self.ownership.take())
        _parallel_each(
            _release_part[Self.T],
            _Context(Pointer(to=job).unsafe_bitcast[NoneType]().unsafe_origin_cast[MutUntrackedOrigin]()),
        )
        # Every element is destroyed; free the buffer without destroying them again.
        var capacity = list.capacity()
        dealloc(ThinAllocation(unsafe_owned_ptr=list.unsafe_take_allocation().unsafe_leak()).unsafe_with_layout(Layout[Self.T](count=capacity)))
        _ = job^


struct _Release[T: ImplicitlyCopyable & Deinitable](Movable):
    var values: Pointer[Self.T, MutUntrackedOrigin]
    var count: Int
    var ownership: _Ownership

    def __init__(out self, values: Pointer[Self.T, MutUntrackedOrigin], count: Int, var ownership: _Ownership):
        self.values = values
        self.count = count
        self.ownership = ownership^


def _release_part[T: ImplicitlyCopyable & Deinitable](context: _Context, worker: Int, end: Int, participant: Int):
    """Destroy the elements participant `worker` computed."""
    ref job = context.value().unsafe_bitcast[_Release[T]]()[]
    ref ownership = job.ownership
    if worker == 0:
        for i in range(min(ownership.offset, job.count)):
            job.values.unsafe_offset(i).unsafe_deinit_pointee()
    for c in range(len(ownership.owners)):
        if Int(ownership.owners[c]) != worker:
            continue
        var begin = ownership.offset + c * ownership.chunk
        for i in range(begin, min(begin + ownership.chunk, job.count)):
            job.values.unsafe_offset(i).unsafe_deinit_pointee()
