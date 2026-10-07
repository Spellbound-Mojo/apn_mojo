"""Private, single-worker census of quiescent integer ownership graphs.

Keep every supplied root alive and unchanged until finish; start a fresh census
after mutation or destruction. Addresses are identity keys, never dereferenced.
The census owns no numeric handles and is not imported by the public package.
"""

from std.memory import ArcPointer
from std.sys import size_of

from ..integer.value import Integer
from ..integer._static import _StaticInteger
from ..batch._values import _Values
from ..batch.value import Batch, _BatchIterator
from ..batch._layout import _FlatLayout
from ..batch.mask import Mask, _MaskStorage, _MaskIterator
from ..integer._tiles import _Mask
from ._sizes import _checked_sum, _checked_count


comptime _CURRENT = 1
comptime _STAGING = 2
comptime _WORKSPACE = 4
comptime _SNAPSHOT = 8
comptime _STATIC = 16


def _memory_bytes(count: Int, width: Int) raises -> Int:
    if count < 0 or width <= 0 or count > Int.MAX // width:
        raise Error(
            "Cannot account for numeric memory: byte count exceeds the"
            " addressable range; inspect a smaller set of values."
        )
    return count * width


struct _MemoryBytes(ImplicitlyCopyable):
    var payload: Int
    var padding: Int
    var spare: Int
    var metadata: Int
    var buffers: Int
    var allocations: Int

    def __init__(
        out self,
        *,
        payload: Int = 0,
        padding: Int = 0,
        spare: Int = 0,
        metadata: Int = 0,
        buffers: Int = 0,
        allocations: Int = 0,
    ):
        self.payload = payload
        self.padding = padding
        self.spare = spare
        self.metadata = metadata
        self.buffers = buffers
        self.allocations = allocations

    def total(self) raises -> Int:
        return _checked_sum(
            _checked_sum(self.payload, self.padding),
            _checked_sum(_checked_sum(self.spare, self.metadata), self.buffers),
        )

    def add(mut self, value: Self) raises:
        self.payload = _checked_sum(self.payload, value.payload)
        self.padding = _checked_sum(self.padding, value.padding)
        self.spare = _checked_sum(self.spare, value.spare)
        self.metadata = _checked_sum(self.metadata, value.metadata)
        self.buffers = _checked_sum(self.buffers, value.buffers)
        self.allocations = _checked_sum(self.allocations, value.allocations)


struct _MemoryReport(ImplicitlyCopyable):
    var current: _MemoryBytes
    var staging: _MemoryBytes
    var workspace: _MemoryBytes
    var snapshot_only: _MemoryBytes
    var static_data: _MemoryBytes
    var inline_bytes: Int
    var diagnostic_bytes: Int
    var snapshot_reachable_bytes: Int
    var shared_role_bytes: Int

    def __init__(out self, *, inline_bytes: Int = 0):
        self.current = _MemoryBytes()
        self.staging = _MemoryBytes()
        self.workspace = _MemoryBytes()
        self.snapshot_only = _MemoryBytes()
        self.static_data = _MemoryBytes()
        self.inline_bytes = inline_bytes
        self.diagnostic_bytes = 0
        self.snapshot_reachable_bytes = 0
        self.shared_role_bytes = 0

    def heap(self) raises -> _MemoryBytes:
        var result = self.current
        result.add(self.staging)
        result.add(self.workspace)
        result.add(self.snapshot_only)
        return result


@fieldwise_init
struct _MemoryEntry(ImplicitlyCopyable):
    var address: Int
    var roles: Int
    var known_value: Bool
    var bytes: _MemoryBytes


@fieldwise_init
struct _MemoryRoot(ImplicitlyCopyable):
    var address: Int
    var kind: Int


struct _MemoryCensus(Movable):
    var _entries: List[_MemoryEntry]
    var _roots: List[_MemoryRoot]
    var _inline_bytes: Int

    def __init__(out self):
        self._entries = List[_MemoryEntry]()
        self._roots = List[_MemoryRoot]()
        self._inline_bytes = 0

    def _root(mut self, address: Int, kind: Int, width: Int) raises -> Bool:
        for root in self._roots:
            if root.address == address and root.kind == kind:
                return False
        self._inline_bytes = _checked_sum(self._inline_bytes, width)
        _ = _checked_count(
            _checked_sum(len(self._roots), 1), size_of[_MemoryRoot]()
        )
        self._roots.append(_MemoryRoot(address, kind))
        return True

    def _record(
        mut self,
        address: Int,
        role: Int,
        bytes: _MemoryBytes,
        known_value: Bool = True,
    ) raises:
        for i in range(len(self._entries)):
            ref entry = self._entries[i]
            if entry.address == address:
                if entry.bytes.total() != bytes.total() or (
                    entry.known_value
                    and known_value
                    and (
                        entry.bytes.payload != bytes.payload
                        or entry.bytes.padding != bytes.padding
                        or entry.bytes.spare != bytes.spare
                        or entry.bytes.metadata != bytes.metadata
                        or entry.bytes.buffers != bytes.buffers
                    )
                ):
                    raise Error(
                        "Numeric memory changed during accounting; discard"
                        " the census and inspect unchanged live values."
                    )
                entry.roles |= role
                # A selection may describe the same buffer first seen as staging.
                # Never interpret partially written staging lengths as a value.
                if known_value:
                    entry.bytes = bytes
                    entry.known_value = True
                return
        _ = _checked_count(
            _checked_sum(len(self._entries), 1), size_of[_MemoryEntry]()
        )
        self._entries.append(_MemoryEntry(address, role, known_value, bytes))

    def _integer(mut self, value: Integer, role: Int) raises:
        if value._storage.isa[_StaticInteger]():
            ref literal = value._storage[_StaticInteger]
            self._record(
                Int(literal.data),
                role | _STATIC,
                _MemoryBytes(
                    payload=_memory_bytes(literal._word_count(), 4),
                    metadata=12,
                ),
            )
            return
        if not value._storage.isa[Integer._Shared]():
            return
        ref shared = value._storage[Integer._Shared]
        ref words = shared.ptr()[].words
        var inline = shared.inline_capacity()
        var external = words.is_external()
        self._record(
            Int(shared.ptr()),
            role,
            _MemoryBytes(
                metadata=size_of[Integer._Shared._inner_type](),
                payload=0 if external else _memory_bytes(len(words), 4),
                spare=_memory_bytes(inline if external else inline - len(words), 4),
                padding=shared.allocation_bytes() - size_of[Integer._Shared._inner_type]() - inline * 4,
                allocations=1,
            ),
        )
        if external and words.capacity():
            self._record(
                Int(words.unsafe_ptr()),
                role,
                _MemoryBytes(
                    payload=_memory_bytes(len(words), size_of[UInt32]()),
                    spare=_memory_bytes(
                        words.capacity() - len(words), size_of[UInt32]()
                    ),
                    allocations=1,
                ),
            )



    def _mask(mut self, shared: ArcPointer[_MaskStorage], role: Int) raises:
        self._record(
            Int(shared.ptr()),
            role,
            _MemoryBytes(
                metadata=size_of[ArcPointer[_MaskStorage]._inner_type](),
                allocations=1,
            ),
        )
        ref storage = shared.ptr()[]
        if storage.tiles.capacity():
            var rows = _memory_bytes(len(storage.tiles), size_of[_Mask]())
            var capacity = _memory_bytes(
                storage.tiles.capacity(), size_of[_Mask]()
            )
            self._record(
                Int(storage.tiles.unsafe_ptr()),
                role,
                _MemoryBytes(
                    payload=storage.length,
                    padding=rows - storage.length,
                    spare=capacity - rows,
                    allocations=1,
                ),
            )

    def _integer_values(mut self, shared: ArcPointer[_Values[Integer]], role: Int) raises:
        self._record(Int(shared.ptr()), role, _MemoryBytes(
            metadata=size_of[ArcPointer[_Values[Integer]]._inner_type](), allocations=1,
        ))
        ref values = shared[].list
        if values.capacity():
            self._record(Int(values.unsafe_ptr()), role, _MemoryBytes(
                metadata=_memory_bytes(len(values), size_of[Integer]()),
                spare=_memory_bytes(values.capacity() - len(values), size_of[Integer]()),
                allocations=1,
            ))
        for value in values:
            self._integer(value, role)

    def add(mut self, value: Integer) raises:
        if self._root(Int(Pointer(to=value)), 0, size_of[Integer]()):
            self._integer(value, _CURRENT)

    def _layout(mut self, shared: ArcPointer[_FlatLayout], role: Int) raises:
        # One immutable layout record, shared by copies and selections.
        self._record(Int(shared.ptr()), role, _MemoryBytes(
            metadata=size_of[ArcPointer[_FlatLayout]._inner_type](), allocations=1,
        ))
        self._ints(shared[].shape, role)
        self._ints(shared[].strides, role)

    def _ints(mut self, values: List[Int], role: Int) raises:
        if values.capacity():
            self._record(Int(values.unsafe_ptr()), role, _MemoryBytes(
                metadata=_memory_bytes(values.capacity(), size_of[Int]()),
                allocations=1,
            ))

    def add(mut self, value: Batch[Integer]) raises:
        if not self._root(Int(Pointer(to=value)), 1, size_of[Batch[Integer]]()):
            return
        self._integer_values(value._owner, _CURRENT)
        self._layout(value._layout, _CURRENT)

    def add(mut self, value: Mask) raises:
        if self._root(Int(Pointer(to=value)), 3, size_of[Mask]()):
            self._mask(value._storage, _CURRENT)

    def add(mut self, value: _BatchIterator[Integer]) raises:
        if not self._root(
            Int(Pointer(to=value)), 4, size_of[_BatchIterator[Integer]]()
        ):
            return
        self._integer_values(value._native._owner, _SNAPSHOT)

    def add(mut self, value: _MaskIterator) raises:
        if self._root(Int(Pointer(to=value)), 5, size_of[_MaskIterator]()):
            self._mask(value._storage, _SNAPSHOT)

    def finish(self) raises -> _MemoryReport:
        var report = _MemoryReport(inline_bytes=self._inline_bytes)
        report.diagnostic_bytes = _checked_sum(
            _memory_bytes(self._entries.capacity(), size_of[_MemoryEntry]()),
            _memory_bytes(self._roots.capacity(), size_of[_MemoryRoot]()),
        )
        for entry in self._entries:
            if entry.roles & _STATIC:
                report.static_data.add(entry.bytes)
                continue
            if entry.roles & _CURRENT:
                report.current.add(entry.bytes)
            elif entry.roles & _STAGING:
                report.staging.add(entry.bytes)
            elif entry.roles & _WORKSPACE:
                report.workspace.add(entry.bytes)
            else:
                report.snapshot_only.add(entry.bytes)
            if entry.roles & _SNAPSHOT:
                report.snapshot_reachable_bytes = _checked_sum(
                    report.snapshot_reachable_bytes, entry.bytes.total()
                )
            if entry.roles & (entry.roles - 1):
                report.shared_role_bytes = _checked_sum(
                    report.shared_role_bytes, entry.bytes.total()
                )
        _ = report.heap().total()
        _ = report.static_data.total()
        return report
