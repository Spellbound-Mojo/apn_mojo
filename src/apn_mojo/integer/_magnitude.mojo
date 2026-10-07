"""Private shared magnitudes with optional in-allocation result limbs."""

from std.atomic import Atomic, Ordering, fence
from std.ffi import external_call, c_size_t
from std.memory import dealloc, Layout, ThinAllocation, MaybeUninit
from std.os import abort
from std.sys import size_of, align_of
from std.builtin.builtin_slice import ContiguousSlice
from ..common._sizes import _checked_count, _checked_sum


struct _MagnitudeWords(Movable, Sized):
    # External buffers are adopted through List's allocation-transfer API.
    # An inline tail belongs to the shared header and is never freed here.
    var _data: Pointer[UInt32, MutUntrackedOrigin]
    var _length: Int
    var _capacity: Int
    var _external: Bool

    def __init__(out self, var words: List[UInt32]):
        self._length = len(words)
        self._capacity = words.capacity()
        self._data = words.unsafe_take_allocation().unsafe_leak()
        self._external = self._capacity != 0

    def __init__(out self, *, data: Pointer[UInt32, MutUntrackedOrigin], capacity: Int):
        self._data = data
        self._length = capacity
        self._capacity = capacity
        self._external = False

    def __deinit__(deinit self):
        if self._external:
            dealloc(ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(Layout[UInt32](count=self._capacity)))

    def __len__(self) -> Int:
        return self._length

    def capacity(self) -> Int:
        return self._capacity

    def is_external(self) -> Bool:
        return self._external

    def unsafe_ptr(ref self) -> Pointer[UInt32, origin_of(self)]:
        return self._data.unsafe_mut_cast[origin_of(self).mut]().unsafe_origin_cast[origin_of(self)]()

    @always_inline
    def span(ref self) -> Span[UInt32, origin_of(self)]:
        return Span(unsafe_ptr=self.unsafe_ptr(), length=self._length)

    def __getitem__(ref self, index: Int) -> ref[self] UInt32:
        debug_assert(0 <= index < self._length, "Magnitude index out of bounds")
        return self.unsafe_ptr().unsafe_offset(index)[]

    def __getitem__(ref self, slice: ContiguousSlice) -> Span[UInt32, origin_of(self)]:
        return Span(unsafe_ptr=self.unsafe_ptr(), length=self._length)[slice]

    def resize(mut self, length: Int, fill: UInt32):
        if length > self._capacity:
            var replacement = List[UInt32](length=length, fill=fill)
            var source = self.span()
            for i in range(self._length):
                replacement[i] = source.unsafe_get(i)
            self = Self(replacement^)
        else:
            for i in range(self._length, length):
                self._data.unsafe_offset(i).unsafe_write(fill)
        self._length = length

    def finish(mut self, used: Int, offset: Int):
        if self._capacity > 128 and (self._capacity - 1) // 16 >= used:
            var compact = List[UInt32](unsafe_uninit_length=used)
            var source = self.span()
            for i in range(used):
                compact.unsafe_ptr().unsafe_offset(i).unsafe_write(source.unsafe_get(offset + i))
            self = Self(compact^)
        else:
            for i in range(used):
                self._data.unsafe_offset(i)[] = self._data.unsafe_offset(offset + i)[]
            self._length = used


struct _LargeInteger(Movable):
    var negative: Bool
    var words: _MagnitudeWords

    def __init__(out self, negative: Bool, var words: List[UInt32]):
        self.negative = negative
        self.words = _MagnitudeWords(words^)

    def __init__(out self, *, negative: Bool, var words: _MagnitudeWords):
        self.negative = negative
        self.words = words^


struct _MagnitudeHeader(Movable):
    var strong: Atomic[UInt64]
    # One implicit weak reference keeps the header alive while strong > 0.
    var weak: Atomic[UInt64]
    var inline_capacity: Int
    var value: MaybeUninit[_LargeInteger]

    def __init__(out self, var value: _LargeInteger, inline_capacity: Int):
        self.strong = Atomic(UInt64(1))
        self.weak = Atomic(UInt64(1))
        self.inline_capacity = inline_capacity
        self.value = MaybeUninit[_LargeInteger](value^)

    def __deinit__(deinit self):
        # Last-strong release already destroyed the payload, before weak release.
        self.value^.unsafe_forget()


@always_inline
def _allocate_magnitude_header(count: Int) -> Pointer[_MagnitudeHeader, MutUntrackedOrigin]:
    # Only the header allocation uses libc; adopted List buffers retain their
    # original allocator. All callers preflight the byte count.
    comptime assert align_of[_MagnitudeHeader]() <= align_of[UInt64]()
    var optional = external_call[
        "malloc", OptionalPointer[_MagnitudeHeader, MutUntrackedOrigin]
    ](c_size_t(count * 8))
    if not optional:
        abort("Cannot allocate numeric storage: physical memory exhausted.")
    return optional.unsafe_value()


def _release_magnitude_weak(ptr: Pointer[_MagnitudeHeader, MutUntrackedOrigin]):
    # The sole remaining weak reference cannot be copied during its destruction.
    # An acquire observation also synchronizes earlier releases of other handles.
    if ptr[].weak.load[ordering=Ordering.ACQUIRE]() != 1:
        if ptr[].weak.fetch_sub[ordering=Ordering.RELEASE](1) != 1:
            return
        fence[ordering=Ordering.ACQUIRE]()
    ptr.unsafe_deinit_pointee()
    external_call["free", NoneType](ptr)


struct _SharedMagnitude(ImplicitlyCopyable):
    comptime _inner_type = _MagnitudeHeader
    comptime WeakPointer = _WeakMagnitude
    var _raw: Pointer[_MagnitudeHeader, MutUntrackedOrigin]

    def __init__(out self, var value: _LargeInteger):
        comptime assert size_of[_MagnitudeHeader]() % 8 == 0
        var count = size_of[_MagnitudeHeader]() // 8
        self._raw = _allocate_magnitude_header(count)
        self._raw.unsafe_write(_MagnitudeHeader(value^, 0))

    def __init__(out self, *, retained: Pointer[_MagnitudeHeader, MutUntrackedOrigin]):
        self._raw = retained

    def __init__(out self, *, copy: Self):
        self._raw = copy._raw
        _ = self._raw[].strong.fetch_add[ordering=Ordering.RELAXED](1)

    def __deinit__(deinit self):
        # The sole owner, with no weak reference besides the implicit one: no
        # other handle exists and none can be made, so the block is released
        # without an atomic write (about 18 cycles here). The acquire loads
        # synchronize with earlier releases by other handles, as the fence does.
        if (
            self._raw[].strong.load[ordering=Ordering.ACQUIRE]() == 1
            and self._raw[].weak.load[ordering=Ordering.ACQUIRE]() == 1
        ):
            self._raw[].value.unsafe_ptr().unsafe_deinit_pointee()
            _release_magnitude_weak(self._raw)
            return
        if self._raw[].strong.fetch_sub[ordering=Ordering.RELEASE](1) == 1:
            fence[ordering=Ordering.ACQUIRE]()
            self._raw[].value.unsafe_ptr().unsafe_deinit_pointee()
            _release_magnitude_weak(self._raw)

    @staticmethod
    @always_inline
    def uninitialized(capacity: Int, negative: Bool) raises -> Self:
        var bytes = _checked_sum(size_of[_MagnitudeHeader](), _checked_count(capacity, 4) * 4)
        var count = _checked_sum(bytes, 7) // 8
        var raw = _allocate_magnitude_header(count)
        var data = raw.unsafe_bitcast[UInt8]().unsafe_offset(size_of[_MagnitudeHeader]()).unsafe_bitcast[UInt32]()
        raw.unsafe_write(_MagnitudeHeader(
            _LargeInteger(negative=negative, words=_MagnitudeWords(data=data, capacity=capacity)), capacity,
        ))
        return Self(retained=raw)

    def __getitem__(ref self) -> ref[self] _LargeInteger:
        return self._raw[].value.unsafe_ptr().unsafe_mut_cast[origin_of(self).mut]().unsafe_origin_cast[origin_of(self)]()[]

    def ptr(self) -> Pointer[_LargeInteger, origin_of(self)]:
        return self._raw[].value.unsafe_ptr().as_imm().unsafe_origin_cast[origin_of(self)]()

    def count(self) -> UInt64:
        return self._raw[].strong.load[ordering=Ordering.RELAXED]()

    def weak_count(self) -> UInt64:
        return self._raw[].weak.load[ordering=Ordering.RELAXED]() - 1

    def inline_capacity(self) -> Int:
        return self._raw[].inline_capacity

    def allocation_bytes(self) -> Int:
        return (size_of[_MagnitudeHeader]() + self._raw[].inline_capacity * 4 + 7) // 8 * 8


struct _WeakMagnitude(ImplicitlyCopyable):
    var _raw: Pointer[_MagnitudeHeader, MutUntrackedOrigin]

    def __init__(out self, owner: _SharedMagnitude):
        self._raw = owner._raw
        _ = self._raw[].weak.fetch_add[ordering=Ordering.RELAXED](1)

    def __init__(out self, *, copy: Self):
        self._raw = copy._raw
        _ = self._raw[].weak.fetch_add[ordering=Ordering.RELAXED](1)

    def __deinit__(deinit self):
        _release_magnitude_weak(self._raw)

    def try_upgrade(self) -> Optional[_SharedMagnitude]:
        var count = self._raw[].strong.load[ordering=Ordering.RELAXED]()
        while count:
            if self._raw[].strong.compare_exchange[
                success_ordering=Ordering.ACQUIRE, failure_ordering=Ordering.RELAXED
            ](count, count + 1):
                return _SharedMagnitude(retained=self._raw)
        return None
