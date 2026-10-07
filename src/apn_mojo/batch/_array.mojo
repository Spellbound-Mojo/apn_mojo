"""Arrow-style untyped arrays shared by every element family and rank.

An _ArrayData is the one record type behind every batch that is not a plain
vector: a runtime family tag, one shared immutable buffer and a runtime
layout, like Arrow's ArrayData. Kernels take these records, or scalar
operands, and dispatch on the tag, so each operation compiles once per result
family instead of once per operand type, rank and view kind. The families and
their tags come from `_families.mojo`; a dispatch on the tag is one loop over
that list.
"""

from std.builtin.builtin_slice import StridedSlice
from std.memory import ArcPointer
from std.os import abort
from std.sys import size_of
from ._values import _Values
from ._families import _Buffer, _Family, _FAMILY_COUNT, _family_of
from ..common._sizes import _checked_count
from ..float.context import FloatFormat
from ..complex.value import Complex
from ..float.value import Float
from ..complex._input import _ComplexArgument
from ._layout import _FlatLayout


def _default_format() -> FloatFormat:
    return FloatFormat(_validated=(128, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX))


struct _ArrayData(ImplicitlyCopyable):
    """One family-tagged buffer viewed through a runtime layout."""
    var family: Int
    var buffer: _Buffer
    var layout: _FlatLayout
    # Default formats describe empty Float and Complex arrays.
    var real_format: FloatFormat
    var imag_format: FloatFormat

    def __init__[T: ImplicitlyCopyable & Deinitable](
        out self, owner: ArcPointer[_Values[T]], var layout: _FlatLayout,
        real_format: FloatFormat = _default_format(), imag_format: FloatFormat = _default_format(),
    ):
        self.family = _family_of[T]()
        self.buffer = _Buffer(owner)
        self.layout = layout^
        self.real_format = real_format
        self.imag_format = imag_format

    def __init__(out self, *, family: Int, var buffer: _Buffer, var layout: _FlatLayout):
        """A record over a buffer whose tag the caller already knows."""
        self.family = family
        self.buffer = buffer^
        self.layout = layout^
        self.real_format = _default_format()
        self.imag_format = _default_format()

    def __init__(out self, *, copy: Self):
        self.family = copy.family
        self.buffer = copy.buffer
        self.layout = copy.layout
        self.real_format = copy.real_format
        self.imag_format = copy.imag_format

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.family = move.family
        self.buffer = move.buffer^
        self.layout = move.layout^
        self.real_format = move.real_format
        self.imag_format = move.imag_format

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    @staticmethod
    def contiguous[T: ImplicitlyCopyable & Deinitable](
        var values: List[T], var shape: List[Int],
    ) raises -> Self:
        var layout = _FlatLayout(shape^)
        _ = _checked_count(layout.size, max(1, size_of[T]()))
        if layout.size != len(values):
            raise Error("Batch shape does not match its values; use dimensions whose product equals the value count.")
        return Self(ArcPointer(_Values[T](values^)), layout^)

    def ndim(self) -> Int:
        return len(self.layout.shape)

    def shape(self) -> List[Int]:
        return self.layout.shape.copy()

    def size(self) -> Int:
        return self.layout.size

    def owner[T: ImplicitlyCopyable & Deinitable](self) -> ArcPointer[_Values[T]]:
        return self.buffer[ArcPointer[_Values[T]]]

    def read[T: ImplicitlyCopyable & Deinitable](self, index: Int) -> T:
        """Read a logical row-major element; callers validate the index."""
        return self.owner[T]()[][self.layout.position(index)]

    def with_layout(self, var layout: _FlatLayout) -> Self:
        var result = self
        result.layout = layout^
        return result^

    def gathered(self) -> Self:
        """A contiguous copy in row-major order, keeping formats."""
        comptime for index in range(_FAMILY_COUNT):
            if self.family == index:
                return self._gathered[_Family[index]]()
        abort("Unknown batch family tag")

    def _gathered[T: ImplicitlyCopyable & Deinitable](self) -> Self:
        var values = List[T](capacity=self.layout.size)
        ref source = self.owner[T]()[]
        var cursor = self.layout.cursor()
        for _ in range(self.layout.size):
            values.append(source[cursor.position])
            cursor.advance()
        var layout = _FlatLayout()
        try:
            layout = _FlatLayout(self.layout.shape.copy())
        except:
            # The shape came from a validated layout.
            pass
        return Self(ArcPointer(_Values[T](values^)), layout^, self.real_format, self.imag_format)

    def reshaped(self, var shape: List[Int]) raises -> Self:
        if self.layout.contiguous:
            return self.with_layout(self.layout.reshaped(shape^))
        var dense = self.gathered()
        return dense.with_layout(dense.layout.reshaped(shape^))

    def transposed(self, axes: List[Int]) raises -> Self:
        return self.with_layout(self.layout.transposed(axes))

    def at(self, axis: Int, index: Int) raises -> Self:
        return self.with_layout(self.layout.at(axis, index))

    def sliced(self, axis: Int, selection: StridedSlice) raises -> Self:
        return self.with_layout(self.layout.sliced(axis, selection))

    def broadcast_to(self, shape: List[Int]) raises -> Self:
        return self.with_layout(self.layout.broadcast_to(shape))

    def format_argument(self) -> _ComplexArgument:
        # Empty arrays select formats from metadata, never fabricated arithmetic.
        var result = _ComplexArgument(Int(0))
        if self.family == _family_of[Float]() or self.family == _family_of[Complex]():
            result.real.format = self.real_format
        if self.family == _family_of[Complex]():
            result.complex = True
            result.imag.format = self.imag_format
        return result


def _empty_buffer(family: Int, capacity: Int) -> _Buffer:
    comptime for index in range(_FAMILY_COUNT):
        if family == index:
            return _Buffer(ArcPointer(_Values[_Family[index]](List[_Family[index]](capacity=capacity))))
    abort("Unknown batch family tag")


struct _ArrayBuilder(Movable):
    """Append elements or whole arrays of one family, then finish with a shape."""
    var family: Int
    var buffer: _Buffer
    var real_format: FloatFormat
    var imag_format: FloatFormat

    def __init__(out self, family: Int, capacity: Int = 0):
        self.family = family
        self.real_format = _default_format()
        self.imag_format = _default_format()
        self.buffer = _empty_buffer(family, capacity)

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.family = move.family
        self.buffer = move.buffer^
        self.real_format = move.real_format
        self.imag_format = move.imag_format

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def _values[T: ImplicitlyCopyable & Deinitable](self) -> ArcPointer[_Values[T]]:
        """The builder's own buffer; it is not shared until finish()."""
        return self.buffer[ArcPointer[_Values[T]]]

    def reserve(mut self, count: Int) raises:
        comptime for index in range(_FAMILY_COUNT):
            if self.family == index:
                _ = _checked_count(count, size_of[_Family[index]]())
                self._values[_Family[index]]()[].list.reserve(count)

    def append[T: ImplicitlyCopyable & Deinitable](mut self, var value: T):
        # Reach the buffer in place: copying its ArcPointer would cost two
        # atomic reference-count updates per element.
        self.buffer[ArcPointer[_Values[T]]][].list.append(value^)

    def append_array(mut self, data: _ArrayData):
        comptime for index in range(_FAMILY_COUNT):
            if self.family == index:
                self._append_array[_Family[index]](data)

    def _append_array[T: ImplicitlyCopyable & Deinitable](mut self, data: _ArrayData):
        var owner = data.owner[T]()
        var target = self._values[T]()
        var cursor = data.layout.cursor()
        for _ in range(data.layout.size):
            target[].append(owner[][cursor.position])
            cursor.advance()

    def finish(deinit self, var shape: List[Int]) raises -> _ArrayData:
        var result = _ArrayData(
            family=self.family, buffer=self.buffer^, layout=_FlatLayout(shape^),
        )
        result.real_format = self.real_format
        result.imag_format = self.imag_format
        return result^
