"""Float batch inputs: runs, complex components and broadcast scalars."""

from std.iter import Iterator
from .value import Float
from .context import FloatFormat
from ..integer.value import Integer
from ..rational.value import Rational
from ..complex.value import Complex
from ..batch._iterator import _iterator_tensor
from ..batch._slices import _Selection
from ..batch._tensor import _Tensor
from ..batch._values import _borrow_values


struct _FloatInput(ImplicitlyCopyable):
    """A Float batch run, a complex component run or one broadcast scalar."""
    var scalar: Optional[Float]
    var selection: _Selection
    var native: Optional[_Tensor[Float]]
    var complex_values: Optional[_Tensor[Complex]]
    var imaginary: Bool
    # The batch's default format; a scalar carries its own.
    var format: FloatFormat

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.scalar = move.scalar^
        self.selection = move.selection^
        self.native = move.native^
        self.complex_values = move.complex_values^
        self.imaginary = move.imaginary
        self.format = move.format

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def __init__(
        out self, scalar: Optional[Float], selection: _Selection,
        native: Optional[_Tensor[Float]] = None,
        complex_values: Optional[_Tensor[Complex]] = None, imaginary: Bool = False,
        format: FloatFormat = FloatFormat(_validated=(128, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX)),
    ):
        self.scalar = scalar
        self.selection = selection
        self.native = native
        self.complex_values = complex_values
        self.imaginary = imaginary
        self.format = format

    def is_native(self) -> Bool:
        return Bool(self.native) or Bool(self.complex_values)

    def default_format(self) -> FloatFormat:
        if self.scalar and not self.is_native():
            return self.scalar.value().format()
        return self.format

    @always_inline
    def _borrow(self, value: Float) -> ref[self] Float:
        # Only for values retained by self. Mojo 1.1 cannot widen the origin
        # of an Optional payload or a Complex field to its enclosing input.
        return Pointer(to=value).as_imm().unsafe_origin_cast[origin_of(self)]()[]

    def element(self, index: Int) -> ref[self] Float:
        """Borrow the stored Float, without copying its significand."""
        if self.complex_values:
            ref value = self.complex_values.value()._read(self.selection.index(index))
            # Return the field directly; a conditional expression copies it.
            if self.imaginary:
                return self._borrow(value._imag)
            return self._borrow(value._real)
        if self.native:
            return self._borrow(self.native.value()._read(self.selection.index(index)))
        return self._borrow(self.scalar.value())

    def run_step(self) -> Optional[Int]:
        """The storage step between consecutive elements when they form one
        arithmetic run of a Float tensor; None otherwise."""
        if not self.native or self.complex_values:
            return None
        # A tensor's layout is one strided run (see _Layout.position).
        return self.native.value()._layout.strides[0] * self.selection.step

    def run_start(self) -> Pointer[Float, origin_of(self)]:
        """Borrow element 0 of a nonempty native run; run_step must be present."""
        ref tensor = self.native.value()
        return _borrow_values[origin_of(self)](tensor._owner[].list).unsafe_ptr().unsafe_offset(
            tensor._layout.position(self.selection.index(0))
        )

    def value(self, index: Int) -> Float:
        return self.element(index)

    def format_at(self, index: Int) -> FloatFormat:
        return self.element(index).format()


def _float_iterator_element[I: Iterator](var value: I.Element) raises -> Float:
    comptime assert conforms_to(I.Element, ImplicitlyCopyable), (
        "Float batches require numeric elements; convert each value to Float"
        " explicitly."
    )
    comptime assert (
        I.Element == Float
        or I.Element == Integer
        or I.Element == Rational
        or I.Element == Int
        or I.Element == Int8
        or I.Element == UInt8
        or I.Element == Int16
        or I.Element == UInt16
        or I.Element == Int32
        or I.Element == UInt32
        or I.Element == Int64
        or I.Element == UInt64
        or I.Element == Float16
        or I.Element == BFloat16
        or I.Element == Float32
        or I.Element == Float64
    ), (
        "Float batches require supported numeric elements; convert text or"
        " other values to Float explicitly."
    )
    comptime if I.Element == Float:
        return rebind_var[Float](value^)
    elif I.Element == Integer:
        return Float(rebind_var[Integer](value^))
    elif I.Element == Rational:
        return Float(rebind_var[Rational](value^))
    elif I.Element == Int:
        return Float(rebind_var[Int](value^))
    elif I.Element == Int8:
        return Float(rebind_var[Int8](value^))
    elif I.Element == UInt8:
        return Float(rebind_var[UInt8](value^))
    elif I.Element == Int16:
        return Float(rebind_var[Int16](value^))
    elif I.Element == UInt16:
        return Float(rebind_var[UInt16](value^))
    elif I.Element == Int32:
        return Float(rebind_var[Int32](value^))
    elif I.Element == UInt32:
        return Float(rebind_var[UInt32](value^))
    elif I.Element == Int64:
        return Float(rebind_var[Int64](value^))
    elif I.Element == UInt64:
        return Float(rebind_var[UInt64](value^))
    elif I.Element == Float16:
        return Float(rebind_var[Float16](value^))
    elif I.Element == BFloat16:
        return Float(rebind_var[BFloat16](value^))
    elif I.Element == Float32:
        return Float(rebind_var[Float32](value^))
    else:
        return Float(rebind_var[Float64](value^))


def _native_float_iterator[I: Iterator](var cursor: I) raises -> _Tensor[Float]:
    return _iterator_tensor[Float, I, _float_iterator_element[I]](cursor^)
