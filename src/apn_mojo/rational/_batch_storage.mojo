"""Native Rational owners and adapters for logical component access."""

from std.iter import Iterator
from ..integer.value import Integer
from .value import Rational
from ..batch._iterator import _iterator_tensor
from ..batch._slices import _Selection
from ..batch._tensor import _Tensor



struct _RationalInput(ImplicitlyCopyable):
    """A Rational run, an Integer run read as fractions, or one scalar."""
    var scalar: Rational
    var selection: _Selection
    var native_integer: Optional[_Tensor[Integer]]
    var native: Optional[_Tensor[Rational]]

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.scalar = move.scalar^
        self.selection = move.selection^
        self.native_integer = move.native_integer^
        self.native = move.native^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def __init__(
        out self, scalar: Rational, selection: _Selection,
        native_integer: Optional[_Tensor[Integer]] = None,
        native: Optional[_Tensor[Rational]] = None,
    ):
        self.scalar = scalar
        self.selection = selection
        self.native_integer = native_integer
        self.native = native

    def value(self, index: Int) -> Rational:
        if self.native:
            return self.native.value()._read(self.selection.index(index))
        if self.native_integer:
            return Rational(self.native_integer.value()._read(self.selection.index(index)))
        return self.scalar

    def component(self, denominator: Bool, index: Int) -> Integer:
        if self.native:
            ref value = self.native.value()._read(self.selection.index(index))
            return value.denominator() if denominator else value.numerator()
        if self.native_integer:
            if denominator:
                return Integer(1)
            return self.native_integer.value()._read(self.selection.index(index))
        return (
            self.scalar.denominator() if denominator else self.scalar.numerator()
        )

    def is_zero(self, index: Int) -> Bool:
        if self.native:
            return not self.native.value()._read(self.selection.index(index))
        if self.native_integer:
            return not self.native_integer.value()._read(self.selection.index(index))
        return not self.scalar

    def sign(self, index: Int) -> Int:
        if self.native:
            return self.native.value()._read(self.selection.index(index)).sign()
        if self.native_integer:
            return self.native_integer.value()._read(self.selection.index(index)).sign()
        return self.scalar.sign()


def _rational_transfer_checkpoint(index: Int, fail: Int) raises:
    if index == fail:
        raise Error(
            String(
                "Injected Rational transfer failure at element ",
                index,
                "; the destination is unchanged. Retry the assignment.",
            )
        )




def _rational_iterator_element[I: Iterator](var value: I.Element) raises -> Rational:
    comptime assert conforms_to(
        I.Element, ImplicitlyCopyable
    ), "Rational batches require exact numeric elements"
    comptime assert (
        I.Element == Rational
        or I.Element == Integer
        or I.Element == Int
        or I.Element == Int8
        or I.Element == UInt8
        or I.Element == Int16
        or I.Element == UInt16
        or I.Element == Int32
        or I.Element == UInt32
        or I.Element == Int64
        or I.Element == UInt64
    ), (
        "Rational batches require Rational, Integer or native integer elements;"
        " convert text or floats explicitly."
    )
    comptime if I.Element == Rational:
        return rebind_var[Rational](value^)
    elif I.Element == Integer:
        return Rational(rebind_var[Integer](value^))
    elif I.Element == Int:
        return Rational(rebind_var[Int](value^))
    elif I.Element == Int8:
        return Rational(rebind_var[Int8](value^))
    elif I.Element == UInt8:
        return Rational(rebind_var[UInt8](value^))
    elif I.Element == Int16:
        return Rational(rebind_var[Int16](value^))
    elif I.Element == UInt16:
        return Rational(rebind_var[UInt16](value^))
    elif I.Element == Int32:
        return Rational(rebind_var[Int32](value^))
    elif I.Element == UInt32:
        return Rational(rebind_var[UInt32](value^))
    elif I.Element == Int64:
        return Rational(rebind_var[Int64](value^))
    else:
        return Rational(rebind_var[UInt64](value^))


def _native_rational_iterator[I: Iterator](var cursor: I, fail: Int = -1) raises -> _Tensor[Rational]:
    return _iterator_tensor[Rational, I, _rational_iterator_element[I], _rational_transfer_checkpoint](cursor^, fail)
