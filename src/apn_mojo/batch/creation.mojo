"""NumPy-style batch constructors.

The element type is a parameter, as `dtype` is in NumPy, so exactness is
explicit: `zeros[Integer]([2, 3])`, `ones[Float](4, context=c)`,
`arange[Rational](0, 1, Rational(1, 10))`. Ranges are computed exactly; a
Float element is the exact value rounded once.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..ball.value import Ball
from ..complex_ball.value import ComplexBall
from ..float.context import ArithmeticContext
from ..complex.context import _ComplexContextArgument
from ..common._traits import _BatchElement
from ..rational.math import ceil as _rational_ceil
from ._layout import _FlatLayout
from .value import Batch
from .functions import _NoContext


struct _Dimensions(Copyable, Movable):
    """A shape: one length, or a list of lengths."""

    var values: List[Int]

    @implicit
    def __init__(out self, length: Int):
        self.values = [length]

    @implicit
    def __init__(out self, length: IntLiteral):
        self.values = [Int(length)]

    @implicit
    def __init__(out self, var values: List[Int]):
        self.values = values^

    def __init__(out self, var *dimensions: Int, __list_literal__: NoneType):
        self.values = List[Int](capacity=len(dimensions))
        for i in range(len(dimensions)):
            self.values.append(dimensions[i])


# The format of a constructed Float or Complex; the other families are exact.
comptime _FormatContext[T: ImplicitlyCopyable & Deinitable] = (
    Optional[ArithmeticContext] if T == Float else _ComplexContextArgument if T == Complex else _NoContext
)


def _whole[T: ImplicitlyCopyable & Deinitable](value: Int, context: _FormatContext[T]) raises -> T:
    """A small integer in the family T, in the context's format."""
    comptime if T == Integer:
        return rebind_var[T](Integer(value))
    elif T == Rational:
        return rebind_var[T](Rational(value))
    elif T == Float:
        return rebind_var[T](Float(value, context=rebind[Optional[ArithmeticContext]](context)))
    elif T == Complex:
        return rebind_var[T](Complex(Integer(value), Integer(0), context=rebind[_ComplexContextArgument](context)))
    elif T == Ball:
        return rebind_var[T](Ball(Integer(value)))
    else:
        comptime assert T == ComplexBall, "Constructors take Integer, Rational, Float, Complex or ball elements."
        return rebind_var[T](ComplexBall(Ball(Integer(value))))


def _exact[T: ImplicitlyCopyable & Deinitable](value: Rational, context: _FormatContext[T]) raises -> T:
    """An exact value in the family T: Integers must be whole; Floats round once."""
    comptime if T == Integer:
        return rebind_var[T](value.to_integer_exact())
    elif T == Rational:
        return rebind_var[T](value)
    else:
        comptime assert T == Float, "Ranges take Integer, Rational or Float elements."
        return rebind_var[T](Float(value, context=rebind[Optional[ArithmeticContext]](context)))


def _filled[T: ImplicitlyCopyable & Deinitable](shape: _Dimensions, value: T) raises -> Batch[T]:
    var count = _FlatLayout(shape.values.copy()).size
    return Batch[T](List[T](length=count, fill=value), shape=shape.values)


def zeros[T: ImplicitlyCopyable & Deinitable](
    shape: _Dimensions, *, context: _FormatContext[T] = _FormatContext[T](),
) raises -> Batch[T]:
    """A batch of zeros, like `numpy.zeros`.

    Parameters:
        T: The element type: Integer, Rational, Float, Complex, Ball or ComplexBall.

    Args:
        shape: A length, or a list of dimensions.
        context: The Float or Complex format; 128 bits by default.

    Returns:
        A batch of the shape, every element zero.

    Raises:
        On a negative dimension or an invalid context.
    """
    return _filled(shape, _whole[T](0, context))


def ones[T: ImplicitlyCopyable & Deinitable](
    shape: _Dimensions, *, context: _FormatContext[T] = _FormatContext[T](),
) raises -> Batch[T]:
    """A batch of ones, like `numpy.ones`.

    Parameters:
        T: The element type: Integer, Rational, Float, Complex, Ball or ComplexBall.

    Args:
        shape: A length, or a list of dimensions.
        context: The Float or Complex format; 128 bits by default.

    Returns:
        A batch of the shape, every element one.

    Raises:
        On a negative dimension or an invalid context.
    """
    return _filled(shape, _whole[T](1, context))


def full[T: ImplicitlyCopyable & Deinitable](shape: _Dimensions, value: T) raises -> Batch[T] where conforms_to(
    T, _BatchElement
):
    """A batch holding `value` everywhere, like `numpy.full`; the family and
    format are the value's.

    Parameters:
        T: The element type, inferred from `value`.

    Args:
        shape: A length, or a list of dimensions.
        value: The element.

    Returns:
        A batch of the shape.

    Raises:
        On a negative dimension.
    """
    return _filled(shape, value)


def full(shape: _Dimensions, value: IntLiteral) raises -> Batch[Integer]:
    """A batch of Integers holding an integer literal of any width."""
    return _filled(shape, Integer(value))


def arange[T: ImplicitlyCopyable & Deinitable](
    stop: Rational, *, context: _FormatContext[T] = _FormatContext[T](),
) raises -> Batch[T]:
    """`0, 1, ...` below `stop`, like `numpy.arange(stop)`."""
    return arange[T](Rational(0), stop, Rational(1), context=context)


def arange[T: ImplicitlyCopyable & Deinitable](
    start: Rational, stop: Rational, step: Rational = Rational(1), *,
    context: _FormatContext[T] = _FormatContext[T](),
) raises -> Batch[T]:
    """`start, start + step, ...` up to but excluding `stop`, like `numpy.arange`.

    Each element is computed exactly: Integer elements must be whole, and a
    Float element is `start + k * step` rounded once, so no rounding error
    accumulates.

    Parameters:
        T: The element type: Integer, Rational or Float.

    Args:
        start: The first value.
        stop: The bound, excluded.
        step: The step, positive or negative.
        context: The Float format; 128 bits by default.

    Returns:
        A vector of `ceil((stop - start) / step)` elements, or none.

    Raises:
        For a zero step, or a value that is not whole for Integer elements.
    """
    if step == Rational(0):
        raise Error("arange needs a nonzero step.")
    var span = (stop - start) / step
    var count = Int(_rational_ceil(span)) if span > Rational(0) else 0
    var values = List[T](capacity=count)
    for k in range(count):
        values.append(_exact[T](start + step * Rational(k), context))
    return Batch[T](values, shape=[count])


def linspace[T: ImplicitlyCopyable & Deinitable](
    start: Rational, stop: Rational, num: Int = 50, *, endpoint: Bool = True,
    context: _FormatContext[T] = _FormatContext[T](),
) raises -> Batch[T]:
    """`num` evenly spaced values from `start` to `stop`, like `numpy.linspace`.

    Each element is computed exactly and the last is `stop` itself, as for
    `arange`.

    Parameters:
        T: The element type: Integer, Rational or Float.

    Args:
        start: The first value.
        stop: The last value, or the bound when `endpoint` is False.
        num: The number of values.
        endpoint: Whether `stop` is included.
        context: The Float format; 128 bits by default.

    Returns:
        A vector of `num` elements.

    Raises:
        For a negative count, or a value that is not whole for Integer elements.
    """
    if num < 0:
        raise Error("linspace needs a nonnegative number of values.")
    var intervals = num - 1 if endpoint else num
    var values = List[T](capacity=num)
    for k in range(num):
        var value = start if intervals == 0 else start + (stop - start) * Rational(k) / Rational(intervals)
        values.append(_exact[T](value, context))
    return Batch[T](values, shape=[num])
