"""Elementary functions of balls, one declaration per name.

Each function returns a ball that contains the function's value at every
point of its input ball, computed from the function's point kernel: at the two
exact ends of the ball for a monotone function, or at the midpoint with a bound
of the derivative. A function undefined somewhere in its input ball gives an
indeterminate ball. A `context=` sets the result precision, by default the
operand's, and the budget of argument reduction, past which `sin` and `cos`
give `[0 +/- 1]` and `tan` an indeterminate ball: a ball function never raises
for its budget.
"""

from ..integer.value import Integer
from .context import BallContext
from .value import Ball, _BallArgument
from ._kernels import (
    _EXP, _EXPM1, _EXP2, _LOG, _LOG1P, _LOG2, _LOG10, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS,
    _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH, _ROOTN,
)
from ._functions import _ball_function, _ball_atan2, _ball_pow, _sin_cos_pair
from ._arithmetic import _working
from .value import _FINITE


def exp(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the exponential.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing exp of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_EXP, value, 0, context)


def expm1(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `exp(x) - 1`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing `exp(x) - 1` of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_EXPM1, value, 0, context)


def exp2(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `2**x`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing `2**x` of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_EXP2, value, 0, context)


def log(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the natural logarithm.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the logarithm of every point; indeterminate unless the ball is above 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_LOG, value, 0, context)


def log1p(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log(1 + x)`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing `log(1 + x)` of every point; indeterminate unless the ball is above -1.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_LOG1P, value, 0, context)


def log2(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the base-2 logarithm.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the logarithm of every point; indeterminate unless the ball is above 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_LOG2, value, 0, context)


def log10(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the base-10 logarithm.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the logarithm of every point; indeterminate unless the ball is above 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_LOG10, value, 0, context)


def sin(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the sine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the sine of every point, within [-1, 1]; `[0 +/- 1]` past the budget.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_SIN, value, 0, context)


def cos(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the cosine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the cosine of every point, within [-1, 1]; `[0 +/- 1]` past the budget.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_COS, value, 0, context)


def tan(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the tangent.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the tangent of every point; indeterminate when the ball may contain a pole, or past the budget.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_TAN, value, 0, context)


def atan(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the arctangent.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the arctangent of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ATAN, value, 0, context)


def asin(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the arcsine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the arcsine of every point; indeterminate unless the ball is within [-1, 1].

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ASIN, value, 0, context)


def acos(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the arccosine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the arccosine of every point; indeterminate unless the ball is within [-1, 1].

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ACOS, value, 0, context)


def sinh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the hyperbolic sine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing sinh of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_SINH, value, 0, context)


def cosh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the hyperbolic cosine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing cosh of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_COSH, value, 0, context)


def tanh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the hyperbolic tangent.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing tanh of every point, within [-1, 1].

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_TANH, value, 0, context)


def asinh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the inverse hyperbolic sine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing asinh of every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ASINH, value, 0, context)


def acosh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the inverse hyperbolic cosine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing acosh of every point; indeterminate unless the ball is at least 1.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ACOSH, value, 0, context)


def atanh(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the inverse hyperbolic tangent.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing atanh of every point; indeterminate unless the ball is within (-1, 1).

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_function(_ATANH, value, 0, context)


def sin_cos(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of the sine and the cosine.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        `(sin(value), cos(value))`, each as `sin` and `cos` give it.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    # A Float midpoint, as `sin` and `cos` would take to their medium kernel:
    # one call for both. Exact Integers and Rationals take the two functions.
    ref a = value.midpoint.value
    if (
        value.kind == _FINITE and a.kind == 1 and a.denominator._is_one() and value.midpoint.format
        and a.numerator.magnitude_bit_length() == value.midpoint.format.value().precision()
    ):
        return _sin_cos_pair(value.ball(), _working(context, value.precision))
    return (_ball_function(_SIN, value, 0, context), _ball_function(_COS, value, 0, context))


def atan2(y: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the angle of `x + i y`, in `[-pi, pi]`.

    Args:
        y: The ordinate.
        x: The abscissa.
        context: The result precision; by default the larger operand
            precision.

    Returns:
        A ball containing the angle of every point; `[0 +/- pi]` when the
        rectangle meets the negative real axis, where the angle jumps, and
        indeterminate when it contains the origin.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_atan2(y, x, context)


def pow(base: _BallArgument, exponent: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `base**exponent`.

    An exact integer exponent gives an integer power; otherwise the base must
    be certainly above 0, and the result is `exp(exponent log base)`.

    Args:
        base: The base.
        exponent: The exponent.
        context: The result precision; by default the larger operand
            precision.

    Returns:
        A ball containing every power; indeterminate when the base may be 0
        or negative for a non-integer exponent (an exact 0 to a positive
        power gives 0).

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_pow(base, exponent, context)


def rootn(value: _BallArgument, n: Int, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the real `n`-th root, `n >= 1`.

    Args:
        value: The operand.
        n: The degree, at least 1; an odd root keeps the sign.
        context: The result precision; by default the operand's precision.

    Returns:
        A ball containing every root; indeterminate when an even root meets a
        negative point.

    Raises:
        When `n` is below 1, or on an invalid precision.
    """
    if n < 1:
        raise Error(String("Cannot take the root of degree ", n, "; rootn needs n >= 1. Use pow for other exponents."))
    return _ball_function(_ROOTN, value, n, context)
