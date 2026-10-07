"""Complex ball arithmetic as functions, one declaration per name.

Each function returns a complex ball containing the result for every point of
its input rectangles. A `context=` sets the precision of the result's parts; by
default it is the largest midpoint precision of the operands.
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._arithmetic import _float_operation
from ..float._functions import _sqrt_float
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._arithmetic import _hull_from, _norm_bounds, _working, _negate, _rounded_ball
from ..ball._functions import _ball_atan2, _atan2_of
from ..ball._kernels import _constant_kernel, _PI
from ..common._stable_hash import _StableHash
from .value import ComplexBall, _add, _sub, _mul, _div


def _bits(value: ComplexBall, context: Optional[BallContext]) -> Int:
    if context and context.value().precision():
        return context.value().precision().value()
    return value.precision()


def add(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `left + right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing every sum.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _add(left, right, context)


def subtract(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `left - right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing every difference.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _sub(left, right, context)


def multiply(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `left * right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing every product.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _mul(left, right, context)


def divide(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `left / right`.

    Args:
        left: The dividend.
        right: The divisor.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing every quotient; indeterminate when the divisor's
        rectangle contains 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _div(left, right, context)


def conjugate(value: ComplexBall) raises -> ComplexBall:
    """The ball of the conjugates, `re - im i`.

    Args:
        value: The ball.

    Returns:
        The conjugate rectangle.

    Raises:
        Only on a checked size error.
    """
    return ComplexBall(_real=value.real(), _imag=_negate(value.imag()))


def abs(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the magnitudes.

    Computed from bounds of the parts' magnitudes, `hypot` of the lower bounds
    rounded down and of the upper bounds rounded up, never from a square root
    of a sum that may reach below 0.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every magnitude.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    if value.is_indeterminate():
        return Ball.indeterminate(w)
    if not value.is_finite():
        return Ball.unbounded(w)
    var norm = _norm_bounds(value.real(), value.imag(), w + 8)
    var down = ArithmeticContext(format=FloatFormat(w + 8), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(w + 8), rounding=RoundingMode.toward_positive)
    var low = Float(_rounded=_sqrt_float(norm[0], down))
    var high = Float(_rounded=_sqrt_float(norm[1], up))
    return _hull_from(low, high, w)


def angle(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the arguments, in `[-pi, pi]`.

    A point of the negative real axis takes `pi`, its counter-clockwise
    continuous value; a rectangle that crosses the axis covers both sides, so
    its result is `[0 +/- pi]`; one that contains 0 is indeterminate.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every argument.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var y = value.imag()
    var x = value.real()
    if y.is_exact() and y._midpoint.is_zero() and x.certainly_negative():
        return _rounded_ball(_BallArgument(_constant_kernel(_PI, w)), w)
    if y.is_finite() and x.is_finite():
        return _atan2_of(value._imag, value._real, w)
    return _ball_atan2(_BallArgument(y), _BallArgument(x), Optional[BallContext](BallContext(w)))


def pow_int(value: ComplexBall, exponent: Integer, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `value**exponent` for an integer exponent.

    Args:
        value: The base.
        exponent: Any integer; a negative one takes the reciprocal.
        context: The result precision; by default the base's.

    Returns:
        A ball containing every power; `value**0` is exactly 1, and a
        negative power of a rectangle containing 0 is indeterminate.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var c = Optional[BallContext](BallContext(w + 8))
    var result = ComplexBall(1)
    var n = -exponent if exponent.sign() < 0 else exponent
    for bit in range(n.magnitude_bit_length() - 1, -1, -1):
        result = _mul(result, result, c)
        if (n._word(bit >> 5) >> UInt32(bit & 31)) & 1:
            result = _mul(result, value, c)
    if exponent.sign() < 0:
        result = _div(ComplexBall(1), result, c)
    return ComplexBall(_real=_rounded_ball(_BallArgument(result.real()), w), _imag=_rounded_ball(_BallArgument(result.imag()), w))


def stable_hash(value: ComplexBall) raises -> UInt64:
    """A 64-bit hash of the representation, stable across processes and releases.

    The algorithm is APNH-64: tag 7, then the real and imaginary balls, each
    encoded as for a Ball without its tag.

    Args:
        value: The ball.

    Returns:
        The hash.

    Raises:
        Never in practice.
    """
    var hash = _StableHash(7)
    value._hash_into(hash)
    return hash.finish()


def real(value: ComplexBall) raises -> Ball:
    """The ball of the real parts (numpy's `real`).

    Args:
        value: The complex ball.

    Returns:
        The real ball.

    Raises:
        Only on a checked size error.
    """
    return value.real()


def imag(value: ComplexBall) raises -> Ball:
    """The ball of the imaginary parts (numpy's `imag`).

    Args:
        value: The complex ball.

    Returns:
        The imaginary ball.

    Raises:
        Only on a checked size error.
    """
    return value.imag()


def reciprocal(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `1 / value` (numpy's `reciprocal`).

    Args:
        value: The complex ball.
        context: The result precision; by default the ball's.

    Returns:
        A complex ball containing every reciprocal; indeterminate when the
        operand contains 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var bits = max(value.real().precision(), value.imag().precision())
    var one = ComplexBall(_real=Ball(Integer(1), precision=bits), _imag=Ball(Integer(0), precision=bits))
    return _div(one, value, context)
