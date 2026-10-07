"""Ball arithmetic as functions, one declaration per name.

Each function returns a ball that contains the exact result for every point
of its inputs. Operands may be balls or exact numbers (Integer, Rational,
Float, native integers and literals); exact numbers are not rounded first. A
`context=` sets the result precision; by default it is the largest midpoint
precision among the ball and Float operands. Where a function is undefined
somewhere in its input ball, the result is indeterminate.
"""

from ..integer.value import Integer
from ..float._arithmetic import _FloatArgument
from ..common._stable_hash import _StableHash
from .context import BallContext
from ._radius import _Radius
from .value import Ball, _BallArgument, _FINITE, _UNBOUNDED, _INDETERMINATE
from ._arithmetic import (
    _sum, _product, _quotient, _square, _power, _sqrt, _abs_ball, _scale2, _fma, _rounded_ball, _hull, _working,
)


def add(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left + right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands'.

    Returns:
        A ball containing every sum; unbounded if an operand is.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _sum(left, right, False, context)


def subtract(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left - right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands'.

    Returns:
        A ball containing every difference.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _sum(left, right, True, context)


def multiply(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left * right`.

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands'.

    Returns:
        A ball containing every product; an exact 0 times an unbounded ball
        is 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _product(left, right, context)


def divide(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left / right`.

    Args:
        left: The dividend.
        right: The divisor.
        context: The result precision; by default the operands'.

    Returns:
        A ball containing every quotient; indeterminate when the divisor
        contains 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _quotient(left, right, context)


def reciprocal(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `1 / value`.

    Args:
        value: The operand.
        context: The result precision; by default the operand's.

    Returns:
        A ball containing every reciprocal; indeterminate when the operand
        contains 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _quotient(_BallArgument(Integer(1)), value, context)


def abs(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `abs(value)`.

    Args:
        value: The operand.
        context: The result precision; by default the operand's.

    Returns:
        The operand or its negation away from 0, otherwise `[0, upper]`.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _abs_ball(value, context)


def square(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `value**2`.

    Args:
        value: The operand.
        context: The result precision; by default the operand's.

    Returns:
        A ball containing every square.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _square(value, context)


def sqrt(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the square root.

    Args:
        value: The operand.
        context: The result precision; by default the operand's.

    Returns:
        A ball containing every root; indeterminate when the operand reaches
        below 0. A lower end of exactly 0 is allowed.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _sqrt(value, context)


def pow_int(value: _BallArgument, exponent: Integer, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `value**exponent` for an integer exponent.

    Args:
        value: The base.
        exponent: Any integer; a negative one takes the reciprocal.
        context: The result precision; by default the operand's.

    Returns:
        A ball containing every power; `value**0` is exactly 1, and a
        negative exponent of a ball containing 0 is indeterminate.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _power(value, exponent, context)


def ldexp(value: _BallArgument, exponent: Integer, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `value * 2**exponent`, scaled exactly.

    Args:
        value: The operand.
        exponent: The power of two.
        context: The result precision; by default the operand's.

    Returns:
        The scaled ball.

    Raises:
        When the exponent does not fit an Int, or on an invalid precision.
    """
    return _scale2(value, exponent, context)


def fma(a: _BallArgument, b: _BallArgument, c: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `a * b + c`, with the midpoint rounded once.

    Args:
        a: The first factor.
        b: The second factor.
        c: The addend.
        context: The result precision; by default the operands'.

    Returns:
        A ball containing every result.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _fma(a, b, c, context)


def round_midpoint(value: Ball, precision: Integer) raises -> Ball:
    """The ball with its midpoint rounded to `precision` bits, the rounding
    error added to the radius.

    Args:
        value: The ball.
        precision: The new midpoint precision, at least 2.

    Returns:
        A ball containing `value`.

    Raises:
        When the precision is invalid.
    """
    if precision < 2:
        raise Error("Cannot round a ball's midpoint below 2 bits; choose at least 2.")
    return _rounded_ball(_BallArgument(value), Int(precision))


def trim(value: Ball) raises -> Ball:
    """The ball with its midpoint rounded to the bits its radius leaves
    meaningful: the relative accuracy plus 8, never more than it has.

    Args:
        value: The ball.

    Returns:
        A ball containing `value`, with a midpoint no longer than needed.

    Raises:
        Only on a checked size error.
    """
    if not value.is_finite() or value.is_exact():
        return value
    var bits = max(2, value.relative_accuracy_bits() + 8)
    if bits >= value.precision():
        return value
    return _rounded_ball(_BallArgument(value), bits)


def add_error(value: Ball, error: _FloatArgument) raises -> Ball:
    """The ball with `abs(error)` added to its radius.

    Args:
        value: The ball.
        error: An exact number or a Float bound.

    Returns:
        A wider ball; unbounded for an infinite error, indeterminate for NaN.

    Raises:
        Only on a checked size error.
    """
    if error.value.kind == 3:
        return Ball.indeterminate(value.precision())
    if not value.is_finite():
        return value
    if error.value.kind == 2:
        return Ball.unbounded(value.precision())
    return Ball(_midpoint=value._midpoint, _radius=value._radius.add(_Radius.upper_input(error.value)), _kind=_FINITE)


def stable_hash(value: Ball) raises -> UInt64:
    """A 64-bit hash of the representation, stable across processes and releases.

    The algorithm is APNH-64: tag 6, the kind (0 finite, 1 unbounded, 2
    indeterminate), then the midpoint and the radius, each encoded as for a
    Float, the radius as a 30-bit Float with the default exponent bounds.

    Args:
        value: The ball.

    Returns:
        The hash.

    Raises:
        Never in practice.
    """
    var hash = _StableHash(6)
    value._hash_into(hash)
    return hash.finish()


def _ball_extreme(left: _BallArgument, right: _BallArgument, maximum: Bool, context: Optional[BallContext]) raises -> Ball:
    """The ball of `max` or `min` over all points: an operand that lies wholly on the
    chosen side of the other, else the least ball over the chosen lower ends
    and the chosen upper ends."""
    var w = _working(context, left.precision, right.precision)
    if left.kind == _INDETERMINATE or right.kind == _INDETERMINATE:
        return Ball.indeterminate(w)
    if left.kind == _UNBOUNDED or right.kind == _UNBOUNDED:
        return Ball.unbounded(w)
    var x = left.ball()
    var y = right.ball()
    var xl = x._exact_lower()
    var xh = x._exact_upper()
    var yl = y._exact_lower()
    var yh = y._exact_upper()
    if maximum:
        if xl >= yh:
            return _rounded_ball(left, w)
        if yl >= xh:
            return _rounded_ball(right, w)
        return _hull(xl if xl >= yl else yl, xh if xh >= yh else yh, w)
    if xh <= yl:
        return _rounded_ball(left, w)
    if yh <= xl:
        return _rounded_ball(right, w)
    return _hull(xl if xl <= yl else yl, xh if xh <= yh else yh, w)


def maximum(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `max(s, t)` over all points of the two balls (numpy's
    `maximum`).

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands' largest.

    Returns:
        The operand lying wholly above the other, or the least ball over the
        larger lower end and the larger upper end; indeterminate when either
        operand is.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_extreme(left, right, True, context)


def minimum(left: _BallArgument, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `min(s, t)` over all points of the two balls (numpy's
    `minimum`).

    Args:
        left: The first operand.
        right: The second operand.
        context: The result precision; by default the operands' largest.

    Returns:
        The operand lying wholly below the other, or the least ball over the
        smaller lower end and the smaller upper end; indeterminate when
        either operand is.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_extreme(left, right, False, context)


def clip(value: _BallArgument, a_min: _BallArgument, a_max: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `value` limited to `[a_min, a_max]`, `minimum(maximum(value,
    a_min), a_max)` (numpy's `clip`).

    Args:
        value: The operand.
        a_min: The lower limit.
        a_max: The upper limit.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing the limited value at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_extreme(_BallArgument(_ball_extreme(value, a_min, True, context)), a_max, False, context)
