"""Balls as sets: three-valued comparison, containment, hulls, splitting,
certain floors, and the simplest rational inside a ball.

Every decision uses the exact ends `midpoint -/+ radius`. An indeterminate
ball may be anything, so it contains and overlaps everything; an unbounded
ball contains every real number.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float._arithmetic import _FloatArgument, _float_operation
from ..float._input import _FloatInput
from .context import BallContext
from ._radius import _Radius
from .value import Ball, BallOrder, _order, _rational_of, _FINITE, _UNBOUNDED, _INDETERMINATE
from ..float._rounding import _finish_round
from ._certified import _certify
from ._arithmetic import _hull


def compare(a: Ball, b: Ball) raises -> BallOrder:
    """How two balls compare, decided on their exact ends.

    Args:
        a: The first ball.
        b: The second ball.

    Returns:
        `less` or `greater` when every pair of points agrees, `equal` for two
        exact equal balls, `overlap` otherwise (including any unbounded ball),
        and `undefined` when either is indeterminate.

    Raises:
        Only on a checked size error.
    """
    return _order(a, b)


def contains(ball: Ball, value: _FloatArgument) raises -> Bool:
    """Whether an exact number lies in the ball, decided exactly.

    Args:
        ball: The ball.
        value: An Integer, Rational, Float or native number.

    Returns:
        True when it lies within the ends; always for an indeterminate ball,
        and for any finite number in an unbounded ball.

    Raises:
        Only on a checked size error.
    """
    if ball.is_indeterminate():
        return True
    if value.value.kind >= 2:
        return False
    if ball.is_unbounded():
        return True
    var q = _rational_of(value.value)
    return ball.lower_rational() <= q and q <= ball.upper_rational()


def contains_zero(ball: Ball) raises -> Bool:
    """Whether 0 lies in the ball.

    Args:
        ball: The ball.

    Returns:
        True unless the ball is certainly nonzero.

    Raises:
        Only on a checked size error.
    """
    if ball.is_finite():
        # Exactly when |m| <= r, compared on the midpoint's significand.
        return ball._midpoint.is_zero() or ball._radius.covers_float(ball._midpoint)
    return contains(ball, Integer(0))


def contains_ball(outer: Ball, inner: Ball) raises -> Bool:
    """Whether `inner` lies within `outer`.

    Args:
        outer: The containing ball.
        inner: The contained ball.

    Returns:
        True when every point of `inner` lies in `outer`; always for an
        indeterminate `outer`, never for an indeterminate `inner` otherwise.

    Raises:
        Only on a checked size error.
    """
    if outer.is_indeterminate():
        return True
    if inner.is_indeterminate():
        return False
    if outer.is_unbounded():
        return True
    if inner.is_unbounded():
        return False
    return outer._exact_lower() <= inner._exact_lower() and inner._exact_upper() <= outer._exact_upper()


def overlaps(a: Ball, b: Ball) raises -> Bool:
    """Whether the balls share a point.

    Args:
        a: The first ball.
        b: The second ball.

    Returns:
        True when the closed intervals meet; always when either ball is not
        finite.

    Raises:
        Only on a checked size error.
    """
    if not a.is_finite() or not b.is_finite():
        return True
    return a._exact_lower() <= b._exact_upper() and b._exact_lower() <= a._exact_upper()


def contains_integer(ball: Ball) raises -> Bool:
    """Whether some integer lies in the ball.

    Args:
        ball: The ball.

    Returns:
        True when the ends enclose an integer; always unless finite.

    Raises:
        Only on a checked size error.
    """
    if not ball.is_finite():
        return True
    return ball._exact_lower().ceil() <= ball._exact_upper().floor()


def _bits(a: Ball, b: Ball, context: Optional[BallContext]) -> Int:
    if context and context.value().precision():
        return context.value().precision().value()
    return max(a.precision(), b.precision())


def union(a: Ball, b: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The least ball at the precision that contains both.

    Args:
        a: The first ball.
        b: The second ball.
        context: The result precision; by default the larger of the two.

    Returns:
        The hull; indeterminate or unbounded if either ball is.

    Raises:
        Only on a checked size error.
    """
    var bits = _bits(a, b, context)
    if a.is_indeterminate() or b.is_indeterminate():
        return Ball.indeterminate(bits)
    if a.is_unbounded() or b.is_unbounded():
        return Ball.unbounded(bits)
    var al = a._exact_lower()
    var bl = b._exact_lower()
    var au = a._exact_upper()
    var bu = b._exact_upper()
    return Ball.from_interval(al if al <= bl else bl, au if au >= bu else bu, precision=bits)


def intersection(a: Ball, b: Ball, *, context: Optional[BallContext] = None) raises -> Optional[Ball]:
    """The least ball at the precision that contains the common points.

    Args:
        a: The first ball.
        b: The second ball.
        context: The result precision; by default the larger of the two.

    Returns:
        None when the balls are disjoint, decided exactly; the other ball when
        one is unbounded; indeterminate when either is.

    Raises:
        Only on a checked size error.
    """
    var bits = _bits(a, b, context)
    if a.is_indeterminate() or b.is_indeterminate():
        return Ball.indeterminate(bits)
    if a.is_unbounded():
        return b
    if b.is_unbounded():
        return a
    var al = a._exact_lower()
    var bl = b._exact_lower()
    var au = a._exact_upper()
    var bu = b._exact_upper()
    var low = al if al >= bl else bl
    var high = au if au <= bu else bu
    if high < low:
        return None
    return Ball.from_interval(low, high, precision=bits)


def split(ball: Ball) raises -> Tuple[Ball, Ball]:
    """The two halves `[m - r/2 +/- r/2]` and `[m + r/2 +/- r/2]`.

    The halves are exact: their midpoints take the precision they need, so
    their union is the ball itself.

    Args:
        ball: A finite ball.

    Returns:
        The lower and the upper half.

    Raises:
        Unless the ball is finite.
    """
    if not ball.is_finite():
        raise Error("Cannot split an unbounded or indeterminate ball; check is_finite() first.")
    var half = ball._radius.scale2(-1)
    var step = half.to_float()
    var exact = _exact_context()
    var halves = List[Ball]()
    for operation in [1, 0]:
        var center = Float(_rounded=_float_operation(ball._midpoint, step, operation, exact))
        var bits = max(ball.precision(), center.precision())
        var midpoint = Float(_rounded=_float_operation(center, Integer(0), 0, ArithmeticContext(format=FloatFormat(bits))))
        halves.append(Ball(_midpoint=midpoint, _radius=half, _kind=_FINITE))
    return (halves[0], halves[1])


def to_float_if_certain(ball: Ball, *, context: Optional[ArithmeticContext] = None) raises -> Optional[Float]:
    """The Float that every point of the ball rounds to, when there is one.

    Both ends of the ball are rounded, once each, to the context's format with
    its rounding mode. When they give the same Float with the same status,
    every point of the ball rounds to that Float, the true value it encloses
    among them. Correctly rounded functions repeat this step at higher
    precision until it succeeds.

    Args:
        ball: The ball.
        context: The format, rounding mode and traps; by default the
            midpoint's format, rounded to nearest-even.

    Returns:
        The Float, or None when the ends round differently or the ball is
        not finite. A ball that contains 0 without being exactly 0 never
        gives a Float: its ends round to values of opposite signs.

    Raises:
        On a trapped condition of the result.
    """
    var target = context.value() if context else ArithmeticContext(format=ball._midpoint.format())
    var rounded = _certify(ball, target)
    if not rounded:
        return None
    return Float(_rounded=_finish_round(rounded.take(), target, False))


def canonical(ball: Ball, precision: Int) raises -> Optional[Ball]:
    """The canonical ball, at `precision` bits, of the value that `ball`
    encloses, when `ball` is narrow enough to decide it.

    The canonical ball is `[round_down(v, p), round_up(v, p)]` for the
    enclosed value `v`: its midpoint is the exact mean of the two ends, at
    `p + 1` bits, and its radius their exact half-width. It depends only on
    `v` and `p`, never on how `ball` was computed, so a cache of the widest ball
    serves every narrower request the same: `canonical(pi_ball(256), 64)` is
    `pi_ball(64)`.

    Args:
        ball: A ball enclosing the value.
        precision: The precision `p` of the two ends, at least 1.

    Returns:
        The canonical ball, or None when either end of `ball` rounds
        differently from the other in its direction.

    Raises:
        When the precision is invalid.
    """
    var format = FloatFormat(precision)
    var low = _certify(ball, ArithmeticContext(format=format, rounding=RoundingMode.toward_negative))
    if not low:
        return None
    var high = _certify(ball, ArithmeticContext(format=format, rounding=RoundingMode.toward_positive))
    if not high:
        return None
    return _hull(Float(_rounded=low.take()), Float(_rounded=high.take()), precision + 1)


def floor_if_certain(ball: Ball) raises -> Optional[Integer]:
    """The floor, when it is the same for every point of the ball.

    Args:
        ball: The ball.

    Returns:
        The floor, or None when it varies or the ball is not finite.

    Raises:
        Only on a checked size error.
    """
    if not ball.is_finite():
        return None
    var low = ball._exact_lower().floor()
    if low == ball._exact_upper().floor():
        return low
    return None


def ceil_if_certain(ball: Ball) raises -> Optional[Integer]:
    """The ceiling, when it is the same for every point of the ball.

    Args:
        ball: The ball.

    Returns:
        The ceiling, or None when it varies or the ball is not finite.

    Raises:
        Only on a checked size error.
    """
    if not ball.is_finite():
        return None
    var low = ball._exact_lower().ceil()
    if low == ball._exact_upper().ceil():
        return low
    return None


def round_half_even_if_certain(ball: Ball) raises -> Optional[Integer]:
    """The nearest integer (ties to even), when it is the same for every point.

    Args:
        ball: The ball.

    Returns:
        The rounded value, or None when it varies or the ball is not finite.

    Raises:
        Only on a checked size error.
    """
    if not ball.is_finite():
        return None
    var low = ball._exact_lower().round()
    if low == ball._exact_upper().round():
        return low
    return None


def simplest_rational_in(ball: Ball) raises -> Rational:
    """The simplest rational in the ball: the least denominator, and among
    those the least absolute numerator.

    A continued-fraction descent on the exact ends finds it; it is a loop over
    an explicit list of partial quotients, not a recursion.
    `simplest_rational_in(Ball.from_interval("3.14059", "3.14259"))` is
    201/64, Mathematica's `Rationalize[3.14159, 10^-3]`.

    Args:
        ball: A finite ball.

    Returns:
        The simplest rational in the closed interval.

    Raises:
        Unless the ball is finite.
    """
    if not ball.is_finite():
        raise Error("Cannot find a rational in an unbounded or indeterminate ball; check is_finite() first.")
    var low = ball.lower_rational()
    var high = ball.upper_rational()
    if low <= 0 and high >= 0:
        return Rational(0)
    var negative = high < 0
    if negative:
        var swap = -high
        high = -low
        low = swap
    var terms = List[Integer]()
    while True:
        var whole = low.floor()
        if Rational(whole) == low:
            terms.append(whole)
            break
        if Rational(whole + 1) <= high:
            terms.append(whole + 1)
            break
        terms.append(whole)
        var next_low = Rational(1) / (high - Rational(whole))
        high = Rational(1) / (low - Rational(whole))
        low = next_low
    var value = Rational(terms[len(terms) - 1])
    for i in range(len(terms) - 2, -1, -1):
        value = Rational(terms[i]) + Rational(1) / value
    return -value if negative else value
