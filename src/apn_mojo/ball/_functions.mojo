"""Ball functions from point kernels.

A monotone function uses the endpoint method: the kernel at the two exact ends
of the ball, and the least ball over the lower end of one result and the upper
end of the other; an exact ball needs one kernel call. `sin` and `cos`
propagate from the midpoint, widening by `r min(1, |cos m| + r)` (or `|sin m|`),
and clamp to `[-1, 1]`. `tan` is indeterminate when `X/pi - 1/2` may contain an
integer (a pole), and otherwise increasing. `cosh` decreases below 0 and
increases above it; a ball containing 0 has `cosh` in `[1, cosh(max |end|)]`.

A function undefined somewhere in its input ball gives an indeterminate ball;
the domain is decided on the exact ends. Past the budget, `sin` and `cos` give
`[0 +/- 1]` and `tan` an indeterminate ball.
"""
from ..float._rounding import _RoundedBinary
from ..float.status import NumericStatus
from ._radius import _up, _down
from ._medium import _exp_medium, _log_medium, _atan_medium, _sin_cos_medium, _sin_or_cos_medium


from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float._arithmetic import _float_operation
from .context import BallContext
from ._radius import _Radius
from .value import Ball, _BallArgument, _FINITE, _UNBOUNDED, _INDETERMINATE
from ._arithmetic import _working, _hull, _rounded_ball, _sum, _product, _quotient, _power, _directed, _scale2
from ._kernels import (
    _point, _constant_kernel, _integral,
    _SQRT, _EXP, _EXPM1, _EXP2, _LOG, _LOG1P, _LOG2, _LOG10, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS,
    _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH, _POW, _ATAN2, _ROOTN, _PI, _HALF_PI,
)
from ._trig import _kernel_sin_cos
from .sets import contains_integer


def _ball_budget(context: Optional[BallContext], w: Int) -> Int:
    if context and context.value().max_precision():
        return context.value().max_precision().value()
    return max(8 * w, w + 4096)


def _unit_ball(w: Int) raises -> Ball:
    """`[0 +/- 1]`."""
    return Ball(Integer(0), Integer(1), precision=w)


def _symmetric(radius: Ball, w: Int) raises -> Ball:
    """`[0 +/- c]` for a ball enclosing a positive constant c."""
    return Ball(_midpoint=Float.zero(context=ArithmeticContext(format=FloatFormat(w))), _radius=_Radius.upper(radius._exact_upper()), _kind=_FINITE)


def _unbounded_image(code: Int, n: Int, w: Int) raises -> Ball:
    """A function of the ball of every real number."""
    if code == _SIN or code == _COS or code == _TANH:
        return _unit_ball(w)
    if code == _ATAN:
        return _symmetric(_constant_kernel(_HALF_PI, 30), w)
    if code == _EXP or code == _EXP2 or code == _EXPM1 or code == _SINH or code == _COSH or code == _ASINH:
        return Ball.unbounded(w)
    if code == _ROOTN and n % 2 == 1:
        return Ball.unbounded(w)
    return Ball.indeterminate(w)


def _outside_domain(code: Int, n: Int, low: Float, high: Float) raises -> Bool:
    """Whether the function is undefined at some point of `[low, high]`."""
    if code == _LOG or code == _LOG2 or code == _LOG10:
        return not (low > 0)
    if code == _LOG1P:
        return not (low > -1)
    if code == _ACOSH:
        return not (low >= 1)
    if code == _ATANH:
        return not (low > -1 and high < 1)
    if code == _ASIN or code == _ACOS:
        return not (low >= -1 and high <= 1)
    if code == _SQRT or (code == _ROOTN and n % 2 == 0):
        return not (low >= 0)
    return False


def _monotone(code: Int, x: Ball, n: Int, w: Int, budget: Int, increasing: Bool) raises -> Ball:
    return _monotone_by[_point](code, x, n, w, budget, increasing)


def _monotone_by[
    point: def(Int, Float, Int, Int, Int) raises thin -> Ball,
](code: Int, x: Ball, n: Int, w: Int, budget: Int, increasing: Bool) raises -> Ball:
    """A monotone function of a ball, from `point` (the function at an exact
    point) at the ball's two exact ends."""
    if x.is_exact():
        var value = point(code, x._midpoint, n, w, budget)
        if not value.is_finite():
            return value^
        return _rounded_ball(_BallArgument(value), w)
    var a = point(code, x._exact_lower(), n, w, budget)
    var b = point(code, x._exact_upper(), n, w, budget)
    if a.is_indeterminate() or b.is_indeterminate():
        return Ball.indeterminate(w)
    if not a.is_finite() or not b.is_finite():
        return Ball.unbounded(w)
    if increasing:
        return _hull(a._exact_lower(), b._exact_upper(), w)
    return _hull(b._exact_lower(), a._exact_upper(), w)


def _clamp_unit(x: Ball, w: Int) raises -> Ball:
    """The least ball containing the part of x within `[-1, 1]`."""
    var low = x._exact_lower()
    var high = x._exact_upper()
    var minus_one = Float(-1)
    var one = Float(1)
    if low >= minus_one and high <= one:
        return x
    return _hull(low if low > minus_one else minus_one, high if high < one else one, w)


def _sin_cos_at(x: Ball, w: Int, budget: Int) raises -> Optional[Tuple[Ball, Ball]]:
    """`(sin m, cos m)` at the midpoint m of a ball of radius below 4, from the
    kernel at `w` bits; None for a wider ball or past the budget."""
    if x._radius.compare(_Radius.power_of_two(2)) >= 0:
        return None
    var mid = x._midpoint
    if mid.is_zero():
        return (Ball(Integer(0), precision=w + 8), Ball(Integer(1), precision=w + 8))
    var pair = _kernel_sin_cos(mid, w, budget)
    if pair[0].is_indeterminate():
        return None
    return pair


def _widened(value: Ball, other: Ball, radius: _Radius) raises -> Ball:
    """sin or cos at the midpoint, widened for an argument radius r by
    `r min(1, |other| + r)`, a bound of the derivative over the ball."""
    if radius.is_zero():
        return value
    var slope = _Radius.upper(other._midpoint).add(other._radius).add(radius)
    if slope.compare(_Radius.power_of_two(0)) > 0:
        slope = _Radius.power_of_two(0)
    return Ball(_midpoint=value._midpoint, _radius=value._radius.add(radius.multiply(slope)), _kind=_FINITE)


def _sin_or_cos(code: Int, x: Ball, w: Int, budget: Int) raises -> Ball:
    var pair = _sin_cos_at(x, w, budget)
    if not pair:
        return _unit_ball(w)
    var value = pair.value()[0] if code == _SIN else pair.value()[1]
    var other = pair.value()[1] if code == _SIN else pair.value()[0]
    if x._radius.is_zero():
        return _rounded_ball(_BallArgument(value), w)
    return _clamp_unit(_rounded_ball(_BallArgument(_widened(value, other, x._radius)), w), w)


def _tan(x: Ball, w: Int, budget: Int) raises -> Ball:
    """tan as sin / cos: both at `w + 4` bits, guard bits for the quotient,
    then one ball division, indeterminate when the cosine reaches 0, as at a
    pole."""
    var pair = _sin_cos_at(x, w + 4, budget)
    if not pair:
        return Ball.indeterminate(w)
    var s = pair.value()[0]
    var c = pair.value()[1]
    return _quotient(
        _BallArgument(_widened(s, c, x._radius)), _BallArgument(_widened(c, s, x._radius)), Optional[BallContext](BallContext(w))
    )


def _cosh(x: Ball, w: Int, budget: Int) raises -> Ball:
    var low = x._exact_lower()
    var high = x._exact_upper()
    if low > 0:
        return _monotone(_COSH, x, 0, w, budget, True)
    if high < 0:
        return _monotone(_COSH, x, 0, w, budget, False)
    var far = abs(low) if abs(low) > abs(high) else abs(high)
    var top = _point(_COSH, far, 0, w, budget)
    if not top.is_finite():
        return Ball.unbounded(w)
    return _hull(Float(1), top._exact_upper(), w)


@always_inline
def _medium_ball(code: Int, x: _BallArgument, w: Int) raises -> Optional[Ball]:
    """A narrow ball of exp, expm1, log, sin, cos, tan or atan: the medium
    kernel at the midpoint, at the ball's precision and rounded
    once, widened by the function's derivative over the radius (Appendix
    E.11), in radius arithmetic. None leaves the ball to the general path:
    other functions, exact points such as 0, midpoints that are not Floats,
    radii above `2**-17` of the midpoint, and what the kernels decline.
    Inlined into its one caller: called, it built its Optional result in
    memory, several hundred instructions a call."""
    if code != _EXP and code != _EXPM1 and code != _LOG and code != _SIN and code != _COS and code != _TAN and code != _ATAN:
        return None
    ref a = x.midpoint.value
    if a.kind != 1 or not a.denominator._is_one() or not x.midpoint.format:
        return None
    var format = x.midpoint.format.value()
    var p = format.precision()
    if a.numerator.magnitude_bit_length() != p:
        return None
    var mid = Float(_rounded=_RoundedBinary(1, a.negative, a.numerator, Int(a.scale) + p, format, NumericStatus()))
    var r = x.radius
    if not r.is_zero() and r.compare(_Radius.power_of_two(mid._exponent - 18)) >= 0:
        return None
    if code == _SIN or code == _COS:
        return _sin_or_cos_medium(mid, r, w, code == _COS)
    if code == _TAN:
        # tan = sin / cos: both at 4 more bits, then one division.
        var pair = _sin_cos_medium(mid, r, w + 4)
        if not pair:
            return None
        ref both = pair.value()
        return _quotient(_BallArgument(both[0]), _BallArgument(both[1]), Optional[BallContext](BallContext(w)))
    var value: Optional[Ball]
    if code == _LOG:
        if mid._negative:
            return None
        value = _log_medium(mid, w)
    elif code == _ATAN:
        value = _atan_medium(mid, w)
    else:
        value = _exp_medium(mid, w, code == _EXPM1)
    if not value or r.is_zero():
        return value^
    ref ball = value.value()
    var one = _Radius.power_of_two(0)
    var extra: _Radius
    if code == _LOG:
        # r / (m - r) <= (r / m)(1 + 2**-16) for r <= m 2**-17.
        extra = r.divide(_Radius.lower(mid)).multiply(_up(UInt128((1 << 16) + 1), -16))
    elif code == _ATAN:
        # r / (1 + d**2) with d = |m| - r >= |m| (1 - 2**-17).
        var d = _Radius.lower(mid).multiply_down(_down(UInt128((1 << 17) - 1), -17))
        extra = r.divide(one.add_down(d.multiply_down(d)))
    else:
        # |f(m + t) - f(m)| <= e**m (e**r - 1) <= e**m r (1 + r) for r <= 1,
        # with e**m within the ball (exp) or one above it (expm1).
        var size = _Radius.upper(ball._midpoint).add(ball._radius)
        if code == _EXPM1:
            size = size.add(one)
        extra = size.multiply(r).multiply(one.add(r))
    ball._radius = ball._radius.add(extra)
    return value^


def _derived_ball(code: Int, x: _BallArgument, w: Int, budget: Int) raises -> Optional[Ball]:
    """A narrow ball of asin, acos, atanh, log1p, asinh, acosh, sinh, cosh or
    tanh: the kernel at the midpoint, widened by `r D` for a bound D of
    `|f'|` over the ball (Appendix E.11), in radius arithmetic, then rounded
    once. The kernel runs at `w + 8` bits, since it promises `w - 8`. With
    `r < 2**-17 |m|`:

    | f | D |
    |---|---|
    | asin, acos | `1 / sqrt(1 - (|m| + r)**2)` |
    | atanh | `1 / (1 - (|m| + r)**2)` |
    | log1p | `1 / (1 + m - r)` |
    | asinh | `min(1, 1 / (|m| - r))` |
    | acosh | `1 / sqrt(2 (m - 1 - r))`, since `(m - r)**2 - 1 >= 2 (m - r - 1)` |
    | sinh | `(|sinh m| + 1)(1 + 2r) >= cosh(m) e**r >= cosh(|m| + r)` |
    | cosh | `cosh(m) (1 + 2r) >= sinh(|m| + r)` |
    | tanh | `(1 - tanh(|m|)**2)(1 + 4r) >= sech(|m| - r)**2` |

    None for other functions, exact balls (one kernel call already), wider
    balls, and balls that may reach a domain's edge."""
    if (
        code != _ASIN and code != _ACOS and code != _ATANH and code != _LOG1P and code != _ASINH
        and code != _ACOSH and code != _SINH and code != _COSH and code != _TANH
    ):
        return None
    var r = x.radius
    ref a = x.midpoint.value
    if r.is_zero() or r.infinite or a.kind != 1 or not a.denominator._is_one():
        return None
    var mid = x.ball()._midpoint
    if r.compare(_Radius.power_of_two(mid._exponent - 18)) >= 0:
        return None
    var one = _Radius.power_of_two(0)
    var high = _Radius.upper(mid).add(r)
    var bound = one
    if code == _ASIN or code == _ACOS or code == _ATANH or (code == _LOG1P and mid._negative):
        # 1 - (|m| + r)**2, or 1 - (|m| + r) for log1p: positive inside the domain.
        var gap = one.sub_down(high if code == _LOG1P else high.multiply(high))
        if gap.is_zero():
            return None
        bound = one.divide(gap.sqrt_down() if code == _ASIN or code == _ACOS else gap)
    elif code == _LOG1P:
        bound = one.divide(one.add_down(_Radius.lower(mid).sub_down(r)))
    elif code == _ASINH:
        var low = _Radius.lower(mid).sub_down(r)
        if low.compare(one) > 0:
            bound = one.divide(low)
    elif code == _ACOSH:
        if mid._negative:
            return None
        var t = Float(_rounded=_float_operation(mid, Integer(1), 1, _exact_context()))
        if t.is_zero() or t._negative:
            return None
        var above = _Radius.lower(t).sub_down(r)
        if above.is_zero():
            return None
        bound = one.divide(above.scale2(1).sqrt_down())
    var value = _point(code, mid, 0, w + 8, budget)
    if not value.is_finite():
        return None
    var size = _Radius.upper(value._midpoint).add(value._radius)
    if code == _SINH:
        bound = size.add(one).multiply(one.add(r.scale2(1)))
    elif code == _COSH:
        bound = size.multiply(one.add(r.scale2(1)))
    elif code == _TANH:
        var least = _Radius.lower(value._midpoint).sub_down(value._radius)
        bound = one.sub_up(least.multiply_down(least)).multiply(one.add(r.scale2(2)))
    value._radius = value._radius.add(r.multiply(bound))
    return _rounded_ball(_BallArgument(value), w)


def _ball_function(code: Int, x: _BallArgument, n: Int, context: Optional[BallContext]) raises -> Ball:
    """A real function of a ball: inclusion, the domain rule, and the budget's
    trivial enclosures; narrow balls of the functions with medium kernels
    first take `_medium_ball`."""
    var w = _working(context, x.precision)
    if x.kind == _INDETERMINATE:
        return Ball.indeterminate(w)
    if x.kind == _UNBOUNDED:
        return _unbounded_image(code, n, w)
    var medium = _medium_ball(code, x, w)
    if medium:
        # A copy: Optional's take is not always inlined, and then moves a Ball
        # out byte by byte, about 400 instructions; a copy costs a few dozen
        # (two atomic counts above 64 bits).
        return medium.value()
    return _general_ball_function(code, x, n, w, context)


def _narrow(x: Ball) -> Bool:
    """Whether a finite ball has a nonzero midpoint and a radius below
    `2**-17` of it: the balls the medium kernels take at their midpoint."""
    return x.is_finite() and not x._midpoint.is_zero() and x._radius.compare(_Radius.power_of_two(x._midpoint._exponent - 18)) < 0


def _sin_cos_pair(x: Ball, w: Int) raises -> Tuple[Ball, Ball]:
    """`(sin x, cos x)` at `w` bits. A narrow ball takes one medium kernel
    call for both, the same values as two calls would give; any other, the
    two functions."""
    if _narrow(x):
        var pair = _sin_cos_medium(x._midpoint, x._radius, w)
        if pair:
            return pair.value()
    var c = Optional[BallContext](BallContext(w))
    return (_ball_function(_SIN, _BallArgument(x), 0, c), _ball_function(_COS, _BallArgument(x), 0, c))


def _sinh_cosh_pair(x: Ball, w: Int) raises -> Tuple[Ball, Ball]:
    """`(sinh x, cosh x)` at `w` bits from one exponential. For a narrow ball,
    with `a = |m|`:
    from `a = 1/2`, `e = exp(a)`, `sinh a = (e - 1/e)/2` and
    `cosh a = (e + 1/e)/2`; below, `u = expm1(a)`, `sinh a = (u + u/(u + 1))/2`
    and `cosh a = 1 + u**2/(2 (u + 1))`, which do not cancel. Both widen by
    `r cosh(m) (1 + 2r)`, a bound of `r cosh(|m| + r)`, which bounds both
    derivatives over the ball. Any other ball takes the two functions."""
    if _narrow(x):
        var a = abs(x._midpoint)
        var c = Optional[BallContext](BallContext(w + 8))
        var large = a._exponent >= 0
        var kernel = _exp_medium(a, w + 8, not large)
        if kernel:
            var e = kernel.value()
            var sinh: Ball
            var cosh: Ball
            if large:
                var inverse = _BallArgument(_quotient(_BallArgument(Integer(1)), _BallArgument(e), c))
                sinh = _scale2(_BallArgument(_sum(_BallArgument(e), inverse, True, c)), Integer(-1), c)
                cosh = _scale2(_BallArgument(_sum(_BallArgument(e), inverse, False, c)), Integer(-1), c)
            else:
                var above = _BallArgument(_sum(_BallArgument(e), _BallArgument(Integer(1)), False, c))
                sinh = _scale2(_BallArgument(_sum(_BallArgument(e), _BallArgument(_quotient(_BallArgument(e), above, c)), False, c)), Integer(-1), c)
                var square = _BallArgument(_product(_BallArgument(e), _BallArgument(e), c))
                cosh = _sum(_BallArgument(Integer(1)), _BallArgument(_scale2(_BallArgument(_quotient(square, above, c)), Integer(-1), c)), False, c)
            var r = x._radius
            var one = _Radius.power_of_two(0)
            var extra = r.multiply(_Radius.upper(cosh._midpoint).add(cosh._radius)).multiply(one.add(r.scale2(1)))
            sinh._radius = sinh._radius.add(extra)
            cosh._radius = cosh._radius.add(extra)
            if x._midpoint._negative:
                sinh = Ball(_midpoint=-sinh._midpoint, _radius=sinh._radius, _kind=sinh._kind)
            return (_rounded_ball(_BallArgument(sinh), w), _rounded_ball(_BallArgument(cosh), w))
    var c = Optional[BallContext](BallContext(w))
    return (_ball_function(_SINH, _BallArgument(x), 0, c), _ball_function(_COSH, _BallArgument(x), 0, c))


@no_inline
def _general_ball_function(code: Int, x: _BallArgument, n: Int, w: Int, context: Optional[BallContext]) raises -> Ball:
    """A finite ball the medium kernels did not take: a narrow ball of a
    derived function by `_derived_ball`, else the domain rule and the
    function's general method. Out of line, so that `_ball_function` stays
    the medium kernels' short path."""
    var budget = _ball_budget(context, w)
    var derived = _derived_ball(code, x, w, budget)
    if derived:
        return derived.value()
    var ball = x.ball()
    var exact = ball.is_exact()
    var low = ball._midpoint if exact else ball._exact_lower()
    var high = ball._midpoint if exact else ball._exact_upper()
    if _outside_domain(code, n, low, high):
        return Ball.indeterminate(w)
    if code == _SIN or code == _COS:
        return _sin_or_cos(code, ball, w, budget)
    if code == _TAN:
        return _tan(ball, w, budget)
    if code == _COSH:
        return _cosh(ball, w, budget)
    return _monotone(code, ball, n, w, budget, code != _ACOS)


def _ball_atan2(y: _BallArgument, x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """atan2 of balls: the angle of every point of `x + i y`. A ball that
    meets the negative real axis covers angles on both sides of the cut, so it
    gives `[0 +/- pi]`; one that contains the origin is indeterminate."""
    var w = _working(context, y.precision, x.precision)
    if y.kind == _INDETERMINATE or x.kind == _INDETERMINATE:
        return Ball.indeterminate(w)
    if y.kind == _UNBOUNDED or x.kind == _UNBOUNDED:
        return _symmetric(_constant_kernel(_PI, 30), w)
    return _atan2_of(y.ball(), x.ball(), w)


def _atan2_of(b: Ball, a: Ball, w: Int) raises -> Ball:
    """atan2 of finite balls, `b` the ordinate and `a` the abscissa, at `w`
    bits; ComplexBall's `angle` and `log` pass their parts here as they are."""
    var c = Optional[BallContext](BallContext(w + 16))
    if a.certainly_positive():
        return _ball_function(_ATAN, _BallArgument(_quotient(_BallArgument(b), _BallArgument(a), c)), 0, Optional[BallContext](BallContext(w)))
    if b.certainly_positive() or b.certainly_negative():
        var angle = _ball_function(_ATAN, _BallArgument(_quotient(_BallArgument(a), _BallArgument(b), c)), 0, Optional[BallContext](BallContext(w + 8)))
        var half_pi = _constant_kernel(_HALF_PI, w + 8)
        if b.certainly_positive():
            return _sum(_BallArgument(half_pi), _BallArgument(angle), True, Optional[BallContext](BallContext(w)))
        return _sum(
            _BallArgument(Ball(_midpoint=-half_pi._midpoint, _radius=half_pi._radius, _kind=half_pi._kind)),
            _BallArgument(angle), True, Optional[BallContext](BallContext(w)),
        )
    if a.certainly_negative():
        return _symmetric(_constant_kernel(_PI, 30), w)
    return Ball.indeterminate(w)


def _ball_pow(x: _BallArgument, y: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """`x**y` of balls: an exact integer `y` is an integer power; otherwise
    `exp(y log x)` for an `x` certainly above 0, and 0 for an exact 0 and a `y`
    certainly above 0. Anything else may meet an undefined point."""
    var w = _working(context, x.precision, y.precision)
    if x.kind == _INDETERMINATE or y.kind == _INDETERMINATE:
        return Ball.indeterminate(w)
    if y.kind == _UNBOUNDED or x.kind == _UNBOUNDED:
        return Ball.indeterminate(w)
    var exponent = y.ball()
    if exponent.is_exact():
        var whole = _integral(exponent._midpoint)
        if whole:
            return _power(x, whole.value(), Optional[BallContext](BallContext(w)))
    var base = x.ball()
    if base.is_exact() and base._midpoint.is_zero():
        if exponent.certainly_positive():
            return Ball(Integer(0), precision=w)
        return Ball.indeterminate(w)
    if not base.certainly_positive():
        return Ball.indeterminate(w)
    var rough = _product(_BallArgument(_ball_function(_LOG, _BallArgument(base), 0, Optional[BallContext](BallContext(32)))), _BallArgument(exponent), Optional[BallContext](BallContext(64)))
    var extra = 0
    if rough.is_finite() and not rough._midpoint.is_zero():
        extra = max(0, rough._midpoint._exponent)
    var bits = w + extra + 16
    var c = Optional[BallContext](BallContext(bits))
    var t = _product(_BallArgument(_ball_function(_LOG, _BallArgument(base), 0, c)), _BallArgument(exponent), c)
    return _ball_function(_EXP, _BallArgument(t), 0, Optional[BallContext](BallContext(w)))
