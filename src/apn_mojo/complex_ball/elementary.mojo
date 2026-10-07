"""Elementary functions of complex balls, one declaration per name.

Each function returns a complex ball containing its value at every point of
the input rectangle, from the real ball functions of the parts:
`exp(x + iy) = e**x (cos y + i sin y)`, `log z = log|z| + i arg z`, and the
like. On a branch cut a point takes its counter-clockwise continuous value:
`log(-2) = log 2 + i pi`, `sqrt(-4) = 2i` and
`asin(2) = pi/2 - i acosh(2)`. A rectangle that crosses a cut gets a result
covering both sides; a function undefined somewhere in the rectangle (`log`
at 0, `atanh` at +-1, `atan` at +-i) gives an indeterminate ball. A `context=`
sets the result precision, by default the ball's.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float._arithmetic import _float_operation
from ..float._functions import _sqrt_float, _scale_float
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball.sets import contains_zero as _ball_contains_zero
from ..ball._radius import _Radius
from ..ball._arithmetic import _sum, _product, _quotient, _scale2, _negate, _rounded_ball, _hull_from, _norm, _norm_bounds, _directed, _magnitude_range
from ..ball._functions import _ball_function, _sin_cos_pair, _sinh_cosh_pair
from ..ball._trig import _atan2_point
from ..ball._medium import _log_medium
from ..float._rounding import _RoundedBinary
from ..float.status import NumericStatus
from ..ball._arithmetic import _from_record, _nearest
from ..complex.context import ComplexContext, _ComplexContextArgument
from ..complex._input import _ComplexArgument
from ..complex._functions import _sqrt_complex
from ..ball._kernels import (
    _constant_kernel, _integral, _EXP, _LOG, _LOG1P, _SIN, _COS, _SINH, _COSH, _TANH, _SQRT, _HALF_PI, _ATANH,
)
from .value import ComplexBall, _add, _sub, _mul, _div
from .math import abs as _abs, angle as _arg, pow_int as _pow_int, _bits


def _at(w: Int) raises -> Optional[BallContext]:
    return Optional[BallContext](BallContext(w))


def _real(code: Int, x: Ball, w: Int) raises -> Ball:
    return _ball_function(code, _BallArgument(x), 0, _at(w))


def _mul_real(a: Ball, b: Ball, w: Int) raises -> Ball:
    return _product(_BallArgument(a), _BallArgument(b), _at(w))


def _rounded(z: ComplexBall, w: Int) raises -> ComplexBall:
    return ComplexBall(_real=_rounded_ball(_BallArgument(z.real()), w), _imag=_rounded_ball(_BallArgument(z.imag()), w))


def _indeterminate(w: Int) raises -> ComplexBall:
    return ComplexBall(_real=Ball.indeterminate(w), _imag=Ball.indeterminate(w))


def _contains_zero(z: ComplexBall) raises -> Bool:
    return _ball_contains_zero(z.real()) and _ball_contains_zero(z.imag())


def _times_i(z: ComplexBall) raises -> ComplexBall:
    return ComplexBall(_real=_negate(z.imag()), _imag=z.real())


def _times_minus_i(z: ComplexBall) raises -> ComplexBall:
    return ComplexBall(_real=z.imag(), _imag=_negate(z.real()))


def _half(z: ComplexBall, w: Int) raises -> ComplexBall:
    return ComplexBall(_real=_scale2(_BallArgument(z.real()), Integer(-1), _at(w)), _imag=_scale2(_BallArgument(z.imag()), Integer(-1), _at(w)))


def _clamp_nonnegative(x: Ball, w: Int) raises -> Ball:
    """The part of a ball at or above 0, for a quantity known to be nonnegative."""
    if not x.is_finite():
        return x
    var low = x._exact_lower()
    if not low._negative or low.is_zero():
        return x
    var high = x._exact_upper()
    if high._negative and not high.is_zero():
        return Ball(Integer(0), precision=w)
    # From 0 exactly: a centred hull would reach below it, and a root of the
    # result would be lost.
    return _hull_from(Float.zero(context=ArithmeticContext(format=FloatFormat(1))), high, w)


def exp(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the exponential, `e**x (cos y + i sin y)`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var e = _real(_EXP, value.real(), work)
    var y = _sin_cos_pair(value._imag, work)
    var re = _mul_real(e, y[1], work)
    var im = _mul_real(e, y[0], work)
    return _rounded(ComplexBall(_real=re, _imag=im), w)


def _half_range(a_low: Float, b_low: Float, a_high: Float, b_high: Float, w: Int) raises -> Ball:
    """`[(a_low + b_low) / 2, (a_high + b_high) / 2]` from 0 at least: bounds
    of a quantity known to be nonnegative, such as `(|z| + x) / 2`."""
    # Halved exactly before the hull: rounding the hull afterwards would undo
    # the midpoint bits that keep it at or above 0.
    var low = Float(_rounded=_scale_float(Float(_rounded=_float_operation(a_low, b_low, 0, _directed(w, False))), Integer(-1), _directed(w, False)))
    var high = Float(_rounded=_scale_float(Float(_rounded=_float_operation(a_high, b_high, 0, _directed(w, True))), Integer(-1), _directed(w, True)))
    if low._negative:
        low = Float.zero(context=ArithmeticContext(format=FloatFormat(1)))
    if high._negative:
        high = Float.zero(context=ArithmeticContext(format=FloatFormat(1)))
    return _hull_from(low, high, w)


def _log_off_cut(value: ComplexBall, w: Int) raises -> Optional[ComplexBall]:
    """log of a finite rectangle off the branch cut `(-inf, 0]`, with a real
    part certainly above 0 or an imaginary part certainly away from it, and
    `R = r_x + r_y` below `|m|`: `log(n)/2` for `n = x**2 + y**2` and
    `atan2(y, x)` at the midpoint, both from the medium kernels at `w` bits,
    widened by `R / (|m| - R)`. Log is analytic on the rectangle, and along
    the segment from m to a point of it `|(log z)'| = 1/|t|` with
    `|t| >= |m| - R`. For an exact rectangle n is exact; otherwise its three
    operations round at `w + 8` bits, one limb at 53, and add `2**(2 - w - 8)`
    to `log n`, below R's share. None for other rectangles, or where a kernel
    declines (n = 1, or n too near it for the kernel's extra bits)."""
    ref x = value._real
    ref y = value._imag
    if not value.is_finite():
        return None
    if not (x.certainly_positive() or y.certainly_positive() or y.certainly_negative()):
        return None
    var r = x._radius.add(y._radius)
    var bound = _Radius.zero()
    var n: Float
    if r.is_zero():
        n = Float(_rounded=_float_operation(
            Float(_rounded=_float_operation(x._midpoint, x._midpoint, 2, _exact_context())),
            Float(_rounded=_float_operation(y._midpoint, y._midpoint, 2, _exact_context())), 0, _exact_context(),
        ))
    else:
        var lx = _Radius.lower(x._midpoint)
        var ly = _Radius.lower(y._midpoint)
        var low = lx.multiply_down(lx).add_down(ly.multiply_down(ly)).sqrt_down().sub_down(r)
        if low.is_zero():
            return None
        bound = r.divide(low)
        var near = _nearest(w + 8)
        n = Float(_rounded=_float_operation(
            Float(_rounded=_float_operation(x._midpoint, x._midpoint, 2, near)),
            Float(_rounded=_float_operation(y._midpoint, y._midpoint, 2, near)), 0, near,
        ))
    var twice = _log_medium(n, w)
    # The kernel declines at n = 1 and just around it (|m| = 1, as at i):
    # there the certified real logarithm of the exact n takes over.
    var log_n = twice.value() if twice else _real(_LOG, Ball(n), w + 8)
    if not log_n.is_finite():
        return None
    if not r.is_zero():
        # Three roundings at w + 8 bits move n by a relative 2**(1 - w - 8) at most.
        log_n._radius = log_n._radius.add(_Radius.power_of_two(-w - 6))
    var re = _scale2(_BallArgument(log_n), Integer(-1), _at(w))
    var angle = _atan2_point(y._midpoint, x._midpoint, w)
    if not angle:
        return None
    var im = angle.value()
    re._radius = re._radius.add(bound)
    im._radius = im._radius.add(bound)
    return ComplexBall(_real=re, _imag=im)


def _log(value: ComplexBall, w: Int) raises -> ComplexBall:
    if value.is_indeterminate():
        return _indeterminate(w)
    var off_cut = _log_off_cut(value, w)
    if off_cut:
        return off_cut.value()
    if _contains_zero(value):
        return _indeterminate(w)
    var work = w + 16
    var x = value.real()
    var y = value.imag()
    var re: Ball
    if value.is_exact():
        # log|z| = log1p(x**2 + y**2 - 1) / 2 from the exact argument.
        var t = Float(_rounded=_float_operation(
            Float(_rounded=_float_operation(x._midpoint, x._midpoint, 2, _exact_context())),
            Float(_rounded=_float_operation(
                Float(_rounded=_float_operation(y._midpoint, y._midpoint, 2, _exact_context())), Integer(1), 1, _exact_context(),
            )),
            0, _exact_context(),
        ))
        re = _real(_LOG1P, Ball(t), work)
    else:
        var norm = _norm(x, y, work) if value.is_finite() else _sum(
            _BallArgument(_mul_real(x, x, work)), _BallArgument(_mul_real(y, y, work)), False, _at(work)
        )
        re = _real(_LOG, norm, work)
    re = _scale2(_BallArgument(re), Integer(-1), _at(work))
    return _rounded(ComplexBall(_real=re, _imag=_arg(value, context=_at(work))), w)


def log(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal logarithm, `log|z| + i arg z`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; the negative real axis takes
        imaginary part `pi`; indeterminate when the rectangle contains 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _log(value, _bits(value, context))


def _sqrt_off_cut(value: ComplexBall, w: Int) raises -> Optional[ComplexBall]:
    """sqrt of a finite rectangle off the branch cut `(-inf, 0]`, with a real
    part certainly above 0 or an imaginary part certainly away from it: the
    root of the midpoint m, each part rounded once, widened by
    `R / (2 sqrt(|m| - R))` for `R = r_x + r_y`, at least the half-diagonal.
    The root is analytic on the rectangle, and along the segment from m to any
    point z of it `|sqrt'| = 1 / (2 sqrt|t|)` with `|t| >= |m| - R`. None for
    other rectangles, or where `|m| <= R` may hold."""
    ref x = value._real
    ref y = value._imag
    if not value.is_finite():
        return None
    if not (x.certainly_positive() or y.certainly_positive() or y.certainly_negative()):
        return None
    var r = x._radius.add(y._radius)
    var bound = _Radius.zero()
    if not r.is_zero():
        var lx = _Radius.lower(x._midpoint)
        var ly = _Radius.lower(y._midpoint)
        var gap = lx.multiply_down(lx).add_down(ly.multiply_down(ly)).sqrt_down().sub_down(r)
        if gap.is_zero():
            return None
        bound = r.divide(gap.sqrt_down().scale2(1))
    var target = ArithmeticContext(format=FloatFormat(w))
    var root = _sqrt_complex(
        _ComplexArgument(real=x._midpoint, imag=y._midpoint), _ComplexContextArgument(ComplexContext(real=target, imag=target))
    )
    return ComplexBall(_real=_from_record(root[0], bound, w), _imag=_from_record(root[1], bound, w))


def _sqrt(value: ComplexBall, w: Int) raises -> ComplexBall:
    if value.is_indeterminate():
        return _indeterminate(w)
    var off_cut = _sqrt_off_cut(value, w)
    if off_cut:
        return off_cut.value()
    var x = value.real()
    var y = value.imag()
    if value.is_exact() and x._midpoint.is_zero() and y._midpoint.is_zero():
        return ComplexBall(0)
    var work = w + 16
    var c = _at(work)
    var half_sum: Ball
    var half_difference: Ball
    if value.is_finite():
        # (|z| +- x) / 2 from exact bounds of |z| and x: at least 0, and not
        # lost to a sum's rounded radius where |z| and x nearly cancel.
        var norm = _norm_bounds(x, y, work + 8)
        var m_low = Float(_rounded=_sqrt_float(norm[0], _directed(work + 8, False)))
        var m_high = Float(_rounded=_sqrt_float(norm[1], _directed(work + 8, True)))
        var x_low = x._exact_lower()
        var x_high = x._exact_upper()
        half_sum = _half_range(m_low, x_low, m_high, x_high, work)
        half_difference = _half_range(m_low, -x_high, m_high, -x_low, work)
    else:
        var magnitude = _abs(value, context=c)
        half_sum = _clamp_nonnegative(_scale2(_BallArgument(_sum(_BallArgument(magnitude), _BallArgument(x), False, c)), Integer(-1), c), work)
        half_difference = _clamp_nonnegative(_scale2(_BallArgument(_sum(_BallArgument(magnitude), _BallArgument(x), True, c)), Integer(-1), c), work)
    var t = _real(_SQRT, half_sum, work)
    var u = _real(_SQRT, half_difference, work)
    var im: Ball
    if y.certainly_positive():
        im = u
    elif y.certainly_negative():
        im = _negate(u)
    elif y.is_exact():
        # On the negative real axis the root is +i sqrt|x|, from above.
        im = u
    else:
        # Either sign: |Im| is at most u. Right of the cut y / (2 Re) also
        # holds, and is the tighter one for a narrow rectangle.
        im = Ball(_midpoint=Float.zero(context=ArithmeticContext(format=FloatFormat(work))), _radius=_Radius.upper(u._midpoint).add(u._radius), _kind=u._kind)
        if x.certainly_positive():
            var ratio = _quotient(_BallArgument(y), _BallArgument(_scale2(_BallArgument(t), Integer(1), c)), c)
            if ratio.is_finite() and ratio._radius.compare(im._radius) < 0:
                im = ratio
    return _rounded(ComplexBall(_real=t, _imag=im), w)


def sqrt(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal square root, with real part at least 0.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every root; the negative real axis takes the root
        with a positive imaginary part.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _sqrt(value, _bits(value, context))


def pow(base: ComplexBall, exponent: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of `base**exponent`, `exp(exponent log base)`.

    An exact integer exponent gives an integer power; an exact 0 base gives 0
    for an exponent whose real part is certainly positive.

    Args:
        base: The base.
        exponent: The exponent.
        context: The result precision; by default the operands' largest.

    Returns:
        A ball containing every power; indeterminate where the power is
        undefined, as for 0 to a power with real part at most 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = max(_bits(base, context), _bits(exponent, context))
    if exponent.is_exact() and exponent.imag()._midpoint.is_zero():
        var whole = _integral(exponent.real()._midpoint)
        if whole:
            return _pow_int(base, whole.value(), context=_at(w))
    if base.is_exact() and base.real()._midpoint.is_zero() and base.imag()._midpoint.is_zero():
        if exponent.real().certainly_positive():
            return ComplexBall(0)
        return _indeterminate(w)
    var work = w + 16
    return exp(_mul(exponent, _log(base, work), _at(work)), context=_at(w))


def sin(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the sine, `sin x cosh y + i cos x sinh y`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var x = _sin_cos_pair(value._real, work)
    var y = _sinh_cosh_pair(value._imag, work)
    var re = _mul_real(x[0], y[1], work)
    var im = _mul_real(x[1], y[0], work)
    return _rounded(ComplexBall(_real=re, _imag=im), w)


def cos(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the cosine, `cos x cosh y - i sin x sinh y`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var x = _sin_cos_pair(value._real, work)
    var y = _sinh_cosh_pair(value._imag, work)
    var re = _mul_real(x[1], y[1], work)
    var im = _negate(_mul_real(x[0], y[0], work))
    return _rounded(ComplexBall(_real=re, _imag=im), w)


def _one_ball(w: Int) raises -> Ball:
    return Ball(Integer(1), precision=w)


def _doubled_ratio(value: ComplexBall, w: Int, trigonometric: Bool) raises -> ComplexBall:
    """`(f(2x) + i g(2y)) / (h(2x) + k(2y))`: tan with (f, g, h, k) = (sin,
    sinh, cos, cosh), tanh with (sinh, sin, cosh, cos). Unlike `sin z / cos z`
    the parts never cancel, so a tiny part keeps its relative accuracy."""
    var work = w + 16
    var c = _at(work)
    var x2 = _scale2(_BallArgument(value.real()), Integer(1), c)
    var y2 = _scale2(_BallArgument(value.imag()), Integer(1), c)
    # (sin, cos) of the trigonometric part and (sinh, cosh) of the other.
    var circular = _sin_cos_pair(x2 if trigonometric else y2, work)
    var hyperbolic = _sinh_cosh_pair(y2 if trigonometric else x2, work)
    var f = circular[0] if trigonometric else hyperbolic[0]
    var g = hyperbolic[0] if trigonometric else circular[0]
    var denominator = _sum(_BallArgument(circular[1]), _BallArgument(hyperbolic[1]), False, c)
    var re = _quotient(_BallArgument(f), _BallArgument(denominator), c)
    var im = _quotient(_BallArgument(g), _BallArgument(denominator), c)
    if re.is_finite() and im.is_finite():
        return _rounded(ComplexBall(_real=re, _imag=im), w)
    # A wide rectangle: cosh varies over orders of magnitude across it, and the
    # ratio of two such balls is lost. Divided through by cosh, the parts are
    # sin * sech and tanh over 1 + cos * sech, which stays near 1 away from
    # the poles; tanh comes bounded from its own kernel, and sech, decreasing
    # in |v|, lies in [0, 1/cosh(min |v|)] (no ball 1/cosh, which a wide
    # cosh ball reaching 0 would lose).
    var v = y2 if trigonometric else x2
    var nearest = _magnitude_range(v)[0]
    var least = _real(_COSH, Ball(nearest), work)._exact_lower()
    var top = Float(_rounded=_float_operation(Float(1), least, 3, _directed(work, True)))
    var sech = _hull_from(Float.zero(context=ArithmeticContext(format=FloatFormat(1))), top, work)
    var bounded = _real(_TANH, v, work)
    var scaled = _product(_BallArgument(circular[0]), _BallArgument(sech), c)
    var near_one = _sum(_BallArgument(_one_ball(work)), _BallArgument(_product(_BallArgument(circular[1]), _BallArgument(sech), c)), False, c)
    var a = _quotient(_BallArgument(scaled), _BallArgument(near_one), c)
    var b = _quotient(_BallArgument(bounded), _BallArgument(near_one), c)
    if trigonometric:
        return _rounded(ComplexBall(_real=a, _imag=b), w)
    return _rounded(ComplexBall(_real=b, _imag=a), w)


def tan(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the tangent, `(sin 2x + i sinh 2y) / (cos 2x + cosh 2y)`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; indeterminate when the rectangle may
        contain a pole.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _doubled_ratio(value, _bits(value, context), True)


def sinh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the hyperbolic sine, `sinh x cos y + i cosh x sin y`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var x = _sinh_cosh_pair(value._real, work)
    var y = _sin_cos_pair(value._imag, work)
    var re = _mul_real(x[0], y[1], work)
    var im = _mul_real(x[1], y[0], work)
    return _rounded(ComplexBall(_real=re, _imag=im), w)


def cosh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the hyperbolic cosine, `cosh x cos y + i sinh x sin y`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var x = _sinh_cosh_pair(value._real, work)
    var y = _sin_cos_pair(value._imag, work)
    var re = _mul_real(x[1], y[1], work)
    var im = _mul_real(x[0], y[0], work)
    return _rounded(ComplexBall(_real=re, _imag=im), w)


def tanh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the hyperbolic tangent, `(sinh 2x + i sin 2y) / (cosh 2x + cos 2y)`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; indeterminate when the rectangle may
        contain a pole.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _doubled_ratio(value, _bits(value, context), False)


def _asin(value: ComplexBall, w: Int) raises -> ComplexBall:
    var work = w + 16
    var c = _at(work)
    var root = _sqrt(_sub(ComplexBall(1), _mul(value, value, c), c), work)
    return _times_minus_i(_log(_add(_times_i(value), root, c), work))


def asin(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal arcsine, `-i log(iz + sqrt(1 - z**2))`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; on the cuts `(-inf, -1)` and
        `(1, inf)` the counter-clockwise continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    return _rounded(_asin(value, w), w)


def acos(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal arccosine, `pi/2 - asin z`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; on the cuts the counter-clockwise
        continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var s = _asin(value, work)
    var half_pi = _constant_kernel(_HALF_PI, work)
    return _rounded(ComplexBall(_real=_sum(_BallArgument(half_pi), _BallArgument(s.real()), True, _at(work)), _imag=_negate(s.imag())), w)


def _twice(x: Float) -> Float:
    """2x exactly, its exponent moved up by one: an exact Float sum takes the
    general path, a thousand instructions and more."""
    return Float(_rounded=_RoundedBinary(x._kind, x._negative, x._significand, x._exponent + 1, x.format(), NumericStatus()))


def _atan_off_cuts(value: ComplexBall, w: Int) raises -> Optional[ComplexBall]:
    """atan of a finite rectangle off the branch cuts `i [1, inf)` and
    `-i [1, inf)`: one whose real part is certainly away from 0, or whose
    imaginary part lies certainly within `(-1, 1)`. Kahan's formulas (Branch
    cuts for complex elementary functions, 1987) at the midpoint `x + iy`:
    `Re = atan2(2x, 1 - x**2 - y**2) / 2` with that difference exact, and
    `Im = atanh(2y / (1 + x**2 + y**2)) / 2`, the quotient of exact numbers
    rounded once. One atan and one atanh where the logarithms took two of
    each. Both widen by `R / ((|m - i| - R)(|m + i| - R))` for `R = r_x + r_y`,
    since `atan'(z) = 1 / ((z - i)(z + i))`. None for other rectangles, or
    where a kernel declines."""
    ref x = value._real
    ref y = value._imag
    if not value.is_finite():
        return None
    var one = _Radius.power_of_two(0)
    var inside = _Radius.upper(y._midpoint).add(y._radius).compare(one) < 0
    if not (inside or x.certainly_positive() or x.certainly_negative()):
        return None
    var r = x._radius.add(y._radius)
    var bound = _Radius.zero()
    if not r.is_zero():
        # |m -+ i| = sqrt(x**2 + (y -+ 1)**2): of |y - 1| and |y + 1| one is at
        # least 1 and the other at least g = |1 - |y||, bounded below here.
        var lx = _Radius.lower(x._midpoint)
        var square = lx.multiply_down(lx)
        var high = _Radius.upper(y._midpoint)
        var low = _Radius.lower(y._midpoint)
        var g = one.sub_down(high) if high.compare(one) < 0 else low.sub_down(one)
        var near = square.add_down(g.multiply_down(g)).sqrt_down().sub_down(r)
        var far = square.add_down(one).sqrt_down().sub_down(r)
        if near.is_zero() or far.is_zero():
            return None
        bound = r.divide(near.multiply_down(far))
    var s = Float(_rounded=_float_operation(
        Float(_rounded=_float_operation(x._midpoint, x._midpoint, 2, _exact_context())),
        Float(_rounded=_float_operation(y._midpoint, y._midpoint, 2, _exact_context())), 0, _exact_context(),
    ))
    var twice_x = _twice(x._midpoint)
    var twice_y = _twice(y._midpoint)
    var angle = _atan2_point(twice_x, Float(_rounded=_float_operation(Integer(1), s, 1, _exact_context())), w)
    if not angle:
        return None
    var ratio = _quotient(
        _BallArgument(twice_y), _BallArgument(Float(_rounded=_float_operation(Integer(1), s, 0, _exact_context()))), _at(w + 16)
    )
    var stretch = _ball_function(_ATANH, _BallArgument(ratio), 0, _at(w))
    if not stretch.is_finite():
        return None
    var re = _scale2(_BallArgument(angle.value()), Integer(-1), _at(w))
    var im = _scale2(_BallArgument(stretch), Integer(-1), _at(w))
    re._radius = re._radius.add(bound)
    im._radius = im._radius.add(bound)
    return ComplexBall(_real=re, _imag=im)


def atan(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal arctangent, `(i/2)(log(1 - iz) - log(1 + iz))`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; indeterminate at +-i; on the cuts of
        the imaginary axis the counter-clockwise continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var off_cuts = _atan_off_cuts(value, w)
    if off_cuts:
        return off_cuts.value()
    var work = w + 16
    var c = _at(work)
    var iz = _times_i(value)
    var difference = _sub(_log(_sub(ComplexBall(1), iz, c), work), _log(_add(ComplexBall(1), iz, c), work), c)
    return _rounded(_half(_times_i(difference), work), w)


def asinh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal inverse hyperbolic sine, `log(z + sqrt(z**2 + 1))`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; on the cuts of the imaginary axis the
        counter-clockwise continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var c = _at(work)
    var root = _sqrt(_add(_mul(value, value, c), ComplexBall(1), c), work)
    return _rounded(_log(_add(value, root, c), work), w)


def acosh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal inverse hyperbolic cosine,
    `log(z + sqrt(z + 1) sqrt(z - 1))`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; on the cut `(-inf, 1)` the
        counter-clockwise continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    var work = w + 16
    var c = _at(work)
    var product = _mul(_sqrt(_add(value, ComplexBall(1), c), work), _sqrt(_sub(value, ComplexBall(1), c), work), c)
    return _rounded(_log(_add(value, product, c), work), w)


def atanh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The ball of the principal inverse hyperbolic tangent,
    `(log(1 + z) - log(1 - z)) / 2`.

    Args:
        value: The ball.
        context: The result precision; by default the ball's.

    Returns:
        A ball containing every value; indeterminate at +-1; on the cuts the
        counter-clockwise continuous value.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _bits(value, context)
    # atanh z = -i atan(iz), off atan's cuts: z off the real cuts (-inf, -1], [1, inf).
    var off_cuts = _atan_off_cuts(_times_i(value), w)
    if off_cuts:
        return _times_minus_i(off_cuts.value())
    var work = w + 16
    var c = _at(work)
    var difference = _sub(_log(_add(ComplexBall(1), value, c), work), _log(_sub(ComplexBall(1), value, c), work), c)
    return _rounded(_half(difference, work), w)

