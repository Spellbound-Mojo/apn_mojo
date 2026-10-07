"""Kernels of sin, cos and tan, and of atan, asin, acos and atan2.

`x = k pi/2 + r` with `|r| <= pi/4`, from `pi` at `scale + E(x)` bits. When
`x` lies near a multiple of `pi/2`, `r` loses relative accuracy; the reduction
then retries with more bits of `pi` until `r` has enough, and gives up past the
budget, as `sin(2**(10**6))` must. The core halves `r` `s` times, sums the
series of `sin` by rectangular splitting, takes `cos = sqrt(1 - sin**2)`, and
doubles with `sin 2a = 2 sin a cos a` and `cos 2a = 1 - 2 sin**2 a`; the
quadrant `k mod 4` picks signs and roles. Below 4608 bits, the
medium-precision kernel (`_medium.mojo`) takes the argument first and leaves this one the cases it
declines: huge arguments, and those near a zero of the sine or cosine.
`tan` divides the two balls.

`atan(x) = pi/2 - atan(1/x)` above 1; the core halves `s` times with
`atan y = 2 atan(y / (1 + sqrt(1 + y**2)))` and sums the alternating series.
Below 4608 bits the medium-precision kernel (`_medium.mojo`) takes `atan` of
a Float first.
`asin x = atan(x / sqrt(1 - x**2))` below `1/sqrt 2` and
`pi/2 - atan(sqrt(1 - x**2) / x)` above, with `sqrt(1 - x)` taken from the
exact `1 - x`; `acos x = 2 atan(sqrt((1 - x)/(1 + x)))` for `x > 0` and
`pi/2 + asin |x|` for `x < 0`. `atan2(y, x)` reduces to `atan` of a ratio at
most 1, then places the quadrant. Kernels are accurate to `w` bits (`c = 0`).
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float.status import NumericStatus
from ..float._arithmetic import _float_operation
from ..float._rounding import _RoundedBinary
from .context import BallContext
from ._radius import _Radius
from .value import Ball, _BallArgument
from ._arithmetic import _sum, _quotient, _scale2, _sqrt, _negate
from ._fixed import _Fix, _FixedPoint, _Wide, _sum_series, _native_words
from ._constants import _pi, _table_fix, _TABLE_BITS
from ._tables import _QUARTER_PI_VALUE
from ._medium import _sin_cos_medium, _atan_medium
from ._exp import _bits, _isqrt, _guard
from ._log import _mantissa, _atan_divisor


def _pi_fix(scale: Int) raises -> _Fix:
    """pi at a fixed-point scale, within one unit up to 4606 bits."""
    if scale + 2 <= _TABLE_BITS:
        return _table_fix(_QUARTER_PI_VALUE, 2, scale)
    return _Fix.of_ball(_pi(scale + 8), scale)


def _sin_divisor(k: Int) -> Int:
    return (2 * k + 2) * (2 * k + 3)


def _sin_cos_series[F: _FixedPoint](r: F) raises -> Tuple[F, F]:
    """`(sin r, cos r)` for `|r| <= 1`, at r's scale; the scale must hold r to
    the relative accuracy wanted for `sin`."""
    var scale = r.scale_of()
    var magnitude = r.magnitude_bits() - scale
    var s = max(0, min(_isqrt(scale) // 4, _isqrt(scale) // 4 + magnitude + 2))
    var work = scale + 2 * s + 2 * _bits(scale) + 8
    var shifted = r.rescale(work)
    var y = shifted.relabel(work + s).rescale(work)
    var one = F.of_int(1, work)
    # sin y = y (1 - y**2/3! + y**4/5! - ...); cos y = sqrt(1 - sin**2 y) >= 1/2 for |y| <= 1.
    var sine = y.mul(_sum_series[F, _sin_divisor, True, True](y.square(), "sin and cos"))
    var cosine = one.sub(sine.square()).sqrt()
    for _ in range(s):
        var doubled = sine.mul(cosine).scale2(1)
        cosine = one.sub(sine.square().scale2(1))
        sine = doubled
    return (sine.rescale(scale), cosine.rescale(scale))


def _sin_cos_core(r: _Fix) raises -> Tuple[_Fix, _Fix]:
    """The series on native words when the scale fits, else on Integers."""
    var words = _native_words(r)
    if words == 4:
        var pair = _sin_cos_series(_Wide[4].of_fix(r))
        return (pair[0].to_fix(), pair[1].to_fix())
    if words == 8:
        var pair = _sin_cos_series(_Wide[8].of_fix(r))
        return (pair[0].to_fix(), pair[1].to_fix())
    return _sin_cos_series(r)



def _atan_series[F: _FixedPoint](t: F) raises -> F:
    """`atan t` for `|t| <= 1`, at t's scale; the scale must hold t to the
    relative accuracy wanted."""
    var scale = t.scale_of()
    var magnitude = t.magnitude_bits() - scale
    # At least one halving unless |t| < 1/8, so that y**2 <= 1/2 below.
    var s = max(0, min(_isqrt(scale) // 5, _isqrt(scale) // 5 + magnitude + 2))
    var work = scale + s + 2 * _bits(scale) + 8
    var y = t.rescale(work)
    var one = F.of_int(1, work)
    for _ in range(s):
        y = y.div(one.add(one.add(y.square()).sqrt()))
    # atan y = y (1 - y**2/3 + y**4/5 - ...)
    var total = y.mul(_sum_series[F, _atan_divisor, False, True](y.square(), "atan"))
    return total.scale2(s).rescale(scale)


def _atan_core(t: _Fix) raises -> _Fix:
    """The series on native words when the scale fits, else on Integers."""
    var words = _native_words(t)
    if words == 4:
        return _atan_series(_Wide[4].of_fix(t)).to_fix()
    if words == 8:
        return _atan_series(_Wide[8].of_fix(t)).to_fix()
    return _atan_series(t)


@fieldwise_init
struct _Reduced(ImplicitlyCopyable):
    var r: _Fix
    var quadrant: Int


def _reduce_half_pi(x: Float, w: Int, budget: Int) raises -> Optional[_Reduced]:
    """`r = x - k pi/2`, `|r| <= pi/4`, at a scale holding r to `w` bits of
    relative accuracy, with `k mod 4`; None when that needs more bits of pi than
    the budget."""
    var scale = _guard(w)
    if x._exponent <= -1:
        return _Reduced(_Fix.of_float(x, scale - x._exponent), 0)
    var extra = 16
    while True:
        var wide = scale + x._exponent + extra
        if wide > budget:
            return None
        var half_pi = _pi_fix(wide).scale2(-1)
        var value = _Fix.of_float(x, wide)
        var q = value.div(half_pi)
        var k = (q.mid + (Integer(1) << (wide - 1))) >> wide
        var r = value.sub(half_pi.mul_integer(k))
        var accuracy = r.mid.magnitude_bit_length() - r.err.ceiling().magnitude_bit_length()
        if accuracy >= scale - 8:
            var magnitude = r.mid.magnitude_bit_length() - wide
            return _Reduced(r.rescale(scale - min(0, magnitude)), Int(k % 4))
        extra *= 2


def _kernel_sin_cos(x: Float, w: Int, budget: Int) raises -> Tuple[Ball, Ball]:
    """`(sin x, cos x)` of a finite nonzero Float; indeterminate past the budget."""
    var medium = _sin_cos_medium(x, _Radius.zero(), w + 8)
    if medium:
        return medium.take()
    var reduced = _reduce_half_pi(x, w, budget)
    if not reduced:
        return (Ball.indeterminate(w + 8), Ball.indeterminate(w + 8))
    var pair = _sin_cos_core(reduced.value().r)
    var s = pair[0].to_ball(w + 8)
    var c = pair[1].to_ball(w + 8)
    var quadrant = reduced.value().quadrant
    if quadrant == 0:
        return (s, c)
    if quadrant == 1:
        return (c, _negate(s))
    if quadrant == 2:
        return (_negate(s), _negate(c))
    return (_negate(c), s)


def _kernel_tan(x: Float, w: Int, budget: Int) raises -> Ball:
    """tan of a finite nonzero Float; indeterminate past the budget."""
    var pair = _kernel_sin_cos(x, w + 4, budget)
    if pair[0].is_indeterminate():
        return pair[0]
    return _quotient(_BallArgument(pair[0]), _BallArgument(pair[1]), Optional[BallContext](BallContext(w + 8)))


def _half_pi_minus(t: _Fix) raises -> _Fix:
    """`pi/2 - atan(t)` for `|t| <= 1`, at t's scale."""
    return _pi_fix(t.scale).scale2(-1).sub(_atan_core(t))


def _kernel_atan(x: Float, w: Int) raises -> Ball:
    """atan of a finite nonzero Float."""
    var medium = _atan_medium(x, w + 8)
    if medium:
        return medium.value()
    var y = abs(x)
    var scale = _guard(w)
    var result: Ball
    if y._exponent <= 0:
        result = _atan_core(_Fix.of_float(y, scale - y._exponent)).to_ball(w + 8)
    elif y._exponent > scale + 8:
        # atan(1/y) < 2**(1 - E(y)) is below the last place: pi/2, widened.
        var half_pi = _pi_fix(scale).scale2(-1)
        result = half_pi.add_error(_Radius.power_of_two(scale + 1 - y._exponent).add(_Radius.power_of_two(0))).to_ball(w + 8)
    else:
        var t = _Fix.exact(Integer(1), scale).div(_Fix.of_float(y, scale))
        result = _half_pi_minus(t).to_ball(w + 8)
    return _negate(result) if x._negative else result^


def _atan_of(t: Ball, w: Int) raises -> Optional[Ball]:
    """atan of a ball with a midpoint in (0, 2] at `w` bits: the medium
    kernel at the midpoint, widened by the radius, since `atan' <= 1`. None
    where the kernel declines."""
    var value = _atan_medium(t._midpoint, w)
    if not value or t._radius.is_zero():
        return value^
    var ball = value.value()
    ball._radius = ball._radius.add(t._radius)
    return ball^


def _complement_root(y: Float, w: Int) raises -> Ball:
    """`sqrt(1 - y**2)` for `0 < y < 1` at `w` bits, from the exact
    `(1 - y)(1 + y)`."""
    var below = Float(_rounded=_float_operation(Integer(1), y, 1, _exact_context()))
    var above = Float(_rounded=_float_operation(Integer(1), y, 0, _exact_context()))
    var d = Float(_rounded=_float_operation(below, above, 2, _exact_context()))
    return _sqrt(_BallArgument(d), Optional[BallContext](BallContext(w)))


def _square_below_half(y: Float) raises -> Bool:
    """Whether `y**2 < 1/2` for `0 < y < 1`."""
    if y._exponent < 0:
        return True
    var p = y.precision()
    return y._significand * y._significand < Integer(1) << (2 * p - 1)


def _inverse_sine_medium(x: Float, w: Int, cosine: Bool) raises -> Optional[Ball]:
    """asin or acos of a nonzero x in (-1, 1) from the medium atan kernel, with
    `s = sqrt(1 - y**2)` for `y = |x|` and every atan argument at most 1:
    asin y is `atan(y / s)` for `y**2 < 1/2`, else `pi/2 - atan(s / y)`; acos y
    is `atan(s / y)` for `y**2 >= 1/2`, else `pi/2 - atan(y / s)`; and
    `asin(-y) = -asin y`, `acos(-y) = pi - acos y`. None where the kernel
    declines."""
    var y = abs(x)
    var c = Optional[BallContext](BallContext(w + 16))
    var s = _complement_root(y, w + 16)
    var below = _square_below_half(y)
    # The atan argument of each branch: y / s when y**2 < 1/2, else s / y.
    var angle = _atan_of(
        _quotient(_BallArgument(y), _BallArgument(s), c) if below else _quotient(_BallArgument(s), _BallArgument(y), c),
        w + 8,
    )
    if not angle:
        return None
    var result = angle.value()
    if below == cosine:
        # pi/2 minus the angle: acos below, asin above.
        var half_pi = _scale2(_BallArgument(_pi(w + 16)), Integer(-1), c)
        result = _sum(_BallArgument(half_pi), _BallArgument(result), True, c)
    if x._negative:
        if not cosine:
            return _negate(result)
        return _sum(_BallArgument(_pi(w + 16)), _BallArgument(result), True, c)
    return result^


def _kernel_asin(x: Float, w: Int) raises -> Ball:
    """asin of a finite nonzero Float of magnitude at most 1."""
    var y = abs(x)
    if y < Float(1):
        var medium = _inverse_sine_medium(x, w, False)
        if medium:
            return medium.value()
    var scale = _guard(w)
    var result: _Fix
    if y == Float(1):
        result = _pi_fix(scale).scale2(-1)
    elif y._exponent < 0:
        scale -= y._exponent
        var v = _Fix.of_float(y, scale)
        var root = _Fix.exact(Integer(1), scale).sub(v.square()).sqrt()
        result = _atan_core(v.div(root))
    else:
        var d = Float(_rounded=_float_operation(Integer(1), y, 1, _exact_context()))
        var narrow = scale + max(0, -d._exponent) + 4
        var below = _Fix.of_float(d, narrow).sqrt().rescale(scale)
        var above = _Fix.of_float(Float(_rounded=_float_operation(Integer(1), y, 0, _exact_context())), scale).sqrt()
        var root = below.mul(above)
        var v = _Fix.of_float(y, scale)
        var p = y.precision()
        if y._significand * y._significand >= Integer(1) << (2 * p - 1):
            result = _half_pi_minus(root.div(v))
        else:
            result = _atan_core(v.div(root))
    var ball = result.to_ball(w + 8)
    return _negate(ball) if x._negative else ball^


def _kernel_acos(x: Float, w: Int) raises -> Ball:
    """acos of a finite Float in `[-1, 1)`."""
    if not x.is_zero() and abs(x) < Float(1):
        var medium = _inverse_sine_medium(x, w, True)
        if medium:
            return medium.value()
    var scale = _guard(w)
    if x.is_zero():
        return _pi_fix(scale).scale2(-1).to_ball(w + 8)
    if x._negative:
        if x == Float(-1):
            return _pi_fix(scale).to_ball(w + 8)
        var c = Optional[BallContext](BallContext(w + 8))
        var half_pi = _pi_fix(scale).scale2(-1).to_ball(w + 12)
        return _sum(_BallArgument(half_pi), _BallArgument(_kernel_asin(abs(x), w + 4)), False, c)
    var d = Float(_rounded=_float_operation(Integer(1), x, 1, _exact_context()))
    var narrow = scale + max(0, -d._exponent) + 4
    var sum = _Fix.of_float(Float(_rounded=_float_operation(Integer(1), x, 0, _exact_context())), narrow)
    var t = _Fix.of_float(d, narrow).div(sum).sqrt()
    return _atan_core(t).scale2(1).to_ball(w + 8)


def _atan2_point(y: Float, x: Float, w: Int) raises -> Optional[Ball]:
    """atan2(y, x) at `w` bits for finite Floats, not both 0, from the medium
    atan kernel. With `t = min(|x|, |y|) / max(|x|, |y|)`, rounded once at
    `w + 8` bits, the angle is `atan t` when `|y| < |x|` (no other rounding)
    and `pi/2 - atan t` when `|y| > |x|`. Then `pi` minus it for `x < 0`, and
    the sign of y. A y of 0 gives 0, or `pi` for `x < 0` (the
    counter-clockwise value on the cut). None where the kernel declines."""
    var c = Optional[BallContext](BallContext(w))
    var ay = abs(y)
    var ax = abs(x)
    var angle: Ball
    var below = not x.is_zero() and ay < ax
    # pi is read from its table once, and only where a branch needs it.
    var pi = _pi(w + 16) if x._negative or not below else Ball._placeholder()
    if y.is_zero():
        # Returned before the sign of y, which a zero's sign must not flip.
        if not x._negative:
            return Ball(Integer(0), precision=w)
        return _scale2(_BallArgument(pi), Integer(0), c)
    if x.is_zero() or ay == ax:
        # pi/2, or pi/4 on the diagonal.
        angle = _scale2(_BallArgument(pi), Integer(-1 if x.is_zero() else -2), c)
    else:
        var t = _quotient(
            _BallArgument(ay if below else ax), _BallArgument(ax if below else ay), Optional[BallContext](BallContext(w + 8))
        )
        var a = _atan_of(t, w)
        if not a:
            return None
        if below:
            angle = a.value()
        else:
            angle = _sum(_BallArgument(_scale2(_BallArgument(pi), Integer(-1), c)), _BallArgument(a.value()), True, c)
    if x._negative:
        angle = _sum(_BallArgument(pi), _BallArgument(angle), True, c)
    return _negate(angle) if y._negative else angle^


def _kernel_atan2(y: Float, x: Float, w: Int) raises -> Ball:
    """atan2(y, x) for finite nonzero Floats."""
    var medium = _atan2_point(y, x, w + 8)
    if medium:
        return medium.value()
    var ay = abs(y)
    var ax = abs(x)
    var scale = _guard(w)
    var high = ay if ay > ax else ax
    # Scale both by 2**-E(high), so the larger lies in [1/2, 1).
    var shift = high._exponent
    var sy = _mantissa(ay, ay._exponent - shift)
    var sx = _mantissa(ax, ax._exponent - shift)
    var angle: _Fix
    if ay <= ax:
        var gap = ay._exponent - ax._exponent
        var s = scale - min(0, gap)
        angle = _atan_core(_Fix.of_float(sy, s).div(_Fix.of_float(sx, s)))
    else:
        angle = _half_pi_minus(_Fix.of_float(sx, scale).div(_Fix.of_float(sy, scale)))
    if x._negative:
        angle = _pi_fix(angle.scale).sub(angle)
    var ball = angle.to_ball(w + 8)
    return _negate(ball) if y._negative else ball^
