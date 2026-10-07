"""Kernels of log, log1p, log2 and log10, and of asinh, acosh and atanh.

`log x = k ln 2 + log m` with `x = 2**k m`, `m` in `[1/sqrt 2, sqrt 2)`, so a
result near 0 comes only from `m` near 1 with `k = 0`. The core takes `r`
square roots of `m` (`r` about `isqrt(scale)/3`, fewer when `m` is already
near 1), then sums `log m_r = 2 atanh(t)`, `t = (m_r - 1)/(m_r + 1)`, until
`t**(2j+1)` falls below the last place, and multiplies by `2**r`. Near
`x = 1` the scale includes `-E(x - 1)` so the result keeps relative accuracy.

`log1p(x)` below `|x| = 1/2` is the core at `1 + x` held exactly at a scale
that includes `-E(x)`; above, the log of the exact `1 + x`. `log2` and
`log10` divide by ln 2 and ln 10. The inverse hyperbolic functions avoid
cancellation: `asinh y = log1p(y + y**2/(1 + sqrt(1 + y**2)))` below `y = 1/2`
and `log(y + sqrt(y**2 + 1))` above, for `y = |x|`; `acosh(1 + t) =
log1p(t + sqrt(t (t + 2)))` below `t = 1/2` and `log(x + sqrt(x**2 - 1))`
above; `atanh y = log1p(2y/(1 - y))/2` with the exact `1 - y`. Every kernel is
accurate to `w` bits (`c = 0`).
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float.status import NumericStatus
from ..float._arithmetic import _float_operation
from ..float._rounding import _RoundedBinary
from ..float._functions import _scale_float
from .context import BallContext
from ._radius import _Radius
from .value import Ball, _BallArgument, _FINITE
from ._arithmetic import _sum, _product, _quotient, _scale2, _sqrt, _directed, _from_record, _nearest, _negate
from ._fixed import _Fix, _FixedPoint, _Wide, _sum_series, _native_words
from ._constants import _ln2, _ln10
from ._exp import _bits, _isqrt, _guard, _ln2_fix
from ._medium import _log_medium


def _atan_divisor(k: Int) -> Int:
    """The divisors of the atan and atanh series, `2k + 1`."""
    return 2 * k + 1


def _log_series[F: _FixedPoint](m: F) raises -> F:
    """`log m` for `m` in `[1/2, 2]`, at m's scale; near `m = 1` the scale must
    hold `m - 1` to the relative accuracy wanted."""
    var scale = m.scale_of()
    var unit = F.of_int(1, scale)
    var near = m.sub(unit).magnitude_bits() - scale
    var r = max(0, min(_isqrt(scale) // 5, _isqrt(scale) // 5 + near + 3))
    var work = scale + r + 2 * _bits(scale) + 8
    var v = m.rescale(work)
    for _ in range(r):
        v = v.sqrt()
    var one = F.of_int(1, work)
    var t = v.sub(one).div(v.add(one))
    # log v = 2 atanh t = 2 t (1 + t**2/3 + t**4/5 + ...), with t**2 < 1/32.
    var total = t.mul(_sum_series[F, _atan_divisor, False, False](t.square(), "log"))
    return total.scale2(r + 1).rescale(scale)


def _log_core(m: _Fix) raises -> _Fix:
    """The series on native words when the scale fits, else on Integers."""
    var words = _native_words(m)
    if words == 4:
        return _log_series(_Wide[4].of_fix(m)).to_fix()
    if words == 8:
        return _log_series(_Wide[8].of_fix(m)).to_fix()
    return _log_series(m)


def _normalized_log(m: _Fix, k: Int) raises -> _Fix:
    """`log(m) + k ln 2` at m's scale, for `m` in `[1/2, 2]`."""
    var result = _log_core(m)
    if k == 0:
        return result^
    var wide = m.scale + _bits(abs(k)) + 4
    return result.rescale(wide).add(_ln2_fix(wide).mul_integer(Integer(k))).rescale(m.scale)


def _log_fix(x: _Fix) raises -> _Fix:
    """`log x` for a certainly positive x, at x's scale: `x = 2**k m` with `m`
    in `[1/sqrt 2, sqrt 2)`."""
    var scale = x.scale
    var k = x.mid.magnitude_bit_length() - scale - 1
    var m = _Fix(x.mid, x.err, scale + k).rescale(scale)
    # m is in [1, 2); halve it above sqrt 2.
    if m.mid * m.mid > Integer(1) << (2 * scale + 1):
        k += 1
        m = _Fix(m.mid, m.err, scale + 1).rescale(scale)
    return _normalized_log(m, k)


def _mantissa(x: Float, exponent: Int) raises -> Float:
    """`abs(x)` with its exponent replaced: in `[2**(exponent-1), 2**exponent)`."""
    return Float(_rounded=_RoundedBinary(1, False, x._significand, exponent, FloatFormat(x.precision()), NumericStatus()))


def _kernel_log(x: Float, w: Int) raises -> Ball:
    """log of a positive finite Float other than 1: the medium-precision
    kernel (`_medium.mojo`) below 4608 bits, else this one."""
    var medium = _log_medium(x, w + 8)
    if medium:
        return medium.value()
    var scale = _guard(w)
    var k = x._exponent - 1
    var mantissa = _mantissa(x, 1)
    # Above sqrt 2, take m in [1/sqrt 2, 1) instead.
    var p = x.precision()
    if x._significand * x._significand >= Integer(1) << (2 * p - 1):
        k += 1
        mantissa = _mantissa(x, 0)
    if k == 0:
        var d = Float(_rounded=_float_operation(x, Integer(1), 1, _exact_context()))
        if d._exponent < 0:
            scale -= d._exponent
    return _normalized_log(_Fix.of_float(mantissa, scale), k).to_ball(w + 8)


def _log1p_point(x: Float, w: Int) raises -> Optional[Ball]:
    """log1p of a finite nonzero Float above -1 at `w` bits from the medium
    kernel: the log of the exact `1 + x`, whose closeness to 1 the kernel
    pays for with extra bits; below `2**-(w/2)`, `x - x**2/2`, the series'
    later terms adding at most `|x|**3` for `|x| <= 1/2`. None where the
    kernel declines."""
    if x._exponent < -(w // 2):
        var half_square = Float(_rounded=_float_operation(x, x, 2, _exact_context()))
        half_square = Float(_rounded=_scale_float(half_square, Integer(-1), _exact_context()))
        var cube = _Radius.upper(x)
        return _from_record(_float_operation(x, half_square, 1, _nearest(w)), cube.multiply(cube).multiply(cube), w)
    return _log_medium(Float(_rounded=_float_operation(Integer(1), x, 0, _exact_context())), w)


def _log1p_of(q: Ball, w: Int) raises -> Optional[Ball]:
    """log1p of a ball with a positive midpoint and a radius below 2**-17 of
    it, at `w` bits: the kernel at the midpoint, widened by
    `r / (1 + m - r) <= r / max(1 - r, m - r)`."""
    var value = _log1p_point(q._midpoint, w)
    if not value or q._radius.is_zero():
        return value^
    var one = _Radius.power_of_two(0)
    var below = one.sub_down(q._radius).max(_Radius.lower(q._midpoint).sub_down(q._radius))
    var ball = value.value()
    ball._radius = ball._radius.add(q._radius.divide(below))
    return ball^


def _at(w: Int) raises -> Optional[BallContext]:
    return Optional[BallContext](BallContext(w))


def _atanh_medium(y: Float, w: Int) raises -> Optional[Ball]:
    """atanh of `0 < y < 1` as `log1p(2 y / (1 - y)) / 2`: the quotient of
    exact numbers rounds once, and log1p has no cancellation."""
    var d = Float(_rounded=_float_operation(Integer(1), y, 1, _exact_context()))
    var twice = Float(_rounded=_scale_float(y, Integer(1), _exact_context()))
    var value = _log1p_of(_quotient(_BallArgument(twice), _BallArgument(d), _at(w + 16)), w + 8)
    if not value:
        return None
    return _scale2(_BallArgument(value.value()), Integer(-1), _at(w + 12))


def _asinh_medium(y: Float, w: Int) raises -> Optional[Ball]:
    """asinh of `y > 0` as `log1p(y + y**2 / (1 + sqrt(1 + y**2)))`, which
    has no cancellation; `y**2` and `1 + y**2` are exact."""
    var c = _at(w + 16)
    var square = Float(_rounded=_float_operation(y, y, 2, _exact_context()))
    var root = _sqrt(_BallArgument(Float(_rounded=_float_operation(Integer(1), square, 0, _exact_context()))), c)
    var ratio = _quotient(_BallArgument(square), _BallArgument(_sum(_BallArgument(Integer(1)), _BallArgument(root), False, c)), c)
    return _log1p_of(_sum(_BallArgument(y), _BallArgument(ratio), False, c), w + 8)


def _acosh_medium(x: Float, w: Int) raises -> Optional[Ball]:
    """acosh of `x > 1` as `log1p(t + sqrt(t (t + 2)))` with the exact
    `t = x - 1`, which has no cancellation."""
    var c = _at(w + 16)
    var t = Float(_rounded=_float_operation(x, Integer(1), 1, _exact_context()))
    var u = Float(_rounded=_float_operation(t, Float(_rounded=_float_operation(t, Integer(2), 0, _exact_context())), 2, _exact_context()))
    return _log1p_of(_sum(_BallArgument(t), _BallArgument(_sqrt(_BallArgument(u), c)), False, c), w + 8)


def _kernel_log1p(x: Float, w: Int) raises -> Ball:
    """log1p of a finite nonzero Float above -1."""
    var medium = _log1p_point(x, w + 8)
    if medium:
        return medium.value()
    if x._exponent >= 0:
        return _kernel_log(Float(_rounded=_float_operation(x, Integer(1), 0, _exact_context())), w)
    var scale = _guard(w) - x._exponent
    var m = _Fix.exact(Integer(1), scale).add(_Fix.of_float(x, scale))
    return _log_core(m).to_ball(w + 8)


def _kernel_log2(x: Float, w: Int) raises -> Ball:
    """log2 of a positive Float that is not a power of 2."""
    var c = Optional[BallContext](BallContext(w + 8))
    return _quotient(_BallArgument(_kernel_log(x, w + 4)), _BallArgument(_ln2(w + 8)), c)


def _kernel_log10(x: Float, w: Int) raises -> Ball:
    """log10 of a positive Float that is not a power of 10."""
    var c = Optional[BallContext](BallContext(w + 8))
    return _quotient(_BallArgument(_kernel_log(x, w + 4)), _BallArgument(_ln10(w + 8)), c)


def _log_ball(x: Ball, w: Int) raises -> Ball:
    """log of a finite ball certainly above 0, from its midpoint: the radius
    grows by `r / (m - r)`; indeterminate if the ball reaches 0."""
    var mid = x._midpoint
    if mid.is_zero() or mid._negative:
        return Ball.indeterminate(w + 8)
    var center: Ball
    if mid == Float(1):
        center = Ball(Integer(0), precision=w + 8)
    else:
        center = _kernel_log(mid, w)
    if x._radius.is_zero():
        return center^
    var gap = Float(_rounded=_float_operation(mid, x._radius.to_float(), 1, _directed(30, False)))
    if gap.is_zero() or gap._negative:
        return Ball.indeterminate(w + 8)
    return Ball(_midpoint=center._midpoint, _radius=center._radius.add(x._radius.divide(_Radius.lower(gap))), _kind=_FINITE)


def _kernel_asinh(x: Float, w: Int) raises -> Ball:
    """asinh of a finite nonzero Float."""
    var y = abs(x)
    var result: Ball
    # Below 1/2, log(y + sqrt(1 + y**2)) would cancel: log1p takes it.
    var medium = _asinh_medium(y, w) if y._exponent < 0 else Optional[Ball](None)
    if medium:
        result = medium.value()
    elif y._exponent < 0:
        var scale = _guard(w) - y._exponent
        var v = _Fix.of_float(y, scale)
        var one = _Fix.exact(Integer(1), scale)
        var square = v.square()
        var z = v.add(square.div(one.add(one.add(square).sqrt())))
        result = _log_core(one.add(z)).to_ball(w + 8)
    else:
        var c = Optional[BallContext](BallContext(w + 16))
        var root = _sqrt(_BallArgument(_sum(_BallArgument(_product(_BallArgument(y), _BallArgument(y), c)), _BallArgument(Integer(1)), False, c)), c)
        result = _log_ball(_sum(_BallArgument(y), _BallArgument(root), False, c), w)
    return _negate(result) if x._negative else result^


def _kernel_acosh(x: Float, w: Int) raises -> Ball:
    """acosh of a finite Float above 1."""
    var medium = _acosh_medium(x, w)
    if medium:
        return medium.value()
    var t = Float(_rounded=_float_operation(x, Integer(1), 1, _exact_context()))
    if t._exponent < 0:
        var scale = _guard(w) - t._exponent
        var u = _Fix.of_float(t, scale)
        var z = u.add(u.mul(u.add(_Fix.exact(Integer(2), scale))).sqrt())
        return _log_fix(_Fix.exact(Integer(1), scale).add(z)).to_ball(w + 8)
    var c = Optional[BallContext](BallContext(w + 16))
    var root = _sqrt(_BallArgument(_sum(_BallArgument(_product(_BallArgument(x), _BallArgument(x), c)), _BallArgument(Integer(1)), True, c)), c)
    return _log_ball(_sum(_BallArgument(x), _BallArgument(root), False, c), w)


def _kernel_atanh(x: Float, w: Int) raises -> Ball:
    """atanh of a finite nonzero Float of magnitude below 1."""
    var y = abs(x)
    # From 1/2 the log of the exact ratio below, at least 3, does not cancel.
    var medium = _atanh_medium(y, w) if y._exponent < 0 else Optional[Ball](None)
    if medium:
        var half = medium.value()
        return _negate(half) if x._negative else half^
    var d = Float(_rounded=_float_operation(Integer(1), y, 1, _exact_context()))
    var result: Ball
    if y._exponent < 0:
        var scale = _guard(w) - y._exponent
        var z = _Fix.of_float(y, scale).scale2(1).div(_Fix.of_float(d, scale))
        result = _log_fix(_Fix.exact(Integer(1), scale).add(z)).to_ball(w + 8)
    else:
        # (1 + y) / (1 - y) >= 3: the log of an exact ratio.
        var c = Optional[BallContext](BallContext(w + 16))
        var ratio = _quotient(_BallArgument(_sum(_BallArgument(Integer(1)), _BallArgument(y), False, c)), _BallArgument(d), c)
        result = _log_ball(ratio, w)
    var half = _scale2(_BallArgument(result), Integer(-1), Optional[BallContext](BallContext(w + 8)))
    return _negate(half) if x._negative else half^
