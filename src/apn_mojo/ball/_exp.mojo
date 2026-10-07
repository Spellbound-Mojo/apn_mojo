"""Kernels of exp, expm1 and exp2, and of sinh, cosh and tanh.

`exp(x) = 2**k exp(r)` with `r = x - k ln 2`, `|r| <= ln 2 / 2`. The core
divides `r` by `2**s` with `s = 2 isqrt(scale) / 3`, sums the Taylor series by
rectangular splitting (`_sum_series`) up to the first term below half a unit,
with twice that term's bound for the tail, then squares `s` times. The fixed-point scale carries `s` extra bits for the error
the squarings double. `exp2(x) = 2**k exp(f ln 2)` with `x = k + f` exactly.

`expm1` keeps relative accuracy near 0: below `|x| = 1/2` its core sums
`expm1(y) = y (1 + y/2! + ...)` for `y = x / 2**s` and doubles with `expm1(2a) = expm1(a)(expm1(a) + 2)`,
whose relative error does not grow; the scale includes `-E(x)` so a tiny
argument keeps its bits. The hyperbolic functions use these two kernels:
`sinh y = (m + m/(m + 1))/2` and `tanh y = m/(m + 2)` with `m = expm1(y)` or
`expm1(2y)` for `y = |x|`, and `cosh y = (e + 1/e)/2` with `e = exp(y)`, so
nothing cancels. Every kernel is accurate to `w` bits (`c = 0`).
"""

from std.bit import count_leading_zeros
from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float._arithmetic import _float_operation
from .context import BallContext
from ._radius import _Radius, _top_bits
from .value import Ball, _BallArgument
from ._arithmetic import _sum, _product, _quotient, _scale2, _hull
from ._fixed import _Fix, _FixedPoint, _Wide, _sum_series, _native_words
from ._constants import _ln2, _table_fix, _TABLE_BITS
from ._tables import _LN2_VALUE
from ._medium import _exp_medium, _hex_limb


def _bits(n: Int) -> Int:
    return 64 - Int(count_leading_zeros(UInt64(n)))


def _isqrt(n: Int) -> Int:
    var r = 0
    while (r + 1) * (r + 1) <= n:
        r += 1
    return r


def _guard(w: Int) -> Int:
    """The fixed-point scale of a kernel accurate to `w` bits."""
    return w + 2 * _bits(w) + 16


def _ln2_fix(scale: Int) raises -> _Fix:
    """ln 2 at a fixed-point scale, within one unit up to 4608 bits."""
    if scale <= _TABLE_BITS:
        return _table_fix(_LN2_VALUE, 0, scale)
    return _Fix.of_ball(_ln2(scale + 8), scale)


def _exp_divisor(k: Int) -> Int:
    return k + 1


def _expm1_divisor(k: Int) -> Int:
    return k + 2


def _exp_series[F: _FixedPoint](r: F) raises -> F:
    """`exp(r)` for `|r| <= 1`, at r's scale, within a few units."""
    var scale = r.scale_of()
    var s = 2 * _isqrt(scale) // 3
    var work = scale + s + 2 * _bits(scale) + 8
    # y = r / 2**s: the same midpoint read at a scale s larger.
    var shifted = r.rescale(work)
    var y = shifted.relabel(work + s).rescale(work)
    var total = _sum_series[F, _exp_divisor, True, False](y, "exp")
    for _ in range(s):
        total = total.square()
    return total.rescale(scale)


def _exp_core(r: _Fix) raises -> _Fix:
    """The series on native words when the scale fits, else on Integers."""
    var words = _native_words(r)
    if words == 4:
        return _exp_series(_Wide[4].of_fix(r)).to_fix()
    if words == 8:
        return _exp_series(_Wide[8].of_fix(r)).to_fix()
    return _exp_series(r)


def _expm1_series[F: _FixedPoint](r: F) raises -> F:
    """`expm1(r)` for `|r| <= 1/2`, at r's scale; the scale must hold r to
    the relative accuracy wanted."""
    var scale = r.scale_of()
    var magnitude = r.magnitude_bits() - scale
    # Halve only as far as the series needs: a tiny r is already small.
    var s = max(0, min(_isqrt(scale) // 3, _isqrt(scale) // 3 + magnitude + 2))
    var work = scale + s + 2 * _bits(scale) + 8
    var shifted = r.rescale(work)
    var y = shifted.relabel(work + s).rescale(work)
    # expm1(y) = y (1 + y/2! + y**2/3! + ...)
    var total = y.mul(_sum_series[F, _expm1_divisor, True, False](y, "expm1"))
    var two = F.of_int(2, work)
    for _ in range(s):
        total = total.mul(total.add(two))
    return total.rescale(scale)


def _expm1_core(r: _Fix) raises -> _Fix:
    """The series on native words when the scale fits, else on Integers."""
    var words = _native_words(r)
    if words == 4:
        return _expm1_series(_Wide[4].of_fix(r)).to_fix()
    if words == 8:
        return _expm1_series(_Wide[8].of_fix(r)).to_fix()
    return _expm1_series(r)


def _kernel_exp(x: Float, w: Int) raises -> Ball:
    """exp of a finite Float whose result exponent stays within about
    `2**62`: the medium-precision kernel (`_medium.mojo`) below 4608 bits,
    else this one."""
    if not x.is_zero():
        var medium = _exp_medium(x, w + 8, False)
        if medium:
            return medium.value()
    var scale = _guard(w)
    var k = Integer(0)
    if x._exponent > 40:
        # k = round(x / ln 2): any integer near it keeps |r| small.
        var rough = x._exponent + 24
        var q = _Fix.of_float(x, rough).div(_ln2_fix(rough))
        k = (q.mid + (Integer(1) << (rough - 1))) >> rough
    elif x._exponent >= 0:
        # Below 2**40, x's top word over ln 2's is within 2**-20 of x / ln 2:
        # x is near top * 2**-shift, and ln 2 near L * 2**-64.
        var bits = x._significand.magnitude_bit_length()
        var top = _top_bits(x._significand, bits) if bits > 64 else x._significand._low_magnitude() << UInt64(64 - bits)
        var shift = 64 + x.precision() - bits - x._exponent
        var quotient = (UInt128(top) << 64) // UInt128(_hex_limb(_LN2_VALUE, 0))
        var nearest = Int((quotient + (UInt128(1) << UInt128(shift - 1))) >> UInt128(shift))
        k = Integer(-nearest if x._negative else nearest)
    var wide = scale + k.magnitude_bit_length() + 4
    var r = _Fix.of_float(x, wide)
    if k:
        r = r.sub(_ln2_fix(wide).mul_integer(k))
    var y = _exp_core(r.rescale(scale)).to_ball(w + 8)
    if not k:
        return y^
    return _scale2(_BallArgument(y), k, Optional[BallContext](BallContext(w + 8)))


def _kernel_exp2(x: Float, w: Int) raises -> Ball:
    """2**x of a finite non-integral Float."""
    var scale = _guard(w)
    var k = x.round()
    var f = Float(_rounded=_float_operation(x, k, 1, _exact_context()))
    var r = _Fix.of_float(f, scale + 4).mul(_ln2_fix(scale + 4)).rescale(scale)
    var y = _exp_core(r).to_ball(w + 8)
    return _scale2(_BallArgument(y), k, Optional[BallContext](BallContext(w + 8)))


def _kernel_expm1(x: Float, w: Int) raises -> Ball:
    """expm1 of a finite nonzero Float."""
    var medium = _exp_medium(x, w + 8, True)
    if medium:
        return medium.value()
    if x._exponent >= 0:
        # |x| >= 1/2: exp(x) - 1 cancels at most 2 bits.
        var c = Optional[BallContext](BallContext(w + 8))
        return _sum(_BallArgument(_kernel_exp(x, w + 4)), _BallArgument(Integer(1)), True, c)
    var scale = _guard(w) - x._exponent
    return _expm1_core(_Fix.of_float(x, scale)).to_ball(w + 8)


def _kernel_sinh(x: Float, w: Int) raises -> Ball:
    """sinh of a finite nonzero Float: sign(x) (m + m/(m + 1))/2 with
    `m = expm1(|x|)`."""
    var y = abs(x)
    var c = Optional[BallContext](BallContext(w + 8))
    var m = _BallArgument(_kernel_expm1(y, w + 6))
    var ratio = _quotient(m, _BallArgument(_sum(m, _BallArgument(Integer(1)), False, c)), c)
    var result = _scale2(_BallArgument(_sum(m, _BallArgument(ratio), False, c)), Integer(-1), c)
    if x._negative:
        return Ball(_midpoint=-result._midpoint, _radius=result._radius, _kind=result._kind)
    return result^


def _kernel_cosh(x: Float, w: Int) raises -> Ball:
    """cosh of a finite nonzero Float: (e + 1/e)/2 with `e = exp(|x|)`."""
    var c = Optional[BallContext](BallContext(w + 8))
    var e = _BallArgument(_kernel_exp(abs(x), w + 6))
    var inverse = _quotient(_BallArgument(Integer(1)), e, c)
    return _scale2(_BallArgument(_sum(e, _BallArgument(inverse), False, c)), Integer(-1), c)


def _kernel_tanh(x: Float, w: Int) raises -> Ball:
    """tanh of a finite nonzero Float: sign(x) m/(m + 2) with
    `m = expm1(2|x|)`."""
    var c = Optional[BallContext](BallContext(w + 8))
    var double = Float(_rounded=_float_operation(abs(x), abs(x), 0, _exact_context()))
    var m = _BallArgument(_kernel_expm1(double, w + 6))
    var result = _quotient(m, _BallArgument(_sum(m, _BallArgument(Integer(2)), False, c)), c)
    if x._negative:
        return Ball(_midpoint=-result._midpoint, _radius=result._radius, _kind=result._kind)
    return result^


def _exp_ball(x: Ball, w: Int) raises -> Ball:
    """exp of a ball: the kernel at the midpoint, widened by
    `upper(exp m) (exp(r) - 1) <= upper(exp m) 2r` for `r <= 1/2`; for a wider
    ball, the kernel at its exact ends, since exp increases."""
    if x.is_indeterminate():
        return Ball.indeterminate(w + 8)
    if x.is_unbounded():
        return Ball.unbounded(w + 8)
    var mid = x._midpoint
    if x._radius.compare(_Radius.power_of_two(-1)) <= 0:
        var center = Ball(Integer(1), precision=w + 8) if mid.is_zero() else _kernel_exp(mid, w)
        if x._radius.is_zero() or not center.is_finite():
            return center^
        var bound = _Radius.upper(center._midpoint).add(center._radius)
        return Ball(_midpoint=center._midpoint, _radius=center._radius.add(bound.multiply(x._radius.scale2(1))), _kind=center._kind)
    var radius = x._radius.to_float()
    var low = Float(_rounded=_float_operation(mid, radius, 1, _exact_context()))
    var high = Float(_rounded=_float_operation(mid, radius, 0, _exact_context()))
    var a = Ball(Integer(1), precision=w + 8) if low.is_zero() else _kernel_exp(low, w)
    var b = Ball(Integer(1), precision=w + 8) if high.is_zero() else _kernel_exp(high, w)
    if not a.is_finite() or not b.is_finite():
        return Ball.unbounded(w + 8)
    return _hull(a._exact_lower(), b._exact_upper(), w + 8)
