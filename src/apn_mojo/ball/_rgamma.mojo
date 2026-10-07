"""The reciprocal gamma function near 1 and its derivative from the tabulated
Taylor series, for Gamma, log Gamma and digamma at low and medium precision
(Johansson, "Arbitrary-precision computation of the gamma function", 2021,
section 5).

`R(u) = 1/Gamma(1 + u) = sum_n b_n u**n`, with the coefficients of
`_rgamma_table.mojo` at 1280 bits, and `R'(u) = sum_n n b_n u**(n-1)` are
summed by Horner's rule in fixed point at `F = 64 wn >= w + 24` fraction
bits for `|u| <= 1/2`. Each coefficient is the table entry's top `wn` limbs,
below the true one by less than a unit (`n` units for `n b_n`), and each
product is truncated, below the true one by less than a unit; the steps after
scale both by `|u| <= 1/2`, so R is within `sum_n 2 |u|**n <= 4` units and R'
within `sum_n (n + 1) 2**(1-n) <= 6`. A step whose result later steps scale
by `|u|**n` (R) or `|u|**(n-1)` (R') ignores the limbs of the partial sum and
of u, and cuts those of its result, below `2**(-n log2 |u| - 66)` units:
less than `2**-64` units a step, under one in all, so R is within 5 units
and R' within 7, and a full-precision u costs about half the products. A u
with bits below the fixed point is
truncated to it, which moves R by at most 3 units and R' by at most 6: on
`t = 1 + u` in [1/2, 3/2], `R' = -psi(t)/Gamma(t)` and `R'' = (psi(t)**2 -
psi'(t))/Gamma(t)`, where psi increases from `psi(1/2) = -gamma - 2 log 2 >
-1.97` to `psi(3/2) < 1`, psi' decreases from `psi'(1/2) = pi**2/2 < 4.94` to
`psi'(3/2) > 0`, and `Gamma(t) > 0.88`.

The truncation (Johansson's Theorem 5.3, a bound on the series' absolute
terms): for complex `|z| <= 20` and `N <= 1000`, `|R(z) - sum_{n<N} b_n z**n|
<= 8 max(1/2, |z|) |b_N| |z|**N` when that is below `2**-8`. So R's tail at
`|u| <= 1/2` is at most `4 |b_N| |u|**N`, and by Cauchy's estimate on the
circle of radius 1/8 about u, where `|z| <= rho = |u| + 1/8 <= 5/8`, R''s
tail is at most `8 (5/8) |b_N| rho**N / (1/8) < 2**6 |b_N| rho**N`. N is the
first index where the bounds fall below `2**-(F + 2)`, with
`|b_N| < 2**(bits_N - 1280)` from the entries' bit lengths.
"""

from std.math import log2, ceil
from std.collections import Array
from std.bit import count_leading_zeros
from ..integer.value import Integer
from ..integer._limbs import (
    _multiply_limbs, _multiply_limb_into, _add_mul_limbs, _multiply_by_limb, _add_limbs, _subtract_limbs, _compare,
    _BASECASE_LIMBS, _used, _add_one,
)
from ..float.value import Float
from .value import Ball
from ._radius import _Radius, _up_word
from ._medium import _Limbs, _fixed_ball, _fixed_from_float, _hex_limb
from ..float._rounding import _short_magnitude
from ._rgamma_table import _RGAMMA_COUNT, _RGAMMA_LIMBS, _RGAMMA_SIGNS, _RGAMMA_MAGNITUDES, _RGAMMA_BITS

comptime _RGAMMA_PRECISION = 64 * _RGAMMA_LIMBS - 24
"""The largest working precision the table serves: `64 wn >= w + 24` fits
its entries."""

comptime _RGAMMA_ROOM = 6 * _RGAMMA_LIMBS + 6
"""Limbs of the stack block: R, R', u and a coefficient at `wn + 1` limbs, and
a product at `2 wn + 2`."""


def _entry_bits(n: Int) -> Int:
    """The bit length of entry n, `|b_n| 2**(64 _RGAMMA_LIMBS)` rounded toward
    zero, from the table's 4 hex digits."""
    var at = _RGAMMA_BITS.unsafe_ptr().unsafe_offset(4 * n)
    var value = 0
    for i in range(4):
        var c = Int(at.unsafe_offset(i)[])
        value = 16 * value + (c & 15) + 9 * (c >> 6)
    return value


def _coefficient(n: Int, target: _Limbs, wn: Int):
    """`floor(|b_n| 2**(64 wn))` into target[0, wn], the top limb the
    integer part: 1 for b_0, else 0."""
    var start = 16 * _RGAMMA_LIMBS * n
    for p in range(wn):
        target.unsafe_offset(wn - 1 - p)[] = _hex_limb(_RGAMMA_MAGNITUDES, start + 16 * p)
    target.unsafe_offset(wn)[] = 1 if n == 0 else 0


def _coefficient_negative(n: Int) -> Bool:
    return _RGAMMA_SIGNS.unsafe_ptr().unsafe_offset(n)[] == 45


def _log2_upper(x: Float64) -> Float64:
    """An upper bound of `log2 x` for a positive double, with a margin far
    above the logarithm's rounding."""
    return log2(x) + 1.0e-9


def _accumulate(s: _Limbs, negative: Bool, b: _Limbs, b_negative: Bool, n: Int) -> Bool:
    """`s += b`, both of n limbs, magnitudes and signs apart; the new sign."""
    if negative == b_negative:
        _ = _add_limbs(s, b, n)
        return negative
    if _compare(s, b, n) >= 0:
        _ = _subtract_limbs(s, b, n)
        return negative
    _ = _subtract_limbs(b, s, n)
    for i in range(n):
        s.unsafe_offset(i)[] = b.unsafe_offset(i)[]
    return b_negative


def _scale_down(s: _Limbs, x: _Limbs, low: Int, cut: Int, wn: Int, product: _Limbs) raises:
    """`s = floor(s |u|)`, s of `wn + 1` limbs and `|u| = x 2**-(64 wn)`,
    with the low `cut` limbs of s and of x taken as zero, and those of the
    result set to zero: Horner's later steps scale them below a unit
    (`_rgamma_series`). x's low `low` limbs are zero, which the product
    skips: a short u such as 1/2 takes one row. Within the basecase size the
    product is taken by rows directly: `_multiply_limbs`' dispatch cost more
    than such few words."""
    var c = max(low, cut)
    var top = x.unsafe_offset(c)
    var xn = wn - c
    # s's significant limbs only: a late partial sum, near its coefficient
    # |b_n|, has leading zero limbs.
    var sn = max(1, _used(s, wn + 1) - cut)
    var from_s = s.unsafe_offset(cut)
    if xn > _BASECASE_LIMBS:
        _ = _multiply_limbs(from_s, sn, top, xn, product)
    else:
        product.unsafe_offset(sn)[] = _multiply_limb_into(product, from_s, sn, top[])
        for j in range(1, xn):
            product.unsafe_offset(sn + j)[] = _add_mul_limbs(product.unsafe_offset(j), from_s, sn, top.unsafe_offset(j)[])
    # The product is `s x 2**-(64 (cut + c))`, its limbs past sn + xn zero.
    var length = sn + xn
    for i in range(cut, wn + 1):
        var at = i + wn - cut - c
        s.unsafe_offset(i)[] = product.unsafe_offset(at)[] if at < length else 0
    for i in range(cut):
        s.unsafe_offset(i)[] = 0


def _rgamma_plan(log_u: Float64, u_upper: Float64, zero: Bool, fraction: Int, derivative: Bool) -> Tuple[Int, Int, Int]:
    """The terms N of R (and R') at `fraction` bits for a u below
    `u_upper <= 2**log_u` (upper bounds), and the exponents of the tails'
    bounds (module docstring); N = -1 where the table is too short."""
    var log_rho = _log2_upper((u_upper + 0.125) * (1.0 + 1.0e-15))
    var never = -(Int(1) << 40)
    for n in range(2, _RGAMMA_COUNT):
        var bits = Float64(_entry_bits(n) - 64 * _RGAMMA_LIMBS)
        var v = never if zero else Int(ceil(2.0 + bits + Float64(n) * log_u))
        var d = Int(ceil(6.0 + bits + Float64(n) * log_rho)) if derivative else never
        if v < -(fraction + 2) and d < -(fraction + 2):
            return (n, v, d)
    return (-1, 0, 0)


def _rgamma_series(u: Float, w: Int, derivative: Bool, guard: Int = 24) raises -> Optional[Tuple[Ball, Ball]]:
    """`(R(u), R'(u))` for `|u| <= 1/2` to about w bits, R' only with
    derivative (else 0), at the fixed point `F = 64 wn >= w + guard`: 24 guard
    bits keep a Float's Ziv steps short, and a ball, whose radius takes the
    error, needs few. None above the table's precision or where it is too
    short."""
    if w > _RGAMMA_PRECISION:
        return None
    var wn = (w + guard + 63) // 64
    var fraction = 64 * wn
    # Upper bounds of log2 |u| and log2 (|u| + 1/8); |u| < 2**exponent.
    var log_u = Float64(u._exponent)
    var u_upper = Float64(2) ** -60
    if u.is_zero():
        u_upper = 0.0
    elif u._exponent > -60:
        u_upper = abs(u.to_native[DType.float64]()) * (1.0 + 1.0e-15)
        log_u = min(log_u, _log2_upper(u_upper))
    var plan = _rgamma_plan(log_u, u_upper, u.is_zero(), fraction, derivative)
    var terms = plan[0]
    var value_tail = plan[1]
    var slope_tail = plan[2]
    if terms < 0:
        return None
    var store = Array[UInt64, _RGAMMA_ROOM](uninitialized=True)
    var block = Span(store).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var s = block
    var t = s.unsafe_offset(wn + 1)
    var x = t.unsafe_offset(wn + 1)
    var b = x.unsafe_offset(wn + 1)
    var product = b.unsafe_offset(wn + 1)
    # |u| at the fixed point, truncated.
    var truncated = _fixed_from_float(u, x, wn, fraction)
    var low = 0
    while low < wn - 1 and x.unsafe_offset(low)[] == 0:
        low += 1
    # Horner from b_{N-1} and (N-1) b_{N-1}.
    _coefficient(terms - 1, s, wn)
    var negative = _coefficient_negative(terms - 1)
    var slope_negative = negative
    if derivative:
        for i in range(wn + 1):
            t.unsafe_offset(i)[] = s.unsafe_offset(i)[]
        _ = _multiply_by_limb(t, wn + 1, UInt64(terms - 1))
    # A step's error reaches R scaled by |u|**n (R' by |u|**(n-1)): limbs
    # below 2**(-n log2|u| - 66) units are cut, less than 2**-64 units a step.
    var depth = -log_u
    for n in range(terms - 2, -1, -1):
        _coefficient(n, b, wn)
        var b_negative = _coefficient_negative(n)
        var cut = min(wn - 1, max(0, Int((Float64(n) * depth - 66.0) / 64.0)))
        if derivative and n >= 1:
            # t = n b_n + u t.
            var slope_cut = min(wn - 1, max(0, Int((Float64(n - 1) * depth - 66.0) / 64.0)))
            _scale_down(t, x, low, slope_cut, wn, product)
            if u._negative:
                slope_negative = not slope_negative
            for i in range(wn + 1):
                product.unsafe_offset(i)[] = b.unsafe_offset(i)[]
            _ = _multiply_by_limb(product, wn + 1, UInt64(n))
            slope_negative = _accumulate(t, slope_negative, product, b_negative, wn + 1)
        # s = b_n + u s.
        _scale_down(s, x, low, cut, wn, product)
        if u._negative:
            negative = not negative
        negative = _accumulate(s, negative, b, b_negative, wn + 1)
    var value_error = UInt64(8) if truncated else UInt64(5)
    var value = _fixed_ball(s, wn + 1, wn, negative, 0, value_error, _Radius.power_of_two(value_tail), w)
    var slope = Ball(Integer(0), precision=w)
    if derivative:
        var slope_error = UInt64(13) if truncated else UInt64(7)
        slope = _fixed_ball(t, wn + 1, wn, slope_negative, 0, slope_error, _Radius.power_of_two(slope_tail), w)
    _ = store^
    return (value^, slope^)


def _rgamma_unit(u: Float, w: Int) raises -> Optional[Ball]:
    """`1/Gamma(1 + u)` for `|u| <= 1/2`, to about w bits; None above the
    table's precision or where the table is too short."""
    var pair = _rgamma_series(u, w, False)
    if not pair:
        return None
    return pair.value()[0]


comptime _FACTOR_LIMBS = 8
"""The widest fixed point, in limbs, at which `_shift_factors` multiplies;
past it the caller's exact product of short factors serves."""

comptime _FACTOR_ROOM = 8 * _FACTOR_LIMBS + 20


def _shift_factors(u: Float, m: Int, w: Int, derivative: Bool, guard: Int = 24) raises -> Optional[Tuple[Ball, Ball]]:
    """`P = (1+u) (2+u) ... (m+u)` and, with derivative, `P' = dP/du =
    P sum_k 1/(k+u)`, for `|u| <= 1/2` exact at `_rgamma_series`'s fixed point
    `F = 64 wn` (else None, as past `_FACTOR_LIMBS` limbs).

    The factors `f_k = (k + u) 2**F` are exact integers. P is kept as an
    integer of n = wn + 2 limbs times `2**e`: each step multiplies by f_k
    and keeps the top n limbs, truncating; while it fits, nothing is cut.
    With P at least `2**(64 (n-1))` after a cut, each loses less than
    `eps = 2**-(64 (n-1))` of itself, so the true P lies within a factor
    `(1 + eps)**m` above the kept one, and below it by nothing: a radius of
    `2 m 2**(64 + e)` covers it. P' steps as `P' f_k + P 2**F` at the same
    scale and cut; all terms are positive, `P'/P = sum 1/(j+u)` stays between
    2/3 and 16 (`m <= 4096`), so P' loses less than `1.5 eps` of itself a step
    and fits n + 1 limbs, and `4 m 2**(128 + e)` covers it."""
    if m > 4096:
        return None
    if m == 0:
        return (Ball(Integer(1), precision=w), Ball(Integer(0), precision=w))
    var wn = (w + guard + 63) // 64
    if wn > _FACTOR_LIMBS:
        return None
    var fraction = 64 * wn
    var n = wn + 2
    var store = Array[UInt64, _FACTOR_ROOM](fill=0)
    var block = Span(store).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var x = block
    var f = x.unsafe_offset(wn)
    var p = f.unsafe_offset(wn + 1)
    var q = p.unsafe_offset(n + 1)
    var prod = q.unsafe_offset(n + 2)
    var prod2 = prod.unsafe_offset(n + wn + 2)
    if _fixed_from_float(u, x, wn, fraction):
        return None
    # The factors' fraction: |u|, or 2**F - |u| below the integer k - 1.
    var negative = u._negative and not u.is_zero()
    var carry = UInt64(1)
    for i in range(wn):
        if negative:
            var v = ~x.unsafe_offset(i)[] + carry
            carry = UInt64(Int(carry == 1 and v == 0))
            f.unsafe_offset(i)[] = v
        else:
            f.unsafe_offset(i)[] = x.unsafe_offset(i)[]
    p[] = 1
    var pn = 1
    var qn = 0
    var exponent = 0
    var cut = False
    var slope_cut = False
    for k in range(1, m + 1):
        f.unsafe_offset(wn)[] = UInt64(k - 1 if negative else k)
        var fl = wn + 1 if f.unsafe_offset(wn)[] else wn
        # prod = P f.
        prod.unsafe_offset(pn)[] = _multiply_limb_into(prod, p, pn, f[])
        for j in range(1, fl):
            prod.unsafe_offset(pn + j)[] = _add_mul_limbs(prod.unsafe_offset(j), p, pn, f.unsafe_offset(j)[])
        var length = _used(prod, pn + fl)
        var drop = max(0, length - n)
        if derivative:
            # prod2 = P' f + P 2**F.
            var length2 = max(qn + fl, pn + wn) + 1
            for i in range(length2):
                prod2.unsafe_offset(i)[] = 0
            if qn:
                prod2.unsafe_offset(qn)[] = _multiply_limb_into(prod2, q, qn, f[])
                for j in range(1, fl):
                    prod2.unsafe_offset(qn + j)[] = _add_mul_limbs(prod2.unsafe_offset(j), q, qn, f.unsafe_offset(j)[])
            var c = _add_limbs(prod2.unsafe_offset(wn), p, pn)
            if c:
                _ = _add_one(prod2.unsafe_offset(wn + pn), length2 - wn - pn, c)
            length2 = _used(prod2, length2)
            if length2 - drop > n + 1:
                return None
            for i in range(drop):
                slope_cut = slope_cut or prod2.unsafe_offset(i)[] != 0
            for i in range(drop, length2):
                q.unsafe_offset(i - drop)[] = prod2.unsafe_offset(i)[]
            qn = length2 - drop
        for i in range(drop):
            cut = cut or prod.unsafe_offset(i)[] != 0
        for i in range(drop, length):
            p.unsafe_offset(i - drop)[] = prod.unsafe_offset(i)[]
        pn = length - drop
        exponent += 64 * drop - fraction
    var size = _Radius.zero()
    var slope_size = _Radius.zero()
    if cut:
        size = _up_word(UInt64(2 * m), False, exponent + 64)
    if cut or slope_cut:
        slope_size = _up_word(UInt64(4 * m), False, exponent + 128)
    var product = _fixed_ball(p, pn, 0, False, exponent, 0, size, w)
    var slope = Ball(Integer(0), precision=w)
    if derivative:
        slope = _fixed_ball(q, qn, 0, False, exponent, 0, slope_size, w)
    _ = store^
    return (product^, slope^)


@fieldwise_init
struct _WordGamma(ImplicitlyCopyable):
    """Gamma and digamma at x from `_gamma_word`: `Gamma(x)` within the
    relative `error` of `quotient 2**exponent`, `|psi(x)| <= psi_bound`, and,
    when asked, `psi(x)` within `psi_error` of `+-psi 2**-64`."""

    var quotient: UInt128
    var exponent: Int
    var error: Float64
    var psi_bound: Float64
    var psi: UInt128
    var psi_negative: Bool
    var psi_error: Float64


@always_inline
def _signed_add(a: UInt128, a_negative: Bool, b: UInt128, b_negative: Bool) -> Tuple[UInt128, Bool]:
    """`a + b` of signed magnitudes, as a magnitude and a sign."""
    if a_negative == b_negative:
        return (a + b, a_negative)
    if a >= b:
        return (a - b, a_negative)
    return (b - a, b_negative)


@always_inline
def _wide_product(a: UInt128, b: UInt128) -> UInt256:
    """`a b` from four 64-bit partial products; a 256-bit multiplication
    computes sixteen."""
    var a0 = UInt128(UInt64(a))
    var a1 = a >> 64
    var b0 = UInt128(UInt64(b))
    var b1 = b >> 64
    return UInt256(a0 * b0) + ((UInt256(a0 * b1) + UInt256(a1 * b0)) << 64) + (UInt256(a1 * b1) << 128)


@always_inline
def _scale_word(s: UInt128, u: UInt64) -> UInt128:
    """`floor(s u / 2**64)` for `s u < 2**128`."""
    return UInt128(s >> 64) * UInt128(u) + ((UInt128(UInt64(s)) * UInt128(u)) >> 64)


def _gamma_word(x: Float, w: Int, digamma: Bool = False) raises -> Optional[_WordGamma]:
    """Gamma and a bound of digamma at `1/2 <= x < 2**20`, exact at 64
    fraction bits, for `w <= 62` (`F = 64 >= w + 2`): the Taylor method of `_rgamma_series`
    and `_shift_factors` in one 64-bit limb of fraction, its values in
    native 128-bit integers. None elsewhere, for the general path.

    With `x = 1 + u + m`, `|u| <= 1/2`: R and R' by Horner's rule, magnitudes
    and signs apart, partial sums below 2 (`|b_1| < 0.58`, `|b_2| < 0.66`, and
    `sum_{j>=3} |b_j| 2**(3-j) < 0.25`), so `s u < 2**128`; their errors are those of `_rgamma_series`,
    4 and 6 units of `2**-64` and the tails. P and P' as 124-bit mantissas at
    a shared exponent, cut each step: true values within a factor
    `(1 + 2**-122)**m` above (`_shift_factors`' argument with these widths).
    Gamma is `P / R` from P's top 64 bits over R, a 128-by-65-bit division:
    its relative error adds P's, R's, the cut of P (below `2**-63`) and the
    quotient's (below `2**-62`), each from above. For the radius, `psi = P'/P
    - R'/R` takes doubles, with margins far above their roundings; with
    digamma, psi itself is `floor(P' 2**64 / P) - floor(R' 2**64 / R)` in
    units of `2**-64`, within those quotients' cuts and the relative errors
    of P', P, R' and R."""
    if x._kind != 1 or x._negative or x._exponent > 20 or w > 62:
        return None
    var significand = _short_magnitude(x._significand)
    if not significand:
        return None
    var shift = x._exponent - x.precision() + 64
    if shift < 0:
        return None
    var big = significand.value() << UInt128(shift)
    var n = Int((big + (UInt128(1) << 63)) >> 64)
    var m = n - 1
    if m < 0 or m + 2 > w // 2 + 10 or m > 126:
        return None
    var whole = UInt128(n) << 64
    var u_negative = big < whole
    var u = UInt64(whole - big) if u_negative else UInt64(big - whole)
    # R and R' by Horner's rule.
    var u_upper = Float64(u) * (1.0 / 18446744073709551616.0) * (1.0 + 1.0e-15)
    var log_u = Float64(-64) if u == 0 else _log2_upper(u_upper)
    var plan = _rgamma_plan(log_u, u_upper, u == 0, 64, True)
    var terms = plan[0]
    if terms < 0:
        return None
    var top = terms - 1
    var s = UInt128(_hex_limb(_RGAMMA_MAGNITUDES, 16 * _RGAMMA_LIMBS * top))
    var s_negative = _coefficient_negative(top)
    var t = s * UInt128(top)
    var t_negative = s_negative
    for k in range(terms - 2, -1, -1):
        var b = UInt128(1) << 64 if k == 0 else UInt128(_hex_limb(_RGAMMA_MAGNITUDES, 16 * _RGAMMA_LIMBS * k))
        var b_negative = _coefficient_negative(k)
        if k >= 1:
            var tk = _signed_add(_scale_word(t, u), t_negative != u_negative, b * UInt128(k), b_negative)
            t = tk[0]
            t_negative = tk[1]
        var sk = _signed_add(_scale_word(s, u), s_negative != u_negative, b, b_negative)
        s = sk[0]
        s_negative = sk[1]
    if s_negative or s == 0:
        return None
    # P and P' as mantissas below 2**124 at a shared exponent.
    var factor_low = (UInt128(1) << 64) - UInt128(u) if u_negative and u != 0 else UInt128(u)
    var pm = UInt128(1)
    var dm = UInt128(0)
    var exponent = 0
    for k in range(1, m + 1):
        var f = (UInt128(k - 1 if u_negative and u != 0 else k) << 64) + factor_low
        var product = _wide_product(pm, f)
        var slope = _wide_product(dm, f) + (UInt256(pm) << 64)
        var bits = 256 - Int(count_leading_zeros(product))
        var cut = max(0, bits - 124)
        pm = UInt128(product >> UInt256(cut))
        dm = UInt128(slope >> UInt256(cut))
        exponent += cut - 64
    # Normalized to 124 bits, exactly, so the quotient keeps 122 or more.
    var lead = 124 - (128 - Int(count_leading_zeros(pm)))
    if lead > 0:
        pm <<= UInt128(lead)
        dm <<= UInt128(lead)
        exponent -= lead
    # Gamma = P / R: q 2**(exponent + 60) with q = floor((pm >> 60) 2**64 / s).
    var q = ((pm >> 60) << 64) // s
    var tail = Float64(2) ** Float64(plan[1] + 64)
    var tail_slope = Float64(2) ** Float64(plan[2] + 64)
    var s_low = Float64(s) * (1.0 - 1.0e-15) - 4.0 - tail
    if not (s_low > 0.0):
        return None
    var r_error = (4.0 + tail) / s_low
    var p_error = Float64(m + 1) * (Float64(2) ** -121) + Float64(2) ** -63
    var q_error = 1.0 / Float64(q) if q else 1.0
    var error = (p_error + r_error + r_error * r_error * 2.0 + q_error) * (1.0 + 1.0e-12) + Float64(2) ** -100
    # psi = P'/P - R'/R, bounded in doubles.
    var a = Float64(dm) / Float64(pm) if pm else 0.0
    var b_value = Float64(t) / Float64(s)
    var psi = a - (-b_value if t_negative else b_value)
    var psi_error = (abs(a) + b_value) * 1.0e-13 + (6.0 + tail_slope) / s_low + b_value * r_error * 2.0 + 1.0e-15
    var exact = UInt128(0)
    var exact_negative = False
    var exact_error = 0.0
    if digamma:
        var ratio = UInt128((UInt256(dm) << 64) // UInt256(pm))
        var slope_ratio = UInt128((UInt256(t) << 64) // UInt256(s))
        var difference = _signed_add(ratio, False, slope_ratio, not t_negative)
        exact = difference[0]
        exact_negative = difference[1]
        # The cuts (a unit each), P' and P's relative errors on P'/P, and R'/R's.
        exact_error = (
            2.0 * (Float64(2) ** -64) + abs(a) * Float64(3 * (m + 1)) * (Float64(2) ** -121)
            + (6.0 + tail_slope) / s_low + b_value * r_error * 2.0
        ) * (1.0 + 1.0e-12)
    return _WordGamma(q, exponent + 60, error, abs(psi) + psi_error, exact, exact_negative, exact_error)
