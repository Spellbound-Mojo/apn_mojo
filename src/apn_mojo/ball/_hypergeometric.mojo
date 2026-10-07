"""Hypergeometric series as balls, and Kummer's confluent hypergeometric
function M (scipy's hyp1f1).

One engine sums `S = sum_k T_k`, `T_0 = 1`,

    T_{k+1} = T_k z prod_i (u_i + k) / (prod_j (l_j + k) (k + 1)),

in ball arithmetic, with at most one more upper parameter u than lower ones
l. Pair each upper parameter with a lower one, the last with the `k!`'s 1
when there is one more. For `l + N > 0` and k >= N,
`|u + k| <= |l + k| + |u - l|`, so every ratio `|T_{k+1} / T_k|` with
k >= N is at most

    D = |z| prod (1 + |u - l| / (l + N)) prod_unpaired 1 / (l + N),

and when D < 1 the tail `sum_{k>=N} T_k` is at most `|T_N| / (1 - D)`
(Johansson, "Computing hypergeometric functions rigorously", 2019,
Theorem 1). A pass in double precision over `log2 |T_k|` picks N, where
D < 1 and `|T_N| / (1 - D)` is `w + 8` bits below the running sum, and the bits the
terms' cancellation takes, `log2(max |T_k| / |S|)`. An exact non-positive
integer upper parameter -n ends the series after `T_n`; with exact rational
parameters that sum is computed exactly.

Each ball operation costs far more than its arithmetic at these precisions,
so the terms take as few as they can. When every parameter is an exact
dyadic `N / 2**d` with N and d near the working precision, as `1/2` or a
Float argument of the target's precision are, the ratio is
`P(k) / Q(k)` in integers: `P = N_z prod (N_u + k 2**d_u)` and
`Q = (k+1) prod (N_l + k 2**d_l)`, with the powers of two moved into one of
them and z folded into P when it is such a dyadic too. A term is then one
multiplication and one division by an Integer, plus the running sum.
Otherwise the shifted parameters `u + k` are balls stepped by 1.

Kummer's U has the expansion, with `sigma = |b - 2a| / |z|` (DLMF 13.7.4-9),

    U(a, b, z) = z**-a sum_{s<n} (a)_s (a-b+1)_s / s! (-z)**-s + e_n(z),
    |e_n(z)| <= 2 alpha C_n |(a)_n (a-b+1)_n / (n! z**(a+n))| exp(2 alpha rho C_1 / |z|).

At z > 0 (region R1), `C_n = 1`, `alpha = 1/(1 - sigma)` and
`rho = |2a**2 - 2ab + b|/2 + sigma (1 + sigma/4) / (1 - sigma)**2`. On the
negative axis beyond `-2 |b - 2a|`, in the closures of R3 and its conjugate,
`C_n = (chi(n) + sigma nu**2 n) nu**n` with
`nu = (1/2 + sqrt(1 - 4 sigma**2)/2)**(-1/2)` and sigma replaced by
`nu sigma` in alpha and rho; `chi(n) = sqrt(pi) Gamma(n/2 + 1) /
Gamma(n/2 + 1/2) < sqrt(pi (n + 2) / 2)` by Gautschi's inequality (DLMF
5.6.4). Both regions need `sigma <= 1/2`, and every bound grows with sigma.

M(a, b, x) at a real x > 0 is the real part of DLMF 13.2.41, which holds
with either sign there:

    M(a, b, x) = Gamma(b) [e**x x**(a-b) / Gamma(a) (S_1 + E_1)
                           + cos(pi a) / Gamma(b-a) x**-a (S_2 + E_2)],

with `S_2 + E_2 = x**a U(a, b, x)` and `S_1 = sum_{s<n} (b-a)_s (1-a)_s /
(s! x**s)`, the expansion of `U(b-a, b, x e**(+-pi i))`: the factor
`e**(+-pi i (b-a))` cancels its phase, so only the remainder E_1 is complex,
and it is bounded on the negative axis. A negative x takes Kummer's
transformation `M(a, b, x) = e**x M(b-a, b, -x)` (DLMF 13.2.39) first, with
`e**x e**-x` cancelled before evaluation. `1/Gamma` is entire: below 0 it is
`sin(pi t) Gamma(1 - t) / pi`. The series runs at `|x|` after Kummer's
transformation, so its terms keep the signs of their parameters; the
expansion takes over where the series would be longer than about the
precision (Johansson's `|z| > w log 2` for small parameters).

Rational values are computed exactly, so that a representable one rounds
correctly in every mode: M = 1 at x = 0 or a = 0; a polynomial for a
non-positive integer a = -n, `sum_{k<=n} (a)_k / (b)_k x**k / k!`, also for
a non-positive integer b at or below a negative integer a, as scipy's (so
`M(-n, -n, x)` is the truncated exponential); 0 where Kummer's
polynomial `M(b-a, b, -x)`, for b - a a non-positive integer, vanishes; and
for integers 0 < a < b, by parts in Euler's integral (DLMF 13.4.1),

    M = C [e**x sum_j (-1)**(j+m) j! C(a-1, j-m) / x**(j+1)
           - (-1)**(a-1) sum_j j! C(m, j-a+1) / x**(j+1)],

with `m = b - a - 1` and `C = (b-1)! / ((a-1)! m!)`, rational where the sum
multiplying `e**x` is 0, such as `M(2, 4, 2) = 3`. Otherwise M is
transcendental there (Lindemann) or not known to be rational, and a value
that no precision separates from a rounding boundary ends in the budget
error. At the other non-positive integers b, M is scipy's `+inf`.
"""

from std.math import log, exp
from std.bit import count_trailing_zeros
from ..integer.value import Integer
from ..integer.math import factorial, comb
from ..integer.powers import iroot_exact
from ..rational.value import Rational
from ..float.value import Float
from ..float._functions import _scale_float
from .context import BallContext
from .value import Ball, _BallArgument, _FINITE
from ._radius import _Radius
from ._arithmetic import _working, _rounded_ball, _power, _negate
from ._kernels import _EXP, _LOG, _SIN, _COS, _SQRT
from ._constants import _pi, _euler_gamma
from ._special import (
    _ball_special, _GAMMA, _DIGAMMA, _MAX_TERMS, _add, _sub, _mul, _div, _mul_n, _div_n, _fn, _c, _one, _scaled, _magnitude, _widen, _sin_pi, _exact_add, _exact_difference, _integral_between,
)
from ._zeta import _int
from ._fixed_series import _fixed_terms, _bit_count
from ._arithmetic import _ball_add_assign
from ._dot import _ball_dot
from ._ratios import _Decided, _neither, _infinite, _nonpositive_integer, _poch_decided, _EXACT_BITS, _RATIONAL, _INFINITE
from ..float.context import _exact_context

comptime _BALL_GUARD = 8
"""The guard bits a ball entry point takes, once: the kernels below compute
at the precision they are given, adding bits only where a formula cancels
(its plan's guard, 1 - Q, the connection's terms). A ball's radius carries
the rounding errors, so a 53-bit ball computes at 61 bits, one word, where
nested guards of 16 bits each had reached past 90 and two-limb paths."""

comptime _LOG2_E = 1.4426950408889634
comptime _LN_2 = 0.6931471805599453
comptime _MODERATE_BITS = 1 << 16
"""The largest exact argument, in bits, that the exact cases convert to a
Rational."""


# ------------------------------------------------------------- planning


@fieldwise_init
struct _Plan(ImplicitlyCopyable):
    """How far a series is summed: `terms` terms (-1 when it is not usable
    at the precision), with `guard` bits for its cancellation; `finished`
    when an upper parameter makes `T_terms` and every later term 0."""

    var terms: Int
    var guard: Int
    var finished: Bool
    var peak: Float64
    """`log2 max |T_k|`."""
    var size: Float64
    """`log2 |S|`, estimated."""


def _unusable() -> _Plan:
    return _Plan(-1, 0, False, 0.0, 0.0)


def _ending(uppers: List[Ball]) raises -> Int:
    """`n + 1` for the smallest exact non-positive integer upper parameter -n,
    after which every term is 0; -1 for none."""
    var end = -1
    for u in uppers:
        var t = u._midpoint
        if not u.is_exact() or not (t.is_zero() or (t._negative and t.is_integer())):
            continue
        if not t.is_zero() and t._exponent > 40:
            continue
        var n = 0 if t.is_zero() else Int((-t).round()._low_magnitude())
        end = n + 1 if end < 0 else min(end, n + 1)
    return end


def _log2_of(x: Float) raises -> Float64:
    """`log2 |x|` in double precision, for planning; very negative at 0."""
    if x.is_zero():
        return -1.0e18
    var m = Float(_rounded=_scale_float(abs(x), Integer(-x._exponent), _exact_context()))
    return Float64(x._exponent) + log(m.to_native[DType.float64]()) * _LOG2_E


def _exp2(v: Float64) -> Float64:
    return exp(v * _LN_2)


def _size(peak: Float64, scaled_sum: Float64) -> Float64:
    """`log2 |S|` from the planning pass's sum scaled by its largest term."""
    return peak + (log(abs(scaled_sum)) * _LOG2_E if scaled_sum != 0.0 else -64.0)


def _guard(scaled_sum: Float64, k: Int) -> Int:
    """The bits that cancellation takes, `-log2 |S| / max |T|`, from the
    scaled sum, with `log2 k` more for the rounding of k terms."""
    var lost = 64.0
    if scaled_sum != 0.0:
        lost = max(0.0, -log(abs(scaled_sum)) * _LOG2_E)
    return Int(lost) + 1 + _bit_count(k) + 8


def _ratio_log2(ups: List[Float64], lows: List[Float64], log_z: Float64, n: Int) -> Float64:
    """`log2 D` at N = n in double precision, very large when some `l + n <= 0`."""
    var d = log_z
    for i in range(len(lows)):
        var l = lows[i] + Float64(n)
        if l <= 0.0:
            return 1.0e18
        if i < len(ups):
            d += log(1.0 + abs(ups[i] - lows[i]) / l) * _LOG2_E
        else:
            d -= log(l) * _LOG2_E
    var one = Float64(n + 1)
    if len(ups) > len(lows):
        d += log(1.0 + abs(ups[len(lows)] - 1.0) / one) * _LOG2_E
    else:
        d -= log(one) * _LOG2_E
    return d


@fieldwise_init
struct _Factor(ImplicitlyCopyable):
    """A parameter p for planning: its double, its nearest integer n (when
    `|p| < 2**50`, else a sentinel), and `log2 |p - n|` from the exact
    difference, since the double loses the distance of a parameter within
    `2**-53` of an integer, where `p + k` with `k = -n` is that distance."""

    var value: Float64
    var near: Int
    var defect: Float64
    var below: Bool

    def log2_at(self, k: Int) -> Float64:
        """`log2 |p + k|`."""
        if self.near + k == 0:
            return self.defect
        return log(max(abs(self.value + Float64(k)), 1.0e-300)) * _LOG2_E

    def negative_at(self, k: Int) -> Bool:
        """Whether `p + k < 0`."""
        if self.near + k == 0:
            return self.below
        return self.value + Float64(k) < 0.0


def _factor(p: Ball) raises -> Optional[_Factor]:
    """The planning record of a parameter ball's midpoint; None beyond
    `2**1000`."""
    var t = p._midpoint
    if t.is_zero():
        return _Factor(0.0, 0, -1.0e18, False)
    if t._exponent > 1000:
        return None
    var value = t.to_native[DType.float64]()
    if t._exponent > 50:
        return _Factor(value, Int(1) << 62, 0.0, False)
    var n = t.round()
    var near = Int(n._low_magnitude()) * (-1 if n.sign() < 0 else 1)
    var defect = _exact_difference(t, Float(n))
    return _Factor(value, near, _log2_of(defect), not defect.is_zero() and defect._negative)


def _plan(uppers: List[Ball], lowers: List[Ball], z: Ball, w: Int, divergent: Bool) raises -> _Plan:
    """The terms and guard bits of `sum_k T_k`, from a double-precision pass
    over `log2 |T_k|` at the parameters' midpoints. A convergent series stops
    where D < 1 and the tail bound `|T_k| / (1 - D)` is `w + 8` bits below
    the running sum; an
    asymptotic one at the first term below `2**-(w+8)`, and is not usable
    when its terms grow for good before that."""
    var end = _ending(uppers)
    if z._midpoint.is_zero() and z._radius.is_zero():
        return _Plan(1, 0, True, 0.0, 0.0)
    var ups = List[Float64]()
    var lows = List[Float64]()
    var up_factors = List[_Factor]()
    var low_factors = List[_Factor]()
    var reach = 2.0
    for u in uppers:
        var f = _factor(u)
        if not f:
            return _unusable()
        up_factors.append(f.value())
        ups.append(f.value().value)
        reach += abs(f.value().value)
    for l in lowers:
        var f = _factor(l)
        if not f:
            return _unusable()
        low_factors.append(f.value())
        lows.append(f.value().value)
        reach += abs(f.value().value)
    var log_z = _log2_of(_magnitude(z).to_float())
    var z_negative = z._midpoint._negative
    var log_t = 0.0
    var peak = 0.0
    var scaled_sum = 1.0
    var sign = 1.0
    var k = 0
    while k < _MAX_TERMS:
        if end >= 0 and k + 1 >= end:
            return _Plan(end, _guard(scaled_sum, k), True, peak, _size(peak, scaled_sum))
        var step = log_z - log(Float64(k + 1)) * _LOG2_E
        var s = -1.0 if z_negative else 1.0
        for i in range(len(up_factors)):
            step += up_factors[i].log2_at(k)
            if up_factors[i].negative_at(k):
                s = -s
        for i in range(len(low_factors)):
            var size = low_factors[i].log2_at(k)
            if size < -1.0e17:
                return _unusable()
            step -= size
            if low_factors[i].negative_at(k):
                s = -s
        log_t += step
        sign *= s
        k += 1
        if log_t > peak:
            scaled_sum *= _exp2(peak - log_t)
            peak = log_t
        scaled_sum += sign * _exp2(log_t - peak)
        if divergent:
            if log_t < -Float64(w + 8):
                return _Plan(k, _guard(scaled_sum, k), False, peak, _size(peak, scaled_sum))
            if step >= 0.0 and Float64(k) > reach:
                return _unusable()
        else:
            # Stop once D < 1 and the tail bound |T_k| / (1 - D) is w + 8 bits
            # below the sum; D tends to |z|, so a series with |z| < 1 stops.
            var size = peak + (log(abs(scaled_sum)) * _LOG2_E if scaled_sum != 0.0 else -64.0)
            if log_t < size - Float64(w + 8):
                var ratio = _ratio_log2(ups, lows, log_z, k)
                if ratio < -0.0015 and log_t - log(1.0 - _exp2(ratio)) * _LOG2_E < size - Float64(w + 8):
                    return _Plan(k, _guard(scaled_sum, k), False, peak, size)
    return _unusable()


# -------------------------------------------------------------- summing


@fieldwise_init
struct _Dyadic(ImplicitlyCopyable):
    """An exact parameter `numerator / 2**shift`, shift >= 0."""

    var numerator: Integer
    var shift: Int


def _dyadic(x: Ball, w: Int) raises -> Optional[_Dyadic]:
    """An exact ball's value as `n / 2**d`, d >= 0, when its odd part has at
    most `w + 64` bits, d at most `w + 128` and its integer part's shift at
    most 64, so that the integers stay near the working precision; None
    otherwise."""
    if not x.is_exact() or not x.is_finite():
        return None
    var t = x._midpoint
    if t.is_zero():
        return _Dyadic(Integer(0), 0)
    # The odd part: wide formats store short values with many trailing zeros.
    var odd = t._significand
    var zeros = 0
    while odd._low_magnitude() == 0:
        odd = odd >> 64
        zeros += 64
    var tail = Int(count_trailing_zeros(odd._low_magnitude()))
    odd = odd >> tail
    zeros += tail
    var q = t._exponent - t.precision() + zeros
    if odd.magnitude_bit_length() > w + 64 or q > 64 or q < -(w + 128):
        return None
    var n = (odd << q) if q > 0 else odd^
    return _Dyadic(-n if t._negative else n^, -q if q < 0 else 0)


struct _ExactRatio(Copyable, Movable):
    """The term ratio `z prod (u + k) / ((k + 1) prod (l + k))` as
    `P(k) / Q(k)` times `2**shift`, in integers, for exact dyadic parameters
    (module docstring); z is folded into P when it is a dyadic too. A shift
    of at most 64 bits goes into P or Q; a larger one scales the term
    exactly."""

    var ups: List[_Dyadic]
    var lows: List[_Dyadic]
    var z_numerator: Integer
    var scale: Int
    var folded: Bool

    def __init__(out self, var ups: List[_Dyadic], var lows: List[_Dyadic], var z_numerator: Integer, scale: Int, folded: Bool):
        self.ups = ups^
        self.lows = lows^
        self.z_numerator = z_numerator^
        self.scale = scale
        self.folded = folded

    def term_shift(self) -> Int:
        """The power of two the term takes after each step."""
        return self.scale if self.scale > 64 or self.scale < -64 else 0

    def at(self, k: Int) raises -> Tuple[Integer, Integer]:
        var p = self.z_numerator
        for u in self.ups:
            p = p * (u.numerator + (Integer(k) << u.shift))
        var q = Integer(k + 1)
        for l in self.lows:
            q = q * (l.numerator + (Integer(k) << l.shift))
        if self.term_shift() == 0:
            if self.scale > 0:
                p = p << self.scale
            elif self.scale < 0:
                q = q << (-self.scale)
        return (p^, q^)


def _size(d: _Dyadic) -> Int:
    """An upper bound of the bits of `N + k 2**d` for `k < 2**22`."""
    return max(d.numerator.magnitude_bit_length(), d.shift + 22) + 1


def _exact_ratio(uppers: List[Ball], lowers: List[Ball], z: Ball, w: Int) raises -> Optional[_ExactRatio]:
    """The integer form of the term ratio, when every parameter is an exact
    dyadic and the integers stay cheap: P within `w + 128 + min(w, 512)`
    bits, and Q within 192 bits, so its division stays short. None
    otherwise."""
    var ups = List[_Dyadic]()
    var lows = List[_Dyadic]()
    var scale = 0
    var p_bits = 0
    var q_bits = 23
    for u in uppers:
        var d = _dyadic(u, w)
        if not d:
            return None
        scale -= d.value().shift
        p_bits += _size(d.value())
        ups.append(d.value())
    for l in lowers:
        var d = _dyadic(l, w)
        if not d:
            return None
        scale += d.value().shift
        q_bits += _size(d.value())
        lows.append(d.value())
    # Below about a thousand bits a ball operation costs more than its
    # arithmetic, so a longer P still saves; above, its multiplication does
    # not (measured at 256 and 4096 bits).
    var limit = w + 128 + min(w, 512)
    var zd = _dyadic(z, w)
    var folded = False
    var z_numerator = Integer(1)
    if zd and p_bits + zd.value().numerator.magnitude_bit_length() <= limit:
        folded = True
        z_numerator = zd.value().numerator
        p_bits += z_numerator.magnitude_bit_length()
        scale -= zd.value().shift
    if p_bits + (scale if 0 < scale <= 64 else 0) > limit or q_bits + (-scale if -64 <= scale < 0 else 0) > 192:
        return None
    return _ExactRatio(ups^, lows^, z_numerator^, scale, folded)


def _advance(
    uppers: List[Ball], lowers: List[Ball], z: Ball, start: Int, stop: Int, first: Ball, partial: Ball, w: Int,
) raises -> Tuple[Ball, Ball]:
    """From `T_start` and `sum_{k<start} T_k`, the sum of the terms below
    `stop` and `T_stop`: by the integer ratio for exact dyadic parameters,
    else with the shifted parameters stepped by 1."""
    var term = first
    var total = partial
    if start >= stop:
        return (total^, term^)
    var exact = _exact_ratio(uppers, lowers, z, w)
    if exact:
        var ratio = exact.take()
        for k in range(start, stop):
            total = _add(total, term, w)
            var pq = ratio.at(k)
            var next = _mul_n(term, pq[0], w)
            if not ratio.folded:
                next = _mul(next, z, w)
            term = _div_n(next, pq[1], w)
            if ratio.term_shift() != 0:
                term = _scaled(term, ratio.term_shift(), w)
        return (total^, term^)
    var one = _one(w)
    var shift = Ball(Integer(start), precision=w)
    var ups = List[Ball]()
    for u in uppers:
        ups.append(_add(u, shift, w))
    var lows = List[Ball]()
    for l in lowers:
        lows.append(_add(l, shift, w))
    for k in range(start, stop):
        total = _add(total, term, w)
        var next = _mul(term, z, w)
        for i in range(len(ups)):
            next = _mul(next, ups[i], w)
            ups[i] = _add(ups[i], one, w)
        if len(lows) == 0:
            term = _div_n(next, Integer(k + 1), w)
            continue
        var bottom = lows[0]
        lows[0] = _add(lows[0], one, w)
        for j in range(1, len(lows)):
            bottom = _mul(bottom, lows[j], w)
            lows[j] = _add(lows[j], one, w)
        term = _div(next, _mul_n(bottom, Integer(k + 1), w), w)
    return (total^, term^)


def _ratio_bound(uppers: List[Ball], lowers: List[Ball], z: Ball, n: Int, w: Int) raises -> Optional[Float]:
    """An upper bound below 1 of every ratio `|T_{k+1} / T_k|` with k >= n,
    D of the module docstring; None when some `l + n` is not certainly
    positive or the bound is not below 1."""
    var d = Ball(_magnitude(z).to_float())
    var shift = Ball(Integer(n), precision=w)
    for i in range(len(lowers)):
        var low = _add(lowers[i], shift, w)._exact_lower()
        if not (low > Float(0)):
            return None
        if i < len(uppers):
            var gap = Ball(_magnitude(_sub(uppers[i], lowers[i], w)).to_float())
            d = _mul(d, _add(_one(w), _div(gap, Ball(low), w), w), w)
        else:
            d = _div(d, Ball(low), w)
    var one = Ball(Integer(n + 1), precision=w)
    if len(uppers) > len(lowers):
        var gap = Ball(_magnitude(_sub(uppers[len(lowers)], _one(w), w)).to_float())
        d = _mul(d, _add(_one(w), _div(gap, one, w), w), w)
    else:
        d = _div(d, one, w)
    var top = d._exact_upper()
    if not (top < Float(1)):
        return None
    return top


def _terms(uppers: List[Ball], lowers: List[Ball], z: Ball, n: Int, w: Int, plan: _Plan) raises -> Tuple[Ball, Ball]:
    """`sum_{k<n} T_k` and `T_n` (or a bound of it): in fixed point
    (`_fixed_series.mojo`) where its block holds the terms and factors, else
    by the integer ratio or the ball steps."""
    var fixed = _fixed_terms(uppers, lowers, z, n, w, plan.size, plan.peak, 8)
    if fixed:
        return fixed.take()
    return _advance(uppers, lowers, z, 0, n, _one(w), Ball(Integer(0), precision=w), w)


def _planned_series(uppers: List[Ball], lowers: List[Ball], z: Ball, w: Int, plan: _Plan) raises -> Ball:
    """A convergent series to about w bits by its plan, widened by the tail
    bound; indeterminate when the plan is not usable."""
    if plan.terms < 0:
        return Ball.indeterminate(w)
    var work = w + plan.guard
    var n = plan.terms
    if plan.finished:
        var fixed = _fixed_terms(uppers, lowers, z, n, work, plan.size, plan.peak, 8)
        if fixed:
            return fixed.value()[0]
        return _finished_sum(uppers, lowers, z, n, work)
    var pair = _terms(uppers, lowers, z, n, work, plan)
    while True:
        if not pair[1].is_finite():
            return Ball.indeterminate(w)
        var bound = _ratio_bound(uppers, lowers, z, n, work)
        if bound:
            var tail = _div(Ball(_magnitude(pair[1]).to_float()), _sub(_one(work), Ball(bound.value()), work), work)
            return _widen(pair[0], tail)
        if 2 * n > _MAX_TERMS:
            return Ball.indeterminate(w)
        n *= 2
        pair = _terms(uppers, lowers, z, n, work, plan)


def _finished_sum(uppers: List[Ball], lowers: List[Ball], z: Ball, n: Int, w: Int) raises -> Ball:
    """`sum_{k<n} T_k` of a series an upper parameter ends there, without
    forming `T_n`, whose lower factors may vanish (scipy's polynomials at a
    pole of b beyond their degree)."""
    var pair = _advance(uppers, lowers, z, 0, n - 1, _one(w), Ball(Integer(0), precision=w), w)
    return _add(pair[0], pair[1], w)


def _series(uppers: List[Ball], lowers: List[Ball], z: Ball, w: Int) raises -> Ball:
    """`sum_k T_k` of a convergent series to about w bits."""
    return _planned_series(uppers, lowers, z, w, _plan(uppers, lowers, z, w, False))


def _rational_bits(q: Rational) raises -> Int:
    return q.numerator().magnitude_bit_length() + q.denominator().magnitude_bit_length()


def _polynomial(uppers: List[Rational], lowers: List[Rational], z: Rational, n: Int) raises -> Optional[Rational]:
    """`sum_{k<=n} T_k` exactly, for a series an upper parameter -n ends;
    None past the exact size bound."""
    var size = _rational_bits(z) + 2 * _bit_count(n) + 2
    for u in uppers:
        size += _rational_bits(u)
    for l in lowers:
        size += _rational_bits(l)
    if n > (1 << 16) or n * size > _EXACT_BITS:
        return None
    var term = Rational(1)
    var total = Rational(1)
    for k in range(n):
        var top = z
        for u in uppers:
            top = top * (u + Rational(k))
        var bottom = Rational(k + 1)
        for l in lowers:
            bottom = bottom * (l + Rational(k))
        term = term * top / bottom
        total = total + term
    return total^


# ------------------------------------------------- Kummer's U expansion


def _u_expansion(a: Ball, b: Ball, y: Ball, cut: Bool, w: Int) raises -> Optional[Ball]:
    """`sum_{s<n} (a)_s (a-b+1)_s / s! (-+1/y)**s` for y > 0, widened by
    Olver's bound of the rest: `y**a U(a, b, y)` (region R1), or with `cut`
    the real expansion of `U(a, b, y e**(+-pi i))` times
    `e**(+-pi i a) y**a` (the closures of R3). None when `|b - 2a| > y/2` or
    the terms grow before the precision."""
    var low = y._exact_lower()
    if not (low > Float(0)):
        return None
    var work = w
    var spread = Ball(_magnitude(_sub(b, _scaled(a, 1, work), work)).to_float())
    var sigma = _div(spread, Ball(low), work)._exact_upper()
    if not (sigma <= Float(Rational(1, 2))):
        return None
    var c = _add(_sub(a, b, work), _one(work), work)
    var inverse = _div(_one(work), y, work)
    var uppers: List[Ball] = [a, c]
    var lowers = List[Ball]()
    var z = inverse if cut else _negate(inverse)
    var plan = _plan(uppers, lowers, z, w, True)
    if plan.terms < 0:
        return None
    work = w + plan.guard
    if plan.finished:
        var fixed = _fixed_terms(uppers, lowers, z, plan.terms, work, plan.size, plan.peak, 8)
        if fixed:
            return fixed.value()[0]
        return _finished_sum(uppers, lowers, z, plan.terms, work)
    var pair = _terms(uppers, lowers, z, plan.terms, work, plan)
    if not pair[1].is_finite():
        return None
    var n = plan.terms
    var s = Ball(sigma)
    var one = _one(work)
    var half_rho = _scaled(Ball(_magnitude(_add(_scaled(_sub(_mul(a, a, work), _mul(a, b, work), work), 1, work), b, work)).to_float()), -1, work)
    var size = _magnitude(pair[1]).to_float()
    var t = s
    var c_n = one
    var c_1 = one
    if cut:
        # nu = (1/2 + sqrt(1 - 4 sigma**2)/2)**(-1/2), sigma -> nu sigma.
        var root = _fn(_SQRT, _sub(one, _scaled(_mul(s, s, work), 2, work), work), work)
        var nu = _div(one, _fn(_SQRT, _scaled(_add(one, root, work), -1, work), work), work)
        var nu2 = _mul(nu, nu, work)
        t = _mul(nu, s, work)
        var pi = _pi(work)
        var chi = _fn(_SQRT, _scaled(_mul(pi, Ball(Integer(n + 2), precision=work), work), -1, work), work)
        var power = _fn(_EXP, _mul(Ball(Integer(n), precision=work), _fn(_LOG, nu, work), work), work)
        c_n = _mul(_add(chi, _mul(_mul(s, nu2, work), Ball(Integer(n), precision=work), work), work), power, work)
        c_1 = _mul(_add(_scaled(pi, -1, work), _mul(s, nu2, work), work), nu, work)
    var alpha = _div(one, _sub(one, t, work), work)
    var rest = _sub(one, t, work)
    var rho = _add(half_rho, _div(_mul(t, _add(one, _scaled(t, -2, work), work), work), _mul(rest, rest, work), work), work)
    var growth = _fn(_EXP, _div(_scaled(_mul(_mul(alpha, rho, work), c_1, work), 1, work), Ball(low), work), work)
    var bound = _mul(_mul(_scaled(_mul(alpha, c_n, work), 1, work), Ball(size), work), growth, work)
    return _widen(pair[0], Ball(bound._exact_upper()))


# ----------------------------------------------------- gamma factors


def _sin_pi_ball(x: Ball, w: Int) raises -> Ball:
    """`sin(pi x)`, reduced exactly for an exact x."""
    if x.is_exact():
        return _sin_pi(x._midpoint, w)
    return _fn(_SIN, _mul(_pi(w + 4), x, w), w)


def _cos_pi_ball(x: Ball, w: Int) raises -> Ball:
    """`cos(pi x) = sin(pi (x + 1/2))`, reduced exactly for an exact x."""
    if x.is_exact():
        return _sin_pi(_exact_add(x._midpoint, Float(Rational(1, 2))), w)
    return _fn(_COS, _mul(_pi(w + 4), x, w), w)


def _rgamma(x: Ball, w: Int) raises -> Ball:
    """`1/Gamma(x)`, entire: 0 at the non-positive integers, directly for
    x > 0, else `sin(pi x) Gamma(1 - x) / pi`."""
    var t = x._midpoint
    if x.is_exact() and (t.is_zero() or (t._negative and t.is_integer())):
        return Ball(Integer(0), precision=w)
    var work = w
    if x._exact_lower() > Float(0):
        return _div(_one(work), _ball_special(_GAMMA, _BallArgument(x), 0, _c(work)), w)
    var reflected = _ball_special(_GAMMA, _BallArgument(_sub(_one(work), x, work)), 0, _c(work))
    return _div(_mul(_sin_pi_ball(x, work), reflected, work), _pi(work), w)


# ---------------------------------------------------------------- hyp1f1


def _hyp1f1_asymptotic(a: Ball, b: Ball, y: Ball, negative: Bool, w: Int) raises -> Optional[Ball]:
    """M(a, b, +-y) for y > 0 from the two expansions of U; None where they
    do not reach the precision. With `u, v = a, b - a` for x = y and
    `b - a, a` for x = -y (after Kummer's transformation),
    `M = Gamma(b) [e**(y or 0) y**(u-b) / Gamma(u) K(v, cut)
    + e**(0 or -y) cos(pi u) y**-u / Gamma(v) K(u)]`."""
    var work = w
    var u = _sub(b, a, work) if negative else a
    var v = a if negative else _sub(b, a, work)
    var first = _u_expansion(v, b, y, True, work)
    if not first:
        return None
    var second = _u_expansion(u, b, y, False, work)
    if not second:
        return None
    var log_y = _fn(_LOG, y, work)
    var e1 = _mul(_sub(u, b, work), log_y, work)
    if not negative:
        e1 = _add(e1, y, work)
    var t1 = _mul(_mul(_fn(_EXP, e1, work), _rgamma(u, work), work), first.value(), work)
    var rest = _mul(_rgamma(v, work), second.value(), work)
    var t2: Ball
    if negative and y._midpoint._exponent > 56:
        # |t2| <= e**-y y**|u| |rest| < 2**(|u| (E + 1) - 1.4426 y + e(rest)),
        # E the exponent of y; kept as a radius, since e**-y would leave the
        # exponent range.
        var size = _magnitude(rest)
        var top = Ball(size.to_float())
        if not top.is_finite() or top._midpoint.is_zero():
            t2 = Ball(Integer(0), precision=work)
        else:
            var limit = Integer(1) << 61
            var exponent = -limit
            if y._midpoint._exponent <= 60:
                var drop = (y._exact_lower().to_rational_exact() * Rational(14426, 10000)).floor()
                var reach = (_magnitude(u).to_float().ceil() + Integer(1)) * Integer(y._midpoint._exponent + 1)
                exponent = reach - drop + Integer(top._midpoint._exponent + 1)
            if exponent < -limit:
                exponent = -limit
            elif exponent > limit:
                exponent = limit^
            t2 = Ball(_midpoint=Float(0), _radius=_Radius.power_of_two(_int(exponent)), _kind=_FINITE)
    else:
        var e2 = _negate(_mul(u, log_y, work))
        if negative:
            e2 = _sub(e2, y, work)
        t2 = _mul(_mul(_fn(_EXP, e2, work), _cos_pi_ball(u, work), work), rest, work)
    var gamma_b = _ball_special(_GAMMA, _BallArgument(b), 0, _c(work))
    return _mul(gamma_b, _add(t1, t2, work), w)


def _hyp1f1_enclosure(a: Ball, b: Ball, x: Ball, w: Int) raises -> Ball:
    """M(a, b, x) for balls with no non-positive integer in b, to about w
    bits: the series at |x|, after Kummer's transformation for x < 0, or the
    asymptotic expansion where the series would be longer than about w
    terms; the series at x itself for a ball x around 0."""
    if x.is_exact() and x._midpoint.is_zero():
        return _one(w)
    var negative = x._exact_upper() < Float(0)
    if not negative and not (x._exact_lower() > Float(0)):
        var around: List[Ball] = [a]
        var below: List[Ball] = [b]
        return _series(around, below, x, w)
    if x._midpoint._exponent > 56 and not negative:
        return Ball.unbounded(w)
    if x._midpoint._exponent > (1 << 20):
        return Ball.indeterminate(w)
    var y = _negate(x) if negative else x
    # Past about w + |a| + |b| the expansion reaches the precision and the
    # series' terms peak near k = |x|: try it before planning the series.
    var far = y._midpoint._exponent > 60
    if not far and a._midpoint._exponent < 60 and b._midpoint._exponent < 60:
        var reach = Float64(w) + abs(a._midpoint.to_native[DType.float64]()) + abs(b._midpoint.to_native[DType.float64]())
        far = y._midpoint.to_native[DType.float64]() > reach
    if far:
        var expansion = _hyp1f1_asymptotic(a, b, y, negative, w)
        if expansion:
            return expansion.value()
    var p = _sub(b, a, w + 16) if negative else a
    var uppers: List[Ball] = [p]
    var lowers: List[Ball] = [b]
    var plan = _plan(uppers, lowers, y, w, False)
    if not far and (plan.terms < 0 or plan.terms > w):
        var expansion = _hyp1f1_asymptotic(a, b, y, negative, w)
        if expansion:
            return expansion.value()
    if plan.terms < 0:
        return Ball.indeterminate(w)
    var sum = _planned_series(uppers, lowers, y, w, plan)
    if not negative:
        return _rounded_ball(_BallArgument(sum), w)
    return _mul(_fn(_EXP, x, w), sum, w)


def _hyp1f1_polynomial(a: Rational, b: Rational, x: Rational) raises -> _Decided:
    """The polynomial M(-n, b, x), exactly, or neither past the size bound."""
    var n = -a
    if n.numerator().magnitude_bit_length() > 20:
        return _neither()
    var uppers: List[Rational] = [a]
    var lowers: List[Rational] = [b]
    var value = _polynomial(uppers, lowers, x, Int(n.numerator()._low_magnitude()))
    if not value:
        return _neither()
    return _Decided(_RATIONAL, value.value())


def _hyp1f1_elementary(a: Int, b: Int, x: Rational) raises -> _Decided:
    """M(a, b, x) for integers 0 < a < b where the sum multiplying `e**x`
    vanishes (module docstring); neither elsewhere and past the size bound."""
    var m = b - a - 1
    var d = b - 2
    if d > 4096 or d * (_rational_bits(x) + 2 * _bit_count(d) + 2) > _EXACT_BITS:
        return _neither()
    # x**(d+1) sum_{j=m}^{d} (-1)**(j+m) j! C(a-1, j-m) / x**(j+1).
    var at_one = Rational(0)
    var power = Rational(1)
    var j = d
    while j >= m:
        var coefficient = factorial(Integer(j)) * comb(Integer(a - 1), Integer(j - m))
        var term = Rational(coefficient) * power
        at_one = at_one - term if (j + m) % 2 == 1 else at_one + term
        power = power * x
        j -= 1
    if at_one.sign() != 0:
        return _neither()
    var inverse = Rational(1) / x
    var step = inverse
    var total = Rational(0)
    for k in range(d + 1):
        step = step * inverse if k > 0 else step
        if k >= a - 1:
            total = total + Rational(factorial(Integer(k)) * comb(Integer(m), Integer(k - a + 1))) * step
    var c = factorial(Integer(b - 1)) / (factorial(Integer(a - 1)) * factorial(Integer(m)))
    var value = c * total
    return _Decided(_RATIONAL, value if (a - 1) % 2 == 1 else -value)


def _scipy_polynomial(a: Rational, b: Rational) raises -> Bool:
    """Whether scipy takes M(a, b, x) at a non-positive integer b as the
    polynomial that a negative integer a >= b ends before the pole."""
    return a.sign() < 0 and a.is_integer() and a >= b


def _hyp1f1_case(a: Rational, b: Rational, x: Rational) raises -> _Decided:
    """M(a, b, x) decided without the series: a rational value (module
    docstring), scipy's `+inf` at a non-positive integer b, or neither."""
    if _nonpositive_integer(b):
        if _scipy_polynomial(a, b):
            return _hyp1f1_polynomial(a, b, x)
        return _infinite()
    if x.sign() == 0 or a.sign() == 0:
        return _Decided(_RATIONAL, Rational(1))
    if _nonpositive_integer(a):
        return _hyp1f1_polynomial(a, b, x)
    if _nonpositive_integer(b - a):
        var kummer = _hyp1f1_polynomial(b - a, b, -x)
        if kummer.kind == _RATIONAL and kummer.value.sign() == 0:
            return kummer
        return _neither()
    if a.is_integer() and b.is_integer() and a.sign() > 0 and a < b and b.numerator().magnitude_bit_length() <= 13:
        return _hyp1f1_elementary(Int(a.numerator()._low_magnitude()), Int(b.numerator()._low_magnitude()), x)
    return _neither()


def _moderate_value(x: Ball) raises -> Optional[Rational]:
    """The value of an exact ball as a Rational, when it is of modest size."""
    if not x.is_exact():
        return None
    var t = x._midpoint
    if t.is_zero():
        return Rational(0)
    if t._exponent > _MODERATE_BITS or t._exponent < -_MODERATE_BITS or t.precision() > _MODERATE_BITS:
        return None
    return t.to_rational_exact()


def _meets_pole(b: Ball) raises -> Bool:
    """Whether the ball b holds a non-positive integer."""
    var low = b._exact_lower()
    if low > Float(0):
        return False
    var high = b._exact_upper()
    if not (high < Float(0)):
        return True
    return _integral_between(low, high)


def _hyp1f1_ball(a: _BallArgument, b: _BallArgument, x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """M(a, b, x) of three balls: a ball containing it at every triple of
    points, exact at its rational values, indeterminate where b may be a
    pole."""
    var w = _working(context, a.precision, b.precision, x.precision)
    if a.kind != _FINITE or b.kind != _FINITE or x.kind != _FINITE:
        return Ball.indeterminate(w)
    var ab = a.ball()
    var bb = b.ball()
    var xb = x.ball()
    var ar = _moderate_value(ab)
    var br = _moderate_value(bb)
    var xr = _moderate_value(xb)
    if ar and br and xr:
        var decided = _hyp1f1_case(ar.value(), br.value(), xr.value())
        if decided.kind == _RATIONAL:
            return Ball(decided.value, precision=w)
        if decided.kind == _INFINITE:
            return Ball.indeterminate(w)
    elif _meets_pole(bb):
        return Ball.indeterminate(w)
    var work = w + _BALL_GUARD
    var value = _hyp1f1_enclosure(_rounded_ball(a, work), _rounded_ball(b, work), _rounded_ball(x, work), work)
    return _rounded_ball(_BallArgument(value), w)


# ---------------------------------------------------------------- hyp2f1

comptime _OUTSIDE = 3
"""A decided case outside the function's real domain."""


def _rational_power(r: Rational, e: Rational) raises -> Optional[Rational]:
    """`r**e` for r > 0 when it is rational: e an integer, or `r` a perfect
    power of e's denominator; None when it is not, or past the size bound."""
    var q = e.denominator()
    var p = e.numerator()
    if q.magnitude_bit_length() > 20 or p.magnitude_bit_length() > 20:
        return None
    var base = r
    if q != Integer(1):
        var top = iroot_exact(r.numerator(), q)
        var bottom = iroot_exact(r.denominator(), q)
        if not top or not bottom:
            return None
        base = Rational(top.value(), bottom.value())
    var n = Int(p._low_magnitude())
    if n * (_rational_bits(base) + 1) > _EXACT_BITS:
        return None
    var value = Rational(1)
    var square = base
    while n > 0:
        if n & 1:
            value = value * square
        n >>= 1
        if n > 0:
            square = square * square
    return Rational(1) / value if p.sign() < 0 else value^


def _gauss_polynomial(a: Rational, b: Rational, c: Rational, x: Rational, m: Int) raises -> _Decided:
    """`F(a, b; c; x)` ended after degree m, exactly, or neither past the size
    bound."""
    var uppers: List[Rational] = [a, b]
    var lowers: List[Rational] = [c]
    var value = _polynomial(uppers, lowers, x, m)
    if not value:
        return _neither()
    return _Decided(_RATIONAL, value.value())


def _degree(q: Rational) raises -> Int:
    """`-q` for a non-positive integer q of modest size, else -1."""
    if not _nonpositive_integer(q) or q.numerator().magnitude_bit_length() > 20:
        return -1
    return Int(q.numerator()._low_magnitude())


def _hyp2f1_case(a: Rational, b: Rational, c: Rational, x: Rational) raises -> _Decided:
    """F(a, b; c; x) decided without the series (module docstring of the
    section): a rational value, scipy's `+inf`, outside the domain (x > 1
    other than a polynomial), or neither."""
    if x.sign() == 0:
        return _Decided(_RATIONAL, Rational(1))
    var da = _degree(a)
    var db = _degree(b)
    if _nonpositive_integer(a) or _nonpositive_integer(b):
        # A polynomial at every x, of the smaller degree; its terms reach a
        # pole of c only past -c.
        var m = da if db < 0 or (da >= 0 and da <= db) else db
        if m < 0:
            return _neither()
        if _nonpositive_integer(c) and -c < Rational(m):
            return _infinite()
        return _gauss_polynomial(a, b, c, x, m)
    if _nonpositive_integer(c):
        return _infinite()
    if x > Rational(1):
        return _Decided(_OUTSIDE, Rational(0))
    var s = c - a - b
    if x == Rational(1):
        if s.sign() <= 0:
            # Chu-Vandermonde at s = 0 when c - a = -k: F(-k, c-b; c; 1) = (b)_k / (c)_k.
            if s.sign() == 0 and (_nonpositive_integer(c - a) or _nonpositive_integer(c - b)):
                var k = _degree(c - a) if _nonpositive_integer(c - a) else _degree(c - b)
                var other = b if _nonpositive_integer(c - a) else a
                if k < 0:
                    return _neither()
                var top = _poch_decided(other, Rational(k))
                var bottom = _poch_decided(c, Rational(k))
                if top.kind == _RATIONAL and bottom.kind == _RATIONAL and bottom.value.sign() != 0:
                    return _Decided(_RATIONAL, top.value / bottom.value)
                return _neither()
            return _infinite()
        # Gauss: Gamma(c) Gamma(s) / (Gamma(c-a) Gamma(c-b)), 0 where c - a or
        # c - b is a non-positive integer, (c-a)_a / (s)_a for a positive
        # integer a.
        if _nonpositive_integer(c - a) or _nonpositive_integer(c - b):
            return _Decided(_RATIONAL, Rational(0))
        for which in range(2):
            var p = a if which == 0 else b
            if p.is_integer() and p.sign() > 0 and p.numerator().magnitude_bit_length() <= 20:
                var top = _poch_decided(c - p, p)
                var bottom = _poch_decided(s, p)
                if top.kind == _RATIONAL and bottom.kind == _RATIONAL and bottom.value.sign() != 0:
                    return _Decided(_RATIONAL, top.value / bottom.value)
        return _neither()
    var rest = Rational(1) - x
    if c == a or c == b:
        var power = _rational_power(rest, -(b if c == a else a))
        if power:
            return _Decided(_RATIONAL, power.value())
        return _neither()
    # Euler's transformation, F = (1-x)**s F(c-a, c-b; c; x), with a
    # polynomial for c - a or c - b a non-positive integer.
    var dca = _degree(c - a)
    var dcb = _degree(c - b)
    if dca >= 0 or dcb >= 0:
        var k = dca if dcb < 0 or (dca >= 0 and dca <= dcb) else dcb
        var inner = _gauss_polynomial(c - a, c - b, c, x, k)
        if inner.kind != _RATIONAL:
            return _neither()
        if inner.value.sign() == 0:
            return inner
        var power = _rational_power(rest, s)
        if power:
            return _Decided(_RATIONAL, power.value() * inner.value)
    return _neither()


def _power_ball(base: Ball, exponent: Ball, w: Int) raises -> Ball:
    """`base**exponent` for a ball base > 0: an integer power for an exact
    integer exponent, else `exp(exponent log base)`."""
    var e = exponent._midpoint
    if exponent.is_exact() and (e.is_zero() or (e.is_integer() and e._exponent <= 40)):
        return _power(_BallArgument(base), e.round(), _c(w))
    return _fn(_EXP, _mul(exponent, _fn(_LOG, base, w), w), w)


def _ends(p: Ball) -> Bool:
    """Whether p is an exact non-positive integer, which ends the series."""
    var t = p._midpoint
    return p.is_exact() and (t.is_zero() or (t._negative and t.is_integer()))


def _gauss_series(a: Ball, b: Ball, c: Ball, x: Ball, w: Int) raises -> Ball:
    var uppers: List[Ball] = [a, b]
    var lowers: List[Ball] = [c]
    return _series(uppers, lowers, x, w)


def _psi_gap(gap: Ball, least: Float, w: Int) raises -> Ball:
    """A bound of `|psi(u) - psi(v)|` for `|u - v| <= |gap|` and
    `u, v >= least > 0`: `psi'` decreases and `psi'(t) < 1/t + 1/t**2`."""
    var t = Ball(least)
    var inverse = _div(_one(w), t, w)
    return _mul(Ball(_magnitude(gap).to_float()), _add(inverse, _mul(inverse, inverse, w), w), w)


def _gauss_degenerate(a: Ball, b: Ball, m: Int, t: Ball, w: Int) raises -> Ball:
    """The regularized `F(a, b; a+b+m; 1-t) / Gamma(a+b+m)` for an integer
    m >= 0 and 0 < t < 1 (DLMF 15.8.10):

        sum_{k<m} (a)_k (b)_k (m-k-1)! / k! (-t)**k / (Gamma(a+m) Gamma(b+m))
        - (-t)**m / (Gamma(a) Gamma(b)) sum_k (a+m)_k (b+m)_k / (k! (k+m)!) t**k L_k,

    `L_k = log t - psi(k+1) - psi(k+m+1) + psi(a+k+m) + psi(b+k+m)`. The
    digammas step by `psi(u+1) = psi(u) + 1/u`, so with `a_k = a+m+k` and
    `b_k = b+m+k`, `L_{k+1} = L_k + (a_k + b_k)/(a_k b_k) - (2k+m+2)/((k+1)
    (k+m+1))`, `a_k b_k` shared with the term's step. Past N, `|L_k|` is at most
    `|log t| + |a+m-1| h(min(a+m, 1) + N) + |b-1| h(min(b, 1) + m + N)`,
    `h(t) = 1/t + 1/t**2` (`_psi_gap`), and the terms obey the ratio bound."""
    var work = w
    var minus_t = _negate(t)
    var finite = Ball(Integer(0), precision=work)
    if m > 0:
        var term = Ball(factorial(Integer(m - 1)), precision=work)
        for k in range(m):
            finite = _add(finite, term, work)
            if k + 1 < m:
                var shift = Ball(Integer(k), precision=work)
                var rise = _mul(_add(a, shift, work), _add(b, shift, work), work)
                term = _div_n(_mul(_mul(term, rise, work), minus_t, work), Integer((k + 1) * (m - k - 1)), work)
    var mb = Ball(Integer(m), precision=work)
    var am = _add(a, mb, work)
    var bm = _add(b, mb, work)
    var part_a = _mul(_mul(_rgamma(am, work), _rgamma(bm, work), work), finite, work)
    var uppers: List[Ball] = [am, bm]
    var lowers: List[Ball] = [Ball(Integer(m + 1), precision=work)]
    var plan = _plan(uppers, lowers, t, work, False)
    if plan.terms < 0:
        return Ball.indeterminate(w)
    var inner = work + plan.guard
    var p1 = _negate(_euler_gamma(inner))
    var pm = p1
    for j in range(1, m + 1):
        pm = _add(pm, _div(_one(inner), Ball(Integer(j), precision=inner), inner), inner)
    var pa = _ball_special(_DIGAMMA, _BallArgument(am), 0, _c(inner))
    var pb = _ball_special(_DIGAMMA, _BallArgument(bm), 0, _c(inner))
    var log_t = _fn(_LOG, t, inner)
    var term = _div(_one(inner), Ball(factorial(Integer(m)), precision=inner), inner)
    var total = Ball(Integer(0), precision=inner)
    var n = plan.terms
    var bracket = _add(_sub(_sub(log_t, p1, inner), pm, inner), _add(pa, pb, inner), inner)
    var one = _one(inner)
    var top_a = am
    var top_b = bm
    # The sum of `term bracket` as one dot product; a_k and b_k step in place.
    var terms = List[Ball](capacity=n)
    var brackets = List[Ball](capacity=n)
    for k in range(n):
        terms.append(term)
        brackets.append(bracket)
        var product = _mul(top_a, top_b, inner)
        var ends = (k + 1) * (k + m + 1)
        term = _div_n(_mul(_mul(term, product, inner), t, inner), Integer(ends), inner)
        var steps = _sub(_div(_add(top_a, top_b, inner), product, inner), _div_n(Ball(Integer(2 * k + m + 2)), Integer(ends), inner), inner)
        bracket = _add(bracket, steps, inner)
        _ball_add_assign(top_a, one, inner)
        _ball_add_assign(top_b, one, inner)
    total = _ball_dot(total, terms, brackets, inner)
    if not plan.finished:
        if not term.is_finite():
            return Ball.indeterminate(w)
        var ratio = _ratio_bound(uppers, lowers, t, n, inner)
        if not ratio:
            return Ball.indeterminate(w)
        var one = Float(1)
        var least_a = _exact_add(am._exact_lower() if am._exact_lower() < one else one, Float(n))
        var least_b = _exact_add(b._exact_lower() if b._exact_lower() < one else one, Float(n + m))
        if not (least_a > Float(0)) or not (least_b > Float(0)):
            return Ball.indeterminate(w)
        var bar = _add(Ball(_magnitude(log_t).to_float()), _add(
            _psi_gap(_sub(am, _one(inner), inner), least_a, inner),
            _psi_gap(_sub(b, _one(inner), inner), least_b, inner), inner), inner)
        var tail = _div(_mul(Ball(_magnitude(term).to_float()), bar, inner), _sub(_one(inner), Ball(ratio.value()), inner), inner)
        total = _widen(total, tail)
    var power = _power(_BallArgument(minus_t), Integer(m), _c(work))
    var part_b = _mul(_mul(_mul(power, _rgamma(a, work), work), _rgamma(b, work), work), total, work)
    return _sub(part_a, part_b, w)


def _meets_integer(x: Ball) raises -> Bool:
    return _integral_between(x._exact_lower(), x._exact_upper())


def _gauss_connection(a: Ball, b: Ball, c: Ball, y: Ball, w: Int) raises -> Ball:
    """F(a, b; c; y) for 1/2 < y < 1 by the connection with `1 - y` (DLMF
    15.8.4), `s = c - a - b`:

        Gamma(c) [Gamma(s) / (Gamma(c-a) Gamma(c-b)) F(a, b; 1-s; 1-y)
                  + (1-y)**s Gamma(-s) / (Gamma(a) Gamma(b)) F(c-a, c-b; 1+s; 1-y)],

    for an integer s the limit 15.8.10 (`_gauss_degenerate`), after Euler's
    transformation `F = (1-y)**s F(c-a, c-b; c; y)` when s < 0;
    indeterminate for a ball s around an integer. Of `Gamma(s)` and
    `Gamma(-s)`, the one at the positive argument is evaluated and the other
    follows by reflection, `Gamma(s) Gamma(-s) = -pi / (s sin(pi s))`."""
    var work = w
    var s = _sub(_sub(c, a, work), b, work)
    var t = _sub(_one(work), y, work)
    var gamma_c = _ball_special(_GAMMA, _BallArgument(c), 0, _c(work))
    if s.is_exact() and (s._midpoint.is_zero() or s._midpoint.is_integer()):
        if not s._midpoint.is_zero() and s._midpoint._exponent > 20:
            return Ball.indeterminate(w)
        var m = Int(s._midpoint.round()._low_magnitude())
        if s._midpoint._negative:
            var inner = _gauss_degenerate(_sub(c, a, work), _sub(c, b, work), m, t, work)
            return _mul(_mul(gamma_c, _power_ball(t, s, work), work), inner, w)
        return _mul(gamma_c, _gauss_degenerate(a, b, m, t, work), w)
    if _meets_integer(s):
        return Ball.indeterminate(w)
    var first = _gauss_series(a, b, _sub(_one(work), s, work), t, work)
    var second = _gauss_series(_sub(c, a, work), _sub(c, b, work), _add(_one(work), s, work), t, work)
    var positive = s._exact_lower() > Float(0)
    var direct = _ball_special(_GAMMA, _BallArgument(s if positive else _negate(s)), 0, _c(work))
    var reflected = _negate(_div(_pi(work), _mul(_mul(s, _sin_pi_ball(s, work), work), direct, work), work))
    var gamma_s = direct if positive else reflected
    var gamma_ms = reflected if positive else direct
    var term1 = _mul(_mul(_mul(gamma_s, _rgamma(_sub(c, a, work), work), work), _rgamma(_sub(c, b, work), work), work), first, work)
    var term2 = _mul(_mul(_mul(_mul(_power_ball(t, s, work), gamma_ms, work), _rgamma(a, work), work), _rgamma(b, work), work), second, work)
    return _mul(gamma_c, _add(term1, term2, work), w)


def _gauss_at_one(a: Ball, b: Ball, c: Ball, w: Int) raises -> Ball:
    """Gauss's `F(a, b; c; 1) = Gamma(c) Gamma(s) / (Gamma(c-a) Gamma(c-b))`
    for `s = c - a - b > 0` (DLMF 15.4.20)."""
    var work = w
    var s = _sub(_sub(c, a, work), b, work)
    if not (s._exact_lower() > Float(0)):
        return Ball.indeterminate(w)
    var top = _mul(_ball_special(_GAMMA, _BallArgument(c), 0, _c(work)), _ball_special(_GAMMA, _BallArgument(s), 0, _c(work)), work)
    return _mul(_mul(top, _rgamma(_sub(c, a, work), work), work), _rgamma(_sub(c, b, work), work), w)


def _same(p: Ball, q: Ball) -> Bool:
    return p.is_exact() and q.is_exact() and p._midpoint == q._midpoint


def _hyp2f1_unit(a: Ball, b: Ball, c: Ball, y: Ball, w: Int) raises -> Ball:
    """F(a, b; c; y) for a ball y in `(-1, 1)` that needs no Pfaff
    transformation: `(1-y)**-b` for c = a (or -a for c = b), the connection
    above 1/2, else the series, which a non-positive integer a or b ends."""
    if _ends(a) or _ends(b):
        return _gauss_series(a, b, c, y, w)
    if _same(a, c) or _same(b, c):
        var rest = _sub(_one(w + 16), y, w + 16)
        return _power_ball(rest, _negate(b if _same(a, c) else a), w)
    if y._exact_lower() > Float(Rational(1, 2)):
        var joined = _gauss_connection(a, b, c, y, w)
        if joined.is_finite():
            return joined
    return _gauss_series(a, b, c, y, w)


def _hyp2f1_enclosure(a: Ball, b: Ball, c: Ball, x: Ball, w: Int) raises -> Ball:
    """F(a, b; c; x) for balls with no non-positive integer in c and x < 1,
    or Gauss's value at an exact x = 1 with `c - a - b > 0`: the argument is
    made positive by Pfaff's transformation `F = (1-x)**-a F(a, c-b; c;
    x/(x-1))` (DLMF 15.8.1) with the smaller of |a| and |b|, then summed up
    to 1/2 and connected with `1 - x` beyond."""
    if x.is_exact() and x._midpoint == Float(1):
        return _gauss_at_one(a, b, c, w)
    if not (x._exact_upper() < Float(1)):
        return Ball.indeterminate(w)
    if _ends(a) or _ends(b) or not (x._exact_upper() < Float(0)) or x._exact_lower() >= Float(Rational(-1, 2)):
        return _hyp2f1_unit(a, b, c, x, w)
    var work = w
    var swap = abs(b._midpoint) < abs(a._midpoint)
    var p = b if swap else a
    var q = a if swap else b
    var rest = _sub(_one(work), x, work)
    # 1 - 1/(1 - x), not -x/(1 - x): x once, so a ball x keeps its width;
    # [-4.87, -3.79] gives [0.79, 0.83], where the two-x form gave [0.58, 1.02].
    var y = _sub(_one(work), _div(_one(work), rest, work), work)
    var inner = _hyp2f1_unit(p, _sub(c, q, work), c, y, work)
    return _mul(_power_ball(rest, _negate(p), work), inner, w)


def _hyp2f1_ball(a: _BallArgument, b: _BallArgument, c: _BallArgument, x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """F(a, b; c; x) of four balls: a ball containing it at every point,
    exact at its rational values; indeterminate where c may be a pole, where
    x reaches 1 (other than Gauss's exact x = 1), and beyond."""
    var w = _working(context, max(a.precision, b.precision), c.precision, x.precision)
    if a.kind != _FINITE or b.kind != _FINITE or c.kind != _FINITE or x.kind != _FINITE:
        return Ball.indeterminate(w)
    var ab = a.ball()
    var bb = b.ball()
    var cb = c.ball()
    var xb = x.ball()
    var ar = _moderate_value(ab)
    var br = _moderate_value(bb)
    var cr = _moderate_value(cb)
    var xr = _moderate_value(xb)
    if ar and br and cr and xr:
        var decided = _hyp2f1_case(ar.value(), br.value(), cr.value(), xr.value())
        if decided.kind == _RATIONAL:
            return Ball(decided.value, precision=w)
        if decided.kind != 0:
            return Ball.indeterminate(w)
    elif _meets_pole(cb) and not (_ends(ab) or _ends(bb)):
        return Ball.indeterminate(w)
    var work = w + _BALL_GUARD
    var value = _hyp2f1_enclosure(
        _rounded_ball(a, work), _rounded_ball(b, work), _rounded_ball(c, work), _rounded_ball(x, work), work,
    )
    return _rounded_ball(_BallArgument(value), w)

