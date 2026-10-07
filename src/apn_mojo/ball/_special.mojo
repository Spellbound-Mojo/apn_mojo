"""Kernels and ball versions of the special functions: Gamma, log Gamma,
digamma, the error functions, the exponential, sine, cosine and hyperbolic
integrals, Fresnel's integrals and Lambert's W.

`_special_kernel(code, x, n, w, budget)` returns a ball containing the
function at a finite Float, with about `w` bits of relative accuracy away
from the function's zeros. Everything is ball arithmetic, so every rounding
is in the radius; a truncated series adds a bound of its remainder, and a
series is chosen so that its cancellation and its length stay within a few
times `w`:

| Function | Method | Remainder |
|---|---|---|
| erf | `(2/sqrt pi) e**(-x**2) sum 2**n x**(2n+1) / (2n+1)!!`, positive terms; 1 within `e**(-x**2)` once that is below the precision | once a term ratio is at most 1/2, the tail is below the last term |
| erfc | `1 + erf|x|` for x < 0; `1 - erf x` with `1.443 x**2` extra bits; for large x, `e**(-x**2)/(x sqrt pi) sum (-1)**k (2k-1)!!/(2x**2)**k` | the first omitted term (DLMF 7.12(i)) |
| erfi | `(2/sqrt pi) sum x**(2n+1) / (n! (2n+1))`; for large x, `(2/sqrt pi) e**(x**2) D(x)` with Dawson's `D(x) = sum_{k<n} (2k-1)!!/(2**(k+1) x**(2k+1))` | `-n e**(-x**2)/x <= D - sum <= 2 sqrt(2n) T_n + (x/sqrt 2) e**(-x**2/2)` for `n <= x**2/2`, from `D(x) = (x/2) int_0^1 e**(-x**2 v) (1-v)**(-1/2) dv` |
| Ei | `gamma + log|x| + sum x**n / (n n!)`; for large x < 0, `-e**x/(-x) sum (-1)**k k!/x**k`; for large x > 0, `e**x sum_{k<n} k!/x**(k+1)` | ratio bound; the first omitted term (DLMF 6.12(ii)); `2 n!/x**(n+1) + 8 e**(-x/2)/x` for `n <= x/4`, `x >= 16`, from `e**(-x) Ei(x) = PV int_0^inf e**(-t)/(x - t) dt` split at `x/2` |
| Si, Ci | their series with `1.443 |x|` extra bits; for large x, `pi/2 - f cos x - g sin x` and `f sin x - g cos x` | ratio bound; the first omitted terms of f and g (DLMF 6.12(ii)) |
| Shi, Chi | their series, positive terms; for large x, `(Ei(x) +- E1(x)) / 2` with `0 < E1(x) < e**(-x)/x` | ratio bound; as Ei |
| Fresnel S, C | series in `u = pi x**2 / 2` with `1.443 u` extra bits; for large x, `1/2 - f cos u - g sin u` and `1/2 + f sin u - g cos u`, u reduced exactly modulo 2 pi | ratio bound; the first omitted terms (DLMF 7.12(ii)) |
| Gamma, log Gamma, digamma | Stirling's series at `z = x + N >= w/2 + 10`, its coefficients integers from the tabulated tangent numbers (`_bernoulli_table.mojo`); back to x by `Gamma(z) / (x (x+1) ... (x+N-1))`, the product's logarithm, or `sum 1/(x+k)`, exact in integers for a short dyadic x; reflection below 1/2 (for log Gamma below 0, `log |Gamma(x)| = log pi - log |sin pi x| - log Gamma(1 - x)`) | the first omitted term, for a real z (DLMF 5.11(ii)) |
| ndtr, log ndtr | `erfc(-x/sqrt 2)/2` from the erfc ball at `x/sqrt 2`; its logarithm, as `log1p(-erfc(x/sqrt 2)/2)` for x > 0 | as erfc |
| erfinv, ndtri | Newton's iteration at doubling precision on erf for `|x| <= 1/2` and on erfc in the tails, from Giles's approximation (2010) or `t = sqrt(L - log(t sqrt pi))` with `L = -log y`; then the signs of `f(t (1 -+ 2**-(w+8))) - x` | none: the root is bracketed |
| Lambert W | Halley's iteration, then the signs of `t e**t - x` at both ends of an interval | none: the root is bracketed |

The ball functions evaluate a monotone function at the two ends of the ball,
and the others at the midpoint, widened by a bound of the derivative; Gamma,
log Gamma and digamma take the midpoint for a narrow ball even where they are
monotone (`_gamma_spread`).
"""

from std.math import log, sqrt
from std.bit import count_trailing_zeros
from ..integer.value import Integer
from ..integer.math import factorial
from ..integer.number_theory import isqrt
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, _exact_context
from ..float._arithmetic import _float_operation, _FloatArgument
from ..float._functions import _scale_float
from ..float.context import RoundingMode
from .context import BallContext
from ._radius import _Radius, _up_word, _up, _double_up
from ..float._rounding import _RoundedBinary, _wide_rounding
from ..float.status import NumericStatus
from .value import Ball, _BallArgument, _FINITE, _INDETERMINATE, _UNBOUNDED
from ._arithmetic import _sum, _product, _quotient, _scale2, _hull, _working, _rounded_ball, _ball_sum, _ball_product, _ball_quotient, _ball_scale2, _ball_product_word, _ball_quotient_word, _negate
from ._constants import _pi, _euler_gamma
from ._functions import _ball_function, _ball_budget, _monotone_by, _unit_ball
from ._kernels import _EXP, _LOG, _LOG1P, _SIN, _COS, _SQRT, _integral
from ._trig import _kernel_sin_cos
from ._certified import _Enclosure
from ._medium import _hex_limb
from ._bernoulli_table import _TANGENT_COUNT, _TANGENT_SIZES, _TANGENT_LIMBS
from ._rgamma import _rgamma_unit, _rgamma_series, _shift_factors, _gamma_word, _WordGamma, _RGAMMA_PRECISION
from ._fixed_series import _bit_count

comptime _GAMMA = 40
comptime _LOG_GAMMA = 41
comptime _DIGAMMA = 42
comptime _ERF = 43
comptime _ERFC = 44
comptime _ERFI = 45
comptime _EI = 46
comptime _SI = 47
comptime _CI = 48
comptime _SHI = 49
comptime _CHI = 50
comptime _FRESNEL_S = 51
comptime _FRESNEL_C = 52
comptime _LAMBERT_W = 53
comptime _NDTR = 54
comptime _LOG_NDTR = 55
comptime _BETA = 56
comptime _LOG_BETA = 57
comptime _POCH = 58
comptime _ERFINV = 59
comptime _NDTRI = 60
comptime _ZETA = 61
comptime _POLYGAMMA = 62

comptime _MAX_TERMS = 1 << 22


def _special_name(code: Int) -> StaticString:
    if code == _GAMMA:
        return "gamma"
    if code == _LOG_GAMMA:
        return "gammaln"
    if code == _DIGAMMA:
        return "digamma"
    if code == _ERF:
        return "erf"
    if code == _ERFC:
        return "erfc"
    if code == _ERFI:
        return "erfi"
    if code == _EI:
        return "expi"
    if code == _SI:
        return "sici"
    if code == _CI:
        return "sici"
    if code == _SHI:
        return "shichi"
    if code == _CHI:
        return "shichi"
    if code == _FRESNEL_S:
        return "fresnel"
    if code == _FRESNEL_C:
        return "fresnel"
    if code == _NDTR:
        return "ndtr"
    if code == _LOG_NDTR:
        return "log_ndtr"
    if code == _BETA:
        return "beta"
    if code == _LOG_BETA:
        return "betaln"
    if code == _POCH:
        return "poch"
    if code == _ERFINV:
        return "erfinv"
    if code == _NDTRI:
        return "ndtri"
    if code == _ZETA:
        return "zeta"
    if code == _POLYGAMMA:
        return "polygamma"
    return "lambertw"


# ------------------------------------------------------------- helpers


def _c(w: Int) raises -> Optional[BallContext]:
    return Optional[BallContext](BallContext(w))


def _add(a: Ball, b: Ball, w: Int) raises -> Ball:
    return _ball_sum(a, b, False, w)


def _sub(a: Ball, b: Ball, w: Int) raises -> Ball:
    return _ball_sum(a, b, True, w)


def _mul(a: Ball, b: Ball, w: Int) raises -> Ball:
    return _ball_product(a, b, w)


def _div(a: Ball, b: Ball, w: Int) raises -> Ball:
    return _ball_quotient(a, b, w)


def _mul_n(a: Ball, n: Integer, w: Int) raises -> Ball:
    if n._storage.isa[Int64]():
        return _ball_product_word(a, Int(n._storage[Int64]), w)
    return _product(_BallArgument(a), _BallArgument(n), _c(w))


def _div_n(a: Ball, n: Integer, w: Int) raises -> Ball:
    if n._storage.isa[Int64]():
        return _ball_quotient_word(a, Int(n._storage[Int64]), w)
    return _quotient(_BallArgument(a), _BallArgument(n), _c(w))


def _scaled(a: Ball, k: Int, w: Int) raises -> Ball:
    """`a * 2**k`, exactly."""
    return _ball_scale2(a, k, w)


def _fn(code: Int, a: Ball, w: Int) raises -> Ball:
    return _ball_function(code, _BallArgument(a), 0, _c(w))


def _one(w: Int) raises -> Ball:
    return Ball(Integer(1), precision=w)


def _magnitude(a: Ball) raises -> _Radius:
    """An upper bound of `|a|`."""
    return _Radius.upper(a._midpoint).add(a._radius)


def _widen(a: Ball, bound: Ball) raises -> Ball:
    """`a` widened by every value of `bound` in absolute value."""
    if not a.is_finite() or not bound.is_finite():
        return Ball.indeterminate(a.precision())
    return Ball(_midpoint=a._midpoint, _radius=a._radius.add(_magnitude(bound)), _kind=_FINITE)


def _below(term: Ball, exponent: Int) raises -> Bool:
    """Whether every point of `term` is below `2**exponent` in magnitude."""
    return term.is_finite() and _magnitude(term).compare(_Radius.power_of_two(exponent)) < 0


def _settled(term: Ball, total: Ball, bits: Int) raises -> Bool:
    """Whether `|term| < 2**-bits max(|total|, 1)`."""
    var e = 1
    if total.is_finite() and not total._midpoint.is_zero():
        e = max(total._midpoint._exponent, 1)
    return _below(term, e - bits - 1)


def _floor_scaled(x: Float, numerator: Int, denominator: Int, square: Bool) raises -> Int:
    """`floor(|x| a / b)`, or `floor(x**2 a / b)`, clamped to `2**60`; a lower
    bound of the bits that `e**|x|` or `e**(x**2)` spans when `a / b` is below
    log2(e)."""
    var limit = Int(1) << 60
    if x.is_zero():
        return 0
    if x._exponent > (29 if square else 58):
        return limit
    var r = abs(x).to_rational_exact()
    var v = r * r if square else r
    var result = (v * Rational(numerator, denominator)).floor()
    return Int(result) if result < Integer(limit) else limit


def _exact_difference(a: Float, b: Float) raises -> Float:
    return Float(_rounded=_float_operation(a, b, 1, _exact_context()))


def _sqrt_pi(w: Int) raises -> Ball:
    return _fn(_SQRT, _pi(w + 4), w)


def _sin_cos_ball(t: Ball, w: Int, budget: Int) raises -> Tuple[Ball, Ball]:
    """`(sin t, cos t)` of a ball: the kernel at the midpoint, widened by the
    radius, since both derivatives are at most 1."""
    if t._midpoint.is_zero():
        return (Ball(_midpoint=t._midpoint, _radius=t._radius, _kind=_FINITE), _widen(_one(w), Ball(_midpoint=Float(0), _radius=t._radius, _kind=_FINITE)))
    var pair = _kernel_sin_cos(t._midpoint, w, budget)
    var spread = Ball(_midpoint=Float(0), _radius=t._radius, _kind=_FINITE)
    return (_widen(pair[0], spread), _widen(pair[1], spread))


def _enveloping_sum(z: Ball, w: Int, kind: Int) raises -> Optional[Ball]:
    """`sum_m t_m` with `t_0 = 1` and `t_m = -t_(m-1) c(m) / z`, for the
    asymptotic series whose remainder, at a real positive argument, is below
    its first omitted term: erfc's (kind 0, `c = 2m - 1`, `z = 2x**2`), E1's
    (1, `c = m`, `z = x`), the f and g of the sine and cosine integrals (2, 3:
    `c = (2m-1) 2m` and `2m (2m+1)`, `z = x**2`) and of Fresnel's integrals (4,
    5: `c = (4m-3)(4m-1)` and `(4m-1)(4m+1)`, `z = (pi x**2)**2`). None when
    the terms grow before they are negligible."""
    var term = _one(w)
    var total = term
    var m = 0
    while m < _MAX_TERMS:
        m += 1
        var c: Integer
        if kind == 0:
            c = Integer(2 * m - 1)
        elif kind == 1:
            c = Integer(m)
        elif kind == 2:
            c = Integer(2 * m - 1) * Integer(2 * m)
        elif kind == 3:
            c = Integer(2 * m) * Integer(2 * m + 1)
        elif kind == 4:
            c = Integer(4 * m - 3) * Integer(4 * m - 1)
        else:
            c = Integer(4 * m - 1) * Integer(4 * m + 1)
        var next = _negate(_div(_mul_n(term, c, w), z, w))
        if _magnitude(next).compare(_magnitude(term)) >= 0:
            return None
        if _settled(next, total, w):
            return _widen(total, next)
        term = next
        total = _add(total, term, w)
    return None


# ----------------------------------------------------- error functions


def _erf_sum(y: Ball, w: Int) raises -> Ball:
    """`sum_n 2**n y**(2n+1) / (2n+1)!!` for y >= 0, all terms positive. Once
    `4 y**2 <= 2n + 3` every later term is at most half the one before."""
    var double_square = _scaled(_mul(y, y, w), 1, w)
    var limit = double_square._exact_upper()
    var term = y
    var total = y
    var n = 0
    while n < _MAX_TERMS:
        n += 1
        term = _div_n(_mul(term, double_square, w), Integer(2 * n + 1), w)
        total = _add(total, term, w)
        if limit * Float(2) <= Float(2 * n + 3) and _settled(term, total, w):
            return _widen(total, term)
    return Ball.indeterminate(w)


def _erf_of(y: Ball, w: Int) raises -> Ball:
    """erf of a ball y >= 0: `(2/sqrt pi) e**(-y**2)` times the positive sum."""
    var factor = _div(_scaled(_fn(_EXP, _negate(_mul(y, y, w)), w), 1, w), _sqrt_pi(w), w)
    return _mul(factor, _erf_sum(y, w), w)


def _kernel_erf(x: Float, w: Int) raises -> Ball:
    """erf at a finite nonzero Float. For `|x| >= 1`, `0 < 1 - erf|x| <
    e**(-x**2)` (DLMF 7.8.2), so once that is below the precision the result
    is 1 with that radius."""
    var work = w + 16
    var a = abs(x)
    var bits = _floor_scaled(x, 14426, 10000, True)
    var result: Ball
    if a >= Float(1) and bits >= work + 8:
        result = Ball(_midpoint=Float(1), _radius=_Radius.power_of_two(-bits), _kind=_FINITE)
    else:
        result = _erf_of(Ball(a), work)
    return _negate(result) if x._negative else result^


def _kernel_erfc(x: Float, w: Int) raises -> Ball:
    """erfc at a finite nonzero Float: `1 + erf|x|` for x < 0; for x > 0 the
    asymptotic series once it reaches the precision, otherwise `1 - erf x`
    with the bits the cancellation loses, at most `x**2 log2(e) + log2(2x+2)`
    since `erfc x > (2/sqrt pi) e**(-x**2) / (x + sqrt(x**2 + 2))`."""
    var work = w + 16
    if x._negative:
        return _add(_one(work), _kernel_erf(abs(x), work), work)
    if x._exponent > 29:
        # erfc x < e**(-x**2) < 2**(-2**58).
        return Ball(_midpoint=Float(0), _radius=_Radius.power_of_two(-(Int(1) << 58)), _kind=_FINITE)
    var y = Ball(x)
    var bits = _floor_scaled(x, 14426, 10000, True)
    if bits >= work + 16:
        var sum = _enveloping_sum(_scaled(_mul(y, y, work), 1, work), work, 0)
        if sum:
            var factor = _div(_fn(_EXP, _negate(_mul(y, y, work)), work), _mul(y, _sqrt_pi(work), work), work)
            return _mul(factor, sum.value(), work)
    var inner = work + bits + max(x._exponent, 0) + 10
    return _sub(_one(inner), _erf_of(y, inner), inner)


def _erfi_sum(y: Ball, w: Int) raises -> Ball:
    """`sum_n y**(2n+1) / (n! (2n+1))`, positive; once `2 y**2 <= n + 1` every
    later term is at most half the one before."""
    var square = _mul(y, y, w)
    var limit = square._exact_upper()
    var power = y
    var total = y
    var n = 0
    while n < _MAX_TERMS:
        n += 1
        power = _div_n(_mul(power, square, w), Integer(n), w)
        var term = _div_n(power, Integer(2 * n + 1), w)
        total = _add(total, term, w)
        if limit * Float(2) <= Float(n + 1) and _settled(term, total, w):
            return _widen(total, term)
    return Ball.indeterminate(w)


def _dawson(y: Ball, a: Float, w: Int) raises -> Optional[Ball]:
    """Dawson's integral at a large y > 0: `sum_{k<n} T_k`, `T_k =
    (2k-1)!! / (2**(k+1) y**(2k+1))`, widened by `n e**(-y**2)/y + 3
    sqrt(n) T_n + y e**(-y**2/2)`, for `n <= y**2/2`. None when the terms are
    not negligible by then."""
    var limit = _floor_scaled(a, 1, 2, True)
    var double_square = _scaled(_mul(y, y, w), 1, w)
    var term = _div(_one(w), _scaled(y, 1, w), w)
    var total = Ball(Integer(0), precision=w)
    var n = 0
    while True:
        var weight = Integer(3) * (isqrt(Integer(n)) + 1)
        if n >= 1 and _settled(_mul_n(term, weight, w), total, w + 2):
            var low = _div(_mul_n(_fn(_EXP, _negate(_mul(y, y, w)), w), Integer(n), w), y, w)
            var middle = _mul(y, _fn(_EXP, _negate(_scaled(_mul(y, y, w), -1, w)), w), w)
            return _widen(_widen(_widen(total, low), _mul_n(term, weight, w)), middle)
        if n >= limit or n >= _MAX_TERMS:
            return None
        total = _add(total, term, w)
        n += 1
        term = _div(_mul_n(term, Integer(2 * n - 1), w), double_square, w)


def _kernel_erfi(x: Float, w: Int) raises -> Ball:
    """erfi at a finite nonzero Float: the positive series, or for large |x|
    `(2/sqrt pi) e**(x**2) D(|x|)` with Dawson's asymptotic series."""
    var work = w + 16
    var a = abs(x)
    if a._exponent > 29:
        return Ball.unbounded(work)
    var y = Ball(a)
    var result = Ball.indeterminate(work)
    var done = False
    if _floor_scaled(a, 7213, 10000, True) >= work + 2 * max(a._exponent, 0) + 16:
        var d = _dawson(y, a, work)
        if d:
            result = _mul(_div(_scaled(_fn(_EXP, _mul(y, y, work), work), 1, work), _sqrt_pi(work), work), d.value(), work)
            done = True
    if not done:
        result = _div(_scaled(_erfi_sum(y, work), 1, work), _sqrt_pi(work), work)
    return _negate(result) if x._negative else result^


# --------------------------------------------------- the integrals


def _series(x: Float, w: Int, step: Int, first: Int, alternating: Bool) raises -> Ball:
    """`sum_m s_m x**m / (m m!)` over `m = first, first + step, ...`: Ei's
    series with step 1 (x carries its sign), and with step 2 the odd terms
    (Si, Shi) or the even ones from 2 (Ci, Chi), alternating for Si and Ci.
    Once `|x|**step <= (m+1)...(m+step) / 2`, every later term is at most half
    the one before, so the tail is below the last term."""
    var y = Ball(x)
    var f = y if step == 1 else _mul(y, y, w)
    if alternating:
        f = _negate(f)
    var size = _magnitude(f).to_float()
    var power = y if first == 1 else _scaled(f, -1, w)
    var m = first
    var total = _div_n(power, Integer(m), w)
    var count = 0
    while count < _MAX_TERMS:
        count += 1
        power = _mul(power, f, w)
        for j in range(1, step + 1):
            power = _div_n(power, Integer(m + j), w)
        m += step
        var term = _div_n(power, Integer(m), w)
        total = _add(total, term, w)
        var next = Integer(m + 1) if step == 1 else Integer(m + 1) * Integer(m + 2)
        if size * Float(2) <= Float(next) and _settled(term, total, w):
            return _widen(total, term)
    return Ball.indeterminate(w)


def _log_part(x: Float, w: Int) raises -> Ball:
    """`gamma + log|x|`."""
    return _add(_euler_gamma(w), _fn(_LOG, Ball(abs(x)), w), w)


def _ei_asymptotic(y: Ball, a: Float, w: Int) raises -> Optional[Ball]:
    """Ei at x >= 16: `e**x (sum_{k<n} k!/x**(k+1) + d)`, `|d| <= 2 n!/x**(n+1)
    + 8 e**(-x/2)/x` for `n <= x/4`."""
    var quarter = _floor_scaled(a, 1, 4, False)
    var term = _div(_one(w), y, w)
    var total = Ball(Integer(0), precision=w)
    var n = 0
    while True:
        if n >= 1 and _settled(_scaled(term, 1, w), total, w + 2):
            break
        if n >= quarter or n >= _MAX_TERMS:
            return None
        total = _add(total, term, w)
        n += 1
        term = _div(_mul_n(term, Integer(n), w), y, w)
    var tail = _div(_mul_n(_fn(_EXP, _negate(_scaled(y, -1, w)), w), Integer(8), w), y, w)
    return _mul(_fn(_EXP, y, w), _widen(_widen(total, _scaled(term, 1, w)), tail), w)


def _kernel_ei(x: Float, w: Int) raises -> Ball:
    """Ei at a finite nonzero Float, the principal value for x < 0. For x < 0
    the series alternates and loses about `1.443 |x|` bits, so a large `|x|`
    takes `-E1(|x|)` by its asymptotic series instead."""
    var work = w + 16
    var a = abs(x)
    var bits = _floor_scaled(x, 14426, 10000, False)
    if x._negative:
        if bits >= work + 16:
            var y = Ball(a)
            var sum = _enveloping_sum(y, work, 1)
            if sum:
                return _negate(_mul(_div(_fn(_EXP, _negate(y), work), y, work), sum.value(), work))
        var inner = work + bits + 16
        return _add(_log_part(x, inner), _series(x, inner, 1, 1, False), inner)
    if a._exponent > 58:
        return Ball.unbounded(work)
    if bits >= 2 * work + 32:
        var r = _ei_asymptotic(Ball(a), a, work)
        if r:
            return r.value()
    return _add(_log_part(x, work), _series(x, work, 1, 1, False), work)


def _trig_auxiliary(y: Ball, w: Int) raises -> Optional[Tuple[Ball, Ball]]:
    """f and g of the sine and cosine integrals at a large y > 0."""
    var square = _mul(y, y, w)
    var f = _enveloping_sum(square, w, 2)
    var g = _enveloping_sum(square, w, 3)
    if not f or not g:
        return None
    return (_div(f.value(), y, w), _div(g.value(), square, w))


def _kernel_si(x: Float, w: Int, budget: Int) raises -> Ball:
    """Si at a finite nonzero Float."""
    var work = w + 16
    var a = abs(x)
    var bits = _floor_scaled(x, 14426, 10000, False)
    var result = Ball.indeterminate(work)
    var done = False
    if bits >= work + 16:
        var fg = _trig_auxiliary(Ball(a), work)
        if fg:
            var sc = _kernel_sin_cos(a, work, budget)
            if sc[0].is_indeterminate():
                return sc[0]
            var half_pi = _scaled(_pi(work), -1, work)
            result = _sub(_sub(half_pi, _mul(fg.value()[0], sc[1], work), work), _mul(fg.value()[1], sc[0], work), work)
            done = True
    if not done:
        result = _series(a, work + bits + 16, 2, 1, True)
    return _negate(result) if x._negative else result^


def _kernel_ci(x: Float, w: Int, budget: Int) raises -> Ball:
    """Ci at a finite Float x > 0."""
    var work = w + 16
    var bits = _floor_scaled(x, 14426, 10000, False)
    if bits >= work + 16:
        var fg = _trig_auxiliary(Ball(x), work)
        if fg:
            var sc = _kernel_sin_cos(x, work, budget)
            if sc[0].is_indeterminate():
                return sc[0]
            return _sub(_mul(fg.value()[0], sc[0], work), _mul(fg.value()[1], sc[1], work), work)
    var inner = work + bits + 16
    return _add(_log_part(x, inner), _series(x, inner, 2, 2, True), inner)


def _kernel_shi(x: Float, w: Int) raises -> Ball:
    """Shi at a finite nonzero Float: positive terms, or `(Ei + E1) / 2` for
    a large `|x|`, with `0 < E1(|x|) < e**(-|x|)/|x|`."""
    var work = w + 16
    var a = abs(x)
    if a._exponent > 58:
        return Ball.unbounded(work)
    var result = Ball.indeterminate(work)
    var done = False
    if _floor_scaled(a, 14426, 10000, False) >= 2 * work + 32:
        var ei = _ei_asymptotic(Ball(a), a, work)
        if ei:
            var e1 = _div(_fn(_EXP, _negate(Ball(a)), work), Ball(a), work)
            result = _widen(_scaled(ei.value(), -1, work), e1)
            done = True
    if not done:
        result = _series(a, work, 2, 1, False)
    return _negate(result) if x._negative else result^


def _kernel_chi(x: Float, w: Int) raises -> Ball:
    """Chi at a finite Float x > 0: `gamma + log x` and positive terms, or
    `(Ei - E1) / 2` for a large x."""
    var work = w + 16
    if x._exponent > 58:
        return Ball.unbounded(work)
    if _floor_scaled(x, 14426, 10000, False) >= 2 * work + 32:
        var ei = _ei_asymptotic(Ball(x), x, work)
        if ei:
            var e1 = _div(_fn(_EXP, _negate(Ball(x)), work), Ball(x), work)
            return _widen(_scaled(ei.value(), -1, work), e1)
    return _add(_log_part(x, work), _series(x, work, 2, 2, False), work)


# ---------------------------------------------------------- Fresnel


def _half_square_mod_two(a: Float) raises -> Float:
    """`(a**2 / 2) mod 2`, exactly: 0 when every bit of a weighs at least 2."""
    var significand = a._significand
    var zeros = 0
    while not (significand & 1):
        significand >>= 1
        zeros += 1
    var lowest = a._exponent - a.precision() + zeros
    if 2 * lowest - 1 >= 1:
        return Float(0)
    var s = a.to_rational_exact()
    s = s * s / Rational(2)
    var r = s - Rational(Integer(2) * (s / Rational(2)).floor())
    return Float(r, context=_exact_context())


def _sin_cos_pi(r: Float, w: Int, budget: Int) raises -> Tuple[Ball, Ball]:
    """`(sin pi r, cos pi r)` for an exact r in [0, 2)."""
    if r.is_zero():
        return (Ball(Integer(0), precision=w), _one(w))
    return _sin_cos_ball(_mul(_pi(w + 4), Ball(r), w), w, budget)


def _fresnel_series(a: Float, w: Int, sine: Bool) raises -> Ball:
    """`x sum_n (-1)**n u**k / (k! (2k+1))` over `k = 2n + 1` (S) or `k = 2n`
    (C), `u = pi x**2 / 2`; once `u**2 <= (k+1)(k+2) / 2` every later term is
    at most half the one before."""
    var y = Ball(a)
    var u = _scaled(_mul(_pi(w), _mul(y, y, w), w), -1, w)
    var f = _negate(_mul(u, u, w))
    var size = _magnitude(f).to_float()
    var k = 1 if sine else 0
    var power = u if sine else _one(w)
    var total = _div_n(power, Integer(2 * k + 1), w)
    var count = 0
    while count < _MAX_TERMS:
        count += 1
        power = _div_n(_div_n(_mul(power, f, w), Integer(k + 1), w), Integer(k + 2), w)
        k += 2
        var term = _div_n(power, Integer(2 * k + 1), w)
        total = _add(total, term, w)
        if size * Float(2) <= Float(Integer(k + 1) * Integer(k + 2)) and _settled(term, total, w):
            return _mul(y, _widen(total, term), w)
    return Ball.indeterminate(w)


def _fresnel_auxiliary(a: Float, w: Int) raises -> Optional[Tuple[Ball, Ball]]:
    """Fresnel's f and g at a large a > 0: `f ~ (1/(pi a)) sum (-1)**m
    (4m-1)!!/(pi a**2)**(2m)`, `g ~ (1/(pi**2 a**3)) sum (-1)**m
    (4m+1)!!/(pi a**2)**(2m)`."""
    var y = Ball(a)
    var pi = _pi(w)
    var v = _mul(pi, _mul(y, y, w), w)
    var z = _mul(v, v, w)
    var sf = _enveloping_sum(z, w, 4)
    var sg = _enveloping_sum(z, w, 5)
    if not sf or not sg:
        return None
    var f = _div(sf.value(), _mul(pi, y, w), w)
    var g = _div(sg.value(), _mul(_mul(pi, v, w), y, w), w)
    return (f, g)


def _kernel_fresnel(x: Float, w: Int, sine: Bool, budget: Int) raises -> Ball:
    """Fresnel's S or C at a finite nonzero Float."""
    var work = w + 16
    var a = abs(x)
    var bits = _floor_scaled(a, 22661, 10000, True)
    var result = Ball.indeterminate(work)
    var done = False
    if bits >= work + 16:
        var fg = _fresnel_auxiliary(a, work)
        if fg:
            var sc = _sin_cos_pi(_half_square_mod_two(a), work, budget)
            var f = fg.value()[0]
            var g = fg.value()[1]
            var half = Ball(Rational(1, 2), precision=work)
            if sine:
                result = _sub(_sub(half, _mul(f, sc[1], work), work), _mul(g, sc[0], work), work)
            else:
                result = _sub(_add(half, _mul(f, sc[0], work), work), _mul(g, sc[1], work), work)
            done = True
    if not done:
        result = _fresnel_series(a, work + bits + 16, sine)
    return _negate(result) if x._negative else result^


# ------------------------------------------------- Gamma and digamma


def _tabulated_tangents(count: Int) raises -> List[Integer]:
    """`T_1 ... T_count` from the generated table, `count <= _TANGENT_COUNT`,
    each assembled from its limbs."""
    var t = List[Integer](capacity=count)
    var at = 0
    for k in range(count):
        var size = Int(_hex_limb(_TANGENT_SIZES, 16 * k))
        var words = List[UInt32](capacity=2 * size)
        for j in range(size):
            var limb = _hex_limb(_TANGENT_LIMBS, 16 * (at + size - 1 - j))
            words.append(UInt32(limb & 0xFFFFFFFF))
            words.append(UInt32(limb >> 32))
        t.append(Integer._from_words(words^, False))
        at += size
    return t^


struct _Tangents(Movable):
    """The tangent numbers in order, read from the table one at a time while
    it reaches and computed by the recurrence beyond, so that a series takes
    only the ones it uses."""

    var count: Int
    var next_k: Int
    var at: Int
    var computed: List[Integer]

    def __init__(out self, count: Int) raises:
        self.count = count
        self.next_k = 1
        self.at = 0
        self.computed = List[Integer]()
        if count > _TANGENT_COUNT:
            self.computed = _tangent_numbers(count)

    def next(mut self) raises -> Integer:
        """T_k for the next k."""
        var k = self.next_k
        self.next_k += 1
        if len(self.computed):
            return self.computed[k - 1]
        var size = Int(_hex_limb(_TANGENT_SIZES, 16 * (k - 1)))
        var words = List[UInt32](capacity=2 * size)
        for j in range(size):
            var limb = _hex_limb(_TANGENT_LIMBS, 16 * (self.at + size - 1 - j))
            words.append(UInt32(limb & 0xFFFFFFFF))
            words.append(UInt32(limb >> 32))
        self.at += size
        return Integer._from_words(words^, False)


def _tangent_numbers(count: Int) raises -> List[Integer]:
    """The tangent numbers `T_1 ... T_count`, `T_k` at index `k - 1`: from the
    table written by scripts/generate_bernoulli_table.py while it reaches,
    else by Brent and Harvey's recurrence (2011) in `O(count**2)` small
    multiplications. Recomputing them on every Stirling series was a third of
    `gammainc`'s time at 53 bits."""
    if count <= _TANGENT_COUNT:
        return _tabulated_tangents(count)
    var t = List[Integer](length=count, fill=Integer(0))
    if count == 0:
        return t^
    t[0] = Integer(1)
    for k in range(2, count + 1):
        t[k - 1] = t[k - 2] * Integer(k - 1)
    for k in range(2, count + 1):
        for j in range(k, count + 1):
            t[j - 1] = t[j - 2] * Integer(j - k) + t[j - 1] * Integer(j - k + 2)
    return t^


def _bernoulli(k: Int, tangent: Integer, w: Int) raises -> Ball:
    """`B_2k = (-1)**(k-1) 2k T_k / (4**k (4**k - 1))`."""
    var four = Integer(1) << (2 * k)
    var value = _div(Ball(Integer(2 * k) * tangent, precision=w), Ball(four * (four - 1), precision=w), w)
    return value if k % 2 == 1 else _negate(value)


def _stirling(z: Ball, w: Int, digamma: Bool) raises -> Ball:
    """log Gamma(z), or digamma(z), at an exact `z >= w/2 + 10` by Stirling's
    series; for a real z the remainder is below the first omitted term."""
    var ln_z = _fn(_LOG, z, w)
    var inverse = _div(_one(w), z, w)
    var inverse_square = _mul(inverse, inverse, w)
    var total: Ball
    var power: Ball
    if digamma:
        total = _sub(ln_z, _scaled(inverse, -1, w), w)
        power = inverse_square
    else:
        var half_log_two_pi = _scaled(_fn(_LOG, _scaled(_pi(w), 1, w), w), -1, w)
        var main = _sub(_mul(_sub(z, Ball(Rational(1, 2), precision=w), w), ln_z, w), z, w)
        total = _add(main, half_log_two_pi, w)
        power = inverse
    # The coefficients in integers: B_2k / (2k (2k-1)) is (-1)**(k-1) T_k /
    # (4**k (4**k - 1) (2k - 1)), and -B_2k / 2k is (-1)**k T_k / (4**k (4**k - 1)).
    var count = w // 8 + 16
    var tangents = _Tangents(count)
    for k in range(1, count + 1):
        var four = Integer(1) << (2 * k)
        var divisor = four * (four - 1)
        if not digamma:
            divisor = divisor * Integer(2 * k - 1)
        var term = _div_n(_mul_n(power, tangents.next(), w), divisor, w)
        if (k % 2 == 1) == digamma:
            term = _negate(term)
        if _settled(term, total, w):
            return _widen(total, term)
        total = _add(total, term, w)
        power = _mul(power, inverse_square, w)
    return Ball.indeterminate(w)


def _short_dyadic(x: Float) raises -> Optional[Tuple[Integer, Int]]:
    """`x = m / 2**d`, d >= 0, for a positive Float whose factors `m + k 2**d`
    stay within 128 bits for k < 2**20; None otherwise."""
    if x.is_zero() or x._negative:
        return None
    var odd = x._significand
    var zeros = 0
    while odd._low_magnitude() == 0:
        odd = odd >> 64
        zeros += 64
    var tail = Int(count_trailing_zeros(odd._low_magnitude()))
    odd = odd >> tail
    var q = x._exponent - x.precision() + zeros + tail
    if q >= 0:
        if odd.magnitude_bit_length() + q > 108:
            return None
        return (odd << q, 0)
    if odd.magnitude_bit_length() > 128 or -q > 108:
        return None
    return (odd^, -q)


def _factor_product(m: Integer, d: Int, lo: Int, hi: Int) raises -> Integer:
    """`prod_{lo <= k < hi} (m + k 2**d)`, by halves."""
    if hi - lo <= 16:
        var p = Integer(1)
        for k in range(lo, hi):
            p = p * (m + (Integer(k) << d))
        return p^
    var middle = (lo + hi) // 2
    return _factor_product(m, d, lo, middle) * _factor_product(m, d, middle, hi)


def _reciprocal_sum(m: Integer, d: Int, lo: Int, hi: Int) raises -> Tuple[Integer, Integer]:
    """`sum_{lo <= k < hi} 1 / (m + k 2**d)` as an unreduced fraction, by
    halves."""
    if hi - lo == 1:
        return (Integer(1), m + (Integer(lo) << d))
    var middle = (lo + hi) // 2
    var left = _reciprocal_sum(m, d, lo, middle)
    var right = _reciprocal_sum(m, d, middle, hi)
    return (left[0] * right[1] + right[0] * left[1], left[1] * right[1])


def _shift_product(x: Float, n: Int, w: Int) raises -> Ball:
    """`x (x+1) ... (x+n-1)` for x > 0 and n >= 1: exactly in integers for a
    short dyadic x, `prod (m + k 2**d) / 2**(d n)`, rounded once; otherwise
    in ball arithmetic."""
    var short = _short_dyadic(x)
    if short:
        var m = short.value()[0]
        var d = short.value()[1]
        return _scaled(Ball(_factor_product(m, d, 0, n), precision=w), -d * n, w)
    var product = Ball(x)
    for k in range(1, n):
        product = _mul(product, Ball(_exact_add(x, Integer(k))), w)
    return product^


def _shift_reciprocals(x: Float, n: Int, w: Int) raises -> Ball:
    """`sum_{k<n} 1/(x+k)` for x > 0 and n >= 1: exactly in integers for a
    short dyadic x, `2**d sum 1/(m + k 2**d)`, with one division; otherwise
    in ball arithmetic."""
    var short = _short_dyadic(x)
    if short:
        var m = short.value()[0]
        var d = short.value()[1]
        var fraction = _reciprocal_sum(m, d, 0, n)
        var quotient = _div(Ball(fraction[0], precision=w + 8), Ball(fraction[1], precision=w + 8), w)
        return _scaled(quotient, d, w)
    var total = Ball(Integer(0), precision=w)
    for k in range(n):
        total = _add(total, _div(_one(w), Ball(_exact_add(x, Integer(k))), w), w)
    return total^


def _shift(x: Float, w: Int) raises -> Int:
    """N with `x + N >= w/2 + 10`."""
    var target = Float(w // 2 + 10)
    if x >= target:
        return 0
    return Int(_exact_difference(target, x).floor()) + 1


def _log_gamma_positive(x: Float, w: Int) raises -> Ball:
    """log Gamma at x > 0: the logarithm of `_gamma_local` where that serves,
    of `Gamma(1 + x) / x` below 1/2; else Stirling at `x + N` minus
    `log(x (x+1) ... (x+N-1))`, with the bits that the magnitudes, about
    `z log z`, take."""
    if _below_half(x):
        var local = _gamma_local(_exact_add(x, Integer(1)), w + 8)
        if local:
            return _fn(_LOG, _div(local.value(), Ball(x), w + 8), w)
    else:
        var local = _gamma_local(x, w + 8)
        if local:
            return _fn(_LOG, local.value(), w)
    var n = _shift(x, w)
    var work = w + 2 * _bit_count(w) + 16
    var result = _stirling(Ball(_exact_add(x, Integer(n))), work, False)
    if n > 0:
        result = _sub(result, _fn(_LOG, _shift_product(x, n, work), work), work)
    return result^


def _below_half(x: Float) -> Bool:
    """`x < 1/2` for a finite Float, `|x|` being below `2**exponent` and at
    least half that."""
    return x._negative or x.is_zero() or x._exponent < 0


def _local_split(x: Float) raises -> Tuple[Int, Float]:
    """`x = 1 + u + m` with `|u| <= 1/2` and `m >= 0`, for `1/2 <= x < 2**20`:
    `1 + m` is x rounded to an integer, at least 1."""
    var n = max(Int(x.round()), 1)
    return (n - 1, _exact_add(x, Integer(-n)))


def _local_parts(x: Float, w: Int, derivative: Bool, guard: Int = 24) raises -> Optional[Tuple[Ball, Ball, Ball, Ball]]:
    """`(R, R', P, P')` at `x = 1 + u + m`: `R(u) = 1/Gamma(1 + u)` and its
    derivative (`_rgamma_series`), `P = (1+u) ... (m+u)` and `P' = dP/du`
    (`_shift_factors`), R' and P' only with derivative; then
    `Gamma(x) = P / R` and `psi(x) = P'/P - R'/R`. None where x is past the
    Taylor method's range or the fixed-point product does not serve."""
    if w > _RGAMMA_PRECISION or x._exponent > 20 or _below_half(x):
        return None
    var split = _local_split(x)
    var m = split[0]
    var u = split[1]
    if m + 2 > w // 2 + 10:
        return None
    var factors = _shift_factors(u, m, w, derivative, guard)
    if not factors:
        return None
    var series = _rgamma_series(u, w, derivative, guard)
    if not series:
        return None
    return (series.value()[0], series.value()[1], factors.value()[0], factors.value()[1])


def _gamma_digamma_local(x: Float, w: Int, guard: Int = 24) raises -> Optional[Tuple[Ball, Ball]]:
    """`(Gamma(x), psi(x))` from one pass of `_local_parts`."""
    var parts = _local_parts(x, w, True, guard)
    if not parts:
        return None
    ref v = parts.value()
    var gamma = _div(v[2], v[0], w)
    var psi = _sub(_div(v[3], v[2], w), _div(v[1], v[0], w), w)
    return (gamma^, psi^)


def _gamma_local(x: Float, w: Int) raises -> Optional[Ball]:
    """Gamma at `1/2 <= x`, rounded to an integer below `w/2 + 9`, from the
    Taylor series of the
    reciprocal (`_rgamma.mojo`): with `x = 1 + u + m`, `|u| <= 1/2`,
    `Gamma(x) = (1+u) (2+u) ... (m+u) / (1/Gamma(1 + u))`, the product exact
    in integers for a short x. Below Stirling's shift, the product has fewer
    factors than Stirling's and the series replaces its exponential and
    logarithm. None at a precision beyond the table."""
    var parts = _local_parts(x, w, False)
    if parts:
        return _div(parts.value()[2], parts.value()[0], w)
    if w > _RGAMMA_PRECISION or x._exponent > 20:
        return None
    var split = _local_split(x)
    var m = split[0]
    var u = split[1]
    if m + 2 > w // 2 + 10:
        return None
    var reciprocal = _rgamma_unit(u, w)
    if not reciprocal:
        return None
    if m == 0:
        return _div(_one(w), reciprocal.value(), w)
    return _div(_shift_product(_exact_add(u, Integer(1)), m, w), reciprocal.value(), w)


def _gamma_positive(x: Float, w: Int) raises -> Ball:
    """Gamma at x >= 1/2: by the reciprocal's Taylor series below Stirling's
    shift and the table's precision (`_gamma_local`), else
    `e**(log Gamma(x + N)) / (x (x+1) ... (x+N-1))`, the logarithm with the
    bits its magnitude, about `z log z`, takes; one division instead of the
    product's logarithm."""
    var local = _gamma_local(x, w)
    if local:
        return local.value()
    var n = _shift(x, w)
    var z = _exact_add(x, Integer(n))
    var magnitude = max(z._exponent, 0) + _bit_count(max(z._exponent, 1)) + 8
    var work = w + 2 * _bit_count(w) + 16
    var top = _fn(_EXP, _stirling(Ball(z), work + magnitude, False), work)
    if n == 0:
        return top^
    return _div(top, _shift_product(x, n, work), w)


def _sin_pi(x: Float, w: Int) raises -> Ball:
    """`sin(pi x)` with x reduced exactly: `(-1)**n sin(pi (x - n))`."""
    var n = x.round()
    var f = _exact_difference(x, Float(n, context=_exact_context()))
    var s = _fn(_SIN, _mul(_pi(w + 4), Ball(f), w), w)
    return _negate(s) if (n & 1).__bool__() else s^


def _kernel_gamma(x: Float, w: Int) raises -> Ball:
    """Gamma at a finite Float that is not 0 or a negative integer."""
    var work = w + 16
    if x._exponent > 56:
        if not x._negative:
            return Ball.unbounded(work)
        # |Gamma(x)| = pi / (|sin pi x| Gamma(1 - x)) < 2**(-2**58).
        return Ball(_midpoint=Float(0), _radius=_Radius.power_of_two(-(Int(1) << 58)), _kind=_FINITE)
    if _below_half(x):
        # Reflection: Gamma(x) = pi / (sin(pi x) Gamma(1 - x)).
        var reflected = _kernel_gamma(_exact_difference(Float(1), x), work)
        return _div(_pi(work), _mul(_sin_pi(x, work), reflected, work), work)
    return _gamma_positive(x, work)


def _kernel_log_gamma(x: Float, w: Int) raises -> Ball:
    """log |Gamma| at a finite Float other than 0, 1, 2 and the negative
    integers; below 0 by reflection, `log pi - log |sin pi x| -
    log Gamma(1 - x)`."""
    var work = w + 16
    if not x._negative:
        return _log_gamma_positive(x, work)
    var s = _sin_pi(x, work)
    if s.certainly_negative():
        s = _negate(s)
    var reflected = _log_gamma_positive(_exact_difference(Float(1), x), work)
    return _sub(_sub(_fn(_LOG, _pi(work), work), _fn(_LOG, s, work), work), reflected, work)


def _half_root(x: Float, negate: Bool, w: Int) raises -> Ball:
    """`x / sqrt 2`, or `-x / sqrt 2` with `negate`."""
    var root = _fn(_SQRT, Ball(Integer(2), precision=w + 8), w + 8)
    return _div(Ball(-x if negate else x), root, w)


def _kernel_ndtr(x: Float, w: Int) raises -> Ball:
    """The standard normal distribution function, `erfc(-x/sqrt 2) / 2`, at
    a finite Float. erfc's relative sensitivity to its argument t is about
    `2 t**2`, so the argument takes `2 exponent(x)` more bits."""
    var work = w + 2 * max(x._exponent, 0) + 16
    var e = _ball_special(_ERFC, _BallArgument(_half_root(x, True, work)), 0, _c(work))
    return _scaled(e, -1, w)


def _giles(x: Float64, w: Float64) -> Float64:
    """Giles's single-precision approximation of erfinv (M. Giles, "Approximating
    the erfinv function", GPU Computing Gems, 2010) from `w = -log((1-x)(1+x))`:
    a start for Newton's iteration, about 7 digits."""
    var p: Float64
    if w < 5.0:
        var v = w - 2.5
        p = 2.81022636e-08
        p = 3.43273939e-07 + p * v
        p = -3.5233877e-06 + p * v
        p = -4.39150654e-06 + p * v
        p = 0.00021858087 + p * v
        p = -0.00125372503 + p * v
        p = -0.00417768164 + p * v
        p = 0.246640727 + p * v
        p = 1.50140941 + p * v
    else:
        var v = sqrt(w) - 3.0
        p = -0.000200214257
        p = 0.000100950558 + p * v
        p = 0.00134934322 + p * v
        p = -0.00367342844 + p * v
        p = 0.00573950773 + p * v
        p = -0.0076224613 + p * v
        p = 0.00943887047 + p * v
        p = 1.00167406 + p * v
        p = 2.83297682 + p * v
    return p * x


def _erfc_seed(y: Float) raises -> Float:
    """A start for the t > 0 with `erfc(t) = y`, 0 < y <= 1: Giles's formula
    while `L = -log y <= 30`, else `t = sqrt(L - log(t sqrt pi))`, from
    `erfc(t) ~ e**(-t**2) / (t sqrt pi)`."""
    var big: Float64
    if y._exponent > -1000:
        var v = y.to_native[DType.float64]()
        big = -log(v)
        if big <= 30.0:
            return Float(_giles(1.0 - v, -log(v * (2.0 - v))))
    else:
        big = -_fn(_LOG, Ball(y), 64)._midpoint.to_native[DType.float64]()
    var t = sqrt(big)
    for _ in range(6):
        var inner = big - log(t * 1.7724538509055159)
        if inner > 0.0:
            t = sqrt(inner)
    return Float(t)


def _exact_add(a: _FloatArgument, b: _FloatArgument) raises -> Float:
    """`a + b`, exactly."""
    return Float(_rounded=_float_operation(a, b, 0, _exact_context()))


def _inverse_root(target: Ball, complement: Bool, seed: Float, w: Int, budget: Int) raises -> Ball:
    """The t with `erf(t) = target`, or `erfc(t) = target` with complement, as
    a ball of relative radius `2**-(w+8)`: Newton's iteration from seed,
    `t -+ (f(t) - target) (sqrt pi / 2) e**(t**2)`, each step at about twice
    the bits the last one showed right, then the root bracketed by the signs
    of `f - target` at the ball's ends."""
    var code = _ERFC if complement else _ERF
    var t = seed
    var final = w + 32
    # The bits of t known right: a step's size shows the last iterate's
    # error, and the next has about twice its bits.
    var good = 2
    for _ in range(256):
        var prec = min(2 * good + 32, final)
        var work = prec + 32
        var value = _special_point(code, t, 0, work, budget)
        var residual = _sub(value, target, work)
        var root_pi = _fn(_SQRT, _pi(work), work)
        var step = _mul(_mul(residual, _scaled(root_pi, -1, work), work), _fn(_EXP, _mul(Ball(t), Ball(t), work), work), work)
        var next = _add(Ball(t), step, work) if complement else _sub(Ball(t), step, work)
        if not next.is_finite():
            return Ball.indeterminate(w)
        t = _rounded_ball(_BallArgument(next), prec)._midpoint
        var moved = step._midpoint
        good = final if moved.is_zero() else min(final, max(2, 2 * (t._exponent - moved._exponent) - 4))
        if prec == final and good >= final - 8:
            break
    if t.is_zero():
        return Ball.indeterminate(w)
    var delta = abs(_scaled(Ball(t), -(w + 8), w)._midpoint)
    var low = _exact_difference(t, delta)
    var high = _exact_add(t, delta)
    var check = w + 40
    for _ in range(2):
        var below = _sub(_special_point(code, low, 0, check, budget), target, check)
        var above = _sub(_special_point(code, high, 0, check, budget), target, check)
        var bracketed = (below.certainly_positive() and above.certainly_negative()) if complement else (below.certainly_negative() and above.certainly_positive())
        if bracketed:
            return Ball(_midpoint=t, _radius=_Radius.upper(delta), _kind=_FINITE)
        check *= 2
    return Ball.indeterminate(w)


def _erfc_root(y: Float, w: Int, budget: Int) raises -> Ball:
    """The t > 0 with `erfc(t) = y`, for 0 < y < 1."""
    return _inverse_root(Ball(y), True, _erfc_seed(y), w, budget)


def _kernel_erfinv(x: Float, w: Int, budget: Int) raises -> Ball:
    """erfinv at a Float with `0 < |x| < 1`: on erf for `|x| <= 1/2`, on erfc
    at `1 - |x|`, exactly, in the tails."""
    var a = abs(x)
    if a <= Float(Rational(1, 2)):
        var v = x.to_native[DType.float64]()
        return _inverse_root(Ball(x), False, Float(_giles(v, -log((1.0 - v) * (1.0 + v)))), w, budget)
    var t = _erfc_root(_exact_difference(Float(1), a), w, budget)
    return _negate(t) if x._negative else t^


def _kernel_ndtri(p: Float, w: Int, budget: Int) raises -> Ball:
    """ndtri at a Float with `0 < p < 1` other than 1/2:
    `sqrt 2 erfinv(2p - 1)` near 1/2, `-+ sqrt 2 erfcinv(2 min(p, 1-p))` in
    the tails, each argument exact."""
    var work = w + 16
    var root_two = _fn(_SQRT, Ball(Integer(2), precision=work), work)
    var x = _exact_difference(_scaled(Ball(p), 1, work)._midpoint, Float(1))
    if abs(x) <= Float(Rational(1, 2)):
        return _mul(root_two, _kernel_erfinv(x, work, budget), w)
    var upper = not x._negative
    var tail = _exact_difference(Float(1), p) if upper else p
    var t = _erfc_root(_scaled(Ball(tail), 1, work)._midpoint, work, budget)
    var result = _mul(root_two, t, w)
    return result if upper else _negate(result)


def _kernel_log_ndtr(x: Float, w: Int) raises -> Ball:
    """log ndtr at a finite Float: `log(erfc(-x/sqrt 2) / 2)` at and below
    0 and `log1p(-erfc(x/sqrt 2) / 2)` above, each keeping its relative
    accuracy."""
    var work = w + 2 * max(x._exponent, 0) + 16
    if x._negative or x.is_zero():
        return _fn(_LOG, _kernel_ndtr(x, work), w)
    var q = _scaled(_ball_special(_ERFC, _BallArgument(_half_root(x, False, work)), 0, _c(work)), -1, work)
    return _fn(_LOG1P, _negate(q), w)


def _digamma_local(x: Float, w: Int) raises -> Optional[Ball]:
    """digamma at `1/2 <= x`, rounded to an integer below `w/2 + 9`, from the
    Taylor series of the
    reciprocal gamma function and its derivative (`_rgamma.mojo`): with
    `x = 1 + u + m`, `psi(1 + u) = -R'(u) / R(u)` and `psi(x) = psi(1 + u) +
    sum_{k<m} 1/(1 + u + k)`. None at a precision beyond the table."""
    var parts = _local_parts(x, w, True)
    if parts:
        ref v = parts.value()
        return _sub(_div(v[3], v[2], w), _div(v[1], v[0], w), w)
    if w > _RGAMMA_PRECISION or x._exponent > 20:
        return None
    var split = _local_split(x)
    var m = split[0]
    var u = split[1]
    if m + 2 > w // 2 + 10:
        return None
    var pair = _rgamma_series(u, w, True)
    if not pair:
        return None
    var result = _negate(_div(pair.value()[1], pair.value()[0], w))
    if m == 0:
        return result^
    return _add(result, _shift_reciprocals(_exact_add(u, Integer(1)), m, w), w)


def _kernel_digamma(x: Float, w: Int) raises -> Ball:
    """digamma at a finite Float that is not 0 or a negative integer."""
    var work = w + 16
    if _below_half(x):
        # Reflection: psi(x) = psi(1 - x) - pi cot(pi x), x reduced exactly.
        var n = x.round()
        var f = _exact_difference(x, Float(n, context=_exact_context()))
        var angle = _mul(_pi(work + 4), Ball(f), work)
        var cot = _div(_fn(_COS, angle, work), _fn(_SIN, angle, work), work)
        return _sub(_kernel_digamma(_exact_difference(Float(1), x), work), _mul(_pi(work), cot, work), work)
    var local = _digamma_local(x, work)
    if local:
        return local.value()
    var n = _shift(x, work)
    var inner = work + _bit_count(max(n, 1)) + 8
    var result = _stirling(Ball(_exact_add(x, Integer(n))), inner, True)
    if n > 0:
        result = _sub(result, _shift_reciprocals(x, n, inner), inner)
    return result^


# ------------------------------------------------------- Lambert's W


def _lambert_residual(t: Float, x: Float, w: Int) raises -> Ball:
    """`t e**t - x`."""
    return _sub(_mul(Ball(t), _fn(_EXP, Ball(t), w), w), Ball(x), w)


def _halley(start: Float, x: Float, precision: Int) raises -> Float:
    """Halley's iteration for `t e**t = x` at a fixed precision, from `start`:
    `t -= f / (e**t (t + 1) - (t + 2) f / (2t + 2))`, `f = t e**t - x`."""
    var context = ArithmeticContext(format=FloatFormat(precision))
    var t = Float(start, context=context)
    for _ in range(64):
        var et = _fn(_EXP, Ball(t), precision + 8)._midpoint
        var f = Float(t * et - x, context=context)
        var t1 = Float(t + Float(1), context=context)
        if f.is_zero() or t1.is_zero():
            return t
        var step = Float(f / (et * t1 - (t + Float(2)) * f / (Float(2) * t1)), context=context)
        t = Float(t - step, context=context)
        if step.is_zero() or step._exponent < t._exponent - precision + 2:
            return t
    return t


def _lambert_start(x: Float, branch: Int) raises -> Float:
    """A start for Halley's iteration: `-1 +- sqrt(2 (1 + e x))` near the
    branch point, `log x - log log x` for a large x on W_0 and
    `log(-x) - log(-log(-x))` on W_-1, else `log(1 + x)`."""
    var c = ArithmeticContext(format=FloatFormat(64))
    var e = _fn(_EXP, _one(64), 64)._midpoint
    var near = Float(Float(1) + e * x, context=c)
    if near < Float(Rational(1, 2)):
        var root = Float(0)
        if near > Float(0):
            root = _fn(_SQRT, Ball(Float(Float(2) * near, context=c)), 64)._midpoint
        return Float(Float(-1) + (root if branch == 0 else -root), context=c)
    if branch == 0:
        if x > Float(3):
            var l1 = _fn(_LOG, Ball(x), 64)._midpoint
            return Float(l1 - _fn(_LOG, Ball(l1), 64)._midpoint, context=c)
        return _fn(_LOG, Ball(Float(Float(1) + x, context=c)), 64)._midpoint
    var l1 = _fn(_LOG, Ball(-x), 64)._midpoint
    return Float(l1 - _fn(_LOG, Ball(-l1), 64)._midpoint, context=c)


def _kernel_lambert_w(x: Float, w: Int, branch: Int) raises -> Ball:
    """W_0 (branch 0) or W_-1 (branch -1) at a nonzero Float of its domain,
    above -1/e: Halley's iteration, then the interval `t (1 -+ 2**-work)` if
    `t e**t - x` has opposite signs at its ends. The residual increases in t
    on W_0's range `t > -1` and decreases on W_-1's `t < -1`, so the root lies
    between. Near the branch point the iterate needs more precision, so each
    failed bracket retries with `work` more bits, three times; then the
    result is indeterminate."""
    var work = w + 16
    var t = _halley(_lambert_start(x, branch), x, 64)
    for attempt in range(4):
        var precision = work + 16 + attempt * work
        t = _halley(t, x, precision)
        var delta = Float(_rounded=_scale_float(abs(t), Integer(-work), _exact_context()))
        var low = _exact_difference(t, delta)
        var high = Float(_rounded=_float_operation(t, delta, 0, _exact_context()))
        var r_low = _lambert_residual(low, x, precision + 16)
        var r_high = _lambert_residual(high, x, precision + 16)
        if branch == 0 and r_low.certainly_negative() and r_high.certainly_positive():
            return _hull(low, high, w + 8)
        if branch != 0 and r_low.certainly_positive() and r_high.certainly_negative():
            return _hull(low, high, w + 8)
    return Ball.indeterminate(w + 8)


# ------------------------------------------------------------ dispatch


def _special_kernel(code: Int, x: Float, n: Int, w: Int, budget: Int) raises -> Ball:
    """The kernel of a special function at a finite Float that is not one of
    its exact points."""
    if code == _GAMMA:
        return _kernel_gamma(x, w)
    if code == _LOG_GAMMA:
        return _kernel_log_gamma(x, w)
    if code == _DIGAMMA:
        return _kernel_digamma(x, w)
    if code == _ERF:
        return _kernel_erf(x, w)
    if code == _ERFC:
        return _kernel_erfc(x, w)
    if code == _ERFI:
        return _kernel_erfi(x, w)
    if code == _EI:
        return _kernel_ei(x, w)
    if code == _SI:
        return _kernel_si(x, w, budget)
    if code == _CI:
        return _kernel_ci(x, w, budget)
    if code == _SHI:
        return _kernel_shi(x, w)
    if code == _CHI:
        return _kernel_chi(x, w)
    if code == _FRESNEL_S:
        return _kernel_fresnel(x, w, True, budget)
    if code == _FRESNEL_C:
        return _kernel_fresnel(x, w, False, budget)
    if code == _NDTR:
        return _kernel_ndtr(x, w)
    if code == _LOG_NDTR:
        return _kernel_log_ndtr(x, w)
    if code == _ERFINV:
        return _kernel_erfinv(x, w, budget)
    if code == _NDTRI:
        return _kernel_ndtri(x, w, budget)
    return _kernel_lambert_w(x, w, n)


def _special_point(code: Int, t: Float, n: Int, w: Int, budget: Int) raises -> Ball:
    """A special function at an exact point of its domain: the exact values
    exactly (zeros, `erfc(0) = 1`, `Gamma(k) = (k-1)!`, `gammaln(1) =
    gammaln(2) = 0`), poles and points outside the domain indeterminate,
    otherwise its kernel."""
    if t.is_zero():
        if code == _ERFC:
            return Ball(Integer(1), precision=w)
        if code == _NDTR:
            return Ball(Rational(1, 2), precision=w)
        if code == _LOG_NDTR:
            return _kernel_log_ndtr(t, w)
        if code == _NDTRI:
            return Ball.indeterminate(w + 8)
        if code == _GAMMA or code == _LOG_GAMMA or code == _DIGAMMA or code == _EI or code == _CI or code == _CHI:
            return Ball.indeterminate(w + 8)
        if code == _LAMBERT_W and n != 0:
            return Ball.indeterminate(w + 8)
        return Ball(Integer(0), precision=w)
    var whole = _integral(t)
    if code == _GAMMA or code == _DIGAMMA or code == _LOG_GAMMA:
        if whole and t._negative:
            return Ball.indeterminate(w + 8)
        if whole and (code == _LOG_GAMMA) and (whole.value() == 1 or whole.value() == 2):
            return Ball(Integer(0), precision=w)
        if whole and code == _GAMMA and whole.value() <= 4096:
            return Ball(factorial(whole.value() - 1), precision=w + 8)
    if (code == _CI or code == _CHI) and t._negative:
        return Ball.indeterminate(w + 8)
    if code == _ERFINV and not (abs(t) < Float(1)):
        return Ball.indeterminate(w + 8)
    if code == _NDTRI:
        if t._negative or not (t < Float(1)):
            return Ball.indeterminate(w + 8)
        if t == Float(Rational(1, 2)):
            return Ball(Integer(0), precision=w)
    if code == _LAMBERT_W:
        if n != 0 and not t._negative:
            return Ball.indeterminate(w + 8)
        # Outside the domain below -1/e: 1 + e t < 0.
        var e = _fn(_EXP, _one(w + 8), w + 8)
        if _add(_one(w + 8), _mul(e, Ball(t), w + 8), w + 8).certainly_negative():
            return Ball.indeterminate(w + 8)
        if not _add(_one(w + 8), _mul(e, Ball(t), w + 8), w + 8).certainly_positive():
            return Ball.indeterminate(w + 8)
    return _special_kernel(code, t, n, w, budget)


@fieldwise_init
struct _SpecialKernel(_Enclosure):
    """A special function at an exact Float, for the Ziv driver."""

    var code: Int
    var x: Float
    var n: Int
    var budget: Int

    def enclosure(self, precision: Int) raises -> Ball:
        return _special_kernel(self.code, self.x, self.n, precision, self.budget)

    def describe(self) raises -> String:
        return String(_special_name(self.code), " at ", self.x)


# ------------------------------------------------------- ball functions


def _increasing(code: Int) -> Bool:
    return code == _ERF or code == _ERFI or code == _SHI or code == _CHI or code == _DIGAMMA or code == _NDTR or code == _LOG_NDTR or code == _ERFINV or code == _NDTRI


def _special_derivative(code: Int, x: Ball, w: Int, budget: Int) raises -> Ball:
    """A bound of `|f'|` over the ball, for the functions that are not
    monotone: 1 for Si, S and C; `1/lower` for Ci; `max |Gamma| max |psi|`
    and `max |psi|` at the ends for Gamma and log Gamma, since on an interval
    without a pole `|Gamma|` is log-convex and psi increases."""
    if code == _SI or code == _FRESNEL_S or code == _FRESNEL_C:
        return _one(w)
    if code == _CI:
        return _div(_one(w), Ball(x._exact_lower()), w)
    var low = x._exact_lower()
    var high = x._exact_upper()
    var psi_low = _special_point(_DIGAMMA, low, 0, w, budget)
    var psi_high = _special_point(_DIGAMMA, high, 0, w, budget)
    var psi = Ball(_midpoint=Float(0), _radius=_magnitude(psi_low).add(_magnitude(psi_high)), _kind=_FINITE)
    if code == _LOG_GAMMA:
        return psi
    var g_low = _special_point(_GAMMA, low, 0, w, budget)
    var g_high = _special_point(_GAMMA, high, 0, w, budget)
    var g = Ball(_midpoint=Float(0), _radius=_magnitude(g_low).add(_magnitude(g_high)), _kind=_FINITE)
    return _mul(g, psi, w)


def _gamma_spread(code: Int, x: Ball, w: Int, budget: Int) raises -> Optional[Ball]:
    """Gamma, log |Gamma| or digamma of a narrow ball without a pole: the
    value at the midpoint m, widened over the radius r. With `M = max |psi|`
    over the ball, psi being log |Gamma|'s derivative, `|log |Gamma(t)| -
    log |Gamma(m)|| <= r M`; then `|Gamma(t) - Gamma(m)| <= r M |Gamma(m)|
    e**(r M)`, and `e**(r M) < 65/64` for `r M < 2**-8`. Above 0, with
    `low <= m - r` from the radius arithmetic, `0 < psi'(t) < 1/t + 1/t**2`
    bounds digamma's change and `M <= |psi(m)| + r (1/low + 1/low**2)`; the
    bounds take radius operations, and Gamma and psi at m one pass of the
    Taylor method (`_gamma_digamma_local`), 8 bits past w, or a 16-bit
    digamma beyond its range. Below 0, psi increases between the poles, so
    M is at an end, both evaluated at 32 bits. None for a ball wide enough
    that the ends are tighter (`r M >= 2**-8`), and for digamma below 0."""
    var r = x._radius
    ref m = x._midpoint
    var low = _Radius.zero() if m._negative or m.is_zero() else _Radius.lower_gap_float(m, r)
    if not low.is_zero():
        var inverse = _Radius.power_of_two(0).divide(low)
        var slope = inverse.add(inverse.multiply(inverse))
        if code == _DIGAMMA:
            var word = _gamma_word(m, w, True)
            if word:
                ref g = word.value()
                var rounded = _wide_rounding(UInt256(g.psi), w, Int128(-64), RoundingMode.nearest_even, g.psi_negative)
                if g.psi == 0:
                    return Ball(_midpoint=Float(_rounded=_RoundedBinary(0, False, Integer(0), 0, FloatFormat(w), NumericStatus())), _radius=_double_up(g.psi_error).add(slope.multiply(r)), _kind=_FINITE)
                var exponent = Int(rounded.exponent)
                var record = _RoundedBinary(1, g.psi_negative, Integer(UInt64(rounded.magnitude)), exponent, FloatFormat(w), NumericStatus())
                var radius = _double_up(g.psi_error).add(slope.multiply(r))
                if rounded.inexact:
                    radius = radius.add(_Radius.power_of_two(exponent - w - 1))
                return Ball(_midpoint=Float(_rounded=record^), _radius=radius, _kind=_FINITE)
            var parts = _local_parts(m, w + 8, True, 0)
            var center: Ball
            if parts:
                ref v = parts.value()
                center = _sub(_div(v[3], v[2], w + 8), _div(v[1], v[0], w + 8), w + 8)
            else:
                center = _special_point(_DIGAMMA, m, 0, w, budget)
            if not center.is_finite():
                return None
            return _widen_by(center, slope.multiply(r))
        var word = _gamma_word(m, w)
        if word:
            return _gamma_word_ball(code, word.value(), slope, r, w)
        # Past one word, psi only bounds the radius: the one-word pass at m
        # rounded to 53 bits, its slope term over r and the rounding's
        # distance; Gamma from the value-only pass.
        var near = Float(_rounded=_float_operation(m, Integer(0), 0, ArithmeticContext(format=FloatFormat(53))))
        var near_word = _gamma_word(near, 53)
        var distance = _Radius.upper(_exact_difference(m, near))
        var psi_bound: _Radius
        if near_word:
            psi_bound = _double_up(near_word.value().psi_bound).add(slope.multiply(distance))
        else:
            var psi = _special_point(_DIGAMMA, m, 0, 16, budget)
            if not psi.is_finite():
                return None
            psi_bound = _magnitude(psi)
        var change = psi_bound.add(slope.multiply(r)).multiply(r)
        if change.infinite or (not change.is_zero() and change.exponent > -8):
            return None
        var parts = _local_parts(m, w + 8, False, 0)
        var center: Ball
        if parts:
            var gamma = _div(parts.value()[2], parts.value()[0], w + 8)
            center = gamma if code == _GAMMA else _fn(_LOG, gamma, w + 8)
        else:
            center = _special_point(code, m, 0, w, budget)
        if not center.is_finite():
            return None
        if code == _LOG_GAMMA:
            return _widen_by(center, change)
        return _widen_by(center, change.multiply(_magnitude(center)).multiply(_up_word(65, False, -6)))
    if code == _DIGAMMA:
        return None
    var low_end = x._exact_lower()
    var high_end = x._exact_upper()
    var spread = Ball(_midpoint=Float(0), _radius=r, _kind=_FINITE)
    var psi_low = _special_point(_DIGAMMA, low_end, 0, 32, budget)
    var psi_high = _special_point(_DIGAMMA, high_end, 0, 32, budget)
    if not psi_low.is_finite() or not psi_high.is_finite():
        return None
    var most = Ball(_magnitude(psi_low).max(_magnitude(psi_high)).to_float())
    var change = Ball(_mul(most, spread, 32)._exact_upper())
    if change._midpoint._negative or change._midpoint._exponent > -8:
        return None
    var center = _special_point(code, m, 0, w, budget)
    if not center.is_finite():
        return None
    if code == _LOG_GAMMA:
        return _widen(center, change)
    var growth = _scaled(Ball(Integer(65)), -6, 32)
    return _widen(center, _mul(_mul(change, Ball(_magnitude(center).to_float()), 32), growth, 32))


def _gamma_word_ball(code: Int, g: _WordGamma, slope: _Radius, r: _Radius, w: Int) raises -> Optional[Ball]:
    """`_gamma_spread`'s ball from `_gamma_word`: Gamma rounded once to w
    bits (to w + 8 for its logarithm), its radius the relative error and the
    rounding, widened by `r M |Gamma| 65/64` or `r M`."""
    var change = _double_up(g.psi_bound).add(slope.multiply(r)).multiply(r)
    if change.infinite or (not change.is_zero() and change.exponent > -8):
        return None
    var bits = w if code == _GAMMA else min(w + 8, 64)
    var rounded = _wide_rounding(UInt256(g.quotient), bits, Int128(g.exponent), RoundingMode.nearest_even, False)
    var exponent = Int(rounded.exponent)
    var record = _RoundedBinary(1, False, Integer(UInt64(rounded.magnitude)), exponent, FloatFormat(bits), NumericStatus())
    var magnitude = _up(g.quotient + 1, g.exponent)
    var radius = magnitude.multiply(_double_up(g.error))
    if rounded.inexact:
        radius = radius.add(_Radius.power_of_two(exponent - bits - 1))
    var center = Ball(_midpoint=Float(_rounded=record^), _radius=radius, _kind=_FINITE)
    if code == _LOG_GAMMA:
        return _widen_by(_fn(_LOG, center, bits), change)
    return _widen_by(center, change.multiply(_magnitude(center)).multiply(_up_word(65, False, -6)))


def _widen_by(a: Ball, extra: _Radius) -> Ball:
    """`a` with `extra` more radius."""
    return Ball(_midpoint=a._midpoint, _radius=a._radius.add(extra), _kind=a._kind)


def _gamma_minimum(code: Int, x: Ball, w: Int, budget: Int) raises -> Optional[Ball]:
    """Gamma or log Gamma of a positive ball by its monotone pieces: it
    decreases to its minimum at 1.4616321449... and increases after, where
    Gamma is 0.8856031944... and log Gamma -0.1214862905..."""
    var low = x._exact_lower()
    var high = x._exact_upper()
    if not (low > Float(0)):
        return None
    var left = Float(Rational(14616321449, 10000000000), context=ArithmeticContext(format=FloatFormat(64)))
    var right = Float(Rational(14616321450, 10000000000), context=ArithmeticContext(format=FloatFormat(64)))
    var a = _special_point(code, low, 0, w, budget)
    var b = _special_point(code, high, 0, w, budget)
    if a.is_indeterminate() or b.is_indeterminate():
        return Ball.indeterminate(w)
    if high <= left:
        return _hull(b._exact_lower(), a._exact_upper(), w)
    if low >= right:
        return _hull(a._exact_lower(), b._exact_upper(), w)
    var floor = Float(Rational(8856031944, 10000000000), context=ArithmeticContext(format=FloatFormat(64))) if code == _GAMMA else Float(Rational(-1214862906, 10000000000), context=ArithmeticContext(format=FloatFormat(64)))
    var top = a._exact_upper() if a._exact_upper() > b._exact_upper() else b._exact_upper()
    return _hull(floor, top, w)


def _special_unbounded(code: Int, w: Int) raises -> Ball:
    """A special function of the ball of every real number."""
    if code == _ERF:
        return _unit_ball(w)
    if code == _ERFC:
        return Ball(Integer(1), Integer(1), precision=w)
    if code == _SI:
        return Ball(Integer(0), Rational(1852, 1000), precision=w)
    if code == _FRESNEL_S or code == _FRESNEL_C:
        return Ball(Integer(0), Rational(78, 100), precision=w)
    if code == _NDTR:
        return Ball(Rational(1, 2), Rational(1, 2), precision=w)
    if code == _ERFI or code == _SHI or code == _LOG_NDTR:
        return Ball.unbounded(w)
    return Ball.indeterminate(w)


def _ball_special(code: Int, x: _BallArgument, n: Int, context: Optional[BallContext]) raises -> Ball:
    """A special function of a ball: inclusion, the domains, and the budget's
    trivial enclosures."""
    var w = _working(context, x.precision)
    if x.kind == _INDETERMINATE:
        return Ball.indeterminate(w)
    if x.kind == _UNBOUNDED:
        return _special_unbounded(code, w)
    var budget = _ball_budget(context, w)
    var ball = x.ball()
    if (
        ball.is_exact() and (code == _GAMMA or code == _LOG_GAMMA or code == _DIGAMMA) and w <= 62
        and ball._midpoint._kind == 1 and not ball._midpoint._negative and not ball._midpoint.is_integer()
    ):
        # An exact point of one word: the one-word Taylor pass, a radius
        # of its error alone. Integers keep their exact values below.
        var point = _gamma_spread(code, ball, w, budget)
        if point:
            if point.value()._kind == _FINITE and point.value()._midpoint.precision() == w:
                return point.value()
            return _rounded_ball(_BallArgument(point.value()), w)
    if ball.is_exact():
        var value = _special_point(code, ball._midpoint, n, w, budget)
        if not value.is_finite():
            return value^
        return _rounded_ball(_BallArgument(value), w)
    if (code == _GAMMA or code == _LOG_GAMMA or code == _DIGAMMA) and ball._midpoint._kind == 1 and not ball._midpoint._negative:
        # A ball above 0, decided in radius arithmetic: no pole, and the
        # midpoint rule applies without the exact ends.
        if not _Radius.lower_gap_float(ball._midpoint, ball._radius).is_zero():
            var narrow = _gamma_spread(code, ball, w, budget)
            if narrow:
                if narrow.value()._kind == _FINITE and narrow.value()._midpoint.precision() == w:
                    return narrow.value()
                return _rounded_ball(_BallArgument(narrow.value()), w)
    var low = ball._exact_lower()
    var high = ball._exact_upper()
    # Poles and domains: Gamma, log Gamma and digamma at 0 and the negative
    # integers, Ei at 0, Ci and Chi at 0 and below.
    if code == _GAMMA or code == _DIGAMMA or code == _LOG_GAMMA:
        if not (low > Float(0)):
            if not (high < Float(0)) or _integral_between(low, high):
                return Ball.indeterminate(w)
    if (code == _EI and not (low > Float(0) or high < Float(0))) or ((code == _CI or code == _CHI) and not (low > Float(0))):
        return Ball.indeterminate(w)
    if code == _ERFINV and not (low > Float(-1) and high < Float(1)):
        return Ball.indeterminate(w)
    if code == _NDTRI and not (low > Float(0) and high < Float(1)):
        return Ball.indeterminate(w)
    if code == _LAMBERT_W and n != 0 and not (high < Float(0)):
        return Ball.indeterminate(w)
    if code == _GAMMA or code == _LOG_GAMMA or code == _DIGAMMA:
        var narrow = _gamma_spread(code, ball, w, budget)
        if narrow:
            return _rounded_ball(_BallArgument(narrow.value()), w)
    if code == _EI:
        return _monotone_by[_special_point](code, ball, n, w, budget, not (high < Float(0)))
    if code == _ERFC or code == _LAMBERT_W:
        return _monotone_by[_special_point](code, ball, n, w, budget, code == _LAMBERT_W and n == 0)
    if _increasing(code):
        return _monotone_by[_special_point](code, ball, n, w, budget, True)
    if code == _GAMMA or code == _LOG_GAMMA:
        var pieces = _gamma_minimum(code, ball, w, budget)
        if pieces:
            return pieces.value()
    # The midpoint value, widened by the radius times a derivative bound.
    var center = _special_point(code, ball._midpoint, n, w, budget)
    if not center.is_finite():
        return Ball.indeterminate(w)
    var slope = _special_derivative(code, ball, w, budget)
    if not slope.is_finite():
        return Ball.indeterminate(w)
    var spread = _mul(slope, Ball(_midpoint=Float(0), _radius=ball._radius, _kind=_FINITE), w)
    var result = _rounded_ball(_BallArgument(_widen(center, spread)), w)
    if code == _SI or code == _FRESNEL_S or code == _FRESNEL_C:
        return _intersect_range(result, _special_unbounded(code, w), w)
    return result^


def _integral_between(low: Float, high: Float) raises -> Bool:
    """Whether an integer lies in `[low, high]`."""
    return low.floor() != high.floor() or _integral(low).__bool__()


def _intersect_range(x: Ball, bound: Ball, w: Int) raises -> Ball:
    """x clipped to a range it is known to lie in."""
    var low = x._exact_lower()
    var high = x._exact_upper()
    var bottom = bound._exact_lower()
    var top = bound._exact_upper()
    if low >= bottom and high <= top:
        return x
    return _hull(low if low > bottom else bottom, high if high < top else top, w)
