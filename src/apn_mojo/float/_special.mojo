"""The correctly rounded special functions.

Special values are the functions' limits: poles signal divide-by-zero, and
arguments outside the domain give NaN. Exact values are
`Gamma(n) = (n-1)!` and `gammaln(1) = gammaln(2) = 0`. Results within a
tiny distance of a Float, which no precision would separate from it, are
decided by `_round_near`:

| Function | Argument | Base | Distance | Side |
|---|---|---|---|---|
| erf | `|x| >= 1` | `+-1` | `erfc|x| < e**(-x**2)` | toward 0 |
| erfc | `x <= -1` | 2 | `erfc|x| < e**(-x**2)` | below |
| erfc | `|x| < 1/2` | 1 | `|erf x| < 1.13 |x|` | above for x < 0 |
| Si, Shi | `|x| < 1/2` | x | `|x|**3 / 16` | toward 0 for Si |
| Fresnel C | `|x| < 1/2` | x | `|x|**5 / 4` | toward 0 |
| Fresnel S, C | large `|x|` | `+-1/2` | `1/|x|` | the sign of `f cos u + g sin u` or `g cos u - f sin u` |
| Lambert W_0 | `|x| <= 1/4` | x | `3.3 x**2` | below |
| Gamma, digamma | `x = +-2**k <= 1/4` | `1/x`, `-1/x` | 2 | below |
| ndtr | `x >= 2` | 1 | `erfc(x/sqrt 2)/2 < e**(-x**2/2)/2` | below |

Overflow and underflow are decided from bounds where the kernels' exponents
would leave the 64-bit range: `Gamma(x) > e**((x-1/2) log x - x)`,
`erfc x < e**(-x**2)`, `erfi x > e**(x**2)/(4x)`, `Ei(x) > e**x/(2x)`,
`0 < E1(x) < e**(-x)/x`, and for ndtr and log ndtr beyond 2, `q < e**(-x**2/2)/2`
for `q = erfc(|x|/sqrt 2)/2` and `|log1p(-q)| < 2q`.

beta, betaln and poch (`_ratio_rounded`) take their rational values exactly
and scipy's values at the poles from `ball/_ratios.mojo`, and the rest from
its gamma ratios through the same driver.

hyp1f1 (`_hyp1f1_rounded`) takes its rational values exactly and `+inf` at
the poles of b from `ball/_hypergeometric.mojo`; NaN for an infinite a and 1
for an infinite b, as scipy; at an infinite x the limit, from the leading
term of the polynomial, of Kummer's polynomial times `e**x`, or of
`Gamma(b)/Gamma(a) e**x x**(a-b)` and `Gamma(b)/Gamma(b-a) (-x)**-a`; and
overflow beyond `x = 2**56` for formats with `emax <= 2**40`, where `e**x`
outgrows every other factor for parameters of at most 20-bit numerators and
denominators; in wider formats such an x raises the budget error.

gammainc and gammaincc (`_gammainc_rounded`) take scipy's values at a = 0,
x = 0 and the infinities, and NaN below 0. `P <= x**a / Gamma(a+1) <
1.13 x**a` (since `e**-x M(1, a+1, x) <= 1`) and, for `x > 2a`,
`Q <= 2 x**(a-1) e**-x / Gamma(a) < 2.26 x**(a-1) e**-x` (DLMF 8.11.3, the
expansion's terms shrinking by `a/x <= 1/2`) decide underflow and the other
function within a tiny distance of 1.

hyp2f1 (`_hyp2f1_rounded`) takes its rational values exactly
(`ball/_hypergeometric.mojo`), `+inf` with divide-by-zero at scipy's poles
(c a non-positive integer the polynomial does not end before, x = 1 with
`c - a - b <= 0`), and NaN for x > 1 other than a polynomial, where scipy
returns `+inf`, and for an infinite parameter. At an infinite x a polynomial
takes its leading term's limit, and anything else is NaN.

betainc (`_betainc_rounded`) takes scipy's values at the edges and its
rational values exactly (`ball/_incomplete.mojo`). Where a double-precision
estimate puts the directly summed side below the precision (or the range), a
64-bit enclosure of it decides `1 - I` by `_round_near`, or underflow.
"""

from std.math import log
from ..integer.value import Integer
from ..integer.math import factorial
from ..rational.value import Rational
from .value import Float
from .context import ArithmeticContext, FloatFormat, _exact_context
from .status import NumericStatus
from ._arithmetic import _FloatArgument, _float_operation, _call_context
from ._input import _FloatInput
from ._rounding import _RoundedBinary, _inexact_error, _round_ratio
from ._elementary import _exact_float, _unit, _special, _invalid, _exactly, _signed_constant, _overflow, _underflow
from ._format import _FormatSlot, _merge_float_formats
from .math import _exact_rational
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._certified import _Enclosure, _round_certified, _round_near, _budget_error
from ..ball._kernels import _HALF_PI, _EXP, _LOG, _integral
from ..ball._special import (
    _GAMMA, _LOG_GAMMA, _DIGAMMA, _ERF, _ERFC, _ERFI, _EI, _SI, _CI, _SHI, _CHI, _FRESNEL_S, _FRESNEL_C,
    _LAMBERT_W, _NDTR, _LOG_NDTR, _BETA, _LOG_BETA, _POCH, _ERFINV, _NDTRI, _ZETA, _POLYGAMMA, _SpecialKernel, _special_name, _ball_special, _floor_scaled, _fresnel_auxiliary,
    _half_square_mod_two, _sin_cos_pi, _add, _sub, _mul, _div, _fn, _one,
)
from ..ball._constants import _ln2
from ..ball._ratios import _ratio_ball, _decided, _RATIONAL, _INFINITE
from ..ball._zeta import _zeta_ball, _polygamma_ball, _zeta_case, _INVALID, _int
from ..ball._ratios import _nonpositive_integer
from ..ball._hypergeometric import (
    _hyp1f1_case, _hyp1f1_enclosure, _scipy_polynomial, _MODERATE_BITS, _hyp2f1_case, _hyp2f1_enclosure, _degree, _OUTSIDE,
)
from ..ball._incomplete import _gammainc_enclosure, _betainc_enclosure, _betainc_case, _betainc_direct, _betainc_flipped, _lgamma_estimate
from ..ball._arithmetic import _negate


@fieldwise_init
struct _SpecialBallPath(_Enclosure):
    """A special function at an exact rational that is not a binary fraction:
    the ball function of an enclosure of the argument at each precision."""

    var code: Int
    var x: _FloatArgument
    var n: Int

    def enclosure(self, precision: Int) raises -> Ball:
        var a = Ball(self.x, precision=precision + 32)
        return _ball_special(self.code, _BallArgument(a), self.n, Optional[BallContext](BallContext(precision + 8)))

    def describe(self) raises -> String:
        return String(_special_name(self.code), " at an exact rational")


def _small_float(value: Int) raises -> Float:
    """A small integer in a format of its own bits, for `_round_near`'s base."""
    return Float(value, context=ArithmeticContext(_format_of=FloatFormat(2)))


def _power_of_two(sign: Bool, exponent: Int) raises -> Float:
    """`+-2**(exponent - 1)` in a 1-bit format."""
    return Float(_rounded=_RoundedBinary(1, sign, Integer(1), exponent, FloatFormat(1), NumericStatus()))


def _at_zero(code: Int, negative: Bool, branch: Int, target: ArithmeticContext) raises -> _RoundedBinary:
    """The function at +-0."""
    if code == _ERFC:
        return _exactly(Integer(1), target)
    if code == _NDTR:
        return _round_ratio(Integer(1), Integer(2), target)
    if code == _NDTRI:
        return _special(2, True, 8, target)
    if code == _GAMMA:
        return _special(2, negative, 8, target)
    if code == _DIGAMMA:
        return _special(2, not negative, 8, target)
    if code == _LOG_GAMMA:
        return _special(2, False, 8, target)
    if code == _EI or code == _CI or code == _CHI or (code == _LAMBERT_W and branch != 0):
        return _special(2, True, 8, target)
    return _special(0, negative, 0, target)


def _at_infinity(code: Int, negative: Bool, branch: Int, target: ArithmeticContext) raises -> _RoundedBinary:
    """The function at +-inf."""
    if code == _NDTR:
        return _special(0, False, 0, target) if negative else _exactly(Integer(1), target)
    if code == _LOG_NDTR:
        return _special(2, True, 0, target) if negative else _special(0, False, 0, target)
    if code == _LOG_GAMMA:
        return _special(2, False, 0, target)
    if code == _ERFINV or code == _NDTRI:
        return _invalid(target)
    if code == _ERF:
        return _exactly(Integer(-1 if negative else 1), target)
    if code == _ERFC:
        return _exactly(Integer(2), target) if negative else _special(0, False, 0, target)
    if code == _ERFI or code == _SHI:
        return _special(2, negative, 0, target)
    if code == _SI:
        return _signed_constant(_HALF_PI, negative, target)
    if code == _FRESNEL_S or code == _FRESNEL_C:
        return _round_ratio(Integer(-1 if negative else 1), Integer(2), target)
    if code == _EI:
        return _special(0, True, 0, target) if negative else _special(2, False, 0, target)
    if negative or (code == _LAMBERT_W and branch != 0):
        return _invalid(target)
    if code == _CI:
        return _special(0, False, 0, target)
    return _special(2, False, 0, target)


def _lambert_side(x: _FloatArgument, budget: Int) raises -> Int:
    """The sign of `1 + e x`: 1 when x lies above -1/e, which no rational
    equals, -1 below, 0 when the budget ends first."""
    var w = 64
    while w <= budget:
        var e = _fn(_EXP, _one(w), w)
        var s = _add(_one(w), _mul(e, Ball(x, precision=w + 32), w), w)
        if s.certainly_positive():
            return 1
        if s.certainly_negative():
            return -1
        w *= 2
    return 0


def _log2_gamma_lower(y: Float) raises -> Float:
    """A lower bound of `log2 Gamma(y)` for y > 0, from
    `log Gamma(y) > (y - 1/2) log y - y` (Binet's remainder is positive)."""
    var w = 64
    var b = Ball(y)
    var bound = _sub(_mul(_sub(b, Ball(Rational(1, 2), precision=w), w), _fn(_LOG, b, w), w), b, w)
    return _div(bound, _ln2(w), w)._exact_lower()


def _gamma_underflows(x: Float, emin: Int) raises -> Optional[Bool]:
    """For x < 0, the sign of Gamma(x) when `|Gamma(x)| < 2**(emin - 2)`
    certainly: `|Gamma(x)| = pi / (|sin pi x| Gamma(1 - x)) <= pi / (2 d
    Gamma(1 - x))` with d the distance from x to the nearest integer, and
    Gamma(x) has the sign of `(-1)**(floor(-x) + 1)`."""
    var n = x.round()
    var d = abs(Float(_rounded=_float_operation(x, n, 1, _exact_context())))
    var one_minus = Float(_rounded=_float_operation(Integer(1), x, 1, _exact_context()))
    var bound = Integer(2 - d._exponent) - _log2_gamma_lower(one_minus).floor()
    if bound >= Integer(emin - 2):
        return None
    var below = (-x).floor()
    return Optional[Bool](not (below & 1).__bool__())


def _fresnel_far(x: Float, sine: Bool, target: ArithmeticContext, budget: Int) raises -> Optional[_RoundedBinary]:
    """S or C at a large |x|, within `1/|x| <= 2**(1 - E)` of `+-1/2`; the
    side is the sign of `S - 1/2 = -(f cos u + g sin u)` or
    `C - 1/2 = f sin u - g cos u` at |x|, odd in x."""
    var e = x._exponent
    var p = target.format().precision()
    if e < p + 5:
        return None
    var a = abs(x)
    var r = _half_square_mod_two(a)
    var w = 64
    while w <= budget:
        var fg = _fresnel_auxiliary(a, w)
        if fg:
            var sc = _sin_cos_pi(r, w, budget)
            var f = fg.value()[0]
            var g = fg.value()[1]
            var d: Ball
            if sine:
                d = _negate(_add(_mul(f, sc[1], w), _mul(g, sc[0], w), w))
            else:
                d = _sub(_mul(f, sc[0], w), _mul(g, sc[1], w), w)
            if d.certainly_positive() or d.certainly_negative():
                var above = d.certainly_positive() != x._negative
                return _round_near(_power_of_two(x._negative, 0), above, 1 - e, target)
        w *= 2
    return None


def _special_rule(code: Int, x: Float, branch: Int, target: ArithmeticContext, budget: Int) raises -> Optional[_RoundedBinary]:
    """The rules for a result near a Float, and overflow and underflow."""
    var e = x._exponent
    var negative = x._negative
    var emin = target.format().emin()
    var emax = target.format().emax()
    if code == _ERF:
        if e >= 1:
            return _round_near(_unit(negative), negative, -_floor_scaled(x, 14426, 10000, True), target)
        return None
    if code == _ERFC:
        if e >= 1:
            var bits = _floor_scaled(x, 14426, 10000, True)
            if negative:
                return _round_near(_small_float(2), False, -bits, target)
            if -bits <= emin - 2:
                return _underflow(False, target)
            return None
        if e < 0:
            return _round_near(_unit(False), negative, e + 1, target)
        return None
    if code == _ERFI:
        if e >= 1 and _floor_scaled(x, 14426, 10000, True) - (e + 2) > emax:
            return _overflow(negative, target)
        return None
    if code == _EI:
        if e >= 5:
            var bits = _floor_scaled(x, 14426, 10000, False)
            if negative and -bits <= emin - 2:
                return _underflow(True, target)
            if not negative and bits - (e + 1) > emax:
                return _overflow(False, target)
        return None
    if code == _SHI or code == _CHI:
        if e >= 5 and _floor_scaled(x, 14426, 10000, False) - (e + 2) > emax:
            return _overflow(negative and code == _SHI, target)
        if code == _SHI and e < 0:
            return _round_near(x, not negative, 3 * e - 4, target)
        return None
    if code == _SI:
        if e < 0:
            return _round_near(x, negative, 3 * e - 4, target)
        return None
    if code == _FRESNEL_C:
        if e < 0:
            return _round_near(x, negative, 5 * e - 2, target)
        return _fresnel_far(x, False, target, budget)
    if code == _FRESNEL_S:
        if e < 0 and 3 * e <= emin - 2:
            return _underflow(negative, target)
        return _fresnel_far(x, True, target, budget)
    if code == _LAMBERT_W:
        if branch == 0 and e <= -2:
            return _round_near(x, False, 2 * e + 2, target)
        return None
    if code == _NDTR or code == _LOG_NDTR:
        # Beyond 2, q = erfc(|x|/sqrt 2)/2 < e**(-x**2/2)/2 < 2**-bits.
        if e < 2:
            return None
        var bits = _floor_scaled(x, 7213, 10000, True) + 1
        if negative:
            if code == _NDTR and -bits <= emin - 2:
                return _underflow(False, target)
            return None
        if code == _NDTR:
            return _round_near(_unit(False), False, -bits, target)
        # log ndtr = log1p(-q), with |log1p(-q)| < 2q.
        if 1 - bits <= emin - 2:
            return _underflow(True, target)
        return None
    if code == _GAMMA or code == _DIGAMMA:
        var sign = negative if code == _GAMMA else not negative
        if e <= -emax - 1:
            return _overflow(sign, target)
        if e <= -1 and (x._significand & (x._significand - 1)) == 0:
            # x = +-2**(e-1): Gamma(x) = 1/x - (gamma + ...) and digamma(x) =
            # -1/x - (gamma + ...), each below its base by less than 2.
            return _round_near(_power_of_two(sign, 2 - e), False, 1, target)
        if code == _GAMMA and e >= 10:
            if not negative:
                if _log2_gamma_lower(x) > Float(emax):
                    return _overflow(False, target)
                if e > 56:
                    raise _budget_error(String("gamma at ", x), budget)
            else:
                var tiny = _gamma_underflows(x, emin)
                if tiny:
                    return _underflow(tiny.value(), target)
                if e > 56:
                    raise _budget_error(String("gamma at ", x), budget)
        return None
    return None


def _special_rounded(
    code: Int, value: _FloatArgument, context: Optional[ArithmeticContext], branch: Int = 0, guard: Int = 0,
) raises -> _RoundedBinary:
    """A special function, correctly rounded."""
    if code == _LAMBERT_W and branch != 0 and branch != -1:
        raise Error("lambertw has the branches k=0 and k=-1.")
    var target = _call_context(value, context)
    var v = value.value
    if v.kind == 3:
        return _special(3, False, 0, target)
    if v.kind == 0 and code == _LOG_NDTR:
        return _round_certified(_SpecialKernel(code, Float.zero(), 0, target._budget()), target, guard)
    if v.kind == 0:
        return _at_zero(code, v.negative, branch, target)
    if v.kind == 2:
        return _at_infinity(code, v.negative, branch, target)
    var budget = target._budget()
    var negative = v.negative
    if negative and (code == _CI or code == _CHI):
        return _invalid(target)
    if code == _ERFINV or code == _NDTRI:
        var r = _exact_rational(v)
        if code == _ERFINV:
            if abs(r) > Rational(1):
                return _invalid(target)
            if abs(r) == Rational(1):
                return _special(2, negative, 8, target)
        else:
            if negative or r > Rational(1):
                return _invalid(target)
            if r == Rational(1):
                return _special(2, False, 8, target)
            if r == Rational(1, 2):
                return _special(0, False, 0, target)
    if code == _LAMBERT_W:
        if branch != 0 and not negative:
            return _invalid(target)
        var side = _lambert_side(value, budget)
        if side < 0:
            return _invalid(target)
        if side == 0:
            raise _budget_error(String("lambertw's domain at its argument"), budget)
    var exact = _exact_float(value)
    if not exact:
        # A rational that is not a binary fraction: never an integer or a pole.
        if target.format()._is_exact():
            raise _inexact_error(_special_name(code))
        return _round_certified(_SpecialBallPath(code, value, branch), target, guard)
    var x = exact.take()
    var whole = _integral(x)
    if (code == _GAMMA or code == _DIGAMMA) and whole and negative:
        return _invalid(target)
    if code == _LOG_GAMMA and whole and negative:
        return _special(2, False, 8, target)
    if code == _GAMMA and whole and not negative and whole.value() <= Integer(65536):
        return _exactly(factorial(whole.value() - 1), target)
    if code == _LOG_GAMMA and whole and (whole.value() == 1 or whole.value() == 2):
        return _special(0, False, 0, target)
    if target.format()._is_exact():
        raise _inexact_error(_special_name(code))
    var rule = _special_rule(code, x, branch, target, budget)
    if rule:
        return rule.take()
    return _round_certified(_SpecialKernel(code, x, branch, budget), target, guard)


@fieldwise_init
struct _RatioEnclosure(_Enclosure):
    """beta, betaln or poch at two exact numbers, for the Ziv driver: the
    ball function at enclosures of the arguments, exact for binary
    fractions."""

    var code: Int
    var a: _FloatArgument
    var b: _FloatArgument

    def enclosure(self, precision: Int) raises -> Ball:
        var x = Ball(self.a, precision=precision + 32)
        var y = Ball(self.b, precision=precision + 32)
        return _ratio_ball(self.code, _BallArgument(x), _BallArgument(y), Optional[BallContext](BallContext(precision + 8)))

    def describe(self) raises -> String:
        return String(_special_name(self.code), " at its arguments")


@fieldwise_init
struct _LogRational(_Enclosure):
    """The logarithm of a positive rational other than 1, for betaln at a
    rational beta value."""

    var value: Rational

    def enclosure(self, precision: Int) raises -> Ball:
        return _fn(_LOG, Ball(self.value, precision=precision + 32), precision + 8)

    def describe(self) raises -> String:
        return String("betaln at a rational value")


def _ratio_at_infinity(code: Int, x: _FloatInput, y: _FloatInput, target: ArithmeticContext) raises -> _RoundedBinary:
    """The limits at an infinite argument: `B(+inf, b) = +0` and
    `betaln(+inf, b) = -inf` for b > 0 (or +inf), and `(+inf)_m` is +inf, 1
    or +0 as m is above, at or below 0; NaN otherwise."""
    if code == _POCH:
        if x.kind != 2 or x.negative or y.kind == 2:
            return _invalid(target)
        if y.kind == 0:
            return _exactly(Integer(1), target)
        return _special(0, False, 0, target) if y.negative else _special(2, False, 0, target)
    var a_ok = (x.kind == 2 and not x.negative) or (x.kind == 1 and not x.negative)
    var b_ok = (y.kind == 2 and not y.negative) or (y.kind == 1 and not y.negative)
    if not (a_ok and b_ok):
        return _invalid(target)
    return _special(2, True, 0, target) if code == _LOG_BETA else _special(0, False, 0, target)


def _ratio_rounded(code: Int, a: _FloatArgument, b: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """beta, betaln or poch, correctly rounded: rational values exactly,
    infinities with divide-by-zero at scipy's poles, the rest certified."""
    var target = _call_context(a, b, context)
    var x = a.value
    var y = b.value
    if x.kind == 3 or y.kind == 3:
        return _special(3, False, 0, target)
    if x.kind == 2 or y.kind == 2:
        return _ratio_at_infinity(code, x, y, target)
    var decided = _decided(code, _exact_rational(x) if x.kind else Rational(0), _exact_rational(y) if y.kind else Rational(0))
    if decided.kind == _INFINITE:
        return _special(2, False, 8, target)
    if decided.kind == _RATIONAL:
        if code != _LOG_BETA:
            return _round_ratio(decided.value.numerator(), decided.value.denominator(), target)
        var size = abs(decided.value)
        if size.sign() == 0:
            return _special(2, True, 8, target)
        if size == Rational(1):
            return _special(0, False, 0, target)
        if target.format()._is_exact():
            raise _inexact_error(_special_name(code))
        return _round_certified(_LogRational(size^), target, 0)
    if target.format()._is_exact():
        raise _inexact_error(_special_name(code))
    return _round_certified(_RatioEnclosure(code, a, b), target, 0)


@fieldwise_init
struct _ZetaEnclosure(_Enclosure):
    """zeta(s, q), or polygamma(n, x) with q unused, at exact arguments, for
    the Ziv driver."""

    var code: Int
    var a: _FloatArgument
    var b: _FloatArgument
    var n: Integer

    def enclosure(self, precision: Int) raises -> Ball:
        var context = Optional[BallContext](BallContext(precision + 8))
        var x = Ball(self.a, precision=precision + 32)
        if self.code == _POLYGAMMA:
            return _polygamma_ball(self.n, _BallArgument(x), context)
        return _zeta_ball(_BallArgument(x), _BallArgument(Ball(self.b, precision=precision + 32)), context)

    def describe(self) raises -> String:
        return String(_special_name(self.code), " at its arguments")


def _zeta_at_infinity(s: _FloatInput, q: _FloatInput, target: ArithmeticContext) raises -> _RoundedBinary:
    """The limits: `zeta(+inf, q)` is +0, 1 or +inf as q is above, at or below
    1 (q > 0), and `zeta(s, +inf)` is +0 for s > 1; NaN otherwise."""
    if s.kind == 2:
        if s.negative or q.kind != 1 or q.negative:
            return _invalid(target)
        var r = _exact_rational(q)
        if r > Rational(1):
            return _special(0, False, 0, target)
        if r == Rational(1):
            return _exactly(Integer(1), target)
        return _special(2, False, 0, target)
    if q.negative or s.kind != 1 or s.negative or not (_exact_rational(s) > Rational(1)):
        return _invalid(target)
    return _special(0, False, 0, target)


def _zeta_rounded(s: _FloatArgument, q: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """zeta(s, q), correctly rounded: Riemann's rational values exactly, +inf
    with divide-by-zero at the poles, NaN outside scipy's domain, and within
    `2**(2 - floor s)` of 1 for the Riemann zeta function at s >= 2."""
    var target = _call_context(s, q, context)
    var x = s.value
    var y = q.value
    if x.kind == 3 or y.kind == 3:
        return _special(3, False, 0, target)
    if x.kind == 2 or y.kind == 2:
        return _zeta_at_infinity(x, y, target)
    var sr = _exact_rational(x) if x.kind else Rational(0)
    var qr = _exact_rational(y) if y.kind else Rational(0)
    var found = _zeta_case(sr, qr)
    if found.kind == _RATIONAL:
        return _round_ratio(found.value.numerator(), found.value.denominator(), target)
    if found.kind == _INFINITE:
        return _special(2, False, 8, target)
    if found.kind == _INVALID:
        return _invalid(target)
    if target.format()._is_exact():
        raise _inexact_error(_special_name(_ZETA))
    if qr == Rational(1) and sr >= Rational(2):
        # zeta(s) - 1 = sum_{k>=2} k**-s < 2**-s (1 + 2/(s-1)) <= 3 2**-s.
        var whole = sr.floor()
        var tiny = -(Int(1) << 61) if whole.magnitude_bit_length() > 60 else 2 - Int(whole)
        var near = _round_near(_unit(False), True, tiny, target)
        if near:
            return near.take()
    return _round_certified(_ZetaEnclosure(_ZETA, s, q, Integer(0)), target, 0)


def _polygamma_rounded(n: Integer, x: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """polygamma(n, x), correctly rounded: digamma for n = 0, NaN for n < 0,
    `(-1)**(n+1) inf` with divide-by-zero at the poles as scipy, and
    `(-1)**(n+1) 0` at +inf."""
    if n.sign() == 0:
        return _special_rounded(_DIGAMMA, x, context)
    var target = _call_context(x, context)
    var v = x.value
    var odd = (n & 1).__bool__()
    if n.sign() < 0 or v.kind == 3 or (v.kind == 2 and v.negative):
        return _special(3, False, 0, target) if v.kind == 3 else _invalid(target)
    if v.kind == 2:
        return _special(0, not odd, 0, target)
    var r = _exact_rational(v) if v.kind else Rational(0)
    if r.is_integer() and r.sign() <= 0:
        return _special(2, not odd, 8, target)
    if target.format()._is_exact():
        raise _inexact_error(_special_name(_POLYGAMMA))
    return _round_certified(_ZetaEnclosure(_POLYGAMMA, x, x, n), target, 0)


def _merged_target3(a: _FloatArgument, b: _FloatArgument, c: _FloatArgument, context: Optional[ArithmeticContext]) raises -> ArithmeticContext:
    """The output context of a three-argument function: the context, else the
    merged argument formats rounded to nearest-even."""
    if context:
        return context.value()
    var first = _FormatSlot(None)
    if a.format or b.format or a.native_precision or b.native_precision:
        first = _FormatSlot(_merge_float_formats(
            a.format, b.format, left_native_precision=a.native_precision, right_native_precision=b.native_precision,
        ))
    return ArithmeticContext(_format_of=_merge_float_formats(first, c.format, right_native_precision=c.native_precision))


def _moderate_input(x: _FloatInput) raises -> Optional[Rational]:
    """The value of a finite exact argument record, when it is of modest
    size; the exact cases look no further."""
    if x.kind == 0:
        return Rational(0)
    var scale = x.scale if x.scale >= 0 else -x.scale
    if scale > Int128(_MODERATE_BITS) or x.numerator.magnitude_bit_length() + x.denominator.magnitude_bit_length() > _MODERATE_BITS:
        return None
    return _exact_rational(x)


def _gamma_negative(t: Rational) raises -> Bool:
    """Whether Gamma(t) < 0, at a t that is not a pole: on `(-n-1, -n)` its
    sign is `(-1)**(n+1)`."""
    if t.sign() > 0:
        return False
    return not ((-t).floor() & 1).__bool__()


def _rising_negative(b: Rational, n: Integer) raises -> Bool:
    """Whether `(b)_n < 0` when no factor is 0: the factors `b + k`, k < n,
    below 0 number `min(n, ceil(-b))`."""
    if b.sign() >= 0:
        return False
    var count = (-b).ceil()
    if n < count:
        count = n
    return (count & 1).__bool__()


def _hyp1f1_limit(a: Rational, b: Rational, upward: Bool) raises -> Tuple[Int, Bool]:
    """M(a, b, x) as x -> +inf (upward) or -inf, b not a pole: `(1, False)`
    for a = 0, else `(0, sign)` or `(2, sign)` for a signed zero or infinity.
    A polynomial follows its leading term `(a)_n / ((b)_n n!) x**n`, Kummer's
    case `e**x M(b-a, b, -x)` the leading term of its polynomial, and the
    rest `Gamma(b)/Gamma(a) e**x x**(a-b)` above and
    `Gamma(b)/Gamma(b-a) (-x)**-a` below (DLMF 13.7.1, 13.7.2)."""
    if a.sign() == 0:
        return (1, False)
    if _nonpositive_integer(a):
        var n = (-a).numerator()
        var odd = (n & 1).__bool__()
        var lead = odd != _rising_negative(b, n)
        return (2, lead != (odd and not upward))
    var c = b - a
    if _nonpositive_integer(c):
        var m = (-c).numerator()
        var odd = (m & 1).__bool__()
        var lead = odd != _rising_negative(b, m)
        if upward:
            return (2, lead != odd)
        return (0, lead)
    if upward:
        return (2, _gamma_negative(b) != _gamma_negative(a))
    var sign = _gamma_negative(b) != _gamma_negative(c)
    return (0, sign) if a.sign() > 0 else (2, sign)


@fieldwise_init
struct _Hyp1f1Enclosure(_Enclosure):
    """M(a, b, x) at exact arguments that are none of its exact cases, for
    the Ziv driver: the enclosure at balls of the arguments, exact for binary
    fractions."""

    var a: _FloatArgument
    var b: _FloatArgument
    var x: _FloatArgument

    def enclosure(self, precision: Int) raises -> Ball:
        var a = Ball(self.a, precision=precision + 32)
        var b = Ball(self.b, precision=precision + 32)
        var x = Ball(self.x, precision=precision + 32)
        return _hyp1f1_enclosure(a, b, x, precision + 8)

    def describe(self) raises -> String:
        return String("hyp1f1 at its arguments")


def _hyp1f1_rounded(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """M(a, b, x), correctly rounded: rational values exactly, `+inf` with
    divide-by-zero at scipy's poles, the limits at infinite arguments, and
    overflow beyond `2**56`."""
    var target = _merged_target3(a, b, x, context)
    var u = a.value
    var v = b.value
    var y = x.value
    if u.kind == 3 or v.kind == 3 or y.kind == 3:
        return _special(3, False, 0, target)
    if u.kind == 2:
        return _invalid(target)
    if v.kind == 2:
        return _invalid(target) if y.kind == 2 else _exactly(Integer(1), target)
    var ar = _moderate_input(u)
    var br = _moderate_input(v)
    if y.kind == 2:
        if not ar or not br:
            return _invalid(target)
        if _nonpositive_integer(br.value()) and not _scipy_polynomial(ar.value(), br.value()):
            return _special(2, False, 8, target)
        var limit = _hyp1f1_limit(ar.value(), br.value(), not y.negative)
        if limit[0] == 1:
            return _exactly(Integer(1), target)
        return _special(limit[0], limit[1], 0, target)
    var xr = _moderate_input(y)
    if ar and br and xr:
        var decided = _hyp1f1_case(ar.value(), br.value(), xr.value())
        if decided.kind == _RATIONAL:
            return _round_ratio(decided.value.numerator(), decided.value.denominator(), target)
        if decided.kind == _INFINITE:
            return _special(2, False, 8, target)
    if target.format()._is_exact():
        raise _inexact_error("hyp1f1")
    if ar and br and not y.negative and not _nonpositive_integer(ar.value()) and target.format().emax() <= (1 << 40):
        # Beyond 2**56, e**x x**(a-b) Gamma(b)/Gamma(a) (or Kummer's e**x P(-x))
        # exceeds 2**(1.44 * 2**55 - 2**30) for parameters of at most 20-bit
        # numerators and denominators, far past any emax <= 2**40.
        var huge = _exact_float(x)
        var small = ar.value().numerator().magnitude_bit_length() <= 20 and ar.value().denominator().magnitude_bit_length() <= 20
        small = small and br.value().numerator().magnitude_bit_length() <= 20 and br.value().denominator().magnitude_bit_length() <= 20
        if small and huge and huge.value()._exponent > 56:
            return _overflow(_hyp1f1_limit(ar.value(), br.value(), True)[1], target)
    return _round_certified(_Hyp1f1Enclosure(a, b, x), target, 0)


@fieldwise_init
struct _GammaincEnclosure(_Enclosure):
    """P(a, x), or Q with `upper`, at exact a > 0 and x > 0, for the Ziv
    driver."""

    var a: _FloatArgument
    var x: _FloatArgument
    var upper: Bool

    def enclosure(self, precision: Int) raises -> Ball:
        var a = Ball(self.a, precision=precision + 32)
        var x = Ball(self.x, precision=precision + 32)
        return _gammainc_enclosure(a, x, self.upper, precision + 8)

    def describe(self) raises -> String:
        return String("gammaincc" if self.upper else "gammainc", " at its arguments")


def _gammainc_rule(a: _FloatArgument, x: _FloatArgument, upper: Bool, target: ArithmeticContext) raises -> Optional[_RoundedBinary]:
    """Underflow, and the other function within a tiny distance of 1, from
    exact bounds of `log2 P < 0.18 + a log2 x <= ceil(a e) + 1` for
    `x < 2**e <= 1`, and for x > 2a and x > 1,
    `log2 Q < 1.2 + (a-1) log2 x - 1.4426 x`, with `log2 x` at most e and at
    least `e - 1`."""
    var ar = _moderate_input(a.value)
    var xr = _moderate_input(x.value)
    var xf = _exact_float(x)
    if not ar or not xr or not xf:
        return None
    var s = ar.value()
    var t = xr.value()
    var e = xf.value()._exponent
    if s >= Rational(Integer(1) << 40):
        return None
    var bound: Integer
    var small_p: Bool
    if e <= 0:
        bound = (s * Rational(e)).ceil() + 1
        small_p = True
    elif t > Rational(2) * s and t > Rational(1):
        var log_x = Rational(e) if s >= Rational(1) else Rational(e - 1)
        bound = ((s - Rational(1)) * log_x).ceil() - (t * Rational(14426, 10000)).floor() + 2
        small_p = False
    else:
        return None
    if bound > Integer(0):
        return None
    # The small one: P when small_p, else Q.
    if upper != small_p:
        if bound < Integer(target.format().emin()) - 2:
            return _underflow(False, target)
        return None
    var limit = -(Integer(1) << 61)
    return _round_near(_unit(False), False, _int(limit if bound < limit else bound), target)


def _gammainc_rounded(a: _FloatArgument, x: _FloatArgument, upper: Bool, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """P(a, x), or Q with `upper`, correctly rounded, with scipy's values:
    NaN below 0 and at a = x = 0, `P(0, x) = 1`, `P(a, 0) = 0`, `P(+inf, x) =
    0` and `P(a, +inf) = 1` (NaN when both are infinite), Q = 1 - P."""
    var target = _call_context(a, x, context)
    var u = a.value
    var v = x.value
    if u.kind == 3 or v.kind == 3:
        return _special(3, False, 0, target)
    if (u.kind != 0 and u.negative) or (v.kind != 0 and v.negative) or (u.kind == 2 and v.kind == 2):
        return _invalid(target)
    var one = _exactly(Integer(1), target)
    var zero = _special(0, False, 0, target)
    if u.kind == 0:
        if v.kind == 0:
            return _invalid(target)
        return zero if upper else one
    if v.kind == 0 or u.kind == 2:
        return one if upper else zero
    if v.kind == 2:
        return zero if upper else one
    if target.format()._is_exact():
        raise _inexact_error("gammaincc") if upper else _inexact_error("gammainc")
    var rule = _gammainc_rule(a, x, upper, target)
    if rule:
        return rule.take()
    return _round_certified(_GammaincEnclosure(a, x, upper), target, 0)


def _merged_slot(a: _FloatArgument, b: _FloatArgument) raises -> _FormatSlot:
    if a.format or b.format or a.native_precision or b.native_precision:
        return _FormatSlot(_merge_float_formats(
            a.format, b.format, left_native_precision=a.native_precision, right_native_precision=b.native_precision,
        ))
    return _FormatSlot(None)


def _merged_target4(
    a: _FloatArgument, b: _FloatArgument, c: _FloatArgument, d: _FloatArgument, context: Optional[ArithmeticContext],
) raises -> ArithmeticContext:
    """The output context of a four-argument function: the context, else the
    merged argument formats rounded to nearest-even."""
    if context:
        return context.value()
    return ArithmeticContext(_format_of=_merge_float_formats(_merged_slot(a, b), _merged_slot(c, d)))


@fieldwise_init
struct _Hyp2f1Enclosure(_Enclosure):
    """F(a, b; c; x) at exact arguments that are none of its exact cases, for
    the Ziv driver."""

    var a: _FloatArgument
    var b: _FloatArgument
    var c: _FloatArgument
    var x: _FloatArgument

    def enclosure(self, precision: Int) raises -> Ball:
        var a = Ball(self.a, precision=precision + 32)
        var b = Ball(self.b, precision=precision + 32)
        var c = Ball(self.c, precision=precision + 32)
        var x = Ball(self.x, precision=precision + 32)
        return _hyp2f1_enclosure(a, b, c, x, precision + 8)

    def describe(self) raises -> String:
        return String("hyp2f1 at its arguments")


def _hyp2f1_infinite_x(a: Rational, b: Rational, c: Rational, negative: Bool, target: ArithmeticContext) raises -> _RoundedBinary:
    """F at `x = +-inf`: a polynomial's leading term
    `(a)_m (b)_m / ((c)_m m!) x**m`, scipy's `+inf` at a pole of c it does
    not end before, and NaN for anything else."""
    var da = _degree(a)
    var db = _degree(b)
    if da < 0 and db < 0:
        return _invalid(target)
    var m = da if db < 0 or (da >= 0 and da <= db) else db
    if _nonpositive_integer(c) and -c < Rational(m):
        return _special(2, False, 8, target)
    if m == 0:
        return _exactly(Integer(1), target)
    var n = Integer(m)
    var sign = (_rising_negative(a, n) != _rising_negative(b, n)) != _rising_negative(c, n)
    if negative and m % 2 == 1:
        sign = not sign
    return _special(2, sign, 0, target)


def _hyp2f1_rounded(a: _FloatArgument, b: _FloatArgument, c: _FloatArgument, x: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """F(a, b; c; x), correctly rounded: rational values exactly, scipy's
    `+inf` with divide-by-zero at the poles, NaN for x > 1 other than a
    polynomial."""
    var target = _merged_target4(a, b, c, x, context)
    var u = a.value
    var v = b.value
    var q = c.value
    var y = x.value
    if u.kind == 3 or v.kind == 3 or q.kind == 3 or y.kind == 3:
        return _special(3, False, 0, target)
    if u.kind == 2 or v.kind == 2 or q.kind == 2:
        return _invalid(target)
    var ar = _moderate_input(u)
    var br = _moderate_input(v)
    var cr = _moderate_input(q)
    if y.kind == 2:
        if not (ar and br and cr):
            return _invalid(target)
        return _hyp2f1_infinite_x(ar.value(), br.value(), cr.value(), y.negative, target)
    var xr = _moderate_input(y)
    if ar and br and cr and xr:
        var decided = _hyp2f1_case(ar.value(), br.value(), cr.value(), xr.value())
        if decided.kind == _RATIONAL:
            return _round_ratio(decided.value.numerator(), decided.value.denominator(), target)
        if decided.kind == _INFINITE:
            return _special(2, False, 8, target)
        if decided.kind == _OUTSIDE:
            return _invalid(target)
    else:
        var xf = _exact_float(x)
        if xf and xf.value() > Float(1):
            return _invalid(target)
    if target.format()._is_exact():
        raise _inexact_error("hyp2f1")
    return _round_certified(_Hyp2f1Enclosure(a, b, c, x), target, 0)


@fieldwise_init
struct _BetaincEnclosure(_Enclosure):
    """`I_x(a, b)` at exact a, b > 0 and 0 < x < 1, for the Ziv driver."""

    var a: _FloatArgument
    var b: _FloatArgument
    var x: _FloatArgument

    def enclosure(self, precision: Int) raises -> Ball:
        var a = Ball(self.a, precision=precision + 32)
        var b = Ball(self.b, precision=precision + 32)
        var x = Ball(self.x, precision=precision + 32)
        return _betainc_enclosure(a, b, x, precision + 8)

    def describe(self) raises -> String:
        return String("betainc at its arguments")


def _betainc_rule(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, target: ArithmeticContext) raises -> Optional[_RoundedBinary]:
    """`1 - I` within a tiny distance of 1, or underflow, from a 64-bit
    enclosure of the directly summed side S, when a double-precision estimate
    of `log2 S`, `a log2 x + b log2(1-x) + log2(Gamma(a+b) / (Gamma(a+1)
    Gamma(b)))` on that side, puts it below the precision or the range."""
    var ab = Ball(a, precision=96)
    var bb = Ball(b, precision=96)
    var xb = Ball(x, precision=96)
    var flip = _betainc_flipped(ab, bb, xb)
    var p = ab if not flip else bb
    var q = bb if not flip else ab
    var s = xb._midpoint.to_native[DType.float64]()
    if flip:
        s = 1.0 - s
    var pf = p._midpoint.to_native[DType.float64]()
    var qf = q._midpoint.to_native[DType.float64]()
    if not (s > 0.0 and s < 1.0) or not (pf > 0.0 and pf < 1.0e300) or not (qf > 0.0 and qf < 1.0e300):
        return None
    var estimate = (pf * log(s) + qf * log(1.0 - s) + _lgamma_estimate(pf + qf) - _lgamma_estimate(pf + 1.0) - _lgamma_estimate(qf)) * 1.4426950408889634
    var precision = target.format().precision()
    var emin = target.format().emin()
    if flip and estimate > -Float64(precision + 16):
        return None
    if not flip and estimate > Float64(emin) + 16.0:
        return None
    var side = xb if not flip else _sub(_one(96), xb, 96)
    var small = _betainc_direct(p, q, side, 64)
    if not small.is_finite() or not small.certainly_positive():
        return None
    var e = small._exact_upper()._exponent
    if flip:
        return _round_near(_unit(False), False, e, target)
    if e < emin - 2:
        return _underflow(False, target)
    return None


def _betainc_rounded(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, context: Optional[ArithmeticContext]) raises -> _RoundedBinary:
    """`I_x(a, b)`, correctly rounded, with scipy's values: NaN for a < 0,
    b < 0 or x outside [0, 1], and at a = b = 0 and a = b = +inf; 1 for x > 0
    where a = 0 or b = +inf; 0 for x < 1 where b = 0 or a = +inf; 0 at x = 0
    and 1 at x = 1."""
    var target = _merged_target3(a, b, x, context)
    var u = a.value
    var v = b.value
    var y = x.value
    if u.kind == 3 or v.kind == 3 or y.kind == 3:
        return _special(3, False, 0, target)
    if (u.kind != 0 and u.negative) or (v.kind != 0 and v.negative) or (y.kind != 0 and y.negative) or y.kind == 2:
        return _invalid(target)
    var x_one = False
    if y.kind == 1:
        var order = _exact_rational(y) if y.scale < Int128(64) else Rational(2)
        if order > Rational(1):
            return _invalid(target)
        x_one = order == Rational(1)
    if (u.kind == 0 and v.kind == 0) or (u.kind == 2 and v.kind == 2):
        return _invalid(target)
    var one = _exactly(Integer(1), target)
    var zero = _special(0, False, 0, target)
    if u.kind == 0 or v.kind == 2:
        return zero if y.kind == 0 else one
    if v.kind == 0 or u.kind == 2:
        return one if x_one else zero
    if y.kind == 0:
        return zero
    if x_one:
        return one
    var ar = _moderate_input(u)
    var br = _moderate_input(v)
    var xr = _moderate_input(y)
    if ar and br and xr:
        var decided = _betainc_case(ar.value(), br.value(), xr.value())
        if decided.kind == _RATIONAL:
            return _round_ratio(decided.value.numerator(), decided.value.denominator(), target)
    if target.format()._is_exact():
        raise _inexact_error("betainc")
    var rule = _betainc_rule(a, b, x, target)
    if rule:
        return rule.take()
    return _round_certified(_BetaincEnclosure(a, b, x), target, 0)

