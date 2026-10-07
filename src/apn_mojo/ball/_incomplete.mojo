"""The regularized incomplete gamma and beta functions as balls (scipy's
gammainc, gammaincc and betainc).

`P(a, x) = gamma(a, x) / Gamma(a)` and `Q(a, x) = Gamma(a, x) / Gamma(a)`,
with `P + Q = 1`, for a > 0 and x > 0. The smaller one is computed directly
and the other as 1 minus it, so neither cancels:

- P by its series (DLMF 8.5.1), `P = x**a e**-x / Gamma(a+1) M(1, a+1, x)`,
  whose terms `x**k / ((a+1)_k)` are positive, through the series engine of
  `_hypergeometric.mojo`; for `x < a + 1`, where `P` is at most about 1/2.
- Q, for larger x, by its asymptotic expansion (DLMF 8.11.2),
  `Gamma(a, x) = x**(a-1) e**-x [sum_{k<n} u_k / x**k + R_n]` with
  `u_k = (a-1)(a-2)...(a-k)`. For x > 0 and `n >= a - 1` the remainder has
  the sign of `u_n / x**n` and is at most it in size (8.11.3), so the sum is
  widened by its first omitted term. An integer a ends it exactly.
- Where the expansion does not reach the precision, Q is `1 - P` with the
  bits it cancels, about `-log2 Q`, estimated from the expansion's first term.

Beyond `x = 2**56`, `e**-x` would leave the exponent range; there
`Q <= 2 x**(a-1) e**-x / Gamma(a) < 2**(-2**55)` for `a < 2**40` (the terms
of the expansion shrink by `a/x <= 1/2`, and `Gamma(a) > 0.88`).

P increases in x and decreases in a (the Gamma distribution's CDF falls as
its shape grows), and Q does the opposite, so a ball of either takes its two
corners. scipy's conventions hold: NaN for a < 0 or x < 0, `P(0, x) = 1` for
x > 0, NaN at a = x = 0, `P(a, 0) = 0`.

`I_x(a, b) = B_x(a, b) / B(a, b)` for a, b > 0 and 0 < x < 1 takes the side
where its series shrinks from the start: for `x <= (a+1)/(a+b+2)` (DLMF
8.17.22's condition)

    I_x(a, b) = x**a (1-x)**b / (a B(a, b)) F(a+b, 1; a+1; x)       (8.17.8)

whose terms `x (a+b+k)/(a+1+k)` times the last are positive, with a first
ratio of at most `(a+b)/(a+b+2)`; beyond it, `1 - I_{1-x}(b, a)` (8.17.4).
`1 / (a B(a, b)) = Gamma(a+b) / (Gamma(a+1) Gamma(b))`. Integer a and b give
the polynomial `sum_{j=a}^{a+b-1} C(a+b-1, j) x**j (1-x)**(a+b-1-j)` (8.17.5),
and one integer b gives `x**a` times a rational function, from the
polynomial `F(a, 1-b; a+1; x)` (8.17.7): exact when `x**a` is rational, such
as `I_{3/4}(1, 1/2) = 1/2` through the symmetry. I increases in x and b and
decreases in a, so a ball takes two corners again. scipy's values: NaN for
a < 0, b < 0 or x outside [0, 1], and at a = b = 0 and a = b = +inf; 1 for
x > 0 where a = 0 or b = +inf; 0 for x < 1 where b = 0 or a = +inf.
"""

from std.math import log
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from .context import BallContext
from .value import Ball, _BallArgument, _FINITE
from ._radius import _Radius
from ._arithmetic import _working, _rounded_ball, _hull, _negate
from ._kernels import _EXP, _LOG, _LOG1P
from ._special import _add, _sub, _mul, _div, _fn, _one, _magnitude, _widen, _intersect_range, _MAX_TERMS, _exact_add
from ._hypergeometric import _plan, _advance, _terms, _finished_sum, _series, _rgamma, _rational_power, _polynomial, _moderate_value, _LOG2_E, _BALL_GUARD
from ._ratios import _Decided, _neither, _beta_decided, _RATIONAL
from ..integer.math import comb
from ._special import _ball_special, _GAMMA, _c


def _lgamma_estimate(a: Float64) -> Float64:
    """`log Gamma(a)` for a > 0 in double precision, roughly: Stirling's
    formula at 8 or beyond, shifted back."""
    var shift = 0.0
    var t = a
    while t < 8.0:
        shift -= log(t)
        t += 1.0
    return shift + (t - 0.5) * log(t) - t + 0.9189385332046727 + 1.0 / (12.0 * t)


def _lower_series(a: Ball, x: Ball, w: Int) raises -> Ball:
    """`P(a, x) = x**a e**-x / Gamma(a+1) M(1, a+1, x)` for a, x > 0."""
    var work = w
    var shifted = _add(a, _one(work), work)
    var uppers: List[Ball] = [_one(work)]
    var lowers: List[Ball] = [shifted]
    var sum = _series(uppers, lowers, x, work)
    var power = _fn(_EXP, _sub(_mul(a, _fn(_LOG, x, work), work), x, work), work)
    return _mul(_mul(power, _rgamma(shifted, work), work), sum, w)


def _upper_expansion(a: Ball, x: Ball, w: Int) raises -> Optional[Ball]:
    """`Q(a, x)` by its asymptotic expansion, widened by the first omitted
    term with at least `a - 1` terms (DLMF 8.11.2-3); None where it does not
    reach the precision."""
    var work = w
    var uppers: List[Ball] = [_sub(_one(work), a, work), _one(work)]
    var lowers = List[Ball]()
    var z = _negate(_div(_one(work), x, work))
    var plan = _plan(uppers, lowers, z, w, True)
    if plan.terms < 0:
        return None
    work = w + plan.guard
    var sum: Ball
    if plan.finished:
        sum = _finished_sum(uppers, lowers, z, plan.terms, work)
    else:
        var n = plan.terms
        var top = a._exact_upper()
        if top > Float(n + 1):
            if top._exponent > 21:
                return None
            n = Int(top.ceil()._low_magnitude()) - 1
        var pair = _terms(uppers, lowers, z, n, work, plan)
        if not pair[1].is_finite():
            return None
        sum = _widen(pair[0], pair[1])
    var power = _fn(_EXP, _sub(_mul(_sub(a, _one(work), work), _fn(_LOG, x, work), work), x, work), work)
    return _mul(_mul(power, _rgamma(a, work), work), sum, w)


def _gammainc_enclosure(a: Ball, x: Ball, upper: Bool, w: Int) raises -> Ball:
    """P(a, x), or Q with `upper`, for balls a > 0 and x > 0, to about w
    bits (module docstring)."""
    var am = a._midpoint
    var xm = x._midpoint
    if xm._exponent > 56 and not (am._exponent > 40):
        var tiny = Ball(_midpoint=Float(0), _radius=_Radius.power_of_two(-(Int(1) << 55)), _kind=_FINITE)
        return tiny if upper else _sub(_one(w), tiny, w)
    var below = xm < _exact_add(am, Float(1))
    if not below:
        var q = _upper_expansion(a, x, w)
        if q:
            return q.value() if upper else _sub(_one(w), q.value(), w)
    var extra = 0
    if upper and not below and am._exponent < 1000 and xm._exponent < 1000:
        var af = am.to_native[DType.float64]()
        var xf = xm.to_native[DType.float64]()
        var lost = -((af - 1.0) * log(xf) - xf - _lgamma_estimate(af)) * _LOG2_E
        if lost > 0.0:
            extra = Int(min(lost, 1.0e7)) + 16
    var p = _lower_series(a, x, w + extra + 8)
    if not upper:
        return p
    return _sub(_one(w + extra + 8), p, w)


def _gammainc_corner(a: Float, x: Float, upper: Bool, w: Int) raises -> Ball:
    """P or Q at an exact corner a >= 0, x >= 0, not both 0: scipy's 1 (or
    0) at a = 0 and 0 (or 1) at x = 0."""
    if a.is_zero():
        return Ball(Integer(0 if upper else 1), precision=w)
    if x.is_zero():
        return Ball(Integer(1 if upper else 0), precision=w)
    return _gammainc_enclosure(Ball(a), Ball(x), upper, w)


def _gammainc_ball(a: _BallArgument, x: _BallArgument, upper: Bool, context: Optional[BallContext]) raises -> Ball:
    """P(a, x), or Q with `upper`, of two balls: their values at the corners
    where each is least and greatest; indeterminate where a ball reaches
    below 0, or both reach 0."""
    var w = _working(context, a.precision, x.precision)
    if a.kind != _FINITE or x.kind != _FINITE:
        return Ball.indeterminate(w)
    var ab = a.ball()
    var xb = x.ball()
    var a_low = ab._exact_lower()
    var a_high = ab._exact_upper()
    var x_low = xb._exact_lower()
    var x_high = xb._exact_upper()
    if a_low._negative and not a_low.is_zero() or x_low._negative and not x_low.is_zero():
        return Ball.indeterminate(w)
    if a_low.is_zero() and x_low.is_zero():
        return Ball.indeterminate(w)
    var work = w + _BALL_GUARD
    # P grows with x and falls with a; Q the reverse.
    var least = _gammainc_corner(a_low if upper else a_high, x_high if upper else x_low, upper, work)
    if ab.is_exact() and xb.is_exact():
        return _rounded_ball(_BallArgument(least), w)
    var most = _gammainc_corner(a_high if upper else a_low, x_low if upper else x_high, upper, work)
    if not least.is_finite() or not most.is_finite():
        return Ball.indeterminate(w)
    var hull = _hull(least._exact_lower(), most._exact_upper(), w)
    return _intersect_range(hull, Ball(Rational(1, 2), Rational(1, 2), precision=w), w)


# ---------------------------------------------------------------- betainc


def _betainc_polynomial(a: Int, b: Int, x: Rational) raises -> Optional[Rational]:
    """`I_x(a, b)` for positive integers: `sum_{j=a}^{n} C(n, j) x**j (1-x)**(n-j)`,
    n = a + b - 1 (DLMF 8.17.5); None past the size bound."""
    var n = a + b - 1
    if n > 4096 or n * (2 * (x.numerator().magnitude_bit_length() + x.denominator().magnitude_bit_length()) + 13) > (1 << 20):
        return None
    var rest = Rational(1) - x
    var total = Rational(0)
    var power_x = Rational(1)
    for _ in range(a):
        power_x = power_x * x
    var power_rest = Rational(1)
    var powers = List[Rational](length=n - a + 1, fill=Rational(1))
    for k in range(1, n - a + 1):
        power_rest = power_rest * rest
        powers[k] = power_rest
    for j in range(a, n + 1):
        total = total + Rational(comb(Integer(n), Integer(j))) * power_x * powers[n - j]
        power_x = power_x * x
    return total^


def _betainc_power_side(a: Rational, b: Int, x: Rational) raises -> Optional[Rational]:
    """`I_x(a, b)` for a positive integer b, when `x**a` is rational:
    `x**a F(a, 1-b; a+1; x) / (a B(a, b))` with the polynomial F (DLMF
    8.17.7) and `B(a, b) = (b-1)! / (a (a+1) ... (a+b-1))`."""
    var power = _rational_power(x, a)
    if not power:
        return None
    var uppers: List[Rational] = [a, Rational(1 - b)]
    var lowers: List[Rational] = [a + Rational(1)]
    var poly = _polynomial(uppers, lowers, x, b - 1)
    var beta = _beta_decided(a, Rational(b))
    if not poly or beta.kind != _RATIONAL:
        return None
    return power.value() * poly.value() / (a * beta.value)


def _betainc_case(a: Rational, b: Rational, x: Rational) raises -> _Decided:
    """`I_x(a, b)` for a, b > 0 and 0 < x < 1 where it is rational this way."""
    var small_a = a.is_integer() and a.numerator().magnitude_bit_length() <= 12
    var small_b = b.is_integer() and b.numerator().magnitude_bit_length() <= 12
    if small_a and small_b:
        var value = _betainc_polynomial(Int(a.numerator()._low_magnitude()), Int(b.numerator()._low_magnitude()), x)
        if value:
            return _Decided(_RATIONAL, value.value())
        return _neither()
    if small_b:
        var value = _betainc_power_side(a, Int(b.numerator()._low_magnitude()), x)
        if value:
            return _Decided(_RATIONAL, value.value())
    elif small_a:
        var value = _betainc_power_side(b, Int(a.numerator()._low_magnitude()), Rational(1) - x)
        if value:
            return _Decided(_RATIONAL, Rational(1) - value.value())
    return _neither()


def _betainc_direct(a: Ball, b: Ball, x: Ball, w: Int, beta: Optional[Ball] = None) raises -> Ball:
    """`x**a (1-x)**b Gamma(a+b) / (Gamma(a+1) Gamma(b)) F(a+b, 1; a+1; x)`;
    with beta, `1/B(a, b) = Gamma(a+b) / (Gamma(a) Gamma(b))` shared by both
    corners of exact a and b, the factor is `(1/B) / a`, as
    `Gamma(a+1) = a Gamma(a)`."""
    var work = w
    var one = _one(work)
    var shifted = _add(a, one, work)
    var total = _add(a, b, work)
    var uppers: List[Ball] = [total, one]
    var lowers: List[Ball] = [shifted]
    var sum = _series(uppers, lowers, x, work)
    var logs = _add(_mul(a, _fn(_LOG, x, work), work), _mul(b, _fn(_LOG1P, _negate(x), work), work), work)
    var factor: Ball
    if beta:
        factor = _div(beta.value(), a, work)
    else:
        factor = _mul(_mul(_ball_special(_GAMMA, _BallArgument(total), 0, _c(work)), _rgamma(shifted, work), work), _rgamma(b, work), work)
    return _mul(_mul(_fn(_EXP, logs, work), factor, work), sum, w)


def _betainc_flipped(a: Ball, b: Ball, x: Ball) raises -> Bool:
    """Whether the series runs on the other side, `x > (a+1)/(a+b+2)`, at the
    midpoints."""
    var a_mid = a._midpoint
    var b_mid = b._midpoint
    if a_mid._exponent > 1000 or b_mid._exponent > 1000:
        return a_mid._exponent < b_mid._exponent
    var af = a_mid.to_native[DType.float64]()
    var bf = b_mid.to_native[DType.float64]()
    var xf = x._midpoint.to_native[DType.float64]()
    return xf * (af + bf + 2.0) > af + 1.0


def _betainc_enclosure(a: Ball, b: Ball, x: Ball, w: Int, beta: Optional[Ball] = None) raises -> Ball:
    """`I_x(a, b)` for balls a, b > 0 and 0 < x < 1, to about w bits; beta
    as in `_betainc_direct`, symmetric in a and b."""
    if _betainc_flipped(a, b, x):
        # 1 - x is exact for x in [1/2, 1) (Sterbenz), and the flipped I
        # is at most about 1/2, so 1 - I loses at most a bit.
        var rest = _sub(_one(w), x, w)
        return _sub(_one(w), _betainc_direct(b, a, rest, w, beta), w)
    return _betainc_direct(a, b, x, w, beta)


def _betainc_corner(a: Float, b: Float, x: Float, w: Int, beta: Optional[Ball] = None) raises -> Ball:
    """`I_x(a, b)` at an exact corner with scipy's values at its edges."""
    if x.is_zero():
        return Ball(Integer(0), precision=w)
    if x == Float(1):
        return Ball(Integer(1), precision=w)
    if a.is_zero():
        return Ball(Integer(1), precision=w)
    if b.is_zero():
        return Ball(Integer(0), precision=w)
    return _betainc_enclosure(Ball(a), Ball(b), Ball(x), w, beta)


def _betainc_ball(a: _BallArgument, b: _BallArgument, x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """`I_x(a, b)` of three balls: its values at the corners where it is
    least and greatest; indeterminate where a ball leaves the domain or both
    a and b reach 0."""
    var w = _working(context, a.precision, b.precision, x.precision)
    if a.kind != _FINITE or b.kind != _FINITE or x.kind != _FINITE:
        return Ball.indeterminate(w)
    var ab = a.ball()
    var bb = b.ball()
    var xb = x.ball()
    var a_low = ab._exact_lower()
    var b_low = bb._exact_lower()
    var x_low = xb._exact_lower()
    var x_high = xb._exact_upper()
    var zero = Float(0)
    if a_low < zero or b_low < zero or x_low < zero or x_high > Float(1):
        return Ball.indeterminate(w)
    if a_low.is_zero() and b_low.is_zero():
        return Ball.indeterminate(w)
    var work = w + _BALL_GUARD
    if ab.is_exact() and bb.is_exact() and xb.is_exact():
        var interior = not (a_low.is_zero() or b_low.is_zero() or x_low.is_zero() or x_low == Float(1))
        var ar = _moderate_value(ab)
        var br = _moderate_value(bb)
        var xr = _moderate_value(xb)
        if interior and ar and br and xr:
            var found = _betainc_case(ar.value(), br.value(), xr.value())
            if found.kind == _RATIONAL:
                return Ball(found.value, precision=w)
        return _rounded_ball(_BallArgument(_betainc_corner(a_low, b_low, x_low, work)), w)
    # I grows with x and b and falls with a. With a and b exact the corners
    # share `1/B(a, b)`, at the precision `_betainc_direct` takes.
    var beta = Optional[Ball](None)
    if ab.is_exact() and bb.is_exact() and not a_low.is_zero() and not b_low.is_zero():
        var inner = work
        beta = _mul(_mul(_ball_special(_GAMMA, _BallArgument(_add(ab, bb, inner)), 0, _c(inner)), _rgamma(ab, inner), inner), _rgamma(bb, inner), inner)
    var least = _betainc_corner(ab._exact_upper(), b_low, x_low, work, beta)
    var most = _betainc_corner(a_low, bb._exact_upper(), x_high, work, beta)
    if not least.is_finite() or not most.is_finite():
        return Ball.indeterminate(w)
    var hull = _hull(least._exact_lower(), most._exact_upper(), w)
    return _intersect_range(hull, Ball(Rational(1, 2), Rational(1, 2), precision=w), w)

