"""Special functions of Floats, correctly rounded, one declaration per name.

Each function returns the exact value rounded once in the context's mode,
certified the way the elementary functions are. Special values are the
functions' limits: poles signal divide-by-zero, and arguments outside the
domain give NaN, which is invalid.
"""

from .value import Float
from .context import ArithmeticContext
from ._arithmetic import _FloatArgument
from ._special import _special_rounded, _ratio_rounded, _zeta_rounded, _polygamma_rounded, _hyp1f1_rounded, _gammainc_rounded, _hyp2f1_rounded, _betainc_rounded
from ..integer.value import Integer
from ..ball._special import (
    _GAMMA, _LOG_GAMMA, _DIGAMMA, _ERF, _ERFC, _ERFI, _EI, _SI, _CI, _SHI, _CHI, _FRESNEL_S, _FRESNEL_C, _LAMBERT_W,
    _NDTR, _LOG_NDTR, _BETA, _LOG_BETA, _POCH, _ERFINV, _NDTRI,
)


def gamma(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Euler's Gamma function, correctly rounded.

    `gamma(+-0)` is `+-inf` with divide-by-zero, a negative integer or `-inf` gives NaN, which is invalid, and `gamma(n)` is `(n-1)!` rounded once.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_GAMMA, value, context))


def gammaln(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The logarithm of the absolute value of Gamma, `log |Gamma(x)|`,
    correctly rounded (scipy's `gammaln`).

    `gammaln(1)` and `gammaln(2)` are exactly +0; at 0, the negative integers
    and both infinities it is `+inf`, with divide-by-zero at the poles.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_LOG_GAMMA, value, context))


def digamma(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The digamma function, `Gamma'/Gamma`, correctly rounded.

    `digamma(+-0)` is `-+inf` with divide-by-zero, and a negative integer or `-inf` gives NaN, which is invalid.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_DIGAMMA, value, context))


def erf(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The error function, correctly rounded.

    `erf(+-0)` is `+-0` and `erf(+-inf)` is `+-1`.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_ERF, value, context))


def erfc(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The complementary error function, `1 - erf`, without its cancellation, correctly rounded.

    `erfc(+-0)` is 1, `erfc(+inf)` is +0 and `erfc(-inf)` is 2.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_ERFC, value, context))


def erfi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The imaginary error function, `-i erf(ix)`, correctly rounded.

    `erfi(+-0)` is `+-0` and `erfi(+-inf)` is `+-inf`.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_ERFI, value, context))


def expi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The exponential integral Ei, the principal value for x < 0, correctly rounded.

    `Ei(+-0)` is `-inf` with divide-by-zero, `Ei(+inf)` is `+inf` and `Ei(-inf)` is -0.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_EI, value, context))


def sici(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """The sine and cosine integrals `(Si(x), Ci(x))`, each correctly rounded
    (scipy's `sici`).

    `Si(x) = int_0^x sin(t)/t dt` and `Ci(x) = gamma + log x + int_0^x
    (cos(t) - 1)/t dt` for x > 0. `Si(+-0)` is `+-0` and `Si(+-inf)` is
    `+-pi/2`, correctly rounded; `Ci(+-0)` is `-inf` with divide-by-zero,
    `Ci(+inf)` is +0, and for a negative argument `Ci` is NaN, which is
    invalid.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        `(Si(x), Ci(x))`, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return (Float(_rounded=_special_rounded(_SI, value, context)), Float(_rounded=_special_rounded(_CI, value, context)))


def shichi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """The hyperbolic sine and cosine integrals `(Shi(x), Chi(x))`, each
    correctly rounded (scipy's `shichi`).

    `Shi(x) = int_0^x sinh(t)/t dt` and `Chi(x) = gamma + log x + int_0^x
    (cosh(t) - 1)/t dt` for x > 0. `Shi(+-0)` is `+-0` and `Shi(+-inf)` is
    `+-inf`; `Chi(+-0)` is `-inf` with divide-by-zero, `Chi(+inf)` is `+inf`,
    and for a negative argument `Chi` is NaN, which is invalid.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        `(Shi(x), Chi(x))`, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return (Float(_rounded=_special_rounded(_SHI, value, context)), Float(_rounded=_special_rounded(_CHI, value, context)))


def fresnel(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """Fresnel's integrals `(S(x), C(x))`, each correctly rounded (scipy's
    `fresnel`).

    `S(x) = int_0^x sin(pi t**2 / 2) dt` and `C(x) = int_0^x cos(pi t**2 / 2)
    dt`. Both are `+-0` at `+-0` and `+-1/2` at `+-inf`.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        `(S(x), C(x))`, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return (Float(_rounded=_special_rounded(_FRESNEL_S, value, context)), Float(_rounded=_special_rounded(_FRESNEL_C, value, context)))


def lambertw(value: _FloatArgument, *, k: Int = 0, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Lambert's W, the solution w of `w e**w = x`, correctly rounded.

    Branch 0 is the principal branch, defined for `x >= -1/e` with
    `W >= -1`; branch -1 is defined for `-1/e <= x < 0` with `W <= -1`.
    `W_0(+-0)` is `+-0` and `W_0(+inf)` is `+inf`; `W_-1(+-0)` is `-inf` with
    divide-by-zero. An argument outside the branch's domain gives NaN, which
    is invalid.

    Args:
        value: The operand; exact operands need a context.
        k: The branch, 0 or -1.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        For another branch, on a trapped condition, past the budget, or for
        exact operands without a context.
    """
    return Float(_rounded=_special_rounded(_LAMBERT_W, value, context, k))


def ndtr(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The standard normal distribution function, `(1 + erf(x/sqrt 2)) / 2`,
    correctly rounded (scipy's `ndtr`).

    `ndtr(+-0)` is exactly 1/2, `ndtr(+inf)` is 1 and `ndtr(-inf)` is +0.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_NDTR, value, context))


def log_ndtr(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The logarithm of the standard normal distribution function, correctly
    rounded (scipy's `log_ndtr`), accurate where `ndtr` is near 0 or 1.

    `log_ndtr(+inf)` is +0 and `log_ndtr(-inf)` is `-inf`.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_LOG_NDTR, value, context))


def beta(a: _FloatArgument, b: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Euler's beta function `B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b)`,
    correctly rounded (scipy's `beta`).

    Rational values, when both arguments are integers or one is a positive
    integer, are exact before the one rounding. At the poles it takes scipy's
    values: for a non-positive integer a, `(-1)**b B(1 - a - b, b)` when b is
    an integer with `1 - a - b > 0` and `+inf` with divide-by-zero otherwise;
    +0 where only `a + b` is a non-positive integer. `B(+inf, b)` is +0 for
    b > 0.

    Args:
        a: The first argument; exact arguments need a context.
        b: The second argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_ratio_rounded(_BETA, a, b, context))


def betaln(a: _FloatArgument, b: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The logarithm of the absolute value of Euler's beta function,
    `log |B(a, b)|`, correctly rounded (scipy's `betaln`).

    At the poles it is `+inf` where `B(a, b)` is infinite and `-inf` where it
    is 0, each with divide-by-zero, and exactly +0 where `|B(a, b)| = 1`.

    Args:
        a: The first argument; exact arguments need a context.
        b: The second argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_ratio_rounded(_LOG_BETA, a, b, context))


def poch(z: _FloatArgument, m: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Pochhammer symbol `(z)_m = Gamma(z + m) / Gamma(z)`, correctly
    rounded (scipy's `poch`).

    For an integer m it is the exact product `z (z+1) ... (z+m-1)`, or
    `1 / ((z-1) ... (z+m))` for m < 0, before the one rounding. It is `+inf`
    with divide-by-zero where a factor of that divisor is 0 or, for a
    non-integer m, where only `z + m` is a non-positive integer, and +0 where
    only z is. `(+inf)_m` is +inf, 1 or +0 as m is above, at or below 0.

    Args:
        z: The first argument; exact arguments need a context.
        m: The second argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_ratio_rounded(_POCH, z, m, context))


def erfinv(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The inverse error function, the t with `erf(t) = x` for `-1 < x < 1`,
    correctly rounded (scipy's `erfinv`).

    `erfinv(+-0)` is `+-0` and `erfinv(+-1)` is `+-inf` with divide-by-zero;
    beyond, and at the infinities, it is NaN, which is invalid.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_ERFINV, value, context))


def ndtri(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The inverse of the standard normal distribution function, the x with
    `ndtr(x) = p` for `0 < p < 1`, correctly rounded (scipy's `ndtri`).

    `ndtri(1/2)` is exactly +0, `ndtri(0)` is `-inf` and `ndtri(1)` is `+inf`,
    each with divide-by-zero; outside `[0, 1]` it is NaN, which is invalid.

    Args:
        value: The probability; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_special_rounded(_NDTRI, value, context))


def zeta(x: _FloatArgument, q: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Hurwitz zeta function `zeta(x, q) = sum_{k>=0} (q + k)**-x`,
    correctly rounded (scipy's `zeta`); `zeta(x, 1)` is the Riemann zeta
    function at every x other than 1, its values at the non-positive integers
    exact.

    It is defined for x > 1 (every x other than 1 when q = 1), takes a negative
    q only for an integer x, and is `+inf` with divide-by-zero at x = 1 and at
    the non-positive integers q; elsewhere it is NaN, which is invalid.

    Args:
        x: The exponent; exact arguments need a context.
        q: The shift.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_zeta_rounded(x, q, context))


def polygamma(n: Integer, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The polygamma function `psi^(n)(x) = (-1)**(n+1) n! zeta(n+1, x)`,
    correctly rounded (scipy's `polygamma`); digamma for n = 0.

    At 0 and the negative integers it is `(-1)**(n+1) inf` with divide-by-zero
    for n >= 1, as scipy's; for n < 0 it is NaN, which is invalid.

    Args:
        n: The order, a non-negative integer.
        x: The argument; exact arguments need a context.
        context: The output format, rounding mode, traps and budget; by
            default the argument's format, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_polygamma_rounded(n, x, context))


def hyp1f1(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Kummer's confluent hypergeometric function
    `M(a, b, x) = sum_k (a)_k / (b)_k x**k / k!`, correctly rounded (scipy's
    `hyp1f1`).

    At a non-positive integer b it is `+inf` with divide-by-zero, except
    that a negative integer `a >= b` ends the series before the pole, so
    that `hyp1f1(-n, -n, x)` is the truncated exponential, as scipy's. An
    infinite a gives NaN, which is invalid, and an infinite b gives 1. At an
    infinite x it is the limit. Rational values, such as the polynomials of
    a non-positive integer a, are exact.

    Args:
        a: The upper parameter; exact arguments need a context.
        b: The lower parameter.
        x: The argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_hyp1f1_rounded(a, b, x, context))


def gammainc(a: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized lower incomplete gamma function
    `P(a, x) = gamma(a, x) / Gamma(a)`, correctly rounded (scipy's
    `gammainc`).

    For a < 0 or x < 0, and at a = x = 0, it is NaN, which is invalid.
    `P(0, x) = 1` for x > 0, `P(a, 0) = 0`, `P(+inf, x) = 0` and
    `P(a, +inf) = 1`, as scipy's.

    Args:
        a: The shape; exact arguments need a context.
        x: The argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_gammainc_rounded(a, x, False, context))


def gammaincc(a: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized upper incomplete gamma function
    `Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)`, correctly rounded
    (scipy's `gammaincc`), computed without cancellation where Q is small.

    For a < 0 or x < 0, and at a = x = 0, it is NaN, which is invalid.
    `Q(0, x) = 0` for x > 0, `Q(a, 0) = 1`, `Q(+inf, x) = 1` and
    `Q(a, +inf) = 0`, as scipy's.

    Args:
        a: The shape; exact arguments need a context.
        x: The argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_gammainc_rounded(a, x, True, context))


def hyp2f1(a: _FloatArgument, b: _FloatArgument, c: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Gauss's hypergeometric function
    `F(a, b; c; x) = sum_k (a)_k (b)_k / (c)_k x**k / k!`, correctly rounded
    (scipy's `hyp2f1`), for real x <= 1 and, where it is a polynomial, every
    x.

    A non-positive integer a or b gives a polynomial, exact for rational
    arguments. At a non-positive integer c it is `+inf` with divide-by-zero
    unless the polynomial ends first; at x = 1 it is Gauss's
    `Gamma(c) Gamma(c-a-b) / (Gamma(c-a) Gamma(c-b))` for `c - a - b > 0`
    and `+inf` otherwise, as scipy's. For x > 1, where the function is not
    real, it is NaN, which is invalid (scipy returns `+inf`). Rational
    values, such as `(1-x)**-b` for c = a at a perfect power, are exact.

    Args:
        a: The first upper parameter; exact arguments need a context.
        b: The second upper parameter.
        c: The lower parameter.
        x: The argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_hyp2f1_rounded(a, b, c, x, context))


def betainc(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized incomplete beta function `I_x(a, b) = B_x(a, b) /
    B(a, b)`, correctly rounded (scipy's `betainc`).

    For a < 0, b < 0 or x outside [0, 1], and at a = b = 0 and a = b = +inf,
    it is NaN, which is invalid. As scipy's, it is 1 for x > 0 where a = 0 or
    b = +inf, 0 for x < 1 where b = 0 or a = +inf, 0 at x = 0 and 1 at
    x = 1. Integer parameters give exact rational values.

    Args:
        a: The first shape; exact arguments need a context.
        b: The second shape.
        x: The argument.
        context: The output format, rounding mode, traps and budget; by
            default the merged argument formats, rounded to nearest-even.

    Returns:
        The value rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact arguments
        without a context.
    """
    return Float(_rounded=_betainc_rounded(a, b, x, context))

