"""Special functions of balls, one declaration per name.

Each function returns a ball that contains the function's value at every
point of its input ball: from the kernel at the two exact ends for a monotone
function, from the midpoint widened by a bound of the derivative otherwise,
and from the monotone pieces around Gamma's minimum. A function undefined
somewhere in its input ball gives an indeterminate ball. A `context=` sets
the result precision, by default the operand's.
"""

from .context import BallContext
from .value import Ball, _BallArgument
from ._special import (
    _ball_special, _GAMMA, _LOG_GAMMA, _DIGAMMA, _ERF, _ERFC, _ERFI, _EI, _SI, _CI, _SHI, _CHI, _FRESNEL_S, _FRESNEL_C, _LAMBERT_W, _NDTR, _LOG_NDTR, _BETA, _LOG_BETA, _POCH, _ERFINV, _NDTRI,
)
from ._ratios import _ratio_ball
from ._zeta import _zeta_ball, _polygamma_ball
from ._hypergeometric import _hyp1f1_ball, _hyp2f1_ball
from ._incomplete import _gammainc_ball, _betainc_ball
from ..integer.value import Integer
from ._arithmetic import _negate


def gamma(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of euler's Gamma function.

    A ball containing 0 or a negative integer is indeterminate.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_GAMMA, value, 0, context)


def gammaln(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the logarithm of Gamma, for x > 0.

    A ball reaching 0 or below is indeterminate.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_LOG_GAMMA, value, 0, context)


def digamma(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the digamma function, `Gamma'/Gamma`.

    A ball containing 0 or a negative integer is indeterminate.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_DIGAMMA, value, 0, context)


def erf(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the error function.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_ERF, value, 0, context)


def erfc(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the complementary error function, `1 - erf`, without its cancellation.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_ERFC, value, 0, context)


def erfi(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the imaginary error function, `-i erf(ix)`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_ERFI, value, 0, context)


def expi(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the exponential integral Ei, the principal value for x < 0.

    A ball containing 0 is indeterminate.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_EI, value, 0, context)


def sici(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of the sine and cosine integrals `(Si(x), Ci(x))`; `Ci` is
    defined for x > 0.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        Two balls, each containing its function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return (_ball_special(_SI, value, 0, context), _ball_special(_CI, value, 0, context))


def shichi(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of the hyperbolic sine and cosine integrals
    `(Shi(x), Chi(x))`; `Chi` is defined for x > 0.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        Two balls, each containing its function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return (_ball_special(_SHI, value, 0, context), _ball_special(_CHI, value, 0, context))


def fresnel(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of Fresnel's integrals `(S(x), C(x))`,
    `int_0^x sin(pi t**2 / 2) dt` and `int_0^x cos(pi t**2 / 2) dt`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        Two balls, each containing its function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return (_ball_special(_FRESNEL_S, value, 0, context), _ball_special(_FRESNEL_C, value, 0, context))


def lambertw(value: _BallArgument, *, k: Int = 0, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of Lambert's W on branch 0 or -1.

    A ball reaching below -1/e, or for branch -1 reaching 0 or above, is
    indeterminate.

    Args:
        value: The operand, a ball or an exact number.
        k: The branch, 0 or -1.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing W at every point.

    Raises:
        For another branch, or on an invalid precision or a checked size error.
    """
    if k != 0 and k != -1:
        raise Error("lambertw has the branches k=0 and k=-1.")
    return _ball_special(_LAMBERT_W, value, k, context)


def ndtr(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the standard normal distribution function, `(1 + erf(x/sqrt 2)) / 2`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_NDTR, value, 0, context)


def log_ndtr(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the logarithm of the standard normal distribution function.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_LOG_NDTR, value, 0, context)


def beta(a: _BallArgument, b: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of Euler's beta function, `Gamma(a) Gamma(b) / Gamma(a + b)`, with scipy's values at the poles.

    Args:
        a: The first argument, a ball or an exact number.
        b: The second argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points;
        indeterminate where it is infinite.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ratio_ball(_BETA, a, b, context)


def betaln(a: _BallArgument, b: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log |B(a, b)|`.

    Args:
        a: The first argument, a ball or an exact number.
        b: The second argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points;
        indeterminate where it is infinite.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ratio_ball(_LOG_BETA, a, b, context)


def poch(z: _BallArgument, m: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the Pochhammer symbol `Gamma(z + m) / Gamma(z)`, with scipy's values at the poles.

    Args:
        z: The first argument, a ball or an exact number.
        m: The second argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points;
        indeterminate where it is infinite.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ratio_ball(_POCH, z, m, context)


def erfinv(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the inverse error function, for a ball inside `(-1, 1)`.

    Args:
        value: The operand, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point; indeterminate unless
        the ball lies inside `(-1, 1)`.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_ERFINV, value, 0, context)


def ndtri(value: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the inverse standard normal distribution function, for a
    ball inside `(0, 1)`.

    Args:
        value: The probability, a ball or an exact number.
        context: The result precision and budget; by default the operand's
            precision.

    Returns:
        A ball containing the function at every point; indeterminate unless
        the ball lies inside `(0, 1)`.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _ball_special(_NDTRI, value, 0, context)


def zeta(x: _BallArgument, q: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the Hurwitz zeta function; `zeta(x, 1)` is the Riemann
    zeta function.

    Args:
        x: The exponent, a ball or an exact number.
        q: The shift, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points; indeterminate
        at the poles and outside the domain.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _zeta_ball(x, q, context)


def polygamma(n: Integer, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the polygamma function of order n.

    Args:
        n: The order, a non-negative integer.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the argument's
            precision.

    Returns:
        A ball containing the function at every point; indeterminate at the
        poles and for n < 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _polygamma_ball(n, x, context)


def hyp1f1(a: _BallArgument, b: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of Kummer's confluent hypergeometric function `M(a, b, x)`.

    Args:
        a: The upper parameter, a ball or an exact number.
        b: The lower parameter, a ball or an exact number.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every triple of points;
        indeterminate where b may be a non-positive integer.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _hyp1f1_ball(a, b, x, context)


def gammainc(a: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the regularized lower incomplete gamma function `P(a, x)`, from its values at the corners of the balls,
    since it is monotone in each argument.

    Args:
        a: The shape, a ball or an exact number.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points;
        indeterminate where a ball reaches below 0 or both reach 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _gammainc_ball(a, x, False, context)


def gammaincc(a: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the regularized upper incomplete gamma function `Q(a, x)`, from its values at the corners of the balls,
    since it is monotone in each argument.

    Args:
        a: The shape, a ball or an exact number.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every pair of points;
        indeterminate where a ball reaches below 0 or both reach 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _gammainc_ball(a, x, True, context)


def hyp2f1(a: _BallArgument, b: _BallArgument, c: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of Gauss's hypergeometric function `F(a, b; c; x)`.

    Args:
        a: The first upper parameter, a ball or an exact number.
        b: The second upper parameter, a ball or an exact number.
        c: The lower parameter, a ball or an exact number.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every point of the balls;
        indeterminate where c may be a pole, and where x reaches 1 other than
        Gauss's exact value at 1, or beyond.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _hyp2f1_ball(a, b, c, x, context)


def betainc(a: _BallArgument, b: _BallArgument, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the regularized incomplete beta function `I_x(a, b)`, from
    its values at the corners of the balls, since it is monotone in each
    argument.

    Args:
        a: The first shape, a ball or an exact number.
        b: The second shape, a ball or an exact number.
        x: The argument, a ball or an exact number.
        context: The result precision and budget; by default the arguments'
            largest precision.

    Returns:
        A ball containing the function at every point of the balls;
        indeterminate where a ball leaves the domain or both shapes reach 0.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    return _betainc_ball(a, b, x, context)

