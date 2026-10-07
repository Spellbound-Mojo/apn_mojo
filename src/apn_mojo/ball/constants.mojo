"""Mathematical constants as canonical balls.

The canonical ball of a constant `c` at precision `p` is
`[round_down(c, p), round_up(c, p)]`: its midpoint is the exact mean of the two
ends, at `p + 1` bits, and its radius their exact half-width. It depends only
on `c` and `p`, so a caller that keeps the widest ball it has computed can
serve every narrower request with `canonical(cached, p)` and get the same
ball as a fresh computation. When `canonical` cannot decide both ends, compute
`pi_ball(2 * p)` and keep that instead.
"""

from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from .value import Ball
from ._arithmetic import _hull
from ._certified import _round_certified
from ._kernels import _RealKernel, _PI, _EULER_E, _LN2, _LOG2_10, _EULER_GAMMA, _CATALAN


def _canonical_constant(code: Int, precision: Int) raises -> Ball:
    var kernel = _RealKernel.constant(code)
    var format = FloatFormat(precision)
    var low = Float(_rounded=_round_certified(kernel, ArithmeticContext(format=format, rounding=RoundingMode.toward_negative)))
    var high = Float(_rounded=_round_certified(kernel, ArithmeticContext(format=format, rounding=RoundingMode.toward_positive)))
    return _hull(low, high, precision + 1)


def pi_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of pi.

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(pi, p), round_up(pi, p)]`, with a midpoint of `p + 1` bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_PI, precision)


def euler_e_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of e.

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(e, p), round_up(e, p)]`, with a midpoint of `p + 1` bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_EULER_E, precision)


def ln2_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of ln 2.

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(ln 2, p), round_up(ln 2, p)]`, with a midpoint of `p + 1`
        bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_LN2, precision)


def log2_10_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of log2(10).

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(log2 10, p), round_up(log2 10, p)]`, with a midpoint of
        `p + 1` bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_LOG2_10, precision)


def euler_gamma_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of Euler's gamma.

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(gamma, p), round_up(gamma, p)]`, with a midpoint of
        `p + 1` bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_EULER_GAMMA, precision)


def catalan_ball(precision: Int = 128) raises -> Ball:
    """The canonical ball of Catalan's constant.

    Args:
        precision: The precision `p` of the two ends, in bits.

    Returns:
        `[round_down(G, p), round_up(G, p)]`, with a midpoint of `p + 1` bits.

    Raises:
        When the precision is invalid.
    """
    return _canonical_constant(_CATALAN, precision)
