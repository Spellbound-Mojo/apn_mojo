"""Mathematical constants as correctly rounded Floats.

Each constant is computed for the call, at a working precision raised until
its rounding is certain, then rounded once; nothing is cached. Without a
context the result has 128 bits, rounded to nearest-even, as Float
construction does. For an enclosing ball, use `apn_mojo.ball.pi_ball` and its
companions.
"""

from .value import Float
from .context import ArithmeticContext
from ._elementary import _constant_rounded
from ..ball._kernels import _PI, _EULER_E, _LN2, _LOG2_10, _EULER_GAMMA, _CATALAN


def pi(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The constant pi, correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        The constant pi, rounded once.

    Raises:
        On a trapped condition (pi is never exact, so `trap_inexact` always
        raises), or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_PI, context))


def euler_e(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Euler's number e = exp(1), correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        The constant e, rounded once.

    Raises:
        On a trapped condition, or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_EULER_E, context))


def ln2(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The natural logarithm of 2, correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        The constant ln 2, rounded once.

    Raises:
        On a trapped condition, or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_LN2, context))


def log2_10(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The base-2 logarithm of 10, correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        The constant log2(10), rounded once.

    Raises:
        On a trapped condition, or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_LOG2_10, context))


def euler_gamma(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Euler-Mascheroni constant gamma = 0.5772..., correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        The constant gamma, rounded once.

    Raises:
        On a trapped condition, or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_EULER_GAMMA, context))


def catalan(*, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Catalan's constant G = 0.9159..., correctly rounded.

    Args:
        context: The format, rounding mode, traps and budget; by default 128
            bits, rounded to nearest-even.

    Returns:
        G rounded once.

    Raises:
        On a trapped condition, or in an exact working format.
    """
    return Float(_rounded=_constant_rounded(_CATALAN, context))
