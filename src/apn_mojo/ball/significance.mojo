"""Support for significance arithmetic: precisions in digits and in bits,
the radius of a relative precision, and the propagation terms of the ball
functions, so that a caller tracking precision bounds errors the way the ball
functions do.

Mathematica's precision of a number is a ball radius on a logarithmic scale:
`d` digits of relative precision mean a radius of `|m| 10**-d`.
`radius_for_relative_digits` gives that radius rounded up, and
`bits_to_digits` and `digits_to_bits` convert precisions, as enclosures,
since `log2 10` is irrational. `propagation_bound[f](m, r)` is the term `P`
that a ball function `f` adds to its kernel's radius at the midpoint, in the
requirements' Appendix E.11.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from .context import BallContext
from .value import Ball, _BallArgument, _FINITE
from ._radius import _Radius
from ._arithmetic import _sum, _product, _quotient, _working
from ._constants import _log2_10, _ln2, _ln10
from ._functions import _ball_function
from ._kernels import _EXP, _EXPM1, _SIN, _COS, _SQRT


def _c(w: Int) raises -> Optional[BallContext]:
    return Optional[BallContext](BallContext(w))


def bits_to_digits(bits: Float, *, context: Optional[BallContext] = None) raises -> Ball:
    """Decimal digits of a precision in bits: `bits * log10(2)`, enclosed.

    Args:
        bits: The precision in bits.
        context: The result precision; 128 bits by default.

    Returns:
        A ball containing `bits / log2(10)`.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _working(context, 128)
    return _quotient(_BallArgument(Ball(bits)), _BallArgument(_log2_10(w + 8)), _c(w))


def digits_to_bits(digits: Float, *, context: Optional[BallContext] = None) raises -> Ball:
    """Bits of a precision in decimal digits: `digits * log2(10)`, enclosed.

    Args:
        digits: The precision in decimal digits.
        context: The result precision; 128 bits by default.

    Returns:
        A ball containing `digits * log2(10)`.

    Raises:
        Only on an invalid precision or a checked size error.
    """
    var w = _working(context, 128)
    return _product(_BallArgument(Ball(digits)), _BallArgument(_log2_10(w + 8)), _c(w))


def _upper(x: Ball) raises -> Float:
    """An upper bound of every point's magnitude, in the 30-bit radius format."""
    if not x.is_finite():
        raise Error("Cannot bound an unbounded or indeterminate value; the inputs are out of range.")
    return _Radius.upper(x._midpoint).add(x._radius).to_float()


def radius_for_relative_digits(midpoint: Float, digits: Float) raises -> Float:
    """The radius of `digits` decimal digits of relative precision,
    `|midpoint| * 10**-digits`, rounded up to the 30-bit radius format.

    Args:
        midpoint: A finite nonzero midpoint.
        digits: The relative precision in decimal digits, finite.

    Returns:
        An upper bound of `|midpoint| 10**-digits` with 30 bits.

    Raises:
        For a zero or nonfinite midpoint, whose relative precision has no
        radius, a nonfinite or huge `digits`.
    """
    if midpoint.is_zero() or not midpoint.is_finite():
        raise Error("A relative precision needs a finite nonzero midpoint; give 0 an absolute radius instead.")
    if not digits.is_finite() or digits._exponent > 60:
        raise Error("Cannot form a radius for that many digits; use a finite precision below 2**60 digits.")
    var w = 64
    var exponent = _product(_BallArgument(Ball(-digits)), _BallArgument(_log2_10(w + 8)), _c(w))
    var scale = _ball_function(_EXP, _BallArgument(_product(_BallArgument(exponent), _BallArgument(_ln2(w + 8)), _c(w))), 0, _c(w))
    return _upper(_product(_BallArgument(Ball(abs(midpoint))), _BallArgument(scale), _c(w)))


def propagation_bound[function: StaticString](midpoint: Float, radius: Float) raises -> Float:
    """The propagation term `P_f(m, r)` of the ball function `f` (Appendix
    E.11): the bound a ball `[m +/- r]` adds to the radius of `f`'s kernel at
    m, rounded up to the 30-bit radius format.

    | f | P |
    |---|---|
    | exp, expm1 | `e**m (e**r - 1)` |
    | exp2 | `2**m (2**r - 1)` |
    | log, log2, log10 | `r / (m - r)`, divided by `ln 2` or `ln 10` |
    | log1p | `r / (1 + m - r)` |
    | sin, cos | `r min(1, |cos m| + r)`, `r min(1, |sin m| + r)` |
    | atan | `r / (1 + d**2)`, `d = max(0, |m| - r)` |
    | tanh | `min(r, 2)` |
    | asinh | `r / sqrt(1 + d**2)` |
    | atanh | `r / (1 - (|m| + r)**2)` |

    Parameters:
        function: The function's name, one of the table's.

    Args:
        midpoint: The ball's finite midpoint.
        radius: The ball's radius, finite and nonnegative.

    Returns:
        An upper bound of the term, with 30 bits.

    Raises:
        For another name, a negative or nonfinite radius, a nonfinite
        midpoint, or a ball reaching outside the function's domain (`m <= r`
        for log, `1 + m <= r` for log1p, `|m| + r >= 1` for atanh).
    """
    if not midpoint.is_finite() or not radius.is_finite() or radius._negative and not radius.is_zero():
        raise Error("propagation_bound needs a finite midpoint and a finite nonnegative radius.")
    var w = 64
    var c = _c(w)
    var m = Ball(midpoint)
    var r = Ball(radius)
    var one = Ball(Integer(1), precision=w)
    var result: Ball
    if function == "exp" or function == "expm1" or function == "exp2":
        var x = m
        var s = r
        if function == "exp2":
            x = _product(_BallArgument(m), _BallArgument(_ln2(w + 8)), c)
            s = _product(_BallArgument(r), _BallArgument(_ln2(w + 8)), c)
        var size = _ball_function(_EXP, _BallArgument(x), 0, c)
        var growth = _ball_function(_EXPM1, _BallArgument(s), 0, c) if not radius.is_zero() else Ball(Integer(0), precision=w)
        result = _product(_BallArgument(size), _BallArgument(growth), c)
    elif function == "log" or function == "log2" or function == "log10":
        var gap = _sum(_BallArgument(m), _BallArgument(r), True, c)
        if not gap.certainly_positive():
            raise Error("log's propagation bound needs m > r: the ball must lie above 0.")
        result = _quotient(_BallArgument(r), _BallArgument(gap), c)
        if function == "log2":
            result = _quotient(_BallArgument(result), _BallArgument(_ln2(w + 8)), c)
        elif function == "log10":
            result = _quotient(_BallArgument(result), _BallArgument(_ln10(w + 8)), c)
    elif function == "log1p":
        var gap = _sum(_BallArgument(_sum(_BallArgument(one), _BallArgument(m), False, c)), _BallArgument(r), True, c)
        if not gap.certainly_positive():
            raise Error("log1p's propagation bound needs 1 + m > r: the ball must lie above -1.")
        result = _quotient(_BallArgument(r), _BallArgument(gap), c)
    elif function == "sin" or function == "cos":
        var other = _ball_function(_COS if function == "sin" else _SIN, _BallArgument(m), 0, c)
        var slope = _upper(_sum(_BallArgument(Ball(_upper(other))), _BallArgument(r), False, c))
        result = _product(_BallArgument(r), _BallArgument(Ball(slope if slope < Float(1) else Float(1))), c)
    elif function == "atan" or function == "asinh":
        # d is a lower bound of max(0, |m| - r), so the term only grows.
        var d = _sum(_BallArgument(Ball(abs(midpoint))), _BallArgument(r), True, c)._exact_lower()
        var square = _product(_BallArgument(Ball(d)), _BallArgument(Ball(d)), c) if d > Float(0) else Ball(Integer(0), precision=w)
        var denominator = _sum(_BallArgument(one), _BallArgument(square), False, c)
        if function == "asinh":
            denominator = _ball_function(_SQRT, _BallArgument(denominator), 0, c)
        result = _quotient(_BallArgument(r), _BallArgument(denominator), c)
    elif function == "tanh":
        result = Ball(radius if radius < Float(2) else Float(2))
    elif function == "atanh":
        var reach = _sum(_BallArgument(Ball(abs(midpoint))), _BallArgument(r), False, c)
        var gap = _sum(_BallArgument(one), _BallArgument(_product(_BallArgument(reach), _BallArgument(reach), c)), True, c)
        if not gap.certainly_positive():
            raise Error("atanh's propagation bound needs |m| + r < 1: the ball must lie inside (-1, 1).")
        result = _quotient(_BallArgument(r), _BallArgument(gap), c)
    else:
        raise Error(String(
            "propagation_bound has no term for ", function,
            "; use exp, expm1, exp2, log, log2, log10, log1p, sin, cos, atan, tanh, asinh or atanh.",
        ))
    return _upper(result)
