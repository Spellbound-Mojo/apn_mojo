"""Correctly rounded elementary functions of Floats, one declaration per name.

Each function computes its exact result and rounds it once, in every rounding
mode. Without a context the result takes the operand's format, rounded to
nearest-even; exact operands (Integer, Rational, native numbers and literals)
need a context, as for `sqrt`, and are never rounded first. Special values
follow C99 Annex F: NaN propagates without a flag, poles signal
divide-by-zero, and arguments outside the domain give NaN, which is invalid.
The only exact results are the trivial ones, such as `exp(0) = 1`,
`log2(2**k) = k` and `pow(16, 0.75) = 8`.

The working precision rises until the rounding is certain, bounded by the
context's `max_precision`; a function that would need more raises an error
naming it. Apply a function to batches with `vmap`, as in
`vmap[apn_mojo.float.exp]()(values, context=c)`.
"""

from ..integer.value import Integer
from .value import Float
from .context import ArithmeticContext
from ._arithmetic import _FloatArgument
from ._functions import _rootn_float
from ._elementary import _real_rounded, _atan2_rounded, _pow_rounded
from ..ball._kernels import (
    _EXP, _EXPM1, _EXP2, _LOG, _LOG1P, _LOG2, _LOG10, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS,
    _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH,
)


def exp(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded exponential.

    `exp(+-0)` is exactly 1, `exp(-inf)` is `+0`, and results beyond the format overflow or underflow.

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
    return Float(_rounded=_real_rounded(_EXP, value, context))


def expm1(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded `exp(x) - 1`, accurate near 0.

    `expm1(+-0)` is `+-0` and `expm1(-inf)` is exactly -1.

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
    return Float(_rounded=_real_rounded(_EXPM1, value, context))


def exp2(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded `2**x`.

    An integral `x` gives the exact power of 2.

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
    return Float(_rounded=_real_rounded(_EXP2, value, context))


def log(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded natural logarithm.

    `log(+-0)` is `-inf` with divide-by-zero, `log(1)` is `+0`, and a negative argument gives NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_LOG, value, context))


def log1p(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded `log(1 + x)`, accurate near 0.

    `log1p(-1)` is `-inf` with divide-by-zero; below -1 the result is NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_LOG1P, value, context))


def log2(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded base-2 logarithm.

    A power of 2 gives its exact exponent; zero and negative arguments are as for `log`.

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
    return Float(_rounded=_real_rounded(_LOG2, value, context))


def log10(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded base-10 logarithm.

    A power of 10 gives its exact exponent; zero and negative arguments are as for `log`.

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
    return Float(_rounded=_real_rounded(_LOG10, value, context))


def sin(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded sine.

    `sin(+-0)` is `+-0` and an infinity gives NaN, which is invalid. A huge argument needs about as many bits of pi as its exponent; past the context's budget the function raises.

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
    return Float(_rounded=_real_rounded(_SIN, value, context))


def cos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded cosine.

    `cos(+-0)` is exactly 1; infinities and the budget are as for `sin`.

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
    return Float(_rounded=_real_rounded(_COS, value, context))


def tan(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded tangent.

    `tan(+-0)` is `+-0`; infinities and the budget are as for `sin`.

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
    return Float(_rounded=_real_rounded(_TAN, value, context))


def atan(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded arctangent, in `[-pi/2, pi/2]`.

    `atan(+-inf)` is `+-pi/2` rounded.

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
    return Float(_rounded=_real_rounded(_ATAN, value, context))


def asin(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded arcsine, in `[-pi/2, pi/2]`.

    Outside `[-1, 1]` the result is NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_ASIN, value, context))


def acos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded arccosine, in `[0, pi]`.

    `acos(1)` is `+0`; outside `[-1, 1]` the result is NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_ACOS, value, context))


def sinh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded hyperbolic sine.

    `sinh(+-0)` is `+-0` and `sinh(+-inf)` is `+-inf`.

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
    return Float(_rounded=_real_rounded(_SINH, value, context))


def cosh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded hyperbolic cosine.

    `cosh(+-0)` is exactly 1 and `cosh(+-inf)` is `+inf`.

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
    return Float(_rounded=_real_rounded(_COSH, value, context))


def tanh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded hyperbolic tangent.

    `tanh(+-inf)` is exactly `+-1`.

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
    return Float(_rounded=_real_rounded(_TANH, value, context))


def asinh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded inverse hyperbolic sine.

    `asinh(+-0)` is `+-0` and `asinh(+-inf)` is `+-inf`.

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
    return Float(_rounded=_real_rounded(_ASINH, value, context))


def acosh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded inverse hyperbolic cosine.

    `acosh(1)` is `+0`; below 1 the result is NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_ACOSH, value, context))


def atanh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded inverse hyperbolic tangent.

    `atanh(+-1)` is `+-inf` with divide-by-zero; outside `[-1, 1]` the result is NaN, which is invalid.

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
    return Float(_rounded=_real_rounded(_ATANH, value, context))


def sin_cos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """The correctly rounded sine and cosine, each rounded independently.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode, traps and budget; by
            default the operand's format, rounded to nearest-even.

    Returns:
        `(sin(value), cos(value))`.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return (Float(_rounded=_real_rounded(_SIN, value, context)), Float(_rounded=_real_rounded(_COS, value, context)))


def atan2(y: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded angle of the point `(x, y)`, in `[-pi, pi]`.

    The special values are those of C99: `atan2(+-0, -0)` is `+-pi`,
    `atan2(+-0, +0)` is `+-0`, and infinite operands give multiples of `pi/4`.

    Args:
        y: The ordinate.
        x: The abscissa.
        context: The output format, rounding mode, traps and budget; by
            default the merged operand formats, rounded to nearest-even.

    Returns:
        The angle rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_atan2_rounded(y, x, context))


def pow(base: _FloatArgument, exponent: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded power `base**exponent`.

    The special values are those of C99 F.9.4.4: a zero exponent gives 1 for
    any base, a base of 1 gives 1 for any exponent, an integral exponent is an
    integer power, and a negative base with a non-integral exponent is
    invalid. A binary-fraction exponent can give an exact result, as in
    `pow(16, 0.75) = 8`.

    Args:
        base: The base.
        exponent: The exponent.
        context: The output format, rounding mode, traps and budget; by
            default the merged operand formats, rounded to nearest-even.

    Returns:
        The power rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact operands without
        a context.
    """
    return Float(_rounded=_pow_rounded(base, exponent, context))


def rootn(value: _FloatArgument, n: Int, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded real `n`-th root, IEEE 754 `rootn`.

    An odd root keeps the sign, an even root of a negative value is NaN, which
    is invalid, and a zero's root is that zero (`+0` for an even `n`). A root
    that is a binary fraction is exact.

    Args:
        value: The operand; exact operands need a context.
        n: The degree, at least 1.
        context: The output format, rounding mode and traps; by default the
            operand's format, rounded to nearest-even.

    Returns:
        The root rounded once.

    Raises:
        When `n` is below 1 (use `pow`), on a trapped condition, or for exact
        operands without a context.
    """
    return Float(_rounded=_rootn_float(value, n, context))
