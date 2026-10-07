"""Correctly rounded elementary functions of Complex numbers, one declaration per name.

Each part of a result is the exact value's part rounded once in its own
context. Special values follow C99 Annex G: NaN
propagates, a real argument keeps a structural zero imaginary part with the
sign the formulas give, and on a branch cut the sign of the zero part selects
the side. Without a context each part keeps its own format, rounded to
nearest-even; a `ComplexContext` sets each part's format, rounding mode,
traps and budget, or one `ArithmeticContext` sets both.
"""

from .value import Complex
from .context import _ComplexContextArgument
from ..float.value import Float
from ..float.context import ArithmeticContext
from ..float._elementary import _atan2_rounded
from ._elementary import _complex_function, _complex_pow
from ..ball._kernels import (
    _EXP, _LOG, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS, _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH,
)


def exp(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The exponential, each part correctly rounded.

    `exp(x + iy) = e**x (cos y + i sin y)`; a real argument keeps its imaginary zero's sign.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_EXP, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def log(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal logarithm, each part correctly rounded.

    The imaginary part is in `[-pi, pi]`; on the negative real axis its sign follows the imaginary zero's, so `log(-1 + 0i)` is `i pi` and `log(-1 - 0i)` is `-i pi`. `log(0)` is `-inf` with divide-by-zero.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_LOG, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def sin(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The sine, each part correctly rounded.

    `sin z = -i sinh(iz)`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_SIN, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def cos(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The cosine, each part correctly rounded.

    `cos z = cosh(iz)`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_COS, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def tan(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The tangent, each part correctly rounded.

    `tan z = -i tanh(iz)`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_TAN, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def sinh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The hyperbolic sine, each part correctly rounded.

    `sinh(x + iy) = sinh x cos y + i cosh x sin y`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_SINH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def cosh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The hyperbolic cosine, each part correctly rounded.

    `cosh(x + iy) = cosh x cos y + i sinh x sin y`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_COSH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def tanh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The hyperbolic tangent, each part correctly rounded.

    `tanh z = sinh z / cosh z`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_TANH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def asin(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal arcsine, each part correctly rounded.

    On the cuts `(-inf, -1)` and `(1, inf)` the imaginary zero's sign selects the side: `asin(2 + 0i)` is `pi/2 + 1.317i`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ASIN, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def acos(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal arccosine, each part correctly rounded.

    On the cuts the imaginary zero's sign selects the side: `acos(2 - 0i)` is `+0 + 1.317i`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ACOS, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def atan(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal arctangent, each part correctly rounded.

    On the cuts of the imaginary axis the real zero's sign selects the side: `atan(+0 + 2i)` is `pi/2 + 0.549i`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ATAN, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def asinh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal inverse hyperbolic sine, each part correctly rounded.

    On the cuts of the imaginary axis the real zero's sign selects the side.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ASINH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def acosh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal inverse hyperbolic cosine, each part correctly rounded.

    On the cut `(-inf, 1)` the imaginary zero's sign selects the side: `acosh(-2 + 0i)` is `1.317 + i pi`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ACOSH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def atanh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal inverse hyperbolic tangent, each part correctly rounded.

    `atanh(+-1)` has an infinite real part with divide-by-zero; on the cuts the imaginary zero's sign selects the side.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default each part's format, rounded to nearest-even.

    Returns:
        The value, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_function(_ATANH, value, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def pow(base: Complex, exponent: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The principal power `base**exponent`, each part correctly rounded.

    A zero exponent gives `1 + 0i`, an integral real exponent is an integer
    power, and otherwise the power is `exp(exponent log base)` with the
    principal logarithm. Powers that are exact, such as `(-4)**0.25 = 1 + i`
    and `(-2)**0.5 = +0 + sqrt(2) i`, come out exactly.

    Args:
        base: The base.
        exponent: The exponent.
        context: An `ArithmeticContext` for both parts or a `ComplexContext`
            for each; by default the base's part formats.

    Returns:
        The power, each part rounded once.

    Raises:
        On a trapped condition in either part, or past the budget.
    """
    var parts = _complex_pow(base, exponent, context)
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def angle(value: Complex, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The argument, `atan2(imag, real)`, correctly rounded, in `[-pi, pi]`.

    Args:
        value: The Complex.
        context: The output format, rounding mode, traps and budget; by
            default the larger part format, rounded to nearest-even.

    Returns:
        The angle rounded once; `arg(-1 + 0i)` is `pi` and `arg(-1 - 0i)`
        is `-pi`.

    Raises:
        On a trapped condition, or past the budget.
    """
    return Float(_rounded=_atan2_rounded(value._imag, value._real, context))
