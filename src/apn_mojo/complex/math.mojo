"""Complex arithmetic and functions with an explicit numerical context.

Each name is one declaration, so vmap[add]() maps it over Complex batches.
Arguments accept Complex values and any exact real number.
"""

from .value import Complex
from ..common._stable_hash import _StableHash
from .context import _ComplexContextArgument
from ._input import _ComplexArgument
from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext


def add(
    left: _ComplexArgument,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The add `left + right`, each component rounded once.

    Either operand may be a Complex, Float, Integer, Rational, integer literal or
    typed native number; real operands keep real semantics, so a real zero never
    changes the other operand's imaginary zero sign.

    Args:
        left: The first operand.
        right: The second operand.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default the merged component formats, rounded to nearest-even.

    Returns:
        The Complex result; both components are published together.

    Raises:
        When a trapped condition occurs in either component.
    """
    return Complex._calculate(left, right, 0, context)


def subtract(
    left: _ComplexArgument,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The subtract `left - right`, each component rounded once.

    Either operand may be a Complex, Float, Integer, Rational, integer literal or
    typed native number; real operands keep real semantics, so a real zero never
    changes the other operand's imaginary zero sign.

    Args:
        left: The first operand.
        right: The second operand.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default the merged component formats, rounded to nearest-even.

    Returns:
        The Complex result; both components are published together.

    Raises:
        When a trapped condition occurs in either component.
    """
    return Complex._calculate(left, right, 1, context)


def multiply(
    left: _ComplexArgument,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The multiply `left * right`, each component rounded once.

    Either operand may be a Complex, Float, Integer, Rational, integer literal or
    typed native number; real operands keep real semantics, so a real zero never
    changes the other operand's imaginary zero sign.

    Args:
        left: The first operand.
        right: The second operand.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default the merged component formats, rounded to nearest-even.

    Returns:
        The Complex result; both components are published together.

    Raises:
        When a trapped condition occurs in either component.
    """
    return Complex._calculate(left, right, 2, context)


def divide(
    left: _ComplexArgument,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The divide `left / right`, each component rounded once.

    Division by complex zero makes each finite nonzero component of the
    dividend an infinity, with divide-by-zero.

    Either operand may be a Complex, Float, Integer, Rational, integer literal or
    typed native number; real operands keep real semantics, so a real zero never
    changes the other operand's imaginary zero sign.

    Args:
        left: The first operand.
        right: The second operand.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default the merged component formats, rounded to nearest-even.

    Returns:
        The Complex result; both components are published together.

    Raises:
        When a trapped condition occurs in either component.
    """
    return Complex._calculate(left, right, 3, context)


def sqrt(
    value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()
) raises -> Complex:
    """The principal square root, each component rounded once.

    The real part is nonnegative, and the imaginary part follows the sign of the
    input's imaginary part on both sides of the negative real axis, signed zero
    included: `sqrt(Complex(-4, +0))` is `(0, 2)` and the `-0` side gives `(0, -2)`.

    Args:
        value: The Complex.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default each component's format, rounded to nearest-even.

    Returns:
        The principal root.

    Raises:
        When a trapped condition occurs in either component.
    """
    return value._sqrt(context)[0]


def pow_int(
    value: Complex,
    exponent: Integer,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The power for a signed integral exponent, each component rounded once.

    A zero exponent gives a real one. Complex zero to a negative power is
    `(+inf, NaN)` with divide-by-zero; an infinite input gives `(+inf, NaN)` for a
    positive power and complex zero for a negative one. A part that is exactly
    zero takes MPC's sign for a base on an axis, and for an inexact power of a
    diagonal base (`|re| == |im|`); an exact power of a diagonal base follows
    the exponent's phase, which can differ from MPC's beyond the eighth power.
    Either can differ from repeated multiplication.

    Args:
        value: The base.
        exponent: The exponent, of any size.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default each component's format, rounded to nearest-even.

    Returns:
        `value ** exponent`.

    Raises:
        When a trapped condition occurs in either component.
    """
    return value._pow(exponent, context)[0]


def abs(
    value: Complex, *, context: Optional[ArithmeticContext] = None
) raises -> Float:
    """The magnitude, `sqrt(real**2 + imag**2)`, rounded once (numpy's `abs`).

    The squared magnitude is never rounded first, so it cannot overflow or
    underflow early. An infinite component gives `+inf` even when the other is NaN.

    Args:
        value: The Complex.
        context: The output format; by default the larger component precision.

    Returns:
        The nonnegative magnitude as a Float.

    Raises:
        On a trapped condition.
    """
    return Float(_rounded=value._real_function(1, context))


def norm_sqr(
    value: Complex, *, context: Optional[ArithmeticContext] = None
) raises -> Float:
    """The squared magnitude, `real**2 + imag**2`, rounded once.

    Args:
        value: The Complex.
        context: The output format; by default the larger component precision.

    Returns:
        The squared magnitude as a Float.

    Raises:
        On a trapped condition.
    """
    return Float(_rounded=value._real_function(0, context))


def stable_hash(z: Complex) -> UInt64:
    """A 64-bit hash of the representation, stable across processes and releases.

    The algorithm is APNH-64: tag 4, then the real and the imaginary component,
    each encoded as for a Float. Unlike Mojo's `Hasher`, its output never
    changes, so stored hashes stay valid.

    Args:
        z: The Complex number.

    Returns:
        The hash.
    """
    var hash = _StableHash(4)
    z._hash_into(hash)
    return hash.finish()


def conjugate(value: Complex) raises -> Complex:
    """The conjugate, `real - imag i` (numpy's `conjugate`).

    Args:
        value: The Complex.

    Returns:
        The conjugate, exactly.

    Raises:
        Only on a checked size error.
    """
    return value.conjugate()


def real(value: Complex) raises -> Float:
    """The real part (numpy's `real`).

    Args:
        value: The Complex.

    Returns:
        The real component, exactly.

    Raises:
        Only on a checked size error.
    """
    return value.real()


def imag(value: Complex) raises -> Float:
    """The imaginary part (numpy's `imag`).

    Args:
        value: The Complex.

    Returns:
        The imaginary component, exactly.

    Raises:
        Only on a checked size error.
    """
    return value.imag()


def reciprocal(
    value: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """`1 / value`, each component rounded once (numpy's `reciprocal`).

    Args:
        value: The operand.
        context: An `ArithmeticContext` for both components or a `ComplexContext` for each; by default the operand's component formats, rounded to nearest-even.

    Returns:
        The Complex reciprocal.

    Raises:
        When a trapped condition occurs in either component.
    """
    return Complex._calculate(Integer(1), value, 3, context)
