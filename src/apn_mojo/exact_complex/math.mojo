"""Exact complex functions, one declaration per name."""

from ..integer.value import Integer
from ..rational.value import Rational
from ..rational.math import root_exact
from ..common._stable_hash import _StableHash
from .value import ExactComplex


def add(left: ExactComplex, right: ExactComplex) raises -> ExactComplex:
    """The exact sum.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left + right`.

    Raises:
        Only on a checked size error.
    """
    return left + right


def subtract(left: ExactComplex, right: ExactComplex) raises -> ExactComplex:
    """The exact difference.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left - right`.

    Raises:
        Only on a checked size error.
    """
    return left - right


def multiply(left: ExactComplex, right: ExactComplex) raises -> ExactComplex:
    """The exact product.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left * right`.

    Raises:
        Only on a checked size error.
    """
    return left * right


def divide(left: ExactComplex, right: ExactComplex) raises -> ExactComplex:
    """The exact quotient.

    Args:
        left: The dividend.
        right: The divisor, nonzero.

    Returns:
        `left / right`.

    Raises:
        `division by zero` for a zero divisor.
    """
    return left / right


def conjugate(value: ExactComplex) raises -> ExactComplex:
    """The complex conjugate, `re - im i`.

    Args:
        value: The number.

    Returns:
        The conjugate.

    Raises:
        Only on a checked size error.
    """
    return ExactComplex(value.real(), -value.imag())


def norm_sqr(value: ExactComplex) raises -> Rational:
    """The squared magnitude, `re**2 + im**2`, exactly.

    Args:
        value: The number.

    Returns:
        The squared magnitude.

    Raises:
        Only on a checked size error.
    """
    return value.real() * value.real() + value.imag() * value.imag()


def pow_int(value: ExactComplex, exponent: Integer) raises -> ExactComplex:
    """The exact power for an integral exponent of any sign.

    `pow_int(z, 0)` is 1, even for `z = 0`; a negative exponent takes the
    reciprocal, so `0` to a negative power raises.

    Args:
        value: The base.
        exponent: The exponent.

    Returns:
        `value**exponent`.

    Raises:
        `division by zero` for a zero base and a negative exponent.
    """
    var result = ExactComplex(Rational(1))
    var base = value
    var n = abs(exponent)
    var bits = n.magnitude_bit_length()
    for bit in range(bits - 1, -1, -1):
        result = result * result
        if (n._word(bit >> 5) >> UInt32(bit & 31)) & 1:
            result = result * base
    if exponent.sign() < 0:
        return ExactComplex(Rational(1)) / result
    return result^


def sqrt_exact(value: ExactComplex) raises -> Optional[ExactComplex]:
    """The principal square root, when both parts are rational.

    The principal root has a nonnegative real part, and a nonnegative
    imaginary part when the real part is 0. Either both parts of the root are
    rational or neither is, so `sqrt_exact(2i)` is `1 + i`, `sqrt_exact(-4)` is
    `2i`, and `sqrt_exact(i)` is None.

    Args:
        value: The number.

    Returns:
        The root, or None when it is not Gaussian-rational.

    Raises:
        Only on a checked size error.
    """
    var x = value.real()
    var y = value.imag()
    if y.sign() == 0:
        if x.sign() >= 0:
            var r = root_exact(x, Integer(2))
            if not r:
                return None
            return ExactComplex(r.take(), Rational(0))
        var s = root_exact(-x, Integer(2))
        if not s:
            return None
        return ExactComplex(Rational(0), s.take())
    var norm = root_exact(x * x + y * y, Integer(2))
    if not norm:
        return None
    var n = norm.take()
    var a = root_exact((n + x) / 2, Integer(2))
    var b = root_exact((n - x) / 2, Integer(2))
    if not a or not b:
        return None
    var imag = b.take()
    return ExactComplex(a.take(), -imag if y.sign() < 0 else imag)


def stable_hash(value: ExactComplex) -> UInt64:
    """A 64-bit hash of the value, stable across processes and releases.

    The algorithm is APNH-64: tag 5, then the real part's numerator and
    denominator and the imaginary part's, each encoded as an Integer. Equal
    values hash equal.

    Args:
        value: The number.

    Returns:
        The hash.
    """
    var hash = _StableHash(5)
    value._hash_into(hash)
    return hash.finish()


def real(value: ExactComplex) raises -> Rational:
    """The real part (numpy's `real`).

    Args:
        value: The number.

    Returns:
        The real part.

    Raises:
        Only on a checked size error.
    """
    return value.real()


def imag(value: ExactComplex) raises -> Rational:
    """The imaginary part (numpy's `imag`).

    Args:
        value: The number.

    Returns:
        The imaginary part.

    Raises:
        Only on a checked size error.
    """
    return value.imag()


def reciprocal(value: ExactComplex) raises -> ExactComplex:
    """`1 / value`, exactly (numpy's `reciprocal`).

    Args:
        value: A nonzero number.

    Returns:
        The exact reciprocal.

    Raises:
        When `value` is zero.
    """
    return divide(ExactComplex(Rational(1)), value)
