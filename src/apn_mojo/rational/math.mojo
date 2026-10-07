"""Exact Rational arithmetic as functions; batches apply them with vmap."""

from .value import Rational, pow_rational
from ..integer.value import Integer
from ..common._stable_hash import _StableHash
from ..integer.number_theory import gcd as _integer_gcd, lcm as _integer_lcm
from ..integer.powers import iroot_exact


def add(left: Rational, right: Rational) raises -> Rational:
    """The exact sum; the same as `left + right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left + right` in lowest terms.

    Raises:
        Only on a checked size error.
    """
    return left + right


def subtract(left: Rational, right: Rational) raises -> Rational:
    """The exact difference; the same as `left - right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left - right` in lowest terms.

    Raises:
        Only on a checked size error.
    """
    return left - right


def multiply(left: Rational, right: Rational) raises -> Rational:
    """The exact product; the same as `left * right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left * right` in lowest terms.

    Raises:
        Only on a checked size error.
    """
    return left * right


def divide(left: Rational, right: Rational) raises -> Rational:
    """The exact quotient; the same as `left / right`.

    Args:
        left: The dividend.
        right: The divisor.

    Returns:
        `left / right` in lowest terms.

    Raises:
        When `right` is zero.
    """
    return left / right


def abs(value: Rational) raises -> Rational:
    """The absolute value.

    Args:
        value: The fraction.

    Returns:
        The nonnegative magnitude.

    Raises:
        Only on a checked size error.
    """
    return value.__abs__()


def gcd(a: Rational, b: Rational) raises -> Rational:
    """The greatest common divisor of two fractions.

    It is the largest fraction of which both are integer multiples:
    `gcd(1/2, 1/3)` is 1/6 and `gcd(4/9, 2/5)` is 2/45. In lowest terms it is
    `gcd(numerators) / lcm(denominators)`, already reduced.

    Args:
        a: The first fraction.
        b: The second fraction.

    Returns:
        The nonnegative gcd; `gcd(0, 0)` is zero.

    Raises:
        Only on a checked size error.
    """
    var numerator = _integer_gcd(a.numerator(), b.numerator())
    if not numerator:
        return Rational(0)
    # A prime of the numerators' gcd divides neither denominator.
    return Rational(
        _numerator=numerator,
        _denominator=_integer_lcm(a.denominator(), b.denominator()),
    )


def lcm(a: Rational, b: Rational) raises -> Rational:
    """The least common multiple of two fractions.

    It is the smallest nonnegative fraction that is an integer multiple of
    both: `lcm(1/2, 1/3)` is 1. In lowest terms it is
    `lcm(numerators) / gcd(denominators)`, already reduced.

    Args:
        a: The first fraction.
        b: The second fraction.

    Returns:
        The nonnegative lcm; zero when either input is zero.

    Raises:
        Only on a checked size error.
    """
    var numerator = _integer_lcm(a.numerator(), b.numerator())
    if not numerator:
        return Rational(0)
    # A prime of the denominators' gcd divides neither numerator.
    return Rational(
        _numerator=numerator,
        _denominator=_integer_gcd(a.denominator(), b.denominator()),
    )


def root_exact(value: Rational, n: Integer) raises -> Optional[Rational]:
    """The exact `n`-th root of a fraction, when it has one.

    `root_exact(9/4, 2)` is 3/2 and `root_exact(-27/8, 3)` is -3/2;
    `root_exact(2, 2)` is None. A fraction in lowest terms is a perfect power
    exactly when its numerator and denominator are. Through `vmap` it gives
    the roots, 0 where there is none, and a Mask of where there is one.

    Args:
        value: The radicand; a negative one has a root only for odd `n`.
        n: The degree, positive.

    Returns:
        The fraction `r` with `r**n == value`, or None when there is none.

    Raises:
        When `n` is not positive.
    """
    var top = iroot_exact(value.numerator(), n)
    if not top:
        return None
    var bottom = iroot_exact(value.denominator(), n)
    if not bottom:
        return None
    # Roots of coprime integers are coprime.
    return Rational(_numerator=top.value(), _denominator=bottom.value())


def stable_hash(x: Rational) -> UInt64:
    """A 64-bit hash of the value, stable across processes and releases.

    The algorithm is APNH-64: tag 2, then the numerator and the denominator in
    lowest terms, each encoded as an Integer. Unlike Mojo's `Hasher`, its output
    never changes, so stored hashes stay valid. Equal fractions hash equal.

    Args:
        x: The fraction.

    Returns:
        The hash.
    """
    var hash = _StableHash(2)
    hash.integer(x.numerator())
    hash.integer(x.denominator())
    return hash.finish()


def floor(value: Rational) raises -> Integer:
    """Round a Rational toward negative infinity, to an Integer (numpy's `floor`).

    Args:
        value: The operand.

    Returns:
        The largest Integer not above the value.

    Raises:
        Only on a checked size error.
    """
    return value.floor()


def ceil(value: Rational) raises -> Integer:
    """Round a Rational toward positive infinity, to an Integer (numpy's `ceil`).

    Args:
        value: The operand.

    Returns:
        The smallest Integer not below the value.

    Raises:
        Only on a checked size error.
    """
    return value.ceil()


def trunc(value: Rational) raises -> Integer:
    """Round a Rational toward zero, to an Integer (numpy's `trunc`).

    Args:
        value: The operand.

    Returns:
        The integer part.

    Raises:
        Only on a checked size error.
    """
    return value.trunc()


def round(value: Rational) raises -> Integer:
    """Round a Rational to the nearest Integer, a half to the even neighbour, to an Integer (numpy's `round`).

    Args:
        value: The operand.

    Returns:
        The nearest Integer: 2.5 gives 2, 3.5 gives 4 and -2.5 gives -2.

    Raises:
        Only on a checked size error.
    """
    return value.round()


def reciprocal(value: Rational) raises -> Rational:
    """`1 / value`, exactly (numpy's `reciprocal`).

    Args:
        value: A nonzero Rational.

    Returns:
        The exact reciprocal.

    Raises:
        When `value` is zero.
    """
    return divide(Rational(1), value)


def maximum(left: Rational, right: Rational) raises -> Rational:
    """The larger of two Rationals, the first when they are equal (numpy's
    `maximum`).

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left` if `left >= right`, else `right`.

    Raises:
        Only on a checked size error.
    """
    return left if left >= right else right


def minimum(left: Rational, right: Rational) raises -> Rational:
    """The smaller of two Rationals, the first when they are equal (numpy's
    `minimum`).

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left` if `left <= right`, else `right`.

    Raises:
        Only on a checked size error.
    """
    return left if left <= right else right


def clip(value: Rational, a_min: Rational, a_max: Rational) raises -> Rational:
    """`value` limited to `[a_min, a_max]`: `minimum(maximum(value, a_min),
    a_max)` (numpy's `clip`), so `a_max` when `a_min > a_max`.

    Args:
        value: The operand.
        a_min: The lower limit.
        a_max: The upper limit.

    Returns:
        The limited value.

    Raises:
        Only on a checked size error.
    """
    return minimum(maximum(value, a_min), a_max)
