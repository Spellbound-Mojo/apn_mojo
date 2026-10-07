"""Exact roots and perfect powers of integers."""

from .value import Integer
from ._root import _integer_root
from .number_theory import _sqrt_rem
from ._word_math import _trailing_zero_bits, _power_residue_witness, _next_prime_word


def iroot_exact(x: Integer, n: Integer) raises -> Optional[Integer]:
    """The exact `n`-th root of `x`, when `x` is a perfect `n`-th power.

    `iroot_exact(-27, 3)` is -3; `iroot_exact(2, 2)` and `iroot_exact(-4, 2)`
    are None. Cheap tests reject most inputs before any root is taken: the
    trailing zero bits must be a multiple of `n`, and residues modulo a few
    small numbers must be `n`-th power residues. Through `vmap` it gives the
    roots, 0 where there is none, and a Mask of where there is one.

    Args:
        x: The radicand; a negative one has a root only for odd `n`.
        n: The degree, positive.

    Returns:
        The integer `r` with `r**n == x`, or None when there is none.

    Raises:
        When `n` is not positive.
    """
    if n <= 0:
        raise Error("Cannot compute iroot_exact: use a positive integer degree n.")
    var negative = x._negative()
    if negative and not (n._word(0) & 1):
        return None
    if n == 1:
        return x
    var magnitude = abs(x)
    if magnitude <= 1:
        return x
    # A root of 1 is the only one whose power stays below 2**n.
    var bits = magnitude.magnitude_bit_length()
    if n >= bits:
        return None
    var degree = Int(n)
    if _trailing_zero_bits(magnitude) % degree:
        return None
    if _power_residue_witness(magnitude, degree):
        return None
    var root: Integer
    if degree == 2:
        var pair = _sqrt_rem(magnitude)
        if pair[1]:
            return None
        root = pair[0]
    else:
        var found = _integer_root(magnitude, degree)
        if not found[1]:
            return None
        root = found[0]
    return -root if negative else root


def perfect_power(x: Integer) raises -> Tuple[Integer, Integer]:
    """`x` as `b**e` with the largest exponent `e`.

    `perfect_power(1000)` is `(10, 3)` and `perfect_power(-512)` is `(-2, 9)`;
    `perfect_power(12)` is `(12, 1)`. The search tries each prime exponent up
    to the bit length of `x`, odd ones only for negative `x`, and continues on
    every root it finds, which builds composite exponents.

    Args:
        x: Any integer.

    Returns:
        `(b, e)` with `b**e == x` and `e` maximal; `(x, 1)` when `x` is not a
        perfect power, and for -1, 0 and 1.

    Raises:
        Only on a checked size error.
    """
    var magnitude = abs(x)
    if magnitude <= 1:
        return (x, Integer(1))
    var negative = x._negative()
    var exponent = 1
    var prime = UInt64(3) if negative else UInt64(2)
    while Int(prime) <= magnitude.magnitude_bit_length():
        var root = iroot_exact(magnitude, Integer(prime))
        if root:
            # The root may be a power of the same prime again.
            magnitude = root.value()
            exponent *= Int(prime)
        else:
            prime = _next_prime_word(prime)
    return (-magnitude if negative else magnitude, Integer(exponent))
