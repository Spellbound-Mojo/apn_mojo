"""Primality: proven below 2**64, Baillie-PSW above."""

from .value import Integer
from .math import pow_mod
from .number_theory import jacobi
from .powers import iroot_exact
from ._word_math import (
    _fits_word,
    _is_prime_word,
    _next_prime_word,
    _remainder_by_word,
    _trailing_zero_bits,
)


def is_prime(n: Integer) raises -> Bool:
    """Whether `n` is prime.

    Below 2**64 the answer is proven: Miller-Rabin with the first twelve prime
    bases has no false positive below about 3.2 * 10**23 (Sorenson and Webster
    2017). Above, the test is Baillie-PSW: trial division by the primes below
    1000, a strong probable-prime test to base 2, a perfect-square check, and a
    strong Lucas test with Selfridge's parameters. No composite is known to
    pass it, but that is not a proof.

    Args:
        n: Any integer; negative numbers, 0 and 1 are not prime.

    Returns:
        True for a prime.

    Raises:
        Only on a checked size error.
    """
    if n < 2:
        return False
    if _fits_word(n):
        return _is_prime_word(n._low_magnitude())
    if not (n._word(0) & 1):
        return False
    var p = UInt64(3)
    while p < 1000:
        if _remainder_by_word(n, p) == 0:
            return False
        p = _next_prime_word(p)
    return _baillie_psw(n)


def _baillie_psw(n: Integer) raises -> Bool:
    """The base-2 strong test, the square check and the strong Lucas test, for
    an odd `n` above 5."""
    if not _strong_probable_prime(n, Integer(2)):
        return False
    if iroot_exact(n, 2):
        return False
    return _strong_lucas_probable_prime(n)


def _strong_probable_prime(n: Integer, base: Integer) raises -> Bool:
    """The Miller-Rabin test of an odd `n` to `base`."""
    var minus_one = n - 1
    var shifts = _trailing_zero_bits(minus_one)
    var x = pow_mod(base, minus_one >> shifts, n)
    if x == 1 or x == minus_one:
        return True
    for _ in range(shifts - 1):
        x = (x * x) % n
        if x == minus_one:
            return True
    return False


def _halve_mod(x: Integer, n: Integer) raises -> Integer:
    """`x / 2` modulo an odd `n`, for `0 <= x < n`."""
    return (x + n) >> 1 if (x._word(0) & 1) != 0 else x >> 1


def _strong_lucas_probable_prime(n: Integer) raises -> Bool:
    """The strong Lucas test of an odd `n` that is not a perfect square, with
    Selfridge's method A: the first D in 5, -7, 9, -11, ... with (D/n) = -1,
    P = 1 and Q = (1 - D) / 4.

    Write n + 1 = d 2**s with d odd. A prime n has U_d = 0, or V_(d 2**r) = 0
    for some r < s. The sequences are doubled and stepped along the bits of d
    with U_2k = U_k V_k, V_2k = V_k**2 - 2 Q**k, U_(k+1) = (P U_k + V_k) / 2
    and V_(k+1) = (D U_k + P V_k) / 2.
    """
    var d_value = 5
    while True:
        var symbol = jacobi(Integer(d_value), n)
        if symbol == -1:
            break
        if symbol == 0 and abs(Integer(d_value)) != n:
            return False
        d_value = -(d_value + 2) if d_value > 0 else -(d_value - 2)
    var D = Integer(d_value) % n
    var Q = Integer((1 - d_value) // 4) % n
    var plus_one = n + 1
    var shifts = _trailing_zero_bits(plus_one)
    var d = plus_one >> shifts
    var u = Integer(1)
    var v = Integer(1)
    var q_power = Q
    for bit in range(d.magnitude_bit_length() - 2, -1, -1):
        u = (u * v) % n
        v = (v * v - (q_power << 1)) % n
        q_power = (q_power * q_power) % n
        if (d._word(bit >> 5) >> UInt32(bit & 31)) & 1:
            var next_u = _halve_mod((u + v) % n, n)
            v = _halve_mod((D * u + v) % n, n)
            u = next_u
            q_power = (q_power * Q) % n
    if not u or not v:
        return True
    for _ in range(shifts - 1):
        v = (v * v - (q_power << 1)) % n
        q_power = (q_power * q_power) % n
        if not v:
            return True
    return False
