"""Number theory on native words: remainders of magnitudes by one word,
modular powers, and a deterministic primality test below 2**64."""

from std.bit import count_trailing_zeros
from .value import Integer


def _trailing_zero_bits(x: Integer) -> Int:
    """The number of trailing zero bits of a nonzero `x`."""
    var index = 0
    while x._word(index) == 0:
        index += 1
    return index * 32 + Int(count_trailing_zeros(x._word(index)))


def _fits_word(x: Integer) -> Bool:
    """Whether the magnitude of `x` fits one 64-bit word."""
    return x._word_count() <= 2


def _remainder_by_word(x: Integer, divisor: UInt64) -> UInt64:
    """`abs(x) mod divisor` for a nonzero divisor, without allocating."""
    var small = x._inline_words()
    var words = x._words_span(small)
    if divisor >> 32 == 0:
        # The remainder stays below 2**32, so each step fits 64 bits.
        var remainder = UInt64(0)
        for i in range(len(words) - 1, -1, -1):
            remainder = ((remainder << 32) | UInt64(words.unsafe_get(i))) % divisor
        return remainder
    var wide = UInt128(0)
    for i in range(len(words) - 1, -1, -1):
        wide = ((wide << 32) | UInt128(words.unsafe_get(i))) % UInt128(divisor)
    return UInt64(wide)


@always_inline
def _multiply_mod(a: UInt64, b: UInt64, modulus: UInt64) -> UInt64:
    return UInt64((UInt128(a) * UInt128(b)) % UInt128(modulus))


def _power_mod(base: UInt64, var exponent: UInt64, modulus: UInt64) -> UInt64:
    """`base ** exponent mod modulus` for `modulus >= 2`."""
    var result = UInt64(1) % modulus
    var factor = base % modulus
    while exponent:
        if exponent & 1:
            result = _multiply_mod(result, factor, modulus)
        exponent >>= 1
        if exponent:
            factor = _multiply_mod(factor, factor, modulus)
    return result


def _strong_probable_prime(n: UInt64, base: UInt64) -> Bool:
    """The Miller-Rabin test of an odd `n > base` to one base."""
    var odd = n - 1
    var shifts = Int(count_trailing_zeros(odd))
    odd >>= UInt64(shifts)
    var x = _power_mod(base, odd, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(shifts - 1):
        x = _multiply_mod(x, x, n)
        if x == n - 1:
            return True
    return False


def _is_prime_word(n: UInt64) -> Bool:
    """Whether `n` is prime, deterministically.

    Trial division by the primes up to 37 settles every `n` below 41**2. Above,
    Miller-Rabin with the first k prime bases is correct below the bound
    psi_k (Jaeschke 1993; Sorenson and Webster 2017); the first 12 bases cover
    every 64-bit `n`, since psi_12 is about 3.2 * 10**23.
    """
    var primes: List[UInt64] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 1681:
        return n >= 2
    var bounds: List[UInt64] = [
        2047,
        1373653,
        25326001,
        3215031751,
        2152302898747,
        3474749660383,
        341550071728321,
        341550071728321,
        3825123056546413051,
        3825123056546413051,
        3825123056546413051,
    ]
    for i in range(len(primes)):
        if not _strong_probable_prime(n, primes[i]):
            return False
        if i < len(bounds) and n < bounds[i]:
            return True
    return True


def _next_prime_word(n: UInt64) -> UInt64:
    """The smallest prime above `n`, for `n` well below 2**64."""
    var candidate = n + 1
    while not _is_prime_word(candidate):
        candidate += 1
    return candidate


def _jacobi_word(var a: UInt64, var n: UInt64) -> Int:
    """The Jacobi symbol (a/n) for an odd positive `n`."""
    a %= n
    var result = 1
    while a:
        var twos = count_trailing_zeros(a)
        a >>= twos
        var low = n & 7
        if (twos & 1) == 1 and (low == 3 or low == 5):
            result = -result
        if (a & 3) == 3 and (low & 3) == 3:
            result = -result
        var remainder = n % a
        n = a
        a = remainder
    return result if n == 1 else 0


def _power_residue_witness(x: Integer, degree: Int) -> Bool:
    """Whether a residue proves that a positive `x` is not a perfect power of
    this degree, for `degree >= 2`.

    Squares are tested against the residues modulo 64, 63, 65 and 11, which
    together pass fewer than 1% of non-squares. Other degrees d use the two
    smallest primes l = k d + 1: an x coprime to l is a d-th power modulo l
    only when x**((l - 1) / d) is 1 modulo l.
    """
    if degree == 2:
        if ((UInt64(0x202021202030213) >> (UInt64(x._word(0)) & 63)) & 1) == 0:
            return True
        var r = _remainder_by_word(x, 45045)
        return (
            ((UInt64(0x402483012450293) >> (r % 63)) & 1) == 0
            or ((UInt128(0x1218A019866014613) >> UInt128(r % 65)) & 1) == 0
            or ((UInt64(0x23B) >> (r % 11)) & 1) == 0
        )
    var found = 0
    var step = UInt64(degree)
    var candidate = step + 1
    while found < 2:
        if _is_prime_word(candidate):
            found += 1
            var r = _remainder_by_word(x, candidate)
            if r != 0 and _power_mod(r, (candidate - 1) // step, candidate) != 1:
                return True
        candidate += step
    return False
