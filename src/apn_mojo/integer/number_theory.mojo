"""Exact gcd, lcm, floor square roots, factor removal, trial division and the
Jacobi symbol."""

from std.memory import ArcPointer
from .value import Integer
from ._square_root import _sqrt_rem, _sqrt_rem_native, _wide_magnitude
from ._word_math import _trailing_zero_bits, _fits_word, _remainder_by_word, _jacobi_word


def _sqrt_domain(index: Int = -1) raises:
    var location = String()
    if index >= 0:
        location = String(" at logical element ", index, " (zero-based)")
    raise Error(
        String(
            "Cannot take integer square root of a negative value",
            location,
            "; use a nonnegative input. The destination is unchanged.",
        )
    )


def gcd(a: Integer, b: Integer) raises -> Integer:
    """The greatest common divisor.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The nonnegative gcd; `gcd(0, 0)` is zero.

    Raises:
        Only on a checked size error.
    """
    return a._gcd(b)


def lcm(a: Integer, b: Integer) raises -> Integer:
    """The least common multiple.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The nonnegative lcm; zero when either input is zero.

    Raises:
        Only on a checked size error.
    """
    if a._word_count() == 0 or b._word_count() == 0:
        return Integer(0)
    return (abs(a) // gcd(a, b)) * abs(b)


def isqrt(a: Integer) raises -> Integer:
    """The integer square root, `floor(sqrt(a))`.

    Args:
        a: A nonnegative integer.

    Returns:
        The largest `r` with `r * r <= a`.

    Raises:
        When `a` is negative.
    """
    if a._negative():
        _sqrt_domain()
    if a <= 1:
        return a
    if a.magnitude_bit_length() <= 256:
        return Integer._from_wide_magnitude128(_sqrt_rem_native(_wide_magnitude(a))[0])
    return _sqrt_rem(a)[0]


def remove_factor(n: Integer, p: Integer) raises -> Tuple[Integer, Integer]:
    """Divide `n` by the highest power of `p` that divides it.

    `remove_factor(-72, 2)` is `(-9, 3)`, since -72 is -9 * 2**3. The search
    divides by p, p**2, p**4, ... and then back down through the same powers,
    so a multiplicity k costs about 2 log2(k) divisions rather than k. Factors
    of 2 are counted from trailing zero bits instead.

    Args:
        n: A nonzero integer.
        p: The factor, at least 2; it need not be prime.

    Returns:
        `(n / p**k, k)` with k as large as possible; the cofactor keeps the
        sign of `n`.

    Raises:
        When `p` is below 2, or `n` is zero, which every power divides.
    """
    if p < 2:
        raise Error("Cannot remove a factor below 2; use a factor p of at least 2.")
    if not n:
        raise Error(
            "Cannot remove a factor from 0, which every power divides; use a"
            " nonzero integer."
        )
    if p == 2:
        var twos = _trailing_zero_bits(n)
        return (n >> twos, Integer(twos))
    var first = n._div_rem_trunc(p)
    if first[1]:
        return (n, Integer(0))
    var rest = first[0]
    var count = 1
    # powers[i] is p**(2**i). After the ascent the multiplicity left is below
    # 2**len(powers), so the descent removes it greedily.
    var powers = List[Integer]()
    powers.append(p)
    while True:
        var bits = powers[len(powers) - 1].magnitude_bit_length()
        if 2 * bits - 1 > rest.magnitude_bit_length():
            break
        var square = powers[len(powers) - 1] * powers[len(powers) - 1]
        var step = rest._div_rem_trunc(square)
        if step[1]:
            break
        rest = step[0]
        count += 1 << len(powers)
        powers.append(square^)
    for i in range(len(powers) - 1, -1, -1):
        var step = rest._div_rem_trunc(powers[i])
        if not step[1]:
            rest = step[0]
            count += 1 << i
    return (rest^, Integer(count))


@fieldwise_init
struct TrialDivision(ImplicitlyCopyable, Writable):
    """The prime factors of an integer below a bound, and what remains.

    `trial_division` returns it. Copies share one immutable list of factors.
    """

    var _factors: ArcPointer[List[Tuple[Integer, Integer]]]
    var _cofactor: Integer

    def factors(self) -> List[Tuple[Integer, Integer]]:
        """The primes found, ascending, each with its multiplicity.

        Returns:
            `(prime, multiplicity)` pairs.
        """
        return self._factors[].copy()

    def cofactor(self) -> Integer:
        """What remains after dividing out the factors.

        Returns:
            `n` divided by every prime power found. It keeps the sign of `n`,
            and is 1 or -1 when the factorization is complete.
        """
        return self._cofactor

    def write_to(self, mut writer: Some[Writer]):
        """Write the factors and the cofactor, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("TrialDivision(factors=[")
        for i in range(len(self._factors[])):
            ref factor = self._factors[][i]
            if i:
                writer.write(", ")
            writer.write("(", factor[0], ", ", factor[1], ")")
        writer.write("], cofactor=", self._cofactor, ")")


def trial_division(n: Integer, bound: Integer) raises -> TrialDivision:
    """The prime factors of `n` below `bound`, found by trial division.

    `trial_division(360, 4)` finds 2**3 and 3**2 and leaves 5. The candidates
    are 2, 3, 5 and the numbers coprime to 30; a composite candidate never
    divides, because its prime factors were divided out before it. The search
    stops once a candidate's square exceeds what remains, which is then 1 or a
    prime, and that prime counts as a factor when it is below `bound`. A
    batch cannot hold its result, so apply it to one value at a time rather
    than through `vmap`.

    Args:
        n: A nonzero integer.
        bound: Only primes below it are tried; below 3, none are.

    Returns:
        The factors and the cofactor; see `TrialDivision`.

    Raises:
        When `n` is zero, which every prime divides.
    """
    if not n:
        raise Error(
            "Cannot factor 0 by trial division, since every prime divides it;"
            " use a nonzero integer."
        )
    var factors = List[Tuple[Integer, Integer]]()
    if bound <= 2:
        return TrialDivision(ArcPointer(factors^), n)
    var negative = n._negative()
    var rest = abs(n)
    # Candidates past 2**62 would take longer than any caller can wait.
    var stop = UInt64(1) << 62
    if bound < Integer(stop):
        stop = bound._low_magnitude()
    var twos = _trailing_zero_bits(rest)
    if twos:
        factors.append((Integer(2), Integer(twos)))
        rest = rest >> twos
    # Once the rest fits a word, the search continues on native integers.
    var small = _fits_word(rest)
    var native = rest._low_magnitude() if small else UInt64(0)
    var increments: List[UInt64] = [4, 2, 4, 2, 4, 6, 2, 6]
    var phase = 0
    var candidate = UInt64(3)
    var prime_rest = False
    while candidate < stop:
        if small:
            if native == 1:
                break
            if UInt128(candidate) * UInt128(candidate) > UInt128(native):
                prime_rest = True
                break
        elif (candidate >> 32) != 0 and Integer(candidate) * Integer(candidate) > rest:
            prime_rest = True
            break
        var remainder = native % candidate if small else _remainder_by_word(rest, candidate)
        if remainder == 0:
            var count = 0
            if small:
                while native % candidate == 0:
                    native //= candidate
                    count += 1
            else:
                var divisor = Integer(candidate)
                while True:
                    var step = rest._div_rem_trunc(divisor)
                    if step[1]:
                        break
                    rest = step[0]
                    count += 1
                if _fits_word(rest):
                    small = True
                    native = rest._low_magnitude()
            factors.append((Integer(candidate), Integer(count)))
        if candidate < 7:
            candidate += 2
        else:
            candidate += increments[phase]
            phase = (phase + 1) & 7
    if small:
        rest = Integer(native)
    if prime_rest and rest > 1 and rest < bound:
        factors.append((rest, Integer(1)))
        rest = Integer(1)
    return TrialDivision(ArcPointer(factors^), -rest if negative else rest)


def jacobi(a: Integer, n: Integer) raises -> Integer:
    """The Jacobi symbol (a/n).

    For a prime `n` it is the Legendre symbol: 1 when `a` is a nonzero square
    modulo `n`, -1 when it is not a square, and 0 when `n` divides `a`.

    Args:
        a: Any integer.
        n: An odd positive modulus.

    Returns:
        -1, 0 or 1.

    Raises:
        When `n` is even or not positive.
    """
    if n < 1 or not (n._word(0) & 1):
        raise Error("Cannot compute the Jacobi symbol: use an odd positive modulus n.")
    var x = a % n
    var m = n
    var result = 1
    while x:
        if _fits_word(x) and _fits_word(m):
            return Integer(result * _jacobi_word(x._low_magnitude(), m._low_magnitude()))
        var twos = _trailing_zero_bits(x)
        x = x >> twos
        var low = m._word(0) & 7
        if (twos & 1) == 1 and (low == 3 or low == 5):
            result = -result
        if (x._word(0) & 3) == 3 and (low & 3) == 3:
            result = -result
        var remainder = m % x
        m = x
        x = remainder
    return Integer(result if m == 1 else 0)
