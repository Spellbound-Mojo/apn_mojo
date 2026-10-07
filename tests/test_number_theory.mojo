"""Number theory: factor removal, trial division, exact roots, perfect powers,
primality and the Jacobi symbol, checked against tests/fixtures/number_theory.txt
(written by gmpy2) and against independent computations here."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    Batch,
    Integer,
    Rational,
    TrialDivision,
    integer,
    rational,
    vmap,
    is_prime,
    iroot,
    iroot_exact,
    jacobi,
    perfect_power,
    remove_factor,
    root_exact,
    trial_division,
)
from apn_mojo.integer.primality import _baillie_psw, _strong_lucas_probable_prime


def _fixture(name: String) raises -> List[Integer]:
    """The values of one fixture line; the runner starts tests in the project root."""
    var text: String
    with open("tests/fixtures/number_theory.txt", "r") as source:
        text = source.read()
    for row in text.split("\n"):
        var fields = row.split(" ")
        if len(fields) and String(fields[0]) == name:
            var values = List[Integer]()
            for i in range(1, len(fields)):
                values.append(Integer.parse(String(fields[i])))
            return values^
    raise Error(String("No fixture named ", name))


def _sieve(limit: Int) -> List[Bool]:
    var prime = List[Bool](length=limit, fill=True)
    prime[0] = False
    prime[1] = False
    var p = 2
    while p * p < limit:
        if prime[p]:
            for multiple in range(p * p, limit, p):
                prime[multiple] = False
        p += 1
    return prime^


def _rebuilt(division: TrialDivision) raises -> Integer:
    var value = division.cofactor()
    for factor in division.factors():
        value *= factor[0] ** Int(factor[1])
    return value


def test_remove_factor() raises:
    var cofactor, count = remove_factor(-72, 2)
    assert_equal(cofactor, -9)
    assert_equal(count, 3)
    var big = (Integer(1) << 100) * Integer(3) ** 1000 * 5
    var rest, threes = remove_factor(big, 3)
    assert_equal(rest, (Integer(1) << 100) * 5)
    assert_equal(threes, 1000)
    # Composite factors, factors that do not divide, and every multiplicity
    # up to 70, which exercises each ascent and descent of the doubling.
    var twelves = remove_factor(Integer(12) ** 5 * 7, 12)
    assert_equal(twelves[0], 7)
    assert_equal(twelves[1], 5)
    assert_equal(remove_factor(35, 3)[1], 0)
    for k in range(71):
        var value = Integer(10**9 + 7) ** k * 6
        var removed = remove_factor(value, 10**9 + 7)
        assert_equal(removed[0], 6)
        assert_equal(removed[1], Integer(k))
    with assert_raises(contains="factor p of at least 2"):
        _ = remove_factor(12, 1)
    with assert_raises(contains="every power divides"):
        _ = remove_factor(0, 3)


def test_trial_division() raises:
    # The proposal's example: the cofactor 86148338324417741 is
    # 2161 * 3607 * 3803 * 2906161, so no factor below 1000 is left.
    var division = trial_division(Integer.parse("123456789012345678901234567890"), 1000)
    assert_equal(
        String(division),
        "TrialDivision(factors=[(2, 1), (3, 3), (5, 1), (7, 1), (13, 1), (31, 1),"
        " (37, 1), (211, 1), (241, 1)], cofactor=86148338324417741)",
    )
    assert_equal(String(trial_division(360, 4)), "TrialDivision(factors=[(2, 3), (3, 2)], cofactor=5)")
    assert_equal(String(trial_division(-360, 2)), "TrialDivision(factors=[], cofactor=-360)")
    assert_equal(String(trial_division(-1, 100)), "TrialDivision(factors=[], cofactor=-1)")
    # A prime left after the square test counts only when it is below the bound.
    assert_equal(String(trial_division(2 * 1009, 1000)), "TrialDivision(factors=[(2, 1)], cofactor=1009)")
    assert_equal(String(trial_division(-2 * 1009, 2000)), "TrialDivision(factors=[(2, 1), (1009, 1)], cofactor=-1)")
    # Multiword values, which switch to native words once the rest fits.
    var p64 = Integer.parse("18446744073709551557")
    var wide = (Integer(1) << 200) * 3**5 * 1000003 * p64
    var found = trial_division(wide, 1 << 20)
    assert_equal(String(found.factors()[2][0]), "1000003")
    assert_equal(found.cofactor(), p64)
    assert_equal(_rebuilt(found), wide)
    var prime = _sieve(5000)
    for n in range(1, 5000):
        for sign in [1, -1]:
            var result = trial_division(sign * n, 50)
            assert_equal(_rebuilt(result), Integer(sign * n))
            for factor in result.factors():
                assert_true(factor[0] < 50 and prime[Int(factor[0])])
            var left = Int(abs(result.cofactor()))
            for p in range(2, 50):
                assert_true(not prime[p] or left % p != 0)
    with assert_raises(contains="every prime divides"):
        _ = trial_division(0, 10)


def test_iroot_exact() raises:
    assert_equal(iroot_exact(-27, 3).value(), -3)
    assert_true(not iroot_exact(2, 2))
    assert_true(not iroot_exact(-4, 2))
    for x in [-1, 0, 1]:
        assert_equal(iroot_exact(x, 5).value(), Integer(x))
    var bases = List[Integer]()
    for b in range(2, 300):
        bases.append(b)
    var state = UInt64(88172645463325252)
    for _ in range(20):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        bases.append(Integer(state | 2))
    for b in bases:
        for e in range(2, 65):
            var power = b**e
            assert_equal(iroot_exact(power, e).value(), b)
            assert_true(not iroot_exact(power + 1, e))
            assert_true(not iroot_exact(power - 1, e))
            if e % 2:
                assert_equal(iroot_exact(-power, e).value(), -b)
    with assert_raises(contains="positive integer degree"):
        _ = iroot_exact(8, 0)


def _dense(bits: Int, mut state: UInt64) raises -> Integer:
    """A pseudo-random value of exactly `bits` bits."""
    var value = Integer(0)
    for _ in range(bits // 64 + 2):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        value = (value << 64) | Integer(state)
    return (value >> (value.magnitude_bit_length() - bits)) | (Integer(1) << (bits - 1))


def _check_root(x: Integer, degree: Int) raises:
    """iroot against its definition, r**k <= x < (r + 1)**k, by powers alone,
    and iroot_exact against r**k == x."""
    var root = iroot(x, degree)
    var power = root**degree
    assert_true(power <= x)
    assert_true((root + 1) ** degree > x)
    var exact = iroot_exact(x, degree)
    if power == x:
        assert_equal(exact.value(), root)
    else:
        assert_true(not exact)
    if degree % 2:
        assert_equal(iroot(-x, degree), -root)


def test_iroot_levels_and_certification() raises:
    # Degrees 3 to 12 and wider ones; radicands in 128-bit and 256-bit native
    # arithmetic, at limb boundaries, where seeds end and past the stack
    # scratch; perfect powers and their neighbours, all-ones radicands and
    # powers of two, whose roots sit at the edges of their bit ranges.
    var state = UInt64(0x9E3779B97F4A7C15)
    var degrees: List[Int] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 31, 32, 63, 64, 65, 100, 257]
    for degree in degrees:
        var lengths: List[Int] = [
            8, 16, 22, 31, 32, 40, 48, 56, 63, 64, 65, 127, 128, 129, 192, 255, 256, 257, 320, 512,
            1024, 2048, 4096, 9000, degree * 44, degree * 45, degree * 46, degree * 64 + 1,
        ]
        for bits in lengths:
            if bits <= degree:
                continue
            var dense = _dense(bits, state)
            for x in [(Integer(1) << bits) - 1, Integer(1) << (bits - 1), dense]:
                _check_root(x, degree)
            var root = iroot(dense, degree)
            var power = root**degree
            for x in [power - 1, power, power + 1]:
                _check_root(x, degree)
        # Just below 2**256, where powers of a root one too large overflow the
        # 256-bit native arithmetic: Newton steps must still reach the root.
        var top = (Integer(1) << 256) - 1
        for offset in range(4):
            _check_root(top - offset, degree)
        var highest = iroot(top, degree)
        for x in [highest**degree - 1, highest**degree, highest**degree + 1]:
            _check_root(x, degree)


def _check_power(x: Integer, base: Integer, exponent: Integer) raises:
    var found = perfect_power(x)
    assert_equal(found[0], base)
    assert_equal(found[1], exponent)


def test_perfect_power() raises:
    _check_power(Integer(-8) ** 3, -2, 9)
    _check_power((Integer(1) << 64) + 1, (Integer(1) << 64) + 1, 1)
    _check_power(1000, 10, 3)
    _check_power(-512, -2, 9)
    _check_power(-64, -4, 3)
    _check_power(64, 2, 6)
    _check_power(12, 12, 1)
    _check_power(Integer(1) << 210, 2, 210)
    _check_power(Integer(3) ** 77 * Integer(5) ** 77, 15, 77)
    for x in [-1, 0, 1]:
        _check_power(x, x, 1)
    # Bases that are not perfect powers keep their exponent.
    var bases: List[Integer] = [6, 10, 12, 18446744073709551617, -6, -10]
    for b in bases:
        for e in range(2, 41):
            if b > 0 or e % 2:
                _check_power(b**e, b, e)


def test_rational_roots_gcd_and_rounding() raises:
    assert_equal(root_exact(Rational(9, 4), 2).value(), Rational(3, 2))
    assert_equal(root_exact(Rational(-27, 8), 3).value(), Rational(-3, 2))
    assert_true(not root_exact(Rational(2), 2))
    assert_true(not root_exact(Rational(4, 3), 2))
    assert_equal(rational.gcd(Rational(1, 2), Rational(1, 3)), Rational(1, 6))
    assert_equal(rational.lcm(Rational(1, 2), Rational(1, 3)), Rational(1))
    assert_equal(rational.gcd(Rational(4, 9), Rational(6, 15)), Rational(2, 45))
    assert_equal(rational.gcd(Rational(-1, 2), Rational(1, 3)), Rational(1, 6))
    assert_equal(rational.gcd(Rational(0), Rational(0)), Rational(0))
    assert_equal(rational.gcd(Rational(0), Rational(-2, 3)), Rational(2, 3))
    assert_equal(rational.lcm(Rational(0), Rational(1, 2)), Rational(0))
    assert_equal(rational.lcm(Rational(4, 9), Rational(-6, 15)), Rational(4))
    var halves = [(5, 2), (-5, 2), (7, 2), (3, 2), (1, 2), (-1, 2), (-3, 2), (7, 3), (-7, 3), (5, 3), (4, 1)]
    var expected = [2, -2, 4, 2, 0, 0, -2, 2, -2, 2, 4]
    for i in range(len(halves)):
        assert_equal(Rational(halves[i][0], halves[i][1]).round(), Integer(expected[i]))


def test_is_prime_below_a_million() raises:
    var prime = _sieve(1000000)
    for n in range(1000000):
        assert_equal(is_prime(n), prime[n])
    assert_true(not is_prime(-7))
    # Composites that pass Miller-Rabin to bases 2 through 23, and through 37.
    assert_true(not is_prime(Integer.parse("3825123056546413051")))
    assert_true(not is_prime(Integer.parse("318665857834031151167461")))
    assert_true(is_prime(Integer.parse("18446744073709551557")))
    assert_true(not is_prime(Integer.parse("18446744073709551617")))


def test_is_prime_pseudoprimes_and_large_values() raises:
    for n in _fixture("strong_pseudoprimes_base_2"):
        assert_true(not is_prime(n))
    for n in _fixture("carmichael"):
        assert_true(not is_prime(n))
    # Every composite Mersenne number is a strong pseudoprime to base 2, so
    # the Lucas test decides these.
    var mersenne = _fixture("mersenne_prime_exponents")
    var exponents = _sieve(1300)
    for p in range(2, 1300):
        if exponents[p]:
            var expected = False
            for q in mersenne:
                expected = expected or q == p
            assert_equal(is_prime((Integer(1) << p) - 1), expected)
    var primes = _fixture("large_primes")
    for p in primes:
        assert_true(is_prime(p))
    for i in range(1, len(primes)):
        assert_true(not is_prime(primes[i - 1] * primes[i]))
    var verdicts = _fixture("large_odd_verdicts")
    for i in range(0, len(verdicts), 2):
        assert_equal(is_prime(verdicts[i]), verdicts[i + 1] == 1)


def test_baillie_psw_parts() raises:
    # Below 2**64, is_prime never reaches Baillie-PSW, so test it directly.
    var prime = _sieve(100000)
    for n in range(7, 100000, 2):
        assert_equal(_baillie_psw(n), prime[n])
    for n in _fixture("strong_lucas_pseudoprimes"):
        assert_true(_strong_lucas_probable_prime(n))
        assert_true(not _baillie_psw(n))


def test_jacobi() raises:
    var triples = _fixture("jacobi_triples")
    for i in range(0, len(triples), 3):
        assert_equal(jacobi(triples[i], triples[i + 1]), triples[i + 2])
    # Euler's criterion for primes: (a/p) is a**((p - 1) / 2) modulo p.
    for p in [3, 5, 7, 11, 101, 65537]:
        for a in range(-40, 40):
            var euler = integer.pow_mod(a, (p - 1) // 2, p)
            assert_equal(jacobi(a, p), Integer(-1) if euler == p - 1 else euler)
    with assert_raises(contains="odd positive modulus"):
        _ = jacobi(2, 8)
    with assert_raises(contains="odd positive modulus"):
        _ = jacobi(2, -3)


def test_number_theory_maps() raises:
    # trial_division returns a struct, which a batch cannot hold, so it stays
    # scalar; Optional results give the values and a Mask of where they exist.
    var values = Batch[Integer]([360, -72, 49, 1000003])
    var cofactors, counts = vmap[integer.remove_factor]()(values, 2)
    assert_equal(cofactors.to_list(), [45, -9, 49, 1000003])
    assert_equal(counts.to_list(), [3, 3, 0, 0])
    assert_equal(vmap[integer.is_prime]()(values).to_list(), [False, False, False, True])
    var bases, exponents = vmap[integer.perfect_power]()(values)
    assert_equal(bases.to_list(), [360, -72, 7, 1000003])
    assert_equal(exponents.to_list(), [1, 1, 2, 1])
    assert_equal(vmap[integer.jacobi]()(values, 7).to_list(), [-1, -1, 0, 1])
    var fractions = Batch[Rational]([Rational(1, 2), Rational(4, 9)])
    assert_equal(vmap[rational.gcd]()(fractions, Rational(1, 3)).to_list(), [Rational(1, 6), Rational(1, 9)])
    var roots, found = vmap[integer.iroot_exact]()(values, 2)
    assert_equal(roots.to_list(), [0, 0, 7, 0])
    assert_equal(found.to_list(), [False, False, True, False])
    assert_equal(roots[found].to_list(), [7])
    var fraction_roots, exact = vmap[rational.root_exact]()(fractions, 2)
    assert_equal(fraction_roots.to_list(), [Rational(0), Rational(2, 3)])
    assert_equal(exact.to_list(), [False, True])
    # A nested mapping over a matrix, with the degree shared by every call.
    var cubes = Batch[Integer]([8, 9, -27, 64, 10, -1]).reshape([2, 3])
    var cube_roots, cubic = vmap[integer.iroot_exact]().vmap()(cubes, 3)
    assert_equal(String(cube_roots), "[[2, 0, -3], [4, 0, -1]]")
    assert_equal(String(cubic), "[[True, False, True], [True, False, True]]")
    # A long mapping runs on the worker pool and must match element by element.
    var count = 30000
    var many = List[Integer](capacity=count)
    for i in range(count):
        many.append(Integer(i) * i + Integer(1 if i % 3 else 0))
    var square_roots, squares = vmap[integer.iroot_exact]()(Batch[Integer](many), 2)
    var root_list = square_roots.to_list()
    var bits = squares.to_list()
    for i in range(count):
        assert_equal(bits[i], i % 3 == 0)
        assert_equal(root_list[i], Integer(i if i % 3 == 0 else 0))
    with assert_raises(contains="vmap failed at mapped index 0"):
        _ = vmap[integer.iroot_exact]()(values, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
