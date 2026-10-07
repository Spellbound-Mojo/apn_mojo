"""Factorization and primes (T-NT2): factor within a budget, primes_below
and next_prime."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, is_prime, factor, FactorBudget, Factorization, FactorTerm, primes_below, next_prime


def _product(f: Factorization) raises -> Integer:
    var value = Integer(f.sign())
    for term in f.terms():
        value *= term.base ** term.multiplicity
    return value


def test_known_factorizations() raises:
    var fermat = factor((Integer(1) << 64) + 1)
    var terms = fermat.terms()
    assert_equal(len(terms), 2)
    assert_true(terms[0].base == 274177 and terms[1].base == Integer(67280421310721))
    assert_true(fermat.is_complete() and terms[0].multiplicity == 1)
    var negative = factor(Integer(-360))
    assert_equal(negative.sign(), -1)
    assert_equal(String(negative), "-1 * 2^3 * 3^2 * 5")
    assert_true(factor(Integer(1)).is_complete() and len(factor(Integer(1)).terms()) == 0)
    assert_equal(String(factor(Integer(1))), "1")
    with assert_raises(contains="Cannot factor 0"):
        _ = factor(Integer(0))
    # A perfect power of a large prime reduces to its base.
    var p = next_prime(Integer(1) << 80)
    var power = factor(p ** 7)
    assert_true(len(power.terms()) == 1 and power.terms()[0].multiplicity == 7 and power.terms()[0].base == p)
    with assert_raises(contains="elliptic-curve"):
        _ = factor(Integer(91), budget=FactorBudget(ecm_curves=3))
    with assert_raises(contains="trial bound"):
        _ = factor(Integer(91), budget=FactorBudget(trial_bound=1))


def test_random_products() raises:
    # Products of random primes up to 40 bits, with multiplicities, factor
    # completely and multiply back within the default budget.
    var state = UInt64(0x853C49E6748FEA9B)
    for trial in range(24):
        var n = Integer(1)
        var expected = List[Integer]()
        var count = 2 + trial % 3
        for _ in range(count):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var prime = next_prime(Integer(Int(state >> 24)))
            var multiplicity = 1 + Int(state % 3)
            n *= prime ** multiplicity
            expected.append(prime)
        var f = factor(n)
        assert_true(f.is_complete(), String("incomplete: ", n))
        assert_true(_product(f) == n, String("product: ", n))
        for term in f.terms():
            assert_true(is_prime(term.base))
        var bases = f.terms()
        for i in range(1, len(bases)):
            assert_true(bases[i - 1].base < bases[i].base)


def test_budget_exhaustion() raises:
    # Two 128-bit primes do not split within a small budget: the part is
    # reported, not as prime, and the factorization is incomplete.
    var p = next_prime(Integer(0x1234567890ABCDEF) << 64)
    var q = next_prime(Integer(0x7EDCBA0987654321) << 64)
    var f = factor(p * q, budget=FactorBudget(rho_iterations=1 << 12))
    assert_false(f.is_complete())
    var terms = f.terms()
    assert_true(len(terms) == 1 and terms[0].base == p * q and not terms[0].is_prime)
    assert_true(String(f).endswith("?"))


def test_primes() raises:
    var small = primes_below(30)
    var expected: List[Int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert_equal(len(small), len(expected))
    for i in range(len(expected)):
        assert_equal(small[i], expected[i])
    assert_equal(len(primes_below(1000000)), 78498)
    assert_equal(len(primes_below(2)), 0)
    assert_equal(len(primes_below(3)), 1)
    var big = primes_below(100003)
    assert_equal(big[len(big) - 1], 99991)
    assert_true(next_prime(Integer(-5)) == 2 and next_prime(Integer(2)) == 3 and next_prime(Integer(3)) == 5)
    assert_true(next_prime(Integer(13)) == 17 and next_prime(Integer(29)) == 31)
    assert_true(next_prime(Integer(10) ** 20) == Integer(10) ** 20 + 39)
    assert_true(next_prime(Integer(1) << 64) == (Integer(1) << 64) + 13)
    for i in range(1, len(big)):
        assert_true(next_prime(Integer(big[i - 1])) == big[i])
        if i > 500:
            break


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
