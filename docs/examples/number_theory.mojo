"""Factors, exact roots, perfect powers and primes, on scalars and batches."""

from apn_mojo import Batch, Integer, Rational, integer, rational, vmap
from apn_mojo import remove_factor, trial_division, iroot_exact, perfect_power
from apn_mojo import is_prime, jacobi, root_exact


def main() raises:
    var cofactor, count = remove_factor(-72, 2)
    print("-72 without 2s:", cofactor, "times 2 **", count)
    print(trial_division(360, 4))
    var division = trial_division(Integer.parse("123456789012345678901234567890"), 1000)
    print("primes below 1000:", len(division.factors()), "cofactor:", division.cofactor())
    print("cube root of -27:", iroot_exact(-27, 3).value())
    print("2 is a square:", Bool(iroot_exact(2, 2)))
    var base, exponent = perfect_power(-512)
    print("-512 =", base, "**", exponent)
    print("2**127 - 1 is prime:", is_prime((Integer(1) << 127) - 1))
    print("primes:", vmap[integer.is_prime]()(Batch[Integer].from_native([97, 561, 7919])))
    print("jacobi(2, 15):", jacobi(2, 15))
    print("root of 9/4:", root_exact(Rational(9, 4), 2).value())
    print("gcd(1/2, 1/3):", rational.gcd(Rational(1, 2), Rational(1, 3)))
    print("5/2 rounds to", Rational(5, 2).round())
