"""Write tests/fixtures/number_theory.txt from gmpy2 (MPIR/GMP), the oracle.

Run in the comparison environment: pixi run -e comparison python3
tests/fixtures/number_theory.py. Each line is a name and its decimal values.
"""

from pathlib import Path
import random

import gmpy2

OUT = Path(__file__).with_suffix(".txt")


def composite_odd(limit):
    return (n for n in range(9, limit, 2) if not gmpy2.is_prime(n))


def is_carmichael(n):
    # Korselt: n squarefree, and p - 1 divides n - 1 for every prime p | n.
    rest, p = n, 3
    while p * p <= rest:
        if rest % p == 0:
            rest //= p
            if rest % p == 0 or (n - 1) % (p - 1):
                return False
        p += 2
    return rest != n and (n - 1) % (rest - 1) == 0


def main():
    rng = random.Random(20261004)
    rows = {}
    rows["strong_pseudoprimes_base_2"] = [n for n in composite_odd(10**7) if gmpy2.is_strong_prp(n, 2)]
    rows["carmichael"] = [n for n in range(561, 10**6, 2) if is_carmichael(n)]
    rows["strong_lucas_pseudoprimes"] = [
        n for n in composite_odd(10**6) if not gmpy2.is_square(n) and gmpy2.is_strong_selfridge_prp(n)]
    rows["mersenne_prime_exponents"] = [
        p for p in range(2, 1300) if gmpy2.is_prime(p) and gmpy2.is_prime(2**p - 1, 50)]
    primes = []
    for bits in (64, 65, 96, 128, 256, 512, 1024, 2048, 4096):
        for _ in range(3):
            primes.append(int(gmpy2.next_prime(rng.getrandbits(bits) | (1 << (bits - 1)))))
    rows["large_primes"] = primes
    # Odd numbers just above each large prime, with gmpy2's verdict.
    rows["large_odd_verdicts"] = [v for p in primes for n in (p + 2, p + 4) for v in (n, int(gmpy2.is_prime(n, 50)))]
    jacobi = []
    for _ in range(40):
        n = rng.getrandbits(rng.choice([65, 200, 700])) | 1
        a = rng.getrandbits(rng.choice([10, 300, 900])) * rng.choice([1, -1])
        jacobi += [a, n, int(gmpy2.jacobi(a, n))]
    rows["jacobi_triples"] = jacobi
    OUT.write_text("".join(f"{name} {' '.join(map(str, values))}\n" for name, values in rows.items()))
    for name, values in rows.items():
        print(name, len(values))


if __name__ == "__main__":
    main()
