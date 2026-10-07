"""Write src/apn_mojo/ball/_bernoulli_table.mojo: the tangent numbers
T_1 ... T_K, from which Stirling's series takes its Bernoulli numbers,
`B_2k = (-1)**(k-1) 2k T_k / (4**k (4**k - 1))`.

The numbers come from Brent and Harvey's recurrence ("Fast computation of
Bernoulli, tangent and secant numbers", 2011), the one the library runs past
the table. Each is then checked two independent ways: against the Bernoulli
numbers of the exact recurrence `sum_{j<=n} C(n+1, j) B_j = 0` for k <= 64,
and for every k against the rounding of `2 (2k)! zeta(2k) / (2 pi)**2k`
times `4**k (4**k - 1) / 2k`, evaluated by mpmath with 64 bits beyond T_k's
length.

The file holds each number as 16-hex-digit limbs, most significant first,
and the limb count of each entry as one 16-digit word.

Run in the comparison environment:
    pixi run -e comparison python3 scripts/generate_bernoulli_table.py
"""

import math
import pathlib
import sys
from fractions import Fraction

import mpmath

COUNT = 256
"""Stirling's series takes `w/8 + 16` terms, so the table covers working
precisions up to `8 (COUNT - 16)` bits, 1920."""

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "apn_mojo" / "ball" / "_bernoulli_table.mojo"


def tangent_numbers(count):
    """T_1 ... T_count by Brent and Harvey's recurrence."""
    t = [0] * count
    t[0] = 1
    for k in range(2, count + 1):
        t[k - 1] = t[k - 2] * (k - 1)
    for k in range(2, count + 1):
        for j in range(k, count + 1):
            t[j - 1] = t[j - 2] * (j - k) + t[j - 1] * (j - k + 2)
    return t


def bernoulli_exact(n):
    """B_0 ... B_n from `sum_{j<=m} C(m+1, j) B_j = 0`."""
    b = [Fraction(1)]
    for m in range(1, n + 1):
        b.append(-sum(math.comb(m + 1, j) * b[j] for j in range(m)) / (m + 1))
    return b


def check(t):
    b = bernoulli_exact(128)
    for k in range(1, 65):
        four = 4 ** k
        expected = Fraction((-1) ** (k - 1) * 2 * k * t[k - 1], four * (four - 1))
        assert b[2 * k] == expected, f"T_{k} disagrees with the exact Bernoulli number"
    for k in range(1, len(t) + 1):
        bits = t[k - 1].bit_length() + 64
        with mpmath.workprec(bits + 2 * k.bit_length() + 64):
            value = 2 * mpmath.factorial(2 * k) * mpmath.zeta(2 * k) / (2 * mpmath.pi) ** (2 * k)
            value *= mpmath.mpf(4 ** k) * (4 ** k - 1) / (2 * k)
            assert int(mpmath.nint(value)) == t[k - 1], f"T_{k} disagrees with the zeta formula"
            assert abs(value - t[k - 1]) < mpmath.mpf(1) / 4, f"T_{k} is not certified by the zeta formula"


def limbs(n):
    words = []
    while n:
        words.append(n & (2 ** 64 - 1))
        n >>= 64
    return list(reversed(words)) or [0]


def main():
    t = tangent_numbers(COUNT)
    check(t)
    entries = [limbs(n) for n in t]
    body = "".join(f"{w:016x}" for entry in entries for w in entry)
    sizes = "".join(f"{len(entry):016x}" for entry in entries)
    OUT.write_text(f'''"""The tangent numbers T_1 ... T_{COUNT} of Stirling's series, for its
Bernoulli numbers `B_2k = (-1)**(k-1) 2k T_k / (4**k (4**k - 1))`; written by
scripts/generate_bernoulli_table.py, do not edit.

`_TANGENT_LIMBS` holds each number as 16-hex-digit limbs, most significant
first, entry after entry; `_TANGENT_SIZES` holds each entry's limb count as
one 16-digit word."""

comptime _TANGENT_COUNT = {COUNT}

comptime _TANGENT_SIZES: StaticString = "{sizes}"

comptime _TANGENT_LIMBS: StaticString = "{body}"
''')
    print(f"wrote {OUT}: {COUNT} numbers, {sum(map(len, entries))} limbs, {len(body) // 1024} KiB")


if __name__ == "__main__":
    sys.exit(main())
