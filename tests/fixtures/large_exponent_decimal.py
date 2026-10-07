"""Write tests/fixtures/large_exponent_decimal.txt with Arb (python-flint) as
the oracle.

Decimal conversions of Floats whose exponents are too large for exact powers
of ten, which apn_mojo bounds instead (float/_bounds10.mojo), and a few just
below that threshold, which take the exact path. MPFR, through gmpy2, keeps
exponents within its default 2**30 here; Arb's are unbounded, and its balls
are rigorous. Every answer below is decided from balls at a rising precision:
a comparison counts only when it is certain, so a wrong line would need a
wrong Arb enclosure. Each line is a case:

- `shortest p x digits exponent10`: the shortest decimal that reads back as x
  at p bits (nearest-even), and of those the closest to x;
- `digits p x n mode text`: x to n significant digits rounded with a mode, as
  `to_string(digits=n, notation="scientific")` writes it;
- `parse p mode text x`: decimal text read at p bits with a rounding mode.

x is `0x<hexadecimal significand>p<binary exponent>`, exact at p bits; values
use apn_mojo's default exponent range, which none of them leaves.

Run with the comparison environment:
pixi run --locked -e comparison python tests/fixtures/large_exponent_decimal.py
"""

import random
from pathlib import Path

import flint
from flint import arb, fmpz

OUT = Path(__file__).with_suffix(".txt")
SEED = 20261006
MODES = ["nearest_even", "toward_zero", "toward_positive", "toward_negative", "away_from_zero"]
PRECISIONS = [2, 11, 24, 53, 64, 113, 200]
# Binary exponents (x = 0.1... * 2**e, as apn_mojo counts them): around the
# threshold of the bounded path, then far beyond it in both directions.
EXPONENTS = [60000, 70000, 1000000, 123456789, 10**12, 2**61 - 12345,
             -60000, -70000, -1000000, -10**12, -(2**61)]


class Undecided(Exception):
    pass


def less(a, b):
    """Whether a < b, when the balls decide it."""
    if a < b:
        return True
    if a >= b:
        return False
    raise Undecided


def decided(function, *args):
    """function(*args) at 256, 512, ... bits, until every comparison it makes
    is certain."""
    precision = 256
    while precision < 1 << 16:
        flint.ctx.prec = precision
        try:
            return function(*args)
        except Undecided:
            precision *= 2
    raise AssertionError("undecided")


def ten(k):
    return arb(10) ** fmpz(k)


def dyadic(m, e):
    """The exact ball m * 2**e."""
    return arb(fmpz(m)) * arb(2) ** fmpz(e)


def to_int(ball):
    man, exp = ball.mid().man_exp()
    assert ball.rad() == 0 and exp >= 0
    return int(man) << int(exp)


def floor_of(y):
    """floor(y) for a ball y that is certainly not an integer."""
    f = y.floor()
    if f.rad() != 0 or not (f < y and y < f + 1):
        raise Undecided
    return to_int(f)


def log10_floor(x):
    """E with 10**E <= x < 10**(E + 1), for x > 0."""
    m, e = x.mid().man_exp()
    bits = int(e) + int(m).bit_length()
    # log10(2) to 33 digits: the estimate is within one of the answer.
    k = bits * 301029995663981195213738894724493 // 10**33 - 1
    while not less(x, ten(k + 1)):
        k += 1
    while less(x, ten(k)):
        k -= 1
    return k


def shortest(m, e, p):
    """The shortest decimal reading back as m * 2**e (m of p bits, positive)."""
    x = dyadic(m, e)
    # The rounding interval: half an ulp each side, a quarter below at a power of two.
    below = dyadic(1, e - 2) if m == 1 << (p - 1) else dyadic(1, e - 1)
    low, high = x - below, x + dyadic(1, e - 1)
    k = log10_floor(x)
    for d in range(1, p * 302 // 1000 + 4):
        unit = k - d + 1
        f = floor_of(x / ten(unit))
        inside = []
        for c in (f, f + 1):
            value = arb(fmpz(c)) * ten(unit)
            if less(low, value) and less(value, high):
                inside.append((c, value))
        if inside:
            if len(inside) == 2:
                inside = [inside[0] if less(abs(inside[0][1] - x), abs(inside[1][1] - x)) else inside[1]]
            c = inside[0][0]
            while c % 10 == 0:
                c //= 10
                unit += 1
            return c, unit
    raise AssertionError("no shortest decimal")


def rounds_up(mode, negative, beyond_half):
    if mode == "nearest_even":
        return beyond_half
    return mode == "away_from_zero" or (mode == "toward_positive" and not negative) or (
        mode == "toward_negative" and negative)


def scientific(m, e, negative, n, mode):
    x = dyadic(m, e)
    k = log10_floor(x)
    y = x / ten(k - n + 1)
    q = floor_of(y)
    if rounds_up(mode, negative, less(arb(fmpz(q)) + arb(1) / 2, y)):
        q += 1
    if q == 10**n:
        q, k = 10**(n - 1), k + 1
    digits = str(q)
    body = digits[0] + ("." + digits[1:] if n > 1 else "")
    return f"{'-' if negative else ''}{body}e{'+' if k >= 0 else '-'}{abs(k):02d}"


def parse(text, p, mode):
    """(negative, m, e): the text read at p bits, m * 2**e with m of p bits."""
    negative = text.startswith("-")
    mantissa, _, exponent = text.lstrip("-").partition("e")
    whole, _, fraction = mantissa.partition(".")
    v = arb(fmpz(int(whole + fraction))) * ten(int(exponent) - len(fraction))
    mid_m, mid_e = v.mid().man_exp()
    b = int(mid_e) + int(mid_m).bit_length() - 1
    while not less(v, dyadic(1, b + 1)):
        b += 1
    while less(v, dyadic(1, b)):
        b -= 1
    s = v * dyadic(1, p - 1 - b)
    q = floor_of(s)
    if rounds_up(mode, negative, less(arb(fmpz(q)) + arb(1) / 2, s)):
        q += 1
    if q == 1 << p:
        q, b = 1 << (p - 1), b + 1
    return negative, q, b - p + 1


def hex_text(negative, m, e):
    return f"{'-' if negative else ''}0x{m:x}p{e}"


def main():
    r = random.Random(SEED)
    lines = []
    for p in PRECISIONS:
        for top in EXPONENTS:
            for shape in ("random", "power_of_two"):
                m = r.getrandbits(p) | (1 << (p - 1)) if shape == "random" else 1 << (p - 1)
                negative = r.random() < 0.3
                e = top - p
                digits, exponent10 = decided(shortest, m, e, p)
                lines.append(f"shortest {p} {hex_text(negative, m, e)} {digits} {exponent10}")
                n = r.choice([1, 2, 3, 17, 25, 40])
                mode = r.choice(MODES)
                text = decided(scientific, m, e, negative, n, mode)
                lines.append(f"digits {p} {hex_text(negative, m, e)} {n} {mode} {text}")
    texts = ["1e1000000000000", "1e-1000000000000", "7.2e-123456789", "9.999999999999999e99999",
             "123456789012345678901234567890e700000", "1e100000000000000000", "-3e-100000000000000000",
             "5e-70000", "2.5e70000"]
    for _ in range(12):
        digits = "".join(r.choice("0123456789") for _ in range(r.randint(1, 60))).lstrip("0") or "7"
        exponent = r.choice([1, -1]) * r.randint(70000, 10**17)
        texts.append(f"{'-' if r.random() < 0.3 else ''}{digits}e{exponent}")
    for p in PRECISIONS:
        for text in texts:
            for mode in MODES:
                lines.append(f"parse {p} {mode} {text} {hex_text(*decided(parse, text, p, mode))}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
