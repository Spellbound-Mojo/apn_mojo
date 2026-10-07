"""Write src/apn_mojo/ball/_rgamma_table.mojo: the Taylor coefficients of the
reciprocal gamma function, `1/Gamma(1 + u) = sum_n b_n u**n`, for Gamma at
low and medium precision by the local method of Johansson, "Arbitrary-
precision computation of the gamma function" (2021), section 5.1.

`1/Gamma(z) = sum_{n>=1} a_n z**n` has `a_1 = 1` and, for n >= 2 (DLMF 5.7.1;
Wrench 1968), `(n-1) a_n = gamma a_{n-1} - zeta(2) a_{n-2} + zeta(3) a_{n-3}
- ... + (-1)**n zeta(n-1) a_1`; then `b_n = a_{n+1}`. The recurrence cancels
about `-log2 |a_n|` bits, so it runs in mpmath at two working precisions far
above that, and the table's entries, `round toward zero (|b_n| 2**F)` with
F = 64 LIMBS, must agree between them. The truncated series is also checked
against mpmath's rgamma at sample points of [-1/2, 1/2].

Each entry is LIMBS limbs of 16 hex digits, most significant first; the
signs are one character per entry.

Run in the comparison environment:
    pixi run -e comparison python3 scripts/generate_rgamma_table.py
"""

import pathlib
import sys

import mpmath

LIMBS = 20
"""Entries at 1280 bits: the method serves working precisions up to 1256
bits, which a 1024-bit Float's guard bits, a reflection and a retry reach;
Stirling's series beyond."""

FRACTION = 64 * LIMBS

OUT = pathlib.Path(__file__).resolve().parent.parent / "src" / "apn_mojo" / "ball" / "_rgamma_table.mojo"


def coefficients(count, precision):
    """b_0 ... b_{count-1} at `precision` bits."""
    with mpmath.workprec(precision):
        gamma = mpmath.euler
        zetas = [None, None] + [mpmath.zeta(k) for k in range(2, count + 2)]
        a = [mpmath.mpf(0), mpmath.mpf(1)]
        for n in range(2, count + 2):
            total = gamma * a[n - 1]
            for k in range(2, n):
                term = zetas[k] * a[n - k]
                total += -term if k % 2 == 0 else term
            a.append(total / (n - 1))
        return [a[n + 1] for n in range(count)]


def entries(b, precision):
    """The rounded magnitudes and signs, converted at the coefficients' own
    precision: at mpmath's default 53 bits the products would keep 53 bits."""
    out = []
    with mpmath.workprec(precision):
        for value in b:
            magnitude = int(mpmath.floor(abs(value) * mpmath.mpf(2) ** FRACTION))
            out.append((magnitude, value < 0))
    return out


def main():
    # Enough terms that |b_N| 2**-N is far below 2**-F.
    count = 300
    first = entries(coefficients(count, 2 * FRACTION + 1600), 2 * FRACTION + 1600)
    second = entries(coefficients(count, 2 * FRACTION + 2400), 2 * FRACTION + 2400)
    assert first == second, "the table depends on the working precision"
    while count > 1 and first[count - 1][0] == 0:
        count -= 1
    table = first[:count]
    # The truncated series against rgamma at sample points of [-1/2, 1/2].
    with mpmath.workprec(FRACTION + 256):
        for u in ("-0.5", "-0.3", "-0.0625", "0.125", "0.3", "0.5"):
            x = mpmath.mpf(u)
            series = sum((-m if negative else m) * x ** n for n, (m, negative) in enumerate(table)) / mpmath.mpf(2) ** FRACTION
            gap = abs(series - mpmath.rgamma(1 + x)) * mpmath.mpf(2) ** FRACTION
            assert gap < 2 * count, f"the series misses rgamma(1 + {u}) by {gap} units"
    # b_0 = 1 is implicit, the only coefficient not below 1; its entry is 0.
    assert table[0] == (2 ** FRACTION, False)
    assert all(m < 2 ** FRACTION for m, _ in table[1:])
    body = "".join(f"{m:0{16 * LIMBS}x}" for m, _ in [(0, False)] + table[1:])
    assert len(body) == 16 * LIMBS * count
    # Each entry's bit length, 4 hex digits: the term count's search reads one.
    lengths = "".join(f"{m.bit_length():04x}" for m, _ in [(0, False)] + table[1:])
    signs = "".join("-" if negative else "+" for _, negative in table)
    OUT.write_text(f'''"""The Taylor coefficients of `1/Gamma(1 + u) = sum_n b_n u**n`, b_0 ... b_{count - 1},
at {FRACTION} bits: each entry is `|b_n| 2**{FRACTION}` rounded toward zero, {LIMBS}
limbs of 16 hex digits, most significant first; `_RGAMMA_SIGNS` holds each
sign. `b_0 = 1`, the only coefficient not below 1, is implicit: its entry is
0. Written by scripts/generate_rgamma_table.py, do not edit."""

comptime _RGAMMA_COUNT = {count}

comptime _RGAMMA_LIMBS = {LIMBS}

comptime _RGAMMA_SIGNS: StaticString = "{signs}"

comptime _RGAMMA_BITS: StaticString = "{lengths}"
"""Each entry's bit length, 4 hex digits."""

comptime _RGAMMA_MAGNITUDES: StaticString = "{body}"
''')
    print(f"wrote {OUT}: {count} coefficients, {len(body) // 1024} KiB")


if __name__ == "__main__":
    sys.exit(main())
