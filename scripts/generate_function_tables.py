"""Write src/apn_mojo/ball/_tables.mojo: the tables of the medium-precision
kernels, whose sizes follow F. Johansson, "Efficient implementation of
elementary functions in the medium-precision range", ARITH 22, 2015.

Function tables: entry i of a table of L limbs is floor(f(i / 2**b) * 2**(64 L))
for a value below 1, certified by mpmath's interval arithmetic (a development
tool only, with outward rounding): an entry is written only when every point
of the enclosure has the same floor. mpmath has no interval atan, so atan
halves its argument, `atan x = 2 atan(x / (1 + sqrt(1 + x**2)))`, to below
2**-16 and sums the alternating Taylor series, whose next term bounds the
rest. cos(0) = 1 is stored as 2**(64 L) - 1, one unit
below. Each table is one string literal of hex digits, entries in order, each
entry's limbs most significant first, 16 digits to a limb: static data the
kernels read in place. (String literals encode bytes above 0x7f as UTF-8, so
raw bytes would not survive.)

Coefficient tables: 1/k! = numer[k] / denom[k]
and 1/(2k+1) = numer[k] / denom[k], with the terms in groups that share a
denominator: groups grow from k = 0 up while the product of the integers
(for 1/k!) or the lcm of the odd numbers (for 1/(2k+1)) stays below 2**64.
A Taylor sum then takes one word multiply-add per term and a word division
only between groups.

Run with `pixi run -e comparison python3 scripts/generate_function_tables.py`;
an argument names another output file.
"""

import math
import pathlib
import sys

import mpmath
from mpmath import iv, libmp

SHORT_LIMBS = 8      # 512 bits
LONG_LIMBS = 72     # 4608 bits

def interval_atan(x):
    """atan of an interval within [0, 1], with outward rounding throughout."""
    halvings = 0
    while x.b > mpmath.mpf(2) ** -16:
        x = x / (1 + iv.sqrt(1 + x * x))
        halvings += 1
    square = x * x
    total, term, k = iv.mpf(0), x, 0
    while term.b > mpmath.mpf(2) ** -(iv.prec + 8):
        total += term / (2 * k + 1) if k % 2 == 0 else -term / (2 * k + 1)
        term *= square
        k += 1
    # The alternating tail lies within the first omitted term.
    total += iv.mpf([-term.b, term.b])
    return total * 2**halvings


# name, limbs, count, step bits, function of an interval
TABLES = [
    ("_EXP_SHORT", SHORT_LIMBS, 178, 8, lambda x: iv.exp(x) / 2),
    ("_EXP_LONG_COARSE", LONG_LIMBS, 23, 5, lambda x: iv.exp(x) / 2),
    ("_EXP_LONG_FINE", LONG_LIMBS, 32, 10, lambda x: iv.exp(x) / 2),
    ("_LOG_SHORT_COARSE", SHORT_LIMBS, 128, 7, lambda x: iv.log(1 + x)),
    ("_LOG_SHORT_FINE", SHORT_LIMBS, 128, 14, lambda x: iv.log(1 + x)),
    ("_LOG_LONG_COARSE", LONG_LIMBS, 32, 5, lambda x: iv.log(1 + x)),
    ("_LOG_LONG_FINE", LONG_LIMBS, 32, 10, lambda x: iv.log(1 + x)),
    ("_ATAN_SHORT", SHORT_LIMBS, 256, 8, interval_atan),
    ("_ATAN_LONG_COARSE", LONG_LIMBS, 32, 5, interval_atan),
    ("_ATAN_LONG_FINE", LONG_LIMBS, 32, 10, interval_atan),
]
# sin and cos interleaved: entry 2i is sin(i / 2**b), entry 2i + 1 is cos(i / 2**b)
SIN_COS = [
    ("_SIN_COS_SHORT", SHORT_LIMBS, 203, 8),
    ("_SIN_COS_LONG_COARSE", LONG_LIMBS, 26, 5),
    ("_SIN_COS_LONG_FINE", LONG_LIMBS, 32, 10),
]
CONSTANTS = [
    ("_LN2_VALUE", lambda: iv.log(2)),
    ("_QUARTER_PI_VALUE", lambda: iv.pi / 4),
    ("_HALF_PI_MINUS_ONE_VALUE", lambda: iv.pi / 2 - 1),
    ("_EULER_GAMMA_VALUE", lambda: iv.euler),
]


def _floor_scaled(raw, shift):
    """floor(e * 2**shift) of an interval endpoint, exactly, from its raw
    (sign, mantissa, exponent): mpmath's own arithmetic on an endpoint would
    round to its default 53 bits."""
    sign, mantissa, exponent, _ = raw
    value = -mantissa if sign else mantissa
    e = exponent + shift
    return value << e if e >= 0 else value >> -e


def fixed(value_of, limbs):
    """floor(v 2**(64 limbs)) for 0 <= v < 1, or 2**(64 limbs) - 1 for v = 1."""
    iv.prec = 64 * limbs + 128
    v = value_of()
    if v._mpi_ == (libmp.fone, libmp.fone):
        return 2 ** (64 * limbs) - 1
    low, high = (_floor_scaled(e, 64 * limbs) for e in v._mpi_)
    if low != high:
        sys.exit("The enclosure of an entry spans a unit; raise the working precision.")
    assert 0 <= low < 2 ** (64 * limbs)
    return low


def hex_entries(values, limbs):
    return "".join(f"{v:0{16 * limbs}x}" for v in values)


def factorial_groups(size):
    numer, denom, b = [], [], 0
    while len(numer) < size:
        e, product = b, max(b, 1)
        while product * (e + 1) < 2**64:
            e += 1
            product *= e
        for k in range(b, e + 1):
            numer.append(math.prod(range(k + 1, e + 1)))
            denom.append(product)
        b = e + 1
    return numer[:size], denom[:size]


def odd_reciprocal_groups(size):
    numer, denom, b = [], [], 0
    while len(numer) < size:
        e, lcm = b, 2 * b + 1
        while math.lcm(lcm, 2 * e + 3) < 2**64:
            e += 1
            lcm = math.lcm(lcm, 2 * e + 1)
        for k in range(b, e + 1):
            numer.append(lcm // (2 * k + 1))
            denom.append(lcm)
        b = e + 1
    return numer[:size], denom[:size]


def words(values):
    return "".join(f"{v:016x}" for v in values)


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    lines = [
        '"""The tables of the medium-precision kernels (`_medium.mojo`); written by',
        "scripts/generate_function_tables.py, do not edit.",
        "",
        "A function table holds floor(f(i / 2**b) * 2**(64 L)) for entries of L limbs,",
        "most significant limb first, 16 hex digits to a limb. A coefficient table",
        'holds one 64-bit word per entry, 16 hex digits each."""',
        "",
        f"comptime _SHORT_LIMBS = {SHORT_LIMBS}",
        f"comptime _LONG_LIMBS = {LONG_LIMBS}",
        "",
    ]
    for name, limbs, count, bits, function in TABLES:
        values = [fixed(lambda i=i: function(iv.mpf(i) / 2**bits), limbs) for i in range(count)]
        lines.append(f"# {count} entries of {limbs} limbs at i / 2**{bits}")
        lines.append(f'comptime {name}: StaticString = "{hex_entries(values, limbs)}"')
    for name, limbs, count, bits in SIN_COS:
        values = []
        for i in range(count):
            values.append(fixed(lambda i=i: iv.sin(iv.mpf(i) / 2**bits), limbs))
            values.append(fixed(lambda i=i: iv.cos(iv.mpf(i) / 2**bits), limbs))
        lines.append(f"# sin and cos at i / 2**{bits}, interleaved: {count} pairs of {limbs} limbs")
        lines.append(f'comptime {name}: StaticString = "{hex_entries(values, limbs)}"')
    for name, value_of in CONSTANTS:
        lines.append(f'comptime {name}: StaticString = "{hex_entries([fixed(value_of, LONG_LIMBS)], LONG_LIMBS)}"')
    numer, denom = factorial_groups(288)
    lines.append("# 1/k! = numer[k] / denom[k], k < 288")
    lines.append(f'comptime _FACTORIAL_NUMER: StaticString = "{words(numer)}"')
    lines.append(f'comptime _FACTORIAL_DENOM: StaticString = "{words(denom)}"')
    lines.append("comptime _FACTORIAL_SIZE = 288")
    numer, denom = odd_reciprocal_groups(256)
    lines.append("# 1/(2k+1) = numer[k] / denom[k], k < 256")
    lines.append(f'comptime _ODD_NUMER: StaticString = "{words(numer)}"')
    lines.append(f'comptime _ODD_DENOM: StaticString = "{words(denom)}"')
    lines.append("comptime _ODD_SIZE = 256")
    # Upper bounds of log2(1/n!) for n < 600: -floor(log2(n!)).
    bounds = [-(math.factorial(n).bit_length() - 1) for n in range(600)]
    lines.append("# -floor(log2(n!)) for n < 600, an upper bound of log2(1/n!), as 16-bit two's complement")
    lines.append(f'comptime _RECIPROCAL_FACTORIAL_BITS: StaticString = "{"".join(f"{b & 0xFFFF:04x}" for b in bounds)}"')
    lines.append("")
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / "src" / "apn_mojo" / "ball" / "_tables.mojo"
    target.write_text("\n".join(lines))
    print(f"wrote {target} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
