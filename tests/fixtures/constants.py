"""Write tests/fixtures/constants.txt from MPFR (gmpy2), the oracle.

Each line is `name precision exponent down nearest_up`: the constant rounded
down to `precision` bits as a hexadecimal significand of exactly `precision`
bits with apn_mojo's exponent (value = significand * 2**(exponent -
precision)), and 1 when rounding to nearest goes up. The constants are positive
and irrational, so toward_zero equals toward_negative (down) and
away_from_zero equals toward_positive (one unit above down).

Precisions: the H.1 table's 64 and 128, small ones, and seeded random ones from
2 to 4096; then pi, e and ln 2 at 100000 bits and gamma at 20000 bits, to
nearest only.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/constants.py
"""

import random
from pathlib import Path

import gmpy2
from gmpy2 import mpfr

OUT = Path(__file__).with_suffix(".txt")
CONSTANTS = {
    "pi": gmpy2.const_pi,
    "euler_e": lambda: gmpy2.exp(mpfr(1)),
    "ln2": gmpy2.const_log2,
    "log2_10": lambda: gmpy2.log2(mpfr(10)),
    "euler_gamma": gmpy2.const_euler,
    "catalan": gmpy2.const_catalan,
}


def triple(x, precision):
    m, e = x.as_mantissa_exp()
    m = int(m)
    shift = precision - m.bit_length()
    return m << shift, int(e) - shift + precision


def rounded(name, precision, mode):
    with gmpy2.context(precision=precision, round=mode, emin=gmpy2.get_emin_min(), emax=gmpy2.get_emax_max()):
        return triple(CONSTANTS[name](), precision)


def main():
    rng = random.Random(20261005)
    precisions = [2, 3, 4, 5, 8, 24, 53, 64, 113, 128, 256]
    precisions += sorted(rng.randrange(2, 400) for _ in range(60))
    precisions += sorted(rng.randrange(400, 4097) for _ in range(40))
    lines = []
    for name in CONSTANTS:
        for p in precisions:
            down = rounded(name, p, gmpy2.RoundDown)
            near = rounded(name, p, gmpy2.RoundToNearest)
            lines.append(f"{name} {p} {down[1]} {down[0]:x} {int(near != down)}\n")
    for name, p in [("pi", 100000), ("euler_e", 100000), ("ln2", 100000), ("euler_gamma", 20000)]:
        near = rounded(name, p, gmpy2.RoundToNearest)
        lines.append(f"near {name} {p} {near[1]} {near[0]:x}\n")
    OUT.write_text("".join(lines))
    print(len(lines), "lines")


if __name__ == "__main__":
    main()
