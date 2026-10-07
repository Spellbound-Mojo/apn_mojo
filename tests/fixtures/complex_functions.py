"""Write tests/fixtures/complex_functions.txt from MPC (gmpy2), the oracle.

Each line is

    function p_re p_im mode_re mode_im z... result_re result_im

with each complex argument as two parts `sign significand exponent precision`
(value `significand * 2**(exponent - precision)`, or `zero`/`inf`/`nan`), and
each result part as `kind sign significand exponent` (kind 0 zero, 1 finite,
2 infinity, 3 NaN). `arg` has one real result, written in both result slots.
Modes are 0 nearest_even, 1 toward_zero, 2 toward_positive and 3
toward_negative, set per part as MPC does; gmpy2's mpc has no
away_from_zero. Its results keep MPFR's default exponent range,
[1 - 2**30, 2**30 - 1], whatever the context says, so the tests use that range. MPC's flags are
shared by both parts, so only values are recorded.

Arguments: general points with parts in [-4, 4] and +-2**U for U in [-40, 40],
points on the axes with signed zeros, and points on and near the branch cuts.
Appendix B and C rows follow at 53 bits, then every pair of parts from
{+-0, +-2, +-inf, NaN} for each function, and the powers of 1, -1, i and -i
that MPC decides structurally.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/complex_functions.py
"""

import random
from pathlib import Path

import gmpy2
from gmpy2 import mpc, mpfr

OUT = Path(__file__).with_suffix(".txt")
MODES = [gmpy2.RoundToNearest, gmpy2.RoundToZero, gmpy2.RoundUp, gmpy2.RoundDown, gmpy2.RoundAwayZero]
FUNCTIONS = {
    "exp": gmpy2.exp, "log": gmpy2.log, "sin": gmpy2.sin, "cos": gmpy2.cos, "tan": gmpy2.tan,
    "sinh": gmpy2.sinh, "cosh": gmpy2.cosh, "tanh": gmpy2.tanh, "asin": gmpy2.asin, "acos": gmpy2.acos,
    "atan": gmpy2.atan, "asinh": gmpy2.asinh, "acosh": gmpy2.acosh, "atanh": gmpy2.atanh,
}


def encode(x):
    sign = "-" if gmpy2.is_signed(x) else "+"
    if gmpy2.is_nan(x):
        return "+ nan 0 1"
    if gmpy2.is_infinite(x):
        return f"{sign} inf 0 1"
    if x == 0:
        return f"{sign} zero 0 1"
    m, e = x.as_mantissa_exp()
    m = abs(int(m))
    q = m.bit_length()
    return f"{sign} {m:x} {int(e) + q} {q}"


def result(y, p):
    if gmpy2.is_nan(y):
        return "3 + 0 0"
    sign = "-" if gmpy2.is_signed(y) else "+"
    if gmpy2.is_infinite(y):
        return f"2 {sign} 0 0"
    if y == 0:
        return f"0 {sign} 0 0"
    m, e = y.as_mantissa_exp()
    m = abs(int(m))
    shift = p - m.bit_length()
    return f"1 {sign} {m << shift:x} {int(e) - shift + p}"


def real(rng, low, high):
    q = rng.randrange(1, 100)
    m = rng.getrandbits(q) | (1 << (q - 1))
    e = rng.randrange(low, high + 1)
    with gmpy2.context(precision=q):
        x = gmpy2.mul_2exp(mpfr(m), e - q)
    return -x if rng.random() < 0.5 else x


def signed_zero(rng):
    return -mpfr(0) if rng.random() < 0.5 else mpfr(0)


def argument(name, rng, kind):
    if kind == 0:
        return real(rng, -2, 3), real(rng, -2, 3)
    if kind == 1:
        return real(rng, -40, 40), real(rng, -40, 40)
    if kind == 2:
        # On an axis.
        if rng.random() < 0.5:
            return real(rng, -10, 4), signed_zero(rng)
        return signed_zero(rng), real(rng, -10, 4)
    # On or near a cut.
    with gmpy2.context(precision=200):
        big = abs(real(rng, 1, 4)) + 1
        near = mpfr(2) ** -rng.randrange(20, 120)
    if name in ("log", "acosh"):
        return -big, signed_zero(rng) if rng.random() < 0.5 else near
    if name in ("asin", "acos", "atanh"):
        x = big if rng.random() < 0.5 else -big
        return x, signed_zero(rng) if rng.random() < 0.5 else near
    if name in ("atan", "asinh"):
        y = big if rng.random() < 0.5 else -big
        return signed_zero(rng) if rng.random() < 0.5 else near, y
    return real(rng, -2, 3), signed_zero(rng)


def precision(rng):
    return rng.randrange(2, 129) if rng.random() < 0.8 else rng.randrange(129, 513)


def emit(lines, name, pr, pi, mr, mi, args):
    # `round` rounds mpfr results such as phase; real_round and imag_round, mpc's.
    with gmpy2.context(precision=pr, round=MODES[mr], real_prec=pr, imag_prec=pi, real_round=MODES[mr], imag_round=MODES[mi],
                       emin=gmpy2.get_emin_min(), emax=gmpy2.get_emax_max()):
        if name == "pow":
            value = args[0] ** args[1]
        elif name == "arg":
            value = gmpy2.phase(args[0])
        else:
            value = FUNCTIONS[name](args[0])
    encoded = " ".join(encode(a.real) + " " + encode(a.imag) for a in args)
    if name == "arg":
        text = result(value, pr)
        lines.append(f"{name} {pr} {pi} {mr} {mi} {encoded} {text} {text}\n")
    else:
        lines.append(f"{name} {pr} {pi} {mr} {mi} {encoded} {result(value.real, pr)} {result(value.imag, pi)}\n")


def complex_of(re, im):
    with gmpy2.context(precision=400):
        return mpc(re, im)


def main():
    rng = random.Random(20261005)
    lines = []
    for name in FUNCTIONS:
        for i in range(60):
            re, im = argument(name, rng, i % 4)
            z = complex_of(re, im)
            p = precision(rng)
            for _ in range(3):
                pr = p if rng.random() < 0.8 else precision(rng)
                emit(lines, name, pr, p, rng.randrange(4), rng.randrange(4), [z])
    for i in range(120):
        z = complex_of(real(rng, -2, 3), real(rng, -2, 3))
        w = complex_of(real(rng, -3, 2), real(rng, -3, 2) if i % 3 else mpfr(0))
        p = precision(rng)
        emit(lines, "pow", p, p, rng.randrange(4), rng.randrange(4), [z, w])
    for i in range(100):
        re, im = argument("log", rng, i % 4)
        p = precision(rng)
        emit(lines, "arg", p, p, rng.randrange(4), 0, [complex_of(re, im)])
    # Appendix B and C rows at 53 bits, every mode pair on the diagonal.
    inf = mpfr("inf")
    nan = mpfr("nan")
    rows = {
        "exp": [(0, 0), (-mpfr(0), -mpfr(0)), (1, inf), (inf, 0), (-inf, 2), (inf, 2), (-inf, inf), (inf, inf),
                (nan, 0), (nan, -mpfr(0)), (3, -mpfr(0)), (0, 2)],
        "log": [(-mpfr(0), 0), (0, 0), (2, inf), (-inf, 3), (inf, 3), (-inf, inf), (inf, inf), (-1, 0), (-1, -mpfr(0)),
                (1, -mpfr(0)), (0, 1), (-2, 0), (-2, -mpfr(0))],
        "asin": [(2, 0), (2, -mpfr(0)), (-2, 0)],
        "acos": [(2, -mpfr(0)), (-2, 0), (2, 0), (0, 0)],
        "atan": [(0, 2), (-mpfr(0), 2), (-mpfr(0), -2), (0, -2)],
        "asinh": [(0, 2), (-mpfr(0), -2), (0, -2)],
        "acosh": [(-2, 0), (mpfr("0.5"), 0), (-2, -mpfr(0))],
        "atanh": [(2, 0), (2, -mpfr(0)), (-2, 0), (1, 0)],
        "sinh": [(0, 0), (inf, 0), (0, inf), (inf, 2), (nan, 0)],
        "cosh": [(0, 0), (inf, 0), (0, inf), (inf, 2), (nan, 0)],
        "tanh": [(0, 0), (inf, 2), (inf, inf), (nan, 0), (0, 2)],
        "sin": [(0, 0), (2, 0), (0, 2)],
        "cos": [(0, 0), (2, 0), (0, 2)],
        "tan": [(0, 0), (2, 0), (0, 2)],
    }
    for name, cases in rows.items():
        for re, im in cases:
            for mode in range(4):
                emit(lines, name, 53, 53, mode, mode, [complex_of(mpfr(re), mpfr(im))])
    for base, power in [((-4, 0), (mpfr("0.25"), 0)), ((-2, 0), (mpfr("0.5"), 0)), ((0, 1), (0, 1)),
                        ((-1, 0), (mpfr("0.5"), 1)), ((2, 0), (3, 0)), ((0, 2), (2, 0))]:
        for mode in range(4):
            emit(lines, "pow", 53, 53, mode, mode, [complex_of(mpfr(base[0]), mpfr(base[1])),
                                                     complex_of(mpfr(power[0]), mpfr(power[1]))])
    # Special values: every pair of parts from S, nearest and toward -inf.
    S = [mpfr(0), -mpfr(0), mpfr(2), mpfr(-2), inf, -inf, nan]
    for name in FUNCTIONS:
        for re in S:
            for im in S:
                for mode in (0, 3):
                    emit(lines, name, 53, 53, mode, mode, [complex_of(re, im)])
    # Powers of 1, -1, i and -i with signed zeros, which MPC's pow decides structurally.
    unit = [(1, 0), (1, -0.0), (-1, 0), (-1, -0.0), (0, 1), (-0.0, 1), (0, -1), (-0.0, -1)]
    powers = [(0, 1), (0, -1), (-0.0, 1), (1, 1), (2, 1), (-1, 1), (0.5, 1), (0.5, -1), (1.5, 1), (0, 0.5)]
    for base in unit:
        for power in powers:
            for mode in range(4):
                emit(lines, "pow", 53, 53, mode, mode, [complex_of(mpfr(base[0]), mpfr(base[1])),
                                                         complex_of(mpfr(power[0]), mpfr(power[1]))])
    OUT.write_text("".join(lines))
    print(len(lines), "lines", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
