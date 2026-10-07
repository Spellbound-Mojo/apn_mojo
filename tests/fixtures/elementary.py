"""Write tests/fixtures/elementary.txt from MPFR (gmpy2), the oracle.

Each line is

    function p mode n x... result flags

with `x` one or two arguments, each `sign significand exponent precision`
(value `significand * 2**(exponent - precision)`, significand hexadecimal of
exactly `precision` bits, or `zero`/`inf`/`nan` with a sign), `n` the degree
of rootn (0 otherwise), `result` as `kind sign significand exponent` (kind 0
zero, 1 finite, 2 infinity, 3 NaN) and `flags` five digits: inexact,
underflow, overflow, divide-by-zero, invalid. As in apn_mojo, a NaN input
gives no invalid flag. Modes: 0 nearest_even, 1 toward_zero, 2
toward_positive, 3 toward_negative, 4 away_from_zero.

Arguments are drawn per function from [-4, 4], from +-2**U with U in
[-200, 200], and near the domain's boundaries; output precisions from 2 to
4096, mostly small. Special and exact inputs (Appendix A and D.1) follow at 53
bits.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/elementary.py
"""

import random
from pathlib import Path

import gmpy2
from gmpy2 import mpfr

OUT = Path(__file__).with_suffix(".txt")
MODES = [gmpy2.RoundToNearest, gmpy2.RoundToZero, gmpy2.RoundUp, gmpy2.RoundDown, gmpy2.RoundAwayZero]
UNARY = {
    "exp": gmpy2.exp, "expm1": gmpy2.expm1, "exp2": gmpy2.exp2, "log": gmpy2.log, "log1p": gmpy2.log1p,
    "log2": gmpy2.log2, "log10": gmpy2.log10, "sin": gmpy2.sin, "cos": gmpy2.cos, "tan": gmpy2.tan,
    "atan": gmpy2.atan, "asin": gmpy2.asin, "acos": gmpy2.acos, "sinh": gmpy2.sinh, "cosh": gmpy2.cosh,
    "tanh": gmpy2.tanh, "asinh": gmpy2.asinh, "acosh": gmpy2.acosh, "atanh": gmpy2.atanh,
}


def context(p, mode):
    return gmpy2.context(precision=p, round=mode, emin=gmpy2.get_emin_min(), emax=gmpy2.get_emax_max(),
                         subnormalize=False)


def encode(x):
    """`sign significand exponent precision` of an exact mpfr input."""
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


def flags(ctx, nan_input):
    bits = [ctx.inexact, ctx.underflow, ctx.overflow, ctx.divzero, ctx.invalid and not nan_input]
    return "".join("1" if b else "0" for b in bits)


def make(rng, low_exp, high_exp, q=None, sign=None):
    """A random Float of precision q with exponent in [low_exp, high_exp]."""
    if q is None:
        q = rng.choice([1, 2, 3, 8, 24, 53, 64, 100, 128, 160]) if rng.random() < 0.5 else rng.randrange(1, 161)
    m = rng.getrandbits(q) | (1 << (q - 1))
    e = rng.randrange(low_exp, high_exp + 1)
    with gmpy2.context(precision=q):
        x = gmpy2.mul_2exp(mpfr(m), e - q)
    negative = rng.random() < 0.5 if sign is None else sign
    return -x if negative else x


def near(rng, center, scale_low, scale_high, side=None):
    """center +- 2**-k times a random mantissa, exact."""
    k = rng.randrange(scale_low, scale_high + 1)
    with gmpy2.context(precision=400):
        delta = gmpy2.mul_2exp(mpfr(rng.getrandbits(20) | (1 << 19)), -k - 20)
        s = side if side is not None else (1 if rng.random() < 0.5 else -1)
        return mpfr(center) + s * delta


def precision(rng):
    r = rng.random()
    if r < 0.7:
        return rng.randrange(2, 129)
    if r < 0.95:
        return rng.randrange(129, 1025)
    return rng.randrange(1025, 4097)


def arguments(name, rng, count):
    xs = []
    for i in range(count):
        kind = i % 4
        if name in ("log", "log2", "log10"):
            x = abs(make(rng, -200, 200)) if kind < 2 else near(rng, 1, 1, 150)
            if kind == 3:
                x = abs(make(rng, -3, 3))
        elif name == "log1p":
            x = make(rng, -200, 2) if kind < 2 else near(rng, -1, 1, 100, 1)
            if x <= -1:
                x = abs(x)
        elif name in ("asin", "acos", "atanh"):
            x = make(rng, -200, 0) if kind < 2 else near(rng, 1 if rng.random() < 0.5 else -1, 1, 150)
            if abs(x) >= 1:
                x = make(rng, -200, 0)
        elif name == "acosh":
            x = abs(make(rng, 1, 200)) if kind < 2 else near(rng, 1, 1, 150, 1)
            if x <= 1:
                x = abs(make(rng, 1, 200))
        elif name in ("exp", "exp2", "expm1", "sinh", "cosh"):
            x = make(rng, -200, 8) if kind < 3 else make(rng, -3, 3)
        elif name in ("sin", "cos", "tan"):
            x = make(rng, -200, 200) if kind < 2 else make(rng, -3, 3)
        else:
            x = make(rng, -200, 200) if kind < 2 else make(rng, -3, 3)
        xs.append(x)
    return xs


def line(name, p, mode_index, n, args, value, nan_input, ctx):
    encoded = " ".join(encode(a) for a in args)
    return f"{name} {p} {mode_index} {n} {encoded} {result(value, p)} {flags(ctx, nan_input)}\n"


def evaluate(name, args, n):
    if name == "pow":
        return args[0] ** args[1]
    if name == "atan2":
        return gmpy2.atan2(args[0], args[1])
    if name == "rootn":
        return gmpy2.rootn(args[0], n)
    return UNARY[name](args[0])


def emit(lines, name, p, args, n=0):
    nan_input = any(gmpy2.is_nan(a) for a in args)
    for mode_index, mode in enumerate(MODES):
        with context(p, mode) as ctx:
            ctx.clear_flags()
            value = evaluate(name, args, n)
            lines.append(line(name, p, mode_index, n, args, value, nan_input, ctx))


def main():
    rng = random.Random(20261005)
    lines = []
    for name in UNARY:
        for x in arguments(name, rng, 120):
            emit(lines, name, precision(rng), [x])
    for i in range(160):
        x = abs(make(rng, -40, 40)) if i % 4 else make(rng, 1, 8)
        y = make(rng, -60, 6)
        if i % 5 == 0:
            y = mpfr(rng.randrange(-40, 41))
            x = make(rng, -10, 10)
        emit(lines, "pow", precision(rng), [x, y])
    for i in range(160):
        emit(lines, "atan2", precision(rng), [make(rng, -100, 100), make(rng, -100, 100)])
    for i in range(160):
        n = rng.choice([2, 3, 4, 5, 7, 10, 17])
        x = make(rng, -200, 200)
        if n % 2 == 0:
            x = abs(x)
        emit(lines, "rootn", precision(rng), [x], n)
    # Special and exact inputs at 53 bits.
    specials = [mpfr(0), -mpfr(0), mpfr("inf"), mpfr("-inf"), mpfr("nan"), mpfr(1), mpfr(-1), mpfr(2), mpfr(-2),
                mpfr("0.5"), mpfr(1024), mpfr(1000), mpfr(100000), mpfr(3), mpfr("-0.5"), mpfr(8), mpfr(16)]
    for name in UNARY:
        for x in specials:
            emit(lines, name, 53, [x])
    pow_cases = [(0, -3), (-0.0, -3), (0, -2), (-0.0, 3), (0, 2), (-1, "inf"), (-1, "-inf"), (1, "nan"),
                 ("nan", 0), (-2, 0.5), (0.5, "-inf"), (2, "-inf"), (0.5, "inf"), (2, "inf"), ("-inf", -3),
                 ("-inf", -2), ("-inf", 3), ("-inf", 2), ("inf", -1), ("inf", 1), (2.25, 0.5), (16, 0.75),
                 (0.0625, -0.25), (4, -0.5), (2, 0.5), (8, "0.3333333333333333333"), (-8, 3), (27, 1.5),
                 (0, -0.5), (-0.0, 0.5), (0, "inf"), (0, "-inf")]
    for a, b in pow_cases:
        emit(lines, "pow", 53, [mpfr(a), mpfr(b)])
    atan2_cases = [(0, -0.0), (-0.0, -0.0), (0, 0), (-0.0, 0), (0, -1), (-0.0, -1), (0, 1), (-1, 0), (1, 0),
                   (1, "-inf"), (-1, "-inf"), (1, "inf"), (-1, "inf"), ("inf", 1), ("-inf", 1), ("inf", "-inf"),
                   ("-inf", "-inf"), ("inf", "inf"), ("-inf", "inf"), (1, 1), (-1, -1)]
    for a, b in atan2_cases:
        emit(lines, "atan2", 53, [mpfr(a), mpfr(b)])
    for x, n in [(0, 3), (-0.0, 3), (-0.0, 2), ("inf", 4), ("-inf", 3), ("-inf", 4), (-2, 4), (16, 4), (-27, 3),
                 (-8, 3), (2, 3), (1, 1), (-5, 1)]:
        emit(lines, "rootn", 53, [mpfr(x)], n)
    OUT.write_text("".join(lines))
    print(len(lines), "lines", OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
