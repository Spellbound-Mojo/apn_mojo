"""Write tests/fixtures/special_functions.txt, the oracle for the special functions.

Lines have the format of elementary.txt:

    function p mode n x result flags

with `n` the branch of lambert_w (0 or -1) and 0 otherwise. MPFR (gmpy2) is
the oracle for gamma, log_gamma (MPFR's lngamma, for x > 0), digamma, erf, erfc
and exp_integral_ei (MPFR's eint), with its flags. Arb (python-flint) is the
oracle for erfi, the sine, cosine and hyperbolic integrals, Fresnel's integrals
and Lambert's W: its enclosure of the value, at p + 64 bits and then at
doubling precisions, is certified, so when both of its ends round to the same
Float in a mode, that Float is the correct rounding; a case that no precision
up to 16p + 4096 decides is dropped. Their flags are just inexact, as no value
is exact. MPFR and Arb are development references only.

Arguments are drawn per function from its typical ranges, tiny and large
magnitudes, and the neighbourhoods of its zeros, poles and branch points;
precisions from 2 to 512, mostly small. Special values at 53 bits follow for
the MPFR functions.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/special_functions.py
"""

import random
from pathlib import Path

import gmpy2
from flint import arb, ctx
from gmpy2 import mpfr

from elementary import MODES, context, encode, result, flags, make, near

OUT = Path(__file__).with_suffix(".txt")
MPFR = {
    "gamma": gmpy2.gamma, "log_gamma": gmpy2.lngamma, "digamma": gmpy2.digamma, "erf": gmpy2.erf,
    "erfc": gmpy2.erfc, "exp_integral_ei": gmpy2.eint,
}
ARB = {
    "erfi": lambda a: a.erfi(), "sin_integral": lambda a: a.si(), "cos_integral": lambda a: a.ci(),
    "sinh_integral": lambda a: a.shi(), "cosh_integral": lambda a: a.chi(),
    "fresnel_s": lambda a: a.fresnel_s(normalized=True), "fresnel_c": lambda a: a.fresnel_c(normalized=True),
}


def to_arb(x):
    """The mpfr as an exact arb."""
    m, e = x.as_mantissa_exp()
    return arb((int(m), int(e)))


def from_arb(v):
    """An exact arb (a bound) as an exact mpfr."""
    m, e = v.man_exp()
    m = int(m)
    if not m:
        return mpfr(0)
    with gmpy2.context(precision=max(abs(m).bit_length(), 2)):
        return gmpy2.mul_2exp(mpfr(m), int(e))


def arb_rounded(name, x, n, p, mode):
    """The correct rounding of the function at x, from Arb's enclosures at
    rising precision, or None when none decides it."""
    bits = p + 64
    while bits <= 16 * p + 4096:
        ctx.prec = bits
        a = to_arb(x)
        v = a.lambertw(n) if name == "lambert_w" else ARB[name](a)
        if v.is_finite():
            low, high = from_arb(v.lower()), from_arb(v.upper())
            with context(p, mode):
                a_low = +low
                a_high = +high
            if result(a_low, p) == result(a_high, p):
                return a_low
        bits *= 2
    return None


def emit_mpfr(lines, name, p, x):
    nan_input = gmpy2.is_nan(x)
    for mode_index, mode in enumerate(MODES):
        with context(p, mode) as ctx:
            ctx.clear_flags()
            value = MPFR[name](x)
            lines.append(f"{name} {p} {mode_index} 0 {encode(x)} {result(value, p)} {flags(ctx, nan_input)}\n")


def emit_arb(lines, name, p, x, n=0):
    rows = []
    for mode_index, mode in enumerate(MODES):
        value = arb_rounded(name, x, n, p, mode)
        if value is None:
            return False
        rows.append(f"{name} {p} {mode_index} {n} {encode(x)} {result(value, p)} 10000\n")
    lines.extend(rows)
    return True


def precision(rng):
    r = rng.random()
    if r < 0.75:
        return rng.randrange(2, 129)
    if r < 0.97:
        return rng.randrange(129, 257)
    return rng.randrange(257, 513)


def argument(name, rng, i):
    kind = i % 5
    if name == "gamma":
        if kind == 0:
            return abs(make(rng, -40, 7))
        if kind == 1:
            return make(rng, -3, 3)
        if kind == 2:
            return -abs(make(rng, -3, 7)) - mpfr(rng.randrange(0, 30))
        if kind == 3:
            return mpfr(rng.randrange(1, 60)) + mpfr("0.5") * rng.choice([0, 1])
        return gmpy2.mul_2exp(mpfr(rng.choice([1, -1])), -rng.randrange(3, 120))
    if name == "log_gamma":
        if kind == 0:
            return abs(make(rng, -40, 30))
        if kind == 1:
            return near(rng, rng.choice([1, 2]), 2, 60)
        return abs(make(rng, -3, 7))
    if name == "digamma":
        if kind == 0:
            return abs(make(rng, -40, 30))
        if kind == 1:
            return near(rng, "1.4616321449683623412626595423", 2, 60)
        if kind == 2:
            return -abs(make(rng, -3, 7)) - mpfr(rng.randrange(0, 20))
        if kind == 3:
            return gmpy2.mul_2exp(mpfr(rng.choice([1, -1])), -rng.randrange(3, 120))
        return make(rng, -3, 4)
    if name in ("erf", "erfc"):
        if kind == 0:
            return make(rng, -60, 0)
        if kind == 1:
            return make(rng, 1, 5)
        return make(rng, -3, 3)
    if name == "exp_integral_ei":
        if kind == 0:
            return make(rng, -60, 0)
        if kind == 1:
            return make(rng, 1, 9)
        if kind == 2:
            return near(rng, "0.37250741078136663446", 2, 60, None)
        return make(rng, -3, 4)
    if name in ("sin_integral", "sinh_integral", "erfi"):
        if kind == 0:
            return make(rng, -60, 0)
        if kind == 1 and name == "sin_integral":
            return make(rng, 8, 24)
        if kind == 1:
            return make(rng, 1, 5 if name == "erfi" else 9)
        return make(rng, -3, 4)
    if name in ("cos_integral", "cosh_integral"):
        if kind == 0:
            return abs(make(rng, -60, 0))
        if kind == 1:
            return abs(make(rng, 8, 24)) if name == "cos_integral" else abs(make(rng, 1, 9))
        if kind == 2:
            return near(rng, "0.61650548562467" if name == "cos_integral" else "0.52382257138986", 2, 40, None)
        return abs(make(rng, -3, 4))
    if name in ("fresnel_s", "fresnel_c"):
        if kind == 0:
            return make(rng, -60, 0)
        if kind == 1:
            return make(rng, 6, 20)
        return make(rng, -3, 4)
    raise ValueError(name)


def edge_plus(k):
    """-1/e + 2**-k, rounded to 100 bits."""
    ctx.prec = 300
    v = arb(-1) / arb(1).exp() + arb((1, -k))
    with gmpy2.context(precision=100):
        return +from_arb(v.mid())


def lambert_argument(rng, i, branch):
    kind = i % 4
    if branch == 0:
        if kind == 0:
            return make(rng, -60, -2)
        if kind == 1:
            return abs(make(rng, 1, 60))
        if kind == 2:
            return edge_plus(rng.randrange(4, 60))
        return make(rng, -2, 4) if rng.random() < 0.5 else abs(make(rng, -2, 4))
    if kind < 2:
        return -abs(make(rng, -120, -2))
    return edge_plus(rng.randrange(4, 60))


def main():
    rng = random.Random(20261005)
    lines = []
    for name in MPFR:
        for i in range(60):
            x = argument(name, rng, i)
            if name == "gamma" and gmpy2.is_integer(x) and x <= 0:
                continue
            emit_mpfr(lines, name, precision(rng), x)
    kept = dropped = 0
    for name in ARB:
        for i in range(60):
            x = argument(name, rng, i)
            if emit_arb(lines, name, precision(rng), x):
                kept += 1
            else:
                dropped += 1
    for branch in (0, -1):
        for i in range(60):
            x = lambert_argument(rng, i, branch)
            if (1 + to_arb(x) * arb(1).exp()) < 0 or (branch == -1 and x >= 0) or x == 0:
                continue
            if emit_arb(lines, "lambert_w", precision(rng), x, branch):
                kept += 1
            else:
                dropped += 1
    specials = [mpfr(0), -mpfr(0), mpfr("inf"), mpfr("-inf"), mpfr("nan"), mpfr(1), mpfr(2), mpfr(3), mpfr(-1),
                mpfr(-2), mpfr("0.5"), mpfr("-0.5"), mpfr("1.5"), mpfr(10), mpfr(100), mpfr(171), mpfr(-170.5)]
    for name in MPFR:
        for x in specials:
            if name == "log_gamma" and (x < 0 or (x == 0 and gmpy2.is_signed(x)) or gmpy2.is_infinite(x) and x < 0):
                continue
            emit_mpfr(lines, name, 53, x)
    OUT.write_text("".join(lines))
    print(len(lines), "lines;", kept, "Arb cases kept,", dropped, "dropped as undecided")


if __name__ == "__main__":
    main()
