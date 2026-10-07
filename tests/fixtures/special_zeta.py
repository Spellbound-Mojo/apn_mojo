"""Write tests/fixtures/special_zeta.txt, the oracle for zeta, polygamma,
erfinv and ndtri.

Lines have the format of special_ratios.txt:

    function p mode 0 x [y] result flags

with `zeta x q` and `polygamma n x` taking two arguments. MPFR (gmpy2's zeta,
with its flags) is the oracle for the Riemann zeta function, q = 1, at every
x. Arb (python-flint) is the oracle for the Hurwitz zeta function (a negative
q shifted above 0: `zeta(s, q) = sum_{k<m} (q+k)**-s + zeta(s, q+m)`), for
`polygamma(n, x) = (-1)**(n+1) n! zeta(n+1, x)` and for erfinv and ndtri
(`-sqrt 2 erfcinv(2p)`): its enclosure at p + 64 bits and then at doubling
precisions is certified, so when both ends round to the same Float in a mode,
that Float is the correct rounding; a case no precision up to 8p + 2048
decides is dropped, and so is one that takes more than 30 seconds. Poles and
points outside the domain are written as scipy defines them: infinities with
divide-by-zero, NaN with invalid. MPFR and Arb are development references
only.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/special_zeta.py
"""

import math
import random
import signal
from fractions import Fraction
from pathlib import Path

import gmpy2
from flint import arb, ctx
from gmpy2 import mpfr

from elementary import MODES, context, encode, result, flags, make, near
from special_functions import to_arb, from_arb

OUT = Path(__file__).with_suffix(".txt")


def frac(x):
    m, e = x.as_mantissa_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


def is_int(x):
    return gmpy2.is_finite(x) and gmpy2.is_integer(x)


def hurwitz(s, q):
    """zeta(s, q) as an arb, q shifted above 0."""
    total = arb(0)
    a = to_arb(q)
    while not a > 0:
        total += a ** (-to_arb(s))
        a += 1
    return total + to_arb(s).zeta(a)


def special_case(name, args):
    """A pole or a point outside the domain: ('inf', sign), ('nan',), or None."""
    if name == "zeta":
        s, q = args
        if s == 1:
            return ("inf", "+")
        if q == 1:
            return None
        if s < 1:
            return ("nan",)
        if is_int(q) and q <= 0:
            return ("inf", "+")
        if q < 0 and not is_int(s):
            return ("nan",)
        return None
    if name == "polygamma":
        n, x = args
        if is_int(x) and x <= 0:
            return ("inf", "+" if int(n) % 2 else "-")
        return None
    if name == "erfinv":
        # Comparisons, not abs(), which rounds to the context's precision.
        x = args[0]
        if x > 1 or x < -1:
            return ("nan",)
        if x == 1 or x == -1:
            return ("inf", "-" if x < 0 else "+")
        return None
    if name == "ndtri":
        p = args[0]
        if p < 0 or p > 1:
            return ("nan",)
        if p == 0:
            return ("inf", "-")
        if p == 1:
            return ("inf", "+")
        return None
    raise ValueError(name)


def arb_value(name, args):
    if name == "zeta":
        return hurwitz(*args)
    if name == "polygamma":
        n, x = args
        n = int(n)
        sign = 1 if n % 2 else -1
        return sign * math.factorial(n) * hurwitz(mpfr(n + 1), x)
    if name == "erfinv":
        return to_arb(args[0]).erfinv()
    return -arb(2).sqrt() * (2 * to_arb(args[0])).erfcinv()


def arb_rounded(name, args, p, mode):
    bits = p + 64
    while bits <= 8 * p + 2048:
        ctx.prec = bits
        v = arb_value(name, args)
        if v.is_finite():
            low, high = from_arb(v.lower()), from_arb(v.upper())
            with context(p, mode):
                a_low = +low
                a_high = +high
            if result(a_low, p) == result(a_high, p):
                return a_low
        bits *= 2
    return None


def line(name, p, mode_index, args, encoded_result, flag_text):
    xs = " ".join(encode(x) for x in args)
    return f"{name} {p} {mode_index} 0 {xs} {encoded_result} {flag_text}\n"


def emit(lines, name, p, args):
    rows = []
    special = special_case(name, args)
    for mode_index, mode in enumerate(MODES):
        if special is not None:
            if special[0] == "nan":
                rows.append(line(name, p, mode_index, args, "3 + 0 0", "00001"))
            else:
                rows.append(line(name, p, mode_index, args, f"2 {special[1]} 0 0", "00010"))
        elif name == "zeta" and args[1] == 1:
            with context(p, mode) as c:
                c.clear_flags()
                value = gmpy2.zeta(args[0])
                rows.append(line(name, p, mode_index, args, result(value, p), flags(c, False)))
        elif name == "ndtri" and args[0] == mpfr("0.5"):
            rows.append(line(name, p, mode_index, args, "0 + 0 0", "00000"))
        else:
            value = arb_rounded(name, args, p, mode)
            if value is None:
                return False
            rows.append(line(name, p, mode_index, args, result(value, p), "10000"))
    lines.extend(rows)
    return True


def precision(rng):
    r = rng.random()
    if r < 0.8:
        return rng.randrange(2, 129)
    return rng.randrange(129, 321)


def positive(rng, low, high):
    return abs(make(rng, low, high))


def arguments(name, rng, i):
    kind = i % 6
    if name == "zeta":
        if kind == 0:
            return [make(rng, -4, 6), mpfr(1)]
        if kind == 1:
            return [-mpfr(rng.randrange(0, 40)), mpfr(1)]
        if kind == 2:
            return [near(rng, 1, 4, 60), mpfr(1) if rng.random() < 0.5 else positive(rng, -3, 3)]
        if kind == 3:
            return [mpfr(1) + positive(rng, -6, 5), positive(rng, -6, 6)]
        if kind == 4:
            return [mpfr(rng.randrange(2, 11)), -positive(rng, -3, 3)]
        return [rng.choice([make(rng, -2, 2), mpfr(rng.randrange(150, 400))]), rng.choice([mpfr(1), mpfr(-rng.randrange(0, 4)), positive(rng, -2, 2)])]
    if name == "polygamma":
        n = mpfr(rng.randrange(1, 7))
        if kind < 2:
            return [n, positive(rng, -6, 6)]
        if kind < 4:
            return [n, -positive(rng, -3, 3)]
        if kind == 4:
            return [n, -mpfr(rng.randrange(0, 6))]
        return [n, make(rng, -2, 3)]
    if name == "erfinv":
        if kind == 0:
            return [make(rng, -60, -1)]
        if kind == 1:
            return [make(rng, -2, 0)]
        if kind == 2:
            return [(mpfr(1) - gmpy2.mul_2exp(mpfr(1), -rng.randrange(2, 200))) * rng.choice([1, -1])]
        if kind == 3:
            return [mpfr(rng.choice([1, -1, 2, "1.5"]))]
        return [make(rng, -6, 0)]
    if name == "ndtri":
        if kind == 0:
            return [gmpy2.mul_2exp(positive(rng, 0, 0), -rng.randrange(2, 300))]
        if kind == 1:
            return [mpfr("0.5") + make(rng, -40, -2)]
        if kind == 2:
            return [mpfr(1) - gmpy2.mul_2exp(mpfr(1), -rng.randrange(2, 200))]
        if kind == 3:
            return [mpfr(rng.choice([0, 1, "0.5", 2, -1]))]
        return [positive(rng, -8, -1)]
    raise ValueError(name)


class Slow(Exception):
    pass


def on_alarm(signum, frame):
    raise Slow()


def main():
    signal.signal(signal.SIGALRM, on_alarm)
    rng = random.Random(20261006)
    lines = []
    for name, count in (("zeta", 240), ("polygamma", 160), ("erfinv", 160), ("ndtri", 160)):
        written = 0
        attempts = 0
        while written < count and attempts < 4 * count:
            p = precision(rng)
            with gmpy2.context(precision=p):
                args = [+x for x in arguments(name, rng, attempts)]
            attempts += 1
            if any(gmpy2.is_nan(x) for x in args):
                continue
            signal.alarm(30)
            try:
                if emit(lines, name, p, args):
                    written += 1
            except Slow:
                print("skipped a slow case", name, p, [str(x) for x in args], flush=True)
            finally:
                signal.alarm(0)
            if attempts % 40 == 0:
                print(name, "attempts", attempts, "written", written, flush=True)
        print(name, written, flush=True)
        OUT.write_text("".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)", flush=True)


if __name__ == "__main__":
    main()
