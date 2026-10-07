"""Write tests/fixtures/special_ratios.txt, the oracle for gammaln below 0,
ndtr, log_ndtr, beta, betaln and poch.

Lines have the format of elementary.txt, with a second argument for the
two-argument functions:

    function p mode 0 x [y] result flags

MPFR (gmpy2's lgamma, log |Gamma| with its flags) is the oracle for gammaln
below 0. Values that are rational, `B(a, b)` with an integer argument and
`(z)_m` for an integer m, are computed exactly with Python's fractions and
rounded once by MPFR, with its flags; scipy's values at the poles are written
as infinities with divide-by-zero. Arb (python-flint) is the oracle for the
rest: its enclosure at p + 64 bits and then at doubling precisions is
certified, so when both ends round to the same Float in a mode, that Float is
the correct rounding; a case no precision up to 8p + 2048 decides is
dropped. Their flags are just inexact. MPFR and Arb are development
references only.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/special_ratios.py
"""

import math
import random
import signal
from fractions import Fraction
from pathlib import Path

import gmpy2
from flint import arb, ctx
from gmpy2 import mpfr, mpq

from elementary import MODES, context, encode, result, flags, make, near
from special_functions import to_arb, from_arb

OUT = Path(__file__).with_suffix(".txt")
ARITY = {"log_gamma": 1, "ndtr": 1, "log_ndtr": 1, "beta": 2, "betaln": 2, "poch": 2}


def is_int(x):
    return gmpy2.is_finite(x) and gmpy2.is_integer(x)


def frac(x):
    m, e = x.as_mantissa_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


def beta_exact(a, b):
    """B(a, b) as a Fraction when it is rational this way, 'inf' at scipy's
    poles, None otherwise: both arguments integers, or one a positive
    integer."""
    if is_int(a) and is_int(b):
        a, b = int(a), int(b)
        if a <= 0 or b <= 0:
            if a > 0:
                a, b = b, a
            if b > 0 and a + b < 1:
                sign = -1 if b % 2 else 1
                return sign * beta_exact(mpfr(1 - a - b), mpfr(b))
            return "inf"
        k, other = min(a, b), max(a, b)
        d = 1
        for i in range(k):
            d *= other + i
        return Fraction(math.factorial(k - 1), d)
    for k, other in ((a, b), (b, a)):
        # As the library, only products of up to 4096 factors are exact;
        # a larger k (a large argument that rounds to an integer) is left to
        # Arb.
        if is_int(k) and 0 < k <= 4096:
            k = int(k)
            z = frac(other)
            d = Fraction(1)
            for i in range(k):
                d *= z + i
            return Fraction(math.factorial(k - 1)) / d
    return None


def poch_exact(z, m):
    """(z)_m as a Fraction for an integer m, 'inf' where a divisor factor is 0."""
    if not is_int(m) or abs(m) > 4096:
        return None
    m = int(m)
    zf = frac(z)
    v = Fraction(1)
    if m >= 0:
        for i in range(m):
            v *= zf + i
        return v
    for i in range(1, -m + 1):
        f = zf - i
        if f == 0:
            return "inf"
        v /= f
    return v


def is_nonpos_int(x):
    """Whether an mpfr, or an exact Fraction, is a non-positive integer."""
    if isinstance(x, Fraction):
        return x.denominator == 1 and x <= 0
    return is_int(x) and x <= 0


def exact_case(name, args):
    """The exact value of a rational case, 'inf', '-inf' or '0' at the
    poles, or None for a transcendental one."""
    if name in ("beta", "betaln"):
        a, b = args
        v = beta_exact(a, b)
        if v is None:
            if is_nonpos_int(a) or is_nonpos_int(b):
                return "inf"
            if is_nonpos_int(frac(a) + frac(b)):
                return "0" if name == "beta" else "-inf"
            return None
        if v == "inf":
            return "inf"
        if name == "betaln":
            return "zero" if abs(v) == 1 else None
        return v
    if name == "poch":
        z, m = args
        v = poch_exact(z, m)
        if v is None:
            if is_nonpos_int(frac(z) + frac(m)) and not is_nonpos_int(z):
                return "inf"
            if is_nonpos_int(z) and not is_nonpos_int(frac(z) + frac(m)):
                return "0"
            return None
        return v
    return None


def arb_value(name, args):
    xs = [to_arb(x) for x in args]
    if name == "ndtr":
        return (-xs[0] / arb(2).sqrt()).erfc() / 2
    if name == "log_ndtr":
        if xs[0] > 0:
            return (-(xs[0] / arb(2).sqrt()).erfc() / 2).log1p()
        return ((-xs[0] / arb(2).sqrt()).erfc() / 2).log()
    if name in ("beta", "betaln"):
        a, b = xs
        v = a.gamma() * b.gamma() * (a + b).rgamma()
        return abs(v).log() if name == "betaln" else v
    if name == "poch":
        z, m = xs
        return (z + m).gamma() * z.rgamma()
    if name == "betaln_exact":
        return abs(to_arb_fraction(args[0])).log()
    raise ValueError(name)


def to_arb_fraction(f):
    return arb(f.numerator) / arb(f.denominator)


def arb_rounded(name, args, p, mode, value_of=None):
    bits = p + 64
    while bits <= 8 * p + 2048:
        ctx.prec = bits
        v = value_of() if value_of else arb_value(name, args)
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
    exact = exact_case(name, args)
    if name == "log_gamma":
        for mode_index, mode in enumerate(MODES):
            with context(p, mode) as c:
                c.clear_flags()
                value = gmpy2.lgamma(args[0])[0]
                rows.append(line(name, p, mode_index, args, result(value, p), flags(c, False)))
        lines.extend(rows)
        return True
    for mode_index, mode in enumerate(MODES):
        if exact in ("inf", "-inf"):
            rows.append(line(name, p, mode_index, args, f"2 {'-' if exact == '-inf' else '+'} 0 0", "00010"))
        elif exact == "0":
            rows.append(line(name, p, mode_index, args, "0 + 0 0", "00000"))
        elif exact == "zero":
            rows.append(line(name, p, mode_index, args, "0 + 0 0", "00000"))
        elif isinstance(exact, Fraction):
            with context(p, mode) as c:
                c.clear_flags()
                value = mpfr(mpq(exact.numerator, exact.denominator))
                rows.append(line(name, p, mode_index, args, result(value, p), flags(c, False)))
        else:
            value_of = None
            if name == "betaln":
                v = beta_exact(*args)
                if isinstance(v, Fraction):
                    value_of = lambda v=v: abs(to_arb_fraction(v)).log()
            value = arb_rounded(name, args, p, mode, value_of)
            if value is None:
                return False
            rows.append(line(name, p, mode_index, args, result(value, p), "10000"))
    lines.extend(rows)
    return True


def precision(rng):
    r = rng.random()
    if r < 0.8:
        return rng.randrange(2, 129)
    return rng.randrange(129, 385)


def small_int(rng, low, high):
    return mpfr(rng.randrange(low, high + 1))


def positive(rng, low, high):
    return abs(make(rng, low, high))


def arguments(name, rng, i):
    kind = i % 6
    if name == "log_gamma":
        if kind == 0:
            return [-abs(make(rng, -3, 7)) - mpfr(rng.randrange(0, 30))]
        if kind == 1:
            n = rng.randrange(1, 40)
            return [near(rng, -n, 4, 60)]
        if kind == 2:
            return [-(mpfr(rng.randrange(1, 10 ** 6)) + mpfr("0.5"))]
        if kind == 3:
            return [-gmpy2.mul_2exp(mpfr(1), -rng.randrange(3, 120))]
        if kind == 4:
            return [near(rng, "-2.4570247382208005860", 2, 50, None)]
        return [-small_int(rng, 1, 50)]
    if name in ("ndtr", "log_ndtr"):
        if kind == 0:
            return [make(rng, -60, 0)]
        if kind == 1:
            return [make(rng, -3, 3)]
        if kind == 2:
            return [make(rng, 1, 6)]
        if kind == 3:
            return [abs(make(rng, 2, 6))]
        if kind == 4:
            return [-abs(make(rng, 2, 6))]
        return [make(rng, -1, 2)]
    if name in ("beta", "betaln"):
        if kind == 0:
            return [positive(rng, -6, 6), positive(rng, -6, 6)]
        if kind == 1:
            return [make(rng, -3, 4), make(rng, -3, 4)]
        if kind == 2:
            return [small_int(rng, 1, 12), make(rng, -3, 4)]
        if kind == 3:
            return [small_int(rng, -8, 12), small_int(rng, -8, 12)]
        if kind == 4:
            return [positive(rng, 6, 20), positive(rng, -3, 6)]
        x = make(rng, -3, 4)
        return [x, mpfr(-rng.randrange(0, 6)) - x]
    if name == "poch":
        if kind == 0:
            return [make(rng, -3, 5), small_int(rng, -10, 12)]
        if kind == 1:
            return [positive(rng, -4, 6), make(rng, -3, 4)]
        if kind == 2:
            return [make(rng, -3, 4), make(rng, -3, 4)]
        if kind == 3:
            return [small_int(rng, -8, 6), make(rng, -3, 4)]
        if kind == 4:
            z = make(rng, -3, 4)
            return [z, mpfr(-rng.randrange(0, 6)) - z]
        return [positive(rng, 8, 30), make(rng, -2, 2)]
    raise ValueError(name)


class Slow(Exception):
    pass


def on_alarm(signum, frame):
    raise Slow()


def main():
    signal.signal(signal.SIGALRM, on_alarm)
    rng = random.Random(20261005)
    lines = []
    for name, count in (("log_gamma", 180), ("ndtr", 160), ("log_ndtr", 160), ("beta", 200), ("betaln", 200), ("poch", 200)):
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
            if attempts % 25 == 0:
                print(name, "attempts", attempts, "written", written, flush=True)
        print(name, written, flush=True)
        OUT.write_text("".join(lines))
    OUT.write_text("".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
