"""Write tests/fixtures/hypergeometric.txt, the oracle for the hypergeometric
functions.

Lines have the format of special_zeta.txt, with four arguments for hyp2f1,
three for hyp1f1 and betainc, and two for the incomplete gamma functions:

    hyp1f1 p mode 0 a b x result flags
    gammainc p mode 0 a x result flags
    hyp2f1 p mode 0 a b c x result flags
    betainc p mode 0 a b x result flags

Arb (python-flint's hypgeom_1f1 and hypgeom_2f1, and gamma_lower,
gamma_upper and beta_lower regularized) is the oracle: its enclosure at p + 64 bits
and then at doubling precisions is certified, so when both ends round to the
same Float in a mode, that Float is the correct rounding; a case no precision
up to 8p + 2048 decides is dropped, and so is one that takes more than 30
seconds. Exact values are exact Fractions, each checked to lie in Arb's
enclosure: the polynomials of a non-positive integer a (and scipy's at a
non-positive integer b at or below a negative integer a), 0 where Kummer's
polynomial `M(b-a, b, -x)` vanishes, and for integers 0 < a < b the rational
values where the coefficient of `e**x` in Euler's integral vanishes (except
at scipy's polynomials at a pole of b, which Arb defines otherwise). The
other non-positive integers b are scipy's `+inf` with divide-by-zero. The
incomplete gamma functions take scipy's values at a = 0, x = 0 and below 0.
hyp2f1's exact values, checked against Arb the same way: its polynomials
(scipy's +inf where c reaches a pole first), (1-x)**-b for c = a where that
is rational, Gauss's value at x = 1 for a positive integer a or b, 0 or
Chu-Vandermonde's (b)_k/(c)_k there, Euler's polynomial times a rational
power; NaN for x > 1 otherwise. betainc's exact values: the polynomial of
integer a and b, and x**a times a rational function for an integer b (or
by the symmetry for an integer a) where x**a is rational; scipy's values at
its edges. Arb and gmpy2 are development references only.

Run in the comparison environment:
    pixi run -e comparison python3 tests/fixtures/hypergeometric.py
"""

import math
import random
import signal
import sys
from fractions import Fraction
from pathlib import Path

import gmpy2
from flint import arb, ctx
from gmpy2 import mpfr, mpq

from elementary import MODES, context, encode, result, flags, make, near
from special_functions import to_arb, from_arb

OUT = Path(__file__).with_suffix(".txt")


def frac(x):
    m, e = x.as_mantissa_exp()
    return Fraction(int(m)) * Fraction(2) ** int(e)


def is_int(q):
    return q.denominator == 1


def nonpositive_int(q):
    return is_int(q) and q <= 0


def polynomial(a, b, x):
    """sum_{k<=n} (a)_k / (b)_k x**k / k! for a = -n, exactly."""
    n = -int(a)
    term = Fraction(1)
    total = Fraction(1)
    for k in range(n):
        term = term * (a + k) * x / ((b + k) * (k + 1))
        total += term
    return total


def elementary(a, b, x):
    """For integers 0 < a < b, M(a, b, x) when the coefficient of e**x in
    `M = C int_0^1 e**(xt) t**(a-1) (1-t)**(b-a-1) dt` vanishes, else None.
    Integrating by parts, the coefficient is
    `sum_j (-1)**j q^(j)(1) / x**(j+1)` for `q(t) = t**(a-1) (1-t)**m`."""
    m = b - a - 1
    d = b - 2
    # q as coefficients, then its derivatives at 0 and 1 directly.
    q = [0] * (d + 1)
    for i in range(m + 1):
        q[i + a - 1] = math.comb(m, i) * (-1) ** i
    def derivative_at(j, t):
        return sum(q[k] * math.perm(k, j) * t ** (k - j) for k in range(j, d + 1))
    e_part = sum(Fraction((-1) ** j * derivative_at(j, 1)) / x ** (j + 1) for j in range(d + 1))
    if e_part != 0:
        return None
    rest = sum(Fraction((-1) ** j * derivative_at(j, 0)) / x ** (j + 1) for j in range(d + 1))
    c = Fraction(math.factorial(b - 1), math.factorial(a - 1) * math.factorial(m))
    return -c * rest


def exact_hyp1f1(a, b, x):
    """M(a, b, x) as a Fraction where it is rational this way, 'inf' at
    scipy's poles, or None."""
    if nonpositive_int(b):
        if is_int(a) and a < 0 and a >= b:
            return polynomial(a, b, x)
        return "inf"
    if x == 0 or a == 0:
        return Fraction(1)
    if nonpositive_int(a):
        return polynomial(a, b, x)
    if nonpositive_int(b - a):
        if polynomial(b - a, b, -x) == 0:
            return Fraction(0)
        return None
    if is_int(a) and is_int(b) and 0 < a < b <= 64:
        return elementary(int(a), int(b), x)
    return None


def exact_gammainc(name, a, x):
    """scipy's values: NaN below 0 and at a = x = 0, then P(0, x) = 1 and
    P(a, 0) = 0 (Q = 1 - P); None elsewhere."""
    upper = name == "gammaincc"
    if a < 0 or x < 0 or (a == 0 and x == 0):
        return "nan"
    if a == 0:
        return Fraction(0 if upper else 1)
    if x == 0:
        return Fraction(1 if upper else 0)
    return None


def rising(z, k):
    v = Fraction(1)
    for i in range(k):
        v *= z + i
    return v


def poly2(a, b, c, x, m):
    """F(a, b; c; x) ended after degree m, exactly."""
    term = Fraction(1)
    total = Fraction(1)
    for k in range(m):
        term = term * (a + k) * (b + k) * x / ((c + k) * (k + 1))
        total += term
    return total


def integer_root(n, q):
    """The exact q-th root of a non-negative integer, or None."""
    if n < 0:
        return None
    r = round(n ** (1.0 / q)) if n < 2 ** 1000 else int(gmpy2.iroot(n, q)[0])
    for c in (r - 1, r, r + 1):
        if c >= 0 and c ** q == n:
            return c
    root, exact = gmpy2.iroot(n, q)
    return int(root) if exact else None


def rational_power(r, e):
    """r**e for r > 0 when it is rational, else None."""
    p, q = e.numerator, e.denominator
    top = integer_root(r.numerator, q)
    bottom = integer_root(r.denominator, q)
    if top is None or bottom is None:
        return None
    return Fraction(top, bottom) ** p


def exact_hyp2f1(a, b, c, x):
    """F(a, b; c; x) as a Fraction where it is rational this way, 'inf' at
    scipy's poles, 'nan' outside the real domain, or None."""
    if x == 0:
        return Fraction(1)
    ends = [-int(p) for p in (a, b) if nonpositive_int(p)]
    if ends:
        m = min(ends)
        if nonpositive_int(c) and -c < m:
            return "inf"
        return poly2(a, b, c, x, m)
    if nonpositive_int(c):
        return "inf"
    if x > 1:
        return "nan"
    s = c - a - b
    if x == 1:
        if s <= 0:
            if s == 0 and (nonpositive_int(c - a) or nonpositive_int(c - b)):
                k, other = (-int(c - a), b) if nonpositive_int(c - a) else (-int(c - b), a)
                return rising(other, k) / rising(c, k)
            return "inf"
        if nonpositive_int(c - a) or nonpositive_int(c - b):
            return Fraction(0)
        for p in (a, b):
            if is_int(p) and p > 0:
                return rising(c - p, int(p)) / rising(s, int(p))
        return None
    if c == a or c == b:
        return rational_power(1 - x, -(b if c == a else a))
    ks = [-int(q) for q in (c - a, c - b) if nonpositive_int(q)]
    if ks:
        inner = poly2(c - a, c - b, c, x, min(ks))
        if inner == 0:
            return Fraction(0)
        power = rational_power(1 - x, s)
        return None if power is None else power * inner
    return None


def exact_betainc(a, b, x):
    """I_x(a, b) as a Fraction where it is rational this way, 'nan' outside
    the domain, scipy's values at the edges, or None."""
    if a < 0 or b < 0 or x < 0 or x > 1 or (a == 0 and b == 0):
        return "nan"
    if a == 0:
        return Fraction(1 if x > 0 else 0)
    if b == 0:
        return Fraction(0 if x < 1 else 1)
    if x == 0 or x == 1:
        return Fraction(x)
    if is_int(a) and is_int(b):
        n = int(a + b - 1)
        return sum(math.comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(int(a), n + 1))
    def power_side(p, k, z):
        power = rational_power(z, p)
        if power is None:
            return None
        beta = Fraction(math.factorial(k - 1)) / rising(p, k)
        return power * poly2(p, Fraction(1 - k), p + 1, z, k - 1) / (p * beta)
    if is_int(b):
        return power_side(a, int(b), x)
    if is_int(a):
        v = power_side(b, int(a), 1 - x)
        return None if v is None else 1 - v
    return None


def arb_value(name, args):
    if name == "hyp1f1":
        a, b, x = args
        return to_arb(x).hypgeom_1f1(to_arb(a), to_arb(b))
    if name == "gammainc":
        a, x = args
        return to_arb(x).gamma_lower(to_arb(a), regularized=1)
    if name == "gammaincc":
        a, x = args
        return to_arb(x).gamma_upper(to_arb(a), regularized=1)
    if name == "hyp2f1":
        a, b, c, x = args
        return to_arb(x).hypgeom_2f1(to_arb(a), to_arb(b), to_arb(c))
    if name == "betainc":
        a, b, x = args
        return to_arb(x).beta_lower(to_arb(a), to_arb(b), regularized=1)
    raise ValueError(name)


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


def check_exact(name, args, value):
    """An exact value must lie in Arb's enclosure."""
    ctx.prec = 256
    v = arb_value(name, args)
    num = arb(value.numerator) / arb(value.denominator)
    if not v.overlaps(num):
        raise AssertionError(f"exact {name}{tuple(str(a) for a in args)} = {value} outside {v}")


def line(name, p, mode_index, args, encoded_result, flag_text):
    xs = " ".join(encode(x) for x in args)
    return f"{name} {p} {mode_index} 0 {xs} {encoded_result} {flag_text}\n"


def emit(lines, name, p, args):
    values = [frac(a) for a in args]
    if name == "hyp1f1":
        exact = exact_hyp1f1(*values)
        checked = not nonpositive_int(values[1])
    elif name == "hyp2f1":
        exact = exact_hyp2f1(*values)
        # At a pole of c, scipy's polynomial is the definition.
        checked = not nonpositive_int(values[2]) and values[3] <= 1
    elif name == "betainc":
        exact = exact_betainc(*values)
        checked = values[0] > 0 and values[1] > 0 and 0 < values[2] < 1
    else:
        exact = exact_gammainc(name, *values)
        checked = False
    if isinstance(exact, Fraction) and checked:
        # At a pole of b, scipy's polynomial is the definition; Arb takes
        # another limit there.
        check_exact(name, args, exact)
    rows = []
    for mode_index, mode in enumerate(MODES):
        if exact == "inf":
            rows.append(line(name, p, mode_index, args, "2 + 0 0", "00010"))
        elif exact == "nan" or exact == "-nan":
            rows.append(line(name, p, mode_index, args, "3 + 0 0", "00001"))
        elif isinstance(exact, Fraction):
            with context(p, mode) as c:
                c.clear_flags()
                value = mpfr(mpq(exact.numerator, exact.denominator))
                rows.append(line(name, p, mode_index, args, result(value, p), flags(c, False)))
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


def small_int(rng, low, high):
    return mpfr(rng.randrange(low, high + 1))


def hyp1f1_arguments(rng, i):
    kind = i % 10
    if kind == 0:
        # The series around 0, both signs.
        return [make(rng, -3, 3), make(rng, -3, 3), make(rng, -4, 3)]
    if kind == 1:
        # Moderate |x|: the series at |x|, Kummer's transformation below 0.
        return [make(rng, -2, 3), positive(rng, -2, 3), make(rng, 3, 7)]
    if kind == 2:
        # Large |x|: the asymptotic expansion.
        return [make(rng, -2, 3), make(rng, -2, 3), make(rng, 7, 14)]
    if kind == 3:
        # Polynomials of a non-positive integer a; scipy's at a pole b <= a.
        n = rng.randrange(0, 13)
        if rng.random() < 0.3:
            return [mpfr(-n), mpfr(-n - rng.randrange(0, 4)), make(rng, -3, 4)]
        return [mpfr(-n), make(rng, -3, 4), make(rng, -3, 4)]
    if kind == 4:
        # Integers 0 < a < b: the elementary case, with its rational values.
        b = rng.randrange(2, 9)
        a = rng.randrange(1, b)
        if rng.random() < 0.5:
            return [mpfr(a), mpfr(b), small_int(rng, -8, 8)]
        return [mpfr(a), mpfr(b), make(rng, -3, 4)]
    if kind == 5:
        # Kummer's polynomial: a = b + m, zero where M(-m, b, -x) vanishes.
        b = make(rng, -2, 4, q=rng.randrange(1, 7))
        with gmpy2.context(precision=64):
            if rng.random() < 0.5:
                return [b + 1, b, -b]
            return [b + rng.randrange(0, 4), b, make(rng, -3, 6)]
    if kind == 6:
        # Negative non-integer parameters: terms that change sign.
        return [-positive(rng, -2, 4), -positive(rng, -2, 4), make(rng, -2, 6)]
    if kind == 7:
        # Near the poles of b and the polynomial ends of a.
        n = rng.randrange(0, 6)
        return [near(rng, -rng.randrange(0, 6), 8, 60), near(rng, -n, 8, 60), make(rng, -2, 5)]
    if kind == 8:
        # scipy's poles: a non-positive integer b without an ending a.
        return [make(rng, -2, 3), mpfr(-rng.randrange(0, 6)), make(rng, -3, 3)]
    # Around the switch to the expansion, x near w log 2.
    return [make(rng, -2, 2), positive(rng, -2, 2), make(rng, 5, 9)]


def gammainc_arguments(rng, i):
    kind = i % 9
    if kind == 0:
        # Small shapes and arguments: the series.
        return [positive(rng, -4, 3), positive(rng, -6, 3)]
    if kind == 1:
        # x near a: the series for P, Q = 1 - P.
        a = positive(rng, 0, 7)
        with gmpy2.context(precision=200):
            return [a, a * (1 + make(rng, -6, -1))]
    if kind == 2:
        # x past a + 1: the expansion, or 1 - P with the bits it cancels.
        a = positive(rng, -2, 5)
        with gmpy2.context(precision=200):
            return [a, a + 1 + positive(rng, -1, 9)]
    if kind == 3:
        # A large x: Q far below 1, P within a tiny distance of 1.
        return [positive(rng, -3, 4), positive(rng, 9, 20)]
    if kind == 4:
        # A tiny x: P far below 1, Q within a tiny distance of 1.
        return [positive(rng, -3, 3), positive(rng, -80, -10)]
    if kind == 5:
        # Integer and half-integer shapes.
        return [mpfr(rng.randrange(1, 30)) / rng.choice([1, 2]), positive(rng, -3, 6)]
    if kind == 6:
        # Large shapes near their mean.
        a = positive(rng, 8, 12)
        with gmpy2.context(precision=200):
            return [a, a + make(rng, 0, 6)]
    if kind == 7:
        # scipy's values at 0 and below.
        return rng.choice([[mpfr(0), positive(rng, -3, 3)], [positive(rng, -3, 3), mpfr(0)], [mpfr(0), mpfr(0)],
                           [-positive(rng, -3, 3), positive(rng, -3, 3)], [positive(rng, -3, 3), -positive(rng, -3, 3)]])
    # Tiny shapes.
    return [positive(rng, -60, -8), positive(rng, -4, 4)]


def few_bits(rng, low, high):
    """A random Float of at most 6 bits, so sums stay exact."""
    return make(rng, low, high, q=rng.randrange(1, 7))


def hyp2f1_arguments(rng, i):
    kind = i % 10
    if kind == 0:
        # The series, |x| <= 1/2.
        return [make(rng, -3, 3), make(rng, -3, 3), make(rng, -3, 3), make(rng, -6, -1)]
    if kind == 1:
        # The connection with 1 - x, 1/2 < x < 1.
        return [make(rng, -3, 3), make(rng, -3, 3), make(rng, -3, 3), 1 - positive(rng, -40, -2)]
    if kind == 2:
        # Pfaff's transformation, x < -1/2.
        return [make(rng, -3, 3), make(rng, -3, 3), make(rng, -3, 3), -positive(rng, -1, 10)]
    if kind == 3:
        # Integer c - a - b (and b - a after Pfaff): the digamma series.
        a = few_bits(rng, -2, 3)
        b = few_bits(rng, -2, 3)
        with gmpy2.context(precision=64):
            m = rng.randrange(-3, 4)
            if rng.random() < 0.5:
                return [a, b, a + b + m, 1 - positive(rng, -30, -2)]
            return [a, a + m, few_bits(rng, -2, 3), -positive(rng, 0, 8)]
    if kind == 4:
        # Polynomials, at every x, and at poles of c.
        n = rng.randrange(0, 9)
        c = mpfr(-rng.randrange(0, 12)) if rng.random() < 0.3 else make(rng, -3, 3)
        return [mpfr(-n), make(rng, -3, 3), c, make(rng, -3, 4)]
    if kind == 5:
        # c = a: (1-x)**-b, rational at perfect powers.
        a = make(rng, -3, 3)
        x = rng.choice([mpfr(3) / 4, mpfr(15) / 16, mpfr(-3), mpfr(-8), mpfr(7) / 16, make(rng, -3, -1)])
        b = rng.choice([mpfr(1) / 2, mpfr(-3) / 2, mpfr(1) / 4, small_int(rng, -3, 3), make(rng, -2, 2)])
        return [a, b, a, x] if rng.random() < 0.5 else [b, a, a, x]
    if kind == 6:
        # x = 1: Gauss's value, rational for a positive integer a.
        a = rng.choice([small_int(rng, 1, 5), make(rng, -2, 2)])
        return [a, make(rng, -2, 2), make(rng, -1, 4), mpfr(1)]
    if kind == 7:
        # Outside: x > 1, and poles of c.
        if rng.random() < 0.5:
            return [make(rng, -2, 2), make(rng, -2, 2), make(rng, -2, 2), 1 + positive(rng, -10, 3)]
        return [make(rng, -2, 2), make(rng, -2, 2), mpfr(-rng.randrange(0, 6)), make(rng, -3, -1)]
    if kind == 8:
        # Euler's polynomial: c - a a non-positive integer.
        c = few_bits(rng, -2, 3)
        with gmpy2.context(precision=64):
            a = c + rng.randrange(1, 5)
        return [a, make(rng, -2, 2), c, make(rng, -3, -1) * rng.choice([1, -1])]
    # Near-integer c - a - b.
    a = few_bits(rng, -2, 3)
    b = few_bits(rng, -2, 3)
    with gmpy2.context(precision=400):
        c = a + b + rng.randrange(-2, 3) + make(rng, -60, -20)
    return [a, b, c, 1 - positive(rng, -10, -2)]


def betainc_arguments(rng, i):
    kind = i % 8
    if kind == 0:
        return [positive(rng, -3, 4), positive(rng, -3, 4), positive(rng, -8, -1)]
    if kind == 1:
        # Integer shapes: the exact polynomial.
        return [small_int(rng, 1, 12), small_int(rng, 1, 12), positive(rng, -6, -1)]
    if kind == 2:
        # One integer shape at a perfect power: rational values.
        x = rng.choice([mpfr(1) / 4, mpfr(9) / 16, mpfr(1) / 9, mpfr(3) / 4, mpfr(1) / 16])
        p = rng.choice([mpfr(1) / 2, mpfr(3) / 2, mpfr(1) / 4, mpfr(5) / 2])
        k = small_int(rng, 1, 6)
        return [p, k, x] if rng.random() < 0.5 else [k, p, x]
    if kind == 3:
        # A large a and a small b near x = 1: long series.
        return [positive(rng, 5, 9), positive(rng, -3, 1), 1 - positive(rng, -12, -4)]
    if kind == 4:
        # Tiny results and results within a tiny distance of 1.
        x = positive(rng, -60, -10)
        return [positive(rng, 0, 4), positive(rng, -2, 4), x if rng.random() < 0.5 else 1 - x]
    if kind == 5:
        # scipy's edges.
        return rng.choice([[mpfr(0), positive(rng, -2, 2), positive(rng, -4, -1)], [positive(rng, -2, 2), mpfr(0), positive(rng, -4, -1)],
                           [mpfr(0), mpfr(0), positive(rng, -4, -1)], [positive(rng, -2, 2), positive(rng, -2, 2), mpfr(0)],
                           [positive(rng, -2, 2), positive(rng, -2, 2), mpfr(1)], [positive(rng, -2, 2), positive(rng, -2, 2), 1 + positive(rng, -4, 0)],
                           [-positive(rng, -2, 2), positive(rng, -2, 2), positive(rng, -4, -1)]])
    if kind == 6:
        # Small shapes.
        return [positive(rng, -12, -1), positive(rng, -12, -1), positive(rng, -6, -1)]
    # Large shapes near the mean.
    a = positive(rng, 6, 10)
    b = positive(rng, 6, 10)
    with gmpy2.context(precision=200):
        return [a, b, a / (a + b) * (1 + make(rng, -8, -3))]


ARGUMENTS = {"hyp1f1": hyp1f1_arguments, "gammainc": gammainc_arguments, "gammaincc": gammainc_arguments,
             "hyp2f1": hyp2f1_arguments, "betainc": betainc_arguments}


class Slow(Exception):
    pass


def on_alarm(signum, frame):
    raise Slow()


def main():
    signal.signal(signal.SIGALRM, on_alarm)
    rng = random.Random(20261007)
    lines = []
    for name, count in (("hyp1f1", 500), ("gammainc", 300), ("gammaincc", 300), ("hyp2f1", 500), ("betainc", 400)):
        written = 0
        attempts = 0
        while written < count and attempts < 4 * count:
            p = precision(rng)
            with gmpy2.context(precision=p):
                args = [+x for x in ARGUMENTS[name](rng, attempts)]
            attempts += 1
            if any(gmpy2.is_nan(x) or gmpy2.is_infinite(x) for x in args):
                continue
            signal.alarm(30)
            try:
                if emit(lines, name, p, args):
                    written += 1
            except Slow:
                print("skipped a slow case", name, p, [str(x) for x in args], flush=True)
            finally:
                signal.alarm(0)
            if attempts % 50 == 0:
                print(name, "attempts", attempts, "written", written, flush=True)
                OUT.write_text("".join(lines))
        print(name, written, flush=True)
        OUT.write_text("".join(lines))
    print(f"wrote {OUT} ({len(lines)} lines)", flush=True)


if __name__ == "__main__":
    main()
