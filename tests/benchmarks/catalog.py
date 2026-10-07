"""The cases of the consolidated report (tests/benchmarks/report.py).

Each case is generated here, deterministically, from one seed. A case has:

- an area, which is its table in the report;
- a family and an operation;
- a size in bits: an Integer's length, a Float's or Ball's precision, or n
  for factorials and binomials;
- an integer parameter, for operations that take one;
- exact operands as text.

The workers read the same text, so every backend computes on identical
inputs. Operands are written as follows:

- an Integer in decimal;
- a Rational as `p/q`;
- a Float as the dyadic rational it holds, exact at the case's precision;
- a Complex as `re;im`;
- a Ball as `mid;rad`, and a ComplexBall as `re_mid;re_rad;im_mid;im_rad`;
- a batch as its elements joined by `,`.

`references` names the backends that implement the case:

- `rug`: GMP, MPFR and MPC through Rug;
- `flint`: FLINT and Arb through python-flint.

`check` says how a result is compared with each reference:

- `exact`: the values are equal. This covers Integers, Rationals, and the
  correctly rounded Float and Complex results.
- `ball`: the balls overlap. Both enclose the true value, so disjoint balls
  mean an error.
- `decimal`: decimal text with the same value.
"""

import math
import random
import sys
from fractions import Fraction

# Operands of 65536 bits have about 20000 digits.
sys.set_int_max_str_digits(0)

SEED = 20261005

AREAS = {
    "integer": "Integer arithmetic",
    "number_theory": "Integer number theory",
    "rational": "Rational arithmetic",
    "float": "Float arithmetic",
    "float_function": "Float functions",
    "complex": "Complex arithmetic",
    "complex_function": "Complex functions",
    "ball": "Ball arithmetic",
    "ball_function": "Ball functions",
    "complex_ball": "ComplexBall",
    "conversion": "Text conversion",
    "batch": "Batches of 1000",
    "hypergeometric": "Hypergeometric functions",
}
"""Report tables, in order, with their titles."""


class _Source:
    def __init__(self, seed):
        self.random = random.Random(seed)

    def integer(self, bits):
        """A random integer of exactly `bits` bits."""
        return self.random.getrandbits(bits) | (1 << (bits - 1))

    def odd(self, bits):
        return self.integer(bits) | 1

    def prime(self, bits):
        """A random prime of `bits` bits: the first that passes 32 Miller-Rabin rounds."""
        n = self.odd(bits)
        while math.gcd(n, _SMALL_PRIMES) != 1 or not _probable_prime(n, self.random):
            n += 2
        return n

    def float(self, p, low, high):
        """A p-bit dyadic in [low, high), as `numerator/denominator`, with all p
        significant bits random: Arb trims trailing zeros, so a short significand
        would make its high-precision operations look cheap."""
        x = Fraction(low) + (Fraction(high) - Fraction(low)) * Fraction(self.random.getrandbits(p + 16), 2**(p + 16))
        negative, x = x < 0, abs(x)
        e = x.numerator.bit_length() - x.denominator.bit_length() - p
        while Fraction(2) ** (e + p) <= x:
            e += 1
        while Fraction(2) ** (e + p - 1) > x:
            e -= 1
        m = int(x / Fraction(2) ** e) * (-1 if negative else 1)
        value = Fraction(m) * Fraction(2) ** e
        return f"{value.numerator}/{value.denominator}"

    def ball(self, p, low, high):
        """A p-bit midpoint and a radius of 2**-(p+2) relative to it."""
        mid = self.float(p, low, high)
        value = abs(Fraction(mid))
        radius = Fraction(2) ** (value.numerator.bit_length() - value.denominator.bit_length() - p - 2)
        return f"{mid};{radius.numerator}/{radius.denominator}"


_SMALL_PRIMES = math.prod(q for q in range(3, 1000, 2) if all(q % d for d in range(3, math.isqrt(q) + 1, 2)))


def _probable_prime(n, source):
    if n < 4:
        return n in (2, 3)
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for _ in range(32):
        x = pow(source.randrange(2, n - 1), d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _case(area, family, op, bits, operands, *, param=0, references=("rug", "flint"), check="exact", label=None):
    return dict(id=f"{area}.{label or op}.{bits}", area=area, family=family, op=op, bits=bits, param=param,
                operands=[str(o) for o in operands], references=list(references), check=check)


FLOAT_DOMAINS = dict(
    exp=(-20, 20), expm1=(-1, 1), exp2=(-20, 20), log=(0.1, 100), log1p=(-0.5, 10), log2=(0.1, 100),
    log10=(0.1, 100), sin=(0.5, 3), cos=(0.5, 3), tan=(0.2, 1.4), atan=(0.1, 10), asin=(-0.9, 0.9),
    acos=(-0.9, 0.9), sinh=(-5, 5), cosh=(-5, 5), tanh=(-3, 3), asinh=(-10, 10), acosh=(1.5, 10),
    atanh=(-0.9, 0.9), gamma=(1.5, 20), gammaln=(1.5, 50), digamma=(1.5, 50), erf=(0.1, 3),
    erfc=(0.1, 3), expi=(0.5, 10),
)
"""Float functions with MPFR equivalents, and the interval each argument is drawn from."""

SPECIAL = ("gamma", "gammaln", "digamma", "erf", "erfc", "expi", "lambertw")
"""Functions left out at 4096 bits, where one call takes milliseconds."""


def cases():
    s = _Source(SEED)
    out = []

    for bits in (64, 256, 1024, 4096):
        x, y = s.integer(bits), s.integer(bits - 3)
        for op in ("add", "subtract", "multiply", "square", "floordiv", "mod", "and", "or", "xor"):
            out.append(_case("integer", "integer", op, bits, [x, y]))
        out.append(_case("integer", "integer", "shift_left", bits, [x], param=37))
        out.append(_case("integer", "integer", "shift_right", bits, [x], param=37))
        out.append(_case("integer", "integer", "floordiv", bits, [x, 2147483629], label="floordiv_small"))
    for bits in (16384, 65536):
        x, y = s.integer(bits), s.integer(bits - 3)
        out.append(_case("integer", "integer", "multiply", bits, [x, y]))
        out.append(_case("integer", "integer", "floordiv", bits, [x * y + 12345, y]))

    for bits in (64, 256, 1024, 4096):
        x, y = s.integer(bits), s.integer(bits - 3)
        out.append(_case("number_theory", "integer", "gcd", bits, [x, y]))
        out.append(_case("number_theory", "integer", "gcd", bits, [x, 2147483629], label="gcd_small"))
        out.append(_case("number_theory", "integer", "lcm", bits, [x, y]))
        out.append(_case("number_theory", "integer", "isqrt", bits, [x]))
        out.append(_case("number_theory", "integer", "iroot", bits, [x], param=3))
        out.append(_case("number_theory", "integer", "div_exact", bits, [x * y, y]))
        m = s.odd(bits)
        out.append(_case("number_theory", "integer", "inverse_mod", bits, [s.integer(bits - 1), m]))
        if bits <= 1024:
            out.append(_case("number_theory", "integer", "pow_mod", bits, [s.integer(bits - 1), s.integer(bits), m]))
            out.append(_case("number_theory", "integer", "is_prime", bits, [s.prime(bits)]))
    for n in (20, 100, 1000, 10000):
        out.append(_case("number_theory", "integer", "factorial", n, [], param=n))
        out.append(_case("number_theory", "integer", "binomial", n, [n], param=n // 2))

    for bits in (64, 256, 1024):
        x = Fraction(s.integer(bits), s.odd(bits // 2))
        y = Fraction(s.integer(bits - 2), s.odd(bits // 2))
        pair = [f"{x.numerator}/{x.denominator}", f"{y.numerator}/{y.denominator}"]
        for op in ("add", "subtract", "multiply", "divide", "square"):
            out.append(_case("rational", "rational", op, bits, pair))
        out.append(_case("rational", "rational", "pow_int", bits, pair[:1], param=3))

    for p in (53, 113, 256, 1024, 4096):
        x, y, z = s.float(p, 1, 100), s.float(p, 1, 10), s.float(p, -5, 5)
        for op in ("add", "subtract", "multiply", "divide"):
            out.append(_case("float", "float", op, p, [x, y], references=("rug",)))
        out.append(_case("float", "float", "square", p, [x], references=("rug",)))
        out.append(_case("float", "float", "sqrt", p, [x], references=("rug",)))
        out.append(_case("float", "float", "fma", p, [x, y, z], references=("rug",)))
        out.append(_case("float", "float", "pow_int", p, [x], param=19, references=("rug",)))

    for p in (53, 256, 1024, 4096):
        for op, (low, high) in FLOAT_DOMAINS.items():
            if p < 4096 or op not in SPECIAL:
                out.append(_case("float_function", "float", op, p, [s.float(p, low, high)], references=("rug",)))
        out.append(_case("float_function", "float", "atan2", p, [s.float(p, 0.5, 3), s.float(p, 0.5, 3)],
                         references=("rug",)))
        out.append(_case("float_function", "float", "pow", p, [s.float(p, 1.1, 3), s.float(p, 0.5, 5)],
                         references=("rug",)))

    for p in (53, 256, 1024):
        x = f"{s.float(p, 1, 10)};{s.float(p, 1, 10)}"
        y = f"{s.float(p, 1, 10)};{s.float(p, -10, -1)}"
        for op in ("add", "subtract", "multiply", "divide"):
            out.append(_case("complex", "complex", op, p, [x, y], references=("rug",)))
        out.append(_case("complex", "complex", "square", p, [x], references=("rug",)))
        out.append(_case("complex", "complex", "sqrt", p, [x], references=("rug",)))
        out.append(_case("complex", "complex", "pow_int", p, [x], param=19, references=("rug",)))
        for op in ("exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
                   "asinh", "acosh", "atanh"):
            z = f"{s.float(p, 0.2, 1.5)};{s.float(p, 0.2, 1.5)}"
            out.append(_case("complex_function", "complex", op, p, [z], references=("rug",)))
        z = f"{s.float(p, 0.2, 1.5)};{s.float(p, 0.2, 1.5)}"
        out.append(_case("complex_function", "complex", "pow", p, [z, f"{s.float(p, 0.5, 2)};{s.float(p, 0.1, 1)}"],
                         references=("rug",)))

    ball_domains = {k: v for k, v in FLOAT_DOMAINS.items() if k not in ("exp2", "log2", "log10")}
    ball_domains["lambertw"] = (0.1, 10)
    for p in (53, 256, 1024, 4096):
        x, y = s.ball(p, 1, 100), s.ball(p, 1, 10)
        for op in ("add", "subtract", "multiply", "divide"):
            out.append(_case("ball", "ball", op, p, [x, y], references=("flint",), check="ball"))
        out.append(_case("ball", "ball", "sqrt", p, [x], references=("flint",), check="ball"))
        for op, (low, high) in ball_domains.items():
            if p < 4096 or op not in SPECIAL:
                out.append(_case("ball_function", "ball", op, p, [s.ball(p, low, high)], references=("flint",),
                                 check="ball"))

    for p in (53, 256, 1024):
        def complex_ball():
            return f"{s.ball(p, 0.2, 1.5)};{s.ball(p, 0.2, 1.5)}"
        x, y = complex_ball(), complex_ball()
        for op in ("add", "multiply", "divide"):
            out.append(_case("complex_ball", "complex_ball", op, p, [x, y], references=("flint",), check="ball"))
        for op in ("exp", "log", "sqrt", "sin", "cos", "tan", "atan"):
            out.append(_case("complex_ball", "complex_ball", op, p, [complex_ball()], references=("flint",),
                             check="ball"))

    for bits in (64, 1024, 16384):
        x = s.integer(bits)
        out.append(_case("conversion", "integer_text", "parse", bits, [x], label="integer_parse"))
        out.append(_case("conversion", "integer_text", "print", bits, [x], label="integer_print"))
    for p in (53, 256, 1024):
        digits = p * 30103 // 100000 + 2
        text = "1." + "".join(str(s.random.randrange(10)) for _ in range(digits)) + "e7"
        out.append(_case("conversion", "float_text", "parse", p, [text], references=("rug",), label="float_parse"))
        out.append(_case("conversion", "float_text", "print", p, [s.float(p, 1, 100)], param=digits,
                         references=("rug",), check="decimal", label="float_print"))

    n = 1000
    for p in (53, 256):
        xs = ",".join(s.float(p, 1, 100) for _ in range(n))
        ys = ",".join(s.float(p, 1, 10) for _ in range(n))
        for op in ("add", "add_loop", "multiply", "exp", "exp_loop", "sum", "dot"):
            out.append(_case("batch", "float_batch", op, p, [xs, ys], param=n, references=("rug",),
                             label=f"float_{op}"))
    xs = ",".join(str(s.integer(256)) for _ in range(n))
    ys = ",".join(str(s.integer(250)) for _ in range(n))
    for op in ("add", "add_loop", "multiply", "sum"):
        out.append(_case("batch", "integer_batch", op, 256, [xs, ys], param=n, references=("rug",),
                         label=f"integer_{op}"))
    out += hypergeometric_cases()
    assert len({c["id"] for c in out}) == len(out), "duplicate case IDs"
    return out


HYPERGEOMETRIC_SEED = 20261007
"""A seed of their own, so that these cases leave the others' operands unchanged."""


def hypergeometric_cases():
    """scipy's hypergeometric functions in each method's regime: balls against
    Arb, at full-precision parameters and at short exact ones such as 1/2,
    which take the series engine's integer ratio; and the correctly rounded
    Floats at the balls' midpoints, without a reference (MPFR has no such
    functions, and Arb returns balls)."""
    s = _Source(HYPERGEOMETRIC_SEED)
    out = []

    def short(text):
        q = Fraction(text)
        return f"{q.numerator}/{q.denominator};0"

    for p in (53, 256, 1024):
        def full(low, high):
            return s.ball(p, low, high)
        rows = [
            ("hyp1f1", "hyp1f1_short", [short("1/2"), short("3/2"), full(2, 8)]),
            ("hyp1f1", "hyp1f1_series", [full(0.2, 2), full(1, 3), full(2, 8)]),
            ("hyp1f1", "hyp1f1_negative", [full(0.2, 2), full(1, 3), full(-30, -10)]),
            ("hyp1f1", "hyp1f1_large", [full(0.2, 2), full(1, 3), full(2 * p, 2 * p + 100)]),
            ("gammainc", "gammainc_series", [full(2, 10), full(0.5, 2)]),
            ("gammaincc", "gammaincc_middle", [full(2, 5), full(8, 20)]),
            ("gammaincc", "gammaincc_large", [full(1, 5), full(2 * p, 2 * p + 100)]),
            ("hyp2f1", "hyp2f1_short", [short("1/2"), short("1/3"), short("5/4"), full(0.1, 0.4)]),
            ("hyp2f1", "hyp2f1_series", [full(0.2, 2), full(0.2, 2), full(1, 3), full(0.1, 0.4)]),
            ("hyp2f1", "hyp2f1_connection", [full(0.2, 2), full(0.2, 2), full(1, 3), full(0.7, 0.9)]),
            ("hyp2f1", "hyp2f1_pfaff", [full(0.2, 2), full(0.2, 2), full(1, 3), full(-5, -2)]),
            ("hyp2f1", "hyp2f1_degenerate", [short("1/2"), short("3/2"), short("3"), full(0.7, 0.9)]),
            ("betainc", "betainc_short", [short("5/2"), short("7/2"), full(0.1, 0.4)]),
            ("betainc", "betainc_series", [full(1, 5), full(1, 5), full(0.1, 0.4)]),
        ]
        for op, label, operands in rows:
            out.append(_case("hypergeometric", "hypergeometric_ball", op, p, operands, references=("flint",),
                             check="ball", label=label))
            if p <= 256:
                midpoints = [o.split(";")[0] for o in operands]
                out.append(_case("hypergeometric", "hypergeometric_float", op, p, midpoints, references=(),
                                 label="float_" + label))
    return out


if __name__ == "__main__":
    import collections
    rows = cases()
    print(len(rows), "cases")
    for area, count in collections.Counter(r["area"] for r in rows).items():
        print(f"  {area}: {count}")
