"""The python-flint worker of the consolidated report (tests/benchmarks/report.py):
FLINT integers and rationals, and Arb balls. Development only.

The protocol is apn_worker.mojo's: the catalog file as the argument, then
commands on standard input, one JSON line each:

- `check i`
- `time i target repeats`
- `empty arity target repeats`
- `version`
- `quit`

Each case is one call `f(*args)` of a C function: an operator, a method
descriptor such as `arb.exp`, or a constructor. The arguments are prepared
first, and `timeit` times the loop with garbage collection off.

`empty` times the same loop around a C function that does no work:
`callable(a)` for one argument, `operator.is_(a, b)` for two or more. Subtracting it
estimates the time spent in the library itself.
"""

import json
import operator
import sys
import timeit
from fractions import Fraction

import flint

ARB = {name: name for name in (
    "exp", "expm1", "log", "log1p", "sin", "cos", "tan", "atan", "asin", "acos", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "gamma", "digamma", "erf", "erfc", "lambertw", "sqrt")}
ARB.update(gammaln="lgamma", expi="ei")
BINARY = dict(add=operator.add, subtract=operator.sub, multiply=operator.mul, divide=operator.truediv)


def hypergeometric(op, x):
    """Arb's method and arguments for scipy's argument order: Arb takes the
    argument as `self`, and `regularized` positionally."""
    if op == "hyp1f1":
        return flint.arb.hypgeom_1f1, [x[2], x[0], x[1]]
    if op == "gammainc":
        return flint.arb.gamma_lower, [x[1], x[0], 1]
    if op == "gammaincc":
        return flint.arb.gamma_upper, [x[1], x[0], 1]
    if op == "hyp2f1":
        return flint.arb.hypgeom_2f1, [x[3], x[0], x[1], x[2]]
    if op == "betainc":
        return flint.arb.beta_lower, [x[2], x[0], x[1], 1]
    raise ValueError(f"unsupported hypergeometric function: {op}")


def fmpq(text):
    q = Fraction(text)
    return flint.fmpq(q.numerator, q.denominator)


def ball(mid, rad):
    return flint.arb(fmpq(mid), fmpq(rad))


def complex_ball(text):
    p = text.split(";")
    return flint.acb(ball(p[0], p[1]), ball(p[2], p[3]))


def prepare(row):
    """The callable and its arguments; ball cases also set Arb's working precision."""
    f = row.rstrip("\n").split("\t")
    family, op, bits, param, a = f[1], f[2], int(f[3]), int(f[4]), f[5:]
    if family == "integer":
        # Padded, so that every entry of the table below can be built.
        x = [flint.fmpz(t) for t in a] + [flint.fmpz(0)] * 3
        ops = {
            "add": (operator.add, x[:2]), "subtract": (operator.sub, x[:2]), "multiply": (operator.mul, x[:2]),
            "square": (operator.mul, [x[0], x[0]]), "floordiv": (operator.floordiv, x[:2]),
            "mod": (operator.mod, x[:2]), "and": (operator.and_, x[:2]), "or": (operator.or_, x[:2]),
            "xor": (operator.xor, x[:2]), "shift_left": (operator.lshift, [x[0], param]),
            "shift_right": (operator.rshift, [x[0], param]), "gcd": (flint.fmpz.gcd, x[:2]),
            "lcm": (flint.fmpz.lcm, x[:2]), "isqrt": (flint.fmpz.isqrt, x[:1]),
            "iroot": (flint.fmpz.root, [x[0], param]),
            # python-flint exposes no exact division; floor division stands in.
            "div_exact": (operator.floordiv, x[:2]),
            "inverse_mod": (pow, [x[0], -1, x[1]]), "pow_mod": (pow, x[:3]),
            "is_prime": (flint.fmpz.is_probable_prime, x[:1]),
            "factorial": (flint.fmpz.fac_ui, [param]), "binomial": (flint.fmpz.bin_uiui, [bits, param]),
        }
        return ops[op]
    if family == "rational":
        x = [fmpq(t) for t in a]
        ops = dict(BINARY, square=operator.mul, pow_int=operator.pow)
        args = {"square": [x[0], x[0]], "pow_int": [x[0], param]}.get(op, x[:2])
        return ops[op], args
    if family == "integer_text":
        return (flint.fmpz, [a[0]]) if op == "parse" else (str, [flint.fmpz(a[0])])
    flint.ctx.prec = bits
    if family == "ball":
        x = [ball(*t.split(";")) for t in a]
        return (BINARY[op], x[:2]) if op in BINARY else (getattr(flint.arb, ARB[op]), x[:1])
    if family == "hypergeometric_ball":
        return hypergeometric(op, [ball(*t.split(";")) for t in a])
    if family == "complex_ball":
        x = [complex_ball(t) for t in a]
        return (BINARY[op], x[:2]) if op in BINARY else (getattr(flint.acb, op), x[:1])
    raise ValueError(f"unsupported family: {family}")


def exact(x):
    """An exact Arb midpoint or radius as a fraction's text."""
    m, e = (int(v) for v in x.man_exp())
    return str(Fraction(m) * Fraction(2) ** e)


def encode(value):
    if isinstance(value, int):  # also a primality test's bool or 0/1
        return {"integer": str(int(value))}
    if isinstance(value, str):
        return {"text": value}
    if isinstance(value, flint.fmpz):
        return {"integer": str(value)}
    if isinstance(value, flint.fmpq):
        return {"numerator": str(value.p), "denominator": str(value.q)}
    if isinstance(value, flint.arb):
        return {"ball": {"mid": exact(value.mid()), "rad": exact(value.rad())}}
    if isinstance(value, flint.acb):
        return {"real": encode(value.real), "imag": encode(value.imag)}
    raise TypeError(type(value).__name__)


def time(f, args, target, repeats):
    """The timeit protocol: 1, 2, 5, 10, 20, ... loops until one sample takes `target` ns."""
    names = [f"a{i}" for i in range(len(args))]
    timer = timeit.Timer(f"f({', '.join(names)})", globals=dict(zip(names, args), f=f))
    number, decade = 1, 1
    while timer.timeit(number) * 1e9 < target:
        if number == 5 * decade:
            decade *= 10
            number = decade
        else:
            number = 2 * decade if number == decade else 5 * decade
    return {"number": number, "samples": [round(timer.timeit(number) * 1e9) for _ in range(repeats)]}


def main():
    rows = open(sys.argv[1]).read().splitlines()
    for line in sys.stdin:
        words = line.split()
        if words[0] == "quit":
            break
        try:
            if words[0] == "version":
                answer = {"python-flint": flint.__version__, "flint": flint.__FLINT_VERSION__,
                          "python": sys.version.split()[0]}
            elif words[0] == "empty":
                arity = int(words[1])
                f, args = (callable, [flint.fmpz(1)]) if arity == 1 else (operator.is_, [flint.fmpz(1), flint.fmpz(2)])
                answer = time(f, args, int(words[2]), int(words[3]))
            else:
                f, args = prepare(rows[int(words[1])])
                if words[0] == "check":
                    answer = {"result": encode(f(*args)), "arity": len(args)}
                else:
                    answer = time(f, args, int(words[2]), int(words[3]))
        except Exception as error:
            answer = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    main()
