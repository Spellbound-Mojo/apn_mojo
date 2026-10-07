"""Differential check: the same inputs, printed by two builds, must print the same.

Built and run by tests/benchmarks/differential.py against a reference commit
and the working tree. An optional argument runs one section: integer, rational,
float, complex or ball. Each line is one input's results; wide values print as their
residue modulo 1000000007, bit length and sign, Float results with their
inexact status, and balls as their kind, midpoint and the radius's mantissa
and exponent. Ball lines start with the operation's name, so a change meant
to alter one operation's radius shows as differences on its lines alone.
"""

from apn_mojo import (
    Integer, Rational, Float, Complex, FloatFormat, ArithmeticContext, ComplexContext,
    RoundingMode, isqrt, iroot, gcd, lcm, factorial, factorial2,
)
from apn_mojo.integer.number_theory import _sqrt_rem
from apn_mojo.float._functions import _sqrt_float
from apn_mojo.float._arithmetic import _FloatArgument, _float_operation
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.float.math import sqrt as float_sqrt, maximum as float_maximum, minimum as float_minimum, clip as float_clip
from apn_mojo.complex._functions import _sqrt_complex
from apn_mojo.complex._arithmetic import _complex_operation
from apn_mojo.complex._input import _ComplexArgument
from apn_mojo.complex.context import _ComplexContextArgument
from apn_mojo import Ball, BallContext
from apn_mojo import ball
from std.sys import argv


def show(x: Integer) raises -> String:
    return String(x % 1000000007, ":", x.magnitude_bit_length(), ":", x.sign())


def show(x: Rational) raises -> String:
    return show(x.numerator()) + "/" + show(x.denominator())


def show(x: Float) raises -> String:
    return x.to_json()


def show(r: _RoundedBinary) raises -> String:
    return String(r.kind, ":", r.negative, ":", show(r.significand), ":", r.exponent, ":", r.status._flags, ":", r.status._direction)


def context(p: Int, mode: RoundingMode = RoundingMode.nearest_even) raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat(p), rounding=mode)


def modes() -> List[RoundingMode]:
    return [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]


def patterns(bits: Int, mut state: Integer) raises -> List[Integer]:
    """All ones, a power of two, a sparse value, and two dense values (one with
    clear low bits), each of exactly `bits` bits."""
    var result = List[Integer]()
    result.append((Integer(1) << bits) - 1)
    result.append(Integer(1) << (bits - 1))
    result.append((Integer(1) << (bits - 1)) + (Integer(1) << (bits // 2)))
    state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << bits)
    result.append(state | (Integer(1) << (bits - 1)))
    state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << bits)
    result.append(((state | (Integer(1) << (bits - 1))) >> 3) << 3)
    return result^


def integer_section() raises:
    var state = Integer(0x9E3779B97F4A7C15)
    var lengths = List[Int]()
    for bits in range(1, 3300, 7):
        lengths.append(bits)
    for bits in [4095, 4096, 4097, 8191, 8192, 12345, 20000]:
        lengths.append(bits)
    var shifts = [0, 1, 31, 32, 33, 63, 64, 65, 100]
    var smalls = [
        Integer(3), Integer(10), Integer(2147483647), (Integer(1) << 32) + 15,
        (Integer(1) << 63) - 25, Integer(1) << 63, (Integer(1) << 64) - 1,
    ]
    for bits in lengths:
        var xs = patterns(bits, state)
        var ys = patterns(max(1, bits - 5 + (bits % 11)), state)
        for i in range(len(xs)):
            for sign in range(4):
                var a = -xs[i] if sign & 1 else xs[i]
                var b = -ys[i] if sign & 2 else ys[i]
                var line = String("integer ", bits, " ", i, " ", sign, " ")
                line += show(a + b) + " " + show(a - b) + " " + show(b - a)
                line += " " + String(Int(a < b), Int(a == b), Int(b < a))
                line += " " + show(a & b) + " " + show(a | b) + " " + show(a ^ b) + " " + show(~a)
                if a and b:
                    line += " " + show(a // b) + " " + show(a % b) + " " + show(b // a) + " " + show(b % a)
                    line += " " + show(gcd(a, b)) + " " + show(lcm(a, b))
                for s in smalls:
                    line += " " + show(a // s) + " " + show(a % s) + " " + show(gcd(a, s))
                    line += " " + show(a // -s) + " " + show(a % -s)
                var z = a
                z += b
                line += " " + show(z)
                z -= b
                z -= b
                line += " " + show(z)
                for k in shifts:
                    line += " " + show(a << k) + " " + show(a >> k)
                    var w = a
                    w <<= k
                    line += " " + show(w)
                    w = a
                    w >>= k
                    line += " " + show(w)
                if sign == 0:
                    line += " " + show(isqrt(a)) + " " + show(isqrt(a * b))
                    var pair = _sqrt_rem(a)
                    line += " " + show(pair[0]) + " " + show(pair[1])
                    for degree in [3, 5, 31, 64]:
                        line += " " + show(iroot(a, degree))
                    line += " " + String(a.to_string(base=16).byte_length()) + " " + String(Integer(a.to_string(base=16, prefix=True), base=0) == a)
                elif sign == 1:
                    for degree in [3, 5, 31]:
                        line += " " + show(iroot(a, degree))
                print(line)
    for n in range(401):
        print("factorial", n, show(factorial(n)), show(factorial2(n)))


def rational_section() raises:
    var state = Integer(0x2545F4914F6CDD1D)
    var numerators = [1, 20, 31, 32, 33, 62, 63, 64, 65, 100, 127, 128, 129, 200, 256, 512, 1024]
    var denominators = [1, 2, 4, 30, 31, 32, 33, 62, 63, 64, 65, 128, 256]
    for nb in numerators:
        for db in denominators:
            var ns = patterns(nb, state)
            var ds = patterns(db, state)
            for i in range(len(ns)):
                var j = (i + 1) % len(ds)
                # Patterns of fewer than four bits can be zero.
                var x = Rational(ns[i], ds[i] if ds[i] else Integer(1))
                var y = Rational(ns[(i + 2) % len(ns)] + 1, ds[j] if ds[j] else Integer(3))
                for sign in range(4):
                    var a = -x if sign & 1 else x
                    var b = -y if sign & 2 else y
                    var line = String("rational ", nb, " ", db, " ", i, " ", sign, " ")
                    line += show(a + b) + " " + show(a - b) + " " + show(a * b) + " " + show(a / b)
                    line += " " + String(Int(a < b), Int(a == b), Int(b < a))
                    line += " " + show(a * a) + " " + show(a ** Integer(3)) + " " + show(-a) + " " + show(abs(a))
                    print(line)


def float_section() raises:
    var sources = List[Rational]()
    sources.append(Rational(2))
    sources.append(Rational(3))
    sources.append(Rational(1, 3))
    sources.append(Rational(-10))
    sources.append(Rational(Integer(4503599627370499), Integer(8)))
    sources.append(Rational(7, 4))
    sources.append(Rational(-(Integer(1) << 200) - 1))
    sources.append(Rational(Integer(1), (Integer(1) << 300) + 7))
    sources.append(Rational(Integer(3) << 5000))
    sources.append(Rational(Integer(-3), Integer(1) << 5000))
    sources.append(Rational(Integer(3) ** 80))
    sources.append(Rational((Integer(1) << 1001) + 5))
    var precisions = [2, 3, 11, 24, 31, 52, 53, 54, 63, 64, 65, 96, 113, 127, 128, 129, 192, 256, 257, 512, 1024]
    for p in precisions:
        var q = (Integer(1) << (p - 1)) + 5
        var middle = (2 * q + 1) * (2 * q + 1)
        var inputs = sources.copy()
        inputs.append(Rational(middle, Integer(4)))
        inputs.append(Rational(middle + 1, Integer(4)))
        inputs.append(Rational(middle - 1, Integer(4)))
        inputs.append(Rational(((Integer(1) << p) - 1) ** 2))
        for mode in modes():
            var target = context(p, mode)
            for i in range(len(inputs)):
                var x = Float(inputs[i], context=context(p))
                var wide = Float(inputs[i], context=context(1100))
                var line = String("float ", p, " ", mode, " ", i, " ")
                line += show(_sqrt_float(_FloatArgument(abs(inputs[i])), target))
                line += " " + show(_sqrt_float(_FloatArgument(abs(x)), target))
                line += " " + show(_sqrt_float(_FloatArgument(abs(wide)), target))
                var y = Float(inputs[(i + 5) % len(inputs)], context=context(p))
                for operation in range(4):
                    line += " " + show(_float_operation(_FloatArgument(x), _FloatArgument(y), operation, target))
                    line += " " + show(_float_operation(_FloatArgument(wide), _FloatArgument(inputs[(i + 3) % len(inputs)]), operation, target))
                # Exact values rounded once: conversions, and the chosen operand of maximum, minimum and clip.
                line += " " + show(Float(inputs[i], context=target)) + " " + show(Float(wide, context=target))
                line += " " + show(float_maximum(x, y, context=target)) + " " + show(float_minimum(wide, x, context=target))
                line += " " + show(float_clip(inputs[i], y, wide, context=target))
                print(line)
        # Without a context: one format, two merged formats, an exact operand, and the operators.
        for i in range(len(inputs)):
            var x = Float(inputs[i], context=context(p))
            var y = Float(inputs[(i + 5) % len(inputs)], context=context(p))
            var wide = Float(inputs[i], context=context(1100))
            var line = String("float-free ", p, " ", i, " ", show(float_sqrt(abs(x))))
            for operation in range(4):
                line += " " + show(_float_operation(_FloatArgument(x), _FloatArgument(y), operation))
                line += " " + show(_float_operation(_FloatArgument(x), _FloatArgument(wide), operation))
                line += " " + show(_float_operation(_FloatArgument(x), _FloatArgument(inputs[(i + 3) % len(inputs)]), operation))
            line += " " + show(x + y) + " " + show(x * y)
            line += " " + show(float_maximum(x, y)) + " " + show(float_minimum(x, wide)) + " " + show(float_clip(x, y, wide))
            print(line)


def complex_section() raises:
    var pairs = List[Tuple[Rational, Rational]]()
    var tiny = Rational(Integer(1), Integer(1) << 200)
    var far = Rational(Integer(1) << 600)
    var near = Rational(Integer(1), Integer(1) << 600)
    pairs.append((Rational(3), Rational(4)))
    pairs.append((Rational(-3), Rational(4)))
    pairs.append((Rational(3), Rational(-4)))
    pairs.append((Rational(-3), Rational(-4)))
    pairs.append((Rational(0), Rational(2)))
    pairs.append((Rational(0), Rational(-2)))
    pairs.append((Rational(0), Rational(1)))
    pairs.append((Rational(1), Rational(1)))
    pairs.append((Rational(-1), Rational(1)))
    pairs.append((Rational(1), tiny))
    pairs.append((Rational(-1), tiny))
    pairs.append((tiny, Rational(1)))
    pairs.append((Rational(-7), Rational(24)))
    pairs.append((Rational(1, 3), Rational(1, 7)))
    pairs.append((Rational(-5, 11), Rational(2, 9)))
    pairs.append((far, near))
    pairs.append((-far, near))
    pairs.append((near, -far))
    pairs.append((Rational(Integer(4503599627370499), Integer(8)), Rational(3, 2)))
    var formats = [(11, 11), (24, 24), (53, 53), (64, 64), (113, 113), (128, 128), (200, 200), (256, 256), (1024, 1024), (11, 53), (53, 128), (24, 256), (128, 24)]
    var mode_pairs = List[Tuple[RoundingMode, RoundingMode]]()
    for mode in modes():
        mode_pairs.append((mode, mode))
    mode_pairs.append((RoundingMode.nearest_even, RoundingMode.toward_negative))
    mode_pairs.append((RoundingMode.toward_positive, RoundingMode.toward_zero))
    var source = context(1100)
    for format in formats:
        for modes_of in mode_pairs:
            var target = ComplexContext(real=context(format[0], modes_of[0]), imag=context(format[1], modes_of[1]))
            for i in range(len(pairs)):
                var z = Complex(Float(pairs[i][0], context=source), Float(pairs[i][1], context=source))
                var w = Complex(Float(pairs[(i + 7) % len(pairs)][0], context=source), Float(pairs[(i + 7) % len(pairs)][1], context=source))
                var line = String("complex ", format[0], " ", format[1], " ", modes_of[0], " ", modes_of[1], " ", i, " ")
                var root = _sqrt_complex(_ComplexArgument(z), _ComplexContextArgument(target))
                line += show(root[0]) + " | " + show(root[1])
                for operation in range(4):
                    var parts = _complex_operation(_ComplexArgument(z), _ComplexArgument(w), operation, _ComplexContextArgument(target))
                    line += " " + show(parts[0]) + " | " + show(parts[1])
                print(line)


def show(b: Ball) raises -> String:
    var m = b._midpoint
    var r = b._radius
    return String(
        b._kind, ":", m._kind, ":", m._negative, ":", show(m._significand), ":", m._exponent,
        "+-", r.mantissa, ":", r.exponent, ":", r.infinite,
    )


def balls(p: Int, mut state: Integer) raises -> List[Ball]:
    """Balls at p bits: each pattern as a midpoint near 2**-3, 1 and 2**5, of
    both signs, with radii 0, 2**-(p+2) and 2**-10 relative to it, its whole
    magnitude (a lower end of 0 when that is a 30-bit number) and twice it."""
    var result = List[Ball]()
    for v in patterns(p, state):
        for e in [-3, 0, 5]:
            var magnitude = Rational(v, Integer(1) << (p - e))
            var radii = [
                Rational(0), Rational(Integer(1), Integer(1) << (p + 2 - e)), Rational(Integer(1), Integer(1) << (10 - e)),
                magnitude, magnitude * Rational(2),
            ]
            for sign in [1, -1]:
                for r in radii:
                    result.append(Ball(magnitude * Rational(sign), r, precision=p))
    return result^


def ball_section() raises:
    var state = Integer(12345)
    for p in [30, 53, 64, 113, 128, 200, 256, 1024]:
        var xs = balls(p, state)
        var n = len(xs)
        for q in [p // 2 + 1, p, 2 * p]:
            var c = BallContext(q)
            for i in range(n):
                ref x = xs[i]
                ref y = xs[(7 * i + 3) % n]
                ref z = xs[(11 * i + 5) % n]
                var line = String(p, " ", q, " ", i)
                print("ball arithmetic", line, show(ball.add(x, y, context=c)), show(ball.subtract(x, y, context=c)),
                      show(ball.multiply(x, y, context=c)), show(ball.divide(x, y, context=c)),
                      show(ball.square(x, context=c)), show(ball.pow_int(x, 3, context=c)), show(ball.fma(x, y, z, context=c)))
                print("ball sqrt", line, show(ball.sqrt(x, context=c)))
                if q == p and i % 5 == 0 and p <= 256:
                    print("ball functions", line, show(ball.exp(x, context=c)), show(ball.log(x, context=c)),
                          show(ball.sin(x, context=c)), show(ball.atan(x, context=c)), show(ball.erf(x, context=c)),
                          show(ball.gamma(x, context=c)))


def main() raises:
    var only = String(argv()[1]) if len(argv()) > 1 else String()
    if not only or only == "integer":
        integer_section()
    if not only or only == "rational":
        rational_section()
    if not only or only == "float":
        float_section()
    if not only or only == "complex":
        complex_section()
    if not only or only == "ball":
        ball_section()
