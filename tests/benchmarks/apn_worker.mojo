"""The apn_mojo worker of the consolidated report (tests/benchmarks/report.py).

Its argument names the catalog file: one case per line, tab-separated: id,
family, operation, bits, parameter, operands (see catalog.py). It then answers
commands on standard input, one JSON line each:

- `check i`: case i's result, as the library's JSON record.
- `time i target repeats`: the timeit protocol. The number of calls steps
  1, 2, 5, 10, 20, ... until one sample of that many calls takes at least
  `target` nanoseconds. Then `repeats` samples of that many calls each; a
  sample is one timer pair around the loop, and no result is checked.
- `calls i n`: exactly n timed calls and nothing else, so that callgrind
  counts repeat between builds.
- `quit`.

Inputs are prepared before any timing. Each timed call is out of line and
consumes its result's nonzero bit, so it can be neither hoisted nor dropped.
The driver sends a command only after reading the previous answer.
"""

from std.benchmark import keep
from std.sys import argv
from std.time import perf_counter_ns
from apn_mojo import (
    Integer, Rational, Float, Complex, Ball, ComplexBall, Batch,
    FloatFormat, ArithmeticContext, ComplexContext, BallContext,
    gcd, lcm, isqrt, iroot, factorial, comb, div_exact, inverse_mod, pow_mod, is_prime,
    vmap, sum, dot,
)
from apn_mojo import integer, float, complex, ball, complex_ball


trait Workload(Movable):
    def run(self) raises -> Bool:
        """One timed call: the operation, reduced to whether its result is nonzero."""
        ...

    def result(self) raises -> String:
        """The operation's result as JSON."""
        ...


def elapsed[W: Workload](work: W, number: Int) raises -> Int:
    var sink = 0
    var start = perf_counter_ns()
    for _ in range(number):
        sink += Int(work.run())
    var t = perf_counter_ns() - start
    keep(sink)
    return t


def serve[W: Workload](work: W, command: List[String]) raises -> String:
    if command[0] == "check":
        return String('{"result":', work.result(), "}")
    if command[0] == "calls":
        return String('{"elapsed":', elapsed(work, Int(command[2])), "}")
    var target = Int(command[2])
    var repeats = Int(command[3])
    var number = 1
    var decade = 1
    while elapsed(work, number) < target:
        if number == 5 * decade:
            decade *= 10
            number = decade
        else:
            number = 2 * decade if number == decade else 5 * decade
    var samples = String()
    for i in range(repeats):
        if i:
            samples += ","
        samples += String(elapsed(work, number))
    return String('{"number":', number, ',"samples":[', samples, "]}")


def text_json(text: String) -> String:
    return String('{"text":"', text, '"}')


def list_json(values: Batch[Float]) raises -> String:
    var out = String('{"list":[')
    for i in range(len(values)):
        out += String("," if i else "", values[i].to_json())
    return out + "]}"


def list_json(values: Batch[Integer]) raises -> String:
    var out = String('{"list":[')
    for i in range(len(values)):
        out += String("," if i else "", values[i].to_json())
    return out + "]}"


def code_of(op: String, names: List[String]) raises -> Int:
    for i in range(len(names)):
        if names[i] == op:
            return i
    raise Error(String("Unsupported operation: ", op))


# Integers. The operation codes follow `integer_names`.


def integer_names() -> List[String]:
    return [
        "add", "subtract", "multiply", "square", "floordiv", "mod", "and", "or", "xor",
        "shift_left", "shift_right", "gcd", "lcm", "isqrt", "iroot", "div_exact",
        "inverse_mod", "pow_mod", "is_prime", "factorial", "binomial",
    ]


def integer_value[k: Int](x: Integer, y: Integer, z: Integer, n: Integer, count: Int) raises -> Integer:
    comptime if k == 0:
        return x + y
    elif k == 1:
        return x - y
    elif k == 2:
        return x * y
    elif k == 3:
        return x * x
    elif k == 4:
        return x // y
    elif k == 5:
        return x % y
    elif k == 6:
        return x & y
    elif k == 7:
        return x | y
    elif k == 8:
        return x ^ y
    elif k == 9:
        return x << count
    elif k == 10:
        return x >> count
    elif k == 11:
        return gcd(x, y)
    elif k == 12:
        return lcm(x, y)
    elif k == 13:
        return isqrt(x)
    elif k == 14:
        return iroot(x, n)
    elif k == 15:
        return div_exact(x, y)
    elif k == 16:
        return inverse_mod(x, y)
    elif k == 17:
        return pow_mod(x, y, z)
    elif k == 18:
        return Integer(Int(is_prime(x)))
    elif k == 19:
        return factorial(n)
    else:
        return comb(x, n)


struct IntegerWork[k: Int](Workload):
    var x: Integer
    var y: Integer
    var z: Integer
    var n: Integer
    var count: Int

    def __init__(out self, operands: List[String], param: Int) raises:
        self.x = Integer(operands[0]) if len(operands) > 0 else Integer(0)
        self.y = Integer(operands[1]) if len(operands) > 1 else Integer(0)
        self.z = Integer(operands[2]) if len(operands) > 2 else Integer(0)
        self.n = Integer(param)
        self.count = param

    @no_inline
    def run(self) raises -> Bool:
        return integer_value[Self.k](self.x, self.y, self.z, self.n, self.count).__bool__()

    def result(self) raises -> String:
        return integer_value[Self.k](self.x, self.y, self.z, self.n, self.count).to_json()


# Rationals.


def rational_names() -> List[String]:
    return ["add", "subtract", "multiply", "divide", "square", "pow_int"]


def rational_value[k: Int](x: Rational, y: Rational, n: Integer) raises -> Rational:
    comptime if k == 0:
        return x + y
    elif k == 1:
        return x - y
    elif k == 2:
        return x * y
    elif k == 3:
        return x / y
    elif k == 4:
        return x * x
    else:
        return x ** n


struct RationalWork[k: Int](Workload):
    var x: Rational
    var y: Rational
    var n: Integer

    def __init__(out self, operands: List[String], param: Int) raises:
        self.x = Rational(operands[0])
        self.y = Rational(operands[1]) if len(operands) > 1 else Rational(0)
        self.n = Integer(param)

    @no_inline
    def run(self) raises -> Bool:
        return rational_value[Self.k](self.x, self.y, self.n).__bool__()

    def result(self) raises -> String:
        return rational_value[Self.k](self.x, self.y, self.n).to_json()


# Floats: arithmetic and functions, each result rounded once to the case's precision.


def float_names() -> List[String]:
    return [
        "add", "subtract", "multiply", "divide", "square", "sqrt", "fma", "pow_int",
        "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sin", "cos", "tan",
        "atan", "asin", "acos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "gamma", "gammaln", "digamma", "erf", "erfc", "expi", "atan2", "pow",
    ]


def float_value[k: Int](x: Float, y: Float, z: Float, n: Int, c: ArithmeticContext) raises -> Float:
    comptime if k == 0:
        return float.add(x, y, context=c)
    elif k == 1:
        return float.subtract(x, y, context=c)
    elif k == 2:
        return float.multiply(x, y, context=c)
    elif k == 3:
        return float.divide(x, y, context=c)
    elif k == 4:
        return float.square(x, context=c)
    elif k == 5:
        return float.sqrt(x, context=c)
    elif k == 6:
        return float.fma(x, y, z, context=c)
    elif k == 7:
        return float.pow_int(x, n, context=c)
    elif k == 8:
        return float.exp(x, context=c)
    elif k == 9:
        return float.expm1(x, context=c)
    elif k == 10:
        return float.exp2(x, context=c)
    elif k == 11:
        return float.log(x, context=c)
    elif k == 12:
        return float.log1p(x, context=c)
    elif k == 13:
        return float.log2(x, context=c)
    elif k == 14:
        return float.log10(x, context=c)
    elif k == 15:
        return float.sin(x, context=c)
    elif k == 16:
        return float.cos(x, context=c)
    elif k == 17:
        return float.tan(x, context=c)
    elif k == 18:
        return float.atan(x, context=c)
    elif k == 19:
        return float.asin(x, context=c)
    elif k == 20:
        return float.acos(x, context=c)
    elif k == 21:
        return float.sinh(x, context=c)
    elif k == 22:
        return float.cosh(x, context=c)
    elif k == 23:
        return float.tanh(x, context=c)
    elif k == 24:
        return float.asinh(x, context=c)
    elif k == 25:
        return float.acosh(x, context=c)
    elif k == 26:
        return float.atanh(x, context=c)
    elif k == 27:
        return float.gamma(x, context=c)
    elif k == 28:
        return float.gammaln(x, context=c)
    elif k == 29:
        return float.digamma(x, context=c)
    elif k == 30:
        return float.erf(x, context=c)
    elif k == 31:
        return float.erfc(x, context=c)
    elif k == 32:
        return float.expi(x, context=c)
    elif k == 33:
        return float.atan2(x, y, context=c)
    else:
        return float.pow(x, y, context=c)


def float_operand(text: String, c: ArithmeticContext) raises -> Float:
    return Float(Rational(text), context=c)


struct FloatWork[k: Int](Workload):
    var x: Float
    var y: Float
    var z: Float
    var n: Int
    var c: ArithmeticContext

    def __init__(out self, operands: List[String], bits: Int, param: Int) raises:
        self.c = ArithmeticContext(format=FloatFormat(bits))
        self.x = float_operand(operands[0], self.c)
        self.y = float_operand(operands[1], self.c) if len(operands) > 1 else Float(0, context=self.c)
        self.z = float_operand(operands[2], self.c) if len(operands) > 2 else Float(0, context=self.c)
        self.n = param

    @no_inline
    def run(self) raises -> Bool:
        return not float_value[Self.k](self.x, self.y, self.z, self.n, self.c).is_zero()

    def result(self) raises -> String:
        return float_value[Self.k](self.x, self.y, self.z, self.n, self.c).to_json()


# Complex numbers, each part rounded once.


def complex_names() -> List[String]:
    return [
        "add", "subtract", "multiply", "divide", "square", "sqrt", "pow_int",
        "exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
        "asinh", "acosh", "atanh", "pow",
    ]


def complex_value[k: Int](x: Complex, y: Complex, n: Int, c: ComplexContext) raises -> Complex:
    comptime if k == 0:
        return complex.add(x, y, context=c)
    elif k == 1:
        return complex.subtract(x, y, context=c)
    elif k == 2:
        return complex.multiply(x, y, context=c)
    elif k == 3:
        return complex.divide(x, y, context=c)
    elif k == 4:
        return complex.multiply(x, x, context=c)
    elif k == 5:
        return complex.sqrt(x, context=c)
    elif k == 6:
        return complex.pow_int(x, n, context=c)
    elif k == 7:
        return complex.exp(x, context=c)
    elif k == 8:
        return complex.log(x, context=c)
    elif k == 9:
        return complex.sin(x, context=c)
    elif k == 10:
        return complex.cos(x, context=c)
    elif k == 11:
        return complex.tan(x, context=c)
    elif k == 12:
        return complex.sinh(x, context=c)
    elif k == 13:
        return complex.cosh(x, context=c)
    elif k == 14:
        return complex.tanh(x, context=c)
    elif k == 15:
        return complex.asin(x, context=c)
    elif k == 16:
        return complex.acos(x, context=c)
    elif k == 17:
        return complex.atan(x, context=c)
    elif k == 18:
        return complex.asinh(x, context=c)
    elif k == 19:
        return complex.acosh(x, context=c)
    elif k == 20:
        return complex.atanh(x, context=c)
    else:
        return complex.pow(x, y, context=c)


def complex_operand(text: String, c: ArithmeticContext) raises -> Complex:
    var parts = text.split(";")
    return Complex(float_operand(String(parts[0]), c), float_operand(String(parts[1]), c))


struct ComplexWork[k: Int](Workload):
    var x: Complex
    var y: Complex
    var n: Int
    var c: ComplexContext

    def __init__(out self, operands: List[String], bits: Int, param: Int) raises:
        var real = ArithmeticContext(format=FloatFormat(bits))
        self.c = ComplexContext(real=real, imag=real)
        self.x = complex_operand(operands[0], real)
        self.y = complex_operand(operands[1], real) if len(operands) > 1 else Complex(0)
        self.n = param

    @no_inline
    def run(self) raises -> Bool:
        return not complex_value[Self.k](self.x, self.y, self.n, self.c).is_zero()

    def result(self) raises -> String:
        return complex_value[Self.k](self.x, self.y, self.n, self.c).to_json()


# Balls at the case's working precision.


def ball_names() -> List[String]:
    return [
        "add", "subtract", "multiply", "divide", "sqrt",
        "exp", "expm1", "log", "log1p", "sin", "cos", "tan", "atan", "asin", "acos",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "gamma", "gammaln", "digamma", "erf", "erfc", "expi", "lambertw",
    ]


def ball_value[k: Int](x: Ball, y: Ball, c: BallContext) raises -> Ball:
    comptime if k == 0:
        return ball.add(x, y, context=c)
    elif k == 1:
        return ball.subtract(x, y, context=c)
    elif k == 2:
        return ball.multiply(x, y, context=c)
    elif k == 3:
        return ball.divide(x, y, context=c)
    elif k == 4:
        return ball.sqrt(x, context=c)
    elif k == 5:
        return ball.exp(x, context=c)
    elif k == 6:
        return ball.expm1(x, context=c)
    elif k == 7:
        return ball.log(x, context=c)
    elif k == 8:
        return ball.log1p(x, context=c)
    elif k == 9:
        return ball.sin(x, context=c)
    elif k == 10:
        return ball.cos(x, context=c)
    elif k == 11:
        return ball.tan(x, context=c)
    elif k == 12:
        return ball.atan(x, context=c)
    elif k == 13:
        return ball.asin(x, context=c)
    elif k == 14:
        return ball.acos(x, context=c)
    elif k == 15:
        return ball.sinh(x, context=c)
    elif k == 16:
        return ball.cosh(x, context=c)
    elif k == 17:
        return ball.tanh(x, context=c)
    elif k == 18:
        return ball.asinh(x, context=c)
    elif k == 19:
        return ball.acosh(x, context=c)
    elif k == 20:
        return ball.atanh(x, context=c)
    elif k == 21:
        return ball.gamma(x, context=c)
    elif k == 22:
        return ball.gammaln(x, context=c)
    elif k == 23:
        return ball.digamma(x, context=c)
    elif k == 24:
        return ball.erf(x, context=c)
    elif k == 25:
        return ball.erfc(x, context=c)
    elif k == 26:
        return ball.expi(x, context=c)
    else:
        return ball.lambertw(x, context=c)


def ball_operand(mid: String, rad: String, bits: Int) raises -> Ball:
    return Ball(Rational(mid), Rational(rad), precision=bits)


struct BallWork[k: Int](Workload):
    var x: Ball
    var y: Ball
    var c: BallContext

    def __init__(out self, operands: List[String], bits: Int) raises:
        self.c = BallContext(bits)
        var a = operands[0].split(";")
        self.x = ball_operand(String(a[0]), String(a[1]), bits)
        self.y = Ball(0)
        if len(operands) > 1:
            var b = operands[1].split(";")
            self.y = ball_operand(String(b[0]), String(b[1]), bits)

    @no_inline
    def run(self) raises -> Bool:
        return ball_value[Self.k](self.x, self.y, self.c).is_exact()

    def result(self) raises -> String:
        return ball_value[Self.k](self.x, self.y, self.c).to_json()


# scipy's hypergeometric functions, as balls and as correctly rounded Floats.


def hypergeometric_names() -> List[String]:
    return ["hyp1f1", "gammainc", "gammaincc", "hyp2f1", "betainc"]


def hypergeometric_ball[k: Int](a: List[Ball], c: BallContext) raises -> Ball:
    comptime if k == 0:
        return ball.hyp1f1(a[0], a[1], a[2], context=c)
    elif k == 1:
        return ball.gammainc(a[0], a[1], context=c)
    elif k == 2:
        return ball.gammaincc(a[0], a[1], context=c)
    elif k == 3:
        return ball.hyp2f1(a[0], a[1], a[2], a[3], context=c)
    else:
        return ball.betainc(a[0], a[1], a[2], context=c)


def hypergeometric_float[k: Int](a: List[Float], c: ArithmeticContext) raises -> Float:
    comptime if k == 0:
        return float.hyp1f1(a[0], a[1], a[2], context=c)
    elif k == 1:
        return float.gammainc(a[0], a[1], context=c)
    elif k == 2:
        return float.gammaincc(a[0], a[1], context=c)
    elif k == 3:
        return float.hyp2f1(a[0], a[1], a[2], a[3], context=c)
    else:
        return float.betainc(a[0], a[1], a[2], context=c)


struct HypergeometricBallWork[k: Int](Workload):
    var args: List[Ball]
    var c: BallContext

    def __init__(out self, operands: List[String], bits: Int) raises:
        self.c = BallContext(bits)
        self.args = List[Ball]()
        for operand in operands:
            var parts = operand.split(";")
            self.args.append(ball_operand(String(parts[0]), String(parts[1]), bits))

    @no_inline
    def run(self) raises -> Bool:
        return hypergeometric_ball[Self.k](self.args, self.c).is_exact()

    def result(self) raises -> String:
        return hypergeometric_ball[Self.k](self.args, self.c).to_json()


struct HypergeometricFloatWork[k: Int](Workload):
    var args: List[Float]
    var c: ArithmeticContext

    def __init__(out self, operands: List[String], bits: Int) raises:
        self.c = ArithmeticContext(format=FloatFormat(bits))
        self.args = List[Float]()
        for operand in operands:
            self.args.append(float_operand(operand, self.c))

    @no_inline
    def run(self) raises -> Bool:
        return hypergeometric_float[Self.k](self.args, self.c).is_zero()

    def result(self) raises -> String:
        return hypergeometric_float[Self.k](self.args, self.c).to_json()


def complex_ball_names() -> List[String]:
    return ["add", "multiply", "divide", "exp", "log", "sqrt", "sin", "cos", "tan", "atan"]


def complex_ball_value[k: Int](x: ComplexBall, y: ComplexBall, c: BallContext) raises -> ComplexBall:
    comptime if k == 0:
        return complex_ball.add(x, y, context=c)
    elif k == 1:
        return complex_ball.multiply(x, y, context=c)
    elif k == 2:
        return complex_ball.divide(x, y, context=c)
    elif k == 3:
        return complex_ball.exp(x, context=c)
    elif k == 4:
        return complex_ball.log(x, context=c)
    elif k == 5:
        return complex_ball.sqrt(x, context=c)
    elif k == 6:
        return complex_ball.sin(x, context=c)
    elif k == 7:
        return complex_ball.cos(x, context=c)
    elif k == 8:
        return complex_ball.tan(x, context=c)
    else:
        return complex_ball.atan(x, context=c)


def complex_ball_operand(text: String, bits: Int) raises -> ComplexBall:
    var p = text.split(";")
    return ComplexBall(
        ball_operand(String(p[0]), String(p[1]), bits), ball_operand(String(p[2]), String(p[3]), bits)
    )


struct ComplexBallWork[k: Int](Workload):
    var x: ComplexBall
    var y: ComplexBall
    var c: BallContext

    def __init__(out self, operands: List[String], bits: Int) raises:
        self.c = BallContext(bits)
        self.x = complex_ball_operand(operands[0], bits)
        self.y = complex_ball_operand(operands[1], bits) if len(operands) > 1 else ComplexBall(0)

    @no_inline
    def run(self) raises -> Bool:
        return complex_ball_value[Self.k](self.x, self.y, self.c).is_exact()

    def result(self) raises -> String:
        return complex_ball_value[Self.k](self.x, self.y, self.c).to_json()


# Text: Integer and Float parsing and decimal printing.


struct IntegerTextWork[k: Int](Workload):
    """Code 0 parses the decimal text; code 1 prints the Integer in decimal."""

    var text: String
    var x: Integer

    def __init__(out self, operands: List[String]) raises:
        self.text = operands[0]
        self.x = Integer(operands[0])

    @no_inline
    def run(self) raises -> Bool:
        comptime if Self.k == 0:
            return Integer(self.text).__bool__()
        else:
            return String(self.x).byte_length() > 0

    def result(self) raises -> String:
        comptime if Self.k == 0:
            return Integer(self.text).to_json()
        else:
            return text_json(String(self.x))


struct FloatTextWork[k: Int](Workload):
    """Code 0 parses decimal text, rounding once; code 1 prints `digits`
    significant decimal digits, rounded to nearest."""

    var text: String
    var x: Float
    var digits: Int
    var c: ArithmeticContext

    def __init__(out self, operands: List[String], bits: Int, param: Int) raises:
        self.c = ArithmeticContext(format=FloatFormat(bits))
        self.text = operands[0]
        self.x = Float(0, context=self.c) if Self.k == 0 else float_operand(operands[0], self.c)
        self.digits = param

    @no_inline
    def run(self) raises -> Bool:
        comptime if Self.k == 0:
            return not Float(self.text, context=self.c).is_zero()
        else:
            return self.x.to_string(10, digits=self.digits).byte_length() > 0

    def result(self) raises -> String:
        comptime if Self.k == 0:
            return Float(self.text, context=self.c).to_json()
        else:
            return text_json(self.x.to_string(10, digits=self.digits))


# Batches: elementwise maps and reductions; a reduction rounds its exact total once.
# The `_loop` operations call the scalar function in a plain List loop instead,
# which measures the batch layer's own cost.


def float_batch_names() -> List[String]:
    return ["add", "multiply", "exp", "sum", "dot", "add_loop", "exp_loop"]


struct FloatBatchWork[k: Int](Workload):
    var xs: Batch[Float]
    var ys: Batch[Float]
    var lx: List[Float]
    var ly: List[Float]
    var c: ArithmeticContext

    def __init__(out self, operands: List[String], bits: Int) raises:
        self.c = ArithmeticContext(format=FloatFormat(bits))
        var xs = List[Float]()
        var ys = List[Float]()
        for t in operands[0].split(","):
            xs.append(float_operand(String(t), self.c))
        for t in operands[1].split(","):
            ys.append(float_operand(String(t), self.c))
        self.xs = Batch[Float](xs)
        self.ys = Batch[Float](ys)
        self.lx = xs^
        self.ly = ys^

    def looped(self) raises -> List[Float]:
        var out = List[Float](capacity=len(self.lx))
        for i in range(len(self.lx)):
            comptime if Self.k == 5:
                out.append(float.add(self.lx[i], self.ly[i], context=self.c))
            else:
                out.append(float.exp(self.lx[i], context=self.c))
        return out^

    @no_inline
    def run(self) raises -> Bool:
        comptime if Self.k >= 5:
            return len(self.looped()) > 0
        elif Self.k == 0:
            return len(vmap[float.add]()(self.xs, self.ys, context=self.c)) > 0
        elif Self.k == 1:
            return len(vmap[float.multiply]()(self.xs, self.ys, context=self.c)) > 0
        elif Self.k == 2:
            return len(vmap[float.exp]()(self.xs, context=self.c)) > 0
        elif Self.k == 3:
            return not sum(self.xs, context=self.c).is_zero()
        else:
            return not dot(self.xs, self.ys, context=self.c).is_zero()

    def result(self) raises -> String:
        comptime if Self.k >= 5:
            return list_json(Batch[Float](self.looped()))
        elif Self.k == 0:
            return list_json(vmap[float.add]()(self.xs, self.ys, context=self.c))
        elif Self.k == 1:
            return list_json(vmap[float.multiply]()(self.xs, self.ys, context=self.c))
        elif Self.k == 2:
            return list_json(vmap[float.exp]()(self.xs, context=self.c))
        elif Self.k == 3:
            return sum(self.xs, context=self.c).to_json()
        else:
            return dot(self.xs, self.ys, context=self.c).to_json()


def integer_batch_names() -> List[String]:
    return ["add", "multiply", "sum", "add_loop"]


struct IntegerBatchWork[k: Int](Workload):
    var xs: Batch[Integer]
    var ys: Batch[Integer]
    var lx: List[Integer]
    var ly: List[Integer]

    def __init__(out self, operands: List[String]) raises:
        var xs = List[Integer]()
        var ys = List[Integer]()
        for t in operands[0].split(","):
            xs.append(Integer(String(t)))
        for t in operands[1].split(","):
            ys.append(Integer(String(t)))
        self.xs = Batch[Integer](xs)
        self.ys = Batch[Integer](ys)
        self.lx = xs^
        self.ly = ys^

    def looped(self) raises -> List[Integer]:
        var out = List[Integer](capacity=len(self.lx))
        for i in range(len(self.lx)):
            out.append(self.lx[i] + self.ly[i])
        return out^

    @no_inline
    def run(self) raises -> Bool:
        comptime if Self.k == 3:
            return len(self.looped()) > 0
        elif Self.k == 0:
            return len(vmap[integer.add]()(self.xs, self.ys)) > 0
        elif Self.k == 1:
            return len(vmap[integer.multiply]()(self.xs, self.ys)) > 0
        else:
            return sum(self.xs).__bool__()

    def result(self) raises -> String:
        comptime if Self.k == 3:
            return list_json(Batch[Integer](self.looped()))
        elif Self.k == 0:
            return list_json(vmap[integer.add]()(self.xs, self.ys))
        elif Self.k == 1:
            return list_json(vmap[integer.multiply]()(self.xs, self.ys))
        else:
            return sum(self.xs).to_json()


def answer(row: String, command: List[String]) raises -> String:
    """Prepare the case on this catalog line, then check or time it."""
    var f = row.split("\t")
    var family = String(f[1])
    var op = String(f[2])
    var bits = Int(String(f[3]))
    var param = Int(String(f[4]))
    var operands = List[String]()
    for i in range(5, len(f)):
        operands.append(String(f[i]))
    if family == "integer":
        var code = code_of(op, integer_names())
        comptime for k in range(21):
            if code == k:
                return serve(IntegerWork[k](operands, param), command)
    elif family == "rational":
        var code = code_of(op, rational_names())
        comptime for k in range(6):
            if code == k:
                return serve(RationalWork[k](operands, param), command)
    elif family == "float":
        var code = code_of(op, float_names())
        comptime for k in range(35):
            if code == k:
                return serve(FloatWork[k](operands, bits, param), command)
    elif family == "complex":
        var code = code_of(op, complex_names())
        comptime for k in range(22):
            if code == k:
                return serve(ComplexWork[k](operands, bits, param), command)
    elif family == "ball":
        var code = code_of(op, ball_names())
        comptime for k in range(28):
            if code == k:
                return serve(BallWork[k](operands, bits), command)
    elif family == "hypergeometric_ball":
        var code = code_of(op, hypergeometric_names())
        comptime for k in range(5):
            if code == k:
                return serve(HypergeometricBallWork[k](operands, bits), command)
    elif family == "hypergeometric_float":
        var code = code_of(op, hypergeometric_names())
        comptime for k in range(5):
            if code == k:
                return serve(HypergeometricFloatWork[k](operands, bits), command)
    elif family == "complex_ball":
        var code = code_of(op, complex_ball_names())
        comptime for k in range(10):
            if code == k:
                return serve(ComplexBallWork[k](operands, bits), command)
    elif family == "integer_text":
        if op == "parse":
            return serve(IntegerTextWork[0](operands), command)
        return serve(IntegerTextWork[1](operands), command)
    elif family == "float_text":
        if op == "parse":
            return serve(FloatTextWork[0](operands, bits, param), command)
        return serve(FloatTextWork[1](operands, bits, param), command)
    elif family == "float_batch":
        var code = code_of(op, float_batch_names())
        comptime for k in range(7):
            if code == k:
                return serve(FloatBatchWork[k](operands, bits), command)
    elif family == "integer_batch":
        var code = code_of(op, integer_batch_names())
        comptime for k in range(4):
            if code == k:
                return serve(IntegerBatchWork[k](operands), command)
    raise Error(String("Unsupported family: ", family))


def read_line(stdin: FileHandle) raises -> String:
    """One command from standard input. The standard library's `input()`
    declares libc's `free` with attributes that conflict with the library's own
    declaration, so the bytes are read through a file handle; commands are short."""
    var line = String()
    while True:
        var byte = stdin.read(1)
        if not byte.byte_length():
            raise Error("End of input")
        if byte == "\n":
            return line
        line += byte


def main() raises:
    var rows = List[String]()
    with open(argv()[1], "r") as source:
        for line in source.read().split("\n"):
            if line.byte_length():
                rows.append(String(line))
    var stdin = open("/dev/stdin", "r")
    while True:
        var command = List[String]()
        for word in read_line(stdin).split(" "):
            command.append(String(word))
        if command[0] == "quit":
            return
        try:
            print(answer(rows[Int(command[1])], command), flush=True)
        except error:
            print(String('{"error":"', String(error).replace('"', "'").replace("\n", " "), '"}'), flush=True)
