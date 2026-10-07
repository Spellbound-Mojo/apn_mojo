"""Elementary functions (T-ELEM): correct rounding and flags in every mode
against MPFR (tests/fixtures/elementary.txt, written by gmpy2), including the
special and exact inputs of Appendices A and D.1; the retry path; the
Appendix H.2 vectors; ball inclusion and tightness; budgets; and determinism
through vmap."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, Rational, Float, Ball, FloatFormat, ArithmeticContext, RoundingMode, BallContext, Batch, vmap
from apn_mojo import float
from apn_mojo import ball
from apn_mojo import exp, sin_cos, rootn, atan2
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.float._elementary import _real_rounded, _pow_rounded, _atan2_rounded
from apn_mojo.float._functions import _rootn_float
from apn_mojo.ball._kernels import (
    _EXP, _EXPM1, _EXP2, _LOG, _LOG1P, _LOG2, _LOG10, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS,
    _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH,
)


def _modes() -> List[RoundingMode]:
    return [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]


def _code(name: String) -> Int:
    var names: List[String] = [
        "sqrt", "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sin", "cos", "tan",
        "atan", "asin", "acos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ]
    for i in range(len(names)):
        if names[i] == name:
            return i
    return -1


def _input(fields: List[String], at: Int) raises -> Float:
    """A Float from `sign significand exponent precision` fields."""
    var negative = fields[at] == "-"
    var body = fields[at + 1]
    var p = Int(fields[at + 3])
    var context = ArithmeticContext(format=FloatFormat(p))
    if body == "zero":
        var zero = Float.zero(context=context)
        return -zero if negative else zero
    if body == "inf":
        return Float.infinity(negative=negative, context=context)
    if body == "nan":
        return Float.nan(context=context)
    var m = Integer.parse(body, base=16)
    return Float(_rounded=_RoundedBinary(1, negative, m, Int(fields[at + 2]), FloatFormat(p), NumericStatus()))


def _matches(result: _RoundedBinary, fields: List[String], at: Int, p: Int) raises -> Bool:
    var kind = Int(fields[at])
    var negative = fields[at + 1] == "-"
    if result.kind != kind:
        return False
    if kind == 3:
        return True
    if result.negative != negative:
        return False
    if kind != 1:
        return True
    return result.significand == Integer.parse(fields[at + 2], base=16) and result.exponent == Int(fields[at + 3])


def _flags(status: NumericStatus) -> String:
    var bits = [status.inexact(), status.underflow(), status.overflow(), status.divide_by_zero(), status.invalid()]
    var text = String()
    for b in bits:
        text += "1" if b else "0"
    return text


def _lines() raises -> List[String]:
    var text: String
    with open("tests/fixtures/elementary.txt", "r") as source:
        text = source.read()
    var result = List[String]()
    for line in text.splitlines():
        if line.byte_length():
            result.append(String(line))
    return result^


def _evaluate(name: String, args: List[Float], n: Int, context: ArithmeticContext, guard: Int) raises -> _RoundedBinary:
    if name == "pow":
        return _pow_rounded(args[0], args[1], context, guard)
    if name == "atan2":
        return _atan2_rounded(args[0], args[1], context, guard)
    if name == "rootn":
        return _rootn_float(args[0], n, context)
    return _real_rounded(_code(name), args[0], context, guard)


def test_correct_rounding_against_mpfr() raises:
    var modes = _modes()
    var count = 0
    var index = 0
    for line in _lines():
        var words = line.split(" ")
        var fields = List[String]()
        for w in words:
            fields.append(String(w))
        var name = fields[0]
        var p = Int(fields[1])
        var mode = modes[Int(fields[2])]
        var n = Int(fields[3])
        var arity = 2 if name == "pow" or name == "atan2" else 1
        var args = List[Float]()
        for i in range(arity):
            args.append(_input(fields, 4 + 4 * i))
        var at = 4 + 4 * arity
        var context = ArithmeticContext(format=FloatFormat(p), rounding=mode)
        var result = _evaluate(name, args, n, context, 0)
        var description = line
        assert_true(_matches(result, fields, at, p), String(name, " value: ", description, " got ", Float(_rounded=result)))
        assert_equal(_flags(result.status), fields[at + 4], String(name, " flags: ", description))
        if index % 7 == 0 and name != "rootn":
            var retried = _evaluate(name, args, n, context, 1)
            assert_true(_matches(retried, fields, at, p) and retried.status == result.status, String(name, " retry: ", description))
        index += 1
        count += 1
    assert_true(count > 15000)


def _h2(m: Int, exponent: Int) raises -> Float:
    return Float(_rounded=_RoundedBinary(1, False, Integer(m), exponent, FloatFormat(53), NumericStatus()))


def test_appendix_h2_vectors() raises:
    # exp(1), log(3), sin(1), atan(2), cos(1e22) at 53 bits in every mode.
    var modes = _modes()
    var nearest: List[Int] = [0x15bf0a8b145769, 0x1193ea7aad030b, 0x1aed548f090cee, 0x11b6e192ebbe44, 0x10be2cef01c8f4]
    var down: List[Int] = [0x15bf0a8b145769, 0x1193ea7aad030a, 0x1aed548f090cee, 0x11b6e192ebbe44, 0x10be2cef01c8f3]
    var exponents: List[Int] = [2, 1, 0, 1, 0]
    var ten22 = Float(Integer(10) ** 22, context=ArithmeticContext(format=FloatFormat(53)))
    for i in range(5):
        for j in range(5):
            var context = ArithmeticContext(format=FloatFormat(53), rounding=modes[j])
            var value: Float
            if i == 0:
                value = float.exp(Float(1), context=context)
            elif i == 1:
                value = float.log(Float(3), context=context)
            elif i == 2:
                value = float.sin(Float(1), context=context)
            elif i == 3:
                value = float.atan(Float(2), context=context)
            else:
                value = float.cos(ten22, context=context)
            var lower = _h2(down[i], exponents[i])
            var expected = _h2(nearest[i], exponents[i]) if j == 0 else (lower if j == 1 or j == 3 else _h2(down[i] + 1, exponents[i]))
            assert_true(value == expected, String("H.2 row ", i, " mode ", modes[j]))


def _contains(x: Ball, low: Float, high: Float) raises -> Bool:
    return x.lower_rational() <= low.to_rational_exact() and high.to_rational_exact() <= x.upper_rational()


def _bounds(name: String, t: Float) raises -> Tuple[Float, Float]:
    """`f(t)` rounded down and up at 300 bits: a rigorous enclosure, since the
    Float functions round correctly."""
    var down = ArithmeticContext(format=FloatFormat(300), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(300), rounding=RoundingMode.toward_positive)
    var code = _code(name)
    return (Float(_rounded=_real_rounded(code, t, down)), Float(_rounded=_real_rounded(code, t, up)))


def test_ball_inclusion() raises:
    var names: List[String] = [
        "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sin", "cos", "tan", "atan", "asin", "acos",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    ]
    var state = UInt64(0x2545F4914F6CDD1D)
    for name in names:
        for _ in range(25):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var center = Rational(Int(state % 2000) - 1000, 1000)
            if name == "log" or name == "log2" or name == "log10" or name == "acosh":
                center = Rational(Int(state % 3000) + 1001, 1000)
            if name == "asin" or name == "acos" or name == "atanh":
                center = Rational(Int(state % 1800) - 900, 1000)
            var radius = Rational(1, Integer(2) ** Int(4 + (state >> 20) % 50))
            var x = Ball(center, radius, precision=96)
            var y = ball.exp(x)
            if name == "expm1":
                y = ball.expm1(x)
            elif name == "exp2":
                y = ball.exp2(x)
            elif name == "log":
                y = ball.log(x)
            elif name == "log1p":
                y = ball.log1p(x)
            elif name == "log2":
                y = ball.log2(x)
            elif name == "log10":
                y = ball.log10(x)
            elif name == "sin":
                y = ball.sin(x)
            elif name == "cos":
                y = ball.cos(x)
            elif name == "tan":
                y = ball.tan(x)
            elif name == "atan":
                y = ball.atan(x)
            elif name == "asin":
                y = ball.asin(x)
            elif name == "acos":
                y = ball.acos(x)
            elif name == "sinh":
                y = ball.sinh(x)
            elif name == "cosh":
                y = ball.cosh(x)
            elif name == "tanh":
                y = ball.tanh(x)
            elif name == "asinh":
                y = ball.asinh(x)
            elif name == "acosh":
                y = ball.acosh(x)
            elif name == "atanh":
                y = ball.atanh(x)
            if y.is_indeterminate():
                continue
            assert_true(y.is_finite(), String(name, " finite at ", x))
            var points: List[Float] = [x.lower(), x.upper(), x.midpoint()]
            for t in points:
                var pair = _bounds(name, t)
                assert_true(_contains(y, pair[0], pair[1]), String(name, " inclusion at ", t, " of ", x))


def test_tightness_against_arb() raises:
    # Appendix E.12: radii at w = 64 for X = [1 +/- 2**-20], within 4 of Arb's.
    var x = Ball(1, Rational(1, 2**20), precision=64)
    var c = BallContext(64)
    var radii: List[Rational] = [Rational(26040, 10**10), Rational(9564, 10**10), Rational(5162, 10**10), Rational(47684, 10**11)]
    var results: List[Ball] = [ball.exp(x, context=c), ball.log(x, context=c), ball.sin(x, context=c), ball.atan(x, context=c)]
    for i in range(4):
        var ratio = results[i].radius().to_rational_exact() / radii[i]
        assert_true(ratio <= 4, String("tightness ", i, ": ", results[i]))
    var narrow = Ball(Rational(1, 2), Rational(1, 2**30), precision=64)
    assert_true(ball.asin(narrow, context=c).radius().to_rational_exact() / Rational(10754, 10**13) <= 4)
    assert_true(ball.tan(narrow, context=c).radius().to_rational_exact() / Rational(12272, 10**13) <= 4)


def test_domains_and_budgets() raises:
    # A function undefined somewhere in its ball is indeterminate.
    assert_true(ball.log(Ball(0, 1)).is_indeterminate())
    assert_true(ball.asin(Ball(1, Rational(1, 10))).is_indeterminate())
    assert_true(ball.atanh(Ball(1)).is_indeterminate())
    assert_true(ball.tan(Ball(Rational(355, 226), Rational(1, 100))).is_indeterminate())
    # sin(2**(10**6)) needs a million bits of pi: past the default budget, the
    # Float raises and the ball is [0 +/- 1].
    var huge = Float(Integer(1) << 1000000, context=ArithmeticContext(format=FloatFormat(53)))
    with assert_raises(contains="precision budget exceeded: sin"):
        _ = float.sin(huge, context=ArithmeticContext(format=FloatFormat(53)))
    var unit = ball.sin(Ball(huge))
    assert_true(unit.midpoint().is_zero() and unit.radius() == Float(1))
    # A tight budget raises at once for any non-exact input.
    with assert_raises(contains="precision budget exceeded"):
        _ = float.exp(Float(1), context=ArithmeticContext(format=FloatFormat(53), max_precision=61))
    # Exact cases need no budget.
    assert_true(float.exp(Float(0), context=ArithmeticContext(format=FloatFormat(53), max_precision=2)) == Float(1))


def test_exact_rationals_and_dispatch() raises:
    # A Rational that is not a binary fraction is evaluated exactly, as 1/3.
    var c = ArithmeticContext(format=FloatFormat(60))
    var third = float.exp(Rational(1, 3), context=c)
    var direct = float.exp(Float(Rational(1, 3), context=ArithmeticContext(format=FloatFormat(400))), context=c)
    assert_true(third == direct)
    assert_true(float.pow(8, Rational(1, 3), context=c) == Float(2))
    assert_true(float.pow(Rational(1, 9), Rational(-1, 2), context=c) == Float(3))
    # The package-level names choose the family.
    assert_true(exp(Float(1)) == float.exp(Float(1)))
    assert_true(exp(Ball(1)).same_representation(ball.exp(Ball(1))))
    var pair = sin_cos(Float(1))
    assert_true(pair[0] == float.sin(Float(1)) and pair[1] == float.cos(Float(1)))
    assert_true(rootn(Float(-27), 3) == Float(-3))
    assert_true(atan2(Float(0), Float(-1)) == float.atan2(Float(0), Float(-1)))


def test_determinism_through_vmap() raises:
    var values = List[Float]()
    var state = UInt64(88172645463325252)
    for _ in range(3000):
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        values.append(Float(Rational(Int(state % 20000) - 10000, 997), context=ArithmeticContext(format=FloatFormat(64))))
    var batch = Batch[Float](values)
    var c = ArithmeticContext(format=FloatFormat(64))
    var mapped_exp = vmap[float.exp]()(batch, context=c)
    var mapped_sin = vmap[float.sin]()(batch, context=c)
    var mapped_atan = vmap[float.atan]()(batch, context=c)
    for i in range(len(values)):
        assert_true(mapped_exp[i] == float.exp(values[i], context=c))
        assert_true(mapped_sin[i] == float.sin(values[i], context=c))
        assert_true(mapped_atan[i] == float.atan(values[i], context=c))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
