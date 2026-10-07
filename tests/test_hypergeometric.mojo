"""Hypergeometric functions (T-HYP): hyp1f1, gammainc, gammaincc, hyp2f1
and betainc correctly rounded in every mode against Arb, and their exact
values against exact fractions (tests/fixtures/hypergeometric.txt); scipy's
poles and conventions, the limits at infinite arguments, the exact cases,
ball inclusion and dispatch through vmap, four arguments included."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, Rational, Float, Ball, FloatFormat, ArithmeticContext, RoundingMode, Batch, vmap
from apn_mojo import float
from apn_mojo import ball
from apn_mojo import hyp1f1, gammainc, hyp2f1, betainc
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.float._special import _hyp1f1_rounded, _gammainc_rounded, _hyp2f1_rounded, _betainc_rounded


def _modes() -> List[RoundingMode]:
    return [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]


def _input(fields: List[String], at: Int) raises -> Float:
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
    return Float(_rounded=_RoundedBinary(1, negative, Integer.parse(body, base=16), Int(fields[at + 2]), FloatFormat(p), NumericStatus()))


def _matches(result: _RoundedBinary, fields: List[String], at: Int) raises -> Bool:
    var kind = Int(fields[at])
    if result.kind != kind:
        return False
    if kind == 3:
        return True
    if result.negative != (fields[at + 1] == "-"):
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


def _contains(b: Ball, x: Float) raises -> Bool:
    return b.is_finite() and b.lower() <= x and x <= b.upper()


def _arity(name: String) -> Int:
    if name == "hyp2f1":
        return 4
    return 3 if name == "hyp1f1" or name == "betainc" else 2


def _evaluate(name: String, fields: List[String], context: ArithmeticContext) raises -> _RoundedBinary:
    if name == "hyp1f1":
        return _hyp1f1_rounded(_input(fields, 4), _input(fields, 8), _input(fields, 12), context)
    if name == "hyp2f1":
        return _hyp2f1_rounded(_input(fields, 4), _input(fields, 8), _input(fields, 12), _input(fields, 16), context)
    if name == "betainc":
        return _betainc_rounded(_input(fields, 4), _input(fields, 8), _input(fields, 12), context)
    return _gammainc_rounded(_input(fields, 4), _input(fields, 8), name == "gammaincc", context)


def test_correct_rounding() raises:
    var modes = _modes()
    var text: String
    with open("tests/fixtures/hypergeometric.txt", "r") as source:
        text = source.read()
    var failures = List[String]()
    var count = 0
    for line in text.splitlines():
        if not line.byte_length():
            continue
        var fields = List[String]()
        for w in line.split(" "):
            fields.append(String(w))
        var context = ArithmeticContext(format=FloatFormat(Int(fields[1])), rounding=modes[Int(fields[2])])
        var at = 4 + 4 * _arity(fields[0])
        try:
            var result = _evaluate(fields[0], fields, context)
            if not _matches(result, fields, at):
                failures.append(String(line, "  got ", Float(_rounded=result)))
            elif _flags(result.status) != fields[at + 4]:
                failures.append(String(line, "  flags ", _flags(result.status)))
        except e:
            failures.append(String(line, "  raised ", e))
        count += 1
    for i in range(min(len(failures), 30)):
        print(failures[i])
    assert_equal(len(failures), 0, String(len(failures), " mismatches"))
    assert_true(count > 8000)


def test_hyp1f1_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var up = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_positive)
    var inf = Float.infinity(context=c)
    var one = Float(1)
    # Exact values in directed modes: polynomials, scipy's truncated
    # exponential, Kummer's zero and the elementary integer case.
    assert_true(float.hyp1f1(Float(-2), Float(3), Float(Rational(3, 2)), context=down) == Float(Rational(3, 16), context=c))
    assert_true(float.hyp1f1(Float(-2), Float(-2), Float(Rational(3, 2)), context=up) == Float(Rational(29, 8), context=c))
    assert_true(float.hyp1f1(Float(-2), Float(-3), Float(Rational(3, 2)), context=down) == Float(Rational(19, 8), context=c))
    assert_true(float.hyp1f1(Float(4), Float(3), Float(-3), context=down).is_zero())
    assert_true(float.hyp1f1(Float(2), Float(4), Float(2), context=down) == Float(3))
    assert_true(float.hyp1f1(Float(2), Float(4), Float(2), context=up) == Float(3))
    assert_true(float.hyp1f1(Float(Rational(5, 2)), Float(3), Float(0), context=down) == one)
    assert_true(float.hyp1f1(Float(0), Float(Rational(7, 2)), Float(5), context=down) == one)
    # scipy's poles and parameters.
    assert_true(float.hyp1f1(one, Float(-2), one) == inf and float.hyp1f1(Float(0), Float(0), one) == inf)
    assert_true(float.hyp1f1(Float(-3), Float(-2), one) == inf and float.hyp1f1(one, Float(0), Float(0)) == inf)
    assert_true(float.hyp1f1(inf, Float(2), one).is_nan() and float.hyp1f1(one, inf, one) == one)
    # Limits at infinite x.
    assert_true(float.hyp1f1(one, Float(2), inf) == inf and float.hyp1f1(one, Float(2), -inf).is_zero())
    assert_true(float.hyp1f1(Float(Rational(-3, 2)), Float(2), -inf) == inf)
    assert_true(float.hyp1f1(Float(-2), Float(2), -inf) == inf and float.hyp1f1(Float(-3), Float(2), -inf) == inf)
    assert_true(float.hyp1f1(Float(-3), Float(2), inf) == -inf)
    assert_true(float.hyp1f1(Float(2), one, -inf).is_zero() and float.hyp1f1(Float(Rational(1, 2)), Float(Rational(-1, 2)), inf) == -inf)
    # e**x for a = b, and overflow far out.
    assert_true(float.hyp1f1(Float(Rational(3, 2)), Float(Rational(3, 2)), one, context=c) == float.exp(one, context=c))
    # Overflow far out where the format's emax is small; in the default
    # format, emax = 2**62 - 1, e**(2**60) is representable and that x raises
    # the budget error rather than overflow.
    var far = Float(Integer(1) << 60, context=c)
    var binary64 = ArithmeticContext(format=FloatFormat(53, emin=-1021, emax=1024))
    assert_true(float.hyp1f1(one, Float(2), far, context=binary64) == inf)
    with assert_raises(contains="budget"):
        _ = float.hyp1f1(one, Float(2), far, context=c)
    # M(1, 2, x) = (e**x - 1)/x, through the series and the expansion.
    var xs: List[Float] = [Float(Rational(1, 3), context=c), Float(Rational(-7, 2), context=c), Float(45, context=c), Float(-300, context=c), Float(2000, context=c)]
    var wide = ArithmeticContext(format=FloatFormat(300))
    for x in xs:
        var expected = Float((float.exp(x, context=wide) - Float(1)) / x, context=c)
        assert_true(float.hyp1f1(one, Float(2), x, context=c) == expected, String("M(1, 2, ", x, ")"))
    # M(a, b, -x) for a large x follows Gamma(b)/Gamma(b-a) x**-a.
    var big = float.hyp1f1(Float(Rational(1, 2)), Float(Rational(3, 2)), Float(-(Integer(1) << 70), context=c), context=c)
    assert_true(big > Float(0) and big < Float(Rational(1, 1 << 30)))


def test_hyp1f1_balls() raises:
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var params: List[Rational] = [Rational(1, 3), Rational(-5, 2), Rational(7, 4), Rational(3)]
    var lows: List[Rational] = [Rational(3, 2), Rational(1, 5), Rational(-7, 3), Rational(9, 2)]
    var points: List[Rational] = [Rational(5, 7), Rational(-13, 2), Rational(60), Rational(-150)]
    for i in range(len(params)):
        var a = Ball(params[i], Rational(1, 1 << 30), precision=96)
        var b = Ball(lows[i], Rational(1, 1 << 30), precision=96)
        var x = Ball(points[i], Rational(1, 1 << 30), precision=96)
        var m = ball.hyp1f1(a, b, x)
        assert_false(m.is_indeterminate(), String("hyp1f1 at ", a, ", ", b, ", ", x))
        var ends: List[Float] = [a.lower(), a.upper()]
        var lowers: List[Float] = [b.lower(), b.upper()]
        var xs: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for p in ends:
            for q in lowers:
                for t in xs:
                    var low = Float(_rounded=_hyp1f1_rounded(p, q, t, down))
                    var high = Float(_rounded=_hyp1f1_rounded(p, q, t, up))
                    assert_true(_contains(m, low) and _contains(m, high), String("hyp1f1 at ", p, ", ", q, ", ", t, " in ", m))
    # Exact balls: rational values exactly, poles indeterminate.
    var exact = ball.hyp1f1(Ball(-2), Ball(3), Ball(Rational(3, 2)))
    assert_true(exact.is_exact() and exact.midpoint() == Float(Rational(3, 16)))
    assert_true(ball.hyp1f1(Ball(1), Ball(-1), Ball(1)).is_indeterminate())
    assert_true(ball.hyp1f1(Ball(1), Ball(Rational(-1), Rational(1, 1 << 20)), Ball(1)).is_indeterminate())
    # The root dispatcher and vmap.
    var c = ArithmeticContext(format=FloatFormat(64))
    assert_true(hyp1f1(Float(1), Float(2), Float(3), context=c) == float.hyp1f1(Float(1), Float(2), Float(3), context=c))
    var xs = List[Float]()
    for i in range(1, 12):
        xs.append(Float(Rational(i, 3), context=c))
    var mapped = vmap[float.hyp1f1]()(Float(Rational(1, 2), context=c), Float(Rational(5, 2), context=c), Batch[Float](xs))
    for i in range(len(xs)):
        assert_true(mapped[i] == float.hyp1f1(Float(Rational(1, 2), context=c), Float(Rational(5, 2), context=c), xs[i]))


def test_gammainc_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var inf = Float.infinity(context=c)
    var one = Float(1)
    var zero = Float.zero()
    # scipy's conventions.
    assert_true(float.gammainc(zero, one) == one and float.gammaincc(zero, one).is_zero())
    assert_true(float.gammainc(zero, zero).is_nan() and float.gammaincc(zero, zero).is_nan())
    assert_true(float.gammainc(one, zero).is_zero() and float.gammaincc(one, zero) == one)
    assert_true(float.gammainc(-one, one).is_nan() and float.gammaincc(one, -one).is_nan())
    assert_true(float.gammainc(inf, one).is_zero() and float.gammaincc(inf, one) == one)
    assert_true(float.gammainc(one, inf) == one and float.gammaincc(one, inf).is_zero())
    assert_true(float.gammainc(inf, inf).is_nan())
    # P(1, x) = 1 - e**-x and Q(1, x) = e**-x, through the series, the
    # expansion and 1 - P.
    var wide = ArithmeticContext(format=FloatFormat(300))
    var xs: List[Float] = [Float(Rational(1, 7), context=c), Float(Rational(5, 2), context=c), Float(30, context=c), Float(700, context=c)]
    for x in xs:
        var e = float.exp(-x, context=wide)
        assert_true(float.gammaincc(one, x, context=c) == Float(e, context=c), String("Q(1, ", x, ")"))
        assert_true(float.gammainc(one, x, context=c) == Float(Float(1) - e, context=c), String("P(1, ", x, ")"))
    # P(1/2, x) = erf(sqrt x): at x = 1/4, erf(1/2).
    var half = Float(Rational(1, 2))
    assert_true(float.gammainc(half, Float(Rational(1, 4)), context=c) == float.erf(half, context=c))
    # Tiny distances from 1, and underflow.
    assert_true(float.gammaincc(one, Float(Rational(1, 1 << 80), context=c), context=c) == one)
    assert_true(float.gammaincc(one, Float(Rational(1, 1 << 80), context=c), context=down) < one)
    assert_true(float.gammainc(Float(2), Float(Integer(1) << 70, context=c), context=down) < one)
    assert_true(float.gammaincc(Float(2), Float(Integer(1) << 70, context=c), context=c).is_zero())


def test_gammainc_balls() raises:
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var shapes: List[Rational] = [Rational(1, 3), Rational(5, 2), Rational(40), Rational(7, 4)]
    var points: List[Rational] = [Rational(5, 7), Rational(3), Rational(38), Rational(90)]
    for i in range(len(shapes)):
        var a = Ball(shapes[i], Rational(1, 1 << 30), precision=96)
        var x = Ball(points[i], Rational(1, 1 << 30), precision=96)
        var p = ball.gammainc(a, x)
        var q = ball.gammaincc(a, x)
        assert_false(p.is_indeterminate() or q.is_indeterminate())
        var ends: List[Float] = [a.lower(), a.midpoint(), a.upper()]
        var xs: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for s in ends:
            for t in xs:
                assert_true(_contains(p, Float(_rounded=_gammainc_rounded(s, t, False, down))) and _contains(p, Float(_rounded=_gammainc_rounded(s, t, False, up))))
                assert_true(_contains(q, Float(_rounded=_gammainc_rounded(s, t, True, down))) and _contains(q, Float(_rounded=_gammainc_rounded(s, t, True, up))))
    # A ball reaching 0 in x, and one reaching below 0.
    var near_zero = ball.gammainc(Ball(2), Ball(Rational(1, 1 << 20), Rational(1, 1 << 20), precision=64))
    assert_true(near_zero.is_finite() and near_zero.lower() <= Float(0))
    assert_true(ball.gammainc(Ball(2), Ball(Rational(0), Rational(1, 8))).is_indeterminate())
    var c = ArithmeticContext(format=FloatFormat(64))
    assert_true(gammainc(Float(3), Float(2), context=c) == float.gammainc(Float(3), Float(2), context=c))
    var xs = List[Float]()
    for i in range(1, 12):
        xs.append(Float(Rational(i, 2), context=c))
    var mapped = vmap[float.gammaincc]()(Float(Rational(5, 2), context=c), Batch[Float](xs))
    for i in range(len(xs)):
        assert_true(mapped[i] == float.gammaincc(Float(Rational(5, 2), context=c), xs[i]))


def test_hyp2f1_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var up = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_positive)
    var inf = Float.infinity(context=c)
    var one = Float(1)
    var half = Float(Rational(1, 2))
    # Exact values in directed modes.
    assert_true(float.hyp2f1(Float(-2), one, Float(3), Float(2), context=down) == Float(Rational(1, 3), context=c))
    assert_true(float.hyp2f1(Float(-1), Float(2), Float(-2), half, context=up) == Float(Rational(3, 2)))
    assert_true(float.hyp2f1(half, one, one, Float(Rational(3, 4)), context=down) == Float(2))
    assert_true(float.hyp2f1(one, one, Float(3), one, context=up) == Float(2))
    assert_true(float.hyp2f1(one, Float(2), one, half, context=down) == Float(4))
    # scipy's poles; NaN past 1 where scipy returns +inf.
    assert_true(float.hyp2f1(one, Float(2), Float(-2), half) == inf and float.hyp2f1(Float(-2), Float(2), Float(-1), half) == inf)
    assert_true(float.hyp2f1(one, one, Float(2), one) == inf and float.hyp2f1(one, Float(2), Float(2), one) == inf)
    assert_true(float.hyp2f1(one, one, Float(2), Float(2)).is_nan() and float.hyp2f1(half, one, one, Float(2)).is_nan())
    assert_true(float.hyp2f1(one, one, Float(0), Float(0)) == one)
    # F(1, 1; 2; x) = -log(1-x)/x: the series, the integer c - a - b at
    # x = 0.9, and Pfaff's transformation with b - a = 0 at x = -5.
    var wide = ArithmeticContext(format=FloatFormat(300))
    var xs: List[Float] = [Float(Rational(3, 10), context=c), Float(Rational(9, 10), context=c), Float(-5, context=c), Float(Rational(-7, 10), context=c)]
    for x in xs:
        var rest = Float(Float(1) - x, context=wide)
        var expected = Float(-float.log(rest, context=wide) / x, context=c)
        assert_true(float.hyp2f1(one, one, Float(2), x, context=c) == expected, String("F(1, 1; 2; ", x, ")"))
    # F(1/2, 1/2; 3/2; x**2) = asin(x)/x, a non-integer c - a - b.
    var t = Float(Rational(9, 10), context=c)
    var square = Float(t * t, context=wide)
    assert_true(float.hyp2f1(half, half, Float(Rational(3, 2)), square, context=c) == Float(float.asin(t, context=wide) / t, context=c))
    # Infinite x: a polynomial's leading term; NaN otherwise.
    assert_true(float.hyp2f1(Float(-2), one, Float(3), inf) == inf and float.hyp2f1(Float(-1), one, Float(3), inf) == -inf)
    assert_true(float.hyp2f1(one, one, Float(2), -inf).is_nan())


def test_hyp2f1_balls() raises:
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var a_values: List[Rational] = [Rational(1, 3), Rational(-5, 2), Rational(1), Rational(7, 4)]
    var b_values: List[Rational] = [Rational(3, 2), Rational(1, 5), Rational(1), Rational(-1, 3)]
    var c_values: List[Rational] = [Rational(5, 2), Rational(7, 3), Rational(2), Rational(9, 2)]
    var x_values: List[Rational] = [Rational(1, 3), Rational(-13, 2), Rational(9, 10), Rational(-3, 4)]
    for i in range(len(a_values)):
        var a = Ball(a_values[i], Rational(1, 1 << 30), precision=96)
        var b = Ball(b_values[i], Rational(1, 1 << 30), precision=96)
        var cb = Ball(c_values[i], Rational(1, 1 << 30), precision=96)
        var x = Ball(x_values[i], Rational(1, 1 << 30), precision=96)
        var f = ball.hyp2f1(a, b, cb, x)
        assert_false(f.is_indeterminate(), String("hyp2f1 at ", a, ", ", b, ", ", cb, ", ", x))
        var points: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for p in points:
            var low = Float(_rounded=_hyp2f1_rounded(a.lower(), b.upper(), cb.midpoint(), p, down))
            var high = Float(_rounded=_hyp2f1_rounded(a.lower(), b.upper(), cb.midpoint(), p, up))
            assert_true(_contains(f, low) and _contains(f, high), String("hyp2f1 at ", p, " in ", f))
    assert_true(ball.hyp2f1(Ball(1), Ball(1), Ball(2), Ball(Rational(1), Rational(1, 1 << 20))).is_indeterminate())
    assert_true(ball.hyp2f1(Ball(1), Ball(1), Ball(Rational(-1), Rational(1, 1 << 20)), Ball(Rational(1, 3))).is_indeterminate())
    # Four-argument vmap and the root dispatcher.
    var c = ArithmeticContext(format=FloatFormat(64))
    assert_true(hyp2f1(Float(1), Float(2), Float(3), Float(Rational(1, 4)), context=c) == float.hyp2f1(Float(1), Float(2), Float(3), Float(Rational(1, 4)), context=c))
    var xs = List[Float]()
    for i in range(1, 12):
        xs.append(Float(Rational(i - 6, 7), context=c))
    var p = Float(Rational(1, 2), context=c)
    var q = Float(Rational(3, 2), context=c)
    var r = Float(Rational(5, 2), context=c)
    var mapped = vmap[float.hyp2f1]()(p, q, r, Batch[Float](xs))
    for i in range(len(xs)):
        assert_true(mapped[i] == float.hyp2f1(p, q, r, xs[i]))


def test_betainc_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var up = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_positive)
    var inf = Float.infinity(context=c)
    var one = Float(1)
    var zero = Float.zero()
    var half = Float(Rational(1, 2))
    # scipy's edges.
    assert_true(float.betainc(zero, one, half) == one and float.betainc(zero, one, zero).is_zero())
    assert_true(float.betainc(one, zero, half).is_zero() and float.betainc(one, zero, one) == one)
    assert_true(float.betainc(zero, zero, half).is_nan() and float.betainc(inf, inf, half).is_nan())
    assert_true(float.betainc(inf, one, half).is_zero() and float.betainc(one, inf, half) == one)
    assert_true(float.betainc(-one, one, half).is_nan() and float.betainc(one, one, Float(Rational(3, 2))).is_nan())
    assert_true(float.betainc(one, one, zero).is_zero() and float.betainc(one, one, one) == one)
    # Exact values: integer shapes, and x**a rational through the symmetry.
    assert_true(float.betainc(Float(2), Float(3), Float(Rational(1, 4)), context=down) == Float(Rational(67, 256)))
    assert_true(float.betainc(one, half, Float(Rational(3, 4)), context=up) == half)
    assert_true(float.betainc(half, one, Float(Rational(1, 4)), context=down) == half)
    # I_x(a, 1) = x**a and I_x(1, b) = 1 - (1-x)**b.
    var wide = ArithmeticContext(format=FloatFormat(300))
    var x = Float(Rational(2, 7), context=c)
    var a = Float(Rational(7, 3), context=c)
    assert_true(float.betainc(a, one, x, context=c) == float.pow(x, a, context=c))
    assert_true(float.betainc(one, a, x, context=c) == Float(Float(1) - float.pow(Float(Float(1) - x, context=wide), a, context=wide), context=c))
    # Within a tiny distance of 1.
    var tiny = Float(Rational(1, 1 << 70), context=c)
    assert_true(float.betainc(Float(2), Float(3), Float(Float(1) - tiny, context=wide), context=down) < one)


def test_betainc_balls() raises:
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var a_values: List[Rational] = [Rational(1, 3), Rational(5, 2), Rational(40), Rational(7, 4)]
    var b_values: List[Rational] = [Rational(3, 2), Rational(1, 5), Rational(30), Rational(9)]
    var x_values: List[Rational] = [Rational(1, 3), Rational(13, 14), Rational(1, 2), Rational(1, 9)]
    for i in range(len(a_values)):
        var a = Ball(a_values[i], Rational(1, 1 << 30), precision=96)
        var b = Ball(b_values[i], Rational(1, 1 << 30), precision=96)
        var x = Ball(x_values[i], Rational(1, 1 << 30), precision=96)
        var f = ball.betainc(a, b, x)
        assert_false(f.is_indeterminate())
        var a_points: List[Float] = [a.lower(), a.upper()]
        var b_points: List[Float] = [b.lower(), b.upper()]
        var x_points: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for p in a_points:
            for q in b_points:
                for t in x_points:
                    assert_true(_contains(f, Float(_rounded=_betainc_rounded(p, q, t, down))) and _contains(f, Float(_rounded=_betainc_rounded(p, q, t, up))))
    assert_true(ball.betainc(Ball(Rational(0), Rational(1, 8)), Ball(Rational(0), Rational(1, 8)), Ball(Rational(1, 2))).is_indeterminate())
    var exact = ball.betainc(Ball(2), Ball(3), Ball(Rational(1, 4)))
    assert_true(exact.is_exact() and exact.midpoint() == Float(Rational(67, 256)))
    var c = ArithmeticContext(format=FloatFormat(64))
    assert_true(betainc(Float(2), Float(Rational(5, 2)), Float(Rational(1, 3)), context=c) == float.betainc(Float(2), Float(Rational(5, 2)), Float(Rational(1, 3)), context=c))
    var xs = List[Float]()
    for i in range(1, 12):
        xs.append(Float(Rational(i, 12), context=c))
    var mapped = vmap[float.betainc]()(Float(Rational(3, 2), context=c), Float(Rational(5, 2), context=c), Batch[Float](xs))
    for i in range(len(xs)):
        assert_true(mapped[i] == float.betainc(Float(Rational(3, 2), context=c), Float(Rational(5, 2), context=c), xs[i]))


def test_parameter_balls_around_an_ending() raises:
    # A parameter ball around 0 or a negative integer: the series at its
    # midpoint ends there, the ball's series does not (found by the release
    # fuzz, where the fixed-point engine took the ball as the exact value).
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var centres: List[Rational] = [Rational(0), Rational(-3)]
    for centre in centres:
        var a = Ball(centre, Rational(1, 1 << 30), precision=128)
        var b = Ball(Rational(-7, 2))
        var x = Ball(Rational(-10), Rational(20), precision=128)
        var m = ball.hyp1f1(a, b, x)
        var ends: List[Float] = [a.lower(), a.upper()]
        var xs: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for p in ends:
            for s in xs:
                var low = Float(_rounded=_hyp1f1_rounded(p, b.midpoint(), s, down))
                var high = Float(_rounded=_hyp1f1_rounded(p, b.midpoint(), s, up))
                assert_true(_contains(m, low) and _contains(m, high), String("hyp1f1 at ", p, ", ", s, " in ", m))
    var a = Ball(Rational(-13, 2))
    var b = Ball(Rational(0), Rational(1, 1 << 115), precision=113)
    var c = Ball(Rational(99, 25))
    var x = Ball(Rational(-577, 100))
    var f = ball.hyp2f1(a, b, c, x)
    var ends: List[Float] = [b.lower(), b.upper()]
    for q in ends:
        var low = Float(_rounded=_hyp2f1_rounded(a.midpoint(), q, c.midpoint(), x.midpoint(), down))
        var high = Float(_rounded=_hyp2f1_rounded(a.midpoint(), q, c.midpoint(), x.midpoint(), up))
        assert_true(_contains(f, low) and _contains(f, high), String("hyp2f1 at b = ", q, " in ", f))


def test_pfaff_keeps_a_ball_argument_narrow() raises:
    # Pfaff's y = 1 - 1/(1 - x) uses x once: for x in [-4.83, -3.83] it stays
    # within (0.79, 0.83) and the connection applies; x/(x - 1) reached 1.
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var a = Ball(6)
    var b = Ball(Rational(11, 24))
    var c = Ball(Rational(21, 11))
    var x = Ball(Rational(-13, 3), Rational(1, 2), precision=53)
    var f = ball.hyp2f1(a, b, c, x)
    assert_false(f.is_indeterminate())
    var points: List[Float] = [x.lower(), x.midpoint(), x.upper()]
    for p in points:
        var low = Float(_rounded=_hyp2f1_rounded(a.midpoint(), b.midpoint(), c.midpoint(), p, down))
        var high = Float(_rounded=_hyp2f1_rounded(a.midpoint(), b.midpoint(), c.midpoint(), p, up))
        assert_true(_contains(f, low) and _contains(f, high), String("hyp2f1 at ", p, " in ", f))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
