"""Special functions (T-SPEC): correct rounding in every mode against MPFR's
gamma, lngamma, digamma, erf, erfc and eint with their flags, and against
Arb for erfi, the sine, cosine and hyperbolic integrals, Fresnel's integrals
and Lambert's W (tests/fixtures/special_functions.txt); gammaln below 0
against MPFR's lgamma, ndtr and log_ndtr against Arb, and beta, betaln and
poch against exact fractions and Arb (tests/fixtures/special_ratios.txt);
the Riemann zeta function against MPFR, the Hurwitz zeta function,
polygamma, erfinv and ndtri against Arb (tests/fixtures/special_zeta.txt);
special and exact values; ball inclusion, domains and poles; dispatch through
vmap."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, Rational, Float, Ball, FloatFormat, ArithmeticContext, RoundingMode, BallContext, Batch, vmap
from apn_mojo import float
from apn_mojo import ball
from apn_mojo import gamma, lambertw, erf, zeta
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.float._special import _special_rounded, _ratio_rounded, _zeta_rounded, _polygamma_rounded
from apn_mojo.ball._special import (
    _GAMMA, _LOG_GAMMA, _DIGAMMA, _ERF, _ERFC, _ERFI, _EI, _SI, _CI, _SHI, _CHI, _FRESNEL_S, _FRESNEL_C, _LAMBERT_W,
    _NDTR, _LOG_NDTR, _BETA, _LOG_BETA, _POCH, _ERFINV, _NDTRI,
)


def _modes() -> List[RoundingMode]:
    return [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]


def _names() -> List[String]:
    return [
        "gamma", "log_gamma", "digamma", "erf", "erfc", "erfi", "exp_integral_ei", "sin_integral",
        "cos_integral", "sinh_integral", "cosh_integral", "fresnel_s", "fresnel_c", "lambert_w",
    ]


def _code(name: String) -> Int:
    var names = _names()
    for i in range(len(names)):
        if names[i] == name:
            return _GAMMA + i
    return -1


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


def test_correct_rounding() raises:
    var modes = _modes()
    var text: String
    with open("tests/fixtures/special_functions.txt", "r") as source:
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
        var x = _input(fields, 4)
        try:
            var result = _special_rounded(_code(fields[0]), x, context, Int(fields[3]))
            if not _matches(result, fields, 8):
                failures.append(String(line, "  got ", Float(_rounded=result)))
            elif _flags(result.status) != fields[12]:
                failures.append(String(line, "  flags ", _flags(result.status)))
        except e:
            failures.append(String(line, "  raised ", e))
        count += 1
    for i in range(min(len(failures), 30)):
        print(failures[i])
    assert_equal(len(failures), 0, String(len(failures), " mismatches"))
    assert_true(count > 4500)


def _f(x: Float, code: Int, branch: Int = 0) raises -> Float:
    return Float(_rounded=_special_rounded(code, x, ArithmeticContext(format=FloatFormat(53)), branch))


def test_special_and_exact_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var inf = Float.infinity(context=c)
    var zero = Float.zero(context=c)
    var half = Float(Rational(1, 2), context=c)
    var pi_half = float.atan(inf, context=c)
    # Limits where MPFR has no function.
    assert_true(_f(inf, _SI) == pi_half and _f(-inf, _SI) == -pi_half)
    assert_true(_f(inf, _FRESNEL_S) == half and _f(-inf, _FRESNEL_C) == -half)
    assert_true(_f(inf, _CI).is_zero() and _f(inf, _CHI).is_infinite() and _f(-inf, _SHI) == -inf)
    assert_true(_f(zero, _CI) == -inf and _f(zero, _CHI) == -inf and _f(zero, _ERFI).is_zero())
    assert_true(_f(Float(-1), _CI).is_nan() and _f(Float(-1), _CHI).is_nan() and _f(Float(-1), _LOG_GAMMA).is_infinite())
    assert_true(_f(-zero, _SI).is_zero() and _f(-zero, _SI)._negative)
    # Lambert's W: domains, branches and the exact zero.
    assert_true(_f(zero, _LAMBERT_W) == zero and _f(zero, _LAMBERT_W, -1) == -inf)
    assert_true(_f(Float(-1), _LAMBERT_W).is_nan() and _f(Float(1), _LAMBERT_W, -1).is_nan())
    assert_true(_f(Float(Rational(-3, 8)), _LAMBERT_W).is_nan())
    assert_true(_f(Float(Rational(-1, 4)), _LAMBERT_W, -1) < Float(-1) and _f(Float(Rational(-1, 4)), _LAMBERT_W) > Float(-1))
    with assert_raises(contains="branches k=0 and k=-1"):
        _ = float.lambertw(Float(1), k=1)
    # Exact values: Gamma at the positive integers, log Gamma at 1 and 2.
    assert_true(float.gamma(Float(5)) == Float(24) and float.gamma(Float(1)) == Float(1))
    assert_true(float.gamma(Float(30), context=ArithmeticContext(format=FloatFormat(200))) == Float(Integer(8841761993739701954543616000000)))
    assert_true(float.gammaln(Float(2)).is_zero() and not float.gammaln(Float(2))._negative)
    with assert_raises(contains="Cannot compute gamma exactly"):
        _ = float.gamma(Float(Rational(1, 2)), context=ArithmeticContext(format=FloatFormat._exact_format(RoundingMode.nearest_even)))
    # Results near a Float: erf of a large argument, Si of a tiny one, W of a tiny one.
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    assert_true(float.erf(Float(30), context=down) < Float(1) and float.erf(Float(30)) == Float(1))
    var tiny = Float(_rounded=_RoundedBinary(1, False, Integer(1), -10000, FloatFormat(1), NumericStatus()))
    assert_true(float.sici(tiny, context=down)[0] < tiny and float.sici(tiny)[0] == tiny)
    assert_true(float.lambertw(tiny, context=down) < tiny)
    # tiny = 2**-10001, so Gamma(tiny) = 2**10001 - 0.577..., which rounds to 2**10001 (exponent 10002).
    assert_true(float.gamma(tiny, context=ArithmeticContext(format=FloatFormat(53))) == Float(_rounded=_RoundedBinary(1, False, Integer(1), 10002, FloatFormat(1), NumericStatus())))
    # Exact rationals are evaluated exactly, never rounded first.
    var third = float.erf(Rational(1, 3), context=c)
    assert_true(third == float.erf(Float(Rational(1, 3), context=ArithmeticContext(format=FloatFormat(300))), context=c))


def _contains(x: Ball, value: Float) raises -> Bool:
    return x.lower_rational() <= value.to_rational_exact() and value.to_rational_exact() <= x.upper_rational()


def test_ball_inclusion() raises:
    # The ball of each function contains its correctly rounded values, both
    # directions, at the ends and the midpoint of the input ball.
    var names = _names()
    var low: List[Rational] = [Rational(1, 10), Rational(1, 10), Rational(1, 10), Rational(-3), Rational(-3), Rational(-3), Rational(1, 10), Rational(-30), Rational(1, 10), Rational(-5), Rational(1, 10), Rational(-6), Rational(-6), Rational(-1, 4)]
    var high: List[Rational] = [Rational(8), Rational(30), Rational(30), Rational(3), Rational(6), Rational(4), Rational(30), Rational(30), Rational(30), Rational(5), Rational(20), Rational(6), Rational(6), Rational(50)]
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var state = UInt64(0x2545F4914F6CDD1D)
    for i in range(len(names)):
        var code = _GAMMA + i
        for _ in range(6):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var t = Rational(Int(state % 1000), 1000)
            var center = low[i] + (high[i] - low[i]) * t
            var radius = Rational(1, Integer(2) ** Int(10 + (state >> 32) % 30))
            var b = Ball(center, radius, precision=96)
            var y = ball.gamma(b)
            if code == _LOG_GAMMA:
                y = ball.gammaln(b)
            elif code == _DIGAMMA:
                y = ball.digamma(b)
            elif code == _ERF:
                y = ball.erf(b)
            elif code == _ERFC:
                y = ball.erfc(b)
            elif code == _ERFI:
                y = ball.erfi(b)
            elif code == _EI:
                y = ball.expi(b)
            elif code == _SI:
                y = ball.sici(b)[0]
            elif code == _CI:
                y = ball.sici(b)[1]
            elif code == _SHI:
                y = ball.shichi(b)[0]
            elif code == _CHI:
                y = ball.shichi(b)[1]
            elif code == _FRESNEL_S:
                y = ball.fresnel(b)[0]
            elif code == _FRESNEL_C:
                y = ball.fresnel(b)[1]
            elif code == _LAMBERT_W:
                y = ball.lambertw(b)
            if y.is_indeterminate():
                continue
            var points: List[Float] = [b.lower(), b.upper(), b.midpoint()]
            for point in points:
                if code == _GAMMA and point._exponent <= 0 and point < Float(0):
                    continue
                var a = Float(_rounded=_special_rounded(code, point, down))
                var z = Float(_rounded=_special_rounded(code, point, up))
                assert_true(_contains(y, a) and _contains(y, z), String(names[i], " at ", point, " in ", y))


def test_domains_and_poles() raises:
    assert_true(ball.gamma(Ball(-2, Rational(1, 10))).is_indeterminate())
    assert_true(ball.gamma(Ball(0, Rational(1, 10))).is_indeterminate())
    assert_false(ball.gamma(Ball(Rational(-5, 2), Rational(1, 10))).is_indeterminate())
    assert_true(ball.digamma(Ball(-1, Rational(1, 100))).is_indeterminate())
    assert_true(ball.gammaln(Ball(0, Rational(1, 10))).is_indeterminate())
    assert_true(ball.expi(Ball(0, Rational(1, 10))).is_indeterminate())
    assert_true(ball.sici(Ball(Rational(1, 20), Rational(1, 10)))[1].is_indeterminate())
    assert_true(ball.lambertw(Ball(Rational(-1, 2))).is_indeterminate())
    assert_true(ball.lambertw(Ball(Rational(-1, 10)), k=-1).lower_rational() < -3)
    assert_true(ball.erf(Ball.unbounded()).upper_rational() <= 1)
    # Gamma on [1.45, 1.47] reaches its minimum 0.8856031944... and Gamma(1.45) = 0.885661...
    var around = ball.gamma(Ball(Rational(146, 100), Rational(1, 100)))
    assert_true(around.lower_rational() <= Rational(8856031944, 10**10) and around.upper_rational() >= Rational(885661, 10**6))


def test_dispatch_and_vmap() raises:
    var x = Float(Rational(3, 2), context=ArithmeticContext(format=FloatFormat(64)))
    assert_true(gamma(x) == float.gamma(x))
    assert_true(gamma(Ball(Rational(3, 2))).same_representation(ball.gamma(Ball(Rational(3, 2)))))
    assert_true(lambertw(Float(1), k=0) == float.lambertw(Float(1)))
    assert_true(erf(Ball(1)).same_representation(ball.erf(Ball(1))))
    var values = List[Float]()
    for i in range(1, 40):
        values.append(Float(Rational(i, 7), context=ArithmeticContext(format=FloatFormat(64))))
    var mapped = vmap[float.erf]()(Batch[Float](values))
    for i in range(len(values)):
        assert_true(mapped[i] == float.erf(values[i]))


def _arity(name: String) -> Int:
    return 2 if name == "beta" or name == "betaln" or name == "poch" else 1


def _ratio_code(name: String) -> Int:
    if name == "log_gamma":
        return _LOG_GAMMA
    if name == "ndtr":
        return _NDTR
    if name == "log_ndtr":
        return _LOG_NDTR
    if name == "beta":
        return _BETA
    if name == "betaln":
        return _LOG_BETA
    return _POCH


def test_ratios_and_normal_correct_rounding() raises:
    var modes = _modes()
    var text: String
    with open("tests/fixtures/special_ratios.txt", "r") as source:
        text = source.read()
    var failures = List[String]()
    var count = 0
    for line in text.splitlines():
        if not line.byte_length():
            continue
        var fields = List[String]()
        for w in line.split(" "):
            fields.append(String(w))
        var arity = _arity(fields[0])
        var code = _ratio_code(fields[0])
        var context = ArithmeticContext(format=FloatFormat(Int(fields[1])), rounding=modes[Int(fields[2])])
        var x = _input(fields, 4)
        var at = 4 + 4 * arity
        try:
            var result: _RoundedBinary
            if arity == 1:
                result = _special_rounded(code, x, context)
            else:
                result = _ratio_rounded(code, x, _input(fields, 8), context)
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
    assert_true(count > 3000)


def test_ratio_and_normal_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var inf = Float.infinity(context=c)
    # gammaln is log |Gamma|: +inf at the poles and both infinities.
    assert_true(float.gammaln(Float(-1)) == inf and float.gammaln(-inf) == inf)
    assert_true(float.gammaln(Float(Rational(-1, 2)), context=c) > Float(Rational(5, 4)))
    # ndtr: exactly 1/2 at 0, the limits, and within 2**-1154 of 1 at 40.
    assert_true(float.ndtr(Float.zero(), context=down) == Float(Rational(1, 2)))
    assert_true(float.ndtr(inf) == Float(1) and float.ndtr(-inf).is_zero() and not float.ndtr(-inf)._negative)
    assert_true(float.ndtr(Float(40), context=down) < Float(1) and float.ndtr(Float(40), context=c) == Float(1))
    assert_true(float.ndtr(Float(-40), context=c) > Float(0))
    assert_true(float.log_ndtr(inf).is_zero() and float.log_ndtr(-inf) == -inf)
    assert_true(float.log_ndtr(Float.zero(), context=c) == float.log(Float(Rational(1, 2)), context=c))
    assert_true(float.log_ndtr(Float(40), context=c) < Float(0))
    # beta: rational values exactly, scipy's values at the poles, the limit at infinity.
    assert_true(float.beta(Float(1), Float(Rational(1, 2)), context=down) == Float(2))
    assert_true(float.beta(Float(2), Float(3), context=c) == Float(Rational(1, 12), context=c))
    assert_true(float.beta(Integer(2), Integer(3), context=c) == Float(Rational(1, 12), context=c))
    assert_true(float.beta(Float(-3), Float(2), context=c) == Float(Rational(1, 6), context=c))
    assert_true(float.beta(Float(-3), Float(1), context=c) == Float(Rational(-1, 3), context=c))
    assert_true(float.beta(Float(-2), Float(3)) == inf)
    assert_true(float.beta(Float(-2), Float(Rational(1, 2))) == inf)
    assert_true(float.beta(Float(Rational(1, 2)), Float(Rational(-1, 2))).is_zero())
    assert_true(float.beta(inf, Float(2)).is_zero() and float.beta(-inf, Float(2)).is_nan())
    assert_true(float.betaln(Float(1), Float(1)).is_zero() and float.betaln(Float(-1), Float(1)).is_zero())
    assert_true(float.betaln(Float(-2), Float(3)) == inf and float.betaln(Float(Rational(1, 2)), Float(Rational(-1, 2))) == -inf)
    assert_true(float.betaln(Float(2), Float(3), context=c) == float.log(Float(Rational(1, 12), context=ArithmeticContext(format=FloatFormat(200))), context=c))
    # poch: integer m exactly, scipy's poles and zeros.
    assert_true(float.poch(Float(Rational(1, 2)), Float(3), context=down) == Float(Rational(15, 8)))
    assert_true(float.poch(Float(4), Float(-2), context=c) == Float(Rational(1, 6), context=c))
    assert_true(float.poch(Float(7), Float(0)) == Float(1) and float.poch(Float(1), Float(-1)) == inf)
    assert_true(float.poch(Float(-3), Float(5)).is_zero() and float.poch(Float(-2), Float(Rational(1, 2))).is_zero())
    assert_true(float.poch(Float(Rational(-5, 2)), Float(Rational(1, 2))) == inf)
    assert_true(float.poch(inf, Float(2)) == inf and float.poch(inf, Float(-2)).is_zero() and float.poch(inf, Float(0)) == Float(1))
    with assert_raises():
        _ = float.beta(Integer(2), Integer(3))


def test_ratio_and_normal_balls() raises:
    # Each ball contains the correctly rounded values, both directions, at
    # the ends and midpoints of its input balls.
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var firsts: List[Rational] = [Rational(3, 2), Rational(-7, 3), Rational(11, 4), Rational(40), Rational(1, 7)]
    var seconds: List[Rational] = [Rational(5, 2), Rational(4, 5), Rational(-13, 10), Rational(1, 3), Rational(-23, 10)]
    var codes: List[Int] = [_BETA, _LOG_BETA, _POCH]
    for i in range(len(firsts)):
        var a = Ball(firsts[i], Rational(1, 1 << 20), precision=96)
        var b = Ball(seconds[i], Rational(1, 1 << 20), precision=96)
        for code in codes:
            var y: Ball
            if code == _BETA:
                y = ball.beta(a, b)
            elif code == _LOG_BETA:
                y = ball.betaln(a, b)
            else:
                y = ball.poch(a, b)
            assert_false(y.is_indeterminate(), String(code, " at ", a, ", ", b))
            var xs: List[Float] = [a.lower(), a.midpoint(), a.upper()]
            var ys: List[Float] = [b.lower(), b.midpoint(), b.upper()]
            for p in xs:
                for q in ys:
                    var lo = Float(_rounded=_ratio_rounded(code, p, q, down))
                    var hi = Float(_rounded=_ratio_rounded(code, p, q, up))
                    assert_true(_contains(y, lo) and _contains(y, hi), String(code, " at ", p, ", ", q, " in ", y))
    var centers: List[Rational] = [Rational(-5, 2), Rational(-31, 10), Rational(-1, 3), Rational(-6), Rational(1, 2), Rational(3), Rational(9)]
    for center in centers:
        var x = Ball(center, Rational(1, 1 << 24), precision=96)
        var points: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        var g = ball.gammaln(x)
        var n = ball.ndtr(x)
        var l = ball.log_ndtr(x)
        for point in points:
            if center.sign() < 0 and not center.is_integer():
                assert_true(_contains(g, Float(_rounded=_special_rounded(_LOG_GAMMA, point, down))) and _contains(g, Float(_rounded=_special_rounded(_LOG_GAMMA, point, up))))
            assert_true(_contains(n, Float(_rounded=_special_rounded(_NDTR, point, down))) and _contains(n, Float(_rounded=_special_rounded(_NDTR, point, up))))
            assert_true(_contains(l, Float(_rounded=_special_rounded(_LOG_NDTR, point, down))) and _contains(l, Float(_rounded=_special_rounded(_LOG_NDTR, point, up))))
    # Exact integer arguments give exact or tight balls; poles are indeterminate.
    var twelfth = ball.beta(Ball(2), Ball(3))
    assert_true(twelfth.lower_rational() <= Rational(1, 12) and Rational(1, 12) <= twelfth.upper_rational())
    assert_true(ball.beta(Ball(-2), Ball(3)).is_indeterminate() and ball.betaln(Ball(-2), Ball(3)).is_indeterminate())
    assert_true(ball.poch(Ball(-3), Ball(5)).is_exact() and ball.poch(Ball(-3), Ball(5)).midpoint().is_zero())
    assert_true(ball.gammaln(Ball(-2, Rational(1, 10))).is_indeterminate())
    var vs = List[Float]()
    var ws = List[Float]()
    for i in range(1, 12):
        vs.append(Float(Rational(i, 3), context=ArithmeticContext(format=FloatFormat(64))))
        ws.append(Float(Rational(2 * i + 1, 5), context=ArithmeticContext(format=FloatFormat(64))))
    var betas = vmap[float.beta]()(Batch[Float](vs), Batch[Float](ws))
    var normals = vmap[float.ndtr]()(Batch[Float](vs))
    for i in range(len(vs)):
        assert_true(betas[i] == float.beta(vs[i], ws[i]) and normals[i] == float.ndtr(vs[i]))


def test_zeta_and_inverses_correct_rounding() raises:
    var modes = _modes()
    var text: String
    with open("tests/fixtures/special_zeta.txt", "r") as source:
        text = source.read()
    var failures = List[String]()
    var count = 0
    for line in text.splitlines():
        if not line.byte_length():
            continue
        var fields = List[String]()
        for w in line.split(" "):
            fields.append(String(w))
        var name = fields[0]
        var arity = 2 if name == "zeta" or name == "polygamma" else 1
        var context = ArithmeticContext(format=FloatFormat(Int(fields[1])), rounding=modes[Int(fields[2])])
        var x = _input(fields, 4)
        var at = 4 + 4 * arity
        try:
            var result: _RoundedBinary
            if name == "zeta":
                result = _zeta_rounded(x, _input(fields, 8), context)
            elif name == "polygamma":
                result = _polygamma_rounded(x.round(), _input(fields, 8), context)
            else:
                result = _special_rounded(_ERFINV if name == "erfinv" else _NDTRI, x, context)
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
    assert_true(count > 2500)


def test_zeta_and_inverses_values() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var down = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_zero)
    var up = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_positive)
    var inf = Float.infinity(context=c)
    var one = Float(1)
    # Riemann's values at the non-positive integers are exact; zeta(x) is zeta(x, 1).
    assert_true(float.zeta(Float(-1), one, context=down) == Float(Rational(-1, 12), context=down))
    assert_true(float.zeta(Float(0), one, context=down) == Float(Rational(-1, 2)))
    assert_true(float.zeta(Float(-2), one).is_zero())
    assert_true(zeta(Float(3), context=c) == float.zeta(Float(3), one, context=c))
    assert_true(float.zeta(Float(200), one, context=down) == one and float.zeta(Float(200), one, context=up) > one)
    # Poles and scipy's domain.
    assert_true(float.zeta(one, Float(2)) == inf and float.zeta(Float(3), Float(0)) == inf)
    assert_true(float.zeta(Float(Rational(1, 2)), Float(2)).is_nan() and float.zeta(Float(Rational(5, 2)), Float(Rational(-1, 2))).is_nan())
    assert_true(float.zeta(Float(3), Float(Rational(-1, 2)), context=c).is_finite())
    assert_true(float.zeta(inf, Float(2)).is_zero() and float.zeta(inf, one) == one)
    # polygamma: digamma at n = 0, psi'(1) = zeta(2), scipy's signed poles.
    assert_true(float.polygamma(Integer(0), Float(3), context=c) == float.digamma(Float(3), context=c))
    assert_true(float.polygamma(Integer(1), one, context=c) == float.zeta(Float(2), one, context=c))
    assert_true(float.polygamma(Integer(2), Float(0)) == -inf and float.polygamma(Integer(1), Float(-1)) == inf)
    assert_true(float.polygamma(Integer(-1), one).is_nan())
    # erfinv and ndtri: exact points, poles, and round trips.
    assert_true(float.erfinv(Float.zero()).is_zero() and float.erfinv(one) == inf and float.erfinv(-one) == -inf)
    assert_true(float.erfinv(Float(2)).is_nan() and float.erfinv(inf).is_nan())
    var wide = ArithmeticContext(format=FloatFormat(200))
    assert_true(float.erf(float.erfinv(Float(Rational(1, 3), context=c), context=wide), context=c) == Float(Rational(1, 3), context=c))
    assert_true(float.ndtri(Float(Rational(1, 2))).is_zero() and float.ndtri(Float.zero()) == -inf and float.ndtri(one) == inf)
    assert_true(float.ndtri(Float(2)).is_nan())
    var tiny = Float(_rounded=_RoundedBinary(1, False, Integer(1), -1000, FloatFormat(1), NumericStatus()))
    var q = float.ndtri(tiny, context=wide)
    assert_true(q < Float(-30) and float.ndtr(q, context=c) == tiny)


def test_zeta_and_inverses_balls() raises:
    var down = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(160), rounding=RoundingMode.toward_positive)
    var inverse_centers: List[Rational] = [Rational(-9, 10), Rational(-1, 7), Rational(1, 3), Rational(7, 8)]
    for center in inverse_centers:
        var x = Ball(center, Rational(1, 1 << 24), precision=96)
        var e = ball.erfinv(x)
        var probability = Ball((center + Rational(1)) / Rational(2), Rational(1, 1 << 24), precision=96)
        var n = ball.ndtri(probability)
        var points: List[Float] = [x.lower(), x.midpoint(), x.upper()]
        for point in points:
            assert_true(_contains(e, Float(_rounded=_special_rounded(_ERFINV, point, down))) and _contains(e, Float(_rounded=_special_rounded(_ERFINV, point, up))))
        var ps: List[Float] = [probability.lower(), probability.midpoint(), probability.upper()]
        for point in ps:
            assert_true(_contains(n, Float(_rounded=_special_rounded(_NDTRI, point, down))) and _contains(n, Float(_rounded=_special_rounded(_NDTRI, point, up))))
    var exponents: List[Rational] = [Rational(3, 2), Rational(5), Rational(-7, 2), Rational(1, 2)]
    var shifts: List[Rational] = [Rational(1, 3), Rational(2), Rational(1), Rational(1)]
    for i in range(len(exponents)):
        var s = Ball(exponents[i], Rational(1, 1 << 24), precision=96)
        var q = Ball(shifts[i]) if shifts[i] == Rational(1) else Ball(shifts[i], Rational(1, 1 << 24), precision=96)
        var z = ball.zeta(s, q)
        assert_false(z.is_indeterminate(), String("zeta at ", s, ", ", q))
        var ss: List[Float] = [s.lower(), s.midpoint(), s.upper()]
        var qs: List[Float] = [q.lower(), q.midpoint(), q.upper()]
        for a in ss:
            for b in qs:
                assert_true(_contains(z, Float(_rounded=_zeta_rounded(a, b, down))) and _contains(z, Float(_rounded=_zeta_rounded(a, b, up))), String("zeta at ", a, ", ", b, " in ", z))
    var g = ball.polygamma(Integer(2), Ball(Rational(-5, 2), Rational(1, 1 << 24), precision=96))
    assert_true(_contains(g, Float(_rounded=_polygamma_rounded(Integer(2), Float(Rational(-5, 2)), down))))
    assert_true(ball.zeta(Ball(1, Rational(1, 10)), Ball(1)).is_indeterminate() and ball.erfinv(Ball(1)).is_indeterminate())
    assert_true(ball.ndtri(Ball(Rational(1, 2))).is_exact() and ball.ndtri(Ball(Rational(1, 2))).midpoint().is_zero())
    var xs = List[Float]()
    for i in range(1, 12):
        xs.append(Float(Rational(i, 13), context=ArithmeticContext(format=FloatFormat(64))))
    var inverses = vmap[float.erfinv]()(Batch[Float](xs))
    var exponents_above = List[Float]()
    for i in range(1, 12):
        exponents_above.append(Float(Rational(13 + i, 13), context=ArithmeticContext(format=FloatFormat(64))))
    var zetas = vmap[float.zeta]()(Batch[Float](exponents_above), Integer(2))
    for i in range(len(xs)):
        assert_true(inverses[i] == float.erfinv(xs[i]) and zetas[i] == float.zeta(exponents_above[i], Integer(2)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
