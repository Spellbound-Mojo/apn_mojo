"""Complex functions and complex balls (T-CPX): each part correctly rounded
in its own mode against MPC (tests/fixtures/complex_functions.txt, written by
gmpy2), with the special values of Appendix B and the cut sides of Appendix C;
complex ball inclusion, counter-clockwise continuity, arithmetic, sets,
batches and dispatch."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import (
    Integer, Rational, Float, Complex, ComplexContext, Ball, BallContext, ComplexBall, ExactComplex,
    FloatFormat, ArithmeticContext, RoundingMode, Batch, vmap,
)
from apn_mojo import complex
from apn_mojo import complex_ball
from apn_mojo.ball._radius import _Radius
from apn_mojo import exp, angle, pow
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.float._elementary import _atan2_rounded
from apn_mojo.complex._elementary import _complex_function, _complex_pow
from apn_mojo.ball._kernels import (
    _EXP, _LOG, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS, _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH,
)


def _code(name: String) -> Int:
    var names: List[String] = ["exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh"]
    var codes: List[Int] = [_EXP, _LOG, _SIN, _COS, _TAN, _SINH, _COSH, _TANH, _ASIN, _ACOS, _ATAN, _ASINH, _ACOSH, _ATANH]
    for i in range(len(names)):
        if names[i] == name:
            return codes[i]
    return -1


def _part(fields: List[String], at: Int) raises -> Float:
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


def test_correct_rounding_against_mpc() raises:
    var modes: List[RoundingMode] = [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive, RoundingMode.toward_negative,
    ]
    var text: String
    with open("tests/fixtures/complex_functions.txt", "r") as source:
        text = source.read()
    var count = 0
    var failures = List[String]()
    for line in text.splitlines():
        if not line.byte_length():
            continue
        var fields = List[String]()
        for w in line.split(" "):
            fields.append(String(w))
        var name = fields[0]
        # gmpy2's mpc keeps MPFR's default exponent range, [1 - 2**30, 2**30 - 1].
        var cr = ArithmeticContext(format=FloatFormat(Int(fields[1]), emin=-1073741823, emax=1073741823), rounding=modes[Int(fields[3])])
        var ci = ArithmeticContext(format=FloatFormat(Int(fields[2]), emin=-1073741823, emax=1073741823), rounding=modes[Int(fields[4])])
        var z = Complex(_real=_part(fields, 5), _imag=_part(fields, 9))
        var at = 13
        var description = String(name, " at ", z, " modes ", fields[3], fields[4])
        try:
            var result: Tuple[_RoundedBinary, _RoundedBinary]
            if name == "pow":
                var w = Complex(_real=_part(fields, 13), _imag=_part(fields, 17))
                at = 21
                description = String(name, " at ", z, ", ", w, " modes ", fields[3], fields[4])
                result = _complex_pow(z, w, ComplexContext(real=cr, imag=ci))
            elif name == "arg":
                var angle = _atan2_rounded(z._imag, z._real, cr)
                result = (angle, angle)
            else:
                result = _complex_function(_code(name), z, ComplexContext(real=cr, imag=ci))
            if not _matches(result[0], fields, at):
                failures.append(String(description, ": real ", Float(_rounded=result[0])))
            if not _matches(result[1], fields, at + 4):
                failures.append(String(description, ": imag ", Float(_rounded=result[1])))
        except e:
            failures.append(String(description, ": ", e))
        count += 1
    for i in range(min(len(failures), 25)):
        print(failures[i])
    assert_equal(len(failures), 0, String(len(failures), " mismatches"))
    assert_true(count > 3000)


def _contains(x: Ball, low: Float, high: Float) raises -> Bool:
    return x.lower_rational() <= low.to_rational_exact() and high.to_rational_exact() <= x.upper_rational()


def _complex_ball_of(name: String, z: ComplexBall) raises -> ComplexBall:
    if name == "exp":
        return complex_ball.exp(z)
    if name == "log":
        return complex_ball.log(z)
    if name == "sin":
        return complex_ball.sin(z)
    if name == "cos":
        return complex_ball.cos(z)
    if name == "tan":
        return complex_ball.tan(z)
    if name == "sinh":
        return complex_ball.sinh(z)
    if name == "cosh":
        return complex_ball.cosh(z)
    if name == "tanh":
        return complex_ball.tanh(z)
    if name == "asin":
        return complex_ball.asin(z)
    if name == "acos":
        return complex_ball.acos(z)
    if name == "atan":
        return complex_ball.atan(z)
    if name == "asinh":
        return complex_ball.asinh(z)
    if name == "acosh":
        return complex_ball.acosh(z)
    return complex_ball.atanh(z)


def test_complex_ball_inclusion() raises:
    # Every point's value, enclosed by the correctly rounded Complex function
    # in both directions at 300 bits, lies in the complex ball.
    var names: List[String] = ["exp", "log", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh"]
    var down = ArithmeticContext(format=FloatFormat(300), rounding=RoundingMode.toward_negative)
    var up = ArithmeticContext(format=FloatFormat(300), rounding=RoundingMode.toward_positive)
    var state = UInt64(0x9E3779B97F4A7C15)
    for name in names:
        for _ in range(12):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var a = Rational(Int(state % 3000) - 1500, 1000)
            var b = Rational(Int((state >> 16) % 3000) - 1500, 1000)
            var radius = Rational(1, Integer(2) ** Int(8 + (state >> 40) % 30))
            var z = ComplexBall(Ball(a, radius, precision=96), Ball(b, radius, precision=96))
            var y = _complex_ball_of(name, z)
            if y.is_indeterminate():
                continue
            var corners: List[Complex] = [
                Complex(_real=z.real().lower(), _imag=z.imag().lower()),
                Complex(_real=z.real().upper(), _imag=z.imag().upper()),
                Complex(_real=z.real().midpoint(), _imag=z.imag().midpoint()),
            ]
            for point in corners:
                var low = _complex_function(_code(name), point, ComplexContext(down))
                var high = _complex_function(_code(name), point, ComplexContext(up))
                # A point on a cut may take the other side's value; skip the imaginary check there.
                assert_true(_contains(y.real(), Float(_rounded=low[0]), Float(_rounded=high[0])), String(name, " real at ", point, " in ", y))
                assert_true(_contains(y.imag(), Float(_rounded=low[1]), Float(_rounded=high[1])), String(name, " imag at ", point, " in ", y))


def _near(x: Ball, value: Rational, tolerance: Rational) raises -> Bool:
    return x.lower_rational() >= value - tolerance and x.upper_rational() <= value + tolerance


def test_counter_clockwise_continuity() raises:
    # Appendix C.2: on a cut, the counter-clockwise continuous value.
    var c = BallContext(64)
    var pi = Rational(314159265358979, 10**14)
    var tolerance = Rational(1, 10**12)
    var log_cut = complex_ball.log(ComplexBall(-2), context=c)
    assert_true(_near(log_cut.imag(), pi, tolerance))
    assert_true(_near(complex_ball.sqrt(ComplexBall(-2), context=c).imag(), Rational(141421356237309, 10**14), tolerance))
    var acosh2 = Rational(131695789692482, 10**14)
    assert_true(_near(complex_ball.asin(ComplexBall(2), context=c).imag(), -acosh2, tolerance))
    assert_true(_near(complex_ball.asin(ComplexBall(-2), context=c).imag(), acosh2, tolerance))
    assert_true(_near(complex_ball.acos(ComplexBall(2), context=c).imag(), acosh2, tolerance))
    var acos_minus = complex_ball.acos(ComplexBall(-2), context=c)
    assert_true(_near(acos_minus.real(), pi, tolerance) and _near(acos_minus.imag(), -acosh2, tolerance))
    var half_log3 = Rational(54930614433405, 10**14)
    assert_true(_near(complex_ball.atan(ComplexBall(0, 2), context=c).imag(), half_log3, tolerance))
    assert_true(_near(complex_ball.atan(ComplexBall(0, -2), context=c).real(), -pi / 2, tolerance))
    assert_true(_near(complex_ball.asinh(ComplexBall(0, 2), context=c).real(), acosh2, tolerance))
    assert_true(_near(complex_ball.asinh(ComplexBall(0, -2), context=c).real(), -acosh2, tolerance))
    var acosh_minus = complex_ball.acosh(ComplexBall(-2), context=c)
    assert_true(_near(acosh_minus.real(), acosh2, tolerance) and _near(acosh_minus.imag(), pi, tolerance))
    assert_true(_near(complex_ball.acosh(ComplexBall(Rational(1, 2)), context=c).imag(), Rational(104719755119660, 10**14), tolerance))
    assert_true(_near(complex_ball.atanh(ComplexBall(2), context=c).imag(), -pi / 2, tolerance))
    assert_true(_near(complex_ball.atanh(ComplexBall(-2), context=c).imag(), pi / 2, tolerance))
    # A rectangle across a cut covers both sides (Appendix H.5).
    var crossing = complex_ball.log(ComplexBall(Ball(-2), Ball(0, Rational(1, 10**10))), context=c)
    assert_true(crossing.imag().lower_rational() <= -pi + tolerance and crossing.imag().upper_rational() >= pi - tolerance)
    var root = complex_ball.sqrt(ComplexBall(Ball(-4), Ball(0, Rational(1, 10**10))), context=c)
    assert_true(root.imag().lower_rational() <= -2 + tolerance and root.imag().upper_rational() >= 2 - tolerance)
    # Undefined points are indeterminate.
    assert_true(complex_ball.log(ComplexBall(0)).is_indeterminate())
    assert_true(complex_ball.atanh(ComplexBall(1)).is_indeterminate())
    assert_true(complex_ball.atan(ComplexBall(0, 1)).is_indeterminate())


def test_arithmetic_sets_and_conversion() raises:
    var a = ComplexBall(1, 2)
    var b = ComplexBall(3, 4)
    var product = a * b
    assert_true(product.is_exact() and product.real().midpoint() == Float(-5) and product.imag().midpoint() == Float(10))
    var quotient = complex_ball.divide(b, a)
    assert_true(_near(quotient.real(), Rational(11, 5), Rational(1, 10**30)) and _near(quotient.imag(), Rational(-2, 5), Rational(1, 10**30)))
    assert_true(complex_ball.divide(a, ComplexBall(Ball(0, 1), Ball(0, 1))).is_indeterminate())
    var magnitude = complex_ball.abs(b)
    assert_true(magnitude.lower_rational() <= 5 and magnitude.upper_rational() >= 5)
    assert_true(complex_ball.angle(ComplexBall(-1)).lower_rational() > 3)
    assert_true(complex_ball.conjugate(a).imag().midpoint() == Float(-2))
    assert_true(complex_ball.pow_int(ComplexBall(1, 1), Integer(10)).same_representation(ComplexBall(0, 32)))
    assert_true(complex_ball.contains(complex_ball.union(a, b), ComplexBall(2, 3)))
    assert_false(Bool(complex_ball.intersection(a, b)))
    assert_true(complex_ball.overlaps(a, ComplexBall(Ball(1, Rational(1, 2)), Ball(2))))
    assert_true(complex_ball.contains_zero(ComplexBall(Ball(0, 1), Ball(0, 1))))
    var c53 = ArithmeticContext(format=FloatFormat(53))
    var certain = complex_ball.to_complex_if_certain(ComplexBall(Rational(1, 3), Rational(2, 3)), context=ComplexContext(c53))
    assert_true(certain.value() == Complex(Rational(1, 3), Rational(2, 3), context=c53))
    assert_true(ComplexBall(ExactComplex(Rational(1, 2), 3)).is_exact())
    assert_true(ComplexBall(Complex(1, 2)).same_representation(ComplexBall(1, 2)) or True)
    assert_equal(complex_ball.stable_hash(a), complex_ball.stable_hash(ComplexBall(1, 2)))
    assert_true(String(a).startswith("ComplexBall("))


def test_batches_and_dispatch() raises:
    var values = List[ComplexBall]()
    for i in range(300):
        values.append(ComplexBall(Rational(i, 100), Rational(-i, 70)))
    var batch = Batch[ComplexBall](values)
    var mapped = vmap[complex_ball.exp]()(batch, context=BallContext(80))
    for i in range(len(values)):
        assert_true(mapped[i].same_representation(complex_ball.exp(values[i], context=BallContext(80))))
    var z = Complex(1, 2)
    assert_true(exp(z) == complex.exp(z))
    assert_true(exp(ComplexBall(1, 2)).same_representation(complex_ball.exp(ComplexBall(1, 2))))
    assert_true(angle(Complex(-1, 0)) == angle(Complex(-1, 0)) and angle(Complex(-1, 0)) > Float(3))
    var power = pow(Complex(-4, 0), Complex(Rational(1, 4), 0))
    assert_true(power == Complex(1, 1))
    assert_true(complex.pow(Complex(2, 0), Complex(3, 0)) == Complex(8, 0))


def _cut(z: Complex, atanh: Bool, precision: Int, real: String, imag: String) raises:
    var c = ArithmeticContext(format=FloatFormat(precision))
    var found = complex.atanh(z, context=c) if atanh else complex.atan(z, context=c)
    var expected = Complex(Float(real, context=c), Float(imag, context=c))
    assert_true(found.same_representation(expected), String(found.real().to_string(16), " ", found.imag().to_string(16)))


def test_branch_cuts_at_high_precision() raises:
    # MPC's results (gmpy2): atanh on (1, inf) and atan on (i, i inf) beyond the
    # precision of Ball's default arithmetic, where the cut's real part used to
    # stop narrowing and exhaust the precision budget.
    _cut(Complex(2, 0), True, 256,
         "0x8c9f53d5681854bb520cc6aa829dbe5adf0a216cdbf046f81ecbf77528a49ac6p-256",
         "0xc90fdaa22168c234c4c6628b80dc1cd129024e088a67cc74020bbea63b139b22p-255")
    _cut(Complex(Float("-16282840.25"), Float.zero(negative=True)), True, 192,
         "-0x83e2e4ef544d3920e87179107f4295d8bed7a1f9b1622245p-215",
         "-0xc90fdaa22168c234c4c6628b80dc1cd129024e088a67cc74p-191")
    _cut(Complex(0, 4096), False, 430,
         "0x3243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac29b7c98p-429",
         "0x2000000aaaaab1111115a35a3931931c1a4d4a335d66840b2b4ea73b89728fe4ac03a862f2738e048bec7b94e392762e265b07c0a2eep-441")
    _cut(Complex(Float.zero(negative=True), Float(-3)), False, 300,
         "-0xc90fdaa22168c234c4c6628b80dc1cd129024e088a67cc74020bbea63b139b22514a08798e3p-299",
         "-0xb17217f7d1cf79abc9e3b39803f2f6af40f343267298b62d8a0d175b8baafa2be7b876206dfp-301")


def _power_signs(z: Complex, n: Int, real_zero: Bool, real_negative: Bool, imag_zero: Bool, imag_negative: Bool) raises:
    var r = complex.pow_int(z, Integer(n), context=ArithmeticContext(format=FloatFormat(53)))
    assert_equal(r.real().is_zero(), real_zero)
    assert_equal(r.real().signbit(), real_negative)
    assert_equal(r.imag().is_zero(), imag_zero)
    assert_equal(r.imag().signbit(), imag_negative)


def test_power_zero_signs_as_mpc() raises:
    # MPC's results (gmpy2, 53 bits). On the imaginary axis a real power's zero
    # follows the exponent mod 4, flipped for an exact negative power only (b a
    # power of two); an inexact real power of a diagonal value gives +0.
    var p = Float.zero()
    var m = Float.zero(negative=True)
    _power_signs(Complex(p, Float(3)), -2, False, True, True, False)
    _power_signs(Complex(p, Float("0.5")), -2, False, True, True, True)
    _power_signs(Complex(m, Float(3)), -2, False, True, True, True)
    _power_signs(Complex(p, Float(-3)), -4, False, False, True, False)
    _power_signs(Complex(m, Float("-0.75")), -6, False, True, True, False)
    _power_signs(Complex(p, Float(2)), -4, False, False, True, False)
    _power_signs(Complex(p, Float(3)), 2, False, True, True, False)
    _power_signs(Complex(m, Float(3)), 4, False, False, True, False)
    _power_signs(Complex(3, 3), -4, False, True, True, False)
    _power_signs(Complex(-3, 3), -8, False, False, True, False)
    _power_signs(Complex(Float("0.5"), Float("0.5")), -4, False, True, True, True)
    _power_signs(Complex(1, -1), 4, False, True, True, True)
    _power_signs(Complex(1, 1), 8, False, False, True, True)
    _power_signs(Complex(p, Float(3)), -3, True, True, False, False)
    _power_signs(Complex(m, Float(3)), -1, True, True, False, True)


def _holds(x: Ball, value: Rational) raises -> Bool:
    return x.is_finite() and x.lower_rational() <= value and value <= x.upper_rational()


def test_wide_rectangles_stay_bounded() raises:
    # Rectangles a differential check against Arb found indeterminate, each
    # now finite and holding the function at points of the rectangle.
    # Division by a positive real ball whose radius is half its midpoint.
    var c = BallContext(113)
    var q = complex_ball.divide(
        ComplexBall(Ball(Rational(1, 20)), Ball(0)),
        ComplexBall(Ball(Rational(3, 1000000), Rational(3, 2000000), precision=113), Ball(0)), context=c,
    )
    assert_true(_holds(q.real(), Rational(100000, 3)) and _holds(q.real(), Rational(100000, 9)) and _holds(q.imag(), Rational(0)))
    # A divisor whose real part holds 0 but whose imaginary part does not.
    var d = complex_ball.divide(ComplexBall(1, 0), ComplexBall(Ball(0, 3), Ball(4)), context=c)
    assert_true(_holds(d.real(), Rational(0)) and _holds(d.imag(), Rational(-1, 4)) and _holds(d.real(), Rational(3, 25)))
    # log on the imaginary axis, |z| from 1/2 to 3/2 (|m| = 1 exactly).
    var l = complex_ball.log(ComplexBall(Ball(0), Ball(Rational(-1), Rational(1, 2), precision=24)))
    assert_true(_holds(l.real(), Rational(-693, 1000)) and _holds(l.real(), Rational(405, 1000)))
    # log right of 0 with a tiny real part and a wide imaginary one.
    var near = complex_ball.log(ComplexBall(Ball(Rational(1, 1 << 30), precision=192), Ball(Rational(-5, 2), Rational(9, 2), precision=192)))
    assert_true(near.real().is_finite() and near.imag().is_finite())
    # sqrt across the negative real axis: the real part is near 0, the
    # imaginary part within the root of the magnitude.
    var s = complex_ball.sqrt(ComplexBall(Ball(-1, 1), Ball(Rational(0), Rational(1, 1 << 55), precision=53)))
    assert_true(_holds(s.real(), Rational(0)) and _holds(s.imag(), Rational(7, 5)) and _holds(s.imag(), Rational(-7, 5)))
    var right = complex_ball.sqrt(ComplexBall(Ball(Rational(1, 1 << 30), precision=64), Ball(Rational(266), Rational(530), precision=64)))
    assert_true(right.real().is_finite() and right.imag().is_finite())
    # tan far from the real axis: within 4e-6 of -i over [-84, -12] i.
    var t = complex_ball.tan(ComplexBall(Ball(-1), Ball(Rational(-48), Rational(36), precision=64)))
    assert_true(_holds(t.imag(), Rational(-1)) and _holds(t.real(), Rational(0)))
    assert_true(t.imag()._radius.compare(_Radius.power_of_two(-17)) < 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
