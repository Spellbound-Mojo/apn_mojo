"""Serialization and significance (T-SIG): the Ball and ComplexBall JSON
records of Appendix G.1 and G.2, their canonical form and round trips; the
precision conversions, the radius of a relative precision and the Appendix
E.11 propagation terms."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import (
    Integer, Rational, Float, Ball, ComplexBall, FloatFormat, ArithmeticContext, BallContext,
    bits_to_digits, digits_to_bits, radius_for_relative_digits, propagation_bound,
)

comptime _EXAMPLE = '{"version":1,"family":"ball","kind":"finite","midpoint":{"version":1,"family":"float","precision":"53","emin":"-4611686018427387904","emax":"4611686018427387903","class":"finite","sign":"+","significand":"4503599627370496","exponent":"1"},"radius":{"version":1,"family":"float","precision":"30","emin":"-4611686018427387904","emax":"4611686018427387903","class":"finite","sign":"+","significand":"536870912","exponent":"-29"}}'


def test_appendix_g1_example() raises:
    var x = Ball.from_json(_EXAMPLE)
    assert_true(x.midpoint() == Float(1) and x.radius() == Float(Rational(1, 1 << 30)))
    assert_equal(x.to_json(), _EXAMPLE)


def test_round_trips() raises:
    var balls: List[Ball] = [
        Ball(Rational(1, 3), Rational(1, 10**20), precision=80),
        Ball(Integer(-7)),
        Ball(Integer(0)),
        Ball.unbounded(64),
        Ball.indeterminate(96),
        Ball(Integer(1) << 1000, Rational(3)),
    ]
    for b in balls:
        var back = Ball.from_json(b.to_json())
        assert_true(back.same_representation(b), String("round trip of ", b))
        assert_equal(back.to_json(), b.to_json())
    var z = ComplexBall(Ball(Rational(1, 3), Rational(1, 10**9), precision=70), Ball.unbounded(32))
    var text = z.to_json()
    assert_true(text.startswith('{"version":1,"family":"complex_ball","real":{"version":1,"family":"ball"'))
    assert_true(ComplexBall.from_json(text).same_representation(z))


def test_canonical_form() raises:
    # Each record differs from a canonical one in one place.
    # A well-formed 31-bit Float of the same value: canonical as a Float, not as a radius.
    var radius31 = String(_EXAMPLE).replace('"precision":"30"', '"precision":"31"').replace('"significand":"536870912"', '"significand":"1073741824"')
    with assert_raises(contains="non-canonical radius"):
        _ = Ball.from_json(radius31)
    var negative = String(_EXAMPLE).replace('"sign":"+","significand":"536870912"', '"sign":"-","significand":"536870912"')
    with assert_raises(contains="non-canonical radius"):
        _ = Ball.from_json(negative)
    var unbounded = String(_EXAMPLE).replace('"kind":"finite"', '"kind":"unbounded"')
    with assert_raises(contains="non-canonical unbounded ball"):
        _ = Ball.from_json(unbounded)
    var kind = String(_EXAMPLE).replace('"kind":"finite"', '"kind":"exact"')
    with assert_raises(contains="unknown kind"):
        _ = Ball.from_json(kind)
    var extra = String(_EXAMPLE).replace('"kind":"finite",', '"kind":"finite","note":"1",')
    with assert_raises(contains="unknown field"):
        _ = Ball.from_json(extra)
    var missing = String(_EXAMPLE).replace('"kind":"finite",', '')
    with assert_raises(contains="missing required field"):
        _ = Ball.from_json(missing)
    with assert_raises(contains="wrong number family"):
        _ = Ball.from_json(String(_EXAMPLE).replace('"family":"ball"', '"family":"float"'))


def _within(x: Ball, value: Rational, tolerance: Rational) raises -> Bool:
    return x.lower_rational() >= value - tolerance and x.upper_rational() <= value + tolerance and x.lower_rational() <= value + tolerance


def test_precision_conversions() raises:
    # 53 log10(2) = 15.954589770191003...; 15 log2(10) = 49.828921423310435...
    var digits = bits_to_digits(Float(53))
    assert_true(digits.lower_rational() < Rational(15954589770191004, 10**15) and digits.upper_rational() > Rational(15954589770191003, 10**15))
    var bits = digits_to_bits(Float(15), context=BallContext(200))
    assert_true(bits.lower_rational() < Rational(49828921423310436, 10**15) and bits.upper_rational() > Rational(49828921423310435, 10**15))
    assert_true(bits.precision() == 200)
    var r = radius_for_relative_digits(Float(1), Float(10))
    assert_true(r.to_rational_exact() >= Rational(1, 10**10) and r.to_rational_exact() <= Rational(1, 10**10) * Rational(1 + (1 << 27), 1 << 27))
    assert_equal(r.precision(), 30)
    var negative = radius_for_relative_digits(Float(-1000), Float(Rational(5, 2)))
    # 1000 10**-2.5 = sqrt(10) = 3.16227766016...
    assert_true(negative.to_rational_exact() >= Rational(316227766, 10**8) and negative.to_rational_exact() < Rational(316227767, 10**8))
    with assert_raises(contains="nonzero midpoint"):
        _ = radius_for_relative_digits(Float(0), Float(3))


def test_propagation_terms() raises:
    var small = Float(Rational(1, 1 << 20))
    # exp at 0: e**0 (e**r - 1), just above r.
    var exp_term = propagation_bound["exp"](Float(0), small).to_rational_exact()
    assert_true(exp_term >= Rational(1, 1 << 20) + Rational(1, 1 << 41) and exp_term < Rational(1, 1 << 20) * Rational((1 << 25) + 1, 1 << 25) + Rational(1, 1 << 40))
    # log at [2 +/- 1/2]: r / (m - r) = 1/3.
    var log_term = propagation_bound["log"](Float(2), Float(Rational(1, 2))).to_rational_exact()
    assert_true(log_term >= Rational(1, 3) and log_term < Rational(1, 3) * Rational((1 << 25) + 1, 1 << 25))
    assert_true(propagation_bound["tanh"](Float(5), Float(3)) == Float(2))
    assert_true(propagation_bound["sin"](Float(0), small).to_rational_exact() >= Rational(1, 1 << 20))
    var atan_term = propagation_bound["atan"](Float(3), Float(1)).to_rational_exact()
    assert_true(atan_term >= Rational(1, 5) and atan_term < Rational(1, 5) * Rational((1 << 25) + 1, 1 << 25))
    with assert_raises(contains="inside (-1, 1)"):
        _ = propagation_bound["atanh"](Float(Rational(1, 2)), Float(Rational(1, 2)))
    with assert_raises(contains="m > r"):
        _ = propagation_bound["log"](Float(1), Float(1))
    with assert_raises(contains="no term for"):
        _ = propagation_bound["gamma"](Float(1), small)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
