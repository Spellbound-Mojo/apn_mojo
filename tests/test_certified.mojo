"""Certified rounding (T-ZIV): the Ziv driver against direct rounding, its
retry path and budget, to_float_if_certain, rounding near a Float, binary
splitting and fixed-point balls."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, Rational, Float, Ball, FloatFormat, ArithmeticContext, RoundingMode, to_float_if_certain
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._functions import _sqrt_float
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.ball._certified import _round_certified, _round_near
from apn_mojo.ball._kernels import _RealKernel, _SQRT
from apn_mojo.ball._fixed import _Fix
from apn_mojo.common._binary_splitting import _binary_split, _SeriesTerms


def _modes() -> List[RoundingMode]:
    return [
        RoundingMode.nearest_even,
        RoundingMode.toward_zero,
        RoundingMode.toward_positive,
        RoundingMode.toward_negative,
        RoundingMode.away_from_zero,
    ]


struct _Random(Movable):
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt64:
        self.state ^= self.state << 13
        self.state ^= self.state >> 7
        self.state ^= self.state << 17
        return self.state

    def below(mut self, n: Int) -> Int:
        return Int(self.next() % UInt64(n))

    def integer(mut self, bits: Int) raises -> Integer:
        """A random integer of exactly `bits` bits."""
        var value = Integer(0)
        var have = 0
        while have < bits:
            value = (value << 64) + Integer(self.next())
            have += 64
        value >>= have - bits
        return value | (Integer(1) << (bits - 1))


def _float(m: Integer, exponent: Int, precision: Int) raises -> Float:
    """The Float `m * 2**(exponent - precision)` for an m of `precision` bits."""
    return Float(_rounded=_RoundedBinary(1, False, m, exponent, FloatFormat(precision), NumericStatus()))


def _same(a: _RoundedBinary, b: _RoundedBinary) -> Bool:
    return (
        a.kind == b.kind and a.negative == b.negative and a.significand == b.significand
        and a.exponent == b.exponent and a.status == b.status
    )


def test_ziv_sqrt_matches_direct_rounding() raises:
    # Ziv over the sqrt kernel must reproduce the directly rounded root, status
    # included, in every mode; with one initial guard bit it retries first.
    var rng = _Random(0x9E3779B97F4A7C15)
    var precisions: List[Int] = [2, 3, 7, 24, 53, 64, 65, 113, 128, 200, 500, 1000, 4096]
    for p in precisions:
        for trial in range(10):
            var x: Float
            if trial % 3 == 0:
                # A perfect square, exact at the precision or not.
                var root_bits = 1 + rng.below(2 * p)
                var r = rng.integer(root_bits)
                var square = r * r
                x = _float(square, square.magnitude_bit_length() + 2 * (rng.below(200) - 100), square.magnitude_bit_length())
            else:
                var q = 1 + rng.below(2 * p + 8)
                x = _float(rng.integer(q), rng.below(600) - 300, q)
            for mode in _modes():
                var context = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                var direct = _sqrt_float(x, context)
                var kernel = _RealKernel.unary(_SQRT, x)
                assert_true(_same(direct, _round_certified(kernel, context)), String("sqrt mismatch at ", x, " p=", p))
                assert_true(_same(direct, _round_certified(kernel, context, guard=1)), String("retry mismatch at ", x))


def test_budget() raises:
    # Past the budget the driver raises, naming the function and the budget.
    var kernel = _RealKernel.unary(_SQRT, Float(2, context=ArithmeticContext(format=FloatFormat(53))))
    var tight = ArithmeticContext(format=FloatFormat(53), max_precision=61)
    with assert_raises(contains="precision budget exceeded: sqrt at 2.0 needed more than 61 bits"):
        _ = _round_certified(kernel, tight)
    # The default budget is max(8p, p + 4096).
    assert_equal(ArithmeticContext(format=FloatFormat(53))._budget(), 4149)
    assert_equal(ArithmeticContext(format=FloatFormat(1000))._budget(), 8000)
    with assert_raises(contains="max_precision below 1"):
        _ = ArithmeticContext(max_precision=0)


def test_to_float_if_certain() raises:
    var c53 = ArithmeticContext(format=FloatFormat(53))
    # An exact ball rounds like its midpoint.
    var third = to_float_if_certain(Ball(Rational(1, 3), precision=200), context=c53)
    assert_true(Bool(third))
    assert_true(third.value() == Float(Rational(1, 3), context=c53))
    # A ball across a rounding boundary does not decide.
    var boundary = Float(1) + Float(Rational(1, 2**53), context=ArithmeticContext(format=FloatFormat(200)))
    assert_false(Bool(to_float_if_certain(Ball(boundary, Rational(1, 2**80)), context=c53)))
    # A ball that contains 0 never decides; exact 0 does.
    assert_false(Bool(to_float_if_certain(Ball(0, Rational(1, 2**100)), context=c53)))
    assert_true(to_float_if_certain(Ball(0), context=c53).value().is_zero())
    assert_false(Bool(to_float_if_certain(Ball.unbounded(), context=c53)))
    # Directed rounding decides from the side of the ball.
    var up = ArithmeticContext(format=FloatFormat(53), rounding=RoundingMode.toward_positive)
    var near_one = Ball(Float(1) + Float(Rational(1, 2**60), context=ArithmeticContext(format=FloatFormat(100))), Rational(1, 2**70))
    assert_true(to_float_if_certain(near_one, context=up).value() == Float(1) + Float(Rational(1, 2**52), context=c53))
    with assert_raises(contains="inexact"):
        _ = to_float_if_certain(Ball(Rational(1, 3), precision=200), context=ArithmeticContext(format=FloatFormat(53), trap_inexact=True))


def test_round_near() raises:
    # v = 1 + d or 1 - d with a tiny d: no boundary lies between.
    var one = Float(1, context=ArithmeticContext(format=FloatFormat(1)))
    var c53 = ArithmeticContext(format=FloatFormat(53))
    var above = Float(1) + Float(Rational(1, 2**52), context=c53)
    var below = Float(1) - Float(Rational(1, 2**53), context=c53)
    var expected_up: List[Float] = [Float(1), Float(1), above, Float(1), above]
    var expected_down: List[Float] = [Float(1), below, Float(1), below, Float(1)]
    var modes = _modes()
    for i in range(5):
        var context = ArithmeticContext(format=FloatFormat(53), rounding=modes[i])
        var up = _round_near(one, True, -100, context)
        var down = _round_near(one, False, -100, context)
        assert_true(Float(_rounded=up.value()) == expected_up[i])
        assert_true(Float(_rounded=down.value()) == expected_down[i])
        assert_true(up.value().status.inexact() and down.value().status.inexact())
    # The nearest result lies below an upward value, and above a downward one.
    var nearest_up = _round_near(one, True, -100, c53).value()
    assert_equal(nearest_up.status.direction(), "below")
    # Too large a distance does not decide.
    assert_false(Bool(_round_near(one, True, -50, c53)))
    # A base with more bits than the target: the granularity of the base counts.
    var wide = Float(Rational(Integer(2) ** 80 + 1, Integer(2) ** 80), context=ArithmeticContext(format=FloatFormat(81)))
    assert_false(Bool(_round_near(wide, True, -82, c53)))
    var decided = _round_near(wide, True, -90, c53).value()
    assert_true(Float(_rounded=decided) == Float(1))


struct _ExpTerms(_SeriesTerms):
    """1/0! + 1/1! + ...: p = 1, q(k) = k."""

    def __init__(out self):
        pass

    def p(self, k: Int) raises -> Integer:
        return Integer(1)

    def q(self, k: Int) raises -> Integer:
        return Integer(max(k, 1))

    def a(self, k: Int) raises -> Integer:
        return Integer(1)

    def b(self, k: Int) raises -> Integer:
        return Integer(1)


struct _OddTerms(_SeriesTerms):
    """atanh(1/3) = sum 1 / ((2k + 1) 3**(2k + 1)), with b(k) = 2k + 1."""

    def __init__(out self):
        pass

    def p(self, k: Int) raises -> Integer:
        return Integer(1)

    def q(self, k: Int) raises -> Integer:
        return Integer(3 if k == 0 else 9)

    def a(self, k: Int) raises -> Integer:
        return Integer(1)

    def b(self, k: Int) raises -> Integer:
        return Integer(2 * k + 1)


def test_binary_splitting() raises:
    # The splits equal the exact partial sums, for every count (odd counts
    # merge unequal halves at the end).
    for n in range(1, 40):
        var split = _binary_split(_ExpTerms(), 0, n)
        var expected = Rational(0)
        var factorial = Integer(1)
        for k in range(n):
            if k > 0:
                factorial *= k
            expected += Rational(Integer(1), factorial)
        assert_true(Rational(split.t, split.b * split.q) == expected, String("e partial sum ", n))
        var odd = _binary_split(_OddTerms(), 0, n)
        var expected_odd = Rational(0)
        for k in range(n):
            expected_odd += Rational(Integer(1), Integer(2 * k + 1) * Integer(3) ** (2 * k + 1))
        assert_true(Rational(odd.t, odd.b * odd.q) == expected_odd, String("atanh partial sum ", n))


def _contains(x: Ball, value: Rational) raises -> Bool:
    return x.lower_rational() <= value and value <= x.upper_rational()


def test_fixed_point_balls() raises:
    var s = 200
    var two = _Fix.exact(Integer(2), s)
    var third = _Fix.ratio(Integer(1), Integer(3), s)
    # Every operation's ball contains the exact result.
    var root = two.sqrt().to_ball(150)
    assert_true(root.lower_rational() ** 2 <= 2 and root.upper_rational() ** 2 >= 2)
    assert_true(root.relative_accuracy_bits() > 140)
    assert_true(_contains(third.to_ball(150), Rational(1, 3)))
    assert_true(_contains(third.mul(third).to_ball(150), Rational(1, 9)))
    assert_true(_contains(third.square().to_ball(150), Rational(1, 9)))
    assert_true(_contains(two.div(third).to_ball(150), Rational(6)))
    assert_true(_contains(third.div_integer(Integer(-7)).to_ball(150), Rational(-1, 21)))
    assert_true(_contains(third.mul_integer(Integer(12)).to_ball(150), Rational(4)))
    assert_true(_contains(third.scale2(-5).to_ball(150), Rational(1, 96)))
    assert_true(_contains(third.rescale(120).to_ball(100), Rational(1, 3)))
    assert_true(_contains(third.sub(two).add(third).neg().to_ball(150), Rational(4, 3)))
    # Errors compound: a long chain still encloses.
    var x = third
    var exact = Rational(1, 3)
    for k in range(1, 30):
        x = x.mul(third).add(_Fix.ratio(Integer(1), Integer(k), s))
        exact = exact * Rational(1, 3) + Rational(1, k)
    assert_true(_contains(x.to_ball(180), exact))
    # Division by a ball that may contain 0, and roots of one that may reach
    # below 0, are indeterminate.
    var tiny = _Fix(Integer(1), x.err.add(x.err), s)
    assert_true(two.div(tiny).to_ball(64).is_indeterminate())
    assert_true(tiny.neg().sqrt().to_ball(64).is_indeterminate())
    assert_equal(third.sign(), 1)
    assert_equal(third.neg().sign(), -1)
    assert_equal(tiny.sign(), 0)
    # A Float that fits the scale is exact; one that does not is rounded.
    var exact_float = _Fix.of_float(Float(Rational(3, 8), context=ArithmeticContext(format=FloatFormat(8))), 10)
    assert_true(exact_float.err.is_zero() and exact_float.mid == 384)
    var fine = _Fix.of_float(Float(Rational(1, 3), context=ArithmeticContext(format=FloatFormat(60))), 10)
    assert_false(fine.err.is_zero())
    assert_true(_contains(fine.to_ball(10), Float(Rational(1, 3), context=ArithmeticContext(format=FloatFormat(60))).to_rational_exact()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
