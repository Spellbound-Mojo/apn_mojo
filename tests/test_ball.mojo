"""Ball arithmetic: construction, inclusion of exact results, tightness against
Arb's reference radii, kinds and domains, three-valued comparison, sets,
accuracy, text and stable hashes."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    ArithmeticContext,
    Ball,
    Batch,
    Mask,
    vmap,
    BallContext,
    BallKey,
    BallOrder,
    Float,
    FloatFormat,
    Integer,
    Rational,
    add,
    ceil_if_certain,
    compare,
    contains,
    contains_ball,
    contains_integer,
    floor_if_certain,
    intersection,
    overlaps,
    pow_int,
    round_half_even_if_certain,
    simplest_rational_in,
    split,
    sqrt,
    stable_hash,
    union,
)
from apn_mojo import ball


def _at(precision: Int) raises -> Optional[BallContext]:
    return BallContext(precision)


def test_construction() raises:
    var five = Ball(5)
    assert_true(five.is_exact() and five.precision() == 128 and five.midpoint() == 5)
    var wide = Ball(Integer(1) << 200)
    assert_true(wide.is_exact() and wide.precision() == 201)
    var third = Ball(Rational(1, 3))
    assert_true(not third.is_exact() and third.precision() == 128)
    assert_true(contains(third, Rational(1, 3)))
    assert_true(third.radius() == Float("0x1p-130"))
    assert_true(Ball(Rational(3, 4)).is_exact())
    var tenth = Ball(Float("0.1", context=ArithmeticContext(format=FloatFormat(53))))
    assert_true(tenth.is_exact() and tenth.precision() == 53)
    assert_true(Ball(Rational(1, 3), precision=10).radius() == Float("0x1p-12"))
    var near_one = Ball(1, Rational(1, 1000))
    assert_true(contains(near_one, Rational(1001, 1000)) and contains(near_one, Rational(999, 1000)))
    assert_true(not contains(near_one, Rational(10011, 10000)))
    var parsed = Ball("[3.14 +/- 0.01]")
    assert_true(contains(parsed, Rational(3135, 1000)) and contains(parsed, Rational(3149, 1000)))
    assert_true(contains(Ball("1/3"), Rational(1, 3)))
    assert_true(Ball("0x3p-1").is_exact() and Ball("0x3p-1").midpoint() == Rational(3, 2))
    assert_true(Ball("[+/- inf]").is_unbounded())
    assert_true(Ball("[nan +/- inf]").is_indeterminate())
    var interval = Ball.from_interval(Rational(1, 3), Rational(1, 2))
    assert_true(contains(interval, Rational(1, 3)) and contains(interval, Rational(1, 2)) and contains(interval, Rational(5, 12)))
    with assert_raises(contains="unbounded"):
        _ = Ball(Float.infinity())
    with assert_raises(contains="negative"):
        _ = Ball(1, -1)
    with assert_raises(contains="low > high"):
        _ = Ball.from_interval(2, 1)


def test_exact_arithmetic() raises:
    var product = Ball(5) * Ball(7)
    assert_true(product.is_exact() and product.midpoint() == 35)
    assert_true((Ball(1) + Ball(2)).is_exact())
    assert_true((Ball(1) / Ball(4)).is_exact() and (Ball(1) / Ball(4)).midpoint() == Rational(1, 4))
    var third = Ball(1) / Ball(3)
    assert_true(not third.is_exact() and contains(third, Rational(1, 3)))
    assert_true((Ball(5) * 2).midpoint() == 10 and (Ball(5) * 2).is_exact())
    assert_true((2 - Ball(5)).midpoint() == -3)
    assert_true((Ball(3) * Rational(1, 3)).is_exact())
    assert_true((-Ball(4)).midpoint() == -4)
    var root = ball.sqrt(Ball(4))
    assert_true(root.is_exact() and root.midpoint() == 2)
    assert_true(ball.pow_int(Ball(2), 100).is_exact())
    assert_true(ball.ldexp(Ball(3), -2).midpoint() == Rational(3, 4))


def _points(x: Ball) raises -> List[Rational]:
    var low = x.lower_rational()
    var high = x.upper_rational()
    var mid = x.midpoint_rational()
    var radius = x.radius().to_rational_exact()
    return [low, high, mid, mid - radius / 3, mid + radius / 7, mid + radius * 5 / 9]


def _random_ball(mut state: UInt64, positive: Bool) raises -> Ball:
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    var center = Int(state % 10000) - 5000
    if positive:
        center = Int(state % 5000) + 200
    var radius = Int((state >> 20) % 50)
    return Ball(Rational(center, 1024), Rational(radius, 4096), precision=64)


def test_inclusion_of_exact_results() raises:
    # For points of the inputs, the exact result is a Rational inside the result.
    var state = UInt64(88172645463325252)
    var context = _at(40)
    for _ in range(150):
        var x = _random_ball(state, False)
        var y = _random_ball(state, False)
        var p = _random_ball(state, True)
        var sum = ball.add(x, y, context=context)
        var difference = ball.subtract(x, y, context=context)
        var product = ball.multiply(x, y, context=context)
        var squared = ball.square(x, context=context)
        var cube = ball.pow_int(x, 3, context=context)
        var magnitude = ball.abs(x, context=context)
        var quotient = ball.divide(x, p, context=context)
        var reciprocal = ball.reciprocal(p, context=context)
        var root = ball.sqrt(p, context=context)
        for a in _points(x):
            for b in _points(y):
                assert_true(contains(sum, a + b) and contains(difference, a - b) and contains(product, a * b))
            assert_true(contains(squared, a * a) and contains(cube, a * a * a) and contains(magnitude, abs(a)))
            for c in _points(p):
                assert_true(contains(quotient, a / c))
        for c in _points(p):
            assert_true(contains(reciprocal, 1 / c))
            # sqrt(c) lies in [lo, hi] when lo**2 <= c <= hi**2 (and lo <= 0 or).
            var high = root.upper_rational()
            var low = root.lower_rational()
            assert_true(high * high >= c and (low <= 0 or low * low <= c))
    # fma over every combination of points of its three operands.
    for _ in range(40):
        var x = _random_ball(state, False)
        var y = _random_ball(state, False)
        var z = _random_ball(state, False)
        var fused = ball.fma(x, y, z, context=context)
        for a in _points(x):
            for b in _points(y):
                for c in _points(z):
                    assert_true(contains(fused, a * b + c))


def test_tightness_against_arb() raises:
    # Arb's radii for X = [1 +/- 2**-20] at 64 bits (python-flint 0.9.0); ours
    # must be within 1.5 times.
    var x = Ball(1, Float("0x1p-20"), precision=64)
    var cases = List[Tuple[Ball, Float]]()
    cases.append((x * x, Float("1.9073e-6")))
    cases.append((ball.reciprocal(x), Float("9.5833e-7")))
    cases.append((x - x, Float("1.9073e-6")))
    cases.append((ball.sqrt(x), Float("4.8056e-7")))
    for entry in cases:
        assert_true(entry[0].radius() <= entry[1] * Rational(3, 2))


def test_kinds_and_domains() raises:
    var nothing = Ball.indeterminate()
    var anything = Ball.unbounded()
    var one = Ball(1)
    assert_true((nothing + one).is_indeterminate() and (nothing * Ball(0)).is_indeterminate())
    assert_true((anything + one).is_unbounded() and (anything * one).is_unbounded())
    var zero_times = Ball(0) * anything
    assert_true(zero_times.is_exact() and zero_times.midpoint() == 0)
    assert_true((one / anything).is_indeterminate() and (anything / Ball(2)).is_unbounded())
    assert_true(ball.abs(anything).is_unbounded() and ball.sqrt(anything).is_indeterminate())
    assert_true(ball.sqrt(Ball(-1, Rational(1, 2))).is_indeterminate())
    var edge = ball.sqrt(Ball(1, 1))
    assert_true(contains(edge, 0) and edge.upper_rational() * edge.upper_rational() >= 2)
    assert_true((one / Ball(0, 1)).is_indeterminate() and (one / Ball(0)).is_indeterminate())
    assert_true(ball.pow_int(anything, 0).is_exact() and ball.pow_int(Ball(0, 1), -1).is_indeterminate())
    assert_true(ball.reciprocal(Ball(0)).is_indeterminate())
    # atan2 takes the larger operand precision by default, on each branch: a
    # positive abscissa (which once returned 16 bits more), a nonzero
    # ordinate, and a negative abscissa.
    var y = Ball(Rational(1, 3), Rational(1, 1 << 60), precision=53)
    var x = Ball(Rational(2, 3), Rational(1, 1 << 60), precision=40)
    assert_true(ball.atan2(y, x).precision() == 53 and ball.atan2(y, -x).precision() == 53)
    assert_true(ball.atan2(-x, y).precision() == 53)


def test_comparisons() raises:
    var low = Ball(1, Rational(1, 2))
    var high = Ball(3, Rational(1, 2))
    assert_true(compare(low, high) == BallOrder.less and compare(high, low) == BallOrder.greater)
    # Touching ends share the point 2.
    var left = Ball(1, 1)
    var right = Ball(3, 1)
    assert_true(compare(left, right) == BallOrder.overlap and left.certainly_le(right) and not left.certainly_lt(right))
    assert_true(compare(Ball(2), Ball(Rational(4, 2))) == BallOrder.equal)
    assert_true(compare(low, low) == BallOrder.overlap and not low.certainly_eq(low))
    assert_true(compare(Ball.indeterminate(), low) == BallOrder.undefined)
    assert_true(compare(Ball.unbounded(), low) == BallOrder.overlap)
    assert_true(low.certainly_positive() and low.certainly_nonzero() and not Ball(0, 1).certainly_nonzero())
    assert_true(Ball(-2, 1).certainly_negative() and Ball(0, 1).certainly_nonnegative() == False)
    assert_true(Ball(1).certainly_lt(2) and Ball(2).certainly_ge(Rational(3, 2)) and low.certainly_ne(high))
    assert_equal(low.sign_if_certain().value(), 1)
    assert_equal(Ball(0).sign_if_certain().value(), 0)
    assert_true(not Ball(0, 1).sign_if_certain())


def test_sets() raises:
    var a = Ball(1, Rational(1, 2))
    var b = Ball(2, Rational(3, 4))
    var hull = union(a, b)
    assert_true(contains_ball(hull, a) and contains_ball(hull, b))
    var common = intersection(a, b)
    assert_true(common and contains(common.value(), Rational(3, 2)) and contains(common.value(), Rational(5, 4)))
    assert_true(not contains(common.value(), Rational(7, 4)))
    assert_true(not intersection(a, Ball(5, 1)))
    assert_true(overlaps(a, b) and not overlaps(a, Ball(5, 1)))
    assert_true(not contains_integer(Ball(Rational(2, 5), Rational(1, 10))) and contains_integer(Ball(Rational(9, 10), Rational(1, 5))))
    var halves = split(Ball(1, Rational(1, 3)))
    assert_true(contains_ball(union(halves[0], halves[1]), Ball(1, Rational(1, 3))))
    assert_true(halves[0].upper_rational() == halves[1].lower_rational())
    assert_equal(floor_if_certain(Ball(Rational(3, 2), Rational(1, 4))).value(), 1)
    assert_true(not floor_if_certain(Ball(Rational(19, 10), Rational(1, 5))))
    assert_equal(ceil_if_certain(Ball(Rational(3, 2), Rational(1, 4))).value(), 2)
    assert_equal(round_half_even_if_certain(Ball(Rational(12, 5), Rational(1, 20))).value(), 2)
    assert_equal(round_half_even_if_certain(Ball(Rational(5, 2))).value(), 2)
    # Mathematica: Rationalize[3.14159, 10^-3] is 201/64.
    assert_equal(simplest_rational_in(Ball.from_interval(Rational(314059, 100000), Rational(314259, 100000))), Rational(201, 64))
    assert_equal(simplest_rational_in(Ball.from_interval(Rational(-3, 5), Rational(-2, 5))), Rational(-1, 2))
    assert_equal(simplest_rational_in(Ball(0, 1)), Rational(0))
    assert_equal(simplest_rational_in(Ball(3, Rational(1, 10))), Rational(3))
    with assert_raises(contains="unbounded or indeterminate"):
        _ = split(Ball.unbounded())


def test_accuracy_and_text() raises:
    var x = Ball(1, Float("0x1p-20"), precision=64)
    assert_equal(x.relative_accuracy_bits(), 19)
    assert_equal(x.accuracy_bits(), 20)
    assert_equal(Ball(3).accuracy_bits(), Int.MAX)
    var rounded = ball.round_midpoint(Ball(Rational(1, 3)), 20)
    assert_true(rounded.precision() == 20 and contains_ball(rounded, Ball(Rational(1, 3))))
    var trimmed = ball.trim(x * Ball(Rational(1, 3)))
    assert_true(trimmed.precision() < 64 and contains_ball(trimmed, x * Ball(Rational(1, 3))))
    var widened = ball.add_error(Ball(1), Rational(1, 8))
    assert_true(contains(widened, Rational(9, 8)) and not contains(widened, Rational(10, 8)))
    assert_equal(x.to_string(), "[1.0000 +/- 9.54e-07]")
    assert_equal(Ball(3).to_string(digits=3), "[3.00 +/- 0]")
    assert_equal(String(Ball.unbounded()), "[+/- inf]")
    assert_equal(String(Ball.indeterminate()), "[nan +/- inf]")
    for value in [x, Ball(Rational(1, 3)), Ball(-7, Rational(1, 1000)), Ball(Float("0x1p-60"), Float("0x1p-90"))]:
        assert_true(contains_ball(Ball(value.to_string(), precision=128), value))


def test_hash_package_names_and_keys() raises:
    # The requirements' Appendix F.4 vectors for balls.
    var double = ArithmeticContext(format=FloatFormat(53))
    assert_equal(ball.stable_hash(Ball(Float(1, context=double))), 0xCE7F2A1DCFEC0F44)
    assert_equal(ball.stable_hash(Ball(Float(1, context=double), Float("0x1p-30"))), 0x1126EE35A67BF9F2)
    assert_equal(stable_hash(Ball(Float(1, context=double))), 0xCE7F2A1DCFEC0F44)
    assert_true(add(Ball(1), Ball(2)).midpoint() == 3 and add(Ball(1), 2).is_exact() and add(2, Ball(1)).midpoint() == 3)
    assert_true(sqrt(Ball(9)).midpoint() == 3 and pow_int(Ball(2), 10).midpoint() == 1024)
    var keys = Dict[BallKey, Int]()
    keys[BallKey(Ball(1))] = 1
    keys[BallKey(Ball(Float(1, context=double)))] = 2
    keys[BallKey(Ball(1, Rational(1, 8)))] = 3
    keys[BallKey(Ball(1))] = 4
    assert_equal(len(keys), 3)
    assert_equal(keys[BallKey(Ball(1))], 4)


def _meet(a: Ball, b: Ball) raises -> Optional[Ball]:
    return intersection(a, b)


def test_ball_batches() raises:
    var originals: List[Ball] = [Ball(1), Ball(2, Rational(1, 8)), Ball(Rational(1, 3))]
    var balls = Batch[Ball](originals)
    assert_equal(len(balls), 3)
    assert_true(balls[1].same_representation(originals[1]))
    assert_equal(String(balls), String("[", originals[0], ", ", originals[1], ", ", originals[2], "]"))
    assert_equal(String(balls.reshape([3, 1])), String("[[", originals[0], "], [", originals[1], "], [", originals[2], "]]"))
    assert_true(balls[1:][0].same_representation(originals[1]))
    var picked = balls[Mask([True, False, True])]
    assert_true(len(picked) == 2 and picked[1].same_representation(originals[2]))
    var updated = balls
    updated[Mask([False, True, False])] = Ball(7)
    assert_true(updated[1].midpoint() == 7 and balls[1].same_representation(originals[1]))
    updated[0:2] = balls[1:3]
    assert_true(updated[0].same_representation(originals[1]) and updated[1].same_representation(originals[2]))
    var count = 0
    for element in balls:
        assert_true(element.same_representation(originals[count]))
        count += 1
    assert_equal(count, 3)
    # vmap maps ball functions over batches of balls or of exact numbers.
    var sums = vmap[ball.add]()(balls, balls)
    assert_true(sums[0].is_exact() and sums[0].midpoint() == 2 and contains(sums[2], Rational(2, 3)))
    var roots = vmap[ball.sqrt]()(Batch[Integer]([4, 9, 2]), context=BallContext(64))
    assert_true(roots[1].midpoint() == 3 and roots[2].precision() == 64)
    assert_true(roots[2].upper_rational() * roots[2].upper_rational() >= 2 and roots[2].lower_rational() * roots[2].lower_rational() <= 2)
    assert_equal(vmap[contains]()(balls, Rational(1, 3)).to_list(), [False, False, True])
    var floors, certain = vmap[floor_if_certain]()(balls)
    assert_equal(floors.to_list(), [1, 0, 0])
    assert_equal(certain.to_list(), [True, False, True])
    # Positions without a result hold the family's zero.
    var meets, met = vmap[_meet]()(balls, Ball(2))
    assert_equal(met.to_list(), [False, True, False])
    assert_true(contains(meets[1], 2) and meets[0].is_exact() and meets[0].midpoint() == 0)
    # A long mapping runs on the worker pool and matches the scalar results.
    var many = List[Ball]()
    for i in range(3000):
        many.append(Ball(Rational(i, 7), Rational(1, 1000 + i)))
    var products = vmap[ball.multiply]()(Batch[Ball](many), Batch[Ball](many), context=BallContext(80))
    for i in range(3000):
        assert_true(products[i].same_representation(ball.multiply(many[i], many[i], context=BallContext(80))))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
