"""Lift shares vmap's exact numeric inputs, including ordered Ball folds."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import Batch, Integer, Rational, Float, Complex, Ball, ComplexBall, BallContext, ArithmeticContext, ComplexContext, FloatFormat, lift, vmap
from apn_mojo import integer, rational, float, complex, ball, complex_ball


def test_calls_share_exact_adapters_and_scalar_broadcasting() raises:
    var xs = Batch[Integer]([1, 2, 3])
    assert_equal(lift[integer.add]()(xs, 1267650600228229401496703205377)[0], Integer(1267650600228229401496703205378))
    assert_equal(lift[rational.add]()(xs, Rational(1, 3)).to_list(), [Rational(4, 3), Rational(7, 3), Rational(10, 3)])
    assert_equal(lift[integer.add]()(2, 3).shape(), List[Int]())
    assert_equal(lift[integer.add]()(2, 3).item(), Integer(5))
    var c = ArithmeticContext(format=FloatFormat(8))
    var fs = Batch[Float]([Float(256, context=c), Float(-256, context=c)])
    var exact = Batch[Rational]([Rational(-511, 2), Rational(513, 2)])
    var result = lift[float.add]()(fs, exact, context=c)
    assert_equal(result.to_list(), [Float(Rational(1, 2)), Float(Rational(1, 2))])
    assert_equal(result.to_json(), vmap[float.add]()(fs, exact, context=c).to_json())
    var zs = Batch[Complex]([Complex(256, 1, context=c), Complex(-256, 1, context=c)])
    assert_equal(lift[complex.add]()(zs, exact, context=c)[0], Complex(Rational(1, 2), 1))
    assert_equal(lift[complex.add]()(zs, exact, context=c).to_json(), vmap[complex.add]()(zs, exact, context=c).to_json())
    with assert_raises(contains="requires a Complex operand"):
        _ = lift[complex.add]()(fs, exact, context=c)
    with assert_raises(contains="Exact operands do not select a Float format"):
        _ = lift[float.add]()(xs, 1)
    var rounded = lift[float.add]()(xs, Rational(1, 3), context=c)
    assert_equal(rounded[2], float.add(Integer(3), Rational(1, 3), context=c))


def test_broadcast_outer_views_and_empty_shapes() raises:
    var xs = Batch[Integer]([1, 2, 3, 4, 5, 6]).reshape([2, 3]).transpose()
    var ys = Batch[Rational]([Rational(1, 3), Rational(2, 3)])
    var f = lift[rational.add]()
    var calls = f(xs, ys)
    assert_equal(calls.shape(), [3, 2])
    for i in range(3):
        for j in range(2):
            assert_equal(calls[i, j], Rational(xs[i, j]) + ys[j])
    var out = f.outer(xs.slice(0, step=-1), ys[::-1])
    assert_equal(out.shape(), [3, 2, 2])
    assert_equal(out[0, 1, 0], Rational(20, 3))
    assert_equal(f.outer(2, ys).to_list(), [Rational(7, 3), Rational(8, 3)])
    assert_equal(f.outer(xs, Rational(1, 3)).shape(), [3, 2])
    assert_equal(f(xs.slice(0, stop=0), ys).shape(), [0, 2])
    assert_equal(f.outer(xs.slice(0, stop=0), ys).shape(), [0, 2, 2])
    with assert_raises(contains="Cannot broadcast shapes"):
        _ = f(xs, Batch[Integer]([1, 2, 3]))


def test_ball_calls_and_folds_match_scalar_enclosures() raises:
    var context = BallContext(precision=17)
    var xs = Batch[Ball]([Ball(Rational(1, 3)), Ball(2), Ball(3, Rational(1, 8))])
    var exact = Batch[Rational]([Rational(2, 3), Rational(1, 7), Rational(3, 5)])
    var f = lift[ball.add]()
    var calls = f(xs, exact, context=context)
    var out = f.outer(xs[::-1], 1267650600228229401496703205377, context=context)
    for i in range(3):
        assert_true(calls[i].same_representation(ball.add(xs[i], exact[i], context=context)))
        assert_true(out[i].same_representation(ball.add(xs[2 - i], Integer(1267650600228229401496703205377), context=context)))
    var running = f.accumulate(xs, context=context)
    var expected = xs[0]
    assert_true(running[0].same_representation(expected))
    for i in range(1, 3):
        expected = ball.add(expected, xs[i], context=context)
        assert_true(running[i].same_representation(expected))
    assert_true(f.reduce(xs, axis=None, context=context).same_representation(expected))
    assert_true(lift[ball.add](exact=False).reduce(xs, axis=None, context=context).same_representation(expected))
    assert_true(f.reduce(xs[:1], axis=None, context=context).same_representation(xs[0]))
    assert_equal(f(xs[:0], 2, context=context).shape(), [0])


def test_mixed_folds_keep_the_first_operand_exact() raises:
    var thirds = Batch[Rational]([Rational(1, 3), Rational(2, 3), Rational(2)])
    var c = ArithmeticContext(format=FloatFormat(8))
    var f = lift[float.add]()
    assert_equal(f.reduce(thirds, axis=None, context=c), Float(3))
    var prefixes = f.accumulate(thirds, context=c)
    assert_equal(prefixes.to_list(), [Float(Rational(1, 3), context=c), Float(1), Float(3)])
    var cancellation = Batch[Rational]([Rational(4096), Rational(-4095)])
    assert_equal(f.reduce(cancellation, axis=None, context=c), Float(1))
    assert_equal(lift[float.add](exact=False).reduce(cancellation, axis=None, context=c), Float(1))
    assert_equal(lift[float.add](exact=False).accumulate(cancellation, context=c)[1], Float(1))
    var pairs = ComplexContext(real=c, imag=ArithmeticContext(format=FloatFormat(13)))
    var halves = Batch[Rational]([Rational(1, 2), Rational(3, 2)])
    assert_equal(lift[complex.add]().reduce(halves, axis=None, initial=Complex(0, 1), context=pairs), Complex(2, 1, context=pairs))
    var initial = Complex(0, 1, context=pairs)
    var merged = lift[complex.add]().reduce(halves, axis=None, initial=initial)
    assert_equal(merged.real_format(), initial.real_format())
    assert_equal(merged.imag_format(), initial.imag_format())
    assert_equal(lift[complex.add]().accumulate(thirds[:1], context=pairs)[0], Complex(Rational(1, 3), context=pairs))
    with assert_raises(contains="requires a Complex operand"):
        _ = lift[complex.add]().reduce(thirds, axis=None, context=pairs)
    with assert_raises(contains="accumulate failed at index [1]"):
        _ = lift[complex.add]().accumulate(thirds, context=pairs)
    var bc = BallContext(precision=8)
    var balls = lift[ball.add]().accumulate(thirds, context=bc)
    assert_true(ball.contains(balls[0], Rational(1, 3)))
    assert_true(balls[1].is_exact())
    assert_true(ball.contains(balls[1], 1))
    assert_true(ball.contains(lift[ball.add]().reduce(thirds, axis=None, context=bc), 3))
    assert_equal(lift[rational.add]().reduce(Batch[Integer]([1, 2, 3]), axis=None), Rational(6))
    assert_equal(lift[rational.add]().accumulate(Batch[Integer]([1, 2, 3])).to_list(), [Rational(1), Rational(3), Rational(6)])


def test_mixed_fold_contexts_empty_singleton_initial_and_identity() raises:
    var c = ArithmeticContext(format=FloatFormat(9))
    var xs = Batch[Integer]([1, 2, 3])
    var f = lift[float.add](identity=Float(17))
    assert_equal(f.reduce(xs[:0], axis=None, context=c).to_json(), Float(17).to_json())
    assert_equal(f.reduce(xs[:0], axis=None, initial=Float(5), context=c), Float(5))
    assert_equal(f.reduce(xs, axis=None, initial=Float(5)), Float(11))
    assert_equal(f.reduce(xs[:1], axis=None, context=c).format(), c.format())
    assert_equal(f.accumulate(xs[:0], context=c).shape(), [0])
    with assert_raises(contains="no identity"):
        _ = lift[ball.add]().reduce(xs[:0], axis=None)
    with assert_raises(contains="Exact operands do not select a Float format"):
        _ = f.reduce(xs, axis=None)
    with assert_raises(contains="Exact operands do not select a Float format"):
        _ = f.reduce(xs[:1], axis=None)
    var bf = lift[ball.add](identity=Ball(7))
    assert_true(bf.reduce(xs[:0], axis=None).same_representation(Ball(7)))
    assert_true(ball.contains(bf.reduce(xs, axis=None, initial=Ball(5)), 11))
    assert_equal(bf.reduce(xs[:1], axis=None, context=BallContext(precision=19)).precision(), 19)


def test_mixed_fold_axes_and_ordered_rounding() raises:
    var xs = Batch[Integer]([1, 2, 3, 4, 5, 6]).reshape([2, 3]).transpose()
    var c = ArithmeticContext(format=FloatFormat(8))
    var f = lift[float.subtract](exact=False)
    assert_equal(f.reduce(xs, axis=1, keepdims=True, context=c).shape(), [3, 1])
    assert_equal(f.reduce(xs, axis=1, context=c).to_list(), [Float(-3), Float(-3), Float(-3)])
    assert_equal(f.accumulate(xs, axis=1, context=c)[2, 1], Float(-3))
    assert_equal(lift[rational.add](associative=True).reduce(xs, axis=[0, 1]).item(), Rational(21))
    var long = Batch[Integer]([Integer(i % 7) for i in range(20001)])[::-1]
    var expected = Integer(0)
    for value in long:
        expected += value
    assert_equal(lift[float.add](associative=True).reduce(long, axis=None, context=c), Float(expected, context=c))
    assert_equal(lift[rational.add](associative=True).reduce(long, axis=None), Rational(expected))


def _fail(x: Rational, y: Rational) raises -> Rational:
    if y == Rational(13):
        raise Error("unlucky")
    return x + y


def test_mixed_failures_report_coordinates_and_keep_inputs() raises:
    var xs = Batch[Integer]([1, 2, 3, 4, 13, 6]).reshape([2, 3])
    var saved = xs.to_json()
    var destination = Batch[Rational]([Rational(99)])
    with assert_raises(contains="call failed at index [1, 1]"):
        destination = lift[_fail]()(Rational(0), xs)
    with assert_raises(contains="outer failed at index [0, 1, 1]"):
        destination = lift[_fail]().outer(Batch[Integer]([1]), xs)
    with assert_raises(contains="reduce failed at index [1, 1]"):
        destination = lift[_fail]().reduce(xs, axis=1)
    with assert_raises(contains="accumulate failed at index [1, 1]"):
        destination = lift[_fail]().accumulate(xs, axis=1)
    assert_equal(destination[0], Rational(99))
    assert_equal(xs.to_json(), saved)
    var thirds = Batch[Rational]([Rational(1, 3), Rational(1, 3)])
    with assert_raises(contains="reduce failed at index [1]"):
        _ = lift[float.add]().reduce(thirds, axis=None, context=ArithmeticContext())


def _below(x: Rational, y: Rational) raises -> Bool:
    return x < y


def _update(mut x: Rational, y: Rational) raises:
    x += y


def _ball_sum(x: Ball, y: Ball) raises -> Ball:
    return x + y


def test_predicates_plain_ball_callbacks_and_in_place_updates() raises:
    var xs = Batch[Integer]([1, 2, 3])
    assert_equal(lift[_below]()(xs, Rational(5, 2)).to_list(), [True, True, False])
    assert_equal(lift[_below]().outer(2, xs).to_list(), [False, False, True])
    assert_equal(lift[_update]()(xs, Rational(1, 2))[0], Rational(3, 2))
    assert_equal(lift[_update]().outer(2, xs)[0], Rational(3))
    assert_equal(lift[_update]().reduce(xs, axis=None), Rational(6))
    assert_equal(lift[_update]().accumulate(xs).to_list(), [Rational(1), Rational(3), Rational(6)])
    var balls = Batch[Ball]([Ball(1), Ball(2)])
    assert_true(ball.contains(lift[_ball_sum]().reduce(balls, axis=None), 3))
    with assert_raises(contains="takes no context"):
        _ = lift[_below]()(xs, 2, context=ArithmeticContext())


def test_ball_specials_order_axes_and_empty_selections() raises:
    var xs = Batch[Ball]([Ball(10), Ball(1), Ball(2), Ball(3)]).reshape([2, 2]).transpose()
    var f = lift[ball.subtract]()
    assert_true(ball.contains(f.reduce(xs, axis=None), 4))
    assert_true(ball.contains(f.reduce(xs, axis=1)[0], 8))
    assert_true(ball.contains(f.accumulate(xs, axis=1)[1, 1], -2))
    var special = Batch[Ball]([Ball.unbounded(), Ball.indeterminate()])
    assert_equal(lift[ball.contains]()(xs, Integer(10))[0, 0], True)
    var result = lift[ball.add]()(special, 1)
    assert_true(result[0].is_unbounded())
    assert_true(result[1].is_indeterminate())
    assert_true(lift[ball.add]().reduce(special, axis=None).is_indeterminate())
    var empty = xs.slice(0, stop=0)
    assert_equal(f.accumulate(empty, axis=1).shape(), [0, 2])
    assert_equal(f.reduce(empty, axis=1).shape(), [0])
    with assert_raises(contains="no identity"):
        _ = f.reduce(empty, axis=0)


def test_parallel_mixed_failures_clean_up_results() raises:
    var wide = (Integer(1) << 600) + 7
    var xs = Batch[Integer]([wide for _ in range(20000)])
    xs[19997] = 13
    var destination = Batch[Rational]([Rational(wide)])
    var count = wide._storage[Integer._Shared].count()
    for _ in range(20):
        with assert_raises(contains="call failed at index [19997]"):
            destination = lift[_fail]()(Rational(wide), xs)
        assert_equal(destination[0], Rational(wide))
        assert_equal(wide._storage[Integer._Shared].count(), count)
        with assert_raises(contains="accumulate failed at index [19, 997]"):
            destination = lift[_fail]().accumulate(xs.reshape([20, 1000]), axis=1)
        assert_equal(destination[0], Rational(wide))
        assert_equal(wide._storage[Integer._Shared].count(), count)
    assert_equal(xs[19997], Integer(13))
    assert_equal(destination[0], Rational(wide))


def _borrowed_add(x: Integer, y: Integer) raises -> Integer:
    assert_equal(y._storage[Integer._Shared].count(), UInt64(1))
    return x + y


def test_concrete_callbacks_keep_borrowed_elements() raises:
    var xs = Batch[Integer]([Integer(i + 1) << 200 for i in range(128)])
    var f = lift[_borrowed_add]()
    assert_equal(f(xs, xs)[0], xs[0] + xs[0])
    assert_equal(f.outer(xs[:2], xs[::-1])[0, 0], xs[0] + xs[-1])
    var expected = Integer(0)
    for value in xs:
        expected += value
    assert_equal(f.reduce(xs, axis=None), expected)
    assert_equal(f.reduce(xs[::-1], axis=None), expected)
    assert_equal(f.accumulate(xs)[-1], expected)
    assert_equal(xs[0], Integer(1) << 200)


def test_complex_balls_share_numeric_mapping_and_fold_contexts() raises:
    var values = Batch[ComplexBall]([ComplexBall(1, 2), ComplexBall(Rational(1, 3), -1)])
    var context = BallContext(precision=24)
    var f = lift[complex_ball.add]()
    var scalar = complex_ball.add(values[0], values[1], context=context)
    assert_true(f.reduce(values, axis=None, context=context).same_representation(scalar))
    var running = f.accumulate(values, context=context)
    assert_true(running[0].same_representation(values[0]))
    assert_true(running[1].same_representation(scalar))
    var calls = f(values, values[1], context=context)
    var mapped = vmap[complex_ball.add]()(values, values[1], context=context)
    assert_true(calls[0].same_representation(scalar))
    assert_true(calls[1].same_representation(mapped[1]))
    assert_true(f.outer(values[::-1], values, context=context)[0, 0].same_representation(scalar))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
