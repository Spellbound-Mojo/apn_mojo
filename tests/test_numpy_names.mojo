"""NumPy's names and semantics: maximum, minimum and clip in each family; the
max, min, argmax and argmin reductions; the rounding functions; reciprocal;
conjugate, real, imag, angle and abs of complex values; comb, factorial2,
ldexp, nextafter and spacing; and vmap over the family declarations."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    Integer, Rational, Float, Complex, ExactComplex, Ball, ComplexBall, Batch,
    FloatFormat, ArithmeticContext,
    maximum, minimum, clip, floor, ceil, trunc, round, reciprocal, conjugate,
    real, imag, angle, abs, comb, factorial2, ldexp, nextafter, spacing,
    argmin, argmax, vmap,
)
# numpy's own aliases, which keep Mojo's built-in min and max visible.
from apn_mojo import min as amin, max as amax
from apn_mojo import integer, rational, float


def _c() raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat.binary64())


def test_exact_extremes() raises:
    assert_equal(maximum(Integer(3), Integer(-5)), 3)
    assert_equal(minimum(3, -5), -5)
    assert_equal(maximum(Rational(1, 3), Rational(1, 2)), Rational(1, 2))
    assert_equal(minimum(Rational(1, 3), Rational(1, 2)), Rational(1, 3))
    assert_equal(clip(Integer(10), Integer(0), Integer(5)), 5)
    assert_equal(clip(Integer(-3), Integer(0), Integer(5)), 0)
    assert_equal(clip(Integer(2), Integer(0), Integer(5)), 2)
    # numpy applies the upper limit last: a_max wins when the limits cross.
    assert_equal(clip(Integer(2), Integer(5), Integer(0)), 0)
    assert_equal(clip(Rational(7, 2), Rational(1, 2), Rational(3)), 3)


def test_float_extremes() raises:
    var c = _c()
    var one = Float(1, context=c)
    var nan = Float.nan(context=c)
    assert_true(maximum(one, nan).is_nan() and maximum(nan, one).is_nan())
    assert_true(minimum(one, nan).is_nan() and minimum(nan, one).is_nan())
    # numpy's x if x >= y else y: equal values give the first, signed zeros included.
    var positive = Float.zero(context=c)
    var negative = Float.zero(negative=True, context=c)
    assert_true(maximum(negative, positive).signbit() and not maximum(positive, negative).signbit())
    assert_true(minimum(positive, negative).signbit() == False and minimum(negative, positive).signbit())
    # Formats merge as for add: the wider precision.
    var narrow = Float(3, context=ArithmeticContext(format=FloatFormat(24, emin=-1021, emax=1024)))
    var larger = maximum(narrow, Float(2, context=c))
    assert_equal(larger, Float(3, context=c))
    assert_equal(larger.precision(), 53)
    # Exact operands compare exactly; the chosen one is rounded once.
    assert_equal(maximum(one, Rational(4, 3)), Float(Rational(4, 3), context=c))
    assert_equal(minimum(Float(1, context=c), Integer(-2)), Float(-2, context=c))
    assert_equal(maximum(Integer(1), Integer(2), context=c), Float(2, context=c))
    with assert_raises():
        _ = float.maximum(Integer(1), Integer(2))
    assert_equal(clip(Float(7, context=c), Integer(0), Integer(5)), Float(5, context=c))
    assert_equal(clip(Float(-7, context=c), Integer(0), Integer(5)), Float(0, context=c))
    assert_equal(clip(Float(3, context=c), Integer(0), Integer(5)), Float(3, context=c))
    assert_true(clip(Float(3, context=c), nan, Integer(5)).is_nan())


def test_ball_extremes() raises:
    var x = Ball(1, Rational(1, 10))
    var above = maximum(x, Ball(3))
    assert_true(above.is_exact() and above.midpoint() == 3)
    var below = minimum(x, Ball(3))
    assert_true(below.lower_rational() == x.lower_rational() and below.upper_rational() == x.upper_rational())
    var overlap = maximum(x, Ball(1, Rational(1, 2)))
    assert_true(overlap.lower_rational() <= Rational(9, 10) and overlap.upper_rational() >= Rational(3, 2))
    assert_true(overlap.lower_rational() >= Rational(4, 5))
    var low = minimum(x, Ball(1, Rational(1, 2)))
    assert_true(low.lower_rational() <= Rational(1, 2) and low.upper_rational() >= Rational(11, 10))
    assert_true(maximum(Ball.indeterminate(64), x).is_indeterminate())
    var limited = clip(Ball(10), 0, 5)
    assert_true(limited.is_exact() and limited.midpoint() == 5)


def test_reductions() raises:
    var xs = Batch[Integer]([3, -1, 7, 7, 2])
    assert_equal(amax(xs), 7)
    assert_equal(amin(xs), -1)
    assert_equal(argmax(xs), 2)
    assert_equal(argmin(xs), 1)
    var matrix = Batch[Integer]([1, 9, 3, 8, 2, 7], shape=[2, 3])
    assert_equal(argmax(matrix), 1)
    assert_equal(argmax(matrix, axis=1).to_list(), [Integer(1), Integer(0)])
    assert_equal(argmin(matrix, axis=0).to_list(), [Integer(0), Integer(1), Integer(0)])
    assert_equal(argmin(matrix, axis=-1, keepdims=True).shape(), [2, 1])
    assert_equal(amax(matrix, axis=0).to_list(), [Integer(8), Integer(9), Integer(7)])
    var qs = Batch[Rational]([Rational(1, 2), Rational(-1, 3), Rational(2, 3)])
    assert_equal(amax(qs), Rational(2, 3))
    assert_equal(argmin(qs), 1)
    var c = _c()
    var fs = Batch[Float]([Float(1, context=c), Float(5, context=c), Float(-2, context=c)])
    assert_equal(amax(fs), Float(5, context=c))
    assert_equal(amin(fs), Float(-2, context=c))
    assert_equal(argmax(fs), 1)
    var with_nan = Batch[Float]([Float(1, context=c), Float.nan(context=c), Float(5, context=c), Float.nan(context=c)])
    assert_true(amax(with_nan).is_nan() and amin(with_nan).is_nan())
    assert_equal(argmax(with_nan), 1)
    assert_equal(argmin(with_nan), 1)
    var zeros = Batch[Float]([Float.zero(negative=True, context=c), Float.zero(context=c)])
    assert_true(amax(zeros).signbit())
    var balls = Batch[Ball]([Ball(1), Ball(3, Rational(1, 2)), Ball(2)])
    var top = amax(balls)
    assert_true(top.lower_rational() <= Rational(5, 2) and top.upper_rational() >= Rational(7, 2))
    assert_true(amin(balls).midpoint() == 1)
    with assert_raises(contains="Cannot compute max of an empty Float"):
        _ = amax(Batch[Float]())
    with assert_raises(contains="Cannot compute argmin of an empty"):
        _ = argmin(Batch[Integer]())


def test_rounding_and_reciprocal() raises:
    assert_equal(floor(Float("-2.5")), -3)
    assert_equal(ceil(Float("-2.5")), -2)
    assert_equal(trunc(Float("-2.5")), -2)
    assert_equal(round(Float("-2.5")), -2)
    assert_equal(round(Float("3.5")), 4)
    assert_equal(floor(Rational(-7, 2)), -4)
    assert_equal(round(Rational(5, 2)), 2)
    assert_equal(floor(Integer(5)), 5)
    assert_equal(round(7), 7)
    with assert_raises():
        _ = floor(Float.nan())
    assert_equal(reciprocal(Integer(4)), Rational(1, 4))
    assert_equal(reciprocal(Rational(-2, 3)), Rational(-3, 2))
    assert_equal(reciprocal(Float(4, context=_c())), Float("0.25", context=_c()))
    with assert_raises():
        _ = reciprocal(Integer(0))
    assert_equal(imag(reciprocal(Complex(0, 2))), Float(Rational(-1, 2)))
    assert_equal(reciprocal(ExactComplex(Rational(0), Rational(2))), ExactComplex(Rational(0), Rational(-1, 2)))
    assert_true(reciprocal(Ball(4)).midpoint() == Rational(1, 4))
    assert_true(reciprocal(Ball(0, Rational(1, 10))).is_indeterminate())
    assert_true(Rational(4, 2).is_integer() and not Rational(1, 2).is_integer())
    assert_true(Float("3").is_integer() and not Float("2.5").is_integer())
    assert_true(Float("0x1p100").is_integer() and not Float("0x3p-1").is_integer() and not Float("0x1p-100").is_integer())
    assert_true(Float.zero().is_integer() and not Float.infinity().is_integer() and not Float.nan().is_integer())


def test_complex_parts() raises:
    var z = Complex(3, -4)
    assert_equal(abs(z), Float(5))
    assert_equal(real(z), Float(3))
    assert_equal(imag(z), Float(-4))
    assert_equal(conjugate(z), Complex(3, 4))
    assert_true(angle(Complex(-1, 0)) > Float(3))
    var e = ExactComplex(Rational(1, 2), Rational(-3))
    assert_equal(real(e), Rational(1, 2))
    assert_equal(imag(e), Rational(-3))
    assert_equal(conjugate(e), ExactComplex(Rational(1, 2), Rational(3)))
    var b = ComplexBall(1, 2)
    assert_true(real(b).midpoint() == 1 and imag(b).midpoint() == 2 and imag(conjugate(b)).midpoint() == -2)
    var size = abs(ComplexBall(3, 4))
    assert_true(size.lower_rational() <= 5 and size.upper_rational() >= 5)


def test_integer_and_float_names() raises:
    assert_equal(comb(5, 2), 10)
    assert_equal(comb(5, 2, repetition=True), 15)
    # As scipy's exact comb: zero outside 0 <= k <= N.
    assert_equal(comb(-3, 3), 0)
    assert_equal(comb(10, -1), 0)
    assert_equal(comb(3, 5), 0)
    assert_equal(factorial2(7), 105)
    assert_equal(ldexp(Float(3), Integer(-1)), Float(Rational(3, 2)))
    var c = _c()
    var one = Float(1, context=c)
    assert_equal(Rational(nextafter(one, 2)) - 1, Rational(1, Integer(1) << 52))
    assert_equal(Rational(1) - Rational(nextafter(one, Rational(1, 2))), Rational(1, Integer(1) << 53))
    assert_true(nextafter(one, 1) == one)
    assert_true(not nextafter(Float.zero(negative=True, context=c), Float.zero(context=c)).signbit())
    assert_true(nextafter(one, Float.nan()).is_nan())
    assert_equal(Rational(spacing(Float(-1, context=c))), -Rational(1, Integer(1) << 52))
    assert_true(spacing(Float.nan()).is_nan())


def test_vmap_family_functions() raises:
    var xs = Batch[Integer]([1, 5, -2])
    var ys = Batch[Integer]([3, 4, -7])
    assert_equal(vmap[integer.maximum]()(xs, ys).to_list(), [Integer(3), Integer(5), Integer(-2)])
    assert_equal(vmap[integer.minimum]()(xs, ys).to_list(), [Integer(1), Integer(4), Integer(-7)])
    assert_equal(vmap[integer.clip]()(xs, Integer(0), Integer(4)).to_list(), [Integer(1), Integer(4), Integer(0)])
    assert_equal(vmap[rational.reciprocal]()(Batch[Rational]([Rational(2), Rational(-1, 3)])).to_list(), [Rational(1, 2), Rational(-3)])
    var c = _c()
    var fs = Batch[Float]([Float(1, context=c), Float.nan(context=c), Float(3, context=c)])
    var top = vmap[float.maximum]()(fs, Float(2, context=c))
    assert_true(top[0] == Float(2, context=c) and top[1].is_nan() and top[2] == Float(3, context=c))
    assert_equal(vmap[float.floor]()(Batch[Float]([Float("2.5"), Float("-2.5")])).to_list(), [Integer(2), Integer(-3)])
    var integrals = vmap[float.sici]()(Batch[Float]([Float(1, context=c), Float(2, context=c)]))
    var pair = float.sici(Float(2, context=c))
    assert_true(integrals[0][1] == pair[0] and integrals[1][1] == pair[1])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
