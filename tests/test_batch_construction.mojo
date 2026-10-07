"""Iterator construction preserves values, ownership and failure boundaries."""

from std.iter import iter
from std.testing import TestSuite, assert_equal, assert_raises, assert_true
from apn_mojo import Batch, Integer, Rational, Float, Complex, Ball, ArithmeticContext, FloatFormat, RoundingMode
from apn_mojo.rational._batch_storage import _native_rational_iterator
from apn_mojo.complex._batch_storage import _native_complex_iterator


def test_iterator_conversions_keep_unsigned_width_and_empty_shape() raises:
    var native: List[UInt64] = [UInt64.MAX, 0]
    assert_equal(Batch[Integer](native)[0], Integer(UInt64.MAX))
    assert_equal(Batch[Rational](native)[0], Rational(Integer(UInt64.MAX)))
    assert_equal(Batch[Float](native)[0], Float(UInt64.MAX))
    assert_equal(Batch[Complex](native^)[0], Complex(UInt64.MAX))
    var empty = List[Int]()
    assert_equal(Batch[Integer](empty).shape(), [0])
    assert_equal(Batch[Rational](empty).shape(), [0])
    assert_equal(Batch[Float](empty).shape(), [0])
    assert_equal(Batch[Complex](empty^).shape(), [0])
    assert_equal(Batch[Ball](List[Ball]()).shape(), [0])


def test_iterator_failures_release_partial_results_and_keep_sources() raises:
    var wide = (Integer(1) << 200) + 3
    var rationals = [Rational(wide, 7), Rational(2), Rational(-wide, 11)]
    var complex = [Complex(wide, -wide), Complex(2), Complex(-wide, wide)]
    var rational_owners = rationals[0]._numerator._storage[Integer._Shared].count()
    var complex_owners = complex[0]._real._significand._storage[Integer._Shared].count()
    for index in range(3):
        with assert_raises(contains=String("Injected Rational transfer failure at element ", index)):
            _ = _native_rational_iterator(iter(rationals), index)
        with assert_raises(contains=String("Injected Complex conversion failure at element ", index)):
            _ = _native_complex_iterator(iter(complex), index)
        var owned_rationals = rationals.copy()
        var owned_complex = complex.copy()
        with assert_raises(contains=String("Injected Rational transfer failure at element ", index)):
            _ = _native_rational_iterator(iter(owned_rationals^), index)
        with assert_raises(contains=String("Injected Complex conversion failure at element ", index)):
            _ = _native_complex_iterator(iter(owned_complex^), index)
        assert_equal(rationals[0]._numerator._storage[Integer._Shared].count(), rational_owners)
        assert_equal(complex[0]._real._significand._storage[Integer._Shared].count(), complex_owners)
    assert_equal(_native_rational_iterator(iter(rationals), 3)._read(2), rationals[2])
    assert_equal(_native_complex_iterator(iter(complex), 3)._read(2), complex[2])
    var expected_rational = rationals[0]
    var expected_complex = complex[0]
    var rational_batch = Batch[Rational](rationals^)
    var complex_batch = Batch[Complex](complex^)
    assert_equal(rational_batch[0], expected_rational)
    assert_equal(complex_batch[0], expected_complex)


def test_to_native() raises:
    var c = ArithmeticContext(format=FloatFormat.binary64())
    # Integers: exact to integer types, rounded once to floating-point ones.
    var ints = Batch[Integer]([Integer(1) << 60, -7]).reshape([2])
    assert_equal(ints.to_native[DType.int64](), [Int64(1) << 60, Int64(-7)])
    assert_equal(Batch[Integer]([(Integer(1) << 60) + 1]).to_native[DType.float64](), [Float64(1152921504606846976.0)])
    with assert_raises():
        _ = Batch[Integer]([300]).to_native[DType.int8]()
    # Rationals: one rounding, subnormals included; whole values to integer types.
    var thirds = Batch[Rational]([Rational(1, 3), Rational(-2, 3), Rational(0)])
    assert_equal(thirds.to_native[DType.float64](), [Float64(1) / 3, Float64(-2) / 3, Float64(0)])
    # The nearest Float64 to 1/10 lies above it, to 1/3 below it.
    var tenth = Batch[Rational]([Rational(1, 10)])
    assert_true(tenth.to_native[DType.float64](rounding=RoundingMode.toward_zero)[0] < Float64(0.1))
    assert_true(thirds.to_native[DType.float64](rounding=RoundingMode.toward_positive)[0] > Float64(1) / 3)
    var tiny = Batch[Rational]([Rational(Integer(3), Integer(1) << 1076)])
    assert_equal(tiny.to_native[DType.float64](), [Float64(5e-324)])
    assert_equal(Batch[Rational]([Rational(4, 2)]).to_native[DType.int32](), [Int32(2)])
    with assert_raises():
        _ = Batch[Rational]([Rational(1, 2)]).to_native[DType.int32]()
    # Floats and balls; row-major order of any view.
    var floats = Batch[Float]([Float(1, context=c), Float("2.5", context=c), Float(3, context=c), Float(4, context=c)]).reshape([2, 2])
    assert_equal(floats.to_native[DType.float32](), [Float32(1), Float32(2.5), Float32(3), Float32(4)])
    assert_equal(floats.transpose([1, 0]).to_native[DType.float64](), [Float64(1), Float64(3), Float64(2.5), Float64(4)])
    with assert_raises():
        _ = floats.to_native[DType.int64]()
    assert_equal(Batch[Float]([Float(3, context=c)]).to_native[DType.int64](), [Int64(3)])
    assert_equal(Batch[Ball]([Ball(Float("0.1", context=c))]).to_native[DType.float64](), [Float64(0.1)])
    var native: List[Float64] = [0.1, -2.0, 1e300]
    assert_equal(Batch[Float].from_native(native).to_native[DType.float64](), native)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
