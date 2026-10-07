"""Check the result types chosen by actual public arithmetic expressions."""

from std.testing import TestSuite, assert_equal
from apn_mojo import Batch, Integer, Rational, Float, Complex


def _check_pair[
    A: ImplicitlyCopyable & Deinitable,
    B: ImplicitlyCopyable & Deinitable,
    Sum: ImplicitlyCopyable & Deinitable,
    Quotient: ImplicitlyCopyable & Deinitable,
](a: Batch[A], b: Batch[B]) raises:
    var added = a + b
    var subtracted = a - b
    var multiplied = a * b
    var divided = a / b
    comptime assert type_of(added) == Batch[Sum]
    comptime assert type_of(subtracted) == Batch[Sum]
    comptime assert type_of(multiplied) == Batch[Sum]
    comptime assert type_of(divided) == Batch[Quotient]
    assert_equal(added.shape(), [2, 2])
    assert_equal(subtracted.shape(), [2, 2])
    assert_equal(multiplied.shape(), [2, 2])
    assert_equal(divided.shape(), [2, 2])


def test_batch_operators_choose_result_families() raises:
    var integers = Batch[Integer]([2, 3]).reshape([2, 1])
    var rationals = Batch[Rational]([Rational(2), Rational(3)]).reshape([2, 1])
    var floats = Batch[Float]([Float(2), Float(3)]).reshape([2, 1])
    var complex = Batch[Complex]([Complex(2), Complex(3)]).reshape([2, 1])
    _check_pair[Integer, Integer, Integer, Rational](integers, integers.transpose())
    _check_pair[Rational, Rational, Rational, Rational](rationals, rationals.transpose())
    _check_pair[Float, Float, Float, Float](floats, floats.transpose())
    _check_pair[Complex, Complex, Complex, Complex](complex, complex.transpose())
    _check_pair[Integer, Rational, Rational, Rational](integers, rationals.transpose())
    _check_pair[Rational, Integer, Rational, Rational](rationals, integers.transpose())
    _check_pair[Integer, Float, Float, Float](integers, floats.transpose())
    _check_pair[Float, Integer, Float, Float](floats, integers.transpose())
    _check_pair[Integer, Complex, Complex, Complex](integers, complex.transpose())
    _check_pair[Complex, Integer, Complex, Complex](complex, integers.transpose())
    _check_pair[Rational, Float, Float, Float](rationals, floats.transpose())
    _check_pair[Float, Rational, Float, Float](floats, rationals.transpose())
    _check_pair[Rational, Complex, Complex, Complex](rationals, complex.transpose())
    _check_pair[Complex, Rational, Complex, Complex](complex, rationals.transpose())
    _check_pair[Float, Complex, Complex, Complex](floats, complex.transpose())
    _check_pair[Complex, Float, Complex, Complex](complex, floats.transpose())


def test_native_and_wide_literal_results_preserve_exactness() raises:
    var integers = Batch[Integer]([1, 2])
    var unsigned = integers + UInt64.MAX
    var literal = 340282366920938463463374607431768211456 + integers
    var quotient = integers / 2
    var real = Float64(0.5) + integers
    var complex = integers + Complex(0, 1)
    comptime assert type_of(unsigned) == Batch[Integer]
    comptime assert type_of(literal) == Batch[Integer]
    comptime assert type_of(quotient) == Batch[Rational]
    comptime assert type_of(real) == Batch[Float]
    comptime assert type_of(complex) == Batch[Complex]
    assert_equal(unsigned[0], Integer(1) << 64)
    assert_equal(literal[1], (Integer(1) << 128) + 2)
    assert_equal(quotient[0], Rational(1, 2))
    assert_equal(real[0], Float(Rational(3, 2)))
    assert_equal(complex[0], Complex(1, 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
