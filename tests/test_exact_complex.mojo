"""Exact complex numbers (T-EXC): exact arithmetic and powers, principal
square roots, text, JSON, hashing and conversion."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import Integer, Rational, Float, Complex, ComplexContext, ArithmeticContext, FloatFormat, ExactComplex
from apn_mojo import exact_complex
from apn_mojo import stable_hash


def test_arithmetic() raises:
    var z = ExactComplex(1, 1)
    assert_true(exact_complex.pow_int(z, Integer(10)) == ExactComplex(0, 32))
    assert_true(Rational(1) / z == ExactComplex(Rational(1, 2), Rational(-1, 2)))
    assert_true(z * z == ExactComplex(0, 2))
    assert_true(z + 2 == ExactComplex(3, 1) and 2 - z == ExactComplex(1, -1))
    assert_true(z * Rational(1, 2) == ExactComplex(Rational(1, 2), Rational(1, 2)))
    assert_true(-z == ExactComplex(-1, -1) and +z == z)
    assert_true(exact_complex.conjugate(z) == ExactComplex(1, -1))
    assert_true(exact_complex.norm_sqr(ExactComplex(3, 4)) == Rational(25))
    assert_true(exact_complex.divide(ExactComplex(1, 2), ExactComplex(3, 4)) == ExactComplex(Rational(11, 25), Rational(2, 25)))
    assert_true(exact_complex.pow_int(ExactComplex(0), Integer(0)) == ExactComplex(1))
    assert_true(exact_complex.pow_int(ExactComplex(1, 2), Integer(-2)) == ExactComplex(Rational(-3, 25), Rational(-4, 25)))
    with assert_raises(contains="division by zero"):
        _ = z / ExactComplex(0)
    with assert_raises(contains="division by zero"):
        _ = exact_complex.pow_int(ExactComplex(0), Integer(-1))
    assert_true(ExactComplex(5).is_real() and not z.is_real())
    assert_true(z.real() == Rational(1) and z.imag() == Rational(1))


def test_square_roots() raises:
    var two_i = exact_complex.sqrt_exact(ExactComplex(0, 2))
    assert_true(two_i.value() == ExactComplex(1, 1))
    assert_true(exact_complex.sqrt_exact(ExactComplex(3, 4)).value() == ExactComplex(2, 1))
    assert_true(exact_complex.sqrt_exact(ExactComplex(-4)).value() == ExactComplex(0, 2))
    assert_false(Bool(exact_complex.sqrt_exact(ExactComplex(-2))))
    assert_false(Bool(exact_complex.sqrt_exact(ExactComplex(0, 1))))
    assert_true(exact_complex.sqrt_exact(ExactComplex(3, -4)).value() == ExactComplex(2, -1))
    assert_true(exact_complex.sqrt_exact(ExactComplex(Rational(9, 4))).value() == ExactComplex(Rational(3, 2)))
    # Every principal root squares back.
    for a in range(-6, 7):
        for b in range(-6, 7):
            var w = ExactComplex(a, b)
            var square = w * w
            var root = exact_complex.sqrt_exact(square)
            assert_true(Bool(root) and root.value() * root.value() == square)
            var r = root.value()
            assert_true(r.real().sign() > 0 or (r.real().sign() == 0 and r.imag().sign() >= 0))


def test_text_json_and_hashing() raises:
    var z = ExactComplex(Rational(1, 2), 3)
    assert_equal(String(z), "ExactComplex(1/2, 3)")
    assert_true(ExactComplex("ExactComplex(1/2, 3)") == z)
    assert_true(ExactComplex(" ExactComplex(-7/3, 0) ") == ExactComplex(Rational(-7, 3)))
    with assert_raises(contains="ExactComplex(re, im)"):
        _ = ExactComplex("1 + 2i")
    var record = z.to_json()
    assert_equal(
        record,
        '{"version":1,"family":"exact_complex","real":{"version":1,"family":"rational","numerator":"1","denominator":"2"},'
        + '"imag":{"version":1,"family":"rational","numerator":"3","denominator":"1"}}',
    )
    assert_true(ExactComplex.from_json(record) == z)
    var swapped = '{"family":"exact_complex","imag":{"version":1,"family":"rational","numerator":"3","denominator":"1"},'
    swapped += '"real":{"version":1,"family":"rational","numerator":"1","denominator":"2"},"version":1}'
    assert_true(ExactComplex.from_json(swapped) == z)
    with assert_raises(contains="noncanonical"):
        _ = ExactComplex.from_json('{"version":1,"family":"exact_complex","real":{"version":1,"family":"rational","numerator":"2","denominator":"4"},"imag":{"version":1,"family":"rational","numerator":"0","denominator":"1"}}')
    with assert_raises(contains="exact_complex"):
        _ = ExactComplex.from_json('{"version":1,"family":"complex","real":{},"imag":{}}')
    # Appendix F: 1 + 2i hashes to 0xd3f2712260baa8ff; equal values hash equal.
    assert_equal(exact_complex.stable_hash(ExactComplex(1, 2)), 0xD3F2712260BAA8FF)
    assert_equal(stable_hash(ExactComplex(1, 2)), 0xD3F2712260BAA8FF)
    assert_equal(hash(ExactComplex(Rational(2, 4), 1)), hash(ExactComplex(Rational(1, 2), 1)))
    var table = Dict[ExactComplex, Int]()
    table[ExactComplex(1, 1)] = 7
    assert_equal(table[ExactComplex(Rational(2, 2), 1)], 7)


def test_conversion() raises:
    var z = ExactComplex(Rational(1, 3), -2)
    var c53 = ArithmeticContext(format=FloatFormat(53))
    var converted = z.to_complex(context=ComplexContext(c53))
    assert_true(converted.real() == Float(Rational(1, 3), context=c53) and converted.imag() == Float(-2, context=c53))
    assert_equal(z.to_complex().real().precision(), 128)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
