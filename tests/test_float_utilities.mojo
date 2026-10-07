"""Float utilities: machine formats, exact Rational conversion, half-even
rounding, neighbours and ulps, and the shortest round-trip decimal, checked
against tests/fixtures/shortest_decimal.txt (written by Python's repr); decimal
conversions at large exponents against tests/fixtures/large_exponent_decimal.txt
(written from Arb's balls)."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    ArithmeticContext,
    Batch,
    Float,
    FloatFormat,
    Integer,
    Rational,
    equal_at_precision,
    nextafter,
    shortest_decimal,
    spacing,
    ulp_distance,
    vmap,
    Complex,
    ConversionLimits,
    RoundingMode,
)
from apn_mojo import float
from apn_mojo.float.context import _exact_context


def next_up(x: Float) raises -> Float:
    return nextafter(x, Float.infinity())


def next_down(x: Float) raises -> Float:
    return nextafter(x, Float.infinity(negative=True))


def _binary64() raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat.binary64())


def _same(a: Float, b: Float) raises:
    if not a.same_representation(b):
        raise Error(String("Expected ", b, " in its format, got ", a))


def test_formats() raises:
    var double = FloatFormat.binary64()
    assert_equal(double.precision(), 53)
    assert_equal(double.emin(), -1021)
    assert_equal(double.emax(), 1024)
    var single = FloatFormat.binary32()
    assert_equal(single.precision(), 24)
    assert_equal(single.emin(), -125)
    assert_equal(single.emax(), 128)


def test_rational_conversion() raises:
    var tenth = Float("0.1", context=_binary64())
    assert_equal(tenth.to_rational_exact(), Rational(3602879701896397, Integer(1) << 55))
    assert_equal(Rational(tenth), Rational(3602879701896397, Integer(1) << 55))
    assert_equal(Rational(Float(-3, context=_binary64())), Rational(-3))
    assert_equal(Float.zero(negative=True).to_rational_exact(), Rational(0))
    assert_equal(Rational(Float("0x3p100")), Rational(Integer(3) << 100))
    assert_equal(Rational(Float("-0x3p-1074")), Rational(-3, Integer(1) << 1074))
    with assert_raises(contains="infinity or NaN"):
        _ = Float.infinity().to_rational_exact()
    with assert_raises(contains="infinity or NaN"):
        _ = Rational(Float.nan())
    # Back to the same format, the value is exact.
    var texts: List[String] = ["1e-300", "-123.456", "7e22", "0.3"]
    for text in texts:
        var x = Float(text, context=_binary64())
        _same(Float(x.to_rational_exact(), context=_binary64()), x)


def test_round_half_even() raises:
    var texts: List[String] = ["2.5", "3.5", "-2.5", "0.5", "-0.5", "1.5", "0.49", "0.51", "-7.5", "1e30", "-0"]
    var expected: List[Integer] = [2, 4, -2, 0, 0, 2, 0, 1, -8, 1000000000000000019884624838656, 0]
    for i in range(len(texts)):
        assert_equal(Float(texts[i], context=_binary64()).round(), expected[i])
    var big = Float(Rational((Integer(1) << 60) * 2 + 1, 2), context=ArithmeticContext(format=FloatFormat(128)))
    assert_equal(big.round(), Integer(1) << 60)
    assert_equal(Float("0x1p-100").round(), 0)
    assert_equal(Float("0x3p-1").round(), 2)
    with assert_raises(contains="nonfinite"):
        _ = Float.nan().round()


def test_neighbors() raises:
    var c = _binary64()
    var smallest = Float("0x1p-1022", context=c)
    var largest = Float("0x1fffffffffffffp971", context=c)
    var one = Float(1, context=c)
    _same(next_up(Float.zero(context=c)), smallest)
    _same(next_up(Float.zero(negative=True, context=c)), smallest)
    _same(next_up(-smallest), Float.zero(negative=True, context=c))
    _same(next_up(largest), Float.infinity(context=c))
    _same(next_up(Float.infinity(context=c)), Float.infinity(context=c))
    _same(next_up(Float.infinity(negative=True, context=c)), -largest)
    assert_true(next_up(Float.nan(context=c)).is_nan())
    assert_equal(Rational(next_up(one)) - 1, Rational(1, Integer(1) << 52))
    assert_equal(Rational(1) - Rational(next_down(one)), Rational(1, Integer(1) << 53))
    _same(next_down(Float.zero(context=c)), -smallest)
    var texts: List[String] = ["1", "-1", "0.1", "1e300", "-3e-300", "7", "0x1p-1021"]
    for text in texts:
        var x = Float(text, context=c)
        _same(next_down(next_up(x)), x)
        _same(next_up(next_down(x)), x)
    var tiny = Float("0x1p-3", context=ArithmeticContext(format=FloatFormat(1)))
    _same(next_up(tiny), Float("0x1p-2", context=ArithmeticContext(format=FloatFormat(1))))
    # spacing is the signed distance at x, in x's format (numpy's spacing).
    _same(spacing(one), Float("0x1p-52", context=c))
    _same(spacing(Float("1.5", context=c)), Float("0x1p-52", context=c))
    _same(spacing(-largest), -Float("0x1p971", context=c))
    _same(spacing(Float.zero(context=c)), smallest)
    with assert_raises(contains="below the smallest"):
        _ = spacing(smallest)
    assert_true(spacing(Float.infinity(context=c)).is_nan())
    # Distances in steps of the coarser format.
    assert_equal(ulp_distance(one, Float("1.0000000000000002", context=c)), 1)
    assert_equal(ulp_distance(Float.zero(context=c), Float.zero(negative=True, context=c)), 0)
    assert_equal(ulp_distance(largest, Float.infinity(context=c)), 1)
    assert_equal(ulp_distance(-smallest, smallest), 2)
    assert_equal(ulp_distance(Float.infinity(negative=True, context=c), Float.infinity(context=c)), (Integer(2046) << 53) + 2)
    var wide = Float("1.0000000000000002", context=ArithmeticContext(format=FloatFormat(128, emin=-1021, emax=1024)))
    assert_equal(ulp_distance(one, wide), 1)
    with assert_raises(contains="NaN"):
        _ = ulp_distance(one, Float.nan(context=c))
    with assert_raises(contains="exponent bounds"):
        _ = ulp_distance(one, Float(1))
    assert_true(equal_at_precision(one, Float("1.0000000000000002", context=c), 52))
    assert_true(not equal_at_precision(one, Float("1.0000000000000002", context=c), 53))
    assert_true(equal_at_precision(Float.zero(context=c), Float.zero(negative=True), 10))
    with assert_raises(contains="NaN"):
        _ = equal_at_precision(one, Float.nan(), 10)


def test_shortest_decimal_against_repr() raises:
    var c = _binary64()
    var text: String
    with open("tests/fixtures/shortest_decimal.txt", "r") as source:
        text = source.read()
    var count = 0
    for row in text.split("\n"):
        if not row.byte_length():
            continue
        var fields = row.split(" ")
        var x = Float(String(fields[0]), context=c)
        var found = shortest_decimal(x)
        if found.digits() != String(fields[1]) or found.exponent10() != Int(String(fields[2])):
            raise Error(String("shortest_decimal(", fields[0], ") gave ", found, ", repr gives ", fields[1], "e", fields[2]))
        assert_equal(found.negative(), String(fields[0]).startswith("-"))
        _same(Float(String(found), context=c), x)
        count += 1
    assert_equal(count, 6000)
    var pi = shortest_decimal(Float("3.141592653589793", context=c))
    assert_equal(pi.digits(), "3141592653589793")
    assert_equal(pi.exponent10(), -15)
    # Without subnormals, everything from half the smallest value rounds to it.
    assert_equal(String(shortest_decimal(Float("0x1p-1022", context=c))), "2e-308")
    _same(Float("2e-308", context=c), Float("0x1p-1022", context=c))
    assert_equal(String(shortest_decimal(Float.zero(negative=True))), "-0e0")
    with assert_raises(contains="infinity or NaN"):
        _ = shortest_decimal(Float.nan())


def test_shortest_decimal_any_precision() raises:
    # In every precision the decimal reads back as x, and no decimal with one
    # digit fewer near x does.
    var state = UInt64(88172645463325252)
    for precision in [1, 2, 7, 24, 64, 113, 200, 1000]:
        var context = ArithmeticContext(format=FloatFormat(precision))
        for k in range(40):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var x = Float(Rational(Integer(state) * Integer(state | 1), Integer(1) << (k * 7)), context=context)
            if k % 3 == 1:
                x = -x
            var found = shortest_decimal(x)
            _same(Float(String(found), context=context), x)
            var digits = Integer.parse(found.digits())
            if digits >= 10:
                for shorter in [digits // 10, digits // 10 + 1]:
                    var text = String("-" if found.negative() else "", shorter, "e", found.exponent10() + 1)
                    assert_true(not Float(text, context=context).same_representation(x))


def test_float_utilities_maps() raises:
    var c = _binary64()
    var xs = Batch[Float]([Float(1, context=c), Float("0.1", context=c), Float(-2, context=c)])
    var ups = vmap[float.nextafter]()(xs, Float.infinity())
    assert_equal(vmap[float.ulp_distance]()(xs, ups).to_list(), [1, 1, 1])
    assert_equal(vmap[float.equal_at_precision]()(xs, ups, 20).to_list(), [True, True, True])
    assert_equal(vmap[float.equal_at_precision]()(xs, ups, 53).to_list(), [False, False, False])


def test_decimal_printing() raises:
    var c = _binary64()
    # print and to_string(): the shortest decimal, positional from 1e-6 up to 1e21.
    assert_equal(String(Float("0.1", context=c)), "0.1")
    assert_equal(String(Float("0.00001", context=c)), "0.00001")
    assert_equal(String(Float("0.000001", context=c)), "0.000001")
    assert_equal(String(Float("1e-7", context=c)), "1e-07")
    assert_equal(String(Float("1.5e-8", context=c)), "1.5e-08")
    assert_equal(String(Float(3, context=c)), "3.0")
    assert_equal(String(Float("123456.789", context=c)), "123456.789")
    assert_equal(String(Float("-2.5", context=c)), "-2.5")
    assert_equal(String(Float(Integer(1) << 70, context=c)), "1.1805916207174113e+21")
    assert_equal(String(Float("1e21", context=c)), "1e+21")
    assert_equal(String(Float("1e20", context=c)), "100000000000000000000.0")
    assert_equal(String(Float.nan(context=c)), "nan")
    assert_equal(String(Float.infinity(negative=True, context=c)), "-inf")
    assert_equal(String(-Float(0, context=c)), "-0.0")
    var x = Float("0.00001", context=c)
    assert_equal(x.to_string(), String(x))
    # Notations and digits.
    assert_equal(x.to_string(notation="scientific"), "1e-05")
    assert_equal(x.to_string(notation="hexadecimal"), x.to_string(16))
    assert_equal(Float("1e21", context=c).to_string(notation="positional"), "1000000000000000000000.0")
    assert_equal(Float(1, context=c).to_string(digits=3), "1.00")
    assert_equal(Float(1, context=c).to_string(digits=3, notation="scientific"), "1.00e+00")
    assert_equal(Float("0.125", context=c).to_string(digits=2), "0.12")
    with assert_raises():
        _ = x.to_string(notation="engineering")
    with assert_raises():
        _ = x.to_string(16, notation="scientific")
    # The decimal reads back as the same Float in its format.
    assert_equal(Float(x.to_string(), context=c), x)
    assert_equal(Float(Float("1.1", context=c).to_string(), context=c), Float("1.1", context=c))
    # Complex numbers and batches print their parts the same way.
    assert_equal(String(Complex(Float("0.5", context=c), Float(-2, context=c))), "Complex(0.5, -2.0)")
    assert_equal(Complex(Float("0.5", context=c), Float(-2, context=c)).to_string(notation="scientific"), "Complex(5e-01, -2e+00)")
    assert_equal(String(Batch[Float]([Float("0.25", context=c), Float(1, context=c)])), "[0.25, 1.0]")


def _mode(name: String) raises -> RoundingMode:
    if name == "nearest_even":
        return RoundingMode.nearest_even
    if name == "toward_zero":
        return RoundingMode.toward_zero
    if name == "toward_positive":
        return RoundingMode.toward_positive
    if name == "toward_negative":
        return RoundingMode.toward_negative
    assert_equal(name, "away_from_zero")
    return RoundingMode.away_from_zero


def test_large_exponent_decimal() raises:
    # Exponents from 60000 bits (the exact path) up to 2**61 (bounds): the
    # shortest decimal, fixed digits in every rounding mode, and parsing, with
    # round trips of each printed value.
    var text: String
    with open("tests/fixtures/large_exponent_decimal.txt", "r") as source:
        text = source.read()
    var count = 0
    for row in text.split("\n"):
        if not row.byte_length():
            continue
        var fields = row.split(" ")
        var kind = String(fields[0])
        var precision = Int(String(fields[1]))
        var c = ArithmeticContext(format=FloatFormat(precision))
        if kind == "shortest":
            var x = Float(String(fields[2]), context=c)
            var found = shortest_decimal(x)
            if found.digits() != String(fields[3]) or found.exponent10() != Int(String(fields[4])):
                raise Error(String("shortest_decimal(", fields[2], ") gave ", found, ", Arb gives ", fields[3], "e", fields[4]))
            _same(Float(String(found), context=c), x)
            _same(Float(String(x), context=c), x)
        elif kind == "digits":
            var x = Float(String(fields[2]), context=c)
            var found = x.to_string(digits=Int(String(fields[3])), rounding=_mode(String(fields[4])), notation="scientific")
            if found != String(fields[5]):
                raise Error(String(fields[2], " to ", fields[3], " digits ", fields[4], " gave ", found, ", Arb gives ", fields[5]))
        else:
            assert_equal(kind, "parse")
            var read = ArithmeticContext(format=FloatFormat(precision), rounding=_mode(String(fields[2])))
            var found = Float(String(fields[3]), context=read)
            if not found.same_representation(Float(String(fields[4]), context=c)):
                raise Error(String(fields[3], " at ", precision, " bits ", fields[2], " gave ", found.to_string(16), ", Arb gives ", fields[4]))
        count += 1
    assert_equal(count, 1043)
    # The ends of the default exponent range print and read back.
    for literal in ["0x1p4611686018427387000", "-0x3p-4611686018427387000"]:
        var x = Float(literal)
        _same(Float(String(x)), x)


def test_conversion_budgets() raises:
    # shortest_decimal's exact work counts against max_allocated_bytes.
    var x = Float("0x1p60000", context=ArithmeticContext(format=FloatFormat(53)))
    with assert_raises(contains="max_allocated_bytes"):
        _ = shortest_decimal(x, limits=ConversionLimits(max_allocated_bytes=1000))
    _same(Float(String(shortest_decimal(x)), context=ArithmeticContext(format=FloatFormat(53))), x)
    # Output limits count each byte once.
    assert_equal(Float("0.1").to_string(limits=ConversionLimits(max_output_bytes=3)), "0.1")
    with assert_raises(contains="max_output_bytes"):
        _ = Float("0.1").to_string(limits=ConversionLimits(max_output_bytes=2))
    assert_equal(Float(1).to_string(digits=3, limits=ConversionLimits(max_output_bytes=4)), "1.00")
    assert_equal(Float("0.00001").to_string(limits=ConversionLimits(max_output_bytes=7)), "0.00001")
    # One Complex value: one budget for both parts and the framing.
    var z = Complex(Float("0.1"), Float("0.2"))
    assert_equal(z.to_string(limits=ConversionLimits(max_output_bytes=17)), "Complex(0.1, 0.2)")
    with assert_raises(contains="max_output_bytes"):
        _ = z.to_string(limits=ConversionLimits(max_output_bytes=16))
    with assert_raises(contains="Hexadecimal notation writes base 16"):
        _ = z.to_string(2, notation="hexadecimal")
    # In an exact format, 5**12 cannot divide one digit: inexact at once,
    # without forming 10**(10**12).
    with assert_raises(contains="exactly"):
        _ = Float("1e-1000000000000", context=_exact_context())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
