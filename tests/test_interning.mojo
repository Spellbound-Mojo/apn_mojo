"""Interning: representation identity and order, keys, and stable hashes."""

from std.testing import TestSuite, assert_equal, assert_true
from apn_mojo import (
    ArithmeticContext,
    Complex,
    ComplexKey,
    Float,
    FloatFormat,
    FloatKey,
    Integer,
    Rational,
    add,
    stable_hash,
)
from apn_mojo import complex, float, integer, rational


def _context(precision: Int) raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat(precision))


def _binary64() raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat(53, emin=-1021, emax=1024))


def test_stable_hash_vectors() raises:
    # The vectors of the requirements' Appendix F.4, from its Python reference.
    var p53 = _context(53)
    assert_equal(integer.stable_hash(0), 0x1389514056E86A8C)
    assert_equal(integer.stable_hash(1), 0xAA06DF08BA932F11)
    assert_equal(integer.stable_hash(-1), 0x62CC5C6C1010DFDD)
    assert_equal(integer.stable_hash(Integer(1) << 64), 0x8F9E62E4CB1CCA4B)
    assert_equal(integer.stable_hash(-((Integer(1) << 128) + 5)), 0xE8E1FE963C96B59F)
    assert_equal(rational.stable_hash(Rational(1, 3)), 0x8ED789C93242A389)
    assert_equal(rational.stable_hash(Rational(-7, 2)), 0x0F5AD03044F82960)
    assert_equal(float.stable_hash(Float.zero(context=p53)), 0x15A578F5F3692D20)
    assert_equal(float.stable_hash(Float.zero(negative=True, context=p53)), 0x9D02448C9F6B3B2D)
    assert_equal(float.stable_hash(Float(1, context=p53)), 0xFE1C360BD17BD38D)
    assert_equal(float.stable_hash(Float(1, context=_context(128))), 0xBDE3E4C5F613EC27)
    assert_equal(float.stable_hash(Float(1, context=_binary64())), 0x712552B5D97BFB71)
    assert_equal(float.stable_hash(Float.infinity(negative=True, context=p53)), 0xF394FAE762F2415E)
    assert_equal(float.stable_hash(Float.nan(context=p53)), 0xC209B16E6B923411)
    var z = Complex(Float(1, context=p53), Float.zero(negative=True, context=p53))
    assert_equal(complex.stable_hash(z), 0x7BCA0003A957C261)
    # The package-level name picks the family.
    assert_equal(stable_hash(1), 0xAA06DF08BA932F11)
    assert_equal(stable_hash(Rational(1, 3)), 0x8ED789C93242A389)
    assert_equal(stable_hash(Float(1, context=p53)), 0xFE1C360BD17BD38D)
    assert_equal(stable_hash(z), 0x7BCA0003A957C261)


def test_equal_values_hash_equal() raises:
    assert_equal(stable_hash(Integer.parse("18446744073709551616")), stable_hash(Integer(1) << 64))
    assert_equal(stable_hash(Integer.parse("-12345678901234567890123")), stable_hash(Integer(-12345678901234567890) * 1000 - 123))
    assert_equal(stable_hash(Rational(2, 4)), stable_hash(Rational(1, 2)))
    assert_equal(stable_hash(Rational(-14, 4)), 0x0F5AD03044F82960)
    assert_true(stable_hash(Rational(1, 2)) != stable_hash(Rational(-1, 2)))
    assert_true(stable_hash(Integer(1) << 64) != stable_hash((Integer(1) << 64) + 1))


def test_same_representation() raises:
    var p53 = _context(53)
    var p128 = _context(128)
    var one = Float(1, context=p53)
    var zero = Float.zero(context=p53)
    var negative_zero = Float.zero(negative=True, context=p53)
    assert_true(zero == negative_zero and not zero.same_representation(negative_zero))
    assert_true(one == Float(1, context=p128) and not one.same_representation(Float(1, context=p128)))
    assert_true(one == Float(1, context=_binary64()) and not one.same_representation(Float(1, context=_binary64())))
    var nan = Float.nan(context=p53)
    assert_true(not (nan == nan) and nan.same_representation(Float.nan(context=p53)))
    assert_true(not nan.same_representation(Float.nan(context=p128)))
    # One representation reached from text, JSON and arithmetic.
    var half = Float("0.5", context=p53)
    var sources = [Float("1", context=p53), Float("0x1p0", context=p53), Float.from_json(one.to_json()), add(half, half)]
    for value in sources:
        assert_true(value.same_representation(one))
        assert_equal(stable_hash(value), stable_hash(one))
    var third = Float("0.1", context=p53) * 3
    assert_true(third.same_representation(Float.from_json(third.to_json())))
    var z = Complex(one, zero)
    assert_true(not z.same_representation(Complex(one, negative_zero)))
    assert_true(z.same_representation(Complex.from_json(z.to_json())))


def test_keys() raises:
    var p53 = _context(53)
    var p128 = _context(128)
    var table = Dict[FloatKey, Int]()
    table[FloatKey(Float.zero(context=p53))] = 1
    table[FloatKey(Float.zero(negative=True, context=p53))] = 2
    table[FloatKey(Float(1, context=p53))] = 3
    table[FloatKey(Float(1, context=p128))] = 4
    table[FloatKey(Float.nan(context=p53))] = 5
    table[FloatKey(Float.nan(context=p53))] = 6
    assert_equal(len(table), 5)
    var half = Float("0.5", context=p53)
    assert_equal(table[FloatKey(add(half, half))], 3)
    assert_equal(table[FloatKey(Float.nan(context=p53))], 6)
    assert_equal(table[FloatKey(Float.zero(negative=True, context=p53))], 2)
    assert_equal(String(FloatKey(Float(3, context=p53))), "FloatKey(3.0)")
    var complexes = Dict[ComplexKey, Int]()
    var one = Float(1, context=p53)
    complexes[ComplexKey(Complex(one, Float.zero(context=p53)))] = 1
    complexes[ComplexKey(Complex(one, Float.zero(negative=True, context=p53)))] = 2
    complexes[ComplexKey(Complex(one, Float.zero(context=p53)))] = 3
    assert_equal(len(complexes), 2)
    assert_equal(complexes[ComplexKey(Complex(one, Float.zero(context=p53)))], 3)
    assert_true(FloatKey(Float.zero(negative=True, context=p53)) < FloatKey(Float.zero(context=p53)))
    assert_true(FloatKey(Float(1, context=p53)) < FloatKey(Float(1, context=p128)))


def _corpus() raises -> List[Float]:
    var p53 = _context(53)
    var p128 = _context(128)
    var values = List[Float]()
    for context in [p53, p128, _binary64()]:
        values.append(Float.infinity(negative=True, context=context))
        values.append(Float(-1, context=context))
        values.append(Float.zero(negative=True, context=context))
        values.append(Float.zero(context=context))
        values.append(Float("0.1", context=context))
        values.append(Float(1, context=context))
        values.append(Float(2, context=context))
        values.append(Float.infinity(context=context))
        values.append(Float.nan(context=context))
    values.append(Float("0.5", context=_context(2)))
    values.append(Float(-3, context=_context(7)))
    return values^


def _sorted(var keys: List[FloatKey]) -> List[FloatKey]:
    for i in range(1, len(keys)):
        var j = i
        while j > 0 and keys[j] < keys[j - 1]:
            var swap = keys[j]
            keys[j] = keys[j - 1]
            keys[j - 1] = swap
            j -= 1
    return keys^


def test_representation_order() raises:
    var values = _corpus()
    var n = len(values)
    for i in range(n):
        for j in range(n):
            var order = values[i].representation_cmp(values[j])
            assert_equal(order, -values[j].representation_cmp(values[i]))
            assert_equal(order == 0, values[i].same_representation(values[j]))
            assert_equal(order == 0, i == j)
            for k in range(n):
                if order <= 0 and values[j].representation_cmp(values[k]) <= 0:
                    assert_true(values[i].representation_cmp(values[k]) <= 0)
    # Equal values order by precision, then by the lower exponent bound.
    var p53 = _context(53)
    assert_equal(Float(1, context=p53).representation_cmp(Float(1, context=_context(128))), -1)
    assert_equal(Float(1, context=p53).representation_cmp(Float(1, context=_binary64())), -1)
    assert_equal(Float.zero(negative=True, context=p53).representation_cmp(Float.zero(context=p53)), -1)
    # Sorting gives one result from any input order.
    var keys = List[FloatKey]()
    for value in values:
        keys.append(FloatKey(value))
    var expected = _sorted(keys.copy())
    var state = UInt64(12345)
    for round in range(6):
        var shuffled = keys.copy()
        if round == 1:
            shuffled.reverse()
        for i in range(len(shuffled) - 1, 0, -1):
            state ^= state << 13
            state ^= state >> 7
            state ^= state << 17
            var j = Int(state % UInt64(i + 1)) if round > 1 else i
            var swap = shuffled[i]
            shuffled[i] = shuffled[j]
            shuffled[j] = swap
        var result = _sorted(shuffled^)
        for i in range(len(result)):
            assert_true(result[i] == expected[i])
    var z1 = Complex(Float(1, context=p53), Float(2, context=p53))
    var z2 = Complex(Float(1, context=p53), Float(3, context=p53))
    assert_equal(z1.representation_cmp(z2), -1)
    assert_equal(z2.representation_cmp(z1), 1)
    assert_equal(z1.representation_cmp(z1), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
