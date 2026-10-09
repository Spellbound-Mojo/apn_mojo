"""Bounded public-API regression scenarios; independent timings live in benchmarks/."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from std.memory import ArcPointer, bitcast
from std.sys import size_of
from std.builtin.builtin_slice import StridedSlice
from apn_mojo.common._memory import _MemoryCensus
from apn_mojo.integer._multiplication import _multiply_into, _product_scratch_size, _high_product_into
from apn_mojo.float._multiplication import _try_high_product
from apn_mojo.float._multiplication import _binary_product
from apn_mojo.float._rounding import _round_ratio, _finish_round, _RoundedBinary
from apn_mojo.rational._batch_storage import _native_rational_iterator
from apn_mojo.rational._batch_json import _parse_rational_batch_json
from apn_mojo.float._batch_json import _parse_float_batch_json
from apn_mojo.complex._batch_storage import _native_complex_iterator
from apn_mojo.complex._batch_json import _parse_complex_batch_json
from apn_mojo.complex.context import _ComplexContextArgument
from apn_mojo.common.conversion import _ConversionBudget
from apn_mojo.batch._layout import _FlatLayout, _broadcast_shapes, _layout_record_bytes
from apn_mojo.batch._tensor import _Tensor
from apn_mojo.batch._values import _Values
from apn_mojo.batch._parallel import _map_tensor
from apn_mojo.batch.value import _float_integral, _float_sign, _float_comparison
from apn_mojo.integer._limits import _MAX_RESULT_BITS
from apn_mojo.batch.value import _complex_result, _complex_comparison, _complex_power, _complex_unary
from apn_mojo.batch.value import _rational_operand
from apn_mojo.batch.value import _rational_result, _rational_power, _rational_unary
from apn_mojo.integer.math import _math_scalar
from apn_mojo.integer.number_theory import _sqrt_rem
from apn_mojo.integer.compound import _axpy
from apn_mojo.batch.reductions import _reduce, _dot
from apn_mojo.batch.mapping import _Result
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._arithmetic import _float_operation, _float_sum
from apn_mojo.float._accumulator import _ExactAccumulator
from apn_mojo.float._input import _FloatInput
from apn_mojo import (
    Integer, Rational, Float, Complex, Batch, Mask,
    FloatFormat, ArithmeticContext, ComplexContext, RoundingMode,
    ConversionLimits, abs, factorial, factorial2, comb, iroot,
    inverse_mod, pow_mod, div_exact, gcd, lcm, isqrt,
    div_rem_floor, div_rem_trunc, div_rem_euclid, pow_rational,
    add, subtract, multiply, divide, square, sqrt, pow_int, ldexp, fma,
    norm_sqr, sum, prod, dot, vdot, axpy,
    sum_sequential, sum_tree, dot_sequential, vmap, lift,
)
from apn_mojo import integer, rational, float, complex
from apn_mojo.complex import pow_int as complex_pow_int, abs as complex_abs
# numpy's own aliases: min and max would hide Mojo's built-ins here.
from apn_mojo import min as amin, max as amax
from apn_mojo.float._arithmetic import _FloatArgument
from apn_mojo.float._input import _float_input
from apn_mojo.float._functions import _sqrt_float
from apn_mojo.float._parse import _parse_float


# Status capture is not public; these read the internal rounding flags.
def _flags[T: Copyable](value: T, context: Optional[ArithmeticContext] = None) raises -> NumericStatus:
    return Float._rounded_input(_float_input(value), context.value() if context else ArithmeticContext()).status


def _op_flags(
    left: _FloatArgument, right: _FloatArgument, operation: Int,
    context: Optional[ArithmeticContext] = None,
) raises -> NumericStatus:
    return _float_operation(left, right, operation, context).status


def _sqrt_flags(value: _FloatArgument, context: Optional[ArithmeticContext] = None) raises -> NumericStatus:
    return _sqrt_float(value, context).status


def _parse_flags(text: String) raises -> NumericStatus:
    var budget = _ConversionBudget(None)
    return _parse_float(text, ArithmeticContext(), False, False, budget).status


def test_integer_widths_carry_copy_and_native_conversion() raises:
    for bits in [31, 32, 63, 64, 127, 256]:
        var boundary = Integer(1) << bits
        var a = boundary - 1
        var saved = a
        a += 1
        assert_equal(a, boundary)
        assert_equal(saved, boundary - 1)
        assert_equal(a * a - 1, (a - 1) * (a + 1))
        assert_equal((a * 17 + 3) // a, 17)
        assert_equal((a * 17 + 3) % a, 3)
        assert_equal(a.magnitude_bit_length(), bits + 1)
        for sign in [-1, 1]:
            var signed = sign * a
            var copy = signed * Integer(1)
            assert_equal(Integer(1) * signed, signed)
            assert_equal(signed * Integer(-1), -signed)
            assert_equal(Integer(-1) * signed, -signed)
            copy += 1
            assert_equal(copy - 1, signed)
            assert_equal(signed, sign * boundary)
            for factor in [Integer(2), Integer(7), Integer(UInt32.MAX), Integer(1) << 32,
                           Integer(Int64.MAX), Integer(Int64.MIN), Integer(UInt64.MAX)]:
                var expected = (factor << bits) - factor
                if sign < 0:
                    expected = -expected
                assert_equal(sign * (boundary - 1) * factor, expected)
                assert_equal(factor * (sign * (boundary - 1)), expected)
                assert_equal(sign * (boundary - 1) * -factor, -expected)
    assert_equal(Integer(18446744073709551616), Integer(1) << 64)
    assert_equal(Integer(1) + UInt64.MAX, Integer(1) << 64)
    assert_equal(18446744073709551616 - Integer(1), Integer(UInt64.MAX))
    assert_equal(Integer(Int64.MIN).to_native_exact[DType.int64](), Int64.MIN)
    assert_equal(Integer(UInt64.MAX).to_native_exact[DType.uint64](), UInt64.MAX)
    var native_edges: List[Int64] = [
        Int64.MIN, Int64.MIN + 1, -(1 << 32), -1, 0, 1, 1 << 32, Int64.MAX - 1, Int64.MAX,
    ]
    for a in native_edges:
        for b in native_edges:
            var signed_product = Int128(a) * Int128(b)
            var magnitude = UInt128(-signed_product if signed_product < 0 else signed_product)
            var expected = Integer(UInt64(magnitude)) + (Integer(UInt64(magnitude >> 64)) << 64)
            if signed_product < 0:
                expected = -expected
            assert_equal(Integer(a) * Integer(b), expected)
    with assert_raises():
        _ = Int(Integer(1) << 64)
    var keys = Dict[Integer, String]()
    var key = Integer(1) << 128
    keys[key] = "wide"
    key += 1
    assert_equal(keys[key - 1], "wide")


def test_integer_wide_products_squares_and_update_rollback() raises:
    for bits in [63, 64, 65, 127, 223, 224, 225, 255, 256, 257, 1024, 2016, 2048, 2049, 4097, 16385]:
        var top = Integer(1) << bits
        var a = top - 1
        var b = (top >> 2) + 17
        var expected = (b << bits) - b
        var squared = (Integer(1) << (2 * bits)) - (top << 1) + 1
        var saved = a
        assert_equal(a * b, expected)
        assert_equal(b * a, expected)
        assert_equal(-a * b, -expected)
        assert_equal(a * -b, -expected)
        assert_equal(-a * -b, expected)
        assert_equal(a * a, squared)
        var independent = (a - 1) + 1
        assert_equal(a * independent, squared)
        var update = a
        update *= update
        assert_equal(update, squared)
        var unique = top - 1
        unique *= unique
        assert_equal(unique, squared)
        update = a
        update *= b
        assert_equal(update, expected)
        update = a
        update **= 3
        assert_equal(update, (Integer(1) << (3 * bits)) - 3 * (top << bits) + 3 * top - 1)
        assert_equal(a, saved)
    var literal = Integer(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF)
    var dynamic = (Integer(1) << 256) - 1
    var squared = (Integer(1) << 512) - (Integer(1) << 257) + 1
    assert_equal(literal * dynamic, squared)
    assert_equal(dynamic * literal, squared)
    assert_equal(literal * literal, squared)
    for stage in [0, 1]:
        var value = (Integer(1) << 256) - 1
        with assert_raises():
            value._assign(value, 2, fail_stage=stage)
        assert_equal(value, dynamic)
        value *= value
        assert_equal(value, squared)


def test_product_owner_sharing_growth_retirement_and_weak_lifetime() raises:
    for capacity in [0, 1, 2, 3, 7, 8, 9, 31, 32, 33, 128, 129, 513]:
        var owner = Integer._Shared.uninitialized(capacity, False)
        for i in range(capacity):
            owner[].words[i] = UInt32(i + 1)
        var weak_owner = Integer._Shared.WeakPointer(owner)
        var upgrade = weak_owner.try_upgrade()
        assert_true(Bool(upgrade))
        assert_equal(owner.count(), 2)
        assert_equal(upgrade.value()[].words.capacity(), capacity)
        for i in range(capacity):
            assert_equal(upgrade.value()[].words[i], UInt32(i + 1))
    for capacity in [-1, Int.MAX]:
        with assert_raises():
            _ = Integer._Shared.uninitialized(capacity, False)
    var a = (Integer(1) << 256) - 1
    var value = a * a
    var census = _MemoryCensus()
    census.add(value)
    assert_equal(census.finish().current.allocations, 1)
    assert_equal(census.finish().current.payload, 64)
    var weak = Integer._Shared.WeakPointer(value._storage[Integer._Shared])
    var weak_copy = weak
    var upgraded = weak.try_upgrade()
    assert_true(Bool(upgraded))
    assert_equal(upgraded.value()[].words[0], UInt32(1))
    value += 1
    assert_equal(upgraded.value()[].words[0], UInt32(1))
    assert_equal(value, a * a + 1)
    upgraded = None
    assert_true(not upgraded)
    assert_true(not weak.try_upgrade())
    assert_true(not weak_copy.try_upgrade())
    for stage in [0, 1]:
        var unique = a * a
        with assert_raises():
            unique._assign(Integer(1) << 1024, 2, fail_stage=stage)
        assert_equal(unique, a * a)
        unique *= unique
        assert_equal(unique, (a * a) * (a * a))
    var grown = a * a
    grown <<= 2048
    assert_equal(grown, (a * a) << 2048)
    var external = _MemoryCensus()
    external.add(grown)
    assert_equal(external.finish().current.allocations, 2)
    var wide = (Integer(1) << 8192) + 3
    var retired = wide * wide
    retired >>= 16320
    assert_equal(retired, Integer(1) << 64)
    var compact = _MemoryCensus()
    compact.add(retired)
    assert_true(compact.finish().current.spare <= 128 * 4)
    var shared = a * a
    var saved = shared
    shared *= shared
    assert_equal(saved, a * a)
    assert_equal(shared, saved * saved)


def test_multiplication_kernels_against_word_reference() raises:
    # Independent 32-bit schoolbook reference; also checks output/scratch guards.
    var seed = UInt64(0xD1B54A32D192ED03)
    for left in [0, 1, 2, 3, 10, 14, 15, 16, 17, 18, 30, 31, 32, 33, 62, 63, 64, 65, 66, 95, 96, 97, 127, 128, 129, 192, 257, 513]:
        for right in [left, max(1, left - 2), max(1, left // 2), 3]:
            var a = List[UInt32](length=left, fill=0)
            var b = List[UInt32](length=right, fill=0)
            for i in range(left):
                seed = seed * 6364136223846793005 + 1442695040888963407
                a[i] = UInt32(seed >> 32)
            for i in range(right):
                seed = seed * 6364136223846793005 + 1442695040888963407
                b[i] = UInt32(seed >> 32)
            for same in [False, True]:
                if same:
                    b = a.copy()
                var count = len(a) + len(b)
                var expected = List[UInt32](length=count, fill=0)
                for i in range(len(a)):
                    var carry = UInt64(0)
                    for j in range(len(b)):
                        var total = UInt64(a[i]) * UInt64(b[j]) + UInt64(expected[i + j]) + carry
                        expected[i + j] = UInt32(total)
                        carry = total >> 32
                    expected[i + len(b)] = UInt32(carry)
                var capacity = _product_scratch_size(len(a), len(b))
                for available in [0, capacity // 2, capacity]:
                    var work = List[UInt32](length=capacity + 2, fill=0xA5A5A5A5)
                    var actual = List[UInt32](length=count + 4, fill=0xA5A5A5A5)
                    var scratch = Span(
                        unsafe_ptr=work.unsafe_ptr().unsafe_offset(1)
                            .unsafe_origin_cast[MutAnyOrigin](),
                        length=available,
                    )
                    _ = _multiply_into(Span(a), Span(b), Span(actual)[1:count + 3], same, scratch)
                    for i in range(count):
                        assert_equal(actual[i + 1], expected[i])
                    assert_equal(actual[count + 1], UInt32(0))
                    assert_equal(actual[count + 2], UInt32(0))
                    assert_equal(actual[0], UInt32(0xA5A5A5A5))
                    assert_equal(actual[count + 3], UInt32(0xA5A5A5A5))
                    assert_equal(work[0], UInt32(0xA5A5A5A5))
                    assert_equal(work[capacity + 1], UInt32(0xA5A5A5A5))


def test_integer_signed_division() raises:
    for a in [-7, 0, 7]:
        for b in [-3, 3]:
            var qf, rf = div_rem_floor(a, b)
            var qt, rt = div_rem_trunc(a, b)
            var qe, re = div_rem_euclid(a, b)
            assert_equal(qf, Integer(a // b))
            assert_equal(rf, Integer(a % b))
            assert_equal(qt * b + rt, a)
            assert_equal(qe * b + re, a)
            assert_true(re >= 0 and re < abs(b))
            assert_true(rt == 0 or rt.sign() == Integer(a).sign())
            assert_equal(Integer(a) / b, Rational(a, b))
    var value = Integer(17)
    with assert_raises():
        value //= 0
    assert_equal(value, 17)
    value //= 3
    assert_equal(value, 5)


def test_one_limb_division_conventions() raises:
    # Dividends of three or more words by divisors of one limb, which round in
    # the division itself: every sign, the three conventions and the
    # operators, against q b + r = a and each convention's remainder range;
    # all-ones dividends, whose rounded quotients carry through their words,
    # and exact multiples, which need no rounding.
    var divisors: List[Integer] = [
        2, 3, 7, 2147483629, (Integer(1) << 32) + 1, (Integer(1) << 63) + 5, (Integer(1) << 64) - 1,
    ]
    var dividends = List[Integer]()
    for bits in [65, 96, 127, 128, 129, 256, 1024, 2048]:
        dividends.append((Integer(0x9E3779B97F4A7C15) << (bits - 64)) + 12345)
        dividends.append((Integer(1) << bits) - 1)
        dividends.append(Integer(1) << bits)
    for magnitude in dividends:
        for divisor in divisors:
            for multiple in [False, True]:
                var a0 = magnitude * divisor if multiple else magnitude
                for sa in [1, -1]:
                    for sb in [1, -1]:
                        var a = a0 * sa
                        var b = divisor * sb
                        var qf, rf = div_rem_floor(a, b)
                        assert_equal(qf * b + rf, a)
                        assert_true((rf == 0 or rf.sign() == b.sign()) and abs(rf) < abs(b))
                        assert_equal(a // b, qf)
                        assert_equal(a % b, rf)
                        var qt, rt = div_rem_trunc(a, b)
                        assert_equal(qt * b + rt, a)
                        assert_true((rt == 0 or rt.sign() == a.sign()) and abs(rt) < abs(b))
                        var qe, re = div_rem_euclid(a, b)
                        assert_equal(qe * b + re, a)
                        assert_true(re >= 0 and re < abs(b))
                        if multiple:
                            assert_true(not rf and not rt and not re)


def test_integer_bits_and_compounds() raises:
    var wide = (Integer(1) << 129) + 5
    for value in [Integer(-5), Integer(0), wide]:
        assert_equal(~value, -value - 1)
        assert_equal(value & ~value, 0)
        assert_equal(value | ~value, -1)
        assert_equal(value ^ value, 0)
        assert_equal((value << 33) >> 33, value)
    assert_equal(Integer(-5) >> 1, -3)
    var n = Integer(13)
    n += 5
    n -= 3
    n *= 2
    n //= 4
    n %= 5
    n **= 3
    n <<= 2
    n >>= 1
    n &= 31
    n |= 1
    n ^= 3
    assert_equal(n, 18)
    assert_equal(Integer(-1) ** ((Integer(1) << 128) + 1), -1)
    with assert_raises():
        n <<= -1
    with assert_raises():
        n **= -1
    assert_equal(n, 18)


def test_integer_math_fixed_answers_and_boundaries() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(30), 265252859812191058636308480000000)
    assert_equal(factorial2(10), 3840)
    assert_equal(factorial2(11), 10395)
    assert_equal(comb(100, 50), 100891344545564193334812497256)
    assert_equal(comb(-3, 3), 0)
    assert_equal(pow_mod(7, -5, -1000), 943)
    assert_equal(inverse_mod(3, -11), 4)
    assert_equal(div_exact(-84, 7), -12)
    assert_equal(gcd(-84, 30), 6)
    assert_equal(gcd(Integer(Int64.MIN), Integer(Int64.MIN)), Integer(1) << 63)
    assert_equal(gcd(Integer(Int64.MIN), 0), Integer(1) << 63)
    assert_equal(gcd(0, Integer(Int64.MIN)), Integer(1) << 63)
    assert_equal(gcd(Integer(Int64.MIN), Integer(Int64.MAX)), 1)
    assert_equal(gcd(0, 0), 0)
    assert_equal(lcm(-84, 30), 420)
    assert_equal(isqrt(26), 5)
    for degree in [2, 3, 5, 31]:
        var root = (Integer(1) << 65) + 1
        var power = root ** degree
        assert_equal(iroot(power - 1, degree), root - 1)
        assert_equal(iroot(power, degree), root)
        assert_equal(iroot(power + 1, degree), root)
        if degree % 2:
            assert_equal(iroot(-power + 1, degree), -root + 1)
            assert_equal(iroot(-power - 1, degree), -root)
    assert_equal(iroot(-9, 3), -2)
    var huge = Integer(1) << 128
    assert_equal(comb(huge, huge), 1)
    assert_equal(comb(huge, huge - 1), huge)
    assert_equal(comb(huge, huge + 1), 0)
    assert_equal(iroot(huge, huge), 1)
    assert_equal(pow_mod(2, -huge, 17), 1)
    assert_equal(pow_mod(0, -1, 1), 0)


def test_square_root_remainders_across_sizes() raises:
    # Every length from 100 to 700 bits, native and limb-level, and wide ones
    # past the stack scratch (about 5800 bits) and the Karatsuba squaring
    # threshold: perfect squares and their neighbours, powers of two, all-ones
    # radicands (whose uncorrected root overflows) and dense values.
    # s**2 + r == a with 0 <= r <= 2 s holds for s = floor(sqrt(a)) alone.
    var state = Integer(0x9E3779B97F4A7C15)
    var lengths = List[Int]()
    for bits in range(100, 701):
        lengths.append(bits)
    for bits in [2000, 5759, 5761, 6000, 12345, 70001]:
        lengths.append(bits)
    for bits in lengths:
        var root = (Integer(1) << (bits // 2)) - 3
        state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << bits)
        var values = [
            (Integer(1) << bits) - 1, Integer(1) << (bits - 1), root * root,
            root * root - 1, root * root + 2 * root, state | (Integer(1) << (bits - 1)),
        ]
        for value in values:
            var pair = _sqrt_rem(value)
            assert_equal(pair[0] * pair[0] + pair[1], value)
            assert_true(pair[1].sign() >= 0 and pair[1] <= pair[0] << 1)


def test_integer_root_staging_and_native_boundaries() raises:
    assert_equal(isqrt(0), 0)
    assert_equal(isqrt(1), 1)
    assert_equal(isqrt(Integer(UInt64.MAX)), 4294967295)
    with assert_raises():
        _ = isqrt(-1)
    for bits in [31, 63, 64, 65, 129, 256, 1024]:
        for degree in [2, 3, 31, 65]:
            var root = (Integer(1) << max(1, bits // degree)) + 1
            var power = root ** degree
            for offset in [-1, 0, 1]:
                var expected = root - Int(offset < 0)
                assert_equal(iroot(power + offset, degree), expected)
                if degree == 2:
                    assert_equal(isqrt(power + offset), expected)
                if degree % 2:
                    assert_equal(iroot(-power - offset, degree), -expected)
            var x = (Integer(1) << bits) - 1
            var saved = x
            var floor = iroot(x, degree)
            assert_true(floor ** degree <= x)
            assert_true((floor + 1) ** degree > x)
            assert_equal(x, saved)
            assert_equal(iroot(x, bits), 1)
    for value in range(2, 128):
        for degree in [3, 5, 7]:
            var root = iroot(value, degree)
            assert_true(root ** degree <= value)
            assert_true((root + 1) ** degree > value)


def test_factorials_against_running_products() raises:
    # Odd parts and their shifts, limb chains, halved products past 96 limbs
    # and heap scratch past about 1,500!: every n up to 400, then sparse ones
    # to 3,000, against running products; double factorials of both parities.
    var product = Integer(1)
    var odd = Integer(1)
    var even = Integer(1)
    for n in range(1, 3001):
        product *= n
        if n % 2:
            odd *= n
        else:
            even *= n
        if n <= 400 or n % 97 == 0 or n in [512, 1024, 1500, 1501, 2048, 2049, 3000]:
            assert_equal(factorial(n), product)
            assert_equal(factorial2(n), odd if n % 2 else even)


def test_integer_math_domains_and_diagnostics() raises:
    var remedies = [
        "nonnegative", "nonnegative", "nonnegative", "odd positive degree",
        "greatest common divisor", "floor division", "nonzero modulus",
    ]
    for operation in range(7):
        var destination = Integer(123)
        var failed = False
        try:
            if operation == 0:
                destination = factorial(-1)
            elif operation == 1:
                destination = factorial2(-1)
            elif operation == 2:
                destination = factorial2(-3)
            elif operation == 3:
                destination = iroot(-8, 2)
            elif operation == 4:
                destination = inverse_mod(2, 4)
            elif operation == 5:
                destination = div_exact(7, 3)
            else:
                destination = pow_mod(2, 3, 0)
        except error:
            failed = True
            assert_true(remedies[operation] in String(error))
        assert_true(failed)
        assert_equal(destination, 123)


def test_integer_batch_math_broadcasts_and_selections() raises:
    # Batches apply the scalar functions through vmap; scalars are shared.
    for length in [0, 1, 8, 9, 17]:
        var a = Batch[Integer].from_iterable(range(length))
        var f = vmap[factorial]()(a)
        var d = vmap[factorial2]()(a[::-1])
        for i in range(length):
            assert_equal(f[i], factorial(i))
            assert_equal(d[i], factorial2(length - i - 1))
        assert_equal(len(vmap[iroot]()(a, 2)), length)
    var a = Batch[Integer]([-1, 3, -1, 4, -1, 5])
    var selected = a[1::2][::-1]
    assert_equal(String(vmap[factorial]()(selected)), "[120, 24, 6]")
    assert_equal(String(vmap[integer.comb]()(10, selected)), "[252, 210, 120]")
    assert_equal(String(vmap[pow_mod]()(selected, -1, 11)), "[9, 3, 4]")
    assert_equal(String(vmap[inverse_mod]()(selected, 11)), "[9, 3, 4]")
    assert_equal(String(vmap[div_exact]()(selected * 12, selected)), "[12, 12, 12]")
    assert_equal(String(vmap[iroot]()(selected ** 3, 3)), "[5, 4, 3]")
    assert_equal(String(vmap[gcd]()(selected, 6)), "[1, 2, 3]")
    assert_equal(String(vmap[lcm]()(selected, 6)), "[30, 12, 6]")
    assert_equal(String(vmap[isqrt]()(selected)), "[2, 2, 1]")
    assert_equal(String(vmap[factorial]()(a[a > 0])), "[6, 24, 120]")
    assert_equal(String(vmap[gcd]()(selected, (Integer(1) << 200) * 3)), "[1, 4, 3]")
    var before = a.to_json()
    with assert_raises():
        a = vmap[factorial]()(a[::-1])
    assert_equal(a.to_json(), before)
    assert_equal(len(vmap[pow_mod]()(a[:0], -1, 0)), 0)


def test_rational_canonicalization_arithmetic_and_copy() raises:
    var wide = (Integer(1) << 129) + 1
    assert_equal(Rational(wide * 8, -wide * 12), Rational(-2, 3))
    assert_equal(Rational(0, -7).denominator(), 1)
    var a = Rational(2, 3)
    var b = Rational(-5, 7)
    assert_equal(a + b, Rational(-1, 21))
    assert_equal(a - b, Rational(29, 21))
    assert_equal(a * b, Rational(-10, 21))
    assert_equal(a / b, Rational(-14, 15))
    for denominator in [Integer(1), Integer(15), (Integer(1) << 129) - 1]:
        var left = Rational(1, denominator)
        var right = Rational(2, denominator)
        assert_equal(left + right, Rational(3, denominator))
        assert_equal(left - right, -left)
        assert_equal(right - left, left)
        assert_equal(left - left, Rational())
        assert_equal(left + -left, Rational())
        assert_equal(-left - right, Rational(-3, denominator))
        var changed = left + right
        changed += left
        assert_equal(changed, Rational(4, denominator))
        assert_equal(left, Rational(1, denominator))
        assert_equal(right, Rational(2, denominator))
    assert_equal(Rational(wide, 3) * Rational(3, wide), Rational(1))
    assert_true(Rational(wide, wide + 1) < Rational(wide + 1, wide + 2))
    assert_equal(1 - a, Rational(1, 3))
    assert_equal(a + UInt64.MAX, Rational(Integer(UInt64.MAX) * 3 + 2, 3))
    var saved = a
    var numerator = a.numerator()
    numerator += 1
    a += a
    a -= saved
    a *= a
    a /= saved
    a **= -2
    assert_equal(a, Rational(9, 4))
    assert_equal(saved, Rational(2, 3))
    with assert_raises():
        a /= 0
    assert_equal(a, Rational(9, 4))
    with assert_raises():
        _ = Rational(0, 0)
    assert_equal(hash(Rational(4, 6)), hash(saved))
    assert_equal(hash(Rational(7)), hash(Integer(7)))


def test_rational_rounding_powers_and_batches() raises:
    var a = Batch[Rational]([Rational(-7, 3), -2, 0, Rational(7, 3)])
    assert_equal(String(a.floor()), "[-3, -2, 0, 2]")
    assert_equal(String(a.ceil()), "[-2, -2, 0, 3]")
    assert_equal(String(a.trunc()), "[-2, -2, 0, 2]")
    assert_equal(String(vmap[rational.abs]()(a)), "[7/3, 2, 0, 7/3]")
    assert_equal(String((-a).sign()), "[1, 1, 0, -1]")
    assert_equal(Int(Rational(-6, 3)), -2)
    with assert_raises():
        _ = Int(Rational(1, 2))
    var bases = Batch[Rational]([Rational(-2, 3), Rational(3, 2), 0])
    var exponents = Batch[Integer]([-3, 2, 0])
    assert_equal(String(bases ** exponents), "[-27/8, 9/4, 1]")
    assert_equal(String(vmap[pow_rational]()(2, exponents)), "[1/8, 4, 1]")
    assert_equal(String(Batch[Integer]([2, 3]) / 2), "[1, 3/2]")
    assert_equal(Rational(-1) ** (-(Integer(1) << 128) - 1), Rational(-1))
    var before = bases.to_json()
    with assert_raises():
        bases **= -1
    assert_equal(bases.to_json(), before)
    bases += Rational(1)
    bases **= -1
    assert_equal(bases[0], Rational(3))


def test_float_rounding_ties_and_all_modes() raises:
    var nearest = ArithmeticContext(format=FloatFormat(1))
    for n in [3, 5, 7, -3]:
        var expected = 4 if n == 7 else -2 if n < 0 else 2
        assert_equal(Float(Rational(n, 2), context=nearest), Float(expected))
    for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                 RoundingMode.toward_positive, RoundingMode.toward_negative,
                 RoundingMode.away_from_zero]:
        for negative in [False, True]:
            var up = (mode == RoundingMode.toward_positive and not negative
                      or mode == RoundingMode.toward_negative and negative
                      or mode == RoundingMode.away_from_zero)
            var c = ArithmeticContext(format=FloatFormat(3), rounding=mode)
            var sign = -1 if negative else 1
            var result = Float(Rational(sign, 3), context=c)
            var status = _flags(Rational(sign, 3), c)
            assert_equal(result, Float(Rational(sign * (6 if up else 5), 16)))
            assert_true(status.inexact())
            assert_equal(status.direction(), "below" if negative == up else "above")
    assert_true(Float(1, context=nearest) < Rational(5, 4))
    assert_true(Float(Integer(1) << 128, context=nearest) < (Integer(1) << 128) + 1)


def test_float_range_specials_and_native_decoding() raises:
    var c = ArithmeticContext(format=FloatFormat(2, emin=0, emax=4))
    var tiny = Float(Rational(-1, 4), context=c)
    assert_true(tiny.is_zero() and tiny.signbit())
    var status = _flags(Rational(-1, 4), c)
    assert_true(status.underflow() and status.inexact())
    assert_true(Float(14, context=c).is_infinite())
    status = _flags(Integer(14), c)
    assert_true(status.overflow() and status.inexact())
    assert_equal(Float(bitcast[DType.float64](UInt64(1))).to_string(16), "0x1p-1074")
    assert_true(Float(bitcast[DType.float64](UInt64(1) << 63)).signbit())
    var nan = Float.nan()
    assert_true(nan != nan and not (nan < Float(0)))
    assert_true(Float.zero(negative=True) == Float.zero())
    assert_equal(Float.zero(negative=True).total_cmp(Float.zero()), -1)
    assert_true(Float.infinity() > Float(Integer(1) << 256))
    assert_true(divide(Float(1), 0).is_infinite())
    assert_true(_op_flags(Float(1), 0, 3).divide_by_zero())
    assert_true(sqrt(Float(-1)).is_nan())
    assert_true(_sqrt_flags(Float(-1)).invalid())


def test_dyadic_rounding_limb_boundaries() raises:
    for p in [3, 65]:
        for odd in [False, True]:
            var q = (Integer(1) << p) - 1 if odd else Integer(1) << (p - 1)
            for discard in [2, 31, 32, 33, 65]:
                var denominator = Integer(1) << discard
                var half = denominator >> 1
                for offset in [-1, 0, 1]:
                    var n = q * denominator + half + offset
                    for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                                 RoundingMode.toward_positive, RoundingMode.toward_negative,
                                 RoundingMode.away_from_zero]:
                        for negative in [False, True]:
                            var up = (offset > 0 or (offset == 0 and odd)) if mode == RoundingMode.nearest_even else (
                                mode == RoundingMode.away_from_zero
                                or (mode == RoundingMode.toward_positive and not negative)
                                or (mode == RoundingMode.toward_negative and negative)
                            )
                            var sign = -1 if negative else 1
                            var c = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                            var value = Float(Rational(sign * n, denominator), context=c)
                            var status = _flags(Rational(sign * n, denominator), c)
                            var exact = ArithmeticContext(format=FloatFormat(p + 1))
                            assert_equal(value, Float(sign * (q + Int(up)), context=exact))
                            assert_true(status.inexact())
                            assert_equal(status.direction(), "below" if negative == up else "above")
    assert_equal(Float("1.000000000"), Float(1))
    assert_true(not _parse_flags("1.000000000").inexact())
    var value = Float(17)
    with assert_raises():
        value = Float("1.125", context=ArithmeticContext(format=FloatFormat(3), trap_inexact=True))
    assert_equal(value, Float(17))


def test_float_product_rounding_and_sparse_significands() raises:
    var source = (Integer(1) << 127) + 5
    var target = ArithmeticContext(format=FloatFormat(65))
    var reference = _round_ratio(source * 7, Integer(1), target)
    for padding in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33]:
        var padded = source << (32 * padding)
        for reverse in [False, True]:
            var actual = _binary_product(
                Integer(7) if reverse else padded, padded if reverse else Integer(7),
                False, Int128(-32 * padding), target, False,
            )
            assert_equal(actual.significand, reference.significand)
            assert_equal(actual.exponent, reference.exponent)
            assert_equal(actual.status, reference.status)
        assert_equal(padded, source << (32 * padding))
    for words in [1, 2, 3, 8, 33]:
        var n = (Integer(1) << (32 * words)) - 1
        for factor in [Integer(1), Integer(UInt32.MAX), (Integer(1) << 32) + 1, Integer(UInt64.MAX)]:
            var product = (factor << (32 * words)) - factor
            for p in [31, 64, 32 * words + 65]:
                for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                             RoundingMode.toward_positive, RoundingMode.toward_negative,
                             RoundingMode.away_from_zero]:
                    var negative = (words + p) % 2 == 0
                    var c = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                    var expected = _round_ratio(-product if negative else product, Integer(1), c)
                    var full = _binary_product(n, factor, negative, Int128(0), c, False, path=1)
                    assert_equal(full.significand, expected.significand)
                    assert_equal(full.exponent, expected.exponent)
                    assert_equal(full.status, expected.status)
                    for reverse in [False, True]:
                        var actual = _binary_product(
                            factor if reverse else n, n if reverse else factor,
                            negative, Int128(0), c, False,
                        )
                        assert_equal(actual.negative, negative)
                        assert_equal(actual.significand, expected.significand)
                        assert_equal(actual.exponent, expected.exponent)
                        assert_equal(actual.status, expected.status)
            assert_equal(n, (Integer(1) << (32 * words)) - 1)
    for p in [1, 3, 31, 32, 33, 63, 64, 65, 95, 96, 97, 256]:
        for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                     RoundingMode.toward_positive, RoundingMode.toward_negative,
                     RoundingMode.away_from_zero]:
            var c = ArithmeticContext(format=FloatFormat(p, emin=-4, emax=4), rounding=mode)
            for odd in [False, True]:
                var q = (Integer(1) << p) - 1 if odd else Integer(1) << (p - 1)
                for discard in [1, 31, 32, 33, 65]:
                    for offset in [-1, 0, 1]:
                        var n = (q << discard) + (Integer(1) << (discard - 1)) + offset
                        for negative in [False, True]:
                            for scale in [-p - discard - 6, -p - discard, 4 - p - discard]:
                                var expected = _round_ratio(-n if negative else n, Integer(1), c, scale=Int128(scale))
                                var actual = _binary_product(n, Integer(1), negative, Int128(scale), c, False)
                                assert_equal(actual.kind, expected.kind)
                                assert_equal(actual.negative, expected.negative)
                                assert_equal(actual.significand, expected.significand)
                                assert_equal(actual.exponent, expected.exponent)
                                assert_equal(actual.status, expected.status)
    for p in [1, 32, 65, 256, 1024, 2049]:
        var c = ArithmeticContext(format=FloatFormat(p))
        var exact = ArithmeticContext(format=FloatFormat(2 * p + 3))
        var n = (Integer(1) << p) - 1
        var a = Float(n, context=exact)
        var b = Float(Rational(7, 4), context=exact)
        assert_equal(multiply(a, b, context=c), Float(Rational(n * 7, 4), context=c))
        assert_equal(_op_flags(a, b, 2, c), _flags(Rational(n * 7, 4), c))
        assert_equal(square(a, context=c), Float(n * n, context=c))
        assert_equal(a * b, multiply(a, b, context=exact))
        var saved = a
        var updated = a
        updated *= updated
        assert_equal(updated, square(a, context=exact))
        assert_equal(a, Float(n, context=exact))
        assert_equal(saved, a)
        var narrow = Float(3, context=c)
        assert_equal((narrow * b).format(), exact.format())
        var narrow_before = narrow
        narrow *= b
        assert_equal(narrow, multiply(narrow_before, b, context=c))
        assert_equal(narrow.format(), c.format())
        assert_equal(multiply(Float(1), Float(1), context=c), Float(1))
        assert_equal(multiply(a, Rational(1, 3), context=c), Float(Rational(n, 3), context=c))
    var value = Float(17)
    with assert_raises():
        value = multiply(Float(7), Float(3), context=ArithmeticContext(format=FloatFormat(3), trap_inexact=True))
    assert_equal(value, Float(17))
    with assert_raises():
        _ = _binary_product(Integer(7), Integer(3), False, Int128(0), ArithmeticContext(), True)
    with assert_raises():
        _ = Float._product(Float(7), Float(3), fail=True)
    var limited = Float(3, context=ArithmeticContext(format=FloatFormat(65, emin=-4, emax=4)))
    with assert_raises():
        _ = limited * value
    assert_equal(multiply(limited, value, context=ArithmeticContext()), Float(51))
    assert_true(multiply(Float.zero(negative=True), Float(2)).signbit())
    assert_true(multiply(Float.infinity(), Float.zero()).is_nan())
    assert_true(_op_flags(Float.infinity(), Float.zero(), 2).invalid())


def test_high_product_bounds_and_certification() raises:
    var seed = UInt64(0xD1B54A32D192ED03)
    var certified = 0
    for left in [3, 8, 9, 16, 32, 33, 128, 129]:
        for right in [left, max(1, left // 4), 3]:
            var a = List[UInt32](length=left, fill=0)
            var b = List[UInt32](length=right, fill=0)
            for i in range(left):
                seed = seed * 6364136223846793005 + 1442695040888963407
                a[i] = UInt32(seed >> 32)
            for i in range(right):
                seed = seed * 6364136223846793005 + 1442695040888963407
                b[i] = UInt32(seed >> 32)
            a[left - 1] |= 0x80000000
            b[right - 1] |= 0x80000000
            var x = Integer._from_words(a.copy(), False)
            var y = Integer._from_words(b.copy(), False)
            var product = x * y
            var n = (left + 1) // 2
            var m = (right + 1) // 2
            for cut in [1, (n + m) // 2, n + m - 1]:
                var output = List[UInt32](length=2 * (n + m - cut + 1) + 2, fill=0xA5A5A5A5)
                _ = _high_product_into(Span(a), Span(b), cut, Span(output)[1:len(output) - 1])
                assert_equal(output[0], UInt32(0xA5A5A5A5))
                assert_equal(output[len(output) - 1], UInt32(0xA5A5A5A5))
                var digits = List[UInt32]()
                for i in range(1, len(output) - 1):
                    digits.append(output[i])
                var low = Integer._from_words(digits^, False)
                assert_true((low << (64 * cut)) <= product)
                assert_true(product < ((low + (Integer(min(n, m)) << 64)) << (64 * cut)))
            for p in [65, 256, left * 32]:
                for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                             RoundingMode.toward_positive, RoundingMode.toward_negative,
                             RoundingMode.away_from_zero]:
                    var c = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                    for negative in [False, True]:
                        var actual = _binary_product(x, y, negative, Int128(0), c, False, path=2)
                        var expected = _binary_product(x, y, negative, Int128(0), c, False, path=1)
                        assert_equal(actual.kind, expected.kind)
                        assert_equal(actual.exponent, expected.exponent)
                        assert_equal(actual.significand, expected.significand)
                        assert_equal(actual.negative, expected.negative)
                        assert_equal(actual.status, expected.status)
                        certified += Int(_try_high_product(Span(a), Span(b), negative, Int128(0), c, False, actual))
    assert_true(certified > 0)
    var exact = Integer(1) << 1023
    var words = exact._words_copy()
    var c = ArithmeticContext(format=FloatFormat(256))
    var rounded = _binary_product(exact, exact, False, Int128(0), c, False, path=1)
    assert_true(not _try_high_product(Span(words), Span(words), False, Int128(0), c, False, rounded))
    with assert_raises():
        _ = _binary_product(exact + 1, exact + 3, False, Int128(0), c, True, path=2)
    for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                 RoundingMode.toward_positive, RoundingMode.toward_negative,
                 RoundingMode.away_from_zero]:
        for negative in [False, True]:
            for offset in [-1, 0, 1]:
                var n = ((Integer(1) << 64) + 3) << 256
                n += (Integer(1) << 255) + offset
                var target = ArithmeticContext(format=FloatFormat(65, emin=-4, emax=4), rounding=mode)
                for scale in [-327, -321, -315]:
                    var actual = _binary_product(n, Integer(1), negative, Int128(scale), target, False, path=2)
                    var expected = _round_ratio(-n if negative else n, Integer(1), target, scale=Int128(scale))
                    assert_equal(actual.kind, expected.kind)
                    assert_equal(actual.negative, expected.negative)
                    assert_equal(actual.significand, expected.significand)
                    assert_equal(actual.exponent, expected.exponent)
                    assert_equal(actual.status, expected.status)
    with assert_raises():
        _ = _binary_product(exact + 1, exact + 3, False, Int128(0),
                            ArithmeticContext(format=FloatFormat(256), trap_inexact=True), False, path=2)


def test_float_sqrt_exact_halfway_and_neighbors() raises:
    for p in [7, 65, 256]:
        for odd in [0, 1]:
            var q = (Integer(1) << (p - 1)) + 2 + odd
            var middle = 2 * q + 1
            for offset in [-1, 0, 1]:
                var x = Rational(middle * middle * 4 + offset, 16)
                for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                             RoundingMode.toward_positive, RoundingMode.toward_negative,
                             RoundingMode.away_from_zero]:
                    var up = (offset > 0 or (offset == 0 and odd != 0)) if mode == RoundingMode.nearest_even else (
                        mode == RoundingMode.toward_positive or mode == RoundingMode.away_from_zero
                    )
                    var c = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                    assert_equal(sqrt(x, context=c), Float(q + Int(up), context=c))
                    var status = _sqrt_flags(x, c)
                    assert_true(status.inexact())
                    assert_equal(status.direction(), "above" if up else "below")
            var c = ArithmeticContext(format=FloatFormat(p))
            assert_equal(sqrt(q * q, context=c), Float(q, context=c))
            assert_equal(_sqrt_flags(q * q, c), NumericStatus())
    var result = Float(17)
    with assert_raises():
        result = sqrt(Float(2), context=ArithmeticContext(trap_inexact=True))
    assert_equal(result, Float(17))


def test_float_arithmetic_functions_and_transactions() raises:
    var magnitude = (Integer(1) << 127) + 1
    var all_traps = ArithmeticContext(
        trap_inexact=True, trap_underflow=True, trap_overflow=True,
        trap_divide_by_zero=True, trap_invalid=True,
    )
    var names = ["inexact", "underflow", "overflow", "divide_by_zero", "invalid"]
    for highest in range(5):
        var flags = NumericStatus._make((1 << (highest + 1)) - 1, 1)
        var rounded = _RoundedBinary(1, False, magnitude, 1, all_traps.format(), flags)
        assert_equal(_finish_round(rounded, ArithmeticContext(), False).status, flags)
        for fail in [False, True]:
            var failed = False
            try:
                _ = _finish_round(rounded, all_traps, fail)
            except error:
                failed = True
                assert_true(String("trapped ", names[highest], ";") in String(error))
            assert_true(failed)
        assert_equal(rounded.significand, magnitude)
    with assert_raises():
        _ = _finish_round(
            _RoundedBinary(1, False, magnitude, 1, all_traps.format(), NumericStatus()),
            all_traps, True,
        )
    var a = Float(Rational(3, 2))
    assert_equal(add(a, 2), Float(Rational(7, 2)))
    assert_equal(subtract(2, a), Float(Rational(1, 2)))
    assert_equal(multiply(a, 2), Float(3))
    assert_equal(divide(3, a), Float(2))
    assert_equal(square(a), Float(Rational(9, 4)))
    assert_equal(sqrt(Float(9)), Float(3))
    assert_equal(pow_int(Float(2), -3), Float(Rational(1, 8)))
    assert_equal(ldexp(a, -1), Float(Rational(3, 4)))
    var c = ArithmeticContext(format=FloatFormat(3))
    assert_equal(fma(Float("1.25"), Float("1.25"), Float("-1.5"), context=c), Float("0.0625"))
    with assert_raises():
        a = divide(Float(1), 3, context=ArithmeticContext(trap_inexact=True))
    assert_equal(a, Float(Rational(3, 2)))
    a += 1
    a -= 1
    a *= 2
    a /= 2
    a **= 2
    assert_equal(a, Float(Rational(9, 4)))
    var narrow = Float(1, context=ArithmeticContext(format=FloatFormat(7, emin=-4, emax=4)))
    with assert_raises():
        _ = a + narrow
    assert_equal(add(a, narrow, context=c), Float(3))


def test_complex_components_arithmetic_and_functions() raises:
    var context = ComplexContext(
        real=ArithmeticContext(format=FloatFormat(11)),
        imag=ArithmeticContext(format=FloatFormat(53)),
    )
    var z = Complex(3, 4, context=context)
    assert_equal(z.real_format().precision(), 11)
    assert_equal(z.imag_format().precision(), 53)
    assert_equal(z.conjugate(), Complex(3, -4))
    assert_equal(norm_sqr(z), Float(25))
    assert_equal(abs(z), Float(5))
    assert_equal(sqrt(z), Complex(2, 1))
    assert_equal(Complex(1, 2) * Complex(3, 4), Complex(-5, 10))
    assert_equal(divide(Integer(5), Complex(1, 2)), Complex(1, -2))
    assert_equal(Rational(2) + z, Complex(5, 4))
    assert_equal(Float(2) - z, Complex(-1, -4))
    assert_equal(pow_int(Complex(1, 1), -2), Complex(0, Rational(-1, 2)))
    assert_equal(Complex(2, 3) ** 3, Complex(-46, 9))
    var saved = z
    var real = z.real()
    real += 1
    z += 1
    z -= 1
    z *= Complex(0, 1)
    z /= Complex(0, 1)
    assert_equal(z, saved)
    z **= 2
    assert_equal(z, Complex(-7, 24))
    assert_equal(saved, Complex(3, 4))


def test_complex_products_of_zero_parts_sign_as_mpc() raises:
    # Up to 128 bits zero parts take the native product path, beyond it the
    # exact path: both give the same values and zero signs.
    var parts = [Float(3), Float(-3), Float.zero(), Float.zero(negative=True)]
    for mode in [RoundingMode.nearest_even, RoundingMode.toward_negative]:
        var native = ComplexContext(ArithmeticContext(format=FloatFormat(53), rounding=mode))
        var exact = ComplexContext(ArithmeticContext(format=FloatFormat(200), rounding=mode))
        for i in range(256):
            var x = Complex(parts[i & 3], parts[(i >> 2) & 3])
            var y = Complex(parts[(i >> 4) & 3], parts[i >> 6])
            var a = multiply(x, y, context=native)
            var b = multiply(x, y, context=exact)
            assert_equal(a, b)
            assert_equal(a.real().signbit(), b.real().signbit())
            assert_equal(a.imag().signbit(), b.imag().signbit())
    # MPC's results (gmpy2, 31 bits): (+0) + (+0) is -0 toward negative.
    var down = ComplexContext(ArithmeticContext(format=FloatFormat(31), rounding=RoundingMode.toward_negative))
    var product = multiply(Complex(3, 0), Complex(5, 0), context=down)
    assert_equal(product, Complex(15, 0))
    assert_true(product.imag().signbit())
    product = multiply(Complex(3, 0), Complex(0, 5), context=down)
    assert_equal(product, Complex(0, 15))
    assert_true(product.real().signbit())
    product = multiply(Complex(-3, 0), Complex(0, 5), context=ArithmeticContext(format=FloatFormat(31)))
    assert_equal(product, Complex(0, -15))
    assert_true(product.real().signbit())
    product = multiply(Complex(3, 5), Complex(0, 0), context=ArithmeticContext(format=FloatFormat(31)))
    assert_true(not product.real().signbit() and not product.imag().signbit())


def test_complex_branch_sides_and_transactional_traps() raises:
    assert_equal(sqrt(Complex(-4, Float.zero())), Complex(0, 2))
    assert_equal(sqrt(Complex(-4, Float.zero(negative=True))), Complex(0, -2))
    assert_true(sqrt(Complex(Float.zero(), Float.zero(negative=True))).imag().signbit())
    assert_true(abs(Complex(Float.nan(), Float.infinity())).is_infinite())
    var result = Complex(7, 8)
    var trap = ArithmeticContext(format=FloatFormat(7), trap_inexact=True)
    for side in [0, 1]:
        var settings = ComplexContext(
            real=trap if side == 0 else ArithmeticContext(),
            imag=trap if side == 1 else ArithmeticContext(),
        )
        with assert_raises():
            result = sqrt(Complex(1, 2), context=settings)
        assert_equal(result, Complex(7, 8))
        with assert_raises():
            result = divide(Complex(1, 2), Complex(3, 7), context=settings)
        assert_equal(result, Complex(7, 8))
    result = sqrt(Complex(3, 4))
    assert_equal(result, Complex(2, 1))
    assert_true(not Complex(3, 4)._sqrt()[1].inexact())


def test_complex_division_exact_rounding_and_sparse_gaps() raises:
    var source = ArithmeticContext(format=FloatFormat(128))
    for gap in [0, 63, 127, 129, 130, 257]:
        for sign in [-1, 1]:
            var a = sign * ((Integer(1) << 67) + 11)
            var b = -(Integer(1) << 33) + 3
            var c = Integer(1) << gap
            var d = Integer(-3)
            var left = Complex(a, b, context=source)
            var right = Complex(c, d, context=source)
            var saved = left
            var denominator = c * c + d * d
            var real = Rational(a * c + b * d, denominator)
            var imag = Rational(b * c - a * d, denominator)
            for p in [7, 65]:
                for narrow in [False, True]:
                    for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                                 RoundingMode.toward_positive, RoundingMode.toward_negative,
                                 RoundingMode.away_from_zero]:
                        var rc = ArithmeticContext(
                            format=FloatFormat(p, emin=-3 if narrow else -1024, emax=4 if narrow else 1024),
                            rounding=mode,
                        )
                        var ic = ArithmeticContext(
                            format=FloatFormat(p + 1, emin=-3 if narrow else -1024, emax=4 if narrow else 1024),
                            rounding=mode,
                        )
                        var expected = Complex(Float(real, context=rc), Float(imag, context=ic))
                        var settings = ComplexContext(real=rc, imag=ic)
                        assert_equal(divide(left, right, context=settings), expected)
                        var status = Complex._calculate_status(left, right, 3, settings)
                        assert_equal(status.real(), _flags(real, rc))
                        assert_equal(status.imag(), _flags(imag, ic))
            assert_equal(left, saved)
            assert_equal(divide(left, left), Complex(1))
    assert_equal(divide(Rational(2, 3), Complex(5, -7)), Complex(Rational(5, 111), Rational(7, 111)))
    var signed_zero = divide(Complex(Float.zero(), Float.zero(negative=True)), Complex(1, 1))
    assert_true(signed_zero.is_zero() and not signed_zero.real().signbit() and signed_zero.imag().signbit())
    var wide = ArithmeticContext(format=FloatFormat(7))
    var tiny = ldexp(Float(1, context=wide), -(Int(1) << 40), context=wide)
    var result = divide(Complex(1), Complex(Float(1, context=wide), tiny), context=wide)
    assert_equal(result, Complex(Float(1, context=wide), -tiny))
    var status = Complex._calculate_status(Complex(1), Complex(Float(1, context=wide), tiny), 3, wide)
    assert_true(status.real().inexact() and status.imag().inexact())
    assert_equal(status.real().direction(), "above")
    assert_equal(status.imag().direction(), "below")


def test_complex_powers_against_exact_components() raises:
    for sign in [-1, 1]:
        var real = Integer(1)
        var imag = Integer(0)
        for count in range(1, 10):
            var next_real = real * 3 - imag * (sign * 4)
            imag = real * (sign * 4) + imag * 3
            real = next_real
            if count != 3 and count != 9:
                continue
            for inverse in [False, True]:
                var denominator = real * real + imag * imag if inverse else Integer(1)
                var r = Rational(real, denominator)
                var i = Rational(-imag if inverse else imag, denominator)
                for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                             RoundingMode.toward_positive, RoundingMode.toward_negative,
                             RoundingMode.away_from_zero]:
                    var c = ArithmeticContext(format=FloatFormat(7), rounding=mode)
                    var expected = Complex(Float(r, context=c), Float(i, context=c))
                    assert_equal(pow_int(Complex(3, sign * 4), -count if inverse else count, context=c), expected)


def scalar[T: ImplicitlyCopyable & Deinitable](value: Int) raises -> T:
    comptime if T == Integer:
        return rebind[T](Integer(value))
    elif T == Rational:
        return rebind[T](Rational(value))
    elif T == Float:
        return rebind[T](Float(value))
    else:
        comptime assert T == Complex
        return rebind[T](Complex(value))


def batch_contract[T: ImplicitlyCopyable & Deinitable & Writable & Equatable]() raises:
    for length in [0, 1, 8, 9]:
        var values = Batch[T]([scalar[T](i + 1) for i in range(length)])
        var saved = values[::-1]
        var selected = saved[::2]
        var expected = selected.to_json()
        var mask = values != 2
        assert_equal(mask.count(), length - (1 if length >= 2 else 0))
        assert_true((mask | ~mask).all())
        assert_true(not (mask & ~mask).any())
        assert_equal(len(values[mask]), mask.count())
        values[:] = saved
        values += 2
        values -= 2
        values *= 2
        for i in range(length):
            assert_equal(values[i], scalar[T](2 * (length - i)))
            assert_equal(saved[i], scalar[T](length - i))
        assert_equal(selected.to_json(), expected)
        assert_equal(Batch[T].from_json(expected).to_json(), expected)
        assert_equal(len(Batch[T].from_json(values[:0].to_json())), 0)
        assert_equal(len(values + 3), length)
        if length:
            values[-1] = scalar[T](99)
            assert_equal(values[length - 1], scalar[T](99))
        with assert_raises():
            _ = values[length]
    var values = Batch[T]([scalar[T](1), scalar[T](2), scalar[T](3)])
    var snapshot = values[:]
    values[1:] = values[:-1]
    assert_equal(values[1], scalar[T](1))
    assert_equal(values[2], scalar[T](2))
    values[Mask([True, False, True])] = scalar[T](7)
    assert_equal(values[0], scalar[T](7))
    assert_equal(values[2], scalar[T](7))
    assert_equal(snapshot[0], scalar[T](1))
    assert_equal(snapshot[2], scalar[T](3))
    var before = values.to_json()
    with assert_raises():
        values += Batch[T]([scalar[T](1)])
    assert_equal(values.to_json(), before)


def test_integer_batch_storage_contract() raises:
    batch_contract[Integer]()
    var values = Batch[Integer]([4, -1, 0, -3, 8])
    values[0] = 18446744073709551616
    assert_equal(String(values), "[18446744073709551616, -1, 0, -3, 8]")
    var accumulated = Integer(0)
    for value in values[::-1]:
        accumulated += value
    assert_equal(accumulated, (Integer(1) << 64) + 4)


def test_rational_batch_storage_contract() raises:
    batch_contract[Rational]()


def test_float_batch_storage_contract() raises:
    batch_contract[Float]()


def test_complex_batch_storage_contract() raises:
    batch_contract[Complex]()


def test_integer_batch_operators_and_failure_recovery() raises:
    var a = Batch[Integer]([-7, 0, 7, 1, 2, 3, 4, 5, 6])
    var b = Batch[Integer]([3, -3, -3, 2, 3, 4, 5, 6, 7])
    var q, r = vmap[div_rem_floor]()(a, b)
    var qt, rt = vmap[div_rem_trunc]()(a, b)
    var qe, re = vmap[div_rem_euclid]()(a, b)
    assert_true((q * b + r == a).all())
    assert_true((qt * b + rt == a).all())
    assert_true((qe * b + re == a).all())
    assert_true((re >= 0).all())
    assert_true((~a == -a - 1).all())
    assert_true(((a & ~a) == 0).all())
    assert_true(((a | ~a) == -1).all())
    assert_true(((a ^ a) == 0).all())
    assert_true((((a << 33) >> 33) == a).all())
    assert_equal(a.sign()[0], -1)
    assert_equal(a.magnitude_bit_length()[0], 3)
    var n = Batch[Integer]([13, 13, 13, 13, 13, 13, 13, 13, 13])
    n += 5
    n -= 3
    n *= 2
    n //= 4
    n %= 5
    n **= 3
    n <<= 2
    n >>= 1
    n &= 31
    n |= 1
    n ^= 3
    assert_true((n == 18).all())
    var snapshot = n[:]
    b[-1] = 0
    with assert_raises():
        n //= b
    assert_true((n == snapshot).all())
    b[-1] = 3
    n //= b
    assert_equal(n[-1], 6)
    assert_equal(snapshot[-1], 18)
    assert_true((axpy(3, a[::-1], a) == 3 * a[::-1] + a).all())


def test_float_batch_functions_and_traps() raises:
    # Batches apply the scalar Float functions through vmap.
    var a = Batch[Float]([Float(1), Float(4), Float(9)])
    assert_equal(vmap[float.sqrt]()(a)[2], Float(3))
    assert_equal(vmap[square]()(a)[2], Float(81))
    assert_equal(vmap[float.pow_int]()(a, -1)[1], Float(Rational(1, 4)))
    assert_equal(vmap[ldexp]()(a, -1)[0], Float(Rational(1, 2)))
    assert_equal(vmap[fma]()(a, 2, 1)[2], Float(19))
    assert_equal(vmap[float.abs]()(-a)[2], Float(9))
    assert_equal(vmap[float.divide]()(a, 3)[0], divide(Float(1), 3))
    assert_true(a.is_finite().all())
    var before = a.to_json()
    var trap = ArithmeticContext(trap_invalid=True)
    var failed = False
    try:
        a = vmap[float.sqrt]()(Batch[Float]([Float(4), Float(-1)]), context=trap)
    except error:
        failed = True
        assert_true("mapped index 1" in String(error))
    assert_true(failed)
    assert_equal(a.to_json(), before)
    a /= 2
    a **= 2
    assert_equal(a[2], Float(Rational(81, 4)))


def test_complex_batch_functions_and_traps() raises:
    # Batches apply the scalar Complex functions through vmap.
    var a = Batch[Complex]([Complex(3, 4), Complex(1, 2)])
    assert_equal(vmap[norm_sqr]()(a)[0], Float(25))
    assert_equal(vmap[complex_abs]()(a)[0], Float(5))
    assert_equal(vmap[complex.sqrt]()(a)[0], Complex(2, 1))
    assert_equal(vmap[complex.pow_int]()(a, 2)[0], Complex(-7, 24))
    assert_equal(vmap[complex.divide]()(a, 3)[0], divide(Complex(3, 4), 3))
    assert_equal(a.conjugate()[1], Complex(1, -2))
    assert_equal(a.real()[0], Float(3))
    assert_equal(a.imag()[1], Float(2))
    assert_true(a.is_finite().all())
    assert_equal((Batch[Rational]([Rational(1, 2), 1]) + a)[0], Complex(Rational(7, 2), 4))
    var before = a.to_json()
    with assert_raises():
        a = vmap[complex.sqrt]()(a, context=ArithmeticContext(trap_inexact=True))
    assert_equal(a.to_json(), before)
    a *= Complex(0, 1)
    a /= Complex(0, 1)
    a **= -1
    assert_equal(a[1], divide(Complex(1), Complex(1, 2)))


def test_exact_reductions_and_complex_conjugation() raises:
    var wide = Integer(1) << 129
    var a = Batch[Integer]([wide, 7, -wide])
    var b = Batch[Rational]([Rational(1, 3), Rational(2, 7), Rational(1, 3)])
    assert_equal(sum(a), 7)
    assert_equal(sum_sequential(a), 7)
    assert_equal(sum_tree(a), 7)
    assert_equal(prod(Batch[Integer]([2, -3, 4])), -24)
    assert_equal(amin(a), -wide)
    assert_equal(amax(a), wide)
    assert_equal(sum(b), Rational(20, 21))
    assert_equal(sum_sequential(b), Rational(20, 21))
    assert_equal(sum_tree(b), Rational(20, 21))
    assert_equal(prod(b), Rational(2, 63))
    assert_equal(dot(a, b), Rational(2))
    assert_equal(dot_sequential(a, b), Rational(2))
    assert_equal(sum(Batch[Integer]()), 0)
    assert_equal(prod(Batch[Rational]()), Rational(1))
    with assert_raises():
        _ = amin(Batch[Integer]())
    with assert_raises():
        _ = dot(a, b[:1])
    var z = Batch[Complex]([Complex(3, 4), Complex(1, 2)])
    assert_equal(dot(z, z), Complex(-10, 28))
    assert_equal(vdot(z, z), Complex(30))
    assert_equal(dot_sequential(z, z), Complex(-10, 28))
    assert_equal(sum_tree(z[::-1]), Complex(4, 6))


def test_ordered_reduction_rounding_is_explicit() raises:
    var c = ArithmeticContext(format=FloatFormat(3))
    var a = Batch[Float]([Float(8), Float(1), Float(1), Float(-8)])
    assert_equal(sum(a, context=c), Float(2))
    assert_equal(sum_sequential(a, context=c), Float(0))
    assert_equal(sum_tree(a, context=c), Float(1))
    var b = Batch[Float]([Float("1.25"), Float("-1.5")])
    var weights = Batch[Rational]([Rational(5, 4), Rational(1)])
    assert_equal(dot(b, weights, context=c), Float("0.0625"))
    assert_equal(dot_sequential(b, weights, context=c), Float(0))
    var z = Batch[Complex]([Complex(8, 8), Complex(1, 1), Complex(1, 1), Complex(-8, -8)])
    assert_equal(sum(z, context=c), Complex(2, 2))
    assert_equal(sum_sequential(z, context=c), Complex(0))
    assert_equal(sum_tree(z, context=c), Complex(1, 1))


def test_text_json_limits_and_malformed_input() raises:
    var integer = Integer(1) << 129
    var rational = Rational(-7, 3)
    var real = Float("-0x1.8p-3")
    var complex = Complex("3-4j")
    assert_equal(Integer.from_json(integer.to_json()), integer)
    assert_equal(Rational.from_json(rational.to_json()), rational)
    assert_equal(Float.from_json(real.to_json()), real)
    assert_equal(Complex.from_json(complex.to_json()), complex)
    assert_equal(Rational(rational.to_string()), rational)
    assert_equal(Float(real.to_string(16)), real)
    assert_equal(Float(real.to_string(), context=ArithmeticContext(format=real.format())), real)
    assert_equal(Complex(complex.to_string(16)), complex)
    with assert_raises():
        _ = Integer.from_json("{}")
    with assert_raises():
        _ = Rational.from_json("{}")
    with assert_raises():
        _ = Float.from_json("{}")
    with assert_raises():
        _ = Complex.from_json("{}")
    for base in [2, 8, 10, 16, 36]:
        var value = -(Integer(1) << 129) + 15
        assert_equal(Integer(value.to_string(base), base=base), value)
    for digits in [8, 9, 10, 18, 19, 26, 27, 28, 77, 309]:
        var value = Integer(10) ** digits - 1
        var text = value.to_string()
        assert_equal(Integer(text), value)
        assert_equal(Integer(String("-000", text)), -value)
        assert_equal(Integer(String("0_", text), allow_underscores=True), value)
        assert_equal(Integer.from_json(value.to_json()), value)
        with assert_raises():
            _ = Integer(text, limits=ConversionLimits(max_digits=digits - 1))
        with assert_raises():
            _ = Integer(text, limits=ConversionLimits(max_allocated_bytes=1))
    assert_equal(Float("123456789.125"), Float(Rational(987654313, 8)))
    assert_equal(Float("123456789."), Float(123456789))
    assert_equal(Float(".000000001"), Float(Rational(1, 1_000_000_000)))
    assert_equal(Integer("123_456_789_012_345_678_901", allow_underscores=True), 123456789012345678901)
    assert_equal(Float("12345678901234567890.125"), Float(Rational(98765431209876543121, 8)))
    var before = real
    with assert_raises():
        real = Float("12345678901234567890.125", limits=ConversionLimits(max_allocated_bytes=1))
    assert_equal(real, before)
    real = Float("12345678901234567890.125")
    assert_equal(real, Float(Rational(98765431209876543121, 8)))
    assert_equal(Integer(" -0xFF_FF ", base=0, allow_whitespace=True, allow_underscores=True), -65535)
    for text in ["", "1__0", "1.5", "0x10", " 1"]:
        with assert_raises():
            _ = Integer(text)
    for text in ["1/0", "1//2", "1/2/3"]:
        with assert_raises():
            _ = Rational(text)
    for text in ["1e", "0x1p", "--1"]:
        with assert_raises():
            _ = Float(text)
    var saved = Integer(17)
    try:
        saved = Integer("1234", limits=ConversionLimits(max_digits=3))
        assert_true(False)
    except error:
        assert_true("max_digits" in String(error))
    assert_equal(saved, 17)
    var values = Batch[Integer]([1, 2, 3])
    with assert_raises():
        _ = values.to_json(limits=ConversionLimits(max_values=2))
    assert_true(Float.from_json(Float.zero(negative=True).to_json()).signbit())
    assert_true(Float.from_json(Float.nan().to_json()).is_nan())


@fieldwise_init
struct _TensorPair[T: ImplicitlyCopyable & Deinitable](Copyable, Movable):
    var left: _Tensor[Self.T]
    var right: _Tensor[Self.T]


def _radix_values() raises -> List[Integer]:
    """Values at word, limb and chunk boundaries, and seeded multiword ones."""
    var values: List[Integer] = [Integer(0), Integer(1), Integer(-1), Integer(Int64.MIN), Integer(UInt64.MAX)]
    for bits in [31, 32, 33, 63, 64, 65, 127, 128, 129]:
        values.append((Integer(1) << bits) - 1)
        values.append(Integer(1) << bits)
        values.append(-((Integer(1) << bits) + 1))
    for base in [3, 7, 10, 36]:
        for k in [12, 19, 22, 40]:
            values.append(Integer(base) ** k - 1)
            values.append(Integer(base) ** k)
    var seed = UInt64(20260101)
    for bits in [95, 160, 1000, 4099]:
        var words = List[UInt32]()
        for _ in range((bits + 31) // 32):
            seed = seed * 6364136223846793005 + 1442695040888963407
            words.append(UInt32(seed >> 32))
        words[len(words) - 1] >>= UInt32(31 - (bits - 1) % 32)
        values.append(Integer._from_words(words^, bits % 2 == 1))
    return values^


def test_integer_text_in_every_base() raises:
    """Every base round-trips, with and without limits; power-of-two digits are the bits."""
    var counted = ConversionLimits(max_digits=10_000_000, max_allocated_bytes=1 << 40)
    for value in _radix_values():
        var binary = value.to_string(2)
        for base in range(2, 37):
            var text = value.to_string(base)
            assert_equal(value.to_string(base, limits=counted), text)
            assert_equal(Integer(text, base=base), value)
            assert_equal(Integer(text, base=base, limits=counted), value)
            assert_equal(value.to_string(base, uppercase=True), text.upper())
            if base == 4 or base == 8 or base == 16 or base == 32:
                # Each digit is the next group of bits, from the least significant.
                var shift = 2 if base == 4 else 3 if base == 8 else 4 if base == 16 else 5
                var bits = binary.removeprefix("-")
                var padded = String("0" * ((shift - bits.byte_length() % shift) % shift), bits)
                var expected = String("-") if value.sign() < 0 else String()
                for i in range(0, padded.byte_length(), shift):
                    var digit = Int(Integer(String(padded[byte=i:i + shift]), base=2))
                    expected += String(Integer(digit).to_string(base))
                assert_equal(text, expected)
        # The decimal budgeted path divides by 10**19 as _decimal does, separately.
        assert_equal(value.to_string(limits=counted), String(value))
    assert_equal(Integer(35).to_string(36), "z")
    assert_equal((Integer(36) ** 12 - 1).to_string(36), "z" * 12)
    assert_equal((Integer(36) ** 12).to_string(36, uppercase=True), "1" + "0" * 12)
    assert_equal((Integer(3) ** 40).to_string(3), "1" + "0" * 40)
    assert_equal(Integer(-255).to_string(16, prefix=True, uppercase=True), "-0XFF")
    assert_equal(Integer(0).to_string(2, prefix=True), "0b0")
    assert_equal(Integer(0).to_string(7, limits=counted), "0")
    assert_equal(Integer(-5).to_string(8, prefix=True, limits=counted), "-0o5")
    assert_equal(Integer("1_0000_0000_0000_0000", base=16, allow_underscores=True), Integer(1) << 64)
    assert_equal(Integer("-0b1_0", base=0, allow_underscores=True), -2)
    assert_equal(Integer("z_z", base=36, allow_underscores=True), 36 * 35 + 35)
    assert_equal(Integer("000000000000000000000000000000000001", base=16), 1)
    assert_equal(Integer("0" * 40 + "1" + "0" * 40, base=7), Integer(7) ** 40)
    # Digit and output limits are exact: the count fits, one less does not.
    var wide = (Integer(1) << 1000) - 1
    for base in [2, 16, 10, 36]:
        var text = wide.to_string(base)
        assert_equal(wide.to_string(base, limits=ConversionLimits(max_digits=text.byte_length())), text)
        assert_equal(wide.to_string(base, limits=ConversionLimits(max_output_bytes=text.byte_length())), text)
        with assert_raises(contains="max_digits"):
            _ = wide.to_string(base, limits=ConversionLimits(max_digits=text.byte_length() - 1))
        with assert_raises(contains="max_output_bytes"):
            _ = wide.to_string(base, limits=ConversionLimits(max_output_bytes=text.byte_length() - 1))
        with assert_raises(contains="max_allocated_bytes"):
            _ = wide.to_string(base, limits=ConversionLimits(max_allocated_bytes=text.byte_length()))
        with assert_raises(contains="max_allocated_bytes"):
            _ = Integer(text, base=base, limits=ConversionLimits(max_allocated_bytes=8))


def test_integer_bytes() raises:
    """Bytes are the magnitude in base 256, either way round; the sign is separate."""
    var two_bytes = Integer(0x0102).to_bytes()
    assert_true(len(two_bytes) == 2 and two_bytes[0] == 2 and two_bytes[1] == 1)
    var big = Integer(0x0102).to_bytes(big_endian=True)
    assert_true(len(big) == 2 and big[0] == 1 and big[1] == 2)
    assert_true(len(Integer(0).to_bytes()) == 0)
    var negative = Integer(-258).to_bytes()
    assert_true(len(negative) == 2 and negative[0] == 2 and negative[1] == 1)
    var padded: List[UInt8] = [0, 0, 1, 0]
    assert_equal(Integer.from_bytes(Span(padded)), 65536)
    assert_equal(Integer.from_bytes(Span(padded), big_endian=True), 256)
    var nothing = List[UInt8]()
    assert_equal(Integer.from_bytes(Span(nothing), negative=True), 0)
    var zeros: List[UInt8] = [0, 0, 0]
    assert_equal(Integer.from_bytes(Span(zeros), negative=True).sign(), 0)
    var top: List[UInt8] = [0, 0, 0, 0, 0, 0, 0, 0x80]
    assert_equal(Integer.from_bytes(Span(top), negative=True), Integer(Int64.MIN))
    assert_equal(Integer.from_bytes(Span(top)), Integer(1) << 63)
    for value in _radix_values():
        var bytes = value.to_bytes()
        assert_equal(len(bytes), (value.magnitude_bit_length() + 7) // 8)
        assert_equal(Integer.from_bytes(Span(bytes), negative=value.sign() < 0), value)
        var reversed = value.to_bytes(big_endian=True)
        assert_equal(Integer.from_bytes(Span(reversed), negative=value.sign() < 0, big_endian=True), value)
        for i in range(len(bytes)):
            assert_true(bytes[i] == reversed[len(bytes) - 1 - i])
        # Each byte is two hexadecimal digits.
        if len(bytes):
            var hex = String()
            for i in range(len(reversed)):
                var pair = Integer(Int(reversed[i])).to_string(16)
                hex += pair if i == 0 or pair.byte_length() == 2 else String("0", pair)
            assert_equal(hex, abs(value).to_string(16))


def _tensor_call[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, T) raises thin -> R,
](job: _TensorPair[T], index: Int) raises -> R:
    return function(job.left._read(index), job.right._read(index))


def _tensor_map[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, T) raises thin -> R,
](left: _Tensor[T], right: _Tensor[T], diagnostic: String = "Batch") raises -> _Tensor[R]:
    return _map_tensor[_tensor_call[function]](
        _TensorPair(left, right), left._layout.size, diagnostic=diagnostic, index_prefix=" element ",
    )


def _tensor_add(a: Int, b: Int) raises -> Int:
    return a + b


def _tensor_divide(a: Int, b: Int) raises -> Int:
    if b == 0:
        raise Error("Cannot divide by zero; use a nonzero divisor.")
    return a // b


@fieldwise_init
struct _CopyTracked(ImplicitlyCopyable):
    var value: Int
    var copies: ArcPointer[Int]

    def __init__(out self, *, copy: Self):
        self.value = copy.value
        self.copies = copy.copies
        self.copies[] += 1


def _positions(layout: _FlatLayout) -> List[Int]:
    var result = List[Int]()
    var cursor = layout.cursor()
    for _ in range(layout.size):
        result.append(cursor.position)
        cursor.advance()
    return result^


def test_tensor_layout_broadcast_snapshot_and_transaction() raises:
    # Storage is one strided run; lifting reads elements in place without copies.
    var copies = ArcPointer[Int](0)
    var tracked = _Tensor[_CopyTracked]([_CopyTracked(1, copies), _CopyTracked(2, copies)], [2])
    var before_copies = copies[]
    def add_values(a: _CopyTracked, b: _CopyTracked) raises -> Int:
        return a.value + b.value
    for view in [tracked, tracked.slice(0, StridedSlice(None, None, -1))]:
        assert_equal(_tensor_map[add_values](view, view).item(0), view._read(0).value * 2)
    assert_equal(copies[], before_copies)

    var columns = _Tensor[Int]([10, 20, 30], [3])
    var reversed = columns.slice(0, StridedSlice(None, None, -1))
    assert_equal(reversed.to_list(), [30, 20, 10])
    assert_equal(_tensor_map[_tensor_add](columns, reversed).to_list(), [40, 40, 40])
    assert_equal(reversed.slice(-1, StridedSlice(1, None, 1)).item(-1), 10)
    def forbidden_binary(a: Int, b: Int) raises -> Int:
        raise Error("The callback must not run.")
    var empty = _Tensor[Int]([], [0])
    assert_equal(_tensor_map[forbidden_binary](empty, empty)._layout.size, 0)
    var shape_failed = False
    try:
        _ = Batch[Float]([Float(10), Float(20), Float(30)]) + Batch[Float]([Float(0), Float(0)])
    except error:
        shape_failed = True
        assert_true("batches" in String(error) and "lengths" in String(error))
    assert_true(shape_failed)
    var numerators = _Tensor[Int]([8, 4, 0], [3])
    var divisors = _Tensor[Int]([2, 0, 0], [3])
    for view in [numerators, numerators.slice(0, StridedSlice(None, None, -1))]:
        for label in [String("Checked"), String("")]:
            var destination = _Tensor[Int]([99], [1])
            var failed = False
            try:
                destination = _tensor_map[_tensor_divide](view, divisors, diagnostic=label)
            except error:
                failed = True
                assert_equal(String(error), label + " element 1: Cannot divide by zero; use a nonzero divisor.")
            assert_true(failed)
            assert_equal(destination.to_list(), [99])
    assert_equal(numerators.to_list(), [8, 4, 0])
    assert_equal(divisors.to_list(), [2, 0, 0])
    with assert_raises():
        _ = _Tensor[Int]([1], [2])
    with assert_raises():
        _ = columns.slice(1, StridedSlice(None, None, 1))

    # Shapes are runtime data: one layout type serves every rank.
    var matrix = _FlatLayout([2, 3])
    assert_equal(matrix.strides, [3, 1])
    var transposed = matrix.transposed([1, 0])
    assert_equal(transposed.shape, [3, 2])
    assert_equal(_positions(transposed), [0, 3, 1, 4, 2, 5])
    assert_true(not transposed.contiguous)
    assert_true(not transposed.progression())
    var reverse = transposed.sliced(0, StridedSlice(None, None, -1)).sliced(1, StridedSlice(None, None, -1))
    assert_equal(_positions(reverse), [5, 2, 4, 1, 3, 0])
    assert_equal(reverse.at(-1, -1).position(2), 0)
    assert_equal(_positions(reverse.reversed_axes()), [5, 4, 3, 2, 1, 0])
    assert_equal(reverse.reversed_axes().progression().value(), -1)
    var last_row = matrix.at(0, -1)
    assert_equal(last_row.shape, [3])
    assert_equal(_positions(last_row), [3, 4, 5])
    assert_equal(last_row.progression().value(), 1)
    assert_equal(_positions(matrix.sliced(1, StridedSlice(None, None, 2))), [0, 2, 3, 5])
    var column = _FlatLayout([3]).broadcast_to([2, 3])
    assert_equal(column.strides, [0, 1])
    assert_equal(_positions(column), [0, 1, 2, 0, 1, 2])
    var singleton = _FlatLayout([2, 1]).broadcast_to([2, 3])
    assert_equal(_positions(singleton), [0, 0, 0, 1, 1, 1])
    var scalar = _FlatLayout(List[Int]())
    assert_equal(scalar.size, 1)
    assert_equal(scalar.ndim(), 0)
    assert_equal(_positions(scalar.broadcast_to([2, 3])), [0, 0, 0, 0, 0, 0])
    assert_equal(_broadcast_shapes([2, 1], [3]), [2, 3])
    assert_equal(_broadcast_shapes(List[Int](), [2, 3]), [2, 3])
    assert_equal(_broadcast_shapes([0, 3], [1, 3]), [0, 3])
    assert_equal(matrix.reshaped([3, 2]).strides, [2, 1])
    assert_equal(_FlatLayout([Int.MAX, 0, Int.MAX]).size, 0)
    assert_true(_FlatLayout([Int.MAX, 0, Int.MAX]).contiguous)
    with assert_raises():
        _ = _FlatLayout([Int.MAX, 2])
    with assert_raises():
        _ = _FlatLayout([-1, 0])
    with assert_raises():
        _ = _broadcast_shapes([0], [2])
    with assert_raises():
        _ = _FlatLayout([2]).broadcast_to([3])
    with assert_raises():
        _ = matrix.transposed([0, 0])
    with assert_raises():
        _ = matrix.sliced(2, StridedSlice(None, None, 1))
    with assert_raises():
        _ = matrix.sliced(0, StridedSlice(None, None, 0))
    with assert_raises():
        _ = _FlatLayout([0, 3]).at(0, 0)
    with assert_raises():
        _ = matrix.reshaped([7])


def _tensor_polynomial(x: Integer) raises -> Integer:
    var result = x * x
    if x < 0:
        return -result
    for i in range(3):
        result += i
    return result


def _tensor_pair(x: Integer) raises -> Tuple[Integer, Bool]:
    return (_tensor_polynomial(x), x > 0)


def _tensor_return(x: Integer) raises -> Batch[Integer]:
    return Batch[Integer]([x, x * x])


def _tensor_bad_shape(x: Integer) raises -> Batch[Integer]:
    if x == 0:
        return Batch[Integer]([x])
    return Batch[Integer]([x, x])


def _tensor_failure(x: Integer) raises -> Integer:
    if x <= 0:
        raise Error("Use a positive input.")
    return x


def test_tensor_mapping_callables_shapes_and_errors() raises:
    var values = Batch[Integer]([-2, 0, 3])
    var mapped = vmap[_tensor_polynomial](in_axes=-1, out_axes=-1)
    var answer = mapped(values)
    for i in range(3):
        assert_equal(answer[i], _tensor_polynomial(values[i]))
    def shifted(x: Integer, shift: Integer) raises -> Integer:
        return _tensor_polynomial(x) + shift
    # A shared value is passed as an argument; scalars are shared by every call.
    assert_equal(vmap[shifted]()(values, Integer(17))[2], 29)
    var pairs = vmap[_tensor_pair]()(values)
    assert_equal(pairs[0][0], -4)
    assert_equal(pairs[1][2], True)
    var tensors = vmap[_tensor_return](out_axes=-1)(values)
    assert_equal(tensors.shape(), [2, 3])
    assert_equal(tensors[1, 2], 9)
    var matrix = Batch[Integer]([1, 2, 3, 4], shape=[2, 2])
    var twice = vmap[_tensor_polynomial]().vmap(in_axes=1, out_axes=1)
    var nested: Batch[Integer] = twice(matrix)
    for i in range(2):
        for j in range(2):
            assert_equal(nested[i, j], _tensor_polynomial(matrix[i, j]))
    assert_equal(twice(matrix).to_list(), nested.to_list())
    var three = vmap[_tensor_polynomial]().vmap().vmap()
    assert_equal(three(matrix.reshape([1, 2, 2]))[0, 1, 1], 19)
    var empty = Batch[Integer]()
    # This callback would fail on zero: empty mapping must never probe it.
    assert_equal(len(vmap[_tensor_failure]()(empty)), 0)
    var empty_tensors = vmap[_tensor_return](out_shape=[2])(empty)
    assert_equal(empty_tensors.shape(), [0, 2])
    with assert_raises():
        _ = vmap[_tensor_return]()(empty)
    with assert_raises():
        _ = vmap[_tensor_bad_shape]()(values)
    with assert_raises():
        _ = vmap[_tensor_failure]()(values)
    try:
        _ = vmap[_tensor_failure]()(values)
    except error:
        assert_true(String(error).__contains__("mapped index 0"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_equal(values[0], -2)
    var x = Batch[Integer]([1, 2, 3])
    def add(a: Integer, b: Integer) raises -> Integer:
        return a + b
    assert_equal(vmap[add](in_axes=(0, None))(x, Integer(7))[2], 10)
    with assert_raises():
        _ = vmap[add]()(x, x[:1])
    with assert_raises():
        _ = vmap[_tensor_polynomial](out_axes=1)(x)


def _tensor_family_product[T: ImplicitlyCopyable & Deinitable](x: T, y: T) raises -> T:
    comptime if T == Integer:
        return rebind[T](rebind[Integer](x) * rebind[Integer](y))
    elif T == Rational:
        return rebind[T](rebind[Rational](x) * rebind[Rational](y))
    elif T == Float:
        return rebind[T](rebind[Float](x) * rebind[Float](y))
    else:
        return rebind[T](rebind[Complex](x) * rebind[Complex](y))


def _check_tensor_family[T: ImplicitlyCopyable & Deinitable & Equatable & Writable](values: List[T]) raises:
    comptime assert T == Integer or T == Rational or T == Float or T == Complex
    comptime assert _Result[T].Output == Batch[T]
    var tensor = _Tensor[T](values.copy(), [len(values)])
    var result = _tensor_map[_tensor_family_product[T]](tensor, tensor)
    def square_value(x: T) raises -> T:
        return _tensor_family_product(x, x)
    assert_equal(rebind_var[Batch[T]](vmap[square_value]()(Batch[T](_tensor=tensor))).to_list(), result.to_list())
    for i in range(len(values)):
        assert_equal(result.item(i), _tensor_family_product(values[i], values[i]))
        assert_equal(tensor.item(i), values[i])


def test_tensor_executor_uses_existing_scalar_families() raises:
    var wide = (Integer(1) << 1024) + 7
    _check_tensor_family[Integer]([Integer(3), -wide, Integer(0)])
    _check_tensor_family[Rational]([Rational(3, 7), Rational(-wide, wide + 1), Rational(0)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    _check_tensor_family[Float]([Float(3), Float(wide - 8, context=context), Float(-7)])
    _check_tensor_family[Complex]([Complex(1, 2), Complex(-7, 3), Complex(0)])


def _check_public_ranked_family[T: ImplicitlyCopyable & Deinitable & Equatable & Writable](
    values: List[T],
) raises:
    var matrix = Batch[T](values, shape=[2, 2])
    assert_equal(matrix.ndim(), 2)
    assert_equal(matrix.shape(), [2, 2])
    assert_equal(matrix.size(), 4)
    assert_equal(len(matrix), 4)
    assert_equal(matrix[3], values[3])
    var shape = matrix.shape()
    shape[0] = 9
    assert_equal(shape[0], 9)
    assert_equal(matrix.shape()[0], 2)
    assert_equal(matrix[-1, -1], values[3])
    assert_equal(matrix.at(0, -1)[-1], values[3])
    var transposed = matrix.transpose()
    assert_equal(transposed[0, 1], values[2])
    assert_equal(transposed.at(-1, -1)[0], values[2])
    assert_equal(matrix.slice(1, step=-1)[0, 0], values[1])
    var reversed = transposed.slice(0, step=-1).slice(-1, step=-1)
    assert_equal(reversed.to_list(), [values[3], values[1], values[2], values[0]])
    var shared = matrix.reshape([4])
    assert_equal(Int(shared._tensor_view()._owner.ptr()), Int(matrix._tensor_view()._owner.ptr()))
    var materialized = transposed.reshape([4])
    assert_true(Int(materialized._tensor_view()._owner.ptr()) != Int(matrix._tensor_view()._owner.ptr()))
    assert_equal(materialized.to_list(), [values[0], values[2], values[1], values[3]])
    # Views of any layout share storage; only flat kernels gather.
    var copied = Batch[T](reversed)
    assert_equal(Int(copied._owner.ptr()), Int(matrix._owner.ptr()))
    assert_equal(Int(transposed._owner.ptr()), Int(matrix._owner.ptr()))
    assert_equal(Int(matrix.at(0, 1)._tensor_view()._owner.ptr()), Int(matrix._tensor_view()._owner.ptr()))
    var snapshot = copied
    copied[1, 1] = values[3]
    assert_equal(copied[1, 1], values[3])
    assert_equal(snapshot[1, 1], values[0])
    assert_equal(matrix[0, 0], values[0])
    # Iteration visits elements in row-major order, like Mask.
    var cursor = iter(matrix)
    var second_cursor = cursor
    assert_equal(rebind_var[T](next(cursor)), values[0])
    assert_equal(rebind_var[T](next(second_cursor)), values[0])
    assert_equal(rebind_var[T](next(cursor)), values[1])
    matrix[0, 0] = values[3]
    assert_equal(shared[0], values[0])
    assert_equal(matrix.to_list(), [values[3], values[1], values[2], values[3]])
    for i in range(matrix.shape()[0]):
        assert_equal(matrix.at(0, i).shape(), [2])
    var cube = matrix.reshape([1, 2, 2])
    assert_equal(cube.at(0, 0).at(0, 0)[1], values[1])
    assert_equal(cube[0, 1, 1], values[3])
    var plane = cube.at(0, 0)
    assert_equal(plane.shape(), [2, 2])
    assert_equal(plane[0, 1], values[1])
    var scalar = Batch[T]([values[0]], shape=[])
    assert_equal(scalar.ndim(), 0)
    assert_equal(scalar.shape(), [])
    assert_equal(scalar.size(), 1)
    assert_equal(scalar[()], values[0])
    assert_equal(String(scalar), String(values[0]))
    var scalar_snapshot = scalar.transpose()
    scalar[()] = values[3]
    assert_equal(scalar.item(), values[3])
    assert_equal(scalar_snapshot.item(), values[0])
    assert_equal(scalar_snapshot.reshape([1])[0], values[0])
    assert_equal(shared.at(0, -1).item(), values[3])
    var old = Batch[T](values, shape=[4])
    assert_equal(old.size(), 4)
    assert_equal(old.ndim(), 1)
    assert_equal(old.transpose().to_list(), values)
    assert_equal(old.slice(0, step=-1)[0], values[3])
    assert_equal(old[1:2].item(), values[1])
    assert_equal(old.reshape([2, 2]).to_list(), values)
    var packed_empty = Batch[T]([], shape=[2, 0])
    assert_equal(packed_empty.size(), 0)
    assert_equal(len(packed_empty), 0)
    assert_equal(packed_empty.shape()[0], 2)
    assert_equal(String(packed_empty), "[[], []]")
    assert_equal(packed_empty.at(0, 1).size(), 0)
    assert_equal(packed_empty.transpose().shape(), [0, 2])
    assert_equal(packed_empty.reshape([0]).size(), 0)
    assert_equal(Batch[T]().shape(), [0])
    var before = matrix.to_list()
    var owner = matrix._tensor_view()._owner
    with assert_raises():
        matrix[0, 2] = values[0]
    assert_equal(Int(matrix._tensor_view()._owner.ptr()), Int(owner.ptr()))
    assert_equal(matrix.to_list(), before)
    with assert_raises():
        _ = matrix.item()
    with assert_raises():
        _ = old.item()
    with assert_raises():
        _ = matrix.reshape([3])
    with assert_raises():
        _ = matrix.transpose([0, 0])
    with assert_raises():
        _ = matrix.at(2, 0)
    with assert_raises():
        _ = matrix.at(0, -3)
    with assert_raises():
        _ = matrix.slice(0, step=0)
    with assert_raises():
        _ = matrix[0, 0, 0]
    with assert_raises():
        _ = Batch[T](values, shape=[1, 2])
    with assert_raises():
        _ = Batch[T](values, shape=[-2, -2])
    with assert_raises():
        _ = Batch[T](values, shape=[Int.MAX, 2])


def test_public_ranked_construction_views_and_updates() raises:
    var wide = (Integer(1) << 1024) + 7
    _check_public_ranked_family[Integer]([0, 1, -wide, wide])
    _check_public_ranked_family[Rational]([Rational(0), Rational(1, 3), Rational(-wide, 7), Rational(wide, 9)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    _check_public_ranked_family[Float]([Float.zero(negative=True), Float(1), Float(-wide, context=context), Float(wide, context=context)])
    _check_public_ranked_family[Complex]([Complex(0), Complex(1, 2), Complex(-wide, 3, context=context), Complex(0, wide, context=context)])
    var integers = Batch[Integer].from_native([1, 2, 3, 4, 5, 6], shape=[2, 3])
    assert_equal(integers.to_list(), Batch[Integer].from_iterable(range(1, 7)).to_list())
    assert_equal(Batch[Rational].from_iterable(range(6), shape=[2, 3])[1, 2], Rational(5))
    assert_equal(Batch[Float].from_native([UInt64.MAX], shape=[1, 1]).item(), Float(UInt64.MAX))
    assert_equal(Batch[Complex].from_native([Float64(1.5)], shape=[1, 1]).item(), Complex(Float64(1.5)))
    var flat = Batch[Integer]([1, 2, 3, 4, 5, 6])
    var cursor = iter(flat)
    _ = next(cursor)
    _ = next(cursor)
    assert_equal(Batch[Integer].from_iterable(cursor^, shape=[2, 2]).to_list(), flat[2:].to_list())
    assert_equal(Batch[Integer]([], shape=[Int.MAX, 0, Int.MAX]).size(), 0)
    assert_equal(String(integers), "[[1, 2, 3], [4, 5, 6]]")
    var floating = Batch[Float]([Float(-1, context=context), Float.zero(negative=True), Float(3), Float(4)])
    floating._real_format = context.format()
    var packed = Batch[Float](floating[:])
    assert_equal(packed.reshape([2, 2]).reshape([4]).to_json(), floating.to_json())
    var empty_float = floating[:0].reshape([2, 0])
    assert_equal(empty_float.at(0, 1).to_json(), floating[:0].to_json())
    assert_equal(empty_float.reshape([0]).to_json(), floating[:0].to_json())
    var complex = Batch[Complex]([Complex(1, 2)])
    complex._real_format = FloatFormat(256)
    complex._imag_format = FloatFormat(1024)
    var empty_complex = complex[:0].reshape([0, 2]).transpose()
    assert_equal(empty_complex.at(0, 0).to_json(), complex[:0].to_json())
    assert_equal(empty_complex.at(0, 1).to_json(), complex[:0].to_json())
    assert_equal(len(empty_complex), 0)


def _check_ranked_arithmetic[T: ImplicitlyCopyable & Deinitable & Equatable & Writable](values: List[T]) raises:
    var matrix = Batch[T](values, shape=[2, 2])
    var row = Batch[T]([values[0], values[1]])
    var flat = Batch[T](values)
    var expanded = Batch[T]([values[0], values[1], values[0], values[1]])
    var added = matrix + row
    comptime assert type_of(added) == Batch[T]
    assert_equal(added.to_list(), (flat + expanded).to_list())
    assert_equal(rebind_var[List[T]]((row[:] - matrix).to_list()), (expanded - flat).to_list())
    assert_equal(rebind_var[List[T]]((matrix * row[:]).to_list()), (flat * expanded).to_list())
    assert_equal(String((row / matrix).to_list()), String((expanded / flat).to_list()))
    assert_equal(String((matrix / row).to_list()), String((flat / expanded).to_list()))
    assert_equal((matrix == row).to_list(), (flat == expanded).to_list())
    assert_equal((row != matrix).to_list(), (expanded != flat).to_list())
    assert_equal((matrix == row).shape(), [2, 2])
    comptime if T != Complex:
        assert_equal((matrix < row).to_list(), (flat < expanded).to_list())
        assert_equal((row <= matrix).to_list(), (expanded <= flat).to_list())
        assert_equal((matrix > row).to_list(), (flat > expanded).to_list())
        assert_equal((row >= matrix).to_list(), (expanded >= flat).to_list())
    # Layout traversal is type-independent; qualify these axes once.
    comptime if T == Integer:
        var reversed = matrix.transpose().slice(0, step=-1)
        var square = reversed * reversed
        for i in range(2):
            for j in range(2):
                assert_equal(rebind[T](square[i, j]), _tensor_family_product(reversed[i, j], reversed[i, j]))
        var cube = matrix.reshape([2, 1, 2]) + row.reshape([1, 2])
        assert_equal(cube.shape(), [2, 1, 2])
        assert_equal(rebind_var[List[T]](cube.to_list()), (flat + expanded).to_list())
        var scalar = Batch[T]([values[0]], shape=[])
        var constant = Batch[T]([values[0], values[0], values[0], values[0]])
        assert_equal(rebind_var[List[T]]((matrix * scalar).to_list()), (flat * constant).to_list())
        assert_equal(rebind_var[List[T]]((scalar * matrix).to_list()), (constant * flat).to_list())
        assert_equal(rebind_var[T]((scalar * scalar).item()), _tensor_family_product(values[0], values[0]))
        assert_equal((scalar == scalar).shape(), [])
        assert_true((scalar == scalar).all())
        var row_expected = List[T]()
        for i in range(2):
            row_expected.append(_tensor_family_product(values[0], row[i]))
        assert_equal(rebind_var[List[T]]((scalar * row).to_list()), row_expected)
        assert_equal(rebind_var[List[T]]((row * scalar).to_list()), row_expected)
        assert_equal((scalar == row).shape(), [2])
        assert_equal((Batch[T]([], shape=[0, 2]) * row).shape(), [0, 2])
        assert_equal((Batch[T]([], shape=[Int.MAX, 0, Int.MAX]) == scalar).size(), 0)
        var snapshot = matrix[:]
        var result = matrix + row
        matrix[0, 0] = values[3]
        assert_equal(snapshot.to_list(), values)
        assert_equal(rebind_var[List[T]](result.to_list()), (flat + expanded).to_list())


def test_ranked_arithmetic_broadcast_promotions_and_masks() raises:
    var wide = (Integer(1) << 1024) + 17
    _check_ranked_arithmetic[Integer]([1, -3, wide, -wide])
    _check_ranked_arithmetic[Rational]([Rational(1, 3), Rational(-2, 7), Rational(wide, 11), Rational(-wide, 9)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    _check_ranked_arithmetic[Float]([Float(1), Float(-3), Float(wide, context=context), Float(-wide, context=context)])
    _check_ranked_arithmetic[Complex]([Complex(1, 2), Complex(-3, 1), Complex(wide, 1, context=context), Complex(1, -wide, context=context)])
    var a = Batch[Integer].from_native([1, 2, 3, 4], shape=[2, 2])
    var mapped = vmap[_tensor_family_product[Integer]]().vmap()
    assert_equal((a * a).to_list(), mapped(a, a).to_list())
    var halves: Batch[Rational] = a / 2
    assert_equal(halves[0, 0], Rational(1, 2))
    assert_equal((2 / a)[1, 0], Rational(2, 3))
    assert_equal((a + 18446744073709551616)[0, 0], (Integer(1) << 64) + 1)
    assert_equal((18446744073709551616 - a)[0, 0], Integer(UInt64.MAX))
    assert_equal((a * UInt64.MAX)[1, 1], Integer(UInt64.MAX) * 4)
    assert_equal((UInt64.MAX + a)[0, 0], Integer(1) << 64)
    assert_equal((Int64.MIN - a)[0, 0], Integer(Int64.MIN) - 1)
    assert_equal((a + Rational(1, 3))[0, 1], Rational(7, 3))
    assert_equal((Rational(1, 3) - a)[0, 1], Rational(-5, 3))
    assert_equal((Float(1) / a)[1, 1], Float("0.25"))
    assert_equal((a + Float64(0.5))[0, 0], Float("1.5"))
    assert_equal((Complex(1, 2) - a)[0, 1], Complex(-1, 2))
    assert_equal((a * Complex(1, 2))[0, 1], Complex(2, 4))
    assert_equal((Integer(3) < a).to_list(), [False, False, False, True])
    assert_equal((Rational(3, 2) >= a).to_list(), [True, False, False, False])
    assert_equal((Float(2) <= a).to_list(), [False, True, True, True])
    assert_equal((Complex(2) == a).to_list(), [False, True, False, False])
    assert_equal((a > UInt64.MAX).count(), 0)
    var rational = Batch[Rational]([Rational(1, 3), Rational(2, 3)], shape=[2, 1])
    assert_equal((a + rational)[1, 0], Rational(11, 3))
    var floats = Batch[Float]([Float(1, context=context), Float(2)])
    var mixed: Batch[Float] = a + floats
    assert_equal(mixed[0, 0].precision(), 1024)
    assert_equal(mixed[0, 1].precision(), 128)
    var complex = Batch[Complex]([Complex(1, 1), Complex(2, -1)])
    assert_equal((mixed * complex)[1, 1], mixed[1, 1] * complex[1])
    var exact = Batch[Integer]([wide], shape=[1, 1])
    var rounded = Float(wide)
    assert_true((exact > rounded).all())
    assert_true((rounded < exact).all())
    assert_true((exact + -rounded == Integer(17)).all())
    var specials = Batch[Float]([Float.nan(), Float.zero(negative=True)], shape=[1, 2])
    assert_equal((specials == Float(0)).to_list(), [False, True])
    assert_equal((specials != Float(0)).to_list(), [True, False])
    assert_equal((specials < Float(1)).to_list(), [False, True])


def test_ranked_arithmetic_formats_errors_and_publication() raises:
    var context = ArithmeticContext(format=FloatFormat(256, emin=-100, emax=100))
    var floats = Batch[Float]([Float(1, context=context)])
    floats._real_format = context.format()
    var empty = floats[:0].reshape([0, 2])
    var result = empty + Integer(1)
    assert_equal(result.reshape([0]).to_json(), floats[:0].to_json())
    var scalar = Batch[Float]([Float(3, context=context)], shape=[])
    assert_equal((empty * scalar).reshape([0]).to_json(), floats[:0].to_json())
    with assert_raises():
        _ = empty + Float(1)
    var complex = Batch[Complex]([Complex(1, 2)])
    complex._real_format = FloatFormat(256)
    complex._imag_format = FloatFormat(1024)
    assert_equal((complex[:0].reshape([2, 0]) + 1).reshape([0]).to_json(), complex[:0].to_json())
    with assert_raises():
        _ = Integer(1) < complex[:0].reshape([2, 0])
    var a = Batch[Integer].from_native([1, 2, 3, 4], shape=[2, 2])
    var before = a.to_list()
    var destination = a / 1
    var owner = destination._tensor_view()._owner
    try:
        destination = a / Batch[Integer]([1, 0])
        assert_true(False)
    except error:
        assert_true(String(error).__contains__("element 1"))
        assert_true(String(error).__contains__("zero"))
    assert_equal(a.to_list(), before)
    assert_equal(Int(destination._tensor_view()._owner.ptr()), Int(owner.ptr()))
    assert_equal(destination[1, 1], Rational(4))
    try:
        _ = a / Batch[Integer]([0, 0, 0])
        assert_true(False)
    except error:
        assert_true(String(error).__contains__("Incompatible batch shapes"))
        assert_true(not String(error).__contains__("element"))
    assert_equal((a / Batch[Integer]([2]))[0, 0], Rational(1, 2))
    # Vector-only length-one sequences keep their existing no-broadcast rule.
    with assert_raises():
        _ = a.reshape([4]) + Batch[Integer]([1])
    var mismatch = Batch[Float]([Float(1), Float(1, context=context)], shape=[1, 2])
    try:
        _ = mismatch + Batch[Float]([Float(1), Float(1)])
        assert_true(False)
    except error:
        assert_true(String(error).__contains__("element 1"))
        assert_true(String(error).__contains__("FormatMismatch"))


def test_shaped_functions_broadcasting_masks_json_and_axis_reductions() raises:
    # Elementwise operators keep the operand shape; nested vmap maps a
    # scalar function over every axis.
    var m = Batch[Integer]([1, -2, 3, -4, 5, -6], shape=[2, 3])
    for result in [-m, vmap[integer.abs]().vmap()(m), m.sign(), ~m, m.magnitude_bit_length(), vmap[gcd]().vmap()(m, 4), m // 2, m % 4, m ** 2]:
        assert_equal(result.shape(), [2, 3])
    assert_equal((-m)[1, 2], 6)
    var quotient, remainder = vmap[div_rem_floor]().vmap()(m, 4)
    assert_equal(quotient.shape(), [2, 3])
    assert_equal(remainder[0, 1], 2)
    var r = Batch[Rational]([Rational(1, 2), Rational(-3, 2), Rational(5), Rational(7, 3)], shape=[2, 2])
    assert_equal(r.floor().shape(), [2, 2])
    assert_equal(r.floor()[0, 1], -2)
    var f = Batch[Float]([Float(1), Float(4), Float(9), Float(16)], shape=[2, 2])
    assert_equal(vmap[float.sqrt]().vmap()(f).shape(), [2, 2])
    assert_equal(vmap[float.sqrt]().vmap()(f)[1, 1], Float(4))
    assert_equal(f.is_zero().shape(), [2, 2])
    var c = Batch[Complex]([Complex(3, 4), Complex(0, 1)], shape=[1, 2])
    assert_equal(vmap[complex_abs]().vmap()(c).shape(), [1, 2])
    assert_equal(c.real()[0, 0], Float(3))
    with assert_raises():
        _ = vmap[gcd]().vmap()(m, m.reshape([3, 2]))

    # Operators broadcast through the family kernels; vmap shares an operand with in_axes.
    var row = Batch[Integer]([10, 20, 30])
    var column = Batch[Integer]([100, 200], shape=[2, 1])
    assert_equal((m + row)[1, 2], 24)
    assert_equal((m * column)[1, 0], -800)
    var floats = Batch[Float]([Float(1), Float(2), Float(3), Float(4), Float(5), Float(6)], shape=[2, 3])
    var frow = Batch[Float]([Float(1), Float(2), Float(3)])
    var context = ArithmeticContext(format=FloatFormat(64))
    var total = vmap[float.add]().vmap(in_axes=(0, None))(floats, frow, context=context)
    assert_equal(total.shape(), [2, 3])
    assert_equal(total[1, 2], Float(9))
    assert_equal(total[1, 2].precision(), 64)
    with assert_raises():
        _ = m // row

    # Compound updates broadcast their source but never expand the destination.
    var destination = m
    destination += row
    assert_equal(destination[1, 2], 24)
    destination *= column
    assert_equal(destination[0, 0], 1100)
    assert_equal(destination.shape(), [2, 3])
    var exponents = f
    exponents **= Batch[Integer]([2, 0])
    assert_equal(exponents[1, 0], Float(81))
    var vector = row
    with assert_raises():
        vector += m
    assert_equal(vector.to_list(), row.to_list())

    # A shaped mask selects row-major values; masks never broadcast.
    var positive = m > 0
    assert_equal(m[positive].to_list(), [Integer(1), Integer(3), Integer(5)])
    assert_equal(m.transpose()[positive.transpose()].to_list(), [Integer(1), Integer(5), Integer(3)])
    var cleared = m
    cleared[positive] = Integer(0)
    assert_equal(cleared.shape(), [2, 3])
    assert_equal(cleared.to_list(), [Integer(0), Integer(-2), Integer(0), Integer(-4), Integer(0), Integer(-6)])
    with assert_raises():
        _ = m[Mask([True, False, True, False, True, False])]

    # Version 2 records carry the shape; vectors keep version 1.
    var text = m.to_json()
    assert_true(text.startswith('{"version":2,"family":"integer-batch","shape":["2","3"]'))
    assert_equal(Batch[Integer].from_json(text).shape(), [2, 3])
    assert_true(row.to_json().startswith('{"version":1,'))
    assert_equal(Batch[Integer].from_json(m.transpose().to_json())[2, 1], -6)
    assert_equal(Batch[Integer].from_json(m.to_json(limits=ConversionLimits(max_values=6))).to_list(), m.to_list())
    assert_equal(Batch[Rational].from_json(r.to_json()).to_json(), r.to_json())
    assert_equal(Batch[Float].from_json(f.to_json()).to_json(), f.to_json())
    var scalar = Batch[Complex]([Complex(1, 2)], shape=[])
    assert_equal(Batch[Complex].from_json(scalar.to_json()).shape(), List[Int]())
    for bad in [String('{"version":2,"family":"integer-batch","values":["1"]}'),
                String('{"version":1,"family":"integer-batch","shape":["1"],"values":["1"]}'),
                String('{"version":2,"family":"integer-batch","shape":["2"],"values":["1"]}'),
                String('{"version":3,"family":"integer-batch","shape":["1"],"values":["1"]}')]:
        with assert_raises():
            _ = Batch[Integer].from_json(bad)

    # Axis reductions reduce zero-copy lanes with the ordinary reductions.
    var matrix = Batch[Integer]([1, 2, 3, 4, 5, 6], shape=[2, 3])
    assert_equal(sum(matrix), 21)
    assert_equal(sum(matrix, axis=0).to_list(), [Integer(5), Integer(7), Integer(9)])
    assert_equal(sum(matrix, axis=-1, keepdims=True).shape(), [2, 1])
    assert_equal(prod(matrix, axis=1).to_list(), [Integer(6), Integer(120)])
    assert_equal(amin(matrix.transpose(), axis=1).to_list(), [Integer(1), Integer(2), Integer(3)])
    assert_equal(amax(matrix, axis=0).to_list(), [Integer(4), Integer(5), Integer(6)])
    assert_equal(sum_tree(r, axis=0).to_list(), [Rational(11, 2), Rational(5, 6)])
    assert_equal(sum_sequential(f, axis=1).to_list(), [Float(5), Float(25)])
    assert_equal(sum(Batch[Integer]([], shape=[0, 3]), axis=0).to_list(), [Integer(0), Integer(0), Integer(0)])
    assert_equal(sum(Batch[Integer]([], shape=[2, 0]), axis=1).to_list(), [Integer(0), Integer(0)])
    with assert_raises():
        _ = sum(matrix, axis=2)

    # Transposed and reversed views share the owner's values.
    var shared = matrix.transpose().slice(0, step=-1)
    assert_equal(Int(shared._owner.ptr()), Int(matrix._owner.ptr()))
    assert_equal(shared[0, 1], 6)


def test_selections_share_values_and_scalars_fill() raises:
    # Slices, axis selections and transposes are batches sharing the source.
    var values = Batch[Integer]([1, -2, 3, -4, 5])
    var odd = values[::2]
    assert_equal(Int(odd._owner.ptr()), Int(values._owner.ptr()))
    assert_equal(Int(Batch[Integer](values)._owner.ptr()), Int(values._owner.ptr()))
    var matrix = Batch[Integer].from_iterable(range(6), shape=[2, 3])
    var columns = matrix.transpose()
    assert_equal(Int(columns._owner.ptr()), Int(matrix._owner.ptr()))
    assert_equal(Batch[Integer](columns).shape(), [3, 2])
    # Updates publish new values; every sharing batch keeps its own.
    odd[0] = Integer(100)
    assert_equal(values[0], 1)
    matrix[matrix > 2] = -1
    assert_equal(columns.to_list(), [Integer(0), Integer(3), Integer(1), Integer(4), Integer(2), Integer(5)])
    assert_equal(matrix.to_list(), [Integer(0), Integer(1), Integer(2), Integer(-1), Integer(-1), Integer(-1)])

    # A scalar converts to a rank-zero batch: it fills a selection and
    # broadcasts in arithmetic; a one-element vector is a sequence.
    var rank_zero: Batch[Integer] = Integer(5)
    assert_equal(rank_zero.ndim(), 0)
    assert_equal((rank_zero + Batch[Integer]([1, 2])).to_list(), [Integer(6), Integer(7)])
    values[values < 0] = 0
    assert_equal(values.to_list(), [Integer(1), Integer(0), Integer(3), Integer(0), Integer(5)])
    var count = 9
    values[::-2] = count
    values[1:2] = Integer(7)
    values[3:] = 18446744073709551616
    assert_equal(values.to_list(), [Integer(9), Integer(7), Integer(9), Integer(1) << 64, Integer(1) << 64])
    with assert_raises():
        values[1:3] = Batch[Integer]([Integer(1)])
    values[:2] = [Integer(10), Integer(11)]
    var listed: List[Integer] = [Integer(4), Integer(5)]
    values[3:] = listed
    assert_equal(values.to_list(), [Integer(10), Integer(11), Integer(9), Integer(4), Integer(5)])
    var rationals = Batch[Rational]([Rational(1, 2), Rational(3, 4)])
    rationals[rationals > Rational(1, 2)] = Integer(5)
    assert_equal(rationals[1], Rational(5))
    var floats = Batch[Float]([Float(1), Float(2)])
    floats[:] = Float64(0.5)
    assert_equal(floats[1], Float(Rational(1, 2)))
    var complex = Batch[Complex]([Complex(1, 2), Complex(3, 4)])
    complex[1:] = Rational(1, 4)
    assert_equal(complex[1], Complex(Rational(1, 4)))

    # Compound updates on a selection read, update and write it back.
    values[values > 9] += 100
    values[3:] *= 2
    assert_equal(values.to_list(), [Integer(110), Integer(111), Integer(9), Integer(8), Integer(10)])

    # Scalar calls stay scalar; batch overloads are generic, so literals are unambiguous.
    assert_equal(gcd(12, 18), 6)
    assert_equal(isqrt(Integer(16)), 4)
    assert_equal(div_rem_floor(7, 2)[0], 3)
    assert_equal(vmap[gcd]()(Batch[Integer]([12, 18]), 4).to_list(), [Integer(4), Integer(2)])
    assert_equal(vmap[div_rem_floor]()(Batch[Integer]([7, -7]), 2)[1].to_list(), [Integer(1), Integer(1)])
    assert_equal(vmap[pow_rational]()(Batch[Integer]([2, 4]), -1).to_list(), [Rational(1, 2), Rational(1, 4)])
    assert_equal(add(Float(1), 2), Float(3))


def test_public_scalar_vmap_axes_masks_and_transaction() raises:
    var values = Batch[Integer]([1, 2, 3, 4])
    var snapshot = values[::-2]
    def polynomial(x: Integer) raises -> Integer:
        return x * x + 17
    var mapped = vmap[polynomial](in_axes=-1, out_axes=-1)
    assert_equal(mapped(snapshot)[0], 33)
    assert_equal(mapped(snapshot)[1], 21)
    values[:] = 9
    assert_equal(mapped(snapshot)[0], 33)
    def positive(x: Integer) raises -> Bool:
        return x > 2
    var mask = vmap[positive]()(snapshot)
    assert_equal(mask.count(), 1)
    assert_equal(snapshot[mask][0], 4)
    assert_equal(len(vmap[_tensor_failure]()(Batch[Integer]())), 0)
    assert_equal(len(vmap[_tensor_failure](in_axes=None, axis_size=0)(Integer(0))), 0)
    assert_equal(vmap[polynomial](in_axes=None, axis_size=2)(Integer(3))[1], 26)
    for count in [-1, 3]:
        with assert_raises():
            _ = vmap[polynomial](axis_size=count)(snapshot)
    with assert_raises():
        _ = vmap[polynomial](in_axes=1)(snapshot)
    with assert_raises():
        _ = vmap[polynomial](out_axes=1)(snapshot)
    with assert_raises():
        _ = vmap[polynomial](in_axes=None)(Integer(3))
    with assert_raises():
        _ = vmap[polynomial]()(Integer(3))
    with assert_raises():
        _ = vmap[polynomial](in_axes=None)(snapshot)
    def difference(a: Integer, b: Integer) raises -> Integer:
        return a - b
    assert_equal(vmap[difference]()(snapshot, snapshot[::-1])[0], 2)
    assert_equal(vmap[difference](in_axes=(0, None))(snapshot, Integer(1))[1], 1)
    assert_equal(vmap[difference](in_axes=(None, -1))(Integer(1), snapshot)[0], -3)
    assert_equal(vmap[difference](in_axes=(None, None), axis_size=3)(Integer(7), Integer(2))[2], 5)
    with assert_raises():
        _ = vmap[difference]()(snapshot, Batch[Integer]([Integer(1)]))
    with assert_raises():
        _ = vmap[difference](in_axes=(None, None))(Integer(7), Integer(2))
    var invalid = Batch[Integer]([2, 1, 0, -1])
    var before = values.to_json()
    var failed = False
    try:
        values = vmap[_tensor_failure]()(invalid)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 2"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(values.to_json(), before)
    assert_equal(invalid[2], 0)
    assert_equal(vmap[_tensor_family_product[Rational]]()(
        Batch[Rational]([Rational(2, 3)]), Batch[Rational]([Rational(3, 5)])
    )[0], Rational(2, 5))
    assert_equal(vmap[_tensor_family_product[Complex]]()(
        Batch[Complex]([Complex(1, 2)]), Batch[Complex]([Complex(1, -2)])
    )[0], Complex(5))


def _check_public_tensor_vmap_family[T: ImplicitlyCopyable & Deinitable & Equatable & Writable](
    values: List[T],
) raises:
    var source = Batch[T](values)
    var owner = source._tensor_view()._owner
    var references = owner.count()
    def reverse(row: Batch[T]) raises -> Batch[T]:
        return row[::-1]
    var result = vmap[reverse](in_axes=-1, out_axes=-1)(source.reshape([2, 2]).transpose())
    assert_equal(result.shape(), [2, 2])
    assert_equal(result[0, 0], values[1])
    assert_equal(result[0, 1], values[3])
    assert_equal(result[1, 0], values[0])
    assert_equal(result[1, 1], values[2])
    def first(row: Batch[T]) raises -> T:
        return row[0]
    assert_equal(rebind_var[Batch[T]](vmap[first]()(result)).to_list(), [values[1], values[0]])
    def identity(cell: Batch[T]) raises -> Batch[T]:
        return cell
    assert_equal(vmap[identity]()(source).to_json(), source.to_json())
    var repeated = vmap[reverse](in_axes=None, axis_size=2)(source[::-1])
    assert_equal(repeated.at(0, 0).to_json(), source.to_json())
    assert_equal(repeated.at(0, 1).to_json(), source.to_json())
    def local_edit(row: Batch[T]) raises -> Batch[T]:
        var local = row
        local[0] = row[-1]
        return local
    assert_equal(vmap[local_edit]()(source.reshape([2, 2]))[0, 0], values[1])
    assert_equal(owner.count(), references)
    # Keep the source live through the count check; Mojo destroys at last use.
    assert_equal(source.to_list(), values)


def test_public_tensor_vmap_axes_shapes_and_snapshots() raises:
    var wide = (Integer(1) << 1024) + 7
    _check_public_tensor_vmap_family[Integer]([1, -wide, 3, wide])
    _check_public_tensor_vmap_family[Rational]([Rational(1, 3), Rational(-wide, 7), Rational(3), Rational(wide, 9)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    _check_public_tensor_vmap_family[Float]([Float.zero(negative=True), Float(-wide, context=context), Float(3), Float(wide, context=context)])
    _check_public_tensor_vmap_family[Complex]([Complex(0), Complex(-wide, 2, context=context), Complex(3, 4), Complex(0, wide, context=context)])

    var matrix = Batch[Integer].from_iterable(range(1, 7), shape=[2, 3])
    def reverse(row: Batch[Integer]) raises -> Batch[Integer]:
        return row[::-1]
    def combine(row: Batch[Integer], shared: Batch[Integer]) raises -> Batch[Integer]:
        return row + shared
    var offset = Batch[Integer]([10, 20, 30])
    var combined = vmap[combine](in_axes=(0, None), out_shape=[3])(matrix, offset)
    assert_equal(combined[1, 2], 36)
    assert_equal(vmap[combine](in_axes=(None, 0))(offset, matrix)[0, 0], 11)
    assert_equal(vmap[combine](in_axes=(None, None), axis_size=2)(offset, offset)[1, 2], 60)
    assert_equal(vmap[combine]()(matrix, matrix)[1, 2], 12)
    def large(row: Batch[Integer]) raises -> Bool:
        return row[0] > 2
    assert_equal(vmap[large]()(matrix).count(), 1)
    def transpose(block: Batch[Integer]) raises -> Batch[Integer]:
        return block.transpose()
    var cube = matrix.reshape([1, 2, 3])
    var flipped = vmap[transpose](out_axes=1)(cube)
    assert_equal(flipped.shape(), [3, 1, 2])
    assert_equal(flipped[2, 0, 1], 6)
    assert_equal(vmap[transpose](in_axes=None, axis_size=1)(matrix)[0, 2, 1], 6)
    var snapshot = matrix.slice(0, step=-1).slice(1, step=-2)
    var retained = vmap[reverse]()(snapshot)
    matrix[1, 2] = 99
    assert_equal(retained[0, 1], 6)
    assert_equal(snapshot[0, 0], 6)

    def never(row: Batch[Integer]) raises -> Batch[Integer]:
        raise Error("An empty mapping must not call its function.")
    var empty = Batch[Integer]([], shape=[0, 3])
    assert_equal(vmap[never](out_shape=[3], out_axes=-1)(empty).shape(), [3, 0])
    assert_equal(vmap[transpose](out_shape=[3, 2])(empty.reshape([0, 2, 3])).shape(), [0, 3, 2])
    assert_equal(vmap[combine](out_shape=[3])(empty, empty).shape(), [0, 3])
    assert_equal(vmap[combine](in_axes=(None, None), axis_size=0, out_shape=[3])(offset, offset).size(), 0)
    # A nonempty mapped axis may still contain empty rows.
    assert_equal(vmap[reverse]()(empty.transpose()).shape(), [3, 0])
    def scalar(cell: Batch[Integer]) raises -> Batch[Integer]:
        return Batch[Integer]([cell.item() + 1], shape=[])
    # Batch results have runtime rank; an empty mapping needs out_shape.
    assert_equal(len(vmap[scalar](out_shape=List[Int]())(Batch[Integer]())), 0)
    with assert_raises():
        _ = vmap[scalar]()(Batch[Integer]())
    assert_equal(vmap[scalar](in_axes=None, axis_size=2)(Batch[Integer]([Integer(7)], shape=[]))[1], 8)
    with assert_raises():
        _ = vmap[scalar]()(Batch[Integer]([Integer(7)], shape=[]))
    with assert_raises():
        _ = vmap[never]()(empty)
    with assert_raises():
        _ = vmap[never](out_shape=[-1])(empty)
    with assert_raises():
        _ = vmap[reverse](out_shape=[Int.MAX])(matrix)
    with assert_raises():
        _ = vmap[reverse](out_shape=[2])(matrix)
    with assert_raises():
        _ = vmap[reverse](in_axes=2)(empty)
    with assert_raises():
        _ = vmap[reverse](out_axes=2)(matrix)
    # A batch argument accepts whatever rank remains after mapping.
    assert_equal(vmap[reverse]()(offset).to_list(), offset.to_list())
    assert_equal(vmap[reverse](in_axes=None, axis_size=1)(matrix).shape(), [1, 6])
    with assert_raises():
        _ = vmap[combine]()(matrix, matrix.slice(0, stop=1))

    var target = combined
    var before = target.to_list()
    def variable(row: Batch[Integer]) raises -> Batch[Integer]:
        return row[:Int(row[0])]
    var failed = False
    try:
        target = vmap[variable]()(matrix)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("same tensor shape"))
    assert_true(failed)
    assert_equal(target.to_list(), before)
    var failures = Batch[Integer].from_iterable([2, 0, -1], shape=[3, 1])
    var failure_owner = failures._tensor_view()._owner
    var failure_references = failure_owner.count()
    def checked(row: Batch[Integer]) raises -> Batch[Integer]:
        return Batch[Integer]([_tensor_failure(row[0])])
    failed = False
    try:
        target = vmap[checked]()(failures)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(target.to_list(), before)
    assert_equal(failure_owner.count(), failure_references)
    assert_equal(failures[1, 0], 0)

    var floats = Batch[Float]([Float(1, context=context)])
    floats._real_format = context.format()
    var complex = Batch[Complex]([Complex(1, 2)])
    complex._real_format = FloatFormat(256)
    complex._imag_format = FloatFormat(1024)
    def empty_float(x: Integer, source: Batch[Float]) raises -> Batch[Float]:
        return source[:0]
    def empty_complex(x: Integer, source: Batch[Complex]) raises -> Batch[Complex]:
        return source[:0]
    assert_equal(vmap[empty_float](in_axes=(0, None))(offset, floats).at(0, 1).to_json(), floats[:0].to_json())
    assert_equal(vmap[empty_complex](in_axes=(0, None))(offset, complex).at(0, 2).to_json(), complex[:0].to_json())
    var empty_result = vmap[empty_float](in_axes=(0, None), out_shape=[0])(Batch[Integer](), floats)
    assert_equal(empty_result.reshape([0])._float_input().default_format(), FloatFormat(128))


def _check_structured_vmap_family[T: ImplicitlyCopyable & Deinitable](values: List[T]) raises where (_Result[T].leaf):
    var source = Batch[T](values)
    def identity(x: T) raises -> T:
        return x
    def pair(x: T) raises -> Tuple[T, Bool]:
        return (x, True)
    var paired = vmap[pair]()(source[::-1])
    assert_equal(rebind[Batch[T]](paired[0]).to_json(), source[::-1].to_json())
    assert_equal(paired[1].count(), len(values))
    var nested = vmap[identity]().vmap().vmap()(source.reshape([1, 2, 2]))
    assert_equal(rebind_var[Batch[T]](nested^).reshape([4]).to_json(), source.to_json())
    def both(x: T, y: T) raises -> Tuple[T, T]:
        return (x, y)
    var matrix = source.reshape([2, 2])
    var paired_matrix = vmap[both]().vmap()(matrix, matrix.slice(1, step=-1))
    assert_equal(rebind[Batch[T]](paired_matrix[0]).reshape([4]).to_json(), source.to_json())
    assert_equal(rebind[Batch[T]](paired_matrix[1]).reshape([4]).to_json(), matrix.slice(1, step=-1).reshape([4]).to_json())


def test_public_structured_vmap_pairs_nesting_and_failure() raises:
    var wide = (Integer(1) << 1024) + 7
    _check_structured_vmap_family[Integer]([1, -wide, 3, wide])
    _check_structured_vmap_family[Rational]([Rational(1, 3), Rational(-wide, 7), Rational(3), Rational(wide, 9)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    _check_structured_vmap_family[Float]([Float.zero(negative=True), Float(-wide, context=context), Float(3), Float(wide, context=context)])
    _check_structured_vmap_family[Complex]([Complex(0), Complex(-wide, 2, context=context), Complex(3, 4), Complex(0, wide, context=context)])

    var source = Batch[Integer]([1, 2, 3, 4, 5, 6])
    var matrix = source.reshape([2, 3])
    var owner = matrix._tensor_view()._owner
    var references = owner.count()
    def square(x: Integer) raises -> Integer:
        return x * x
    def add_pair(pair: Tuple[Integer, Integer]) raises -> Integer:
        return pair[0] + pair[1]
    def split_pair(pair: Tuple[Integer, Integer]) raises -> Tuple[Integer, Bool]:
        return (pair[0] + pair[1], pair[0] > pair[1])
    def split(a: Integer, b: Integer) raises -> Tuple[Integer, Bool]:
        return (a - b, a > b)
    assert_equal(vmap[add_pair]()((source, source[::-1]))[0], 7)
    assert_equal(vmap[add_pair](in_axes=(0, None))((source, Integer(10)))[5], 16)
    assert_equal(vmap[add_pair](in_axes=(None, -1))((Integer(10), source))[0], 11)
    assert_equal(vmap[add_pair](in_axes=(None, None), axis_size=2)((Integer(3), Integer(4)))[1], 7)
    var pair = vmap[split_pair]()((source, source[::-1]))
    assert_equal(pair[0][5], 7)
    assert_equal(pair[1].count(), 3)
    var binary = vmap[split](in_axes=(0, None))(source, Integer(3))
    assert_equal(binary[0][0], -2)
    assert_equal(binary[1].count(), 3)
    with assert_raises():
        _ = vmap[add_pair]()((source, source[:1]))
    with assert_raises():
        _ = vmap[add_pair](in_axes=(None, None))((Integer(3), Integer(4)))

    def rows(row: Batch[Integer]) raises -> Tuple[Batch[Integer], Integer]:
        return (row[::-1], row[0])
    var rows_result = vmap[rows](out_axes=(-1, 0))(matrix)
    assert_equal(rows_result[0].shape(), [3, 2])
    assert_equal(rows_result[0][0, 1], 6)
    assert_equal(rows_result[1][1], 4)
    var bad_pair_shape: List[Int] = [1]
    var rejected = False
    try:
        _ = vmap[rows](out_axes=(2, 0), out_shape=([3], bad_pair_shape.copy()))(matrix)
    except error:
        rejected = True
        assert_true(String(error).__contains__("axis is out of range"))
    assert_true(rejected)
    def shared_row(pair: Tuple[Batch[Integer], Integer]) raises -> Batch[Integer]:
        return pair[0][::-1]
    def shared_pair(pair: Tuple[Batch[Integer], Integer]) raises -> Tuple[Batch[Integer], Integer]:
        return (pair[0][::-1], pair[1])
    def binary_rows(row: Batch[Integer], x: Integer) raises -> Tuple[Batch[Integer], Integer]:
        return (row[::-1], x)
    assert_equal(vmap[shared_row](in_axes=(0, None), out_shape=[3])((matrix, Integer(9)))[1, 0], 6)
    var tuple_rows = vmap[shared_pair](in_axes=(0, None), out_axes=0, out_shape=([3], None))((matrix, Integer(9)))
    assert_equal(tuple_rows[0][1, 0], 6)
    assert_equal(tuple_rows[1][1], 9)
    var two_rows = vmap[binary_rows](in_axes=(0, None), out_shape=([3], None))(matrix, Integer(9))
    assert_equal(two_rows[0].to_list(), tuple_rows[0].to_list())
    assert_equal(two_rows[1].to_list(), tuple_rows[1].to_list())
    def never(x: Integer) raises -> Tuple[Batch[Integer], Bool]:
        raise Error("An empty mapping must not call its function.")
    var empty = vmap[never](out_shape=([3], None), out_axes=(1, 0))(source[:0])
    assert_equal(empty[0].shape(), [3, 0])
    assert_equal(len(empty[1]), 0)
    with assert_raises():
        _ = vmap[never]()(source[:0])
    with assert_raises():
        var bad_shape: List[Int] = [3, -1]
        _ = vmap[never](out_shape=(bad_shape^, None))(source[:0])
    with assert_raises():
        _ = vmap[rows](out_axes=(0, 1))(matrix)

    var nested: Batch[Integer] = vmap[square](in_axes=-1).vmap(out_axes=-1)(matrix.slice(1, step=-1))
    assert_equal(nested.shape(), [3, 2])
    assert_equal(nested[0, 1], 36)
    assert_equal(nested[2, 0], 1)
    assert_equal(vmap[square]().vmap(in_axes=None, axis_size=2)(source)[1, 5], 36)
    assert_equal(vmap[square](in_axes=None, axis_size=2).vmap()(source)[5, 1], 36)
    assert_equal(vmap[square]().vmap(out_shape=[3])(matrix.slice(0, stop=0)).shape(), [0, 3])
    assert_equal(vmap[square]().vmap()(Batch[Integer]([], shape=[2, 0])).shape(), [2, 0])
    # Scalar leaves have a known shape, so empty nested mappings infer it.
    assert_equal(vmap[square]().vmap()(matrix.slice(0, stop=0)).shape(), [0, 3])
    with assert_raises():
        _ = vmap[square]().vmap(out_shape=[2])(matrix.slice(0, stop=0))
    with assert_raises():
        _ = vmap[square]().vmap(axis_size=3)(matrix)
    with assert_raises():
        _ = vmap[square]().vmap()(Integer(1))

    var target = (Batch[Integer]([99]), Batch[Integer]([99], shape=[1, 1]))
    def variable(x: Integer, whole: Batch[Integer]) raises -> Tuple[Integer, Batch[Integer]]:
        return (x, whole[:Int(x)])
    var failed = False
    try:
        target = vmap[variable](in_axes=(0, None))(source, source)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("same tensor shape"))
    assert_true(failed)
    assert_equal(target[0][0], 99)
    assert_equal(target[1][0, 0], 99)
    var invalid = Batch[Integer]([2, 1, 0, -1], shape=[2, 2])
    failed = False
    try:
        nested = vmap[_tensor_failure]().vmap()(invalid)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1: vmap failed at mapped index 0"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(nested[0, 1], 36)
    assert_equal(invalid[1, 0], 0)
    assert_equal(owner.count(), references)
    assert_equal(matrix[1, 2], 6)
    assert_equal(source[5], 6)


def test_public_nested_binary_and_paired_mapping() raises:
    var matrix = Batch[Integer].from_iterable(range(1, 7), shape=[2, 3])
    var shared = Batch[Integer]([10, 20, 30])
    var owner = matrix._tensor_view()._owner
    var references = owner.count()
    def add(x: Integer, y: Integer) raises -> Integer:
        return x + y
    def both(x: Integer, y: Integer) raises -> Tuple[Integer, Integer]:
        return (x + y, x * y)
    def tuple_add(pair: Tuple[Integer, Integer]) raises -> Integer:
        return add(pair[0], pair[1])
    def tuple_both(pair: Tuple[Integer, Integer]) raises -> Tuple[Integer, Integer]:
        return both(pair[0], pair[1])
    var aligned = vmap[add](in_axes=(-1, -1)).vmap(in_axes=(-1, 0), out_axes=-1)(matrix.transpose(), matrix)
    assert_equal(aligned.shape(), [3, 2])
    assert_equal(aligned[2, 1], 12)
    var result: Tuple[Batch[Integer], Batch[Integer]] = vmap[both]().vmap(in_axes=(0, None), out_axes=(0, -1))(matrix, shared)
    assert_equal(result[0][1, 2], 36)
    assert_equal(result[1].shape(), [3, 2])
    assert_equal(result[1][2, 1], 180)
    assert_equal(vmap[both]().vmap(in_axes=(None, 0))(shared, matrix)[0][1, 2], 36)
    assert_equal(vmap[both]().vmap(in_axes=(None, None), axis_size=2)(shared, shared)[1][1, 2], 900)
    assert_equal(vmap[add](in_axes=(0, None)).vmap(in_axes=(0, None))(matrix, Integer(7))[1, 2], 13)
    assert_equal(vmap[add](in_axes=(None, 0)).vmap(in_axes=(None, 0))(Integer(7), matrix)[1, 2], 13)
    assert_equal(vmap[add](in_axes=(None, None), axis_size=2).vmap()(shared, shared)[2, 1], 60)
    assert_equal(vmap[tuple_add]().vmap(in_axes=(0, None))((matrix, shared))[1, 2], 36)
    var tuple_result = vmap[tuple_both]().vmap(in_axes=(0, None), out_axes=(0, -1))((matrix, shared))
    assert_equal(tuple_result[0].to_list(), result[0].to_list())
    assert_equal(tuple_result[1].to_list(), result[1].to_list())
    assert_equal(vmap[tuple_add]().vmap().vmap()((matrix.reshape([1, 2, 3]), matrix.reshape([1, 2, 3])))[0, 1, 2], 12)

    var context = ArithmeticContext(format=FloatFormat(256))
    def mixed(x: Integer, offset: Integer) raises -> Tuple[Rational, Float]:
        return (Rational(x, 3), Float(x + offset, context=ArithmeticContext(format=FloatFormat(256))))
    var mixed_result = vmap[mixed]().vmap(out_axes=(-1, 0))(matrix.slice(1, step=-1), shared[0])
    assert_equal(mixed_result[0][0, 1], Rational(2))
    assert_equal(mixed_result[1][1, 0].to_json(), Float(16, context=context).to_json())

    var empty = matrix.slice(0, stop=0)
    var shape: List[Int] = [3]
    assert_equal(vmap[add]().vmap(out_shape=shape.copy())(empty, empty).shape(), [0, 3])
    assert_equal(vmap[tuple_add]().vmap(out_shape=[3])((empty, empty)).shape(), [0, 3])
    var empty_binary = vmap[both]().vmap(out_shape=([3], [3]))(empty, empty)
    var empty_tuple = vmap[tuple_both]().vmap(out_shape=([3], [3]))((empty, empty))
    var empty_unary = vmap[mixed]().vmap(out_shape=([3], [3]))(empty, shared[0])
    assert_equal(empty_binary[0].shape(), [0, 3])
    assert_equal(empty_tuple[1].shape(), [0, 3])
    assert_equal(empty_unary[0].shape(), [0, 3])
    assert_equal(empty_unary[1].reshape([0])._float_input().default_format(), FloatFormat(128))
    var empty_rows = Batch[Integer]([], shape=[2, 0])
    assert_equal(vmap[both]().vmap()(empty_rows, empty_rows)[1].shape(), [2, 0])
    assert_equal(vmap[both]().vmap(in_axes=(None, None), axis_size=0, out_shape=([3], [3]))(shared, shared)[0].size(), 0)
    assert_equal(vmap[both]().vmap(in_axes=(0, None), out_shape=(None, [3]))(matrix, shared)[1][1, 2], 180)
    assert_equal(vmap[both]().vmap()(empty, empty)[0].shape(), [0, 3])
    assert_equal(vmap[both]().vmap(out_shape=(None, [3]))(empty, empty)[1].shape(), [0, 3])
    with assert_raises():
        _ = vmap[both]().vmap(out_shape=([3], [-1]))(empty, empty)
    with assert_raises():
        _ = vmap[both]().vmap(out_shape=([3], [2]))(matrix, matrix)
    with assert_raises():
        _ = vmap[both]().vmap(out_axes=(0, 2))(matrix, matrix)
    with assert_raises():
        _ = vmap[add]().vmap(in_axes=(None, None))(shared, shared)
    with assert_raises():
        _ = vmap[add]().vmap(axis_size=3)(matrix, matrix)
    with assert_raises():
        _ = vmap[add]().vmap(in_axes=(0, 2))(matrix, matrix)
    # An unmapped scalar is shared by every call at every level.
    assert_equal(vmap[add]().vmap()(matrix, Integer(1)).to_list(), (matrix + 1).to_list())
    with assert_raises():
        _ = vmap[add]().vmap()(matrix, matrix.slice(0, stop=1))

    def checked(x: Integer, y: Integer) raises -> Tuple[Integer, Integer]:
        return (x, _tensor_failure(y))
    var invalid = Batch[Integer]([2, 3, 4, 5, 0, -1], shape=[2, 3])
    var before_first = result[0].to_list()
    var before_second = result[1].to_list()
    var failed = False
    try:
        result = vmap[checked]().vmap(out_axes=(0, -1))(matrix, invalid)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1: vmap failed at mapped index 1"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(result[0].to_list(), before_first)
    assert_equal(result[1].to_list(), before_second)
    assert_equal(invalid[1, 1], 0)
    assert_equal(owner.count(), references)
    assert_equal(matrix[1, 2], 6)
    assert_equal(shared[2], 30)


def test_public_variadic_tuple_results() raises:
    var wide = (Integer(1) << 1024) + 7
    var source = Batch[Integer]([wide, -3, 0, 5])
    var matrix = source.reshape([2, 2])
    var owner = matrix._tensor_view()._owner
    var references = owner.count()
    var context = ArithmeticContext(format=FloatFormat(1024))
    def families(x: Integer) raises -> Tuple[Integer, Rational, Float, Complex]:
        var wide = ArithmeticContext(format=FloatFormat(1024))
        return (x, Rational(x, 3), Float(x, context=wide), Complex(x, -x, context=wide))
    var result = vmap[families]()(source[::-1])
    for i in range(4):
        var expected = families(source[3 - i])
        assert_equal(result[0][i], expected[0])
        assert_equal(result[1][i], expected[1])
        assert_equal(result[2][i].to_json(), expected[2].to_json())
        assert_equal(result[3][i].to_json(), expected[3].to_json())
    var nested: Tuple[Batch[Integer], Batch[Rational], Batch[Float], Batch[Complex]] = vmap[families]().vmap(out_axes=(0, -1, 0, -1))(matrix)
    assert_equal(nested[0][0, 0], wide)
    assert_equal(nested[1][1, 0], Rational(-1))
    assert_equal(nested[2][1, 1].to_json(), Float(5, context=context).to_json())
    assert_equal(nested[3][0, 0].to_json(), Complex(wide, -wide, context=context).to_json())
    assert_equal(vmap[families]().vmap(out_shape=([2], [2], [2], [2]))(matrix.slice(0, stop=0))[3].shape(), [0, 2])

    def three(x: Integer, y: Integer) raises -> Tuple[Integer, Integer, Integer]:
        return (x + y, x - y, x * y)
    def tuple_three(pair: Tuple[Integer, Integer]) raises -> Tuple[Integer, Integer, Integer]:
        return three(pair[0], pair[1])
    assert_equal(vmap[three](in_axes=(0, None))(source, Integer(2))[2][0], wide * 2)
    assert_equal(vmap[tuple_three]()((source, source[::-1]))[1][0], wide - 5)
    assert_equal(vmap[tuple_three]().vmap()((matrix, matrix))[2][1, 1], 25)
    assert_equal(vmap[three]().vmap(out_shape=([2], [2], [2]))(matrix.slice(0, stop=0), matrix.slice(0, stop=0))[2].shape(), [0, 2])
    assert_equal(vmap[tuple_three]().vmap(out_shape=([2], [2], [2]))((matrix.slice(0, stop=0), matrix.slice(0, stop=0)))[0].shape(), [0, 2])
    def one(x: Integer) raises -> Tuple[Integer]:
        return Tuple(x)
    def flags(x: Integer) raises -> Tuple[Integer, Bool, Bool]:
        return (x, x > 0, x == 0)
    def nothing(x: Integer) raises -> Tuple[]:
        _ = _tensor_failure(x)
        return Tuple()
    assert_equal(vmap[one]()(source)[0][0], wide)
    assert_equal(vmap[flags]()(source)[1].count(), 2)
    assert_equal(vmap[flags]()(source)[2].count(), 1)
    assert_equal(len(vmap[nothing]()(source[:1])), 0)
    with assert_raises():
        _ = vmap[nothing]()(source)

    def rows(row: Batch[Integer]) raises -> Tuple[Batch[Integer], Integer, Batch[Integer]]:
        return (row[::-1], row[0], row)
    var rows_result = vmap[rows](out_axes=(-1, 0, 0), out_shape=([2], None, [2]))(matrix)
    assert_equal(rows_result[0][1, 0], wide)
    assert_equal(rows_result[2][1, 1], 5)
    var listed_shape: List[Int] = [2]
    var array_shape = Array[Int, 1](fill=2)
    var listed_map = vmap[rows](out_axes=0, out_shape=(listed_shape.copy(), None, listed_shape.copy()))
    var array_map = vmap[rows](out_axes=(0, 0, 0), out_shape=(Optional[Array[Int, 1]](array_shape.copy()), Optional[List[Int]](), array_shape.copy()))
    comptime assert type_of(listed_map) == type_of(array_map), "Shape and axis syntax must not change the mapper type."
    listed_shape[0] = 9
    array_shape[0] = 9
    var listed_result = listed_map(matrix)
    var array_result = array_map(matrix)
    assert_equal(listed_result[0].shape(), [2, 2])
    assert_equal(listed_result[0].to_list(), array_result[0].to_list())
    assert_equal(listed_result[1].to_list(), array_result[1].to_list())
    assert_equal(listed_result[2].to_list(), array_result[2].to_list())
    assert_equal(vmap[rows](out_shape=([2], None, [2]))(matrix.slice(0, stop=0))[2].shape(), [0, 2])
    with assert_raises():
        _ = vmap[rows]()(matrix.slice(0, stop=0))
    with assert_raises():
        _ = vmap[rows](out_axes=(0, 0, 3))(matrix)
    with assert_raises():
        _ = vmap[rows](out_shape=([2], None, [-1]))(matrix.slice(0, stop=0))
    def variable(x: Integer, whole: Batch[Integer]) raises -> Tuple[Integer, Integer, Batch[Integer]]:
        return (x, -x, whole[:Int(x)])
    var target = (Batch[Integer]([99]), Batch[Integer]([98]), Batch[Integer]([97], shape=[1, 1]))
    var failed = False
    try:
        target = vmap[variable](in_axes=(0, None))(Batch[Integer]([1, 2]), source)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("same tensor shape"))
    assert_true(failed)
    assert_equal(target[0][0], 99)
    assert_equal(target[1][0], 98)
    assert_equal(target[2][0, 0], 97)
    assert_equal(owner.count(), references)
    assert_equal(matrix[0, 0], wide)
    assert_equal(source[0], wide)


def test_public_variadic_tuple_inputs() raises:
    var wide = (Integer(1) << 1024) + 7
    var source = Batch[Integer]([wide, -3, 0, 5])
    var matrix = Batch[Integer]([1, 2, 3, 4, 5, 6, 7, 8], shape=[2, 4])
    var rational = Batch[Rational]([Rational(1, 3), Rational(2, 3), Rational(4, 3), Rational(5, 3)])
    var context = ArithmeticContext(format=FloatFormat(1024))
    var floating = Float(wide, context=context)
    var complex_value = Complex(wide, -wide, context=context)
    var owner = source._tensor_view()._owner
    var references = owner.count()
    def run(checked: Bool) raises {source, matrix, rational, floating, complex_value} -> Batch[Integer]:
        var values = (source[::-1], rational[::-1], floating, complex_value, matrix, source[:], checked, rational)
        return vmap[_evaluate_parts](in_axes=(-1, 0, None, None, -1, None, None, None))(values)

    var result = run(False)
    for i in range(4):
        assert_equal(result[i], source[3 - i] + matrix[1, i] + wide)
    var before = result.to_list()
    var failed = False
    try:
        result = run(True)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(result.to_list(), before)

    comptime Three = Tuple[Integer, Integer, Integer]
    def total(parts: Three) raises -> Integer:
        return parts[0] + parts[1] + parts[2]
    def identity(parts: Three) raises -> Three:
        return parts
    assert_equal(vmap[total]()((source, source, source))[0], wide * 3)
    assert_equal(vmap[total](in_axes=-1)((source, source, source))[3], 15)
    assert_equal(vmap[identity]()((source, source, source))[2][0], wide)
    assert_equal(vmap[total](in_axes=(0, 0, 0))((source, source, source))[0], wide * 3)
    assert_equal(vmap[identity](in_axes=(0, 0, 0), out_axes=(0, -1, 0))((source, source, source))[2][0], wide)
    assert_equal(vmap[identity](in_axes=(0, 0, 0), out_shape=(None, None, None))((source, source, source))[1][3], 5)
    def flag(parts: Three) raises -> Bool:
        return parts[0] > 0
    def no_results(parts: Three) raises -> Tuple[]:
        _ = _tensor_failure(parts[0])
        return Tuple()
    assert_equal(vmap[flag](in_axes=(0, 0, 0))((source, source, source)).count(), 2)
    with assert_raises():
        _ = vmap[no_results](in_axes=(0, 0, 0))((source, source, source))
    var views = (source[:], source[:1], source[:])
    failed = False
    try:
        _ = vmap[total](in_axes=(0, 0, 2), axis_size=-1)(views)
    except error:
        failed = True
        assert_true(String(error).__contains__("axis"))
        assert_true(not String(error).__contains__("different lengths"))
    assert_true(failed)
    failed = False
    try:
        _ = vmap[total](in_axes=(0, 0, 0), axis_size=-1)(views)
    except error:
        failed = True
        assert_true(String(error).__contains__("different lengths"))
    assert_true(failed)
    var empty = (source[:0], source[:0], source[:0])
    assert_equal(len(vmap[total](in_axes=(0, 0, 0))(empty)), 0)
    assert_equal(len(vmap[no_results](in_axes=(0, 0, 0))(empty)), 0)
    with assert_raises():
        _ = vmap[total](in_axes=(0, 0))(empty)
    with assert_raises():
        _ = vmap[total](in_axes=(0, 0, 0), axis_size=1)(empty)
    var shared = (wide, Integer(2), Integer(3))
    assert_equal(vmap[total](in_axes=None, axis_size=2)(shared)[1], wide + 5)
    with assert_raises():
        _ = vmap[total](in_axes=(None, None, None))(shared)
    with assert_raises():
        _ = vmap[total](in_axes=(None, None, None), axis_size=-1)(shared)
    assert_equal(vmap[total](in_axes=(None, None, None), axis_size=2)(shared)[1], wide + 5)
    var singleton = Tuple(source[::2])
    def one(parts: Tuple[Integer]) raises -> Integer:
        return parts[0]
    assert_equal(vmap[one]()(singleton)[1], 0)
    assert_equal(vmap[one](in_axes=Tuple(0))(singleton)[1], 0)
    def constant(parts: Tuple[]) raises -> Integer:
        return Integer(7)
    assert_equal(len(vmap[constant](axis_size=0)(Tuple())), 0)
    assert_equal(vmap[constant](axis_size=2)(Tuple())[1], 7)
    with assert_raises():
        _ = vmap[constant](in_axes=Tuple())(Tuple())
    assert_equal(vmap[constant](in_axes=Tuple(), axis_size=2)(Tuple()).to_list(), [Integer(7), Integer(7)])
    assert_equal(len(vmap[constant](in_axes=Tuple(), axis_size=0)(Tuple())), 0)

    def rows(parts: Tuple[Batch[Integer], Integer, Batch[Integer]]) raises -> Tuple[Batch[Integer], Integer, Bool]:
        return (parts[0][::-1], parts[1] + parts[2][0], parts[0][0] > 2)
    var axes = (0, None, None)
    var dims: List[Int] = [4]
    var map_rows = vmap[rows](in_axes=axes, out_axes=(-1, 0, 0), out_shape=(dims.copy(), None, None))
    dims[0] = 1
    var row_result = map_rows((matrix, Integer(2), source[:]))
    assert_equal(row_result[0].shape(), [4, 2])
    assert_equal(row_result[0][0, 1], 8)
    assert_equal(row_result[1][1], wide + 2)
    assert_equal(row_result[2].count(), 1)
    assert_equal(map_rows((matrix.slice(0, stop=0), Integer(2), source[:]))[0].shape(), [4, 0])
    var shape_hint = Optional(source.shape())
    assert_equal(vmap[rows](in_axes=axes, out_shape=(shape_hint.copy(), None, None))((matrix.slice(0, stop=0), Integer(2), source[:]))[0].shape(), [0, 4])
    assert_equal(vmap[rows](in_axes=axes, out_shape=(source.shape(), None, None))((matrix.slice(0, stop=0), Integer(2), source[:]))[0].shape(), [0, 4])
    var invalid_shape = vmap[rows](in_axes=axes, out_shape=([-1], None, None))
    with assert_raises():
        _ = invalid_shape((matrix.slice(0, stop=0), Integer(2), source[:]))
    with assert_raises():
        _ = vmap[rows](in_axes=axes)((matrix.slice(0, stop=0), Integer(2), source[:]))
    def reversed_row(parts: Tuple[Batch[Integer], Integer, Batch[Integer]]) raises -> Batch[Integer]:
        return parts[0][::-1]
    assert_equal(vmap[reversed_row](in_axes=axes, out_shape=[4])((matrix.slice(0, stop=0), Integer(2), source[:])).shape(), [0, 4])

    # Reuse the same scalar tuple callbacks through additional mapped axes.
    var nested: Batch[Integer] = vmap[total](in_axes=(0, 0, 0)).vmap()((matrix, matrix, matrix))
    assert_equal(nested.shape(), [2, 4])
    assert_equal(nested[1, 3], 24)
    # All axis combinations reuse one mapper type; axes are runtime metadata.
    for modes in range(8):
        var outer = Array[Optional[Int], 3](fill=None)
        var middle = Array[Optional[Int], 3](fill=None)
        for leaf in range(3):
            if modes & (1 << leaf):
                outer[leaf] = 0
            else:
                middle[leaf] = 0
        var mapped = vmap[total]().vmap(in_axes=(middle[0], middle[1], middle[2]), axis_size=2).vmap(in_axes=(outer[0], outer[1], outer[2]), axis_size=2)
        var combinations: Batch[Integer] = mapped((matrix, matrix, matrix))
        assert_equal(combinations.shape(), [2, 2, 4])
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    var expected = Integer(0)
                    for leaf in range(3):
                        expected += matrix[i if modes & (1 << leaf) else j, k]
                    assert_equal(combinations[i, j, k], expected)
    assert_equal(matrix[1, 3], 8)
    var columns = vmap[total](in_axes=(0, 0, 0)).vmap(in_axes=(-1, 0, None), out_axes=-1)((matrix, matrix.transpose(), source[:2]))
    assert_equal(columns.shape(), [2, 4])
    assert_equal(columns[0, 3], wide + 8)
    assert_equal(columns[1, 3], 13)
    var triples = vmap[identity](in_axes=(0, 0, 0)).vmap(out_axes=(0, -1, 0))((matrix, matrix, matrix))
    assert_equal(triples[0][1, 3], 8)
    assert_equal(triples[1].shape(), [4, 2])
    assert_equal(triples[1][3, 1], 8)
    var empty_matrices = (matrix.slice(0, stop=0), matrix.slice(0, stop=0), matrix.slice(0, stop=0))
    assert_equal(vmap[total](in_axes=(0, 0, 0)).vmap(out_shape=[4])(empty_matrices).shape(), [0, 4])
    var hint_list: List[Int] = [4]
    var hints = (source.shape(), Optional(source.shape()), hint_list.copy())
    assert_equal(vmap[identity](in_axes=(0, 0, 0)).vmap(out_shape=hints^)(empty_matrices)[2].shape(), [0, 4])
    assert_equal(vmap[total](in_axes=(0, 0, 0)).vmap()(empty_matrices).shape(), [0, 4])
    var inner_empty = Batch[Integer](List[Integer](), shape=[2, 0])
    assert_equal(vmap[total](in_axes=(0, 0, 0)).vmap()((inner_empty, inner_empty, inner_empty)).shape(), [2, 0])
    var outer_shared = vmap[total](in_axes=(0, 0, 0)).vmap(in_axes=None, axis_size=2)((source, source, source))
    assert_equal(outer_shared[1, 0], wide * 3)
    with assert_raises():
        _ = vmap[total](in_axes=(0, 0, 0)).vmap(in_axes=None)((source, source, source))
    assert_equal(vmap[one](in_axes=Tuple(0)).vmap(in_axes=-1)(Tuple(matrix))[3, 1], 8)
    assert_equal(vmap[constant](in_axes=Tuple(), axis_size=2).vmap(axis_size=3)(Tuple()).shape(), [3, 2])

    var volume = Batch[Integer].from_iterable(range(1, 9), shape=[2, 2, 2])
    var deep = vmap[total](in_axes=(0, None, None)).vmap(in_axes=(0, None, None)).vmap(in_axes=(0, None, None))((volume, Integer(2), Integer(3)))
    assert_equal(deep.shape(), [2, 2, 2])
    assert_equal(deep[1, 1, 1], 13)
    def mixed(parts: Tuple[Integer, Rational, Float, Complex]) raises -> Tuple[Rational, Float, Complex]:
        return (Rational(parts[0]) + parts[1], parts[2], parts[3])
    comptime assert type_of(vmap[mixed]()) == type_of(vmap[mixed](in_axes=(0, None, None, None)))
    comptime assert type_of(vmap[mixed]().vmap()) == type_of(vmap[mixed](in_axes=(0, None, None, None)).vmap(in_axes=(0, None, None, None)))
    var mixed_result = vmap[mixed](in_axes=(0, None, None, None)).vmap(in_axes=(0, None, None, None))((matrix, Rational(1, 3), floating, complex_value))
    assert_equal(mixed_result[0][1, 3], Rational(25, 3))
    assert_equal(mixed_result[1][1, 3].to_json(), floating.to_json())
    assert_equal(mixed_result[2][1, 3].to_json(), complex_value.to_json())

    var lengths_differ = (matrix, matrix.slice(0, stop=1), matrix)
    failed = False
    try:
        _ = vmap[total](in_axes=(0, 0, 0)).vmap(in_axes=(0, 0, 3), axis_size=-1)(lengths_differ)
    except error:
        failed = True
        assert_true(String(error).__contains__("axis"))
        assert_true(not String(error).__contains__("different lengths"))
    assert_true(failed)
    failed = False
    try:
        _ = vmap[total](in_axes=(0, 0, 0)).vmap(axis_size=-1)(lengths_differ)
    except error:
        failed = True
        assert_true(String(error).__contains__("different lengths"))
    assert_true(failed)
    with assert_raises():
        _ = vmap[total](in_axes=(0, 0, 0)).vmap(in_axes=(0, 0))((matrix, matrix, matrix))
    with assert_raises():
        _ = vmap[total](in_axes=(0, 0, 0)).vmap(axis_size=1)((matrix, matrix, matrix))
    assert_equal(vmap[total](in_axes=(0, None, None)).vmap()((matrix, Integer(2), Integer(3))).to_list(), (matrix + 5).to_list())

    var failures = Batch[Integer]([1, 2, 0, -1], shape=[2, 2])
    var failure_owner = failures._tensor_view()._owner
    var failure_references = failure_owner.count()
    def checked(parts: Three) raises -> Three:
        return (parts[0], parts[1], _tensor_failure(parts[0]))
    var destination = vmap[checked](in_axes=(0, None, None)).vmap(in_axes=(0, None, None))((matrix, Integer(2), Integer(3)))
    var original = destination[2].to_list()
    failed = False
    try:
        destination = vmap[checked](in_axes=(0, None, None)).vmap(in_axes=(0, None, None))((failures, Integer(2), Integer(3)))
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1: vmap failed at mapped index 0"))
        assert_true(String(error).__contains__("Use a positive input"))
    assert_true(failed)
    assert_equal(destination[2].to_list(), original)
    assert_equal(failure_owner.count(), failure_references)
    assert_equal(failures[1, 1], -1)

    # Explicitly retire local snapshots before checking the retained source.
    _ = views^
    _ = empty^
    _ = singleton^
    assert_equal(owner.count(), references)
    assert_equal(source[0], wide)
    assert_equal(matrix[1, 3], 8)


def test_public_boolean_tensor_mapping() raises:
    var values = Batch[Integer]([1, -2, 3, 0, 5, -6], shape=[2, 3])
    def positive(value: Integer) raises -> Bool:
        return value > 0
    def invert(value: Bool) raises -> Bool:
        return not value
    var mask: Mask = vmap[positive]().vmap()(values)
    assert_equal(mask.shape(), [2, 3])
    assert_equal(mask.ndim(), 2)
    assert_equal(mask.size(), 6)
    assert_equal(String(mask), "[[True, False, True], [False, True, False]]")
    assert_equal(mask.to_list(), [True, False, True, False, True, False])
    assert_true(mask[0, -1])
    assert_true(not mask[-1, 0])
    assert_true(mask[4])
    assert_equal(mask.count(), 3)
    assert_equal(vmap[invert]().vmap()(mask).to_list(), (~mask).to_list())
    var row = mask.at(0, 0)
    var repeated_rows: Mask = vmap[invert]().vmap(in_axes=None, axis_size=2)(row)
    assert_equal(repeated_rows.shape(), [2, 3])
    assert_equal(repeated_rows.to_list(), [False, True, False, False, True, False])
    var transposed = vmap[positive]().vmap(out_axes=-1)(values)
    assert_equal(transposed.shape(), [3, 2])
    assert_equal(transposed.to_list(), mask.transpose().to_list())
    assert_equal(mask.transpose([1, 0]).transpose().to_list(), mask.to_list())
    assert_equal(vmap[invert]().vmap(in_axes=-1, out_axes=-1)(mask).to_list(), (~mask).to_list())
    assert_equal(mask.at(1, -1).to_list(), [True, False])
    var scalar = mask.at(0, 0).at(0, 0)
    assert_equal(scalar.ndim(), 0)
    assert_equal(scalar.shape(), List[Int]())
    assert_true(scalar[Tuple()])
    assert_equal(String(scalar), "True")
    var flat = mask.reshape([6])
    assert_true(not Bool(flat._shape))
    assert_true(flat._storage.ptr() == mask._storage.ptr())
    assert_equal(vmap[invert]()(flat).to_list(), (~flat).to_list())
    assert_equal(mask[::2].shape(), [3])
    assert_equal((mask & ~mask).count(), 0)
    assert_true((mask | ~mask).all())
    assert_equal((mask ^ mask).shape(), mask.shape())
    with assert_raises():
        _ = mask & flat
    with assert_raises():
        _ = values.reshape([6])[mask]
    with assert_raises():
        _ = vmap[invert]()(mask)
    with assert_raises():
        _ = mask.item()
    with assert_raises():
        _ = mask.reshape([-1, 0])
    with assert_raises():
        _ = mask.transpose([0, 0])
    with assert_raises():
        _ = mask[0, 3]
    with assert_raises():
        _ = mask[0, 0, 0]
    with assert_raises():
        _ = Mask(List[Bool](), shape=[Int.MAX, 2])
    var empty_mask = Mask(List[Bool](), shape=[Int.MAX, 0, Int.MAX])
    assert_equal(empty_mask.transpose().shape(), [Int.MAX, 0, Int.MAX])
    assert_equal(empty_mask.at(0, -1).shape(), [0, Int.MAX])
    assert_equal(vmap[positive]().vmap(out_shape=[3])(values.slice(0, stop=0)).shape(), [0, 3])
    assert_equal(vmap[positive]().vmap(out_shape=Array[Int, 1](fill=3), out_axes=-1)(values.slice(0, stop=0)).shape(), [3, 0])
    var inner_empty = Batch[Integer](List[Integer](), shape=[2, 0])
    assert_equal(vmap[positive]().vmap()(inner_empty).shape(), [2, 0])
    # Bool leaves have a known shape, so empty nested mappings infer it.
    assert_equal(vmap[positive]().vmap()(values.slice(0, stop=0)).shape(), [0, 3])
    with assert_raises():
        _ = vmap[positive]().vmap(out_shape=[-1])(values.slice(0, stop=0))
    with assert_raises():
        _ = vmap[positive]().vmap(out_shape=[3], out_axes=2)(values.slice(0, stop=0))
    def whole(value: Mask) raises -> Mask:
        return ~value
    var repeated = vmap[whole](in_axes=None, axis_size=2)(mask)
    assert_equal(repeated.shape(), [2, 2, 3])
    assert_equal(repeated.at(0, 1).to_list(), (~mask).to_list())
    assert_equal(vmap[whole](out_shape=Array[Int, 1](fill=3))(Mask(List[Bool](), shape=[0, 3])).shape(), [0, 3])
    def both(a: Bool, b: Bool) raises -> Bool:
        return a and b
    assert_equal(vmap[both]().vmap()(mask, mask).to_list(), mask.to_list())
    var row_filter: Mask = vmap[both]().vmap(in_axes=(0, None))(mask, row)
    assert_equal(row_filter.shape(), [2, 3])
    assert_equal(row_filter.to_list(), [True, False, True, False, False, False])
    assert_equal(vmap[both]().vmap(in_axes=(None, 0))(row, mask).to_list(), row_filter.to_list())
    assert_equal(vmap[both]().vmap(in_axes=(None, None), axis_size=2)(row, row).to_list(), [True, False, True, True, False, True])
    assert_equal(vmap[both](in_axes=(0, None))(row, True).to_list(), row.to_list())
    assert_equal(vmap[both](in_axes=(None, 0))(True, row).to_list(), row.to_list())
    def selected_row(row: Mask, enabled: Bool) raises -> Tuple[Mask, Bool]:
        if not enabled:
            raise Error("Set enabled=True to return this row.")
        return (row, enabled)
    def unary_row(row: Mask) raises -> Tuple[Mask, Bool]:
        return (row, False)
    def tuple_row(parts: Tuple[Mask, Bool]) raises -> Tuple[Mask, Bool]:
        return parts
    var rows = vmap[selected_row](in_axes=(0, None), out_axes=(1, 0), out_shape=([3], None))
    var no_rows = rows(Mask(List[Bool](), shape=[0, 3]), False)
    assert_equal(no_rows[0].shape(), [3, 0])
    assert_equal(no_rows[1].shape(), [0])
    var zero_shared = vmap[selected_row](in_axes=(None, None), axis_size=0, out_shape=([3], None))(row, False)
    assert_equal(zero_shared[0].shape(), [0, 3])
    assert_equal(zero_shared[1].size(), 0)
    for signature in range(3):
        var rejected = False
        try:
            if signature == 0:
                _ = vmap[selected_row](in_axes=(0, None), out_axes=(2, 0), out_shape=([3], List[Int]([1])))(mask, False)
            elif signature == 1:
                _ = vmap[unary_row](out_axes=(2, 0), out_shape=([3], List[Int]([1])))(mask)
            else:
                _ = vmap[tuple_row](in_axes=(0, None), out_axes=(2, 0), out_shape=([3], List[Int]([1])))((mask, False))
        except error:
            rejected = True
            assert_true(String(error).__contains__("axis is out of range"))
        assert_true(rejected)
    def mixed(x: Integer) raises -> Tuple[Integer, Bool, Bool]:
        return (x, x > 0, x == 0)
    var mixed_result = vmap[mixed]().vmap(out_axes=(0, -1, 0))(values)
    assert_equal(mixed_result[0][1, 1], 5)
    assert_equal(mixed_result[1].to_list(), transposed.to_list())
    assert_equal(mixed_result[2].count(), 1)
    assert_equal(vmap[mixed]().vmap(out_shape=([3], [3], [3]))(values.slice(0, stop=0))[1].shape(), [0, 3])
    def tuple_flags(parts: Tuple[Bool, Bool, Bool]) raises -> Tuple[Bool, Bool, Bool]:
        return parts
    var flags = vmap[tuple_flags]().vmap()((mask, mask, mask))
    assert_equal(flags[2].shape(), [2, 3])
    assert_equal(flags[2].to_list(), mask.to_list())
    var shared_flags = vmap[tuple_flags]().vmap(in_axes=(0, None, None))((mask, row, ~row))
    assert_equal(shared_flags[0].to_list(), mask.to_list())
    assert_equal(shared_flags[1].shape(), [2, 3])
    assert_equal(shared_flags[1].to_list(), [True, False, True, True, False, True])
    assert_equal(shared_flags[2].shape(), [2, 3])
    assert_equal(shared_flags[2].to_list(), repeated_rows.to_list())
    var cube = values.reshape([1, 2, 3])
    assert_equal(vmap[positive]().vmap().vmap()(cube).shape(), [1, 2, 3])
    var source = Batch[Integer]([1, 2])
    var owner = mask._storage
    var references = owner.count()
    def varying(x: Integer, flags: Mask) raises -> Tuple[Integer, Mask, Mask]:
        return (x, flags, flags[:Int(x)])
    var destination = (Batch[Integer]([99]), Mask([False]), Mask([True]))
    var failed = False
    try:
        destination = vmap[varying](in_axes=(0, None))(source, mask)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
        assert_true(String(error).__contains__("inconsistent mask shapes"))
    assert_true(failed)
    assert_equal(destination[0][0], 99)
    assert_equal(destination[1].shape(), [1])
    assert_true(destination[2][0])
    assert_equal(owner.count(), references)
    assert_equal(mask.shape(), [2, 3])
    assert_equal(mask.count(), 3)


def test_native_float_batch_arithmetic_and_mapping() raises:
    var context = ArithmeticContext(format=FloatFormat(1024))
    var wide = Float((Integer(1) << 1024) - 17, context=context)
    var source = Batch[Float]([wide, Float(-3), Float(0), Float(7)])
    var packed = Batch[Float](source[:])
    var saved = source[::-1]
    # A strided selection reads the copy's run in place.
    var gathered = packed[3:0:-2]._tensor_view()
    assert_equal(Int(gathered._owner.ptr()), Int(packed._owner.ptr()))
    assert_equal(gathered.item(1), Float(-3))
    var products = source * packed
    var sums = source + saved
    var differences = source - saved
    var divisors = Batch[Float]([Float(3), Float(4), Float(5), Float(6)])
    var quotients = source / divisors
    var mapped = vmap[_tensor_family_product[Float]]()(saved, divisors)
    for i in range(4):
        assert_equal(products[i], source[i] * source[i])
        assert_equal(sums[i], source[i] + saved[i])
        assert_equal(differences[i], source[i] - saved[i])
        assert_equal(quotients[i], source[i] / divisors[i])
        assert_equal(mapped[i], saved[i] * divisors[i])
    assert_equal(sum(source).to_json(), sum(packed).to_json())
    assert_equal(sum_tree(source).to_json(), sum_tree(packed).to_json())
    assert_equal(dot(source, divisors).to_json(), dot(packed, divisors).to_json())
    assert_equal(source.to_json(), packed.to_json())
    var rounded = vmap[float.multiply]()(source, divisors, context=context)
    for i in range(4):
        assert_equal(rounded[i].to_json(), multiply(source[i], divisors[i], context=context).to_json())
    var before = source.to_json()
    var trap = ArithmeticContext(trap_invalid=True)
    var failed = False
    try:
        source = vmap[float.divide]()(source, source, context=trap)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 2"))
    assert_true(failed)
    assert_equal(source.to_json(), before)
    source += 1
    assert_equal(saved[3], wide)
    assert_equal(source[2], Float(1))


def test_native_float_integer_results_and_masks() raises:
    var numerators: List[Int] = [-7, -4, -1, 0, 1, 4, 7, 9, -9, 3, -3, 8, -8, 2, -2, 11, -11]
    var source = Batch[Float]([
        Float(Rational(numerators[i], 2), context=ArithmeticContext(format=FloatFormat(256 if i % 2 else 1024)))
        for i in range(len(numerators))
    ])
    source[3] = Float.zero(negative=True)
    var before = source.to_json()
    var packed = Batch[Float](source[:])
    for count in [0, 1, 8, 9, 17]:
        for input in [source[:count], source[:count][::-1], packed[:count]]:
            var floors = input.floor()
            var ceilings = input.ceil()
            var truncations = input.trunc()
            var signs = input.sign()
            for result in [floors, ceilings, truncations, signs]:
                assert_equal(len(result), count)
            for i in range(count):
                assert_equal(floors[i], input[i].floor())
                assert_equal(ceilings[i], input[i].ceil())
                assert_equal(truncations[i], input[i].trunc())
                assert_equal(signs[i], Integer(input[i].sign()))
            assert_equal(_float_integral(input, 0).to_json(), truncations.to_json())
    var floors = source.floor()
    var ceilings = source.ceil()
    var truncations = source.trunc()
    for i in range(len(source)):
        var n = numerators[i]
        assert_equal(floors[i], Integer(n) // 2)
        assert_equal(ceilings[i], -((-Integer(n)) // 2))
        assert_equal(truncations[i], (Integer(abs(n)) // 2) * (-1 if n < 0 else 1))
    var selected = source[Mask([n % 2 == 0 for n in numerators])]
    var exact = selected[::-1].to_integer_exact()
    for i in range(len(exact)):
        assert_equal(exact[i], selected[len(selected) - 1 - i].trunc())
    var top = Integer(1) << 1023
    var wide = Batch[Float]([
        Float(top + 17, context=ArithmeticContext(format=FloatFormat(1024))),
        Float(-(top + 17), context=ArithmeticContext(format=FloatFormat(1024))),
    ])
    assert_equal(wide.to_integer_exact()[0], top + 17)
    assert_equal(wide.to_integer_exact()[1], -(top + 17))
    assert_equal(Batch[Float]([ldexp(Float(3), Integer(2048))]).to_integer_exact()[0], Integer(3) << 2048)
    var tiny = ldexp(Float(1, context=ArithmeticContext(format=FloatFormat(256, emin=Int.MIN))), Integer(Int.MIN))
    var small = Batch[Float]([tiny, -tiny])
    assert_equal(small.floor()[0], 0)
    assert_equal(small.floor()[1], -1)
    assert_equal(small.ceil()[0], 1)
    assert_equal(small.ceil()[1], 0)
    assert_true((small.trunc() == 0).all())
    assert_equal(Batch[Float]([Float(-3)]).sign()[0], -1)
    var snapshot = source[::-2]
    var saved = snapshot.to_json()
    source += 1
    assert_equal(snapshot.to_json(), saved)
    assert_equal(packed.to_json(), before)
    assert_equal(snapshot.floor()[0], -6)

    var specials = Batch[Float]([
        Float.zero(negative=True), Float(0), Float(-2), Float(3),
        Float.infinity(), Float.infinity(negative=True), Float.nan(),
        Float(1, context=ArithmeticContext(format=FloatFormat(3, emin=-4, emax=4))),
        wide[0],
    ])
    for input in [specials[:], specials[::-1], specials[::2], specials[:0]]:
        var queries = [input.signbit(), input.is_zero(), input.is_finite(), input.is_infinite(), input.is_nan()]
        var other = input[::-1]
        var comparisons = [input == other, input != other, input < other, input <= other, input > other, input >= other]
        for query in queries:
            assert_equal(len(query), len(input))
        for comparison in comparisons:
            assert_equal(len(comparison), len(input))
        for i in range(len(input)):
            var a = input[i]
            var b = other[i]
            var expected_queries = [a.signbit(), a.is_zero(), a.is_finite(), a.is_infinite(), a.is_nan()]
            var expected_comparisons = [a == b, a != b, a < b, a <= b, a > b, a >= b]
            for op in range(5):
                assert_equal(queries[op][i], expected_queries[op])
            for op in range(6):
                assert_equal(comparisons[op][i], expected_comparisons[op])
    assert_equal(specials[6:7].is_nan().count(), 1)
    assert_true(not (specials[6:7] == specials[6:7])[0])
    assert_true((specials[6:7] != specials[6:7])[0])
    assert_true((specials[:1] == specials[1:2])[0])
    assert_equal(specials[specials.is_finite()].sign()[0], 0)
    assert_equal(specials[:0].to_integer_exact().to_list(), List[Integer]())
    assert_true((Batch[Float]() == Float.nan()).all())
    var edge = Integer(1) << 256
    var low = Batch[Float]([Float(edge, context=ArithmeticContext(format=FloatFormat(256)))])
    var integers = Batch[Integer]([edge + 1])
    var rationals = Batch[Rational]([Rational(2 * edge + 1, 2)])
    assert_true((low < integers)[0] and (integers > low)[0])
    assert_true((low < rationals)[0] and (rationals > low)[0])
    assert_true((low < Rational(2 * edge + 1, 2))[0])
    assert_true((Rational(2 * edge + 1, 2) > low)[0])
    var native = Batch[Float]([Float(UInt64.MAX), Float(18446744073709551616)])
    assert_true((native <= 18446744073709551616).all())
    assert_true((Integer(18446744073709551616) >= native).all())
    assert_true((native >= UInt64.MAX).all())
    assert_true(_float_comparison(UInt64.MAX, native, 3).all())
    assert_true((native > Float64(1.5)).all())
    assert_true(_float_comparison(Float64(1.5), native, 2).all())
    assert_true(_float_comparison(native, Batch[Float]([Float(1)]), 4, broadcast_b=True).all())
    with assert_raises():
        _ = native < Batch[Float]([Float(1)])
    with assert_raises():
        _ = Batch[Float]() == Batch[Float]([Float.nan()])

    var oversized = ldexp(Float(1), Integer(_MAX_RESULT_BITS))
    var failures = Batch[Float]([Float(2), Float(Rational(-7, 2)), Float.infinity(), oversized, Float.nan()])
    var original = failures.to_json()
    var compatible = Batch[Float](failures[:])
    var valid = Mask([True, True, False, False, False])
    assert_equal(compatible[valid].trunc().to_json(), failures[:2].trunc().to_json())
    var destination = Batch[Integer]([42])
    var unchanged = destination.to_json()
    for op in range(4):
        var message = String()
        try:
            destination = _float_integral(failures, op)
        except error:
            message = String(error)
        assert_true(("fractional part" if op == 3 else "nonfinite Float") in message)
        assert_true(("element 1" if op == 3 else "element 2") in message)
        assert_true("The destination is unchanged." in message)
        assert_equal(destination.to_json(), unchanged)
    for input in [failures[3:4], failures[::-1][1:2]]:
        var message = String()
        try:
            destination = input.trunc()
        except error:
            message = String(error)
        assert_equal(message, "Cannot convert Float to Integer at batch element 0: the result exceeds addressable storage; keep the value as Float or scale it down. The destination is unchanged.")
    for fail in [0, 1, 4]:
        var message = String()
        try:
            destination = _float_sign(failures, fail)
        except error:
            message = String(error)
        if fail == 4:
            assert_equal(message, "Cannot take the sign of NaN at batch element 4; check is_nan() before requesting a numerical sign. The destination is unchanged.")
        else:
            assert_true(String("Injected Float batch failure at element ", fail) in message)
        assert_equal(destination.to_json(), unchanged)
    for fail in [0, 1, 2]:
        var message = String()
        try:
            destination = _float_integral(failures, 0, fail=fail)
        except error:
            message = String(error)
        assert_true(("nonfinite Float to Integer at batch element 2" if fail == 2 else String("Injected Float batch failure at element ", fail)) in message)
        assert_equal(destination.to_json(), unchanged)
    var fractional_message = String()
    try:
        destination = _float_integral(failures, 3, fail=1)
    except error:
        fractional_message = String(error)
    assert_equal(fractional_message, "Cannot convert Float exactly to Integer at batch element 1: the value has a fractional part; keep it as Float or explicitly choose floor(), ceil(), or trunc(). The destination is unchanged.")
    assert_equal(destination.to_json(), unchanged)
    var message = String()
    try:
        destination = _float_integral(failures[::-1], 3, fail=0)
    except error:
        message = String(error)
    assert_true("nonfinite Float to Integer at batch element 0" in message)
    assert_equal(destination.to_json(), unchanged)
    assert_equal(failures.to_json(), original)
    destination = failures[:2].trunc()
    var expected: List[Integer] = [2, -3]
    assert_equal(destination.to_list(), expected)


def _check_mapped_float(
    a: Batch[Float], divisors: Batch[Float], counts: Batch[Integer], context: ArithmeticContext,
) raises:
    """Mapped Float functions agree with element-by-element scalar calls."""
    var c = a[::-1]
    var sums = vmap[float.add]()(a, divisors, context=context)
    var differences = vmap[float.subtract]()(a, divisors, context=context)
    var products = vmap[float.multiply]()(a, divisors, context=context)
    var quotients = vmap[float.divide]()(a, divisors, context=context)
    var roots = vmap[float.sqrt]()(a, context=context)
    var powers = vmap[float.pow_int]()(a, counts, context=context)
    var scaled = vmap[ldexp]()(a, counts, context=context)
    var fused = vmap[fma]()(a, divisors, c, context=context)
    for i in range(len(a)):
        assert_equal(sums[i].to_json(), add(a[i], divisors[i], context=context).to_json())
        assert_equal(differences[i].to_json(), subtract(a[i], divisors[i], context=context).to_json())
        assert_equal(products[i].to_json(), multiply(a[i], divisors[i], context=context).to_json())
        assert_equal(quotients[i].to_json(), divide(a[i], divisors[i], context=context).to_json())
        assert_equal(roots[i].to_json(), sqrt(a[i], context=context).to_json())
        assert_equal(powers[i].to_json(), pow_int(a[i], counts[i], context=context).to_json())
        assert_equal(scaled[i].to_json(), ldexp(a[i], counts[i], context=context).to_json())
        assert_equal(fused[i].to_json(), fma(a[i], divisors[i], c[i], context=context).to_json())


def test_mapped_float_functions_rounding_specials_and_failures() raises:
    var a = Batch[Float]([
        Float(Rational(1, 3)), Float(-7), Float.zero(negative=True), Float(0),
        Float.infinity(), Float.infinity(negative=True), Float.nan(),
        Float("0x1p-20"), Float(513),
    ])
    var divisors = Batch[Float]([
        Float(3), Float(5), Float(-2), Float(0), Float(0), Float(1),
        Float(7), Float(17), Float(513),
    ])
    var counts = Batch[Integer]([3, -1, 0, 2, 1, -2, 5, 4, 3])
    for mode in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                 RoundingMode.toward_positive, RoundingMode.toward_negative,
                 RoundingMode.away_from_zero]:
        _check_mapped_float(a, divisors, counts[::-1], ArithmeticContext(format=FloatFormat(9, emin=-8, emax=8), rounding=mode))
    var unchanged = a.to_json()
    var destination = vmap[float.divide]()(a, 3)
    var before = destination.to_json()
    var trap = ArithmeticContext(trap_invalid=True)
    var errors = Batch[Float]([Float(4), Float(-1), Float(-9)])
    var failed = False
    try:
        destination = vmap[float.sqrt]()(errors, context=trap)
    except error:
        failed = True
        assert_true(String(error).__contains__("mapped index 1"))
    assert_true(failed)
    assert_equal(destination.to_json(), before)
    assert_equal(a.to_json(), unchanged)
    assert_equal(len(vmap[float.sqrt]()(Batch[Float](), context=trap)), 0)
    var huge = (Integer(1) << 256) + 1
    var identities = vmap[float.pow_int]()(Batch[Float]([Float(-1), Float(1)]), huge)
    assert_equal(identities[0], Float(-1))
    assert_equal(identities[1], Float(1))


def test_native_float_compound_formats_overlap_and_rollback() raises:
    var values = List[Float]()
    for i in range(9):
        values.append(Float(Rational(i + 3, 7), context=ArithmeticContext(format=FloatFormat(9 + 8 * (i % 3)))))
    var original = Batch[Float](values)
    var snapshot = original[::-1]
    var encoded = original.to_json()
    var powers = Batch[Integer]([-1, 2, 0, 3, -2, 1, 4, -1, 2])
    var context = ArithmeticContext(format=FloatFormat(256), rounding=RoundingMode.toward_negative)
    for operation in range(5):
        var target = original
        var expected = Batch[Float].from_json(encoded)
        if operation == 4:
            expected._update_float_power(powers, context=context)
            target._update_float_power(powers, context=context)
        else:
            expected._update_float(expected[::-1], operation, context=context)
            target._update_float(target[::-1], operation, context=context)
        assert_equal(target.to_json(), expected.to_json())
        for i in range(9):
            assert_equal(target[i].format(), original[i].format())
        for fail in [0, 4, 8]:
            target = original
            with assert_raises():
                if operation == 4:
                    target._update_float_power(powers, fail=fail)
                else:
                    target._update_float(target[::-1], operation, fail=fail)
            assert_equal(target.to_json(), encoded)
        assert_equal(original.to_json(), encoded)
        assert_equal(snapshot[0], original[8])
    var target = original
    var expected = Batch[Float].from_json(encoded)
    var rhs = Batch[Rational]([Rational(1, i + 3) for i in range(9)])
    expected._update_float(rhs, 2)
    target *= rhs
    assert_equal(target.to_json(), expected.to_json())
    var before = target.to_json()
    with assert_raises():
        target._update_float(Batch[Float]([Float(0)]), 3, context=ArithmeticContext(trap_divide_by_zero=True))
    assert_equal(target.to_json(), before)
    var zeros = Batch[Float]([Float(1) for _ in range(9)])
    zeros[4] = Float(0)
    with assert_raises():
        target._update_float(zeros, 3, context=ArithmeticContext(trap_divide_by_zero=True))
    assert_equal(target.to_json(), before)
    target /= 1


def _check_float_native_transfer[dtype: DType]() raises:
    var numbers: List[SIMD[dtype, 1]] = [SIMD[dtype, 1](3)]
    var result = Batch[Float].from_native(numbers)
    assert_equal(result[0], Float(3))


def test_native_float_transfers_and_checked_json() raises:
    var format = FloatFormat(1024)
    var context = ArithmeticContext(format=format)
    var wide = Float((Integer(1) << 1024) - 17, context=context)
    var source = Batch[Float]([wide, Float.zero(negative=True), Float.nan(), Float.infinity(), Float(7)])
    source._real_format = format
    var encoded = source.to_json()
    var reversed = source[::-1]
    var copied = Batch[Float](reversed)
    assert_equal(copied.to_json(), reversed.to_json())
    var cursor = iter(reversed)
    _ = next(cursor)
    var remaining = Batch[Float].from_iterable(cursor^)
    assert_equal(remaining.to_json(), reversed[1:].to_json())
    var native = Batch[Float].from_native([UInt64.MAX])
    assert_equal(native[0], Float(UInt64.MAX))
    _check_float_native_transfer[DType.int8]()
    _check_float_native_transfer[DType.uint8]()
    _check_float_native_transfer[DType.int16]()
    _check_float_native_transfer[DType.uint16]()
    _check_float_native_transfer[DType.int32]()
    _check_float_native_transfer[DType.uint32]()
    _check_float_native_transfer[DType.int64]()
    _check_float_native_transfer[DType.uint64]()
    _check_float_native_transfer[DType.float16]()
    _check_float_native_transfer[DType.bfloat16]()
    _check_float_native_transfer[DType.float32]()
    _check_float_native_transfer[DType.float64]()
    assert_equal(Batch[Float].from_iterable(source).to_json(), encoded)
    var exact = Batch[Float](Batch[Rational]([Rational(1, 3)]))
    assert_equal(exact[0], Float(Rational(1, 3)))
    var integers: List[Integer] = [Integer(1) << 256, Integer(-3)]
    assert_equal(Batch[Float](integers)[0], Float(integers[0]))
    assert_equal(Batch[Float](integers^)[1], Float(-3))
    for empty in [False, True]:
        var input = source[0:0] if empty else source[:]
        var json = input.to_json()
        var budget = _ConversionBudget(ConversionLimits(max_allocated_bytes=Int.MAX))
        var values, decoded_format = _parse_float_batch_json(json, budget)
        assert_equal(decoded_format, format)
        assert_equal(values._layout.size, len(input))
        if empty:
            var decoded_bytes = 0
            for token in [String("version"), "family", "float-batch", "default_format",
                          "precision", String(format.precision()), "emin", String(format.emin()),
                          "emax", String(format.emax()), "values"]:
                var capacity = 1
                decoded_bytes += 1
                while capacity < token.byte_length():
                    capacity *= 2
                    decoded_bytes += capacity
            assert_equal(budget.allocated_used,
                         decoded_bytes +
                         size_of[ArcPointer[_Values[Float]]._inner_type]() +
                         _layout_record_bytes())
        var restored = Batch[Float].from_json(json, limits=ConversionLimits(max_allocated_bytes=budget.allocated_used))
        assert_equal(restored.to_json(), json)
        with assert_raises():
            restored = Batch[Float].from_json(json, limits=ConversionLimits(max_allocated_bytes=budget.allocated_used - 1))
        assert_equal(restored.to_json(), json)
        assert_equal(Batch[Float](input).to_json(), json)
    var mask = Mask([False, True, False, False, True])
    var selected = source[mask]
    assert_equal(len(selected._tensor_view()._owner[]), 2)
    assert_equal(selected[0].to_json(), source[1].to_json())
    assert_equal(selected[1], Float(7))
    assert_equal(reversed[mask][0].to_json(), source[3].to_json())
    assert_equal(Batch[Float]([wide], shape=[])[:].to_json(), Batch[Float]([wide], shape=[]).reshape([1]).to_json())
    var before_empty = source._owner
    source[Mask([False, False, False, False, False])] = Float(9)
    assert_equal(Int(source._owner.ptr()), Int(before_empty.ptr()))
    with assert_raises():
        source[Mask([False])] = Float(9)
    with assert_raises():
        source[mask] = Batch[Float]([Float(1)])[:]
    for fail in [0, 1]:
        with assert_raises():
            source._assign_mask(mask, Batch[Float]([Float(11)]), True, fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    for fail in [0, 2, 4]:
        with assert_raises():
            source._assign_slice(StridedSlice(None, None, -1), source[:], fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    source[mask] = Float(11)
    assert_equal(source[1], Float(11))
    assert_equal(source[4], Float(11))
    source[::-1] = reversed
    assert_equal(source.to_json(), encoded)
    source[-1] = wide
    assert_equal(source[-1].precision(), 1024)
    assert_equal(reversed[0], Float(7))
    with assert_raises():
        source[:] = Batch[Float]([Float(1)])[:]
    var before = source.to_json()
    with assert_raises():
        source = Batch[Float].from_json(encoded, limits=ConversionLimits(max_values=4))
    with assert_raises():
        source = Batch[Float].from_json(encoded + " trailing")
    assert_equal(source.to_json(), before)


def test_native_integer_storage_transfers_and_compatibility() raises:
    var wide = (Integer(1) << 1024) - 17
    var source = Batch[Integer]([wide, Integer(-3), Integer(0), Integer(7), wide])
    var snapshot = source[::-1]
    var encoded = source.to_json()
    var scalar_memory = _MemoryCensus()
    scalar_memory.add(wide)
    var memory = _MemoryCensus()
    memory.add(source)
    var allocations = memory.finish().heap().allocations
    # The owner handle and value list, plus one shared layout record.
    assert_equal(allocations, scalar_memory.finish().heap().allocations + 5)
    # A slice shares the values; only its layout record is new.
    memory.add(snapshot)
    var cursor = iter(snapshot)
    memory.add(cursor)
    assert_equal(memory.finish().heap().allocations, allocations + 3)
    assert_equal(memory.finish().snapshot_only.allocations, 0)
    _ = next(cursor)
    var remaining = Batch[Integer].from_iterable(cursor^)
    assert_equal(remaining.to_json(), snapshot[1:].to_json())
    for operation in range(3):
        var result = source._binary(source, operation)
        for i in range(len(source)):
            var x = source[i]
            assert_equal(result[i], x + x if operation == 0 else x - x if operation == 1 else x * x)
    assert_equal((source[::-1] - wide)[1], 7 - wide)
    assert_equal((wide - source[::2])[1], wide)
    with assert_raises():
        _ = source[:] + Batch[Integer]([Integer(1)])
    for count in [0, 1, 8, 9]:
        var values = Batch[Integer].from_native([i for i in range(count)])
        var json = values.to_json()
        var restored = Batch[Integer].from_json(json)
        assert_equal(restored.to_json(), json)
        assert_equal(len(values + 1), count)
        assert_equal(Batch[Integer](values[::-1]).to_json(), values[::-1].to_json())
    var mask = Mask([False, True, False, True, False])
    var gathered = source[mask]
    assert_equal(len(gathered._tensor_view()._owner[]), 2)
    for fail in [0, 1]:
        with assert_raises():
            source._assign_mask(mask, Batch[Integer]([wide]), True, fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    for fail in [0, 2, 4]:
        with assert_raises():
            source._assign_slice(StridedSlice(None, None, -1), source[:], fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    source[1:] = source[:-1]
    assert_equal(source[1], wide)
    assert_equal(source[2], -3)
    source[0] = 18446744073709551616
    assert_equal(source[0], Integer(1) << 64)
    assert_equal(snapshot[-1], wide)
    var before = source.to_json()
    with assert_raises():
        source._assign(source, 2, fail_after_tile=0)
    assert_equal(source.to_json(), before)
    source += 1
    assert_equal(source[0], (Integer(1) << 64) + 1)
    assert_equal(snapshot[-1], wide)
    var low = 0
    var high = 100000
    while low < high:
        var middle = (low + high) // 2
        var accepted = False
        try:
            _ = Batch[Integer].from_json(encoded, limits=ConversionLimits(max_allocated_bytes=middle))
            accepted = True
        except:
            pass
        if accepted:
            high = middle
        else:
            low = middle + 1
    assert_equal(Batch[Integer].from_json(encoded, limits=ConversionLimits(max_allocated_bytes=low)).to_json(), encoded)
    with assert_raises():
        source = Batch[Integer].from_json(encoded, limits=ConversionLimits(max_allocated_bytes=low - 1))
    assert_equal(source[0], (Integer(1) << 64) + 1)
    assert_equal(snapshot.to_json(), Batch[Integer].from_json(encoded)[::-1].to_json())


def _integer_compound(mut destination: Batch[Integer], rhs: Batch[Integer], operation: Int, fail: Int = -1) raises:
    # Compound-update codes: 0-2 + - *, 3 //, 4-6 & | ^, 8-9 shifts, 10 **, 11 %.
    if operation <= 2:
        destination._assign(rhs, operation, fail)
    elif operation == 3 or operation == 11:
        destination._assign_division(rhs, Int(operation == 11), fail_after_tile=fail)
    else:
        destination._assign_bit(rhs, operation - 4, fail)


def _assert_native_integer_result(actual: Batch[Integer], expected: Batch[Integer]) raises:
    assert_equal(actual.to_json(), expected.to_json())


def _integer_bit_reference(x: Integer, y: Integer, operation: Int) raises -> Integer:
    if operation == 0:
        return x & y
    if operation == 1:
        return x | y
    if operation == 2:
        return x ^ y
    if operation == 3:
        return ~x
    if operation == 4:
        return x << y
    if operation == 5:
        return x >> y
    if operation == 6:
        return x ** y
    if operation == 7:
        return Integer(x.sign())
    if operation == 8:
        return Integer(x.magnitude_bit_length())
    if operation == 9:
        return -x
    return abs(x)


def _integer_relation(x: Integer, y: Integer, operation: Int) -> Bool:
    if operation == 0:
        return x == y
    if operation == 1:
        return x != y
    if operation == 2:
        return x < y
    if operation == 3:
        return x <= y
    if operation == 4:
        return x > y
    return x >= y


def test_native_integer_execution_and_transactionality() raises:
    var wide = (Integer(1) << 1024) + 17
    var a = Batch[Integer]([wide, -wide, Integer(0), Integer(-7), Integer(7), Integer(1), Integer(-1), wide, -wide])
    var b = Batch[Integer]([Integer(3), Integer(-3), Integer(2), Integer(5), Integer(-5), Integer(7), Integer(8), Integer(9), Integer(-9)])
    var counts = Batch[Integer].from_native([0, 1, 2, 3, 4, 5, 6, 7, 8])
    for operation in range(11):
        var rhs = counts if 4 <= operation <= 6 else b
        var result = a._bit_result(rhs, operation)
        for i in range(len(a)):
            assert_equal(result[i], _integer_bit_reference(a[i], rhs[i], operation))
    var floor_q = a // b
    var floor_r = a % b
    for i in range(len(a)):
        var sq, sr = a[i]._div_rem(b[i], 0)
        assert_equal(floor_q[i], sq)
        assert_equal(floor_r[i], sr)
    for operation in range(6):
        var relation = a._compare(b, operation)
        var shared = a._compare(Batch[Integer]([Integer(3)]), operation, True)
        for i in range(len(a)):
            assert_equal(relation[i], _integer_relation(a[i], b[i], operation))
            assert_equal(shared[i], _integer_relation(a[i], Integer(3), operation))
    var selected = a[::-2]
    var divisor = b[::2]
    for operation in [3, 7, 8, 10]:
        var unary = ~selected if operation == 3 else selected.sign() if operation == 7 else selected.magnitude_bit_length() if operation == 8 else vmap[integer.abs]()(selected)
        for i in range(len(selected)):
            assert_equal(unary[i], _integer_bit_reference(selected[i], Integer(0), operation))
    var q, r = vmap[div_rem_floor]()(selected, divisor)
    for i in range(len(selected)):
        assert_equal(q[i], selected[i] // divisor[i])
        assert_equal(r[i], selected[i] % divisor[i])
        assert_equal((selected > divisor)[i], selected[i] > divisor[i])
    assert_true((Integer(1) < selected).to_list() == (selected > Integer(1)).to_list())
    _assert_native_integer_result(Integer(29) // divisor, Batch[Integer]([Integer(29) // divisor[i] for i in range(len(divisor))]))
    _assert_native_integer_result(selected % Integer(-5), Batch[Integer]([selected[i] % Integer(-5) for i in range(len(selected))]))
    _assert_native_integer_result(Integer(2) ** counts, Batch[Integer]([Integer(2) ** counts[i] for i in range(len(counts))]))
    for operation in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]:
        var rhs = counts if 8 <= operation <= 10 else b
        var expected = Batch[Integer]([a[i]._compound_value(rhs[len(a) - 1 - i], operation) for i in range(len(a))])
        var destination = a
        var snapshot = destination[:]
        for fail in [0, 1]:
            with assert_raises():
                _integer_compound(destination, rhs[::-1], operation, fail)
            _assert_native_integer_result(destination, a)
        _integer_compound(destination, rhs[::-1], operation)
        _assert_native_integer_result(destination, expected)
        assert_equal(snapshot.to_json(), a.to_json())
    var overlapping = a
    overlapping += overlapping[::-1]
    _assert_native_integer_result(overlapping, a + a[::-1])
    for family in range(3):
        for fail in [0, 1]:
            var destination = a
            var failed = False
            try:
                if family == 0:
                    destination._assign(b, 2, fail_after_tile=fail)
                elif family == 1:
                    destination._assign_bit(b, 0, fail_after_tile=fail)
                else:
                    destination._assign_division(b, 0, fail_after_tile=fail)
            except error:
                failed = True
                assert_true("Injected" in String(error))
            assert_true(failed)
            assert_equal(destination.to_json(), a.to_json())
    var scalar = a
    scalar += 18446744073709551616
    scalar ^= UInt64.MAX
    scalar //= Integer(3)
    assert_equal(scalar[0], ((a[0] + (Integer(1) << 64)) ^ Integer(UInt64.MAX)) // 3)
    var huge = Integer(1) << 256
    assert_true(((Batch[Integer]([Integer(0)]) << huge) == 0).all())
    assert_true(((a >> huge) == Batch[Integer]([Integer(-1 if a[i] < 0 else 0) for i in range(len(a))])).all())
    var identities: List[Integer] = [0, 1, -1]
    assert_equal((Batch[Integer](identities) ** (huge + 1)).to_list(), identities)
    for operation in [4, 5, 6]:
        var bad = counts
        bad[0] = huge
        bad[-1] = -1
        var native_error = String()
        var destination = a
        try:
            destination._assign_bit(bad, operation, fail_after_tile=0)
        except error:
            native_error = String(error)
        assert_equal(destination.to_json(), a.to_json())
        assert_true("logical element 8" in native_error)
    for operation in [4, 6]:
        var native_error = String()
        try:
            _ = a._bit_result(Batch[Integer]([huge]), operation, broadcast=True)
        except error:
            native_error = String(error)
        assert_true("logical element 0" in native_error)
    var zero = b
    zero[2] = 0
    zero[-1] = 0
    var unchanged = a
    try:
        unchanged._assign_division(zero, 0, fail_after_tile=0)
        assert_true(False)
    except error:
        assert_true("logical element 2" in String(error))
    _assert_native_integer_result(unchanged, a)
    with assert_raises():
        unchanged //= Batch[Integer]([Integer(0)])
    try:
        unchanged //= Batch[Integer]([Integer(0)])
        assert_true(False)
    except error:
        assert_true("batches of lengths 9 and 1" in String(error))
    _assert_native_integer_result(unchanged, a)
    for count in [0, 1, 8, 9]:
        var values = Batch[Integer].from_native([i for i in range(count)])
        var empty_q, empty_r = vmap[div_rem_floor]()(values, Integer(3))
        assert_equal(len(empty_q) + len(empty_r), 2 * count)
        assert_equal(len(~values), count)
        assert_equal(len(values < 4), count)
        values += 1
        values **= 2
    var empty = Batch[Integer].from_native(List[Int]())
    empty //= 0
    empty <<= -1
    assert_equal(len(empty), 0)


def _check_native_integer_math[operation: Int](a: Batch[Integer], b: Batch[Integer], c: Batch[Integer]) raises:
    # Mapped public functions agree with the scalar reference element by element.
    var actual: Batch[Integer]
    comptime if operation == 0:
        actual = vmap[factorial]()(a)
    elif operation == 1:
        actual = vmap[factorial2]()(a)
    elif operation == 2:
        actual = vmap[integer.comb]()(a, b)
    elif operation == 3:
        actual = vmap[iroot]()(a, b)
    elif operation == 4:
        actual = vmap[inverse_mod]()(a, b)
    elif operation == 5:
        actual = vmap[div_exact]()(a, b)
    else:
        actual = vmap[pow_mod]()(a, b, c)
    for i in range(len(a)):
        assert_equal(actual[i], _math_scalar[operation](a[i], b[i], c[i]))


def test_native_integer_math_reductions_and_mixed_inputs() raises:
    var wide = (Integer(1) << 256) + 1
    var counts = Batch[Integer].from_native([0, 1, 2, 3, 4, 5, 64, 128, 384])
    var small = Batch[Integer].from_iterable(range(9))
    var odd = small * 2 + 1
    var powers = Batch[Integer]([wide, -wide, Integer(0), Integer(1), Integer(-1), wide**3 - 1, wide**3, wide**3 + 1, -wide**3])
    var degrees = small * 0 + 3
    var moduli = small * 0 + (Integer(1) << 256)
    _check_native_integer_math[0](counts[::-1], small[:], small[:])
    _check_native_integer_math[1](counts[:], small[:], small[:])
    _check_native_integer_math[2](powers[:], small[:], small[:])
    _check_native_integer_math[3](powers[::-1], degrees[:], small[:])
    _check_native_integer_math[4](odd[::-1], moduli[:], small[:])
    _check_native_integer_math[5]((odd * wide)[:], odd[:], small[:])
    _check_native_integer_math[6](odd[:], (small - 4)[:], moduli[::-1])
    var before = powers.to_json()
    var magnitudes = vmap[integer.abs]()(powers)
    var gcds = vmap[gcd]()(powers, odd)
    var lcms = vmap[lcm]()(powers, odd)
    var roots = vmap[isqrt]()(magnitudes)
    for i in range(len(powers)):
        assert_equal(gcds[i], gcd(powers[i], odd[i]))
        assert_equal(lcms[i], lcm(powers[i], odd[i]))
        assert_equal(roots[i], isqrt(magnitudes[i]))
    with assert_raises():
        _ = vmap[isqrt]()(powers)
    _assert_native_integer_result(vmap[gcd]()(wide, powers[::-2]), vmap[gcd]()(powers[::-2], wide))
    _assert_native_integer_result(vmap[lcm]()(powers[::2], wide), vmap[lcm]()(wide, powers[::2]))
    _assert_native_integer_result(vmap[isqrt]()(magnitudes[::-2]), Batch[Integer]([isqrt(magnitudes[::-2][i]) for i in range(5)]))
    var bad = counts
    bad[1] = -1
    bad[8] = -2
    var message = String()
    try:
        _ = vmap[factorial]()(bad)
    except error:
        message = String(error)
    assert_true("mapped index 1" in message)
    with assert_raises():
        _ = vmap[pow_mod]()(odd, small, Batch[Integer]([Integer(0)]))
    with assert_raises():
        _ = vmap[div_exact]()(odd, 2)
    with assert_raises():
        _ = vmap[inverse_mod]()(small, 2)
    with assert_raises():
        _ = vmap[iroot]()(-odd, 2)
    for length in [0, 1, 8, 9]:
        var selected = powers[:length][::-1]
        var total = Integer(0)
        var product = Integer(1)
        for i in range(length):
            total += selected[i]
            product *= selected[i]
        assert_equal(_reduce[0](selected), total)
        assert_equal(_reduce[1](selected), product)
        if length:
            var low = selected[0]
            var high = selected[0]
            for i in range(length):
                low = selected[i] if selected[i] < low else low
                high = selected[i] if selected[i] > high else high
            assert_equal(_reduce[2](selected), low)
            assert_equal(_reduce[3](selected), high)
        assert_equal(sum(selected), sum_sequential(selected))
        assert_equal(sum(selected), sum_tree(selected))
        assert_equal(dot(selected, odd[:length]), dot_sequential(selected, odd[:length]))
        var scaled = axpy(wide, selected, odd[:length])
        for i in range(length):
            assert_equal(scaled[i], wide * selected[i] + odd[i])
        _assert_native_integer_result(axpy(0, selected, odd[:length]), Batch[Integer](odd[:length]))
    with assert_raises():
        _ = amin(powers[:0])
    with assert_raises():
        _ = amax(powers[:0])
    with assert_raises():
        _ = dot(powers, odd[:1])
    with assert_raises():
        _ = axpy(0, powers, odd[:1])
    var reduced = Integer(123)
    with assert_raises():
        reduced = _reduce[0](powers, fail_after_element=8)
    assert_equal(reduced, 123)
    with assert_raises():
        reduced = _dot(powers, odd, fail_after_element=8)
    assert_equal(reduced, 123)
    assert_equal(_reduce[1](powers, fail_after_element=0), 0)
    assert_equal(powers.to_json(), before)
    var selected = powers[::-2]
    var input = _rational_operand(selected)
    assert_true(Bool(input.native_integer) and not input.native)
    assert_equal(Int(input.native_integer.value()._owner.ptr()), Int(powers._owner.ptr()))
    for i in range(len(selected)):
        assert_equal(input.component(False, i), selected[i])
        assert_equal(input.component(True, i), 1)
        assert_equal(input.value(i), Rational(selected[i]))
        assert_equal(input.sign(i), selected[i].sign())
        assert_equal(input.is_zero(i), not selected[i])
    var rational = selected + Rational(1, 3)
    var floating = Float(2) * selected
    var complex = selected + Complex(1, 2)
    for i in range(len(selected)):
        assert_equal(rational[i], selected[i] + Rational(1, 3))
        assert_equal(floating[i], Float(2) * selected[i])
        assert_equal(complex[i], selected[i] + Complex(1, 2))
    assert_equal(dot(rational, selected), dot_sequential(rational, selected))
    assert_equal(dot(floating, selected), dot(floating, Batch[Integer](selected)))
    assert_equal(dot(complex, selected), dot(complex, Batch[Integer](selected)))
    var unit = Batch[Rational](List[Rational](length=9, fill=Rational(1)))
    unit **= small[::-1]
    assert_true((unit == 1).all())
    assert_equal(vmap[float.pow_int]()(Batch[Float](List[Float](length=9, fill=Float(2))), small)[8], Float(256))
    assert_equal(vmap[complex_pow_int]()(Batch[Complex](List[Complex](length=9, fill=Complex(0, 1))), small)[8], Complex(1))
    powers[:] = 7
    assert_equal(input.component(False, 0), selected[0])
    assert_equal(selected.to_json(), Batch[Integer].from_json(before)[::-2].to_json())


def test_native_rational_storage_transfers_and_compatibility() raises:
    var wide = Rational((Integer(1) << 1024) - 1, Integer(1) << 256)
    var source = Batch[Rational]([wide, Rational(-3, 7), Rational(), Rational(5, 2), wide])
    ref first = source._run()._read(0)
    assert_equal(Int(first._numerator._storage[Integer._Shared].ptr()), Int(wide._numerator._storage[Integer._Shared].ptr()))
    assert_equal(Int(first._denominator._storage[Integer._Shared].ptr()), Int(wide._denominator._storage[Integer._Shared].ptr()))
    var snapshot = source[::-1]
    var selected = snapshot[1::2]
    assert_equal(Int(selected._tensor_view()._owner.ptr()), Int(source._tensor_view()._owner.ptr()))
    var encoded = source.to_json()
    var copy = source
    var packed = Batch[Rational](source[:])
    assert_equal(packed.to_json(), encoded)
    assert_equal(Batch[Rational](packed[::-1]).to_json(), snapshot.to_json())
    var gathered = snapshot[Mask([False, True, False, True, False])]
    assert_equal(gathered.to_json(), selected.to_json())
    assert_equal(len(gathered._tensor_view()._owner[]), 2)
    var cursor = iter(snapshot)
    _ = next(cursor)
    assert_equal(Batch[Rational].from_iterable(cursor^).to_json(), snapshot[1:].to_json())
    var unsigned: List[UInt64] = [UInt64.MAX, 0]
    assert_equal(Batch[Rational].from_iterable(unsigned)[0], Rational(Integer(UInt64.MAX)))
    assert_equal(Batch[Rational].from_iterable(range(3))[2], Rational(2))
    var mapped = vmap[_tensor_family_product[Rational]]()(source, source)
    assert_equal(mapped.to_json(), (packed * packed).to_json())
    for count in [0, 1, 8, 9]:
        var values = Batch[Rational].from_native([i for i in range(count)])
        var restored = Batch[Rational].from_json(values.to_json())
        assert_equal(restored.to_json(), values.to_json())
    for fail in [0, 2, 4]:
        with assert_raises():
            source._assign_slice(StridedSlice(None, None, -1), source[:], fail_after_element=fail)
        with assert_raises():
            _ = _native_rational_iterator(iter(source), fail)
        assert_equal(source.to_json(), encoded)
        assert_equal(Int(source._tensor_view()._owner.ptr()), Int(copy._tensor_view()._owner.ptr()))
    var mask = Mask([False, True, False, True, False])
    var before_empty = source._tensor_view()._owner
    source[Mask([False, False, False, False, False])] = Rational(9)
    assert_equal(Int(source._tensor_view()._owner.ptr()), Int(before_empty.ptr()))
    with assert_raises():
        source[Mask([False])] = Rational(9)
    with assert_raises():
        source[mask] = Batch[Rational]([Rational(1)])[:]
    for fail in [0, 1]:
        with assert_raises():
            source._assign_mask(mask, Batch[Rational]([wide]), True, fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    source[1:] = source[:-1]
    assert_equal(source[1], wide)
    assert_equal(source[2], Rational(-3, 7))
    source[mask] = selected
    assert_equal(source[1], Rational(5, 2))
    assert_equal(source[3], Rational(-3, 7))
    source[0] = Rational(18446744073709551616)
    assert_equal(source[0], Rational(Integer(1) << 64))
    var before = source.to_json()
    with assert_raises():
        source[:] = Batch[Rational]([Rational(1)])[:]
    with assert_raises():
        source._update_rational(Rational(1), 0, fail=2)
    assert_equal(source.to_json(), before)
    source += Rational(1, 3)
    assert_equal(source[1], Rational(17, 6))
    assert_equal(copy.to_json(), encoded)
    assert_equal(snapshot.to_json(), packed[::-1].to_json())
    source[:] = snapshot
    assert_equal(source.to_json(), snapshot.to_json())
    var budget = _ConversionBudget(ConversionLimits(max_allocated_bytes=Int.MAX))
    _ = _parse_rational_batch_json(encoded, budget)
    var limit = budget.allocated_used
    assert_equal(Batch[Rational].from_json(encoded, limits=ConversionLimits(max_allocated_bytes=limit)).to_json(), encoded)
    with assert_raises():
        source = Batch[Rational].from_json(encoded, limits=ConversionLimits(max_allocated_bytes=limit - 1))
    assert_equal(source.to_json(), snapshot.to_json())
    assert_equal(sum(copy), sum(packed))
    assert_equal(dot(copy, snapshot), dot(packed, snapshot))
    assert_equal((Float(2) * copy).to_json(), (Float(2) * packed).to_json())
    assert_equal((copy + Complex(1, 2)).to_json(), (packed + Complex(1, 2)).to_json())


def test_native_complex_storage_transfers_and_compatibility() raises:
    var real_format = FloatFormat(1024)
    var imag_format = FloatFormat(256)
    var wide = Complex(
        _real=Float((Integer(1) << 1024) - 17, context=ArithmeticContext(format=real_format)),
        _imag=Float((Integer(1) << 256) - 3, context=ArithmeticContext(format=imag_format)),
    )
    var source = Batch[Complex]([
        wide, Complex(_real=Float.zero(negative=True), _imag=Float.zero()),
        Complex(_real=Float.nan(), _imag=Float.infinity(negative=True)),
        Complex(3, 4), wide,
    ])
    source._real_format = real_format
    source._imag_format = imag_format
    ref first = source._run()._read(0)
    assert_equal(Int(first._real._significand._storage[Integer._Shared].ptr()), Int(wide._real._significand._storage[Integer._Shared].ptr()))
    assert_equal(Int(first._imag._significand._storage[Integer._Shared].ptr()), Int(wide._imag._significand._storage[Integer._Shared].ptr()))
    var encoded = source.to_json()
    var snapshot = source[::-1]
    var selected = snapshot[1::2]
    assert_equal(Int(selected._tensor_view()._owner.ptr()), Int(source._tensor_view()._owner.ptr()))
    var packed = Batch[Complex](source[:])
    assert_equal(packed.to_json(), encoded)
    assert_equal(Batch[Complex](packed[::-1]).to_json(), snapshot.to_json())
    var mask = Mask([False, True, False, True, False])
    var gathered = snapshot[mask]
    assert_equal(len(gathered._tensor_view()._owner[]), 2)
    assert_equal(gathered.to_json(), selected.to_json())
    var cursor = iter(snapshot)
    _ = next(cursor)
    assert_equal(Batch[Complex].from_iterable(cursor).to_json(), snapshot[1:].to_json())
    assert_equal(Batch[Complex].from_iterable(cursor^).to_json(), snapshot[1:].to_json())
    assert_equal(Batch[Complex].from_native([UInt64.MAX])[0], Complex(UInt64.MAX))
    assert_equal(Batch[Complex].from_native([Float64(1.25)])[0], Complex(Float64(1.25)))
    assert_equal(Batch[Complex](Batch[Rational]([Rational(1, 3)]))[0], Complex(Float(Rational(1, 3))))
    assert_equal(Batch[Complex].from_iterable(range(3))[2], Complex(2))
    var numbers: List[Integer] = [Integer(1) << 256]
    assert_equal(Batch[Complex](numbers^)[0], Complex(Float(Integer(1) << 256)))
    assert_equal(Batch[Complex]([wide], shape=[])[:].to_json(), Batch[Complex]([wide], shape=[]).reshape([1]).to_json())
    var mapped = vmap[_tensor_family_product[Complex]]()(selected, selected)
    assert_equal(mapped.to_list(), (selected * selected).to_list())
    for empty in [False, True]:
        var input = source[0:0] if empty else source[:]
        var json = input.to_json()
        var budget = _ConversionBudget(ConversionLimits(max_allocated_bytes=Int.MAX))
        var values, formats = _parse_complex_batch_json(json, budget)
        assert_equal(values._layout.size, len(input))
        assert_equal(formats.real(), real_format)
        assert_equal(formats.imag(), imag_format)
        var restored = Batch[Complex].from_json(json, limits=ConversionLimits(max_allocated_bytes=budget.allocated_used))
        assert_equal(restored.to_json(), json)
        with assert_raises():
            restored = Batch[Complex].from_json(json, limits=ConversionLimits(max_allocated_bytes=budget.allocated_used - 1))
        assert_equal(restored.to_json(), json)
        assert_equal(Batch[Complex](input).to_json(), json)
    for fail in [0, 2, 4]:
        with assert_raises():
            source._assign_slice(StridedSlice(None, None, -1), source[:], fail_after_element=fail)
        with assert_raises():
            _ = _native_complex_iterator(iter(source), fail)
        with assert_raises():
            _ = _native_complex_iterator(iter(source.to_list()), fail)
        assert_equal(source.to_json(), encoded)
    for fail in [0, 1]:
        with assert_raises():
            source._assign_mask(mask, Batch[Complex]([wide]), True, fail_after_element=fail)
        assert_equal(source.to_json(), encoded)
    var before_empty = source._tensor_view()._owner
    source[Mask([False, False, False, False, False])] = Complex(9)
    assert_equal(Int(source._tensor_view()._owner.ptr()), Int(before_empty.ptr()))
    with assert_raises():
        source[Mask([False])] = Complex(9)
    with assert_raises():
        source[mask] = Batch[Complex]([Complex(1)])[:]
    source[1:] = source[:-1]
    assert_equal(source[1].to_json(), wide.to_json())
    source[mask] = selected
    assert_equal(source[1], Complex(3, 4))
    source[::-1] = snapshot
    assert_equal(source.to_json(), encoded)
    source[-1] = wide
    assert_equal(source[-1]._imag.format(), imag_format)
    assert_equal(source[-1]._real.format(), real_format)
    assert_equal(snapshot.to_json(), packed[::-1].to_json())
    with assert_raises():
        source = Batch[Complex].from_json(encoded, limits=ConversionLimits(max_values=4))
    with assert_raises():
        source = Batch[Complex].from_json(encoded + " trailing")
    assert_equal(source.to_json(), encoded)
    var finite = Batch[Complex](selected)
    var finite_packed = Batch[Complex](finite[:])
    assert_equal(sum(finite), sum(finite_packed))
    assert_equal(dot(finite, finite[::-1]), dot(finite_packed, finite_packed[::-1]))
    assert_equal(vdot(finite, finite), vdot(finite_packed, finite_packed))
    var saved = finite[:]
    var before = finite.to_json()
    with assert_raises():
        finite._update_complex(Complex(1, 2), 0, fail=1)
    assert_equal(finite.to_json(), before)
    finite += Complex(1, 2)
    assert_equal(saved.to_json(), before)
    finite[:] = saved
    assert_equal(finite.to_json(), before)


def _assert_native_complex(value: Batch[Complex]) raises:
    # Every result is an owned, row-major native record.
    assert_equal(value._layout[].size, len(value))


def test_native_complex_execution_and_transactionality() raises:
    var numbers = List[Complex]()
    for i in range(9):
        var p = 1024 if i == 8 else 256 if i == 7 else 17 if i % 2 else 5
        var real = Float(Rational(i + 1, 3), context=ArithmeticContext(format=FloatFormat(p)))
        if i == 8:
            real = Float((Integer(1) << 1024) - 17, context=ArithmeticContext(format=FloatFormat(p)))
        numbers.append(Complex(
            _real=real, _imag=Float(Rational(2 - i, 7), context=ArithmeticContext(format=FloatFormat(p + 2))),
        ))
    var a = Batch[Complex](numbers)
    var b = Batch[Complex]([Complex(i + 2, 1) for i in range(9)])
    var before = a.to_json()
    for count in [0, 1, 8, 9]:
        var left = a[:count][::-1]
        var right = b[:count]
        for operation in range(4):
            var result = _complex_result(left, right, operation)
            _assert_native_complex(result)
            assert_equal(len(result), count)
            for i in range(count):
                assert_equal(result[i].to_json(), Complex._calculate(left[i], right[i], operation).to_json())
            var exact = _complex_result(left, Rational(2, 7), operation)
            var reverse = _complex_result(Integer(18446744073709551616), left, operation)
            _assert_native_complex(exact)
            _assert_native_complex(reverse)
            for i in range(count):
                assert_equal(exact[i].to_json(), Complex._calculate(left[i], Rational(2, 7), operation).to_json())
                assert_equal(reverse[i].to_json(), Complex._calculate(Integer(18446744073709551616), left[i], operation).to_json())
        _assert_native_complex(-left)
        _assert_native_complex(+left)
        _assert_native_complex(left.conjugate())
        for result in [left.real(), left.imag()]:
            assert_equal(len(result), len(left))
        assert_true((left == left).all())
        assert_true(not (left != left).any())
    var selected = a[::-1][Mask([False, True, False, True, False, True, False, True, False])]
    var packed = Batch[Complex](selected[:])
    for operation in range(4):
        var native = _complex_result(selected, Float64(1.25), operation)
        assert_equal(native.to_json(), _complex_result(packed, Float64(1.25), operation).to_json())
        _assert_native_complex(native)
    var specials = Batch[Complex]([
        Complex(3, 4), Complex(_real=Float(-4), _imag=Float.zero(negative=True)),
        Complex(_real=Float.zero(negative=True), _imag=Float.zero()),
        Complex(_real=Float.infinity(), _imag=Float(1)),
        Complex(_real=Float.nan(), _imag=Float(2)),
    ])
    var counts = Batch[Integer]([2, -1, 0, 3, 1])
    for rounding in [RoundingMode.nearest_even, RoundingMode.toward_zero,
                     RoundingMode.toward_positive, RoundingMode.toward_negative,
                     RoundingMode.away_from_zero]:
        var context = ArithmeticContext(format=FloatFormat(9, emin=-8, emax=8), rounding=rounding)
        var paired = ComplexContext(real=context, imag=ArithmeticContext(format=FloatFormat(13, emin=-8, emax=8), rounding=rounding))
        for operation in range(4):
            var result = vmap[complex.add]()(specials, Complex(3, 2), context=paired) if operation == 0 else (
                vmap[complex.subtract]()(specials, Complex(3, 2), context=paired) if operation == 1 else (
                    vmap[complex.multiply]()(specials, Complex(3, 2), context=paired) if operation == 2
                    else vmap[complex.divide]()(specials, Complex(3, 2), context=paired)
                )
            )
            for i in range(len(specials)):
                assert_equal(result[i].to_json(), Complex._calculate(specials[i], Complex(3, 2), operation, paired).to_json())
        var roots = vmap[complex.sqrt]()(specials, context=paired)
        var powers = vmap[complex.pow_int]()(specials, counts, context=paired)
        var squares = vmap[norm_sqr]()(specials, context=context)
        var magnitudes = vmap[complex_abs]()(specials, context=context)
        _assert_native_complex(roots)
        _assert_native_complex(powers)
        for i in range(len(specials)):
            assert_equal(roots[i].to_json(), sqrt(specials[i], context=paired).to_json())
            assert_equal(powers[i].to_json(), pow_int(specials[i], counts[i], context=paired).to_json())
            assert_equal(squares[i].to_json(), norm_sqr(specials[i], context=context).to_json())
            assert_equal(magnitudes[i].to_json(), abs(specials[i], context=context).to_json())
    var output = specials.conjugate()
    for i in range(len(specials)):
        assert_equal(output[i].to_json(), specials[i].conjugate().to_json())
        assert_equal(specials.is_zero()[i], specials[i].is_zero())
        assert_equal(specials.is_finite()[i], specials[i].is_finite())
        assert_equal(specials.is_infinite()[i], specials[i].is_infinite())
        assert_equal(specials.is_nan()[i], specials[i].is_nan())
    var saved_output = output.to_json()
    for component in range(2):
        for fail in [0, 2, 4]:
            with assert_raises():
                output = _complex_result(specials, 3, 2, _ComplexContextArgument(), fail, component)
            with assert_raises():
                output = _complex_power(specials, Integer(0), _ComplexContextArgument(), fail, component)
            with assert_raises():
                output = _complex_unary(specials, 2, fail, component)
            assert_equal(output.to_json(), saved_output)
    var trapped = False
    try:
        output = vmap[complex.divide]()(a[:3], Batch[Complex]([Complex(1), Complex(0), Complex(0)]),
                                      context=ArithmeticContext(trap_divide_by_zero=True))
    except error:
        trapped = True
        assert_true(String(error).__contains__("mapped index 1"))
    assert_true(trapped)
    assert_equal(output.to_json(), saved_output)
    for operation in range(5):
        var target = a
        var snapshot = target[::-1]
        var exponents = Batch[Integer]([i % 3 - 1 for i in range(9)])
        for component in range(2):
            for fail in [0, 4, 8]:
                with assert_raises():
                    if operation == 4:
                        target._update_complex_power(exponents, fail, component)
                    else:
                        target._update_complex(snapshot, operation, fail, component)
                assert_equal(target.to_json(), before)
                assert_equal(Int(target._tensor_view()._owner.ptr()), Int(a._tensor_view()._owner.ptr()))
        if operation == 4:
            target **= exponents
        else:
            target._update_complex(snapshot, operation)
        _assert_native_complex(target)
        for i in range(9):
            var context = ComplexContext(real=ArithmeticContext(format=a[i].real_format()), imag=ArithmeticContext(format=a[i].imag_format()))
            var expected: Complex
            if operation == 4:
                expected = pow_int(a[i], exponents[i], context=context)
            else:
                expected = Complex._calculate(a[i], snapshot[i], operation, context)
            assert_equal(target[i].to_json(), expected.to_json())
        assert_equal(snapshot.to_json(), a[::-1].to_json())
    var huge = (Integer(1) << 256) + 1
    var identity = vmap[complex.pow_int]()(Batch[Complex]([Complex(-1), Complex(0, 1)]), huge)
    assert_equal(identity[0], Complex(-1))
    assert_equal(identity[1], Complex(0, 1))
    var empty = a[:0]
    _assert_native_complex(vmap[complex.pow_int]()(empty, huge))
    var empty_owner = Batch[Complex](empty)
    var owner = empty_owner._tensor_view()._owner
    empty_owner += Complex(1)
    empty_owner **= huge
    assert_equal(Int(empty_owner._tensor_view()._owner.ptr()), Int(owner.ptr()))
    empty_owner._real_format = FloatFormat(23)
    empty_owner._imag_format = FloatFormat(37)
    assert_equal((empty_owner + 1)._complex_input().real.default_format(), FloatFormat(23))
    assert_equal((empty_owner ** huge)._complex_input().imag.default_format(), FloatFormat(37))
    assert_equal(empty_owner.real()._float_input().default_format(), FloatFormat(23))
    with assert_raises():
        _ = a + Batch[Complex]([Complex(1)])
    with assert_raises():
        _ = _complex_comparison(a, b[:1], 2)
    var incompatible = b
    incompatible[8] = Complex(1, context=ArithmeticContext(format=FloatFormat(17, emin=-100, emax=100)))
    var failed = False
    try:
        _ = _complex_result(a, incompatible, 0, fail=0)
    except error:
        failed = True
        assert_true(String(error).__contains__("element 8"))
        assert_true(String(error).__contains__("FormatMismatch"))
    assert_true(failed)
    assert_equal(a.to_json(), before)


def _assert_native_rational_result(actual: Batch[Rational], expected: List[Rational]) raises:
    assert_equal(actual.to_list(), expected)


def test_native_rational_execution_and_transactionality() raises:
    var wide = Rational((Integer(1) << 1024) - 1, Integer(1) << 256)
    var a = Batch[Rational]([wide, Rational(-3, 7), 0, Rational(5, 2), -wide, 1, -1, Rational(4, 3), Rational(-7, 2)])
    var b = Batch[Integer].from_native([i + 1 for i in range(9)])
    var before = a.to_json()
    for count in [0, 1, 8, 9]:
        var left = a[:count][::-1]
        var right = b[:count]
        var results = [left + right, left - right, left * right, left / right]
        for op in range(4):
            var expected = List[Rational]()
            for i in range(count):
                var x = left[i]
                var y = Rational(right[i])
                expected.append(x + y if op == 0 else x - y if op == 1 else x * y if op == 2 else x / y)
            _assert_native_rational_result(results[op], expected)
            var reverse = _rational_result(Integer(3), right, op)
            var broadcast = _rational_result(left, Rational(2, 3), op)
            for i in range(count):
                var y = Rational(right[i])
                assert_equal(reverse[i], 3 + y if op == 0 else 3 - y if op == 1 else 3 * y if op == 2 else 3 / y)
                var x = left[i]
                var scale = Rational(2, 3)
                assert_equal(broadcast[i], x + scale if op == 0 else x - scale if op == 1 else x * scale if op == 2 else x / scale)
        var masks = [left == right, left != right, left < right, left <= right, left > right, left >= right]
        for i in range(count):
            var x = left[i]
            var y = Rational(right[i])
            var expected = [x == y, x != y, x < y, x <= y, x > y, x >= y]
            for op in range(6):
                assert_equal(masks[op][i], expected[op])
        for mask in masks:
            assert_equal(len(mask), count)
        _assert_native_rational_result(-left, [-left[i] for i in range(count)])
        _assert_native_rational_result(vmap[rational.abs]()(left), [abs(left[i]) for i in range(count)])
        var queries = [left.sign(), left.floor(), left.ceil(), left.trunc()]
        for result in queries:
            assert_equal(len(result), count)
        for i in range(count):
            assert_equal(queries[0][i], left[i].sign())
            assert_equal(queries[1][i], left[i].floor())
            assert_equal(queries[2][i], left[i].ceil())
            assert_equal(queries[3][i], left[i].trunc())
    var selected = a[::-2]
    var packed = Batch[Rational](selected[:])
    _assert_native_rational_result(packed + selected, [selected[i] * 2 for i in range(len(selected))])
    var masked = a[a != 0]
    _assert_native_rational_result(masked / masked, List[Rational](length=len(masked), fill=Rational(1)))
    var exponents = Batch[Integer].from_native([-1, 0, 0, 1, 2, -2, -3, 0, 1])
    _assert_native_rational_result(a ** exponents, [a[i] ** exponents[i] for i in range(9)])
    _assert_native_rational_result(vmap[pow_rational]()(Rational(2, 3), exponents[::-1]), [Rational(2, 3) ** exponents[8 - i] for i in range(9)])
    _assert_native_rational_result(vmap[pow_rational]()(b[::2], Integer(-1)), [Rational(1, b[i]) for i in range(0, 9, 2)])
    var huge = Integer(1) << 256
    _assert_native_rational_result(Batch[Rational]([0, 1, -1]) ** (huge + 1), [0, 1, -1])
    _assert_native_rational_result(Batch[Rational]([1, -1]) ** -(huge + 1), [1, -1])
    for op in [0, 1, 2, 3, 5]:
        var expected = _rational_power(a, exponents) if op == 5 else _rational_result(a, b, op)
        for fail in [0, 4, 8]:
            var destination = a
            var messages = List[String]()
            for fresh in [True, False]:
                try:
                    if fresh:
                        if op == 5:
                            _ = _rational_power(a, exponents, fail)
                        else:
                            _ = _rational_result(a, b, op, fail)
                    elif op == 5:
                        destination._update_rational_power(exponents, fail)
                    else:
                        destination._update_rational(b, op, fail)
                except error:
                    messages.append(String(error))
            assert_equal(len(messages), 2)
            assert_equal(messages[0], messages[1])
            assert_true(String("failure at element ", fail) in messages[0])
            assert_equal(Int(destination._tensor_view()._owner.ptr()), Int(a._tensor_view()._owner.ptr()))
            assert_equal(destination.to_json(), before)
            if op == 5:
                destination **= exponents
            elif op == 0:
                destination += b
            elif op == 1:
                destination -= b
            elif op == 2:
                destination *= b
            else:
                destination /= b
            _assert_native_rational_result(destination, expected.to_list())
    for op in range(6):
        var message = String()
        try:
            if op < 2:
                _ = _rational_unary[False](selected, op, 2)
            else:
                _ = _rational_unary[True](selected, op, 2)
        except error:
            message = String(error)
        assert_true("unary/power failure at element 2" in message)
    var overlapping = a
    var saved = overlapping[::-1]
    overlapping += saved
    _assert_native_rational_result(overlapping, [a[i] + saved[i] for i in range(9)])
    var native = a
    native += 18446744073709551616
    native *= UInt64.MAX
    native /= Integer(UInt64.MAX)
    native -= Integer(1) << 64
    _assert_native_rational_result(native, a.to_list())
    var bad = b
    bad[2] = 0
    bad[8] = 0
    for shape_error in [True, False]:
        var destination = a
        var message = String()
        try:
            destination._update_rational(bad[2:3] if shape_error else bad[:], 3, fail=0)
        except error:
            message = String(error)
        assert_true(("batches of lengths 9 and 1" if shape_error else "divisor is 0") in message)
        if not shape_error:
            assert_true("element 2" in message)
        assert_equal(destination.to_json(), before)
    var invalid_power = exponents
    invalid_power[0] = huge
    invalid_power[2] = -1
    var message = String()
    try:
        a._update_rational_power(invalid_power, fail=0)
    except error:
        message = String(error)
    assert_true("negative power at element 2" in message)
    with assert_raises():
        a += Batch[Rational]([Rational(1)])
    with assert_raises():
        a **= Batch[Integer]([Integer(1)])
    assert_equal(a.to_json(), before)
    assert_equal(saved.to_json(), a[::-1].to_json())
    var empty = Batch[Rational].from_native(List[Int]())
    var owner = empty._tensor_view()._owner
    empty /= 0
    empty **= -1
    assert_equal(Int(empty._tensor_view()._owner.ptr()), Int(owner.ptr()))



def _sum_case_magnitude(mut seed: UInt64, bits: Int) -> Integer:
    """A nonzero magnitude of exactly `bits` bits from a 64-bit LCG."""
    var words = List[UInt32]()
    for _ in range((bits + 31) // 32):
        seed = seed * 6364136223846793005 + 1442695040888963407
        words.append(UInt32(seed >> 32))
    var top = (bits - 1) % 32
    var last = len(words) - 1
    words[last] = (words[last] & UInt32((UInt64(1) << UInt64(top + 1)) - 1)) | (UInt32(1) << UInt32(top))
    return Integer._from_words(words^, False)


def _assert_same_rounding(fast: _RoundedBinary, slow: _RoundedBinary) raises:
    assert_equal(fast.kind, slow.kind)
    assert_equal(fast.negative, slow.negative)
    assert_equal(fast.significand, slow.significand)
    assert_equal(fast.exponent, slow.exponent)
    assert_equal(fast.status._flags, slow.status._flags)
    assert_equal(fast.status._direction, slow.status._direction)


def test_binary_float_sum_matches_general_path() raises:
    # Binary operands take the one-allocation sum; the same values written as
    # 3n/3 take the general exact path. Both must round identically.
    var seed = UInt64(0x9E3779B97F4A7C15)
    var precisions: List[Int] = [2, 3, 24, 53, 64, 113, 128, 200, 1024]
    for _ in range(1500):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var draw = Int(seed >> 33)
        var p = precisions[draw % len(precisions)]
        var mode = RoundingMode(_code=(draw // 16) % 5)
        var narrow = (draw // 128) % 4 == 0
        var format = FloatFormat(p, emin=-40, emax=40) if narrow else FloatFormat(p)
        var context = ArithmeticContext(format=format, rounding=mode)
        var a_bits = 1 + (draw // 1024) % (p + 40)
        var b_bits = 1 + (draw // 4096) % (p + 40)
        var a = _sum_case_magnitude(seed, a_bits)
        var b = _sum_case_magnitude(seed, b_bits)
        var shape = (draw // 65536) % 8
        if shape == 0:
            b = a
        elif shape == 1:
            b = a + 1
        elif shape == 2 and a > 1:
            b = a - 1
        var a_scale = Int128((draw // 7) % 120) - 60
        var gap = Int128((draw // 13) % (p + 12))
        if shape == 3:
            gap = Int128(p * 3 + b_bits + 50)
        var b_scale = a_scale - gap if (draw // 3) % 2 else a_scale + gap
        var a_negative = (draw // 5) % 2 == 1
        var b_negative = (draw // 11) % 2 == 1 if shape > 2 else not a_negative
        var fast = _float_sum(
            _FloatInput(1, a_negative, a, Integer(1), a_scale),
            _FloatInput(1, b_negative, b, Integer(1), b_scale), context, False,
        )
        var slow = _float_sum(
            _FloatInput(1, a_negative, a * 3, Integer(3), a_scale),
            _FloatInput(1, b_negative, b * 3, Integer(3), b_scale), context, False,
        )
        _assert_same_rounding(fast, slow)



def _significand_count(value: Float) -> UInt64:
    """Owners of value's heap significand, or 0 when it is inline."""
    if not value._significand._storage.isa[Integer._Shared]():
        return 0
    return value._significand._storage[Integer._Shared].count()


def _copying_add(a: _FloatArgument, b: _FloatArgument) raises -> Float:
    # Copies of a lent argument count normally and are released normally.
    var kept = a
    var again = kept
    return Float._from_arguments(again, b, 0)


def _fail_late(a: _FloatArgument, b: _FloatArgument) raises -> Float:
    # Elements (i + 1) << 100 at 128 bits reach scale -22 from index 31 on.
    if a.value.scale >= -22:
        raise Error("late")
    return Float._from_arguments(a, b, 0)


def test_borrowed_sums_match_the_argument_path() raises:
    var narrow = ArithmeticContext(format=FloatFormat(53, emin=-30, emax=30))
    var wide = ArithmeticContext(format=FloatFormat(160))
    var values = List[Float]()
    for text in ["0", "-0", "1", "-1", "3.25", "-1e9", "7e-8", "123456789.125"]:
        values.append(Float(text, context=narrow))
        values.append(Float(text, context=wide))
    var inf = Float("inf", context=wide)
    values.append(inf)
    values.append(-inf)
    values.append(Float("nan", context=narrow))
    values.append(Float((Integer(1) << 200) + 3, context=wide))
    values.append(Float(Integer(1) << 29, context=narrow))
    for i in range(len(values)):
        for j in range(len(values)):
            ref a = values[i]
            ref b = values[j]
            if a.format().emin() != b.format().emin():
                with assert_raises():
                    _ = a + b
                continue
            for negate in range(2):
                var borrowed = a - b if negate else a + b
                var general = Float._from_arguments(a, b, negate)
                assert_equal(borrowed.to_json(), general.to_json())
                var updated = a
                if negate:
                    updated -= b
                else:
                    updated += b
                assert_equal(updated.to_json(), Float._from_arguments(a, b, negate, ArithmeticContext(format=a.format())).to_json())
    var twice = values[3]
    twice += twice
    assert_equal(twice, values[3] + values[3])
    var z = Complex(values[5], values[13])
    var w = Complex(values[7], values[9])
    for negate in range(2):
        var borrowed = z - w if negate else z + w
        assert_equal(borrowed.to_json(), Complex._calculate(z, w, negate).to_json())
    var mismatched = Complex(Float(1, context=narrow), Float(2, context=narrow))
    with assert_raises(contains="FormatMismatch"):
        _ = mismatched + z
    var both = z
    both += both
    assert_equal(both, z + z)


def test_vmap_lends_batch_elements_without_owning_them() raises:
    var context = ArithmeticContext(format=FloatFormat(128))
    var values = List[Float]()
    for i in range(64):
        values.append(Float((Integer(i + 1) << 100) + 1, context=context))
    var a = Batch[Float](values)
    _ = values^
    assert_equal(_significand_count(a[0]), 2)
    var copied = vmap[_copying_add]()(a, a)
    var added = vmap[float.add]()(a, a)
    for i in range(len(a)):
        assert_equal(copied[i], a[i] + a[i])
        assert_equal(added[i], a[i] + a[i])
    with assert_raises(contains="mapped index 31"):
        _ = vmap[_fail_late]()(a, a)
    # Every lent handle was returned: only the batch and this read own each one.
    for i in range(len(a)):
        assert_equal(_significand_count(a[i]), 2)
    var z = Batch[Complex]([Complex(a[i], a[63 - i]) for i in range(64)])
    var sums = vmap[complex.add]()(z, z)
    for i in range(64):
        assert_equal(sums[i], z[i] + z[i])
        assert_equal(_significand_count(z[i].real()), 3)



def _wide_sqrt(x: Float) raises -> Float:
    return float.sqrt(x)


def _fail_marked(x: Float) raises -> Float:
    # Elements 70 and 90 fail; the lowest index must win, as in a loop.
    if x == Float(70) or x == Float(90):
        raise Error(String("marked ", x))
    return float.sqrt(x)


def _nested_sum(x: Float, y: Float) raises -> Float:
    # A batch operation inside a mapped function finds the pool busy and runs inline.
    var pair = Batch[Float]([x, y, x, y])
    return float.add((pair + pair)[1], (vmap[float.add]()(pair, pair))[0])


def test_parallel_batches_match_sequential_results() raises:
    var wide = ArithmeticContext(format=FloatFormat(128))
    var floats = List[Float]()
    var others = List[Float]()
    var rationals = List[Rational]()
    var divisors = List[Rational]()
    var integers = List[Integer]()
    var smaller = List[Integer]()
    for i in range(100):
        floats.append(Float((Integer(i * 7919 + 13) << 100) + (Integer(1) << 127), context=wide))
        others.append(Float(Integer(i * 104729 + 5) << 80, context=wide))
        rationals.append(Rational(Integer(i * 7919 + 1) << 70, Integer(i * 31 + 7)))
        divisors.append(Rational(Integer(i + 3), (Integer(i) << 65) + 1))
        integers.append((Integer(i * 7919 + 3) << 500) + i)
        smaller.append((Integer(i + 5) << 200) + 1)
    var a = Batch[Float](floats)
    var b = Batch[Float](others)
    var p = Batch[Rational](rationals)
    var q = Batch[Rational](divisors)
    var m = Batch[Integer](integers)
    var d = Batch[Integer](smaller)
    # vmap[f]() is a mapped function: store it, copy it and call it again.
    var root_of = vmap[float.sqrt]()
    var again = root_of.copy()
    var roots = root_of(a)
    assert_equal(again(a)[7], roots[7])
    var mapped_roots = vmap[_wide_sqrt]()(a)
    var sums = vmap[float.add]()(a, b, context=wide)
    var floor = m // d
    var ratios = p / q
    var differences = a - b
    var z = Batch[Complex]([Complex(floats[i], others[i]) for i in range(100)])
    var doubled = z + z
    var nested = vmap[_nested_sum]()(a, b)
    for i in range(100):
        assert_equal(roots[i], float.sqrt(floats[i]))
        assert_equal(mapped_roots[i], roots[i])
        assert_equal(sums[i], float.add(floats[i], others[i], context=wide))
        assert_equal(floor[i], integers[i] // smaller[i])
        assert_equal(ratios[i], rationals[i] / divisors[i])
        assert_equal(differences[i], floats[i] - others[i])
        assert_equal(doubled[i], Complex(floats[i] + floats[i], others[i] + others[i]))
        assert_equal(nested[i], float.add(others[i] + others[i], floats[i] + floats[i]))
    var marked = Batch[Float]([Float(i) for i in range(100)])
    with assert_raises(contains="mapped index 70: marked"):
        _ = vmap[_fail_marked]()(marked)
    var zeros = List[Rational]()
    for i in range(100):
        zeros.append(Rational(0) if i == 70 or i == 90 else divisors[i])
    with assert_raises():
        _ = p / Batch[Rational](zeros)
    with assert_raises(contains="Mapped axes have different lengths"):
        _ = vmap[float.add]()(a, a[:5])
    # The batches are intact and reusable after failures.
    assert_equal((p / q)[99], rationals[99] / divisors[99])



comptime _EvaluatedParts = Tuple[Integer, Rational, Float, Complex, Batch[Integer], Batch[Integer], Bool, Batch[Rational]]


def _evaluate_parts(parts: _EvaluatedParts) raises -> Integer:
    """Shared leaves arrive intact; mapped leaves line up by position."""
    var wide = (Integer(1) << 1024) + 7
    var context = ArithmeticContext(format=FloatFormat(1024))
    assert_equal(parts[2].to_json(), Float(wide, context=context).to_json())
    assert_equal(parts[3].to_json(), Complex(wide, -wide, context=context).to_json())
    var position = Int(parts[4][0]) - 1
    assert_equal(parts[1], parts[7][3 - position])
    if parts[6]:
        _ = _tensor_failure(parts[0])
    return parts[0] + parts[4][1] + parts[5][0]



def _rebuilt_inside(x: Rational, y: Rational) raises -> Rational:
    # A large parallel result made and dropped inside a mapped function
    # releases on this thread: the pool is busy with the outer call.
    var inner = Batch[Rational]([x + Rational(i, 7) for i in range(1100)])
    var total = vmap[_rational_square]()(inner)
    return total[1099] + y


def _rational_square(x: Rational) raises -> Rational:
    return x * x


def _squares_kept(source: Batch[Rational]) raises -> Tuple[Rational, Batch[Rational]]:
    var squares = vmap[_rational_square]()(source)
    # A long enough run records who computed what.
    assert_true(Bool(squares._owner[].ownership))
    return (squares[1234], squares)


def test_parallel_results_release_by_their_owners() raises:
    # Long enough to run in parallel on a fast machine too: the executor splits
    # a map estimated at 100 us or more, and 1500 squares of 80-bit numerators
    # fell just short of that on Apple silicon.
    var values = List[Rational]()
    for i in range(1500):
        values.append(Rational(Integer(i + 3) << 700, Integer(i * 31 + 7)))
    var source = Batch[Rational](values)
    var held = _squares_kept(source)
    var kept = held[0]
    var copy = held[1]
    _ = held^
    # The run outlived its batch through the copy; the element outlives both.
    assert_equal(copy[7], values[7] * values[7])
    _ = copy^
    assert_equal(kept, values[1234] * values[1234])
    var sums = source + source
    assert_equal(sums[1499], values[1499] + values[1499])
    var nested = vmap[_rebuilt_inside]()(source[:70], Rational(1))
    assert_equal(nested[3], (values[3] + Rational(1099, 7)) * (values[3] + Rational(1099, 7)) + Rational(1))



def test_batch_with_shared_scalars_matches_scalar_arithmetic() raises:
    var narrow = ArithmeticContext(format=FloatFormat(53))
    var wide = ArithmeticContext(format=FloatFormat(160))
    var values = List[Float]()
    for text in ["0", "-0", "1.5", "-7", "1e30", "3e-20"]:
        values.append(Float(text, context=wide))
    values.append(Float("inf", context=wide))
    values.append(Float("nan", context=wide))
    var batch = Batch[Float](values)
    var scalars: List[Float] = [
        Float("2.25", context=narrow), Float("-0", context=wide), Float("inf", context=wide),
        Float("nan", context=narrow), Float((Integer(1) << 140) + 1, context=wide),
    ]
    for s in scalars:
        var forward = batch + s
        var reverse = s - batch
        for i in range(len(values)):
            assert_equal(forward[i].to_json(), (values[i] + s).to_json())
            assert_equal(reverse[i].to_json(), (s - values[i]).to_json())
    var parts = Batch[Complex]([Complex(values[i], values[(i + 3) % 6]) for i in range(6)])
    var z = Complex(Float("1.25", context=wide), Float("-2", context=wide))
    var right = parts + z
    var left = z - parts
    for i in range(6):
        assert_equal(right[i].to_json(), (parts[i] + z).to_json())
        assert_equal(left[i].to_json(), (z - parts[i]).to_json())
    # Exponent bounds still have to agree.
    var bounded = Complex(Float(1, context=ArithmeticContext(format=FloatFormat(53, emin=-9, emax=9))), Float(1, context=ArithmeticContext(format=FloatFormat(53, emin=-9, emax=9))))
    with assert_raises(contains="FormatMismatch"):
        _ = parts + bounded



def _complex_operator(x: Batch[Complex], y: Batch[Complex], operation: Int) raises -> Batch[Complex]:
    if operation == 0:
        return x + y
    if operation == 1:
        return x - y
    if operation == 2:
        return x * y
    return x / y


def _complex_scalar_operator(x: Complex, y: Complex, operation: Int) raises -> Complex:
    if operation == 0:
        return x + y
    if operation == 1:
        return x - y
    if operation == 2:
        return x * y
    return x / y


def test_stored_complex_pairs_match_scalar_operators() raises:
    var c = ArithmeticContext(format=FloatFormat(96))
    var parts = List[Float]()
    for text in ["1.5", "-2", "0", "-0", "1e40", "3e-30", "inf", "nan"]:
        parts.append(Float(text, context=c))
    var left = List[Complex]()
    var right = List[Complex]()
    for i in range(len(parts)):
        left.append(Complex(parts[i], parts[(i + 3) % len(parts)]))
        right.append(Complex(parts[(i + 1) % len(parts)], parts[(i + 5) % len(parts)]))
    var a = Batch[Complex](left)
    var b = Batch[Complex](right)
    for operation in range(4):
        var result = _complex_operator(a, b, operation)
        for i in range(len(left)):
            assert_equal(result[i].to_json(), _complex_scalar_operator(left[i], right[i], operation).to_json())
    # An exponent-bounds clash is reported for its element, as before.
    var bounded = ArithmeticContext(format=FloatFormat(96, emin=-50, emax=50))
    right[1] = Complex(Float(1, context=bounded), Float(2, context=bounded))
    with assert_raises(contains="element 1"):
        _ = a + Batch[Complex](right)


def test_results_share_matching_operand_layouts() raises:
    var a = Batch[Integer]([1, 2, 3, 4])
    var b = Batch[Integer]([10, 20, 30, 40])
    var sum = a + b
    assert_true(sum._layout.ptr() == a._layout.ptr())
    # Reshaping a result leaves the operand's shape alone.
    var grid = sum.reshape([2, 2])
    assert_equal(grid.shape(), [2, 2])
    assert_equal(a.shape(), [4])
    assert_equal(sum.shape(), [4])
    # A strided view is not the result's layout.
    var tail = a[1::2] + b[1::2]
    assert_equal(tail.shape(), [2])
    assert_equal(tail[0], Integer(22))
    assert_equal(tail[1], Integer(44))
    assert_true(tail._layout.ptr() != a._layout.ptr())
    var f = Batch[Float]([Float(1), Float(2)])
    var g = f[1:] + f[:1]
    assert_equal(g.shape(), [1])
    assert_equal(g[0], Float(3))



def _mixed_integer(i: Int) raises -> Integer:
    var kind = i % 5
    if kind == 0:
        return Integer(i * 37 - 1000)
    if kind == 1:
        return (Integer(1) << 63) + i
    if kind == 2:
        return -((Integer(1) << 63) + 3 * i)
    if kind == 3:
        return (Integer(1) << 100) - i
    return Integer(UInt64.MAX - UInt64(i))


def test_parallel_kernels_match_scalar_loops() raises:
    var n = 20000
    var a = List[Integer]()
    var b = List[Integer]()
    var f = List[Float]()
    var g = List[Float]()
    var narrow = ArithmeticContext(format=FloatFormat(53))
    var wide = ArithmeticContext(format=FloatFormat(200))
    for i in range(n):
        a.append(_mixed_integer(i))
        b.append(_mixed_integer(i * 7 + 3))
        f.append(Float(a[i], context=narrow if i % 3 == 0 else wide))
        g.append(Float(b[i], context=wide))
    var ba = Batch[Integer](a)
    var bb = Batch[Integer](b)
    var bf = Batch[Float](f)
    var bg = Batch[Float](g)
    # Reductions against ordered scalar folds.
    var total = Integer()
    var products = Integer()
    var low = a[0]
    var high = a[0]
    for i in range(n):
        total += a[i]
        products += a[i] * b[i]
        if a[i] < low:
            low = a[i]
        if a[i] > high:
            high = a[i]
    assert_equal(sum(ba), total)
    assert_equal(dot(ba, bb), products)
    # Float sums and dots of integral values round their exact Integer results once.
    var exact_total = Integer()
    var exact_dot = Integer()
    for i in range(n):
        exact_total += f[i].floor()
        exact_dot += f[i].floor() * g[i].floor()
    assert_equal(sum(bf).to_json(), Float(exact_total, context=wide).to_json())
    assert_equal(dot(bf, bg).to_json(), Float(exact_dot, context=wide).to_json())
    assert_equal(amin(ba), low)
    assert_equal(amax(ba), high)
    # Comparisons, unary values and classifications, element by element.
    var less = ba < bb
    var float_less = bf < bg
    var float_equal = bf == bf
    var negated = -bf
    var floors = bf.floor()
    var signs = bf.signbit()
    for i in range(n):
        assert_equal(less[i], a[i] < b[i])
        assert_equal(float_less[i], f[i] < g[i])
        assert_true(float_equal[i])
        assert_equal(negated[i].to_json(), (-f[i]).to_json())
        assert_equal(floors[i], f[i].floor())
        assert_equal(signs[i], f[i].signbit())
    # The lowest failing element is reported, as a sequential loop would.
    f[15000] = Float("inf", context=wide)
    f[12000] = Float("nan", context=wide)
    with assert_raises(contains="element 12000"):
        _ = Batch[Float](f).floor()



def _halved(x: Integer) raises -> Integer:
    if x % 2 != 0:
        raise Error("odd value")
    return x // 2


def _nested_halves(grid: Batch[Integer]) raises -> Batch[Integer]:
    return vmap[_halved](in_axes=0).vmap(in_axes=0)(grid)


def test_nested_vmap_runs_flat_with_nested_errors() raises:
    var values = List[Integer]()
    for i in range(300 * 40):
        values.append(Integer(2 * i))
    var halves = _nested_halves(Batch[Integer](values).reshape([300, 40]))
    assert_equal(halves.shape(), [300, 40])
    var flat = halves.reshape([300 * 40])
    for i in range(300 * 40):
        assert_equal(flat[i], Integer(i))
    # Element (7, 9): the flat run names both levels, for a contiguous or a
    # transposed input, as the general path (forced by axis_size) does.
    values[7 * 40 + 9] = Integer(5)
    var flat_message = String()
    try:
        _ = _nested_halves(Batch[Integer](values).reshape([300, 40]))
    except error:
        flat_message = String(error)
    assert_equal(flat_message, "vmap failed at mapped index 7: vmap failed at mapped index 9: odd value")
    var columns = List[Integer]()
    for j in range(40):
        for i in range(300):
            columns.append(values[i * 40 + j])
    var transposed_message = String()
    try:
        _ = _nested_halves(Batch[Integer](columns).reshape([40, 300]).transpose())
    except error:
        transposed_message = String(error)
    var general_message = String()
    try:
        _ = vmap[_halved](axis_size=40).vmap()(Batch[Integer](values).reshape([300, 40]))
    except error:
        general_message = String(error)
    assert_equal(transposed_message, flat_message)
    assert_equal(general_message, flat_message)



def _cubed(x: Integer) raises -> Integer:
    if x == Integer(-7):
        raise Error("minus seven")
    return x * x * x


def _blend(x: Integer, y: Integer) raises -> Tuple[Integer, Bool]:
    return (x * y + x, x > y)


def _same_batches(a: Batch[Integer], b: Batch[Integer]) raises:
    assert_equal(a.shape(), b.shape())
    assert_equal(a.to_list(), b.to_list())


def test_flat_vmap_over_any_axes() raises:
    # Wide values in a 3-D batch, long enough for the pool. The general path,
    # forced by an innermost axis_size, gives every expected result.
    var values = List[Integer]()
    for i in range(12 * 10 * 30):
        values.append((Integer(i * 7919 + 1) << 400) - i)
    var cube = Batch[Integer](values).reshape([12, 10, 30])
    # Each level maps a different axis; out_axes stack the levels in any order.
    for outer in [0, 1, -1]:
        for middle in [0, -1]:
            var flat = vmap[_cubed]().vmap(in_axes=1, out_axes=middle).vmap(in_axes=2, out_axes=outer)(cube)
            var general = vmap[_cubed](axis_size=12).vmap(in_axes=1, out_axes=middle).vmap(in_axes=2, out_axes=outer)(cube)
            _same_batches(flat, general)
    # Views read in place: transposed, reversed and strided.
    var view = cube.transpose([2, 0, 1]).slice(0, step=-3).slice(2, start=1, step=2)
    _same_batches(vmap[_cubed]().vmap().vmap()(view), vmap[_cubed](axis_size=5).vmap().vmap()(view))
    var cubes = vmap[_cubed]().vmap().vmap()(view).reshape([10 * 12 * 5])
    var expected = view.reshape([10 * 12 * 5])
    for i in range(600):
        assert_true(cubes[i] == expected[i] * expected[i] * expected[i])
    # A shared argument at one level; tuple results with a Bool field.
    var matrix = cube.reshape([120, 30]).transpose()
    var row = Batch[Integer]([Integer(i) << 401 for i in range(120)])
    var pair = vmap[_blend]().vmap(in_axes=(0, None))(matrix, row)
    var general_pair = vmap[_blend](axis_size=120).vmap(in_axes=(0, None))(matrix, row)
    _same_batches(pair[0], general_pair[0])
    assert_equal(pair[1].shape(), [30, 120])
    assert_equal(pair[1].to_list(), general_pair[1].to_list())
    var marks = vmap[_is_square]().vmap(in_axes=1, out_axes=1)(cube.reshape([120, 30]))
    assert_equal(marks.shape(), [120, 30])
    assert_equal(marks.to_list(), vmap[_is_square](axis_size=120).vmap(in_axes=1, out_axes=1)(cube.reshape([120, 30])).to_list())
    # Empty extents keep their shapes.
    var empty = Batch[Integer](List[Integer](), shape=[0, 3])
    _same_batches(vmap[_cubed]().vmap(in_axes=1)(empty), vmap[_cubed](axis_size=0).vmap(in_axes=1)(empty))
    assert_equal(vmap[_cubed]().vmap(in_axes=1)(empty).shape(), [3, 0])
    # The lowest failing call in run order raises the general path's error.
    values[5 * 300 + 7 * 30 + 4] = Integer(-7)
    var broken = Batch[Integer](values).reshape([12, 10, 30])
    var flat_message = String()
    try:
        _ = vmap[_cubed]().vmap(in_axes=1).vmap(in_axes=2)(broken)
    except error:
        flat_message = String(error)
    var general_message = String()
    try:
        _ = vmap[_cubed](axis_size=12).vmap(in_axes=1).vmap(in_axes=2)(broken)
    except error:
        general_message = String(error)
    assert_equal(flat_message, "vmap failed at mapped index 4: vmap failed at mapped index 7: vmap failed at mapped index 5: minus seven")
    assert_equal(general_message, flat_message)


def _row_total(row: Batch[Integer]) raises -> Integer:
    if len(row) and row[0] == Integer(-11):
        raise Error("minus eleven")
    return sum_sequential(row)


def _row_parts(row: Batch[Float]) raises -> Tuple[Float, Bool]:
    var total = sum(row)
    return (total, total > row[0])


def _weighted(row: Batch[Integer], weights: Batch[Integer]) raises -> Integer:
    return dot(row, weights)


def test_flat_vmap_of_batch_arguments() raises:
    # Rows, columns and inner axes of wide values, long enough for the pool;
    # the general path, forced by axis_size, gives every expected result.
    var values = List[Integer]()
    for i in range(80 * 6 * 7):
        values.append((Integer(i * 104729 + 5) << 300) - i)
    var cube = Batch[Integer](values).reshape([80, 6, 7])
    var matrix = cube.reshape([480, 7])
    _same_batches(vmap[_row_total]()(matrix), vmap[_row_total](axis_size=480)(matrix))
    _same_batches(vmap[_row_total](in_axes=1)(matrix), vmap[_row_total](in_axes=1, axis_size=7)(matrix))
    _same_batches(vmap[_row_total](in_axes=1)(matrix.transpose()), vmap[_row_total](axis_size=480)(matrix))
    # Two levels map axes 2 and 1; the function receives axis 0, what is left.
    var nested = vmap[_row_total](in_axes=1).vmap(in_axes=2, out_axes=1)(cube)
    assert_equal(nested.shape(), [6, 7])
    _same_batches(nested, vmap[_row_total](in_axes=1, axis_size=6).vmap(in_axes=2, out_axes=1)(cube))
    for j in range(6):
        for k in range(7):
            assert_true(nested[j, k] == sum_sequential(cube.at(1, j).at(1, k)))
    # A batch argument shared by every call, beside a mapped one.
    var weights = Batch[Integer]([Integer(k + 1) << 70 for k in range(7)])
    var weighted = vmap[_weighted](in_axes=(0, None))(matrix, weights)
    _same_batches(weighted, vmap[_weighted](in_axes=(0, None), axis_size=480)(matrix, weights))
    # Float rows keep their formats in tuple results with a Bool field.
    var context = ArithmeticContext(format=FloatFormat(200))
    var floats = Batch[Float]([Float(values[i], context=context) for i in range(480 * 7)]).reshape([480, 7])
    var parts = vmap[_row_parts]()(floats)
    var general_parts = vmap[_row_parts](axis_size=480)(floats)
    for i in range(480):
        assert_equal(parts[0][i].to_json(), general_parts[0][i].to_json())
    assert_equal(parts[1].to_list(), general_parts[1].to_list())
    # Empty rows and an empty mapping.
    _same_batches(vmap[_row_total]()(Batch[Integer](List[Integer](), shape=[3, 0])), Batch[Integer]([0, 0, 0]))
    assert_equal(len(vmap[_row_total]()(Batch[Integer](List[Integer](), shape=[0, 4]))), 0)
    # The lowest failing row raises the general path's error.
    values[300 * 7] = Integer(-11)
    values[400 * 7] = Integer(-11)
    var broken = Batch[Integer](values).reshape([480, 7])
    var flat_message = String()
    try:
        _ = vmap[_row_total]()(broken)
    except error:
        flat_message = String(error)
    var general_message = String()
    try:
        _ = vmap[_row_total](axis_size=480)(broken)
    except error:
        general_message = String(error)
    assert_equal(flat_message, "vmap failed at mapped index 300: minus eleven")
    assert_equal(general_message, flat_message)


def _affine_parts(parts: Tuple[Integer, Integer, Integer]) raises -> Integer:
    if parts[1] == Integer(-3):
        raise Error("minus three")
    return parts[0] * parts[1] + parts[2]


def _scaled_row(parts: Tuple[Batch[Integer], Integer]) raises -> Tuple[Integer, Bool]:
    var total = sum_sequential(parts[0]) * parts[1]
    return (total, total > parts[1])


def test_flat_vmap_of_tuple_arguments() raises:
    # Wide values, long enough for the pool; the general path, forced by
    # axis_size, gives every expected result.
    var values = List[Integer]()
    for i in range(2000):
        values.append((Integer(i * 7919 + 3) << 400) - i)
    var a = Batch[Integer](values)
    var flat = vmap[_affine_parts](in_axes=(0, 0, None))((a, a[::-1], Integer(7)))
    _same_batches(flat, vmap[_affine_parts](in_axes=(0, 0, None), axis_size=2000)((a, a[::-1], Integer(7))))
    for i in range(2000):
        assert_true(flat[i] == values[i] * values[1999 - i] + 7)
    # Nested levels with per-leaf axes, and a row leaf with tuple results.
    var grid = a.reshape([40, 50])
    var nested = vmap[_affine_parts](in_axes=(0, None, 0)).vmap(in_axes=(1, None, 1))((grid, Integer(5), grid))
    assert_equal(nested.shape(), [50, 40])
    _same_batches(nested, vmap[_affine_parts](in_axes=(0, None, 0), axis_size=40).vmap(in_axes=(1, None, 1))((grid, Integer(5), grid)))
    var parts = vmap[_scaled_row](in_axes=(0, None))((grid, Integer(3)))
    var general_parts = vmap[_scaled_row](in_axes=(0, None), axis_size=40)((grid, Integer(3)))
    _same_batches(parts[0], general_parts[0])
    assert_equal(parts[1].to_list(), general_parts[1].to_list())
    # The lowest failing call raises the general path's error.
    values[1999 - 700] = Integer(-3)
    values[1999 - 1200] = Integer(-3)
    var broken = Batch[Integer](values)
    var flat_message = String()
    try:
        _ = vmap[_affine_parts](in_axes=(0, 0, None))((broken, broken[::-1], Integer(7)))
    except error:
        flat_message = String(error)
    var general_message = String()
    try:
        _ = vmap[_affine_parts](in_axes=(0, 0, None), axis_size=2000)((broken, broken[::-1], Integer(7)))
    except error:
        general_message = String(error)
    assert_equal(flat_message, "vmap failed at mapped index 700: minus three")
    assert_equal(general_message, flat_message)


def test_decimal_text_matches_counted_conversion() raises:
    # Unlimited conversion parses and formats nineteen digits per step; a
    # counted budget keeps the nine-digit parser and digit-by-digit formatter.
    var counted = ConversionLimits(max_digits=10_000_000, max_allocated_bytes=1 << 40)
    var values = List[Integer]()
    var seed = UInt64(0x9E3779B97F4A7C15)
    for bits in range(1, 4100, 37):
        var words = List[UInt32]()
        for _ in range((bits + 31) // 32):
            seed = seed * 6364136223846793005 + 1442695040888963407
            words.append(UInt32(seed >> 32))
        words[len(words) - 1] >>= UInt32(31 - (bits - 1) % 32)
        values.append(Integer._from_words(words^, bits % 3 == 0))
    var ten = Integer(10)
    for k in [18, 19, 20, 37, 38, 39, 57, 95, 300]:
        values.append(ten ** k - 1)
        values.append(ten ** k)
        values.append(ten ** k + 1)
        values.append(-(ten ** k) - 7)
    values.append((Integer(1) << 64) - 1)
    values.append(Integer(1) << 64)
    values.append(ten ** 60 + 7)
    for value in values:
        var text = String(value)
        assert_equal(text, value.to_string(limits=counted))
        assert_equal(value.to_string(), text)
        assert_true(Integer(text) == value)
        assert_true(Integer(text, limits=counted) == value)
        var batch_text = Batch[Integer]([value]).to_json()
        assert_true(Batch[Integer].from_json(batch_text)[0] == value)


def _same_json[T: ImplicitlyCopyable & Deinitable](values: Batch[T], counted: ConversionLimits) raises:
    """Unlimited JSON, parallel for long runs, matches the counted sequential path."""
    var text = values.to_json()
    assert_equal(text, values.to_json(limits=counted))
    assert_equal(Batch[T].from_json(text).to_json(), text)
    assert_equal(Batch[T].from_json(text, limits=counted).to_json(), text)


def test_parallel_json_matches_counted_conversion() raises:
    var counted = ConversionLimits(max_digits=100_000_000, max_allocated_bytes=1 << 40)
    var context = ArithmeticContext(format=FloatFormat(1024))
    var integers = List[Integer]()
    var rationals = List[Rational]()
    var floats = List[Float]()
    var complexes = List[Complex]()
    # 600 wide values: well past the pool's threshold, while the counted
    # reference paths (exact accounting, slower algorithms) stay quick.
    for i in range(600):
        var wide = (Integer(i * 7919 + 3) << 1000) + i * 104729 - 300
        integers.append(-wide if i % 3 == 0 else wide)
        rationals.append(Rational(wide, (Integer(i) << 100) + 1))
        floats.append(Float(wide, context=context))
        complexes.append(Complex(Float(wide, context=context), Float(-wide - 1, context=context)))
    _same_json(Batch[Integer](integers), counted)
    _same_json(Batch[Integer](integers).reshape([30, 20]), counted)
    _same_json(Batch[Rational](rationals), counted)
    _same_json(Batch[Float](floats), counted)
    _same_json(Batch[Complex](complexes), counted)


def _json_error[T: ImplicitlyCopyable & Deinitable](text: String, limits: Optional[ConversionLimits]) -> String:
    try:
        _ = Batch[T].from_json(text, limits=limits)
    except error:
        return String(error)
    return String()


def test_deferred_json_conversion_keeps_errors() raises:
    # Unlimited batch reads convert wide significands after the parse and
    # Rationals on the pool; counted reads convert inline, one at a time.
    var counted = ConversionLimits(max_digits=100_000_000, max_allocated_bytes=1 << 40)
    var wide = ArithmeticContext(format=FloatFormat(1024))
    var narrow = ArithmeticContext(format=FloatFormat(128))
    var unnormalized = String(Integer(1) << 1022)
    # Mixed precisions: only the wide significands wait for the pool.
    var floats = List[Float]()
    for i in range(300):
        floats.append(Float((Integer(i * 7919 + 3) << 1100) + i, context=wide if i % 2 == 0 else narrow))
    var text = Batch[Float](floats).to_json()
    assert_equal(Batch[Float].from_json(text).to_json(), text)
    # An unnormalized wide significand is reported before a later syntax error.
    var good = String(floats[4]._significand)
    var broken = text.replace('"' + good + '"', '"' + unnormalized + '"') + "x"
    var message = _json_error[Float](broken, None)
    assert_true(message.__contains__("unnormalized") and message.__contains__("element 4"))
    assert_equal(message, _json_error[Float](broken, counted))
    # Complex: the same for a component, and records that list imag first.
    var pairs = List[Complex]()
    for i in range(300):
        pairs.append(Complex(floats[i], Float((Integer(i + 11) << 1500) + 1, context=wide)))
    var complex_text = Batch[Complex](pairs).to_json()
    assert_equal(Batch[Complex].from_json(complex_text).to_json(), complex_text)
    var imag_good = String(pairs[3]._imag._significand)
    var complex_broken = complex_text.replace('"' + imag_good + '"', '"' + unnormalized + '"') + "x"
    message = _json_error[Complex](complex_broken, None)
    assert_true(message.__contains__("unnormalized") and message.__contains__("element 3"))
    assert_equal(message, _json_error[Complex](complex_broken, counted))
    var real_json = pairs[0].real().to_json()
    var imag_json = pairs[0].imag().to_json()
    var swapped = Batch[Complex]([pairs[0], pairs[1]]).to_json().replace(
        '"real":' + real_json + ',"imag":' + imag_json, '"imag":' + imag_json + ',"real":' + real_json
    )
    assert_true(swapped != Batch[Complex]([pairs[0], pairs[1]]).to_json())
    var restored = Batch[Complex].from_json(swapped)
    assert_true(restored[0] == pairs[0] and restored[1] == pairs[1])
    # Rational: the lowest noncanonical element is reported.
    var fractions = List[Rational]()
    for i in range(600):
        fractions.append(Rational((Integer(i * 104729 + 1) << 900) + 1, (Integer(i + 2) << 300) + 3))
    var rational_text = Batch[Rational](fractions).to_json()
    assert_equal(Batch[Rational].from_json(rational_text).to_json(), rational_text)
    for i in [400, 250]:
        var n = fractions[i].numerator()
        var d = fractions[i].denominator()
        rational_text = rational_text.replace(
            '{"numerator":"' + String(n) + '","denominator":"' + String(d) + '"}',
            '{"numerator":"' + String(n * 2) + '","denominator":"' + String(d * 2) + '"}',
        )
    message = _json_error[Rational](rational_text, None)
    assert_true(message.__contains__("element 250") and message.__contains__("noncanonical"))
    assert_equal(message, _json_error[Rational](rational_text, counted))


def _below(x: Integer, y: Integer) raises -> Bool:
    return x < y


def _either(x: Bool, y: Bool) raises -> Bool:
    return x or y


def _plus_square(acc: Float, x: Float) raises -> Float:
    return acc + x ** 2


def _hypot(acc: Float, x: Float) raises -> Float:
    return sqrt(acc * acc + x * x)


def _inverse_square(acc: Float, x: Float) raises -> Float:
    return x ** -2


def _root(acc: Float, x: Float) raises -> Float:
    return sqrt(x)


def _complex_cube(acc: Complex, x: Complex) raises -> Complex:
    return x ** 3


def _complex_inverse(acc: Complex, x: Complex) raises -> Complex:
    return x ** -1


def _complex_inverse_square(acc: Complex, x: Complex) raises -> Complex:
    return x ** -2


def _outer_error(a: Batch[Integer], context: ArithmeticContext) -> String:
    try:
        _ = lift[_below]().outer(a, a, context=context)
    except error:
        return String(error)
    return String()


def test_lift_contexts_masks_and_exact_powers() raises:
    var c = ArithmeticContext(format=FloatFormat(20), rounding=RoundingMode.toward_zero)
    var xs = Batch[Float]([Float(1), Float(Integer(3)) / 7, Float(Integer(10) ** 30), Float(-5)])
    # Exact folds round once with the context; step folds pass it to each call.
    assert_equal(lift[float.add]().reduce(xs, axis=None, context=c).to_json(), sum(xs, context=c).to_json())
    var running = lift[float.add]().accumulate(xs, context=c)
    assert_equal(running[3].to_json(), sum(xs, context=c).to_json())
    var stepped = float.add(float.add(float.add(xs[0], xs[1], context=c), xs[2], context=c), xs[3], context=c)
    assert_equal(lift[float.add](exact=False).reduce(xs, axis=None, context=c).to_json(), stepped.to_json())
    assert_equal(lift[float.add]().outer(xs, xs, context=c)[1, 2].to_json(), float.add(xs[1], xs[2], context=c).to_json())
    var pairs = ComplexContext(real=c, imag=ArithmeticContext(format=FloatFormat(40)))
    var zs = Batch[Complex]([Complex(1, 2), Complex(Integer(1) << 70, -3), Complex(Float(Integer(1)) / 3, 0)])
    assert_equal(lift[complex.add]().reduce(zs, axis=None, context=pairs).to_json(), sum(zs, context=pairs).to_json())
    # A context nothing can use is an error.
    var a = Batch[Integer]([1, 5, 3])
    assert_true(_outer_error(a, c).__contains__("takes no context"))
    with assert_raises(contains="takes no context"):
        _ = lift[_plus_square](exact=False).reduce(xs, axis=None, context=c)
    with assert_raises(contains="takes no context"):
        _ = lift[lambda (x: Integer, y: Integer) raises -> Integer: x + y]().reduce(a, axis=None, context=c)
    with assert_raises(contains="ArithmeticContext for Float"):
        _ = lift[float.add]().reduce(xs, axis=None, context=pairs)
    # Bool results are Masks; Bool functions fold Masks.
    var m = lift[_below]().outer(a, a)
    assert_equal(m.shape(), [3, 3])
    assert_equal(m.to_list(), [False, True, True, False, False, False, False, True, False])
    var column = Batch[Integer]([2, 4]).reshape([2, 1])
    var broadcast = lift[_below]()(column, a)
    assert_equal(broadcast.shape(), [2, 3])
    assert_equal(broadcast.to_list(), [False, True, True, False, True, False])
    var any = lift[_either](identity=False)
    assert_equal(any.reduce(m, axis=1).to_list(), [True, False, True])
    assert_equal(any.reduce(m, axis=1, keepdims=True).shape(), [3, 1])
    assert_equal(any.reduce(m, axis=[0, 1]).to_list(), [True])
    assert_true(any.reduce(m, axis=None))
    assert_true(not any.reduce(Mask(List[Bool]()), axis=None))
    assert_true(any.reduce(Mask(List[Bool]()), axis=None, initial=True))
    assert_equal(any.accumulate(m, axis=1).to_list(), [False, True, True, False, False, False, False, True, True])
    with assert_raises(contains="no identity"):
        _ = lift[_either]().reduce(Mask(List[Bool]()), axis=None)
    # Exact folds compute powers and square roots that are finite binary fractions.
    var ys = Batch[Float]([Float(3), Float(4), Float(12)])
    assert_equal(lift[_plus_square]().reduce(ys, axis=None), Float(163))
    assert_equal(lift[_hypot]().reduce(ys, axis=None), Float(13))
    assert_equal(lift[_hypot]().reduce(Batch[Float]([Float(0), Float(Integer(1)) / 4]), axis=None), Float(Integer(1)) / 4)
    assert_equal(lift[_inverse_square]().reduce(Batch[Float]([Float(0), Float(Integer(1)) / 8]), axis=None), Float(64))
    with assert_raises(contains="reduce failed at index [1]: Cannot compute sqrt exactly"):
        _ = lift[_hypot]().reduce(Batch[Float]([Float(1), Float(1)]), axis=None)
    with assert_raises(contains="Cannot compute"):
        _ = lift[_inverse_square]().reduce(Batch[Float]([Float(0), Float(3)]), axis=None)
    # 1/4 is 2**-2, so its root is 1/2; 1/8 = 2**-3 has an odd power of two.
    assert_equal(lift[_root]().reduce(Batch[Float]([Float(0), Float(Integer(1)) / 4]), axis=None), Float(Integer(1)) / 2)
    with assert_raises(contains="Cannot compute sqrt exactly"):
        _ = lift[_root]().reduce(Batch[Float]([Float(0), Float(Integer(1)) / 8]), axis=None)
    # Complex powers agree with rounded powers, signed zeros included, on the
    # axes, the diagonals and in general position; inverses need a dyadic norm.
    var points = [
        Complex(0, 2), Complex(0, -2), Complex(-2, 0), Complex(2, Float.zero(negative=True)),
        Complex(1, 1), Complex(-1, 1), Complex(1, -1), Complex(-1, -1), Complex(Float.zero(negative=True), 4),
        Complex(3, -1), Complex(-5, 7),
    ]
    for z in points:
        var both = Batch[Complex]([Complex(0), z])
        assert_equal(lift[_complex_cube]().reduce(both, axis=None).to_json(), (z ** 3).to_json())
        if z.real().is_zero() or z.imag().is_zero() or abs(z.real()) == abs(z.imag()):
            assert_equal(lift[_complex_inverse]().reduce(both, axis=None).to_json(), (z ** -1).to_json())
            assert_equal(lift[_complex_inverse_square]().reduce(both, axis=None).to_json(), (z ** -2).to_json())
    with assert_raises(contains="Cannot compute"):
        _ = lift[_complex_inverse]().reduce(Batch[Complex]([Complex(0), Complex(1, 2)]), axis=None)


def _quotient_parts(a: Integer, b: Integer) raises -> Tuple[Integer, Rational, Float, Complex, Bool]:
    """One field of every kind a flat run moves out of its tuples."""
    var q = a // b
    return (q, Rational(a, b), Float(a), Complex(q, a % b), isqrt(q) * isqrt(q) == q)


def _is_square(x: Integer) raises -> Bool:
    return isqrt(x) * isqrt(x) == x


def test_flat_vmap_tuple_and_boolean_results() raises:
    # Wide values and long runs, so the calls go to the worker pool.
    var numerators = List[Integer]()
    var denominators = List[Integer]()
    for i in range(1500):
        var root = (Integer(i * 7919 + 3) << 350) + i
        numerators.append(root * root if i % 5 == 0 else root * root + i + 1)
        denominators.append((Integer(i % 37 + 1) << 300) + 1)
    var a = Batch[Integer](numerators)
    var b = Batch[Integer](denominators)
    var parts = vmap[_quotient_parts]()(a, b)
    var pairs = vmap[div_rem_floor]()(a, b)
    var squares = vmap[_is_square]()(a)
    # Every field of a long heap result records who computed its elements.
    assert_true(Bool(pairs[0]._owner[].ownership) and Bool(pairs[1]._owner[].ownership))
    for i in range(1500):
        var expected = _quotient_parts(numerators[i], denominators[i])
        assert_true(parts[0][i] == expected[0] and parts[1][i] == expected[1])
        assert_true(parts[2][i] == expected[2] and parts[3][i] == expected[3])
        assert_equal(parts[4][i], expected[4])
        var pair = div_rem_floor(numerators[i], denominators[i])
        assert_true(pairs[0][i] == pair[0] and pairs[1][i] == pair[1])
        assert_equal(squares[i], i % 5 == 0)
    assert_equal(squares.count(), 300)
    # Nested levels keep their shape; strided inputs and shared scalars read in place.
    var grid = vmap[div_rem_floor]().vmap()(a.reshape([30, 50]), b.reshape([30, 50]))
    var marks = vmap[_is_square]().vmap()(a.reshape([30, 50]))
    assert_equal(grid[0].shape(), [30, 50])
    assert_equal(marks.shape(), [30, 50])
    assert_equal(grid[1].reshape([1500]).to_list(), pairs[1].to_list())
    assert_equal(marks.reshape([1500]).to_list(), squares.to_list())
    var strided = vmap[div_rem_floor](in_axes=(0, None))(a[::-3], Integer(97))
    for i in range(500):
        var pair = div_rem_floor(numerators[1499 - 3 * i], Integer(97))
        assert_true(strided[0][i] == pair[0] and strided[1][i] == pair[1])
    var empty = vmap[div_rem_floor]()(a[:0], b[:0])
    assert_equal(len(empty[0]) + len(empty[1]) + len(vmap[_is_square]()(a[:0])), 0)
    # The lowest failing element raises the general path's error, and nothing is published.
    var zeros = List[Integer]()
    for i in range(1500):
        zeros.append(Integer(0) if i == 700 or i == 1200 else denominators[i])
    var flat_message = String()
    try:
        pairs = vmap[div_rem_floor]()(a, Batch[Integer](zeros))
    except error:
        flat_message = String(error)
    var general_message = String()
    try:
        pairs = vmap[div_rem_floor](axis_size=1500)(a, Batch[Integer](zeros))
    except error:
        general_message = String(error)
    assert_true(flat_message.startswith("vmap failed at mapped index 700: "))
    assert_equal(flat_message, general_message)
    assert_true(pairs[1][700] == numerators[700] % denominators[700])


def _first(x: Integer, y: Integer) raises -> Integer:
    return x


def _last(x: Integer, y: Integer) raises -> Integer:
    return y


def _unlucky_add(x: Integer, y: Integer) raises -> Integer:
    if y == Integer(13):
        raise Error("unlucky")
    return x + y


def test_lift_reduce_accumulate_outer_and_call() raises:
    var add = lift[lambda (x: Integer, y: Integer) raises -> Integer: x + y](identity=Integer(0), associative=True)
    var plain = lift[lambda (x: Integer, y: Integer) raises -> Integer: x + y]()
    var values = List[Integer]()
    for i in range(24):
        values.append(Integer(i * i - 50))
    var cube = Batch[Integer](values).reshape([2, 3, 4])
    for axis in [0, 1, 2, -1]:
        assert_equal(add.reduce(cube, axis=axis).to_list(), sum(cube, axis=axis).to_list())
        assert_equal(plain.reduce(cube, axis=axis).to_list(), sum(cube, axis=axis).to_list())
    assert_equal(add.reduce(cube, axis=[0, 2]).to_list(), sum(sum(cube, axis=2), axis=0).to_list())
    assert_equal(add.reduce(cube, axis=1, keepdims=True).shape(), [2, 1, 4])
    assert_equal(add.reduce(cube, axis=None), sum(cube.reshape([24])))
    assert_equal(plain.reduce(cube, axis=0, initial=Integer(1000)).to_list()[0], Integer(1000 - 50 + 94))
    with assert_raises(contains="twice"):
        _ = add.reduce(cube, axis=[1, -2])
    # Empty selections: the identity, then initial, else an error naming both remedies.
    var empty = Batch[Integer](List[Integer]()).reshape([0, 3])
    assert_equal(add.reduce(empty, axis=0).to_list(), [Integer(0), Integer(0), Integer(0)])
    assert_equal(plain.reduce(empty, axis=0, initial=Integer(7)).to_list(), [Integer(7), Integer(7), Integer(7)])
    with assert_raises(contains="identity= or pass initial="):
        _ = plain.reduce(empty, axis=0)
    assert_equal(plain.reduce(empty, axis=1).shape(), [0])
    # Chunked single lanes keep operand order: projections are associative
    # but not commutative.
    var numbers = List[Integer]()
    for i in range(100000):
        numbers.append(Integer(i))
    var long = Batch[Integer](numbers)
    assert_equal(lift[_first](associative=True).reduce(long, axis=None), Integer(0))
    assert_equal(lift[_last](associative=True).reduce(long, axis=None), Integer(99999))
    assert_equal(add.reduce(long, axis=None), sum(long))
    # Many lanes run in parallel, each a strict left fold.
    var grid = long.reshape([1000, 100])
    var differences = lift[lambda (x: Integer, y: Integer) raises -> Integer: x - y]().reduce(grid, axis=1)
    var running = add.accumulate(grid, axis=0)
    for r in range(1000):
        assert_equal(differences[r], Integer(r * 100 - (99 * r * 100 + 4950)))
        assert_equal(running[r, 7], Integer((r * (r + 1) // 2) * 100 + 7 * (r + 1)))
    # Errors name the coordinates of the first failing operand.
    with assert_raises(contains="reduce failed at index [0, 13]: unlucky"):
        _ = lift[_unlucky_add]().reduce(grid, axis=1)
    with assert_raises(contains="accumulate failed at index [0, 13]: unlucky"):
        _ = lift[_unlucky_add]().accumulate(grid, axis=1)
    # Outer products and broadcasting calls, here with Float.
    var multiply = lift[lambda (x: Float, y: Float) raises -> Float: x * y]()
    var xs = Batch[Float]([Float(1), Float(2), Float(3)])
    var table = multiply.outer(xs, Batch[Float]([Float(10), Float(20)]))
    assert_equal(table.shape(), [3, 2])
    assert_equal(table[2, 1], Float(60))
    var shifted = add(cube, Batch[Integer]([1, 2, 3, 4]))
    assert_equal(shifted.shape(), [2, 3, 4])
    assert_equal(shifted[1, 2, 3], values[23] + 4)
    assert_equal(add(cube, Integer(1))[0, 0, 0], values[0] + 1)
    with assert_raises(contains="Cannot broadcast shapes [2, 3, 4] and [3]"):
        _ = add(cube, Batch[Integer]([1, 2, 3]))



def test_integer_in_place_addition_keeps_value_semantics() raises:
    var a = (Integer(1) << 100) - 1
    var copy = a
    a += Integer(1)
    assert_equal(copy, (Integer(1) << 100) - 1)
    assert_equal(a, Integer(1) << 100)
    var b = (Integer(1) << 64) - 1
    b += Integer(1)
    assert_equal(b, Integer(1) << 64)
    var c = Integer(1) << 64
    c += c
    assert_equal(c, Integer(1) << 65)
    var d = -(Integer(1) << 80)
    d += Integer(5)
    assert_equal(d, Integer(5) - (Integer(1) << 80))
    d -= Integer(1) << 80
    assert_equal(d, Integer(5) - (Integer(1) << 81))
    var e = Integer(1) << 70
    e += Integer(1) << 200
    assert_equal(e, (Integer(1) << 200) + (Integer(1) << 70))
    var running = Integer(1) << 62
    var total = running
    for i in range(1, 2000):
        var term = (Integer(1) << 62) + i * 7919
        running += term
        total = total + term
    assert_equal(running, total)


def test_integer_in_place_addition_carries_and_grows() raises:
    # Uniquely owned values add in place, carrying through all-ones words,
    # into spare capacity and past it.
    var z = (Integer(1) << 128) - 1
    z += Integer(1)
    assert_equal(z, Integer(1) << 128)
    z += (Integer(1) << 300) + 5
    assert_equal(z, (Integer(1) << 300) + (Integer(1) << 128) + 5)
    var w = (Integer(1) << 96) - 1
    w -= -((Integer(1) << 96) - 1)
    assert_equal(w, (Integer(1) << 97) - 2)


def _twos_complement_bits(x: Integer, y: Integer, operation: Int) raises -> Integer:
    """x op y from 32-bit digits of x and y modulo 2**512, without bitwise Integer
    operations: operation 0 is and, 1 or, 2 xor, 3 the complement of x."""
    var modulus = Integer(1) << 512
    var digit = Integer(1) << 32
    var a = x % modulus
    var b = y % modulus
    var result = Integer(0)
    var scale = Integer(1)
    for _ in range(16):
        var p = UInt32(Int(a % digit))
        var q = UInt32(Int(b % digit))
        var r = p & q if operation == 0 else p | q if operation == 1 else p ^ q if operation == 2 else ~p
        result += Integer(Int(r)) * scale
        a = a // digit
        b = b // digit
        scale = scale * digit
    return result - modulus if result >= modulus // 2 else result


def test_integer_bitwise_matches_twos_complement_digits() raises:
    var state = Integer(0x9E3779B97F4A7C15)
    var sizes = [1, 31, 32, 33, 63, 64, 65, 95, 96, 97, 128, 200, 300]
    for i in range(len(sizes)):
        for j in range(len(sizes)):
            state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 400)
            var x = (state % (Integer(1) << sizes[i])) | (Integer(1) << (sizes[i] - 1))
            state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 400)
            var y = ((state % (Integer(1) << sizes[j])) | (Integer(1) << (sizes[j] - 1))) >> (j % 3) << (j % 3)
            for signs in range(4):
                var a = -x if signs & 1 else x
                var b = -y if signs & 2 else y
                assert_equal(a & b, _twos_complement_bits(a, b, 0))
                assert_equal(a | b, _twos_complement_bits(a, b, 1))
                assert_equal(a ^ b, _twos_complement_bits(a, b, 2))
                assert_equal(~a, _twos_complement_bits(a, b, 3))


def test_inline_gcd_and_small_factorials() raises:
    assert_equal(gcd(Integer(Int64.MIN), Integer(0)), Integer(1) << 63)
    assert_equal(gcd(Integer(Int64.MIN), Integer(Int64.MIN)), Integer(1) << 63)
    assert_equal(gcd(Integer(Int64.MIN), Integer(6)), Integer(2))
    assert_equal(gcd(Integer(-12), Integer(18)), Integer(6))
    assert_equal(gcd(Integer(0), Integer(0)), Integer(0))
    assert_equal(gcd(Integer(0), Integer(-7)), Integer(7))
    var state = 88172645463325252
    for _ in range(200):
        state = (state * 2862933555777941757 + 3037000493) & 0x7FFFFFFFFFFFFFFF
        var a = Integer(state >> (state & 63))
        state = (state * 2862933555777941757 + 3037000493) & 0x7FFFFFFFFFFFFFFF
        var b = Integer(state >> (state & 31)) * (1 + (state & 7))
        assert_equal(gcd(a, b), _euclid_gcd(a, b))
    # n! and n!! come from tables while they fit an Int64.
    var f = Integer(1)
    var d = List[Integer]()
    d.append(Integer(1))
    d.append(Integer(1))
    for n in range(41):
        if n:
            f *= n
        if n >= 2:
            d.append(d[n - 2] * n)
        assert_equal(factorial(n), f)
        assert_equal(factorial2(n), d[n])
    assert_equal(factorial(20), Integer(2432902008176640000))
    assert_equal(factorial2(33), Integer(6332659870762850625))


def test_rational_inline_arithmetic_matches_integer_reference() raises:
    # Parts that fit Int64 take 128-bit paths; each result must equal the
    # fraction built from Integer cross products and reduced by the constructor.
    var numerators: List[Int64] = [
        0, 1, -1, 7, -12, 2147483647, -2147483648, (1 << 62) + 3,
        -((1 << 62) + 5), Int64.MAX, Int64.MIN, Int64.MIN + 1,
    ]
    var denominators: List[Int64] = [
        1, 2, 3, 6, 4294967297, (1 << 61) + 1, Int64.MAX - 1, Int64.MAX,
    ]
    var values = List[Rational]()
    for n in numerators:
        for d in denominators:
            values.append(Rational(Integer(n), Integer(d)))
    for a in values:
        for b in values:
            var an = a.numerator()
            var ad = a.denominator()
            var bn = b.numerator()
            var bd = b.denominator()
            assert_equal(a + b, Rational(an * bd + bn * ad, ad * bd))
            assert_equal(a - b, Rational(an * bd - bn * ad, ad * bd))
            assert_equal(a * b, Rational(an * bn, ad * bd))
            if bn:
                assert_equal(a / b, Rational(an * bd, ad * bn))
            assert_equal(a < b, an * bd < bn * ad)
            assert_equal(a > b, an * bd > bn * ad)


def test_division_by_one_limb_divisors() raises:
    # Dividends of three or more words divide by a divisor of at most 64 bits
    # in one pass; floor and truncating results must both reconstruct the dividend.
    var dividends = List[Integer]()
    dividends.append((Integer(1) << 64) + 1)
    dividends.append((Integer(1) << 95) - 1)
    dividends.append((Integer(3) << 200) + 12345)
    dividends.append(-((Integer(1) << 129) + 7))
    dividends.append((Integer(1) << 2000) - 1)
    dividends.append(-(Integer(1) << 160))
    var divisors = List[Integer]()
    for b in [Integer(1), Integer(3), Integer(2147483647), Integer(4294967296),
              (Integer(1) << 63) - 1, Integer(1) << 63, (Integer(1) << 64) - 1]:
        divisors.append(b)
        divisors.append(-b)
    for a in dividends:
        for b in divisors:
            var q = a // b
            var r = a % b
            assert_equal(q * b + r, a)
            assert_true(abs(r) < abs(b) and (not r or r.sign() == b.sign()))
            var t, s = div_rem_trunc(a, b)
            assert_equal(t * b + s, a)
            assert_true(abs(s) < abs(b) and (not s or s.sign() == a.sign()))
            var e, u = div_rem_euclid(a, b)
            assert_equal(e * b + u, a)
            assert_true(u >= 0 and u < abs(b))


def _round_to_bits(value: Rational, p: Int, mode: RoundingMode) raises -> Rational:
    """`value` rounded once to p significant bits in `mode`, from Integer
    arithmetic alone: the reference for the native quotient paths."""
    if not value:
        return value
    var negative = value.sign() < 0
    var n = abs(value.numerator())
    var d = value.denominator()
    # 2**e <= n / d < 2**(e + 1).
    var e = n.magnitude_bit_length() - d.magnitude_bit_length()
    if (n << max(-e, 0)) < (d << max(e, 0)):
        e -= 1
    var shift = p - 1 - e
    var divisor = d << max(-shift, 0)
    var q, r = div_rem_trunc(n << max(shift, 0), divisor)
    var up = False
    if r:
        if mode == RoundingMode.nearest_even:
            up = r * 2 > divisor or (r * 2 == divisor and q % 2 == 1)
        elif mode == RoundingMode.toward_positive:
            up = not negative
        elif mode == RoundingMode.toward_negative:
            up = negative
        elif mode == RoundingMode.away_from_zero:
            up = True
    if up:
        q += 1
    var magnitude = Rational(q << max(-shift, 0), Integer(1) << max(shift, 0))
    return -magnitude if negative else magnitude


def test_native_quotients_round_once() raises:
    # Operands of at most 256 bits round through a native quotient, and Complex
    # division with short binary parts forms its numerators and denominator
    # natively; each result must be the exact value rounded once.
    var modes = [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]
    var precisions = [2, 11, 24, 53, 64, 100, 113, 128]
    var state = Integer(0x9E3779B97F4A7C15)
    var limit = Integer(1) << 256
    for trial in range(60):
        state = (state * 6364136223846793005 + 1442695040888963407) % limit
        var n = state >> (trial * 3 % 250)
        state = (state * 6364136223846793005 + 1442695040888963407) % limit
        var d = (state >> (trial * 7 % 250)) | 1
        if not n:
            continue
        var ratio = Rational(-n if trial % 3 == 0 else n, d)
        for p in precisions:
            for mode in modes:
                var context = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                assert_equal(Float(ratio, context=context).to_rational_exact(), _round_to_bits(ratio, p, mode))
    var source = ArithmeticContext(format=FloatFormat(64))
    for trial in range(40):
        var parts = List[Rational]()
        for k in range(4):
            state = (state * 6364136223846793005 + 1442695040888963407) % limit
            var significand = Integer(-1 if (trial >> k) & 1 else 1) * (state >> (194 + (trial + k) % 40))
            # Parts a few bits apart, and every tenth divisor's parts 200 apart.
            var scale = (trial * (k + 3)) % 17 - 8 - (200 if trial % 10 == 0 and k == 3 else 0)
            parts.append(ldexp(Float(significand, context=source), scale, context=source).to_rational_exact())
        if not parts[2] and not parts[3]:
            continue
        var left = Complex(Float(parts[0], context=source), Float(parts[1], context=source))
        var right = Complex(Float(parts[2], context=source), Float(parts[3], context=source))
        var denominator = parts[2] * parts[2] + parts[3] * parts[3]
        var real = (parts[0] * parts[2] + parts[1] * parts[3]) / denominator
        var imag = (parts[1] * parts[2] - parts[0] * parts[3]) / denominator
        for p in precisions:
            for mode in modes:
                var rc = ArithmeticContext(format=FloatFormat(p), rounding=mode)
                var ic = ArithmeticContext(format=FloatFormat(max(2, p - 9)), rounding=mode)
                var result = divide(left, right, context=ComplexContext(real=rc, imag=ic))
                assert_equal(result.real().to_rational_exact(), _round_to_bits(real, p, mode))
                assert_equal(result.imag().to_rational_exact(), _round_to_bits(imag, max(2, p - 9), mode))


def _add_into(mut total: Integer, value: Integer) raises:
    total += value


def test_built_in_reductions_share_the_lane_driver() raises:
    var values = List[Integer]()
    for i in range(24):
        values.append(Integer((i * 7) % 11 - 5))
    var cube = Batch[Integer](values).reshape([2, 3, 4])
    assert_equal(sum(cube, axis=[0, 2]).to_list(), sum(sum(cube, axis=2), axis=0).to_list())
    assert_equal(prod(cube, axis=[1, 2], keepdims=True).shape(), [2, 1, 1])
    assert_equal(amin(cube, axis=[0, 1, 2]).shape(), List[Int]())
    assert_equal(amin(cube, axis=[0, 1, 2]).item(), amin(cube.reshape([24])))
    assert_equal(amax(cube, axis=-1).to_list(), amax(cube, axis=2).to_list())
    var numbers = List[Integer]()
    for i in range(100000):
        numbers.append((Integer(1) << 62) + i * 7919)
    var long = Batch[Integer](numbers)
    var grid = long.reshape([1000, 100])
    var into = lift[_add_into](identity=Integer(0), associative=True)
    assert_equal(into.reduce(long, axis=None), sum(long))
    assert_equal(into.reduce(grid, axis=1).to_list(), sum(grid, axis=1).to_list())
    assert_equal(into.accumulate(grid, axis=1)[999, 99], sum(grid, axis=1)[999])
    assert_equal(into.outer(Batch[Integer]([1, 2]), Batch[Integer]([10]))[1, 0], Integer(12))
    assert_equal(into(grid, Integer(1))[0, 0], numbers[0] + 1)



def test_lift_rounds_float_and_complex_results_once() raises:
    var c = ArithmeticContext(format=FloatFormat(53))
    var xs = Batch[Float]([Float(1, context=c), Float("1e-16", context=c), Float("1e-16", context=c), Float("1e-16", context=c)])
    var add = lift[lambda (x: Float, y: Float) raises -> Float: x + y](associative=True)
    var stepwise = lift[lambda (x: Float, y: Float) raises -> Float: x + y](exact=False)
    assert_equal(add.reduce(xs, axis=None), sum(xs))
    assert_equal(add.reduce(xs, axis=None).format(), sum(xs).format())
    assert_true(add.reduce(xs, axis=None) != Float(1, context=c))
    assert_equal(stepwise.reduce(xs, axis=None), sum_sequential(xs))
    # Exact folds are associative, so chunked results equal one rounding.
    var values = List[Float]()
    for i in range(20000):
        values.append(Float(Integer(i * 7919 - 50000) << (i % 97), context=c))
    var many = Batch[Float](values)
    assert_equal(add.reduce(many, axis=None), sum(many))
    var grid = many.reshape([200, 100])
    assert_equal(add.reduce(grid, axis=1).to_list(), sum(grid, axis=1).to_list())
    # Each running value is rounded once from its exact prefix.
    var running = add.accumulate(xs)
    for k in range(4):
        assert_equal(running[k], sum(xs[: k + 1]))
    # Products round once too.
    var factors = Batch[Float]([Float(Integer(3) ** 30 + 1, context=c), Float(Integer(5) ** 20 + 7, context=c), Float(Integer(7) ** 18 + 3, context=c)])
    var exact_product = ((Integer(3) ** 30 + 1) * (Integer(5) ** 20 + 7)) * (Integer(7) ** 18 + 3)
    var multiply = lift[lambda (x: Float, y: Float) raises -> Float: x * y]()
    assert_equal(multiply.reduce(factors, axis=None), Float(exact_product, context=c))
    # Inexact steps raise with coordinates; exact=False rounds them instead.
    var divide = lift[lambda (x: Float, y: Float) raises -> Float: x / y]()
    var thirds = Batch[Float]([Float(1, context=c), Float(4, context=c), Float(3, context=c)])
    with assert_raises(contains="reduce failed at index [2]: Cannot compute the quotient exactly"):
        _ = divide.reduce(thirds, axis=None)
    _ = lift[lambda (x: Float, y: Float) raises -> Float: x / y](exact=False).reduce(thirds, axis=None)
    with assert_raises(contains="Cannot compute sqrt exactly"):
        _ = lift[lambda (x: Float, y: Float) raises -> Float: float.sqrt(x * x + y * y)]().reduce(thirds, axis=None)
    var bounded = ArithmeticContext(format=FloatFormat(53, emin=-9, emax=9))
    with assert_raises(contains="reduce failed at index [1]: FormatMismatch"):
        _ = add.reduce(Batch[Float]([Float(1, context=c), Float(1, context=bounded)]), axis=None)
    # Complex results round each component once.
    var zs = Batch[Complex]([Complex(xs[i], xs[3 - i]) for i in range(4)])
    assert_equal(lift[lambda (x: Complex, y: Complex) raises -> Complex: x + y]().reduce(zs, axis=None), sum(zs))
    # The built-in rounded sums take several axes and stay rounded once.
    var cube = many.reshape([20, 10, 100])
    assert_equal(sum(cube, axis=[0, 2]).to_list(), add.reduce(cube, axis=[0, 2]).to_list())
    assert_equal(sum_sequential(cube, axis=[1, 2]).shape(), [20])



def _scaled(significand: Integer, scale: Int, negative: Bool) raises -> Rational:
    var value = -significand if negative else significand
    if scale >= 0:
        return Rational(value << scale)
    return Rational(value, Integer(1) << -scale)


def _same_rounding(actual: Float, exact: Rational, context: ArithmeticContext) raises:
    """actual is exact rounded once in context; an exact zero is +0, or -0
    when rounding toward negative."""
    if exact:
        assert_equal(actual.to_json(), Float(exact, context=context).to_json())
    else:
        assert_true(actual.is_zero())
        assert_equal(actual.signbit(), context.rounding() == RoundingMode.toward_negative)


def _check_short_operations(
    x: Float, xv: Rational, y: Float, yv: Rational, w: Float, wv: Rational, context: ArithmeticContext,
) raises:
    """Sums, products, quotients, square roots and Complex products of Floats
    holding exactly xv, yv and wv, against one rounding of the exact results."""
    _same_rounding(float.add(x, y, context=context), xv + yv, context)
    _same_rounding(float.multiply(x, y, context=context), xv * yv, context)
    _same_rounding(Float(_rounded=Float._quotient(x, y, context)), xv / yv, context)
    # The reference root rounds the same exact value from a wider Float.
    var wide = ArithmeticContext(format=FloatFormat(context.format().precision() * 2 + 8))
    assert_equal(float.sqrt(abs(y), context=context).to_json(), float.sqrt(Float(abs(yv), context=wide), context=context).to_json())
    # w shares x's significand at another scale and sign.
    var z = complex.multiply(Complex(x, y), Complex(y, w), context=context)
    _same_rounding(z.real(), xv * yv - yv * wv, context)
    _same_rounding(z.imag(), xv * wv + yv * yv, context)


def test_short_float_arithmetic_rounds_like_exact_values() raises:
    var modes: List[RoundingMode] = [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]
    var state = 12345
    for precision in [53, 113, 128]:
        var exact = ArithmeticContext(format=FloatFormat(precision))
        for trial in range(300):
            var a = Integer(1)
            var b = Integer(1)
            for _ in range(3):
                state = (state * 1103515245 + 12345) % 2147483648
                a = (a << 31) + state
                state = (state * 1103515245 + 12345) % 2147483648
                b = (b << 31) + state
            a = a % (Integer(1) << precision) + 1
            b = b % (Integer(1) << precision) + 1
            if trial % 7 == 0:
                b = a - trial % 3    # near or exact cancellation
            var sa = (state % 300) - 150
            var sb = sa - (trial % 280) + 60    # gaps up to beyond 128 bits
            var na = trial % 2 == 0
            var nb = trial % 3 == 0
            # Every fourth x is a shorter Float, so merged formats mix precisions.
            var short = max(2, precision // 3)
            if trial % 4 == 3:
                a = a >> (precision - short)
            var x = Float(_scaled(a, sa, na), context=ArithmeticContext(format=FloatFormat(short)) if trial % 4 == 3 else exact)
            var y = Float(_scaled(b, sb, nb), context=exact)
            var w = Float(_scaled(a, sa + 7, not na), context=exact)
            for mode in modes:
                _check_short_operations(
                    x, _scaled(a, sa, na), y, _scaled(b, sb, nb), w, _scaled(a, sa + 7, not na),
                    ArithmeticContext(format=FloatFormat(precision), rounding=mode),
                )



def test_exact_sums_across_dense_windows() raises:
    """Float sums and dots round the exact total once, whether values share one
    dense window, lie too far apart for it, or arrive in merged chunks."""
    var bits = ArithmeticContext(format=FloatFormat(64))
    var values = List[Float]()
    var twos = List[Float]()
    var exact = Rational(0)
    var even = Rational(0)
    var state = 7
    for i in range(3000):
        state = (state * 1103515245 + 12345) % 2147483648
        var value = Rational(Integer(state) * (1 if i % 2 else -1) << 100, Integer(1) << ((state % 200) + 1))
        # Pairs too far apart for one window, cancelling across chunks; each
        # is padded with a zero so both halves sit at even indices.
        if i == 100 or i == 2900:
            values.append(Float(Rational(Integer(3 if i == 100 else -3) << 600000), context=bits))
            values.append(Float(0))
        if i == 1500 or i == 2000:
            values.append(Float(Rational(Integer(5 if i == 1500 else -5), Integer(1) << 600000), context=bits))
            values.append(Float(0))
        values.append(Float(value, context=bits))
        exact = exact + value
        if len(values) % 2 == 1:
            even = even + value
    # A small value only a wide window keeps exactly.
    var small = Rational(Integer(1), Integer(1) << 3000)
    values.append(Float(small, context=bits))
    exact = exact + small
    if len(values) % 2 == 1:
        even = even + small
    for _ in range(len(values)):
        twos.append(Float(2))
    var xs = Batch[Float](values)
    var ys = Batch[Float](twos)
    for mode in [RoundingMode.nearest_even, RoundingMode.toward_positive, RoundingMode.toward_negative]:
        var c = ArithmeticContext(format=FloatFormat(64), rounding=mode)
        assert_equal(sum(xs, context=c).to_json(), Float(exact, context=c).to_json())
        assert_equal(dot(xs, ys, context=c).to_json(), Float(exact * 2, context=c).to_json())
        # Strided runs of stored values.
        assert_equal(dot(xs[::2], ys[::2], context=c).to_json(), Float(even * 2, context=c).to_json())
    # Carries through the window, a negative total and merged sums.
    var total = _ExactAccumulator()
    var check = Rational(0)
    for i in range(200):
        var term = Integer(0xFFFFFFFF) * (i + 1) << (i % 70)
        total.add(term, 0)
        check = check + Rational(term)
    var other = _ExactAccumulator()
    other.add(-(Integer(1) << 300), -3)
    total.merge(other)
    check = check - Rational(Integer(1) << 300, Integer(8))
    var before = Float(_rounded=total.magnitude().rounded(bits))
    total._carry()
    assert_equal(Float(_rounded=total.magnitude().rounded(bits)).to_json(), before.to_json())
    assert_equal(before.to_json(), Float(check, context=bits).to_json())


def _euclid_gcd(a: Integer, b: Integer) raises -> Integer:
    var x = abs(a)
    var y = abs(b)
    while y._word_count() != 0:
        var r = x % y
        x = y
        y = r
    return x


def test_gcd_matches_euclid_across_sizes() raises:
    var state = 987654321
    var sizes: List[Int] = [40, 64, 100, 128, 200, 256, 300, 600, 1024, 2000]
    for trial in range(80):
        var values = List[Integer]()
        for which in range(3):
            var bits = sizes[(trial * 3 + which * 7) % len(sizes)]
            var value = Integer(1)
            while value.magnitude_bit_length() < bits:
                state = (state * 1103515245 + 12345) % 2147483648
                value = (value << 31) + state
            values.append(value >> (value.magnitude_bit_length() - bits))
        var shared = values[2]
        var x = values[0] * shared if trial % 2 else values[0]
        var y = values[1] * shared if trial % 3 else values[1]
        if trial % 5 == 0:
            x = -x
        assert_equal(gcd(x, y), _euclid_gcd(x, y))
        assert_equal(gcd(y, x), _euclid_gcd(x, y))
        # Floor square roots and long division on the same operands.
        var square = abs(x * y) + trial
        var root = isqrt(square)
        assert_true(root * root <= square and (root + 1) * (root + 1) > square)
        var quotient, remainder = div_rem_trunc(x, y)
        assert_equal(quotient * y + remainder, x)
        assert_true(abs(remainder) < abs(y) and (not remainder or remainder.sign() == x.sign()))
    assert_equal(gcd(Integer(0), Integer(1) << 300), Integer(1) << 300)
    assert_equal(gcd((Integer(1) << 300) * 3, Integer(0)), (Integer(1) << 300) * 3)


def test_lehmer_gcd_edge_cases() raises:
    # Consecutive Fibonacci numbers make every quotient 1, the longest run of
    # small steps; with a planted factor the gcd spans several limbs.
    var f = Integer(0)
    var g = Integer(1)
    var factor = (Integer(1) << 200) + 12345
    for k in range(1, 3001):
        var following = f + g
        f = g
        g = following
        if k in [64, 93, 94, 127, 128, 200, 500, 1000, 1500, 3000]:
            assert_equal(gcd(g, f), Integer(1))
            assert_equal(gcd(f * factor, g * factor), factor)
    var state = Integer(0x2545F4914F6CDD1D)
    for bits in [65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 500, 1024, 2048, 4096, 5000]:
        state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 5000)
        var x = (state >> (5000 - bits)) | (Integer(1) << (bits - 1))
        state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 5000)
        var y = (state >> (5000 - bits + 3)) | 1
        # A multiple, equal and nearly equal operands, powers of two.
        assert_equal(gcd(x, x * 7), x)
        assert_equal(gcd(x * 7, x), x)
        assert_equal(gcd(x, x), x)
        assert_equal(gcd(x, x + 1), Integer(1))
        assert_equal(gcd(x * 2, x * 2 + 6), _euclid_gcd(x * 2, x * 2 + 6))
        assert_equal(gcd(x << 70, y << 33), _euclid_gcd(x, y) << 33)
        # Random pairs with and without a large common factor, any signs.
        assert_equal(gcd(x, y), _euclid_gcd(x, y))
        assert_equal(gcd(-x * factor, y * factor), _euclid_gcd(x, y) * factor)
        assert_equal(gcd(x * y, y), y)
        # Operands far apart in size.
        assert_equal(gcd(x << 300, y), _euclid_gcd(x << 300, y))


def test_two_limb_gcd_edge_cases() raises:
    # Stein's steps on two limbs: shared and unshared powers of two.
    for i in [0, 1, 2, 63, 64, 65, 100, 117]:
        for j in [0, 1, 64, 117]:
            assert_equal(gcd(Integer(591) << i, Integer(753) << j), Integer(3) << min(i, j))
    # Differences whose low limb is zero, and two-limb results.
    var x = (Integer(1) << 127) + 12345
    for gap in [Integer(1) << 64, Integer(3) << 64, Integer(5) << 70]:
        assert_equal(gcd(x, x - gap), _euclid_gcd(x, x - gap))
    var g = (Integer(1) << 70) + 31
    assert_equal(gcd(g * 101, g * 103), g)
    assert_equal(gcd(g << 20, (g * 3) << 5), g << 5)
    # The 128-bit limit, results above Int64.MAX, and every storage form:
    # inline UInt64, a static literal, shared words.
    var top = (Integer(1) << 128) - 1
    var wide = Integer(UInt64(0xFFFFFFFFFFFFFFFF))
    assert_equal(gcd(top, wide), wide)
    assert_equal(gcd(Integer(340282366920938463463374607431768211455), wide * 3), wide)
    assert_equal(gcd(top, (Integer(1) << 64) + 1), (Integer(1) << 64) + 1)
    assert_equal(gcd(top, top - 2), Integer(1))
    assert_equal(gcd(top, Integer(1) << 127), Integer(1))
    assert_equal(gcd(Integer(1) << 127, Integer(1) << 64), Integer(1) << 64)
    assert_equal(gcd(Integer(1) << 127, Integer(1) << 127), Integer(1) << 127)
    # Gaps either side of the 16 bits that take a division first, and zero,
    # one and one-limb operands with signs.
    var big = (Integer(1) << 127) + 0x123456789
    for shift in [14, 15, 16, 17, 18, 63, 64, 65]:
        var small = (big >> shift) | 1
        assert_equal(gcd(big, small), _euclid_gcd(big, small))
        assert_equal(gcd(-small, big), _euclid_gcd(big, small))
    assert_equal(gcd(big, Integer(0)), big)
    assert_equal(gcd(Integer(0), -big), big)
    assert_equal(gcd(big, Integer(1)), Integer(1))
    assert_equal(gcd(Integer(6), (Integer(1) << 100) * 9), Integer(6))
    # Random pairs of up to two limbs, with and without a common factor.
    var state = Integer(0x2545F4914F6CDD1D)
    for trial in range(64):
        state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 128)
        var a = state >> (trial % 50)
        state = (state * 6364136223846793005 + 1442695040888963407) % (Integer(1) << 128)
        var b = state >> (trial % 37 + 30)
        assert_equal(gcd(a, b), _euclid_gcd(a, b))
        var factor = Integer(trial * 2 + 1) << (trial % 7)
        assert_equal(gcd((a >> 20) * factor, (b >> 20) * factor), _euclid_gcd(a >> 20, b >> 20) * factor)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
