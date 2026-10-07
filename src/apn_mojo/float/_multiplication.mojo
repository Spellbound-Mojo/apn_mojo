"""Finite binary products: borrowed significant limbs and one final rounding."""

from std.bit import count_leading_zeros, count_trailing_zeros
from ..integer.value import Integer
from ..integer._multiplication import (
    _multiply_with_scratch,
    _product_scratch_size,
    _high_product_into,
    _load_pair,
    _put_pair,
    _multiply_short_into,
)
from ..common._sizes import _checked_count, _checked_sum
from .context import ArithmeticContext, RoundingMode
from .status import NumericStatus
from ._rounding import (
    _RoundedBinary,
    _exponent_add,
    _finish_round,
    _overflow_result,
    _round_away,
    _rounded_direction,
    _round_native,
    _round_wide,
    _short_magnitude,
)


def _product_rounding_decision(
    digits: Span[mut=False, UInt32, _], discard: Int,
    mode: RoundingMode, negative: Bool,
) -> Tuple[Bool, Bool]:
    if discard <= 0:
        return False, False
    var index = (discard - 1) // 32
    var bit = UInt32((discard - 1) % 32)
    var word = digits.unsafe_get(index)
    var round_bit = Bool(word & (UInt32(1) << bit))
    var sticky = Bool(word & ((UInt32(1) << bit) - 1))
    if not sticky:
        for i in range(index):
            if digits.unsafe_get(i) != 0:
                sticky = True
                break
    var inexact = round_bit or sticky
    var up = inexact and _round_away(mode, negative)
    if mode == RoundingMode.nearest_even:
        var odd = Bool(digits.unsafe_get(discard // 32) & (UInt32(1) << UInt32(discard % 32)))
        up = round_bit and (sticky or odd)
    return inexact, up


def _shift_product_down(
    digits: Span[mut=True, UInt32, _], used: Int, count: Int, discard: Int,
):
    var whole = discard // 32
    var part = UInt64(discard % 32)
    var i = 0
    # Read each pair and its incoming bits before overwriting the lower words.
    # The final one or two words handle a possibly absent upper source word.
    if part:
        while i + 2 < count:
            var low = _load_pair(digits, i + whole)
            var high = UInt64(digits.unsafe_get(i + whole + 2))
            _put_pair(digits, i, (low >> part) | (high << (UInt64(64) - part)))
            i += 2
        while i < count:
            var low = UInt64(digits.unsafe_get(i + whole))
            var high = UInt64(digits.unsafe_get(i + whole + 1)) if i + whole + 1 < used else UInt64(0)
            digits.unsafe_get(i) = UInt32((low >> part) | (high << (UInt64(32) - part)))
            i += 1
    else:
        while i + 1 < count:
            _put_pair(digits, i, _load_pair(digits, i + whole))
            i += 2
        if i < count:
            digits.unsafe_get(i) = digits.unsafe_get(i + whole)


def _first_product_word(digits: Span[mut=False, UInt32, _]) -> Int:
    # Finite nonzero significands can have long exact zero-padded tails.
    # Scan groups only after the common nonzero-low-word check.
    if digits.unsafe_get(0) != 0:
        return 0
    var first = 1
    while first + 8 < len(digits):
        var group = (_load_pair(digits, first) | _load_pair(digits, first + 2)
                     | _load_pair(digits, first + 4) | _load_pair(digits, first + 6))
        if group:
            break
        first += 8
    while digits.unsafe_get(first) == 0:
        first += 1
    return first


def _try_high_product(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    negative: Bool, scale: Int128, context: ArithmeticContext, fail: Bool,
    mut result: _RoundedBinary,
) raises -> Bool:
    result = _RoundedBinary(0, False, Integer(0), 0, context.format(), NumericStatus())
    var abits = (len(a) - 1) * 32 + 32 - Int(count_leading_zeros(a.unsafe_get(len(a) - 1)))
    var bbits = (len(b) - 1) * 32 + 32 - Int(count_leading_zeros(b.unsafe_get(len(b) - 1)))
    var omitted_bits = Int128(abits) + Int128(bbits) - Int128(context.format().precision()) - 128
    if omitted_bits < 64:
        return False
    var omitted = Int(omitted_bits // 64)
    var n = (len(a) + 1) // 2
    var m = (len(b) + 1) // 2
    var capacity = _checked_count(_checked_sum(_checked_sum(n, m) - omitted, 1), 8) * 2
    var owner = Integer._Shared.uninitialized(capacity, False)
    var words = owner[].words.span()
    var used = _high_product_into(a, b, omitted, words)
    if not used:
        return False
    var bits = (used - 1) * 32 + 32 - Int(count_leading_zeros(words[used - 1]))
    var adjusted_scale = _exponent_add(scale, Int128(omitted) * 64)
    var exponent = _exponent_add(adjusted_scale, Int128(bits))
    if exponent < Int128(context.format().emin()) or exponent > Int128(context.format().emax()):
        return False
    var discard = bits - context.format().precision()
    if discard <= 64:
        return False
    var lower = _product_rounding_decision(words, discard, context.rounding(), negative)
    if not lower[0]:
        return False
    # Add the proven error bound in place. Reject if it reaches any retained
    # bit, or changes the rounding decision. A certified interval lies strictly
    # inside one rounding region, proving inexactness and its direction too.
    var addition = UInt64(min(n, m))
    var i = 2
    var boundary = discard // 32
    var part = UInt32(discard % 32)
    while addition:
        var old = words[i]
        var total = UInt64(old) + UInt64(UInt32(addition))
        words[i] = UInt32(total)
        if i > boundary or (i == boundary and (old >> part) != (words[i] >> part)):
            return False
        addition = (addition >> 32) + (total >> 32)
        i += 1
    var upper = _product_rounding_decision(words, discard, context.rounding(), negative)
    if not upper[0] or lower[1] != upper[1]:
        return False
    result = _round_product_owner(owner^, used, negative, adjusted_scale, context, fail)
    return True




@always_inline
def _round_product_owner(
    var owner: Integer._Shared, used: Int, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    # The caller preflights initialized storage for both product and output.
    var digits = owner[].words.span()
    var format = context.format()
    var p = format.precision()
    var bits = (used - 1) * 32 + 32 - Int(count_leading_zeros(digits.unsafe_get(used - 1)))
    if format._is_exact():
        # Exact working results keep every significant bit (trailing zero
        # bits carry no information and are dropped).
        var low = 0
        while digits.unsafe_get(low) == 0:
            low += 1
        p = max(1, bits - low * 32 - Int(count_trailing_zeros(digits.unsafe_get(low))))
        format = format._with_precision(p)
    var exponent = _exponent_add(scale, Int128(bits))
    if exponent > Int128(format.emax()):
        return _finish_round(_overflow_result(negative, context), context, fail)
    if exponent < Int128(format.emin()):
        var up = _round_away(context.rounding(), negative)
        if context.rounding() == RoundingMode.nearest_even:
            up = False
            if exponent == Int128(format.emin()) - 1:
                var top = digits.unsafe_get(used - 1)
                up = Bool(top & (top - 1))
                for i in range(used - 1):
                    up |= digits.unsafe_get(i) != 0
        return _finish_round(_RoundedBinary(
            1 if up else 0, negative,
            (Integer(1) << (p - 1)) if up else Integer(0),
            format.emin() if up else 0, format,
            NumericStatus._make(3, _rounded_direction(negative, up)),
        ), context, fail)

    var count = (p + 31) // 32
    var discard = bits - p
    var inexact = False
    var up = False
    if discard > 0:
        var decision = _product_rounding_decision(digits, discard, context.rounding(), negative)
        inexact = decision[0]
        up = decision[1]
        _shift_product_down(digits, used, count, discard)
    elif discard < 0:
        var whole = (-discard) // 32
        var part = UInt64((-discard) % 32)
        for i in range(count - 1, -1, -1):
            var index = i - whole
            var low = UInt64(digits.unsafe_get(index)) if 0 <= index < used else UInt64(0)
            var high = UInt64(digits.unsafe_get(index - 1)) if 0 < index <= used else UInt64(0)
            digits.unsafe_get(i) = UInt32((low << part) | (high >> (UInt64(32) - part)))
    if up:
        var carry = UInt64(1)
        var i = 0
        while carry and i < count:
            var total = UInt64(digits.unsafe_get(i)) + carry
            digits.unsafe_get(i) = UInt32(total)
            carry = total >> 32
            i += 1
        var rounded_bits = (count - 1) * 32 + 32 - Int(count_leading_zeros(digits.unsafe_get(count - 1)))
        if carry or rounded_bits > p:
            for i in range(count):
                digits.unsafe_get(i) = 0
            digits.unsafe_get(count - 1) = UInt32(1) << UInt32((p - 1) % 32)
            exponent = _exponent_add(exponent, Int128(1))
    if exponent > Int128(format.emax()):
        return _finish_round(_overflow_result(negative, context), context, fail)
    return _finish_round(_RoundedBinary(
        1, negative, Integer._from_product(owner^, count), Int(exponent), format,
        NumericStatus._make(Int(inexact), _rounded_direction(negative, up) if inexact else 0),
    ), context, fail)


@always_inline
def _binary_product(
    left: Integer, right: Integer, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool, *, path: Int = 0,
) raises -> _RoundedBinary:
    # Short significands multiply natively here; the rest in _binary_product_general.
    context.rounding()._validate()
    if (
        path == 0 and context.format().precision() <= 63 and not context.format()._is_exact()
        and left._storage.isa[Int64]() and right._storage.isa[Int64]()
    ):
        # Short significands: the exact product fits a 128-bit native integer.
        var product = UInt128(UInt64(left._storage[Int64])) * UInt128(UInt64(right._storage[Int64]))
        var native = _round_native(product, negative, scale, context, fail)
        if native.kind >= 0:
            return native^
    elif path == 0 and context.format().precision() <= 128 and not context.format()._is_exact():
        # Significands up to 128 bits: the product fits a 256-bit native integer.
        var x = _short_magnitude(left)
        var y = _short_magnitude(right)
        if x and y:
            var wide = _round_wide(UInt256(x.value()) * UInt256(y.value()), negative, scale, context, fail)
            if wide.kind >= 0:
                return wide^
    return _binary_product_general(left, right, negative, scale, context, fail, path=path)


@always_inline
def _binary_product_general(
    left: Integer, right: Integer, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool, *, path: Int = 0,
) raises -> _RoundedBinary:
    # Both magnitudes are nonzero, with denominator one. Removing whole zero
    # limbs is exact and leaves the original values and their lifetimes intact.
    context.rounding()._validate()
    var small_a = left._inline_words()
    var small_b = right._inline_words()
    var a = left._words_span(small_a)
    var b = right._words_span(small_b)
    var first_a = _first_product_word(a)
    var first_b = _first_product_word(b)
    var x = a[first_a:]
    var y = b[first_b:]
    var adjusted_scale = _exponent_add(scale, 32 * (Int128(first_a) + Int128(first_b)))
    var product_count = _checked_sum(len(x), len(y))
    var output_count = _checked_sum(context.format().precision(), 31) // 32
    var capacity = _checked_count(max(product_count, output_count), 4)
    var same = Int(x.unsafe_ptr()) == Int(y.unsafe_ptr()) and len(x) == len(y)
    # Measured crossover: avoid certificate overhead for short operands, and
    # retain subquadratic full products above the qualified triangular range.
    var shorter = min(len(x), len(y))
    var longer = max(len(x), len(y))
    # The high-half product only rounds; exact working formats need it all.
    if path == 2 or (path == 0 and not same and shorter >= 64 and longer <= 256
                    and shorter >= longer // 2 and output_count <= longer
                    and not context.format()._is_exact()):
        var rounded = _RoundedBinary(0, False, Integer(0), 0, context.format(), NumericStatus())
        if _try_high_product(x, y, negative, adjusted_scale, context, fail, rounded):
            return rounded^
    var scratch_size = _product_scratch_size(len(x), len(y), same)
    var owner = Integer._Shared.uninitialized(capacity, False)
    var used: Int
    if shorter <= 2 and path != 1:
        if len(y) <= 2:
            used = _multiply_short_into(x, y, owner[].words.span())
        else:
            used = _multiply_short_into(y, x, owner[].words.span())
    else:
        used = _multiply_with_scratch(x, y, owner[].words.span(), same, scratch_size)
    return _round_product_owner(owner^, used, negative, adjusted_scale, context, fail)
