"""Budgeted preparation around the shared allocation-free Integer kernels.

These conversion-only helpers operate on nonnegative magnitudes. All new
buffers and final owners are charged before allocation, including gcd work.
"""

from .value import Integer
from ..common.conversion import _ConversionBudget
from ..common._sizes import _checked_count, _checked_sum
from ._division import _divide_into
from ._scalar_update import _multiply_into, _shift_into, _sum_into
from ._limits import _MAX_RESULT_BITS


def _conversion_words(
    value: Integer, mut budget: _ConversionBudget
) raises -> List[UInt32]:
    budget.allocate(value._word_count(), 4)
    return value._words_copy()


def _conversion_sign(
    value: Integer, negative: Bool, mut budget: _ConversionBudget
) raises -> Integer:
    if not negative or not value:
        return value
    var words = _conversion_words(value, budget)
    return Integer._from_budgeted_words(words^, True, budget)


def _conversion_div_rem(
    a: Integer, b: Integer, mut budget: _ConversionBudget
) raises -> Tuple[Integer, Integer]:
    # Internal callers establish a >= 0 and b > 0.
    var order = a._compare_magnitude(b)
    if order < 0:
        return (Integer(0), a)
    if order == 0:
        return (Integer(1), Integer(0))
    if b == 1:
        return (a, Integer(0))
    var size = _checked_count(_checked_sum(a._word_count(), 1), 4)
    var count = _checked_count(a._word_count() - b._word_count() + 1, 4)
    budget.allocate(size, 4)
    var dividend = List[UInt32](length=size, fill=0)
    for i in range(a._word_count()):
        dividend[i] = a._word(i)
    var divisor = _conversion_words(b, budget)
    budget.allocate(count, 4)
    var quotient = List[UInt32](length=count, fill=0)
    var used = _divide_into(Span(dividend), Span(divisor), Span(quotient))
    dividend.resize(used, 0)
    var q = Integer._from_budgeted_words(quotient^, False, budget)
    var r = Integer._from_budgeted_words(dividend^, False, budget)
    return (q, r)


def _conversion_gcd(
    var a: Integer, var b: Integer, mut budget: _ConversionBudget
) raises -> Integer:
    while b:
        var _, remainder = _conversion_div_rem(a, b, budget)
        a = b
        b = remainder
    return a


def _conversion_multiply(
    a: Integer, b: Integer, mut budget: _ConversionBudget
) raises -> Integer:
    if a == 1:
        return b
    if b == 1:
        return a
    var count = _checked_count(
        _checked_sum(a._word_count(), b._word_count()), 4
    )
    var left = _conversion_words(a, budget)
    var right = _conversion_words(b, budget)
    budget.allocate(count, 4)
    var words = List[UInt32](length=count, fill=0)
    _ = _multiply_into(Span(left), Span(right), Span(words))
    return Integer._from_budgeted_words(words^, False, budget)


def _conversion_power10(
    var exponent: Int, mut budget: _ConversionBudget
) raises -> Integer:
    var result = Integer(1)
    var factor = Integer(10)
    while exponent:
        if exponent & 1:
            result = _conversion_multiply(result, factor, budget)
        exponent >>= 1
        if exponent:
            factor = _conversion_multiply(factor, factor, budget)
    return result


def _conversion_shift(
    value: Integer, shift: Int, right: Bool, mut budget: _ConversionBudget
) raises -> Integer:
    if not value or not shift:
        return value
    if right and shift >= value.magnitude_bit_length():
        return Integer(0)
    if shift < 0 or (
        not right and shift > _MAX_RESULT_BITS - value.magnitude_bit_length()
    ):
        raise Error(
            "Cannot convert: shifted value exceeds addressable storage; use a"
            " smaller exponent or exact hexadecimal output. The destination is"
            " unchanged."
        )
    var count = _checked_count(
        _checked_sum(
            value._word_count()
            - shift
            // 32 if right else _checked_sum(value._word_count(), shift // 32),
            1,
        ),
        4,
    )
    var source = _conversion_words(value, budget)
    budget.allocate(count, 4)
    var words = List[UInt32](length=count, fill=0)
    _shift_into(Span(source), False, shift, right, Span(words))
    return Integer._from_budgeted_words(words^, False, budget)


def _conversion_sum(
    a: Integer, b: Integer, subtract: Bool, mut budget: _ConversionBudget
) raises -> Integer:
    # Magnitudes only; subtraction callers establish a >= b.
    var count = _checked_count(
        _checked_sum(max(a._word_count(), b._word_count()), 1), 4
    )
    var left = _conversion_words(a, budget)
    var right = _conversion_words(b, budget)
    budget.allocate(count, 4)
    var words = List[UInt32](length=count, fill=0)
    _ = _sum_into(Span(left), Span(right), False, subtract, Span(words))
    return Integer._from_budgeted_words(words^, False, budget)
