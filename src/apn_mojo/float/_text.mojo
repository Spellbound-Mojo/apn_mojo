"""Exact radix output without exponent-sized or copied significand buffers."""

from ..integer.value import Integer
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from ._rounding import _RoundedBinary


def _float_trailing_zeros(significand: Integer) -> Int:
    var zeros = 0
    while not significand._word(zeros // 32):
        zeros += 32
    var low = significand._word(zeros // 32)
    while not (low & 1):
        zeros += 1
        low >>= 1
    return zeros


def _write_float_exact(
    value: _RoundedBinary, base: Int, mut writer: Some[Writer]
):
    if value.kind == 3:
        writer.write("nan")
        return
    if value.negative:
        writer.write("-")
    if value.kind == 0:
        writer.write("0")
    elif value.kind == 2:
        writer.write("inf")
    else:
        var zeros = _float_trailing_zeros(value.significand)
        var bits = value.format.precision() - zeros
        var width = 4 if base == 16 else 1
        comptime digits: StaticString = "0123456789abcdef"
        writer.write("0x" if base == 16 else "0b")
        for i in range((bits - 1) // width, -1, -1):
            var offset = zeros + i * width
            var word = UInt64(value.significand._word(offset // 32))
            if offset // 32 + 1 < value.significand._word_count():
                word |= UInt64(value.significand._word(offset // 32 + 1)) << 32
            writer.write(
                digits[
                    byte=Int(
                        (word >> UInt64(offset % 32)) & UInt64((1 << width) - 1)
                    )
                ]
            )
        writer.write(
            "p",
            Int128(value.exponent)
            - Int128(value.format.precision())
            + Int128(zeros),
        )


def _format_float_exact(
    value: _RoundedBinary,
    base: Int,
    mut budget: _ConversionBudget,
    *,
    count_value: Bool = True,
) raises -> String:
    if base != 2 and base != 16:
        raise Error(
            "Cannot format Float in this base; choose base=2 or base=16 for"
            " exact output, or base=10 with an explicit digits count."
        )
    if count_value:
        budget.values(1)
    var count = 3 if value.kind >= 2 else 1
    var digits = Int(value.kind == 0)
    if value.kind == 1:
        var zeros = _float_trailing_zeros(value.significand)
        var bits = value.format.precision() - zeros
        var width = 4 if base == 16 else 1
        var exponent = (
            Int128(value.exponent)
            - Int128(value.format.precision())
            + Int128(zeros)
        )
        var magnitude = -exponent if exponent < 0 else exponent
        var exponent_digits = 1
        while magnitude >= 10:
            exponent_digits += 1
            magnitude //= 10
        digits = _checked_sum((bits - 1) // width + 1, exponent_digits)
        count = _checked_sum(digits, 3 + Int(exponent < 0))
    count = _checked_sum(count, Int(value.negative and value.kind != 3))
    budget.digits(digits)
    budget.output(count)
    var result = String()
    budget.grow_string(result, count)
    result.reserve_bytes(count)
    _write_float_exact(value, base, result)
    return result
