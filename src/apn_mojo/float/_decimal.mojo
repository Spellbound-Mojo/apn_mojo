"""Explicitly rounded scientific decimal output from the exact stored dyadic."""

from ..integer.value import Integer
from ..integer._conversion import (
    _conversion_shift,
    _conversion_multiply,
    _conversion_div_rem,
    _conversion_sum,
)
from ..integer._limits import _MAX_RESULT_BITS
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from .context import RoundingMode
from ._rounding import _RoundedBinary, _round_away
from ._parse import _text_power10
from ._bounds10 import _EXACT_POWER_BITS, _floor_scaled


def _decimal_compare(
    a: Integer, b: Integer, exponent: Int128, mut budget: _ConversionBudget
) raises -> Int:
    if exponent >= 0:
        var power = _conversion_multiply(
            b, _text_power10(exponent, budget), budget
        )
        return a._compare_magnitude(power)
    var power = _conversion_multiply(
        a, _text_power10(-exponent, budget), budget
    )
    return power._compare_magnitude(b)


def _check_notation(notation: StaticString) raises:
    if notation != "auto" and notation != "positional" and notation != "scientific":
        raise Error(String(
            "Unknown notation '", notation, "'; choose auto, positional, scientific or",
            " hexadecimal. The destination is unchanged.",
        ))


def _layout_decimal(
    negative: Bool, digits: String, exponent: Int, notation: StaticString,
    mut budget: _ConversionBudget,
) raises -> String:
    """Write `d1.d2...dn * 10**exponent` for the digit string `d1d2...dn`,
    whose output the caller has counted: the layout counts the rest.

    Positional notation writes `0.00001` and `3.0`; scientific writes `1.5e-08`
    and `1e+21`, the exponent signed and of two digits at least. Auto is
    positional from 1e-6 up to but excluding 1e21, as JavaScript prints numbers,
    and scientific beyond.
    """
    _check_notation(notation)
    var n = digits.byte_length()
    var positional = notation == "positional" or (notation == "auto" and exponent >= -6 and exponent < 21)
    var point = exponent + 1
    var exponent_text = String(exponent if exponent >= 0 else -exponent)
    var count = Int(negative)
    if positional:
        count += n + 2 - point if point <= 0 else (point + 2 if point >= n else n + 1)
    else:
        count += n + Int(n > 1) + 2 + max(2, exponent_text.byte_length())
    budget.output(count - n)
    var result = String()
    budget.grow_string(result, count)
    result.reserve_bytes(count)
    if negative:
        result.write("-")
    if not positional:
        result.write(digits[byte=0])
        if n > 1:
            result.write(".")
            for i in range(1, n):
                result.write(digits[byte=i])
        result.write("e", "-" if exponent < 0 else "+")
        if exponent_text.byte_length() < 2:
            result.write("0")
        result.write(exponent_text)
    elif point <= 0:
        result.write("0.")
        for _ in range(-point):
            result.write("0")
        result.write(digits)
    elif point >= n:
        result.write(digits)
        for _ in range(point - n):
            result.write("0")
        result.write(".0")
    else:
        for i in range(n):
            if i == point:
                result.write(".")
            result.write(digits[byte=i])
    return result


def _format_float_decimal(
    value: _RoundedBinary,
    digits: Int,
    rounding: RoundingMode,
    mut budget: _ConversionBudget,
    notation: StaticString = "scientific",
) raises -> String:
    if digits < 1:
        raise Error(
            "Cannot format Float with fewer than one significant digit; choose"
            " digits >= 1. The destination is unchanged."
        )
    rounding._validate()
    budget.values(1)
    if value.kind >= 2:
        var count = 3 + Int(value.negative and value.kind != 3)
        budget.output(count)
        return String(
            "nan" if value.kind == 3 else "-inf" if value.negative else "inf"
        )
    # Reject impossible output limits before exact-ratio preparation; charge
    # final counts only once, after the exponent and rounding carry are known.
    # The shortest layout of any notation: the digits, a point and the sign
    # (positional `1.00`); scientific adds an exponent.
    var framing = 1 + Int(value.negative)
    var minimum = _checked_sum(digits, framing)
    var preview = budget
    preview.digits(_checked_sum(digits, 1))
    preview.output(minimum)
    var q = Integer(0)
    var exponent = Int128(0)
    if value.kind == 1:
        var bounded = _bounded_decimal(value, digits, rounding, budget)
        if bounded:
            q = bounded.value()[0]
            exponent = Int128(bounded.value()[1])
        else:
            var scale = Int128(value.exponent) - Int128(value.format.precision())
            if scale > Int128(_MAX_RESULT_BITS) or scale < -Int128(
                _MAX_RESULT_BITS
            ):
                raise Error(
                    "Cannot format this decimal value within addressable working"
                    " storage; use exact hexadecimal output or JSON, or scale the"
                    " value down. The destination is unchanged."
                )
            var a = value.significand
            var b = Integer(1)
            if scale >= 0:
                a = _conversion_shift(a, Int(scale), False, budget)
            else:
                b = _conversion_shift(b, Int(-scale), False, budget)
            # Integer-only log10 estimate. Exact comparisons correct its bounded
            # error before choosing the decimal rounding unit.
            var estimate = (Int128(value.exponent) - 1) * Int128(301029995663981195)
            comptime denominator = Int128(1000000000000000000)
            exponent = estimate // denominator if estimate >= 0 else -(
                (-estimate + denominator - 1) // denominator
            )
            while _decimal_compare(a, b, exponent, budget) < 0:
                exponent -= 1
            while _decimal_compare(a, b, exponent + 1, budget) >= 0:
                exponent += 1
            var shift = Int128(digits) - 1 - exponent
            if shift >= 0:
                a = _conversion_multiply(a, _text_power10(shift, budget), budget)
            else:
                b = _conversion_multiply(b, _text_power10(-shift, budget), budget)
            var remainder: Integer
            q, remainder = _conversion_div_rem(a, b, budget)
            if remainder != 0:
                var up = _round_away(rounding, value.negative)
                if rounding == RoundingMode.nearest_even:
                    var other = _conversion_sum(b, remainder, True, budget)
                    up = remainder > other or (
                        remainder == other and Bool(q._word(0) & 1)
                    )
                if up:
                    q = _conversion_sum(q, Integer(1), False, budget)
            var carry = _text_power10(Int128(digits), budget)
            if q == carry:
                q, _ = _conversion_div_rem(q, Integer(10), budget)
                exponent += 1
    budget.digits(String(exponent).byte_length() - Int(exponent < 0))
    var mantissa = String()
    if value.kind == 1:
        mantissa = q._format_with_budget(10, False, False, budget)
    else:
        budget.digits(digits)
        budget.output(digits)
        for _ in range(digits):
            mantissa.write("0")
    return _layout_decimal(value.negative, mantissa, Int(exponent), notation, budget)


def _bounded_decimal(
    value: _RoundedBinary, digits: Int, rounding: RoundingMode, mut budget: _ConversionBudget
) raises -> Optional[Tuple[Integer, Int]]:
    """`digits` rounded significant digits and the decimal exponent of a
    finite value whose exact powers of ten would be long, from bounds (see
    _bounds10); None for a value the exact path handles. Beyond the threshold
    neither the value over a power of ten nor twice it is ever an integer, so
    floors decide every rounding mode without a tie."""
    var precision = value.format.precision()
    var scale = value.exponent - precision
    if Int128(abs(scale)) <= Int128(_EXACT_POWER_BITS + 8 * precision) + 16 * Int128(digits):
        return None
    var estimate = (Int128(value.exponent) - 1) * Int128(301029995663981195)
    comptime denominator = Int128(1000000000000000000)
    var exponent = Int(
        estimate // denominator if estimate >= 0 else -((-estimate + denominator - 1) // denominator)
    )
    var working = precision + 64 + 4 * digits
    for _ in range(12):
        var found = _bounded_digits(value, scale, exponent, digits, rounding, working, budget)
        if found:
            return found
        working *= 2
    return None


def _bounded_digits(
    value: _RoundedBinary, scale: Int, estimate: Int, digits: Int, rounding: RoundingMode,
    precision: Int, mut budget: _ConversionBudget,
) raises -> Optional[Tuple[Integer, Int]]:
    """One try of _bounded_decimal at `precision` bits; None when the bounds
    do not decide."""
    var exponent = estimate
    # 10**exponent <= x < 10**(exponent + 1); the estimate is off by at most one.
    while True:
        var below = _floor_scaled(value.significand, scale, -exponent, precision, budget)
        if not below:
            return None
        if not below.value():
            exponent -= 1
            continue
        var above = _floor_scaled(value.significand, scale, -(exponent + 1), precision, budget)
        if not above:
            return None
        if above.value():
            exponent += 1
            continue
        break
    var unit = exponent - digits + 1
    var floor = _floor_scaled(value.significand, scale, -unit, precision, budget)
    var twice = _floor_scaled(value.significand << 1, scale, -unit, precision, budget)
    if not floor or not twice:
        return None
    var q = floor.value()
    var up = _round_away(rounding, value.negative)
    if rounding == RoundingMode.nearest_even:
        # The dropped part exceeds one half exactly when floor(2y) is odd.
        up = twice.value() - (q << 1) == 1
    if up:
        q += 1
    budget.allocate(digits // 8 + 4, 4)
    if q == Integer(10) ** digits:
        q = Integer(10) ** (digits - 1)
        exponent += 1
    return (q^, exponent)
