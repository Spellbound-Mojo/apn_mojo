"""Allocation-free lexical validation for exact decimal and fraction input."""

from ..common._text import _text_bounds
from ..integer._limits import _MAX_RESULT_BITS
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget


def _rational_text_error(offset: Int) raises:
    raise Error(
        String(
            "Cannot parse Rational at byte ",
            offset,
            (
                "; use signed decimal integers, a fraction such as 3/4, or an"
                " exact decimal such as 1.25e-3. Enable allow_whitespace for"
                " surrounding ASCII spaces or allow_underscores for single"
                " underscores between digits. The destination is unchanged."
            ),
        )
    )


def _rational_scale_error() raises:
    raise Error(
        "Cannot parse Rational: decimal exponent requires unaddressable"
        " storage; use a smaller exponent. The destination is unchanged."
    )


@fieldwise_init
struct _RationalText(ImplicitlyCopyable):
    var start: Int
    var end: Int
    var denominator_start: Int
    var denominator_end: Int
    var exponent_start: Int
    var exponent_end: Int
    var fractional_digits: Int
    var negative: Bool
    var exponent_negative: Bool
    var zero: Bool

    def scale(self, text: String) raises -> Int:
        # A syntactically valid zero needs no power, even for a huge exponent.
        if self.zero:
            return 0
        var exponent = 0
        for i in range(self.exponent_start, self.exponent_end):
            var byte = text.as_bytes()[i]
            if byte == 95:
                continue
            var digit = Int(byte) - 48
            if exponent > (Int.MAX - digit) // 10:
                _rational_scale_error()
            exponent = exponent * 10 + digit
        var scale = exponent - self.fractional_digits
        if self.exponent_negative:
            scale = -_checked_sum(exponent, self.fractional_digits)
        if abs(scale) > _MAX_RESULT_BITS // 4:
            _rational_scale_error()
        return scale


def _rational_digits(
    text: String,
    start: Int,
    end: Int,
    point: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> Tuple[Int, Bool]:
    var bytes = text.as_bytes()
    var seen_point = False
    var count = 0
    var fractional = 0
    var zero = True
    for i in range(start, end):
        var byte = bytes[i]
        if byte == 46 and point and not seen_point:
            seen_point = True
            continue
        if byte == 95 and underscores and i > start and i + 1 < end:
            if (
                bytes[i - 1] >= 48
                and bytes[i - 1] <= 57
                and bytes[i + 1] >= 48
                and bytes[i + 1] <= 57
            ):
                continue
        if byte < 48 or byte > 57:
            _rational_text_error(i)
        budget.digits(1)
        count += 1
        fractional += Int(seen_point)
        zero &= byte == 48
    if not count:
        _rational_text_error(start)
    return (fractional, zero)


def _rational_text(
    text: String,
    whitespace: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> _RationalText:
    var start, end = _text_bounds(text, whitespace)
    var bytes = text.as_bytes()
    if start == end:
        _rational_text_error(start)
    var negative = bytes[start] == 45
    if negative or bytes[start] == 43:
        start += 1
    var slash = -1
    for i in range(start, end):
        if bytes[i] == 47:
            if slash >= 0:
                _rational_text_error(i)
            slash = i
    if slash >= 0:
        var _, zero = _rational_digits(
            text, start, slash, False, underscores, budget
        )
        var denominator = slash + 1
        if denominator < end and (
            bytes[denominator] == 45 or bytes[denominator] == 43
        ):
            negative ^= bytes[denominator] == 45
            denominator += 1
        var _, denominator_zero = _rational_digits(
            text, denominator, end, False, underscores, budget
        )
        if denominator_zero:
            raise Error(
                "Cannot parse Rational: denominator is 0; use a nonzero"
                " denominator. The destination is unchanged."
            )
        return _RationalText(
            start, slash, denominator, end, end, end, 0, negative, False, zero
        )
    var marker = end
    for i in range(start, end):
        if bytes[i] == 101 or bytes[i] == 69:
            marker = i
            break
    var fractional, zero = _rational_digits(
        text, start, marker, True, underscores, budget
    )
    var exponent = marker
    var exponent_negative = False
    if marker < end:
        exponent += 1
        if exponent < end and (bytes[exponent] == 45 or bytes[exponent] == 43):
            exponent_negative = bytes[exponent] == 45
            exponent += 1
        _ = _rational_digits(text, exponent, end, False, underscores, budget)
    return _RationalText(
        start,
        marker,
        -1,
        -1,
        exponent,
        end,
        fractional,
        negative,
        exponent_negative,
        zero,
    )
