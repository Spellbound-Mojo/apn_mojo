"""Checked text grammar and size estimates for exact integer conversion."""

from ..common._sizes import _checked_count, _checked_sum
from ..common.conversion import _ConversionBudget
from ..common._text import _text_digit, _text_bounds


def _text_prefix(base: Int, uppercase: Bool) -> UInt8:
    return UInt8(
        (66 if uppercase else 98) if base
        == 2 else (79 if uppercase else 111) if base
        == 8 else (88 if uppercase else 120)
    )


def _format_bound(words: Int) raises -> Int:
    # Binary output is the longest supported representation; include -0x.
    return _checked_count(_checked_sum(_checked_count(words, 32) * 32, 3), 1)


def _integer_text(
    text: String,
    base: Int,
    allow_prefix: Bool,
    allow_whitespace: Bool,
    allow_underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> Tuple[Int, Int, Int, Bool]:
    if base != 0 and (base < 2 or base > 36):
        raise Error(
            String(
                "Cannot parse Integer in base ",
                base,
                "; choose a base from 2 through 36, or 0 to detect a prefix.",
            )
        )
    var start, end = _text_bounds(text, allow_whitespace)
    var bytes = text.as_bytes()
    if start == end:
        raise Error(
            "Cannot parse an empty Integer; provide at least one digit."
        )
    var negative = bytes[start] == 45
    if bytes[start] == 43 or negative:
        start += 1
    var radix = 10 if base == 0 else base
    if (allow_prefix or base == 0) and end - start >= 2 and bytes[start] == 48:
        var marker = bytes[start + 1]
        var detected = (
            2 if marker == 98
            or marker == 66 else 8 if marker == 111
            or marker == 79 else 16 if marker == 120
            or marker == 88 else 0
        )
        if detected:
            if base != 0 and base != detected:
                raise Error(
                    String(
                        "Cannot parse Integer: prefix does not match base ",
                        base,
                        (
                            "; use the matching base, base=0 for detection, or"
                            " disable prefix handling for bare digits."
                        ),
                    )
                )
            radix = detected
            start += 2
    if start == end:
        raise Error(
            "Cannot parse an Integer sign or prefix without digits; add at"
            " least one digit after the sign or prefix."
        )
    for i in range(start, end):
        if bytes[i] == 95 and allow_underscores:
            if i > start and i + 1 < end:
                var before = _text_digit(bytes[i - 1])
                var after = _text_digit(bytes[i + 1])
                if (
                    0 <= before
                    and before < radix
                    and 0 <= after
                    and after < radix
                ):
                    continue
            raise Error(
                String(
                    "Cannot parse Integer: misplaced underscore at byte ",
                    i,
                    (
                        "; put one underscore only between two digits valid for"
                        " the selected base."
                    ),
                )
            )
        var digit = _text_digit(bytes[i])
        if digit < 0 or digit >= radix:
            raise Error(
                String(
                    "Cannot parse Integer: invalid digit at byte ",
                    i,
                    " for base ",
                    radix,
                    (
                        "; use valid digits, base=0 for prefixes,"
                        " allow_whitespace=True for surrounding ASCII spaces,"
                        " or allow_underscores=True for underscores between"
                        " digits."
                    ),
                )
            )
        budget.digits(1)
    return (start, end, radix, negative)
