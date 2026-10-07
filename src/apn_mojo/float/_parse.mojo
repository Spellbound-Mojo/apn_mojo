"""Exact-source Float grammar, range classification and one budgeted rounding."""

from ..common._text import _text_bounds, _text_digit
from ..common.conversion import _ConversionBudget
from ..integer.value import Integer
from ..integer._conversion import _conversion_multiply, _conversion_power10
from ..integer._limits import _MAX_RESULT_BITS
from .context import ArithmeticContext
from .status import NumericStatus
from ._rounding import _RoundedBinary, _round_magnitude, _finish_round, _inexact_error
from ._bounds10 import _EXACT_POWER_BITS, _scaled_bounds


def _float_text_error(offset: Int, family: StaticString = "Float") raises:
    raise Error(
        String(
            "Cannot parse ",
            family,
            " at byte ",
            offset,
            (
                "; use decimal digits with an optional e exponent, 0x"
                " hexadecimal or 0b binary digits with a p exponent, or"
                " inf/nan. Enable allow_whitespace for surrounding ASCII spaces"
                " and allow_underscores for single separators between digits."
                " The destination is unchanged."
            ),
        )
    )


@fieldwise_init
struct _FloatText(ImplicitlyCopyable):
    var start: Int
    var end: Int
    var radix: Int
    var fractional: Int
    var significant: Int
    var exponent: Int128
    var negative: Bool
    var kind: Int


def _float_text(
    text: String,
    whitespace: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> _FloatText:
    var start, end = _text_bounds(text, whitespace)
    return _float_text_range(text, start, end, underscores, budget)


def _float_text_range(
    text: String,
    var start: Int,
    end: Int,
    underscores: Bool,
    mut budget: _ConversionBudget,
    family: StaticString = "Float",
) raises -> _FloatText:
    var bytes = text.as_bytes()
    if start == end:
        _float_text_error(start, family)
    var negative = bytes[start] == 45
    if negative or bytes[start] == 43:
        start += 1
    if start == end:
        _float_text_error(start, family)
    if end - start == 3:
        var a = bytes[start] | 32
        var b = bytes[start + 1] | 32
        var c = bytes[start + 2] | 32
        if (a == 105 and b == 110 and c == 102) or (
            a == 110 and b == 97 and c == 110
        ):
            var kind = 2 if a == 105 else 3
            return _FloatText(
                start, end, 10, 0, 0, 0, negative and kind != 3, kind
            )
    var radix = 10
    if end - start >= 2 and bytes[start] == 48:
        if (bytes[start + 1] | 32) == 120:
            radix = 16
            start += 2
        elif (bytes[start + 1] | 32) == 98:
            radix = 2
            start += 2
    var point = False
    var count = 0
    var fractional = 0
    var significant = 0
    var mantissa_end = end
    for i in range(start, end):
        var byte = bytes[i]
        if (byte | 32) == UInt8(101 if radix == 10 else 112):
            mantissa_end = i
            break
        if byte == 46 and not point:
            point = True
            continue
        if byte == 95 and underscores and i > start and i + 1 < end:
            var before = _text_digit(bytes[i - 1])
            var after = _text_digit(bytes[i + 1])
            if 0 <= before < radix and 0 <= after < radix:
                continue
        var digit = _text_digit(byte)
        if digit < 0 or digit >= radix:
            _float_text_error(i, family)
        budget.digits(1)
        count += 1
        fractional += Int(point)
        if digit or significant:
            significant += 1
    if not count:
        _float_text_error(start, family)
    if radix != 10 and mantissa_end == end:
        _float_text_error(end, family)
    var exponent = Int128(0)
    if mantissa_end < end:
        var index = mantissa_end + 1
        var exponent_negative = False
        if index < end and (bytes[index] == 43 or bytes[index] == 45):
            exponent_negative = bytes[index] == 45
            index += 1
        if index == end:
            _float_text_error(index, family)
        # Input lengths and target bounds fit Int. Beyond 2**80, even their
        # largest corrections cannot bring an exponent back into range.
        comptime cap = Int128(1) << 80
        for i in range(index, end):
            var byte = bytes[i]
            if byte == 95 and underscores and i > index and i + 1 < end:
                if (
                    bytes[i - 1] >= 48
                    and bytes[i - 1] <= 57
                    and bytes[i + 1] >= 48
                    and bytes[i + 1] <= 57
                ):
                    continue
            if byte < 48 or byte > 57:
                _float_text_error(i, family)
            budget.digits(1)
            exponent = min(cap, exponent * 10 + Int128(byte - 48))
        if exponent_negative:
            exponent = -exponent
    return _FloatText(
        start,
        mantissa_end,
        radix,
        fractional,
        significant,
        exponent,
        negative,
        1 if significant else 0,
    )


def _text_power10(
    exponent: Int128, mut budget: _ConversionBudget
) raises -> Integer:
    if exponent < 0 or exponent > Int128(_MAX_RESULT_BITS // 4):
        raise Error(
            "Cannot convert decimal text: the exact power exceeds addressable"
            " storage; use a smaller exponent, or hexadecimal text/JSON for a"
            " compact binary value. The destination is"
            " unchanged."
        )
    return _conversion_power10(Int(exponent), budget)


def _parse_float(
    text: String,
    context: ArithmeticContext,
    whitespace: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> _RoundedBinary:
    context.rounding()._validate()
    budget.input(text.byte_length())
    budget.values(1)
    var form = _float_text(text, whitespace, underscores, budget)
    return _parse_float_form(text, form, context, budget)


def _parse_float_form(
    text: String,
    form: _FloatText,
    context: ArithmeticContext,
    mut budget: _ConversionBudget,
) raises -> _RoundedBinary:
    if form.kind != 1:
        return _finish_round(
            _RoundedBinary(
                form.kind,
                form.negative,
                Integer(0),
                0,
                context.format(),
                NumericStatus._make(0, 2 if form.kind == 3 else 0),
            ),
            context,
            False,
        )
    var scale = form.exponent - Int128(form.fractional) * Int128(
        4 if form.radix == 16 else 1
    )
    var lower: Int128
    var upper: Int128
    if form.radix == 10:
        var order = scale + Int128(form.significant) - 1
        # 10**order <= |x| < 10**(order+1); 3 < log2(10) < 4.
        lower = order * Int128(3 if order >= 0 else 4)
        upper = (order + 1) * Int128(4 if order + 1 >= 0 else 3)
        if (
            lower < Int128(context.format().emax())
            and upper > Int128(context.format().emin()) - 2
        ):
            # The coarse test bounds order by native Int-sized format limits,
            # so these tighter certified log2(10) bounds cannot overflow Int128.
            comptime lo = Int128(3321928094887362)
            comptime hi = Int128(3321928094887363)
            comptime unit = Int128(1000000000000000)
            var a = order * (lo if order >= 0 else hi)
            var b = (order + 1) * (hi if order + 1 >= 0 else lo)
            lower = a // unit if a >= 0 else -((-a + unit - 1) // unit)
            upper = (b + unit - 1) // unit if b >= 0 else -((-b) // unit)
    else:
        var width = Int128(4 if form.radix == 16 else 1)
        lower = scale + (Int128(form.significant) - 1) * width
        upper = lower + width
    if lower >= Int128(context.format().emax()):
        return _round_magnitude[True](
            Integer(1),
            Integer(1),
            form.negative,
            context,
            budget,
            scale=Int128(context.format().emax()) + 2,
        )
    if upper <= Int128(context.format().emin()) - 2:
        return _round_magnitude[True](
            Integer(1),
            Integer(1),
            form.negative,
            context,
            budget,
            scale=Int128(context.format().emin()) - 4,
        )
    var numerator = Integer._parse_words(
        text,
        form.start,
        form.end,
        form.radix,
        False,
        budget,
        skip_decimal_point=True,
    )
    var denominator = Integer(1)
    if form.radix == 10 and scale != 0:
        var digits = numerator.magnitude_bit_length()
        if context.format()._is_exact():
            # 5**-scale > the digits cannot divide them: the value is not a
            # finite binary fraction, whatever the power's size.
            if scale < 0 and Int128(232) * (-scale) >= Int128(100) * Int128(digits):
                raise _inexact_error()
        elif Int128(10) * abs(scale) // 3 > Int128(_EXACT_POWER_BITS) + 4 * Int128(
            digits + context.format().precision()
        ):
            var bounded = _bounded_parse(numerator, Int(scale), form.negative, context, budget)
            if bounded:
                return bounded.value()
    if form.radix == 10:
        if scale > 0:
            numerator = _conversion_multiply(
                numerator, _text_power10(scale, budget), budget
            )
        elif scale < 0:
            denominator = _text_power10(-scale, budget)
        scale = 0
    return _round_magnitude[True](
        numerator, denominator, form.negative, context, budget, scale=scale
    )


def _bounded_parse(
    digits: Integer, scale: Int, negative: Bool, context: ArithmeticContext, mut budget: _ConversionBudget
) raises -> Optional[_RoundedBinary]:
    """`digits * 10**scale` rounded to the context from bounds (see _bounds10),
    without the exact power of ten; None when bounds at the largest working
    precision tried still round apart. Beyond the threshold the value lies on
    no rounding boundary of the format, so bounds that round to the same
    result with the same status decide it. Traps apply once, to that result."""
    var quiet = context
    quiet._traps = 0
    var precision = context.format().precision() + 64
    for _ in range(12):
        var bounds = _scaled_bounds(digits, 0, scale, precision, budget)
        var low = _round_magnitude[True](
            bounds[0].mantissa, Integer(1), negative, quiet, budget, scale=Int128(bounds[0].exponent)
        )
        var high = _round_magnitude[True](
            bounds[1].mantissa, Integer(1), negative, quiet, budget, scale=Int128(bounds[1].exponent)
        )
        if (
            low.kind == high.kind
            and low.exponent == high.exponent
            and low.significand == high.significand
            and low.status == high.status
        ):
            return _finish_round(low, context, False)
        precision *= 2
    return None
