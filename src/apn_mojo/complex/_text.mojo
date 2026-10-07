"""Complex grammar over shared Float token spans and one conversion budget."""

from ..common._text import _text_bounds, _text_space
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from ..integer.value import Integer
from ..float._parse import _FloatText, _float_text_range, _parse_float_form
from ..float._text import _format_float_exact
from ..float._rounding import _RoundedBinary, _round_magnitude, _finish_round
from .context import ComplexContext, _ComplexContextArgument


def _complex_text_error(offset: Int) raises:
    raise Error(
        String(
            "Cannot parse Complex at byte ",
            offset,
            (
                "; use a real number, a+bj (or a+bi), or Complex(real, imag)."
                " Use decimal, hexadecimal or binary Float components; enable"
                " allow_whitespace for surrounding/component spaces. The"
                " destination is unchanged."
            ),
        )
    )


def _component_bounds(
    text: String, var start: Int, var end: Int, whitespace: Bool
) -> Tuple[Int, Int]:
    if whitespace:
        var bytes = text.as_bytes()
        while start < end and _text_space(bytes[start]):
            start += 1
        while end > start and _text_space(bytes[end - 1]):
            end -= 1
    return start, end


def _complex_text(
    text: String,
    whitespace: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
) raises -> Tuple[_FloatText, _FloatText, Bool]:
    var start, end = _text_bounds(text, whitespace)
    var bytes = text.as_bytes()
    if start == end:
        _complex_text_error(start)
    comptime prefix: StaticString = "Complex("
    var named = end - start >= prefix.byte_length()
    if named:
        for i in range(prefix.byte_length()):
            if bytes[start + i] != prefix.as_bytes()[i]:
                named = False
                break
    var wrapped = named or bytes[start] == 40
    if wrapped:
        if bytes[end - 1] != 41:
            _complex_text_error(end)
        start += prefix.byte_length() if named else 1
        end -= 1
        start, end = _component_bounds(text, start, end, whitespace)
    if start == end:
        _complex_text_error(start)
    var comma = -1
    for i in range(start, end):
        if bytes[i] == 44:
            if not wrapped or comma != -1:
                _complex_text_error(i)
            comma = i
    var zero = _FloatText(0, 0, 10, 0, 0, 0, False, 0)
    if comma != -1:
        var a_start, a_end = _component_bounds(text, start, comma, whitespace)
        var b_start = comma + 1
        # The ordinary Writable spelling includes one structural space.
        if b_start < end and bytes[b_start] == 32:
            b_start += 1
        var b_end = end
        b_start, b_end = _component_bounds(text, b_start, b_end, whitespace)
        var a = _float_text_range(
            text, a_start, a_end, underscores, budget, "Complex real component"
        )
        var b = _float_text_range(
            text,
            b_start,
            b_end,
            underscores,
            budget,
            "Complex imaginary component",
        )
        return a, b, False
    if named:
        _complex_text_error(end)
    if bytes[end - 1] != 105 and bytes[end - 1] != 106:
        return (
            _float_text_range(
                text, start, end, underscores, budget, "Complex real component"
            ),
            zero,
            False,
        )
    end -= 1
    var separator = -1
    for i in range(start + 1, end):
        if (
            (bytes[i] == 43 or bytes[i] == 45)
            and (bytes[i - 1] | 32) != 101
            and (bytes[i - 1] | 32) != 112
        ):
            if separator != -1:
                _complex_text_error(i)
            separator = i
    var a = zero
    var negative = False
    if separator != -1:
        var a_start, a_end = _component_bounds(
            text, start, separator, whitespace
        )
        a = _float_text_range(
            text, a_start, a_end, underscores, budget, "Complex real component"
        )
        negative = bytes[separator] == 45
        start = separator + 1
    start, end = _component_bounds(text, start, end, whitespace)
    if (
        separator != -1
        and start < end
        and (bytes[start] == 43 or bytes[start] == 45)
    ):
        _complex_text_error(start)
    if (
        separator == -1
        and end - start == 1
        and (bytes[start] == 43 or bytes[start] == 45)
    ):
        negative = bytes[start] == 45
        start += 1
    if start == end:
        var unit = zero
        unit.negative = negative
        return a, unit, True
    var b = _float_text_range(
        text, start, end, underscores, budget, "Complex imaginary component"
    )
    if separator != -1 and b.kind != 3:
        b.negative = negative
    return a, b, False


def _parse_complex(
    text: String,
    context: _ComplexContextArgument,
    whitespace: Bool,
    underscores: Bool,
    mut budget: _ConversionBudget,
    *,
    fail_component: Int = -1,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var target = context.value() if context else ComplexContext()
    budget.input(text.byte_length())
    budget.values(1)
    var forms = _complex_text(text, whitespace, underscores, budget)
    var real = _parse_float_form(text, forms[0], target.real(), budget)
    real = _finish_round(real, target.real(), fail_component == 0)
    var imag: _RoundedBinary
    if forms[2]:
        imag = _round_magnitude[True](
            Integer(1), Integer(1), forms[1].negative, target.imag(), budget
        )
    else:
        imag = _parse_float_form(text, forms[1], target.imag(), budget)
    imag = _finish_round(imag, target.imag(), fail_component == 1)
    return real, imag


def _format_complex_exact(
    real: _RoundedBinary,
    imag: _RoundedBinary,
    base: Int,
    mut budget: _ConversionBudget,
) raises -> String:
    if base != 2 and base != 16:
        raise Error(
            "Cannot format Complex in this base; choose base=2 or base=16 for"
            " exact component output. Use to_json() to preserve component"
            " formats too."
        )
    budget.values(1)
    budget.output(11)
    var a = _format_float_exact(real, base, budget, count_value=False)
    var b = _format_float_exact(imag, base, budget, count_value=False)
    var count = _checked_sum(11, _checked_sum(a.byte_length(), b.byte_length()))
    var result = String()
    budget.grow_string(result, count)
    result.reserve_bytes(count)
    result.write("Complex(", a, ", ", b, ")")
    return result
