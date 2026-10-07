"""Version-1 Complex envelopes containing two canonical Float records."""

from ..common._json import _JSONReader
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from ..float._json import _read_float_json_record, _format_float_json, _Deferred
from ..float._rounding import _RoundedBinary
from ._format import _ComplexFormats


def _read_complex_json(
    text: String, mut budget: _ConversionBudget
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    budget.input(text.byte_length())
    budget.values(1)
    var reader = _JSONReader(
        text, budget, "Invalid complex interchange at byte "
    )
    var result = _read_complex_json_record(reader)
    reader.finish()
    budget = reader.budget
    return result


def _read_complex_json_record(
    mut reader: _JSONReader,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var inline = _Deferred(False)
    return _read_complex_json_record(reader, inline)


def _read_complex_json_record(
    mut reader: _JSONReader, mut deferred: _Deferred,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    """A Complex record. With deferral, adds the real then the imaginary
    significand text, whichever order the record lists them in."""
    var diagnostic = reader.diagnostic_prefix
    var imag_first = False
    reader.expect(123, "provide an object produced by Complex.to_json()")
    var seen = 0
    var real: Optional[_RoundedBinary] = None
    var imag: Optional[_RoundedBinary] = None
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "real":
            flag = 4
        elif key == "imag":
            flag = 8
        else:
            reader.fail(
                "unknown field",
                "use version, family, real and imag from Complex.to_json()",
            )
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one("use Complex interchange version 1")
        elif flag == 2:
            reader.family(
                "complex", "use family complex with Complex.from_json()"
            )
        else:
            reader.diagnostic_prefix = (
                "Invalid Complex real component at byte " if flag
                == 4 else "Invalid Complex imaginary component at byte "
            )
            var component = _read_float_json_record(reader, deferred)
            if flag == 4:
                real = component
            else:
                imag_first = not real
                imag = component
            reader.diagnostic_prefix = diagnostic
        if reader.end_object(
            "separate fields with commas and close the object with }"
        ):
            break
    reader.require_fields(seen, 15, "include version, family, real and imag")
    if deferred.enabled and imag_first:
        var last = len(deferred.texts) - 1
        var imag_text = deferred.texts[last - 1]
        deferred.texts[last - 1] = deferred.texts[last]
        deferred.texts[last] = imag_text^
    _ = _ComplexFormats(real.value().format, imag.value().format)
    return real.value(), imag.value()


def _format_complex_json(
    real: _RoundedBinary,
    imag: _RoundedBinary,
    mut budget: _ConversionBudget,
    element: Int = -1,
    *,
    count_value: Bool = True,
) raises -> String:
    comptime prefix: StaticString = '{"version":1,"family":"complex","real":'
    comptime separator: StaticString = ',"imag":'
    comptime overhead = prefix.byte_length() + separator.byte_length() + 1
    if count_value:
        budget.values(1, element)
    budget.output(overhead, element)
    var a = _format_float_json(real, budget, element, count_value=False)
    var b = _format_float_json(imag, budget, element, count_value=False)
    var count = _checked_sum(
        overhead, _checked_sum(a.byte_length(), b.byte_length())
    )
    var result = String()
    budget.grow_string(result, count, element)
    result.reserve_bytes(count)
    result.write(prefix, a, separator, b, "}")
    return result
