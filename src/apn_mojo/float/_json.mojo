"""Version-1 scalar Float records, independent of physical storage and rounding."""

from ..common._json import _JSONReader
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from ..integer.value import Integer
from .context import FloatFormat
from .status import NumericStatus
from ._rounding import _RoundedBinary


def _float_json_int(
    mut reader: _JSONReader, payload: Bool = False
) raises -> Int:
    var text = reader.decimal(False, count_digits=payload)
    var bytes = text.as_bytes()
    var negative = bytes[0] == 45
    var magnitude = Int128(0)
    var limit = Int128(Int.MAX) + Int128(Int(negative))
    for i in range(Int(negative), len(bytes)):
        magnitude = magnitude * 10 + Int128(bytes[i] - 48)
        if magnitude > limit:
            reader.fail(
                "format or exponent outside supported range",
                (
                    "use native-Int precision/exponent bounds supported by"
                    " FloatFormat"
                ),
            )
    return Int(-magnitude if negative else magnitude)


def _read_float_json(
    text: String, mut budget: _ConversionBudget
) raises -> _RoundedBinary:
    budget.input(text.byte_length())
    budget.values(1)
    var reader = _JSONReader(text, budget, "Invalid float interchange at byte ")
    var result = _read_float_json_record(reader, complete=True)
    budget = reader.budget
    return result


comptime _DEFER_BITS = 1024
"""Batch reads convert significands of this many bits or more after the parse."""


def _decimal_less(a: String, b: String) -> Bool:
    """a < b for canonical positive decimal digits."""
    if a.byte_length() != b.byte_length():
        return a.byte_length() < b.byte_length()
    return a < b


struct _Deferred(Movable):
    """Significands a batch read converts after its parse, on the worker pool.

    When enabled, a finite record of _DEFER_BITS or more keeps its significand
    text. The parse still checks, where it always did, that the significand has
    exactly the precision's bits: it compares the digits with 2**(p-1) and 2**p
    in decimal, computed once per precision. Every record read adds one text,
    empty when the significand was converted inline.
    """
    var enabled: Bool
    var texts: List[String]
    var count: Int
    var precision: Int
    var low: String
    var high: String

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.texts = List[String]()
        self.count = 0
        self.precision = -1
        self.low = String()
        self.high = String()

    def normalized(mut self, text: String, precision: Int) raises -> Bool:
        if precision != self.precision:
            self.low = (Integer(1) << (precision - 1))._decimal()
            self.high = (Integer(1) << precision)._decimal()
            self.precision = precision
        return not _decimal_less(text, self.low) and _decimal_less(text, self.high)


def _read_float_json_record(
    mut reader: _JSONReader, *, complete: Bool = False
) raises -> _RoundedBinary:
    var inline = _Deferred(False)
    return _read_float_json_record(reader, inline, complete=complete)


def _read_float_json_record(
    mut reader: _JSONReader, mut deferred: _Deferred, *, complete: Bool = False
) raises -> _RoundedBinary:
    reader.expect(123, "provide a JSON object produced by Float.to_json()")
    var seen = 0
    var precision = 0
    var emin = 0
    var emax = 0
    var exponent = 0
    var kind = -1
    var negative = False
    var significand_text = String()
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "precision":
            flag = 4
        elif key == "emin":
            flag = 8
        elif key == "emax":
            flag = 16
        elif key == "class":
            flag = 32
        elif key == "sign":
            flag = 64
        elif key == "significand":
            flag = 128
        elif key == "exponent":
            flag = 256
        else:
            reader.fail(
                "unknown field", "use the fields produced by Float.to_json()"
            )
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one("use Float interchange version 1")
        elif flag == 2:
            reader.family("float", "use family float with Float.from_json()")
        elif flag == 4:
            precision = _float_json_int(reader)
        elif flag == 8:
            emin = _float_json_int(reader)
        elif flag == 16:
            emax = _float_json_int(reader)
        elif flag == 32:
            var tag = reader.string()
            if tag == "zero":
                kind = 0
            elif tag == "finite":
                kind = 1
            elif tag == "infinity":
                kind = 2
            elif tag == "nan":
                kind = 3
            else:
                reader.fail(
                    "unknown value class", "use zero, finite, infinity or nan"
                )
        elif flag == 64:
            var sign = reader.string()
            if sign != "+" and sign != "-":
                reader.fail("invalid sign", 'use the string "+" or "-"')
            negative = sign == "-"
        elif flag == 128:
            significand_text = reader.decimal(False)
        else:
            exponent = _float_json_int(reader, True)
        if reader.end_object(
            "separate fields with commas and close the object with }"
        ):
            break
    reader.require_fields(
        seen & 127,
        127,
        "include version, family, precision, emin, emax, class and sign",
    )
    if kind == 1:
        reader.require_fields(
            seen, 511, "include significand and exponent for finite values"
        )
    elif seen != 127:
        reader.fail(
            "unexpected finite payload",
            "omit significand and exponent for zero, infinity and nan",
        )
    if complete:
        reader.finish()
    if kind == 3 and negative:
        reader.fail(
            "noncanonical NaN sign", 'use sign "+" for the canonical quiet NaN'
        )
    var format = FloatFormat(precision, emin=emin, emax=emax)
    var significand = Integer(0)
    var later = String()
    if kind == 1:
        if exponent < emin or exponent > emax:
            reader.fail(
                "exponent outside format bounds", "use emin <= exponent <= emax"
            )
        if significand_text.as_bytes()[0] == 45 or significand_text == "0":
            reader.fail(
                "nonpositive finite significand",
                "use a positive normalized p-bit significand or class zero",
            )
        var normalized: Bool
        if deferred.enabled and precision >= _DEFER_BITS:
            normalized = deferred.normalized(significand_text, precision)
            later = significand_text
        else:
            significand = Integer._parse_json_digits(
                significand_text, reader.budget, reader.element
            )
            normalized = significand.magnitude_bit_length() == precision
        if not normalized:
            reader.fail(
                "unnormalized finite significand",
                (
                    "use exactly precision bits with the leading bit set; JSON"
                    " import never rounds"
                ),
            )
    if deferred.enabled:
        deferred.count += Int(Bool(later))
        deferred.texts.append(later^)
    return _RoundedBinary(
        kind, negative, significand, exponent, format, NumericStatus()
    )


def _format_float_json(
    value: _RoundedBinary,
    mut budget: _ConversionBudget,
    element: Int = -1,
    *,
    count_value: Bool = True,
) raises -> String:
    if count_value:
        budget.values(1, element)
    # Native metadata strings fit the pinned String's inline representation.
    comptime assert String.INLINE_CAPACITY >= 20
    var precision = String(value.format.precision())
    var emin = String(value.format.emin())
    var emax = String(value.format.emax())
    var exponent = String(value.exponent)
    var tag: StaticString = "zero"
    if value.kind == 1:
        tag = "finite"
    elif value.kind == 2:
        tag = "infinity"
    elif value.kind == 3:
        tag = "nan"
    comptime prefix: StaticString = (
        '{"version":1,"family":"float","precision":"'
    )
    comptime min_field: StaticString = '","emin":"'
    comptime max_field: StaticString = '","emax":"'
    comptime class_field: StaticString = '","class":"'
    comptime sign_field: StaticString = '","sign":"'
    comptime significand_field: StaticString = ',"significand":"'
    comptime exponent_field: StaticString = '","exponent":"'
    var header_size = (
        prefix.byte_length()
        + precision.byte_length()
        + min_field.byte_length()
        + emin.byte_length()
        + max_field.byte_length()
        + emax.byte_length()
        + class_field.byte_length()
        + tag.byte_length()
        + sign_field.byte_length()
        + 2
    )
    var overhead = header_size + 1
    if value.kind == 1:
        overhead = (
            header_size
            + significand_field.byte_length()
            + exponent_field.byte_length()
            + exponent.byte_length()
            + 2
        )
        budget.digits(exponent.byte_length() - Int(value.exponent < 0), element)
    budget.output(overhead, element)
    var significand = String()
    if value.kind == 1:
        significand = value.significand._format_with_budget(
            10, False, False, budget, element
        )
    var count = _checked_sum(overhead, significand.byte_length())
    var result = String()
    budget.grow_string(result, count, element)
    result.reserve_bytes(count)
    result.write(
        prefix,
        precision,
        min_field,
        emin,
        max_field,
        emax,
        class_field,
        tag,
        sign_field,
        "-" if value.negative else "+",
        '"',
    )
    if value.kind == 1:
        result.write(
            significand_field, significand, exponent_field, exponent, '"'
        )
    result.write("}")
    return result
