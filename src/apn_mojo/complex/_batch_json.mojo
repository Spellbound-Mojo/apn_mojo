"""Budgeted paired-component interchange without importing public Batch."""

from std.sys import size_of
from ..common._sizes import _checked_count, _checked_sum
from ..common._json import _JSONReader
from ..common.conversion import _ConversionBudget
from ..common._json import _batch_json_header
from ..batch._tensor import _Tensor
from ..batch._json import _BatchJSONCodec, _BatchJSONLabels, _batch_owner_budget, _parse_formatted_batch_json, _write_batch_records
from ..float.value import Float
from ..float.context import FloatFormat
from ..float._json import _Deferred
from ..batch._digits import _parse_digit_texts
from ..float._batch_json import _read_float_json_format, _json_float_value
from ._json import _read_complex_json_record, _format_complex_json
from ._format import _ComplexFormats
from ._batch_storage import _ComplexInput
from .value import Complex


def _read_complex_json_format(
    mut reader: _JSONReader,
) raises -> _ComplexFormats:
    reader.expect(
        123, "provide default_format with real and imag format objects"
    )
    var real = FloatFormat()
    var imag = FloatFormat()
    var seen = 0
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "real":
            flag = 1
        elif key == "imag":
            flag = 2
        else:
            reader.fail(
                "unknown component format", "use real and imag format objects"
            )
        reader.mark_field(seen, flag)
        var format = _read_float_json_format(reader)
        if flag == 1:
            real = format
        else:
            imag = format
        if reader.end_object(
            "separate component formats with a comma and close with }"
        ):
            break
    reader.require_fields(seen, 3, "include both real and imag default formats")
    return _ComplexFormats(real, imag)


struct _ComplexBatchJSON(_BatchJSONCodec):
    comptime Element = Complex
    comptime Format = _ComplexFormats

    comptime State = _Deferred
    comptime formatted = True

    @staticmethod
    def start(budget: _ConversionBudget) -> _Deferred:
        return _Deferred(not budget.bounded_allocation())

    @staticmethod
    def labels() -> _BatchJSONLabels:
        return _BatchJSONLabels(
            "Invalid complex-batch interchange at byte ",
            "provide an object produced by Batch[Complex].to_json()",
            "use version, family, default_format and values",
            "use Complex batch interchange version 1, or 2 for a shaped batch",
            "complex-batch", "use family complex-batch with Batch[Complex].from_json()",
            "provide values as an array of Complex records",
            "separate Complex records with commas and close the array with ]",
            "unknown field",
            "separate envelope fields with commas and close with }",
            "include version, family, default_format and values",
        )

    @staticmethod
    def default_format() raises -> _ComplexFormats:
        return _ComplexFormats(FloatFormat(), FloatFormat())

    @staticmethod
    def read_format(mut reader: _JSONReader) raises -> _ComplexFormats:
        return _read_complex_json_format(reader)

    @staticmethod
    def append_record(mut reader: _JSONReader, mut values: List[Complex], mut deferred: _Deferred, index: Int) raises:
        reader.budget.values(1, index)
        _ = _checked_count(_checked_sum(index, 1), size_of[Complex]())
        try:
            var parts = _read_complex_json_record(reader, deferred)
            reader.budget.append(values, Complex(
                _real=Float(_rounded=parts[0]),
                _imag=Float(_rounded=parts[1]),
            ), index)
        except error:
            raise Error(
                String(
                    "Cannot read complex interchange element ",
                    index,
                    ": ",
                    error,
                )
            )

    @staticmethod
    def finish_records(mut values: List[Complex], mut deferred: _Deferred, mut budget: _ConversionBudget) raises:
        if deferred.count:
            var parsed = _parse_digit_texts(deferred.texts).take_list()
            for i in range(len(values)):
                if deferred.texts[2 * i]:
                    swap(values[i]._real._significand, parsed[2 * i])
                if deferred.texts[2 * i + 1]:
                    swap(values[i]._imag._significand, parsed[2 * i + 1])
        _batch_owner_budget[Complex](budget)


def _parse_complex_batch_json(
    text: String,
    mut budget: _ConversionBudget,
) raises -> Tuple[_Tensor[Complex], _ComplexFormats]:
    var shape = Optional[List[Int]]()
    return _parse_complex_batch_json(text, budget, shape)


def _parse_complex_batch_json(
    text: String,
    mut budget: _ConversionBudget,
    mut shape: Optional[List[Int]],
) raises -> Tuple[_Tensor[Complex], _ComplexFormats]:
    return _parse_formatted_batch_json[_ComplexBatchJSON](text, budget, shape)


def _complex_record(
    input: _ComplexInput, index: Int, mut budget: _ConversionBudget,
) raises -> String:
    var a = _json_float_value(input.real, index, budget)
    var b = _json_float_value(input.imag, index, budget)
    return _format_complex_json(a, b, budget, index, count_value=False)


def _write_complex_batch_json(
    input: _ComplexInput,
    mut budget: _ConversionBudget,
    shape: Optional[List[Int]] = None,
) raises -> String:
    var length = input.real.selection.count
    budget.values(
        length,
        budget.limits._values if 0 <= budget.limits._values < length else -1,
    )
    var prefix = _batch_json_header("complex-batch", shape) + ',"default_format":{"real":'
    comptime format_prefix: StaticString = '{"precision":"'
    comptime min_field: StaticString = '","emin":"'
    comptime max_field: StaticString = '","emax":"'
    comptime format_end: StaticString = '"}'
    comptime imag_field: StaticString = ',"imag":'
    comptime values_field: StaticString = '},"values":['
    comptime assert String.INLINE_CAPACITY >= 20
    var real = input.real.default_format()
    var imag = input.imag.default_format()
    var rp = String(real.precision())
    var ri = String(real.emin())
    var ra = String(real.emax())
    var ip = String(imag.precision())
    var ii = String(imag.emin())
    var ia = String(imag.emax())
    var header = (
        prefix.byte_length()
        + imag_field.byte_length()
        + values_field.byte_length()
        + 2
        * (
            format_prefix.byte_length()
            + min_field.byte_length()
            + max_field.byte_length()
            + format_end.byte_length()
        )
        + rp.byte_length()
        + ri.byte_length()
        + ra.byte_length()
        + ip.byte_length()
        + ii.byte_length()
        + ia.byte_length()
    )
    budget.output(header + 2)
    var result = String()
    budget.grow_string(result, header)
    result.reserve_bytes(header)
    result.write(
        prefix,
        format_prefix,
        rp,
        min_field,
        ri,
        max_field,
        ra,
        format_end,
        imag_field,
        format_prefix,
        ip,
        min_field,
        ii,
        max_field,
        ia,
        format_end,
        values_field,
    )
    return _write_batch_records[_ComplexInput, _complex_record](input, length, result^, budget)
