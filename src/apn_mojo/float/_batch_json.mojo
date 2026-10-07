"""Streaming, budgeted Float batch interchange without importing public Batch."""

from std.sys import size_of
from ..common._sizes import _checked_count, _checked_sum
from ..common._json import _JSONReader
from ..common.conversion import _ConversionBudget
from ..common._json import _batch_json_header
from ..batch._tensor import _Tensor
from ..batch._json import _BatchJSONCodec, _BatchJSONLabels, _batch_owner_budget, _parse_formatted_batch_json, _write_batch_records
from ..integer._text import _format_bound
from .value import Float
from .context import FloatFormat
from ._rounding import _RoundedBinary
from ._json import _float_json_int, _read_float_json_record, _format_float_json, _Deferred
from ..batch._digits import _parse_digit_texts
from ._batch_storage import _FloatInput


def _read_float_json_format(mut reader: _JSONReader) raises -> FloatFormat:
    reader.expect(123, "provide default_format with precision, emin and emax")
    var seen = 0
    var precision = 0
    var emin = 0
    var emax = 0
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "precision":
            flag = 1
        elif key == "emin":
            flag = 2
        elif key == "emax":
            flag = 4
        else:
            reader.fail("unknown format field", "use precision, emin and emax")
        reader.mark_field(seen, flag)
        var value = _float_json_int(reader)
        if flag == 1:
            precision = value
        elif flag == 2:
            emin = value
        else:
            emax = value
        if reader.end_object(
            "separate format fields with commas and close with }"
        ):
            break
    reader.require_fields(seen, 7, "include precision, emin and emax")
    return FloatFormat(precision, emin=emin, emax=emax)



struct _FloatBatchJSON(_BatchJSONCodec):
    comptime Element = Float
    comptime Format = FloatFormat

    comptime State = _Deferred
    comptime formatted = True

    @staticmethod
    def start(budget: _ConversionBudget) -> _Deferred:
        return _Deferred(not budget.bounded_allocation())

    @staticmethod
    def labels() -> _BatchJSONLabels:
        return _BatchJSONLabels(
            "Invalid float-batch interchange at byte ",
            "provide a JSON object produced by Batch[Float].to_json()",
            "use the fields produced by Batch[Float].to_json()",
            "use Float batch interchange version 1, or 2 for a shaped batch",
            "float-batch", "use family float-batch with Batch[Float].from_json()",
            "provide values as an array of Float records",
            "separate Float records with commas and close the array with ]",
            "unknown field",
            "separate envelope fields with commas and close with }",
            "include version, family, default_format and values",
        )

    @staticmethod
    def default_format() raises -> FloatFormat:
        return FloatFormat()

    @staticmethod
    def read_format(mut reader: _JSONReader) raises -> FloatFormat:
        return _read_float_json_format(reader)

    @staticmethod
    def append_record(mut reader: _JSONReader, mut values: List[Float], mut deferred: _Deferred, index: Int) raises:
        reader.budget.values(1, index)
        _ = _checked_count(_checked_sum(index, 1), size_of[Float]())
        var value: Float
        try:
            value = Float(_rounded=_read_float_json_record(reader, deferred))
        except error:
            raise Error(
                String(
                    "Cannot read float interchange element ",
                    index,
                    ": ",
                    error,
                )
            )
        reader.budget.append(values, value^, index)

    @staticmethod
    def finish_records(mut values: List[Float], mut deferred: _Deferred, mut budget: _ConversionBudget) raises:
        if deferred.count:
            var parsed = _parse_digit_texts(deferred.texts).take_list()
            for i in range(len(values)):
                if deferred.texts[i]:
                    swap(values[i]._significand, parsed[i])
        _batch_owner_budget[Float](budget)


def _parse_float_batch_json(
    text: String, mut budget: _ConversionBudget
) raises -> Tuple[_Tensor[Float], FloatFormat]:
    var shape = Optional[List[Int]]()
    return _parse_float_batch_json(text, budget, shape)


def _parse_float_batch_json(
    text: String, mut budget: _ConversionBudget,
    mut shape: Optional[List[Int]],
) raises -> Tuple[_Tensor[Float], FloatFormat]:
    return _parse_formatted_batch_json[_FloatBatchJSON](text, budget, shape)


def _json_float_value(
    input: _FloatInput, index: Int, mut budget: _ConversionBudget
) raises -> _RoundedBinary:
    var value = input.value(index)
    if value._kind == 1:
        _ = _format_bound(value._significand._word_count())
        budget.preflight(value.precision(), 10, 0, index)
    return value._interchange_record()


def _float_record(
    input: _FloatInput, index: Int, mut budget: _ConversionBudget,
) raises -> String:
    var value = _json_float_value(input, index, budget)
    return _format_float_json(value, budget, index, count_value=False)


def _write_float_batch_json(
    input: _FloatInput, mut budget: _ConversionBudget,
    shape: Optional[List[Int]] = None,
) raises -> String:
    var length = input.selection.count
    budget.values(
        length,
        budget.limits._values if 0 <= budget.limits._values < length else -1,
    )
    var prefix = _batch_json_header("float-batch", shape) + ',"default_format":{"precision":"'
    comptime min_field: StaticString = '","emin":"'
    comptime max_field: StaticString = '","emax":"'
    comptime values_field: StaticString = '"},"values":['
    comptime assert String.INLINE_CAPACITY >= 20
    var format = input.default_format()
    var precision = String(format.precision())
    var emin = String(format.emin())
    var emax = String(format.emax())
    var header = (
        prefix.byte_length()
        + min_field.byte_length()
        + max_field.byte_length()
        + values_field.byte_length()
        + precision.byte_length()
        + emin.byte_length()
        + emax.byte_length()
    )
    budget.output(header + 2)
    var result = String()
    budget.grow_string(result, header)
    result.reserve_bytes(header)
    result.write(
        prefix, precision, min_field, emin, max_field, emax, values_field
    )
    return _write_batch_records[_FloatInput, _float_record](input, length, result^, budget)
