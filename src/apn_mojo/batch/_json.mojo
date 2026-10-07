"""Shared framing for numeric batch interchange.

Families own scalar records and their diagnostics. This driver owns field
validation, array traversal and budget accounting, in input order.
"""

from std.memory import ArcPointer
from std.sys import size_of
from ..common._json import _JSONReader, _join_records
from ..common._sizes import _checked_count, _checked_sum
from ..common.conversion import _ConversionBudget
from ._layout import _layout_record_bytes
from ._parallel import _map_values
from ._tensor import _Tensor
from ._values import _Values


@fieldwise_init
struct _BatchJSONLabels(Copyable, Movable):
    var diagnostic: StaticString
    var opening: StaticString
    var unknown: StaticString
    var version: StaticString
    var family: StaticString
    var family_hint: StaticString
    var values: StaticString
    var separator: StaticString
    var unknown_kind: StaticString
    var closing: StaticString
    var required: StaticString


trait _BatchJSONCodec:
    comptime Element: ImplicitlyCopyable & Deinitable
    comptime Format: ImplicitlyCopyable & Deinitable
    comptime State: Movable & Deinitable
    comptime formatted: Bool

    @staticmethod
    def start(budget: _ConversionBudget) -> Self.State: ...

    @staticmethod
    def labels() -> _BatchJSONLabels: ...

    @staticmethod
    def default_format() raises -> Self.Format: ...

    @staticmethod
    def read_format(mut reader: _JSONReader) raises -> Self.Format: ...

    @staticmethod
    def append_record(mut reader: _JSONReader, mut values: List[Self.Element], mut state: Self.State, index: Int) raises: ...

    @staticmethod
    def finish_records(mut values: List[Self.Element], mut state: Self.State, mut budget: _ConversionBudget) raises: ...


@fieldwise_init
struct _BatchJSONRecords[Codec: _BatchJSONCodec](Movable):
    var values: List[Self.Codec.Element]
    var format: Self.Codec.Format

    def take_values(deinit self) -> List[Self.Codec.Element]:
        return self.values^


def _read_batch_json[Codec: _BatchJSONCodec](
    text: String, mut budget: _ConversionBudget, mut shape: Optional[List[Int]],
) raises -> _BatchJSONRecords[Codec]:
    var labels = Codec.labels()
    budget.input(text.byte_length())
    var reader = _JSONReader(text, budget, labels.diagnostic)
    reader.expect(123, String(labels.opening))
    var values = List[Codec.Element]()
    var format = Codec.default_format()
    var version = 1
    var state = Codec.start(budget)
    var seen = 0
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif Codec.formatted and key == "default_format":
            flag = 4
        elif key == "values":
            flag = 8
        elif key == "shape":
            flag = 16
        else:
            reader.fail(String(labels.unknown_kind), String(labels.unknown))
        reader.mark_field(seen, flag)
        if flag == 1:
            version = reader.version(labels.version)
        elif flag == 16:
            shape = reader.shape()
        elif flag == 2:
            reader.family(labels.family, labels.family_hint)
        elif flag == 4:
            format = Codec.read_format(reader)
        else:
            reader.expect(91, String(labels.values))
            reader.whitespace()
            if reader.peek() != 93:
                while True:
                    var index = len(values)
                    reader.element = index
                    Codec.append_record(reader, values, state, index)
                    reader.whitespace()
                    if reader.peek() == 93:
                        break
                    reader.expect(44, String(labels.separator))
            reader.expect(93, "close the values array with ]")
            reader.element = -1
        if reader.end_object(labels.closing):
            break
    comptime required = 15 if Codec.formatted else 11
    reader.require_fields(seen & required, required, labels.required)
    reader.check_shape_field(version, Bool(seen & 16))
    reader.finish()
    Codec.finish_records(values, state, reader.budget)
    budget = reader.budget
    return _BatchJSONRecords[Codec](values^, format)


def _batch_owner_budget[T: ImplicitlyCopyable & Deinitable](mut budget: _ConversionBudget) raises:
    budget.allocate(size_of[ArcPointer[_Values[T]]._inner_type]())
    budget.allocate(_layout_record_bytes())


def _parse_formatted_batch_json[Codec: _BatchJSONCodec](
    text: String, mut budget: _ConversionBudget, mut shape: Optional[List[Int]],
) raises -> Tuple[_Tensor[Codec.Element], Codec.Format]:
    var records = _read_batch_json[Codec](text, budget, shape)
    var count = len(records.values)
    var format = records.format
    return _Tensor[Codec.Element](records^.take_values(), [count]), format


def _uncounted_record[
    Input: ImplicitlyCopyable & Deinitable,
    record: def(Input, Int, mut _ConversionBudget) raises thin -> String,
](input: Input, index: Int) raises -> String:
    var budget = _ConversionBudget(None)
    return record(input, index, budget)


def _write_batch_records[
    Input: ImplicitlyCopyable & Deinitable,
    record: def(Input, Int, mut _ConversionBudget) raises thin -> String,
](input: Input, length: Int, var result: String, mut budget: _ConversionBudget) raises -> String:
    """Append records after a charged header; bounded output stays sequential."""
    if not budget.bounded_output():
        return _join_records(result^, _map_values[_uncounted_record[Input, record]](input, length).take_list())
    return _write_bounded_batch_records[Input, _append_batch_record[Input, record]](input, length, result^, budget)


def _append_batch_record[
    Input: ImplicitlyCopyable & Deinitable,
    record: def(Input, Int, mut _ConversionBudget) raises thin -> String,
](input: Input, index: Int, mut result: String, mut budget: _ConversionBudget) raises:
    budget.output(Int(index != 0), index)
    var text = record(input, index, budget)
    var extra = _checked_sum(text.byte_length(), Int(index != 0))
    _ = _checked_count(_checked_sum(result.byte_length(), extra), 1)
    budget.grow_string(result, extra, index)
    if index:
        result += ","
    result += text


def _write_bounded_batch_records[
    Input: ImplicitlyCopyable & Deinitable,
    append: def(Input, Int, mut String, mut _ConversionBudget) raises thin -> None,
](input: Input, length: Int, var result: String, mut budget: _ConversionBudget) raises -> String:
    """Stream records into one output buffer, preserving each codec's charges."""
    for index in range(length):
        append(input, index, result, budget)
    budget.grow_string(result, 2)
    result += "]}"
    return result^
