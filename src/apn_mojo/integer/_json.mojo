"""Strict version-1 Integer records using the shared JSON reader."""

from std.sys import size_of

from ..common._json import _JSONReader
from ..batch._json import _BatchJSONCodec, _BatchJSONLabels, _read_batch_json
from ..common._sizes import _checked_count, _checked_sum
from ..common.conversion import _ConversionBudget


def _json_decimal_bound(words: Int) raises -> Int:
    # Each used 32-bit magnitude word needs at most ten decimal digits.
    # Check before multiplication; the extra byte covers a sign or zero.
    return _checked_count(words, 10) * 10 + 1


def _read_integer_json(
    text: String, batch: Bool, mut budget: _ConversionBudget
) raises -> List[String]:
    var shape = Optional[List[Int]]()
    return _read_integer_json(text, batch, budget, shape)


def _read_integer_json(
    text: String, batch: Bool, mut budget: _ConversionBudget,
    mut shape: Optional[List[Int]],
) raises -> List[String]:
    if batch:
        var records = _read_batch_json[_IntegerBatchJSON](text, budget, shape)
        return records^.take_values()
    budget.input(text.byte_length())
    var reader = _JSONReader(
        text, budget, "Invalid integer interchange at byte "
    )
    reader.expect(123, "provide a JSON object produced by to_json()")
    var seen = 0
    var version = 1
    var values = List[String]()
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "value":
            flag = 4
        else:
            reader.fail(
                "unknown field or wrong scalar/batch schema",
                (
                    "use version, family, and values for a batch or value for"
                    " an Integer"
                ),
            )
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one(
                "use the integer interchange version 1 or a compatible library"
            )
        elif flag == 2:
            reader.family(
                "integer",
                "use family integer-batch with Batch[Integer].from_json() or integer with Integer.from_json()",
            )
        else:
            var value = reader.decimal()
            reader.budget.append(values, value^)
        if reader.end_object(
            "separate object fields with commas and close the object with }"
        ):
            break
    reader.require_fields(
        seen & 7, 7, "include version, family, and value or values"
    )
    reader.check_shape_field(version, Bool(seen & 8))
    reader.finish()
    budget = reader.budget
    return values^


struct _IntegerBatchJSON(_BatchJSONCodec):
    comptime Element = String
    comptime Format = Bool
    comptime State = Bool
    comptime formatted = False

    @staticmethod
    def labels() -> _BatchJSONLabels:
        return _BatchJSONLabels(
            "Invalid integer interchange at byte ",
            "provide a JSON object produced by to_json()",
            "use version, family, and values for a batch or value for an Integer",
            "use integer interchange version 1, or 2 for a shaped batch",
            "integer-batch",
            "use family integer-batch with Batch[Integer].from_json() or integer with Integer.from_json()",
            "provide values as an array of decimal strings",
            "separate array elements with commas and close the array with ]",
            "unknown field or wrong scalar/batch schema",
            "separate object fields with commas and close the object with }",
            "include version, family, and value or values",
        )

    @staticmethod
    def start(budget: _ConversionBudget) -> Bool:
        return False

    @staticmethod
    def default_format() raises -> Bool:
        return False

    @staticmethod
    def read_format(mut reader: _JSONReader) raises -> Bool:
        return False

    @staticmethod
    def append_record(mut reader: _JSONReader, mut values: List[String], mut state: Bool, index: Int) raises:
        _ = _checked_count(_checked_sum(index, 1), size_of[String]())
        var value = reader.decimal()
        reader.budget.append(values, value^, index)
        reader.element = -1

    @staticmethod
    def finish_records(mut values: List[String], mut state: Bool, mut budget: _ConversionBudget) raises:
        pass
