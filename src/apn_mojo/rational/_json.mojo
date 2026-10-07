"""Strict version-1 Rational records using the shared JSON tokenizer."""

from ..common._json import _JSONReader
from ..batch._json import _BatchJSONCodec, _BatchJSONLabels, _read_batch_json
from ..common.conversion import _ConversionBudget


def _read_rational_json(
    text: String, mut budget: _ConversionBudget
) raises -> Tuple[String, String]:
    budget.input(text.byte_length())
    budget.values(1)
    var reader = _JSONReader(
        text, budget, "Invalid rational interchange at byte "
    )
    var result = _read_rational_json_record(reader)
    reader.finish()
    budget = reader.budget
    return result^


def _read_rational_json_record(mut reader: _JSONReader) raises -> Tuple[String, String]:
    """A complete Rational record within a larger document: the numerator and
    denominator text, with the denominator checked positive."""
    reader.expect(123, "provide a JSON object produced by Rational.to_json()")
    var seen = 0
    var numerator = String()
    var denominator = String()
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "numerator":
            flag = 4
        elif key == "denominator":
            flag = 8
        else:
            reader.fail(
                "unknown field",
                "use only version, family, numerator and denominator",
            )
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one(
                "use Rational interchange version 1 or a compatible library"
            )
        elif flag == 2:
            reader.family(
                "rational", "use family rational with Rational.from_json()"
            )
        elif flag == 4:
            numerator = reader.decimal(False)
        else:
            denominator = reader.decimal(False)
        if reader.end_object(
            "separate fields with commas and close the object with }"
        ):
            break
    reader.require_fields(
        seen, 15, "include version, family, numerator and denominator"
    )
    if denominator == "0" or denominator.as_bytes()[0] == 45:
        reader.fail(
            "nonpositive denominator", "use a positive canonical denominator"
        )
    return (numerator, denominator)


def _read_rational_components(
    mut reader: _JSONReader,
) raises -> Tuple[String, String]:
    reader.budget.values(1, reader.element)
    reader.expect(
        123, "provide an object with numerator and denominator strings"
    )
    var seen = 0
    var numerator = String()
    var denominator = String()
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "numerator":
            flag = 1
        elif key == "denominator":
            flag = 2
        else:
            reader.fail(
                "unknown component field", "use only numerator and denominator"
            )
        reader.mark_field(seen, flag)
        if flag == 1:
            numerator = reader.decimal(False)
        else:
            denominator = reader.decimal(False)
        if reader.end_object(
            "separate fields with commas and close the object with }"
        ):
            break
    reader.require_fields(
        seen,
        3,
        "include numerator and denominator",
        reason="missing component field",
    )
    if denominator == "0" or denominator.as_bytes()[0] == 45:
        reader.fail(
            "nonpositive denominator", "use a positive canonical denominator"
        )
    return numerator, denominator


def _read_rational_batch_json(
    text: String, mut budget: _ConversionBudget
) raises -> List[Tuple[String, String]]:
    var shape = Optional[List[Int]]()
    return _read_rational_batch_json(text, budget, shape)


def _read_rational_batch_json(
    text: String, mut budget: _ConversionBudget, mut shape: Optional[List[Int]],
) raises -> List[Tuple[String, String]]:
    var records = _read_batch_json[_RationalBatchJSON](text, budget, shape)
    return records^.take_values()


struct _RationalBatchJSON(_BatchJSONCodec):
    comptime Element = Tuple[String, String]
    comptime Format = Bool
    comptime State = Bool
    comptime formatted = False

    @staticmethod
    def labels() -> _BatchJSONLabels:
        return _BatchJSONLabels(
            "Invalid rational interchange at byte ",
            "provide a JSON object produced by Batch[Rational].to_json()",
            "use only version, family, shape and values",
            "use Rational interchange version 1, or 2 for a shaped batch",
            "rational-batch",
            "use family rational-batch with Batch[Rational].from_json()",
            "provide an array of numerator/denominator records",
            "separate values with commas and close the array with ]",
            "unknown field",
            "separate fields with commas and close the object with }",
            "include version, family and values",
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
    def append_record(mut reader: _JSONReader, mut values: List[Tuple[String, String]], mut state: Bool, index: Int) raises:
        var value = _read_rational_components(reader)
        reader.budget.append(values, value, index)

    @staticmethod
    def finish_records(mut values: List[Tuple[String, String]], mut state: Bool, mut budget: _ConversionBudget) raises:
        pass
