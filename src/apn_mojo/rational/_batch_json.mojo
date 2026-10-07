"""Budgeted canonical Rational interchange without importing public Batch."""

from ..batch._values import _Values
from std.sys import size_of
from ..integer.value import Integer
from .value import Rational
from ..common.conversion import _ConversionBudget
from ..common._json import _batch_json_header, _join_records
from ..batch._parallel import _map_values
from ._json import _read_rational_batch_json
from ._batch_storage import _RationalInput
from ..batch._tensor import _Tensor
from ..batch._json import _write_bounded_batch_records, _batch_owner_budget
from ..integer._text import _format_bound
from ..common._sizes import _checked_count, _checked_sum


@fieldwise_init
struct _RationalTexts(ImplicitlyCopyable):
    """Validated fraction texts, read in place while they convert."""
    var records: Pointer[List[Tuple[String, String]], MutUntrackedOrigin]


def _rational_at(texts: _RationalTexts, index: Int) raises -> Rational:
    ref record = texts.records[][index]
    var budget = _ConversionBudget(None)
    return Rational._from_json_components(record[0], record[1], budget)


def _parse_rational_records(mut records: List[Tuple[String, String]]) raises -> _Values[Rational]:
    """Each record as a canonical Rational, long runs on the worker pool; the
    lowest failing element raises as the sequential loop does."""
    var state = _RationalTexts(Pointer(to=records).unsafe_origin_cast[MutUntrackedOrigin]())
    return _map_values[_rational_at](state, len(records), diagnostic="Cannot read rational interchange element ")


def _parse_rational_batch_json(
    text: String, mut budget: _ConversionBudget
) raises -> _Tensor[Rational]:
    var shape = Optional[List[Int]]()
    return _parse_rational_batch_json(text, budget, shape)


def _parse_rational_batch_json(
    text: String, mut budget: _ConversionBudget,
    mut shape: Optional[List[Int]],
) raises -> _Tensor[Rational]:
    var records = _read_rational_batch_json(text, budget, shape)
    var length = len(records)
    if not budget.bounded_allocation():
        # Unlimited: the texts are validated, so long runs convert in parallel.
        return _Tensor[Rational](_parse_rational_records(records), [length])
    budget.allocate(length, size_of[Rational]())
    var values = List[Rational](capacity=_checked_count(length, size_of[Rational]()))
    for index in range(length):
        try:
            values.append(Rational._from_json_components(
                records[index][0], records[index][1], budget
            ))
        except error:
            raise Error(String("Cannot read rational interchange element ", index, ": ", error))
    _batch_owner_budget[Rational](budget)
    return _Tensor[Rational](values^, [length])


def _json_rational_component(
    input: _RationalInput,
    denominator: Bool,
    index: Int,
    mut budget: _ConversionBudget,
) raises -> Integer:
    var value = input.component(denominator, index)
    _ = _format_bound(value._word_count())
    budget.preflight(
        value.magnitude_bit_length(), 10, Int(value._negative()), index
    )
    return value


def _rational_record(input: _RationalInput, index: Int) raises -> String:
    """One element's record, when nothing counts the conversion."""
    var budget = _ConversionBudget(None)
    var numerator = _json_rational_component(input, False, index, budget)._decimal()
    var denominator = _json_rational_component(input, True, index, budget)._decimal()
    return String('{"numerator":"', numerator, '","denominator":"', denominator, '"}')


def _write_rational_batch_json(
    input: _RationalInput, mut budget: _ConversionBudget,
    shape: Optional[List[Int]] = None,
) raises -> String:
    var length = input.selection.count
    budget.values(
        length,
        budget.limits._values if length > budget.limits._values
        and budget.limits._values >= 0 else -1,
    )
    var prefix = _batch_json_header("rational-batch", shape) + ',"values":['
    budget.output(prefix.byte_length() + 2)
    # StaticString construction borrows the literal; the first growth allocates.
    var result = String(prefix)
    if not budget.bounded_output():
        # Unlimited: records are independent, so long runs format in parallel.
        return _join_records(result^, _map_values[_rational_record](input, length).take_list())
    return _write_bounded_batch_records[_RationalInput, _append_rational_record](input, length, result^, budget)


def _append_rational_record(input: _RationalInput, index: Int, mut result: String, mut budget: _ConversionBudget) raises:
    comptime first: StaticString = '{"numerator":"'
    comptime middle: StaticString = '","denominator":"'
    var framing = (
        first.byte_length() + middle.byte_length() + 2 + Int(index != 0)
    )
    budget.output(framing, index)
    var n = _json_rational_component(input, False, index, budget)
    var numerator = n._format_with_budget(10, False, False, budget, index)
    var d = _json_rational_component(input, True, index, budget)
    var denominator = d._format_with_budget(10, False, False, budget, index)
    var extra = _checked_sum(
        framing,
        _checked_sum(numerator.byte_length(), denominator.byte_length()),
    )
    _ = _checked_count(_checked_sum(result.byte_length(), extra), 1)
    budget.grow_string(result, extra, index)
    if index:
        result += ","
    result.write(first, numerator, middle, denominator, '"}')
