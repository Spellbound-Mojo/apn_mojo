"""Decimal digit texts converted on the worker pool, for batch JSON reads."""

from ._parallel import _map_values
from ._values import _Values
from ..integer.value import Integer
from ..common.conversion import _ConversionBudget


@fieldwise_init
struct _DigitTexts(ImplicitlyCopyable):
    """Validated digit texts, read in place while they convert."""
    var texts: Pointer[List[String], MutUntrackedOrigin]


def _digits_at(texts: _DigitTexts, index: Int) raises -> Integer:
    ref text = texts.texts[][index]
    if not text:
        return Integer(0)
    var budget = _ConversionBudget(None)
    return Integer._parse_json_digits(text, budget, index)


def _parse_digit_texts(mut texts: List[String], diagnostic: String = "") raises -> _Values[Integer]:
    """Each validated canonical decimal text as an Integer, an empty text as
    zero; long runs convert on the worker pool. For readers whose budget does
    not count allocations: the conversions charge nothing."""
    var state = _DigitTexts(Pointer(to=texts).unsafe_origin_cast[MutUntrackedOrigin]())
    return _map_values[_digits_at](state, len(texts), diagnostic=diagnostic)
