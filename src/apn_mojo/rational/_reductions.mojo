"""Exact folds over retained component storage, without a product batch."""

from .value import Rational
from ._batch_storage import _RationalInput


def _rational_reduction_checkpoint(position: Int, fail: Int) raises:
    if position == fail:
        raise Error(
            String(
                "Injected Rational reduction failure after logical element ",
                position,
                "; the destination is unchanged. Retry the operation.",
            )
        )


def _dot_rational(
    left: _RationalInput, right: _RationalInput, fail: Int = -1
) raises -> Rational:
    var length = left.selection.count
    if length != right.selection.count:
        raise Error(
            String(
                "Cannot compute Rational dot product for lengths ",
                length,
                " and ",
                right.selection.count,
                "; use equal-length batches.",
                " No truncation or broadcasting is performed.",
                " The destination is unchanged.",
            )
        )
    var result = Rational()
    for position in range(length):
        # Metadata checks avoid extracting a wide operand paired with zero.
        if not left.is_zero(position) and not right.is_zero(position):
            var term = left.value(position) * right.value(position)
            result = result + term if result else term
        _rational_reduction_checkpoint(position, fail)
    return result
