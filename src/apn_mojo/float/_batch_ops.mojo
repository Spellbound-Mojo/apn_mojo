"""Exact-source Float operands, native comparisons, and packed arithmetic fixtures."""

from ..batch.mask import Mask
from ..batch._parallel import _parallel_mask
from ..rational._batch_storage import _RationalInput
from ..batch._slices import _Selection
from ._batch_storage import _FloatInput
from ._arithmetic import _FloatArgument, _float_argument
from ._format import _merge_float_formats
from ._rounding import (
    _RoundedBinary,
    _round_ratio,
    _compare_scaled,
    _round_away,
    _rounded_direction,
    _overflow_result,
    _finish_round,
)
from ._comparison import _float_relation
from .context import FloatFormat
from .value import Float


@fieldwise_init
struct _FloatOperand(ImplicitlyCopyable):
    var floating: Optional[_FloatInput]
    var exact: Optional[_RationalInput]
    var scalar: Optional[_FloatArgument]
    var length: Int
    var broadcast: Bool

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.floating = move.floating^
        self.exact = move.exact^
        self.scalar = move.scalar^
        self.length = move.length
        self.broadcast = move.broadcast

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def value(self, index: Int) -> _FloatArgument:
        if self.scalar:
            return self.scalar.value()
        var position = 0 if self.broadcast else index
        if self.floating:
            return _float_argument(self.floating.value().value(position))
        return _float_argument(self.exact.value().value(position))

    def format(self) -> Optional[FloatFormat]:
        if self.floating:
            return self.floating.value().default_format()
        if self.scalar and self.scalar.value().format:
            return self.scalar.value().format.value()
        return None


def _float_scalar_operand(value: Float) -> _FloatOperand:
    """A shared Float as a one-element broadcast run, so kernels borrow it."""
    return _FloatOperand(_FloatInput(value, _Selection(0, 1, 1)), None, None, 1, True)


@fieldwise_init
struct _FloatInputs(Movable):
    var a: _FloatOperand
    var b: _FloatOperand

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.a = move.a^
        self.b = move.b^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def length(self) -> Int:
        return self.b.length if self.a.broadcast else self.a.length


    def check_shape(self) raises:
        if (
            not self.a.broadcast
            and not self.b.broadcast
            and self.a.length != self.b.length
        ):
            raise Error(
                String(
                    "Cannot operate on Float batches of lengths ",
                    self.a.length,
                    " and ",
                    self.b.length,
                    (
                        "; use equal lengths or a scalar operand. The"
                        " destination is unchanged."
                    ),
                )
            )

    def default_format(self) raises -> FloatFormat:
        return _merge_float_formats(
            self.a.format(),
            self.b.format(),
            left_native_precision=self.a.scalar.value().native_precision if self.a.scalar else 0,
            right_native_precision=self.b.scalar.value().native_precision if self.b.scalar else 0,
        )




@fieldwise_init
struct _FloatUnaryJob(Copyable, Movable):
    """One Float input and an operation, for elementwise kernels."""
    var source: _FloatInput
    var operation: Int
    var mode: Int
    var fail: Int


def _float_math_checkpoint(index: Int, fail: Int) raises:
    if index == fail:
        raise Error(
            String(
                "Injected Float batch failure at element ",
                index,
                "; inputs and destination are unchanged. Retry the operation.",
            )
        )














def _float_argument_order(a: _FloatArgument, b: _FloatArgument) raises -> Int:
    var left = a.value
    var right = b.value
    if left.kind == 3 or right.kind == 3:
        return 2
    if left.kind == 0 and right.kind == 0:
        return 0
    if left.negative != right.negative:
        return -1 if left.negative else 1
    var order = 0
    if left.kind != right.kind:
        order = -1 if left.kind < right.kind else 1
    elif left.kind == 1:
        order = _compare_scaled(
            left.numerator * right.denominator,
            right.numerator * left.denominator,
            right.scale - left.scale,
        )
    return -order if left.negative else order


@fieldwise_init
struct _FloatOrderJob(Copyable, Movable):
    var left: _FloatInput
    var right: _FloatInput
    var left_broadcast: Bool
    var right_broadcast: Bool
    var operation: Int


@always_inline
def _float_order(job: _FloatOrderJob, index: Int) raises -> Bool:
    ref x = job.left.element(0 if job.left_broadcast else index)
    ref y = job.right.element(0 if job.right_broadcast else index)
    return _float_relation(x._compare_float(y), job.operation)


@fieldwise_init
struct _FloatMixedOrderJob(Copyable, Movable):
    var a: _FloatOperand
    var b: _FloatOperand
    var operation: Int


def _float_mixed_order(job: _FloatMixedOrderJob, index: Int) raises -> Bool:
    """Floats against exact Integer or Rational values, compared exactly."""
    return _float_relation(
        _float_argument_order(job.a.value(index), job.b.value(index)),
        job.operation,
    )


def _native_float_comparison(inputs: _FloatInputs, operation: Int) raises -> Mask:
    inputs.check_shape()
    if inputs.a.floating and inputs.b.floating:
        # Two Float operands: compare them in place, without exact conversion.
        return _parallel_mask[_float_order](
            _FloatOrderJob(
                inputs.a.floating.value(), inputs.b.floating.value(),
                inputs.a.broadcast, inputs.b.broadcast, operation,
            ),
            inputs.length(),
        )
    return _parallel_mask[_float_mixed_order](_FloatMixedOrderJob(inputs.a, inputs.b, operation), inputs.length())
