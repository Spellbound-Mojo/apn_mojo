"""Paired arithmetic adapter for the family-independent ordered executor."""

from ..batch._reduce_exec import _ReductionInputs, _ReductionOperation
from ..float._arithmetic import _FloatArgument
from ..float._batch_ops import _FloatOperand
from ..float._rounding import _RoundedBinary
from ..float._ordered_reductions import _FloatReductionInputs, _FloatOrderedReduction
from ..float.context import ArithmeticContext
from ..float.value import Float
from ..float._reductions import _dot_shape
from ._arithmetic import _complex_operation
from ._batch_ops import _ComplexOperand
from ._format import _ComplexFormats
from ._input import _ComplexArgument
from ._reductions import _dot_component
from .context import ComplexContext, _ComplexContextArgument
from .status import ComplexStatus
from .value import Complex


@fieldwise_init
struct _ComplexReductionInputs(_ReductionInputs):
    var left: _ComplexOperand
    var right: Optional[_ComplexOperand]

    def length(self) -> Int:
        return self.left.length


def _ordered_pair(
    real: _FloatArgument, imag: _FloatArgument
) -> _ComplexArgument:
    var result = _ComplexArgument(Int(0))
    result.real = real
    result.imag = imag
    result.complex = True
    return result


struct _ComplexOrderedReduction[dot: Bool](_ReductionOperation):
    comptime Inputs = _ComplexReductionInputs
    comptime Value = _ComplexArgument
    comptime Result = Tuple[Complex, ComplexStatus]
    var real: _FloatOrderedReduction[Self.dot]
    var imag: _FloatOrderedReduction[Self.dot]
    var name: String

    def __init__(
        out self,
        name: String,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        *,
        fail_after_step: Int = -1,
        fail_component: Int = 0,
        fail: Bool = False,
    ) raises:
        self.name = String("Complex ", name)
        var real_context: Optional[ArithmeticContext] = None
        var imag_context: Optional[ArithmeticContext] = None
        if context:
            real_context = context.value().real()
            imag_context = context.value().imag()
        self.real = _FloatOrderedReduction[Self.dot](
            String(self.name, " real component"),
            real_context,
            fail_after_step if fail_component == 0 else -1,
            fail and fail_component == 0,
        )
        self.imag = _FloatOrderedReduction[Self.dot](
            String(self.name, " imaginary component"),
            imag_context,
            fail_after_step if fail_component == 1 else -1,
            fail and fail_component == 1,
        )

    def validate(self, inputs: Self.Inputs) raises:
        comptime if Self.dot:
            _dot_shape(inputs.length(), inputs.right.value().length, self.name)

    def prepare(mut self, inputs: Self.Inputs) raises:
        var real_right: Optional[_FloatOperand] = None
        var imag_right: Optional[_FloatOperand] = None
        comptime if Self.dot:
            real_right = _dot_component(inputs.right.value(), False)
            imag_right = _dot_component(inputs.right.value(), True)
        # Both component metadata scans finish before any arithmetic can trap.
        self.real.prepare(
            _FloatReductionInputs(
                _dot_component(inputs.left, False), real_right
            )
        )
        self.imag.prepare(
            _FloatReductionInputs(_dot_component(inputs.left, True), imag_right)
        )
        _ = _ComplexFormats(
            self.real.target.format(), self.imag.target.format()
        )

    def identity(self) raises -> Self.Value:
        return _ordered_pair(self.real.identity(), self.imag.identity())

    def element(mut self, inputs: Self.Inputs, index: Int) raises -> Self.Value:
        comptime if not Self.dot:
            return inputs.left.value(index)
        else:
            var target = ComplexContext(
                real=self.real.target, imag=self.imag.target
            )
            var result: Tuple[_RoundedBinary, _RoundedBinary]
            try:
                result = _complex_operation(
                    inputs.left.value(index),
                    inputs.right.value().value(index),
                    2,
                    _ComplexContextArgument(target),
                )
            except error:
                raise Error(
                    String(
                        self.name, " product at element ", index, ": ", error
                    )
                )
            var real = self.real.record(result[0], "product", index, index + 1)
            var imag = self.imag.record(result[1], "product", index, index + 1)
            return _ordered_pair(real, imag)

    def combine(
        mut self, left: Self.Value, right: Self.Value, start: Int, end: Int
    ) raises -> Self.Value:
        var real = self.real.combine(left.real, right.real, start, end)
        var imag = self.imag.combine(left.imag, right.imag, start, end)
        return _ordered_pair(real, imag)

    def singleton(mut self, value: Self.Value) raises -> Self.Value:
        var real = self.real.singleton(value.real)
        var imag = self.imag.singleton(value.imag)
        return _ordered_pair(real, imag)

    def finish(mut self, var value: Self.Value) raises -> Self.Result:
        var real = self.real.finish(value.real)
        var imag = self.imag.finish(value.imag)
        return (
            Complex(_real=Float(_rounded=real), _imag=Float(_rounded=imag)),
            ComplexStatus._make(real.status, imag.status),
        )
