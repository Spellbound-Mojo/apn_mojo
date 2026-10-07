"""Float arithmetic/status adapter; traversal belongs to batch._reduce_exec."""

from ..batch._reduce_exec import _ReductionInputs, _ReductionOperation
from ..integer.value import Integer
from ._batch_ops import _FloatOperand
from ._reductions import _dot_metadata
from ._format import _merge_float_formats
from ._arithmetic import _FloatArgument, _float_argument, _float_operation
from ._input import _FloatInput
from ._rounding import _RoundedBinary, _finish_round
from .context import ArithmeticContext, FloatFormat
from .status import NumericStatus
from .value import Float


@fieldwise_init
struct _FloatReductionInputs(_ReductionInputs):
    var left: _FloatOperand
    var right: Optional[_FloatOperand]

    def length(self) -> Int:
        return self.left.length


def _metadata_argument(input: _FloatOperand, index: Int) -> _FloatArgument:
    var metadata = _dot_metadata(input, index)
    return _FloatArgument(
        _FloatInput(
            metadata.kind,
            metadata.negative,
            Integer(1 if metadata.kind == 1 else 0),
            Integer(1),
            0,
        ),
        metadata.format,
        0,
    )


struct _FloatOrderedReduction[dot: Bool](_ReductionOperation):
    comptime Inputs = _FloatReductionInputs
    comptime Value = _FloatArgument
    comptime Result = _RoundedBinary
    var context: Optional[ArithmeticContext]
    var target: ArithmeticContext
    var name: String
    var flags: Int
    var direction: Int
    var steps: Int128
    var fail_after_step: Int
    var fail: Bool

    def __init__(
        out self,
        name: String,
        context: Optional[ArithmeticContext],
        fail_after_step: Int = -1,
        fail: Bool = False,
    ) raises:
        self.context = context
        self.target = ArithmeticContext()
        self.name = name
        self.flags = 0
        self.direction = 0
        self.steps = 0
        self.fail_after_step = fail_after_step
        self.fail = fail

    def validate(self, inputs: Self.Inputs) raises:
        comptime if Self.dot:
            if inputs.length() != inputs.right.value().length:
                raise Error(
                    String(
                        "Cannot compute ",
                        self.name,
                        " for lengths ",
                        inputs.length(),
                        " and ",
                        inputs.right.value().length,
                        (
                            "; use equal-length batches. No"
                            " broadcasting is performed. The destination is"
                            " unchanged."
                        ),
                    )
                )

    def prepare(mut self, inputs: Self.Inputs) raises:
        var format: Optional[FloatFormat] = None
        for index in range(inputs.length()):
            var left = _dot_metadata(inputs.left, index).format
            var right: Optional[FloatFormat] = None
            comptime if Self.dot:
                right = _dot_metadata(inputs.right.value(), index).format
            try:
                var current = _merge_float_formats(
                    left, right, context=self.context
                )
                format = _merge_float_formats(
                    format, current, context=self.context
                )
            except error:
                raise Error(String(self.name, " element ", index, ": ", error))
        if not format:
            var right: Optional[FloatFormat] = None
            comptime if Self.dot:
                right = inputs.right.value().format()
            format = _merge_float_formats(
                inputs.left.format(), right, context=self.context
            )
        self.target = (
            self.context.value() if self.context else ArithmeticContext(
                _format_of=format.value()
            )
        )

    def identity(self) raises -> Self.Value:
        return _float_argument(Float.zero(context=self.target))

    def record(
        mut self,
        result: _RoundedBinary,
        stage: StaticString,
        start: Int,
        end: Int,
    ) raises -> Self.Value:
        if self.steps == Int128(self.fail_after_step):
            raise Error(
                String(
                    "Injected ",
                    self.name,
                    " ",
                    stage,
                    " failure at logical range [",
                    start,
                    ", ",
                    end,
                    (
                        "); retry the operation. The destination is"
                        " unchanged."
                    ),
                )
            )
        self.steps += 1
        self.flags |= result.status._flags
        self.direction = result.status._direction
        return _float_argument(Float(_rounded=result))

    def element(mut self, inputs: Self.Inputs, index: Int) raises -> Self.Value:
        comptime if not Self.dot:
            return inputs.left.value(index)
        else:
            var a = _metadata_argument(inputs.left, index)
            var b = _metadata_argument(inputs.right.value(), index)
            if a.value.kind == 1 and b.value.kind == 1:
                a = inputs.left.value(index)
                b = inputs.right.value().value(index)
            var result: _RoundedBinary
            try:
                result = _float_operation(a, b, 2, self.target)
            except error:
                raise Error(
                    String(
                        self.name, " product at element ", index, ": ", error
                    )
                )
            return self.record(result, "product", index, index + 1)

    def combine(
        mut self, left: Self.Value, right: Self.Value, start: Int, end: Int
    ) raises -> Self.Value:
        var result: _RoundedBinary
        try:
            result = _float_operation(left, right, 0, self.target)
        except error:
            raise Error(
                String(
                    self.name,
                    " addition at logical range [",
                    start,
                    ", ",
                    end,
                    "): ",
                    error,
                )
            )
        return self.record(result, "addition", start, end)

    def singleton(mut self, value: Self.Value) raises -> Self.Value:
        var result: _RoundedBinary
        try:
            result = Float._rounded_input(value.value, self.target)
        except error:
            raise Error(String(self.name, " singleton at element 0: ", error))
        return self.record(result, "singleton", 0, 1)

    def finish(mut self, var value: Self.Value) raises -> Self.Result:
        # Identity, combine and singleton already produce the target format.
        var source = value.value
        var result = _RoundedBinary(
            source.kind,
            source.negative,
            source.numerator,
            Int(
                source.scale + Int128(self.target.format().precision())
            ) if source.kind
            == 1 else 0,
            self.target.format(),
            NumericStatus._make(self.flags, self.direction),
        )
        return _finish_round(result, self.target, self.fail)
