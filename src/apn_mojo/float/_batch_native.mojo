"""Float batch operators: arithmetic, powers and signs over shared execution
with transactional result staging."""

from ..batch._tensor import _Tensor
from ..batch._parallel import _map_tensor
from .value import Float
from .context import ArithmeticContext
from ._rounding import _RoundedBinary
from ._arithmetic import _float_operation
from ._functions import _destination_context, _pow_float
from ._batch_ops import _FloatInputs, _FloatUnaryJob, _float_math_checkpoint
from ._batch_storage import _FloatInput


@always_inline
def _float_rounded(
    inputs: _FloatInputs, operation: Int, context: Optional[ArithmeticContext],
    destination_format: Bool, initial_work: Int, index: Int,
) raises -> _RoundedBinary:
    var target = context
    if operation <= 2 and inputs.a.floating and inputs.b.floating:
        # Stored Floats are borrowed, as the scalar operators borrow theirs.
        ref a = inputs.a.floating.value().element(0 if inputs.a.broadcast else index)
        ref b = inputs.b.floating.value().element(0 if inputs.b.broadcast else index)
        if destination_format:
            target = _destination_context(a.format(), context)
        if operation == 2:
            return Float._product(a, b, target)
        return Float._sum(a, b, operation == 1, target)
    var a = inputs.a.value(index)
    var b = inputs.b.value(index)
    if destination_format:
        target = _destination_context(a.format.value(), context)
    if operation == 4:
        var count = b.value.numerator
        return _pow_float(
            a, -count if b.value.negative else count, target,
            initial_work=initial_work,
        )
    return _float_operation(a, b, operation, target)


@fieldwise_init
struct _FloatJob(Copyable, Movable):
    # The operator's borrowed inputs outlive every element of the run.
    var inputs: Pointer[_FloatInputs, MutUntrackedOrigin]
    var operation: Int
    var context: Optional[ArithmeticContext]
    var destination_format: Bool
    var initial_work: Int
    var fail: Int


@always_inline
def _float_element(job: _FloatJob, index: Int) raises -> Float:
    var value = Float(_rounded=_float_rounded(
        job.inputs[], job.operation, job.context, job.destination_format,
        job.initial_work, index,
    ))
    _float_math_checkpoint(index, job.fail)
    return value^


def _native_float_arithmetic(
    inputs: _FloatInputs, operation: Int, context: Optional[ArithmeticContext],
    *, fail: Int = -1, destination_format: Bool = False,
    initial_work: Int = 0,
) raises -> _Tensor[Float]:
    inputs.check_shape()
    var job = _FloatJob(
        Pointer(to=inputs).unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin](),
        operation, context, destination_format, initial_work, fail,
    )
    return _map_tensor[_float_element](
        job, inputs.length(), diagnostic="Float batch element ",
    )


@always_inline
def _float_unary_value(job: _FloatUnaryJob, index: Int) raises -> Float:
    ref value = job.source.element(index)
    _float_math_checkpoint(index, job.fail)
    if job.operation == 1:
        return -value
    return value


def _native_float_unary(source: _FloatInput, operation: Int, fail: Int) raises -> _Tensor[Float]:
    return _map_tensor[_float_unary_value](
        _FloatUnaryJob(source, operation, 0, fail), source.selection.count,
        diagnostic="Float batch element ",
    )
