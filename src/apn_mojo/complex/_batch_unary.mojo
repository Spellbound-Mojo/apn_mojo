"""Native Complex sign transforms and direct classification masks."""

from ..batch.mask import Mask
from ..batch._parallel import _map_tensor, _parallel_mask
from ..float._batch_ops import _float_math_checkpoint
from ._batch_storage import _ComplexInput
from ..batch._tensor import _Tensor
from .value import Complex


@fieldwise_init
struct _ComplexUnaryJob(Copyable, Movable):
    var source: _ComplexInput
    var operation: Int
    var fail: Int
    var fail_component: Int


@always_inline
def _complex_unary_value(job: _ComplexUnaryJob, index: Int) raises -> Complex:
    var value = job.source.value(index)
    if job.fail_component == 0 or job.fail_component == 1:
        _float_math_checkpoint(index, job.fail)
    if job.operation == 1:
        value = -value
    elif job.operation == 2:
        value = value.conjugate()
    return value


@always_inline
def _complex_query_value(job: _ComplexUnaryJob, index: Int) raises -> Bool:
    var value = job.source.value(index)
    var operation = job.operation
    if operation == 0:
        return value.is_zero()
    if operation == 1:
        return value.is_finite()
    if operation == 2:
        return value.is_infinite()
    return value.is_nan()


def _native_complex_unary(
    source: _ComplexInput, operation: Int,
    *, fail: Int = -1, fail_component: Int = 1,
) raises -> _Tensor[Complex]:
    return _map_tensor[_complex_unary_value](
        _ComplexUnaryJob(source, operation, fail, fail_component), source.real.selection.count,
    )


def _native_complex_query(source: _ComplexInput, operation: Int) raises -> Mask:
    return _parallel_mask[_complex_query_value](
        _ComplexUnaryJob(source, operation, -1, -1), source.real.selection.count,
    )
