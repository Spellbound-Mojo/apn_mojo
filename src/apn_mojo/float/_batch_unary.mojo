"""Native Float sign results and direct classification masks."""

from ..batch._tensor import _Tensor
from ..batch.mask import Mask
from ..integer.value import Integer
from ._batch_storage import _FloatInput
from ._batch_ops import _FloatUnaryJob, _float_math_checkpoint
from ..batch._parallel import _map_tensor, _parallel_mask


@always_inline
def _float_sign_value(job: _FloatUnaryJob, index: Int) raises -> Integer:
    ref value = job.source.element(index)
    if value.is_nan():
        raise Error(String(
            "Cannot take the sign of NaN at batch element ", index,
            "; check is_nan() before requesting a numerical sign."
            " The destination is unchanged.",
        ))
    _float_math_checkpoint(index, job.fail)
    return Integer(value.sign())


def _native_float_sign(source: _FloatInput, fail: Int) raises -> _Tensor[Integer]:
    return _map_tensor[_float_sign_value](_FloatUnaryJob(source, 0, 0, fail), source.selection.count)


@always_inline
def _float_query_value(job: _FloatUnaryJob, index: Int) raises -> Bool:
    ref value = job.source.element(index)
    var operation = job.operation
    if operation == 0:
        return value.signbit()
    if operation == 1:
        return value.is_zero()
    if operation == 2:
        return value.is_finite()
    if operation == 3:
        return value.is_infinite()
    return value.is_nan()


def _native_float_query(source: _FloatInput, operation: Int) raises -> Mask:
    return _parallel_mask[_float_query_value](_FloatUnaryJob(source, operation, 0, -1), source.selection.count)
