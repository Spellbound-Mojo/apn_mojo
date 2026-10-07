"""Selected Float values to native Integer results with checked conversion."""

from ..batch._tensor import _Tensor
from ..integer.value import Integer
from ..integer._limits import _MAX_RESULT_BITS
from ._batch_storage import _FloatInput
from ._batch_ops import _FloatUnaryJob, _float_math_checkpoint
from ..batch._parallel import _map_tensor
from ._functions import _float_integer
from ._arithmetic import _float_argument


def _float_integral_value(job: _FloatUnaryJob, index: Int) raises -> Integer:
    var value = _float_argument(job.source.element(index)).value
    if not job.mode:
        var result = _float_integer(value, job.operation)
        _float_math_checkpoint(index, job.fail)
        return result
    # Preserve batch validation and checkpoint precedence before arithmetic.
    if value.kind >= 2:
        raise Error(String(
            "Cannot convert a nonfinite Float to Integer at batch element ",
            index,
            "; check is_finite() before floor(), ceil(), trunc(), or exact"
            " conversion. The destination is unchanged.",
        ))
    if value.kind == 1:
        var bits = value.numerator.magnitude_bit_length()
        if value.scale >= 0:
            if value.scale > Int128(_MAX_RESULT_BITS - bits):
                raise Error(String(
                    "Cannot convert Float to Integer at batch element ", index,
                    ": the result exceeds addressable storage; keep the value"
                    " as Float or scale it down. The destination is unchanged.",
                ))
        elif job.operation == 3:
            var count = bits if -value.scale >= Int128(bits) else Int(-value.scale)
            var remainder = False
            for word in range(count // 32):
                remainder |= value.numerator._word(word) != 0
            if count % 32:
                remainder |= (
                    value.numerator._word(count // 32)
                    & ((UInt32(1) << UInt32(count % 32)) - 1)
                ) != 0
            if remainder:
                raise Error(String(
                    "Cannot convert Float exactly to Integer at batch element ",
                    index,
                    ": the value has a fractional part; keep it as Float or"
                    " explicitly choose floor(), ceil(), or trunc(). The"
                    " destination is unchanged.",
                ))
    _float_math_checkpoint(index, job.fail)
    return _float_integer(value, job.operation)


def _native_float_integral(
    source: _FloatInput, operation: Int, mode: Int = 1, fail: Int = -1,
) raises -> _Tensor[Integer]:
    return _map_tensor[_float_integral_value](
        _FloatUnaryJob(source, operation, mode, fail), source.selection.count,
    )
