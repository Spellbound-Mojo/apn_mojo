"""Certified signed integral Complex powers through the shared mapping executor."""

from ..batch._tensor import _Tensor
from ..batch._parallel import _map_tensor
from .context import _ComplexContextArgument
from ._batch_ops import (
    _ComplexInputs,
    _native_complex_parts,
    _complex_destination_context,
)
from .value import Complex
from ._power import _pow_complex


@fieldwise_init
struct _ComplexPowerJob(Copyable, Movable):
    var inputs: _ComplexInputs
    var context: _ComplexContextArgument
    var fail: Int
    var fail_component: Int
    var destination_format: Bool


def _complex_power_value(job: _ComplexPowerJob, index: Int) raises -> Complex:
    var value = job.inputs.a.value(index)
    var target = job.context
    if job.destination_format:
        target = _ComplexContextArgument(_complex_destination_context(value, job.context))
    var count = job.inputs.b.value(index).real.value
    return _native_complex_parts(_pow_complex(
        value, -count.numerator if count.negative else count.numerator,
        target, fail_component=job.fail_component if index == job.fail else -1,
    ))


def _native_complex_power(
    inputs: _ComplexInputs,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *, fail: Int = -1, fail_component: Int = 1,
    destination_format: Bool = False,
) raises -> _Tensor[Complex]:
    inputs.check_shape()
    return _map_tensor[_complex_power_value](
        _ComplexPowerJob(inputs, context, fail, fail_component, destination_format), inputs.length(),
        diagnostic="Complex batch element ",
    )
