"""Exact Rational execution over retained inputs and unpublished native results."""

from ..batch._parallel import _map_tensor, _parallel_mask
from ..batch._tensor import _Tensor
from ..batch.mask import Mask
from ._batch_storage import _RationalInput
from .value import Rational
from ..integer.value import Integer


@fieldwise_init
struct _RationalInputs(ImplicitlyCopyable):
    var a: _RationalInput
    var b: _RationalInput
    var broadcast_a: Bool
    var broadcast_b: Bool

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.a = move.a^
        self.b = move.b^
        self.broadcast_a = move.broadcast_a
        self.broadcast_b = move.broadcast_b

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def length(self) -> Int:
        return self.b.selection.count if self.broadcast_a else self.a.selection.count

    def check_shape(self, operation: Int) raises:
        if (
            self.broadcast_a
            or self.broadcast_b
            or self.a.selection.count == self.b.selection.count
        ):
            return
        var name = (
            "add" if operation
            == 0 else "subtract" if operation
            == 1 else "multiply" if operation
            == 2 else "divide" if operation
            == 3 else "compare" if operation
            == 4 else "raise powers of"
        )
        raise Error(
            String(
                "Cannot ", name, " batches of lengths ", self.a.selection.count,
                " and ", self.b.selection.count,
                "; use equal lengths or a scalar operand.",
                "" if operation == 4 else " The destination is unchanged.",
            )
        )

    def validate(self, operation: Int) raises:
        self.check_shape(operation)
        if operation == 3:
            for i in range(self.length()):
                if self.b.is_zero(0 if self.broadcast_b else i):
                    raise Error(String(
                        "Cannot divide Rational batch at element ", i,
                        ": divisor is 0; use a nonzero divisor. The destination is unchanged.",
                    ))
        elif operation == 5:
            for i in range(self.length()):
                if (self.a.is_zero(0 if self.broadcast_a else i)
                    and self.b.sign(0 if self.broadcast_b else i) < 0):
                    raise Error(String(
                        "Cannot raise zero Rational to a negative power at element ", i,
                        "; use a nonzero base or a nonnegative exponent. The destination is unchanged.",
                    ))


def _rational_math_checkpoint(index: Int, fail: Int) raises:
    if index == fail:
        raise Error(String(
            "Injected Rational unary/power failure at element ", index,
            "; inputs are unchanged. Retry the operation.",
        ))


@fieldwise_init
struct _RationalJob(Copyable, Movable):
    var inputs: _RationalInputs
    var operation: Int
    var fail: Int


@always_inline
def _rational_value(
    inputs: _RationalInputs, operation: Int, fail: Int, index: Int
) raises -> Rational:
    var a = inputs.a.value(0 if inputs.broadcast_a else index)
    var right = 0 if inputs.broadcast_b else index
    var result: Rational
    if operation == 5:
        result = a ** inputs.b.component(False, right)
        _rational_math_checkpoint(index, fail)
    else:
        var b = inputs.b.value(right)
        if operation == 0:
            result = a + b
        elif operation == 1:
            result = a - b
        elif operation == 2:
            result = a * b
        else:
            result = a / b
        if index == fail:
            raise Error(String(
                "Injected Rational arithmetic failure at element ", index,
                "; inputs are unchanged. Retry the operation.",
            ))
    return result


@always_inline
def _rational_element(job: _RationalJob, index: Int) raises -> Rational:
    return _rational_value(job.inputs, job.operation, job.fail, index)


def _native_rational_arithmetic(
    inputs: _RationalInputs, operation: Int, fail: Int = -1,
) raises -> _Tensor[Rational]:
    inputs.validate(operation)
    return _map_tensor[_rational_element](
        _RationalJob(inputs, operation, fail), inputs.length(),
    )


@fieldwise_init
struct _RationalUnaryJob(Copyable, Movable):
    var input: _RationalInput
    var operation: Int
    var fail: Int


@always_inline
def _rational_unary_value[integer_result: Bool](
    job: _RationalUnaryJob, index: Int,
) raises -> Integer if integer_result else Rational:
    comptime Result = Integer if integer_result else Rational
    var operation = job.operation
    var result: Result
    comptime if integer_result:
        if operation == 2:
            result = rebind[Result](Integer(job.input.sign(index)))
        else:
            var value = job.input.value(index)
            result = rebind[Result](
                value.floor() if operation == 3 else value.ceil() if operation == 4 else value.trunc()
            )
    else:
        var value = job.input.value(index)
        result = rebind[Result](-value if operation == 0 else abs(value))
    _rational_math_checkpoint(index, job.fail)
    return result^


def _native_rational_unary[integer_result: Bool](
    input: _RationalInput, operation: Int, fail: Int = -1,
) raises -> _Tensor[Integer if integer_result else Rational]:
    return _map_tensor[_rational_unary_value[integer_result]](
        _RationalUnaryJob(input, operation, fail), input.selection.count,
    )


@fieldwise_init
struct _RationalOrderJob(Copyable, Movable):
    var inputs: _RationalInputs
    var operation: Int


@always_inline
def _rational_order(job: _RationalOrderJob, index: Int) raises -> Bool:
    var a = job.inputs.a.value(0 if job.inputs.broadcast_a else index)
    var b = job.inputs.b.value(0 if job.inputs.broadcast_b else index)
    var operation = job.operation
    if operation == 0:
        return a == b
    if operation == 1:
        return a != b
    var order = a._compare(b)
    return (
        order < 0 if operation == 2 else order <= 0 if operation == 3
        else order > 0 if operation == 4 else order >= 0
    )


def _native_rational_comparison(inputs: _RationalInputs, operation: Int) raises -> Mask:
    inputs.check_shape(4)
    return _parallel_mask[_rational_order](_RationalOrderJob(inputs, operation), inputs.length())
