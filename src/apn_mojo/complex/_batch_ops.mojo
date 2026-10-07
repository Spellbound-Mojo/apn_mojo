"""Scalar Complex arithmetic, exact comparisons and transactional mapped results."""

from ..batch._parallel import _map_tensor, _parallel_mask, _parallel_ready
from ..batch._tensor import _Tensor
from ..batch.mask import Mask
from ..float._batch_ops import _FloatOperand, _float_argument_order
from ..float._functions import _destination_context
from ..float.context import ArithmeticContext, FloatFormat
from ..float.value import Float
from ..float._rounding import _RoundedBinary
from ._batch_storage import _ComplexInput
from ._input import _ComplexArgument
from ._arithmetic import _complex_operation, _target
from .context import ComplexContext, _ComplexContextArgument
from .value import Complex


def _complex_destination_context(
    value: _ComplexArgument, context: _ComplexContextArgument
) raises -> ComplexContext:
    var real: Optional[ArithmeticContext] = None
    var imag: Optional[ArithmeticContext] = None
    if context:
        real = context.value().real()
        imag = context.value().imag()
    return ComplexContext(
        real=_destination_context(value.real.format.value(), real),
        imag=_destination_context(value.imag.format.value(), imag),
    )


@fieldwise_init
struct _ComplexOperand(ImplicitlyCopyable):
    var components: Optional[_ComplexInput]
    var real: Optional[_FloatOperand]
    var scalar: Optional[_ComplexArgument]
    var length: Int
    var broadcast: Bool
    # A shared Complex scalar itself, so additions borrow it (scalar is its record).
    var held: Optional[Complex]

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.components = move.components^
        self.real = move.real^
        self.scalar = move.scalar^
        self.length = move.length
        self.broadcast = move.broadcast
        self.held = move.held^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def value(self, index: Int) -> _ComplexArgument:
        if self.scalar:
            return self.scalar.value()
        if self.components:
            return _ComplexArgument(
                self.components.value().value(0 if self.broadcast else index)
            )
        return _ComplexArgument(self.real.value().value(index))

    def formats(self, index: Int = -1) -> _ComplexArgument:
        if self.scalar:
            return self.scalar.value()
        var result = _ComplexArgument(Int(0))
        if self.components:
            var parts = self.components.value()
            result.complex = True
            result.real.format = (
                parts.real.default_format() if index
                < 0 else parts.real.format_at(index)
            )
            result.imag.format = (
                parts.imag.default_format() if index
                < 0 else parts.imag.format_at(index)
            )
        elif self.real.value().floating:
            var source = self.real.value().floating.value()
            result.real.format = (
                source.default_format() if index
                < 0 else source.format_at(index)
            )
        elif self.real.value().scalar:
            result = _ComplexArgument(self.real.value().scalar.value())
        return result


@fieldwise_init
struct _ComplexInputs(ImplicitlyCopyable):
    var a: _ComplexOperand
    var b: _ComplexOperand

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
                    "Cannot operate on Complex batches of lengths ",
                    self.a.length,
                    " and ",
                    self.b.length,
                    (
                        "; use equal lengths or a scalar operand. The"
                        " destination is unchanged."
                    ),
                )
            )

    def target(
        self, index: Int, context: _ComplexContextArgument
    ) raises -> ComplexContext:
        return _target(self.a.formats(index), self.b.formats(index), context)


def _native_complex_parts(parts: Tuple[_RoundedBinary, _RoundedBinary]) -> Complex:
    return Complex(_real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1]))


def _complex_values(operand: _ComplexOperand) -> Optional[_Tensor[Complex]]:
    """The stored values behind a Complex batch operand; both parts view them."""
    if operand.components:
        return operand.components.value().real.complex_values
    return None


@always_inline
def _stored_complex(
    operand: _ComplexOperand, values: _Tensor[Complex], index: Int,
) -> ref[values] Complex:
    ref selection = operand.components.value().real.selection
    return values._read(selection.index(0 if operand.broadcast else index))


def _real_format_at(
    operand: _ComplexOperand, values: Optional[_Tensor[Complex]], index: Int,
) -> FloatFormat:
    if values:
        return _stored_complex(operand, values.value(), index)._real.format()
    return operand.scalar.value().real.format.value()


struct _ComplexJob(Copyable, Movable):
    var inputs: _ComplexInputs
    var left: Optional[_Tensor[Complex]]
    var right: Optional[_Tensor[Complex]]
    var operation: Int
    var context: _ComplexContextArgument
    # Plain flags: `left and right` would evaluate to a copied Optional tensor,
    # an ArcPointer refcount pair on every element.
    var stored: Bool
    var additive: Bool
    # A stored batch plus or minus a shared Complex scalar, on either side.
    var right_scalar_sum: Bool
    var left_scalar_sum: Bool

    def __init__(
        out self, inputs: _ComplexInputs, left: Optional[_Tensor[Complex]],
        right: Optional[_Tensor[Complex]], operation: Int,
        context: _ComplexContextArgument,
    ):
        self.inputs = inputs
        self.left = left
        self.right = right
        self.operation = operation
        self.context = context
        self.stored = Bool(left) and Bool(right)
        self.additive = self.stored and operation <= 1
        self.right_scalar_sum = Bool(left) and Bool(inputs.b.held) and operation <= 1
        self.left_scalar_sum = Bool(right) and Bool(inputs.a.held) and operation <= 1


@always_inline
def _complex_value(
    inputs: _ComplexInputs, left: Optional[_Tensor[Complex]],
    right: Optional[_Tensor[Complex]], operation: Int,
    context: _ComplexContextArgument, stored: Bool, additive: Bool,
    right_scalar_sum: Bool, left_scalar_sum: Bool, index: Int,
) raises -> Complex:
    if right_scalar_sum:
        return Complex._sum(
            _stored_complex(inputs.a, left.value(), index),
            inputs.b.held.value(), operation == 1, context,
        )
    if left_scalar_sum:
        return Complex._sum(
            inputs.a.held.value(),
            _stored_complex(inputs.b, right.value(), index),
            operation == 1, context,
        )
    if additive:
        return Complex._sum(
            _stored_complex(inputs.a, left.value(), index),
            _stored_complex(inputs.b, right.value(), index),
            operation == 1, context,
        )
    if stored:
        return _native_complex_parts(_complex_operation(
            _ComplexArgument(_stored_complex(inputs.a, left.value(), index)),
            _ComplexArgument(_stored_complex(inputs.b, right.value(), index)),
            operation, context,
        ))
    if left:
        return _native_complex_parts(_complex_operation(
            _ComplexArgument(_stored_complex(inputs.a, left.value(), index)),
            inputs.b.scalar.value(), operation, context,
        ))
    return _native_complex_parts(_complex_operation(
        inputs.a.scalar.value(),
        _ComplexArgument(_stored_complex(inputs.b, right.value(), index)),
        operation, context,
    ))


@always_inline
def _complex_element(job: _ComplexJob, index: Int) raises -> Complex:
    return _complex_value(
        job.inputs, job.left, job.right, job.operation, job.context,
        job.stored, job.additive, job.right_scalar_sum, job.left_scalar_sum, index,
    )


@always_inline
def _complex_pair_value(x: Complex, y: Complex, operation: Int) raises -> Complex:
    if operation <= 1:
        return Complex._sum(x, y, operation == 1, _ComplexContextArgument())
    return _native_complex_parts(_complex_operation(
        _ComplexArgument(x), _ComplexArgument(y), operation, _ComplexContextArgument(),
    ))


@fieldwise_init
struct _ComplexPairJob(Copyable, Movable):
    var left: _Tensor[Complex]
    var right: _Tensor[Complex]
    var operation: Int


@always_inline
def _complex_pair(job: _ComplexPairJob, index: Int) raises -> Complex:
    return _complex_pair_value(job.left._read(index), job.right._read(index), job.operation)


def _complex_pairs(
    left: _Tensor[Complex], right: _Tensor[Complex], operation: Int,
) raises -> Optional[_Tensor[Complex]]:
    """Two stored Complex runs without a context, element by element.

    No format pre-check: on any failure this returns None and the caller runs
    the general path, which raises that path's error.
    """
    var count = left._layout.size
    try:
        return _map_tensor[_complex_pair](_ComplexPairJob(left, right, operation), count)
    except:
        return None


def _complex_pair_arithmetic(
    inputs: _ComplexInputs,
    left: Optional[_Tensor[Complex]],
    right: Optional[_Tensor[Complex]],
    operation: Int,
    context: _ComplexContextArgument,
) raises -> _Tensor[Complex]:
    """Complex batches and scalars: borrow each stored value and do exactly the
    scalar operator's work, instead of rebuilding values from components."""
    var count = inputs.length()
    # Keep the sequential format preflight. Eligible parallel runs report the
    # same lowest mismatch through the kernel; the target computation supplies
    # the established diagnostic for incompatible exponent bounds.
    if not context and not _parallel_ready(count):
        for index in range(count):
            var a = _real_format_at(inputs.a, left, index)
            var b = _real_format_at(inputs.b, right, index)
            if a.emin() != b.emin() or a.emax() != b.emax():
                try:
                    _ = inputs.target(index, context)
                except error:
                    raise Error(String("Complex batch element ", index, ": ", error))
    return _map_tensor[_complex_element](
        _ComplexJob(inputs, left, right, operation, context), count,
        diagnostic="Complex batch", index_prefix=" element ",
    )


@fieldwise_init
struct _ComplexGeneralJob(Copyable, Movable):
    var inputs: _ComplexInputs
    var operation: Int
    var context: _ComplexContextArgument
    var fail: Int
    var fail_component: Int
    var destination_format: Bool


def _complex_general_value(job: _ComplexGeneralJob, index: Int) raises -> Complex:
    var left = job.inputs.a.value(index)
    var target = job.context
    if job.destination_format:
        target = _ComplexContextArgument(_complex_destination_context(left, job.context))
    return _native_complex_parts(_complex_operation(
        left, job.inputs.b.value(index), job.operation, target,
        fail_component=job.fail_component if index == job.fail else -1,
    ))


def _native_complex_arithmetic(
    inputs: _ComplexInputs, operation: Int,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *, fail: Int = -1, fail_component: Int = 1,
    destination_format: Bool = False,
) raises -> _Tensor[Complex]:
    inputs.check_shape()
    var left = _complex_values(inputs.a)
    var right = _complex_values(inputs.b)
    if not destination_format and fail < 0 and (left or inputs.a.scalar) and (right or inputs.b.scalar):
        return _complex_pair_arithmetic(inputs, left, right, operation, context)
    # Preserve whole-input format preflight before arithmetic or injected failures.
    for index in range(inputs.length()):
        try:
            if destination_format:
                _ = _complex_destination_context(inputs.a.formats(index), context)
            else:
                _ = inputs.target(index, context)
        except error:
            raise Error(String("Complex batch element ", index, ": ", error))
    return _map_tensor[_complex_general_value](
        _ComplexGeneralJob(inputs, operation, context, fail, fail_component, destination_format),
        inputs.length(), diagnostic="Complex batch element ",
    )


@fieldwise_init
struct _ComplexEqualJob(Copyable, Movable):
    var inputs: _ComplexInputs
    var left: _Tensor[Complex]
    var right: _Tensor[Complex]
    var negate: Bool


@always_inline
def _complex_equal(job: _ComplexEqualJob, index: Int) raises -> Bool:
    ref x = _stored_complex(job.inputs.a, job.left, index)
    ref y = _stored_complex(job.inputs.b, job.right, index)
    var equal = x._real._compare_float(y._real) == 0 and x._imag._compare_float(y._imag) == 0
    return equal != job.negate


@fieldwise_init
struct _ComplexMixedEqualJob(Copyable, Movable):
    var inputs: _ComplexInputs
    var negate: Bool


def _complex_mixed_equal(job: _ComplexMixedEqualJob, index: Int) raises -> Bool:
    var a = job.inputs.a.value(index)
    var b = job.inputs.b.value(index)
    var equal = (
        _float_argument_order(a.real, b.real) == 0
        and _float_argument_order(a.imag, b.imag) == 0
    )
    return equal != job.negate


def _native_complex_comparison(inputs: _ComplexInputs, operation: Int) raises -> Mask:
    if operation > 1:
        raise Error(
            "Complex values have no ordinary ordering; compare equality or"
            " explicitly choose a real quantity such as magnitude."
        )
    inputs.check_shape()
    var left = _complex_values(inputs.a)
    var right = _complex_values(inputs.b)
    if left and right:
        return _parallel_mask[_complex_equal](
            _ComplexEqualJob(inputs, left.value(), right.value(), operation == 1), inputs.length(),
        )
    return _parallel_mask[_complex_mixed_equal](
        _ComplexMixedEqualJob(inputs, operation == 1), inputs.length(),
    )
