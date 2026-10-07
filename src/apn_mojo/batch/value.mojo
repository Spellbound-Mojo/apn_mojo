"""Numeric batches: shared values read through a runtime layout of any rank."""

from std.builtin.builtin_slice import ContiguousSlice, StridedSlice
from std.builtin.rebind import downcast
from std.memory import ArcPointer
from std.sys import size_of
from std.os import abort
from std.iter import (
    Iterable,
    IterableOwned,
    Iterator,
    StopIteration,
    iter,
    next,
)

from ._json import _write_bounded_batch_records, _batch_owner_budget
from ._iterator import _iterator_tensor, _same_element
from ._comparison import _ExactBatchComparison
from ._values import _Values
from ._tensor import _Tensor, _TensorSequence, _write_axis
from ._parallel import _parallel_ready, _map_tensor, _map_values
from ._digits import _parse_digit_texts
from ._layout import _Layout, _FlatLayout, _broadcast_shapes, _linear_index, _list_ints
from ._array import _ArrayData, _default_format
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..complex._input import _ComplexArgument
from ..complex.context import _ComplexContextArgument
from ..complex._batch_protocol import _ComplexBatchOperand
from ..complex._batch_unary import _native_complex_unary, _native_complex_query
from ..complex._batch_power import _native_complex_power
from ..complex._batch_ops import (
    _complex_pairs,
    _ComplexOperand,
    _ComplexInputs,
    _native_complex_arithmetic,
    _native_complex_comparison,
)
from ..complex._format import _ComplexFormats
from ..complex._arithmetic import _target as _complex_target
from ..complex._batch_storage import _ComplexInput, _native_complex_iterator
from ..float._arithmetic import _FloatArgument, _float_argument
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ._natives import _native_element
from ..float._batch_protocol import _FloatBatchOperand
from ..float._batch_integral import _native_float_integral
from ..float._batch_native import _native_float_arithmetic, _native_float_unary
from ..float._batch_unary import _native_float_sign, _native_float_query
from ..float._batch_ops import (
    _float_scalar_operand,
    _FloatOperand,
    _FloatInputs,
    _native_float_comparison,
    _float_argument_order,
)
from ..float._batch_storage import _FloatInput, _native_float_iterator
from ..rational._batch_ops import (
    _RationalInputs,
    _native_rational_arithmetic,
    _native_rational_comparison,
    _native_rational_unary,
)
from ..rational._batch_storage import _RationalInput, _native_rational_iterator
from ..common._traits import _BatchElement, _MapArgument
from ._families import _Arithmetic
from ..integer._batch_native import (
    _NativeIntegerInputs,
    _native_integer_arithmetic,
    _native_integer_comparison,
)
from ..integer._batch_native import _length_mismatch
from ..integer._operand import _IntegerOperand, _LiteralOperand
from ..common._sizes import _checked_count, _checked_sum
from ..integer._json import _json_decimal_bound, _read_integer_json
from ..rational._batch_json import _parse_rational_batch_json, _write_rational_batch_json
from ..integer._text import _format_bound
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..common._json import _batch_json_header, _join_records
from ._slices import _Selection
from .mask import Mask
from ..integer._tiles import _WIDTH



struct _BatchIterator[T: ImplicitlyCopyable & Deinitable](
    ImplicitlyCopyable, Iterable, IterableOwned, Iterator
):
    """A batch's elements in row-major order, read from one strided run.

    It carries the batch's default formats, so a batch built from what remains
    keeps them.
    """

    comptime Element = Self.T
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _selection: _Selection
    var _position: Int
    var _native: _Tensor[Self.T]
    var _real_format: FloatFormat
    var _imag_format: FloatFormat

    def __init__(
        out self, selection: _Selection, position: Int, native: _Tensor[Self.T],
        real_format: FloatFormat, imag_format: FloatFormat,
    ):
        self._selection = selection
        self._position = position
        self._native = native
        self._real_format = real_format
        self._imag_format = imag_format

    def __iter__(ref self) -> Self:
        return self

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> Self.T:
        if self._position == self._selection.count:
            raise StopIteration()
        var index = self._selection.index(self._position)
        self._position += 1
        return self._native._read(index)

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self._selection.count - self._position
        return remaining, Optional(remaining)

    def _rest(self) raises -> Batch[Self.T]:
        """The elements not read yet, as a batch with the source's formats."""
        var values = List[Self.T](capacity=_checked_count(self._selection.count - self._position, size_of[Self.T]()))
        for i in range(self._position, self._selection.count):
            values.append(self._native._read(self._selection.index(i)))
        var count = len(values)
        var result = Batch[Self.T](_tensor=_Tensor[Self.T](values^, [count]))
        result._real_format = self._real_format
        result._imag_format = self._imag_format
        return result^


def _same_family_tensor[T: ImplicitlyCopyable & Deinitable, I: Iterator](var cursor: I) raises -> _Tensor[T]:
    """The elements of an iterator of the batch's own family, for a family
    with no conversions from others."""
    return _iterator_tensor[T, I, _same_element[T, I]](cursor^)


def _integer_iterator_element[I: Iterator](var value: I.Element) raises -> Integer:
    comptime assert conforms_to(I.Element, ImplicitlyCopyable)
    comptime assert (I.Element == Integer or I.Element == Int or
                    I.Element == Int8 or I.Element == UInt8 or
                    I.Element == Int16 or I.Element == UInt16 or
                    I.Element == Int32 or I.Element == UInt32 or
                    I.Element == Int64 or I.Element == UInt64), (
        "Integer batches require Integer or native integer elements; convert other elements to Integer explicitly."
    )
    comptime if I.Element == Integer:
        return rebind_var[Integer](value^)
    elif I.Element == Int:
        return Integer(rebind_var[Int](value^))
    elif I.Element == Int8:
        return Integer(rebind_var[Int8](value^))
    elif I.Element == UInt8:
        return Integer(rebind_var[UInt8](value^))
    elif I.Element == Int16:
        return Integer(rebind_var[Int16](value^))
    elif I.Element == UInt16:
        return Integer(rebind_var[UInt16](value^))
    elif I.Element == Int32:
        return Integer(rebind_var[Int32](value^))
    elif I.Element == UInt32:
        return Integer(rebind_var[UInt32](value^))
    elif I.Element == Int64:
        return Integer(rebind_var[Int64](value^))
    else:
        return Integer(rebind_var[UInt64](value^))


def _native_integer_iterator[I: Iterator](var cursor: I) raises -> _Tensor[Integer]:
    return _iterator_tensor[Integer, I, _integer_iterator_element[I]](cursor^)


def _native_integer_inputs(
    a: Batch[Integer], b: Batch[Integer], operation: Int,
    broadcast_a: Bool = False, broadcast_b: Bool = False,
    *, family: Int = 0,
) raises -> _NativeIntegerInputs:
    # Check sequence lengths before reading either input.
    if not broadcast_a and not broadcast_b and len(a) != len(b):
        _length_mismatch(family, operation, len(a), len(b))
    return _NativeIntegerInputs(a._tensor_view(), b._tensor_view(), broadcast_a, broadcast_b)


def _native_integer_result(
    a: Batch[Integer], b: Batch[Integer], operation: Int,
    broadcast_a: Bool = False, broadcast_b: Bool = False,
    *, family: Int = 0, fail: Int = -1, label: StaticString = "staging",
) raises -> Batch[Integer]:
    var inputs = _native_integer_inputs(a, b, operation, broadcast_a, broadcast_b, family=family)
    return Batch[Integer](
        _tensor=_native_integer_arithmetic(inputs, family, operation, fail=fail, label=label),
        _like=_result_layout(a, b),
    )


def _integer_batch[
    T: ImplicitlyCopyable
](value: T) raises -> Batch[Integer]:
    """An Integer operand as a batch; callers flag a scalar as broadcast."""
    comptime assert T != Batch[Float], (
        "This operation requires Integer elements; use Float arithmetic for"
        " real values instead of implicitly narrowing a Float batch."
    )
    comptime if T == Integer:
        return Batch[Integer]([rebind[Integer](value)])
    else:
        comptime assert T == Batch[Integer], (
            "This operation requires Integer elements; Rational batches"
            " support +, -, *, / and comparisons."
        )
        return rebind[Batch[Integer]](value)


def _rational_operand[T: ImplicitlyCopyable](value: T) raises -> _RationalInput:
    comptime assert T != Batch[Float], (
        "This operation requires exact Integer/Rational inputs; use Float"
        " arithmetic instead of implicitly narrowing a Float batch."
    )
    comptime if T == Rational:
        return _RationalInput(
            rebind[Rational](value), _Selection(0, 1, 1)
        )
    elif T == Integer:
        return _RationalInput(
            Rational(rebind[Integer](value)), _Selection(0, 1, 1)
        )
    elif T == Batch[Rational]:
        return rebind[Batch[Rational]](value)._rational_input()
    else:
        comptime assert (
            T == Batch[Integer]
        ), "Expected an exact batch or scalar"
        ref owner = rebind[Batch[Integer]](value)
        return _RationalInput(Rational(), _Selection(0, len(owner), 1), owner._run())


def _rational_inputs[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B) raises -> _RationalInputs:
    return _RationalInputs(
        _rational_operand(a),
        _rational_operand(b),
        A == Integer or A == Rational,
        B == Integer or B == Rational,
    )


def _rational_result[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A, b: B, operation: Int, fail: Int = -1, broadcast_b: Bool = False,
    *, flat: Bool = False,
) raises -> Batch[Rational]:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, operation, broadcast_b)
        if plan.broadcast:
            return _shaped_result(_rational_result(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, fail, broadcast_b, flat=True), plan.shape)
        return _shaped_result(_rational_result(a, b, operation, fail, broadcast_b, flat=True), plan.shape)
    var inputs = _rational_inputs(a, b)
    inputs.broadcast_b |= broadcast_b
    return Batch[Rational](_tensor=_native_rational_arithmetic(inputs, operation, fail), _like=_result_layout(a, b))


def _exact_result[
    T: ImplicitlyCopyable & Deinitable,
    A: ImplicitlyCopyable,
    B: ImplicitlyCopyable,
](a: A, b: B, operation: Int) raises -> Batch[T]:
    comptime if T == Complex:
        return rebind_var[Batch[T]](_complex_result(a, b, operation))
    elif T == Float:
        return rebind_var[Batch[T]](_float_result(a, b, operation))
    elif T == Rational:
        return rebind_var[Batch[T]](_rational_result(a, b, operation))
    else:
        comptime assert T == Integer, _NO_ARITHMETIC
        return rebind_var[Batch[T]](_integer_result[0, False](a, b, operation))


def _rational_unary[
    integer_result: Bool, A: ImplicitlyCopyable
](
    value: A,
    operation: Int,
    fail: Int = -1,
) raises -> Batch[
    Integer if integer_result else Rational
]:
    var shape = _operand_shape(value, value, value)
    return _shape_like(Batch[Integer if integer_result else Rational](
        _tensor=_native_rational_unary[integer_result](_rational_operand(value), operation, fail)
    ), shape)


def _rational_power[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](base: A, exponent: B, fail: Int = -1,) raises -> Batch[Rational]:
    var shape = _operand_shape(base, exponent, base)
    # The Integer adapter enforces the exponent family without narrowing it.
    var inputs = _rational_inputs(base, _integer_batch(exponent))
    inputs.broadcast_b = B == Integer
    return _shape_like(Batch[Rational](_tensor=_native_rational_arithmetic(inputs, 5, fail)), shape)


def _exact_comparison[
    rational: Bool, A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, operation: Int, broadcast_b: Bool = False, *, flat: Bool = False) raises -> Mask:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, -1, broadcast_b)
        if plan.broadcast:
            return _exact_comparison[rational](_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, broadcast_b, flat=True).reshape(plan.shape)
        return _exact_comparison[rational](a, b, operation, broadcast_b, flat=True).reshape(plan.shape)
    comptime if A == Batch[Complex] or B == Batch[Complex]:
        return _complex_comparison(a, b, operation)
    elif A == Batch[Float] or B == Batch[Float] or A == Float or B == Float:
        return _float_comparison(a, b, operation, broadcast_b=broadcast_b)
    elif rational:
        var inputs = _rational_inputs(a, b)
        inputs.broadcast_b |= broadcast_b
        return _native_rational_comparison(inputs, operation)
    else:
        return _integer_comparison(a, b, operation)


def _integer_result[
    family: Int, unary: Bool, A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, operation: Int, *, flat: Bool = False) raises -> Batch[Integer]:
    comptime assert family == 0 or family == 1
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, operation if family == 0 else 4)
        if plan.broadcast:
            return _shaped_result(_integer_result[family, unary](_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, flat=True), plan.shape)
        return _shaped_result(_integer_result[family, unary](a, b, operation, flat=True), plan.shape)
    return _native_integer_result(_integer_batch(a), _integer_batch(b), operation, A == Integer, B == Integer, family=family)


def _integer_comparison[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, operation: Int, *, flat: Bool = False) raises -> Mask:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, -1)
        if plan.broadcast:
            return _integer_comparison(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, flat=True).reshape(plan.shape)
        return _integer_comparison(a, b, operation, flat=True).reshape(plan.shape)
    return _native_integer_comparison(
        _native_integer_inputs(_integer_batch(a), _integer_batch(b), operation, A == Integer, B == Integer, family=4), operation,
    )


def _integer_division_part[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, part: Int) raises -> Batch[Integer]:
    """The floor quotient (part 0) or remainder (part 1) of each pair."""
    var shape = _operand_shape(a, b, a)
    var inputs = _native_integer_inputs(
        _integer_batch(a), _integer_batch(b), part, A == Integer, B == Integer, family=2,
    )
    return _shape_like(Batch[Integer](_tensor=_native_integer_arithmetic(inputs, 2, part)), shape)


def _float_operand[T: ImplicitlyCopyable](value: T) raises -> _FloatOperand:
    comptime if T == Batch[Float]:
        ref source = rebind[Batch[Float]](value)
        return _FloatOperand(
            source._float_input(), None, None, len(source), False
        )
    elif T == Batch[Integer] or T == Batch[Rational]:
        var source = _rational_operand(value)
        return _FloatOperand(None, source, None, source.selection.count, False)
    elif T == _FloatArgument:
        # A scalar-left operator passes its Float as a record; rebuild the
        # Float once so every element borrows it.
        ref argument = rebind[_FloatArgument](value)
        var float = Float._from_argument(argument)
        if float:
            return _float_scalar_operand(float.value())
        return _FloatOperand(None, None, argument, 1, True)
    elif T == Float:
        return _float_scalar_operand(rebind[Float](value))
    else:
        return _FloatOperand(None, None, _float_argument(value), 1, True)


def _complex_operand[T: ImplicitlyCopyable](value: T) raises -> _ComplexOperand:
    comptime if T == Batch[Complex]:
        ref source = rebind[Batch[Complex]](value)
        return _ComplexOperand(
            source._complex_input(), None, None, len(source), False, None
        )
    elif T == Complex:
        return _ComplexOperand(
            None, None, _ComplexArgument(value), 1, True, rebind[Complex](value)
        )
    elif T == _ComplexArgument:
        ref argument = rebind[_ComplexArgument](value)
        var held: Optional[Complex] = None
        if argument.complex:
            var real = Float._from_argument(argument.real)
            var imag = Float._from_argument(argument.imag)
            if real and imag:
                held = Complex(_real=real.value(), _imag=imag.value())
        return _ComplexOperand(None, None, argument, 1, True, held^)
    else:
        var source = _float_operand(value)
        return _ComplexOperand(
            None, source, None, source.length, source.broadcast, None
        )


def _complex_result[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    operation: Int,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    fail: Int = -1,
    fail_component: Int = 1,
    *,
    flat: Bool = False,
) raises -> Batch[Complex]:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, operation)
        if plan.broadcast:
            return _shaped_result(_complex_result(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, context, fail, fail_component, flat=True), plan.shape)
        return _shaped_result(_complex_result(a, b, operation, context, fail, fail_component, flat=True), plan.shape)
    comptime if A == Batch[Complex] and B == Batch[Complex]:
        # Two stored batches: borrow both runs and call the scalar operator,
        # as Float does. On failure the general path raises its error.
        ref left = rebind[Batch[Complex]](a)
        ref right = rebind[Batch[Complex]](b)
        var count = len(left)
        if not context and fail < 0 and count and count == len(right):
            var values = _complex_pairs(left._run(), right._run(), operation)
            if values:
                return _complex_from_tensor(
                    values.value(), _ComplexFormats(FloatFormat(), FloatFormat()), _result_layout(a, b),
                )
    return _complex_arithmetic(
        _ComplexInputs(_complex_operand(a), _complex_operand(b)),
        operation, context, fail, fail_component, _result_layout(a, b),
    )


# Operand adapters stay generic; the arithmetic body compiles once.
def _complex_arithmetic(
    var inputs: _ComplexInputs,
    operation: Int,
    context: _ComplexContextArgument,
    fail: Int,
    fail_component: Int,
    like: Optional[ArcPointer[_FlatLayout]] = None,
) raises -> Batch[Complex]:
    inputs.check_shape()
    var formats = _ComplexFormats(FloatFormat(), FloatFormat())
    if context or not inputs.length():
        var target = inputs.target(-1, context)
        formats = _ComplexFormats(target.real().format(), target.imag().format())
    var values = _native_complex_arithmetic(
        inputs, operation, context, fail=fail, fail_component=fail_component,
    )
    return _complex_from_tensor(values, formats, like)


def _complex_comparison[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, operation: Int, *, flat: Bool = False) raises -> Mask:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, -1)
        if plan.broadcast:
            return _complex_comparison(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, flat=True).reshape(plan.shape)
        return _complex_comparison(a, b, operation, flat=True).reshape(plan.shape)
    var inputs = _ComplexInputs(_complex_operand(a), _complex_operand(b))
    return _native_complex_comparison(inputs, operation)


def _complex_unary[
    A: ImplicitlyCopyable
](
    value: A,
    operation: Int,
    fail: Int = -1,
    fail_component: Int = 1,
) raises -> Batch[Complex]:
    var shape = _operand_shape(value, value, value)
    return _shape_like(_complex_unary_of(_complex_operand(value), operation, fail, fail_component), shape)


def _complex_unary_of(
    operand: _ComplexOperand,
    operation: Int,
    fail: Int,
    fail_component: Int,
) raises -> Batch[Complex]:
    var source = operand.components.value()
    var values = _native_complex_unary(
        source, operation, fail=fail, fail_component=fail_component,
    )
    return _complex_from_tensor(values, _ComplexFormats(
        source.real.default_format(), source.imag.default_format(),
    ))


def _complex_component[
    A: ImplicitlyCopyable
](value: A, imaginary: Bool,) raises -> Batch[Float]:
    var shape = _operand_shape(value, value, value)
    var pair = _complex_operand(value).components.value()
    var source = pair.imag if imaginary else pair.real
    return _shape_like(_float_from_tensor(
        _native_float_unary(source, 0, -1), source.default_format(),
    ), shape)


def _complex_query[
    A: ImplicitlyCopyable
](value: A, operation: Int) raises -> Mask:
    var shape = _operand_shape(value, value, value)
    return _shape_like(_native_complex_query(_complex_operand(value).components.value(), operation), shape)


def _complex_count_operand[
    B: ImplicitlyCopyable
](exponent: B) raises -> _ComplexOperand:
    comptime assert B == Integer or B == Batch[Integer], (
        "Complex powers require Integer exponents; use an Integer scalar or"
        " batch."
    )
    var count = _complex_operand(_integer_batch(exponent))
    count.broadcast = B == Integer
    return count


def _complex_power[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    base: A,
    exponent: B,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    fail: Int = -1,
    fail_component: Int = 1,
) raises -> Batch[Complex]:
    comptime assert A == Complex or A == _ComplexArgument or A == Batch[Complex]
    var shape = _operand_shape(base, exponent, base)
    var inputs = _ComplexInputs(_complex_operand(base), _complex_count_operand(exponent))
    inputs.check_shape()
    var defaults = inputs.a.formats()
    var formats = _ComplexFormats(
        context.value().real().format() if context else defaults.real.format.value(),
        context.value().imag().format() if context else defaults.imag.format.value(),
    )
    var values = _native_complex_power(inputs, context, fail=fail, fail_component=fail_component)
    return _shape_like(_complex_from_tensor(values, formats), shape)


def _approx_result[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, operation: Int) raises -> Batch[
    Complex if A == Batch[Complex] or B == Batch[Complex] else Float
]:
    comptime if A == Batch[Complex] or B == Batch[Complex]:
        return rebind_var[
            Batch[Complex if A == Batch[Complex] or B == Batch[Complex] else Float]
        ](_complex_result(a, b, operation))
    else:
        return rebind_var[
            Batch[Complex if A == Batch[Complex] or B == Batch[Complex] else Float]
        ](_float_result(a, b, operation))


def _float_tensor[T: ImplicitlyCopyable](value: T) raises -> _Tensor[Float]:
    comptime assert T == Batch[Float]
    return rebind[Batch[Float]](value)._tensor_view()


def _complex_from_tensor(
    values: _Tensor[Complex], formats: _ComplexFormats, like: Optional[ArcPointer[_FlatLayout]] = None,
) raises -> Batch[Complex]:
    var result = Batch[Complex](_tensor=values, _like=like)
    result._real_format = formats.real()
    result._imag_format = formats.imag()
    return result


def _float_from_tensor(
    values: _Tensor[Float], default_format: FloatFormat, like: Optional[ArcPointer[_FlatLayout]] = None,
) raises -> Batch[Float]:
    var result = Batch[Float](_tensor=values, _like=like)
    result._real_format = default_format
    return result


def _float_result[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    operation: Int,
    fail: Int = -1,
    broadcast_b: Bool = False,
    context: Optional[ArithmeticContext] = None,
    *,
    flat: Bool = False,
) raises -> Batch[Float]:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, operation, broadcast_b)
        if plan.broadcast:
            return _shaped_result(_float_result(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, fail, broadcast_b, context, flat=True), plan.shape)
        return _shaped_result(_float_result(a, b, operation, fail, broadcast_b, context, flat=True), plan.shape)
    var inputs = _FloatInputs(_float_operand(a), _float_operand(b))
    inputs.b.broadcast |= broadcast_b
    inputs.check_shape()
    comptime if A == Batch[Float] and B == Batch[Float]:
        if fail < 0:
            return _float_tensor_arithmetic(_float_tensor(a), _float_tensor(b), operation, context, inputs, _result_layout(a, b))
    var values = _native_float_arithmetic(inputs, operation, context, fail=fail)
    return _float_from_tensor(
        values,
        context.value().format() if context else inputs.default_format() if not inputs.length() else FloatFormat(),
        _result_layout(a, b),
    )


@always_inline
def _float_pair_value(
    x: Float, y: Float, operation: Int, context: Optional[ArithmeticContext]
) raises -> Float:
    if operation == 2:
        return Float(_rounded=Float._product(x, y, context))
    if operation <= 1:
        return Float(_rounded=Float._sum(x, y, operation == 1, context))
    return Float._from_arguments(x, y, operation, context)


@fieldwise_init
struct _FloatPairJob(Copyable, Movable):
    var left: _Tensor[Float]
    var right: _Tensor[Float]
    var operation: Int
    var context: Optional[ArithmeticContext]


@always_inline
def _float_pair(job: _FloatPairJob, index: Int) raises -> Float:
    return _float_pair_value(job.left._read(index), job.right._read(index), job.operation, job.context)


def _float_tensor_arithmetic(
    left: _Tensor[Float], right: _Tensor[Float], operation: Int,
    context: Optional[ArithmeticContext], inputs: _FloatInputs,
    like: Optional[ArcPointer[_FlatLayout]] = None,
) raises -> Batch[Float]:
    if left._layout.size != right._layout.size:
        raise Error("Incompatible batch shapes; align trailing axes with equal or singleton dimensions.")
    var values = _map_tensor[_float_pair](
        _FloatPairJob(left, right, operation, context), left._layout.size,
        diagnostic="Float batch", index_prefix=" element ",
    )
    return _float_from_tensor(
        values^,
        context.value().format() if context else inputs.default_format() if not inputs.length() else FloatFormat(),
        like,
    )


def _float_comparison[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A, b: B, operation: Int, broadcast_b: Bool = False,
    *, flat: Bool = False,
) raises -> Mask:
    if not flat and _any_nd(a, b):
        var plan = _prepare_binary_shape(a, b, -1, broadcast_b)
        if plan.broadcast:
            return _float_comparison(_broadcast_operand(a, plan.shape), _broadcast_operand(b, plan.shape), operation, broadcast_b, flat=True).reshape(plan.shape)
        return _float_comparison(a, b, operation, broadcast_b, flat=True).reshape(plan.shape)
    comptime if A == Batch[Complex] or B == Batch[Complex]:
        return _complex_comparison(a, b, operation)
    return _float_compare(_FloatInputs(_float_operand(a), _float_operand(b)), operation, broadcast_b)


def _float_compare(var inputs: _FloatInputs, operation: Int, broadcast_b: Bool) raises -> Mask:
    inputs.b.broadcast |= broadcast_b
    return _native_float_comparison(inputs, operation)


def _float_unary[
    A: ImplicitlyCopyable
](value: A, operation: Int, fail: Int = -1) raises -> Batch[Float]:
    var shape = _operand_shape(value, value, value)
    var source = _float_operand(value).floating.value()
    return _shape_like(_float_from_tensor(_native_float_unary(source, operation, fail), source.default_format()), shape)


def _float_sign[
    A: ImplicitlyCopyable
](value: A, fail: Int = -1) raises -> Batch[Integer]:
    var shape = _operand_shape(value, value, value)
    return _shape_like(Batch[Integer](_tensor=_native_float_sign(
        _float_operand(value).floating.value(), fail,
    )), shape)


def _float_query[
    A: ImplicitlyCopyable
](value: A, operation: Int) raises -> Mask:
    var shape = _operand_shape(value, value, value)
    return _shape_like(_native_float_query(_float_operand(value).floating.value(), operation), shape)


def _float_count_operand[
    B: ImplicitlyCopyable
](exponent: B) raises -> _FloatOperand:
    comptime assert B == Integer or B == Batch[Integer], (
        "Float powers require Integer exponents; use an Integer scalar or"
        " Batch[Integer]."
    )
    var operand = _float_operand(_integer_batch(exponent))
    operand.broadcast = B == Integer
    return operand


def _float_integral[
    A: ImplicitlyCopyable
](value: A, operation: Int, fail: Int = -1) raises -> Batch[
    Integer
]:
    var shape = _operand_shape(value, value, value)
    return _shape_like(Batch[Integer](_tensor=_native_float_integral(
        _float_operand(value).floating.value(), operation, 1, fail,
    )), shape)


def _float_power[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    base: A,
    exponent: B,
    context: Optional[ArithmeticContext] = None,
    fail: Int = -1,
    initial_work: Int = 0,
) raises -> Batch[Float]:
    var shape = _operand_shape(base, exponent, base)
    var inputs = _FloatInputs(_float_operand(base), _float_count_operand(exponent))
    inputs.check_shape()
    var values = _native_float_arithmetic(inputs, 4, context, fail=fail, initial_work=initial_work)
    var result = _float_from_tensor(
        values,
        context.value().format() if context else inputs.default_format() if not inputs.length() else FloatFormat(),
    )
    return _shape_like(result^, shape)


trait _BatchShape(ImplicitlyCopyable):
    """A batch of any family; its rank and shape are runtime values."""
    comptime Element: ImplicitlyCopyable & Deinitable

    def ndim(self) -> Int: ...

    def shape(self) -> List[Int]: ...

    def _array(self) raises -> _ArrayData: ...

    def _flat(self) -> Self: ...


def _scalar_formats[T: ImplicitlyCopyable & Deinitable](mut data: _ArrayData, value: T):
    """A rank-zero batch built from one scalar keeps that scalar's format."""
    comptime if T == Float:
        data.real_format = rebind[Float](value).format()
    elif T == Complex:
        ref number = rebind[Complex](value)
        data.real_format = number._real.format()
        data.imag_format = number._imag.format()


def _set_formats[T: ImplicitlyCopyable & Deinitable](
    mut owner: Batch[T], real: FloatFormat, imag: FloatFormat,
):
    owner._real_format = real
    owner._imag_format = imag


def _quoted_decimal(values: Batch[Integer], index: Int) raises -> String:
    return String('"', values[index]._decimal(), '"')


def _append_integer_record(input: Batch[Integer], i: Int, mut result: String, mut budget: _ConversionBudget) raises:
    budget.output(2 + Int(i != 0), i)
    var value = input._json_value(i, budget)
    var digits = value._format_with_budget(10, False, False, budget, i)
    _ = _checked_count(
        _checked_sum(
            result.byte_length(), _checked_sum(digits.byte_length(), 5)
        ),
        1,
    )
    if i:
        budget.grow_string(result, 1, i)
        result += ","
    budget.grow_string(result, _checked_sum(digits.byte_length(), 2), i)
    result.write('"', digits, '"')


def _batch_from[T: ImplicitlyCopyable & Deinitable](data: _ArrayData) raises -> Batch[T]:
    """A typed owner over an untyped array; it shares the buffer and layout."""
    var owner = Batch[T](_owner=data.owner[T](), _layout=data.layout)
    _set_formats(owner, data.real_format, data.imag_format)
    return owner^


def _coordinates(shape: List[Int], indices: List[Int]) raises -> Int:
    if len(indices) != len(shape):
        raise Error("Supply one integer coordinate per axis, or use at(axis, index) to select an axis.")
    return _linear_index(_list_ints(shape), _list_ints(indices), _FlatLayout(shape.copy()).size)


def _is_nd[V: ImplicitlyCopyable](value: V) -> Bool:
    comptime if conforms_to(V, _BatchShape):
        return value.ndim() != 1
    else:
        return False


def _any_nd[A: ImplicitlyCopyable, B: ImplicitlyCopyable](a: A, b: B) -> Bool:
    return _is_nd(a) or _is_nd(b)


def _same_shape[A: ImplicitlyCopyable, B: ImplicitlyCopyable](
    a: A, b: B, scalar_b: Bool = False,
) -> Optional[List[Int]]:
    """The shared shape when flat kernels apply; None when shapes must broadcast.

    Equal shapes, or a scalar operand, let every vector kernel run unchanged
    on the row-major data; the result then takes the shared shape.
    """
    comptime if conforms_to(A, _BatchShape) and conforms_to(B, _BatchShape):
        if scalar_b or a.shape() == b.shape():
            return a.shape()
        return None
    elif conforms_to(A, _BatchShape):
        return a.shape()
    elif conforms_to(B, _BatchShape):
        return b.shape()
    else:
        return None


def _layout_of[A: ImplicitlyCopyable](a: A) -> Optional[ArcPointer[_FlatLayout]]:
    """A batch's layout, of any element family, or None for a scalar."""
    comptime if conforms_to(A, _BatchShape):
        return rebind[Batch[downcast[A, _BatchShape].Element]](a)._layout
    else:
        return None


def _result_layout[A: ImplicitlyCopyable, B: ImplicitlyCopyable](a: A, b: B) -> Optional[ArcPointer[_FlatLayout]]:
    return _shared_layout(_layout_of(a), _layout_of(b))


@no_inline
def _shared_layout(
    var left: Optional[ArcPointer[_FlatLayout]], var right: Optional[ArcPointer[_FlatLayout]],
) -> Optional[ArcPointer[_FlatLayout]]:
    """An operand layout that a fresh row-major result may share."""
    if left and left.value()[].contiguous and left.value()[].offset == 0:
        return left^
    return right^


def _shaped_result[T: ImplicitlyCopyable & Deinitable](var result: Batch[T], shape: List[Int]) raises -> Batch[T]:
    result._keep_shape(shape)
    return result^


def _operand_shape[A: ImplicitlyCopyable, B: ImplicitlyCopyable, C: ImplicitlyCopyable](
    a: A, b: B, c: C,
) raises -> Optional[List[Int]]:
    """The shared shape of batch operands when any is shaped, else None.

    Elementwise functions run on row-major data, so every batch operand must
    have the same shape; scalars apply to every element.
    """
    if not (_is_nd(a) or _is_nd(b) or _is_nd(c)):
        return None
    var shape = Optional[List[Int]]()
    comptime if conforms_to(A, _BatchShape):
        shape = a.shape()
    comptime if conforms_to(B, _BatchShape):
        if shape and shape.value() != b.shape():
            _check_broadcast_operation(4)
        shape = b.shape()
    comptime if conforms_to(C, _BatchShape):
        if shape and shape.value() != c.shape():
            _check_broadcast_operation(4)
        shape = c.shape()
    return shape^


def _shape_like[T: ImplicitlyCopyable & Deinitable](var result: Batch[T], shape: Optional[List[Int]]) raises -> Batch[T]:
    if shape:
        result._keep_shape(shape.value())
    return result^


def _shape_like(var result: Mask, shape: Optional[List[Int]]) raises -> Mask:
    if shape:
        return result.reshape(shape.value())
    return result^


def _vector_mask[V: ImplicitlyCopyable](values: V, mask: Mask) raises -> Mask where conforms_to(V, _BatchShape):
    """A mask over a batch's row-major values; a shaped mask must match exactly."""
    if mask.ndim() == 1 and values.ndim() == 1:
        return mask
    if mask.shape() != values.shape():
        raise Error("A mask must have the batch's shape; masks select row-major values and never broadcast. The destination is unchanged.")
    return mask.reshape([mask.size()])


def _check_broadcast_operation(operation: Int) raises:
    if operation > 3:
        raise Error("Batches of different shapes broadcast only for +, -, *, / and comparisons; use equal shapes or a scalar for this operation.")


comptime _NO_ARITHMETIC = (
    "Batch operators apply to Integer, Rational, Float and Complex elements;"
    " map the family's scalar functions with vmap for other families."
)

def _shaped_integer[V: ImplicitlyCopyable](value: V) -> Integer:
    comptime if V == Integer:
        return rebind[Integer](value)
    elif V == Int:
        return Integer(rebind[Int](value))
    elif V == Int8:
        return Integer(rebind[Int8](value))
    elif V == UInt8:
        return Integer(rebind[UInt8](value))
    elif V == Int16:
        return Integer(rebind[Int16](value))
    elif V == UInt16:
        return Integer(rebind[UInt16](value))
    elif V == Int32:
        return Integer(rebind[Int32](value))
    elif V == UInt32:
        return Integer(rebind[UInt32](value))
    elif V == Int64:
        return Integer(rebind[Int64](value))
    elif V == UInt64:
        return Integer(rebind[UInt64](value))
    else:
        comptime assert False, "Use an apn_mojo number or a native integer through 64 bits. Convert text or Bool explicitly."


comptime _Broadcast[V: ImplicitlyCopyable] = (
    Batch[downcast[V, _BatchShape].Element] if conforms_to(V, _BatchShape)
    else downcast[V, ImplicitlyCopyable & Deinitable]
)


def _shape_of[V: ImplicitlyCopyable](value: V) -> List[Int]:
    """A batch's shape; a scalar broadcasts like a rank-zero batch."""
    comptime if conforms_to(V, _BatchShape):
        return value.shape()
    else:
        return List[Int]()


def _broadcast_target[A: ImplicitlyCopyable, B: ImplicitlyCopyable](a: A, b: B) raises -> List[Int]:
    return _broadcast_shapes(_shape_of(a), _shape_of(b))


def _broadcast_operand[V: ImplicitlyCopyable](value: V, shape: List[Int]) raises -> _Broadcast[V]:
    """A zero-copy view of a batch operand at the broadcast shape; scalars pass through.

    Broadcast operands then take the family kernels' equal-shape path, so
    contexts and diagnostics behave exactly as for vectors.
    """
    comptime if conforms_to(V, _BatchShape):
        return rebind_var[_Broadcast[V]](_batch_from[V.Element](value._array().broadcast_to(shape.copy())))
    else:
        return rebind[_Broadcast[V]](value)


@fieldwise_init
struct _BinaryShape(Copyable, Movable):
    var shape: List[Int]
    var broadcast: Bool


def _prepare_binary_shape[A: ImplicitlyCopyable, B: ImplicitlyCopyable](
    a: A, b: B, operation: Int, scalar_b: Bool = False,
) raises -> _BinaryShape:
    """Plan shaped operands before a family's flat kernel runs.

    Callers enter here when at least one operand is not rank one; vector
    operators keep their equal-length rule. A scalar RHS may be represented
    by a one-element batch. Comparisons pass -1 as the operation code.
    """
    var shape = _same_shape(a, b, scalar_b)
    if shape:
        return _BinaryShape(shape.take(), False)
    _check_broadcast_operation(operation)
    return _BinaryShape(_broadcast_target(a, b), True)


def _update_source[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable](
    destination: Batch[T], rhs: V,
) raises -> _Broadcast[V]:
    """A compound update's source at the destination shape; it never expands the destination."""
    var target = destination.shape()
    if _broadcast_shapes(target, _shape_of(rhs)) != target:
        raise Error("Compound updates may broadcast their source but cannot expand the destination; use a source whose shape broadcasts to the destination's. The destination is unchanged.")
    return _broadcast_operand(rhs, target)


def _check_update_shape[T: ImplicitlyCopyable & Deinitable, V: AnyType](
    destination: Batch[T], rhs: V,
) raises:
    """Compound updates keep the destination shape and never broadcast N-D operands."""
    comptime if conforms_to(V, _BatchShape):
        if (destination.ndim() != 1 or rhs.ndim() != 1) and destination.shape() != rhs.shape():
            raise Error("Compound updates on shaped batches need an operand of the same shape or a scalar; the destination is unchanged.")


# ---------------------------------------------------------------- operators
# Operand normalization and promotion shared by every batch operator.

comptime _NativeInteger[V: AnyType] = (
    V == Int or V == Int8 or V == UInt8 or V == Int16 or V == UInt16
    or V == Int32 or V == UInt32 or V == Int64 or V == UInt64
)
comptime _NativeReal[V: AnyType] = V == Float16 or V == BFloat16 or V == Float32 or V == Float64
# Native integers become Integer; every other operand passes through.
comptime _OperandType[V: ImplicitlyCopyable & Deinitable] = Integer if _NativeInteger[V] else V
comptime _ElementOf[V: ImplicitlyCopyable & Deinitable] = (
    downcast[V, _BatchShape].Element if conforms_to(V, _BatchShape) else
    Integer if _NativeInteger[V] else
    Float if _NativeReal[V] or V == _FloatArgument else
    Complex if V == _ComplexArgument else V
)
# A family's place in the numeric tower; Complex orders nothing.
comptime _FamilyOf[E: ImplicitlyCopyable & Deinitable] = (
    3 if E == Complex else 2 if E == Float else 1 if E == Rational else 0
)
# Promotion for + - *: the wider family wins.
comptime _Sum[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable] = (
    Complex if A == Complex or B == Complex else
    Float if A == Float or B == Float else
    Rational if A == Rational or B == Rational else Integer
)
# Exact division of integers is Rational.
comptime _Quotient[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable] = (
    Complex if A == Complex or B == Complex else
    Float if A == Float or B == Float else Rational
)


def _operand[V: ImplicitlyCopyable & Deinitable](value: V) -> _OperandType[V]:
    comptime assert (
        conforms_to(V, _BatchShape) or V == Integer or V == Rational or V == Float
        or V == Complex or V == _FloatArgument or V == _ComplexArgument
        or _NativeInteger[V] or _NativeReal[V]
    ), "Combine a batch with a batch, an apn_mojo number or a native number; convert other values explicitly."
    comptime if _NativeInteger[V]:
        return rebind_var[_OperandType[V]](_shaped_integer(value))
    else:
        return rebind[_OperandType[V]](value)


# Scalars that convert exactly to a batch element: natives, plus wider
# apn_mojo families for Rational and Complex. Float never rounds implicitly.
comptime _Exact[T: ImplicitlyCopyable & Deinitable, V: AnyType] = _Arithmetic[T] and (
    _NativeInteger[V]
    or (_NativeReal[V] and (T == Float or T == Complex))
    or (V == Integer and (T == Rational or T == Complex))
    or ((V == Rational or V == Float) and T == Complex)
)


def _exact_element[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](value: V) raises -> T:
    comptime assert _Exact[T, V]
    comptime if T == Complex:
        return rebind[T](Complex(_float_argument(value)))
    elif T == Float:
        return rebind[T](Float(value))
    elif T == Rational:
        return rebind[T](Rational(_shaped_integer(value)))
    else:
        return rebind[T](_shaped_integer(value))


def _compare_values[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable](
    a: A, b: B, operation: Int,
) raises -> Mask:
    comptime assert _Arithmetic[_ElementOf[A]] and _Arithmetic[_ElementOf[B]], _NO_ARITHMETIC
    comptime family = max(_FamilyOf[_ElementOf[A]], _FamilyOf[_ElementOf[B]])
    comptime if family == 3:
        return _complex_comparison(a, b, operation)
    elif family == 2:
        return _float_comparison(a, b, operation)
    else:
        return _exact_comparison[family == 1](a, b, operation)


def _power[T: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable, E: ImplicitlyCopyable](
    base: A, exponent: E,
) raises -> Batch[T]:
    comptime if T == Complex:
        return rebind_var[Batch[T]](_complex_power(base, exponent))
    elif T == Float:
        return rebind_var[Batch[T]](_float_power(base, exponent))
    elif T == Rational:
        return rebind_var[Batch[T]](_rational_power(base, exponent))
    else:
        comptime assert T == Integer, _NO_ARITHMETIC
        return rebind_var[Batch[T]](_integer_result[1, False](base, exponent, 6))


def _power_update[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    mut destination: Batch[T], rhs: V,
) raises:
    """Compound ** with Integer exponents in the destination's family."""
    comptime if conforms_to(V, _BatchShape):
        if (destination.ndim() != 1 or rhs.ndim() != 1) and destination.shape() != rhs.shape():
            _power_update(destination, _update_source(destination, rhs))
            return
    comptime if T == Complex:
        destination._update_complex_power(_operand(rhs))
    elif T == Float:
        destination._update_float_power(_operand(rhs))
    elif T == Rational:
        destination._update_rational_power(_operand(rhs))
    else:
        _integer_update[10](destination, rhs)


def _update[operation: Int, T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    mut destination: Batch[T], rhs: V,
) raises:
    """Compound + - * / in the destination's family; the shape never changes."""
    comptime if conforms_to(V, _BatchShape):
        if (destination.ndim() != 1 or rhs.ndim() != 1) and destination.shape() != rhs.shape():
            _update[operation](destination, _update_source(destination, rhs))
            return
    comptime if T == Complex:
        destination._update_complex(_operand(rhs), operation)
    elif T == Float:
        destination._update_float(_operand(rhs), operation)
    elif T == Rational:
        destination._update_rational(_operand(rhs), operation)
    elif operation == 3:
        # Reports that Integer elements cannot become Rational in place.
        destination._update_rational(_operand(rhs), operation)
    else:
        _integer_update[operation](destination, rhs)


def _integer_update[code: Int, T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    mut destination: Batch[T], rhs: V,
) raises:
    """Integer compound updates: 0-2 + - *, 3 //, 4-6 & | ^, 8-9 shifts, 10 **, 11 %."""
    comptime assert T == Integer, (
        "This Integer-only update requires Integer elements; use"
        " Rational arithmetic and integral powers for fractions"
    )
    comptime assert _ElementOf[V] == Integer, (
        "Compound updates cannot change Integer elements to another family;"
        " construct a batch of the wider family before updating."
    )
    comptime division = code == 3 or code == 11
    comptime if conforms_to(V, _BatchShape):
        if (destination.ndim() != 1 or rhs.ndim() != 1) and destination.shape() != rhs.shape():
            _integer_update[code](destination, _update_source(destination, rhs))
            return
    comptime if V == Batch[Integer]:
        var source = rebind[Batch[T]](rhs)
        comptime if code <= 2:
            destination._assign(source, code)
        elif division:
            destination._assign_division(source, Int(code == 11))
        else:
            destination._assign_bit(source, code - 4)
    else:
        var scalar = rebind[Integer](_operand(rhs))
        comptime if code <= 2:
            destination._assign_scalar(scalar, code)
        elif division:
            destination._assign_division_scalar(scalar, Int(code == 11))
        else:
            destination._assign_bit_scalar(scalar, code - 4)


struct Batch[T: ImplicitlyCopyable & Deinitable](
    ImplicitlyCopyable,
    Iterable,
    IterableOwned,
    Sized,
    Writable,
    _ComplexBatchOperand,
    _ExactBatchComparison,
    _FloatBatchOperand,
    _TensorSequence,
    _BatchShape,
    _MapArgument,
):
    """A batch of numbers of any rank, with value semantics.

    Rank and shape are run-time values, so one type serves rank-zero scalars,
    vectors, matrices and higher-rank tensors. Slices, axis selections, transposes
    and reshapes share the values without copying them, and a batch never sees a
    later update to another one: updates publish new storage.

    | Operation | Contract |
    |---|---|
    | `a + b`, `a - b`, `a * b`, `a / b` | Elementwise; shapes broadcast from the trailing axis; scalars broadcast |
    | `a // b`, `a % b`, `a ** b`, bit operations | Elementwise on equal shapes or a scalar |
    | `==`, `!=`, `<`, `<=`, `>`, `>=` | A `Mask` of the broadcast shape |
    | `values[i]`, `values[i, j]` | One element; negative indices count from the end |
    | `values[a:b:c]` | A vector over the flat row-major values, sharing them |
    | `values[mask]` | A new vector of the selected values |
    | `values[selection] = source` | A scalar fills; a sequence must match the selection's length |
    | `values += x` and the other compound forms | Keep the destination's shape and formats; unchanged on error |

    Two vectors combine only at equal lengths: a length-one vector is a sequence,
    not a scalar. Exact `/` returns a `Batch[Rational]`. A failed operation reports
    the earliest failing element and publishes nothing. Long operations run on
    the worker pool with the same results and errors.

    Named math functions are scalar; apply them with `vmap`, as in
    `vmap[apn_mojo.float.sqrt]()(values)`.

    Parameters:
        T: The element type: `Integer`, `Rational`, `Float`, `Complex` or
            `Ball`. A `Batch[Ball]` is a container, mapped with `vmap`; it has
            no operators or JSON.

    Limitations:
        A native number cannot be the left operand of a comparison; write
        `batch > native`. Batches are not hash keys. A slice keeps its whole
        source alive; copy a small slice with `Batch[T](values.to_list())`.
    """

    # Arrow-style typed array: a shared immutable value list read through a
    # runtime layout of any rank. Updates publish a new list (copy-on-write).
    var _owner: ArcPointer[_Values[Self.T]]
    # Immutable and shared by copies, slices and other selections.
    var _layout: ArcPointer[_FlatLayout]
    # Default formats for Float (real) and Complex (real/imaginary) results.
    var _real_format: FloatFormat
    var _imag_format: FloatFormat

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self._owner = move._owner^
        self._layout = move._layout^
        self._real_format = move._real_format
        self._imag_format = move._imag_format

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    comptime FloatBatch = Batch[Complex if Self.T == Complex else Float]
    comptime ComplexBatch = Batch[Complex]
    comptime Element = Self.T

    def __init__(out self, values: List[Self.T], *, shape: List[Int]) raises:
        """A batch with a shape, from row-major values.

        Args:
            values: The values in row-major order.
            shape: The dimensions, outermost first; their product must equal the value count.

        Raises:
            When the shape does not match the value count.
        """
        var data = _ArrayData.contiguous(values.copy(), shape.copy())
        if not shape:
            _scalar_formats(data, values[0])
        self = _batch_from[Self.T](data)

    def __init__[I: Iterable](out self, values: I, *, shape: List[Int]) raises where I != List[Self.T]:
        """A batch with a shape, from a finite iterable consumed once."""
        self = Self(values)._with_shape(shape)

    def __init__(out self, var values: Some[IterableOwned], *, shape: List[Int]) raises:
        """A batch with a shape, from an owned finite iterable."""
        self = Self(values^)._with_shape(shape)

    @staticmethod
    def from_iterable[I: Iterable](values: I, *, shape: List[Int]) raises -> Self:
        """A batch from a finite iterable, consumed once.

        Parameters:
            I: The iterable type; its items must convert exactly to the element type.

        Args:
            values: The values in row-major order.
            shape: The dimensions, outermost first; their product must equal the value count.

        Returns:
            The batch.

        Raises:
            When an item does not convert or the shape does not match.
        """
        comptime if I == List[Self.T]:
            return Self(rebind[List[Self.T]](values), shape=shape)
        else:
            return Self(values, shape=shape)

    @staticmethod
    def from_iterable(var values: Some[IterableOwned], *, shape: List[Int]) raises -> Self:
        """A batch with a shape from an owned finite iterable."""
        return Self(values^, shape=shape)

    @staticmethod
    def from_native(values: List[Int], *, shape: List[Int]) raises -> Self:
        """A batch from native integers.

        Args:
            values: The values in row-major order.
            shape: The dimensions, outermost first; their product must equal the value count.

        Returns:
            The batch.

        Raises:
            When the shape does not match the value count.
        """
        return Self.from_native(values)._with_shape(shape)

    @staticmethod
    def from_native[dtype: DType](
        values: List[SIMD[dtype, 1]], *, shape: List[Int],
    ) raises -> Self where (Self.T == Float or Self.T == Complex):
        """A Float or Complex batch with a shape, from native numbers."""
        return Self.from_native(values)._with_shape(shape)

    def to_native[dtype: DType](
        self, *, rounding: RoundingMode = RoundingMode.nearest_even,
    ) raises -> List[SIMD[dtype, 1]]:
        """The values as native numbers in row-major order, the counterpart of
        `from_native`; `shape()` gives the dimensions.

        To `float64`, `float32`, `float16` or `bfloat16`, each value is rounded
        once with `rounding`, subnormals included; beyond the range it becomes an
        infinity. A Ball converts through its midpoint. To an integer type, each
        value converts exactly: a value that is not whole or does not fit raises.

        Parameters:
            dtype: The native type.

        Args:
            rounding: The rounding mode for floating-point types.

        Returns:
            One native value per element.

        Raises:
            For an integer type, at the first value that is not a whole number in
            its range.
        """
        var values = self.to_list()
        var result = List[SIMD[dtype, 1]](capacity=len(values))
        for value in values:
            result.append(_native_element[dtype](value, rounding))
        return result^

    def _with_shape(self, shape: List[Int]) raises -> Self:
        var data = self._array().reshaped(shape.copy())
        if not shape:
            _scalar_formats(data, self[0])
        return _batch_from[Self.T](data)

    def ndim(self) -> Int:
        """The rank.

        Returns:
            The number of axes; zero for a rank-zero batch.
        """
        return len(self._layout[].shape)

    def shape(self) -> List[Int]:
        """The dimensions.

        Returns:
            A copy of the shape, outermost axis first.
        """
        return self._layout[].shape.copy()

    def size(self) -> Int:
        """The number of elements.

        Returns:
            The product of the dimensions; `len(values)` is the same.
        """
        return len(self)

    def item(self) raises -> Self.T:
        """The only element.

        Returns:
            The element of a batch of size one.

        Raises:
            When the size is not one.
        """
        if len(self) != 1:
            raise Error("item() requires exactly one value; select an element or a singleton batch first.")
        return self[0]

    def _array(self) raises -> _ArrayData:
        return _ArrayData(self._owner, self._layout[], self._real_format, self._imag_format)

    def _flat(self) -> Self:
        """The same values as a vector; one strided run needs no copy."""
        var result = self
        var run = self._run()
        result._owner = run._owner
        result._layout = ArcPointer(_FlatLayout.run(run._layout.size, run._layout.offset, run._layout.strides[0]))
        return result^

    def _run(self) -> _Tensor[Self.T]:
        """The row-major values as one strided run, gathered only when needed."""
        ref layout = self._layout[]
        var step = layout.progression()
        if step:
            return _Tensor[Self.T](self._owner, _Layout.run(layout.size, layout.offset, step.value()))
        var values = List[Self.T](capacity=layout.size)
        for i in range(layout.size):
            values.append(self._owner[].list[layout.position(i)])
        var count = len(values)
        return _Tensor[Self.T](ArcPointer(_Values[Self.T](values^)), _Layout.run(count, 0, 1))

    def _keep_shape(mut self, shape: List[Int]) raises:
        """Give a freshly computed result the destination's shape."""
        if self._layout[].shape == shape:
            return
        if not self._layout[].contiguous:
            self = self._flat()
        self._layout = ArcPointer(self._layout[].reshaped(shape.copy()))

    @implicit
    def __init__(out self, value: Self.T):
        """A rank-zero batch holding one value; it fills a selection when assigned."""
        var values = List[Self.T](capacity=1)
        values.append(value)
        self._owner = ArcPointer(_Values[Self.T](values^))
        # The row-major rank-zero layout, as Batch([value], shape=[]) builds.
        var layout = _FlatLayout()
        layout.scalar = False
        layout.contiguous = True
        self._layout = ArcPointer(layout^)
        self._real_format = _default_format()
        self._imag_format = _default_format()
        comptime if Self.T == Float:
            self._real_format = rebind[Float](value).format()
        elif Self.T == Complex:
            ref number = rebind[Complex](value)
            self._real_format = number._real.format()
            self._imag_format = number._imag.format()

    @implicit
    def __init__(out self, value: IntLiteral) raises:
        """A rank-zero batch from an integer literal of any width."""
        comptime if Self.T == Integer:
            self = Self(rebind[Self.T](Integer(type_of(value)())))
        elif Self.T == Float:
            # Rounds to the default format, as Float(literal) does.
            self = Self(rebind[Self.T](Float(value)))
        else:
            self = Self(_exact_element[Self.T](Integer(type_of(value)())))

    @implicit
    def __init__[V: ImplicitlyCopyable & Deinitable](out self, value: V) raises where (
        V != Self.T and _Exact[Self.T, V]
    ):
        """A rank-zero batch from a value that widens exactly to the element type."""
        self = Self(_exact_element[Self.T](value))

    def __init__(out self, *, var _owner: ArcPointer[_Values[Self.T]], var _layout: _FlatLayout):
        self._owner = _owner^
        self._layout = ArcPointer(_layout^)
        self._real_format = _default_format()
        self._imag_format = _default_format()

    def reshape(self, shape: List[Int]) raises -> Self:
        """The same values with another shape, in row-major order.

        Args:
            shape: The new dimensions; the size must not change.

        Returns:
            A batch sharing the values when they form one strided run.

        Raises:
            When the sizes differ.
        """
        return _batch_from[Self.T](self._array().reshaped(shape.copy()))

    def transpose(self, axes: List[Int]) raises -> Self:
        """Permute the axes.

        Args:
            axes: A permutation of the axes; negative entries count from the end.

        Returns:
            A batch sharing the values.

        Raises:
            When `axes` is not a permutation.
        """
        return _batch_from[Self.T](self._array().transposed(axes))

    def transpose(self) raises -> Self:
        """Reverse the axes, sharing the values."""
        var data = self._array()
        return _batch_from[Self.T](data.with_layout(data.layout.reversed_axes()))

    def at(self, axis: Int, index: Int) raises -> Self:
        """Select one index along an axis, dropping that axis.

        Args:
            axis: The axis; negative counts from the end.
            index: The index along it; negative counts from the end.

        Returns:
            A batch of one rank lower, sharing the values; `values.at(0, i)` is row `i`.

        Raises:
            When the axis or index is out of range.
        """
        return _batch_from[Self.T](self._array().at(axis, index))

    def slice(
        self, axis: Int, start: Optional[Int] = None,
        stop: Optional[Int] = None, step: Int = 1,
    ) raises -> Self:
        """Slice one axis, keeping the rank.

        Args:
            axis: The axis; negative counts from the end.
            start: The first index, as in Python slicing.
            stop: The end index, as in Python slicing.
            step: The step; not zero.

        Returns:
            A batch sharing the values.

        Raises:
            When the axis is out of range or the step is zero.
        """
        return _batch_from[Self.T](self._array().sliced(axis, StridedSlice(start, stop, step)))

    def __getitem__(self, *indices: Int) raises -> Self.T:
        var coordinates = List[Int](capacity=len(indices))
        for i in range(len(indices)):
            coordinates.append(indices[i])
        return self[_coordinates(self.shape(), coordinates)]

    def __getitem__(self, empty: Tuple[]) raises -> Self.T:
        if self.ndim() != 0:
            raise Error("Empty coordinates select a rank-zero batch; supply one coordinate per axis.")
        return self[0]

    def __setitem__(mut self, *indices: Int, var value: Self.T) raises:
        var coordinates = List[Int](capacity=len(indices))
        for i in range(len(indices)):
            coordinates.append(indices[i])
        self[_coordinates(self.shape(), coordinates)] = value^

    def __setitem__(mut self, empty: Tuple[], var value: Self.T) raises:
        if self.ndim() != 0:
            raise Error("Empty coordinates select a rank-zero batch; supply one coordinate per axis.")
        self[0] = value^

    def _tensor_view(self) raises -> _Tensor[Self.T]:
        return self._run()

    def __init__(out self, *, _tensor: _Tensor[Self.T]):
        self = Self(_tensor=_tensor, _like=None)

    def __init__(out self, *, _tensor: _Tensor[Self.T], _like: Optional[ArcPointer[_FlatLayout]]):
        """A batch over a run, sharing an operand's layout when it is that run.

        Layouts are immutable once shared, and a result shaped like its operand
        is common: sharing saves building one (three allocations).
        """
        self._owner = _tensor._owner
        ref run = _tensor._layout
        if _like and _like.value()[].is_run(run.size, run.offset, run.strides[0]):
            self._layout = _like.value()
        else:
            self._layout = ArcPointer(_FlatLayout.run(run.size, run.offset, run.strides[0]))
        self._real_format = _default_format()
        self._imag_format = _default_format()

    def _complex_power_reverse(
        self, base: _ComplexArgument
    ) raises -> Batch[Complex]:
        comptime assert (
            Self.T == Integer
        ), "Complex powers require an Integer batch exponent."
        return _complex_power(base, self)

    def _complex_reverse(
        self, lhs: _ComplexArgument, operation: Int
    ) raises -> Batch[Complex]:
        return _complex_result(lhs, self, operation)

    def _compare_complex(
        self, rhs: _ComplexArgument, operation: Int
    ) raises -> Mask:
        return _complex_comparison(self, rhs, operation)

    # Binary operators: one generic form plus literal forms each. Operands
    # normalize natives to Integer or _FloatArgument and promote by family.

    # Same-family operands keep the exact result type in generic code.
    def __add__(self, rhs: Batch[Self.T]) raises -> Batch[Self.T]:
        return _exact_result[Self.T](self, rhs, 0)

    def __add__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[_Sum[Self.T, _ElementOf[V]]]:
        return _exact_result[_Sum[Self.T, _ElementOf[V]]](self, _operand(rhs), 0)

    def __add__(self, rhs: IntLiteral) raises -> Batch[_Sum[Self.T, Integer]]:
        return _exact_result[_Sum[Self.T, Integer]](self, Integer(type_of(rhs)()), 0)

    def __add__(self, rhs: FloatLiteral) raises -> Batch[_Sum[Self.T, Float]]:
        return _exact_result[_Sum[Self.T, Float]](self, _FloatArgument(rhs), 0)

    def __radd__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[_Sum[_ElementOf[V], Self.T]]:
        return _exact_result[_Sum[_ElementOf[V], Self.T]](_operand(lhs), self, 0)

    def __radd__(self, lhs: IntLiteral) raises -> Batch[_Sum[Integer, Self.T]]:
        return _exact_result[_Sum[Integer, Self.T]](Integer(type_of(lhs)()), self, 0)

    def __radd__(self, lhs: FloatLiteral) raises -> Batch[_Sum[Float, Self.T]]:
        return _exact_result[_Sum[Float, Self.T]](_FloatArgument(lhs), self, 0)

    def __iadd__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _update[0](self, rhs)

    def __iadd__(mut self, rhs: IntLiteral) raises:
        comptime if Self.T == Integer:
            self._assign_scalar(_LiteralOperand[type_of(rhs)()](), 0)
        else:
            _update[0](self, Integer(type_of(rhs)()))

    def __iadd__(mut self, rhs: FloatLiteral) raises:
        comptime if Self.T == Complex:
            _update[0](self, _ComplexArgument(rhs))
        else:
            _update[0](self, _FloatArgument(rhs))

    # Same-family operands keep the exact result type in generic code.
    def __sub__(self, rhs: Batch[Self.T]) raises -> Batch[Self.T]:
        return _exact_result[Self.T](self, rhs, 1)

    def __sub__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[_Sum[Self.T, _ElementOf[V]]]:
        return _exact_result[_Sum[Self.T, _ElementOf[V]]](self, _operand(rhs), 1)

    def __sub__(self, rhs: IntLiteral) raises -> Batch[_Sum[Self.T, Integer]]:
        return _exact_result[_Sum[Self.T, Integer]](self, Integer(type_of(rhs)()), 1)

    def __sub__(self, rhs: FloatLiteral) raises -> Batch[_Sum[Self.T, Float]]:
        return _exact_result[_Sum[Self.T, Float]](self, _FloatArgument(rhs), 1)

    def __rsub__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[_Sum[_ElementOf[V], Self.T]]:
        return _exact_result[_Sum[_ElementOf[V], Self.T]](_operand(lhs), self, 1)

    def __rsub__(self, lhs: IntLiteral) raises -> Batch[_Sum[Integer, Self.T]]:
        return _exact_result[_Sum[Integer, Self.T]](Integer(type_of(lhs)()), self, 1)

    def __rsub__(self, lhs: FloatLiteral) raises -> Batch[_Sum[Float, Self.T]]:
        return _exact_result[_Sum[Float, Self.T]](_FloatArgument(lhs), self, 1)

    def __isub__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _update[1](self, rhs)

    def __isub__(mut self, rhs: IntLiteral) raises:
        comptime if Self.T == Integer:
            self._assign_scalar(_LiteralOperand[type_of(rhs)()](), 1)
        else:
            _update[1](self, Integer(type_of(rhs)()))

    def __isub__(mut self, rhs: FloatLiteral) raises:
        comptime if Self.T == Complex:
            _update[1](self, _ComplexArgument(rhs))
        else:
            _update[1](self, _FloatArgument(rhs))

    # Same-family operands keep the exact result type in generic code.
    def __mul__(self, rhs: Batch[Self.T]) raises -> Batch[Self.T]:
        return _exact_result[Self.T](self, rhs, 2)

    def __mul__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[_Sum[Self.T, _ElementOf[V]]]:
        return _exact_result[_Sum[Self.T, _ElementOf[V]]](self, _operand(rhs), 2)

    def __mul__(self, rhs: IntLiteral) raises -> Batch[_Sum[Self.T, Integer]]:
        return _exact_result[_Sum[Self.T, Integer]](self, Integer(type_of(rhs)()), 2)

    def __mul__(self, rhs: FloatLiteral) raises -> Batch[_Sum[Self.T, Float]]:
        return _exact_result[_Sum[Self.T, Float]](self, _FloatArgument(rhs), 2)

    def __rmul__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[_Sum[_ElementOf[V], Self.T]]:
        return _exact_result[_Sum[_ElementOf[V], Self.T]](_operand(lhs), self, 2)

    def __rmul__(self, lhs: IntLiteral) raises -> Batch[_Sum[Integer, Self.T]]:
        return _exact_result[_Sum[Integer, Self.T]](Integer(type_of(lhs)()), self, 2)

    def __rmul__(self, lhs: FloatLiteral) raises -> Batch[_Sum[Float, Self.T]]:
        return _exact_result[_Sum[Float, Self.T]](_FloatArgument(lhs), self, 2)

    def __imul__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _update[2](self, rhs)

    def __imul__(mut self, rhs: IntLiteral) raises:
        comptime if Self.T == Integer:
            self._assign_scalar(_LiteralOperand[type_of(rhs)()](), 2)
        else:
            _update[2](self, Integer(type_of(rhs)()))

    def __imul__(mut self, rhs: FloatLiteral) raises:
        comptime if Self.T == Complex:
            _update[2](self, _ComplexArgument(rhs))
        else:
            _update[2](self, _FloatArgument(rhs))

    def __truediv__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[_Quotient[Self.T, _ElementOf[V]]]:
        return _exact_result[_Quotient[Self.T, _ElementOf[V]]](self, _operand(rhs), 3)

    def __truediv__(self, rhs: IntLiteral) raises -> Batch[_Quotient[Self.T, Integer]]:
        return _exact_result[_Quotient[Self.T, Integer]](self, Integer(type_of(rhs)()), 3)

    def __truediv__(self, rhs: FloatLiteral) raises -> Batch[_Quotient[Self.T, Float]]:
        return _exact_result[_Quotient[Self.T, Float]](self, _FloatArgument(rhs), 3)

    def __rtruediv__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[_Quotient[_ElementOf[V], Self.T]]:
        return _exact_result[_Quotient[_ElementOf[V], Self.T]](_operand(lhs), self, 3)

    def __rtruediv__(self, lhs: IntLiteral) raises -> Batch[_Quotient[Integer, Self.T]]:
        return _exact_result[_Quotient[Integer, Self.T]](Integer(type_of(lhs)()), self, 3)

    def __rtruediv__(self, lhs: FloatLiteral) raises -> Batch[_Quotient[Float, Self.T]]:
        return _exact_result[_Quotient[Float, Self.T]](_FloatArgument(lhs), self, 3)

    def __itruediv__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _update[3](self, rhs)

    def __itruediv__(mut self, rhs: IntLiteral) raises:
        comptime if Self.T == Integer:
            self._assign_scalar(_LiteralOperand[type_of(rhs)()](), 3)
        else:
            _update[3](self, Integer(type_of(rhs)()))

    def __itruediv__(mut self, rhs: FloatLiteral) raises:
        comptime if Self.T == Complex:
            _update[3](self, _ComplexArgument(rhs))
        else:
            _update[3](self, _FloatArgument(rhs))

    def __eq__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        return _compare_values(self, _operand(rhs), 0)

    def __eq__(self, rhs: IntLiteral) raises -> Mask:
        return _compare_values(self, Integer(type_of(rhs)()), 0)

    def __eq__(self, rhs: FloatLiteral) raises -> Mask:
        return _compare_values(self, _FloatArgument(rhs), 0)

    def __ne__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        return _compare_values(self, _operand(rhs), 1)

    def __ne__(self, rhs: IntLiteral) raises -> Mask:
        return _compare_values(self, Integer(type_of(rhs)()), 1)

    def __ne__(self, rhs: FloatLiteral) raises -> Mask:
        return _compare_values(self, _FloatArgument(rhs), 1)

    def __lt__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        comptime assert _FamilyOf[Self.T] != 3 and _FamilyOf[_ElementOf[V]] != 3, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _operand(rhs), 2)

    def __lt__(self, rhs: IntLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, Integer(type_of(rhs)()), 2)

    def __lt__(self, rhs: FloatLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _FloatArgument(rhs), 2)

    def __le__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        comptime assert _FamilyOf[Self.T] != 3 and _FamilyOf[_ElementOf[V]] != 3, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _operand(rhs), 3)

    def __le__(self, rhs: IntLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, Integer(type_of(rhs)()), 3)

    def __le__(self, rhs: FloatLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _FloatArgument(rhs), 3)

    def __gt__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        comptime assert _FamilyOf[Self.T] != 3 and _FamilyOf[_ElementOf[V]] != 3, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _operand(rhs), 4)

    def __gt__(self, rhs: IntLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, Integer(type_of(rhs)()), 4)

    def __gt__(self, rhs: FloatLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _FloatArgument(rhs), 4)

    def __ge__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Mask:
        comptime assert _FamilyOf[Self.T] != 3 and _FamilyOf[_ElementOf[V]] != 3, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _operand(rhs), 5)

    def __ge__(self, rhs: IntLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, Integer(type_of(rhs)()), 5)

    def __ge__(self, rhs: FloatLiteral) raises -> Mask:
        comptime assert Self.T != Complex, (
            "Complex values have no ordinary ordering; use equality or"
            " explicitly compare a real quantity."
        )
        return _compare_values(self, _FloatArgument(rhs), 5)

    def __and__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, _operand(rhs), 0))

    def __and__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, Integer(type_of(rhs)()), 0))

    def __rand__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](_operand(lhs), self, 0))

    def __rand__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 0))

    def __iand__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[4](self, rhs)

    def __iand__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 0)

    def __or__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, _operand(rhs), 1))

    def __or__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, Integer(type_of(rhs)()), 1))

    def __ror__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](_operand(lhs), self, 1))

    def __ror__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 1))

    def __ior__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[5](self, rhs)

    def __ior__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 1)

    def __xor__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, _operand(rhs), 2))

    def __xor__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, Integer(type_of(rhs)()), 2))

    def __rxor__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](_operand(lhs), self, 2))

    def __rxor__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 2))

    def __ixor__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[6](self, rhs)

    def __ixor__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 2)

    def __lshift__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, _operand(rhs), 4))

    def __lshift__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, Integer(type_of(rhs)()), 4))

    def __rlshift__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](_operand(lhs), self, 4))

    def __rlshift__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 4))

    def __ilshift__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[8](self, rhs)

    def __ilshift__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 4)

    def __rshift__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, _operand(rhs), 5))

    def __rshift__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](self, Integer(type_of(rhs)()), 5))

    def __rrshift__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](_operand(lhs), self, 5))

    def __rrshift__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 5))

    def __irshift__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[9](self, rhs)

    def __irshift__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 5)

    def __floordiv__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(self, _operand(rhs), 0))

    def __floordiv__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(self, Integer(type_of(rhs)()), 0))

    def __rfloordiv__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(_operand(lhs), self, 0))

    def __rfloordiv__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(Integer(type_of(lhs)()), self, 0))

    def __ifloordiv__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[3](self, rhs)

    def __ifloordiv__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_division_scalar(_LiteralOperand[type_of(rhs)()](), 0)

    def __mod__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(self, _operand(rhs), 1))

    def __mod__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(self, Integer(type_of(rhs)()), 1))

    def __rmod__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(_operand(lhs), self, 1))

    def __rmod__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        return rebind_var[Batch[Self.T]](_integer_division_part(Integer(type_of(lhs)()), self, 1))

    def __imod__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _integer_update[11](self, rhs)

    def __imod__(mut self, rhs: IntLiteral) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._assign_division_scalar(_LiteralOperand[type_of(rhs)()](), 1)

    def __pow__[V: ImplicitlyCopyable & Deinitable](self, rhs: V) raises -> Batch[Self.T]:
        return _power[Self.T](self, _operand(rhs))

    def __pow__(self, rhs: IntLiteral) raises -> Batch[Self.T]:
        return _power[Self.T](self, Integer(type_of(rhs)()))

    def __rpow__[V: ImplicitlyCopyable & Deinitable](self, lhs: V) raises -> Batch[_ElementOf[V]]:
        comptime assert Self.T == Integer, "Powers of a scalar base require Integer exponents."
        comptime assert _ElementOf[V] == Integer or _ElementOf[V] == Rational, (
            "Use Float or Complex scalar powers through their own types."
        )
        comptime if _ElementOf[V] == Rational:
            return rebind_var[Batch[_ElementOf[V]]](_rational_power(_operand(lhs), self))
        else:
            return rebind_var[Batch[_ElementOf[V]]](_integer_result[1, False](_operand(lhs), self, 6))

    def __rpow__(self, lhs: IntLiteral) raises -> Batch[Self.T]:
        comptime assert Self.T == Integer, "Powers of a scalar base require Integer exponents."
        return rebind_var[Batch[Self.T]](_integer_result[1, False](Integer(type_of(lhs)()), self, 6))

    def __ipow__[V: ImplicitlyCopyable & Deinitable](mut self, var rhs: V) raises:
        _power_update(self, rhs)

    def __ipow__(mut self, rhs: IntLiteral) raises:
        comptime if Self.T == Integer:
            self._assign_bit_scalar(_LiteralOperand[type_of(rhs)()](), 6)
        else:
            self.__ipow__(Integer(type_of(rhs)()))


    def _float_reverse(
        self, lhs: _FloatArgument, operation: Int
    ) raises -> Batch[Complex if Self.T == Complex else Float]:
        return rebind_var[Batch[Complex if Self.T == Complex else Float]](
            _approx_result(lhs, self, operation)
        )

    def _float_power_reverse(
        self, base: _FloatArgument
    ) raises -> Batch[Complex if Self.T == Complex else Float]:
        comptime assert Self.T == Integer, (
            "Float powers require Integer exponents; use a Batch[Integer]."
        )
        return rebind_var[Batch[Complex if Self.T == Complex else Float]](
            _float_power(base, self)
        )

    def _compare_float(
        self, rhs: _FloatArgument, operation: Int
    ) raises -> Mask:
        return _float_comparison(self, rhs, operation)

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _BatchIterator[Self.T]
    comptime IteratorOwnedType: Iterator = _BatchIterator[Self.T]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return _BatchIterator[Self.T](
            _Selection(0, len(self), 1), 0, self._run(), self._real_format, self._imag_format
        )

    def __iter__(var self) -> Self.IteratorOwnedType:
        return _BatchIterator[Self.T](
            _Selection(0, len(self), 1), 0, self._run(), self._real_format, self._imag_format
        )

    def __init__(out self):
        """An empty vector."""
        comptime assert conforms_to(Self.T, _BatchElement), (
            "Batch elements must be one of the families listed in"
            " batch/_families.mojo; add a new family there."
        )
        self._owner = ArcPointer(_Values[Self.T](List[Self.T]()))
        self._layout = ArcPointer(_FlatLayout.run(0, 0, 1))
        self._real_format = _default_format()
        self._imag_format = _default_format()

    @implicit
    def __init__(out self, values: List[Self.T]) raises:
        """A vector of the values; a batch of the same family shares its values.

        Args:
            values: The values.

        Raises:
            Only on a checked size error.
        """
        self = Self(_tensor=_Tensor[Self.T](values.copy(), [len(values)]))

    def __init__(out self, var *values: Self.T, __list_literal__: NoneType) raises:
        """A vector from a list literal, such as a slice assignment source."""
        var prepared = List[Self.T](capacity=_checked_count(len(values), size_of[Self.T]()))
        for i in range(len(values)):
            prepared.append(values[i])
        self = Self(prepared)

    def __init__(out self, *, copy: Self):
        self._owner = copy._owner
        self._layout = copy._layout
        self._real_format = copy._real_format
        self._imag_format = copy._imag_format

    def _complex_input(self) -> _ComplexInput:
        comptime assert Self.T == Complex
        var selection = _Selection(0, len(self), 1)
        var values = rebind[_Tensor[Complex]](self._run())
        return _ComplexInput(
            _FloatInput(None, selection, complex_values=values, format=self._real_format),
            _FloatInput(None, selection, complex_values=values, imaginary=True, format=self._imag_format),
        )

    def _float_input(self) -> _FloatInput:
        comptime assert Self.T == Float
        return _FloatInput(
            None, _Selection(0, len(self), 1),
            rebind[_Tensor[Float]](self._run()), format=self._real_format,
        )

    def _rational_input(self) -> _RationalInput:
        comptime assert Self.T == Rational
        return _RationalInput(
            Rational(), _Selection(0, len(self), 1),
            native=rebind[_Tensor[Rational]](self._run()),
        )

    def __init__[I: Iterable](out self, values: I) raises:
        """A vector from a finite iterable, consumed once; another batch converts each element."""
        comptime if I == Self:
            self = rebind[Self](values)
            return
        comptime if I == _BatchIterator[Self.T]:
            self = rebind[_BatchIterator[Self.T]](values)._rest()
            return
        comptime if Self.T == Complex:
            self = Self(_complex_cursor=iter(values))
        elif Self.T == Float:
            comptime if I == List[Float]:
                var list_values = rebind[List[Float]](values).copy()
                var count = len(list_values)
                self = Self(_tensor=rebind[_Tensor[Self.T]](_Tensor[Float](list_values^, [count])))
            else:
                self = Self(_float_cursor=iter(values))
        elif Self.T == Rational:
            self = Self(_tensor=rebind[_Tensor[Self.T]](_native_rational_iterator(iter(values))))
        elif Self.T == Integer:
            comptime if conforms_to(I, _BatchShape):
                comptime assert I.Element == Integer, (
                    "Use Batch[Rational] to keep fractional elements; convert"
                    " individual values explicitly for Integer batches"
                )
            self = Self(_tensor=rebind[_Tensor[Self.T]](_native_integer_iterator(iter(values))))
        else:
            self = Self(_tensor=_same_family_tensor[Self.T](iter(values)))
        comptime if conforms_to(I, _BatchShape):
            if values.ndim() != 1:
                self._keep_shape(values.shape())

    def __init__(out self, var values: Some[IterableOwned]) raises:
        """A vector from an owned finite iterable."""
        self = Self._packed(values^)


    @staticmethod
    def _packed[I: IterableOwned](var values: I) raises -> Self:
        comptime if I == _BatchIterator[Self.T]:
            return rebind_var[_BatchIterator[Self.T]](iter(values^))._rest()
        elif Self.T == Complex:
            return Self(_complex_cursor=iter(values^))
        elif Self.T == Float:
            return Self(_float_cursor=iter(values^))
        elif Self.T == Rational:
            return Self(_tensor=rebind[_Tensor[Self.T]](_native_rational_iterator(iter(values^))))
        elif Self.T == Integer:
            return Self(_tensor=rebind[_Tensor[Self.T]](_native_integer_iterator(iter(values^))))
        else:
            return Self(_tensor=_same_family_tensor[Self.T](iter(values^)))

    def __init__[I: Iterator](out self, *, var _complex_cursor: I) raises:
        comptime assert Self.T == Complex
        var formats = _ComplexFormats(FloatFormat(), FloatFormat())
        self = rebind[Self](_complex_from_tensor(_native_complex_iterator(_complex_cursor^), formats))

    def __init__[I: Iterator](out self, *, var _float_cursor: I) raises:
        comptime assert Self.T == Float
        self = Self(_tensor=rebind[_Tensor[Self.T]](_native_float_iterator(_float_cursor^)))
        self._real_format = FloatFormat()

    @staticmethod
    def from_json(
        text: String, *, limits: Optional[ConversionLimits] = None
    ) raises -> Self:
        """Read a batch from its JSON record.

        Version 1 holds a vector; version 2 adds an explicit shape.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            An independent batch.

        Raises:
            When the text is not exactly that schema, naming the byte offset and element.
        """
        comptime assert _Arithmetic[Self.T], "Batch JSON covers Integer, Rational, Float and Complex elements."
        var shape = Optional[List[Int]]()
        var result: Self
        comptime if Self.T == Complex:
            from ..complex._batch_json import _parse_complex_batch_json

            var budget = _ConversionBudget(limits)
            var values, formats = _parse_complex_batch_json(text, budget, shape)
            result = rebind[Self](_complex_from_tensor(values, formats))
        elif Self.T == Float:
            from ..float._batch_json import _parse_float_batch_json

            var budget = _ConversionBudget(limits)
            var values, format = _parse_float_batch_json(text, budget, shape)
            result = rebind[Self](_float_from_tensor(values, format))
        elif Self.T == Rational:
            var budget = _ConversionBudget(limits)
            result = Self(_tensor=rebind[_Tensor[Self.T]](_parse_rational_batch_json(text, budget, shape)))
        else:
            result = Self._from_integer_json(text, limits, shape)
        if shape:
            if _FlatLayout(shape.value().copy()).size != len(result):
                raise Error("Batch interchange shape does not match its values; use dimensions whose product equals the value count.")
            result._keep_shape(shape.value())
        return result^

    @staticmethod
    def _from_integer_json(
        text: String, limits: Optional[ConversionLimits], mut shape: Optional[List[Int]],
    ) raises -> Self:
        var budget = _ConversionBudget(limits)
        var strings = _read_integer_json(text, True, budget, shape)
        if not budget.bounded_allocation():
            # Unlimited: the texts are validated, so long runs convert in parallel.
            var parsed = _parse_digit_texts(strings, "Cannot read integer interchange element ")
            var count = len(parsed)
            return Self(_tensor=rebind[_Tensor[Self.T]](_Tensor[Integer](parsed^, [count])))
        budget.allocate(len(strings), size_of[Integer]())
        var values = List[Integer](
            capacity=_checked_count(len(strings), size_of[Integer]())
        )
        for i in range(len(strings)):
            try:
                values.append(Integer._parse_json_digits(strings[i], budget, i))
            except error:
                raise Error(
                    String(
                        "Cannot read integer interchange element ",
                        i,
                        ": ",
                        error,
                    )
                )
        _batch_owner_budget[Integer](budget)
        var count = len(values)
        return Self(_tensor=rebind[_Tensor[Self.T]](_Tensor[Integer](values^, [count])))

    def to_json(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """Write the JSON record: version 1 for a vector, version 2 with its shape otherwise.

        Selections write their values in logical order.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON.

        Raises:
            When the output exceeds `limits`.
        """
        comptime assert _Arithmetic[Self.T], "Batch JSON covers Integer, Rational, Float and Complex elements."
        comptime if Self.T == Complex:
            from ..complex._batch_json import _write_complex_batch_json

            var budget = _ConversionBudget(limits)
            return _write_complex_batch_json(self._complex_input(), budget, self._json_shape())
        elif Self.T == Float:
            from ..float._batch_json import _write_float_batch_json

            var budget = _ConversionBudget(limits)
            return _write_float_batch_json(self._float_input(), budget, self._json_shape())
        elif Self.T == Rational:
            var budget = _ConversionBudget(limits)
            return _write_rational_batch_json(self._rational_input(), budget, self._json_shape())
        else:
            return self._integer_json(limits)

    def _json_shape(self) -> Optional[List[Int]]:
        """Version 2 records carry the shape of any batch that is not a vector."""
        if self.ndim() == 1:
            return None
        return self.shape()

    def _integer_json(
        self, limits: Optional[ConversionLimits]
    ) raises -> String:
        if limits:
            return self._limited_json(limits.value())
        var result = _batch_json_header("integer-batch", self._json_shape()) + ',"values":['
        if _parallel_ready(len(self)):
            # Records are independent, so long runs format in parallel.
            var integers = rebind[Batch[Integer]](self)
            return _join_records(result^, _map_values[_quoted_decimal](integers, len(self)).take_list())
        for i in range(len(self)):
            var value = rebind[Integer](self[i])
            _ = _checked_count(
                _checked_sum(
                    result.byte_length(),
                    _checked_sum(_json_decimal_bound(value._word_count()), 8),
                ),
                1,
            )
            if i:
                result += ","
            result.write('"', value._decimal(), '"')
        result += "]}"
        return result

    def _json_value(
        self, index: Int, mut budget: _ConversionBudget
    ) raises -> Integer:
        var scalar = rebind[Integer](self[index])
        _ = _format_bound(scalar._word_count())
        budget.preflight(
            scalar.magnitude_bit_length(),
            10,
            Int(scalar._negative()),
            index,
        )
        return scalar

    def _limited_json(self, limits: ConversionLimits) raises -> String:
        var budget = _ConversionBudget(limits)
        var count = len(self)
        budget.values(
            count,
            limits._values if count > limits._values
            and limits._values >= 0 else -1,
        )
        var result = _batch_json_header("integer-batch", self._json_shape()) + ',"values":['
        budget.output(result.byte_length() + 2)
        return _write_bounded_batch_records[Batch[Integer], _append_integer_record](
            rebind[Batch[Integer]](self), count, result^, budget,
        )

    @staticmethod
    def from_iterable(values: List[Self.T]) raises -> Self:
        """A vector of the values."""
        return Self(values)

    @staticmethod
    def from_iterable[I: Iterable](values: I) raises -> Self:
        """A vector from a finite iterable, consumed once."""
        return Self(values)

    @staticmethod
    def from_iterable(var values: Some[IterableOwned]) raises -> Self:
        """A vector from an owned finite iterable."""
        return Self(values^)

    @staticmethod
    def from_native(values: List[Int]) raises -> Self:
        """A vector from native integers."""
        comptime if Self.T == Complex:
            return Self(_complex_cursor=iter(values))
        elif Self.T == Float:
            return Self(_float_cursor=iter(values))
        elif Self.T == Rational:
            return Self(_tensor=rebind[_Tensor[Self.T]](_native_rational_iterator(iter(values))))
        else:
            return Self(_tensor=rebind[_Tensor[Self.T]](_native_integer_iterator(iter(values))))

    @staticmethod
    def from_native[
        dtype: DType
    ](values: List[SIMD[dtype, 1]]) raises -> Self where (
        Self.T == Float or Self.T == Complex
    ):
        """A Float or Complex vector from native numbers."""
        comptime if Self.T == Complex:
            return Self(_complex_cursor=iter(values))
        else:
            return Self(_float_cursor=iter(values))

    def __len__(self) -> Int:
        return self._layout[].size

    def write_to(self, mut writer: Some[Writer]):
        """Write the values as nested lists, such as `[[1, 2], [3, 4]]`, in full.

        Args:
            writer: The destination.
        """
        comptime assert conforms_to(Self.T, Writable)
        ref layout = self._layout[]
        _write_axis(self._owner, _list_ints(layout.shape), _list_ints(layout.strides), 0, layout.offset, writer)

    def _checked_index(self, index: Int) raises -> Int:
        var adjusted = index
        if adjusted < 0:
            adjusted += len(self)
        if adjusted < 0 or adjusted >= len(self):
            raise Error(
                String(
                    "Cannot access batch element ",
                    index,
                    " in a batch of length ",
                    len(self),
                    "; use an index within the batch bounds.",
                )
            )
        return adjusted

    def __getitem__(self, index: Int) raises -> Self.T:
        return self._owner[].list[self._layout[].position(self._checked_index(index))]

    def __getitem__(self, selection: StridedSlice) raises -> Self:
        """A flat slice of the row-major values; it shares them without copying."""
        return _batch_from[Self.T](self._flat()._array().sliced(0, selection))

    def __getitem__(self, selection: ContiguousSlice) raises -> Self:
        return self[StridedSlice(selection.start, selection.end, 1)]

    def __setitem__(mut self, index: Int, var value: Self.T) raises:
        var adjusted = self._checked_index(index)
        self._assign_selection(_Selection(adjusted, 1, 1), Self(value^), True, None)

    def __getitem__(self, selection: Mask) raises -> Self:
        """The mask-selected row-major values as a new vector."""
        var mask = _vector_mask(self, selection)
        mask._check_length(len(self))
        var count = mask.count()
        var values = List[Self.T](capacity=_checked_count(count, size_of[Self.T]()))
        ref layout = self._layout[]
        for i in range(len(self)):
            if mask[i]:
                values.append(self._owner[].list[layout.position(i)])
        var result = Self(_tensor=_Tensor[Self.T](values^, [count]))
        result._real_format = self._real_format
        result._imag_format = self._imag_format
        return result^

    # Mojo converts an assigned value to the getter's type, so a scalar
    # arrives as a rank-zero batch; a rank-zero source fills the selection.

    def __setitem__(mut self, selection: Mask, var value: Self) raises:
        """Assign row-major values of the selected count, or fill the selection."""
        self._assign_mask(selection, value, value.ndim() == 0)

    def __setitem__(mut self, selection: StridedSlice, var value: Self) raises:
        """Assign row-major values of the slice's length, or fill the slice."""
        self._assign_slice(selection, value, value.ndim() == 0)

    def __setitem__(mut self, selection: ContiguousSlice, var value: Self) raises:
        self._assign_slice(StridedSlice(selection.start, selection.end, 1), value, value.ndim() == 0)

    def _assign_mask(
        mut self,
        selection: Mask,
        rhs: Self,
        broadcast: Bool = False,
        fail_after_element: Int = -1,
    ) raises:
        var mask = _vector_mask(self, selection)
        self._assign_selection(_Selection(0, len(self), 1), rhs, broadcast, mask, fail_after_element)

    def _assign_slice(
        mut self,
        selection: StridedSlice,
        rhs: Self,
        broadcast: Bool = False,
        fail_after_element: Int = -1,
    ) raises:
        self._assign_selection(_Selection(len(self), selection), rhs, broadcast, None, fail_after_element)

    def _assign_selection(
        mut self, target: _Selection, rhs: Self, broadcast: Bool,
        mask: Optional[Mask], fail: Int = -1,
    ) raises:
        """Write the selected positions from `rhs`, or fill them from its one
        value. The values are staged privately and published only when every
        write succeeds; the destination keeps its shape and default formats.
        `fail` injects a failure after that many writes, for tests."""
        var count = target.count
        if mask:
            mask.value()._check_length(len(self))
            count = mask.value().count()
        if not broadcast and count != len(rhs):
            raise Error(
                String(
                    "Cannot assign ",
                    len(rhs),
                    " values to ",
                    count,
                    (
                        " selected elements; use a source with the same length"
                        " or a scalar to fill the selection. The destination is"
                        " unchanged."
                    ),
                )
            )
        if not count:
            return
        var staged = self.to_list()
        var source = rhs._run()
        var selected = 0
        for i in range(len(self) if mask else target.count):
            if mask and not mask.value()[i]:
                continue
            var position = i if mask else target.index(i)
            staged[position] = source._read(0 if broadcast else selected)
            if selected == fail:
                raise Error(String(
                    "Injected ", "mask" if mask else "slice",
                    " assignment failure at batch element ", position,
                    "; the destination is unchanged. Retry the assignment.",
                ))
            selected += 1
        var length = len(staged)
        var kept_shape = self.shape()
        var real_format = self._real_format
        var imag_format = self._imag_format
        self = Self(_tensor=_Tensor[Self.T](staged^, [length]))
        self._real_format = real_format
        self._imag_format = imag_format
        self._keep_shape(kept_shape)

    def to_list(self) raises -> List[Self.T]:
        """The values, in row-major order.

        Returns:
            An independent list.

        Raises:
            Only on a checked size error.
        """
        _ = _checked_count(len(self), size_of[Self.T]())
        return self._run().to_list()

    def _binary(
        self,
        rhs: Self,
        operation: Int,
        broadcast: Bool = False,
        fail_after_tile: Int = -1,
    ) raises -> Self:
        if self.ndim() != 1 or rhs.ndim() != 1:
            var plan = _prepare_binary_shape(self, rhs, operation, broadcast)
            if plan.broadcast:
                return rebind_var[Self](_exact_result[Self.T](_broadcast_operand(self, plan.shape), _broadcast_operand(rhs, plan.shape), operation))
            return _shaped_result(self._flat()._binary(rhs._flat(), operation, broadcast, fail_after_tile), plan.shape)
        comptime if Self.T == Complex:
            return rebind_var[Self](
                _complex_result(
                    self,
                    rhs,
                    operation,
                    fail=fail_after_tile * _WIDTH,
                )
            )
        elif Self.T == Float:
            return rebind_var[Self](
                _float_result(
                    self,
                    rhs,
                    operation,
                    fail=fail_after_tile * _WIDTH,
                    broadcast_b=broadcast,
                )
            )
        elif Self.T == Rational:
            return rebind_var[Self](
                _rational_result(
                    self, rhs, operation, fail_after_tile * _WIDTH, broadcast
                )
            )
        else:
            comptime assert Self.T == Integer
            self._check_shape(rhs, operation, broadcast)
            return rebind[Self](_native_integer_result(
                rebind[Batch[Integer]](self), rebind[Batch[Integer]](rhs),
                operation, broadcast_b=broadcast, fail=fail_after_tile,
            ))

    def _check_shape(
        self, rhs: Self, operation: Int, broadcast: Bool = False
    ) raises:
        if not broadcast:
            self._check_length(len(rhs), operation)

    def _check_length(self, length: Int, operation: Int) raises:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        if len(self) != length:
            var name = (
                "add" if operation
                == 0 else "subtract" if operation
                == 1 else "multiply" if operation
                == 2 else "divide" if operation
                == 3 else "combine with &" if operation
                == 4 else "combine with |" if operation
                == 5 else "combine with ^" if operation
                == 6 else "left-shift" if operation
                == 8 else "right-shift" if operation
                == 9 else "raise powers of"
            )
            raise Error(
                String(
                    "Cannot ",
                    name,
                    " batches of lengths ",
                    len(self),
                    " and ",
                    length,
                    (
                        "; use equal lengths or a scalar operand. The"
                        " destination is unchanged."
                    ),
                )
            )

    def _stage_complex(
        mut self,
        rhs: _ComplexOperand,
        operation: Int,
        fail: Int = -1,
        fail_component: Int = 1,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        comptime assert (
            Self.T == Complex
        ), "Convert the destination to Batch[Complex] before a Complex update."
        var inputs = _ComplexInputs(_complex_operand(self), rhs)
        inputs.check_shape()
        if not len(self):
            return
        var source = inputs.a.components.value()
        var formats = _ComplexFormats(source.real.default_format(), source.imag.default_format())
        var kept_shape = self.shape()
        if operation == 4:
            var output = _native_complex_power(
                inputs, context, fail=fail, fail_component=fail_component,
                destination_format=True,
            )
            self = rebind[Self](_complex_from_tensor(output, formats))
        else:
            var output = _native_complex_arithmetic(
                inputs, operation, context, fail=fail, fail_component=fail_component,
                destination_format=True,
            )
            self = rebind[Self](_complex_from_tensor(rebind_var[_Tensor[Complex]](output^), formats))
        self._keep_shape(kept_shape)

    def _update_complex[
        R: ImplicitlyCopyable
    ](
        mut self,
        rhs: R,
        operation: Int,
        fail: Int = -1,
        fail_component: Int = 1,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        _check_update_shape(self, rhs)
        self._stage_complex(
            _complex_operand(rhs),
            operation,
            fail,
            fail_component,
            context,
        )

    def _update_complex_power[
        R: ImplicitlyCopyable
    ](
        mut self,
        rhs: R,
        fail: Int = -1,
        fail_component: Int = 1,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        _check_update_shape(self, rhs)
        self._stage_complex(
            _complex_count_operand(rhs), 4, fail, fail_component, context
        )

    def _stage_float(
        mut self,
        rhs: _FloatOperand,
        operation: Int,
        fail: Int = -1,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        comptime assert (
            Self.T == Float
        ), "Convert the destination to Batch[Float] before a Float update."
        var inputs = _FloatInputs(_float_operand(self), rhs)
        inputs.check_shape()
        if not len(self):
            return
        var output = _native_float_arithmetic(inputs, operation, context, fail=fail, destination_format=True)
        var staged = Self(_tensor=rebind[_Tensor[Self.T]](output))
        staged._real_format = self._real_format
        var kept_shape = self.shape()
        self = staged
        self._keep_shape(kept_shape)

    def _update_float[
        R: ImplicitlyCopyable
    ](
        mut self,
        rhs: R,
        operation: Int,
        fail: Int = -1,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        _check_update_shape(self, rhs)
        self._stage_float(_float_operand(rhs), operation, fail, context)

    def _update_float_power[
        R: ImplicitlyCopyable
    ](
        mut self,
        rhs: R,
        fail: Int = -1,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        _check_update_shape(self, rhs)
        self._stage_float(_float_count_operand(rhs), 4, fail, context)

    def _update_rational[
        R: ImplicitlyCopyable
    ](mut self, rhs: R, operation: Int, fail: Int = -1) raises:
        _check_update_shape(self, rhs)
        comptime assert Self.T == Rational, (
            "Compound updates cannot change Integer elements to Rational;"
            " construct Batch[Rational](values) before updating"
        )
        var inputs = _rational_inputs(self, rhs)
        if not len(self):
            inputs.validate(operation)
            return
        var values = _native_rational_arithmetic(inputs, operation, fail)
        var kept_shape = self.shape()
        self = Self(_tensor=rebind[_Tensor[Self.T]](values))
        self._keep_shape(kept_shape)

    def _update_rational_power[
        R: ImplicitlyCopyable
    ](mut self, rhs: R, fail: Int = -1) raises:
        _check_update_shape(self, rhs)
        # Enforce integral exponents even for a Rational destination.
        comptime if R == Integer:
            self._update_rational(rhs, 5, fail)
        else:
            self._update_rational(_integer_batch(rhs), 5, fail)

    def _stage_integer(
        mut self, source: Batch[Integer], family: Int, operation: Int,
        *, broadcast: Bool = False, fail: Int = -1,
        label: StaticString = "staging",
    ) raises:
        comptime assert Self.T == Integer
        var inputs = _native_integer_inputs(
            rebind[Batch[Integer]](self), source, operation,
            broadcast_b=broadcast, family=family,
        )
        var staged = _native_integer_arithmetic(inputs, family, operation, fail=fail, label=label)
        var kept_shape = self.shape()
        self = Self(_tensor=rebind_var[_Tensor[Self.T]](staged^))
        self._keep_shape(kept_shape)

    def _assign(
        mut self,
        var rhs: Self,
        operation: Int,
        fail_after_tile: Int = -1,
    ) raises:
        _check_update_shape(self, rhs)
        comptime assert Self.T == Integer, (
            "This Integer update requires Integer elements; Float compound"
            " updates use Float operands."
        )
        self._check_shape(rhs, operation)
        self._stage_integer(rebind[Batch[Integer]](rhs), 0, operation, fail=fail_after_tile)

    def _assign_scalar[
        R: _IntegerOperand
    ](mut self, rhs: R, operation: Int, fail_after_tile: Int = -1) raises:
        _check_update_shape(self, rhs)
        comptime assert Self.T == Integer, (
            "This Integer-only update requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._stage_integer(Batch[Integer]([Integer._from_operand(rhs)]), 0, operation, broadcast=True, fail=fail_after_tile)

    def _bit_result(
        self,
        rhs: Self,
        operation: Int,
        broadcast: Bool = False,
    ) raises -> Self:
        comptime assert Self.T == Integer, (
            "This Integer-only operation requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        var shape = _operand_shape(self, self, self) if broadcast else _operand_shape(self, rhs, self)
        self._check_shape(rhs, operation + 4, broadcast)
        var a = rebind[Batch[Integer]](self)
        var b = rebind[Batch[Integer]](rhs)
        return _shape_like(rebind[Self](_native_integer_result(
            a, b, operation, broadcast_b=broadcast, family=1,
        )), shape)

    def _assign_bit(
        mut self, var rhs: Self, operation: Int, fail_after_tile: Int = -1
    ) raises:
        _check_update_shape(self, rhs)
        comptime assert Self.T == Integer, (
            "This Integer-only update requires Integer elements; use"
            " Rational arithmetic and integral powers for fractions"
        )
        self._check_shape(rhs, operation + 4)
        self._stage_integer(rebind[Batch[Integer]](rhs), 1, operation, fail=fail_after_tile, label="integer operation")

    def _assign_bit_scalar[
        R: _IntegerOperand
    ](mut self, rhs: R, operation: Int, fail_after_tile: Int = -1) raises:
        _check_update_shape(self, rhs)
        comptime assert (
            Self.T == Integer
        ), "Rational batches do not support Integer bit operations"
        self._stage_integer(Batch[Integer]([Integer._from_operand(rhs)]), 1, operation, broadcast=True, fail=fail_after_tile, label="integer operation")

    def __invert__(self) raises -> Self:
        return self._bit_result(self, 3)

    def __neg__(self) raises -> Self:
        comptime if Self.T == Complex:
            return rebind_var[Self](_complex_unary(self, 1))
        elif Self.T == Float:
            return rebind_var[Self](_float_unary(self, 1))
        elif Self.T == Rational:
            return rebind_var[Self](_rational_unary[False](self, 0))
        else:
            return self._bit_result(self, 9)

    def sign(self) raises -> Batch[Integer]:
        """The sign of each element: -1, 0 or 1.

        Returns:
            A `Batch[Integer]` of the same shape.

        Raises:
            For a NaN element.
        """
        comptime if Self.T == Float:
            return _float_sign(self)
        elif Self.T == Rational:
            return _rational_unary[True](self, 2)
        else:
            return rebind_var[Batch[Integer]](self._bit_result(self, 7))

    def __pos__(self) raises -> Self where Self.T == Float or Self.T == Complex:
        comptime if Self.T == Complex:
            return rebind_var[Self](_complex_unary(self, 0))
        else:
            return rebind_var[Self](_float_unary(self, 0))

    def conjugate(self) raises -> Self where Self.T == Complex:
        """The complex conjugate of each element.

        Returns:
            A Complex batch of the same shape.

        Raises:
            Only on a checked size error.
        """
        return rebind_var[Self](_complex_unary(self, 2))

    def real(self) raises -> Batch[Float] where Self.T == Complex:
        """The real part of each element.

        Returns:
            A `Batch[Float]` of the same shape.

        Raises:
            Only on a checked size error.
        """
        return _complex_component(self, False)

    def imag(self) raises -> Batch[Float] where Self.T == Complex:
        """The imaginary part of each element.

        Returns:
            A `Batch[Float]` of the same shape.

        Raises:
            Only on a checked size error.
        """
        return _complex_component(self, True)

    def signbit(self) raises -> Mask where Self.T == Float:
        """Whether each element is negative, including negative zero.

        Returns:
            A `Mask` of the same shape.

        Raises:
            Only on a checked size error.
        """
        return _float_query(self, 0)

    def is_zero(self) raises -> Mask where Self.T == Float or Self.T == Complex:
        """Whether each element is zero.

        Returns:
            A `Mask` of the same shape.

        Raises:
            Only on a checked size error.
        """
        comptime if Self.T == Complex:
            return _complex_query(self, 0)
        else:
            return _float_query(self, 1)

    def is_finite(
        self,
    ) raises -> Mask where Self.T == Float or Self.T == Complex:
        """Whether each element is finite.

        Returns:
            A `Mask` of the same shape.

        Raises:
            Only on a checked size error.
        """
        comptime if Self.T == Complex:
            return _complex_query(self, 1)
        else:
            return _float_query(self, 2)

    def is_infinite(
        self,
    ) raises -> Mask where Self.T == Float or Self.T == Complex:
        """Whether each element is infinite.

        Returns:
            A `Mask` of the same shape.

        Raises:
            Only on a checked size error.
        """
        comptime if Self.T == Complex:
            return _complex_query(self, 2)
        else:
            return _float_query(self, 3)

    def is_nan(self) raises -> Mask where Self.T == Float or Self.T == Complex:
        """Whether each element is NaN.

        Returns:
            A `Mask` of the same shape.

        Raises:
            Only on a checked size error.
        """
        comptime if Self.T == Complex:
            return _complex_query(self, 3)
        else:
            return _float_query(self, 4)

    def floor(
        self,
    ) raises -> Batch[Integer] where Self.T == Rational or Self.T == Float:
        """Round each element toward negative infinity.

        Returns:
            A `Batch[Integer]` of the same shape.

        Raises:
            For an infinite or NaN element, naming its index.
        """
        comptime if Self.T == Float:
            return _float_integral(self, 1)
        else:
            return _rational_unary[True](self, 3)

    def ceil(
        self,
    ) raises -> Batch[Integer] where Self.T == Rational or Self.T == Float:
        """Round each element toward positive infinity.

        Returns:
            A `Batch[Integer]` of the same shape.

        Raises:
            For an infinite or NaN element, naming its index.
        """
        comptime if Self.T == Float:
            return _float_integral(self, 2)
        else:
            return _rational_unary[True](self, 4)

    def trunc(
        self,
    ) raises -> Batch[Integer] where Self.T == Rational or Self.T == Float:
        """Round each element toward zero.

        Returns:
            A `Batch[Integer]` of the same shape.

        Raises:
            For an infinite or NaN element, naming its index.
        """
        comptime if Self.T == Float:
            return _float_integral(self, 0)
        else:
            return _rational_unary[True](self, 5)

    def to_integer_exact(self) raises -> Batch[Integer] where Self.T == Float:
        """Convert each integral element to Integer.

        Returns:
            A `Batch[Integer]` of the same shape.

        Raises:
            For a fractional, infinite or NaN element, naming its index.
        """
        return _float_integral(self, 3)

    def magnitude_bit_length(self) raises -> Self:
        """The bit length of each element's absolute value.

        Returns:
            A batch of the same shape.

        Raises:
            Only on a checked size error.
        """
        return self._bit_result(self, 8)

    def _assign_division(
        mut self,
        var rhs: Self,
        output: Int,
        fail_after_tile: Int = -1,
    ) raises:
        _check_update_shape(self, rhs)
        self._check_shape(rhs, 3)
        self._stage_integer(rebind[Batch[Integer]](rhs), 2, output, fail=fail_after_tile, label="division")

    def _assign_division_scalar[
        R: _IntegerOperand
    ](
        mut self,
        rhs: R,
        output: Int,
        fail_after_tile: Int = -1,
    ) raises:
        _check_update_shape(self, rhs)
        comptime assert (
            Self.T == Integer
        ), "Rational batches do not support Integer floor division or remainder"
        self._stage_integer(Batch[Integer]([Integer._from_operand(rhs)]), 2, output, broadcast=True, fail=fail_after_tile, label="division")

    def _compare(
        self, rhs: Self, operation: Int, broadcast: Bool = False
    ) raises -> Mask:
        if self.ndim() != 1 or rhs.ndim() != 1:
            var plan = _prepare_binary_shape(self, rhs, -1, broadcast)
            if plan.broadcast:
                return _compare_values(_broadcast_operand(self, plan.shape), _broadcast_operand(rhs, plan.shape), operation)
            return self._flat()._compare(rhs._flat(), operation, broadcast).reshape(plan.shape)
        comptime if Self.T == Complex:
            return _complex_comparison(self, rhs, operation)
        elif Self.T == Float:
            return _float_comparison(
                self, rhs, operation, broadcast_b=broadcast
            )
        elif Self.T == Rational:
            return _exact_comparison[True](self, rhs, operation, broadcast)
        else:
            comptime assert Self.T == Integer
            if not broadcast and len(self) != len(rhs):
                raise Error(
                    String(
                        "Cannot compare batches of lengths ",
                        len(self),
                        " and ",
                        len(rhs),
                        "; use equal lengths or a scalar operand.",
                    )
                )
            return _native_integer_comparison(_native_integer_inputs(
                rebind[Batch[Integer]](self), rebind[Batch[Integer]](rhs),
                operation, broadcast_b=broadcast, family=4,
            ), operation)

    def _compare_integer(self, rhs: Integer, operation: Int) raises -> Mask:
        return self._compare_scalar(rhs, operation)

    def _compare_rational(self, rhs: Rational, operation: Int) raises -> Mask:
        return _exact_comparison[True](self, rhs, operation)

    def _compare_scalar(self, rhs: Integer, operation: Int) raises -> Mask:
        comptime if Self.T == Complex:
            return _complex_comparison(self, rhs, operation)
        elif Self.T == Float:
            return _float_comparison(self, rhs, operation)
        elif Self.T == Rational:
            return _exact_comparison[True](self, rhs, operation)
        else:
            return self._compare(
                Self([_batch_scalar[Self.T](rhs)]), operation, broadcast=True
            )


def _batch_scalar[T: ImplicitlyCopyable & Deinitable](value: Integer) -> T:
    comptime if T == Rational:
        return rebind[T](Rational(value))
    else:
        comptime assert T == Integer
        return rebind[T](value)
