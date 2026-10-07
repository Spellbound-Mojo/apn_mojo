"""Exact and approximate reductions through family-specific arithmetic adapters."""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat
from ..float._rounding import _RoundedBinary
from ..float._batch_storage import _FloatInput
from ..float._reductions import _dot_float
from ..complex.value import Complex
from ..ball.value import Ball, _BallArgument
from ..float._arithmetic import _FloatArgument
from ..complex._input import _ComplexArgument
from ..ball.context import BallContext
from ..complex_ball.value import ComplexBall
from ..integer.math import add as _integer_add, multiply as _integer_multiply
from ..rational.math import add as _rational_add, multiply as _rational_multiply
from ..float.math import add as _float_add, multiply as _float_multiply
from ..complex.math import add as _complex_add, multiply as _complex_multiply
from ..ball.math import add as _ball_add, multiply as _ball_multiply
from ..complex_ball.math import add as _complex_ball_add, multiply as _complex_ball_multiply
from .lift import lift
from ..complex.context import ComplexContext, _ComplexContextArgument
from ..complex.status import ComplexStatus
from ..complex._batch_storage import _ComplexInput
from ..complex._reductions import _dot_complex
from ._float_folds import _ComplexSumFold, _FloatSumFold
from ._lanes import _Fold, _Lanes, _reduce_lanes
from ..complex._ordered_reductions import _ComplexReductionInputs, _ComplexOrderedReduction
from ..float._ordered_reductions import _FloatReductionInputs, _FloatOrderedReduction
from ._reduce_exec import _execute_reduction
from ._exact_reduce import _ExactReductionInputs, _ExactReduction, _exact_fold, _integer_dot, _ExtremeFold, _ArgExtremeFold, _BallExtremeFold
from ._lanes import _ReduceAxes
from .value import Batch, _BatchShape, _batch_from
from ._layout import _FlatLayout, _axis
from std.builtin.rebind import downcast
from .value import _rational_operand, _float_operand, _complex_operand
from ..rational._reductions import _dot_rational


struct _ReductionType[A: ImplicitlyCopyable, B: ImplicitlyCopyable = Int]:
    comptime rational = Self.A == Batch[Rational] or Self.B == Batch[Rational]
    comptime Value = Rational if Self.rational else Integer


def _reduce[
    operation: Int, T: ImplicitlyCopyable
](values: T, fail_after_element: Int = -1) raises -> Integer:
    comptime assert T == Batch[Integer], (
        "Integer reductions require Batch[Integer]; construct"
        " Batch[Integer](values) from an integer iterable first."
    )
    var inputs = _ExactReductionInputs(_rational_operand(values), None)
    comptime if operation == 1:
        # Preserve the early zero shortcut, including its checkpoint behavior.
        for i in range(inputs.length()):
            if inputs.left.is_zero(i):
                return Integer()
    return _exact_fold[operation](rebind[Batch[Integer]](values), List[Int](), True, False, fail_after_element).item()


def _reduce_exact[
    operation: Int, T: ImplicitlyCopyable
](values: T, fail: Int = -1) raises -> _ReductionType[T].Value:
    comptime if _ReductionType[T].rational:
        return rebind[_ReductionType[T].Value](
            _exact_fold[operation](rebind[Batch[Rational]](values), List[Int](), True, False, fail).item()
        )
    else:
        return rebind[_ReductionType[T].Value](_reduce[operation](values, fail))


struct _SumType[T: ImplicitlyCopyable]:
    comptime complex = Self.T == Batch[Complex]
    comptime approximate = Self.T == Batch[Float]
    comptime Value = Complex if Self.complex else Float if Self.approximate else _ReductionType[
        Self.T
    ].Value


def _float_sum_input[T: ImplicitlyCopyable](values: T) -> _FloatInput:
    comptime assert _SumType[T].approximate, (
        "Float sum contexts require Batch[Float];"
        " omit these options for exact Integer/Rational sums."
    )
    return rebind[Batch[Float]](values)._float_input()


def sum[T: ImplicitlyCopyable](values: T) raises -> _SumType[T].Value:
    """Sum every element exactly.

    Integer and Rational sums are exact. Float and Complex sums add the stored
    values exactly and round once, per component for Complex, so the result does
    not depend on the order of the elements. An empty sum is zero.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The total, in the element family.

    Raises:
        When Float formats have different exponent bounds without a context, or on
        a checked size error.
    """
    comptime if _SumType[T].complex or _SumType[T].approximate:
        return _rounded_sum(values, List[Int](), True, False, None, _ComplexContextArgument()).item()
    else:
        return rebind[_SumType[T].Value](_reduce_exact[0](values))


def _rounded_sum[T: ImplicitlyCopyable](
    values: T, axes: List[Int], every: Bool, keepdims: Bool,
    context: Optional[ArithmeticContext], complex_context: _ComplexContextArgument,
) raises -> Batch[_SumType[T].Value]:
    """Float and Complex sums through the lane driver: exact, rounded once."""
    comptime if _SumType[T].complex:
        ref batch = rebind[Batch[Complex]](values)
        return rebind[Batch[_SumType[T].Value]](_reduce_lanes(
            _ComplexSumFold(complex_context, batch._real_format, batch._imag_format),
            batch, axes, every, keepdims, True,
        ))
    else:
        comptime assert _SumType[T].approximate, "Rounded sums take Float or Complex batches."
        ref batch = rebind[Batch[Float]](values)
        return rebind[Batch[_SumType[T].Value]](_reduce_lanes(
            _FloatSumFold(context, batch._real_format), batch, axes, every, keepdims, True,
        ))


def _uniform_reduction_context[
    C: ImplicitlyCopyable
](context: C) -> Optional[ArithmeticContext]:
    comptime if C == ArithmeticContext:
        return rebind[ArithmeticContext](context)
    elif C == Optional[ArithmeticContext]:
        return rebind[Optional[ArithmeticContext]](context)
    else:
        comptime assert C == NoneType or C == type_of(None), (
            "Uniform reduction contexts require ArithmeticContext or None; use"
            " ComplexContext only with a Complex batch."
        )
        return None


def _complex_reduction_context[
    C: ImplicitlyCopyable
](context: C) -> _ComplexContextArgument:
    comptime if C == ComplexContext:
        return _ComplexContextArgument(rebind[ComplexContext](context))
    elif C == Optional[ComplexContext]:
        return _ComplexContextArgument(
            rebind[Optional[ComplexContext]](context)
        )
    elif C == _ComplexContextArgument:
        return rebind[_ComplexContextArgument](context)
    else:
        return _ComplexContextArgument(_uniform_reduction_context(context))


def sum[
    T: ImplicitlyCopyable, C: ImplicitlyCopyable
](values: T, *, context: C) raises -> _SumType[T].Value:
    """Sum a Float or Complex batch exactly and round once with `context`."""
    # Dispatch in the body so callers need no family constraints of their own.
    comptime if _SumType[T].complex:
        _ = _complex_sum_input(values)
        return _rounded_sum(
            values, List[Int](), True, False, None, _complex_reduction_context(context),
        ).item()
    else:
        _ = _float_sum_input(values)
        return _rounded_sum(
            values, List[Int](), True, False, _uniform_reduction_context(context), _ComplexContextArgument(),
        ).item()


def _complex_sum_input[T: ImplicitlyCopyable](values: T) -> _ComplexInput:
    comptime assert T == Batch[Complex], (
        "Complex sum requires a Complex batch; pack scalar or"
        " iterable inputs with Batch[Complex](values) first."
    )
    return rebind[Batch[Complex]](values)._complex_input()


def prod[T: ImplicitlyCopyable](values: T) raises -> _ReductionType[T].Value:
    """The exact product of every element.

    Parameters:
        T: The batch type, of Integers or Rationals.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The product; an empty product is one.

    Raises:
        Only on a checked size error.
    """
    return _reduce_exact[1](values)


struct _ExtremeType[T: ImplicitlyCopyable]:
    """The element type of a batch that `max` and `min` reduce."""

    comptime Value = Rational if Self.T == Batch[Rational] else (Float if Self.T == Batch[Float] else (Ball if Self.T == Batch[Ball] else Integer))


def _extreme_lanes[maximum: Bool, T: ImplicitlyCopyable](
    values: T, axes: List[Int], every: Bool, keepdims: Bool,
) raises -> Batch[_ExtremeType[T].Value]:
    comptime if T == Batch[Integer] or T == Batch[Rational]:
        return rebind[Batch[_ExtremeType[T].Value]](_exact_axes[3 if maximum else 2](values, axes, keepdims, every))
    elif T == Batch[Float]:
        return rebind[Batch[_ExtremeType[T].Value]](_reduce_lanes(
            _ExtremeFold[Float, maximum, False](-1), rebind[Batch[Float]](values), axes, every, keepdims, True,
        ))
    else:
        comptime assert T == Batch[Ball], (
            "max and min take Integer, Rational, Float or Ball batches; complex values have no order."
        )
        return rebind[Batch[_ExtremeType[T].Value]](_reduce_lanes(
            _BallExtremeFold[maximum](), rebind[Batch[Ball]](values), axes, every, keepdims, True,
        ))


def min[T: ImplicitlyCopyable](values: T) raises -> _ExtremeType[T].Value:
    """The least element (numpy's `min`).

    Integers and Rationals compare exactly; a Float batch gives the first
    least element, or its first NaN; a Ball batch gives the ball of the least
    value over all points of its balls.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The least value.

    Raises:
        When the batch is empty; check `len(values)` first.
    """
    comptime if T == Batch[Integer] or T == Batch[Rational]:
        return rebind[_ExtremeType[T].Value](_reduce_exact[2](values))
    else:
        return _extreme_lanes[False](values, List[Int](), True, False).item()


def max[T: ImplicitlyCopyable](values: T) raises -> _ExtremeType[T].Value:
    """The greatest element (numpy's `max`).

    Integers and Rationals compare exactly; a Float batch gives the first
    greatest element, or its first NaN; a Ball batch gives the ball of the
    greatest value over all points of its balls.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The greatest value.

    Raises:
        When the batch is empty; check `len(values)` first.
    """
    comptime if T == Batch[Integer] or T == Batch[Rational]:
        return rebind[_ExtremeType[T].Value](_reduce_exact[3](values))
    else:
        return _extreme_lanes[True](values, List[Int](), True, False).item()


def _arg_lanes[maximum: Bool, T: ImplicitlyCopyable](
    values: T, axes: List[Int], every: Bool, keepdims: Bool,
) raises -> Batch[Integer]:
    comptime if T == Batch[Integer]:
        return _reduce_lanes(_ArgExtremeFold[Integer, maximum](), rebind[Batch[Integer]](values), axes, every, keepdims, True)
    elif T == Batch[Rational]:
        return _reduce_lanes(_ArgExtremeFold[Rational, maximum](), rebind[Batch[Rational]](values), axes, every, keepdims, True)
    else:
        comptime assert T == Batch[Float], (
            "argmax and argmin take Integer, Rational or Float batches; balls and complex values have no order."
        )
        return _reduce_lanes(_ArgExtremeFold[Float, maximum](), rebind[Batch[Float]](values), axes, every, keepdims, True)


def argmin[T: ImplicitlyCopyable](values: T) raises -> Integer:
    """The row-major index of the first least element (numpy's `argmin`);
    of the first NaN in a Float batch.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The flat index.

    Raises:
        When the batch is empty.
    """
    return _arg_lanes[False](values, List[Int](), True, False).item()


def argmax[T: ImplicitlyCopyable](values: T) raises -> Integer:
    """The row-major index of the first greatest element (numpy's `argmax`);
    of the first NaN in a Float batch.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The flat index.

    Raises:
        When the batch is empty.
    """
    return _arg_lanes[True](values, List[Int](), True, False).item()


def _dot_operand[T: ImplicitlyCopyable](values: T) -> Batch[Integer]:
    comptime assert T == Batch[Integer], (
        "Integer dot products require integer batches; construct"
        " Batch[Integer](values) from an integer iterable first."
    )
    return rebind[Batch[Integer]](values)


def _dot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    fail_after_element: Int = -1,
) raises -> Integer:
    return _integer_dot(_dot_operand(a), _dot_operand(b), fail_after_element)


struct _DotType[A: ImplicitlyCopyable, B: ImplicitlyCopyable]:
    comptime complex = _SumType[Self.A].complex or _SumType[Self.B].complex
    comptime approximate = _SumType[Self.A].approximate or _SumType[
        Self.B
    ].approximate
    comptime Value = Complex if Self.complex else Float if Self.approximate else _ReductionType[
        Self.A, Self.B
    ].Value


def _check_float_dot_operands[A: ImplicitlyCopyable, B: ImplicitlyCopyable]():
    comptime assert _DotType[A, B].approximate, (
        "Float dot contexts require at least one Float batch;"
        " omit these options for exact Integer/Rational dot products."
    )
    comptime assert (
        A == Batch[Float]
        or A == Batch[Integer]
        or A == Batch[Rational]
    ) and (
        B == Batch[Float]
        or B == Batch[Integer]
        or B == Batch[Rational]
    ), (
        "Float dot requires two numeric batches; pack"
        " scalar/iterable inputs explicitly. No broadcasting is performed."
    )


def _dot_approximate[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    context: Optional[ArithmeticContext] = None,
    *,
    fail_after_element: Int = -1,
    fail: Bool = False,
) raises -> Float:
    _check_float_dot_operands[A, B]()
    return Float(
        _rounded=_dot_float(
            _float_operand(a),
            _float_operand(b),
            context,
            fail_after_element=fail_after_element,
            fail=fail,
        )
    )


def _check_complex_dot_input[T: ImplicitlyCopyable]():
    comptime assert (
        T == Batch[Complex]
        or T == Batch[Float]
        or T == Batch[Integer]
        or T == Batch[Rational]
    ), (
        "Complex dot/vdot requires two numeric batches; pack"
        " scalar/iterable inputs explicitly. No broadcasting is performed."
    )


def _complex_dot_result[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *,
    conjugate: Bool = False,
) raises -> Tuple[Complex, ComplexStatus]:
    comptime assert _DotType[A, B].complex, (
        "Complex dot/vdot contexts require a Complex batch; use"
        " ArithmeticContext for Float results or omit the context for exact results."
    )
    _check_complex_dot_input[A]()
    _check_complex_dot_input[B]()
    return _dot_complex(
        _complex_operand(a), _complex_operand(b), context, conjugate=conjugate
    )


def dot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B) raises -> _DotType[A, B].Value:
    """The sum of pairwise products, exactly.

    Elements pair by logical position. Exact families give an exact result; with a
    Float or Complex operand the exact sum of exact products is rounded once.
    Integer and Rational operands may pair with Float or Complex ones.

    Parameters:
        A: The first batch type.
        B: The second batch type.

    Args:
        a: The first batch.
        b: The second batch, of the same length.

    Returns:
        The dot product; two empty batches give zero.

    Raises:
        When the lengths differ.
    """
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](_complex_dot_result(a, b)[0])
    elif _DotType[A, B].approximate:
        return rebind[_DotType[A, B].Value](_dot_approximate(a, b))
    else:
        return rebind[_DotType[A, B].Value](_dot_exact(a, b))


def dot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable, C: ImplicitlyCopyable
](a: A, b: B, *, context: C) raises -> _DotType[A, B].Value:
    """The dot product rounded once with `context`."""
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](
            _complex_dot_result(a, b, _complex_reduction_context(context))[0]
        )
    else:
        return rebind[_DotType[A, B].Value](
            _dot_approximate(a, b, _uniform_reduction_context(context))
        )


def vdot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B,) raises -> _DotType[A, B].Value:
    """The dot product with the first operand conjugated.

    For real operands it equals `dot`.

    Parameters:
        A: The first batch type.
        B: The second batch type.

    Args:
        a: The batch to conjugate.
        b: The second batch, of the same length.

    Returns:
        The sum of `conj(a[i]) * b[i]`, rounded once per component.

    Raises:
        When the lengths differ.
    """
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](
            _complex_dot_result(a, b, conjugate=True)[0]
        )
    else:
        return dot(a, b)


def vdot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable, C: ImplicitlyCopyable
](a: A, b: B, *, context: C,) raises -> _DotType[A, B].Value:
    """The conjugated dot product rounded once with `context`."""
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](
            _complex_dot_result(
                a, b, _complex_reduction_context(context), conjugate=True
            )[0]
        )
    else:
        return dot(a, b, context=context)


def _dot_exact[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, fail: Int = -1) raises -> _ReductionType[A, B].Value:
    comptime if _ReductionType[A, B].rational:
        comptime assert (
            A == Batch[Integer]
            or A == Batch[Rational]
        ) and (
            B == Batch[Integer]
            or B == Batch[Rational]
        ), (
            "Exact dot products require Integer or Rational batches;"
            " construct Batch[Integer](values) or"
            " Batch[Rational](values) from an exact iterable first."
        )
        return rebind[_ReductionType[A, B].Value](
            _dot_rational(_rational_operand(a), _rational_operand(b), fail)
        )
    else:
        return rebind[_ReductionType[A, B].Value](_dot(a, b, fail))


def _ordered_exact[
    tree: Bool, dot_product: Bool, A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B,) raises -> _ReductionType[A, B].Value:
    comptime assert (
        A == Batch[Integer]
        or A == Batch[Rational]
    ) and (
        B == Batch[Integer]
        or B == Batch[Rational]
    ), (
        "Exact reductions require Integer/Rational batches; pack"
        " scalar or iterable inputs explicitly."
    )
    var left = _rational_operand(a)
    var right: Optional[type_of(left)] = None
    comptime if dot_product:
        right = _rational_operand(b)
    var inputs = _ExactReductionInputs(left, right)
    var operation = _ExactReduction[
        _ReductionType[A, B].rational, dot_product
    ]()
    return rebind[_ReductionType[A, B].Value](
        _execute_reduction[tree](operation, inputs)
    )


def _ordered_float_sum[
    tree: Bool, T: ImplicitlyCopyable
](values: T, context: Optional[ArithmeticContext],) raises -> _RoundedBinary:
    _ = _float_sum_input(values)
    var inputs = _FloatReductionInputs(_float_operand(values), None)
    var operation = _FloatOrderedReduction[False](
        "sum_tree" if tree else "sum_sequential", context
    )
    return _execute_reduction[tree](operation, inputs)


def _ordered_float_dot[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B, context: Optional[ArithmeticContext],) raises -> _RoundedBinary:
    _check_float_dot_operands[A, B]()
    var inputs = _FloatReductionInputs(_float_operand(a), _float_operand(b))
    var operation = _FloatOrderedReduction[True]("dot_sequential", context)
    return _execute_reduction[False](operation, inputs)


def _ordered_complex[
    tree: Bool, dot_product: Bool, A: ImplicitlyCopyable, B: ImplicitlyCopyable
](
    a: A,
    b: B,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Tuple[Complex, ComplexStatus]:
    var left = _complex_operand(a)
    var right: Optional[type_of(left)] = None
    comptime if dot_product:
        comptime assert _DotType[A, B].complex, (
            "Complex ordered dot contexts require at least one Complex batch;"
            " use ArithmeticContext for Float results."
        )
        _check_complex_dot_input[A]()
        _check_complex_dot_input[B]()
        right = _complex_operand(b)
    else:
        _ = _complex_sum_input(a)
    var inputs = _ComplexReductionInputs(left, right)
    var operation = _ComplexOrderedReduction[dot_product](
        "dot_sequential" if dot_product else "sum_tree" if tree else "sum_sequential",
        context,
    )
    return _execute_reduction[tree](operation, inputs)


def sum_sequential[
    T: ImplicitlyCopyable
](values: T) raises -> _SumType[T].Value:
    """Sum in logical order, rounding every step.

    For a reproducible sequence of rounded additions; `sum` rounds once instead.
    Exact families give the exact sum.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The running sum after the last element.

    Raises:
        On a trapped condition at a step, or mismatched Float formats.
    """
    comptime if _SumType[T].complex:
        return rebind[_SumType[T].Value](
            _ordered_complex[False, False](values, values)[0]
        )
    elif _SumType[T].approximate:
        return rebind[_SumType[T].Value](
            Float(_rounded=_ordered_float_sum[False](values, None))
        )
    else:
        return rebind[_SumType[T].Value](
            _ordered_exact[False, False](values, values)
        )


def sum_sequential[
    T: ImplicitlyCopyable, C: ImplicitlyCopyable
](values: T, *, context: C,) raises -> _SumType[T].Value:
    """Sum in logical order with `context`, rounding every step."""
    comptime if _SumType[T].complex:
        return rebind[_SumType[T].Value](
            _ordered_complex[False, False](
                values, values, _complex_reduction_context(context)
            )[0]
        )
    else:
        return rebind[_SumType[T].Value](
            Float(
                _rounded=_ordered_float_sum[False](
                    values, _uniform_reduction_context(context)
                )
            )
        )


def sum_tree[T: ImplicitlyCopyable](values: T) raises -> _SumType[T].Value:
    """Sum as a fixed binary tree, rounding every step.

    Pairs adjacent elements, padded with zeros to a power of two, so the rounding
    sequence is reproducible and partial sums stay logarithmic in depth.

    Parameters:
        T: The batch type.

    Args:
        values: A batch, or any selection of one.

    Returns:
        The tree sum.

    Raises:
        On a trapped condition at a step, or mismatched Float formats.
    """
    comptime if _SumType[T].complex:
        return rebind[_SumType[T].Value](
            _ordered_complex[True, False](values, values)[0]
        )
    elif _SumType[T].approximate:
        return rebind[_SumType[T].Value](
            Float(_rounded=_ordered_float_sum[True](values, None))
        )
    else:
        return rebind[_SumType[T].Value](
            _ordered_exact[True, False](values, values)
        )


def sum_tree[
    T: ImplicitlyCopyable, C: ImplicitlyCopyable
](values: T, *, context: C,) raises -> _SumType[T].Value:
    """Sum as a fixed binary tree with `context`, rounding every step."""
    comptime if _SumType[T].complex:
        return rebind[_SumType[T].Value](
            _ordered_complex[True, False](
                values, values, _complex_reduction_context(context)
            )[0]
        )
    else:
        return rebind[_SumType[T].Value](
            Float(
                _rounded=_ordered_float_sum[True](
                    values, _uniform_reduction_context(context)
                )
            )
        )


def dot_sequential[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable
](a: A, b: B,) raises -> _DotType[A, B].Value:
    """The dot product in logical order, rounding each product and each addition.

    Parameters:
        A: The first batch type.
        B: The second batch type.

    Args:
        a: The first batch.
        b: The second batch, of the same length.

    Returns:
        The running sum after the last pair.

    Raises:
        When the lengths differ, or on a trapped condition at a step.
    """
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](
            _ordered_complex[False, True](a, b)[0]
        )
    elif _DotType[A, B].approximate:
        return rebind[_DotType[A, B].Value](
            Float(_rounded=_ordered_float_dot(a, b, None))
        )
    else:
        return rebind[_DotType[A, B].Value](_ordered_exact[False, True](a, b))


def dot_sequential[
    A: ImplicitlyCopyable, B: ImplicitlyCopyable, C: ImplicitlyCopyable
](a: A, b: B, *, context: C,) raises -> _DotType[A, B].Value:
    """The ordered dot product with `context`."""
    comptime if _DotType[A, B].complex:
        return rebind[_DotType[A, B].Value](
            _ordered_complex[False, True](
                a, b, _complex_reduction_context(context)
            )[0]
        )
    else:
        return rebind[_DotType[A, B].Value](
            Float(
                _rounded=_ordered_float_dot(
                    a, b, _uniform_reduction_context(context)
                )
            )
        )


def _exact_axes[operation: Int, T: ImplicitlyCopyable](
    values: T, axes: List[Int], keepdims: Bool, every: Bool = False,
) raises -> Batch[_ReductionType[T].Value]:
    comptime if T == Batch[Integer]:
        return rebind[Batch[_ReductionType[T].Value]](
            _exact_fold[operation](rebind[Batch[Integer]](values), axes, every, keepdims)
        )
    else:
        comptime assert T == Batch[Rational], (
            "Exact reductions require Batch[Integer] or Batch[Rational]."
        )
        return rebind[Batch[_ReductionType[T].Value]](
            _exact_fold[operation](rebind[Batch[Rational]](values), axes, every, keepdims)
        )


def sum[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_SumType[T].Value] where conforms_to(T, _BatchShape):
    """Sum exactly along an axis or a list of axes."""
    comptime if T == Batch[Integer] or T == Batch[Rational]:
        return rebind[Batch[_SumType[T].Value]](_exact_axes[0](values, axis.axes, keepdims))
    else:
        return _rounded_sum(values, axis.axes, False, keepdims, None, _ComplexContextArgument())


@fieldwise_init
struct _OrderedFold[T: ImplicitlyCopyable & Deinitable, tree: Bool](_Fold):
    """An ordered sum as a lane kernel: each lane is summed whole, in its
    named order (one task per lane; lanes run in parallel)."""

    comptime Element = Self.T
    comptime Partial = Self.T
    comptime Result = Self.T
    # The source batch's formats, for empty lanes.
    var real: FloatFormat
    var imag: FloatFormat

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.T:
        var values = List[Self.T](capacity=end - start)
        var base = lanes.base(j)
        for k in range(start, end):
            values.append(lanes.at(base, k))
        var lane = Batch[Self.T](values^, shape=[end - start])
        lane._real_format = self.real
        lane._imag_format = self.imag
        comptime if Self.tree:
            return rebind[Self.T](sum_tree(lane))
        else:
            return rebind[Self.T](sum_sequential(lane))

    def combine(self, left: Self.T, right: Self.T, lanes: _Lanes[Self.T], j: Int, k: Int) raises -> Self.T:
        raise Error("Ordered sums do not split lanes.")

    def finish(self, var partial: Self.T) raises -> Self.T:
        return partial^


def _ordered_axes[tree: Bool, T: ImplicitlyCopyable](
    values: T, axes: List[Int], keepdims: Bool,
) raises -> Batch[_SumType[T].Value] where conforms_to(T, _BatchShape):
    comptime E = downcast[T, _BatchShape].Element
    ref batch = rebind[Batch[E]](values)
    return rebind[Batch[_SumType[T].Value]](_reduce_lanes(
        _OrderedFold[E, tree](batch._real_format, batch._imag_format), batch, axes, False, keepdims, False,
    ))


def sum_sequential[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_SumType[T].Value] where conforms_to(T, _BatchShape):
    """Left-to-right sums, rounding each step, along an axis or a list of axes."""
    return _ordered_axes[False](values, axis.axes, keepdims)


def sum_tree[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_SumType[T].Value] where conforms_to(T, _BatchShape):
    """Fixed binary-tree sums along an axis or a list of axes."""
    return _ordered_axes[True](values, axis.axes, keepdims)


def prod[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_ReductionType[T].Value] where conforms_to(T, _BatchShape):
    """Exact products along an axis or a list of axes."""
    return _exact_axes[1](values, axis.axes, keepdims)


def min[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_ExtremeType[T].Value] where conforms_to(T, _BatchShape):
    """The least elements along an axis or a list of axes."""
    return _extreme_lanes[False](values, axis.axes, False, keepdims)


def max[T: ImplicitlyCopyable](values: T, *, var axis: _ReduceAxes, keepdims: Bool = False) raises -> Batch[_ExtremeType[T].Value] where conforms_to(T, _BatchShape):
    """The greatest elements along an axis or a list of axes."""
    return _extreme_lanes[True](values, axis.axes, False, keepdims)


def argmin[T: ImplicitlyCopyable](values: T, *, axis: Int, keepdims: Bool = False) raises -> Batch[Integer] where conforms_to(T, _BatchShape):
    """The index along `axis` of the first least element of each lane."""
    return _arg_lanes[False](values, [axis], False, keepdims)


def argmax[T: ImplicitlyCopyable](values: T, *, axis: Int, keepdims: Bool = False) raises -> Batch[Integer] where conforms_to(T, _BatchShape):
    """The index along `axis` of the first greatest element of each lane."""
    return _arg_lanes[True](values, [axis], False, keepdims)


# ------------------------------------------------------------ running folds


# Ball folds take a ball context; the others take the context lift takes.
comptime _FoldContext[T: ImplicitlyCopyable & Deinitable] = (
    Optional[BallContext] if T == Ball or T == ComplexBall else _ComplexContextArgument
)


def _running[
    T: ImplicitlyCopyable & Deinitable, //,
    integer: def(Integer, Integer) raises thin -> Integer,
    rational: def(Rational, Rational) raises thin -> Rational,
    real: def(_FloatArgument, _FloatArgument, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    complex: def(_ComplexArgument, _ComplexArgument, /, *, context: _ComplexContextArgument) raises thin -> Complex,
    ball: def(_BallArgument, _BallArgument, /, *, context: Optional[BallContext]) raises thin -> Ball,
    complex_ball: def(ComplexBall, ComplexBall, /, *, context: Optional[BallContext]) raises thin -> ComplexBall,
](values: Batch[T], axis: Optional[Int], context: _FoldContext[T]) raises -> Batch[T]:
    """Running folds through `lift(...).accumulate`, over the flattened batch without an axis."""
    var source = values if axis else values.reshape([values.size()])
    var a = axis.value() if axis else 0
    comptime if T == Integer:
        return rebind_var[Batch[T]](lift[integer]().accumulate(source, axis=a, context=rebind[_ComplexContextArgument](context)))
    elif T == Rational:
        return rebind_var[Batch[T]](lift[rational]().accumulate(source, axis=a, context=rebind[_ComplexContextArgument](context)))
    elif T == Float:
        return rebind_var[Batch[T]](lift[real]().accumulate(source, axis=a, context=rebind[_ComplexContextArgument](context)))
    elif T == Complex:
        return rebind_var[Batch[T]](lift[complex]().accumulate(source, axis=a, context=rebind[_ComplexContextArgument](context)))
    elif T == Ball:
        return rebind_var[Batch[T]](lift[ball]().accumulate(source, axis=a, context=rebind[Optional[BallContext]](context)))
    else:
        comptime assert T == ComplexBall, "Running folds take Integer, Rational, Float, Complex or ball elements."
        return rebind_var[Batch[T]](lift[complex_ball]().accumulate(source, axis=a, context=rebind[Optional[BallContext]](context)))


def cumsum[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, axis: Optional[Int] = None, context: _FoldContext[T] = _FoldContext[T](),
) raises -> Batch[T]:
    """Running sums along an axis, like `numpy.cumsum`; without an axis, over
    the batch flattened in row-major order.

    Integer and Rational sums are exact. Each Float or Complex sum is the exact
    sum of its elements rounded once, with `context` when given, so the result
    does not depend on how the elements round; ball sums enclose the exact ones.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        axis: The axis; negative axes count from the end. None flattens first.
        context: The rounding of Float, Complex and ball sums.

    Returns:
        A batch of the same shape, or a vector without an axis.

    Raises:
        On an invalid axis, or a context for exact elements.
    """
    return _running[_integer_add, _rational_add, _float_add, _complex_add, _ball_add, _complex_ball_add](
        values, axis, context
    )


def cumprod[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, axis: Optional[Int] = None, context: _FoldContext[T] = _FoldContext[T](),
) raises -> Batch[T]:
    """Running products along an axis, like `numpy.cumprod`; without an
    axis, over the batch flattened in row-major order.

    Integer and Rational products are exact. Each Float or Complex product is
    exact and rounded once, so its working size grows with the number of
    factors; ball products enclose the exact ones.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        axis: The axis; negative axes count from the end. None flattens first.
        context: The rounding of Float, Complex and ball products.

    Returns:
        A batch of the same shape, or a vector without an axis.

    Raises:
        On an invalid axis, or a context for exact elements.
    """
    return _running[
        _integer_multiply, _rational_multiply, _float_multiply, _complex_multiply, _ball_multiply,
        _complex_ball_multiply,
    ](values, axis, context)
