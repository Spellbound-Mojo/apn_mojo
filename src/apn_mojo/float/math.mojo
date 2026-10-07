"""Float operations with an explicit numerical context.

Each function is one scalar declaration. Arguments accept any exact number
(Integer, Rational, Float, native values and integer literals of any width);
batches apply them with vmap, for example vmap[add]()(xs, ys, context=c).
"""

from .value import Float
from ..common._stable_hash import _StableHash
from .context import ArithmeticContext
from ._input import _FloatInput
from ._format import _FormatSlot, _merge_float_formats
from ._rounding import _compare_scaled, _round_exact_binary
from ..rational.value import Rational
from ._arithmetic import _FloatArgument, _float_operation, _float_operation_of, _call_context
from ..integer.value import Integer
from ._functions import _sqrt_float, _scale_float, _pow_float, _fma_float


def add(
    left: _FloatArgument,
    right: _FloatArgument,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded add, `left + right`.

    The exact result is rounded once. Operands may be Float, Integer, Rational,
    integer literals of any width or typed native numbers, in either order, and are
    never rounded first.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        A Float in the context's format, or the merged operand format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_float_operation_of[0](left, right, context))


def subtract(
    left: _FloatArgument,
    right: _FloatArgument,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded subtract, `left - right`.

    The exact result is rounded once. Operands may be Float, Integer, Rational,
    integer literals of any width or typed native numbers, in either order, and are
    never rounded first.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        A Float in the context's format, or the merged operand format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_float_operation_of[1](left, right, context))


def multiply(
    left: _FloatArgument,
    right: _FloatArgument,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded multiply, `left * right`.

    The exact result is rounded once. Operands may be Float, Integer, Rational,
    integer literals of any width or typed native numbers, in either order, and are
    never rounded first.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        A Float in the context's format, or the merged operand format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_float_operation_of[2](left, right, context))


def divide(
    left: _FloatArgument,
    right: _FloatArgument,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded divide, `left / right`.

    The exact result is rounded once. Operands may be Float, Integer, Rational,
    integer literals of any width or typed native numbers, in either order, and are
    never rounded first. A finite nonzero value over zero is a signed infinity.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        A Float in the context's format, or the merged operand format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_float_operation_of[3](left, right, context))


def square(
    value: _FloatArgument, *, context: Optional[ArithmeticContext] = None
) raises -> Float:
    """The correctly rounded square.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode and traps; by default the operand's format, rounded to nearest-even.

    Returns:
        `value * value`, rounded once.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_float_operation_of[2](value, value, context))


def sqrt(
    value: _FloatArgument, *, context: Optional[ArithmeticContext] = None
) raises -> Float:
    """The correctly rounded square root.

    `-0` stays `-0`; a negative nonzero value gives NaN, which is invalid.

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode and traps; by default the operand's format, rounded to nearest-even.

    Returns:
        The square root, rounded once.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_sqrt_float(value, context))


def pow_int(
    value: _FloatArgument,
    exponent: Integer,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded power for a signed integral exponent.

    A zero exponent gives one, even for NaN and infinity. A negative power of zero
    is an infinity with divide-by-zero.

    Args:
        value: The base; exact bases need a context.
        exponent: The exponent, of any size.
        context: The output format, rounding mode and traps; by default the operand's format, rounded to nearest-even.

    Returns:
        `value ** exponent`, rounded once.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_pow_float(value, exponent, context))


def ldexp(
    value: _FloatArgument,
    exponent: Integer,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The correctly rounded `value * 2**exponent`.

    Args:
        value: The operand; exact operands need a context.
        exponent: The power of two, of any size.
        context: The output format, rounding mode and traps; by default the operand's format, rounded to nearest-even.

    Returns:
        The scaled value, rounded once.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_scale_float(value, exponent, context))


def fma(
    a: _FloatArgument,
    b: _FloatArgument,
    c: _FloatArgument,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """Fused multiply-add: `a * b + c` with one rounding.

    The exact product is added to `c` before the only rounding, so its
    cancellation and range never round early. Zero times infinity is invalid even
    with a NaN addend.

    Args:
        a: The first factor.
        b: The second factor.
        c: The addend.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        `a * b + c`, rounded once.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    return Float(_rounded=_fma_float(a, b, c, context))


def abs(value: Float) raises -> Float:
    """The absolute value, in the same format.

    Args:
        value: The Float.

    Returns:
        The value with its sign cleared; NaN stays NaN.

    Raises:
        Never for a valid Float.
    """
    return value.__abs__()


def stable_hash(x: Float) -> UInt64:
    """A 64-bit hash of the representation, stable across processes and releases.

    The algorithm is APNH-64: tag 3, then the precision, the exponent bounds,
    the class and the sign, and for a finite nonzero value the exponent and the
    significand. Unlike Mojo's `Hasher`, its output never changes, so stored
    hashes stay valid. It hashes representations, as `FloatKey` compares them:
    `0.0` and `-0.0`, or 1 at 53 and at 128 bits, hash differently.

    Args:
        x: The Float.

    Returns:
        The hash.
    """
    var hash = _StableHash(3)
    x._hash_into(hash)
    return hash.finish()


def floor(value: Float) raises -> Integer:
    """Round a Float toward negative infinity, to an Integer (numpy's `floor`).

    Args:
        value: The operand.

    Returns:
        The largest Integer not above the value.

    Raises:
        For infinity and NaN.
    """
    return value.floor()


def ceil(value: Float) raises -> Integer:
    """Round a Float toward positive infinity, to an Integer (numpy's `ceil`).

    Args:
        value: The operand.

    Returns:
        The smallest Integer not below the value.

    Raises:
        For infinity and NaN.
    """
    return value.ceil()


def trunc(value: Float) raises -> Integer:
    """Round a Float toward zero, to an Integer (numpy's `trunc`).

    Args:
        value: The operand.

    Returns:
        The integer part.

    Raises:
        For infinity and NaN.
    """
    return value.trunc()


def round(value: Float) raises -> Integer:
    """Round a Float to the nearest Integer, a half to the even neighbour, to an Integer (numpy's `round`).

    Args:
        value: The operand.

    Returns:
        The nearest Integer: 2.5 gives 2, 3.5 gives 4 and -2.5 gives -2.

    Raises:
        For infinity and NaN.
    """
    return value.round()


def reciprocal(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The correctly rounded `1 / value` (numpy's `reciprocal`).

    Args:
        value: The operand; exact operands need a context.
        context: The output format, rounding mode and traps; by default the operand's format, rounded to nearest-even.

    Returns:
        The reciprocal, rounded once.

    Raises:
        When a trapped condition occurs, or an exact operand is given without a context.
    """
    return Float(_rounded=_float_operation_of[3](Integer(1), value, context))


def _exact_rational(x: _FloatInput) raises -> Rational:
    """The value of a finite exact argument record."""
    var numerator = x.numerator
    var denominator = x.denominator
    if x.scale >= 0:
        numerator = numerator << Int(x.scale)
    else:
        denominator = denominator << Int(-x.scale)
    var value = Rational(numerator, denominator)
    return -value if x.negative else value


def _order(left: _FloatArgument, right: _FloatArgument) raises -> Int:
    """-1, 0 or 1 as `left` lies below, at or above `right`, compared exactly;
    2 when either is NaN."""
    ref a = left.value
    ref b = right.value
    # Two finite binary values, as Floats give: the signs, then the magnitudes.
    if a.kind == 1 and b.kind == 1 and a.denominator._is_one() and b.denominator._is_one():
        if a.negative != b.negative:
            return -1 if a.negative else 1
        var order = _compare_scaled(a.numerator, b.numerator, b.scale - a.scale)
        return -order if a.negative else order
    var x = Float._from_argument(left)
    if x:
        return x.value()._compare_input(right.value)
    var y = Float._from_argument(right)
    if y:
        var order = y.value()._compare_input(left.value)
        return order if order == 2 else -order
    if a.kind == 3 or b.kind == 3:
        return 2
    var sa = 0 if a.kind == 0 else (-1 if a.negative else 1)
    var sb = 0 if b.kind == 0 else (-1 if b.negative else 1)
    if sa != sb or sa == 0:
        return -1 if sa < sb else Int(sa > sb)
    if a.kind == 2 or b.kind == 2:
        if a.kind == b.kind:
            return 0
        return sa if a.kind == 2 else -sa
    return _exact_rational(a)._compare(_exact_rational(b))


def _chosen(value: _FloatInput, target: ArithmeticContext) raises -> Float:
    """The operand maximum, minimum or clip chose, in the output format: a
    Float of that format as it stands, anything else rounded once."""
    var format = target.format()
    if (
        value.kind == 1 and value.denominator._is_one() and not format._is_exact()
        and value.numerator.magnitude_bit_length() == format.precision()
    ):
        var exact = _round_exact_binary(value.numerator, value.negative, value.scale, target, False)
        if exact.kind >= 0:
            return Float(_rounded=exact^)
    return Float(_rounded=Float._rounded_input(value, target))


def _selects_first(left: _FloatArgument, right: _FloatArgument, maximum: Bool) raises -> Bool:
    """numpy's choice: `left` when it is NaN or `left >= right` (`<=` for
    the minimum), else `right`."""
    if left.value.kind == 3:
        return True
    var order = _order(left, right)
    return order != 2 and (order >= 0 if maximum else order <= 0)


def maximum(left: _FloatArgument, right: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The larger of two numbers (numpy's `maximum`): compared exactly, the
    first when they are equal, and NaN when either is NaN; rounded once.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        The chosen operand in the output format, unchanged when it is a Float of that format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    var first = _selects_first(left, right, True)
    return _chosen(left.value if first else right.value, _call_context(left, right, context))


def minimum(left: _FloatArgument, right: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The smaller of two numbers (numpy's `minimum`): compared exactly, the
    first when they are equal, and NaN when either is NaN; rounded once.

    Args:
        left: The first operand.
        right: The second operand.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        The chosen operand in the output format, unchanged when it is a Float of that format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    var first = _selects_first(left, right, False)
    return _chosen(left.value if first else right.value, _call_context(left, right, context))


def clip(value: _FloatArgument, a_min: _FloatArgument, a_max: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`value` limited to `[a_min, a_max]`: `minimum(maximum(value, a_min),
    a_max)` (numpy's `clip`), with both choices exact and one rounding; NaN
    in any operand gives NaN.

    Args:
        value: The operand.
        a_min: The lower limit.
        a_max: The upper limit.
        context: The output format, rounding mode and traps; by default the merged operand formats, rounded to nearest-even.

    Returns:
        The chosen operand in the output format.

    Raises:
        When a trapped condition occurs, or exact operands are given without a context.
    """
    var low = value if _selects_first(value, a_min, True) else a_min
    var chosen = low if _selects_first(low, a_max, False) else a_max
    var target = context.value() if context else ArithmeticContext(_format_of=_merge_float_formats(
        _FormatSlot(_merge_float_formats(
            value.format, a_min.format,
            left_native_precision=value.native_precision, right_native_precision=a_min.native_precision,
        ), True) if (value.format or a_min.format or value.native_precision or a_min.native_precision) else _FormatSlot(None),
        a_max.format,
        right_native_precision=a_max.native_precision,
    ))
    return _chosen(chosen.value, target)
