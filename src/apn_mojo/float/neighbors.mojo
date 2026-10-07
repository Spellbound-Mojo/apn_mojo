"""Neighbouring values, units in the last place, and closeness of Floats.

A format's finite values, from the most negative to the most positive, are
numbered consecutively: zero is 0, the smallest positive value 1, and each
infinity lies one step beyond the largest finite value. Neighbours and
distances follow that numbering, within one format.
"""

from ..integer.value import Integer
from .value import Float
from .context import ArithmeticContext, FloatFormat
from .status import NumericStatus
from ._rounding import _RoundedBinary
from ._arithmetic import _FloatArgument


def _make(kind: Int, negative: Bool, significand: Integer, exponent: Int, format: FloatFormat) -> Float:
    return Float(_rounded=_RoundedBinary(kind, negative, significand, exponent, format, NumericStatus()))


def _next_up(x: Float) raises -> Float:
    """The next value above `x` in its format.

    `_next_up` of a zero is the smallest positive value, `2**(emin - 1)`; of the
    largest finite value, `inf`; of the negative value nearest zero, `-0`.
    Infinity stays infinity, NaN stays NaN, and `-inf` gives the most negative
    finite value.

    Args:
        x: Any Float.

    Returns:
        The next representable value, in `x`'s format.

    Raises:
        Only on a checked size error.
    """
    if x.is_nan():
        return x
    var format = x._format
    var precision = format.precision()
    var top = Integer(1) << precision
    var low = Integer(1) << (precision - 1)
    if x._kind == 2:
        return x if not x._negative else _make(1, True, top - 1, format.emax(), format)
    if x._kind == 0:
        return _make(1, False, low, format.emin(), format)
    var significand = x._significand
    var exponent = x._exponent
    if not x._negative:
        significand += 1
        if significand == top:
            significand = low
            exponent += 1
        if exponent > format.emax():
            return _make(2, False, Integer(0), 0, format)
        return _make(1, False, significand, exponent, format)
    significand -= 1
    if significand < low:
        significand = top - 1
        exponent -= 1
    if exponent < format.emin():
        return _make(0, True, Integer(0), 0, format)
    return _make(1, True, significand, exponent, format)


def _next_down(x: Float) raises -> Float:
    """The next value below `x` in its format: `-_next_up(-x)`.

    Args:
        x: Any Float.

    Returns:
        The next representable value below, in `x`'s format.

    Raises:
        Only on a checked size error.
    """
    return -_next_up(-x)


def _ulp(x: Float) raises -> Float:
    """The unit in the last place of `x`: the spacing of its format at `x`.

    For a finite nonzero `x` it is `2**(exponent(x) - precision)`; for a zero,
    the distance to the smallest positive value, which is that value. A format
    has no subnormals, so in the lowest `precision - 1` binades the spacing is
    below the smallest positive value and cannot be represented.

    Args:
        x: A finite Float.

    Returns:
        The spacing, in `x`'s format.

    Raises:
        For infinity and NaN, and when the spacing is below the smallest
        positive value of the format.
    """
    if not x.is_finite():
        raise Error(
            "Cannot take the ulp of infinity or NaN; check is_finite() first."
            " The destination is unchanged."
        )
    var format = x._format
    var low = Integer(1) << (format.precision() - 1)
    if x._kind == 0:
        return _make(1, False, low, format.emin(), format)
    var exponent = x._exponent - format.precision() + 1
    if exponent < format.emin():
        raise Error(
            "Cannot represent ulp(x) in x's format: it is below the smallest"
            " positive value, 2**(emin - 1); use a format with a lower emin."
            " The destination is unchanged."
        )
    return _make(1, False, low, exponent, format)


def nextafter(x: Float, toward: _FloatArgument) raises -> Float:
    """The next value after `x` in its format toward `toward` (numpy's
    `nextafter`).

    `toward` may be any number and is compared with `x` exactly. When they are
    equal the result is `toward` (so `nextafter(-0, +0)` is +0); a NaN in
    either gives NaN. Past the largest finite value the next value is
    infinity, and from an infinity toward a finite value it is the largest
    finite value.

    Args:
        x: Any Float.
        toward: The direction, any number.

    Returns:
        A Float in `x`'s format.

    Raises:
        Only on a checked size error.
    """
    var order = x._compare_input(toward.value)
    if order == 2:
        return x if x.is_nan() else _make(3, False, Integer(0), 0, x._format)
    if order == 0:
        return _make(0, toward.value.negative, Integer(0), 0, x._format) if x.is_zero() else x
    return _next_up(x) if order < 0 else _next_down(x)


def spacing(x: Float) raises -> Float:
    """The distance from `x` to the next value of its format away from zero,
    with `x`'s sign (numpy's `spacing`): `2**(exponent(x) - precision)` for
    a finite nonzero `x`, and the smallest positive value for a zero; NaN for
    infinity and NaN.

    A format has no subnormals, so in the lowest `precision - 1` binades the
    spacing is below the smallest positive value and cannot be represented.

    Args:
        x: Any Float.

    Returns:
        The signed spacing, in `x`'s format.

    Raises:
        When the spacing is below the smallest positive value of the format.
    """
    if not x.is_finite():
        return _make(3, False, Integer(0), 0, x._format)
    var size = _ulp(x)
    return -size if x._negative else size


def _ordinal(x: Float) raises -> Integer:
    """The position of `x` in its format's numbering of values."""
    if x._kind == 0:
        return Integer(0)
    var format = x._format
    var low = Integer(1) << (format.precision() - 1)
    var index: Integer
    if x._kind == 2:
        index = (Integer(format.emax()) - format.emin() + 1) * low + 1
    else:
        index = (Integer(x._exponent) - format.emin()) * low + (x._significand - low) + 1
    return -index if x._negative else index


def ulp_distance(a: Float, b: Float) raises -> Integer:
    """How many steps of the coarser format separate `a` and `b`.

    Both round (nearest-even) to the smaller of the two precisions; the result
    is the number of representable values strictly between them, plus one if
    they differ, so equal values give 0 and neighbours 1. The two zeros are
    equal, and each infinity is one step beyond the largest finite value.

    Args:
        a: A Float that is not NaN.
        b: A Float that is not NaN, with the same exponent bounds as `a`.

    Returns:
        The nonnegative distance.

    Raises:
        For NaN, or when the exponent bounds differ.
    """
    if a.is_nan() or b.is_nan():
        raise Error(
            "Cannot measure the ulp distance to NaN; check is_nan() first."
            " The destination is unchanged."
        )
    var fa = a._format
    var fb = b._format
    if fa.emin() != fb.emin() or fa.emax() != fb.emax():
        raise Error(
            "Cannot measure the ulp distance across exponent bounds; round"
            " both values into one format first. The destination is unchanged."
        )
    var context = ArithmeticContext(_format_of=FloatFormat(min(fa.precision(), fb.precision()), emin=fa.emin(), emax=fa.emax()))
    return abs(_ordinal(Float(a, context=context)) - _ordinal(Float(b, context=context)))


def equal_at_precision(a: Float, b: Float, bits: Integer) raises -> Bool:
    """Whether `a` and `b` round (nearest-even) to the same value at `bits` bits.

    The comparison is by value, so the two zeros are equal. It rounds into the
    default exponent bounds, which hold every Float without overflow.

    Args:
        a: A Float that is not NaN.
        b: A Float that is not NaN.
        bits: The precision to compare at, at least 1.

    Returns:
        True when both round to the same value.

    Raises:
        For NaN, or when `bits` is not a valid precision.
    """
    if a.is_nan() or b.is_nan():
        raise Error(
            "Cannot compare NaN at a precision; check is_nan() first. The"
            " destination is unchanged."
        )
    var context = ArithmeticContext(_format_of=FloatFormat(Int(bits)))
    return Float(a, context=context) == Float(b, context=context)
