"""Ball arithmetic: each midpoint rounds once to nearest, and each radius
bounds the propagated radii plus that rounding's error, rounded up.

The radius formulas are those of the requirements' Appendix E. A rounding
error is added only when the midpoint's rounding was inexact, so arithmetic on
exact balls stays exact while it fits the precision.
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from std.atomic import Ordering
from ..float._arithmetic import _FloatArgument, _float_argument, _float_operation, _binary_float_sum, _binary_quotient, _aligned_sum
from ..float._multiplication import _binary_product
from ..float._functions import _sqrt_float, _pow_float, _scale_float, _fma_float
from ..float._rounding import _RoundedBinary, _round_exact_binary, _exponent_add, _wide_rounding, _short_magnitude, _put_magnitude
from .context import BallContext
from ._radius import _Radius, _products_up, _up_word, _down_word
from .value import Ball, _BallArgument, _FINITE, _UNBOUNDED, _INDETERMINATE


def _working(context: Optional[BallContext], a: Int, b: Int = 0, c: Int = 0) -> Int:
    """The result precision: the context's, else the largest operand's (at
    least 2), else 128 for exact operands alone."""
    if context and context.value().precision():
        return context.value().precision().value()
    var bits = max(a, max(b, c))
    return max(2, bits) if bits else 128


def _nearest(bits: Int) raises -> ArithmeticContext:
    """Nearest-even at `bits` bits with the default exponent bounds: a
    precision in range skips the public constructors' checks, about 60
    instructions an operation."""
    if bits >= 1 and bits <= FloatFormat.MAX_PRECISION:
        return ArithmeticContext(
            _format_of=FloatFormat(_validated=(bits, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX))
        )
    return ArithmeticContext(format=FloatFormat(bits))


def _directed(bits: Int, up: Bool) raises -> ArithmeticContext:
    return ArithmeticContext(
        format=FloatFormat(bits),
        rounding=RoundingMode.toward_positive if up else RoundingMode.toward_negative,
    )


def _error(record: _RoundedBinary, bits: Int) -> _Radius:
    """Half a unit in the last place of an inexact nearest rounding."""
    if not record.status.inexact():
        return _Radius.zero()
    if record.kind != 1:
        return _Radius.power_of_two(FloatFormat.DEFAULT_EMIN)
    return _Radius.power_of_two(record.exponent - bits - 1)


def _from_record(var record: _RoundedBinary, radius: _Radius, bits: Int) raises -> Ball:
    """A finite ball from a rounded midpoint, or unbounded on overflow."""
    if record.kind == 2 or radius.infinite:
        return Ball.unbounded(bits)
    var error = _error(record, bits)
    return Ball(_midpoint=Float(_rounded=record^), _radius=radius.add(error), _kind=_FINITE)


def _hull(low: Float, high: Float, bits: Int) raises -> Ball:
    """A ball at `bits` bits containing `[low, high]`, for finite `low <= high`:
    the midpoint is their mean rounded to nearest, and the radius covers both
    ends."""
    var sum = Float(_rounded=_float_operation(low, high, 0, _nearest(bits + 1)))
    var middle = Float(_rounded=_scale_float(sum, Integer(-1), _nearest(bits)))
    var above = Float(_rounded=_float_operation(high, middle, 1, _directed(30, True)))
    var below = Float(_rounded=_float_operation(middle, low, 1, _directed(30, True)))
    return Ball(_midpoint=middle, _radius=_Radius.upper(above).max(_Radius.upper(below)), _kind=_FINITE)


def _hull_from(low: Float, high: Float, bits: Int) raises -> Ball:
    """A ball at `bits` bits containing `[low, high]`, for `0 <= low <= high`,
    whose lower end stays within one unit in the last place of its midpoint
    below `low`, and is exactly 0 from `low = 0`. `_hull` centres its midpoint,
    so the rounding of its radius, a 2**-29 share of the whole width, can
    carry the lower end of a wide range such as [1e-9, 50] past 0; this one
    is for bounds a ball must not cross that way: a norm, a magnitude, a
    nonnegative half-sum. The radius is half the width plus that unit,
    rounded up, and the midpoint `low` plus the radius, rounded down; so the
    lower end is at most `low`, and the upper end at least `high`. The
    midpoint takes enough bits beyond `bits` (up to 4096) for that unit to
    stay below half of `low`, so a positive range stays positive."""
    if low.is_zero():
        # The midpoint equals the radius: both hold high/2 rounded up to 30 bits.
        var half = Float(_rounded=_scale_float(high, Integer(-1), _directed(min(bits, 30), True)))
        return Ball(_midpoint=half, _radius=_Radius.upper(half), _kind=_FINITE)
    var precision = max(bits, min(bits + 4096, high._exponent - low._exponent + 3))
    var width = Float(_rounded=_float_operation(high, low, 1, _directed(30, True)))
    # A unit in the last place of any midpoint up to 2 high.
    var unit = _Radius.power_of_two(high._exponent + 1 - precision)
    var radius = _Radius.upper(width).scale2(-1).add(unit)
    var middle = Float(_rounded=_float_operation(low, radius.to_float(), 0, _directed(precision, False)))
    return Ball(_midpoint=middle, _radius=radius, _kind=_FINITE)


def _magnitude_range(x: Ball) raises -> Tuple[Float, Float]:
    """The least and greatest absolute value over a finite ball, exactly."""
    var low = x._exact_lower()
    var high = x._exact_upper()
    var a = low.__abs__()
    var b = high.__abs__()
    var top = a if a > b else b
    if not low._negative or low.is_zero():
        return (low, high)
    if high._negative and not high.is_zero():
        return (b, a)
    return (Float.zero(context=ArithmeticContext(format=FloatFormat(1))), top)


def _norm_bounds(re: Ball, im: Ball, bits: Int) raises -> Tuple[Float, Float]:
    """Bounds of `re**2 + im**2` over two finite balls: the squares of the
    least magnitudes summed rounding down, of the greatest rounding up."""
    var x = _magnitude_range(re)
    var y = _magnitude_range(im)
    var down = _directed(bits, False)
    var up = _directed(bits, True)
    var low = _float_operation(
        Float(_rounded=_float_operation(x[0], x[0], 2, down)), Float(_rounded=_float_operation(y[0], y[0], 2, down)), 0, down,
    )
    var high = _float_operation(
        Float(_rounded=_float_operation(x[1], x[1], 2, up)), Float(_rounded=_float_operation(y[1], y[1], 2, up)), 0, up,
    )
    return (Float(_rounded=low), Float(_rounded=high))


def _norm(re: Ball, im: Ball, bits: Int) raises -> Ball:
    """`re**2 + im**2` over two finite balls, from the parts' magnitude
    bounds and never below the least of them. Midpoint-radius squares reach
    below 0 once a radius passes about 0.41 of its midpoint, and a sum's radius
    rounds up past a small term: either makes the norm of a rectangle away from
    0 seem to reach it, which loses a quotient by it or its logarithm."""
    var bounds = _norm_bounds(re, im, bits)
    return _hull_from(bounds[0], bounds[1], bits)


def _magnitude(x: _FloatArgument) -> _FloatArgument:
    var result = x
    result.value.negative = False
    return result


def _is_exact_zero(x: _BallArgument) -> Bool:
    return x.kind == _FINITE and x.radius.is_zero() and x.midpoint.value.kind == 0


def _reaches_zero(x: _BallArgument) raises -> Bool:
    """Whether a finite operand contains 0, decided exactly: in radius
    arithmetic for a dyadic midpoint, by an exact subtraction otherwise."""
    if x.radius.is_zero():
        return x.midpoint.value.kind == 0
    if x.midpoint.value.denominator._is_one():
        return x.radius.covers(x.midpoint.value)
    var gap = _float_operation(_magnitude(x.midpoint), x.radius.to_float(), 1, _exact_context())
    return gap.kind == 0 or gap.negative


def _gap_down(x: _BallArgument) raises -> _Radius:
    """A lower bound of `abs(m) - r` for an operand away from 0."""
    return _Radius.lower_gap(x.midpoint.value, x.radius)


def _rounded_ball(x: _BallArgument, bits: Int) raises -> Ball:
    """The operand with its midpoint rounded to `bits`."""
    if x.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if x.kind == _UNBOUNDED:
        return Ball.unbounded(bits)
    return _from_record(_float_operation(x.midpoint, Integer(0), 0, _nearest(bits)), x.radius, bits)


def _negate(x: Ball) raises -> Ball:
    return Ball(_midpoint=-x._midpoint, _radius=x._radius, _kind=x._kind)


def _sum(a: _BallArgument, b: _BallArgument, subtract: Bool, context: Optional[BallContext]) raises -> Ball:
    # E.2: R = r_x + r_y + e(M).
    var bits = _working(context, a.precision, b.precision)
    if a.kind == _INDETERMINATE or b.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if a.kind == _UNBOUNDED or b.kind == _UNBOUNDED:
        return Ball.unbounded(bits)
    var record = _float_operation(a.midpoint, b.midpoint, 1 if subtract else 0, _nearest(bits))
    return _from_record(record^, a.radius.add(b.radius), bits)


def _product_radius(upper_x: _Radius, r_x: _Radius, upper_y: _Radius, r_y: _Radius) -> _Radius:
    """E.3's `|m_x| r_y + |m_y| r_x + r_x r_y`, from upper bounds of the
    midpoints' magnitudes, rounded up once."""
    return _products_up(upper_x, r_y, upper_y, r_x, r_x, r_y)


def _quotient_numerator(upper_x: _Radius, r_x: _Radius, upper_y: _Radius, r_y: _Radius) -> _Radius:
    """E.6's numerator, `|m_x| r_y + |m_y| r_x`, rounded up once."""
    return _products_up(upper_x, r_y, upper_y, r_x, _Radius.zero(), _Radius.zero())


def _product(a: _BallArgument, b: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # E.3: R = |m_x| r_y + |m_y| r_x + r_x r_y + e(M); exact 0 times unbounded is 0.
    var bits = _working(context, a.precision, b.precision)
    if a.kind == _INDETERMINATE or b.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if a.kind == _UNBOUNDED or b.kind == _UNBOUNDED:
        if _is_exact_zero(a) or _is_exact_zero(b):
            return Ball(Integer(0), precision=bits)
        return Ball.unbounded(bits)
    var record = _float_operation(a.midpoint, b.midpoint, 2, _nearest(bits))
    var radius = _product_radius(
        _Radius.upper_input(a.midpoint.value), a.radius, _Radius.upper_input(b.midpoint.value), b.radius
    )
    return _from_record(record^, radius, bits)


def _quotient(a: _BallArgument, b: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # E.6: with d = |m_y| - r_y > 0, R = (|m_x| r_y + |m_y| r_x) / (|m_y| d) + e(M).
    var bits = _working(context, a.precision, b.precision)
    if a.kind == _INDETERMINATE or b.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if b.kind == _UNBOUNDED or _reaches_zero(b):
        return Ball.indeterminate(bits)
    if a.kind == _UNBOUNDED:
        return Ball.unbounded(bits)
    var record = _float_operation(a.midpoint, b.midpoint, 3, _nearest(bits))
    var numerator = _quotient_numerator(
        _Radius.upper_input(a.midpoint.value), a.radius, _Radius.upper_input(b.midpoint.value), b.radius
    )
    var denominator = _Radius.lower_input(b.midpoint.value).multiply_down(_gap_down(b))
    return _from_record(record^, numerator.divide(denominator), bits)


def _square(x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # E.4: R = 2 |m_x| r_x + r_x**2 + e(M).
    var bits = _working(context, x.precision)
    if x.kind != _FINITE:
        return Ball.indeterminate(bits) if x.kind == _INDETERMINATE else Ball.unbounded(bits)
    var record = _float_operation(x.midpoint, x.midpoint, 2, _nearest(bits))
    var radius = _Radius.upper_input(x.midpoint.value).multiply(x.radius).scale2(1).add(x.radius.multiply(x.radius))
    return _from_record(record^, radius, bits)


def _power(x: _BallArgument, n: Integer, context: Optional[BallContext]) raises -> Ball:
    # E.7: for n >= 3, R = n (|m_x| + r_x)**(n - 1) r_x + e(M).
    var bits = _working(context, x.precision)
    if x.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if not n:
        return Ball(Integer(1), precision=bits)
    if n < 0:
        return _quotient(_BallArgument(Integer(1)), _BallArgument(_power(x, -n, context)), context)
    if x.kind == _UNBOUNDED:
        return Ball.unbounded(bits)
    if n == 1:
        return _rounded_ball(x, bits)
    if n == 2:
        return _square(x, context)
    var record = _pow_float(x.midpoint, n, _nearest(bits))
    var base = _Radius.upper_input(x.midpoint.value).add(x.radius)
    var radius = _Radius.upper_input(_float_argument(n).value).multiply(base.power(n - 1)).multiply(x.radius)
    return _from_record(record^, radius, bits)


def _sqrt(x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # E.9: the domain is m_x >= r_x; a lower end of exactly 0 gives [0, sqrt(m_x + r_x)].
    var bits = _working(context, x.precision)
    if x.kind != _FINITE:
        return Ball.indeterminate(bits)
    ref m = x.midpoint.value
    if x.radius.is_zero():
        if m.negative and m.kind != 0:
            return Ball.indeterminate(bits)
        return _from_record(_sqrt_float(x.midpoint, _nearest(bits)), _Radius.zero(), bits)
    if m.kind == 1 and not m.negative and m.denominator._is_one() and not x.radius.covers(m):
        # m - r > 0, decided exactly. For |t| <= r,
        # |sqrt(m + t) - sqrt(m)| = |t| / (sqrt(m + t) + sqrt(m)) <= r / (sqrt(m - r) + sqrt(m)),
        # with the two roots bounded below in radius arithmetic.
        var low = _Radius.lower_gap(m, x.radius).sqrt_down().add_down(_Radius.lower_input(m).sqrt_down())
        return _from_record(_sqrt_float(x.midpoint, _nearest(bits)), x.radius.divide(low), bits)
    var radius = x.radius.to_float()
    var low = _float_operation(x.midpoint, radius, 1, _exact_context())
    if low.negative and low.kind != 0:
        return Ball.indeterminate(bits)
    if low.kind == 0:
        var top = Float(_rounded=_float_operation(x.midpoint, radius, 0, _directed(bits, True)))
        var root = Float(_rounded=_sqrt_float(top, _directed(bits, True)))
        var middle = Float(_rounded=_scale_float(root, Integer(-1), _nearest(bits)))
        return Ball(_midpoint=middle, _radius=_Radius.upper(middle), _kind=_FINITE)
    var record = _sqrt_float(x.midpoint, _nearest(bits))
    var below = Float(_rounded=_float_operation(x.midpoint, radius, 1, _directed(bits, False)))
    var root_low = _Radius.lower(Float(_rounded=_sqrt_float(below, _directed(30, False))))
    var root_mid = _Radius.lower(Float(_rounded=_sqrt_float(x.midpoint, _directed(30, False))))
    return _from_record(record^, x.radius.divide(root_low.add_down(root_mid)), bits)


def _abs_ball(x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # E.8: away from 0, +-X; otherwise [0, |m_x| + r_x].
    var bits = _working(context, x.precision)
    if x.kind != _FINITE:
        return Ball.indeterminate(bits) if x.kind == _INDETERMINATE else Ball.unbounded(bits)
    if not _reaches_zero(x):
        return _from_record(_float_operation(_magnitude(x.midpoint), Integer(0), 0, _nearest(bits)), x.radius, bits)
    var top = Float(_rounded=_float_operation(_magnitude(x.midpoint), x.radius.to_float(), 0, _directed(bits, True)))
    var middle = Float(_rounded=_scale_float(top, Integer(-1), _nearest(bits)))
    return Ball(_midpoint=middle, _radius=_Radius.upper(middle), _kind=_FINITE)


def _scale2(x: _BallArgument, k: Integer, context: Optional[BallContext]) raises -> Ball:
    # E.1: both midpoint and radius scale exactly.
    var bits = _working(context, x.precision)
    if x.kind != _FINITE:
        return Ball.indeterminate(bits) if x.kind == _INDETERMINATE else Ball.unbounded(bits)
    var record = _scale_float(x.midpoint, k, _nearest(bits))
    return _from_record(record^, x.radius.scale2(Int(k)), bits)


def _fma(a: _BallArgument, b: _BallArgument, c: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    # The midpoint fma rounds once; R = |m_x| r_y + |m_y| r_x + r_x r_y + r_z + e(M).
    var bits = _working(context, a.precision, b.precision, c.precision)
    if a.kind == _INDETERMINATE or b.kind == _INDETERMINATE or c.kind == _INDETERMINATE:
        return Ball.indeterminate(bits)
    if a.kind == _UNBOUNDED or b.kind == _UNBOUNDED:
        if _is_exact_zero(a) or _is_exact_zero(b):
            return _rounded_ball(c, bits)
        return Ball.unbounded(bits)
    if c.kind == _UNBOUNDED:
        return Ball.unbounded(bits)
    var record = _fma_float(a.midpoint, b.midpoint, c.midpoint, _nearest(bits))
    var radius = _Radius.upper_input(a.midpoint.value).multiply(b.radius).add(
        _Radius.upper_input(b.midpoint.value).multiply(a.radius)
    ).add(a.radius.multiply(b.radius)).add(c.radius)
    return _from_record(record^, radius, bits)


# ------------------------------------------------ Balls at a given precision
#
# The kernels' arithmetic: two Balls at `bits` bits, with the results of
# `_sum`, `_product`, `_quotient` and `_scale2` at that precision, the same
# midpoints and radii. A finite nonzero midpoint goes straight to the Float
# operation's binary core, which `_float_operation` reaches for it too,
# without argument records, which copy each significand, or context objects;
# a midpoint's magnitude is bounded only where a radius multiplies it. Any
# other operand takes the general function.


def _scale_of(x: Float) -> Int128:
    """The scale of a finite Float's significand: `x = m 2**scale`."""
    return Int128(x._exponent) - Int128(x.precision())


@always_inline
def _plain(a: Ball, b: Ball) -> Bool:
    """Whether both are finite balls with finite nonzero midpoints."""
    return a._kind == _FINITE and b._kind == _FINITE and a._midpoint._kind == 1 and b._midpoint._kind == 1


def _at(bits: Int) raises -> Optional[BallContext]:
    return Optional[BallContext](BallContext(bits))


def _ball_sum(a: Ball, b: Ball, subtract: Bool, bits: Int) raises -> Ball:
    """`a + b`, or `a - b`, at `bits` bits."""
    if not _plain(a, b):
        return _sum(_BallArgument(a), _BallArgument(b), subtract, _at(bits))
    ref x = a._midpoint
    ref y = b._midpoint
    var record = _binary_float_sum(
        x._significand, x._negative, _scale_of(x), y._significand, y._negative != subtract, _scale_of(y), _nearest(bits), False
    )
    return _from_record(record^, a._radius.add(b._radius), bits)


def _ball_product(a: Ball, b: Ball, bits: Int) raises -> Ball:
    """`a * b` at `bits` bits."""
    if not _plain(a, b):
        return _product(_BallArgument(a), _BallArgument(b), _at(bits))
    ref x = a._midpoint
    ref y = b._midpoint
    var record = _binary_product(
        x._significand, y._significand, x._negative != y._negative, _exponent_add(_scale_of(x), _scale_of(y)), _nearest(bits), False
    )
    var upper_x = _Radius.upper(x) if not b._radius.is_zero() else _Radius.zero()
    var upper_y = _Radius.upper(y) if not a._radius.is_zero() else _Radius.zero()
    return _from_record(record^, _product_radius(upper_x, a._radius, upper_y, b._radius), bits)


def _ball_quotient(a: Ball, b: Ball, bits: Int) raises -> Ball:
    """`a / b` at `bits` bits; indeterminate where b contains 0."""
    if not _plain(a, b):
        return _quotient(_BallArgument(a), _BallArgument(b), _at(bits))
    ref x = a._midpoint
    ref y = b._midpoint
    if not b._radius.is_zero() and b._radius.covers_float(y):
        return Ball.indeterminate(bits)
    var record = _binary_quotient(
        x._significand, y._significand, x._negative != y._negative, _exponent_add(_scale_of(x), -_scale_of(y)), _nearest(bits), False
    )
    var upper_x = _Radius.upper(x) if not b._radius.is_zero() else _Radius.zero()
    var upper_y = _Radius.upper(y) if not a._radius.is_zero() else _Radius.zero()
    var numerator = _quotient_numerator(upper_x, a._radius, upper_y, b._radius)
    if numerator.is_zero():
        return _from_record(record^, numerator, bits)
    var denominator = _Radius.lower(y).multiply_down(_Radius.lower_gap_float(y, b._radius))
    return _from_record(record^, numerator.divide(denominator), bits)


def _ball_scale2(a: Ball, k: Int, bits: Int) raises -> Ball:
    """`a * 2**k` at `bits` bits."""
    if a._kind == _FINITE and a._midpoint._kind == 1 and k > -(Int(1) << 60) and k < (Int(1) << 60):
        ref x = a._midpoint
        var record = _round_exact_binary(x._significand, x._negative, _scale_of(x) + Int128(k), _nearest(bits), False)
        if record.kind >= 0:
            return _from_record(record^, a._radius.scale2(k), bits)
    return _scale2(_BallArgument(a), Integer(k), _at(bits))


def _ball_product_word(a: Ball, n: Int, bits: Int) raises -> Ball:
    """`a * n` at `bits` bits for a word n: the midpoint by the Float
    product's core with n inline, the radius `|n| r`."""
    if a._kind != _FINITE or a._midpoint._kind != 1 or n == 0 or n == Int.MIN:
        return _product(_BallArgument(a), _BallArgument(Integer(n)), _at(bits))
    ref x = a._midpoint
    var magnitude = UInt64(abs(n))
    var record = _binary_product(
        x._significand, Integer(magnitude), x._negative != (n < 0), _scale_of(x), _nearest(bits), False
    )
    return _from_record(record^, a._radius.multiply(_up_word(magnitude, False, 0)), bits)


def _ball_quotient_word(a: Ball, n: Int, bits: Int) raises -> Ball:
    """`a / n` at `bits` bits for a nonzero word n: E.6 with an exact
    divisor, whose radius is `r / |n|`."""
    if a._kind != _FINITE or a._midpoint._kind != 1 or n == 0 or n == Int.MIN:
        return _quotient(_BallArgument(a), _BallArgument(Integer(n)), _at(bits))
    ref x = a._midpoint
    var magnitude = UInt64(abs(n))
    var record = _binary_quotient(
        x._significand, Integer(magnitude), x._negative != (n < 0), _scale_of(x), _nearest(bits), False
    )
    return _from_record(record^, a._radius.divide(_down_word(magnitude, 0)), bits)


# ------------------------------------------------------------- in place
#
# A kernel's loop overwrites its running values: `_ball_mul_assign` and
# `_ball_add_assign` write the result's significand into the destination's
# own block where the destination is its only owner, so a step past 64 bits
# allocates nothing, as an output argument does in Arb. Results are those of
# `_ball_product` and `_ball_sum`.


def _format_at(bits: Int) -> FloatFormat:
    """`FloatFormat(bits)` for a precision in range, without its checks."""
    return FloatFormat(_validated=(bits, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX))


def _store_magnitude(mut n: Integer, magnitude: UInt128) raises:
    """`n = magnitude`: in n's block when n holds its only reference and the
    block has four words, else a new value."""
    if magnitude >> 64 == 0:
        n = Integer(UInt64(magnitude))
        return
    if n._storage.isa[Integer._Shared]():
        ref shared = n._storage[Integer._Shared]
        if (
            shared._raw[].strong.load[ordering=Ordering.ACQUIRE]() == 1
            and shared._raw[].weak.load[ordering=Ordering.ACQUIRE]() == 1
            and shared[].words.capacity() >= 4
            and not shared[].negative
        ):
            shared[].words._length = _put_magnitude(shared[].words.span(), magnitude)
            return
    var owner = Integer._Shared.uninitialized(4, False)
    var used = _put_magnitude(owner[].words.span(), magnitude)
    n = Integer._from_product(owner^, used)


def _set_midpoint(mut z: Ball, magnitude: UInt128, exponent: Int, negative: Bool, inexact: Bool, radius: _Radius, bits: Int) raises:
    """z's midpoint and radius from a rounded result, as `_from_record` makes
    them."""
    if radius.infinite:
        z = Ball.unbounded(bits)
        return
    _store_magnitude(z._midpoint._significand, magnitude)
    z._midpoint._exponent = exponent
    z._midpoint._negative = negative
    z._midpoint._kind = 1
    z._midpoint._format = _format_at(bits)
    z._radius = radius.add(_Radius.power_of_two(exponent - bits - 1)) if inexact else radius
    z._kind = _FINITE


def _ball_mul_assign(mut z: Ball, y: Ball, bits: Int) raises:
    """`z = z * y` at `bits` bits."""
    if not _plain(z, y) or bits < 2 or bits > 128:
        z = _ball_product(z, y, bits)
        return
    var a = _short_magnitude(z._midpoint._significand)
    var b = _short_magnitude(y._midpoint._significand)
    if not a or not b:
        z = _ball_product(z, y, bits)
        return
    var negative = z._midpoint._negative != y._midpoint._negative
    var rounded = _wide_rounding(
        UInt256(a.value()) * UInt256(b.value()), bits, _exponent_add(_scale_of(z._midpoint), _scale_of(y._midpoint)),
        RoundingMode.nearest_even, negative,
    )
    if rounded.exponent > Int128(FloatFormat.DEFAULT_EMAX) or rounded.exponent < Int128(FloatFormat.DEFAULT_EMIN):
        z = _ball_product(z, y, bits)
        return
    var upper_x = _Radius.upper(z._midpoint) if not y._radius.is_zero() else _Radius.zero()
    var upper_y = _Radius.upper(y._midpoint) if not z._radius.is_zero() else _Radius.zero()
    var radius = _product_radius(upper_x, z._radius, upper_y, y._radius)
    _set_midpoint(z, rounded.magnitude, Int(rounded.exponent), negative, rounded.inexact, radius, bits)


def _ball_add_assign(mut z: Ball, y: Ball, bits: Int) raises:
    """`z = z + y` at `bits` bits. At most 64 bits the sum allocates nothing
    anyway and takes `_ball_sum`."""
    if not _plain(z, y) or bits <= 64 or bits > 128:
        z = _ball_sum(z, y, False, bits)
        return
    var a = _short_magnitude(z._midpoint._significand)
    var b = _short_magnitude(y._midpoint._significand)
    if not a or not b:
        z = _ball_sum(z, y, False, bits)
        return
    var aligned = _aligned_sum(
        a.value(), z._midpoint._negative, _scale_of(z._midpoint), b.value(), y._midpoint._negative, _scale_of(y._midpoint), bits
    )
    if not aligned[3] or not aligned[0]:
        z = _ball_sum(z, y, False, bits)
        return
    var rounded = _wide_rounding(aligned[0], bits, aligned[2], RoundingMode.nearest_even, aligned[1])
    if rounded.exponent > Int128(FloatFormat.DEFAULT_EMAX) or rounded.exponent < Int128(FloatFormat.DEFAULT_EMIN):
        z = _ball_sum(z, y, False, bits)
        return
    _set_midpoint(z, rounded.magnitude, Int(rounded.exponent), aligned[1], rounded.inexact, z._radius.add(y._radius), bits)
