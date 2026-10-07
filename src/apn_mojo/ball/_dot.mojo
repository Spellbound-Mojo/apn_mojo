"""The dot product of balls, `initial + sum_i x_i y_i`, rounded once
(as Arb's `arb_dot`; Johansson, "Faster arbitrary-precision dot product and
matrix multiplication", ARITH 26, 2019)."""

from std.collections import Array
from ..integer._limbs import _add_one, _subtract_limbs, _compare
from ..float._rounding import _short_magnitude
from .value import Ball, _FINITE
from ._radius import _Radius
from ._arithmetic import _plain, _scale_of, _product_radius, _ball_sum, _ball_product
from ._medium import _fixed_ball


comptime _DOT_LIMBS = 8
"""Limbs of a dot product's accumulators: 64 guard bits below a result of at
most 128 bits, its terms of up to 256 bits, and room for their sum."""


def _ball_dot(initial: Ball, xs: List[Ball], ys: List[Ball], bits: Int) raises -> Ball:
    """`initial + sum_i xs[i] ys[i]` at `bits` bits, rounded once. The products
    of the midpoints are exact in 256 bits; they add exactly at the unit
    `2**U`, `U` 64 bits below the precision under the largest term, a term's
    bits below the unit costing one unit of radius; the radius adds each
    product's E.3 bound. Chained products and sums round 2n times and cost a
    ball operation each. Any operand without a short midpoint takes them."""
    var count = len(xs)
    var usable = bits >= 2 and bits <= 128 and initial._kind == _FINITE
    var top = Int.MIN
    if usable and initial._midpoint._kind == 1:
        var s = _short_magnitude(initial._midpoint._significand)
        usable = Bool(s)
        top = initial._midpoint._exponent
    for i in range(count):
        if not usable:
            break
        if not _plain(xs[i], ys[i]) or not _short_magnitude(xs[i]._midpoint._significand) or not _short_magnitude(ys[i]._midpoint._significand):
            usable = False
            break
        top = max(top, xs[i]._midpoint._exponent + ys[i]._midpoint._exponent)
    if not usable or top == Int.MIN:
        var total = initial
        for i in range(count):
            total = _ball_sum(total, _ball_product(xs[i], ys[i], bits), False, bits)
        return total^
    var unit = top - bits - 64
    var store = Array[UInt64, 2 * _DOT_LIMBS](fill=0)
    var block = Span(store).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var positive = block
    var negative = block.unsafe_offset(_DOT_LIMBS)
    var lost = 0
    var radius = initial._radius
    if initial._midpoint._kind == 1:
        ref m = initial._midpoint
        lost += _dot_place(
            UInt256(_short_magnitude(m._significand).value()), Int(_scale_of(m)) - unit,
            negative if m._negative else positive,
        )
    for i in range(count):
        ref x = xs[i]._midpoint
        ref y = ys[i]._midpoint
        var product = UInt256(_short_magnitude(x._significand).value()) * UInt256(_short_magnitude(y._significand).value())
        lost += _dot_place(product, Int(_scale_of(x)) + Int(_scale_of(y)) - unit, negative if x._negative != y._negative else positive)
        var upper_x = _Radius.upper(x) if not ys[i]._radius.is_zero() else _Radius.zero()
        var upper_y = _Radius.upper(y) if not xs[i]._radius.is_zero() else _Radius.zero()
        radius = radius.add(_product_radius(upper_x, xs[i]._radius, upper_y, ys[i]._radius))
    var result_negative = False
    var difference = positive
    if _compare(positive, negative, _DOT_LIMBS) >= 0:
        _ = _subtract_limbs(positive, negative, _DOT_LIMBS)
    else:
        _ = _subtract_limbs(negative, positive, _DOT_LIMBS)
        difference = negative
        result_negative = True
    var result = _fixed_ball(difference, _DOT_LIMBS, 0, result_negative, unit, UInt64(lost), radius, bits)
    _ = store^
    return result^


def _dot_place(value: UInt256, shift: Int, target: Pointer[UInt64, MutUntrackedOrigin]) -> Int:
    """Add `floor(value 2**shift)` into the accumulator; 1 when bits fell
    below its unit, else 0. Every term lies below `2**(bits + 64)` units and
    the accumulators have room for their sum."""
    if shift <= -256:
        return 1
    var cut = False
    var v = value
    var at = shift
    if at < 0:
        cut = (v & ((UInt256(1) << UInt256(-at)) - 1)) != 0
        v >>= UInt256(-at)
        at = 0
    var q = at >> 6
    var r = UInt256(at & 63)
    var limbs = Array[UInt64, 5](fill=0)
    var shifted_low = v << r
    limbs[0] = UInt64(shifted_low)
    limbs[1] = UInt64(shifted_low >> 64)
    limbs[2] = UInt64(shifted_low >> 128)
    limbs[3] = UInt64(shifted_low >> 192)
    limbs[4] = UInt64(v >> (UInt256(256) - r)) if r else 0
    var carry = UInt64(0)
    for i in range(5):
        if q + i >= _DOT_LIMBS:
            break
        var sum = UInt128(target.unsafe_offset(q + i)[]) + UInt128(limbs[i]) + UInt128(carry)
        target.unsafe_offset(q + i)[] = UInt64(sum)
        carry = UInt64(sum >> 64)
    if carry and q + 5 < _DOT_LIMBS:
        _ = _add_one(target.unsafe_offset(q + 5), _DOT_LIMBS - q - 5, carry)
    return Int(cut)
