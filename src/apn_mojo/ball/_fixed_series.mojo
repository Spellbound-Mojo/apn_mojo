"""Hypergeometric series in fixed point: the series engine's terms without a
ball operation each.

`sum_k T_k`, `T_{k+1} = T_k z prod (u + k) / ((k + 1) prod (l + k))`, at
dyadic midpoints `u = N_u / 2**d_u`, `l = N_l / 2**d_l`, `z = N_z / 2**d_z`,
has the integer ratio

    T_{k+1} / T_k = 2**s N_z prod f_u(k) / ((k + 1) prod f_l(k)),
    f(k) = N + k 2**d,   s = sum d_l - sum d_u - d_z.

The terms are held as integers `V_k ~ T_k 2**-E`, magnitude and sign apart,
in one stack block of limbs, and so are the factors, each stepped by `2**d`
in place. A step takes the factors one at a time, keeping V at the term's
width: it multiplies by `|N_z|` or `|f_u(k)|` and shifts right by that
factor's d, shifts left by `d_l` and divides by `|f_l(k)|`, and divides by
`k + 1`, a word at a time for a one-limb factor. Each operation with a
truncation leaves the exact result less a part below 1, and the operations
after it scale that part by their own factors, so an operation of true
factor g (`|u + k|`, `|z|`, `1/|l + k|`, `1/(k+1)`) updates the error bound
as `e -> g e + 1`, or `g e` when it is exact. From `e_0 = 0` this bounds
`|V_k - T_k 2**-E|` at every k, and the sum's error by `sum e_k`; the
accumulators add exactly. The bounds are computed in doubles, from the
limbs with `2**-d` folded in so they stay in range, each result raised by
`1 + 2**-47`, more than its roundings can take.

A ball parameter or argument with radius r changes each ratio by a factor
within `1 +- delta_k`, `1 + delta_k = (1 + r_z/|z|) prod (1 + r_u/|u+k|)
prod 1 / (1 - r_l/|l+k|)`, so every term at every point of the balls is
within `R_k |T_k|` of the midpoints', `R_{k+1} = R_k + delta_k + R_k delta_k`
(kept in that form, since `1 + delta_k` rounds to 1 in a double);
those add `sum R_k (|V_k| 2**E + e_k 2**E)`, its first part kept in value
units so that a double holds it at any precision.

The scale is set from the planning pass: `E = min(0, log2 |S|) - w - guard`,
fine enough for the sum and for `T_0 = 1`, from which every term follows,
and the block holds the largest term. Past the block, or for parameters
whose binary exponents leave its range, the caller sums the balls.
"""

from std.collections import Array
from std.memory import bitcast
from std.bit import count_trailing_zeros
from ..integer._limbs import (
    _add_limbs, _add_one, _subtract_one, _subtract_limbs, _multiply_by_limb, _divide_limbs_by_limb,
    _multiply_limbs, _shift_left_limbs, _shift_right_limbs, _used, _compare,
)
from ..float.value import Float
from std.bit import count_leading_zeros
from .value import Ball, _FINITE
from ._radius import _Radius
from ._medium import _Limbs, _fixed_ball, _fixed_from_float, _copy, _divide_fixed

comptime _BLOCK = 4096
"""Limbs of the stack block, as the medium kernels' scratch."""

comptime _MARGIN = 1.0 + 1.0 / Float64(1 << 47)
comptime _LOWER = 1.0 - 1.0 / Float64(1 << 47)


def _two(e: Int) -> Float64:
    """`2**e` as a double: 0 below the normal range, inf above."""
    if e < -1022:
        return 0.0
    if e > 1023:
        return Float64.MAX * 2.0
    return bitcast[DType.float64](UInt64(e + 1023) << 52)


def _limbs_upper(v: _Limbs, n: Int, scale: Int) -> Float64:
    """An upper bound of `v[0, n) 2**scale` as a double; below the doubles'
    normal range, the scale is raised to it, which still bounds the value."""
    var used = _used(v, n)
    if used == 0:
        return 0.0
    var top = Float64(v.unsafe_offset(used - 1)[])
    if used == 1:
        return (top + 1.0) * _MARGIN * _two(max(scale, -1022))
    var next = Float64(v.unsafe_offset(used - 2)[])
    return (top * 18446744073709551616.0 + next + 1.0) * _MARGIN * _two(max(64 * (used - 2) + scale, -1022))


def _limbs_lower(v: _Limbs, n: Int, scale: Int) -> Float64:
    """A lower bound of `v[0, n) 2**scale` as a double."""
    var used = _used(v, n)
    if used == 0:
        return 0.0
    var top = Float64(v.unsafe_offset(used - 1)[])
    if used == 1:
        return top * _LOWER * _two(scale)
    var next = Float64(v.unsafe_offset(used - 2)[])
    return (top * 18446744073709551616.0 + next) * _LOWER * _two(64 * (used - 2) + scale)


@fieldwise_init
struct _Factor(ImplicitlyCopyable):
    """A factor `N + k 2**d` held in the block: its magnitude's limbs at `at`,
    `size` of them, its sign, and its step, a `step_bit` in limb `step_limb`;
    `radius` bounds the parameter's radius r."""

    var at: Int
    var size: Int
    var negative: Bool
    var shift: Int
    var step_limb: Int
    var step_bit: UInt64
    var radius: Float64


def _dyadic_shift(x: Float) raises -> Optional[Int]:
    """d >= 0 with `x 2**d` an integer, the least; None for a zero."""
    if x.is_zero():
        return 0
    var odd = x._significand
    var zeros = 0
    while odd._low_magnitude() == 0:
        odd = odd >> 64
        zeros += 64
    var q = x._exponent - x.precision() + zeros + Int(count_trailing_zeros(odd._low_magnitude()))
    return -q if q < 0 else 0


def _radius_f64(x: Ball, shift: Int) raises -> Float64:
    """An upper bound of the radius times `2**shift`, as a double."""
    if x._radius.is_zero():
        return 0.0
    var r = x._radius.to_float().to_native[DType.float64]()
    return r * _MARGIN * _two(shift)


def _place(x: Ball, terms: Int, at: Int, block: _Limbs, w: Int) raises -> Optional[_Factor]:
    """The midpoint's factor `N + k 2**d`, with room for every k <= terms, at
    `at` in the block (the caller checks the room); None for a binary
    exponent beyond `w + 128` bits."""
    var t = x._midpoint
    var d = _dyadic_shift(t).value()
    if d > w + 128 or (not t.is_zero() and t._exponent > w + 128):
        return None
    var bits = max(0 if t.is_zero() else t._exponent + d, d + 1 + _bit_count(terms + 1)) + 1
    var size = (bits + 63) // 64 + 1
    var target = block.unsafe_offset(at)
    for i in range(size):
        target.unsafe_offset(i)[] = 0
    if not t.is_zero():
        _ = _fixed_from_float(t, target, size, d)
    return _Factor(at, size, t._negative, d, d >> 6, UInt64(1) << UInt64(d & 63), _radius_f64(x, 0))


def _bit_count(n: Int) -> Int:
    """The bit length of n, or 0 when n <= 0."""
    return 64 - Int(count_leading_zeros(UInt64(n))) if n > 0 else 0


def _room(x: Ball, terms: Int, w: Int) raises -> Int:
    """The limbs `_place` takes for x, or -1 when it would refuse."""
    var t = x._midpoint
    var d = _dyadic_shift(t).value()
    if d > w + 128 or (not t.is_zero() and t._exponent > w + 128):
        return -1
    var bits = max(0 if t.is_zero() else t._exponent + d, d + 1 + _bit_count(terms + 1)) + 1
    return (bits + 63) // 64 + 1


def _step(mut f: _Factor, block: _Limbs):
    """`f(k) -> f(k + 1) = f(k) + 2**d`, the magnitude and sign in place."""
    var limbs = block.unsafe_offset(f.at)
    var reach = f.size - f.step_limb
    if not f.negative:
        _ = _add_one(limbs.unsafe_offset(f.step_limb), reach, f.step_bit)
        return
    # -|f| + 2**d: |f| - 2**d while it stays positive, else its negation.
    if _subtract_one(limbs.unsafe_offset(f.step_limb), reach, f.step_bit):
        for i in range(f.size):
            limbs.unsafe_offset(i)[] = ~limbs.unsafe_offset(i)[]
        _ = _add_one(limbs, f.size, 1)
        f.negative = False


def _fixed_terms(
    uppers: List[Ball], lowers: List[Ball], z: Ball, n: Int, w: Int, log2_size: Float64, log2_peak: Float64, guard: Int,
) raises -> Optional[Tuple[Ball, Ball]]:
    """`sum_{k<n} T_k` and a bound of `|T_n|` as balls, in fixed point (module
    docstring); None past the block or the parameters' range."""
    if w > 4500:
        return None
    # The scale and the term's limbs.
    var top = log2_size if log2_size < 0.0 else 0.0
    var exponent = Int(top) - 1 - w - guard
    var peak = log2_peak if log2_peak > 0.0 else 0.0
    var limbs = (Int(peak) - exponent + 66) // 64 + 1
    # The room each factor takes, then the layout.
    var factor_room = 0
    var z_room = _room(z, 0, w)
    if z_room < 0:
        return None
    for u in uppers:
        var r = _room(u, n, w)
        if r < 0:
            return None
        factor_room += r
    for l in lowers:
        var r = _room(l, n, w)
        if r < 0:
            return None
        factor_room += r
    var widest = z_room
    var deepest = 0
    for u in uppers:
        widest = max(widest, _room(u, n, w))
    for l in lowers:
        widest = max(widest, _room(l, n, w))
        deepest = max(deepest, _dyadic_shift(l._midpoint).value())
    var term_room = limbs + widest + (deepest >> 6) + 4
    var largest_divisor = 1
    for l in lowers:
        largest_divisor = max(largest_divisor, _room(l, n, w))
    var work_room = 2 * term_room + largest_divisor + 6
    var accumulator = limbs + 2
    var total = 2 * term_room + work_room + factor_room + z_room + 2 * accumulator
    if total > _BLOCK:
        return None
    var radii = not z._radius.is_zero()
    for u in uppers:
        radii = radii or not u._radius.is_zero()
    for l in lowers:
        radii = radii or not l._radius.is_zero()
    var store = Array[UInt64, _BLOCK](uninitialized=True)
    var block = Span(store).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var v = block
    var spare = v.unsafe_offset(term_room)
    var work = spare.unsafe_offset(term_room)
    var positive = work.unsafe_offset(work_room)
    var negative_sum = positive.unsafe_offset(accumulator)
    var at = 2 * term_room + work_room + 2 * accumulator
    var zf = _place(z, 0, at, block, w).value()
    at += zf.size
    var ups = List[_Factor]()
    for u in uppers:
        var f = _place(u, n, at, block, w).value()
        at += f.size
        ups.append(f)
    var lows = List[_Factor]()
    for l in lowers:
        var f = _place(l, n, at, block, w).value()
        at += f.size
        lows.append(f)
    var z_limbs = block.unsafe_offset(zf.at)
    var z_used = _used(z_limbs, zf.size)
    var z_magnitude = _limbs_upper(z_limbs, zf.size, -zf.shift)
    var z_relative = 0.0
    if radii and not z._radius.is_zero():
        var z_low = _limbs_lower(z_limbs, zf.size, -zf.shift)
        if not (z_low > 0.0):
            return None
        z_relative = zf.radius / z_low * _MARGIN
    for i in range(2 * accumulator):
        positive.unsafe_offset(i)[] = 0
    for i in range(term_room):
        v.unsafe_offset(i)[] = 0
    # T_0 = 1 = 2**-E units.
    var one_at = -exponent
    v.unsafe_offset(one_at >> 6)[] = UInt64(1) << UInt64(one_at & 63)
    var vn = (one_at >> 6) + 1
    var sign_negative = False
    var error = 0.0
    var error_sum = 0.0
    var drift = 0.0
    # The drift's part: units of 2**E from the errors, and value units from
    # the terms themselves, `|V_k| 2**E`, which stay below the largest term.
    var drift_units = 0.0
    var drift_value = 0.0
    var ended = z_used == 0
    for k in range(n):
        # Add T_k.
        if vn:
            var target = negative_sum if sign_negative else positive
            var carry = _add_limbs(target, v, vn)
            if _add_one(target.unsafe_offset(vn), accumulator - vn, carry):
                return None
        error_sum += error
        if radii:
            drift_units += drift * error
            drift_value += drift * _limbs_upper(v, vn, exponent)
        if ended:
            for i in range(vn):
                v.unsafe_offset(i)[] = 0
            vn = 0
            error = 0.0
            continue
        for i in range(len(ups)):
            if _used(block.unsafe_offset(ups[i].at), ups[i].size) == 0:
                if ups[i].radius > 0.0:
                    # u + k is 0 at the midpoint but not across the ball: the
                    # midpoints' series ends there and the ball's does not,
                    # and a radius relative to 0 bounds nothing. Sum balls.
                    return None
                ended = True
        if ended:
            for i in range(vn):
                v.unsafe_offset(i)[] = 0
            vn = 0
            error = 0.0
            continue
        # delta_k itself, not 1 + delta_k, which a double would round to 1.
        var delta = z_relative
        var flip = z._midpoint._negative
        # z: multiply by |N_z|, then the floor of 2**-d_z.
        if vn:
            if z_used == 1:
                v.unsafe_offset(vn)[] = _multiply_by_limb(v, vn, z_limbs[])
                vn += 1
            else:
                vn = _multiply_limbs(v, vn, z_limbs, z_used, spare)
                var swap = v
                v = spare
                spare = swap
            if zf.shift:
                _shift_right_limbs(v, vn, zf.shift, v, vn)
            vn = _used(v, vn)
        error = (error * z_magnitude + (1.0 if zf.shift else 0.0)) * _MARGIN
        # The upper factors.
        for i in range(len(ups)):
            var f = ups[i]
            var fl = block.unsafe_offset(f.at)
            var fu = _used(fl, f.size)
            if f.negative:
                flip = not flip
            var m_up = _limbs_upper(fl, f.size, -f.shift)
            if f.radius > 0.0:
                var m_low = _limbs_lower(fl, f.size, -f.shift)
                if not (m_low > 0.0):
                    return None
                var a = f.radius / m_low
                delta = (delta + a + delta * a) * _MARGIN
            if vn:
                if fu == 1:
                    v.unsafe_offset(vn)[] = _multiply_by_limb(v, vn, fl[])
                    vn += 1
                else:
                    vn = _multiply_limbs(v, vn, fl, fu, spare)
                    var swap = v
                    v = spare
                    spare = swap
                if f.shift:
                    _shift_right_limbs(v, vn, f.shift, v, vn)
                vn = _used(v, vn)
            error = (error * m_up + (1.0 if f.shift else 0.0)) * _MARGIN
        # The lower factors: the floor of `V 2**d_l / |f_l|`.
        for i in range(len(lows)):
            var f = lows[i]
            var fl = block.unsafe_offset(f.at)
            var fu = _used(fl, f.size)
            var m_low = _limbs_lower(fl, f.size, -f.shift)
            if fu == 0 or not (m_low > 0.0):
                return None
            if f.negative:
                flip = not flip
            if f.radius > 0.0:
                var relative = f.radius / m_low
                if not (relative < 1.0):
                    return None
                var a = relative / (1.0 - relative)
                delta = (delta + a + delta * a) * _MARGIN
            if vn:
                if f.shift:
                    vn = _shift_left_limbs(v, vn, f.shift)
                if fu == 1:
                    _ = _divide_limbs_by_limb(v, vn, fl[])
                elif vn < fu:
                    vn = 0
                else:
                    _divide_fixed(v, vn, fl, fu, v, vn - fu + 1, work)
                    vn = vn - fu + 1
                vn = _used(v, vn)
            error = (error / m_low + 1.0) * _MARGIN
        # k! : the floor of V / (k + 1).
        if vn and k > 0:
            _ = _divide_limbs_by_limb(v, vn, UInt64(k + 1))
            vn = _used(v, vn)
        if k > 0:
            error = (error / Float64(k + 1) + 1.0) * _MARGIN
        if vn > limbs:
            return None
        if flip:
            sign_negative = not sign_negative
        if radii:
            drift = (drift + delta + drift * delta) * _MARGIN
        if not (error < 1.0e300) or not (drift < 1.0e300) or not (drift_value < 1.0e300):
            return None
        for i in range(len(ups)):
            _step(ups[i], block)
        for i in range(len(lows)):
            _step(lows[i], block)
    # The sum, positive minus negative.
    var result_negative = False
    var difference = positive
    if _compare(positive, negative_sum, accumulator) >= 0:
        _ = _subtract_limbs(positive, negative_sum, accumulator)
    else:
        _ = _subtract_limbs(negative_sum, positive, accumulator)
        difference = negative_sum
        result_negative = True
    var units = (error_sum + drift_units + 1.0) * _MARGIN
    var extra_radius = _Radius.upper(Float(units)).multiply(_Radius.power_of_two(exponent))
    if drift_value > 0.0:
        extra_radius = extra_radius.add(_Radius.upper(Float(drift_value * _MARGIN)))
    var sum = _fixed_ball(difference, accumulator, 0, result_negative, exponent, 0, extra_radius, w)
    # |T_n| <= (|V_n| + e_n) (1 + R_n), the first part in value units.
    var growth = (1.0 + drift) * _MARGIN
    var last = _Radius.upper(Float(error * growth)).multiply(_Radius.power_of_two(exponent))
    var top_value = _limbs_upper(v, vn, exponent) * growth
    if not (top_value < 1.0e300):
        return None
    last = last.add(_Radius.upper(Float(top_value)))
    var tail = Ball(_midpoint=Float(0), _radius=last, _kind=_FINITE)
    _ = store^
    return (sum^, tail^)
