"""Medium-precision kernels: exp, expm1, log, atan, sin and cos of a finite
Float up to 4608 bits, by the method of F. Johansson, "Efficient
implementation of elementary functions in the medium-precision range",
ARITH 22, 2015.

Values are fixed-point numbers of whole 64-bit limbs in one stack block, at
the target precision plus a few bits. An argument is reduced modulo ln 2 or
pi/4 against a stored constant, then by one table lookup on its top 8 bits
(up to 512 bits) or two on 5 bits each (up to 4608 bits); the rest takes a
Taylor series by rectangular splitting (`_series_sum`) whose coefficients
are integers over denominators shared by groups of terms, so a term costs
one word multiply-add. The error is one count of units in the last limb,
raised at each step by the bound derived there, and becomes the ball's
radius.

Each kernel returns None where it does not apply (beyond the tables, tiny
or huge arguments, or a sine or cosine that cancels), and the general
kernels take over."""

from std.bit import count_leading_zeros, byte_swap
from std.memory import bitcast
from std.collections import Array
from ..integer.value import Integer
from ..integer.number_theory import isqrt
from ..integer._division import _divide_limbs, _load_limbs
from ..integer._limbs import (
    _add_limbs, _subtract_limbs, _subtract_one, _add_mul_limbs, _subtract_mul_limbs, _multiply_by_limb,
    _multiply_limbs, _divide_normalized, _divide_limbs_by_limb, _shift_left_limbs, _shift_right_limbs,
)
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat
from ..float._rounding import _round_ratio, _RoundedBinary, _rounded_direction
from ..float.status import NumericStatus
from ._radius import _Radius, _up
from .value import Ball, _FINITE
from ._arithmetic import _from_record
from ._tables import (
    _SHORT_LIMBS, _LONG_LIMBS, _EXP_SHORT, _EXP_LONG_COARSE, _EXP_LONG_FINE, _LOG_SHORT_COARSE, _LOG_SHORT_FINE, _LOG_LONG_COARSE,
    _LOG_LONG_FINE, _ATAN_SHORT, _ATAN_LONG_COARSE, _ATAN_LONG_FINE, _SIN_COS_SHORT, _SIN_COS_LONG_COARSE, _SIN_COS_LONG_FINE,
    _LN2_VALUE, _QUARTER_PI_VALUE, _HALF_PI_MINUS_ONE_VALUE, _FACTORIAL_NUMER, _FACTORIAL_DENOM, _FACTORIAL_SIZE,
    _ODD_NUMER, _ODD_DENOM, _ODD_SIZE, _RECIPROCAL_FACTORIAL_BITS,
)

comptime _Limbs = Pointer[UInt64, MutUntrackedOrigin]
comptime _SHORT_PREC = 64 * _SHORT_LIMBS
comptime _LONG_PREC = 64 * _LONG_LIMBS
comptime _SCRATCH = 4096
"""Limbs of one call's stack block: the largest kernel (atan at 4608 bits)
takes about 2,000."""


# ---------------------------------------------------------------- the tables


@always_inline
def _hex_limb(table: StaticString, at: Int) -> UInt64:
    """The word written by the 16 hex digits at `at`: nibbles by
    `(c & 15) + 9 (c >> 6)`, paired into bytes, most significant first."""
    var c = table.unsafe_ptr().unsafe_offset(at).unsafe_load[width=16, alignment=1]()
    var nibbles = (c & 15) + (c >> 6) * 9
    var halves = nibbles.deinterleave()
    var bytes = (halves[0] << 4) | halves[1]
    return byte_swap(bitcast[DType.uint64, 1](bytes))


def _read_limbs(table: StaticString, entry: Int, limbs: Int, count: Int, target: _Limbs):
    """The top `count` limbs of entry `entry`, of `limbs` limbs, into
    target[0, count): `floor(f 2**(64 count))` of the entry's value f."""
    var start = 16 * limbs * entry
    for j in range(count):
        target.unsafe_offset(count - 1 - j)[] = _hex_limb(table, start + 16 * j)


@always_inline
def _coefficient(table: StaticString, k: Int) -> UInt64:
    return _hex_limb(table, 16 * k)


def _reciprocal_factorial_bits(n: Int) -> Int:
    """An upper bound of log2(1/n!), for n < 600."""
    var p = _RECIPROCAL_FACTORIAL_BITS.unsafe_ptr()
    var v = 0
    for i in range(4):
        var c = Int(p.unsafe_offset(4 * n + i)[])
        v = (v << 4) | (c - 48 if c < 58 else c - 87)
    return v - 65536 if v >= 32768 else v


def _exp_taylor_bound(mag: Int, prec: Int) -> Int:
    """The terms of the exponential series for `|x| <= 2**mag`, `mag <= -2`,
    that leave a truncation error below `2**(-prec - 1)`."""
    var i = 1
    while mag * i + _reciprocal_factorial_bits(i) >= -prec - 1:
        i += 1
    return i


# ---------------------------------------------------------------- limbs


@always_inline
def _copy(target: _Limbs, source: _Limbs, n: Int):
    for i in range(n):
        target.unsafe_offset(i)[] = source.unsafe_offset(i)[]


@always_inline
def _zero(target: _Limbs, n: Int):
    for i in range(n):
        target.unsafe_offset(i)[] = 0


def _leading_zeros(w: _Limbs, n: Int) -> Int:
    var zeros = 0
    for i in range(n - 1, -1, -1):
        var limb = w.unsafe_offset(i)[]
        if limb:
            return zeros + Int(count_leading_zeros(limb))
        zeros += 64
    return zeros


def _bits_at(words: Span[UInt32, _], position: Int) -> UInt64:
    """Bits [position, position + 64) of a magnitude's words, zero outside."""
    var q = position >> 5
    var o = UInt128(position & 31)
    var window = UInt128(0)
    for j in range(3):
        var i = q + j
        if i >= 0 and i < len(words):
            window |= UInt128(words[i]) << UInt128(32 * j)
    return UInt64(window >> o)


def _fixed_from_float(x: Float, target: _Limbs, n: Int, shift: Int) -> Bool:
    """target[0, n) = floor(|x| 2**shift), which must fit n limbs; whether
    bits were dropped."""
    var small = x._significand._inline_words()
    var words = x._significand._words_span(small)
    # Bit 0 of the significand lands at bit k of the result.
    var k = x._exponent - x.precision() + shift
    for i in range(n):
        target.unsafe_offset(i)[] = _bits_at(words, 64 * i - k)
    if k >= 0:
        return False
    for i in range(min(len(words), ((-k) + 31) // 32)):
        var below = (-k) - 32 * i
        var word = words[i]
        if below < 32:
            word &= (UInt32(1) << UInt32(below)) - 1
        if word:
            return True
    return False


def _limbs_integer(t: _Limbs, n: Int, negative: Bool) raises -> Integer:
    return Integer._from_span(
        Span(unsafe_ptr=t.unsafe_bitcast[UInt32]().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=2 * n), negative
    )


def _fixed_ball(t: _Limbs, tn: Int, wn: Int, negative: Bool, exponent: Int, error: UInt64, extra: _Radius, prec: Int) raises -> Ball:
    """The ball of `t[0, tn) 2**(exponent - 64 wn)`, signed, with `error`
    units of `2**(exponent - 64 wn)` and `extra` more. The midpoint is rounded
    once to nearest at `prec` bits straight from the limbs: its top `prec`
    bits, then the half bit and the sticky bits below; the rounding is added
    to the radius."""
    var radius = _up(UInt128(error), exponent - 64 * wn).add(extra)
    var n = tn
    while n and t.unsafe_offset(n - 1)[] == 0:
        n -= 1
    if not n:
        var zero = _RoundedBinary(0, negative, Integer(0), 0, FloatFormat(prec), NumericStatus())
        return _from_record(zero^, radius, prec)
    var bits = 64 * n - Int(count_leading_zeros(t.unsafe_offset(n - 1)[]))
    var limbs = (prec + 63) // 64
    var buffer = Array[UInt64, _LONG_LIMBS + 2](uninitialized=True)
    var top = Span(buffer).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var discard = bits - prec
    var inexact = False
    var up = False
    if discard > 0:
        _shift_right_limbs(t, n, discard, top, limbs)
        # The limb above the top `prec` bits, read when a carry out of them
        # shifts the limbs back down: zero unless that carry sets it.
        top.unsafe_offset(limbs)[] = 0
        var at = discard - 1
        var half = (t.unsafe_offset(at >> 6)[] >> UInt64(at & 63)) & 1
        var sticky = (t.unsafe_offset(at >> 6)[] & ((UInt64(1) << UInt64(at & 63)) - 1)) != 0
        for i in range(at >> 6):
            sticky = sticky or t.unsafe_offset(i)[] != 0
        inexact = half == 1 or sticky
        up = half == 1 and (sticky or (top.unsafe_offset(0)[] & 1) == 1)
        if up:
            var carry = UInt64(1)
            for i in range(limbs):
                var v = top.unsafe_offset(i)[] + carry
                carry = UInt64(Int(v < carry))
                top.unsafe_offset(i)[] = v
            # A carry into bit prec: 2**prec, which is 2**(prec - 1) at the next exponent.
            var overflow: Bool
            if prec & 63 == 0:
                overflow = carry == 1
                top.unsafe_offset(limbs)[] = carry
            else:
                overflow = ((top.unsafe_offset(limbs - 1)[] >> UInt64(prec & 63)) & 1) == 1
            if overflow:
                _shift_right_limbs(top, limbs + 1, 1, top, limbs)
                bits += 1
    else:
        _copy(top, t, n)
        for i in range(n, limbs + 1):
            top.unsafe_offset(i)[] = 0
        _ = _shift_left_limbs(top, n, -discard)
    var significand = _limbs_integer(top, limbs, False)
    _ = buffer^
    var e = bits + exponent - 64 * wn
    if inexact:
        radius = radius.add(_Radius.power_of_two(e - prec - 1))
    if radius.infinite:
        return Ball.unbounded(prec)
    var status = NumericStatus._make(Int(inexact), _rounded_direction(negative, up) if inexact else 0)
    var record = _RoundedBinary(1, negative, significand^, e, FloatFormat(prec), status)
    return Ball(_midpoint=Float(_rounded=record^), _radius=radius, _kind=_FINITE)

def _square_root_limbs(source: _Limbs, sn: Int, target: _Limbs, tn: Int) raises:
    """target[0, tn) = floor(sqrt(source[0, sn)))."""
    var root = isqrt(_limbs_integer(source, sn, False))
    var small = root._inline_words()
    _load_limbs(root._words_span(small), 0, target, tn)


# ---------------------------------------------------------------- series


@always_inline
def _row(rows: _Limbs, i: Int, xn: Int) -> _Limbs:
    """Row i of a table of powers: y**i, at i xn."""
    return rows.unsafe_offset(i * xn)


def _powers(rows: _Limbs, xn: Int, m: Int, scratch: _Limbs) raises:
    """Rows 2 to m (m even) of the powers of the fraction y in row 1, each a
    truncated product: y**k = (y**(k/2))**2 and y**(k-1) = y**(k/2)
    y**(k/2 - 1), with each power at most one unit and a fraction low for
    y < 1. scratch holds 2 xn limbs."""
    _ = _multiply_limbs(_row(rows, 1, xn), xn, _row(rows, 1, xn), xn, scratch, square=True)
    _copy(_row(rows, 2, xn), scratch.unsafe_offset(xn), xn)
    var k = 4
    while k <= m:
        _ = _multiply_limbs(_row(rows, k // 2, xn), xn, _row(rows, k // 2 - 1, xn), xn, scratch)
        _copy(_row(rows, k - 1, xn), scratch.unsafe_offset(xn), xn)
        _ = _multiply_limbs(_row(rows, k // 2, xn), xn, _row(rows, k // 2, xn), xn, scratch, square=True)
        _copy(_row(rows, k, xn), scratch.unsafe_offset(xn), xn)
        k += 2


def _series_sum(
    s: _Limbs, rows: _Limbs, xn: Int, m: Int, terms: Int,
    numer: StaticString, denom: StaticString, first: Int, stride: Int,
    independent: Bool, alternating: Bool, t: _Limbs,
) raises:
    """The sum of c_j y**j for j < terms by rectangular splitting (Paterson and
    Stockmeyer, 1973; Smith, 1989; with integer coefficients over shared
    denominators, Johansson, Efficient implementation of elementary functions
    in the medium-precision range, ARITH 22, 2015), signs alternating from +
    when asked, left in s scaled by the first group's denominator.

    Coefficient j is numer[i] / denom[i] for i = first + stride j, within a
    group of terms that share the denominator. Blocks of m terms take one word
    multiply-add each against the rows of powers, from the highest block
    down, and each block's sum is multiplied by y**m (row m) before the next.
    At a group boundary the sum moves to the lower group's denominator: for
    nested groups (factorials) by one word division, for independent ones
    (odd reciprocals) by multiplying by the new denominator and dividing by
    the old. A negative partial sum, which only alternating signs give, is
    lifted by the old denominator before the division and lowered after.

    s holds xn + 1 limbs, the top one the integer part (one more limb for
    independent groups); t holds 2 xn + 2."""
    for i in range(xn + 1 + Int(independent)):
        s.unsafe_offset(i)[] = 0
    var power = (terms - 1) % m
    var j = terms - 1
    while j >= 0:
        var index = first + stride * j
        var c = _coefficient(numer, index)
        var new_denom = _coefficient(denom, index)
        var old_denom = _coefficient(denom, index + stride)
        var negative = alternating and j % 2 == 1
        if new_denom != old_denom and j < terms - 1:
            var lift = alternating and not negative
            if lift:
                s.unsafe_offset(xn)[] += old_denom
            if independent:
                s.unsafe_offset(xn + 1)[] = _multiply_by_limb(s, xn + 1, new_denom)
                _ = _divide_limbs_by_limb(s, xn + 2, old_denom)
                if s.unsafe_offset(xn + 1)[] != 0:
                    raise Error("apn_mojo internal error: a series sum lost a carry; please report it.")
                if lift:
                    s.unsafe_offset(xn)[] -= new_denom
            else:
                _ = _divide_limbs_by_limb(s, xn + 1, old_denom)
                if lift:
                    s.unsafe_offset(xn)[] -= 1
        if power == 0:
            if negative:
                s.unsafe_offset(xn)[] -= c
            else:
                s.unsafe_offset(xn)[] += c
            if j != 0:
                _ = _multiply_limbs(s, xn + 1, _row(rows, m, xn), xn, t)
                _copy(s, t.unsafe_offset(xn), xn + 1)
            power = m - 1
        else:
            if negative:
                s.unsafe_offset(xn)[] -= _subtract_mul_limbs(s, _row(rows, power, xn), xn, c)
            else:
                s.unsafe_offset(xn)[] += _add_mul_limbs(s, _row(rows, power, xn), xn, c)
            power -= 1
        j -= 1


@always_inline
def _block_size(terms: Int) -> Int:
    """The even m near sqrt(terms): rows of powers against blocks of products."""
    var m = 2
    while m * m < terms:
        m += 2
    return m


def _exp_series(y: _Limbs, x: _Limbs, xn: Int, terms: Int, work: _Limbs) raises -> UInt64:
    """y[0, xn + 1) = the sum of x**k / k! for k < terms, x a fraction of xn
    limbs; returns the error in units of the last limb, at most 2. The
    coefficients' scaled errors (each power's below a unit and a fraction,
    weighted by 1/k!, which sum below e - 2 past the exact first two terms)
    stay below a unit through the divisions, and the final division adds one.
    work holds (m + 1) xn + 3 xn + 4 limbs, m ~ sqrt(terms)."""
    if terms <= 3:
        if terms <= 1:
            _zero(y, xn)
            y.unsafe_offset(xn)[] = UInt64(terms)
            return 0
        if terms == 2:
            _copy(y, x, xn)
            y.unsafe_offset(xn)[] = 1
            return 0
        # 1 + x + x**2 / 2
        var square = work
        _ = _multiply_limbs(x, xn, x, xn, square, square=True)
        _shift_right_limbs(square.unsafe_offset(xn), xn, 1, square.unsafe_offset(xn), xn)
        _copy(y, x, xn)
        y.unsafe_offset(xn)[] = _add_limbs(y, square.unsafe_offset(xn), xn) + 1
        return 2
    var m = _block_size(terms)
    var rows = work
    var s = rows.unsafe_offset((m + 1) * xn)
    var t = s.unsafe_offset(xn + 2)
    _copy(_row(rows, 1, xn), x, xn)
    _powers(rows, xn, m, t)
    _series_sum(s, rows, xn, m, terms, _FACTORIAL_NUMER, _FACTORIAL_DENOM, 0, 1, False, False, t)
    _ = _divide_limbs_by_limb(s, xn + 1, _coefficient(_FACTORIAL_DENOM, 0))
    _copy(y, s, xn + 1)
    return 2


def _sin_cos_series(ysin: _Limbs, ycos: _Limbs, x: _Limbs, xn: Int, terms: Int, sin_only: Bool, alternating: Bool, work: _Limbs) raises -> UInt64:
    """`ysin[0, xn)` = the sum of (-1)**k x**(2k+1) / (2k+1)! and `ycos[0, xn)`
    = the sum of (-1)**k x**(2k) / (2k)! for k < terms (without the signs,
    sinh and cosh), x a fraction of xn limbs; returns the error in units, at
    most 2, as for the exponential's series. Both sums share the powers of
    x**2. A cosine of 1, which the limbs cannot hold, is 1 - ulp."""
    if terms <= 1:
        if terms == 0:
            _zero(ysin, xn)
            if not sin_only:
                _zero(ycos, xn)
            return 0
        _copy(ysin, x, xn)
        if not sin_only:
            for i in range(xn):
                ycos.unsafe_offset(i)[] = UInt64.MAX
        return 1
    var m = _block_size(terms)
    var rows = work
    var s = rows.unsafe_offset((m + 1) * xn)
    var t = s.unsafe_offset(xn + 2)
    _ = _multiply_limbs(x, xn, x, xn, t, square=True)
    _copy(_row(rows, 1, xn), t.unsafe_offset(xn), xn)
    _powers(rows, xn, m, t)
    var denominator = _coefficient(_FACTORIAL_DENOM, 0)
    if not sin_only:
        _series_sum(s, rows, xn, m, terms, _FACTORIAL_NUMER, _FACTORIAL_DENOM, 0, 2, False, alternating, t)
        _copy(t, s, xn + 1)
        _ = _divide_limbs_by_limb(t, xn + 1, denominator)
        if t.unsafe_offset(xn)[] == 0:
            _copy(ycos, t, xn)
        else:
            for i in range(xn):
                ycos.unsafe_offset(i)[] = UInt64.MAX
    _series_sum(s, rows, xn, m, terms, _FACTORIAL_NUMER, _FACTORIAL_DENOM, 1, 2, False, alternating, t)
    _ = _divide_limbs_by_limb(s, xn + 1, denominator)
    _ = _multiply_limbs(s, xn + 1, x, xn, t)
    _copy(ysin, t.unsafe_offset(xn), xn)
    return 2


def _atan_series(y: _Limbs, x: _Limbs, xn: Int, terms: Int, alternating: Bool, work: _Limbs) raises -> UInt64:
    """y[0, xn) = the sum of (-1)**k x**(2k+1) / (2k+1) for k < terms
    (without the signs, atanh's), x a fraction of xn limbs; returns the error
    in units: at most 2, and 3 for the two-term closed form, whose division
    by 3 adds one."""
    if terms <= 2:
        if terms == 0:
            _zero(y, xn)
            return 0
        if terms == 1:
            _copy(y, x, xn)
            return 0
        # x (1 -+ x**2 / 3)
        var cube = work
        _ = _multiply_limbs(x, xn, x, xn, cube.unsafe_offset(xn), square=True)
        _ = _multiply_limbs(cube.unsafe_offset(2 * xn), xn, x, xn, cube)
        _ = _divide_limbs_by_limb(cube.unsafe_offset(xn), xn, 3)
        _copy(y, x, xn)
        if alternating:
            _ = _subtract_limbs(y, cube.unsafe_offset(xn), xn)
        else:
            _ = _add_limbs(y, cube.unsafe_offset(xn), xn)
        return 3
    var m = _block_size(terms)
    var rows = work
    var s = rows.unsafe_offset((m + 1) * xn)
    var t = s.unsafe_offset(xn + 2)
    _ = _multiply_limbs(x, xn, x, xn, t, square=True)
    _copy(_row(rows, 1, xn), t.unsafe_offset(xn), xn)
    _powers(rows, xn, m, t)
    _series_sum(s, rows, xn, m, terms, _ODD_NUMER, _ODD_DENOM, 0, 1, True, alternating, t)
    _ = _divide_limbs_by_limb(s, xn + 1, _coefficient(_ODD_DENOM, 0))
    _ = _multiply_limbs(s, xn + 1, x, xn, t)
    _copy(y, t.unsafe_offset(xn), xn)
    return 2


# ---------------------------------------------------------------- reductions


@fieldwise_init
struct _Reduced(ImplicitlyCopyable):
    var ok: Bool
    var quotient: Int
    var error: UInt64


def _reduce_log2(w: _Limbs, wn: Int, x: Float, work: _Limbs) raises -> _Reduced:
    """`w = |x| - q ln 2` as a fraction of wn limbs in `[0, ln 2)`, with x's
    sign: x = (w + xi error) +- q ln 2 with |xi| <= 1. Within 2 units: one
    from truncating x, and one from ln 2, which is read to tn more limbs
    than w so that a quotient below 2**62 scales its truncation below a unit;
    3 for a negative x, whose ln 2 - r reads ln 2 once more."""
    var exp = x._exponent
    if exp <= -1:
        var error = UInt64(Int(_fixed_from_float(x, w, wn, 64 * wn)))
        if not x._negative:
            return _Reduced(True, 0, error)
        if wn > _LONG_LIMBS:
            return _Reduced(False, 0, 0)
        var d = work
        _read_limbs(_LN2_VALUE, 0, _LONG_LIMBS, wn, d)
        _ = _subtract_limbs(d, w, wn)
        _copy(w, d, wn)
        return _Reduced(True, -1, error + 1)
    var tn = ((exp + 2) + 63) // 64
    var dn = wn + tn
    var nn = wn + 2 * tn
    if dn > _LONG_LIMBS or tn > 1:
        return _Reduced(False, 0, 0)
    var d = work
    var u = d.unsafe_offset(dn)
    var q = u.unsafe_offset(nn + 1)
    _read_limbs(_LN2_VALUE, 0, _LONG_LIMBS, dn, d)
    _ = _fixed_from_float(x, u, nn, 64 * dn)
    u.unsafe_offset(nn)[] = 0
    _divide_limbs(u, nn + 1, d, dn, q)
    # The quotient has nn + 1 - dn = tn + 1 limbs; |x| < 2**63 keeps it in one.
    var quotient = q.unsafe_offset(0)[]
    if q.unsafe_offset(1)[] != 0 or quotient >> 62:
        return _Reduced(False, 0, 0)
    if not x._negative:
        _copy(w, u.unsafe_offset(tn), wn)
        return _Reduced(True, Int(quotient), 2)
    # x < 0: w = ln 2 - r, q = -(q + 1).
    _copy(w, d.unsafe_offset(tn), wn)
    _ = _subtract_limbs(w, u.unsafe_offset(tn), wn)
    return _Reduced(True, -Int(quotient) - 1, 3)


def _reduce_pi4(w: _Limbs, wn: Int, x: Float, work: _Limbs) raises -> _Reduced:
    """`|x|` reduced modulo pi/4 into `[0, pi/4)` as a fraction of wn limbs,
    with the octant `floor(|x| / (pi/4)) mod 8` as the quotient; an odd octant
    stores `pi/4 - r`. Within 2 units, as `_reduce_log2` with pi/4 for ln 2,
    and 3 for an odd octant, whose pi/4 - r reads pi/4 once more."""
    var exp = x._exponent
    if exp <= -1:
        var error = UInt64(Int(_fixed_from_float(x, w, wn, 64 * wn)))
        return _Reduced(True, 0, error)
    if exp == 0:
        if wn > _LONG_LIMBS:
            return _Reduced(False, 0, 0)
        var error = UInt64(Int(_fixed_from_float(x, w, wn, 64 * wn)))
        var d = work
        _read_limbs(_QUARTER_PI_VALUE, 0, _LONG_LIMBS, wn, d)
        var below = False
        for i in range(wn - 1, -1, -1):
            var a = w.unsafe_offset(i)[]
            var b = d.unsafe_offset(i)[]
            if a != b:
                below = a < b
                break
        if below:
            return _Reduced(True, 0, error)
        # pi/4 <= |x| < 1: octant 1, w = pi/4 - (|x| - pi/4).
        _ = _subtract_limbs(w, d, wn)
        _ = _subtract_limbs(d, w, wn)
        _copy(w, d, wn)
        return _Reduced(True, 1, error + 2)
    var tn = ((exp + 2) + 63) // 64
    var dn = wn + tn
    var nn = wn + 2 * tn
    if dn > _LONG_LIMBS or tn > 1:
        return _Reduced(False, 0, 0)
    var d = work
    var u = d.unsafe_offset(dn)
    var q = u.unsafe_offset(nn + 1)
    _read_limbs(_QUARTER_PI_VALUE, 0, _LONG_LIMBS, dn, d)
    _ = _fixed_from_float(x, u, nn, 64 * dn)
    u.unsafe_offset(nn)[] = 0
    _divide_limbs(u, nn + 1, d, dn, q)
    var octant = Int(q.unsafe_offset(0)[] % 8)
    if octant % 2 == 0:
        _copy(w, u.unsafe_offset(tn), wn)
        return _Reduced(True, octant, 2)
    _copy(w, d.unsafe_offset(tn), wn)
    _ = _subtract_limbs(w, u.unsafe_offset(tn), wn)
    return _Reduced(True, octant, 3)


# ---------------------------------------------------------------- table steps


@always_inline
def _split_top(w: _Limbs, wn: Int, wp: Int) -> Tuple[UInt64, UInt64]:
    """Take the table indices off the top of the fraction w: its top 8 bits
    up to 512 bits of working precision, else its top 5 bits and the next 5
    (tables of 2**8, or two of 2**5, entries). w keeps the rest, below
    2**-8 or 2**-10."""
    var top = w.unsafe_offset(wn - 1)[]
    if wp <= _SHORT_PREC:
        var p1 = top >> 56
        w.unsafe_offset(wn - 1)[] = top - (p1 << 56)
        return (p1, UInt64(0))
    var p1 = top >> 59
    var p2 = (top - (p1 << 59)) >> 54
    w.unsafe_offset(wn - 1)[] = top - (p1 << 59) - (p2 << 54)
    return (p1, p2)


def _rotate(
    sin_a: _Limbs, cos_a: _Limbs, sin_b: _Limbs, cos_b: _Limbs,
    sin_out: _Limbs, cos_out: _Limbs, ta: _Limbs, tb: _Limbs, wn: Int,
) raises:
    """sin(a + b) = sin a cos b + cos a sin b and cos(a + b) = cos a cos b -
    sin a sin b, as fractions of wn limbs. With e_a and e_b units of error in
    the inputs, the results are within 2 e_a + 2 e_b + 3 units: each product
    is off by the sum of its factors' errors (both factors are below 1), by
    their product (below a unit) and by its truncation (one unit). ta and tb
    hold 2 wn limbs each; the outputs may not alias the inputs."""
    _ = _multiply_limbs(sin_a, wn, cos_b, wn, ta)
    _ = _multiply_limbs(cos_a, wn, sin_b, wn, tb)
    _copy(sin_out, ta.unsafe_offset(wn), wn)
    _ = _add_limbs(sin_out, tb.unsafe_offset(wn), wn)
    _ = _multiply_limbs(cos_a, wn, cos_b, wn, ta)
    _ = _multiply_limbs(sin_a, wn, sin_b, wn, tb)
    _copy(cos_out, ta.unsafe_offset(wn), wn)
    _ = _subtract_limbs(cos_out, tb.unsafe_offset(wn), wn)


# ---------------------------------------------------------------- exp


def _exp_medium(x: Float, prec: Int, minus_one: Bool) raises -> Optional[Ball]:
    """exp(x), or expm1(x) with minus_one, of a finite nonzero Float, at
    `prec` bits below 4608 bits; None outside."""
    var exp = x._exponent
    # Tiny and huge arguments go to the general kernel.
    if exp < (-prec - 4 if minus_one else -(prec // 2) - 4) or exp > 60:
        return None
    var wp = prec + 8
    if minus_one and exp <= 0:
        wp += -exp
    var wn = (wp + 63) // 64
    var wprounded = 64 * wn
    wp = max(wp, wprounded - 60)
    if wp > _LONG_PREC:
        return None
    var scratch = Array[UInt64, _SCRATCH](uninitialized=True)
    var block = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var result = _exp_fixed(x, prec, minus_one, wp, wn, block)
    _ = scratch^
    return result


def _exp_fixed(x: Float, prec: Int, minus_one: Bool, wp: Int, wn: Int, block: _Limbs) raises -> Optional[Ball]:
    var wprounded = 64 * wn
    var w = block
    var t = w.unsafe_offset(wn + 1)
    var u = t.unsafe_offset(wn + 1)
    var work = u.unsafe_offset(2 * wn + 1)
    var reduced = _reduce_log2(w, wn, x, work)
    if not reduced.ok:
        return None
    var n = reduced.quotient
    # err(w) propagates through exp' < 3.
    var error = reduced.error * 3
    var p1, p2 = _split_top(w, wn, wp)
    var r = _leading_zeros(w, wn)
    var big_n = _exp_taylor_bound(-r, wp)
    if big_n < 60:
        error += _exp_series(t, w, wn, big_n, work)
        error += UInt64(1) << UInt64(wprounded - wp)
    else:
        # cosh from sinh by a square root.
        big_n = (big_n + 1) // 2
        error += _sin_cos_series(t, u, w, wn, big_n, True, False, work)
        error += UInt64(1) << UInt64(wprounded - wp)
        _ = _multiply_limbs(t, wn, t, wn, u, square=True)
        u.unsafe_offset(2 * wn)[] = 1
        _square_root_limbs(u, 2 * wn + 1, w, wn + 1)
        t.unsafe_offset(wn)[] = w.unsafe_offset(wn)[] + _add_limbs(t, w, wn)
        # cosh's error is below sinh's, plus 1 from the square root.
        error = 2 * error + 1
    var final = t
    var final_n = wn + 1
    if wp <= _SHORT_PREC:
        if p1 != 0:
            # t / 2 < 1 times exp(p1/256) / 2: (t + e1)(u + 1) + 1 within e1 + 4.
            _shift_right_limbs(t, wn + 1, 1, t, wn + 1)
            error = (error >> 1) + 2
            _read_limbs(_EXP_SHORT, Int(p1), _SHORT_LIMBS, wn, work)
            _ = _multiply_limbs(t, wn, work, wn, u)
            error += 4
            final = u.unsafe_offset(wn)
            final_n = wn
            n += 2
    elif p1 != 0 or p2 != 0:
        _shift_right_limbs(t, wn + 1, 1, t, wn + 1)
        error = (error >> 1) + 2
        var a = work
        var b = a.unsafe_offset(wn)
        _read_limbs(_EXP_LONG_COARSE, Int(p1), _LONG_LIMBS, wn, a)
        _read_limbs(_EXP_LONG_FINE, Int(p2), _LONG_LIMBS, wn, b)
        _ = _multiply_limbs(a, wn, b, wn, u)
        # The table product is within 4 units; times t, within e1 + 6.
        _copy(w, u.unsafe_offset(wn), wn)
        _ = _multiply_limbs(t, wn, w, wn, u)
        error += 6
        final = u.unsafe_offset(wn)
        final_n = wn
        n += 3
    if not minus_one:
        return _fixed_ball(final, final_n, wn, False, n, error, _Radius.zero(), prec)
    # expm1: final 2**(n - 64 wn) - 1, rounded once on the limbs rather than
    # subtracting the one from a rounded ball.
    var shift = 64 * wn - n
    var difference = work
    if shift < 0:
        # exp(x) is far above 1 and the one lies below the last limb. The exact
        # value 2**g - 1, g = -shift, lies strictly inside ((value - 1) 2**g,
        # value 2**g), which holds no point or midpoint of the grid at prec
        # bits since the value has more than prec + 8 bits; so does
        # value 2**64 - 1, one limb lower, which rounds the same way in every
        # mode.
        difference.unsafe_offset(0)[] = UInt64.MAX
        _copy(difference.unsafe_offset(1), final, final_n)
        _ = _subtract_one(difference.unsafe_offset(1), final_n, 1)
        return _fixed_ball(difference, final_n + 1, wn + 1, False, n, 0, _up(UInt128(error), n - 64 * wn), prec)
    if shift >= 64 * final_n + prec + 2:
        # exp(x) < 2**-(prec + 2): expm1 is -(1 - v) with v below a quarter
        # unit of 1, in the same rounding cell as -(1 - 2**-(64 L)) for
        # 64 L >= prec + 3.
        var limbs = (prec + 66) // 64
        for i in range(limbs):
            difference.unsafe_offset(i)[] = UInt64.MAX
        return _fixed_ball(difference, limbs, limbs, True, 0, 0, _up(UInt128(error), n - 64 * wn), prec)
    if shift < 64 * final_n:
        var limb = shift >> 6
        var bit = UInt64(1) << UInt64(shift & 63)
        var above = final.unsafe_offset(limb)[] >= bit
        for i in range(limb + 1, final_n):
            above = above or final.unsafe_offset(i)[] != 0
        if above:
            _copy(difference, final, final_n)
            _ = _subtract_one(difference.unsafe_offset(limb), final_n - limb, bit)
        else:
            _zero(difference, final_n)
            difference.unsafe_offset(limb)[] = bit
            _ = _subtract_limbs(difference, final, final_n)
        return _fixed_ball(difference, final_n, wn, not above, n, error, _Radius.zero(), prec)
    # 2**shift - final with the bit up to prec + 2 limbs above: a few limbs more.
    var value = _limbs_integer(final, final_n, False)
    var record = _round_ratio(value - (Integer(1) << shift), Integer(1), ArithmeticContext(format=FloatFormat(prec)), scale=Int128(-shift))
    return _from_record(record, _up(UInt128(error), n - 64 * wn), prec)


# ---------------------------------------------------------------- sin and cos


@fieldwise_init
struct _SinCos(ImplicitlyCopyable):
    """`|sin x|` and `|cos x|` as fixed-point fractions of wn limbs in a
    kernel's block, their signs, and one error count in units of the last
    limb; `ok` is False where the kernel declines."""

    var sine: _Limbs
    var cosine: _Limbs
    var sin_negative: Bool
    var cos_negative: Bool
    var error: UInt64
    var ok: Bool


def _sin_cos_size(x: Float, prec: Int) -> Tuple[Int, Int]:
    """The working precision and limbs for sin and cos of x at `prec` bits;
    no limbs where the kernel does not apply."""
    var exp = x._exponent
    if exp < -(prec // 2) - 2 or exp > 60:
        return (0, 0)
    var wp = prec + 8
    if exp <= 0:
        wp += -exp
    var wn = (wp + 63) // 64
    wp = max(wp, 64 * wn - 60)
    if wp > _LONG_PREC:
        return (0, 0)
    return (wp, wn)


def _sin_cos_medium(x: Float, radius: _Radius, prec: Int) raises -> Optional[Tuple[Ball, Ball]]:
    """`(sin x, cos x)` at `prec` bits for a finite nonzero Float x, or of the
    ball of x with radius `radius`, below 4608 bits; None outside, and None
    when x lies so near a zero of either that the result would lose relative
    accuracy (the general kernel then takes more bits of pi)."""
    var size = _sin_cos_size(x, prec)
    if not size[1]:
        return None
    var scratch = Array[UInt64, _SCRATCH](uninitialized=True)
    var block = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var f = _sin_cos_fixed(x, size[0], size[1], block)
    var result = Optional[Tuple[Ball, Ball]](None)
    if f.ok:
        result = (_sin_cos_ball(f, False, radius, size[1], prec), _sin_cos_ball(f, True, radius, size[1], prec))
    _ = scratch^
    return result^


def _sin_or_cos_medium(x: Float, radius: _Radius, prec: Int, cosine: Bool) raises -> Optional[Ball]:
    """sin x, or cos x with `cosine`, as `_sin_cos_medium`, rounding only the
    one asked for."""
    var size = _sin_cos_size(x, prec)
    if not size[1]:
        return None
    var scratch = Array[UInt64, _SCRATCH](uninitialized=True)
    var block = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var f = _sin_cos_fixed(x, size[0], size[1], block)
    var result = Optional[Ball](None)
    if f.ok:
        result = _sin_cos_ball(f, cosine, radius, size[1], prec)
    _ = scratch^
    return result^


def _sin_cos_ball(f: _SinCos, cosine: Bool, radius: _Radius, wn: Int, prec: Int) raises -> Ball:
    """sin x, or cos x with `cosine`, from the kernel's limbs, for an argument
    within `radius` r of x. By Taylor's theorem the value moves by at most
    `A r + r**2 / 2`, A above the other function's absolute value (the
    derivative's, and the second derivative is at most 1), here from that
    value's top limb and the error; and by at most r, the derivative being at
    most 1. The smaller bound joins the radius."""
    var value = f.cosine if cosine else f.sine
    var other = f.sine if cosine else f.cosine
    var extra = _Radius.zero()
    if not radius.is_zero():
        var bound = _up(UInt128(other.unsafe_offset(wn - 1)[]) + 1, -64).add(_up(UInt128(f.error), -64 * wn))
        extra = bound.multiply(radius).add(radius.multiply(radius).scale2(-1))
        if extra.compare(radius) > 0:
            extra = radius
    return _fixed_ball(value, wn, wn, f.cos_negative if cosine else f.sin_negative, 0, f.error, extra, prec)


def _sin_cos_fixed(x: Float, wp: Int, wn: Int, block: _Limbs) raises -> _SinCos:
    var wprounded = 64 * wn
    var w = block
    var sina = w.unsafe_offset(wn)
    var cosa = sina.unsafe_offset(wn)
    var sinb = cosa.unsafe_offset(wn)
    var cosb = sinb.unsafe_offset(wn)
    var ta = cosb.unsafe_offset(wn)
    var tb = ta.unsafe_offset(2 * wn + 1)
    var work = tb.unsafe_offset(2 * wn + 1)
    var reduced = _reduce_pi4(w, wn, x, work)
    if not reduced.ok:
        return _SinCos(w, w, False, False, 0, False)
    var octant = reduced.quotient
    var error = reduced.error
    var sin_negative = (octant >= 4) != x._negative
    var cos_negative = octant >= 2 and octant <= 5
    var swap = octant == 1 or octant == 2 or octant == 5 or octant == 6
    var p1, p2 = _split_top(w, wn, wp)
    var r = _leading_zeros(w, wn)
    var big_n = (_exp_taylor_bound(-r, wp) + 1) // 2
    if big_n < 14:
        error += _sin_cos_series(sina, cosa, w, wn, big_n, False, True, work)
        error += UInt64(1) << UInt64(wprounded - wp)
    else:
        # cos from sin by a square root, 1 more unit.
        error += _sin_cos_series(sina, cosa, w, wn, big_n, True, True, work)
        error += UInt64(1) << UInt64(wprounded - wp)
        var nonzero = False
        for i in range(wn):
            nonzero = nonzero or sina.unsafe_offset(i)[] != 0
        if not nonzero:
            for i in range(wn):
                cosa.unsafe_offset(i)[] = UInt64.MAX
            error = max(error, 1)
        else:
            _ = _multiply_limbs(sina, wn, sina, wn, ta, square=True)
            # 1 - s**2: the negation of the 2 wn limbs, with a borrow.
            var carry = UInt64(1)
            for i in range(2 * wn):
                var v = ~ta.unsafe_offset(i)[] + carry
                carry = UInt64(1) if carry == 1 and v == 0 else UInt64(0)
                ta.unsafe_offset(i)[] = v
            _square_root_limbs(ta, 2 * wn, cosa, wn)
            error += 1
    # The table's angle added back (`_rotate`): within 2 e + 2 e_table + 3.
    var sin_out = sina
    var cos_out = cosa
    if p1 != 0 or p2 != 0:
        var sinc = work
        var cosc = sinc.unsafe_offset(wn)
        if p1 == 0 or p2 == 0:
            if wp <= _SHORT_PREC:
                _read_limbs(_SIN_COS_SHORT, 2 * Int(p1), _SHORT_LIMBS, wn, sinc)
                _read_limbs(_SIN_COS_SHORT, 2 * Int(p1) + 1, _SHORT_LIMBS, wn, cosc)
            elif p1 != 0:
                _read_limbs(_SIN_COS_LONG_COARSE, 2 * Int(p1), _LONG_LIMBS, wn, sinc)
                _read_limbs(_SIN_COS_LONG_COARSE, 2 * Int(p1) + 1, _LONG_LIMBS, wn, cosc)
            else:
                _read_limbs(_SIN_COS_LONG_FINE, 2 * Int(p2), _LONG_LIMBS, wn, sinc)
                _read_limbs(_SIN_COS_LONG_FINE, 2 * Int(p2) + 1, _LONG_LIMBS, wn, cosc)
            error = 2 * error + 2 + 3
        else:
            var sind = cosc.unsafe_offset(wn)
            var cosd = sind.unsafe_offset(wn)
            _read_limbs(_SIN_COS_LONG_COARSE, 2 * Int(p1), _LONG_LIMBS, wn, sinb)
            _read_limbs(_SIN_COS_LONG_COARSE, 2 * Int(p1) + 1, _LONG_LIMBS, wn, cosb)
            _read_limbs(_SIN_COS_LONG_FINE, 2 * Int(p2), _LONG_LIMBS, wn, sind)
            _read_limbs(_SIN_COS_LONG_FINE, 2 * Int(p2) + 1, _LONG_LIMBS, wn, cosd)
            # Two table entries, each within a unit: within 2 + 2 + 3 units.
            _rotate(sinb, cosb, sind, cosd, sinc, cosc, ta, tb, wn)
            error = 2 * error + 2 * (2 + 2 + 3) + 3
        # sin into sinb, cos into cosb (free now).
        _rotate(sina, cosa, sinc, cosc, sinb, cosb, ta, tb, wn)
        sin_out = sinb
        cos_out = cosb
    if swap:
        var held = sin_out
        sin_out = cos_out
        cos_out = held
    # Near a zero (only past pi/4, where x itself is not small), the general
    # kernel keeps the relative accuracy that correct rounding needs.
    if x._exponent > 0 and (_leading_zeros(sin_out, wn) > 10 or _leading_zeros(cos_out, wn) > 10):
        return _SinCos(w, w, False, False, 0, False)
    return _SinCos(sin_out, cos_out, sin_negative, cos_negative, error, True)


# ---------------------------------------------------------------- atan


def _atan_medium(x: Float, prec: Int) raises -> Optional[Ball]:
    """atan of a finite nonzero Float at `prec` bits below 4608 bits; None
    outside, and for `|x| = 1`."""
    var exp = x._exponent
    if exp < -(prec // 2) - 2 or exp > prec + 2:
        return None
    var p = x.precision()
    var unit = x._exponent == 1 and x._significand == Integer(1) << (p - 1)
    if unit:
        return None
    var wp = prec - min(0, exp) + 4
    if wp > _LONG_PREC:
        return None
    var wn = (wp + 63) // 64
    var scratch = Array[UInt64, _SCRATCH](uninitialized=True)
    var block = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var result = _atan_fixed(x, prec, wp, wn, block)
    _ = scratch^
    return result


def _atan_fixed(x: Float, prec: Int, wp: Int, wn: Int, block: _Limbs) raises -> Ball:
    var w = block
    var t = w.unsafe_offset(wn + 2)
    var u = t.unsafe_offset(wn + 2)
    var work = u.unsafe_offset(2 * wn + 2)
    var reciprocal = x._exponent > 0
    var error: UInt64
    if not reciprocal:
        error = UInt64(Int(_fixed_from_float(x, w, wn, 64 * wn)))
    else:
        # w = floor(2**(64 wn) / |x|) < 1.
        var p = x.precision()
        var quotient = (Integer(1) << (64 * wn + p - x._exponent)) // x._significand
        var small = quotient._inline_words()
        _load_limbs(quotient._words_span(small), 0, w, wn)
        error = 1
    var q1bits = 8 if wp <= _SHORT_PREC else 5
    var p1 = w.unsafe_offset(wn - 1)[] >> UInt64(64 - q1bits)
    # atan(w) = atan(p/q) + atan((q w - p)/(q + p w)), within 1 more unit.
    if p1 != 0:
        _atan_step(w, wn, p1, q1bits, t, u, work)
        error += 1
    var p2 = UInt64(0)
    if wp > _SHORT_PREC:
        p2 = w.unsafe_offset(wn - 1)[] >> 54
        if p2 != 0:
            _atan_step(w, wn, p2, 10, t, u, work)
            error += 1
    var r = _leading_zeros(w, wn)
    var big_n = (wp - r + (2 * r - 1)) // (2 * r)
    error += _atan_series(t, w, wn, big_n, True, work)
    var tn = wn
    if p1 != 0:
        if wp <= _SHORT_PREC:
            _read_limbs(_ATAN_SHORT, Int(p1), _SHORT_LIMBS, wn, work)
        else:
            _read_limbs(_ATAN_LONG_COARSE, Int(p1), _LONG_LIMBS, wn, work)
        _ = _add_limbs(t, work, wn)
        error += 1
    if p2 != 0:
        _read_limbs(_ATAN_LONG_FINE, Int(p2), _LONG_LIMBS, wn, work)
        _ = _add_limbs(t, work, wn)
        error += 1
    if reciprocal:
        # pi/2 - atan(1/x) = (pi/2 - 1) - t + 1.
        _read_limbs(_HALF_PI_MINUS_ONE_VALUE, 0, _LONG_LIMBS, wn, work)
        var borrow = _subtract_limbs(work, t, wn)
        _copy(t, work, wn)
        t.unsafe_offset(wn)[] = 1 - borrow
        tn += Int(t.unsafe_offset(wn)[] != 0)
        error += 1
    # The series' truncation: below 2**(-r (2N + 1)).
    var truncation = _Radius.power_of_two(-r * (2 * big_n + 1))
    return _fixed_ball(t, tn, wn, x._negative, 0, error, truncation, prec)


def _atan_step(w: _Limbs, wn: Int, p: UInt64, qbits: Int, t: _Limbs, u: _Limbs, work: _Limbs) raises:
    """w = (q w - p) / (q + p w) for q = 2**qbits and p the top qbits of w,
    truncated."""
    # t = q + p w, with its integer part in t[wn].
    _copy(t, w, wn)
    t.unsafe_offset(wn)[] = (UInt64(1) << UInt64(qbits)) + _multiply_by_limb(t, wn, p)
    # u = (q w - p) 2**(64 wn), 2 wn + 1 limbs.
    _zero(u, wn)
    _copy(u.unsafe_offset(wn), w, wn)
    u.unsafe_offset(2 * wn)[] = 0
    _ = _shift_left_limbs(u.unsafe_offset(wn), wn, qbits)
    u.unsafe_offset(2 * wn)[] -= p
    _divide_fixed(u, 2 * wn + 1, t, wn + 1, w, wn, work)


def _divide_fixed(num: _Limbs, nn: Int, den: _Limbs, dn: Int, target: _Limbs, tn: Int, work: _Limbs) raises:
    """target[0, tn) = floor(num / den) for a quotient below 2**(64 tn)."""
    var dn_used = dn
    while dn_used > 1 and den.unsafe_offset(dn_used - 1)[] == 0:
        dn_used -= 1
    var q = work
    var scratch = q.unsafe_offset(nn + 3)
    _ = _divide_normalized(num, nn, den, dn_used, q, scratch)
    _copy(target, q, tn)


# ---------------------------------------------------------------- log


def _log_medium(x: Float, prec: Int) raises -> Optional[Ball]:
    """log of a positive finite Float other than 1 at `prec` bits below 4608
    bits; None outside."""
    var exp = x._exponent
    if exp > 60 or exp < -60:
        return None
    # c >= 0 with |x - 1| <= 2**-c when c > 0.
    var closeness = 0
    if exp == 0 or exp == 1:
        var e = Float(1) - x if exp == 0 else x - Float(1)
        if e.is_zero():
            return None
        closeness = -e._exponent
    if closeness > prec // 2:
        return None
    var wp = prec + closeness + 5
    if wp > _LONG_PREC:
        return None
    var wn = (wp + 63) // 64
    var scratch = Array[UInt64, _SCRATCH](uninitialized=True)
    var block = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var result = _log_fixed(x, prec, wp, wn, block)
    _ = scratch^
    return result


def _log_fixed(x: Float, prec: Int, wp: Int, wn: Int, block: _Limbs) raises -> Ball:
    var w = block
    var t = w.unsafe_offset(wn + 2)
    var u = t.unsafe_offset(wn + 2)
    var work = u.unsafe_offset(2 * wn + 2)
    var exp = x._exponent
    # w = 2 m - 1 for x = m 2**exp, m in [1/2, 1): the fraction of x 2**(1 - exp).
    var error = UInt64(Int(_fixed_from_float(x, w, wn + 1, 64 * wn + 1 - exp)))
    w.unsafe_offset(wn)[] = 0
    var q1bits = 7 if wp <= _SHORT_PREC else 5
    var p1 = w.unsafe_offset(wn - 1)[] >> UInt64(64 - q1bits)
    var p2 = UInt64(0)
    var used_series = False
    var big_n = 0
    var r = 0
    var exact_table = True
    for i in range(wn - 1):
        exact_table = exact_table and w.unsafe_offset(i)[] == 0
    exact_table = exact_table and w.unsafe_offset(wn - 1)[] == p1 << UInt64(64 - q1bits) and error == 0
    if exact_table:
        _zero(t, wn)
    else:
        # log(1 + w) = log(1 + p/q) + log(1 + (q w - p)/(p + q)), 1 more unit.
        w.unsafe_offset(wn)[] = _multiply_by_limb(w, wn, UInt64(1) << UInt64(q1bits)) - p1
        _ = _divide_limbs_by_limb(w, wn + 1, p1 + (UInt64(1) << UInt64(q1bits)))
        error += 1
        # The second reduction, fused with log(1 + v) = 2 atanh(v / (2 + v)):
        # w = (Q w - p2) / (Q w + p2 + 2 Q), Q = 2**q2bits, 3 more units.
        var q2bits = 14 if wp <= _SHORT_PREC else 10
        p2 = w.unsafe_offset(wn - 1)[] >> UInt64(64 - q2bits)
        _zero(u, wn)
        _copy(u.unsafe_offset(wn), w, wn)
        u.unsafe_offset(2 * wn)[] = 0
        _ = _shift_left_limbs(u.unsafe_offset(wn), wn, q2bits)
        _copy(t, u.unsafe_offset(wn), wn + 1)
        t.unsafe_offset(wn)[] += p2 + (UInt64(1) << UInt64(q2bits + 1))
        u.unsafe_offset(2 * wn)[] -= p2
        _divide_fixed(u, 2 * wn + 1, t, wn + 1, w, wn, work)
        error += 3
        r = _leading_zeros(w, wn)
        big_n = max((wp - r + (2 * r - 1)) // (2 * r), 0)
        var series_error = _atan_series(t, w, wn, big_n, False, work)
        _ = _shift_left_limbs(t, wn, 1)
        error += series_error * 2
        used_series = True
    var tn = wn
    t.unsafe_offset(wn)[] = 0
    if p1 != 0:
        if wp <= _SHORT_PREC:
            _read_limbs(_LOG_SHORT_COARSE, Int(p1), _SHORT_LIMBS, wn, work)
        else:
            _read_limbs(_LOG_LONG_COARSE, Int(p1), _LONG_LIMBS, wn, work)
        t.unsafe_offset(wn)[] += _add_limbs(t, work, wn)
        error += 1
    if p2 != 0:
        if wp <= _SHORT_PREC:
            _read_limbs(_LOG_SHORT_FINE, Int(p2), _SHORT_LIMBS, wn, work)
        else:
            _read_limbs(_LOG_LONG_FINE, Int(p2), _LONG_LIMBS, wn, work)
        t.unsafe_offset(wn)[] += _add_limbs(t, work, wn)
        error += 1
    # Add (exp - 1) ln 2.
    var k = exp - 1
    var negative = False
    var ln2 = work
    _read_limbs(_LN2_VALUE, 0, _LONG_LIMBS, wn, ln2)
    if k > 0:
        ln2.unsafe_offset(wn)[] = _multiply_by_limb(ln2, wn, UInt64(k))
        t.unsafe_offset(wn)[] += _add_limbs(t, ln2, wn) + ln2.unsafe_offset(wn)[]
        error += UInt64(k)
    elif k < 0:
        ln2.unsafe_offset(wn)[] = _multiply_by_limb(ln2, wn, UInt64(-k))
        var order = 0
        for i in range(wn, -1, -1):
            var a = t.unsafe_offset(i)[]
            var b = ln2.unsafe_offset(i)[]
            if a != b:
                order = -1 if a < b else 1
                break
        if order >= 0:
            _ = _subtract_limbs(t, ln2, wn + 1)
        else:
            _ = _subtract_limbs(ln2, t, wn + 1)
            _copy(t, ln2, wn + 1)
            negative = True
        error += UInt64(-k)
    tn += Int(t.unsafe_offset(wn)[] != 0)
    var truncation = _Radius.power_of_two(-r * (2 * big_n + 1) + 1) if used_series else _Radius.zero()
    return _fixed_ball(t, tn, wn, negative, 0, error, truncation, prec)
