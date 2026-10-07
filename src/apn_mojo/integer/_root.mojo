"""Integer k-th roots for k >= 3.

Zimmermann's root on 64-bit limbs (Brent and Zimmermann, Modern Computer
Arithmetic, Cambridge University Press, 2010, chapter 1). From the floor
root S of a prefix of the radicand, one Newton step on the remainder gives the
next b bits: their quotient by k S**(k - 1). The root's bits grow from a
seed, roughly doubling per level. Radicands of up to 256 bits, and the seeds
of longer ones for degrees up to 8, are settled natively: Newton steps on the
exact residual from a Float64 estimate, with overflow-checked powers. Each
level's root is certified before the next step: from above while S**k exceeds
the prefix, and from below by the remainder, since (S + 1)**k - S**k >
k S**(k - 1); (S + 1)**k is computed only when the remainder exceeds
k S**(k - 1). The book's bound makes one correction per level at most, but
exactness depends neither on it nor on the seed.
"""

from std.bit import count_leading_zeros
from std.collections import Array
from std.math import exp2, log2
from std.memory import bitcast
from std.sys import bit_width_of
from .value import Integer
from ._division import _load_limbs, _store_words
from ._square_root import _wide_magnitude
from ._limbs import (
    _add_one, _compare_used, _divide_normalized, _multiply_by_limb, _multiply_limbs, _power_limbs,
    _shift_left_limbs, _shift_right_limbs, _subtract_limbs, _subtract_one, _used,
)

comptime _LIMB_SEED_BITS = 30
"""Seed root bits for degrees from 9, whose prefixes pass 256 bits: within a
unit of the Float64 estimate, so their certification on limbs, a unit at a
time, takes a step or two. Mojo's exp2 and log2 make the estimate good to
about 2**-31 relatively (the measured worst case)."""

comptime _BITWISE_ROOT_BITS = 11
"""Roots of radicands up to 64 bits shorter than this take one power per bit,
about 114 + 27 cycles a bit (measured), instead of an estimate and its
certification, about 410 cycles: a chain of Float64 operations."""

comptime _MAX_LEVELS = 64
"""Schedule entries: the root's bit count about halves per level."""

comptime _ROOT_STACK_LIMBS = 640
"""Scratch limbs on the stack: cube roots of radicands up to 2,880 bits."""


def _schedule(xnb: Int, k: Int, mut sizes: Array[Int, _MAX_LEVELS]) -> Int:
    """sizes[0] = xnb, then the bit counts less one of each level's root down
    to one the seed settles; returns that level's index. A level from c + 1 to
    x + 1 bits keeps the book's bound of one correction when c is at least
    (x + log2 k) / 2; logk is ceil(log2 k) + 1, a margin over it."""
    var logk = 3
    var half = ((k - 1) // 2) >> 1
    while half:
        logk += 1
        half >>= 1
    # Degrees up to 8: the largest seed whose prefix fits 256 bits.
    var limit = 256 // k - 1 if k <= 8 else _LIMB_SEED_BITS - 1
    sizes[0] = xnb
    var level = 0
    while sizes[level] > limit:
        var x = sizes[level]
        sizes[level + 1] = (x + logk) // 2 if x > logk else x - 1
        level += 1
    return level


@always_inline
def _power_of_two(n: Int) -> Float64:
    """2**n exactly, for -1022 <= n <= 1023, from its bits: Mojo's exp2 is not
    exact at integers."""
    return bitcast[DType.float64](UInt64(1023 + n) << 52)


def _float_power(base: Float64, exponent: Int) -> Float64:
    """base ** exponent by squarings."""
    var result = 1.0
    var factor = base
    var e = exponent
    while e:
        if e & 1:
            result *= factor
        e >>= 1
        if e:
            factor *= factor
    return result


def _root_estimate(leading: UInt64, nb: Int, k: Int, t: Int) -> UInt128:
    """A Float64 estimate of the k-th root of u >> k t, for u of nb bits whose
    leading 64 bits (all of them, below 64) are leading. The exponent splits so
    that the root taken, c of z = leading 2**rest, lies below 2**(1 + 64 / k):
    long radicands lose no accuracy to it. Mojo's exp2 and log2 give c to
    about 2**-31 (measured), within a unit for roots of up to 28 bits; longer
    ones take one Newton step in Float64, to about 2**-50, while z stays a
    Float64 (rest below 960)."""
    var shift = max(nb - 64, 0)
    var whole = shift // k
    var rest = shift - whole * k
    var c = exp2((log2(Float64(leading)) + Float64(rest)) / Float64(k))
    if (nb - 1) // k - t >= 28 and rest < 960:
        var z = Float64(leading) * _power_of_two(rest)
        c += (z / _float_power(c, k - 1) - c) / Float64(k)
    var estimate = c * _power_of_two(whole - t)
    if estimate < 4.0e18:
        # The hardware conversion, not the 128-bit one's library call.
        return UInt128(estimate.cast[DType.int64]())
    return estimate.cast[DType.uint128]()


@always_inline
def _checked_multiply[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Tuple[Scalar[dtype], Bool]:
    """(a b, False) when the product fits the type, else (0, True). Only bit
    lengths summing to one past the width need more than the lengths: one
    factor then fits half the width, and the other splits there."""
    comptime width = bit_width_of[dtype]()
    comptime half = width // 2
    if not a or not b:
        return (0, False)
    var a_bits = width - Int(count_leading_zeros(a))
    var b_bits = width - Int(count_leading_zeros(b))
    if a_bits + b_bits <= width:
        return (a * b, False)
    if a_bits + b_bits > width + 1:
        return (0, True)
    var split = Scalar[dtype](half)
    var narrow = a if a_bits <= half else b
    var wide = b if a_bits <= half else a
    var high = (wide >> split) * narrow
    if high >> split:
        return (0, True)
    var low = (wide & ((Scalar[dtype](1) << split) - 1)) * narrow
    var total = (high << split) + low
    if total < low:
        return (0, True)
    return (total, False)


@always_inline
def _multiply[dtype: DType, checked: Bool](a: Scalar[dtype], b: Scalar[dtype]) -> Tuple[Scalar[dtype], Bool]:
    """_checked_multiply, or the plain product where the caller knows it fits."""
    comptime if checked:
        return _checked_multiply(a, b)
    else:
        return (a * b, False)


def _native_power[dtype: DType, checked: Bool](base: Scalar[dtype], exponent: Int) -> Tuple[Scalar[dtype], Bool]:
    """(base ** exponent, False) by squarings when the power fits the type,
    else (0, True)."""
    var result = Scalar[dtype](0)
    var started = False
    var factor = base
    var e = exponent
    while True:
        if e & 1:
            if started:
                var product = _multiply[dtype, checked](result, factor)
                if product[1]:
                    return product
                result = product[0]
            else:
                result = factor
                started = True
        e >>= 1
        if not e:
            return (result, False)
        var square = _multiply[dtype, checked](factor, factor)
        if square[1]:
            return square
        factor = square[0]


def _to_float[dtype: DType](value: Scalar[dtype]) -> Float64:
    """value as a Float64, from its leading 64 bits."""
    var bits = bit_width_of[dtype]() - Int(count_leading_zeros(value))
    if bits <= 64:
        return UInt64(value).cast[DType.float64]()
    return UInt64(value >> Scalar[dtype](bits - 64)).cast[DType.float64]() * _power_of_two(bits - 64)


def _ratio[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """About floor(a / b), below 2**128, from Float64 values of both; 0 when
    a < b."""
    if a < b:
        return 0
    return Scalar[dtype]((_to_float(a) / _to_float(b)).cast[DType.uint128]())


def _below_next_power(prefix: Float64, power: Float64, s: Float64, k: Int) -> Bool:
    """Whether prefix < (s + 1)**k is certain, from Float64 values of prefix
    and power = s**k (s >= 2), each within 2**-52 relatively: when
    prefix / power falls below (1 + 1 / s)**k by more than either side's
    rounding. Correctly rounded operations err by at most 2**-53 each: the
    ratio by under 2**-50.9 in all; the base 1 + 1 / s by 1.5 2**-52 (with s
    itself rounded), so its k-th power by k 1.5 2**-52 plus 2**-53 for each of
    at most 2 log2 k products. The margin, (k + 64) 2**-50, exceeds their sum
    for every k up to 2**20; an overflowing power is infinite and passes."""
    if k > 1 << 20:
        return False
    var growth = _float_power(1.0 + 1.0 / s, k)
    return prefix / power < growth * (1.0 - Float64(k + 64) * _power_of_two(-50))


@always_inline
def _leading(x: Pointer[mut=True, UInt64, _], n: Int) -> Tuple[Float64, Int]:
    """x[0, n) (n >= 1, top limb nonzero) as (f, e): x is f 2**e within
    2**-52, from its leading 64 bits."""
    var bits = 64 * n - Int(count_leading_zeros(x.unsafe_offset(n - 1)[]))
    if bits <= 64:
        return (x.unsafe_offset(0)[].cast[DType.float64](), 0)
    var shift = bits - 64
    var limb = x.unsafe_offset(shift >> 6)[]
    if shift & 63:
        limb = (limb >> UInt64(shift & 63)) | (x.unsafe_offset((shift >> 6) + 1)[] << UInt64(64 - (shift & 63)))
    return (limb.cast[DType.float64](), shift)


def _native_certify[dtype: DType, checked: Bool = True](
    var s: Scalar[dtype], prefix: Scalar[dtype], k: Int, low: Scalar[dtype], high: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    """As _certify, natively, for the root of prefix in [low, high], a root
    below 2**128: (root, remainder, k root**(k - 1), saturated at the type's
    maximum). A gap of more than k s**(k - 1) moves s by Newton's correction,
    the gap's quotient by k s**(k - 1): from an estimate off by 2**-31
    relatively, two steps reach the root's unit. A power past the type takes
    the same step from Float64 values, s (1 - prefix / s**k) / k, or s >> 52
    if that is larger. Upward steps stay below the least value known to be
    too large, and s is the root once s + 1 is that value: for small roots of
    high degrees, where the gap overstates the distance, steps would
    otherwise cycle. Unchecked, the caller knows every power fits."""
    var upper = high + 1
    while True:
        var w = _native_power[dtype, checked](s, k - 1)
        var power = _multiply[dtype, checked](w[0], s) if not w[1] else w
        if power[1] or power[0] > prefix:
            upper = s
            var step = Scalar[dtype](1)
            if power[1]:
                # Below 2**-52 the Float64 step is noise, and s may be millions
                # of units above a root just under the type's limit: at least
                # s >> 52 then reaches below the root, from where one step on
                # the exact residual lands within a unit.
                var root = _to_float(s)
                var guess = root * (1.0 - _to_float(prefix) / _float_power(root, k)) / Float64(k)
                step = max(Scalar[dtype](guess.cast[DType.uint128]()) if guess >= 1.0 else 1, s >> 52)
            else:
                var kw = _multiply[dtype, checked](w[0], Scalar[dtype](k))
                if not kw[1]:
                    step = max(_ratio(power[0] - prefix, kw[0]), 1)
            s = max(s - step, low)
            continue
        var rest = prefix - power[0]
        var kw = _multiply[dtype, checked](w[0], Scalar[dtype](k))
        if kw[1]:
            # k s**(k - 1) exceeds any remainder.
            return (s, rest, Scalar[dtype].MAX)
        if rest <= kw[0] or s + 1 == upper:
            return (s, rest, kw[0])
        var step = _ratio(rest, kw[0])
        if step > 1:
            s = min(s + step, upper - 1)
            continue
        if _below_next_power(_to_float(prefix), _to_float(power[0]), _to_float(s), k):
            return (s, rest, kw[0])
        var next = _native_power[dtype, checked](s + 1, k)
        if next[1] or next[0] > prefix:
            return (s, rest, kw[0])
        s += 1


@always_inline
def _store_native(value: UInt256, target: Pointer[mut=True, UInt64, _]) -> Int:
    """value's four limbs into target; the length without leading zero limbs."""
    for i in range(4):
        target.unsafe_offset(i)[] = UInt64(value >> UInt256(64 * i))
    return _used(target, 4)


def _certify(
    s: Pointer[mut=True, UInt64, _], mut sn: Int, p: Pointer[mut=True, UInt64, _], pn: Int, k: Int,
    w: Pointer[mut=True, UInt64, _], q: Pointer[mut=True, UInt64, _], other: Pointer[mut=True, UInt64, _],
    rest: Pointer[mut=True, UInt64, _],
) raises -> Tuple[Int, Int]:
    """Make s[0, sn) the floor k-th root of p[0, pn) from within a few units:
    down while s**k > p, and up while p - s**k exceeds k s**(k - 1) and
    (s + 1)**k <= p, unless a step down showed s + 1 too large. Leaves the
    remainder p - s**k in rest and k s**(k - 1) in w; returns their
    lengths."""
    var above = False
    while True:
        var wn = _power_limbs(s, sn, k - 1, w, other)
        var qn = _multiply_limbs(w, wn, s, sn, q)
        if _compare_used(q, qn, p, pn) > 0:
            _ = _subtract_one(s, sn, 1)
            sn = _used(s, sn)
            above = True
            continue
        for i in range(pn):
            rest.unsafe_offset(i)[] = p.unsafe_offset(i)[]
        var borrow = _subtract_limbs(rest, q, qn)
        _ = _subtract_one(rest.unsafe_offset(qn), pn - qn, borrow)
        var rn = _used(rest, pn)
        var carry = _multiply_by_limb(w, wn, UInt64(k))
        if carry:
            w.unsafe_offset(wn)[] = carry
            wn += 1
        if above or _compare_used(rest, rn, w, wn) <= 0:
            return (rn, wn)
        # (s + 1)**k - s**k may still exceed the remainder; Float64 values of p
        # and s**k settle most cases without (s + 1)**k.
        var pf = _leading(p, pn)
        var qf = _leading(q, qn)
        var sf = _leading(s, sn)
        if pf[1] - qf[1] < 960 and sf[1] < 960:
            var scale = _power_of_two(pf[1] - qf[1])
            if _below_next_power(pf[0] * scale, qf[0], sf[0] * _power_of_two(sf[1]), k):
                return (rn, wn)
        var top = _add_one(s, sn, 1)
        if top:
            s.unsafe_offset(sn)[] = top
            sn += 1
        var tn = _power_limbs(s, sn, k, q, other)
        if _compare_used(q, tn, p, pn) <= 0:
            continue
        _ = _subtract_one(s, sn, 1)
        sn = _used(s, sn)
        return (rn, wn)


def _integer_root(magnitude: Integer, degree: Int) raises -> Tuple[Integer, Bool]:
    """(floor(magnitude ** (1 / degree)), whether that root is exact), for
    magnitude >= 2 and 3 <= degree < magnitude.magnitude_bit_length()."""
    var k = degree
    var nb = magnitude.magnitude_bit_length()
    # The root has xnb + 1 bits.
    var xnb = (nb - 1) // k
    if nb <= 256:
        var value = _wide_magnitude(magnitude)
        if nb <= 64 and xnb < _BITWISE_ROOT_BITS:
            # A short root's bits from the top, one power each, chosen by
            # selects rather than branches, so varying radicands cost the
            # same. Powers of roots of at most xnb + 1 bits fit 128 bits:
            # 2**(k (xnb + 1)) <= 2**(nb - 1 + k) <= 2**127.
            var x = UInt128(value)
            var root = UInt128(1) << UInt128(xnb)
            var root_power = UInt128(1) << UInt128(k * xnb)
            for bit in range(xnb - 1, -1, -1):
                var trial = root | (UInt128(1) << UInt128(bit))
                var power = _native_power[DType.uint128, False](trial, k)[0]
                var fits = power <= x
                root = trial if fits else root
                root_power = power if fits else root_power
            return (Integer(UInt64(root)), root_power == x)
        var low = UInt256(1) << UInt256(xnb)
        var leading = UInt64(value >> UInt256(max(nb - 64, 0)))
        var estimate = min(max(UInt256(_root_estimate(leading, nb, k, 0)), low), low + (low - 1))
        if nb <= 64:
            # Every power of a root of at most xnb + 1 bits fits 128 bits.
            var short = _native_certify[checked=False](
                UInt128(estimate), UInt128(value), k, UInt128(low), UInt128(low + (low - 1))
            )
            return (Integer(UInt64(short[0])), short[1] == 0)
        var native = _native_certify(estimate, value, k, low, low + (low - 1))
        return (Integer._from_wide_magnitude128(UInt128(native[0])), native[1] == 0)
    var sizes = Array[Int, _MAX_LEVELS](fill=0)
    var level = _schedule(xnb, k, sizes)
    var un = (nb + 63) // 64
    var rootn = (xnb + 64) // 64
    # Powers of up to twice the root: k / 64 limbs more than the radicand.
    var big = un + rootn + 5 + k // 64
    var wide = big + rootn + 3
    var prefix_limbs = max(un, 4)
    var root_limbs = max(rootn + 2, 4)
    # u, the prefix, the root, the remainder, k s**(k - 1), products and
    # quotients, power scratch, the dividend, the division's work.
    var limbs = un + prefix_limbs + root_limbs + big + (big + 1) + (wide + 2) + big + wide + (big + wide + 4)
    var stack = Array[UInt64, _ROOT_STACK_LIMBS](uninitialized=True)
    var heap = List[UInt64]()
    var scratch: Pointer[UInt64, MutUntrackedOrigin]
    if limbs <= _ROOT_STACK_LIMBS:
        scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        heap = List[UInt64](unsafe_uninit_length=limbs)
        scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var u = scratch
    var p = u.unsafe_offset(un)
    var s = p.unsafe_offset(prefix_limbs)
    var rest = s.unsafe_offset(root_limbs)
    var w = rest.unsafe_offset(big)
    var q = w.unsafe_offset(big + 1)
    var other = q.unsafe_offset(wide + 2)
    var dividend = other.unsafe_offset(big)
    var work = dividend.unsafe_offset(wide)
    var small = magnitude._inline_words()
    _load_limbs(magnitude._words_span(small), 0, u, un)
    # The seed: the root of u >> k t, which has sizes[level] + 1 bits.
    var t = xnb - sizes[level]
    var low = UInt128(1) << UInt128(sizes[level])
    var shift = nb - 64
    var leading = u.unsafe_offset(shift >> 6)[]
    if shift & 63:
        leading = (leading >> UInt64(shift & 63)) | (u.unsafe_offset((shift >> 6) + 1)[] << UInt64(64 - (shift & 63)))
    var estimate = min(max(_root_estimate(leading, nb, k, t), low), low + (low - 1))
    var sn = 1
    var rn: Int
    var wn: Int
    if k * (sizes[level] + 1) <= 256:
        # The prefix fits 256 bits.
        _shift_right_limbs(u, un, k * t, p, 4)
        var prefix = UInt256(0)
        for i in range(4):
            prefix |= UInt256(p.unsafe_offset(i)[]) << UInt256(64 * i)
        var native = _native_certify(UInt256(estimate), prefix, k, UInt256(low), UInt256(low + (low - 1)))
        if not level:
            _ = heap^
            _ = stack^
            return (Integer._from_wide_magnitude128(UInt128(native[0])), native[1] == 0)
        sn = _store_native(native[0], s)
        rn = _store_native(native[1], rest)
        wn = _store_native(native[2], w)
    else:
        # Degrees from 9: a seed of at most 30 bits.
        s.unsafe_offset(0)[] = UInt64(estimate)
        var pn = (nb - k * t + 63) // 64
        _shift_right_limbs(u, un, k * t, p, pn)
        var certified = _certify(s, sn, p, pn, k, w, q, other, rest)
        rn = certified[0]
        wn = certified[1]
    while level:
        # The dividend: the remainder followed by the b bits of u below the
        # current prefix.
        var b = sizes[level - 1] - sizes[level]
        var bn = (b + 63) // 64
        for i in range(bn):
            dividend.unsafe_offset(i)[] = 0
        for i in range(rn):
            dividend.unsafe_offset(i)[] = rest.unsafe_offset(i)[]
        var nn = max(_shift_left_limbs(dividend, rn, b), bn)
        _shift_right_limbs(u, un, k * t - b, other, bn)
        if b & 63:
            other.unsafe_offset(bn - 1)[] &= (UInt64(1) << UInt64(b & 63)) - 1
        for i in range(bn):
            dividend.unsafe_offset(i)[] |= other.unsafe_offset(i)[]
        nn = _used(dividend, nn)
        # The next b root bits: the quotient by k s**(k - 1), below 2**b.
        var qn = 0
        if _compare_used(dividend, nn, w, wn) >= 0:
            _ = _divide_normalized(dividend, nn, w, wn, q, work)
            qn = _used(q, nn + 2 - wn)
        if qn > bn or (qn == bn and (b & 63) != 0 and (q.unsafe_offset(bn - 1)[] >> UInt64(b & 63)) != 0):
            for i in range(bn):
                q.unsafe_offset(i)[] = UInt64.MAX
            if b & 63:
                q.unsafe_offset(bn - 1)[] = (UInt64(1) << UInt64(b & 63)) - 1
            qn = bn
        sn = _shift_left_limbs(s, sn, b)
        for i in range(qn):
            s.unsafe_offset(i)[] |= q.unsafe_offset(i)[]
        level -= 1
        t = xnb - sizes[level]
        if t:
            var pn = (nb - k * t + 63) // 64
            _shift_right_limbs(u, un, k * t, p, pn)
            var certified = _certify(s, sn, p, pn, k, w, q, other, rest)
            rn = certified[0]
            wn = certified[1]
        else:
            var certified = _certify(s, sn, u, un, k, w, q, other, rest)
            rn = certified[0]
            wn = certified[1]
    var root: Integer
    if sn == 1:
        root = Integer(s.unsafe_offset(0)[])
    elif sn == 2:
        root = Integer._from_wide_magnitude128((UInt128(s.unsafe_offset(1)[]) << 64) | UInt128(s.unsafe_offset(0)[]))
    else:
        var owner = Integer._Shared.uninitialized(2 * sn, False)
        var used = _store_words(s, sn, 0, owner[].words.unsafe_ptr())
        root = Integer._from_product[False](owner^, used)
    _ = heap^
    _ = stack^
    return (root^, rn == 0)
