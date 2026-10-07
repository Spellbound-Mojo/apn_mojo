"""Greatest common divisors of magnitudes.

Magnitudes of up to two limbs take Stein's binary gcd. Longer ones take
Lehmer's algorithm on 64-bit limbs, in one scratch block: each step reads
the leading 128 bits of both operands, reduces them by Euclid's algorithm
into a 2 x 2 matrix of determinant 1 with entries below 2**64 (the
double-digit step of Moller, On Schonhage's algorithm and subquadratic
integer gcd computation, Mathematics of Computation 77, 2008), and applies
the
matrix's inverse to the full operands in one pass, which shortens both by
about 62 bits.
"""

from std.bit import count_leading_zeros, count_trailing_zeros
from std.collections import Array
from std.sys import bit_width_of
from ._division import _load_limbs, _store_words
from ._limbs import _compare, _divide_normalized, _used
from .value import Integer

comptime _GCD_STACK_LIMBS = 400
"""Scratch limbs on the stack: operands up to 64 limbs (4096 bits)."""


def _native_gcd[dtype: DType](var a: Scalar[dtype], var b: Scalar[dtype]) -> Scalar[dtype]:
    """gcd of two native magnitudes of up to two limbs. Stein's steps, a shift
    and a subtraction each, cost less than Euclid's divisions; but they take
    about one step per bit the larger operand has beyond the smaller, so a
    larger operand with 16 more bits first takes one division. Two-limb
    operands that fit one limb take the one-limb steps and hardware division."""
    comptime assert bit_width_of[dtype]() <= 128, "native gcds take at most two limbs"
    comptime if bit_width_of[dtype]() > 64:
        if not (a | b) >> 64:
            return Scalar[dtype](_native_gcd(UInt64(a), UInt64(b)))
    if a < b:
        swap(a, b)
    if b and a >> 16 >= b:
        a %= b
    comptime if bit_width_of[dtype]() > 64:
        return Scalar[dtype](_binary_gcd2(UInt128(a), UInt128(b)))
    else:
        return _binary_gcd(a, b)


def _binary_gcd[dtype: DType](var a: Scalar[dtype], var b: Scalar[dtype]) -> Scalar[dtype]:
    """Stein's binary gcd on native unsigned integers. Each step takes its
    shift from the wrapped difference itself, whose trailing zeros are those of
    its magnitude, so counting them overlaps taking the smaller operand and the
    difference's magnitude (after Lemire's variant)."""
    if not a:
        return b
    if not b:
        return a
    var a_zeros = count_trailing_zeros(a)
    var b_zeros = count_trailing_zeros(b)
    var shift = min(a_zeros, b_zeros)
    a >>= a_zeros
    while True:
        b >>= b_zeros
        var difference = b - a
        b_zeros = count_trailing_zeros(difference)
        if not difference:
            break
        var smaller = min(a, b)
        b = max(a, b) - smaller
        a = smaller
    return a << shift


def _binary_gcd2(var a: UInt128, var b: UInt128) -> UInt128:
    """Stein's binary gcd of two magnitudes below 2**128: steps on limb pairs
    while either operand exceeds one limb, then _binary_gcd's (Knuth, The
    Art of Computer Programming, vol. 2, 4.5.2, Algorithm B). Each step forms
    both differences and picks one by the operands' order, without a branch,
    and counts the trailing zeros of the low limb alone. On 128-bit
    integers the compiler took the absolute difference with a mask and a
    negation, after a 128-bit count, and shifted with tests for counts past
    63."""
    if not a:
        return b
    if not b:
        return a
    var a_zeros = count_trailing_zeros(a)
    var b_zeros = count_trailing_zeros(b)
    var shift = min(a_zeros, b_zeros)
    a >>= a_zeros
    b >>= b_zeros
    var a0 = UInt64(a)
    var a1 = UInt64(a >> 64)
    var b0 = UInt64(b)
    var b1 = UInt64(b >> 64)
    while a1 | b1:
        # Both odd, so b - a is even; unless its low limb is zero, that limb's
        # trailing zeros are all of its zeros, at most 63.
        var low = b0 - a0
        var high = b1 - a1 - UInt64(b0 < a0)
        var b_smaller = ((UInt128(b1) << 64) | UInt128(b0)) < ((UInt128(a1) << 64) | UInt128(a0))
        if not low:
            # Equal low limbs: the difference is a multiple of 2**64.
            if not high:
                return ((UInt128(a1) << 64) | UInt128(a0)) << shift
            var magnitude = a1 - b1 if b_smaller else high
            if b_smaller:
                a1 = b1
            b0 = magnitude >> count_trailing_zeros(magnitude)
            b1 = 0
            continue
        var zeros = count_trailing_zeros(low)
        var magnitude0 = a0 - b0 if b_smaller else low
        var magnitude1 = a1 - b1 - UInt64(a0 < b0) if b_smaller else high
        if b_smaller:
            a0 = b0
            a1 = b1
        b0 = (magnitude0 >> zeros) | (magnitude1 << (64 - zeros))
        b1 = magnitude1 >> zeros
    return UInt128(_binary_gcd(a0, b0)) << shift


@fieldwise_init
struct _Matrix(ImplicitlyCopyable):
    """A reduction step's matrix ((u00, u01), (u10, u11)), determinant 1:
    the old pair is the matrix times the new pair."""

    var u00: UInt64
    var u01: UInt64
    var u10: UInt64
    var u11: UInt64


@no_inline
def _divide2(var n: UInt128, d: UInt128) -> Tuple[UInt64, UInt128]:
    """n // d and n % d for n >= d. Quotients of 1 and 2 take subtractions
    alone; larger ones shift and subtract once per quotient bit. Kept out of
    line: inlined, its work was computed ahead of every step's branch,
    including the many steps that never divide."""
    n -= d
    if n < d:
        return (1, n)
    n -= d
    if n < d:
        return (2, n)
    var shifted = d
    var count = Int(count_leading_zeros(d)) - Int(count_leading_zeros(n))
    shifted <<= UInt128(count)
    var q = UInt64(0)
    for _ in range(count + 1):
        q <<= 1
        if n >= shifted:
            n -= shifted
            q |= 1
        shifted >>= 1
    return (q + 2, n)


@no_inline
def _divide1(var n: UInt64, d: UInt64) -> Tuple[UInt64, UInt64]:
    """n // d and n % d for n >= d, as _divide2 does on one limb."""
    n -= d
    if n < d:
        return (1, n)
    n -= d
    if n < d:
        return (2, n)
    var shifted = d
    var count = Int(count_leading_zeros(d)) - Int(count_leading_zeros(n))
    shifted <<= UInt64(count)
    var q = UInt64(0)
    for _ in range(count + 1):
        q <<= 1
        if n >= shifted:
            n -= shifted
            q |= 1
        shifted >>= 1
    return (q + 2, n)


@no_inline
def _hgcd2(ah: UInt64, al: UInt64, bh: UInt64, bl: UInt64, mut m: _Matrix) -> Bool:
    """Reduce the leading parts (ah, al) and (bh, bl) of two operands, taken at
    the same bit position, by Euclid's algorithm, recording the quotients in
    m; False when no step is safe. The steps stop while the remainders' high
    limbs are still at least 2 (and, on single limbs, the remainders at least
    2**33), so the quotients are those of the full operands: applying m's
    inverse leaves both nonnegative. Kept out of line: inlined into
    _limb_gcd's loop, a 1024-bit gcd took 4% more instructions and 6% more
    cycles."""
    if ah < 2 or bh < 2:
        return False
    var a = (UInt128(ah) << 64) | UInt128(al)
    var b = (UInt128(bh) << 64) | UInt128(bl)
    var u00 = UInt64(1)
    var u01: UInt64
    var u10: UInt64
    var u11 = UInt64(1)
    if a > b:
        a -= b
        if a >> 64 < 2:
            return False
        u01 = 1
        u10 = 0
    else:
        b -= a
        if b >> 64 < 2:
            return False
        u01 = 0
        u10 = 1
    # Double-limb steps, a -= q b then b -= q a, while the larger leading limb
    # holds at least 32 bits; then single-limb steps on 64 significant bits.
    comptime half = UInt128(1) << 32
    var skip_a = a >> 64 < b >> 64
    var single = False
    var single_a = True
    while True:
        if not skip_a:
            if a >> 64 == b >> 64:
                break
            if a >> 64 < half:
                single = True
                break
            a -= b
            if a >> 64 < 2:
                break
            if a >> 64 > b >> 64:
                var q, rest = _divide2(a, b)
                a = rest
                if a >> 64 < 2:
                    u01 += q * u00
                    u11 += q * u10
                    break
                q += 1
                u01 += q * u00
                u11 += q * u10
            else:
                u01 += u00
                u11 += u10
        skip_a = False
        if a >> 64 == b >> 64:
            break
        if b >> 64 < half:
            single = True
            single_a = False
            break
        b -= a
        if b >> 64 < 2:
            break
        if b >> 64 > a >> 64:
            var q, rest = _divide2(b, a)
            b = rest
            if b >> 64 < 2:
                u00 += q * u01
                u10 += q * u11
                break
            q += 1
            u00 += q * u01
            u10 += q * u11
        else:
            u00 += u01
            u10 += u11
    if single:
        # x and y are a and b without their low 32 bits, so the bound of a high
        # limb of at least 2 becomes 2**33.
        comptime limit = UInt64(1) << 33
        var x = UInt64(a >> 32)
        var y = UInt64(b >> 32)
        var skip_x = not single_a
        while True:
            if not skip_x:
                x -= y
                if x < limit:
                    break
                if x > y:
                    var q, rest = _divide1(x, y)
                    x = rest
                    if x < limit:
                        u01 += q * u00
                        u11 += q * u10
                        break
                    q += 1
                    u01 += q * u00
                    u11 += q * u10
                else:
                    u01 += u00
                    u11 += u10
            skip_x = False
            y -= x
            if y < limit:
                break
            if y > x:
                var q, rest = _divide1(y, x)
                y = rest
                if y < limit:
                    u00 += q * u01
                    u10 += q * u11
                    break
                q += 1
                u00 += q * u01
                u10 += q * u11
            else:
                u00 += u01
                u10 += u11
    m = _Matrix(u00, u01, u10, u11)
    return True


@always_inline
def _apply_inverse(
    m: _Matrix, a: Pointer[mut=True, UInt64, _], b: Pointer[mut=True, UInt64, _],
    t: Pointer[mut=True, UInt64, _], n: Int,
):
    """t = u11 a - u01 b and b = u00 b - u10 a over n limbs, in one pass. Both
    are nonnegative and fit n limbs (_hgcd2's guarantee), so the products'
    high parts cancel against the borrows."""
    var t_carry = UInt64(0)
    var t_debt = UInt64(0)
    var t_borrow = UInt64(0)
    var b_carry = UInt64(0)
    var b_debt = UInt64(0)
    var b_borrow = UInt64(0)
    for i in range(n):
        var x = a.unsafe_offset(i)[]
        var y = b.unsafe_offset(i)[]
        var plus = UInt128(m.u11) * UInt128(x) + UInt128(t_carry)
        var minus = UInt128(m.u01) * UInt128(y) + UInt128(t_debt)
        t_carry = UInt64(plus >> 64)
        t_debt = UInt64(minus >> 64)
        var high = UInt64(plus)
        var low = UInt64(minus)
        var difference = high - low
        var borrow = UInt64(high < low) | UInt64(difference < t_borrow)
        t.unsafe_offset(i)[] = difference - t_borrow
        t_borrow = borrow
        plus = UInt128(m.u00) * UInt128(y) + UInt128(b_carry)
        minus = UInt128(m.u10) * UInt128(x) + UInt128(b_debt)
        b_carry = UInt64(plus >> 64)
        b_debt = UInt64(minus >> 64)
        high = UInt64(plus)
        low = UInt64(minus)
        difference = high - low
        borrow = UInt64(high < low) | UInt64(difference < b_borrow)
        b.unsafe_offset(i)[] = difference - b_borrow
        b_borrow = borrow


def _reduce(
    x: Pointer[mut=True, UInt64, _], xn: Int, y: Pointer[mut=True, UInt64, _], yn: Int,
    work: Pointer[mut=True, UInt64, _],
):
    """x = x mod y for x of xn limbs and nonzero y of yn limbs: the general
    long division, on normalized copies in work (yn + xn + 2 + xn + 2 limbs)."""
    var w = work.unsafe_origin_cast[MutUntrackedOrigin]()
    var u = w.unsafe_offset(yn)
    var norm = _divide_normalized(x, xn, y, yn, u.unsafe_offset(xn + 2), w)
    for i in range(xn):
        var limb = UInt64(0)
        if i < yn:
            limb = u.unsafe_offset(i)[] >> norm
            if norm and i + 1 < yn:
                limb |= u.unsafe_offset(i + 1)[] << (64 - norm)
        x.unsafe_offset(i)[] = limb


def _limb_gcd(x: Span[mut=False, UInt32, _], y: Span[mut=False, UInt32, _]) raises -> Integer:
    """gcd of two nonzero magnitudes, given as words: Lehmer's algorithm on
    64-bit limbs in one scratch block, a division when a step makes no
    progress, and the binary gcd once both fit two limbs."""
    var n = max((len(x) + 1) // 2, (len(y) + 1) // 2)
    # a, b, the step's output, and the division's dividend, divisor and quotient.
    var size = 3 * n + (n + 2) + n + (n + 2)
    var stack = Array[UInt64, _GCD_STACK_LIMBS](uninitialized=True)
    var heap = List[UInt64]()
    var scratch: Pointer[UInt64, MutUntrackedOrigin]
    if size <= _GCD_STACK_LIMBS:
        scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        heap = List[UInt64](unsafe_uninit_length=size)
        scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var a = scratch
    var b = scratch.unsafe_offset(n)
    var t = scratch.unsafe_offset(2 * n)
    var work = scratch.unsafe_offset(3 * n)
    _load_limbs(x, 0, a, n)
    _load_limbs(y, 0, b, n)
    var result: Integer
    while True:
        var an = _used(a, n)
        var bn = _used(b, n)
        if not an or not bn:
            # The other operand is the gcd.
            var source = b if not an else a
            var used = bn if not an else an
            var owner = Integer._Shared.uninitialized(2 * used, False)
            var words = _store_words(source, used, 0, owner[].words.unsafe_ptr())
            result = Integer._from_product[False](owner^, words)
            break
        n = max(an, bn)
        if n <= 2:
            result = Integer._from_wide_magnitude128(
                _native_gcd(
                    UInt128(a.unsafe_offset(0)[]) | (UInt128(a.unsafe_offset(1)[]) << 64 if n > 1 else 0),
                    UInt128(b.unsafe_offset(0)[]) | (UInt128(b.unsafe_offset(1)[]) << 64 if n > 1 else 0),
                )
            )
            break
        # The leading 128 bits of both, at the larger's top bit.
        var shift = UInt64(count_leading_zeros(a.unsafe_offset(n - 1)[] | b.unsafe_offset(n - 1)[]))
        var ah = a.unsafe_offset(n - 1)[]
        var al = a.unsafe_offset(n - 2)[]
        var bh = b.unsafe_offset(n - 1)[]
        var bl = b.unsafe_offset(n - 2)[]
        if shift:
            ah = (ah << shift) | (al >> (64 - shift))
            al = (al << shift) | (a.unsafe_offset(n - 3)[] >> (64 - shift))
            bh = (bh << shift) | (bl >> (64 - shift))
            bl = (bl << shift) | (b.unsafe_offset(n - 3)[] >> (64 - shift))
        var m = _Matrix(0, 0, 0, 0)
        if _hgcd2(ah, al, bh, bl, m):
            _apply_inverse(m, a, b, t, n)
            swap(a, t)
        elif _compare(a, b, n) >= 0:
            # No safe step: a quotient past 64 bits, or nearly equal operands.
            _reduce(a, an, b, bn, work)
        else:
            _reduce(b, bn, a, an, work)
    # The scratch block lives until here.
    _ = stack^
    _ = heap^
    return result^
