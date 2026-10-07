"""Floor square roots with remainders: native up to 256 bits, beyond that
Zimmermann's Karatsuba square root on 64-bit limbs."""

from std.bit import count_leading_zeros
from std.math import sqrt
from .value import Integer
from ._division import _load_limbs, _store_words, _divide_limbs
from ._multiplication import _multiply_with_scratch, _product_scratch_size
from ._limbs import _add_limbs, _subtract_limbs, _add_one, _subtract_one, _add_mul_limbs
from ..common._sizes import _checked_count


def _isqrt_native(n: UInt128) -> UInt128:
    """floor(sqrt(n)): a Float64 estimate, one Newton step, then a final
    correction of at most a few units."""
    if not n:
        return 0
    comptime top = UInt128(UInt64.MAX)
    var estimate = sqrt(n.cast[DType.float64]())
    var root = top if estimate >= 18446744073709551615.0 else UInt128(estimate.cast[DType.uint64]())
    if root:
        root = min((root + n // root) >> 1, top)
    while root * root > n:
        root -= 1
    while root < top and (root + 1) * (root + 1) <= n:
        root += 1
    return root


@always_inline
def _limb_square_root(u: UInt64) -> Tuple[UInt64, Bool]:
    """floor(sqrt(u * 2**64)) and whether the root is inexact, for
    2**62 <= u < 2**64 (the root has 64 bits).

    The hardware Float64 root of u, scaled by 2**32, is within about 3 * 2**11
    units: u and its root each round by at most 2**-53, relatively. One Newton
    step adds (N - s**2) / (2 s), whose Float64 value is within far less than
    a unit of the remaining error (the step's own error, (s* - s)**2 / 2s,
    is below 2**-40); its reciprocal factor does not wait for the remainder.
    A last exact correction moves the root by at most one unit."""
    var radicand = UInt128(u) << 64
    var estimate = sqrt(u.cast[DType.float64]()) * 4294967296.0
    var half_inverse = 0.5 / estimate
    var root = UInt64.MAX if estimate >= 18446744073709551615.0 else estimate.cast[DType.uint64]()
    # The signed remainder, below 2**77 in magnitude; its low 32 bits do not
    # matter to the step.
    var rest = (radicand - UInt128(root) * UInt128(root)).cast[DType.int128]()
    var step = (rest >> 32).cast[DType.int64]().cast[DType.float64]() * 4294967296.0 * half_inverse
    var corrected = Int128(root) + Int128(Int(step + 0.5) if step >= 0 else -Int(0.5 - step))
    var top = Int128(UInt64.MAX)
    root = max(Int128(1) << 63, min(corrected, top)).cast[DType.uint64]()
    rest = (radicand - UInt128(root) * UInt128(root)).cast[DType.int128]()
    if rest < 0:
        root -= 1
        rest += 2 * Int128(root) + 1
    elif rest > 2 * Int128(root):
        rest -= 2 * Int128(root) + 1
        root += 1
    return (root, rest != 0)


def _wide_magnitude(a: Integer) -> UInt256:
    """The magnitude of an Integer of at most 256 bits, natively."""
    if a._storage.isa[Int64]():
        return UInt256(UInt64(abs(a._storage[Int64])))
    # Four limbs from the words at once, not a storage dispatch per word.
    var small = a._inline_words()
    var limbs = Array[UInt64, 4](fill=0)
    var target = Span(limbs).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _load_limbs(a._words_span(small), 0, target, 4)
    var n = UInt256(target.unsafe_offset(0)[])
    for i in range(1, 4):
        n |= UInt256(target.unsafe_offset(i)[]) << UInt256(64 * i)
    _ = limbs^
    return n


def _sqrt_rem_native(n: UInt256) -> Tuple[UInt128, UInt256]:
    """`(s, n - s**2)` for `s = floor(sqrt(n))`, in native integers.

    Up to 128 bits, `_isqrt_native`. Beyond, one step of the Karatsuba square
    root (see `_sqrt_rem`) with b = 2**k and k = floor((bits + 1) / 4), from
    32 to 64: the top part n >> 2k has at most 128 bits, so its root s' is
    below 2**64 and r' <= 2 s'. Then r' b + a1 < 2**129, and its quotient by
    2 s' is that of (r' b + a1) >> 1 by s', which fits 128 bits; the dropped
    low bit returns in the remainder. q <= b keeps q**2 within 2**128.
    """
    if not n >> 128:
        var small = UInt128(n)
        var root = _isqrt_native(small)
        return (root, UInt256(small - root * root))
    var bits = 256 - Int(count_leading_zeros(n))
    var k = UInt256((bits + 1) // 4)
    var mask = (UInt256(1) << k) - 1
    var top = UInt128(n >> (k + k))
    var high = _isqrt_native(top)
    var numerator = (UInt256(top - high * high) << k) | ((n >> k) & mask)
    var half = UInt128(numerator >> 1)
    var q = half // high
    var u = ((half - q * high) << 1) | UInt128(numerator & 1)
    var root = (UInt256(high) << k) + UInt256(q)
    var rest = (UInt256(u) << k) + (n & mask)
    var square = UInt256(q) * UInt256(q)
    if rest >= square:
        return (UInt128(root), rest - square)
    return (UInt128(root - 1), rest + 2 * root - 1 - square)


comptime _STACK_LIMBS = 160
"""Scratch limbs kept on the stack: roots of radicands up to about 5800 bits."""


def _dc_sqrt_rem(
    s: Pointer[mut=True, UInt64, _], x: Pointer[mut=True, UInt64, _], n: Int,
    q: Pointer[mut=True, UInt64, _],
) raises -> UInt64:
    """The floor root of x[0, 2n) into s[0, n), the remainder's low n limbs
    into x[0, n); returns its high limb, 0 or 1 (the remainder is at most 2s).
    Needs x[2n - 1] >= 2**62, and q of n // 2 + 1 limbs for quotients.

    As `_sqrt_rem` describes (Zimmermann, Karatsuba Square Root, INRIA
    Research Report 3805, 1999), on limbs and in place: with l = n // 2 and
    h = n - l, the top 2h limbs give
    (s', r') into s[l, n) and x[2l, n + l), so x[l, n + l) holds r' B**l + a1.
    s' >= B**h / 2 is a normalized divisor, so the quotient by 2 s' comes from
    the quotient Q by s', halved: the odd bit adds s' to the remainder. q**2
    goes to x[n, n + 2l), which the division has freed.
    """
    if n == 1:
        var value = (UInt128(x.unsafe_offset(1)[]) << 64) | UInt128(x.unsafe_offset(0)[])
        var root = _isqrt_native(value)
        var rest = value - root * root
        s.unsafe_offset(0)[] = UInt64(root)
        x.unsafe_offset(0)[] = UInt64(rest)
        return UInt64(rest >> 64)
    if n == 2:
        var value = UInt256(0)
        for i in range(4):
            value |= UInt256(x.unsafe_offset(i)[]) << UInt256(64 * i)
        var native = _sqrt_rem_native(value)
        s.unsafe_offset(0)[] = UInt64(native[0])
        s.unsafe_offset(1)[] = UInt64(native[0] >> 64)
        x.unsafe_offset(0)[] = UInt64(native[1])
        x.unsafe_offset(1)[] = UInt64(native[1] >> 64)
        return UInt64(native[1] >> 128)
    var l = n // 2
    var h = n - l
    var high = _dc_sqrt_rem(s.unsafe_offset(l), x.unsafe_offset(2 * l), h, q)
    if high:
        # r' >= B**h: divide (r' - s') B**l + a1 instead, and count B**l more.
        _ = _subtract_limbs(x.unsafe_offset(2 * l), s.unsafe_offset(l), h)
    x.unsafe_offset(l + n)[] = 0
    _divide_limbs(x.unsafe_offset(l), n + 1, s.unsafe_offset(l), h, q)
    # Q = q[0, l) + top B**l <= 2 B**l; the low part of the root is Q >> 1,
    # and Q >> 1 = B**l carries into s'.
    var top = q.unsafe_offset(l)[] + high
    var odd = q.unsafe_offset(0)[] & 1
    for i in range(l):
        var above = q.unsafe_offset(i + 1)[] if i + 1 < l else top
        s.unsafe_offset(i)[] = (q.unsafe_offset(i)[] >> 1) | (above << 63)
    var carry = top >> 1
    var rest_high = 0
    if odd:
        rest_high = Int(_add_limbs(x.unsafe_offset(l), s.unsafe_offset(l), h))
    # The remainder u B**l + a0 - q**2 sits in x[0, n) and rest_high.
    var borrow = UInt64(1)
    if not carry:
        # x[n, n + 2l) is disjoint from x[0, 2l), since 2l <= n.
        var square = x.unsafe_offset(n).unsafe_origin_cast[MutUntrackedOrigin]()
        _ = _multiply_with_scratch(
            Span(unsafe_ptr=s.unsafe_bitcast[UInt32](), length=2 * l),
            Span(unsafe_ptr=s.unsafe_bitcast[UInt32](), length=2 * l),
            Span(unsafe_ptr=square.unsafe_bitcast[UInt32](), length=4 * l),
            True,
            _product_scratch_size(2 * l, 2 * l, True),
        )
        borrow = _subtract_limbs(x, square, 2 * l)
    if h > l:
        borrow = _subtract_one(x.unsafe_offset(2 * l), h - l, borrow)
    rest_high -= Int(borrow)
    var overflow = UInt64(0)
    if carry:
        overflow = _add_one(s.unsafe_offset(l), h, carry)
    if rest_high < 0:
        # One correction: r + 2 s - 1 with s - 1. When s' is all ones and
        # q = B**l, s = B**n overflows its n limbs: it counts 2 B**n in r, and
        # s - 1 is back within them.
        rest_high += Int(_add_limbs(x, s, n))
        rest_high += Int(_add_limbs(x, s, n))
        rest_high += 2 * Int(overflow)
        rest_high -= Int(_subtract_one(x, n, 1))
        _ = _subtract_one(s, n, 1)
    return UInt64(rest_high)


def _sqrt_rem(a: Integer, shift: Int = 0) raises -> Tuple[Integer, Integer]:
    """`(s, a 2**shift - s**2)` for `s = floor(sqrt(a 2**shift))`, nonnegative
    `a` and `shift`: the shift happens as the words load, not in an Integer.

    Zimmermann's Karatsuba square root (Brent and Zimmermann, Modern Computer
    Arithmetic, Algorithm 1.12). Write a = (a3 b + a2) b**2 + a1 b + a0 with
    a3 >= b / 4. With (s', r') the root and remainder of a3 b + a2, and (q, u)
    the quotient and remainder of (r' b + a1) / (2 s'), s = s' b + q has the
    remainder u b + a0 - q**2. Since s' >= b / 2, q <= b and q**2 <= 2 s - 1,
    so a negative remainder needs one correction: s - 1, with r + 2 s - 1.

    Up to 256 bits natively; beyond, on 64-bit limbs in one scratch block
    (`_dc_sqrt_rem`), after shifting a left by 2k bits so that its top limb
    is at least 2**62. With 4**k a = S**2 + R and s0 = S mod 2**k, the root is
    S >> k and the remainder (R + 2 S s0 - s0**2) / 4**k.
    """
    if not a:
        return (Integer(0), Integer(0))
    var bits = a.magnitude_bit_length() + shift
    if bits <= 256:
        var native = _sqrt_rem_native(_wide_magnitude(a) << UInt256(shift))
        return (
            Integer._from_wide_magnitude128(native[0]),
            Integer._from_wide_magnitude256(native[1]),
        )
    var n = (bits + 127) // 128
    var k = (128 * n - bits) // 2
    # x: 2n limbs, then s: n limbs, then q: n // 2 + 1 limbs.
    var limbs = 3 * _checked_count(n, 32) + n // 2 + 1
    var stack = Array[UInt64, _STACK_LIMBS](uninitialized=True)
    var heap = List[UInt64]()
    var scratch: Pointer[UInt64, MutUntrackedOrigin]
    if limbs <= _STACK_LIMBS:
        scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        heap = List[UInt64](unsafe_uninit_length=limbs)
        scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var x = scratch
    var s = scratch.unsafe_offset(2 * n)
    var q = scratch.unsafe_offset(3 * n)
    var small = a._inline_words()
    _load_limbs(a._words_span(small), shift + 2 * k, x, 2 * n)
    var high = UInt128(_dc_sqrt_rem(s, x, n, q))
    if k:
        var s0 = s.unsafe_offset(0)[] & ((UInt64(1) << UInt64(k)) - 1)
        high += UInt128(_add_mul_limbs(x, s, n, 2 * s0))
        var square = UInt128(s0) * UInt128(s0)
        var pair = Array[UInt64, 2](fill=0)
        pair[0] = UInt64(square)
        pair[1] = UInt64(square >> 64)
        var borrow = _subtract_limbs(x, Span(pair).unsafe_ptr(), 2)
        high -= UInt128(_subtract_one(x.unsafe_offset(2), n - 2, borrow))
    x.unsafe_offset(n)[] = UInt64(high)
    var root_owner = Integer._Shared.uninitialized(2 * n, False)
    var root = Integer._from_product[False](root_owner^, _store_words(s, n, k, root_owner[].words.unsafe_ptr()))
    var whole = (2 * k) // 64
    var rest_owner = Integer._Shared.uninitialized(2 * (n + 1 - whole), False)
    var rest_used = _store_words(x.unsafe_offset(whole), n + 1 - whole, 2 * k - 64 * whole, rest_owner[].words.unsafe_ptr())
    var rest = Integer._from_product[False](rest_owner^, rest_used)
    _ = heap^
    _ = stack^
    return (root^, rest^)
