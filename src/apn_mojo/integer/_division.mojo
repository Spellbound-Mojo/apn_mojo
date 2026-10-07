"""Private base-2^32 unsigned long division on owned, normalized magnitudes."""

from std.bit import count_leading_zeros
from ..common._sizes import _checked_count, _checked_sum
from ._words import _word_pair, _put_word_pair


def _divide_into(
    dividend: Span[mut=True, UInt32, _],
    divisor: Span[mut=True, UInt32, _],
    quotient: Span[mut=True, UInt32, _],
) -> Int:
    # Preflighted, disjoint buffers: dividend has one extra high word;
    # normalized input magnitudes satisfy dividend >= divisor > 0.
    var size = len(dividend) - 1
    var n = len(divisor)
    quotient.fill(0)
    if n == 1:
        var rest = UInt64(0)
        var denominator = UInt64(divisor[0])
        for i in range(size - 1, -1, -1):
            var combined = (rest << 32) | UInt64(dividend[i])
            quotient[i] = UInt32(combined // denominator)
            rest = combined % denominator
        dividend[0] = UInt32(rest)
        return Int(rest != 0)
    var shift = 0
    var top = divisor[n - 1]
    while top < UInt32(0x80000000):
        top <<= 1
        shift += 1
    var carry = UInt64(0)
    for i in range(n):
        var word = (UInt64(divisor[i]) << UInt64(shift)) | carry
        divisor[i] = UInt32(word)
        carry = word >> 32
    carry = 0
    for i in range(size):
        var word = (UInt64(dividend[i]) << UInt64(shift)) | carry
        dividend[i] = UInt32(word)
        carry = word >> 32
    dividend[size] = UInt32(carry)

    comptime radix = UInt64(1) << 32
    var high_divisor = UInt64(divisor[n - 1])
    for j in range(size - n, -1, -1):
        var high = UInt64(dividend[j + n])
        var next_word = UInt64(dividend[j + n - 1])
        # Cap the estimate at B-1 before dividing a potentially B-sized quotient.
        var estimate = radix - 1
        var residual = next_word + high_divisor
        if high != high_divisor:
            var numerator = (high << 32) | next_word
            estimate = numerator // high_divisor
            residual = numerator % high_divisor
        while residual < radix and estimate * UInt64(divisor[n - 2]) > (
            residual << 32
        ) + UInt64(dividend[j + n - 2]):
            estimate -= 1
            residual += high_divisor

        var borrow = UInt64(0)
        for i in range(n):
            # estimate, divisor[i], and incoming borrow are each at most B-1.
            var product = estimate * UInt64(divisor[i]) + borrow
            var low = product & (radix - 1)
            var original = UInt64(dividend[j + i])
            dividend[j + i] = UInt32(original - low)
            borrow = (product >> 32) + UInt64(original < low)
        var original_top = UInt64(dividend[j + n])
        dividend[j + n] = UInt32(original_top - borrow)
        if original_top < borrow:
            # The two-word estimate can still be one too large: add back once.
            estimate -= 1
            carry = 0
            for i in range(n):
                var total = UInt64(dividend[j + i]) + UInt64(divisor[i]) + carry
                dividend[j + i] = UInt32(total)
                carry = total >> 32
            dividend[j + n] = UInt32(UInt64(dividend[j + n]) + carry)
        quotient[j] = UInt32(estimate)

    for i in range(n):
        var word = UInt64(dividend[i]) >> UInt64(shift)
        if shift and i + 1 < n:
            word |= UInt64(dividend[i + 1]) << UInt64(32 - shift)
        dividend[i] = UInt32(word)
    var used = n
    while used and dividend[used - 1] == 0:
        used -= 1
    return used


@always_inline
def _pair_at(p: Pointer[mut=False, UInt32, _], n: Int, j: Int) -> UInt64:
    """64-bit pair j of n words (words 2j and 2j + 1), zero outside them."""
    if j < 0 or 2 * j >= n:
        return 0
    if 2 * j + 2 <= n:
        return _word_pair(p, 2 * j)
    return UInt64(p.unsafe_offset(2 * j)[])


def _load_limbs(
    words: Span[mut=False, UInt32, _], shift: Int, target: Pointer[mut=True, UInt64, _], limbs: Int,
):
    """Write words * 2**shift into `limbs` 64-bit limbs at target: one pair
    load per limb, and the previous pair's top bits for a partial shift."""
    var whole = shift >> 6
    var bits = UInt64(shift & 63)
    var p = words.unsafe_ptr()
    var n = len(words)
    var below = UInt64(0)
    for i in range(limbs):
        var limb = _pair_at(p, n, i - whole)
        target.unsafe_offset(i)[] = ((limb << bits) | (below >> (64 - bits))) if bits else limb
        below = limb


def _store_words(
    source: Pointer[mut=False, UInt64, _], limbs: Int, shift: Int, target: Pointer[mut=True, UInt32, _],
) -> Int:
    """Write the limbs shifted right by `shift` < 64 bits as 2 * limbs words at
    target; return the count without leading zero words."""
    for i in range(limbs):
        var limb = source.unsafe_offset(i)[]
        if shift:
            limb >>= UInt64(shift)
            if i + 1 < limbs:
                limb |= source.unsafe_offset(i + 1)[] << UInt64(64 - shift)
        _put_word_pair(target, 2 * i, limb)
    var used = 2 * limbs
    while used and target.unsafe_offset(used - 1)[] == 0:
        used -= 1
    return used


# The seed table of Moller and Granlund's Algorithm 2 (Improved division by
# invariant integers, IEEE Transactions on Computers 60(2), 2011): entry i is
# v0 = floor((2**19 - 3 * 2**8) / (256 + i)) = floor(0x7fd00 / (256 + i)),
# between 1024 and 2045, written as two characters from "0" (48):
# v0 = 1024 + 32 (c0 - 48) + (c1 - 48). A string literal is static data, read
# by two byte loads; an Array or SIMD constant is copied to the stack per call.
comptime _RECIPROCAL_SEEDS: StaticString = "OMOEO=O5NMNEN>N6MOMGM@M8M1LJLBL;L4KMKFK?K8K1JKJDJ=J7J0IIICI<I6I0HIHCH=H7H0GJGDG>G8G2FLFFFAF;F5EOEJEDE>E9E3DNDHDCD=D8D3CMCHCCC>C9C4BNBIBDB?B:B5B0ALAGABA=A8A4@O@J@F@A@<@8@3?O?J?F?A?=?9?4?0>L>G>C>?>;>6>2=N=J=F=B=>=:=6=2<N<J<F<B<><:<6<3;O;K;G;D;@;<;8;5;1:N:J:F:C:?:<:8:5:19N9J9G9D9@9=9:9693908L8I8F8C8?8<898683807L7I7F7C7@7=7:7774716N6K6H6E6B6?6<6:6764615N5K5I5F5C5@5=5;585553504M4K4H4E4C4@4=4;484643413N3L3I3G3D3B3?3=3:383533312N2L2I2G2E2B2@2>2;29272422201N1K1I1G1E1B1@1>1<1:181513110O0M0K0I0G0D0B0@0>0<0:0806040200"


@always_inline
def _reciprocal(d: UInt64) -> UInt64:
    """floor((2**128 - 1) / d) - 2**64 for d with its high bit set.

    Moller and Granlund, Improved division by invariant integers, IEEE
    Transactions on Computers 60(2), 2011, Algorithm 2: an 11-bit seed from
    the table, refined to
    about 22, 40 and 64 bits by multiplications, then one correction. A
    128-by-64 hardware division took about 21 ns a call here, this 10."""
    var i = 2 * (Int(d >> 55) - 256)
    var seeds = _RECIPROCAL_SEEDS.unsafe_ptr()
    var v0 = 1024 + ((UInt64(seeds.unsafe_offset(i)[]) - 48) << 5) + (UInt64(seeds.unsafe_offset(i + 1)[]) - 48)
    var d40 = (d >> 24) + 1
    var v1 = (v0 << 11) - ((v0 * v0 * d40) >> 40) - 1
    var v2 = (v1 << 13) + ((v1 * ((UInt64(1) << 60) - v1 * d40)) >> 47)
    var d0 = d & 1
    var e = ((v2 & (0 - d0)) >> 1) - v2 * ((d >> 1) + d0)
    var v3 = (v2 << 31) + (UInt64((UInt128(v2) * UInt128(e)) >> 64) >> 1)
    return v3 - UInt64((UInt128(v3) * UInt128(d) + UInt128(d)) >> 64) - d


@always_inline
def _divide_2by1(high: UInt64, low: UInt64, d: UInt64, v: UInt64) -> Tuple[UInt64, UInt64]:
    """(high * 2**64 + low) // d and the remainder, for normalized d > high
    and v = _reciprocal(d): two multiplications instead of a division
    (Moller and Granlund, Improved division by invariant integers, 2011)."""
    var estimate = UInt128(v) * UInt128(high) + ((UInt128(high + 1) << 64) | UInt128(low))
    var quotient = UInt64(estimate >> 64)
    var rest = low - quotient * d
    if rest > UInt64(estimate & UInt128(UInt64.MAX)):
        quotient -= 1
        rest += d
    if rest >= d:
        quotient += 1
        rest -= d
    return (quotient, rest)


@always_inline
def _divide_by_limb_into[quotient: Bool](
    words: Span[mut=False, UInt32, _], d: UInt64, target: Pointer[mut=True, UInt32, _],
) -> UInt64:
    """Divide a nonempty magnitude by a nonzero d of at most 64 bits and return
    the remainder; with `quotient`, write the quotient's len(words) words at
    target, which may be the words themselves. One pass from the top limb
    (words 2i and 2i + 1; the top one has a single word when the count is
    odd): each limb, shifted with d's normalization, takes one `_divide_2by1`,
    and is read before its quotient digit replaces it. The top limb goes
    first on its own, so the loop stores whole pairs without a test."""
    var n = len(words)
    var p = words.unsafe_ptr()
    var shift = UInt64(count_leading_zeros(d))
    var divisor = d << shift
    var inverse = _reciprocal(divisor)
    var i = (n - 1) // 2
    var limb = UInt64(p.unsafe_offset(n - 1)[]) if n & 1 else _word_pair(p, 2 * i)
    # The normalized dividend's extra top limb: the bits shifted out of the top one.
    var rest = (limb >> (64 - shift)) if shift else UInt64(0)
    var below = _word_pair(p, 2 * i - 2) if i else UInt64(0)
    var normalized = ((limb << shift) | (below >> (64 - shift))) if shift else limb
    var digit, remainder = _divide_2by1(rest, normalized, divisor, inverse)
    comptime if quotient:
        # A single top word's digit fits it: that limb is below 2**32.
        if n & 1:
            target.unsafe_offset(2 * i)[] = UInt32(digit)
        else:
            _put_word_pair(target, 2 * i, digit)
    rest = remainder
    while i:
        i -= 1
        limb = below
        below = _word_pair(p, 2 * i - 2) if i else UInt64(0)
        normalized = ((limb << shift) | (below >> (64 - shift))) if shift else limb
        digit, remainder = _divide_2by1(rest, normalized, divisor, inverse)
        comptime if quotient:
            _put_word_pair(target, 2 * i, digit)
        rest = remainder
    return rest >> shift


def _divide_limbs(
    u: Pointer[mut=True, UInt64, _], size: Int, d: Pointer[mut=False, UInt64, _], n: Int,
    q: Pointer[mut=True, UInt64, _],
):
    """Knuth's long division on 64-bit limbs: divide u[0, size) by d[0, n),
    whose top limb has its high bit set, with u[size - 1] = 0. Writes the
    size - n quotient limbs to q and leaves the remainder in u[0, n)."""
    var d1 = d.unsafe_offset(n - 1)[]
    var v = _reciprocal(d1)
    if n == 1:
        var rest = UInt64(0)
        for j in range(size - 2, -1, -1):
            var digit, next_rest = _divide_2by1(rest, u.unsafe_offset(j)[], d1, v)
            q.unsafe_offset(j)[] = digit
            rest = next_rest
        u.unsafe_offset(0)[] = rest
        return
    var d0 = d.unsafe_offset(n - 2)[]
    for j in range(size - n - 1, -1, -1):
        var u2 = u.unsafe_offset(j + n)[]
        var u1 = u.unsafe_offset(j + n - 1)[]
        # A two-limb estimate of the digit, corrected by the third limb: at
        # most one too large afterwards (Knuth, TAOCP 4.3.1, Algorithm D).
        var estimate: UInt128
        var rest: UInt128
        if u2 >= d1:
            estimate = UInt128(UInt64.MAX)
            rest = ((UInt128(u2) << 64) | UInt128(u1)) - estimate * UInt128(d1)
        else:
            var digit, remainder = _divide_2by1(u2, u1, d1, v)
            estimate = UInt128(digit)
            rest = UInt128(remainder)
        while rest >> 64 == 0 and estimate * UInt128(d0) > ((rest << 64) | UInt128(u.unsafe_offset(j + n - 2)[])):
            estimate -= 1
            rest += UInt128(d1)
        var digit = UInt64(estimate)
        # The carry fits one limb: digit * d[i] + carry < 2**128 - 2**64.
        var carry = UInt64(0)
        for i in range(n):
            var product = UInt128(digit) * UInt128(d.unsafe_offset(i)[]) + UInt128(carry)
            var low = UInt64(product)
            var word = u.unsafe_offset(j + i)[]
            u.unsafe_offset(j + i)[] = word - low
            carry = UInt64(product >> 64) + UInt64(word < low)
        var last = u.unsafe_offset(j + n)[]
        u.unsafe_offset(j + n)[] = last - carry
        if last < carry:
            # The estimate was one too large: add the divisor back.
            digit -= 1
            var sum_carry = UInt128(0)
            for i in range(n):
                var total = UInt128(u.unsafe_offset(j + i)[]) + UInt128(d.unsafe_offset(i)[]) + sum_carry
                u.unsafe_offset(j + i)[] = UInt64(total & UInt128(UInt64.MAX))
                sum_carry = total >> 64
            u.unsafe_offset(j + n)[] = u.unsafe_offset(j + n)[] + UInt64(sum_carry)
        q.unsafe_offset(j)[] = digit
