"""Magnitude word kernels: 32-bit words, read and written two at a time.

Magnitudes are stored as 32-bit words, but these loops do 64-bit work: one
64-bit load or store covers two adjacent words (alignment 4, so any word
offset is valid), and carries travel through 64-bit additions. Measured, this
runs as fast as the same loops over 64-bit limbs, with half the iterations of
word-at-a-time loops. An odd word count leaves one word to a 32-bit step.
"""

from std.memory import bitcast


@always_inline
def _word_pair(p: Pointer[mut=False, UInt32, _], i: Int) -> UInt64:
    """Words i and i + 1 as one 64-bit value, word i low."""
    return bitcast[DType.uint64, 1](p.unsafe_offset(i).unsafe_load[width=2, alignment=4]())


@always_inline
def _put_word_pair(p: Pointer[mut=True, UInt32, _], i: Int, value: UInt64):
    """Store value's low half at word i and its high half at word i + 1."""
    p.unsafe_offset(i).unsafe_store[alignment=4](bitcast[DType.uint32, 2](value))


@always_inline
def _add_words(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    target: Pointer[mut=True, UInt32, _],
):
    """Write |a| + |b| into max(len) + 1 words at target."""
    var shorter = min(len(a), len(b))
    var length = max(len(a), len(b))
    var x = a.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var y = b.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var rest = x if len(a) > len(b) else y
    var carry = UInt64(0)
    var i = 0
    while i + 2 <= shorter:
        var total = UInt128(_word_pair(x, i)) + UInt128(_word_pair(y, i)) + UInt128(carry)
        _put_word_pair(target, i, UInt64(total))
        carry = UInt64(total >> 64)
        i += 2
    if i < shorter:
        var total = UInt64(x.unsafe_offset(i)[]) + UInt64(y.unsafe_offset(i)[]) + carry
        target.unsafe_offset(i)[] = UInt32(total)
        carry = total >> 32
        i += 1
    while i + 2 <= length:
        var total = UInt128(_word_pair(rest, i)) + UInt128(carry)
        _put_word_pair(target, i, UInt64(total))
        carry = UInt64(total >> 64)
        i += 2
    if i < length:
        var total = UInt64(rest.unsafe_offset(i)[]) + carry
        target.unsafe_offset(i)[] = UInt32(total)
        carry = total >> 32
    target.unsafe_offset(length)[] = UInt32(carry)


@always_inline
def _subtract_words(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    target: Pointer[mut=True, UInt32, _],
):
    """Write |a| - |b| into len(a) words at target; requires |a| >= |b|."""
    var x = a.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var y = b.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin]()
    var shorter = min(len(a), len(b))
    var borrow = UInt64(0)
    var i = 0
    while i + 2 <= shorter:
        var total = UInt128(_word_pair(x, i)) - UInt128(_word_pair(y, i)) - UInt128(borrow)
        _put_word_pair(target, i, UInt64(total))
        borrow = UInt64(total >> 127)
        i += 2
    if i < shorter:
        var p = UInt64(x.unsafe_offset(i)[])
        var q = UInt64(y.unsafe_offset(i)[]) + borrow
        target.unsafe_offset(i)[] = UInt32(p - q)
        borrow = UInt64(p < q)
        i += 1
    while i + 2 <= len(a):
        var total = UInt128(_word_pair(x, i)) - UInt128(borrow)
        _put_word_pair(target, i, UInt64(total))
        borrow = UInt64(total >> 127)
        i += 2
    if i < len(a):
        var p = UInt64(x.unsafe_offset(i)[])
        target.unsafe_offset(i)[] = UInt32(p - borrow)


@always_inline
def _shift_left_words(
    source: Span[mut=False, UInt32, _], whole: Int, part: Int,
    target: Pointer[mut=True, UInt32, _],
):
    """Write source * 2**(32 whole + part), for part < 32, into words whole to
    len(source) + whole (inclusive) at target; the words below are the caller's.
    Each pair takes its low bits from the pair below: a 64-bit funnel shift."""
    var s = source.unsafe_ptr()
    var n = len(source)
    var shift = UInt64(part)
    var carry = UInt64(0)
    var i = 0
    while i + 2 <= n:
        var pair = _word_pair(s, i)
        _put_word_pair(target, i + whole, (pair << shift) | carry)
        carry = (pair >> (64 - shift)) if part else UInt64(0)
        i += 2
    if i < n:
        var value = (UInt64(s.unsafe_offset(i)[]) << shift) | carry
        target.unsafe_offset(i + whole)[] = UInt32(value)
        carry = value >> 32
    target.unsafe_offset(n + whole)[] = UInt32(carry)


@always_inline
def _shift_right_words(
    source: Span[mut=False, UInt32, _], whole: Int, part: Int, carry_in: UInt64,
    target: Pointer[mut=True, UInt32, _],
) -> UInt64:
    """Write floor(source / 2**(32 whole + part)) + carry_in, for part < 32,
    into len(source) - whole words at target; return the carry out. Each pair
    takes its high bits from the word above it (zero above the top)."""
    var s = source.unsafe_ptr()
    var n = len(source) - whole
    var shift = UInt64(part)
    var carry = carry_in
    var i = 0
    while i + 2 <= n:
        var bits = _word_pair(s, i + whole) >> shift
        if part and i + whole + 2 < len(source):
            bits |= UInt64(s.unsafe_offset(i + whole + 2)[]) << (64 - shift)
        var total = bits + carry
        carry = UInt64(total < carry)
        _put_word_pair(target, i, total)
        i += 2
    if i < n:
        var above = UInt64(s.unsafe_offset(i + whole + 1)[]) if i + whole + 1 < len(source) else UInt64(0)
        var digit = UInt32((UInt64(s.unsafe_offset(i + whole)[]) >> shift) | (above << (32 - shift)))
        var total = UInt64(digit) + carry
        target.unsafe_offset(i)[] = UInt32(total)
        carry = total >> 32
    return carry


@always_inline
def _shifted_pair(p: Pointer[mut=False, UInt32, _], n: Int, j: Int, shift: UInt64) -> UInt64:
    """Words j and j + 1 of the n words at p, shifted right by shift < 32 bits
    with word j + 2's low bits above them; zero past the words."""
    var low = _word_pair(p, j) if j + 2 <= n else (UInt64(p.unsafe_offset(j)[]) if j < n else UInt64(0))
    var next = UInt64(p.unsafe_offset(j + 2)[]) if j + 2 < n else UInt64(0)
    return (low >> shift) | ((next << 1) << (63 - shift))


@always_inline
def _add_shifted_words[subtract: Bool](
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _], whole: Int, part: Int,
    borrow: UInt64, target: Pointer[mut=True, UInt32, _],
) -> UInt64:
    """Write |a| + floor(|b| / 2**(32 whole + part)), or with `subtract` |a|
    less that and the borrow, for part < 32, into len(a) words at target; the
    shifted b has at most len(a) words. Return the carry or borrow out. One
    pass, b shifted on the way: pairs over three of
    b's words, then the pairs over b's last words read them checked, then
    a's upper pairs take the carry alone."""
    var x = a.unsafe_ptr()
    var y = b.unsafe_ptr()
    var n = len(a)
    var m = len(b)
    var shift = UInt64(part)
    var carry = borrow
    var i = 0
    while i + 2 <= n and i + whole < m:
        var s: UInt64
        if i + whole + 3 <= m:
            # (y << 1) << (63 - shift) is y << (64 - shift), and zero for shift 0.
            s = (_word_pair(y, i + whole) >> shift) | ((UInt64(y.unsafe_offset(i + whole + 2)[]) << 1) << (63 - shift))
        else:
            s = _shifted_pair(y, m, i + whole, shift)
        comptime if subtract:
            var total = UInt128(_word_pair(x, i)) - UInt128(s) - UInt128(carry)
            _put_word_pair(target, i, UInt64(total))
            carry = UInt64(total >> 127)
        else:
            var total = UInt128(_word_pair(x, i)) + UInt128(s) + UInt128(carry)
            _put_word_pair(target, i, UInt64(total))
            carry = UInt64(total >> 64)
        i += 2
    while i + 2 <= n:
        comptime if subtract:
            var total = UInt128(_word_pair(x, i)) - UInt128(carry)
            _put_word_pair(target, i, UInt64(total))
            carry = UInt64(total >> 127)
        else:
            var total = UInt128(_word_pair(x, i)) + UInt128(carry)
            _put_word_pair(target, i, UInt64(total))
            carry = UInt64(total >> 64)
        i += 2
    if i < n:
        var s = _shifted_pair(y, m, i + whole, shift) & 0xFFFFFFFF
        var word = UInt64(x.unsafe_offset(i)[])
        comptime if subtract:
            var q = s + carry
            target.unsafe_offset(i)[] = UInt32((word - q) & 0xFFFFFFFF)
            carry = UInt64(word < q)
        else:
            var total = word + s + carry
            target.unsafe_offset(i)[] = UInt32(total & 0xFFFFFFFF)
            carry = total >> 32
    return carry


@always_inline
def _compare_words(a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _]) -> Int:
    """The order of two magnitudes without leading zero words: -1, 0 or 1."""
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    var x = a.unsafe_ptr()
    var y = b.unsafe_ptr()
    var i = len(a)
    if i % 2:
        i -= 1
        var p = x.unsafe_offset(i)[]
        var q = y.unsafe_offset(i)[]
        if p != q:
            return -1 if p < q else 1
    while i >= 2:
        i -= 2
        var p = _word_pair(x, i)
        var q = _word_pair(y, i)
        if p != q:
            return -1 if p < q else 1
    return 0


@always_inline
def _extended_pair(p: Pointer[mut=False, UInt32, _], length: Int, i: Int) -> UInt64:
    """Words i and i + 1 of a magnitude of `length` words, zero past its end."""
    if i + 2 <= length:
        return _word_pair(p, i)
    if i < length:
        return UInt64(p.unsafe_offset(i)[])
    return 0


@always_inline
def _bitwise_words(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    a_negative: Bool, b_negative: Bool, operation: Int,
    target: Pointer[mut=True, UInt32, _], count: Int,
) -> Bool:
    """Write the magnitude of `a op b` into `count` words at target and return
    its sign, for signed values with magnitudes a and b: operation 0 is and, 1
    or, 2 xor, 3 the complement of a. Each operand is read in two's complement,
    -m as ~m + 1 extended with ones past its words, and the result converted
    back the same way. `count` must exceed both lengths: the top word holds the
    sign extension, so every magnitude fits."""
    var negative = (
        (a_negative and b_negative) if operation == 0
        else (a_negative or b_negative) if operation == 1
        else (a_negative != b_negative) if operation == 2
        else not a_negative
    )
    var x_words = a.unsafe_ptr()
    var y_words = b.unsafe_ptr()
    var a_carry = UInt64(a_negative)
    var b_carry = UInt64(b_negative)
    var carry = UInt64(negative)
    var i = 0
    while i < count:
        var x = _extended_pair(x_words, len(a), i)
        var y = _extended_pair(y_words, len(b), i)
        if a_negative:
            x = ~x + a_carry
            a_carry = UInt64(x < a_carry)
        if b_negative:
            y = ~y + b_carry
            b_carry = UInt64(y < b_carry)
        var digit = (
            x & y if operation == 0
            else x | y if operation == 1
            else x ^ y if operation == 2
            else ~x
        )
        var total = (~digit if negative else digit) + carry
        carry = UInt64(total < carry)
        # An odd count ends with one word: the pair's high half is extension.
        if i + 2 <= count:
            _put_word_pair(target, i, total)
        else:
            target.unsafe_offset(i)[] = UInt32(total)
        i += 2
    return negative
