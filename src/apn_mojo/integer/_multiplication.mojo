"""Magnitude products over preflighted buffers; no allocation in limb kernels."""

from ..common._sizes import _checked_count, _checked_sum
from std.memory import MaybeUninit

comptime _KARATSUBA_WORDS = 96
comptime _KARATSUBA_SQUARE_WORDS = 512


def _product_scratch_size(left: Int, right: Int, same_operand: Bool = False) raises -> Int:
    var shorter = min(left, right)
    var threshold = _KARATSUBA_SQUARE_WORDS if same_operand else _KARATSUBA_WORDS
    if shorter < threshold:
        return 0
    var longer = max(left, right)
    var size = 0
    if longer // 2 >= shorter:
        size = _checked_count(shorter, 8) * 2
        longer = shorter
    # A frame owns four half-width slices and one carry word. All children
    # reuse its tail; the equal-width middle product bounds both other children.
    while longer >= threshold:
        var half = longer // 2 + longer % 2
        size = _checked_sum(size, _checked_sum(_checked_count(half, 16) * 4, 1))
        longer = half
    return _checked_count(size, 4)


def _multiply_with_scratch(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    result: Span[mut=True, UInt32, _], same_operand: Bool, scratch_size: Int,
) -> Int:
    if not scratch_size:
        return _multiply_into(a, b, result, same_operand)
    return _multiply_using_workspace(a, b, result, same_operand, scratch_size)


@no_inline
def _multiply_using_workspace(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    result: Span[mut=True, UInt32, _], same_operand: Bool, scratch_size: Int,
) -> Int:
    # Keep the recursive stack buffer out of calls that need no workspace.
    if scratch_size <= 512:
        var stack = MaybeUninit[Array[UInt32, 512]]()
        var work = Span(unsafe_ptr=stack.unsafe_ptr().unsafe_bitcast[UInt32](), length=scratch_size)
        return _multiply_into(a, b, result, same_operand, work)
    var work = List[UInt32](unsafe_uninit_length=scratch_size)
    return _multiply_into(a, b, result, same_operand, Span(work))


def _product_used(a: Span[mut=False, UInt32, _]) -> Int:
    var n = len(a)
    while n and a.unsafe_get(n - 1) == 0:
        n -= 1
    return n


@always_inline
def _product_slice[mutable: Bool, //, origin: Origin[mut=mutable]](
    words: Span[UInt32, origin], start: Int, end: Int,
) -> Span[UInt32, origin]:
    # Only internal, preflighted layouts call this unchecked slice constructor.
    return Span(unsafe_ptr=words.unsafe_ptr().unsafe_offset(start), length=end - start)


@always_inline
def _load_pair(a: Span[mut=False, UInt32, _], i: Int) -> UInt64:
    return UInt64(a.unsafe_get(i)) | (UInt64(a.unsafe_get(i + 1)) << 32)


@always_inline
def _put_pair(r: Span[mut=True, UInt32, _], i: Int, x: UInt64):
    r.unsafe_get(i) = UInt32(x)
    r.unsafe_get(i + 1) = UInt32(x >> 32)


@always_inline
def _load_product_digit(a: Span[mut=False, UInt32, _], i: Int) -> UInt64:
    return UInt64(a.unsafe_get(i)) | (
        (UInt64(a.unsafe_get(i + 1)) << 32) if i + 1 < len(a) else UInt64(0)
    )


def _high_product_into(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    omitted: Int, r: Span[mut=True, UInt32, _],
) -> Int:
    # B=2^64. H contains only diagonals i+j >= omitted. For n,m digits,
    # H*B^omitted <= a*b < (H + min(n,m)*B)*B^omitted when omitted>0.
    # Each omitted diagonal has at most min(n,m) products, each < (B-1)^2;
    # summing the geometric tail gives a bound < min(n,m)*(B-1).
    var n = (len(a) + 1) // 2
    var m = (len(b) + 1) // 2
    r.fill(0)
    for j in range(m):
        var factor = _load_product_digit(b, 2 * j)
        var start = max(0, omitted - j)
        if start < n:
            _product_row[True](
                _product_slice(a, 2 * start, len(a)), factor,
                r, 2 * (start + j - omitted),
            )
    return _product_used(r)


@always_inline
def _product_step[add: Bool](
    product: UInt128, carry: UInt64, r: Span[mut=True, UInt32, _], i: Int,
) -> UInt64:
    var previous = _load_pair(r, i) if add else UInt64(0)
    var total = product + UInt128(previous) + UInt128(carry)
    _put_pair(r, i, UInt64(total))
    return UInt64(total >> 64)


def _product_row[add: Bool, short: Bool = False](
    a: Span[mut=False, UInt32, _], word: UInt64,
    r: Span[mut=True, UInt32, _], offset: Int,
):
    var paired = len(a) & -2
    var factor = UInt128(word)
    var carry = UInt64(0)
    var i = 0
    # Load independent products before stores; no input/output alias is allowed.
    while i + 8 <= paired:
        var p0 = UInt128(_load_pair(a, i)) * factor
        var p1 = UInt128(_load_pair(a, i + 2)) * factor
        var p2 = UInt128(_load_pair(a, i + 4)) * factor
        var p3 = UInt128(_load_pair(a, i + 6)) * factor
        carry = _product_step[add](p0, carry, r, offset + i)
        carry = _product_step[add](p1, carry, r, offset + i + 2)
        carry = _product_step[add](p2, carry, r, offset + i + 4)
        carry = _product_step[add](p3, carry, r, offset + i + 6)
        i += 8
    while i < paired:
        carry = _product_step[add](UInt128(_load_pair(a, i)) * factor, carry, r, offset + i)
        i += 2
    if paired != len(a):
        # Only the low word belongs to the preceding partial product.
        var previous = r.unsafe_get(offset + paired) if add else UInt32(0)
        var total = UInt128(a.unsafe_get(paired)) * factor + UInt128(previous) + UInt128(carry)
        _put_pair(r, offset + paired, UInt64(total))
        comptime if not short:
            r.unsafe_get(offset + paired + 2) = UInt32(total >> 64)
    else:
        comptime if short:
            r.unsafe_get(offset + paired) = UInt32(carry)
        else:
            _put_pair(r, offset + paired, carry)


@always_inline
def _multiply_short_into(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    result: Span[mut=True, UInt32, _],
) -> Int:
    # Nonzero normalized magnitudes; b has at most two words. The caller
    # preflights len(a)+len(b) disjoint output words. Only the top can be zero.
    var factor = UInt64(b.unsafe_get(0))
    if len(b) == 1:
        _product_row[False, True](a, factor, result, 0)
    else:
        factor |= UInt64(b.unsafe_get(1)) << 32
        _product_row[False](a, factor, result, 0)
    var count = len(a) + len(b)
    return count - Int(result.unsafe_get(count - 1) == 0)


@always_inline
def _product_two_step[add: Bool](
    product: UInt128, pending: UInt128, carry0: UInt64, carry1: UInt64,
    r: Span[mut=True, UInt32, _], offset: Int,
) -> Tuple[UInt64, UInt64]:
    var previous = _load_pair(r, offset) if add else UInt64(0)
    var first = product + UInt128(previous) + UInt128(carry0)
    var second = pending + UInt128(UInt64(first)) + UInt128(carry1)
    _put_pair(r, offset, UInt64(second))
    return UInt64(first >> 64), UInt64(second >> 64)


def _product_two_rows[add: Bool](
    a: Span[mut=False, UInt32, _], b0: UInt64, b1: UInt64,
    r: Span[mut=True, UInt32, _], offset: Int,
):
    var carry0 = UInt64(0)
    var carry1 = UInt64(0)
    var pending = UInt128(0)
    var i = 0
    while i + 4 <= len(a):
        var word0 = UInt128(_load_pair(a, i))
        var word1 = UInt128(_load_pair(a, i + 2))
        var p00 = word0 * UInt128(b0)
        var p01 = word0 * UInt128(b1)
        var p10 = word1 * UInt128(b0)
        var p11 = word1 * UInt128(b1)
        carry0, carry1 = _product_two_step[add](p00, pending, carry0, carry1, r, offset + i)
        carry0, carry1 = _product_two_step[add](p10, p01, carry0, carry1, r, offset + i + 2)
        pending = p11
        i += 4
    while i < len(a):
        var word = UInt128(_load_pair(a, i))
        var p0 = word * UInt128(b0)
        var p1 = word * UInt128(b1)
        carry0, carry1 = _product_two_step[add](p0, pending, carry0, carry1, r, offset + i)
        pending = p1
        i += 2
    var last = pending + UInt128(carry0) + UInt128(carry1)
    _put_pair(r, offset + len(a), UInt64(last))
    _put_pair(r, offset + len(a) + 2, UInt64(last >> 64))


def _mul_basecase(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    r: Span[mut=True, UInt32, _],
):
    if len(a) >= 8 and len(b) >= 4 and not (len(a) % 2 or len(b) % 2):
        _product_two_rows[False](a, _load_pair(b, 0), _load_pair(b, 2), r, 0)
        var j = 4
        while j + 4 <= len(b):
            _product_two_rows[True](a, _load_pair(b, j), _load_pair(b, j + 2), r, j)
            j += 4
        if j < len(b):
            _product_row[True](a, _load_pair(b, j), r, j)
    else:
        _mul_single_rows(a, b, r)


def _mul_single_rows(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    r: Span[mut=True, UInt32, _],
):
    # The first row initializes the product; later rows read only initialized
    # partial products. The caller preflights len(a)+len(b) disjoint words.
    if not len(b):
        r.fill(0)
        return
    var right = len(b) & -2
    if right:
        _product_row[False](a, _load_pair(b, 0), r, 0)
        var j = 2
        while j < right:
            _product_row[True](a, _load_pair(b, j), r, j)
            j += 2
    if right != len(b):
        var factor = UInt64(b.unsafe_get(right))
        if right:
            _product_row[True, True](a, factor, r, right)
        else:
            _product_row[False, True](a, factor, r, 0)


def _square_basecase(
    a: Span[mut=False, UInt32, _], r: Span[mut=True, UInt32, _],
):
    r.fill(0)
    var size = len(a) // 2
    for i in range(size):
        var word = UInt128(_load_pair(a, 2 * i))
        var carry = UInt64(0)
        for j in range(i + 1, size):
            var offset = 2 * (i + j)
            var total = (
                word * UInt128(_load_pair(a, 2 * j))
                + UInt128(_load_pair(r, offset)) + UInt128(carry)
            )
            _put_pair(r, offset, UInt64(total))
            carry = UInt64(total >> 64)
        _put_pair(r, 2 * (i + size), carry)
    var carry = UInt64(0)
    for i in range(2 * len(a)):
        var total = (UInt64(r.unsafe_get(i)) << 1) | carry
        r.unsafe_get(i) = UInt32(total)
        carry = total >> 32
    carry = 0
    for i in range(size):
        var word = UInt128(_load_pair(a, 2 * i))
        var total = word * word + UInt128(_load_pair(r, 4 * i)) + UInt128(carry)
        _put_pair(r, 4 * i, UInt64(total))
        var high = (total >> 64) + UInt128(_load_pair(r, 4 * i + 2))
        _put_pair(r, 4 * i + 2, UInt64(high))
        carry = UInt64(high >> 64)
    if len(a) % 2:
        var offset = len(a) - 1
        var top = UInt128(a.unsafe_get(offset))
        var tail_carry = UInt128(0)
        for i in range(offset):
            var total = 2 * top * UInt128(a.unsafe_get(i)) + UInt128(r.unsafe_get(offset + i)) + tail_carry
            r.unsafe_get(offset + i) = UInt32(total)
            tail_carry = total >> 32
        _put_pair(r, 2 * offset, UInt64(top * top + tail_carry))


@always_inline
def _product_subtract(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    r: Span[mut=True, UInt32, _],
):
    var common = min(len(a), len(b))
    var borrow = UInt64(0)
    for i in range(common):
        var x = UInt64(a.unsafe_get(i))
        var sub = UInt64(b.unsafe_get(i)) + borrow
        r.unsafe_get(i) = UInt32(x - sub)
        borrow = UInt64(x < sub)
    for i in range(common, len(r)):
        var x = UInt64(a.unsafe_get(i)) if i < len(a) else UInt64(0)
        var sub = (UInt64(b.unsafe_get(i)) if i < len(b) else UInt64(0)) + borrow
        r.unsafe_get(i) = UInt32(x - sub)
        borrow = UInt64(x < sub)


def _product_difference(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    r: Span[mut=True, UInt32, _],
) -> Bool:
    var n = max(len(a), len(b))
    var negative = False
    for i in range(n - 1, -1, -1):
        var x = a.unsafe_get(i) if i < len(a) else UInt32(0)
        var y = b.unsafe_get(i) if i < len(b) else UInt32(0)
        if x != y:
            negative = x < y
            break
    if negative:
        _product_subtract(b, a, r)
    else:
        _product_subtract(a, b, r)
    return negative


@always_inline
def _product_cross_words[subtract: Bool](
    r: Span[mut=True, UInt32, _],
    low: Span[mut=False, UInt32, _], high: Span[mut=False, UInt32, _],
):
    # z0+z2 - (a0-a1)(b0-b1); signed carry stays within two limb magnitudes.
    var carry = Int64(0)
    var common = min(len(low), len(high))
    for i in range(common):
        var total = carry + Int64(low.unsafe_get(i)) + Int64(high.unsafe_get(i))
        var difference = Int64(r.unsafe_get(i))
        total += -difference if subtract else difference
        r.unsafe_get(i) = UInt32(total)
        carry = total >> 32
    for i in range(common, len(r)):
        var total = carry
        if i < len(low):
            total += Int64(low.unsafe_get(i))
        if i < len(high):
            total += Int64(high.unsafe_get(i))
        var difference = Int64(r.unsafe_get(i))
        total += -difference if subtract else difference
        r.unsafe_get(i) = UInt32(total)
        carry = total >> 32


def _product_cross(
    r: Span[mut=True, UInt32, _],
    low: Span[mut=False, UInt32, _], high: Span[mut=False, UInt32, _],
    subtract: Bool,
):
    if subtract:
        _product_cross_words[True](r, low, high)
    else:
        _product_cross_words[False](r, low, high)


def _product_accumulate(
    r: Span[mut=True, UInt32, _], a: Span[mut=False, UInt32, _], offset: Int,
):
    var carry = UInt64(0)
    for i in range(len(a)):
        var total = UInt64(r.unsafe_get(offset + i)) + UInt64(a.unsafe_get(i)) + carry
        r.unsafe_get(offset + i) = UInt32(total)
        carry = total >> 32
    var index = offset + len(a)
    while carry:
        var total = UInt64(r.unsafe_get(index)) + carry
        r.unsafe_get(index) = UInt32(total)
        carry = total >> 32
        index += 1


def _product_recursive(
    a: Span[UInt32, ImmutAnyOrigin], b: Span[UInt32, ImmutAnyOrigin],
    r: Span[UInt32, MutAnyOrigin], work: Span[UInt32, MutAnyOrigin],
    same: Bool,
):
    if len(a) < len(b):
        _product_recursive(b, a, r, work, same)
        return
    var threshold = _KARATSUBA_SQUARE_WORDS if same else _KARATSUBA_WORDS
    var basecase = len(b) < threshold or not len(work)
    if not basecase:
        var frame = 2 * len(b) if len(a) >= 2 * len(b) else 4 * ((len(a) + 1) // 2) + 1
        basecase = len(work) < frame
    if basecase:
        if same and len(a) >= 16:
            _square_basecase(a, r)
        else:
            _mul_basecase(a, b, r)
        return
    if len(a) >= 2 * len(b):
        # Chunk the longer operand instead of zero-padding a lopsided product.
        r.fill(0)
        var chunk = len(b)
        for start in range(0, len(a), chunk):
            var end = min(start + chunk, len(a))
            var size = end - start + len(b)
            var product = _product_slice(work, 0, size)
            _product_recursive(_product_slice(a, start, end), b, product, _product_slice(work, 2 * chunk, len(work)), False)
            _product_accumulate(r, _product_slice(product, 0, _product_used(product)), start)
        return
    var half = (len(a) + 1) // 2
    var sa = _product_slice(work, 0, half)
    var sb = _product_slice(work, half, 2 * half)
    var middle = _product_slice(work, 2 * half, 4 * half + 1)
    var tail = _product_slice(work, 4 * half + 1, len(work))
    var a0 = _product_slice(a, 0, half)
    var a1 = _product_slice(a, half, len(a))
    var b0 = _product_slice(b, 0, half)
    var b1 = _product_slice(b, half, len(b))
    var low = _product_slice(r, 0, 2 * half)
    var high = _product_slice(r, 2 * half, len(r))
    var negative_a = _product_difference(a0, a1, sa)
    var negative_b = _product_difference(b0, b1, sb)
    _product_recursive(a0, b0, low, tail, same)
    _product_recursive(a1, b1, high, tail, same)
    _product_recursive(
        Span(unsafe_ptr=sa.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=half),
        Span(unsafe_ptr=sb.unsafe_ptr().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=half),
        _product_slice(middle, 0, 2 * half), tail, same,
    )
    middle.unsafe_get(2 * half) = 0
    _product_cross(middle, low, high, negative_a == negative_b)
    _product_accumulate(r, _product_slice(middle, 0, _product_used(middle)), half)


def _multiply_into(
    a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _],
    result: Span[mut=True, UInt32, _], same_operand: Bool = False,
    workspace: Span[mut=True, UInt32, _] = Span[UInt32, MutAnyOrigin](),
) -> Int:
    # All spans remain live for this non-escaping call. The checked caller layout
    # establishes output/workspace disjointness before their origins are erased.
    var left = _product_used(a)
    var right = _product_used(b)
    result[left + right:].fill(0)
    if not left or not right:
        result.fill(0)
        return 0
    _product_recursive(
        Span(unsafe_ptr=a.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](), length=left),
        Span(unsafe_ptr=b.unsafe_ptr().unsafe_origin_cast[ImmutAnyOrigin](), length=right),
        Span(unsafe_ptr=result.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), length=left + right),
        Span(unsafe_ptr=workspace.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), length=len(workspace)),
        same_operand,
    )
    return _product_used(result[:left + right])
