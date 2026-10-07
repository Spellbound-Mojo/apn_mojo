"""Allocation-free scalar kernels over preflighted, disjoint arena slices."""

from ._division import _divide_into
from ._multiplication import _multiply_into
from ._magnitude import _MagnitudeWords
from ._words import _add_words, _subtract_words, _shift_left_words, _shift_right_words, _compare_words, _bitwise_words


def _finish_words(mut words: _MagnitudeWords, used: Int, offset: Int):
    words.finish(used, offset)


def _finish_words(mut words: List[UInt32], used: Int, offset: Int):
    var capacity = words.capacity()
    # Result/division/power slices need more room than independent results.
    # This overflow-free comparison is capacity > max(128, 16 * used).
    if capacity > 128 and (capacity - 1) // 16 >= used:
        var compact = List[UInt32](length=used, fill=0)
        for i in range(used):
            compact[i] = words[offset + i]
        words = compact^
    else:
        for i in range(used):
            words[i] = words[offset + i]
        words.resize(used, 0)


def _used(words: Span[mut=False, UInt32, _]) -> Int:
    var count = len(words)
    while count and words[count - 1] == 0:
        count -= 1
    return count


def _sum_into(
    a: Span[mut=False, UInt32, _],
    b: Span[mut=False, UInt32, _],
    a_negative: Bool,
    b_negative: Bool,
    result: Span[mut=True, UInt32, _],
) -> Bool:
    result.fill(0)
    if a_negative == b_negative:
        _add_words(a, b, result.unsafe_ptr())
        return a_negative
    var swap = _compare_words(a, b) < 0
    if swap:
        _subtract_words(b, a, result.unsafe_ptr())
    else:
        _subtract_words(a, b, result.unsafe_ptr())
    return b_negative if swap else a_negative




def _bits_into(
    a: Span[mut=False, UInt32, _],
    b: Span[mut=False, UInt32, _],
    a_negative: Bool,
    b_negative: Bool,
    operation: Int,
    result: Span[mut=True, UInt32, _],
) -> Bool:
    # Operations 4, 5 and 6 are the kernel's and, or and xor.
    return _bitwise_words(
        a, b, a_negative, b_negative, operation - 4, result.unsafe_ptr(), len(result)
    )


def _shift_into(
    a: Span[mut=False, UInt32, _],
    negative: Bool,
    shift: Int,
    right: Bool,
    result: Span[mut=True, UInt32, _],
):
    result.fill(0)
    var whole = shift // 32
    var part = shift % 32
    if not right:
        _shift_left_words(a, whole, part, result.unsafe_ptr())
        return
    var discarded = False
    for i in range(whole):
        discarded |= a[i] != 0
    discarded |= (UInt64(a[whole]) & ((UInt64(1) << UInt64(part)) - 1)) != 0
    result[len(a) - whole] = UInt32(
        _shift_right_words(a, whole, part, UInt64(negative and discarded), result.unsafe_ptr())
    )


def _power_into(
    a: Span[mut=False, UInt32, _],
    var exponent: Int,
    arena: Span[UInt32, MutAnyOrigin],
    width: Int,
):
    var accumulator = 0
    var factor = width
    var temporary = width * 2
    arena.fill(0)
    arena[0] = 1
    for i in range(len(a)):
        arena[factor + i] = a[i]
    var a_size = 1
    var f_size = len(a)
    while exponent:
        if exponent & 1:
            a_size = _multiply_into(
                arena[accumulator : accumulator + a_size],
                arena[factor : factor + f_size],
                arena[temporary : temporary + width],
                workspace=arena[3 * width:],
            )
            var old = accumulator
            accumulator = temporary
            temporary = old
        exponent >>= 1
        if exponent:
            f_size = _multiply_into(
                arena[factor : factor + f_size],
                arena[factor : factor + f_size],
                arena[temporary : temporary + width],
                same_operand=True,
                workspace=arena[3 * width:],
            )
            var old = factor
            factor = temporary
            temporary = old
    if accumulator:
        for i in range(a_size):
            arena[i] = arena[accumulator + i]
    for i in range(a_size, width):
        arena[i] = 0


def _update_words(
    arena: Span[UInt32, MutAnyOrigin],
    left: Int,
    right: Int,
    width: Int,
    a_negative: Bool,
    b_negative: Bool,
    operation: Int,
    count: Int,
    same_operand: Bool = False,
) -> Bool:
    var output = left + right
    var a = arena[:left]
    var b = arena[left:output]
    var result = arena[output : output + width]
    if operation <= 1:
        return _sum_into(
            a, b, a_negative, b_negative != (operation == 1), result
        )
    if operation == 2:
        _ = _multiply_into(a, b, result, same_operand, arena[output + width:])
        return a_negative != b_negative
    if 4 <= operation and operation <= 6:
        return _bits_into(a, b, a_negative, b_negative, operation, result)
    if operation == 8 or operation == 9:
        _shift_into(a, a_negative, count, operation == 9, result)
        return a_negative
    if operation == 10:
        _power_into(a, count, arena[output:], width)
        return a_negative and (count & 1) != 0
    var order = _compare_words(a, b)
    var opposite = a_negative != b_negative
    result.fill(0)
    if order <= 0:
        if order == 0:
            result[0] = UInt32(operation == 3)
            return opposite
        if operation == 3:
            result[0] = UInt32(opposite and left != 0)
            return opposite
        if opposite and left:
            return _sum_into(b, a, b_negative, not b_negative, result)
        for i in range(left):
            result[i] = a[i]
        return a_negative
    var offset = output + width
    var dividend = arena[offset : offset + left + 1]
    var divisor = arena[offset + left + 1 :]
    for i in range(left):
        dividend[i] = a[i]
    dividend[left] = 0
    for i in range(right):
        divisor[i] = b[i]
    var rest = _divide_into(dividend, divisor, result)
    if operation == 11:
        if opposite and rest:
            return _sum_into(
                b, dividend[:rest], b_negative, not b_negative, result
            )
        result.fill(0)
        for i in range(rest):
            result[i] = dividend[i]
        return a_negative
    if opposite and rest:
        var carry = UInt64(1)
        for i in range(width):
            var value = UInt64(result[i]) + carry
            result[i] = UInt32(value)
            carry = value >> 32
    return opposite
