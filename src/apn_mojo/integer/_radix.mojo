"""Kernels of exact text conversion in any base, on preflighted buffers.

A power-of-two base reads and writes bits: each digit is `log2(base)` bits of
the magnitude, so formatting and parsing take time linear in the length.
Any other base divides, or multiplies, the whole magnitude once per chunk of
as many digits as fit in a 64-bit limb (`_radix_chunk`), as the decimal
paths do: still quadratic, but with 12 to 63 digits per pass instead of one.
"""
from std.bit import count_leading_zeros, count_trailing_zeros

from ..common._text import _text_digit
from ._division import _divide_2by1, _reciprocal


@always_inline
def _is_power_of_two_base(base: Int) -> Bool:
    return base & (base - 1) == 0


@always_inline
def _base_shift(base: Int) -> Int:
    """Bits per digit of a power-of-two base."""
    return Int(count_trailing_zeros(UInt64(base)))


def _radix_chunk(base: Int, limit: UInt64) -> Tuple[UInt64, Int]:
    """The largest power `base**k <= limit`, and `k`: one division, then multiplications."""
    var bound = limit // UInt64(base)
    var power = UInt64(base)
    var digits = 1
    while power <= bound:
        power *= UInt64(base)
        digits += 1
    return (power, digits)


@always_inline
def _digit_byte(digit: Int, uppercase: Bool) -> UInt8:
    return UInt8(digit + (48 if digit < 10 else 55 if uppercase else 87))


def _power_of_two_digit_count(bits: Int, shift: Int) -> Int:
    """Digits of a magnitude of `bits` bits in base `2**shift`; zero has one."""
    if not bits:
        return 1
    return (bits - 1) // shift + 1


def _write_power_of_two_digits(
    words: Span[mut=False, UInt32, _],
    shift: Int,
    uppercase: Bool,
    mut bytes: List[UInt8],
    offset: Int,
    count: Int,
):
    """Write `count` digits of base `2**shift`, most significant first, at `offset`."""
    var mask = (UInt64(1) << UInt64(shift)) - 1
    var used = len(words)
    var target = bytes.unsafe_ptr().unsafe_offset(offset)
    for d in range(count):
        var bit = d * shift
        var index = bit >> 5
        var within = bit & 31
        var value = UInt64(0)
        if index < used:
            value = UInt64(words.unsafe_get(index)) >> UInt64(within)
            if within + shift > 32 and index + 1 < used:
                value |= UInt64(words.unsafe_get(index + 1)) << UInt64(32 - within)
        target.unsafe_offset(count - 1 - d)[] = _digit_byte(Int(value & mask), uppercase)


def _power_of_two_word_count(digits: Int, shift: Int) -> Int:
    """Words holding `digits` digits of base `2**shift`."""
    return (digits * shift + 31) // 32


def _read_power_of_two_digits(
    text: Span[mut=False, UInt8, _],
    start: Int,
    end: Int,
    shift: Int,
    skip_decimal_point: Bool,
    mut words: List[UInt32],
):
    """Pack validated digits of base `2**shift` into `words`, least significant first.

    `words` holds enough zero words for every digit; underscores, and a
    decimal point when `skip_decimal_point`, are skipped.
    """
    var target = words.unsafe_ptr()
    var accumulator = UInt64(0)
    var filled = 0
    var index = 0
    for position in range(end - 1, start - 1, -1):
        var byte = text.unsafe_get(position)
        if byte == 95 or (skip_decimal_point and byte == 46):
            continue
        accumulator |= UInt64(_text_digit(byte)) << UInt64(filled)
        filled += shift
        if filled >= 32:
            target.unsafe_offset(index)[] = UInt32(accumulator & 0xFFFFFFFF)
            index += 1
            accumulator >>= 32
            filled -= 32
    if filled > 0 and index < len(words):
        target.unsafe_offset(index)[] = UInt32(accumulator)


struct _ChunkDivisor(ImplicitlyCopyable):
    """Division of 64-bit limbs by `base**digits`, normalized for `_divide_2by1`."""

    var power: UInt64
    var digits: Int
    var shift: UInt64
    var normalized: UInt64
    var reciprocal: UInt64

    def __init__(out self, base: Int):
        var chunk = _radix_chunk(base, UInt64.MAX)
        self.power = chunk[0]
        self.digits = chunk[1]
        self.shift = UInt64(count_leading_zeros(self.power))
        self.normalized = self.power << self.shift
        self.reciprocal = _reciprocal(self.normalized)

    @always_inline
    def divide(self, mut limbs: List[UInt64]) -> UInt64:
        """Divide the limbs by `power` in place; return the remainder."""
        var rest = UInt64(0)
        var data = limbs.unsafe_ptr()
        if self.shift == 0:
            for i in range(len(limbs) - 1, -1, -1):
                var step = _divide_2by1(rest, data.unsafe_offset(i)[], self.normalized, self.reciprocal)
                data.unsafe_offset(i)[] = step[0]
                rest = step[1]
            return rest
        # (rest * 2**64 + limb) * 2**shift by the normalized divisor has the same
        # quotient; its remainder is the true one times 2**shift.
        var back = UInt64(64) - self.shift
        for i in range(len(limbs) - 1, -1, -1):
            var limb = data.unsafe_offset(i)[]
            var high = (rest << self.shift) | (limb >> back)
            var step = _divide_2by1(high, limb << self.shift, self.normalized, self.reciprocal)
            data.unsafe_offset(i)[] = step[0]
            rest = step[1] >> self.shift
        return rest


def _digits_in_chunk(var value: UInt64, base: Int) -> Int:
    """Digits of a nonzero chunk value; zero has none."""
    var digits = 0
    while value:
        value //= UInt64(base)
        digits += 1
    return digits


def _write_chunk(
    var value: UInt64, base: Int, uppercase: Bool, mut bytes: List[UInt8], end: Int, digits: Int
):
    """Write `digits` digits of `value`, zero-padded, ending before `end`."""
    var target = bytes.unsafe_ptr()
    for k in range(digits):
        target.unsafe_offset(end - 1 - k)[] = _digit_byte(Int(value % UInt64(base)), uppercase)
        value //= UInt64(base)
