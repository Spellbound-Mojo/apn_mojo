"""Exact signed radix-2**32 accumulation (dense windows, sparse when values
lie far apart) and gap-compressed rounding."""

from std.bit import count_leading_zeros
from std.sys import size_of
from ..common._sizes import _checked_count, _checked_sum
from ..integer.value import Integer
from .context import ArithmeticContext
from ._rounding import _RoundedBinary, _round_ratio, _compare_scaled


def _bin_index(position: Int128) raises -> Int:
    if position < Int128(Int.MIN) or position >= Int128(Int.MAX):
        raise Error(
            "Cannot accumulate Float values: exponent bins exceed the checked"
            " addressable range; use smaller exponents. The destination is"
            " unchanged."
        )
    return Int(position)


@fieldwise_init
struct _DigitRun(ImplicitlyCopyable):
    var low: Int
    var high: Int
    var digit: UInt32


struct _SparseMagnitude(Movable):
    var runs: List[_DigitRun]
    var negative: Bool

    def __init__(out self, negative: Bool):
        self.runs = List[_DigitRun]()
        self.negative = negative

    def append(mut self, low: Int, high: Int, digit: UInt32) raises:
        if not digit or low > high:
            return
        _ = _checked_count(
            _checked_sum(len(self.runs), 1), size_of[_DigitRun]()
        )
        self.runs.append(_DigitRun(low, high, digit))

    def word(self, index: Int) -> UInt32:
        var low = 0
        var high = len(self.runs)
        while low < high:
            var middle = low + (high - low) // 2
            if self.runs[middle].high < index:
                low = middle + 1
            else:
                high = middle
        if low < len(self.runs) and self.runs[low].low <= index:
            return self.runs[low].digit
        return 0

    def rounded(
        self,
        context: ArithmeticContext,
        fail: Bool = False,
        *,
        denominator: Integer = 1,
    ) raises -> _RoundedBinary:
        if not len(self.runs):
            return _round_ratio(
                Integer(0),
                denominator,
                context,
                negative_zero=self.negative,
                fail=fail,
            )
        var last = self.runs[len(self.runs) - 1]
        var top = Int128(last.high) * 32 + Int128(
            32 - Int(count_leading_zeros(last.digit))
        )
        if context.format()._is_exact():
            # Exact working formats keep every occupied digit.
            var span = Int(top - Int128(self.runs[0].low) * 32)
            return _round_ratio(
                self.prefix(top, span), denominator, context, scale=top - Int128(span), fail=fail
            )
        # A denominator widens the retained prefix so all final rounding
        # boundaries remain even integers at the extraction scale.
        var extra = (
            denominator.magnitude_bit_length() if not denominator._is_one() else 0
        )
        var low_exponent = top - Int128(extra)
        var high_exponent = low_exponent + Int128(extra != 0)
        if extra and (
            low_exponent < Int128(context.format().emin()) <= high_exponent
            or low_exponent <= Int128(context.format().emax()) < high_exponent
        ):
            # Resolve the one-bit exponent uncertainty before allocating output
            # precision; an underflow-to-zero may need only denominator storage.
            var prefix_bits = _checked_sum(extra, 3)
            var prefix = self.prefix(top, prefix_bits)
            low_exponent += Int128(
                _compare_scaled(
                    abs(prefix),
                    denominator,
                    low_exponent - top + Int128(prefix_bits),
                )
                >= 0
            )
            high_exponent = low_exponent
        var bits = (
            3 if high_exponent < Int128(context.format().emin())
            or low_exponent
            > Int128(context.format().emax()) else context.format().precision()
            + 2
        )
        bits = _checked_sum(bits, extra)
        var value = self.prefix(top, bits)
        return _round_ratio(
            value, denominator, context, scale=top - Int128(bits), fail=fail
        )

    def prefix(self, top: Int128, bits: Int) raises -> Integer:
        var cutoff = top - Int128(bits)
        var base = cutoff // 32
        var offset = Int(cutoff - base * 32)
        var start = _bin_index(base)
        var count = _checked_count(bits // 32 + Int(bits % 32 != 0), 4)
        var words = List[UInt32](length=count, fill=0)
        for row in range(count):
            var position = _bin_index(base + Int128(row))
            words[row] = self.word(position) >> UInt32(offset)
            if offset:
                words[row] |= self.word(position + 1) << UInt32(32 - offset)
        # The least occupied run tells whether any nonzero bits were omitted.
        var first = self.runs[0]
        var sticky = first.low < start
        if first.low == start and offset:
            sticky = Bool(first.digit & ((UInt32(1) << UInt32(offset)) - 1))
        if sticky:
            words[0] |= 1
        return Integer._from_words(words^, self.negative)


comptime _DENSE_DIGITS = 1 << 12
"""The widest dense window, in radix-2**32 digits (128 Kibit)."""
comptime _CARRY_EVERY = 1 << 30
"""Additions a dense digit absorbs before carries must be propagated."""
comptime _DENSE_POSITIONS = Int(1) << 60
"""Dense windows stay within these digit positions; beyond them, sparse."""


struct _ExactAccumulator(Movable):
    """An exact signed sum of scaled integers, in radix-2**32 digits.

    Digits live in a dense window of Int64 counters that absorb each addition
    without carrying, so adding a value is one pass over its words. Values
    too far apart for one window move the whole sum to a sparse map of
    carried digits, so the cost never depends on exponent gaps.
    """

    comptime RADIX = Int64(1) << 32
    var dense: List[Int64]
    # Digit position of dense[0].
    var low: Int
    # Additions since carries were last propagated through the window.
    var additions: Int
    var sparse: Dict[Int, Int64]
    var is_sparse: Bool

    def __init__(out self):
        self.dense = List[Int64]()
        self.low = 0
        self.additions = 0
        self.sparse = Dict[Int, Int64]()
        self.is_sparse = False

    def _reserve(mut self, first: Int, last: Int) -> Bool:
        """Cover digit positions [first, last] with the dense window, or
        return False when that would exceed _DENSE_DIGITS."""
        var count = len(self.dense)
        if count and first >= self.low and last < self.low + count:
            return True
        if first < -_DENSE_POSITIONS or last > _DENSE_POSITIONS:
            return False
        var new_first = min(first, self.low) if count else first
        var new_last = max(last, self.low + count - 1) if count else last
        if new_last - new_first + 1 > _DENSE_DIGITS:
            return False
        # Slack on each side that grows, so repeated growth copies rarely.
        var slack = max(4, (new_last - new_first + 1) // 2)
        if not count or new_first < self.low:
            new_first = max(new_first - slack, new_last - _DENSE_DIGITS + 1)
        if not count or new_last >= self.low + count:
            new_last = min(new_last + slack, new_first + _DENSE_DIGITS - 1)
        var digits = List[Int64](length=new_last - new_first + 1, fill=0)
        var offset = self.low - new_first
        for i in range(count):
            digits.unsafe_ptr().unsafe_offset(offset + i)[] = self.dense.unsafe_ptr().unsafe_offset(i)[]
        self.dense = digits^
        self.low = new_first
        return True

    def _carry(mut self) raises:
        """Propagate carries so every digit but a signed top one lies in
        [0, 2**32)."""
        var carry = Int64(0)
        var count = len(self.dense)
        for i in range(count):
            var total = self.dense[i] + carry
            carry = total >> 32
            self.dense[i] = total & (Self.RADIX - 1)
        self.additions = 1
        if carry:
            var position = self.low + count
            if self._reserve(position, position):
                self.dense[position - self.low] += carry
            else:
                self._to_sparse()
                self._add_sparse_digit(position, carry)

    def add_magnitude(mut self, words: Span[mut=False, UInt32, _], scale: Int128, negative: Bool) raises:
        """Add (or subtract) words * 2**scale."""
        var count = len(words)
        if not count:
            return
        var base = scale // 32
        var offset = UInt64(scale - base * 32)
        var top = base + Int128(count)
        var dense = (
            not self.is_sparse and base > Int128(-_DENSE_POSITIONS) and top < Int128(_DENSE_POSITIONS)
            and self._reserve(Int(base), Int(top))
        )
        if dense and self.additions >= _CARRY_EVERY:
            self._carry()
            # The carry pass may grow the window or move the sum to the map.
            dense = not self.is_sparse and self._reserve(Int(base), Int(top))
        if dense:
            self.additions += 1
            var target = self.dense.unsafe_ptr().unsafe_offset(Int(base) - self.low)
            var below = UInt64(0)
            for i in range(count):
                var word = UInt64(words.unsafe_get(i))
                var shifted = Int64(((word << offset) | (below >> (32 - offset))) & 0xFFFFFFFF)
                if negative:
                    target.unsafe_offset(i)[] -= shifted
                else:
                    target.unsafe_offset(i)[] += shifted
                below = word
            var spill = Int64(below >> (32 - offset))
            if negative:
                target.unsafe_offset(count)[] -= spill
            else:
                target.unsafe_offset(count)[] += spill
            return
        self._to_sparse()
        for row in range(count):
            self._add_sparse_limb(row, words.unsafe_get(row), scale, negative)

    def add(mut self, value: Integer, scale: Int128) raises:
        var small = value._inline_words()
        self.add_magnitude(value._words_span(small), scale, value._negative())

    def merge(mut self, other: Self) raises:
        """Add another exact sum."""
        if not other.is_sparse and not len(other.dense):
            return
        var count = len(other.dense)
        if not self.is_sparse and not len(self.dense) and not other.is_sparse:
            # Into an empty sum: the same window, without growth slack.
            self.dense = other.dense.copy()
            self.low = other.low
            self.additions = other.additions
            return
        if (
            not self.is_sparse and not other.is_sparse
            and self.additions + other.additions < _CARRY_EVERY
            and self._reserve(other.low, other.low + count - 1)
        ):
            var target = self.dense.unsafe_ptr().unsafe_offset(other.low - self.low)
            for i in range(count):
                target.unsafe_offset(i)[] += other.dense.unsafe_ptr().unsafe_offset(i)[]
            self.additions += other.additions
            return
        if other.is_sparse:
            for entry in other.sparse.items():
                self._add_digit(entry.key, entry.value)
        else:
            for i in range(count):
                self._add_digit(other.low + i, other.dense[i])

    def _add_digit(mut self, position: Int, digit: Int64) raises:
        """Add digit * 2**(32 * position) for any Int64 digit."""
        if not digit:
            return
        var magnitude = UInt64(digit) if digit > 0 else UInt64(0) - UInt64(digit)
        var words = Array[UInt32, 2](fill=0)
        words[0] = UInt32(magnitude & 0xFFFFFFFF)
        words[1] = UInt32(magnitude >> 32)
        self.add_magnitude(Span(words), Int128(position) * 32, digit < 0)

    def _to_sparse(mut self) raises:
        if self.is_sparse:
            return
        self.is_sparse = True
        var digits = self.dense^
        self.dense = List[Int64]()
        for i in range(len(digits)):
            self._add_sparse_digit(self.low + i, digits[i])

    def _add_sparse_digit(mut self, position: Int, digit: Int64) raises:
        if not digit:
            return
        var magnitude = UInt64(digit) if digit > 0 else UInt64(0) - UInt64(digit)
        self._add_sparse_word(position, UInt32(magnitude & 0xFFFFFFFF), digit < 0)
        if magnitude >> 32:
            self._add_sparse_word(_bin_index(Int128(position) + 1), UInt32(magnitude >> 32), digit < 0)

    def _add_sparse_word(mut self, position: Int, word: UInt32, negative: Bool) raises:
        var carry = -Int64(word) if negative else Int64(word)
        var index = position
        while carry:
            var total = self.sparse.get(index, 0) + carry
            # Truncating signed remainders avoid borrowing through empty gaps.
            carry = total // Self.RADIX if total >= 0 else -(
                (-total) // Self.RADIX
            )
            _ = _checked_count(_checked_sum(len(self.sparse), 1), 32)
            self.sparse[index] = total - carry * Self.RADIX
            if carry:
                index = _bin_index(Int128(index) + 1)

    def _add_sparse_limb(
        mut self, row: Int, word: UInt32, scale: Int128, negative: Bool
    ) raises:
        var base = scale // 32
        var offset = Int(scale - base * 32)
        var index = _bin_index(base + Int128(row))
        self._add_sparse_word(index, word << UInt32(offset), negative)
        if offset:
            self._add_sparse_word(
                _bin_index(base + Int128(row) + 1),
                word >> UInt32(32 - offset),
                negative,
            )

    def magnitude(self, negative_zero: Bool = False) raises -> _SparseMagnitude:
        if self.is_sparse:
            return self._sparse_magnitude(negative_zero)
        # Carry a copy of the window; a negative final carry means a negative
        # sum, whose magnitude carries the negated digits.
        var count = len(self.dense)
        var digits = List[Int64](length=count, fill=0)
        var carry = Int64(0)
        for i in range(count):
            var total = self.dense[i] + carry
            carry = total >> 32
            digits[i] = total & (Self.RADIX - 1)
        var negative = carry < 0
        if negative:
            carry = 0
            for i in range(count):
                var total = carry - self.dense[i]
                carry = total >> 32
                digits[i] = total & (Self.RADIX - 1)
        var result = _SparseMagnitude(negative)
        for i in range(count):
            if digits[i]:
                result.append(self.low + i, self.low + i, UInt32(digits[i]))
        # The window keeps headroom above its values, so a remaining carry is
        # rare; it lands above the window.
        var position = self.low + count
        while carry:
            result.append(position, position, UInt32(carry & (Self.RADIX - 1)))
            carry >>= 32
            position += 1
        if not len(result.runs):
            return _SparseMagnitude(negative_zero)
        return result^

    def _sparse_magnitude(self, negative_zero: Bool) raises -> _SparseMagnitude:
        var keys = List[Int](
            capacity=_checked_count(len(self.sparse), size_of[Int]())
        )
        for entry in self.sparse.items():
            if entry.value:
                keys.append(entry.key)
        if not len(keys):
            return _SparseMagnitude(negative_zero)
        sort(keys[:])
        var negative = self.sparse[keys[len(keys) - 1]] < 0
        var result = _SparseMagnitude(negative)
        var carry = Int64(0)
        var previous = keys[0]
        for index in keys:
            if carry and index > previous + 1:
                result.append(previous + 1, index - 1, UInt32.MAX)
            var digit = self.sparse[index]
            if negative:
                digit = -digit
            digit += carry
            carry = -1 if digit < 0 else 0
            if digit < 0:
                digit += Self.RADIX
            result.append(index, index, UInt32(digit))
            previous = index
        return result^
