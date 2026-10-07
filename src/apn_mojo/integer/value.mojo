"""Exact signed integers with inline words and immutable shared large values."""

from std.math import Absable
from std.builtin.comparable import Comparable
from std.builtin.int import IntableRaising
from std.sys import bit_width_of, size_of
from std.utils import Variant
from std.bit import count_leading_zeros, count_trailing_zeros
from std.hashlib import Hasher
from std.os import abort

from ..rational.value import Rational
from ..batch._comparison import _ExactBatchComparison
from ..float._comparison import _FloatComparison
from ..complex._comparison import _ComplexComparison
from ..complex.value import Complex
from ..float.value import Float
from ..batch.mask import Mask
from ..common._sizes import _checked_count, _checked_sum
from ._division import _load_limbs, _store_words, _divide_limbs, _divide_2by1, _divide_by_limb_into
from ._gcd import _native_gcd, _limb_gcd
from ._scalar_update import _finish_words, _update_words, _used
from ._words import _add_words, _subtract_words, _shift_left_words, _shift_right_words, _compare_words, _bitwise_words
from ._multiplication import _product_scratch_size, _multiply_with_scratch, _multiply_short_into, _product_used, _high_product_into
from ._magnitude import _LargeInteger, _SharedMagnitude
from ._operand import (
    _IntegerOperand,
    _NativeOperand,
    _LiteralOperand,
    _operand_count,
    _operand_at_least,
)
from ._limits import _MAX_RESULT_BITS, _count_error
from ._static import _StaticInteger
from ._json import _json_decimal_bound, _read_integer_json
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..common._text import _text_digit
from ._text import _integer_text, _text_prefix, _format_bound
from ..common._traits import _BatchElement

# Decimal text moves nineteen digits per step on 64-bit limbs: 10**19 is the
# largest power of ten below 2**64, and its high bit is set, so dividing by it
# can use a precomputed reciprocal (_division._reciprocal(10**19)).
comptime _TEN19 = UInt64(10_000_000_000_000_000_000)
comptime _TEN19_RECIPROCAL = UInt64(15581492618384294730)


def _unsigned_magnitude(value: Int64) -> UInt64:
    if value < 0:
        return UInt64(-(value + 1)) + 1
    return UInt64(value)


struct Integer(
    Absable,
    Comparable,
    Hashable,
    ImplicitlyCopyable,
    IntableRaising,
    Writable,
    _IntegerOperand,
    _BatchElement,
):
    """An exact signed integer of any size.

    Arithmetic grows the value instead of wrapping, and a copy keeps its value
    when the original changes. Values that fit 64 bits are stored inline; larger
    ones share immutable words, so copies are cheap.

    | Operation | Contract |
    |---|---|
    | `x + y`, `x - y`, `x * y` | Exact |
    | `x / y` | Exact `Rational`, even when integral; a zero divisor raises |
    | `x // y`, `x % y` | Floor division and its remainder; a zero divisor raises |
    | `x ** n` | Nonnegative exponent; `0 ** 0` is one |
    | `x & y`, `x | y`, `x ^ y`, `~x` | Infinite-width two's complement |
    | `x << n`, `x >> n` | Nonnegative count; `>>` rounds toward negative infinity |
    | `==`, `!=`, `<`, `<=`, `>`, `>=` | Exact comparison returning `Bool` |
    | `-x`, `abs(x)`, `Bool(x)` | Negation, magnitude, and false only for zero |

    All eleven compound forms (`+=` through `**=`) leave the destination unchanged
    when they raise. Arithmetic with a native integer or an integer literal is exact
    in either order; with a `Rational` it returns `Rational`, and with a `Float` or
    native float it returns `Float`. `Integer` is `Hashable`: equal values hash
    equally whatever their construction, so it works as a `Dict` or `Set` key.

    Limitations:
        A typed native integer cannot be the left operand of a comparison
        (`native < x` does not compile); write `x > native` or `Integer(native) < x`.
        An Integer destination cannot change family in place: `x /= y` does not
        compile. Very large powers and shifts can raise checked size errors.
    """

    comptime _Shared = _SharedMagnitude
    comptime _Storage = Variant[Int64, UInt64, _StaticInteger, Self._Shared]
    var _storage: Self._Storage

    @implicit
    def __init__(out self, value: Int = 0):
        """An Integer from a native `Int`; `Integer()` is zero.

        Args:
            value: The value.
        """
        self._storage = Self._Storage(Int64(value))

    @implicit
    def __init__(out self, value: Int64):
        """An Integer from a native `Int64`."""
        self._storage = Self._Storage(value)

    @implicit
    def __init__(out self, value: UInt64):
        """An Integer from a native `UInt64`, including values above `Int64.MAX`."""
        if value <= UInt64(Int64.MAX):
            self._storage = Self._Storage(Int64(value))
        else:
            self._storage = Self._Storage(value)

    @implicit
    def __init__[dtype: DType](out self, value: SIMD[dtype, 1]):
        """An Integer from any native integral scalar of at most 64 bits."""
        comptime assert dtype.is_integral() and bit_width_of[dtype]() <= 64, (
            "Integer accepts native integral types up to 64 bits; use text or"
            " an integer literal for wider values"
        )
        comptime if dtype.is_unsigned():
            self = Self(UInt64(value))
        else:
            self = Self(Int64(value))

    @implicit
    def __init__(out self, value: IntLiteral):
        """An Integer from an integer literal of any width, without narrowing it."""
        comptime literal = type_of(value)()
        comptime if -(1 << 63) <= literal and literal < (1 << 63):
            self._storage = Self._Storage(Int64(literal))
        else:
            self._storage = Self._Storage(_StaticInteger(value))

    def __init__(
        out self,
        text: String,
        base: Int = 10,
        *,
        allow_prefix: Bool = False,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises:
        """Parse an Integer from text.

        Text is decimal by default. Bases 2 through 36 use case-insensitive digits,
        and `base=0` detects a `0b`, `0o` or `0x` prefix.

        Args:
            text: The digits, with an optional sign.
            base: The radix, from 2 to 36, or 0 to read a prefix.
            allow_prefix: Accept a prefix that matches `base`.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Raises:
            When the text is not a valid integer in the base, naming the byte offset,
            or when it exceeds `limits`.
        """
        self = Self.parse(
            text,
            base,
            allow_prefix=allow_prefix,
            allow_whitespace=allow_whitespace,
            allow_underscores=allow_underscores,
            limits=limits,
        )

    @staticmethod
    def _placeholder() -> Self:
        return Self(0)

    def __init__(out self, *, copy: Self):
        self._storage = copy._storage

    @staticmethod
    def _from_operand[R: _IntegerOperand](rhs: R) -> Self:
        comptime if R == Self:
            return rebind[Self](rhs)
        else:
            var count = rhs._word_count()
            if count <= 2:
                var magnitude = UInt64(rhs._word(0)) | (
                    UInt64(rhs._word(1)) << 32
                )
                if not rhs._negative() or magnitude <= UInt64(1) << 63:
                    return Self._from_magnitude(magnitude, rhs._negative())
            var words = List[UInt32](length=count, fill=0)
            for i in range(count):
                words[i] = rhs._word(i)
            return Self._from_words(words^, rhs._negative())

    @staticmethod
    def _from_words(var words: List[UInt32], negative: Bool) -> Self:
        while len(words) and words[len(words) - 1] == 0:
            _ = words.pop()
        if not len(words):
            return Self(0)
        if len(words) <= 2:
            var magnitude = UInt64(words[0])
            if len(words) == 2:
                magnitude |= UInt64(words[1]) << 32
            if magnitude <= UInt64(Int64.MAX):
                var small = Int64(magnitude)
                return Self(-small if negative else small)
            if negative and magnitude == UInt64(1) << 63:
                return Self(Int64.MIN)
        # Cancellation can leave a small multiword result in a huge allocation.
        # Compact only disproportionate spare capacity; inline results returned
        # above need no replacement allocation.
        if words.capacity() > max(128, len(words) * 4):
            var compact = List[UInt32](length=len(words), fill=0)
            for i in range(len(words)):
                compact[i] = words[i]
            words = compact^
        var result = Self()
        result._storage = Self._Storage(
            Self._Shared(_LargeInteger(negative, words^))
        )
        return result

    @staticmethod
    def _from_span(words: Span[mut=False, UInt32, _], negative: Bool) raises -> Self:
        """A canonical Integer holding a copy of these magnitude words: inline up
        to 64 bits, otherwise one allocation with the words in it."""
        var used = len(words)
        while used and words.unsafe_get(used - 1) == 0:
            used -= 1
        if used <= 2:
            var magnitude = UInt64(words.unsafe_get(0)) if used else UInt64(0)
            if used == 2:
                magnitude |= UInt64(words.unsafe_get(1)) << 32
            if magnitude <= UInt64(Int64.MAX) or (negative and magnitude == UInt64(1) << 63):
                return Self._from_magnitude(magnitude, negative)
        var owner = Self._Shared.uninitialized(used, negative)
        var target = owner[].words.unsafe_ptr()
        for i in range(used):
            target.unsafe_offset(i)[] = words.unsafe_get(i)
        return Self._from_product[False](owner^, used)

    @staticmethod
    @always_inline
    def _from_product[compact: Bool = True](var owner: Self._Shared, used: Int) -> Self:
        var negative = owner[].negative
        if not used:
            return Self(0)
        if used <= 2:
            var magnitude = UInt64(owner[].words[0])
            if used == 2:
                magnitude |= UInt64(owner[].words[1]) << 32
            if magnitude <= UInt64(Int64.MAX) or (negative and magnitude == UInt64(1) << 63):
                return Self._from_magnitude(magnitude, negative)
        comptime if compact:
            if owner[].words.capacity() > 128 and (owner[].words.capacity() - 1) // 4 >= used:
                var words = List[UInt32](unsafe_uninit_length=used)
                for i in range(used):
                    words[i] = owner[].words[i]
                return Self._from_words(words^, negative)
        owner[].words._length = used
        var result = Self()
        result._storage = Self._Storage(owner^)
        return result

    def _negative(self) -> Bool:
        if self._storage.isa[Int64]():
            return self._storage[Int64] < 0
        if self._storage.isa[UInt64]():
            return False
        if self._storage.isa[_StaticInteger]():
            return self._storage[_StaticInteger]._negative()
        return self._storage[Self._Shared].ptr()[].negative

    def _word_count(self) -> Int:
        if self._storage.isa[UInt64]():
            return 2
        if self._storage.isa[_StaticInteger]():
            return self._storage[_StaticInteger]._word_count()
        if self._storage.isa[Int64]():
            var magnitude = _unsigned_magnitude(self._storage[Int64])
            if magnitude == 0:
                return 0
            return 2 if magnitude >> 32 else 1
        return len(self._storage[Self._Shared].ptr()[].words)

    def _word(self, index: Int) -> UInt32:
        if index < 0:
            return 0
        if self._storage.isa[_StaticInteger]():
            return self._storage[_StaticInteger]._word(index)
        if self._storage.isa[UInt64]():
            if index >= 2:
                return 0
            return UInt32(
                (self._storage[UInt64] >> UInt64(index * 32)) & 0xFFFFFFFF
            )
        if self._storage.isa[Int64]():
            if index >= 2:
                return 0
            var magnitude = _unsigned_magnitude(self._storage[Int64])
            return UInt32((magnitude >> UInt64(index * 32)) & 0xFFFFFFFF)
        if index >= len(self._storage[Self._Shared].ptr()[].words):
            return 0
        return self._storage[Self._Shared].ptr()[].words[index]

    def _words_copy(self) -> List[UInt32]:
        var small = self._inline_words()
        var source = self._words_span(small)
        var words = List[UInt32](unsafe_uninit_length=len(source))
        for i in range(len(source)):
            words.unsafe_ptr().unsafe_offset(i).unsafe_write(source.unsafe_get(i))
        return words^

    def _inline_words(self) -> Array[UInt32, 2]:
        var words = Array[UInt32, 2](fill=0)
        if not self._storage.isa[Self._Shared]() and not self._storage.isa[_StaticInteger]():
            words[0] = self._word(0)
            words[1] = self._word(1)
        return words^

    def _words_span(
        self, small: Array[UInt32, 2]
    ) -> Span[UInt32, origin_of(self, small)]:
        # A borrowed self keeps its shared/static owner alive; small supplies the
        # inline alternative. Neither origin escapes the arithmetic call.
        if self._storage.isa[Self._Shared]():
            return Span(
                unsafe_ptr=self._storage[Self._Shared].ptr()[].words.unsafe_ptr()
                    .unsafe_origin_cast[origin_of(self, small)](),
                length=self._word_count(),
            )
        if self._storage.isa[_StaticInteger]():
            return Span(
                unsafe_ptr=self._storage[_StaticInteger].data.unsafe_offset(3)
                    .unsafe_origin_cast[origin_of(self, small)](),
                length=self._word_count(),
            )
        return Span(
            unsafe_ptr=Span(small).unsafe_ptr().unsafe_origin_cast[origin_of(self, small)](),
            length=self._word_count(),
        )

    @staticmethod
    def _from_budgeted_words(
        var words: List[UInt32],
        negative: Bool,
        mut budget: _ConversionBudget,
        element: Int = -1,
    ) raises -> Self:
        if budget.bounded_allocation():
            while len(words) and words[len(words) - 1] == 0:
                _ = words.pop()
            var inline_value = len(words) == 0
            if 0 < len(words) <= 2:
                var magnitude = UInt64(words[0])
                if len(words) == 2:
                    magnitude |= UInt64(words[1]) << 32
                inline_value = magnitude <= UInt64(Int64.MAX) or (
                    negative and magnitude == UInt64(1) << 63
                )
            if not inline_value:
                if words.capacity() > max(128, len(words) * 4):
                    budget.allocate(len(words), 4, element)
                budget.allocate(size_of[Self._Shared._inner_type](), 1, element)
        return Self._from_words(words^, negative)

    def _compare_magnitude(self, rhs: Self) -> Int:
        var small_a = self._inline_words()
        var small_b = rhs._inline_words()
        return _compare_words(self._words_span(small_a), rhs._words_span(small_b))

    def _compare(self, rhs: Self) -> Int:
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            var a = self._storage[Int64]
            var b = rhs._storage[Int64]
            return -1 if a < b else Int(a > b)
        if self._negative() != rhs._negative():
            return -1 if self._negative() else 1
        var order = self._compare_magnitude(rhs)
        return -order if self._negative() else order

    def __eq__[C: _ComplexComparison](self, rhs: C) raises -> Bool:
        return rhs._equals_real(self)

    def __ne__[C: _ComplexComparison](self, rhs: C) raises -> Bool:
        return not rhs._equals_real(self)

    def __eq__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 0)

    def __ne__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 1)

    def __lt__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 4)

    def __le__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 5)

    def __gt__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 2)

    def __ge__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_integer(self, 3)

    def __eq__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 0)

    def __ne__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 1)

    def __lt__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 4)

    def __le__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 5)

    def __gt__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 2)

    def __ge__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_integer(self, 3)

    def __eq__(self, rhs: Self) -> Bool:
        return self._compare(rhs) == 0

    def __ne__(self, rhs: Self) -> Bool:
        return self._compare(rhs) != 0

    def __eq__(self, rhs: Rational) -> Bool:
        return rhs == self

    def __ne__(self, rhs: Rational) -> Bool:
        return rhs != self

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the canonical value, never its storage or spare capacity.

        Hashes are in-process keys, not persistent or cryptographic identifiers.
        Native inputs to Dict[Integer, ...] are converted to Integer keys.
        """
        self._negative().__hash__(hasher)
        self._word_count().__hash__(hasher)
        for i in range(self._word_count()):
            self._word(i).__hash__(hasher)

    def __lt__(self, rhs: Self) -> Bool:
        return self._compare(rhs) < 0

    def __le__(self, rhs: Self) -> Bool:
        return self._compare(rhs) <= 0

    def __gt__(self, rhs: Self) -> Bool:
        return self._compare(rhs) > 0

    def __ge__(self, rhs: Self) -> Bool:
        return self._compare(rhs) >= 0

    def __lt__(self, rhs: Rational) raises -> Bool:
        return rhs > self

    def __le__(self, rhs: Rational) raises -> Bool:
        return rhs >= self

    def __gt__(self, rhs: Rational) raises -> Bool:
        return rhs < self

    def __ge__(self, rhs: Rational) raises -> Bool:
        return rhs <= self

    # Keep native/literal comparisons unambiguous when both exact families can
    # accept the RHS implicitly. All conversions still go through Integer.
    def __eq__(self, rhs: Int) -> Bool:
        return self == Self(rhs)

    def __ne__(self, rhs: Int) -> Bool:
        return self != Self(rhs)

    def __lt__(self, rhs: Int) -> Bool:
        return self < Self(rhs)

    def __le__(self, rhs: Int) -> Bool:
        return self <= Self(rhs)

    def __gt__(self, rhs: Int) -> Bool:
        return self > Self(rhs)

    def __ge__(self, rhs: Int) -> Bool:
        return self >= Self(rhs)

    def __eq__(self, rhs: IntLiteral) -> Bool:
        return self == Self(rhs)

    def __ne__(self, rhs: IntLiteral) -> Bool:
        return self != Self(rhs)

    def __lt__(self, rhs: IntLiteral) -> Bool:
        return self < Self(rhs)

    def __le__(self, rhs: IntLiteral) -> Bool:
        return self <= Self(rhs)

    def __gt__(self, rhs: IntLiteral) -> Bool:
        return self > Self(rhs)

    def __ge__(self, rhs: IntLiteral) -> Bool:
        return self >= Self(rhs)

    def __eq__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self == Self(rhs)

    def __ne__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self != Self(rhs)

    def __lt__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self < Self(rhs)

    def __le__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self <= Self(rhs)

    def __gt__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self > Self(rhs)

    def __ge__[dtype: DType](self, rhs: SIMD[dtype, 1]) -> Bool:
        return self >= Self(rhs)

    def __bool__(self) -> Bool:
        return self._word_count() != 0

    @always_inline
    def _is_one(self) -> Bool:
        """Whether this is 1; canonical storage keeps small values inline as Int64."""
        return self._storage.isa[Int64]() and self._storage[Int64] == 1

    def sign(self) -> Int:
        """The sign: -1, 0 or 1.

        Returns:
            -1 for a negative value, 0 for zero, 1 for a positive value.
        """
        return -1 if self._negative() else Int(self._word_count() != 0)

    def magnitude_bit_length(self) -> Int:
        """The number of bits in the absolute value.

        Returns:
            The bit length of `abs(self)`; zero has length 0.
        """
        var count = self._word_count()
        if not count:
            return 0
        return (
            (count - 1) * 32
            + 32
            - Int(count_leading_zeros(self._word(count - 1)))
        )

    def _bounded_count(self, limit: Int, operation: Int) raises -> Int:
        if self._word_count() > 2:
            _count_error(operation, too_large=True)
        var value = UInt64(self._word(0)) | (UInt64(self._word(1)) << 32)
        if value > UInt64(limit):
            _count_error(operation, too_large=True)
        return Int(value)

    def _bitwise(self, rhs: Self, operation: Int) raises -> Self:
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            var a = self._storage[Int64]
            var b = rhs._storage[Int64]
            return Self(
                a & b if operation
                == 0 else a | b if operation
                == 1 else a ^ b if operation
                == 2 else ~a
            )
        var small_a = self._inline_words()
        var small_b = rhs._inline_words()
        var a = self._words_span(small_a)
        var b = rhs._words_span(small_b)
        # One extra word holds the infinite sign extension.
        var count = _checked_count(_checked_sum(max(len(a), len(b)), 1), 4)
        var words = _ResultWords(count, zeroed=False)
        var negative = _bitwise_words(
            a, b, self._negative(), rhs._negative(), operation, words.ptr(), count
        )
        return words^.finish(negative)

    def __invert__(self) raises -> Self:
        return self._bitwise(Self(0), 3)

    def __lshift__(self, rhs: Self) raises -> Self:
        if rhs._negative():
            _count_error(4)
        if not self or not rhs:
            return self
        var bits = self.magnitude_bit_length()
        return self._shifted_left(rhs._bounded_count(_MAX_RESULT_BITS - bits, 4), bits)

    def _shifted_left(self, shift: Int, bits: Int) raises -> Self:
        """self * 2**shift for a nonzero self of `bits` bits and a count its
        caller has bounded."""
        if bits + shift <= 63 and self._storage.isa[Int64]():
            return Self._from_magnitude(
                _unsigned_magnitude(self._storage[Int64]) << UInt64(shift),
                self._negative(),
            )
        var whole = shift // 32
        var part = UInt64(shift % 32)
        var count = _checked_count(
            _checked_sum(_checked_sum(self._word_count(), whole), 1), 4
        )
        # Short operands take the native path above, so this result has at least
        # 64 bits and almost always more.
        var words = _ResultWords[wide=True](count, zeroed=False)
        for i in range(whole):
            words.ptr().unsafe_offset(i)[] = 0
        var small = self._inline_words()
        _shift_left_words(self._words_span(small), whole, Int(part), words.ptr())
        return words^.finish(self._negative())

    def __rshift__(self, rhs: Self) raises -> Self:
        if rhs._negative():
            _count_error(5)
        if not self or not rhs:
            return self
        if rhs >= Self(self.magnitude_bit_length()):
            return Self(-1 if self._negative() else 0)
        return self._shifted_right(rhs._bounded_count(self.magnitude_bit_length(), 5))

    def _shifted_right(self, shift: Int) raises -> Self:
        """floor(self / 2**shift) for a nonzero self and 0 < shift < its bit length."""
        if self._storage.isa[Int64]():
            return Self(self._storage[Int64] >> Int64(shift))
        var whole = shift // 32
        var part = UInt64(shift % 32)
        var count = self._word_count() - whole
        # Floor correction can grow an all-ones quotient by one word.
        var words = _ResultWords(
            _checked_count(_checked_sum(count, Int(self._negative())), 4), zeroed=False
        )
        var small = self._inline_words()
        var source = self._words_span(small)
        var target = words.ptr()
        var discarded = False
        for i in range(whole):
            discarded |= source.unsafe_get(i) != 0
        discarded |= (
            UInt64(source.unsafe_get(whole)) & ((UInt64(1) << part) - 1)
        ) != 0
        var carry = _shift_right_words(
            source, whole, Int(part), UInt64(self._negative() and discarded), target
        )
        if self._negative():
            target.unsafe_offset(count)[] = UInt32(carry)
        return words^.finish(self._negative())

    def __pow__(self, rhs: Self) raises -> Self:
        if rhs._negative():
            _count_error(6)
        if not rhs:
            return Self(1)
        if not self:
            return Self(0)
        if self._word_count() == 1 and self._word(0) == 1:
            return Self(
                -1 if self._negative() and (rhs._word(0) & 1) != 0 else 1
            )
        if rhs == Self(1):
            return self
        var exponent = rhs._bounded_count(
            _MAX_RESULT_BITS // self.magnitude_bit_length(), 6
        )
        # From the top bit down: square, and multiply by the short base at
        # each set bit (a cube is two products, not three).
        var result = self
        var bit = 62 - Int(count_leading_zeros(UInt64(exponent)))
        while bit >= 0:
            result = result * result
            if (exponent >> bit) & 1:
                result = result * self
            bit -= 1
        return result

    def __and__(self, rhs: Self) raises -> Self:
        return self._bitwise(rhs, 0)

    def _compound_value(self, rhs: Self, operation: Int) raises -> Self:
        if operation == 0:
            return self + rhs
        if operation == 1:
            return self - rhs
        if operation == 2:
            return self * rhs
        if operation == 3:
            return self // rhs
        if operation == 4:
            return self & rhs
        if operation == 5:
            return self | rhs
        if operation == 6:
            return self ^ rhs
        if operation == 8:
            return self << rhs
        if operation == 9:
            return self >> rhs
        if operation == 10:
            return self**rhs
        return self % rhs

    def _assign(
        mut self, var rhs: Self, operation: Int, fail_stage: Int = -1
    ) raises:
        var aliases = False
        if (
            self._storage.isa[Self._Shared]()
            and rhs._storage.isa[Self._Shared]()
        ):
            aliases = Int(self._storage[Self._Shared].ptr()) == Int(
                rhs._storage[Self._Shared].ptr()
            )
        self._assign_operand(rhs, operation, fail_stage, aliases)

    def _assign_operand[
        R: _IntegerOperand
    ](
        mut self,
        rhs: R,
        operation: Int,
        fail_stage: Int = -1,
        aliases: Bool = False,
    ) raises:
        var small = rhs._word_count() <= 2 and (
            rhs._word(1) < 0x80000000
            or (
                rhs._negative()
                and rhs._word(1) == 0x80000000
                and rhs._word(0) == 0
            )
        )
        if self._storage.isa[Int64]() and small:
            var result = self._compound_value(
                Self._from_operand(rhs), operation
            )
            self = result^
            return
        var left = self._word_count()
        var right = rhs._word_count()
        var a_negative = self._negative()
        var b_negative = rhs._negative()
        var division = operation == 3 or operation == 11
        if division and not right:
            raise Error(
                "Cannot divide Integer: divisor is 0; use a nonzero divisor."
                " The destination is unchanged."
            )
        if 8 <= operation and operation <= 10 and b_negative:
            _count_error(operation - 4)
        if not right:
            if operation == 2 or operation == 4 or operation == 10:
                self = Self(1 if operation == 10 else 0)
            return
        var count = 0
        var width = _checked_sum(max(left, right), 1)
        if operation == 2:
            width = _checked_sum(left, right)
        elif operation == 8:
            if not left:
                return
            count = _operand_count(
                rhs, _MAX_RESULT_BITS - self.magnitude_bit_length(), 4
            )
            width = _checked_sum(_checked_sum(left, count // 32), 1)
        elif operation == 9:
            if _operand_at_least(rhs, self.magnitude_bit_length()):
                self = Self(-1 if a_negative else 0)
                return
            count = _operand_count(rhs, self.magnitude_bit_length(), 5)
            width = _checked_sum(left - count // 32, 1)
        elif operation == 10:
            if not left or (left == 1 and self._word(0) == 1):
                self = Self(
                    -1 if a_negative and rhs._word(0) & 1 else Int(left != 0)
                )
                return
            if right == 1 and rhs._word(0) == 1:
                return
            var bits = self.magnitude_bit_length()
            count = _operand_count(rhs, _MAX_RESULT_BITS // bits, 6)
            width = _checked_sum((bits * count + 31) // 32, 2)
        var output = _checked_sum(left, right)
        var total = _checked_sum(output, width)
        if division:
            total = _checked_sum(total, _checked_sum(output, 1))
        elif operation == 2:
            total = _checked_sum(total, _product_scratch_size(left, right, aliases))
        elif operation == 10:
            total = _checked_sum(total, _checked_sum(width, width))
            # Keep optional recursion scratch within the existing retention
            # envelope. Kernels use only recursive frames that actually fit.
            var limit = max(128, left * 16) if left <= Int.MAX // 16 else Int.MAX
            var scratch = min(_product_scratch_size(width, width), max(0, limit - total))
            total = _checked_sum(total, scratch)
        total = _checked_count(total, 4)

        if aliases and (operation == 4 or operation == 5):
            return
        # rhs is consumed, so a repeated operand may be the sole second owner.
        # Any external copy or weak handle requires independent staging.
        var reusable = (
            self._storage.isa[Self._Shared]()
            and self._storage[Self._Shared].count() == UInt64(1 + Int(aliases))
            and self._storage[Self._Shared].weak_count() == 0
        )
        var owner: Self._Shared
        if reusable:
            owner = self._storage[Self._Shared]
        else:
            # One allocation holds the header and every staged word.
            owner = Self._Shared.uninitialized(total, a_negative)
            var staged = owner[].words.unsafe_ptr()
            for i in range(total):
                staged.unsafe_offset(i)[] = 0
            var small = self._inline_words()
            var source = self._words_span(small)
            for i in range(left):
                owner[].words.unsafe_ptr().unsafe_offset(i)[] = source.unsafe_get(i)
        var negative = False
        var used = 0
        try:
            owner[].words.resize(total, 0)
            for i in range(right):
                owner[].words[left + i] = rhs._word(i)
            if fail_stage == 0:
                raise Error(
                    "Injected Integer update failure during preparation; the"
                    " destination is unchanged. Retry the operation."
                )
            # The checked layout gives disjoint slices inside one allocation.
            # Erase only their shared origin for these non-escaping kernels;
            # no resize, publication or throwing call occurs while they run.
            negative = _update_words(
                Span(
                    unsafe_ptr=owner[]
                    .words.unsafe_ptr()
                    .unsafe_origin_cast[MutAnyOrigin](),
                    length=total,
                ),
                left,
                right,
                width,
                a_negative,
                b_negative,
                operation,
                count,
                same_operand=aliases,
            )
            used = _used(owner[].words[output : output + width])
            if fail_stage == 1:
                raise Error(
                    "Injected Integer update failure after staging; the"
                    " destination is unchanged. Retry the operation."
                )
        except error:
            # The original prefix/sign is untouched until the commit below.
            if reusable:
                _finish_words(owner[].words, left, 0)
            raise error^
        if used <= 2:
            var magnitude = UInt64(0)
            if used:
                magnitude = UInt64(owner[].words[output])
            if used == 2:
                magnitude |= UInt64(owner[].words[output + 1]) << 32
            if magnitude <= UInt64(Int64.MAX) or (
                negative and magnitude == UInt64(1) << 63
            ):
                self = Self._from_magnitude(magnitude, negative)
                return
        # A coallocated tail cannot shrink in place, even after words have grown
        # into external storage. Retire disproportionate tails at publication.
        if owner.inline_capacity() > 128 and (owner.inline_capacity() - 1) // 16 >= used:
            var compact = List[UInt32](unsafe_uninit_length=used)
            for i in range(used):
                compact[i] = owner[].words[output + i]
            self = Self._from_words(compact^, negative)
            return
        _finish_words(owner[].words, used, output)
        owner[].negative = negative
        self._storage = Self._Storage(owner^)

    def __iand__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 4)

    def __and__(self, rhs: Int) raises -> Self:
        return self & Self(rhs)

    def __rand__(self, lhs: Int) raises -> Self:
        return Self(lhs) & self

    def __iand__(mut self, rhs: Int) raises:
        self &= Self(rhs)

    def __and__(self, rhs: IntLiteral) raises -> Self:
        return self & Self(rhs)

    def __rand__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) & self

    def __iand__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 4)

    def __iand__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 4)

    def __rand__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) & self

    def __or__(self, rhs: Self) raises -> Self:
        return self._bitwise(rhs, 1)

    def __ior__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 5)

    def __or__(self, rhs: Int) raises -> Self:
        return self | Self(rhs)

    def __ror__(self, lhs: Int) raises -> Self:
        return Self(lhs) | self

    def __ior__(mut self, rhs: Int) raises:
        self |= Self(rhs)

    def __or__(self, rhs: IntLiteral) raises -> Self:
        return self | Self(rhs)

    def __ror__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) | self

    def __ior__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 5)

    def __ior__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 5)

    def __ror__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) | self

    def __xor__(self, rhs: Self) raises -> Self:
        return self._bitwise(rhs, 2)

    def __ixor__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 6)

    def __xor__(self, rhs: Int) raises -> Self:
        return self ^ Self(rhs)

    def __rxor__(self, lhs: Int) raises -> Self:
        return Self(lhs) ^ self

    def __ixor__(mut self, rhs: Int) raises:
        self ^= Self(rhs)

    def __xor__(self, rhs: IntLiteral) raises -> Self:
        return self ^ Self(rhs)

    def __rxor__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) ^ self

    def __ixor__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 6)

    def __ixor__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 6)

    def __rxor__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) ^ self

    def __ilshift__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 8)

    def __lshift__(self, rhs: Int) raises -> Self:
        if rhs < 0:
            _count_error(4)
        if not self or not rhs:
            return self
        var bits = self.magnitude_bit_length()
        if rhs > _MAX_RESULT_BITS - bits:
            _count_error(4, too_large=True)
        return self._shifted_left(rhs, bits)

    def __rlshift__(self, lhs: Int) raises -> Self:
        return Self(lhs) << self

    def __ilshift__(mut self, rhs: Int) raises:
        self <<= Self(rhs)

    def __lshift__(self, rhs: IntLiteral) raises -> Self:
        # A literal count that fits an Int takes the Int path.
        comptime count = type_of(rhs)()
        comptime if 0 <= count and count <= 9223372036854775807:
            return self << Int(count)
        else:
            return self << Self(rhs)

    def __rlshift__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) << self

    def __ilshift__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 8)

    def __ilshift__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 8)

    def __rlshift__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) << self

    def __irshift__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 9)

    def __rshift__(self, rhs: Int) raises -> Self:
        if rhs < 0:
            _count_error(5)
        if not self or not rhs:
            return self
        if rhs >= self.magnitude_bit_length():
            return Self(-1 if self._negative() else 0)
        return self._shifted_right(rhs)

    def __rrshift__(self, lhs: Int) raises -> Self:
        return Self(lhs) >> self

    def __irshift__(mut self, rhs: Int) raises:
        self >>= Self(rhs)

    def __rshift__(self, rhs: IntLiteral) raises -> Self:
        # A literal count that fits an Int takes the Int path.
        comptime count = type_of(rhs)()
        comptime if 0 <= count and count <= 9223372036854775807:
            return self >> Int(count)
        else:
            return self >> Self(rhs)

    def __rrshift__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) >> self

    def __irshift__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 9)

    def __irshift__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 9)

    def __rrshift__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) >> self

    def __ipow__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 10)

    def __pow__(self, rhs: Int) raises -> Self:
        return self ** Self(rhs)

    def __rpow__(self, lhs: Int) raises -> Self:
        return Self(lhs) ** self

    def __ipow__(mut self, rhs: Int) raises:
        self **= Self(rhs)

    def __pow__(self, rhs: IntLiteral) raises -> Self:
        return self ** Self(rhs)

    def __rpow__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) ** self

    def __ipow__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 10)

    def __ipow__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 10)

    def __rpow__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) ** self

    def to_native_exact[dtype: DType](self) raises -> SIMD[dtype, 1]:
        """Convert to a native integer type, exactly.

        Parameters:
            dtype: The target: `DType.int` or a signed or unsigned 8- to 64-bit integer.

        Returns:
            The same value in the native type.

        Raises:
            When the value does not fit the type; nothing is clamped or wrapped.
        """
        comptime assert dtype.is_integral() and bit_width_of[dtype]() <= 64, (
            "to_native_exact supports native integral types up to 64 bits;"
            " keep wider values as Integer"
        )
        comptime bits = bit_width_of[dtype]()
        if self._word_count() <= 2:
            var magnitude = UInt64(self._word(0)) | (
                UInt64(self._word(1)) << 32
            )
            comptime if dtype.is_unsigned():
                if not self._negative() and magnitude <= UInt64(
                    SIMD[dtype, 1].MAX
                ):
                    return SIMD[dtype, 1](magnitude)
            else:
                var limit = UInt64(1) << UInt64(bits - 1)
                if self._negative() and magnitude <= limit:
                    if magnitude == limit:
                        return SIMD[dtype, 1].MIN
                    return SIMD[dtype, 1](-Int64(magnitude))
                if not self._negative() and magnitude < limit:
                    return SIMD[dtype, 1](magnitude)
        raise Error(
            String(
                "Cannot convert Integer to ",
                "unsigned " if dtype.is_unsigned() else "signed ",
                bits,
                "-bit integer: value is outside [",
                SIMD[dtype, 1].MIN,
                ", ",
                SIMD[dtype, 1].MAX,
                "]; keep the value as Integer or choose a wider integer type.",
            )
        )

    def __int__(self) raises -> Int:
        return self.to_native_exact[DType.int]()

    @staticmethod
    def _from_magnitude(magnitude: UInt64, negative: Bool) -> Self:
        # Inline arithmetic results reach here: magnitude <= 2^63.
        if negative:
            if magnitude == UInt64(1) << 63:
                return Self(Int64.MIN)
            return Self(-Int64(magnitude))
        if magnitude <= UInt64(Int64.MAX):
            return Self(Int64(magnitude))
        # Input UInt64 storage does not change dynamic result demotion policy:
        # two words in one allocation.
        var words = Array[UInt32, 2](fill=0)
        words[0] = UInt32(magnitude & 0xFFFFFFFF)
        words[1] = UInt32(magnitude >> 32)
        try:
            return Self._from_span(Span(words), False)
        except:
            abort("Cannot allocate numeric storage: physical memory exhausted.")

    @staticmethod
    def _from_wide_magnitude(magnitude: UInt64, negative: Bool) raises -> Self:
        """Any 64-bit magnitude with a sign, in at most one allocation."""
        if magnitude <= UInt64(1) << 63:
            return Self._from_magnitude(magnitude, negative)
        var value = Int128(magnitude)
        return Self._from_native_product(-value if negative else value)

    def _div_rem_trunc[with_quotient: Bool = True](self, rhs: Self) raises -> Tuple[Self, Self]:
        """The truncated quotient and remainder. Without `with_quotient`, the
        quotient is zero when a dividend of three or more words has a one-limb
        divisor; every other division computes it anyway, through the one
        instantiation that holds the general path, so its long division keeps
        a single call site and stays inlined."""
        comptime if not with_quotient:
            if self._word_count() > 2 and rhs._word_count() <= 2 and rhs:
                return self._divide_by_limb[False](UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32))
            return self._div_rem_trunc[True](rhs)
        if not rhs:
            raise Error(
                "Cannot divide Integer: divisor is 0; use a nonzero divisor."
                " The destination is unchanged."
            )
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            var a = _unsigned_magnitude(self._storage[Int64])
            var b = _unsigned_magnitude(rhs._storage[Int64])
            return (
                Self._from_magnitude(
                    a // b, self._negative() != rhs._negative()
                ),
                Self._from_magnitude(a % b, self._negative()),
            )
        if self._word_count() <= 2 and rhs._word_count() <= 2:
            # Magnitudes that fit 64 bits divide natively, without word lists.
            var a = UInt64(self._word(0)) | (UInt64(self._word(1)) << 32)
            var b = UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32)
            return (
                Self._from_wide_magnitude(
                    a // b, self._negative() != rhs._negative()
                ),
                Self._from_wide_magnitude(a % b, self._negative()),
            )
        if rhs._word_count() == 1 and rhs._word(0) == 1:
            return (-self if rhs._negative() else self), Self(0)
        if rhs._word_count() <= 2:
            return self._divide_by_limb[True](UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32), rhs._negative())
        var order = self._compare_magnitude(rhs)
        if order < 0:
            return Self(0), self
        if order == 0:
            return Self(-1 if self._negative() != rhs._negative() else 1), Self(
                0
            )
        var parts = _long_divide[True](
            self, rhs, 0, self._negative() != rhs._negative(), self._negative()
        )
        return (_taken(parts[0]), _taken(parts[1]))

    def _divide_by_limb[with_quotient: Bool](
        self, divisor: UInt64, divisor_negative: Bool = False, convention: Int = 1
    ) raises -> Tuple[Self, Self]:
        """Division of three or more words by a nonzero divisor of at most 64
        bits, in one pass, rounded by `_div_rem`'s convention (truncation by
        default). Floor division with operands of opposite signs, and Euclidean
        division of a negative dividend, round a nonzero remainder away from
        zero: one more in the quotient's magnitude, added in its words before
        they become an Integer, and the remainder's magnitude divisor - rest.
        The quotient cannot outgrow the dividend's words: the divisor is at
        least 2 whenever there is a remainder."""
        var small = self._inline_words()
        var words = self._words_span(small)
        var negative = self._negative()
        var away = (convention == 0 and negative != divisor_negative) or (convention == 2 and negative)
        # Floor remainders take the divisor's sign; Euclidean ones are nonnegative.
        var remainder_negative = divisor_negative if convention == 0 else (negative if convention == 1 else False)
        comptime if with_quotient:
            var result = _ResultWords(len(words), zeroed=False)
            var target = result.ptr()
            var rest = _divide_by_limb_into[True](words, divisor, target)
            if rest and away:
                var i = 0
                while True:
                    var word = target.unsafe_offset(i)[] + 1
                    target.unsafe_offset(i)[] = word
                    if word:
                        break
                    i += 1
                rest = divisor - rest
            return (
                result^.finish(negative != divisor_negative),
                Self._from_wide_magnitude(rest, remainder_negative),
            )
        else:
            # The kernel writes nothing at its target without a quotient.
            var unused = UInt32(0)
            var rest = _divide_by_limb_into[False](words, divisor, Pointer(to=unused))
            if rest and away:
                rest = divisor - rest
            return (Self(0), Self._from_wide_magnitude(rest, remainder_negative))

    def _high_product(self, rhs: Self, shift: Int) raises -> Self:
        """`floor(|self rhs| / 2**shift)`, or one less, with the sign of
        `self rhs`, for `shift >= 0`: the fixed-point product. The 64-bit digit
        diagonals more than two digits below `shift` are left out
        (`_high_product_into`), which lowers the product by less than
        `2**(shift - 64)`. The digits go to a stack block up to _STACK_LIMBS
        and the result to its own allocation, shifted on the way."""
        var a_small = self._inline_words()
        var a = self._words_span(a_small)
        var b_small = rhs._inline_words()
        var b = rhs._words_span(b_small)
        if not len(a) or not len(b):
            return Self(0)
        var omitted = max(0, shift // 64 - 2)
        var digits = (len(a) + 1) // 2 + (len(b) + 1) // 2 - omitted
        var rest = shift - 64 * omitted
        var whole = rest // 64
        if digits <= whole:
            return Self(0)
        var stack = Array[UInt64, _STACK_LIMBS](uninitialized=True)
        var heap = List[UInt64]()
        var scratch: Pointer[UInt64, MutUntrackedOrigin]
        if digits <= _STACK_LIMBS:
            scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        else:
            heap = List[UInt64](unsafe_uninit_length=_checked_count(digits, 8))
            scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        _ = _high_product_into(a, b, omitted, Span(unsafe_ptr=scratch.unsafe_bitcast[UInt32](), length=2 * digits))
        var limbs = digits - whole
        var owner = Self._Shared.uninitialized(2 * limbs, self._negative() != rhs._negative())
        var used = _store_words(scratch.unsafe_offset(whole), limbs, rest % 64, owner[].words.unsafe_ptr())
        # The scratch block lives until here.
        _ = stack^
        _ = heap^
        return Self._from_product[False](owner^, used)

    def _divide_by_word_in_place(mut self, d: UInt64) raises:
        """`self = trunc(self / d)` for a nonzero `d < 2**64`. A value whose
        words no other value shares is divided in those words, without an
        allocation; any other takes one."""
        if self._word_count() <= 2:
            var magnitude = UInt64(self._word(0)) | (UInt64(self._word(1)) << 32)
            self = Self._from_wide_magnitude(magnitude // d, self._negative())
            return
        if not (
            self._storage.isa[Self._Shared]()
            and self._storage[Self._Shared].count() == 1
            and self._storage[Self._Shared].weak_count() == 0
        ):
            var quotient, _ = self._divide_by_limb[True](d)
            self = quotient^
            return
        ref large = self._storage[Self._Shared][]
        var n = len(large.words)
        var words = large.words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        _ = _divide_by_limb_into[True](
            Span(unsafe_ptr=words.as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=n), d, words
        )
        # Three or more words over one limb leave at least one word.
        var used = n
        while words.unsafe_offset(used - 1)[] == 0:
            used -= 1
        var negative = large.negative
        if used <= 2:
            var magnitude = UInt64(words.unsafe_offset(0)[])
            if used == 2:
                magnitude |= UInt64(words.unsafe_offset(1)[]) << 32
            if magnitude <= UInt64(Int64.MAX) or (negative and magnitude == UInt64(1) << 63):
                self = Self._from_magnitude(magnitude, negative)
                return
        large.words._length = used

    def _div_rem[with_quotient: Bool = True](
        self, rhs: Self, convention: Int = 0
    ) raises -> Tuple[Self, Self]:
        # 0 = floor, 1 = truncate, 2 = Euclidean. Only the signed correction differs.
        comptime if with_quotient:
            if self._word_count() > 2 and rhs._word_count() <= 2 and rhs:
                var divisor = UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32)
                if divisor == 1:
                    # Exact: the dividend itself, sharing its words, as
                    # _div_rem_trunc gives it (cancelling a gcd of 1 lands here).
                    return (-self if rhs._negative() else self), Self(0)
                # A one-limb divisor rounds its quotient inside the division, in
                # its words, rather than through a 1024-bit Integer subtraction.
                return self._divide_by_limb[True](divisor, rhs._negative(), convention)
        # The parts stay in their tuple: unpacking copies each one (an atomic
        # increment for a heap value) and the tuple's release undoes it.
        var parts = self._div_rem_trunc[with_quotient](rhs)
        if parts[1]:
            if convention == 0 and self._negative() != rhs._negative():
                comptime if with_quotient:
                    parts[0] -= 1
                parts[1] += rhs
            elif convention == 2 and parts[1]._negative():
                if rhs._negative():
                    comptime if with_quotient:
                        parts[0] += 1
                    parts[1] -= rhs
                else:
                    comptime if with_quotient:
                        parts[0] -= 1
                    parts[1] += rhs
        return parts^

    def _gcd(self, rhs: Self) raises -> Self:
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            return Self._from_magnitude(
                _native_gcd(
                    _unsigned_magnitude(self._storage[Int64]),
                    _unsigned_magnitude(rhs._storage[Int64]),
                ),
                False,
            )
        # Magnitudes of up to two limbs take the native gcd.
        if self._word_count() <= 4 and rhs._word_count() <= 4:
            return Self._from_wide_magnitude128(
                _native_gcd(_low_words[DType.uint128](self), _low_words[DType.uint128](rhs))
            )
        # A one-limb operand: one remainder pass over the other's words, then
        # the native gcd, with no Integer arithmetic (gcd(x, 7)).
        var self_count = self._word_count()
        var rhs_count = rhs._word_count()
        if self_count > 4 and 0 < rhs_count <= 2:
            return Self._gcd_by_limb(self, rhs)
        if rhs_count > 4 and 0 < self_count <= 2:
            return Self._gcd_by_limb(rhs, self)
        # Lehmer's steps take operands up to a limb apart, of either sign and
        # order, straight from their words.
        var self_bits = self.magnitude_bit_length()
        var rhs_bits = rhs.magnitude_bit_length()
        if self_bits and rhs_bits and max(self_bits, rhs_bits) <= min(self_bits, rhs_bits) + 64:
            var small_self = self._inline_words()
            var small_rhs = rhs._inline_words()
            return _limb_gcd(self._words_span(small_self), rhs._words_span(small_rhs))
        var x = abs(self)
        var y = abs(rhs)
        if x < y:
            swap(x, y)
        if not y._word_count():
            return x
        # A much longer operand first takes one division, to the other's size.
        if x.magnitude_bit_length() > y.magnitude_bit_length() + 64:
            var remainder = x % y
            x = y
            y = remainder
            if not y._word_count():
                return x
            if x._storage.isa[Int64]() and y._storage.isa[Int64]():
                return Self._from_magnitude(
                    _native_gcd(UInt64(x._storage[Int64]), UInt64(y._storage[Int64])), False
                )
        var small_x = x._inline_words()
        var small_y = y._inline_words()
        return _limb_gcd(x._words_span(small_x), y._words_span(small_y))

    @staticmethod
    def _gcd_by_limb(long: Self, short: Self) raises -> Self:
        """gcd(long, short) for a nonzero short of at most 64 bits."""
        var d = UInt64(short._word(0)) | (UInt64(short._word(1)) << 32)
        var small = long._inline_words()
        # The kernel writes nothing at its target without a quotient.
        var unused = UInt32(0)
        var rest = _divide_by_limb_into[False](long._words_span(small), d, Pointer(to=unused))
        return Self._from_wide_magnitude(_native_gcd(d, rest), False)

    @staticmethod
    def _from_wide_magnitude128(magnitude: UInt128) raises -> Self:
        return Self._from_wide_magnitude256(UInt256(magnitude))

    @staticmethod
    def _from_wide_magnitude256(magnitude: UInt256) raises -> Self:
        """A nonnegative Integer from a native magnitude, canonically stored."""
        if magnitude <= UInt256(Int64.MAX):
            return Self(Int64(magnitude))
        var owner = Self._Shared.uninitialized(8, False)
        var words = owner[].words.span()
        for i in range(8):
            words.unsafe_get(i) = UInt32(magnitude >> UInt256(32 * i))
        return Self._from_product(owner^, _product_used(words))

    def __truediv__(self, rhs: Self) raises -> Rational:
        """Return an exact fraction; use // or div_exact for an Integer result.
        """
        if not rhs:
            raise Error(
                "Cannot divide Integer: divisor is 0; use a nonzero divisor."
                " The destination is unchanged."
            )
        return Rational(self, rhs)

    def __rtruediv__(self, lhs: Self) raises -> Rational:
        return lhs / self

    def __floordiv__(self, rhs: Self) raises -> Self:
        var parts = self._div_rem(rhs)
        return _taken(parts[0])

    def __mod__(self, rhs: Self) raises -> Self:
        var parts = self._div_rem[with_quotient=False](rhs)
        return _taken(parts[1])

    def __ifloordiv__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 3)

    def __imod__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 11)

    def __floordiv__(self, rhs: Int) raises -> Self:
        return self // Self(rhs)

    def __floordiv__(self, rhs: IntLiteral) raises -> Self:
        return self // Self(rhs)

    def __rfloordiv__(self, lhs: Int) raises -> Self:
        return Self(lhs) // self

    def __rfloordiv__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) // self

    def __rfloordiv__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) // self

    def __mod__(self, rhs: Int) raises -> Self:
        return self % Self(rhs)

    def __mod__(self, rhs: IntLiteral) raises -> Self:
        return self % Self(rhs)

    def __rmod__(self, lhs: Int) raises -> Self:
        return Self(lhs) % self

    def __rmod__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) % self

    def __rmod__[dtype: DType](self, lhs: SIMD[dtype, 1]) raises -> Self:
        return Self(lhs) % self

    def __ifloordiv__(mut self, rhs: Int) raises:
        self //= Self(rhs)

    def __ifloordiv__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 3)

    def __ifloordiv__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 3)

    def __imod__(mut self, rhs: Int) raises:
        self %= Self(rhs)

    def __imod__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 11)

    def __imod__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 11)

    def __add__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 0)

    def __sub__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 1)

    def __mul__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 2)

    def __truediv__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 3)

    def __neg__(self) -> Self:
        if self._storage.isa[Int64]() and self._storage[Int64] != Int64.MIN:
            return Self(-self._storage[Int64])
        # One allocation holding the words; the same size as self, so the size
        # checks cannot fail.
        var small = self._inline_words()
        try:
            return Self._from_span(self._words_span(small), not self._negative())
        except:
            abort("Cannot allocate numeric storage: physical memory exhausted.")

    def __abs__(self) -> Self:
        return -self if self._negative() else self

    @always_inline
    def _add(self, rhs: Self, subtract: Bool) raises -> Self:
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            var a = self._storage[Int64]
            var b = rhs._storage[Int64]
            if subtract:
                if not (
                    (b > 0 and a < Int64.MIN + b)
                    or (b < 0 and a > Int64.MAX + b)
                ):
                    return Self(a - b)
            elif not (
                (b > 0 and a > Int64.MAX - b) or (b < 0 and a < Int64.MIN - b)
            ):
                return Self(a + b)
        var right_negative = rhs._negative() != subtract
        var negative = self._negative()
        var small_a = self._inline_words()
        var small_b = rhs._inline_words()
        var a = self._words_span(small_a)
        var b = rhs._words_span(small_b)
        var length = max(len(a), len(b))
        var count = _checked_count(_checked_sum(length, 1), 4)
        var adding = negative == right_negative
        var swapped = not adding and self._compare_magnitude(rhs) < 0
        if swapped:
            negative = right_negative
        var words = _ResultWords(count, zeroed=False)
        if adding:
            _add_words(a, b, words.ptr())
        else:
            if swapped:
                _subtract_words(b, a, words.ptr())
            else:
                _subtract_words(a, b, words.ptr())
            # A difference has at most `length` words.
            words.ptr().unsafe_offset(length)[] = 0
        return words^.finish(negative)

    def __add__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 0)

    def __add__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self + Self(rhs)

    def __radd__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 0)

    def __radd__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) + self

    def __sub__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 1)

    def __sub__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self - Self(rhs)

    def __rsub__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 1)

    def __rsub__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) - self

    def __mul__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 2)

    def __mul__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self * Self(rhs)

    def __rmul__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 2)

    def __rmul__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) * self

    def __truediv__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 3)

    def __truediv__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Rational where not dtype.is_floating_point():
        return self / Self(rhs)

    def __rtruediv__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 3)

    def __rtruediv__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Rational where not dtype.is_floating_point():
        return Self(lhs) / self

    # Keep literals out of the pinned compiler's native-SIMD coercion path.
    def __add__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 0)

    def __radd__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 0)

    def __sub__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 1)

    def __rsub__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 1)

    def __mul__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 2)

    def __rmul__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 2)

    def __truediv__(self, rhs: IntLiteral) raises -> Rational:
        return self / Self(rhs)

    def __truediv__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 3)

    def __rtruediv__(self, lhs: IntLiteral) raises -> Rational:
        return Self(lhs) / self

    def __rtruediv__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 3)

    def __add__(self, rhs: Self) raises -> Self:
        return self._add(rhs, False)

    def __sub__(self, rhs: Self) raises -> Self:
        return self._add(rhs, True)

    @always_inline
    def __mul__(self, rhs: Self) raises -> Self:
        if self._storage.isa[Int64]() and rhs._storage.isa[Int64]():
            return Self._from_int128(Int128(self._storage[Int64]) * Int128(rhs._storage[Int64]))
        return self._multiply_large(rhs)

    @staticmethod
    @always_inline
    def _from_int128(value: Int128) raises -> Self:
        """An Integer from a native 128-bit value, inline when it fits an Int64."""
        if Int128(Int64.MIN) <= value <= Int128(Int64.MAX):
            return Self(Int64(value))
        return Self._from_native_product(value)

    @staticmethod
    @no_inline
    def _from_native_product(product: Int128) raises -> Self:
        var negative = product < 0
        var magnitude = UInt128(-product if negative else product)
        var owner = Self._Shared.uninitialized(4, negative)
        var words = owner[].words.span()
        words.unsafe_get(0) = UInt32(magnitude)
        words.unsafe_get(1) = UInt32(magnitude >> 32)
        words.unsafe_get(2) = UInt32(magnitude >> 64)
        words.unsafe_get(3) = UInt32(magnitude >> 96)
        return Self._from_product(owner^, _product_used(words))

    @no_inline
    def _multiply_large(self, rhs: Self) raises -> Self:
        if self._storage.isa[Self._Shared]() and rhs._storage.isa[Self._Shared]():
            ref a = self._storage[Self._Shared].ptr()[]
            ref b = rhs._storage[Self._Shared].ptr()[]
            return Self._multiply_spans(a.words.span(), b.words.span(), a.negative != b.negative)
        var small_a = self._inline_words()
        var small_b = rhs._inline_words()
        var a = self._words_span(small_a)
        var b = rhs._words_span(small_b)
        if not len(a) or not len(b):
            return Self(0)
        var negative = self._negative() != rhs._negative()
        if len(b) == 1 and b.unsafe_get(0) == 1:
            return -self if rhs._negative() else self
        if len(a) == 1 and a.unsafe_get(0) == 1:
            return -rhs if self._negative() else rhs
        return Self._multiply_spans(a, b, negative)

    @staticmethod
    @always_inline
    def _multiply_spans(
        a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _], negative: Bool,
    ) raises -> Self:
        if len(b) <= 2:
            return Self._multiply_short(a, b, negative)
        if len(a) <= 2:
            return Self._multiply_short(b, a, negative)
        return Self._multiply_full(a, b, negative)

    @staticmethod
    @no_inline
    def _multiply_full(
        a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _], negative: Bool,
    ) raises -> Self:
        var left = len(a)
        var right = len(b)
        var count = _checked_count(_checked_sum(left, right), 4)
        var same = Int(a.unsafe_ptr()) == Int(b.unsafe_ptr())
        var scratch_size = _product_scratch_size(left, right, same)
        var owner = Self._Shared.uninitialized(count, negative)
        var used = _multiply_with_scratch(a, b, owner[].words.span(), same, scratch_size)
        # Full nonzero magnitudes leave at most one unused result word.
        return Self._from_product[False](owner^, used)

    @staticmethod
    @no_inline
    def _multiply_short(
        a: Span[mut=False, UInt32, _], b: Span[mut=False, UInt32, _], negative: Bool,
    ) raises -> Self:
        # Both spans are nonempty; b contains at most one 64-bit digit.
        var count = _checked_sum(len(a), len(b))
        var owner = Self._Shared.uninitialized(count, negative)
        var used = _multiply_short_into(a, b, owner[].words.span())
        return Self._from_product[False](owner^, used)

    def __iadd__(mut self, var rhs: Self) raises:
        self._accumulate(rhs, False)

    def __isub__(mut self, var rhs: Self) raises:
        self._accumulate(rhs, True)

    def _accumulate(mut self, rhs: Self, subtract: Bool) raises:
        """`self += rhs` (or `-=`): in place when this value alone owns its
        words and the signs agree, so the magnitude only grows; otherwise as
        `self + rhs`. Either way the value is unchanged if it raises."""
        if (
            self._storage.isa[Self._Shared]()
            and self._negative() == (rhs._negative() != subtract)
            and self._storage[Self._Shared].count() == 1
            and self._storage[Self._Shared].weak_count() == 0
        ):
            var small = rhs._inline_words()
            var b = rhs._words_span(small)
            ref words = self._storage[Self._Shared][].words
            var used = len(words)
            var length = max(used, len(b))
            _ = _checked_count(_checked_sum(length, 1), 4)
            if length + 1 > words.capacity():
                words.resize(length + 1, 0)
            # The sum overwrites the left operand's own words: the kernel reads
            # each pair before writing it, and its last word lies past them.
            var target = words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            _add_words(Span(unsafe_ptr=target, length=used), b, target)
            words._length = length + Int(target.unsafe_offset(length)[] != 0)
            return
        self = self._add(rhs, subtract)

    def __imul__(mut self, var rhs: Self) raises:
        self._assign(rhs^, 2)

    def __add__(self, rhs: Int) raises -> Self:
        return self + Self(rhs)

    def __add__(self, rhs: IntLiteral) raises -> Self:
        return self + Self(rhs)

    def __radd__(self, lhs: Int) raises -> Self:
        return Self(lhs) + self

    def __radd__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) + self

    def __sub__(self, rhs: Int) raises -> Self:
        return self - Self(rhs)

    def __sub__(self, rhs: IntLiteral) raises -> Self:
        return self - Self(rhs)

    def __rsub__(self, lhs: Int) raises -> Self:
        return Self(lhs) - self

    def __rsub__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) - self

    def __mul__(self, rhs: Int) raises -> Self:
        return self * Self(rhs)

    def __mul__(self, rhs: IntLiteral) raises -> Self:
        return self * Self(rhs)

    def __rmul__(self, lhs: Int) raises -> Self:
        return Self(lhs) * self

    def __rmul__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) * self

    def __iadd__(mut self, rhs: Int) raises:
        self += Self(rhs)

    def __iadd__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 0)

    def __iadd__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 0)

    def __isub__(mut self, rhs: Int) raises:
        self -= Self(rhs)

    def __isub__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 1)

    def __isub__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 1)

    def __imul__(mut self, rhs: Int) raises:
        self *= Self(rhs)

    def __imul__(mut self, rhs: IntLiteral) raises:
        self._assign_operand(_LiteralOperand[type_of(rhs)()](), 2)

    def __imul__[dtype: DType](mut self, rhs: SIMD[dtype, 1]) raises:
        self._assign_operand(_NativeOperand(rhs), 2)

    @staticmethod
    def from_json(
        text: String, *, limits: Optional[ConversionLimits] = None
    ) raises -> Self:
        """Read an Integer from its version-1 JSON record.

        Args:
            text: A record such as `{"version":1,"family":"integer","value":"-7"}`.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The Integer the record holds.

        Raises:
            When the text is not exactly that schema, naming the byte offset.
        """
        var budget = _ConversionBudget(limits)
        var values = _read_integer_json(text, False, budget)
        return Self._parse_json_digits(values[0], budget)

    def to_json(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """Write the version-1 JSON record of this Integer.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON, with the value as a decimal string.

        Raises:
            When the output exceeds `limits`.
        """
        if limits:
            var budget = _ConversionBudget(limits)
            budget.values(1)
            budget.output(43)
            var digits = self._format_with_budget(10, False, False, budget)
            _ = _checked_count(_checked_sum(43, digits.byte_length()), 1)
            budget.string_allocation(_checked_sum(43, digits.byte_length()))
            return String(
                '{"version":1,"family":"integer","value":"', digits, '"}'
            )
        _ = _checked_count(
            _checked_sum(_json_decimal_bound(self._word_count()), 64), 1
        )
        return String('{"version":1,"family":"integer","value":"', self, '"}')

    @staticmethod
    def parse(
        text: String,
        base: Int = 10,
        *,
        allow_prefix: Bool = False,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises -> Self:
        """Parse an Integer from text; the same contract as the text constructor.

        Args:
            text: The digits, with an optional sign.
            base: The radix, from 2 to 36, or 0 to read a prefix.
            allow_prefix: Accept a prefix that matches `base`.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The parsed Integer.

        Raises:
            When the text is not a valid integer in the base, or exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        budget.input(text.byte_length())
        budget.values(1)
        var start, end, radix, negative = _integer_text(
            text,
            base,
            allow_prefix,
            allow_whitespace,
            allow_underscores,
            budget,
        )
        return Self._parse_words(text, start, end, radix, negative, budget)

    @staticmethod
    def _parse_json_digits(
        text: String, mut budget: _ConversionBudget, element: Int = -1
    ) raises -> Self:
        # The JSON reader already validated canonical decimal and counted its
        # digits/values. Only allocations remain to charge while constructing it.
        var negative = text.as_bytes()[0] == 45
        return Self._parse_words(
            text,
            Int(negative),
            text.byte_length(),
            10,
            negative,
            budget,
            element,
        )

    @staticmethod
    @always_inline
    def _parse_group(
        mut words: List[UInt32], radix: UInt64, var carry: UInt64,
        mut budget: _ConversionBudget, element: Int,
    ) raises:
        for j in range(len(words)):
            var total = UInt64(words[j]) * radix + carry
            words[j] = UInt32(total & 0xFFFFFFFF)
            carry = total >> 32
        if carry:
            _ = _checked_count(_checked_sum(len(words), 1), 4)
            budget.append(words, UInt32(carry), element)

    @staticmethod
    @no_inline
    def _parse_decimal_words(
        text: String, start: Int, end: Int, negative: Bool,
        mut budget: _ConversionBudget, element: Int, skip_decimal_point: Bool,
    ) raises -> Self:
        if not budget.bounded_allocation():
            return Self._parse_decimal_limbs(text, start, end, negative, skip_decimal_point)
        # A counted budget charges each growth of the words as it happens.
        var words = List[UInt32]()
        var bytes = text.as_bytes()
        var group = UInt64(0)
        var multiplier = UInt64(1)
        for index in range(start, end):
            if bytes[index] == 95 or (skip_decimal_point and bytes[index] == 46):
                continue
            group = group * 10 + UInt64(_text_digit(bytes[index]))
            multiplier *= 10
            if multiplier == 1_000_000_000:
                Self._parse_group(words, multiplier, group, budget, element)
                group = 0
                multiplier = 1
        if multiplier != 1:
            Self._parse_group(words, multiplier, group, budget, element)
        return Self._from_budgeted_words(words^, negative, budget, element)

    @staticmethod
    @no_inline
    def _parse_decimal_limbs(
        text: String, start: Int, end: Int, negative: Bool, skip_decimal_point: Bool,
    ) -> Self:
        """Decimal digits nineteen at a time, each step one 64-by-64-bit
        multiply-add per limb."""
        var limbs = List[UInt64](capacity=(end - start) // 19 + 2)
        var bytes = text.as_bytes()
        var group = UInt64(0)
        var multiplier = UInt64(1)
        var digits = 0
        for index in range(start, end):
            var byte = bytes[index]
            if byte == 95 or (skip_decimal_point and byte == 46):
                continue
            group = group * 10 + UInt64(_text_digit(byte))
            multiplier *= 10
            digits += 1
            if digits == 19:
                _multiply_add_limbs(limbs, multiplier, group)
                group = 0
                multiplier = 1
                digits = 0
        if digits:
            _multiply_add_limbs(limbs, multiplier, group)
        var words = List[UInt32](capacity=2 * len(limbs))
        for limb in limbs:
            words.append(UInt32(limb & 0xFFFFFFFF))
            words.append(UInt32(limb >> 32))
        while len(words) and words[len(words) - 1] == 0:
            _ = words.pop()
        return Self._from_words(words^, negative)

    @staticmethod
    def _parse_words(
        text: String,
        start: Int,
        end: Int,
        radix: Int,
        negative: Bool,
        mut budget: _ConversionBudget,
        element: Int = -1,
        skip_decimal_point: Bool = False,
    ) raises -> Self:
        if radix == 10 and end - start > 19:
            return Self._parse_decimal_words(
                text, start, end, negative, budget, element, skip_decimal_point
            )
        var words = List[UInt32]()
        var bytes = text.as_bytes()
        for index in range(start, end):
            if bytes[index] == 95 or (
                skip_decimal_point and bytes[index] == 46
            ):
                continue
            var digit = _text_digit(bytes[index])
            Self._parse_group(words, UInt64(radix), UInt64(digit), budget, element)
        return Self._from_budgeted_words(words^, negative, budget, element)

    def to_string(
        self,
        base: Int = 10,
        *,
        prefix: Bool = False,
        uppercase: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises -> String:
        """Write the exact digits in a base.

        Args:
            base: The radix, from 2 to 36.
            prefix: Write `0b`, `0o` or `0x` for bases 2, 8 and 16, after the sign.
            uppercase: Use uppercase letter digits and prefix.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The exact digits; `String(x)` is the decimal form.

        Raises:
            When the base is out of range or the output exceeds `limits`.
        """
        if base < 2 or base > 36:
            raise Error(
                String(
                    "Cannot format Integer in base ",
                    base,
                    "; choose an output base from 2 through 36.",
                )
            )
        if prefix and base != 2 and base != 8 and base != 16:
            raise Error(
                "Cannot format Integer with a prefix in this base; choose base"
                " 2, 8, or 16, or set prefix=False."
            )
        var bound = _format_bound(self._word_count())
        if limits:
            var budget = _ConversionBudget(limits)
            budget.values(1)
            return self._format_with_budget(base, prefix, uppercase, budget)
        if base == 10:
            return self._decimal()
        var words = self._words_copy()
        var digits = List[UInt8]()
        while len(words):
            var remainder = UInt64(0)
            for i in range(len(words) - 1, -1, -1):
                var total = (remainder << 32) | UInt64(words[i])
                words[i] = UInt32(total // UInt64(base))
                remainder = total % UInt64(base)
            var digit = Int(remainder)
            digits.append(
                UInt8(digit + (48 if digit < 10 else 55 if uppercase else 87))
            )
            while len(words) and words[len(words) - 1] == 0:
                _ = words.pop()
        var bytes = List[UInt8](capacity=bound)
        if self._negative():
            bytes.append(45)
        if prefix:
            bytes.append(48)
            bytes.append(_text_prefix(base, uppercase))
        if not len(digits):
            bytes.append(48)
        for i in range(len(digits) - 1, -1, -1):
            bytes.append(digits[i])
        return String(from_utf8=Span(bytes))

    def _format_with_budget(
        self,
        base: Int,
        prefix: Bool,
        uppercase: Bool,
        mut budget: _ConversionBudget,
        element: Int = -1,
    ) raises -> String:
        if not budget.bounded_output():
            return self.to_string(base, prefix=prefix, uppercase=uppercase)
        _ = _format_bound(self._word_count())
        var overhead = Int(self._negative()) + 2 * Int(prefix)
        budget.preflight(self.magnitude_bit_length(), base, overhead, element)
        budget.output(overhead, element)
        budget.allocate(self._word_count(), 4, element)
        var words = self._words_copy()
        var digits = List[UInt8]()
        if not len(words):
            budget.output_digit(element)
            budget.append(digits, UInt8(48), element)
        while len(words):
            budget.output_digit(element)
            var remainder = UInt64(0)
            for i in range(len(words) - 1, -1, -1):
                var total = (remainder << 32) | UInt64(words[i])
                words[i] = UInt32(total // UInt64(base))
                remainder = total % UInt64(base)
            var digit = Int(remainder)
            budget.append(
                digits,
                UInt8(digit + (48 if digit < 10 else 55 if uppercase else 87)),
                element,
            )
            while len(words) and words[len(words) - 1] == 0:
                _ = words.pop()
        budget.allocate(_checked_sum(len(digits), overhead), 1, element)
        var bytes = List[UInt8](
            capacity=_checked_count(_checked_sum(len(digits), overhead), 1)
        )
        if self._negative():
            bytes.append(45)
        if prefix:
            bytes.append(48)
            bytes.append(_text_prefix(base, uppercase))
        for i in range(len(digits) - 1, -1, -1):
            bytes.append(digits[i])
        budget.string_allocation(len(bytes), element)
        return String(from_utf8=Span(bytes))

    def write_to(self, mut writer: Some[Writer]):
        """Write the canonical decimal form, as `print` does.

        Args:
            writer: The destination.
        """
        if self._storage.isa[Int64]():
            writer.write(self._storage[Int64])
            return
        if self._word_count() <= 2:
            # Native formatting needs no allocation.
            if self._negative():
                writer.write("-")
            writer.write(self._low_magnitude())
            return
        writer.write(self._decimal())

    def _low_magnitude(self) -> UInt64:
        """The magnitude of a value of at most two words."""
        return UInt64(self._word(0)) | (UInt64(self._word(1)) << 32)

    def _decimal(self) -> String:
        """The canonical decimal form, built in one pass and one allocation.

        `String(x)` sizes its result by writing once to a counter and then
        writes again, so its callers in conversions use this instead.
        """
        if not self._word_count():
            return String("0")
        if self._storage.isa[Int64]():
            return String(self._storage[Int64])
        if self._word_count() <= 2:
            if self._negative():
                return String("-", self._low_magnitude())
            return String(self._low_magnitude())
        # Groups of nineteen digits, least significant first: each pass divides
        # the 64-bit limbs by 10**19 with two multiplications per limb.
        var count = self._word_count()
        var limbs = List[UInt64](capacity=(count + 1) // 2)
        for i in range((count + 1) // 2):
            limbs.append(UInt64(self._word(2 * i)) | (UInt64(self._word(2 * i + 1)) << 32))
        var groups = List[UInt64](capacity=len(limbs) * 64 // 63 + 1)
        while len(limbs):
            var rest = UInt64(0)
            for i in range(len(limbs) - 1, -1, -1):
                var step = _divide_2by1(rest, limbs[i], _TEN19, _TEN19_RECIPROCAL)
                limbs[i] = step[0]
                rest = step[1]
            groups.append(rest)
            while len(limbs) and limbs[len(limbs) - 1] == 0:
                _ = limbs.pop()
        var top = groups[len(groups) - 1]
        var top_digits = 1
        var rest = top
        while rest >= 10:
            rest //= 10
            top_digits += 1
        var sign = Int(self._negative())
        var length = sign + top_digits + 19 * (len(groups) - 1)
        var bytes = List[UInt8](length=length, fill=48)
        if sign:
            bytes[0] = 45
        var end = length
        for i in range(len(groups) - 1):
            var group = groups[i]
            for k in range(19):
                bytes[end - 1 - k] = UInt8(48 + group % 10)
                group //= 10
            end -= 19
        for k in range(top_digits):
            bytes[end - 1 - k] = UInt8(48 + top % 10)
            top //= 10
        # ASCII digits and a sign are valid UTF-8 by construction.
        return String(unsafe_from_utf8=Span(bytes))


@always_inline
def _multiply_add_limbs(mut limbs: List[UInt64], factor: UInt64, var carry: UInt64):
    """limbs = limbs * factor + carry, growing by a limb when needed."""
    for i in range(len(limbs)):
        var product = UInt128(limbs[i]) * UInt128(factor) + UInt128(carry)
        limbs[i] = UInt64(product & UInt128(UInt64.MAX))
        carry = UInt64(product >> 64)
    if carry:
        limbs.append(carry)


def div_rem_floor(a: Integer, b: Integer) raises -> Tuple[Integer, Integer]:
    """Floor division and its remainder, together.

    The quotient rounds toward negative infinity, so the remainder is zero or has
    the divisor's sign; this is the rule of `//` and `%`.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        `(q, r)` with `a == q * b + r`.

    Raises:
        When `b` is zero.
    """
    return a._div_rem(b)


comptime _STACK_LIMBS = 80


def _long_divide[remainder: Bool](
    a: Integer, b: Integer, shift: Int, quotient_negative: Bool, remainder_negative: Bool,
) raises -> Tuple[Integer, Integer, Int]:
    """Divide |a| * 2**shift by nonzero |b|: the quotient and, with `remainder`,
    the remainder (else 0); then the remainder's place against half of |b|
    when not asked for (0 none, 1 below, 2 half, 3 above).

    One scratch block (on the stack up to _STACK_LIMBS limbs) holds both
    operands, normalized, and the quotient; results go straight into their
    own allocations.
    """
    var a_small = a._inline_words()
    var a_words = a._words_span(a_small)
    var b_small = b._inline_words()
    var b_words = b._words_span(b_small)
    var b_bits = b.magnitude_bit_length()
    var n = (b_bits + 63) // 64
    var norm = 64 * n - b_bits
    var size = _checked_sum(max((_checked_sum(a.magnitude_bit_length(), _checked_sum(shift, norm)) + 63) // 64, n), 1)
    var stack = Array[UInt64, _STACK_LIMBS](uninitialized=True)
    var heap = List[UInt64]()
    var scratch: Pointer[UInt64, MutUntrackedOrigin]
    if 2 * size <= _STACK_LIMBS:
        scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        heap = List[UInt64](unsafe_uninit_length=_checked_count(2 * size, 8))
        scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var u = scratch
    var d = scratch.unsafe_offset(size)
    var q = scratch.unsafe_offset(size + n)
    _load_limbs(a_words, shift + norm, u, size)
    _load_limbs(b_words, norm, d, n)
    _divide_limbs(u, size, d, n, q)
    var limbs = size - n
    var owner = Integer._Shared.uninitialized(2 * limbs, quotient_negative)
    var used = _store_words(q, limbs, 0, owner[].words.unsafe_ptr())
    var quotient = Integer._from_product[False](owner^, used)
    var rest = Integer(0)
    var place = 0
    comptime if remainder:
        var rest_owner = Integer._Shared.uninitialized(2 * n, remainder_negative)
        var rest_used = _store_words(u, n, norm, rest_owner[].words.unsafe_ptr())
        rest = Integer._from_product[False](rest_owner^, rest_used)
    else:
        # Both operands carry the normalizing shift, so 2 * rest against b
        # compares the normalized limbs directly.
        var nonzero = False
        for i in range(n):
            nonzero = nonzero or u.unsafe_offset(i)[] != 0
        if nonzero:
            place = 2
            if u.unsafe_offset(n - 1)[] >> 63:
                place = 3
            else:
                for i in range(n - 1, -1, -1):
                    var twice = u.unsafe_offset(i)[] << 1
                    if i:
                        twice |= u.unsafe_offset(i - 1)[] >> 63
                    var limb = d.unsafe_offset(i)[]
                    if twice != limb:
                        place = 3 if twice > limb else 1
                        break
    # The scratch block lives until here.
    _ = stack^
    _ = heap^
    return (quotient^, rest^, place)


def div_rem_trunc(a: Integer, b: Integer) raises -> Tuple[Integer, Integer]:
    """Truncating division and its remainder, together.

    The quotient rounds toward zero, so the remainder is zero or has the
    dividend's sign.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        `(q, r)` with `a == q * b + r`.

    Raises:
        When `b` is zero.
    """
    return a._div_rem(b, 1)


def div_rem_euclid(a: Integer, b: Integer) raises -> Tuple[Integer, Integer]:
    """Euclidean division and its remainder, together.

    The quotient is chosen so that `0 <= r < abs(b)`.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        `(q, r)` with `a == q * b + r`.

    Raises:
        When `b` is zero.
    """
    return a._div_rem(b, 2)


@always_inline
def _taken(mut value: Integer) -> Integer:
    """Move a value out of a tuple element (or any other place), leaving zero:
    a copy would increment a heap value's reference count, and the tuple's
    release would decrement it, two atomic writes."""
    var result = Integer(0)
    swap(result, value)
    return result^


struct _ResultWords[wide: Bool = False](Movable):
    """Writable, zeroed magnitude words for one result, then the result itself.

    Up to four words live on the stack, so results that end up inline (64 bits
    or less) never touch the heap. Longer results get one allocation holding the
    header and its words together, which becomes the result's storage: no
    temporary List and no second allocation. A `wide` builder, for results its
    caller knows almost always exceed 64 bits, always allocates: building such a
    result on the stack would only add a copy.
    """
    var small: Array[UInt32, 4]
    var owner: Optional[_SharedMagnitude]
    var count: Int

    @always_inline
    def __init__(out self, count: Int, zeroed: Bool = True) raises:
        """Words for a result of at most `count` words. Unless `zeroed`, the
        caller writes every one of them."""
        self.count = count
        if Self.wide or count > 4:
            self.small = Array[UInt32, 4](uninitialized=True)
            var owner = _SharedMagnitude.uninitialized(count, False)
            if zeroed:
                var words = owner[].words.unsafe_ptr()
                for i in range(count):
                    words.unsafe_offset(i)[] = 0
            self.owner = owner^
        else:
            self.small = Array[UInt32, 4](fill=0)
            self.owner = None

    @always_inline
    def ptr(mut self) -> Pointer[UInt32, MutUntrackedOrigin]:
        if Self.wide or self.count > 4:
            return self.owner.unsafe_value()[].words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        return Pointer(to=self.small[0]).unsafe_origin_cast[MutUntrackedOrigin]()

    @always_inline
    def finish(deinit self, negative: Bool) raises -> Integer:
        if not Self.wide and self.count <= 4:
            return Integer._from_span(Span(self.small)[: self.count], negative)
        var owner = self.owner.unsafe_take()
        var words = owner[].words.unsafe_ptr()
        var used = self.count
        while used and words.unsafe_offset(used - 1)[] == 0:
            used -= 1
        owner[].negative = negative
        return Integer._from_product(owner^, used)


def _low_words[dtype: DType](value: Integer) -> Scalar[dtype]:
    """The magnitude of a value that fits `dtype` (callers check the word count)."""
    var small = value._inline_words()
    var words = value._words_span(small)
    var result = Scalar[dtype](0)
    for i in range(len(words)):
        result |= Scalar[dtype](words.unsafe_get(i)) << Scalar[dtype](32 * i)
    return result


