"""A ball's radius: a nonnegative bound with a 30-bit mantissa (the radius
format of Johansson, Arb: efficient arbitrary-precision midpoint-radius interval arithmetic, IEEE Transactions on Computers 66(8), 2017).

Every operation on a radius rounds up, except the `*_down` forms used for
lower bounds in denominators. A radius is one machine word and an exponent, so
radius arithmetic costs a few native instructions where a general Float
rounding would cost hundreds. As a Float it is exact in the 30-bit format with
the default exponent bounds.

Sums and products of two mantissas fit one 64-bit word once aligned, so they
are exact there before the one rounding; wider values reach `_up` and `_down`
as 128-bit integers and are cut to their top word with a sticky bit, which
rounds the same way.
"""

from std.bit import count_leading_zeros
from std.memory import bitcast
from ..integer.value import Integer, _unsigned_magnitude
from ..integer._square_root import _limb_square_root
from ..integer._word_math import _trailing_zero_bits
from ..float.value import Float
from ..float.context import FloatFormat
from ..float.status import NumericStatus
from ..float._rounding import _RoundedBinary
from ..float._input import _FloatInput

comptime _TOP = UInt64(1) << 30
comptime _LOW = UInt64(1) << 29


@fieldwise_init
struct _Radius(ImplicitlyCopyable):
    """`mantissa * 2**(exponent - 30)` with the mantissa 0 or in [2**29,
    2**30), or infinity."""

    var mantissa: UInt64
    var exponent: Int
    var infinite: Bool

    @staticmethod
    def zero() -> Self:
        return Self(0, 0, False)

    @staticmethod
    def infinity() -> Self:
        return Self(0, 0, True)

    @staticmethod
    def power_of_two(k: Int) -> Self:
        """Exactly `2**k`, or the nearest radius above it out of range."""
        return _clamped(_LOW, k + 1, True)

    def is_zero(self) -> Bool:
        return not self.infinite and self.mantissa == 0

    def add(self, other: Self) -> Self:
        """An upper bound of the sum."""
        if self.infinite or other.infinite:
            return Self.infinity()
        if self.mantissa == 0:
            return other
        if other.mantissa == 0:
            return self
        var high = self if self.exponent >= other.exponent else other
        var low = other if self.exponent >= other.exponent else self
        var gap = high.exponent - low.exponent
        if gap > 33:
            # The smaller term is below a sixteenth of a unit of the larger:
            # the least radius above the sum is one unit more.
            return _up_word((high.mantissa << 2) + 1, False, high.exponent - 32)
        # Below 2**63 + 2**30: exact in one word.
        return _up_word((high.mantissa << UInt64(gap)) + low.mantissa, False, low.exponent - 30)

    def add_down(self, other: Self) -> Self:
        """A lower bound of the sum of two finite radii."""
        if self.mantissa == 0:
            return other
        if other.mantissa == 0:
            return self
        var high = self if self.exponent >= other.exponent else other
        var low = other if self.exponent >= other.exponent else self
        var gap = high.exponent - low.exponent
        if gap > 33:
            return high
        return _down_word((high.mantissa << UInt64(gap)) + low.mantissa, low.exponent - 30)

    def multiply(self, other: Self) -> Self:
        """An upper bound of the product. Two mantissas in [2**29, 2**30)
        multiply to 59 or 60 bits, so the product's top 30 bits are one
        shift away, rounded up when the bits below are not all zero: the
        rounding of `_up_word`, without its search for the leading bit."""
        if self.mantissa == 0 or other.mantissa == 0:
            if (self.mantissa == 0 and not self.infinite) or (other.mantissa == 0 and not other.infinite):
                return Self.zero()
            return Self.infinity()
        var value = self.mantissa * other.mantissa
        var high = value >> 59
        var shift = UInt64(29) + high
        var mantissa = (value >> shift) + UInt64(Int((value & ((UInt64(1) << shift) - 1)) != 0))
        var exponent = self.exponent + other.exponent - 1 + Int(high)
        if mantissa == _TOP:
            mantissa = _LOW
            exponent += 1
        return _clamped(mantissa, exponent, True)

    def multiply_down(self, other: Self) -> Self:
        """A lower bound of the product of two finite radii."""
        if self.is_zero() or other.is_zero():
            return Self.zero()
        return _down_word(self.mantissa * other.mantissa, self.exponent + other.exponent - 60)

    def divide(self, other: Self) -> Self:
        """An upper bound of the quotient by a lower bound `other`."""
        if self.is_zero():
            return Self.zero()
        if self.infinite or other.is_zero():
            return Self.infinity()
        if other.infinite:
            return Self.zero()
        # A quotient of at least 33 bits, rounded up; rounding that up to 30
        # bits gives the least radius above the exact quotient, since a ceiling
        # never passes a 30-bit value.
        var numerator = self.mantissa << 33
        var quotient = numerator // other.mantissa
        if numerator % other.mantissa:
            quotient += 1
        return _up_word(quotient, False, self.exponent - other.exponent - 33)

    def sub_down(self, other: Self) -> Self:
        """A lower bound of `self - other` for finite radii, zero where that is
        not positive."""
        if other.mantissa == 0:
            return self
        if self.mantissa == 0 or other.exponent > self.exponent:
            return Self.zero()
        var gap = self.exponent - other.exponent
        if gap > 33:
            # other < 2**(e - 34): one unit less, rounded down, is below self - other.
            return _down_word((self.mantissa << 4) - 1, self.exponent - 34)
        var a = self.mantissa << UInt64(gap)
        if a <= other.mantissa:
            return Self.zero()
        return _down_word(a - other.mantissa, other.exponent - 30)

    def sub_up(self, other: Self) -> Self:
        """An upper bound of `self - other` for finite radii, zero where that is
        not positive."""
        if other.mantissa == 0:
            return self
        if self.mantissa == 0 or other.exponent > self.exponent:
            return Self.zero()
        var gap = self.exponent - other.exponent
        if gap > 33:
            return self
        var a = self.mantissa << UInt64(gap)
        if a <= other.mantissa:
            return Self.zero()
        return _up_word(a - other.mantissa, False, other.exponent - 30)

    def sqrt_down(self) -> Self:
        """A lower bound of the square root of a finite radius."""
        if self.mantissa == 0:
            return self
        # The mantissa moved to [2**62, 2**64) by s, with e - s even so that
        # the root's scale is whole: m 2**(e - 30) = (m << s) 2**64 2**(e - s - 94).
        var s = 33 if (self.exponent - 33) % 2 == 0 else 34
        var root = _limb_square_root(self.mantissa << UInt64(s))[0]
        return _down_word(root, (self.exponent - s - 94) // 2)

    def power(self, exponent: Integer) raises -> Self:
        """An upper bound of `self**exponent` for a nonnegative exponent."""
        var result = _Radius.power_of_two(0)
        var factor = self
        for bit in range(exponent.magnitude_bit_length()):
            if (exponent._word(bit >> 5) >> UInt32(bit & 31)) & 1:
                result = result.multiply(factor)
            factor = factor.multiply(factor)
        return result

    def scale2(self, k: Int) -> Self:
        """Exactly `self * 2**k`, or the nearest radius above it out of range."""
        if self.mantissa == 0 or self.infinite:
            return self
        return _clamped(self.mantissa, self.exponent + k, True)

    def compare(self, other: Self) -> Int:
        if self.infinite or other.infinite:
            return Int(self.infinite) - Int(other.infinite)
        if self.mantissa == 0 or other.mantissa == 0:
            return Int(self.mantissa != 0) - Int(other.mantissa != 0)
        if self.exponent != other.exponent:
            return -1 if self.exponent < other.exponent else 1
        return -1 if self.mantissa < other.mantissa else Int(self.mantissa > other.mantissa)

    def max(self, other: Self) -> Self:
        return other if self.compare(other) < 0 else self

    def same(self, other: Self) -> Bool:
        return self.infinite == other.infinite and self.mantissa == other.mantissa and (
            self.mantissa == 0 or self.exponent == other.exponent
        )

    def ceiling(self) raises -> Integer:
        """The least integer at or above a finite radius."""
        if self.infinite:
            raise Error("An infinite radius has no integer bound.")
        if self.mantissa == 0:
            return Integer(0)
        var shift = self.exponent - 30
        if shift >= 0:
            return Integer(self.mantissa) << shift
        if -shift >= 64:
            return Integer(1)
        var s = UInt64(-shift)
        return Integer((self.mantissa + (UInt64(1) << s) - 1) >> s)

    def to_float(self) raises -> Float:
        """The radius exactly, as a 30-bit Float with the default bounds."""
        var format = FloatFormat(30)
        if self.infinite:
            return Float(_rounded=_RoundedBinary(2, False, Integer(0), 0, format, NumericStatus()))
        if self.mantissa == 0:
            return Float(_rounded=_RoundedBinary(0, False, Integer(0), 0, format, NumericStatus()))
        return Float(_rounded=_RoundedBinary(1, False, Integer(self.mantissa), self.exponent, format, NumericStatus()))

    @staticmethod
    def upper(x: Float) raises -> Self:
        """An upper bound of `abs(x)`. A significand of exactly the format's
        precision p puts `abs(x)` in `[2**(e-1), 2**e)`: its top 64 bits t,
        bit 63 set, give the mantissa `t >> 34`, raised by one unless t's
        low 34 bits and every bit below t are zero, at exponent e."""
        if x.is_nan() or x.is_infinite():
            return Self.infinity()
        if x.is_zero():
            return Self.zero()
        var top = _normal_top(x)
        if top == 0:
            return _up_integer(x._significand, x._exponent - x.precision())
        var mantissa = top >> 34
        var exponent = x._exponent
        if (top & ((UInt64(1) << 34) - 1)) != 0 or (x.precision() > 64 and _normal_rest(x)):
            mantissa += 1
            if mantissa == _TOP:
                mantissa = _LOW
                exponent += 1
        return _clamped(mantissa, exponent, True)

    @staticmethod
    def lower(x: Float) raises -> Self:
        """A lower bound of `abs(x)`, for a finite `x`: `t >> 34` at exponent
        e, as in `upper`."""
        if x.is_zero():
            return Self.zero()
        var top = _normal_top(x)
        if top == 0:
            return _down_integer(x._significand, x._exponent - x.precision())
        return _clamped(top >> 34, x._exponent, False)

    @staticmethod
    def upper_input(x: _FloatInput) raises -> Self:
        """An upper bound of the absolute value of an exact number."""
        if x.kind >= 2:
            return Self.infinity()
        if x.kind == 0:
            return Self.zero()
        if x.denominator._is_one():
            return _up_integer(x.numerator, Int(x.scale))
        var shift = max(0, 64 + x.denominator.magnitude_bit_length() - x.numerator.magnitude_bit_length())
        var quotient = (x.numerator << shift)._div_rem_trunc(x.denominator)
        var bound = quotient[0] + Integer(1 if quotient[1] else 0)
        return _up_integer(bound, Int(x.scale) - shift)

    def covers(self, x: _FloatInput) raises -> Bool:
        """Whether `abs(x) <= self`, exactly, for a finite dyadic `x`
        (denominator 1): by exponents, then by `x`'s top 30 bits against the
        mantissa, then by the bits below them."""
        if self.infinite or x.kind == 0:
            return True
        return self._covers_magnitude(x.numerator, Int(x.scale))

    def covers_float(self, x: Float) raises -> Bool:
        """Whether `abs(x) <= self`, exactly, for a finite Float: the test of
        `covers` on its significand, so a ball's sign needs no Float subtraction."""
        if self.infinite or x.is_zero():
            return True
        return self._covers_magnitude(x._significand, x._exponent - x.precision())

    def _covers_magnitude(self, n: Integer, scale: Int) raises -> Bool:
        """Whether `abs(n) 2**scale <= self` for a nonzero n."""
        if self.mantissa == 0:
            return False
        var bits = n.magnitude_bit_length()
        # abs(x) lies in [2**(top - 1), 2**top), self in [2**(e - 1), 2**e).
        var top = scale + bits
        if top != self.exponent:
            return top < self.exponent
        var head: UInt64
        var exact = True
        if bits <= 30:
            head = n._low_magnitude() << UInt64(30 - bits)
        else:
            var word = _top_word(n)
            head = word[0] >> UInt64(64 - 30) if bits > 64 else word[0] >> UInt64(bits - 30)
            exact = _trailing_zero_bits(n) >= bits - 30
        if head != self.mantissa:
            return head < self.mantissa
        return exact

    @staticmethod
    def lower_gap(x: _FloatInput, r: Self) raises -> Self:
        """A lower bound of `abs(x) - r`, or zero where it is not positive,
        for a finite `x`, from the top 64 bits of a dyadic `x` (its 30-bit
        lower bound otherwise)."""
        if r.is_zero():
            return Self.lower_input(x)
        if r.infinite or x.kind == 0:
            return Self.zero()
        if x.denominator._is_one():
            var word = _top_word(x.numerator)
            return _gap_below(word[0], Int(x.scale) + word[1], r)
        var low = Self.lower_input(x)
        return _gap_below(low.mantissa, low.exponent - 30, r)

    @staticmethod
    def lower_gap_float(x: Float, r: Self) raises -> Self:
        """`lower_gap` for a finite Float."""
        if r.is_zero():
            return Self.lower(x)
        if r.infinite or x.is_zero():
            return Self.zero()
        var word = _top_word(x._significand)
        return _gap_below(word[0], x._exponent - x.precision() + word[1], r)

    @staticmethod
    def lower_input(x: _FloatInput) raises -> Self:
        """A lower bound of the absolute value of a finite exact number."""
        if x.kind == 0:
            return Self.zero()
        if x.denominator._is_one():
            return _down_integer(x.numerator, Int(x.scale))
        var shift = max(0, 64 + x.denominator.magnitude_bit_length() - x.numerator.magnitude_bit_length())
        var quotient = (x.numerator << shift)._div_rem_trunc(x.denominator)
        return _down_integer(quotient[0], Int(x.scale) - shift)


def _gap_below(top: UInt64, e: Int, r: _Radius) -> _Radius:
    """A lower bound of `top 2**e - r`, or zero where it is not positive, for
    a finite nonzero radius r."""
    if top == 0:
        return _Radius.zero()
    # abs(x) >= top 2**e; r = mantissa 2**f.
    var f = r.exponent - 30
    if f >= e:
        # A mantissa of 30 bits moved up 35 or more passes any 64-bit top.
        if f - e > 34:
            return _Radius.zero()
        var shifted = r.mantissa << UInt64(f - e)
        if top <= shifted:
            return _Radius.zero()
        return _down(UInt128(top - shifted), e)
    if e - f > 64:
        # r < 2**(f + 30) < 2**(e - 34), below a unit of top 2**34 at e - 34.
        return _down((UInt128(top) << 34) - 1, e - 34)
    var aligned = UInt128(top) << UInt128(e - f)
    if aligned <= UInt128(r.mantissa):
        return _Radius.zero()
    return _down(aligned - UInt128(r.mantissa), f)

@always_inline
def _normal_top(x: Float) -> UInt64:
    """The top 64 bits of a finite nonzero Float's significand, left-aligned,
    when the significand has exactly the format's precision; 0 otherwise
    (a shorter significand, or one in static storage), for the general path."""
    var p = x.precision()
    ref s = x._significand._storage
    if s.isa[Int64]():
        var m = UInt64(s[Int64])
        if p <= 63 and (m >> UInt64(p - 1)) == 1:
            return m << UInt64(64 - p)
        return 0
    if s.isa[UInt64]():
        return s[UInt64] if p == 64 else 0
    if s.isa[Integer._Shared]():
        var words = s[Integer._Shared].ptr()[].words.span()
        if p <= 64 or _span_bits(words) != p:
            return 0
        return _span_top(words, p - 64)
    return 0


def _normal_rest(x: Float) -> Bool:
    """Whether any significand bit below the top 64 is set, for a shared
    significand of exactly the format's precision p > 64."""
    return _span_sticky(x._significand._storage[Integer._Shared].ptr()[].words.span(), x.precision() - 64)


@always_inline
def _pair(a: _Radius, b: _Radius) -> Tuple[UInt64, Int, Bool]:
    """`a b` as an exact word and scale, `(0, 0, False)` for a zero factor
    (zero times infinity is zero, as in `multiply`), `infinite` otherwise
    when a factor is."""
    if (a.mantissa == 0 and not a.infinite) or (b.mantissa == 0 and not b.infinite):
        return (UInt64(0), 0, False)
    if a.infinite or b.infinite:
        return (UInt64(0), 0, True)
    return (a.mantissa * b.mantissa, a.exponent + b.exponent - 60, False)


def _products_up(a0: _Radius, b0: _Radius, a1: _Radius, b1: _Radius, a2: _Radius, b2: _Radius) -> _Radius:
    """An upper bound of `a0 b0 + a1 b1 + a2 b2`, rounded up once. Two
    mantissas in [2**29, 2**30) multiply exactly to a word in [2**58, 2**60);
    in units of half the largest scale's unit each term is below 2**61 and
    the three below 2**63, so they add in one word, a term shifted down
    adding one unit for any bits it loses. Three products and two sums, each
    normalized and rounded up, cost about three times as much."""
    var t0 = _pair(a0, b0)
    var t1 = _pair(a1, b1)
    var t2 = _pair(a2, b2)
    if t0[2] or t1[2] or t2[2]:
        return _Radius.infinity()
    var base = Int.MIN
    if t0[0]:
        base = t0[1]
    if t1[0]:
        base = max(base, t1[1])
    if t2[0]:
        base = max(base, t2[1])
    if base == Int.MIN:
        return _Radius.zero()
    var unit = base - 1
    return _up_word(_term_word(t0[0], t0[1], unit) + _term_word(t1[0], t1[1], unit) + _term_word(t2[0], t2[1], unit), False, unit)


@always_inline
def _term_word(value: UInt64, scale: Int, unit: Int) -> UInt64:
    """`value 2**scale` in units of `2**unit`, rounded up, for a scale at
    most one above the unit."""
    if not value:
        return 0
    var d = scale - unit
    if d >= 0:
        return value << UInt64(d)
    if d > -64:
        return (value >> UInt64(-d)) + UInt64(Int((value & ((UInt64(1) << UInt64(-d)) - 1)) != 0))
    return 1


def _top_word(n: Integer) -> Tuple[UInt64, Int]:
    """`(t, s)` with `t = floor(abs(n) / 2**s)` of at most 64 bits: all of a
    nonzero `n` up to 64 bits (`s = 0`), else its top 64 bits."""
    var bits = n.magnitude_bit_length()
    if bits <= 64:
        return (n._low_magnitude(), 0)
    return (_top_bits(n, bits), bits - 64)


def _clamped(mantissa: UInt64, exponent: Int, up: Bool) -> _Radius:
    """A normalized radius kept within the default exponent bounds: above them
    it is infinite (or the largest radius, rounding down); below them, the
    smallest positive radius (or zero, rounding down)."""
    if mantissa == 0:
        return _Radius.zero()
    if exponent > FloatFormat.DEFAULT_EMAX:
        return _Radius.infinity() if up else _Radius(_TOP - 1, FloatFormat.DEFAULT_EMAX, False)
    if exponent < FloatFormat.DEFAULT_EMIN:
        return _Radius(_LOW, FloatFormat.DEFAULT_EMIN, False) if up else _Radius.zero()
    return _Radius(mantissa, exponent, False)


@always_inline
def _up_word(value: UInt64, sticky: Bool, scale: Int) -> _Radius:
    """The least radius at or above `(value + f) * 2**scale`, where `0 < f < 1`
    when `sticky` (bits below the word) and `f = 0` otherwise."""
    if value == 0:
        return _Radius.zero()
    var bits = 64 - Int(count_leading_zeros(value))
    if bits <= 30:
        # Only a word cut from a wider value is sticky, and it has 64 bits.
        return _clamped(value << UInt64(30 - bits), scale + bits, True)
    var shift = UInt64(bits - 30)
    var mantissa = value >> shift
    if sticky or (value & ((UInt64(1) << shift) - 1)) != 0:
        mantissa += 1
    var exponent = scale + bits
    if mantissa == _TOP:
        mantissa = _LOW
        exponent += 1
    return _clamped(mantissa, exponent, True)


@always_inline
def _down_word(value: UInt64, scale: Int) -> _Radius:
    """The greatest radius at or below `value * 2**scale`."""
    if value == 0:
        return _Radius.zero()
    var bits = 64 - Int(count_leading_zeros(value))
    if bits <= 30:
        return _clamped(value << UInt64(30 - bits), scale + bits, False)
    return _clamped(value >> UInt64(bits - 30), scale + bits, False)


def _up(wide: UInt128, scale: Int) -> _Radius:
    """The least radius at or above `wide * 2**scale`."""
    var high = UInt64(wide >> 64)
    if high == 0:
        return _up_word(UInt64(wide), False, scale)
    var lead = UInt128(count_leading_zeros(high))
    return _up_word(UInt64((wide << lead) >> 64), UInt64(wide << lead) != 0, scale + 64 - Int(lead))


def _down(wide: UInt128, scale: Int) -> _Radius:
    """The greatest radius at or below `wide * 2**scale`."""
    var high = UInt64(wide >> 64)
    if high == 0:
        return _down_word(UInt64(wide), scale)
    var lead = UInt128(count_leading_zeros(high))
    return _down_word(UInt64((wide << lead) >> 64), scale + 64 - Int(lead))


def _top_bits(n: Integer, bits: Int) -> UInt64:
    """`floor(abs(n) / 2**(bits - 64))` for `abs(n)` of `bits > 64` bits, read
    from its words in place: kernels bound magnitudes this way at every
    fixed-point product, where shifting a copy allocated twice."""
    var small = n._inline_words()
    return _span_top(n._words_span(small), bits - 64)


@always_inline
def _span_top(words: Span[mut=False, UInt32, _], shift: Int) -> UInt64:
    """`floor(m / 2**shift)` of the magnitude m in words, for a shift that
    leaves at most 64 bits; reads within the span, unchecked."""
    var index = shift >> 5
    var count = len(words)
    var at = words.unsafe_ptr()
    var window = UInt128(0)
    for j in range(3):
        if index + j < count:
            window |= UInt128(at.unsafe_offset(index + j)[]) << UInt128(32 * j)
    return UInt64(window >> UInt128(shift & 31))


@always_inline
def _span_bits(words: Span[mut=False, UInt32, _]) -> Int:
    """The bit length of the magnitude in words, its top word nonzero."""
    var count = len(words)
    return 32 * count - Int(count_leading_zeros(words.unsafe_ptr().unsafe_offset(count - 1)[])) if count else 0


@always_inline
def _span_sticky(words: Span[mut=False, UInt32, _], shift: Int) -> Bool:
    """Whether any of the magnitude's `shift` low bits is set."""
    var index = shift >> 5
    var at = words.unsafe_ptr()
    for i in range(index):
        if at.unsafe_offset(i)[]:
            return True
    return (at.unsafe_offset(index)[] & ((UInt32(1) << UInt32(shift & 31)) - 1)) != 0


def _up_integer(n: Integer, scale: Int) raises -> _Radius:
    """The least radius at or above `abs(n) * 2**scale`; exact when `n` is,
    so a power of two with a long significand stays exact. The words are
    read in place: each `_word` read tested the storage again."""
    if n._storage.isa[Int64]():
        # One word, read with one test: a significand below 2**63.
        return _up_word(_unsigned_magnitude(n._storage[Int64]), False, scale)
    if n._storage.isa[Integer._Shared]():
        return _up_span(n._storage[Integer._Shared].ptr()[].words.span(), scale)
    var small = n._inline_words()
    return _up_span(n._words_span(small), scale)


@always_inline
def _up_span(words: Span[mut=False, UInt32, _], scale: Int) -> _Radius:
    """`_up_integer` of the magnitude in words."""
    var bits = _span_bits(words)
    if bits <= 64:
        return _up_word(_span_top(words, 0), False, scale)
    var shift = bits - 64
    return _up_word(_span_top(words, shift), _span_sticky(words, shift), scale + shift)


def _down_integer(n: Integer, scale: Int) raises -> _Radius:
    """The greatest radius at or below `abs(n) * 2**scale`."""
    if n._storage.isa[Int64]():
        return _down_word(_unsigned_magnitude(n._storage[Int64]), scale)
    if n._storage.isa[Integer._Shared]():
        return _down_span(n._storage[Integer._Shared].ptr()[].words.span(), scale)
    var small = n._inline_words()
    return _down_span(n._words_span(small), scale)


@always_inline
def _down_span(words: Span[mut=False, UInt32, _], scale: Int) -> _Radius:
    """`_down_integer` of the magnitude in words."""
    var bits = _span_bits(words)
    if bits <= 64:
        return _down_word(_span_top(words, 0), scale)
    return _down_word(_span_top(words, bits - 64), scale + bits - 64)


def _double_up(d: Float64) -> _Radius:
    """A radius at or above a nonnegative double."""
    if not (d > 0.0):
        return _Radius.zero()
    if d < 2.2250738585072014e-308:
        return _Radius.power_of_two(-1022)
    var bits = bitcast[DType.uint64](d)
    var mantissa = (bits & ((UInt64(1) << 52) - 1)) | (UInt64(1) << 52)
    return _up_word(mantissa, False, Int((bits >> 52) & 0x7FF) - 1075)
