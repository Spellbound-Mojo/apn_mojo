"""Fixed-point balls: the arithmetic inside function kernels.

A `_Fix` is the interval `(mid +/- err) * 2**-scale`: an Integer midpoint at a
binary scale, and an error bound in units of its last place. Each operation
rounds its midpoint once, down, and bounds its error by the operands' errors
plus one unit for that rounding. A kernel's enclosure is therefore rigorous by
construction: the kernel itself adds only the tails of the series it
truncates. Kernels choose the scale for the accuracy they need, and their
argument reductions keep magnitudes near 1, so a midpoint has about `scale`
bits and an operation costs one Integer product and a few native radius steps.
"""

from std.bit import count_leading_zeros
from ..integer.value import Integer
from ..integer.number_theory import isqrt
from ..integer._word_math import _trailing_zero_bits
from ..integer._division import _divide_by_limb_into
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat
from ..float._rounding import _round_ratio
from ._radius import _Radius, _up_integer, _down_integer, _up, _down
from .value import Ball
from ._arithmetic import _from_record


def _check_terms(count: Int, limit: Int, name: StaticString) raises:
    """A series of the kernels reaches its last term within `limit` terms for
    an argument in its range; past that, an internal error rather than a hang."""
    if count > limit:
        raise Error(String("apn_mojo internal error: the ", name, " series did not converge within ", limit, " terms; please report it."))


def _ulp() -> _Radius:
    return _Radius.power_of_two(0)


trait _FixedPoint(ImplicitlyCopyable, Deinitable):
    """The fixed-point arithmetic of the series cores: `_Fix` on Integers at
    any scale, and `_Wide[n]` on `n` native words where the scale fits. Each
    has the semantics of `_Fix`: an interval `(mid +/- err) * 2**-scale` whose
    error grows by the operands' errors and one unit per rounding."""

    @staticmethod
    def of_int(value: Int, scale: Int) raises -> Self:
        """A small integer, exactly."""
        ...

    def scale_of(self) -> Int:
        ...

    def relabel(self, scale: Int) -> Self:
        """The same digits and error read at another scale: a division by
        `2**(scale - old scale)`, exactly."""
        ...

    def magnitude_bits(self) -> Int:
        """The bit length of the midpoint's magnitude."""
        ...

    def add(self, other: Self) raises -> Self:
        ...

    def sub(self, other: Self) raises -> Self:
        ...

    def neg(self) raises -> Self:
        ...

    def add_error(self, units: _Radius) raises -> Self:
        ...

    def mul(self, other: Self) raises -> Self:
        ...

    def square(self) raises -> Self:
        ...

    def div_int(var self, k: Int) raises -> Self:
        """`x / k` for an integer `1 <= k < 2**63`, in x's own words when no
        other value shares them."""
        ...

    def scale2(self, k: Int) raises -> Self:
        ...

    def rescale(self, scale: Int) raises -> Self:
        ...

    def div(self, other: Self) raises -> Self:
        ...

    def sqrt(self) raises -> Self:
        ...

    def bound(self) raises -> _Radius:
        ...


def _sum_series[
    F: _FixedPoint, divisor: def(Int) thin -> Int, cumulative: Bool, alternating: Bool
](z: F, name: StaticString) raises -> F:
    """`sum(c_k z**k for k >= 0)` at z's scale, within a few units: either
    `c_0 = 1` and `c_(k+1) = c_k / divisor(k)` (cumulative), or
    `c_k = 1 / divisor(k)`, each with the sign `(-1)**k` when alternating.
    `divisor` is positive and nondecreasing with `divisor(0) = 1`.

    The sum stops at the first term whose bound is below half a unit and adds
    twice that bound for the tail, which holds while each later term is at
    most half the one before; the function checks that. It runs by
    rectangular splitting (Smith): the powers `z**i` for `i <= m`,
    `m = ceil(sqrt(terms))`, then Horner's rule in `z**m` over blocks of `m`
    terms, so about `2 sqrt(terms)` full products, and each term costs one
    division by an integer and one sum."""
    var work = z.scale_of()
    var q = z.bound().scale2(-work)
    var half = _Radius.power_of_two(-1)
    # Bounds of |c_k z**k| in units, until one falls below half a unit.
    var power = _Radius.power_of_two(work)
    var size = power
    var terms = 0
    while size.compare(half) >= 0:
        terms += 1
        _check_terms(terms, work + 64, name)
        comptime if cumulative:
            size = size.multiply(q).divide(_down(UInt128(divisor(terms - 1)), 0))
        else:
            power = power.multiply(q)
            size = power.divide(_down(UInt128(divisor(terms)), 0))
    comptime if cumulative:
        var ratio = q.divide(_down(UInt128(divisor(terms)), 0))
        if ratio.compare(half) > 0:
            raise Error(String("apn_mojo internal error: the ", name, " series shrinks too slowly for its tail bound; please report it."))
    else:
        if q.compare(half) > 0:
            raise Error(String("apn_mojo internal error: the ", name, " series shrinks too slowly for its tail bound; please report it."))
    var m = 1
    while m * m < terms:
        m += 1
    var top = (terms - 1) // m
    var highest = m if top > 0 else terms - 1
    var powers = List[F](capacity=highest + 1)
    powers.append(F.of_int(1, work))
    for i in range(1, highest + 1):
        powers.append(powers[i // 2].square() if i % 2 == 0 else powers[i - 1].mul(z))
    var total = powers[0]
    for block in range(top, -1, -1):
        var first = block * m
        var count = min(m, terms - first)
        comptime if cumulative:
            # c_(first+i) / c_first = (+/-1)**i / (divisor(first) ... divisor(first+i-1)),
            # nested from the top of the block.
            var i = count - 1
            var v = powers[i]
            if block < top:
                i = m
                v = powers[m].mul(total)
            while i > 0:
                i -= 1
                v = v^.div_int(divisor(first + i))
                comptime if alternating:
                    v = v.neg()
                v = v.add(powers[i])
            total = v
        else:
            var v = powers[m].mul(total) if block < top else F.of_int(0, work)
            for i in range(count):
                var term = powers[i].div_int(divisor(first + i))
                comptime if alternating:
                    if (first + i) % 2 == 1:
                        term = term.neg()
                v = v.add(term)
            total = v
    return total.add_error(size.scale2(1))


@fieldwise_init
struct _Fix(_FixedPoint):
    """`(mid +/- err) * 2**-scale`; `err` counts units of the last place."""

    var mid: Integer
    var err: _Radius
    var scale: Int

    @staticmethod
    def of_int(value: Int, scale: Int) raises -> Self:
        return Self(Integer(value) << scale, _Radius.zero(), scale)

    def scale_of(self) -> Int:
        return self.scale

    def relabel(self, scale: Int) -> Self:
        return Self(self.mid, self.err, scale)

    def magnitude_bits(self) -> Int:
        return self.mid.magnitude_bit_length()

    def div_int(var self, k: Int) raises -> Self:
        # Truncated, not floored: within the unit the error adds either way.
        self.mid._divide_by_word_in_place(UInt64(k))
        self.err = self.err.divide(_down(UInt128(k), 0)).add(_ulp())
        return self^

    @staticmethod
    def exact(value: Integer, scale: Int) raises -> Self:
        """An integer, exactly."""
        return Self(value << scale, _Radius.zero(), scale)

    @staticmethod
    def ratio(numerator: Integer, denominator: Integer, scale: Int) raises -> Self:
        """`numerator / denominator` for a positive denominator."""
        return Self((numerator << scale) // denominator, _ulp(), scale)

    @staticmethod
    def of_float(x: Float, scale: Int) raises -> Self:
        """A finite Float, exactly when its bits fit the scale."""
        if x.is_zero():
            return Self(Integer(0), _Radius.zero(), scale)
        var m = -x._significand if x._negative else x._significand
        var shift = x._exponent - x.precision() + scale
        if shift >= 0:
            return Self(m << shift, _Radius.zero(), scale)
        var exact = _trailing_zero_bits(x._significand) >= -shift
        return Self(m >> (-shift), _Radius.zero() if exact else _ulp(), scale)

    @staticmethod
    def of_ball(x: Ball, scale: Int) raises -> Self:
        """A finite ball, widened by the rounding of its midpoint."""
        var midpoint = Self.of_float(x._midpoint, scale)
        midpoint.err = midpoint.err.add(x._radius.scale2(scale))
        return midpoint^

    def magnitude(self) raises -> _Radius:
        """An upper bound of `abs(mid)`, in units."""
        return _up_integer(self.mid, 0)

    def bound(self) raises -> _Radius:
        """An upper bound of every point's absolute value, in units."""
        return self.magnitude().add(self.err)

    def add(self, other: Self) raises -> Self:
        return Self(self.mid + other.mid, self.err.add(other.err), self.scale)

    def sub(self, other: Self) raises -> Self:
        return Self(self.mid - other.mid, self.err.add(other.err), self.scale)

    def neg(self) raises -> Self:
        return Self(-self.mid, self.err, self.scale)

    def add_error(self, units: _Radius) raises -> Self:
        """The ball widened by `units` units of the last place."""
        return Self(self.mid, self.err.add(units), self.scale)

    def mul(self, other: Self) raises -> Self:
        # |xy - XY| <= |X| e_y + |Y| e_x + e_x e_y, then the high product,
        # within two units below.
        var mid = self.mid._high_product(other.mid, self.scale)
        var err = self.magnitude().multiply(other.err).add(other.magnitude().multiply(self.err)).add(
            self.err.multiply(other.err)
        )
        return Self(mid, err.scale2(-self.scale).add(_Radius.power_of_two(1)), self.scale)

    def square(self) raises -> Self:
        var mid = self.mid._high_product(self.mid, self.scale)
        var err = self.magnitude().multiply(self.err).scale2(1).add(self.err.multiply(self.err))
        return Self(mid, err.scale2(-self.scale).add(_Radius.power_of_two(1)), self.scale)

    def mul_integer(self, k: Integer) raises -> Self:
        """`x * k`, exactly."""
        return Self(self.mid * k, self.err.multiply(_up_integer(k, 0)), self.scale)

    def div_integer(self, k: Integer) raises -> Self:
        """`x / k` for a nonzero integer `k`."""
        return Self(self.mid // k, self.err.divide(_down_integer(k, 0)).add(_ulp()), self.scale)

    def scale2(self, k: Int) raises -> Self:
        """`x * 2**k` at the same scale."""
        if k >= 0:
            return Self(self.mid << k, self.err.scale2(k), self.scale)
        return Self(self.mid >> (-k), self.err.scale2(k).add(_ulp()), self.scale)

    def rescale(self, scale: Int) raises -> Self:
        """The same interval at another scale."""
        var shifted = self.scale2(scale - self.scale)
        shifted.scale = scale
        return shifted^

    def div(self, other: Self) raises -> Self:
        """`x / y`; the error is infinite unless `y` is certainly away from 0."""
        var b = abs(other.mid)
        var gap = b - other.err.ceiling()
        if gap.sign() <= 0:
            return Self(Integer(0), _Radius.infinity(), self.scale)
        # |x/y - X/Y| <= (|X| e_y + |Y| e_x) / (|Y| (|Y| - e_y)), then one floor.
        var mid = (self.mid << self.scale) // other.mid
        var numerator = self.magnitude().multiply(other.err).add(other.magnitude().multiply(self.err))
        var denominator = _down_integer(b, 0).multiply_down(_down_integer(gap, 0))
        return Self(mid, numerator.scale2(self.scale).divide(denominator).add(_ulp()), self.scale)

    def sqrt(self) raises -> Self:
        """The square root; the error is infinite unless `x` is certainly
        positive, or exactly 0."""
        if self.err.is_zero():
            if self.mid.sign() < 0:
                return Self(Integer(0), _Radius.infinity(), self.scale)
            return Self(isqrt(self.mid << self.scale), _ulp() if self.mid else _Radius.zero(), self.scale)
        var low = self.mid - self.err.ceiling()
        if low.sign() <= 0:
            return Self(Integer(0), _Radius.infinity(), self.scale)
        # |sqrt(x) - sqrt(X)| <= e / (sqrt(X - e) + sqrt(X)) <= e / sqrt(X),
        # at most twice the sharper bound, then one floor.
        var mid = isqrt(self.mid << self.scale)
        var err = self.err.scale2(self.scale).divide(_down_integer(mid, 0))
        return Self(mid, err.add(_ulp()), self.scale)

    def sign(self) raises -> Int:
        """1 or -1 when every point has that sign, else 0."""
        if self.err.infinite:
            return 0
        var e = self.err.ceiling()
        if self.mid > e:
            return 1
        if self.mid < -e:
            return -1
        return 0

    def to_ball(self, bits: Int) raises -> Ball:
        """The ball at `bits` bits, its midpoint rounded once to nearest."""
        if self.err.infinite:
            return Ball.indeterminate(bits)
        var record = _round_ratio(
            self.mid, Integer(1), ArithmeticContext(format=FloatFormat(bits)), scale=Int128(-self.scale)
        )
        return _from_record(record^, self.err.scale2(-self.scale), bits)


comptime _WORD = UInt128(0xFFFFFFFFFFFFFFFF)


def _wide_overflow() -> Error:
    return Error("apn_mojo internal error: a fixed-point value outgrew its words; please report it.")


comptime _UNBOUNDED = UInt64.MAX
"""A `_Wide` error past every count: the error is unbounded."""


def _saturated(units: UInt128) -> UInt64:
    return _UNBOUNDED if units >= UInt128(_UNBOUNDED) else UInt64(units)


def _units(x: _Radius) -> UInt64:
    """The least count of units at or above a radius, saturated."""
    if x.infinite or x.exponent - 30 >= 33:
        return _UNBOUNDED
    if x.mantissa == 0:
        return 0
    var shift = x.exponent - 30
    if shift >= 0:
        return x.mantissa << UInt64(shift)
    if -shift >= 64:
        return 1
    var s = UInt64(-shift)
    return (x.mantissa + (UInt64(1) << s) - 1) >> s


def _shifted_units(value: UInt128, k: Int) -> UInt128:
    """An upper bound of `value * 2**k`, saturated at `_UNBOUNDED`."""
    if value == 0:
        return 0
    var bound: UInt128
    if k >= 0:
        if k >= 64 or (value >> UInt128(127 - k)) != 0:
            return UInt128(_UNBOUNDED)
        bound = value << UInt128(k)
    elif -k >= 128:
        return 1
    else:
        var s = UInt128(-k)
        bound = (value >> s) + UInt128(Int((value & ((UInt128(1) << s) - 1)) != 0))
    return min(bound, UInt128(_UNBOUNDED))


@fieldwise_init
struct _Wide[n: Int](_FixedPoint):
    """`(mid +/- err) * 2**-scale` with `mid` in `n` words, two's complement,
    little-endian: the series cores' arithmetic without an allocation per
    step, for midpoints below `2**(64 n - 2)`. Products and quotients
    truncate toward 0 where `_Fix` floors; either rounding is within the one
    unit that every operation adds to the error, so the bounds have the same
    form as `_Fix`'s. The error is a count of units in a word, saturating at
    `_UNBOUNDED`, so bounding it costs a few native steps."""

    var words: SIMD[DType.uint64, Self.n]
    var err: UInt64
    var scale: Int

    # ---- words

    def _negative(self) -> Bool:
        return (self.words[Self.n - 1] >> 63) == 1

    @staticmethod
    def _negated(a: SIMD[DType.uint64, Self.n]) -> SIMD[DType.uint64, Self.n]:
        var result = SIMD[DType.uint64, Self.n](0)
        var carry = UInt64(1)
        comptime for i in range(Self.n):
            var v = ~a[i] + carry
            carry = UInt64(1) if carry == 1 and v == 0 else UInt64(0)
            result[i] = v
        return result

    def _magnitude(self) -> SIMD[DType.uint64, Self.n]:
        return Self._negated(self.words) if self._negative() else self.words

    @staticmethod
    def _signed(magnitude: SIMD[DType.uint64, Self.n], negative: Bool) raises -> SIMD[DType.uint64, Self.n]:
        if (magnitude[Self.n - 1] >> 62) != 0:
            raise _wide_overflow()
        return Self._negated(magnitude) if negative else magnitude

    @staticmethod
    def _bit_length(m: SIMD[DType.uint64, Self.n]) -> Int:
        comptime for k in range(Self.n):
            comptime i = Self.n - 1 - k
            if m[i] != 0:
                return 64 * i + 64 - Int(count_leading_zeros(m[i]))
        return 0

    @staticmethod
    def _top(m: SIMD[DType.uint64, Self.n]) -> Tuple[UInt128, Int]:
        """`(t, e)` with `m <= t * 2**e`, `t <= 2**64`: the top word rounded up."""
        var bits = Self._bit_length(m)
        if bits <= 64:
            return (UInt128(m[0]), 0)
        var shift = bits - 64
        var q = shift >> 6
        var o = shift & 63
        var top = m[q] if o == 0 else (m[q] >> UInt64(o)) | (m[q + 1] << UInt64(64 - o))
        var sticky = False
        for i in range(q):
            sticky = sticky or m[i] != 0
        if o != 0:
            sticky = sticky or (m[q] & ((UInt64(1) << UInt64(o)) - 1)) != 0
        return (UInt128(top) + UInt128(Int(sticky)), shift)

    @staticmethod
    def _spread(m: SIMD[DType.uint64, Self.n], e: UInt64, scale: Int) -> UInt128:
        """An upper bound of `m * e / 2**scale`, saturated: what an error of
        `e` units in one factor adds to a product with magnitude `m`."""
        if e == 0:
            return 0
        if e == _UNBOUNDED:
            return UInt128(_UNBOUNDED)
        var top = Self._top(m)
        return _shifted_units(top[0] * UInt128(e), top[1] - scale)

    @staticmethod
    def _shifted_up(a: SIMD[DType.uint64, Self.n], k: Int) raises -> SIMD[DType.uint64, Self.n]:
        """A magnitude times `2**k`, checked."""
        if Self._bit_length(a) + k > 64 * Self.n - 2:
            raise _wide_overflow()
        var q = k >> 6
        var o = k & 63
        var m = SIMD[DType.uint64, Self.n](0)
        for i in range(q, Self.n):
            var low = a[i - q]
            var lower = a[i - q - 1] if i - q - 1 >= 0 else UInt64(0)
            m[i] = low if o == 0 else (low << UInt64(o)) | (lower >> UInt64(64 - o))
        return m

    @staticmethod
    def _shifted_down(a: SIMD[DType.uint64, Self.n], k: Int) -> SIMD[DType.uint64, Self.n]:
        """A magnitude divided by `2**k`, truncated."""
        var q = k >> 6
        var o = k & 63
        var m = SIMD[DType.uint64, Self.n](0)
        for i in range(0, Self.n - q):
            var low = a[i + q]
            var high = a[i + q + 1] if i + q + 1 < Self.n else UInt64(0)
            m[i] = low if o == 0 else (low >> UInt64(o)) | (high << UInt64(64 - o))
        return m

    # ---- the interface

    @staticmethod
    def of_int(value: Int, scale: Int) raises -> Self:
        var m = SIMD[DType.uint64, Self.n](0)
        m[0] = UInt64(abs(value))
        return Self(Self._signed(Self._shifted_up(m, scale), value < 0), 0, scale)

    def scale_of(self) -> Int:
        return self.scale

    def relabel(self, scale: Int) -> Self:
        return Self(self.words, self.err, scale)

    def magnitude_bits(self) -> Int:
        return Self._bit_length(self._magnitude())

    def bound(self) raises -> _Radius:
        var top = Self._top(self._magnitude())
        var err = _Radius.infinity() if self.err == _UNBOUNDED else _up(UInt128(self.err), 0)
        return _up(top[0], top[1]).add(err)

    def add(self, other: Self) raises -> Self:
        var result = SIMD[DType.uint64, Self.n](0)
        var carry = UInt128(0)
        comptime for i in range(Self.n):
            var s = UInt128(self.words[i]) + UInt128(other.words[i]) + carry
            result[i] = UInt64(s & _WORD)
            carry = s >> 64
        var sum = Self(result, _saturated(UInt128(self.err) + UInt128(other.err)), self.scale)
        var negative = sum._negative()
        # Two's complement overflow, or a sum past the two guard bits.
        if (self._negative() == other._negative() and negative != self._negative()) or (sum._magnitude()[Self.n - 1] >> 62) != 0:
            raise _wide_overflow()
        return sum

    def sub(self, other: Self) raises -> Self:
        return self.add(other.neg())

    def neg(self) raises -> Self:
        return Self(Self._negated(self.words), self.err, self.scale)

    def add_error(self, units: _Radius) raises -> Self:
        return Self(self.words, _saturated(UInt128(self.err) + UInt128(_units(units))), self.scale)

    @staticmethod
    def _times(a: SIMD[DType.uint64, Self.n], b: SIMD[DType.uint64, Self.n], scale: Int, negative: Bool) raises -> SIMD[DType.uint64, Self.n]:
        """The truncated `a * b / 2**scale` with a sign, checked."""
        var p = SIMD[DType.uint64, 2 * Self.n](0)
        comptime for i in range(Self.n):
            var ai = UInt128(a[i])
            if ai != 0:
                var carry = UInt128(0)
                comptime for j in range(Self.n):
                    var t = UInt128(p[i + j]) + ai * UInt128(b[j]) + carry
                    p[i + j] = UInt64(t & _WORD)
                    carry = t >> 64
                p[i + Self.n] = UInt64(carry)
        var q = scale >> 6
        var o = scale & 63
        var m = SIMD[DType.uint64, Self.n](0)
        for i in range(Self.n):
            var low = p[q + i] if q + i < 2 * Self.n else UInt64(0)
            var high = p[q + i + 1] if q + i + 1 < 2 * Self.n else UInt64(0)
            m[i] = low if o == 0 else (low >> UInt64(o)) | (high << UInt64(64 - o))
        # Nothing above the result's words.
        var rest = q + Self.n
        if rest < 2 * Self.n and (p[rest] >> UInt64(o)) != 0:
            raise _wide_overflow()
        for k in range(rest + 1, 2 * Self.n):
            if p[k] != 0:
                raise _wide_overflow()
        return Self._signed(m, negative)

    def mul(self, other: Self) raises -> Self:
        # |xy - XY| <= |X| e_y + |Y| e_x + e_x e_y, then one truncation.
        var a = self._magnitude()
        var b = other._magnitude()
        var cross = _shifted_units(UInt128(self.err) * UInt128(other.err), -self.scale)
        var err = Self._spread(a, other.err, self.scale) + Self._spread(b, self.err, self.scale) + cross + 1
        if self.err == _UNBOUNDED or other.err == _UNBOUNDED:
            err = UInt128(_UNBOUNDED)
        return Self(Self._times(a, b, self.scale, self._negative() != other._negative()), _saturated(err), self.scale)

    def square(self) raises -> Self:
        var a = self._magnitude()
        var cross = _shifted_units(UInt128(self.err) * UInt128(self.err), -self.scale)
        var err = 2 * Self._spread(a, self.err, self.scale) + cross + 1
        if self.err == _UNBOUNDED:
            err = UInt128(_UNBOUNDED)
        return Self(Self._times(a, a, self.scale, False), _saturated(err), self.scale)

    def div_int(var self, k: Int) raises -> Self:
        var negative = self._negative()
        var m = self._magnitude()
        # The words in use, a whole number of limbs, divided in place.
        var used = 2 * ((Self._bit_length(m) + 63) // 64)
        if used:
            var words = Pointer(to=m).unsafe_bitcast[UInt32]().unsafe_origin_cast[MutUntrackedOrigin]()
            _ = _divide_by_limb_into[True](
                Span(unsafe_ptr=words.as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=used), UInt64(k), words
            )
        var quotient = self.err // UInt64(k)
        var err = _UNBOUNDED if self.err == _UNBOUNDED else _saturated(UInt128(quotient) + UInt128(Int(quotient * UInt64(k) != self.err)) + 1)
        return Self(Self._signed(m, negative), err, self.scale)

    def scale2(self, k: Int) raises -> Self:
        var negative = self._negative()
        var err = _UNBOUNDED if self.err == _UNBOUNDED else _saturated(_shifted_units(UInt128(self.err), k) + UInt128(Int(k < 0)))
        if k >= 0:
            return Self(Self._signed(Self._shifted_up(self._magnitude(), k), negative), err, self.scale)
        return Self(Self._signed(Self._shifted_down(self._magnitude(), -k), negative), err, self.scale)

    def rescale(self, scale: Int) raises -> Self:
        var shifted = self.scale2(scale - self.scale)
        shifted.scale = scale
        return shifted

    # ---- through _Fix: the few divisions and square roots of a kernel call

    @staticmethod
    def of_fix(x: _Fix) raises -> Self:
        if x.mid.magnitude_bit_length() > 64 * Self.n - 2:
            raise _wide_overflow()
        var m = SIMD[DType.uint64, Self.n](0)
        comptime for i in range(Self.n):
            m[i] = UInt64(x.mid._word(2 * i)) | (UInt64(x.mid._word(2 * i + 1)) << 32)
        return Self(Self._signed(m, x.mid.sign() < 0), _units(x.err), x.scale)

    def to_fix(self) raises -> _Fix:
        var m = self._magnitude()
        var words = List[UInt32](capacity=2 * Self.n)
        comptime for i in range(Self.n):
            words.append(UInt32(m[i] & 0xFFFFFFFF))
            words.append(UInt32(m[i] >> 32))
        var err = _Radius.infinity() if self.err == _UNBOUNDED else _up(UInt128(self.err), 0)
        return _Fix(Integer._from_words(words^, self._negative()), err, self.scale)

    def div(self, other: Self) raises -> Self:
        return Self.of_fix(self.to_fix().div(other.to_fix()))

    def sqrt(self) raises -> Self:
        return Self.of_fix(self.to_fix().sqrt())


def _native_words(x: _Fix) -> Int:
    """The words of the `_Wide` that runs a series core on x: 4 or 8 when
    the core's working scale fits them and x's error is below `2**32` units,
    so that no error saturates; otherwise 0, for `_Fix`."""
    if x.err.compare(_Radius.power_of_two(32)) >= 0:
        return 0
    var room = _core_room(x.scale)
    return 4 if room < 64 * 4 else (8 if room < 64 * 8 else 0)


def _core_room(scale: Int) -> Int:
    """An upper bound of the working scale of any series core called at
    `scale`, with room for its values, all below 4."""
    var root = 0
    while (root + 1) * (root + 1) <= scale:
        root += 1
    var bits = 0
    var v = scale
    while v > 0:
        bits += 1
        v >>= 1
    return scale + root + 2 * bits + 12
