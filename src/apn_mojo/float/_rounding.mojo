"""Exact-ratio binary rounding. No native floating-point arithmetic or replay."""

from std.bit import count_leading_zeros
from std.collections import Array
from ..integer.value import Integer, _long_divide, _low_words, _taken
from ..integer._division import _divide_limbs
from ..integer.division import div_rem_trunc
from ..integer._conversion import _conversion_shift, _conversion_sum, _conversion_div_rem
from ..integer._word_math import _trailing_zero_bits
from ..common.conversion import _ConversionBudget
from .context import ArithmeticContext, FloatFormat, RoundingMode
from .status import NumericStatus


@fieldwise_init
struct _RoundedBinary(ImplicitlyCopyable):
    # Classes: zero=0, finite=1, infinity=2. NaNs are not exact ratio inputs.
    var kind: Int
    var negative: Bool
    var significand: Integer
    var exponent: Int
    var format: FloatFormat
    var status: NumericStatus


def _inexact_error(operation: StaticString = "") -> Error:
    return Error(String(
        "Cannot compute ", operation if operation else "the result",
        " exactly: exact arithmetic keeps only finite binary fractions (1/3 and"
        " sqrt(2) are not). An exact reduction rounds only its final result;"
        " lift the function with exact=False to round every step.",
    ))


# Scales whose top exponents and their differences fit Int with room to
# spare, for the one-limb paths: Float exponents are within 2**62.
comptime _LIMB_SCALE = Int128(1) << 62


@always_inline
def _round_limb(
    a0: UInt64, rb: UInt64, sb: UInt64, negative: Bool, exponent: Int,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round a one-limb result: a0 holds the significand left-aligned (bit 63
    set, the low 64 - p bits clear), rb the next bit and sb the bits below it,
    for a precision p of at most 64 and the result's top exponent: at most
    one add of a unit in the last place, whose carry out is the next power of
    two. Kind -1 outside the format's exponent range, for the general path."""
    var format = context.format()
    var sh = UInt64(64 - format.precision())
    var a = a0
    var e = exponent
    var inexact = (rb | sb) != 0
    var up = False
    if inexact:
        var mode = context.rounding()
        if mode == RoundingMode.nearest_even:
            up = rb != 0 and (sb != 0 or (a & (UInt64(1) << sh)) != 0)
        else:
            up = _round_away(mode, negative)
        if up:
            a += UInt64(1) << sh
            if not a:
                a = UInt64(1) << 63
                e += 1
    if e > format.emax() or e < format.emin():
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    return _finish_round(
        _RoundedBinary(
            1, negative, Integer(a >> sh), e, format,
            NumericStatus._make(Int(inexact), _rounded_direction(negative, up) if inexact else 0),
        ),
        context,
        fail,
    )


@always_inline
def _round_native(
    value: UInt128, negative: Bool, scale: Int128, context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round a nonzero value * 2**scale to a precision of at most 63 bits in
    native arithmetic, without allocating. Kind -1 when the result leaves the
    format's exponent range: the general path handles overflow and underflow.

    The value is left-aligned once, and its top limb, round bit and sticky go
    to _round_limb: 64-bit operations, where shifts and masks of the 128-bit
    value took variable 128-bit shifts each. Inlined, like _finish_round:
    returning the record through calls costs more than computing it."""
    var format = context.format()
    var bits = 128 - Int(count_leading_zeros(value))
    var exponent = scale + Int128(bits)
    # Rounding raises the exponent by at most one.
    if exponent > Int128(format.emax()) or exponent < Int128(format.emin()) - 1:
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    var aligned = value << UInt128(128 - bits)
    var high = UInt64(aligned >> 64)
    var sh = UInt64(64 - format.precision())
    var mask = (UInt64(1) << sh) - 1
    var rb = high & (UInt64(1) << (sh - 1))
    var sb = ((high & mask) ^ rb) | UInt64(aligned)
    return _round_limb(high & ~mask, rb, sb, negative, Int(exponent), context, fail)


@always_inline
def _short_magnitude(value: Integer) -> Optional[UInt128]:
    """A nonnegative significand of at most 128 bits as a native integer."""
    if value._storage.isa[Int64]():
        return UInt128(UInt64(value._storage[Int64]))
    if value._storage.isa[UInt64]():
        return UInt128(value._storage[UInt64])
    if value._storage.isa[Integer._Shared]():
        ref words = value._storage[Integer._Shared].ptr()[].words
        if len(words) <= 4:
            var magnitude = UInt128(0)
            for i in range(len(words)):
                magnitude |= UInt128(words[i]) << UInt128(32 * i)
            return magnitude
    return None


@fieldwise_init
struct _WideRounding(ImplicitlyCopyable):
    """A value rounded to p <= 128 bits: the significand, its top exponent
    (the value lies below `2**exponent`), and whether the rounding was
    inexact and went up."""

    var magnitude: UInt128
    var exponent: Int128
    var inexact: Bool
    var up: Bool


@always_inline
def _wide_rounding(value: UInt256, p: Int, scale: Int128, mode: RoundingMode, negative: Bool) -> _WideRounding:
    """Round a nonzero `value * 2**scale` to p <= 128 bits. The value's 64-bit
    limbs are read once, and the significand, the round bit and the sticky
    bits come from them by word shifts: shifts of the 256-bit value by a
    variable count, a mask and a comparison with the half took dozens of
    instructions each."""
    var bits = 256 - Int(count_leading_zeros(value))
    var exponent = scale + Int128(bits)
    var discard = bits - p
    var magnitude: UInt128
    var inexact = False
    var up = False
    if discard > 0:
        var limbs = Array[UInt64, 6](fill=0)
        limbs[0] = UInt64(value)
        limbs[1] = UInt64(value >> 64)
        limbs[2] = UInt64(value >> 128)
        limbs[3] = UInt64(value >> 192)
        # The top p <= 128 bits, from limbs q, q + 1 and q + 2.
        var q = discard >> 6
        var r = discard & 63
        magnitude = (UInt128(limbs[q]) | (UInt128(limbs[q + 1]) << 64)) >> UInt128(r)
        if r:
            magnitude |= UInt128(limbs[q + 2]) << UInt128(128 - r)
        # The round bit, at discard - 1, and the sticky bits below it.
        var below = discard - 1
        var word = limbs[below >> 6]
        var bit = UInt64(below & 63)
        var half = (word >> bit) & 1
        var sticky = (word & ((UInt64(1) << bit) - 1)) != 0
        for i in range(below >> 6):
            sticky = sticky or limbs[i] != 0
        inexact = half != 0 or sticky
        if inexact:
            if mode == RoundingMode.nearest_even:
                up = half != 0 and (sticky or Bool(magnitude & 1))
            else:
                up = _round_away(mode, negative)
        if up:
            magnitude += 1
            # A carry out of the top bit: the next power of two.
            if p == 128 and magnitude == 0:
                magnitude = UInt128(1) << 127
                exponent += 1
            elif p < 128 and magnitude >> UInt128(p):
                magnitude >>= 1
                exponent += 1
    else:
        magnitude = UInt128(value) << UInt128(-discard)
    return _WideRounding(magnitude, exponent, inexact, up)


@always_inline
def _round_wide(
    value: UInt256, negative: Bool, scale: Int128, context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """As _round_native, for precisions of at most 128 bits (`_wide_rounding`):
    one allocation for a result past 64 bits, none below."""
    var format = context.format()
    var rounded = _wide_rounding(value, format.precision(), scale, context.rounding(), negative)
    if rounded.exponent > Int128(format.emax()) or rounded.exponent < Int128(format.emin()):
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    var magnitude = rounded.magnitude
    var significand: Integer
    if magnitude >> 64 == 0:
        # Inline: Int64 below 2**63, UInt64 up to 2**64.
        significand = Integer(UInt64(magnitude))
    else:
        var owner = Integer._Shared.uninitialized(4, False)
        var used = _put_magnitude(owner[].words.span(), magnitude)
        significand = Integer._from_product(owner^, used)
    var inexact = rounded.inexact
    return _finish_round(
        _RoundedBinary(
            1, negative, significand^, Int(rounded.exponent), format,
            NumericStatus._make(Int(inexact), _rounded_direction(negative, rounded.up) if inexact else 0),
        ),
        context,
        fail,
    )


@always_inline
def _put_magnitude(words: Span[mut=True, UInt32, _], magnitude: UInt128) -> Int:
    """A magnitude of 65 to 128 bits into four 32-bit words; how many it uses."""
    var high = UInt64(magnitude >> 64)
    var low = UInt64(magnitude)
    words.unsafe_get(0) = UInt32(low & 0xFFFFFFFF)
    words.unsafe_get(1) = UInt32(low >> 32)
    words.unsafe_get(2) = UInt32(high & 0xFFFFFFFF)
    words.unsafe_get(3) = UInt32(high >> 32)
    return 4 if high >> 32 else 3


@always_inline
def _put_limbs(value: UInt256, shift: Int, target: Pointer[mut=True, UInt64, _]):
    """Or the 64-bit limbs of value * 2**shift into zeroed limbs at target."""
    var whole = shift // 64
    var part = UInt64(shift % 64)
    comptime for i in range(4):
        var limb = UInt64(value >> UInt256(64 * i))
        if limb:
            target.unsafe_offset(i + whole)[] |= limb << part
            if part:
                target.unsafe_offset(i + whole + 1)[] |= limb >> (64 - part)


def _native_quotient(numerator: UInt256, denominator: UInt128) -> UInt128:
    """floor(numerator / denominator), for a nonzero denominator and a quotient
    below 2**128: long division on stack limbs, the divisor normalized."""
    var limbs = 2 if denominator >> 64 else 1
    var norm = Int(count_leading_zeros(denominator)) - (0 if limbs == 2 else 64)
    # Dividend (five limbs and the zero limb above it), divisor, quotient.
    var scratch = Array[UInt64, 16](fill=0)
    var u = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var d = u.unsafe_offset(6)
    var q = u.unsafe_offset(10)
    _put_limbs(numerator, norm, u)
    _put_limbs(UInt256(denominator), norm, d)
    _divide_limbs(u, 6, d, limbs, q)
    var quotient = UInt128(q.unsafe_offset(0)[]) | (UInt128(q.unsafe_offset(1)[]) << 64)
    # The scratch block lives until here.
    _ = scratch^
    return quotient


def _round_exact_binary(
    numerator: Integer, negative: Bool, scale: Int128, context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """`numerator * 2**scale` when the numerator has exactly the format's
    precision and the exponent is in range: the same significand, already
    normalized, at its exponent, with nothing to round or allocate. Kind -1
    otherwise, for the general path."""
    var format = context.format()
    var p = format.precision()
    if format._is_exact() or numerator.magnitude_bit_length() != p:
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    var exponent = scale + Int128(p)
    if exponent > Int128(format.emax()) or exponent < Int128(format.emin()):
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    return _finish_round(
        _RoundedBinary(1, negative, numerator, Int(exponent), format, NumericStatus()), context, fail
    )


def _round_quotient_native(
    numerator: UInt256, denominator: UInt256, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round (numerator / denominator) * 2**scale once, for nonzero magnitudes
    and an inexact format of at most 128 bits, without allocating: long
    division on stack limbs gives a quotient of p + 2 or p + 3 bits, its
    remainder a sticky bit below it, and _round_wide rounds that. Kind -1 when
    the result leaves the format's exponent range."""
    var p = context.format().precision()
    var numerator_bits = 256 - Int(count_leading_zeros(numerator))
    var denominator_bits = 256 - Int(count_leading_zeros(denominator))
    if not (denominator & (denominator - 1)):
        # A power-of-two denominator: a binary value, rounded without a division.
        return _round_wide(numerator, negative, scale - Int128(denominator_bits - 1), context, fail)
    # Shift the numerator up, or the denominator when the numerator is the
    # longer, so that their lengths differ by p + 2 bits.
    var shift = p + 2 - numerator_bits + denominator_bits
    var up = max(shift, 0)
    var down = max(-shift, 0)
    # The divisor is normalized: its top limb has its high bit set.
    var limbs = (denominator_bits + down + 63) // 64
    var norm = 64 * limbs - denominator_bits - down
    # One scratch block holds dividend, divisor and quotient, as in
    # _long_divide; every access after the zero fill goes through its pointer.
    var scratch = Array[UInt64, 36](fill=0)
    var u = Span(scratch).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var d = u.unsafe_offset(12)
    var q = u.unsafe_offset(24)
    _put_limbs(numerator, up + norm, u)
    _put_limbs(denominator, down + norm, d)
    # One zero limb above the dividend, as the division requires.
    var size = (numerator_bits + up + norm + 63) // 64 + 1
    _divide_limbs(u, size, d, limbs, q)
    var rest = UInt64(0)
    for i in range(limbs):
        rest |= u.unsafe_offset(i)[]
    var quotient = (
        UInt256(q.unsafe_offset(0)[]) | (UInt256(q.unsafe_offset(1)[]) << 64)
        | (UInt256(q.unsafe_offset(2)[]) << 128)
    )
    # The scratch block lives until here.
    _ = scratch^
    return _round_wide(
        (quotient << 1) | UInt256(rest != 0), negative, scale - Int128(shift) - 1, context, fail
    )


def _exponent_add(a: Int128, b: Int128) raises -> Int128:
    if (b > 0 and a > Int128.MAX - b) or (b < 0 and a < Int128.MIN - b):
        raise Error(
            "Cannot round: working exponent exceeds supported checked range;"
            " use a smaller scaling exponent. The destination is unchanged."
        )
    return a + b


def _shifted_word(value: Integer, index: Int, shift: Int) -> UInt32:
    var position = index - shift // 32
    if position < 0:
        return 0
    var offset = shift % 32
    var word = value._word(
        position
    ) if position < value._word_count() else UInt32(0)
    if not offset:
        return word
    word <<= UInt32(offset)
    if position > 0 and position - 1 < value._word_count():
        word |= value._word(position - 1) >> UInt32(32 - offset)
    return word


def _compare_scaled(a: Integer, b: Integer, shift: Int128) raises -> Int:
    """Compare positive a with b * 2**shift without allocating shifted values.
    """
    var left_bits = Int128(a.magnitude_bit_length())
    var right_bits = _exponent_add(Int128(b.magnitude_bit_length()), shift)
    if left_bits != right_bits:
        return -1 if left_bits < right_bits else 1
    # Equal top exponents bound the shift by the two existing operand lengths.
    var count = a._word_count() if shift >= 0 else b._word_count()
    for index in range(count - 1, -1, -1):
        var left = a._word(index) if shift >= 0 else _shifted_word(
            a, index, Int(-shift)
        )
        var right = _shifted_word(
            b, index, Int(shift)
        ) if shift >= 0 else b._word(index)
        if left != right:
            return -1 if left < right else 1
    return 0


def _round_away(mode: RoundingMode, negative: Bool) -> Bool:
    return (
        mode == RoundingMode.away_from_zero
        or (mode == RoundingMode.toward_positive and not negative)
        or (mode == RoundingMode.toward_negative and negative)
    )


def _is_power_of_two(value: Integer) -> Bool:
    var count = value._word_count()
    var top = value._word(count - 1)
    if top & (top - 1):
        return False
    for index in range(count - 1):
        if value._word(index):
            return False
    return True


def _rounded_direction(negative: Bool, increased_magnitude: Bool) -> Int:
    return -1 if negative == increased_magnitude else 1


def _overflow_result(
    negative: Bool, context: ArithmeticContext
) raises -> _RoundedBinary:
    var format = context.format()
    var infinity = (
        context.rounding() == RoundingMode.nearest_even
        or _round_away(context.rounding(), negative)
    )
    var magnitude = (
        Integer(0) if infinity else (Integer(1) << format.precision()) - 1
    )
    return _RoundedBinary(
        2 if infinity else 1,
        negative,
        magnitude,
        0 if infinity else format.emax(),
        format,
        NumericStatus._make(5, _rounded_direction(negative, infinity)),
    )


@always_inline
@always_inline
def _finish_round(
    var result: _RoundedBinary, context: ArithmeticContext, fail: Bool
) raises -> _RoundedBinary:
    if fail or Bool(context._traps & result.status._flags):
        _check_round_failure(result.status, context, fail)
    return result


@no_inline
def _check_round_failure(
    status: NumericStatus, context: ArithmeticContext, fail: Bool,
) raises:
    # Diagnostics stay out of the success path; numerical traps take priority
    # over the injected pre-publication checkpoint, as in ordinary rounding.
    context._check_status(status)
    if fail:
        raise Error(
            "Injected rounding failure before publication; retry the operation."
            " The destination is unchanged."
        )


def _round_ratio(
    numerator: Integer,
    denominator: Integer,
    context: ArithmeticContext,
    *,
    scale: Int128 = 0,
    negative_zero: Bool = False,
    fail: Bool = False,
) raises -> _RoundedBinary:
    """Round (numerator/denominator)*2**scale directly to the final format."""
    context.rounding()._validate()
    if not denominator:
        raise Error(
            "Cannot round an exact fraction with a zero denominator; supply a"
            " nonzero denominator. The destination is"
            " unchanged."
        )
    var negative = (numerator.sign() < 0 if numerator else negative_zero) != (
        denominator.sign() < 0
    )
    if not numerator:
        var budget = _ConversionBudget(None)
        return _round_magnitude[False](
            numerator,
            Integer(1),
            negative,
            context,
            budget,
            scale=scale,
            fail=fail,
        )
    if not numerator._negative() and not denominator._negative():
        # Magnitudes already: borrowed, where abs() would copy them.
        return _round_magnitudes(numerator, denominator, negative, context, scale=scale, fail=fail)
    return _round_magnitudes(abs(numerator), abs(denominator), negative, context, scale=scale, fail=fail)


def _round_magnitudes(
    numerator: Integer,
    denominator: Integer,
    negative: Bool,
    context: ArithmeticContext,
    *,
    scale: Int128 = 0,
    fail: Bool = False,
) raises -> _RoundedBinary:
    """_round_ratio of a nonzero numerator and a denominator given as
    magnitudes, with the sign apart: callers that hold magnitudes pass them
    borrowed, with no negated or absolute copies (each an atomic increment
    for a heap value, and a decrement on release)."""
    var format = context.format()
    if (
        format.precision() <= 128 and not format._is_exact()
        and numerator._word_count() <= 8 and denominator._word_count() <= 8
    ):
        # Operands of at most 256 bits: one native quotient, without the
        # general path's shifted operands and allocations.
        if denominator._is_one():
            # A binary value needs no quotient: round it as it stands.
            var binary = _round_wide(_low_words[DType.uint256](numerator), negative, scale, context, fail)
            if binary.kind >= 0:
                return binary^
        else:
            var native = _round_quotient_native(
                _low_words[DType.uint256](numerator), _low_words[DType.uint256](denominator),
                negative, scale, context, fail,
            )
            if native.kind >= 0:
                return native^
    var budget = _ConversionBudget(None)
    return _round_magnitude[False](
        numerator,
        denominator,
        negative,
        context,
        budget,
        scale=scale,
        fail=fail,
    )


def _round_shift[
    bounded: Bool
](
    value: Integer, shift: Int, right: Bool, mut budget: _ConversionBudget
) raises -> Integer:
    comptime if bounded:
        return _conversion_shift(value, shift, right, budget)
    else:
        return value >> shift if right else value << shift


def _round_sum[
    bounded: Bool
](
    a: Integer, b: Integer, subtract: Bool, mut budget: _ConversionBudget
) raises -> Integer:
    comptime if bounded:
        return _conversion_sum(a, b, subtract, budget)
    else:
        return a - b if subtract else a + b


def _round_overflow[
    bounded: Bool
](
    negative: Bool, context: ArithmeticContext, mut budget: _ConversionBudget
) raises -> _RoundedBinary:
    comptime if not bounded:
        return _overflow_result(negative, context)
    else:
        var infinity = (
            context.rounding() == RoundingMode.nearest_even
            or _round_away(context.rounding(), negative)
        )
        var magnitude = Integer(0)
        if not infinity:
            magnitude = _round_sum[True](
                _round_shift[True](
                    Integer(1), context.format().precision(), False, budget
                ),
                Integer(1),
                True,
                budget,
            )
        return _RoundedBinary(
            2 if infinity else 1,
            negative,
            magnitude,
            0 if infinity else context.format().emax(),
            context.format(),
            NumericStatus._make(5, _rounded_direction(negative, infinity)),
        )


def _round_magnitude[
    bounded: Bool
](
    numerator: Integer,
    denominator: Integer,
    negative: Bool,
    requested: ArithmeticContext,
    mut budget: _ConversionBudget,
    *,
    scale: Int128 = 0,
    fail: Bool = False,
) raises -> _RoundedBinary:
    # Internal callers supply nonnegative numerator and positive denominator.
    var context = requested
    var format = context.format()
    if not numerator:
        return _finish_round(
            _RoundedBinary(0, negative, Integer(0), 0, format, NumericStatus()),
            context,
            fail,
        )
    var dyadic = _is_power_of_two(denominator)
    if format._is_exact():
        # A ratio is a finite binary fraction only when the denominator's odd
        # part divides the numerator; then keep every significant bit.
        if not dyadic:
            var zeros = _trailing_zero_bits(denominator)
            var quotient = div_rem_trunc(numerator, denominator >> zeros)
            if quotient[1]:
                raise _inexact_error()
            context._format = format._with_precision(
                max(1, quotient[0].magnitude_bit_length() - _trailing_zero_bits(quotient[0]))
            )
            return _round_magnitude_core[bounded](
                quotient[0], Integer(1) << zeros, True, negative, context, budget, scale, fail
            )
        context._format = format._with_precision(
            max(1, numerator.magnitude_bit_length() - _trailing_zero_bits(numerator))
        )
    return _round_magnitude_core[bounded](numerator, denominator, dyadic, negative, context, budget, scale, fail)


def _round_magnitude_core[
    bounded: Bool
](
    a: Integer,
    b: Integer,
    dyadic: Bool,
    negative: Bool,
    context: ArithmeticContext,
    mut budget: _ConversionBudget,
    scale: Int128,
    fail: Bool,
) raises -> _RoundedBinary:
    """_round_magnitude for a nonzero a and the context's final format: the
    operands borrowed, not copied (a copy of a heap part is an atomic
    increment, and its release a decrement)."""
    var format = context.format()
    var base_exponent = a.magnitude_bit_length() - b.magnitude_bit_length()
    base_exponent += 1 if dyadic else Int(_compare_scaled(a, b, Int128(base_exponent)) >= 0)
    var exponent = _exponent_add(scale, Int128(base_exponent))
    if exponent < Int128(format.emin()):
        var up = _round_away(context.rounding(), negative)
        if context.rounding() == RoundingMode.nearest_even:
            up = False
            if exponent == Int128(format.emin()) - 1:
                up = _compare_scaled(a, b, Int128(base_exponent) - 1) > 0
        var magnitude = (
            _round_shift[bounded](
                Integer(1), format.precision() - 1, False, budget
            )
        ) if up else Integer(0)
        return _finish_round(
            _RoundedBinary(
                1 if up else 0,
                negative,
                magnitude,
                format.emin() if up else 0,
                format,
                NumericStatus._make(3, _rounded_direction(negative, up)),
            ),
            context,
            fail,
        )
    if exponent > Int128(format.emax()):
        return _finish_round(
            _round_overflow[bounded](negative, context, budget), context, fail
        )
    # Cancel the scale symbolically: only input lengths and precision set storage.
    var shift = Int128(format.precision()) - Int128(base_exponent)
    if shift > Int128(Int.MAX) or shift < -Int128(Int.MAX):
        raise Error(
            "Cannot round: required precision exceeds supported addressable"
            " storage; choose a smaller precision or exact source. The"
            " destination is unchanged."
        )
    var magnitude: Integer
    var inexact = False
    var up = False
    if dyadic:
        # Denominator scale cancels: the retained significand depends only on
        # numerator length and precision. No shifted divisor is materialized.
        var discard = a.magnitude_bit_length() - format.precision()
        magnitude = _round_shift[bounded](a, abs(discard), discard > 0, budget)
        if discard > 0:
            var index = (discard - 1) // 32
            var bit = UInt32((discard - 1) % 32)
            var word = a._word(index)
            var round_bit = Bool(word & (UInt32(1) << bit))
            var sticky = Bool(word & ((UInt32(1) << bit) - 1))
            for lower in range(index):
                sticky |= a._word(lower) != 0
            inexact = round_bit or sticky
            up = inexact and _round_away(context.rounding(), negative)
            if context.rounding() == RoundingMode.nearest_even:
                up = round_bit and (sticky or Bool(magnitude._word(0) & 1))
    else:
        comptime if bounded:
            var x = a
            var y = b
            if shift >= 0:
                x = _round_shift[bounded](x, Int(shift), False, budget)
            else:
                y = _round_shift[bounded](y, Int(-shift), False, budget)
            var pair = _conversion_div_rem(x, y, budget)
            magnitude = pair[0]
            var remainder = pair[1]
            inexact = remainder != 0
            if inexact:
                up = _round_away(context.rounding(), negative)
                if context.rounding() == RoundingMode.nearest_even:
                    var other_half = _round_sum[bounded](y, remainder, True, budget)
                    up = remainder > other_half or (
                        remainder == other_half and Bool(magnitude._word(0) & 1)
                    )
        else:
            # One division gives the significand and where the remainder lies
            # against half the divisor; no shifted or remainder values.
            var parts: Tuple[Integer, Integer, Int]
            if shift < 0:
                parts = _long_divide[False](a, b << Int(-shift), 0, False, False)
            else:
                parts = _long_divide[False](a, b, Int(shift), False, False)
            magnitude = _taken(parts[0])
            var place = parts[2]
            inexact = place != 0
            if inexact:
                up = _round_away(context.rounding(), negative)
                if context.rounding() == RoundingMode.nearest_even:
                    up = place == 3 or (place == 2 and Bool(magnitude._word(0) & 1))
    if up:
        magnitude = _round_sum[bounded](magnitude, Integer(1), False, budget)
    if magnitude.magnitude_bit_length() > format.precision():
        magnitude = _round_shift[bounded](magnitude, 1, True, budget)
        exponent = _exponent_add(exponent, Int128(1))
    if exponent > Int128(format.emax()):
        return _finish_round(
            _round_overflow[bounded](negative, context, budget), context, fail
        )
    var status = NumericStatus._make(
        Int(inexact), _rounded_direction(negative, up) if inexact else 0
    )
    return _finish_round(
        _RoundedBinary(1, negative, magnitude, Int(exponent), format, status),
        context,
        fail,
    )


