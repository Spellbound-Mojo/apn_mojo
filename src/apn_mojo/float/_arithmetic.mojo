"""Exact-source scalar arithmetic with bounded exponent-gap preparation."""

from ..integer.value import Integer
from ._input import _FloatInput, _float_input
from ._format import _merge_float_formats, _FormatSlot
from .context import ArithmeticContext, FloatFormat, RoundingMode
from .status import NumericStatus
from std.bit import count_leading_zeros
from std.collections import Array
from ..common._sizes import _checked_count, _checked_sum
from ._multiplication import _binary_product, _first_product_word, _round_product_owner
from ..integer._words import _add_shifted_words, _add_words, _compare_words, _shift_left_words, _shift_right_words, _subtract_words
from ..integer._division import _divide_2by1, _divide_by_limb_into, _reciprocal
from ._rounding import (
    _RoundedBinary,
    _round_ratio,
    _round_magnitudes,
    _finish_round,
    _exponent_add,
    _inexact_error,
    _round_away,
    _LIMB_SCALE,
    _round_limb,
    _round_native,
    _round_wide,
    _rounded_direction,
    _short_magnitude,
)
from ..common._traits import _MapAdapter, _MapArgument
from ..rational.value import Rational
from .value import Float


trait _FloatSource(Copyable):
    def _as_float_input(self) -> _FloatInput:
        ...

    def format(self) -> FloatFormat:
        ...


@fieldwise_init
struct _FloatArgument(ImplicitlyCopyable, _MapAdapter, _MapArgument):
    var value: _FloatInput
    var format: _FormatSlot
    var native_precision: Int

    @staticmethod
    def _adapt[V: ImplicitlyCopyable & Deinitable](value: V) raises -> Self:
        comptime assert V == Integer or V == Rational or V == Float, (
            "A real function takes Integer, Rational or Float values; map the"
            " Complex function for Complex values, for example apn_mojo.complex.sqrt."
        )
        return Self(value)

    @implicit
    def __init__[T: Copyable](out self, value: T):
        self = _float_argument(value)

    @implicit
    def __init__(out self, value: IntLiteral):
        self = _float_argument(Integer(value))

    @implicit
    def __init__(out self, value: FloatLiteral):
        self = Self(_float_input(value), None, 0)


def _float_argument[T: Copyable](value: T) -> _FloatArgument:
    comptime if conforms_to(T, _FloatSource):
        return _FloatArgument(value._as_float_input(), value.format(), 0)
    else:
        comptime precision = 53 if T == Float64 else 24 if T == Float32 else 11 if T == Float16 else 8 if T == BFloat16 else 0
        return _FloatArgument(_float_input(value), None, precision)


def _float_special(
    kind: Int,
    negative: Bool,
    flags: Int,
    context: ArithmeticContext,
    fail: Bool,
) raises -> _RoundedBinary:
    return _finish_round(
        _RoundedBinary(
            kind,
            negative if kind != 3 else False,
            Integer(0),
            0,
            context.format(),
            NumericStatus._make(flags, 2 if kind == 3 else 0),
        ),
        context,
        fail,
    )


def _binary_sum(
    high: Span[mut=False, UInt32, _], high_negative: Bool,
    low: Span[mut=False, UInt32, _], low_negative: Bool,
    gap: Int128, low_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round high * 2**gap + low once, scaled by 2**low_scale.

    Both are nonzero binary significands. The exact aligned sum is built in one
    allocation by the shared word kernels, and the shared rounding turns it into
    the result's significand in place: no intermediate Integers.
    """
    if gap > Int128(Int.MAX):
        raise Error(
            "Cannot add Float values: working precision exceeds addressable"
            " storage; choose a smaller output precision or exact source."
            " The destination is unchanged."
        )
    var shift = Int(gap)
    var whole = shift // 32
    var top = _checked_sum(_checked_sum(len(high), whole), 1)
    var length = _checked_sum(max(top, len(low)), 1)
    var count = (context.format().precision() + 31) // 32
    var capacity = _checked_count(max(length, count), 4)
    var owner = Integer._Shared.uninitialized(capacity, False)
    # The shared word kernels work in place: each reads a pair before writing it.
    var words = owner[].words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for i in range(whole):
        words.unsafe_offset(i)[] = 0
    _shift_left_words(high, whole, shift % 32, words)
    var negative = high_negative
    var used = length
    if high_negative == low_negative:
        _add_words(Span(unsafe_ptr=words, length=top), low, words)
    else:
        while not words.unsafe_offset(top - 1)[]:
            top -= 1
        var shifted = Span(unsafe_ptr=words, length=top)
        var order = _compare_words(shifted, low)
        if order == 0:
            # Exact cancellation: a signed zero, as _round_ratio gives it.
            return _float_special(0, context.rounding() == RoundingMode.toward_negative, 0, context, fail)
        if order > 0:
            _subtract_words(shifted, low, words)
            used = top
        else:
            # The shifted high magnitude is the smaller one.
            negative = low_negative
            _subtract_words(low, shifted, words)
            used = len(low)
    while used and words.unsafe_offset(used - 1)[] == 0:
        used -= 1
    return _round_product_owner(owner^, used, negative, low_scale, context, fail)


@no_inline
def _one_limb_quotient(
    x: UInt64, y: UInt64, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round (x / y) * 2**scale once, for nonzero significands of at most 64
    bits and a format of at most 63. With both left-aligned, u < 2v, so
    u * 2**63 / v fits one limb: one 2-by-1 division (Moller and Granlund,
    2011, by the table reciprocal) gives it and its remainder. A quotient of
    63 bits takes its next bit from the remainder, a step of long division.
    The round bit, and a sticky from the bits below and the remainder, go to
    _round_limb: exact for inputs of any precision. Out of line, its operands
    stay in registers."""
    var lx = count_leading_zeros(x)
    var ly = count_leading_zeros(y)
    var u = x << lx
    var v = y << ly
    var quotient, rest = _divide_2by1(u >> 1, u << 63, v, _reciprocal(v))
    # x / y * 2**scale = (quotient + rest / v) * 2**(scale + ly - lx - 63).
    var exponent = Int(scale) + Int(ly) - Int(lx) + 1
    if quotient >> 63 == 0:
        # u < v: one more quotient bit, from twice the remainder.
        var twice = UInt128(rest) << 1
        var bit = UInt64(Int(twice >= UInt128(v)))
        quotient = (quotient << 1) | bit
        rest = UInt64(twice - UInt128(v & (0 - bit)))
        exponent -= 1
    var sh = UInt64(64 - context.format().precision())
    var mask = (UInt64(1) << sh) - 1
    var rb = quotient & (UInt64(1) << (sh - 1))
    var sb = (quotient & (mask >> 1)) | rest
    return _round_limb(quotient & ~mask, rb, sb, negative, exponent, context, fail)


def _limb_quotient(
    dividend: Integer, d: UInt64, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round (dividend / divisor) * 2**scale once, for nonzero significands, a
    divisor of at most 64 bits and an inexact format: the dividend, shifted so
    that the quotient has at least p + 2 bits, is divided in place in the
    result's allocation by one pass of 2-by-1 divisions. A nonzero remainder
    sets the quotient's lowest bit, below its round bit: the same rounding as
    the exact quotient's. Then the shared rounding, in place."""
    var small = dividend._inline_words()
    var words = dividend._words_span(small)
    var bits = dividend.magnitude_bit_length()
    var shift = max(0, context.format().precision() + 2 + (64 - Int(count_leading_zeros(d))) - bits)
    var whole = shift // 32
    var length = _checked_sum(_checked_sum(len(words), whole), 1)
    var count = (context.format().precision() + 31) // 32
    var owner = Integer._Shared.uninitialized(_checked_count(max(length, count), 4), False)
    var target = owner[].words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for i in range(whole):
        target.unsafe_offset(i)[] = 0
    _shift_left_words(words, whole, shift % 32, target)
    # The kernel reads each limb before its quotient digit replaces it.
    var rest = _divide_by_limb_into[True](Span(unsafe_ptr=target, length=length), d, target)
    if rest:
        target[] |= 1
    var used = length
    while not target.unsafe_offset(used - 1)[]:
        used -= 1
    return _round_product_owner(owner^, used, negative, _exponent_add(scale, -Int128(shift)), context, fail)


@always_inline
def _binary_quotient(
    dividend: Integer, divisor: Integer, negative: Bool, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round (dividend / divisor) * 2**scale once, for nonzero significands."""
    context.rounding()._validate()
    var format = context.format()
    if (
        format.precision() <= 63 and not format._is_exact()
        and dividend._storage.isa[Int64]() and divisor._storage.isa[Int64]()
        and scale > -_LIMB_SCALE and scale < _LIMB_SCALE
    ):
        # Short significands: one limb.
        var native = _one_limb_quotient(
            UInt64(dividend._storage[Int64]), UInt64(divisor._storage[Int64]), negative, scale, context, fail
        )
        if native.kind >= 0:
            return native^
    if not format._is_exact():
        # Significands are padded to their precision: the divisor's trailing
        # zero words only scale the quotient, and a divisor of one limb
        # without them (x / 1.75) takes one division pass.
        var small = divisor._inline_words()
        var words = divisor._words_span(small)
        var first = _first_product_word(words)
        if len(words) - first <= 2:
            var d = UInt64(words[first])
            if first + 1 < len(words):
                d |= UInt64(words[first + 1]) << 32
            return _limb_quotient(dividend, d, negative, _exponent_add(scale, -32 * Int128(first)), context, fail)
    if format._is_exact():
        # Name the operation when an exact quotient does not exist.
        try:
            return _round_ratio(-dividend if negative else dividend, divisor, context, scale=scale, fail=fail)
        except error:
            if String(error).startswith("Cannot compute the result exactly"):
                raise _inexact_error("the quotient")
            raise error^
    # Magnitudes and the sign apart: no negated copy of the dividend.
    return _round_magnitudes(dividend, divisor, negative, context, scale=scale, fail=fail)


@always_inline
def _wide_sum(
    x: UInt128, x_negative: Bool, x_scale: Int128,
    y: UInt128, y_negative: Bool, y_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round x * 2**x_scale + y * 2**y_scale once, for nonzero magnitudes and
    an inexact format of at most 128 bits, in 256-bit native integers. Kind
    -1 asks the caller for the general path (terms far apart, or a result
    outside the exponent range)."""
    var aligned = _aligned_sum(x, x_negative, x_scale, y, y_negative, y_scale, context.format().precision())
    if not aligned[3]:
        return _RoundedBinary(-1, False, Integer(), 0, context.format(), NumericStatus())
    if not aligned[0]:
        return _float_special(0, context.rounding() == RoundingMode.toward_negative, 0, context, fail)
    return _round_wide(aligned[0], aligned[1], aligned[2], context, fail)


@always_inline
def _aligned_sum(
    x: UInt128, x_negative: Bool, x_scale: Int128,
    y: UInt128, y_negative: Bool, y_scale: Int128, p: Int,
) -> Tuple[UInt256, Bool, Int128, Bool]:
    """`x 2**x_scale + y 2**y_scale` as a 256-bit magnitude, its sign and
    scale, rounding to p bits as the exact sum does; the last field is False
    for terms too far apart, which the caller sums otherwise."""
    var on_left = x_scale >= y_scale
    var high = UInt256(x if on_left else y)
    var low = UInt256(y if on_left else x)
    var high_negative = x_negative if on_left else y_negative
    var low_negative = y_negative if on_left else x_negative
    var high_scale = x_scale if on_left else y_scale
    var low_scale = y_scale if on_left else x_scale
    var gap = high_scale - low_scale
    if gap <= 127:
        high <<= UInt256(gap)
    elif p <= 127 and gap > Int128(256 - Int(count_leading_zeros(low))) + 128:
        # A low term under one unit of the shifted high term, below a rounding
        # point at least two bits higher, acts only as a sticky bit.
        high <<= 128
        low = 1
        low_scale = high_scale - 128
    else:
        return (UInt256(0), False, Int128(0), False)
    var total, negative = _signed_sum(high, high_negative, low, low_negative)
    return (total, negative, low_scale, True)


@always_inline
def _signed_sum(
    high: UInt256, high_negative: Bool, low: UInt256, low_negative: Bool
) -> Tuple[UInt256, Bool]:
    """The sum of two signed magnitudes, as a magnitude and a sign; the first
    term's sign when they cancel."""
    if high_negative == low_negative:
        return (high + low, high_negative)
    if high >= low:
        return (high - low, high_negative)
    return (low - high, low_negative)


@no_inline
def _one_limb_sum(
    b0: UInt64, b_negative: Bool, b_scale: Int128,
    c0: UInt64, c_negative: Bool, c_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round b + c once, for significands of exactly the format's precision
    p <= 64, left-aligned in one limb (bit 63 set), scaled by 2**scale as
    stored. The textbook addition with a sticky bit (Muller et al., Handbook
    of Floating-Point Arithmetic, 2nd ed., 2018): the term of larger
    magnitude in a 128-bit window with a bit of headroom, the other shifted
    into it by the exponent difference, any bits shifted out of the window
    kept as a sticky flag, which a difference takes as one unit borrowed from
    the window; then normalization by the leading zeros, and one rounding.
    The window holds p + 1 bits and more after any cancellation: below 64
    bits apart the shifted term loses nothing, and further apart a
    difference cancels at most one bit. Out of line, its operands stay in
    registers."""
    var p = context.format().precision()
    var b = b0
    var c = c0
    var bx = Int(b_scale) + p
    var cx = Int(c_scale) + p
    var negative = b_negative
    var subtract = b_negative != c_negative
    if cx > bx or (cx == bx and c > b):
        swap(b, c)
        swap(bx, cx)
        negative = c_negative
    if subtract and bx == cx and b == c:
        # Exact cancellation: a signed zero, as _round_ratio gives it.
        return _float_special(0, context.rounding() == RoundingMode.toward_negative, 0, context, fail)
    var d = bx - cx
    var window = UInt128(b) << 63
    var shifted = UInt128(c) << 63
    # The window's low 63 bits start clear: bits are lost only for d > 63,
    # c's d - 63 lowest, which a 64-bit shift tests.
    var lost = d > 63
    if d >= 127:
        shifted = 0
    else:
        if lost:
            lost = (c << UInt64(127 - d)) != 0
        shifted >>= UInt128(d)
    if subtract:
        window -= shifted + UInt128(Int(lost))
    else:
        window += shifted
    # The window's value is window * 2**(bx - 127); normalized, its top limb
    # holds the significand and round bit, and everything below is sticky.
    var lead = count_leading_zeros(window)
    window <<= lead
    var top = UInt64(window >> 64)
    var sh = UInt64(64 - p)
    if sh == 0:
        # p = 64: the top limb is the significand, and the round bit leads
        # the low limb.
        var low = UInt64(window)
        return _round_limb(top, low >> 63, (low << 1) | UInt64(Int(lost)), negative, bx + 1 - Int(lead), context, fail)
    var mask = (UInt64(1) << sh) - 1
    var rb = top & (UInt64(1) << (sh - 1))
    var sb = (top & (mask >> 1)) | UInt64(window) | UInt64(Int(lost))
    return _round_limb(top & ~mask, rb, sb, negative, bx + 1 - Int(lead), context, fail)


@always_inline
def _short_sum(
    x: UInt64, left_negative: Bool, left_scale: Int128,
    y: UInt64, right_negative: Bool, right_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round x * 2**left_scale + y * 2**right_scale once, for nonzero
    significands of at most 63 bits and an inexact format of at most 63 bits:
    align, add and round in 128-bit native integers. Kind -1 asks for the
    general path (terms far apart, or a result outside the exponent range)."""
    var on_left = left_scale >= right_scale
    var high = UInt128(x if on_left else y)
    var low = UInt128(y if on_left else x)
    var high_negative = left_negative if on_left else right_negative
    var low_negative = right_negative if on_left else left_negative
    var high_scale = left_scale if on_left else right_scale
    var low_scale = right_scale if on_left else left_scale
    var gap = high_scale - low_scale
    var direct = True
    if gap <= 64:
        high <<= UInt128(gap)
    elif gap > Int128(128 - Int(count_leading_zeros(low))) + 64:
        # The low term is worth less than one unit of the shifted high
        # term, whose rounding point lies at least two bits higher (64 >
        # 63 >= precision): it can only act as a sticky bit.
        high <<= 64
        low = 1
        low_scale = high_scale - 64
    else:
        direct = False
    var negative = high_negative
    if direct:
        var total: UInt128
        if high_negative == low_negative:
            total = high + low
        elif high >= low:
            total = high - low
        else:
            total = low - high
            negative = low_negative
        if not total:
            return _float_special(0, context.rounding() == RoundingMode.toward_negative, 0, context, fail)
        var native = _round_native(total, negative, low_scale, context, fail)
        if native.kind >= 0:
            return native^
    return _RoundedBinary(-1, False, Integer(), 0, context.format(), NumericStatus())


@always_inline
def _binary_float_sum(
    left: Integer, left_negative: Bool, left_scale: Int128,
    right: Integer, right_negative: Bool, right_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round left * 2**left_scale + right * 2**right_scale once: two terms at
    a format's precision of at most 64 bits in one limb, here; the rest in
    _other_float_sum, a call apart. Inlined, every alternative's setup was
    computed ahead of the one-limb branch (the compiler hoists it), and a
    53-bit sum ran more instructions than through 128-bit native integers."""
    context.rounding()._validate()
    var format = context.format()
    if (
        format.precision() <= 63 and not format._is_exact()
        and left._storage.isa[Int64]() and right._storage.isa[Int64]()
    ):
        var x = UInt64(left._storage[Int64])
        var y = UInt64(right._storage[Int64])
        var p = UInt64(format.precision())
        if (
            (x | y) >> p == 0 and (x & y) >> (p - 1) == 1
            and left_scale > -_LIMB_SCALE and left_scale < _LIMB_SCALE
            and right_scale > -_LIMB_SCALE and right_scale < _LIMB_SCALE
        ):
            var sum = _one_limb_sum(x << (64 - p), left_negative, left_scale, y << (64 - p), right_negative, right_scale, context, fail)
            # The one-limb path declines results beyond the format's exponent
            # range, such as an underflow, and the other paths round them.
            if sum.kind >= 0:
                return sum^
    elif (
        format.precision() == 64 and not format._is_exact()
        and left._storage.isa[UInt64]() and right._storage.isa[UInt64]()
        and left_scale > -_LIMB_SCALE and left_scale < _LIMB_SCALE
        and right_scale > -_LIMB_SCALE and right_scale < _LIMB_SCALE
    ):
        # A 64-bit significand is a UInt64 of exactly 64 bits, aligned as is.
        var sum = _one_limb_sum(
            left._storage[UInt64], left_negative, left_scale, right._storage[UInt64], right_negative, right_scale, context, fail
        )
        if sum.kind >= 0:
            return sum^
    return _other_float_sum(left, left_negative, left_scale, right, right_negative, right_scale, context, fail)


@no_inline
def _other_float_sum(
    left: Integer, left_negative: Bool, left_scale: Int128,
    right: Integer, right_negative: Bool, right_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """_binary_float_sum's other sums: short significands of other precisions
    in 128-bit native integers, up to 128 bits in 256-bit ones, the rest in
    _binary_float_sum_general."""
    var format = context.format()
    if (
        format.precision() <= 63 and not format._is_exact()
        and left._storage.isa[Int64]() and right._storage.isa[Int64]()
    ):
        var short = _short_sum(
            UInt64(left._storage[Int64]), left_negative, left_scale,
            UInt64(right._storage[Int64]), right_negative, right_scale, context, fail,
        )
        if short.kind >= 0:
            return short^
    elif format.precision() <= 128 and not format._is_exact():
        # Significands up to 128 bits (the default precision): 256-bit native integers.
        var x = _short_magnitude(left)
        var y = _short_magnitude(right)
        var p = UInt128(format.precision())
        # Terms at the format's precision take the same-precision sum of the
        # general path, which measured faster than 256-bit native integers.
        if x and y and not (x.value() >> (p - 1) == 1 and y.value() >> (p - 1) == 1):
            var wide = _wide_sum(x.value(), left_negative, left_scale, y.value(), right_negative, right_scale, context, fail)
            if wide.kind >= 0:
                return wide^
    return _binary_float_sum_general(left, left_negative, left_scale, right, right_negative, right_scale, context, fail)


@always_inline
def _bit_at(words: Span[mut=False, UInt32, _], i: Int) -> Bool:
    """Bit i of a magnitude's words; clear below them and past them."""
    if i < 0 or i >= 32 * len(words):
        return False
    return Bool((words.unsafe_get(i // 32) >> UInt32(i % 32)) & 1)


@always_inline
def _low_bits_nonzero(words: Span[mut=False, UInt32, _], k: Int) -> Bool:
    """Whether any of a magnitude's bits below bit k is set."""
    if k <= 0:
        return False
    var whole = min(k // 32, len(words))
    for i in range(whole):
        if words.unsafe_get(i):
            return True
    var part = k % 32
    return whole < len(words) and part != 0 and Bool(words.unsafe_get(whole) & ((UInt32(1) << UInt32(part)) - 1))


@no_inline
def _same_precision_sum(
    high: Span[mut=False, UInt32, _], high_negative: Bool,
    low: Span[mut=False, UInt32, _], low_negative: Bool,
    gap: Int128, high_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round high * 2**high_scale + low * 2**(high_scale - gap) once, for
    significands of exactly the inexact format's precision p (n words each)
    and gap >= 0: the textbook addition with round and sticky bits (Muller
    et al., Handbook of Floating-Point Arithmetic, 2nd ed., 2018) on n words.
    The low term is shifted by the gap once, added to or subtracted from the
    high one in n words, and the result rounds from the shifted-out bits: no
    exact sum.

    A difference writes its shifted-out remainder rem as the positive fraction
    2**gap - rem below high - floor(low / 2**gap) - 1, so sums and differences
    both round up from a p-bit truncation. Kind -1 asks for the general path:
    terms of opposite signs less than two binades apart, which can cancel, or
    a result outside the exponent range."""
    var format = context.format()
    var p = format.precision()
    var n = len(high)
    var subtract = high_negative != low_negative
    if subtract and gap < 2:
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    # Past p + 2 bits, every gap shifts the low term out alike.
    var d = Int(min(gap, Int128(p + 2)))
    var whole = min(d // 32, n)
    var owner = Integer._Shared.uninitialized(n + 1, False)
    var target = owner[].words.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    # Bit p, one past the significand: in word n when p fills its words.
    var carry_word = p // 32
    var carry_bit = UInt32(1) << UInt32(p % 32)
    var exponent = high_scale + Int128(p)
    var round_bit: Bool
    var sticky: Bool
    if not subtract:
        # high + floor(low / 2**d) in one pass, with the shifted-out bits'
        # round and sticky.
        round_bit = _bit_at(low, d - 1)
        sticky = _low_bits_nonzero(low, d - 1)
        target.unsafe_offset(n)[] = UInt32(_add_shifted_words[False](high, low, whole, d % 32, 0, target))
        if target.unsafe_offset(carry_word)[] & carry_bit:
            # A carry: one binade up, and the sum's last bit joins the tail.
            sticky = sticky or round_bit
            round_bit = Bool(target[] & 1)
            _ = _shift_right_words(Span(unsafe_ptr=target, length=n + 1), 0, 1, 0, target)
            exponent += 1
    else:
        # high - floor(low / 2**d) - 1 when rem = low mod 2**d is nonzero, and
        # the fraction 2**d - rem: its top two bits are four less rem's top
        # two, less one more (with a sticky tail) when rem has lower bits.
        var rest = _low_bits_nonzero(low, d - 2)
        var upper = 2 * Int(_bit_at(low, d - 1)) + Int(_bit_at(low, d - 2))
        var remainder = rest or upper != 0
        var top = (4 - upper - Int(rest)) if remainder else 0
        # No borrow out: with d >= 2 the subtrahend is below 2**(p - 2) + 1.
        _ = _add_shifted_words[True](high, low, whole, d % 32, UInt64(Int(remainder)), target)
        target.unsafe_offset(n)[] = 0
        round_bit = Bool(top & 2)
        sticky = Bool(top & 1) or rest
        var top_word = (p - 1) // 32
        if not (target.unsafe_offset(top_word)[] & (UInt32(1) << UInt32((p - 1) % 32))):
            # One bit lost, no more: the difference is at least 2**(p - 2).
            # The fraction's top bit shifts in.
            _shift_left_words(Span(unsafe_ptr=target, length=n), 0, 1, target)
            target[] |= UInt32(Int(round_bit))
            round_bit = Bool(top & 1)
            sticky = rest
            exponent -= 1
    var inexact = round_bit or sticky
    var up = False
    if inexact:
        var mode = context.rounding()
        if mode == RoundingMode.nearest_even:
            up = round_bit and (sticky or Bool(target[] & 1))
        else:
            up = _round_away(mode, high_negative)
    if up:
        var i = 0
        while True:
            var word = target.unsafe_offset(i)[] + 1
            target.unsafe_offset(i)[] = word
            if word:
                break
            i += 1
        if target.unsafe_offset(carry_word)[] & carry_bit:
            # 2**p: the next binade's first value.
            for j in range(n + 1):
                target.unsafe_offset(j)[] = 0
            target.unsafe_offset((p - 1) // 32)[] = UInt32(1) << UInt32((p - 1) % 32)
            exponent += 1
    if exponent > Int128(format.emax()) or exponent < Int128(format.emin()):
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    return _finish_round(_RoundedBinary(
        1, high_negative, Integer._from_product(owner^, n), Int(exponent), format,
        NumericStatus._make(Int(inexact), _rounded_direction(high_negative, up) if inexact else 0),
    ), context, fail)


@always_inline
def _binary_float_sum_general(
    left: Integer, left_negative: Bool, left_scale: Int128,
    right: Integer, right_negative: Bool, right_scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round left * 2**left_scale + right * 2**right_scale once.

    Both magnitudes are nonzero and borrowed, as the binary significands of
    finite Floats are; only the result is allocated.
    """
    context.rounding()._validate()
    var small_left = left._inline_words()
    var small_right = right._inline_words()
    # Significands are padded to their precision: trailing zero words only
    # scale a term, and without them 1.75 is one word, not eight at 256 bits.
    var l_full = left._words_span(small_left)
    var r_full = right._words_span(small_right)
    var format = context.format()
    var n = (format.precision() + 31) // 32
    var top_bits = format.precision() - 32 * (n - 1)
    if (
        not format._is_exact() and len(l_full) == n and len(r_full) == n
        and 32 - Int(count_leading_zeros(l_full.unsafe_get(n - 1))) == top_bits
        and 32 - Int(count_leading_zeros(r_full.unsafe_get(n - 1))) == top_bits
    ):
        # Both terms at the format's precision, the usual case: the
        # same-precision sum, with no exact sum to build.
        var same: _RoundedBinary
        if left_scale >= right_scale:
            same = _same_precision_sum(l_full, left_negative, r_full, right_negative, left_scale - right_scale, left_scale, context, fail)
        else:
            same = _same_precision_sum(r_full, right_negative, l_full, left_negative, right_scale - left_scale, right_scale, context, fail)
        if same.kind >= 0:
            return same^
    var first_l = _first_product_word(l_full)
    var first_r = _first_product_word(r_full)
    var l = l_full[first_l:]
    var r = r_full[first_r:]
    var l_scale = _exponent_add(left_scale, 32 * Int128(first_l))
    var r_scale = _exponent_add(right_scale, 32 * Int128(first_r))
    var high_is_left = l_scale >= r_scale
    var high_scale = l_scale if high_is_left else r_scale
    var low_scale = r_scale if high_is_left else l_scale
    var gap = _exponent_add(high_scale, -low_scale)
    var low_bits = (
        len(r) * 32 - Int(count_leading_zeros(r.unsafe_get(len(r) - 1)))
        if high_is_left else
        len(l) * 32 - Int(count_leading_zeros(l.unsafe_get(len(l) - 1)))
    )
    # A remote tail rounds like a sticky one-bit just below the cutoff
    # (docs/float_arithmetic.md); the cutoff matches the general path.
    var cutoff = Int128(context.format().precision()) + 5
    var one = Array[UInt32, 1](fill=1)
    # Exact working formats keep a remote tail: it is part of the exact sum.
    if gap > Int128(low_bits) + cutoff and not context.format()._is_exact():
        if high_is_left:
            return _binary_sum(l, left_negative, Span(one), right_negative, cutoff, _exponent_add(high_scale, -cutoff), context, fail)
        return _binary_sum(r, right_negative, Span(one), left_negative, cutoff, _exponent_add(high_scale, -cutoff), context, fail)
    if high_is_left:
        return _binary_sum(l, left_negative, r, right_negative, gap, low_scale, context, fail)
    return _binary_sum(r, right_negative, l, left_negative, gap, low_scale, context, fail)


def _float_sum(
    left: _FloatInput,
    right: _FloatInput,
    context: ArithmeticContext,
    fail: Bool,
    *,
    negate_right: Bool = False,
) raises -> _RoundedBinary:
    # Subtraction passes negate_right instead of a negated copy of right.
    var right_negative = right.negative != negate_right
    if not left.numerator and not right.numerator:
        var negative = (
            left.negative if left.negative
            == right_negative else context.rounding()
            == RoundingMode.toward_negative
        )
        return _float_special(0, negative, 0, context, fail)
    if (
        left.numerator.__bool__() and right.numerator.__bool__()
        and left.denominator._is_one() and right.denominator._is_one()
    ):
        # Binary operands, the Float case: one allocation for the whole sum.
        return _binary_float_sum(
            left.numerator, left.negative, left.scale,
            right.numerator, right_negative, right.scale, context, fail,
        )
    if not left.numerator:
        return _round_ratio(
            -right.numerator if right_negative else right.numerator,
            right.denominator,
            context,
            scale=right.scale,
            fail=fail,
        )
    if not right.numerator:
        return _round_ratio(
            -left.numerator if left.negative else left.numerator,
            left.denominator,
            context,
            scale=left.scale,
            fail=fail,
        )
    var a = left.numerator * right.denominator
    var b = right.numerator * left.denominator
    var denominator = left.denominator * right.denominator
    var high_scale = left.scale
    var low_scale = right.scale
    var negative = left.negative
    var other_negative = right_negative
    if high_scale < low_scale:
        var saved = a
        a = b
        b = saved
        high_scale = right.scale
        low_scale = left.scale
        negative = right_negative
        other_negative = left.negative
    var gap = _exponent_add(high_scale, -low_scale)
    var cutoff = (
        Int128(denominator.magnitude_bit_length())
        + Int128(context.format().precision())
        + 4
    )
    # A remote nonzero tail is replaced on the same side of every rounding
    # boundary. See docs/float_arithmetic.md for the lattice-distance proof.
    if gap > Int128(b.magnitude_bit_length()) + cutoff:
        gap = cutoff
        low_scale = _exponent_add(high_scale, -cutoff)
        b = Integer(1)
    if gap > Int128(Int.MAX):
        raise Error(
            "Cannot add Float values: working precision exceeds addressable"
            " storage; choose a smaller output precision or exact source."
            " The destination is unchanged."
        )
    a <<= Int(gap)
    var numerator = a - b if negative != other_negative else a + b
    if negative:
        numerator = -numerator
    return _round_ratio(
        numerator,
        denominator,
        context,
        scale=low_scale,
        negative_zero=context.rounding() == RoundingMode.toward_negative,
        fail=fail,
    )


def _float_operation(
    left: _FloatArgument,
    right: _FloatArgument,
    operation: Int,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    """_float_operation_of for an operation chosen at run time."""
    if operation == 0:
        return _float_operation_of[0](left, right, context, fail=fail)
    if operation == 1:
        return _float_operation_of[1](left, right, context, fail=fail)
    if operation == 2:
        return _float_operation_of[2](left, right, context, fail=fail)
    return _float_operation_of[3](left, right, context, fail=fail)


@always_inline
def _call_context(value: _FloatArgument, context: Optional[ArithmeticContext]) raises -> ArithmeticContext:
    """The context of a call on one value: the caller's, or the value's own
    format with nearest-even rounding and no traps. Inlined, so a given
    context costs a copy; other operands merge out of line."""
    if context:
        return context.value()
    if value.format and value.native_precision == 0:
        return ArithmeticContext(_format_of=value.format.value())
    return _merged_context(value.format, None, value.native_precision, 0)


@always_inline
def _call_context(
    left: _FloatArgument, right: _FloatArgument, context: Optional[ArithmeticContext],
) raises -> ArithmeticContext:
    """The context of a call on two values: the caller's, or the merged format
    of the operands. Floats of one format, the common case, skip the merge."""
    if context:
        return context.value()
    if (
        left.format and right.format and left.native_precision == 0 and right.native_precision == 0
        and left.format.value() == right.format.value()
    ):
        return ArithmeticContext(_format_of=left.format.value())
    return _merged_context(left.format, right.format, left.native_precision, right.native_precision)


@no_inline
def _merged_context(
    left: _FormatSlot, right: _FormatSlot, left_native_precision: Int, right_native_precision: Int,
) raises -> ArithmeticContext:
    return ArithmeticContext(_format_of=_merge_float_formats(
        left, right, left_native_precision=left_native_precision, right_native_precision=right_native_precision,
    ))


def _float_operation_of[operation: Int](
    left: _FloatArgument,
    right: _FloatArgument,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    """The rounded add (0), subtract (1), multiply (2) or divide (3) of two
    argument records. One function per operation: with the operation chosen
    at run time, all four paths shared one function, and its size and
    register pressure cost a 53-bit sum about 90 instructions."""
    var target = _call_context(left, right, context)
    comptime if operation == 3:
        if target.format()._is_exact():
            # Name the operation when an exact quotient does not exist.
            try:
                return _float_input_operation_of[operation](left.value, right.value, target, fail)
            except error:
                if String(error).startswith("Cannot compute the result exactly"):
                    raise _inexact_error("the quotient")
                raise error^
    return _float_input_operation_of[operation](left.value, right.value, target, fail)


@always_inline
def _float_input_operation(
    a: _FloatInput,
    b: _FloatInput,
    operation: Int,
    target: ArithmeticContext,
    fail: Bool,
) raises -> _RoundedBinary:
    """_float_input_operation_of for an operation chosen at run time."""
    if operation == 0:
        return _float_input_operation_of[0](a, b, target, fail)
    if operation == 1:
        return _float_input_operation_of[1](a, b, target, fail)
    if operation == 2:
        return _float_input_operation_of[2](a, b, target, fail)
    return _float_input_operation_of[3](a, b, target, fail)


@always_inline
def _float_input_operation_of[operation: Int](
    a: _FloatInput,
    b: _FloatInput,
    target: ArithmeticContext,
    fail: Bool,
) raises -> _RoundedBinary:
    # Borrows both inputs; the caller has resolved the target context.
    # Finite binary operands, the Float case, go straight to the binary cores
    # the operators use. Inlined: passing the input records down another call
    # costs about 100 instructions.
    if (
        a.kind == 1 and b.kind == 1
        and a.denominator._is_one() and b.denominator._is_one()
    ):
        comptime if operation <= 1:
            return _binary_float_sum(
                a.numerator, a.negative, a.scale,
                b.numerator, b.negative != (operation == 1), b.scale,
                target, fail,
            )
        elif operation == 2:
            return _binary_product(a.numerator, b.numerator, a.negative != b.negative, _exponent_add(a.scale, b.scale), target, fail)
        else:
            return _binary_quotient(a.numerator, b.numerator, a.negative != b.negative, _exponent_add(a.scale, -b.scale), target, fail)
    return _special_or_ratio_operation(a, b, operation, target, fail)


def _special_or_ratio_operation(
    a: _FloatInput,
    b: _FloatInput,
    operation: Int,
    target: ArithmeticContext,
    fail: Bool,
) raises -> _RoundedBinary:
    # Every case but two finite binary operands: specials, zeros and ratios.
    if a.kind == 3 or b.kind == 3:
        return _float_special(3, False, 0, target, fail)
    if operation <= 1:
        var b_negative = b.negative != (operation == 1)
        if a.kind == 2 or b.kind == 2:
            if a.kind == 2 and b.kind == 2 and a.negative != b_negative:
                return _float_special(3, False, 16, target, fail)
            return _float_special(
                2, a.negative if a.kind == 2 else b_negative, 0, target, fail
            )
        return _float_sum(a, b, target, fail, negate_right=operation == 1)
    var negative = a.negative != b.negative
    if operation == 2:
        if (a.kind == 0 and b.kind == 2) or (a.kind == 2 and b.kind == 0):
            return _float_special(3, False, 16, target, fail)
        if a.kind == 2 or b.kind == 2:
            return _float_special(2, negative, 0, target, fail)
        if a.kind == 0 or b.kind == 0:
            return _float_special(0, negative, 0, target, fail)
        return _round_ratio(
            (-a.numerator if negative else a.numerator) * b.numerator,
            a.denominator * b.denominator,
            target,
            scale=_exponent_add(a.scale, b.scale),
            fail=fail,
        )
    if (a.kind == 0 and b.kind == 0) or (a.kind == 2 and b.kind == 2):
        return _float_special(3, False, 16, target, fail)
    if a.kind == 2 or b.kind == 0:
        return _float_special(
            2, negative, 8 if a.kind == 1 and b.kind == 0 else 0, target, fail
        )
    if a.kind == 0 or b.kind == 2:
        return _float_special(0, negative, 0, target, fail)
    return _round_ratio(
        (-a.numerator if negative else a.numerator) * b.denominator,
        a.denominator * b.numerator,
        target,
        scale=_exponent_add(a.scale, -b.scale),
        fail=fail,
    )
