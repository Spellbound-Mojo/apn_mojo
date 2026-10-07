"""Exact sparse magnitudes and certified principal complex square roots."""

from ..integer.value import Integer
from ..common._sizes import _checked_sum
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._input import _FloatInput
from ..float._arithmetic import _FloatArgument, _float_special
from std.bit import count_leading_zeros
from ..float._rounding import (
    _RoundedBinary, _round_ratio, _finish_round, _inexact_error, _round_away, _rounded_direction,
    _native_quotient, _short_magnitude,
)
from ..integer._square_root import _sqrt_rem_native
from ..float._functions import _sqrt_float
from ..float.status import NumericStatus
from ..integer.number_theory import isqrt
from ..float._accumulator import _ExactAccumulator, _SparseMagnitude
from ._input import _ComplexArgument
from ._format import _ComplexFormats
from .context import ComplexContext, _ComplexContextArgument
from ._finite import _ExactPair, _finite_product, _magnitude_top, _negated


def _squared_magnitude(a: _FloatInput, b: _FloatInput) raises -> _ExactPair:
    return _ExactPair(_finite_product(a, a), _finite_product(b, b), False)


def _positive_input(value: _RoundedBinary) -> _FloatInput:
    return _FloatInput(
        value.kind,
        False,
        value.significand,
        Integer(1),
        Int128(value.exponent) - Int128(value.format.precision()) if value.kind
        == 1 else 0,
    )


def _sparse_sqrt(
    value: _SparseMagnitude, context: ArithmeticContext
) raises -> _RoundedBinary:
    if not len(value.runs):
        return _float_special(0, False, 0, context, False)
    var top = _magnitude_top(value)
    var exponent = (top + 1) // 2 if top >= 0 else -((-top) // 2)
    var bits = 4
    if (
        Int128(context.format().emin())
        <= exponent
        <= Int128(context.format().emax())
    ):
        bits = _checked_sum(
            _checked_sum(
                context.format().precision(), context.format().precision()
            ),
            4,
        )
    # Squared rounding boundaries are even at this scale; sticky jamming keeps
    # their exact side and equality, including perfect-square endpoints.
    var source = _FloatInput(
        1, False, value.prefix(top, bits), Integer(1), top - Int128(bits)
    )
    return _sqrt_float(_FloatArgument(source, None, 0), context)


def _complex_real_function(
    value: _ComplexArgument,
    operation: Int,
    context: Optional[ArithmeticContext] = None,
) raises -> _RoundedBinary:
    var formats = _ComplexFormats(
        value.real.format.value(), value.imag.format.value()
    )
    var target = context.value() if context else ArithmeticContext(
        format=formats.real_result_format()
    )
    var a = value.real.value
    var b = value.imag.value
    if a.kind == 2 or b.kind == 2:
        return _float_special(2, False, 0, target, False)
    if a.kind == 3 or b.kind == 3:
        return _float_special(3, False, 0, target, False)
    var squared = _squared_magnitude(a, b)
    if operation == 0:
        return squared.rounded(target)
    var magnitude = squared.magnitude()
    return _sparse_sqrt(magnitude, target)


def _root_compare(
    a: _FloatInput, b: _FloatInput, threshold: Integer, scale: Int128
) raises -> Int:
    # For x=sqrt((sqrt(a*a+b*b)+a)/2), first guard the sign of
    # 2*t*t-a; squaring then reduces the comparison to b*b+4*a*t*t-4*t**4.
    var square = threshold * threshold
    var guard = _ExactAccumulator()
    guard.add(square, 2 * scale + 1)
    guard.add(a.numerator if a.negative else -a.numerator, a.scale)
    var sign = guard.magnitude()
    if sign.negative:
        return 1
    var difference = _ExactAccumulator()
    difference.add(b.numerator * b.numerator, 2 * b.scale)
    difference.add(
        (-a.numerator if a.negative else a.numerator) * square,
        a.scale + 2 * scale + 2,
    )
    difference.add(-(square * square), 4 * scale + 2)
    var order = difference.magnitude()
    return (-1 if order.negative else 1) if len(order.runs) else 0


@fieldwise_init
struct _RootEstimate(ImplicitlyCopyable):
    var magnitude: Integer
    var scale: Int128


def _root_estimate(a: _FloatInput, b: _FloatInput, work: Int) raises -> _RootEstimate:
    var major = _major_root(a, b, work)
    if not a.negative or a.kind == 0:
        return major
    return _minor_root(b, major, work)


def _scaled(magnitude: Integer, shift: Int128) raises -> Integer:
    """floor(magnitude * 2**shift)."""
    if shift >= 0:
        return magnitude << Int(shift)
    if -shift >= Int128(magnitude.magnitude_bit_length()):
        return Integer(0)
    return magnitude >> Int(-shift)


def _fixed_square(value: _FloatInput, power: Int128) raises -> Integer:
    """floor(value**2 * 4**power); a value below 2**-power is not squared."""
    if not value.kind or value.scale + power + Int128(value.numerator.magnitude_bit_length()) <= 0:
        return Integer(0)
    return _scaled(value.numerator * value.numerator, 2 * (value.scale + power))


def _major_root(a: _FloatInput, b: _FloatInput, work: Int) raises -> _RootEstimate:
    """sqrt((hypot(a, b) + |a|) / 2), the larger component: at least work + 1
    bits, low by less than 2**-(work - 1), relatively.

    In fixed point with f = work + 2 fractional bits, after scaling by an even
    power 2**-k that puts the larger of |a| and |b| in [1/2, 2): the truncated
    root of floor((a**2 + b**2) 4**f), plus floor(|a| 2**f), then the truncated
    root of that sum times 2**(f - 1). The complex chapter bounds the error.
    """
    var top = max(
        a.scale + Int128(a.numerator.magnitude_bit_length()) if a.kind else b.scale,
        b.scale + Int128(b.numerator.magnitude_bit_length()),
    )
    var k = (top // 2) * 2
    var f = Int128(work + 2)
    var power = f - k
    var total = isqrt(_fixed_square(a, power) + _fixed_square(b, power))
    if a.kind:
        total += _scaled(a.numerator, a.scale + power)
    return _RootEstimate(isqrt(total << Int(f - 1)), k // 2 - f)


def _minor_root(b: _FloatInput, major: _RootEstimate, work: Int) raises -> _RootEstimate:
    """|b| / (2 * major), the smaller component without the cancellation in
    (hypot(a, b) - |a|) / 2: a floor quotient of at least work + 2 bits, within
    2**-(work - 2) relatively, given the major root's bound."""
    var shift = Int128(
        work + 2 + major.magnitude.magnitude_bit_length() - b.numerator.magnitude_bit_length()
    )
    return _RootEstimate(
        _scaled(b.numerator, shift) // major.magnitude,
        b.scale - major.scale - 1 - shift,
    )


# Guard bits of the fast estimate, and its relative error bound in units of
# 2**-work: the estimates err by less than 2**2, so 2**4 leaves a margin.
comptime _GUARD_BITS = 32
comptime _ERROR_ULPS_LOG2 = 4


def _uniform_bits(m: Integer, low: Int, high: Int) -> Tuple[Bool, Bool]:
    """Whether bits low to high (exclusive) of a magnitude are all clear, and
    whether they are all set."""
    var clear = True
    var set = True
    var i = low
    while i < high and (clear or set):
        var offset = i % 32
        var count = min(32 - offset, high - i)
        var mask = UInt32((UInt64(1) << UInt64(count)) - 1)
        var part = (m._word(i // 32) >> UInt32(offset)) & mask
        clear = clear and part == 0
        set = set and part == mask
        i += count
    return (clear, set)


def _certified(
    estimate: _RootEstimate, work: Int, negative: Bool, context: ArithmeticContext,
    mut result: _RoundedBinary,
) raises -> Int:
    """A root component rounded from a `work`-bit estimate into `result`, if
    its error bound decides the rounding (0); otherwise 2 when a midpoint kept
    it undecided, for a retry, and 1 when the exact certification decides.

    The root lies within 2**-(work - 4) of the estimate m, relatively: within
    2**s units of m's last bit, for s = bits(m) - work + 4. Rounding to p bits
    changes at boundaries every 2**g units, g = t - 1 for nearest (midpoints and
    representable values) and g = t otherwise, for t = bits(m) - p. With T the
    low t bits of m, (T + 2**s) mod 2**g >= 2**(s + 2) puts every boundary more
    than 2**s units away: the root rounds as m does, inexact in its direction.
    The test reads the tail's words, however long: near-ties need all of it.
    Near the exponent range's ends, the exact certification decides.
    """
    var format = context.format()
    ref m = estimate.magnitude
    var bits = m.magnitude_bit_length()
    var top = estimate.scale + Int128(bits)
    if not m or top <= Int128(format.emin()) + 2 or top >= Int128(format.emax()) - 2:
        return 1
    var p = format.precision()
    var s = bits - work + _ERROR_ULPS_LOG2
    var t = bits - p
    if t < s + 4:
        return 1
    var nearest = context.rounding() == RoundingMode.nearest_even
    # (T + 2**s) mod 2**g < 2**(s + 2) exactly when T < 3 * 2**s (bits s + 2
    # to g clear, bits s and s + 1 not both set) or T >= 2**g - 2**s (bits s
    # to g all set).
    var g = t - 1 if nearest else t
    var upper = _uniform_bits(m, s + 2, g)
    var edge = _uniform_bits(m, s, s + 2)
    var half = Bool((m._word((t - 1) // 32) >> UInt32((t - 1) % 32)) & 1)
    var below = upper[0] and not edge[1]
    if below or (upper[1] and edge[1]):
        # A boundary within the error bound: for nearest, a midpoint when it is
        # an odd multiple of 2**(t - 1); otherwise a representable value,
        # possibly the root itself.
        return 2 if nearest and (half if below else not half) else 1
    var up = half if nearest else _round_away(context.rounding(), negative)
    var significand = m >> t
    var exponent = Int(top)
    if up:
        significand += 1
        if significand.magnitude_bit_length() > p:
            significand >>= 1
            exponent += 1
    result = _finish_round(
        _RoundedBinary(
            1, negative, significand^, exponent, format,
            NumericStatus._make(1, _rounded_direction(negative, up)),
        ),
        context,
        False,
    )
    return 0


# The native twins of _major_root, _minor_root and _certified: the same
# arithmetic, error bound and decisions in 128- and 256-bit integers, for
# working precisions up to _NATIVE_WORK and significands up to 128 bits. With
# f = work + 2 fractional bits, the scaled terms are below 2**(f + 1), the
# sum of squares below 2**(2f + 3), the roots below 2**(f + 2) and the
# shifted dividend of the minor root below 2**(2f + 2): all within 256 bits
# for f <= 122, and both roots within 128.
comptime _NATIVE_WORK = 120


@fieldwise_init
struct _NativeRoots(ImplicitlyCopyable):
    var native: Bool
    var major: UInt128
    var major_scale: Int128
    var minor: UInt128
    var minor_scale: Int128


@always_inline
def _native_scaled(magnitude: UInt256, shift: Int128) -> UInt256:
    """floor(magnitude * 2**shift), for a result below 2**256."""
    if shift >= 0:
        return magnitude << UInt256(shift)
    if -shift >= 256:
        return 0
    return magnitude >> UInt256(-shift)


@always_inline
def _native_fixed_square(value: _FloatInput, magnitude: UInt128, power: Int128) -> UInt256:
    """_fixed_square of a significand of at most 128 bits."""
    var bits = 128 - Int(count_leading_zeros(magnitude))
    if not value.kind or value.scale + power + Int128(bits) <= 0:
        return 0
    return _native_scaled(UInt256(magnitude) * UInt256(magnitude), 2 * (value.scale + power))


def _native_roots(a: _FloatInput, b: _FloatInput, work: Int) -> _NativeRoots:
    """_major_root and then _minor_root, natively when they fit (`native`)."""
    var none = _NativeRoots(False, 0, 0, 0, 0)
    if work > _NATIVE_WORK or not a.denominator._is_one() or not b.denominator._is_one():
        return none
    var short_a = _short_magnitude(a.numerator)
    var short_b = _short_magnitude(b.numerator)
    if not short_a or not short_b:
        return none
    var x = short_a.value()
    var y = short_b.value()
    var y_bits = 128 - Int(count_leading_zeros(y))
    var top = max(
        a.scale + Int128(128 - Int(count_leading_zeros(x))) if a.kind else b.scale,
        b.scale + Int128(y_bits),
    )
    var k = (top // 2) * 2
    var f = Int128(work + 2)
    var power = f - k
    var total = UInt256(_sqrt_rem_native(_native_fixed_square(a, x, power) + _native_fixed_square(b, y, power))[0])
    if a.kind:
        total += _native_scaled(UInt256(x), a.scale + power)
    var major = _sqrt_rem_native(total << UInt256(f - 1))[0]
    var major_scale = k // 2 - f
    var shift = Int128(work + 2 + 128 - Int(count_leading_zeros(major)) - y_bits)
    var minor = _native_quotient(_native_scaled(UInt256(y), shift), major)
    return _NativeRoots(True, major, major_scale, minor, b.scale - major_scale - 1 - shift)


@always_inline
def _native_uniform(m: UInt128, low: Int, high: Int) -> Tuple[Bool, Bool]:
    """_uniform_bits of a native magnitude, for 0 <= low and high <= 128."""
    if low >= high:
        return (True, True)
    var mask = (UInt128.MAX >> UInt128(128 - (high - low))) << UInt128(low)
    var part = m & mask
    return (part == 0, part == mask)


def _native_certified(
    m: UInt128, scale: Int128, work: Int, negative: Bool, context: ArithmeticContext,
    mut result: _RoundedBinary,
) raises -> Int:
    """_certified of a native estimate m * 2**scale: the same test, codes and
    rounding."""
    var format = context.format()
    var bits = 128 - Int(count_leading_zeros(m))
    var top = scale + Int128(bits)
    if not m or top <= Int128(format.emin()) + 2 or top >= Int128(format.emax()) - 2:
        return 1
    var p = format.precision()
    var s = bits - work + _ERROR_ULPS_LOG2
    var t = bits - p
    if t < s + 4 or s < 0:
        return 1
    var nearest = context.rounding() == RoundingMode.nearest_even
    var g = t - 1 if nearest else t
    var upper = _native_uniform(m, s + 2, g)
    var edge = _native_uniform(m, s, s + 2)
    var half = Bool((m >> UInt128(t - 1)) & 1)
    var below = upper[0] and not edge[1]
    if below or (upper[1] and edge[1]):
        return 2 if nearest and (half if below else not half) else 1
    var up = half if nearest else _round_away(context.rounding(), negative)
    var significand = m >> UInt128(t)
    var exponent = Int(top)
    if up:
        significand += 1
        if significand >> UInt128(p):
            significand >>= 1
            exponent += 1
    result = _finish_round(
        _RoundedBinary(
            1, negative, Integer._from_wide_magnitude128(significand), exponent, format,
            NumericStatus._make(1, _rounded_direction(negative, up)),
        ),
        context,
        False,
    )
    return 0


def _root_component(
    a: _FloatInput,
    b: _FloatInput,
    negative: Bool,
    context: ArithmeticContext,
    initial_work: Int,
) raises -> _RoundedBinary:
    var work = (
        Int128(initial_work) if initial_work
        > 0 else Int128(context.format().precision()) + 16
    )
    while work <= Int128(FloatFormat.MAX_PRECISION):
        var estimate = _root_estimate(a, b, Int(work))
        var exponent = estimate.scale + Int128(
            estimate.magnitude.magnitude_bit_length()
        )
        while _root_compare(a, b, Integer(1), exponent - 1) < 0:
            exponent -= 1
        while _root_compare(a, b, Integer(1), exponent) >= 0:
            exponent += 1
        var format = context.format()
        if exponent < Int128(format.emin()):
            var order = _root_compare(
                a, b, Integer(1), Int128(format.emin()) - 2
            )
            var witness = Integer(1 if order < 0 else 2 if order == 0 else 3)
            return _round_ratio(
                -witness if negative else witness,
                Integer(1),
                context,
                scale=Int128(format.emin()) - 3,
            )
        if exponent > Int128(format.emax()):
            return _round_ratio(
                Integer(-1 if negative else 1),
                Integer(1),
                context,
                scale=exponent - 1,
            )
        var scale = exponent - Int128(format.precision())
        var shift = estimate.scale - scale
        var q = (estimate.magnitude << Int(shift)) if shift >= 0 else (
            estimate.magnitude >> Int(-shift)
        )
        var order = _root_compare(a, b, q, scale)
        if order < 0:
            q -= 1
            order = _root_compare(a, b, q, scale)
        elif _root_compare(a, b, q + 1, scale) >= 0:
            q += 1
            order = _root_compare(a, b, q, scale)
        if order < 0 or _root_compare(a, b, q + 1, scale) >= 0:
            work *= 2
            continue
        if order == 0:
            return _round_ratio(
                -q if negative else q,
                Integer(1),
                context,
                scale=scale,
            )
        var middle = _root_compare(a, b, q * 2 + 1, scale - 1)
        var witness = q * 4 + (1 if middle < 0 else 2 if middle == 0 else 3)
        return _round_ratio(
            -witness if negative else witness,
            Integer(1),
            context,
            scale=scale - 2,
        )
    raise Error(
        "Cannot certify Complex square root within addressable precision;"
        " choose a smaller output precision. The destination is"
        " unchanged."
    )


def _axis_root(
    a: _FloatInput, negative: Bool, context: ArithmeticContext
) raises -> _RoundedBinary:
    var mode = context.rounding()
    if negative:
        if mode == RoundingMode.toward_positive:
            mode = RoundingMode.toward_negative
        elif mode == RoundingMode.toward_negative:
            mode = RoundingMode.toward_positive
    var quiet = ArithmeticContext(format=context.format(), rounding=mode)
    var source = a
    source.negative = False
    var result = _sqrt_float(_FloatArgument(source, None, 0), quiet)
    result.negative = negative
    if negative:
        result.status._direction = -result.status._direction
    return _finish_round(result, context, False)


def _sqrt_complex(
    value: _ComplexArgument,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *,
    initial_work: Int = 0,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var target = context.value() if context else ComplexContext(
        real=ArithmeticContext(format=value.real.format.value()),
        imag=ArithmeticContext(format=value.imag.format.value()),
    )
    if target.real().format()._is_exact() or target.imag().format()._is_exact():
        raise _inexact_error("sqrt")
    var a = value.real.value
    var b = value.imag.value
    if a.kind >= 2 or b.kind >= 2:
        var rk = 3
        var ik = 3
        if b.kind == 2:
            rk = 2
            ik = 2
        elif a.kind == 2:
            rk = (3 if b.kind == 3 else 0) if a.negative else 2
            ik = 2 if a.negative else (3 if b.kind == 3 else 0)
        var real = _float_special(rk, False, 0, target.real(), False)
        var imag = _float_special(ik, b.negative, 0, target.imag(), False)
        return (real, imag)
    if b.kind == 0:
        var real: _RoundedBinary
        var imag: _RoundedBinary
        if a.negative and a.kind != 0:
            real = _float_special(0, False, 0, target.real(), False)
            imag = _axis_root(a, b.negative, target.imag())
        else:
            real = _axis_root(a, False, target.real())
            imag = _float_special(0, b.negative, 0, target.imag(), False)
        return (real, imag)
    # Each part's context once, and its result in place: records copied
    # through Optionals and tuples cost more than the native estimate.
    var real_context = target.real()
    var imag_context = target.imag()
    var real = _RoundedBinary(0, False, Integer(0), 0, real_context.format(), NumericStatus())
    var imag = _RoundedBinary(0, False, Integer(0), 0, imag_context.format(), NumericStatus())
    var real_done = False
    var imag_done = False
    if initial_work == 0:
        # Estimate both components with guard bits; the larger is computed
        # once and the smaller is |b| / (2 * larger). A component within the
        # error bound of a midpoint gets one retry with about twice the
        # precision, which settles near-ties short of exact ones; one near a
        # representable value, possibly exact, goes to the exact certification.
        var real_major = not a.negative or a.kind == 0
        var widest = max(real_context.format().precision(), imag_context.format().precision())
        var work = widest + _GUARD_BITS
        while True:
            var retry = False
            # Up to _NATIVE_WORK bits (53-bit parts take 85) natively: the
            # Integer estimate is a dozen allocating operations.
            var native = _native_roots(a, b, work)
            if native.native:
                if not real_done:
                    var outcome = _native_certified(
                        native.major if real_major else native.minor,
                        native.major_scale if real_major else native.minor_scale,
                        work, False, real_context, real,
                    )
                    real_done = outcome == 0
                    retry = outcome == 2
                if not imag_done:
                    var outcome = _native_certified(
                        native.minor if real_major else native.major,
                        native.minor_scale if real_major else native.major_scale,
                        work, b.negative, imag_context, imag,
                    )
                    imag_done = outcome == 0
                    retry = retry or outcome == 2
            else:
                var major = _major_root(a, b, work)
                var minor = _minor_root(b, major, work)
                if not real_done:
                    var outcome = _certified(major if real_major else minor, work, False, real_context, real)
                    real_done = outcome == 0
                    retry = outcome == 2
                if not imag_done:
                    var outcome = _certified(minor if real_major else major, work, b.negative, imag_context, imag)
                    imag_done = outcome == 0
                    retry = retry or outcome == 2
            if (real_done and imag_done) or not retry or work > widest + _GUARD_BITS:
                break
            work = 2 * widest + _GUARD_BITS
    if not real_done:
        real = _root_component(a, b, False, real_context, initial_work)
    if not imag_done:
        imag = _root_component(_negated(a), b, b.negative, imag_context, initial_work)
    return (real^, imag^)
