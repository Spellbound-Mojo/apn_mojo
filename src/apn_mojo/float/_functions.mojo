"""Strict scalar functions: exact roots/fma and certified integral powers."""

from std.bit import count_leading_zeros
from ..integer.value import Integer, _taken
from ..integer.number_theory import isqrt
from ..integer._square_root import _limb_square_root, _sqrt_rem, _sqrt_rem_native, _wide_magnitude
from ..integer.division import div_rem_trunc
from ..integer.math import iroot
from .context import ArithmeticContext, FloatFormat, RoundingMode
from .status import NumericStatus
from ._input import _FloatInput
from ._format import _merge_float_formats
from ._arithmetic import _FloatArgument, _float_special, _float_sum, _call_context
from ._multiplication import _binary_product
from ._rounding import (
    _RoundedBinary,
    _LIMB_SCALE,
    _round_limb,
    _round_ratio,
    _round_exact_binary,
    _compare_scaled,
    _exponent_add,
    _round_away,
    _rounded_direction,
    _overflow_result,
    _finish_round,
    _inexact_error,
    _is_power_of_two,
    _trailing_zero_bits,
)


def _wide_integer(value: Int128) raises -> Integer:
    var negative = value < 0
    var magnitude = UInt128(-value) if negative else UInt128(value)
    var result = Integer(UInt64(magnitude >> 64)) << 64
    result += UInt64(magnitude)
    return -result if negative else result


def _wide_exponent(value: Integer) raises -> Int128:
    if value.magnitude_bit_length() > 126:
        raise Error(
            "Cannot convert working exponent; reduce the scaling exponent."
        )
    var result = Int128(0)
    for i in range(value._word_count() - 1, -1, -1):
        result = (result << 32) | Int128(value._word(i))
    return -result if value.sign() < 0 else result


def _ratio_exponent(a: Integer, b: Integer) raises -> Int:
    if b._is_one():
        # A binary significand: 2**(bits - 1) <= a < 2**bits.
        return a.magnitude_bit_length()
    var k = a.magnitude_bit_length() - b.magnitude_bit_length()
    return k + Int(_compare_scaled(a, b, Int128(k)) >= 0)


def _destination_context(
    format: FloatFormat, context: Optional[ArithmeticContext]
) raises -> ArithmeticContext:
    if not context:
        return ArithmeticContext(_format_of=format)
    return context.value()._with_format(format)


def _round_scaled_ratio(
    a: Integer,
    b: Integer,
    scale: Integer,
    negative: Bool,
    context: ArithmeticContext,
    fail: Bool,
) raises -> _RoundedBinary:
    var e = scale + _ratio_exponent(a, b)
    if e > context.format().emax():
        return _finish_round(_overflow_result(negative, context), context, fail)
    if e < Integer(context.format().emin()) - 1:
        return _round_ratio(
            Integer(-1 if negative else 1),
            Integer(1),
            context,
            scale=Int128(context.format().emin()) - 3,
            fail=fail,
        )
    return _round_ratio(
        -a if negative else a,
        b,
        context,
        scale=_wide_exponent(scale),
        fail=fail,
    )


def _scale_float(
    value: _FloatArgument,
    exponent: Integer,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    var target = _call_context(value, context)
    var a = value.value
    if a.kind != 1:
        return _float_special(a.kind, a.negative, 0, target, fail)
    if a.denominator._is_one() and exponent.magnitude_bit_length() < 62:
        var exact = _round_exact_binary(a.numerator, a.negative, a.scale + Int128(Int(exponent)), target, fail)
        if exact.kind >= 0:
            return exact^
    return _round_scaled_ratio(
        a.numerator,
        a.denominator,
        _wide_integer(a.scale) + exponent,
        a.negative,
        target,
        fail,
    )


@no_inline
def _one_limb_sqrt(
    x: UInt64, scale: Int128, context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """The rounded square root of x * 2**scale, for a nonzero significand of
    at most 63 bits and a format of at most 63. The significand is
    left-aligned, and shifted once more for an odd exponent (no bit is lost
    below 64 bits), so the root of the limb times 2**64 has exactly 64 bits:
    its round bit, and a sticky from the bits below and the remainder, go to
    _round_limb, which also breaks the ties that inputs of more bits than the
    output allow. Out of line, its operands stay in registers."""
    var lx = count_leading_zeros(x)
    var u = x << lx
    var exponent = Int(scale) + 64 - Int(lx)
    if exponent & 1:
        u >>= 1
        exponent += 1
    var root, inexact = _limb_square_root(u)
    var sh = UInt64(64 - context.format().precision())
    var mask = (UInt64(1) << sh) - 1
    var rb = root & (UInt64(1) << (sh - 1))
    var sb = (root & (mask >> 1)) | UInt64(Int(inexact))
    return _round_limb(root & ~mask, rb, sb, False, exponent // 2, context, fail)


def _sqrt_float(
    value: _FloatArgument,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    var target = _call_context(value, context)
    # Borrowed: copies of heap parts cost an atomic increment each.
    ref source = value.value
    if source.kind == 3 or source.kind == 0:
        return _float_special(source.kind, source.negative, 0, target, fail)
    if source.negative:
        return _float_special(3, False, 16, target, fail)
    if source.kind == 2:
        return _float_special(2, False, 0, target, fail)
    ref a = source.numerator
    ref b = source.denominator
    if target.format()._is_exact():
        return _exact_sqrt(a, b, source.scale, target, fail)
    if (
        target.format().precision() <= 63 and a._storage.isa[Int64]() and b._is_one()
        and source.scale > -_LIMB_SCALE and source.scale < _LIMB_SCALE
    ):
        # A significand of at most 63 bits: one limb.
        var native = _one_limb_sqrt(UInt64(a._storage[Int64]), source.scale, target, fail)
        if native.kind >= 0:
            return native^
    var e = _exponent_add(source.scale, Int128(_ratio_exponent(a, b)))
    e = (e + 1) // 2 if e >= 0 else -((-e) // 2)
    var format = target.format()
    if e < Int128(format.emin()):
        var up = _round_away(target.rounding(), False)
        if target.rounding() == RoundingMode.nearest_even:
            up = (
                _compare_scaled(
                    a, b, 2 * (Int128(format.emin()) - 2) - source.scale
                )
                > 0
            )
        return _finish_round(
            _RoundedBinary(
                1 if up else 0,
                False,
                (Integer(1) << (format.precision() - 1)) if up else Integer(0),
                format.emin() if up else 0,
                format,
                NumericStatus._make(3, 1 if up else -1),
            ),
            target,
            fail,
        )
    if e > Int128(format.emax()):
        return _finish_round(_overflow_result(False, target), target, fail)
    var shift = source.scale + 2 * (Int128(format.precision()) - e)
    if shift > Int128(Int.MAX) or shift < -Int128(Int.MAX):
        raise Error(
            "Cannot take square root: working precision exceeds addressable"
            " storage; choose a smaller output precision. The destination is"
            " unchanged."
        )
    if (
        format.precision() <= 128 and b._is_one() and shift >= 0
        and Int128(a.magnitude_bit_length()) + shift <= 256
    ):
        # Significands up to 128 bits: the root and its remainder in native
        # integers. As below, the root lies above the midpoint exactly when
        # rest > q.
        var native = _sqrt_rem_native(_wide_magnitude(a) << UInt256(Int(shift)))
        var root = UInt256(native[0])
        var exact = native[1] == 0
        var raise_root = False
        if not exact:
            raise_root = _round_away(target.rounding(), False)
            if target.rounding() == RoundingMode.nearest_even:
                raise_root = native[1] > root
            if raise_root:
                root += 1
        var exponent = e
        if root >> UInt256(format.precision()):
            root >>= 1
            exponent += 1
        if exponent > Int128(format.emax()):
            return _finish_round(_overflow_result(False, target), target, fail)
        return _finish_round(
            _RoundedBinary(
                1, False, Integer._from_wide_magnitude256(root), Int(exponent), format,
                NumericStatus._make(Int(not exact), (1 if raise_root else -1) if not exact else 0),
            ),
            target,
            fail,
        )
    var q: Integer
    var inexact: Bool
    var up = False
    if b._is_one() and shift >= 0:
        # a 2**shift = q**2 + rest; a tie would need (q + 1/2)**2, not an
        # integer, so the root lies above the midpoint exactly when rest > q.
        var root = _sqrt_rem(a, Int(shift))
        q = _taken(root[0])
        ref rest = root[1]
        inexact = rest.__bool__()
        if inexact:
            up = _round_away(target.rounding(), False)
            if target.rounding() == RoundingMode.nearest_even:
                up = rest > q
    else:
        var x = a
        var y = b
        if shift >= 0:
            x <<= Int(shift)
        else:
            y <<= Int(-shift)
        q = isqrt(x // y)
        inexact = q * q * y != x
        if inexact:
            up = _round_away(target.rounding(), False)
            if target.rounding() == RoundingMode.nearest_even:
                var middle = q * 2 + 1
                var boundary = middle * middle * y
                var four_a = x << 2
                up = four_a > boundary or (
                    four_a == boundary and Bool(q._word(0) & 1)
                )
    if up:
        q += 1
    if q.magnitude_bit_length() > format.precision():
        q >>= 1
        e += 1
    if e > Int128(format.emax()):
        return _finish_round(_overflow_result(False, target), target, fail)
    return _finish_round(
        _RoundedBinary(
            1,
            False,
            q,
            Int(e),
            format,
            NumericStatus._make(
                Int(inexact), (1 if up else -1) if inexact else 0
            ),
        ),
        target,
        fail,
    )


def _exact_sqrt(
    a: Integer, b: Integer, scale: Int128, target: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """The square root of a positive (a / b) * 2**scale in an exact working
    format: a finite binary fraction only when the value's odd part is a
    perfect square and its power of two is even; otherwise inexact."""
    if not _is_power_of_two(b):
        raise _inexact_error("sqrt")
    var zeros = _trailing_zero_bits(a)
    var odd = a >> zeros
    var power = scale + Int128(zeros) - Int128(b.magnitude_bit_length() - 1)
    if power % 2 != 0:
        raise _inexact_error("sqrt")
    var root = _sqrt_rem(odd)
    if root[1]:
        raise _inexact_error("sqrt")
    return _round_ratio(root[0], Integer(1), target, scale=power // 2, fail=fail)


def _rootn_float(
    value: _FloatArgument,
    n: Int,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    """The correctly rounded n-th root, IEEE 754 `rootn`: an odd root keeps the
    sign, an even root of a negative value is invalid, and the root of a zero
    is that zero (`+0` for an even `n`).

    The integer root `q = floor(|x|**(1/n) 2**t)` has at least `p + 2` bits;
    `2q + s`, with `s = 1` when the root is inexact, lies strictly between the
    same two rounding boundaries as the exact root, so one rounding of it is
    correct in every mode, and an exact root rounds exactly."""
    if n < 1:
        raise Error(String(
            "Cannot take the root of degree ", n, "; rootn needs n >= 1. Use pow for other exponents.",
        ))
    var target = _call_context(value, context)
    var source = value.value
    var odd = n % 2 == 1
    if source.kind == 3:
        return _float_special(3, False, 0, target, fail)
    if source.negative and not odd and source.kind != 0:
        return _float_special(3, False, 16, target, fail)
    if source.kind == 0:
        return _float_special(0, source.negative and odd, 0, target, fail)
    if source.kind == 2:
        return _float_special(2, source.negative, 0, target, fail)
    var a = source.numerator
    var b = source.denominator
    var e = _exponent_add(source.scale, Int128(_ratio_exponent(a, b)))
    var p = Int128(target.format().precision())
    # The root is at least 2**er with er = floor((e - 1) / n).
    var er = (e - 1) // Int128(n)
    var t = p + 2 - er
    var k = source.scale + Int128(n) * t
    if k > Int128(Int.MAX) or k < -Int128(Int.MAX):
        raise Error(
            "Cannot take root: working precision exceeds addressable storage;"
            " choose a smaller output precision. The destination is unchanged."
        )
    var numerator = a
    var denominator = b
    if k >= 0:
        numerator <<= Int(k)
    else:
        denominator <<= Int(-k)
    var pair = div_rem_trunc(numerator, denominator)
    var q = iroot(pair[0], Integer(n))
    var exact = not pair[1] and q ** n == pair[0]
    if target.format()._is_exact() and not exact:
        raise _inexact_error("rootn")
    var marked = q * 2 + Integer(0 if exact else 1)
    return _round_ratio(-marked if source.negative else marked, Integer(1), target, scale=-t - 1, fail=fail)


def _fma_context(
    a: _FloatArgument,
    b: _FloatArgument,
    c: _FloatArgument,
    context: Optional[ArithmeticContext] = None,
) raises -> ArithmeticContext:
    var library: Optional[FloatFormat] = None
    if not context:
        for index in range(3):
            var operand = a if index == 0 else b if index == 1 else c
            if operand.format:
                library = _merge_float_formats(library, operand.format)
    return context.value() if context else ArithmeticContext(
        _format_of=_merge_float_formats(
            library,
            None,
            right_native_precision=max(
                a.native_precision, b.native_precision, c.native_precision
            ),
        )
    )


def _fma_float(
    a: _FloatArgument,
    b: _FloatArgument,
    c: _FloatArgument,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
) raises -> _RoundedBinary:
    var target = _fma_context(a, b, c, context)
    var x = a.value
    var y = b.value
    var z = c.value
    # An invalid product must survive a quiet-NaN addend.
    if (x.kind == 0 and y.kind == 2) or (x.kind == 2 and y.kind == 0):
        return _float_special(3, False, 16, target, fail)
    if x.kind == 3 or y.kind == 3 or z.kind == 3:
        return _float_special(3, False, 0, target, fail)
    var negative = x.negative != y.negative
    if x.kind == 2 or y.kind == 2:
        if z.kind == 2 and z.negative != negative:
            return _float_special(3, False, 16, target, fail)
        return _float_special(2, negative, 0, target, fail)
    if z.kind == 2:
        return _float_special(2, z.negative, 0, target, fail)
    var product = _FloatInput(
        0 if x.kind == 0 or y.kind == 0 else 1,
        negative,
        x.numerator * y.numerator,
        x.denominator * y.denominator,
        _exponent_add(x.scale, y.scale),
    )
    return _float_sum(product, z, target, fail)


@fieldwise_init
struct _PowerBound(ImplicitlyCopyable):
    var magnitude: Integer
    var scale: Integer


def _bound_product(
    a: _PowerBound, b: _PowerBound, precision: Int, upper: Bool
) raises -> _PowerBound:
    var m = a.magnitude * b.magnitude
    var scale = a.scale + b.scale
    var discard = m.magnitude_bit_length() - precision
    if discard > 0:
        var q = m >> discard
        if upper and _compare_scaled(m, q, Int128(discard)) != 0:
            q += 1
        m = q
        scale += discard
    return _PowerBound(m, scale)


def _power_bounds(
    a: Integer,
    b: Integer,
    scale: Int128,
    exponent: Integer,
    precision: Int,
    target: ArithmeticContext,
) raises -> Tuple[_PowerBound, _PowerBound]:
    var k = _ratio_exponent(a, b)
    var shift = Int128(precision) - Int128(k)
    if shift > Int128(Int.MAX) or shift < -Int128(Int.MAX):
        raise Error(
            "Cannot compute power: working precision exceeds addressable"
            " storage; choose a smaller precision or exponent. The destination"
            " is unchanged."
        )
    var numerator = a
    var denominator = b
    if shift >= 0:
        numerator <<= Int(shift)
    else:
        denominator <<= Int(-shift)
    var pair = div_rem_trunc(numerator, denominator)
    var s = _wide_integer(scale) + k - precision
    var base_lo = _PowerBound(pair[0], s)
    var base_hi = _PowerBound(pair[0] + Int(pair[1] != 0), s)
    var lo = _PowerBound(Integer(1), Integer(0))
    var hi = lo
    var order = _compare_scaled(a, b, -scale)
    for bit in range(exponent.magnitude_bit_length() - 1, -1, -1):
        lo = _bound_product(lo, lo, precision, False)
        hi = _bound_product(hi, hi, precision, True)
        if exponent._word(bit // 32) & (UInt32(1) << UInt32(bit % 32)):
            lo = _bound_product(lo, base_lo, precision, False)
            hi = _bound_product(hi, base_hi, precision, True)
        # A monotone exact power cannot return from these strict range bounds.
        if (
            order > 0
            and lo.scale + lo.magnitude.magnitude_bit_length()
            > target.format().emax()
        ):
            return lo, lo
        if (
            order < 0
            and hi.scale + hi.magnitude.magnitude_bit_length()
            < Integer(target.format().emin()) - 1
        ):
            return hi, hi
    return lo, hi


def _same_round(a: _RoundedBinary, b: _RoundedBinary) raises -> Bool:
    return (
        a.kind == b.kind
        and a.negative == b.negative
        and a.significand == b.significand
        and a.exponent == b.exponent
        and a.status == b.status
    )


def _pow_float(
    value: _FloatArgument,
    exponent: Integer,
    context: Optional[ArithmeticContext] = None,
    *,
    fail: Bool = False,
    initial_work: Int = 0,
) raises -> _RoundedBinary:
    var target = _call_context(value, context)
    if not exponent:
        return _round_ratio(Integer(1), Integer(1), target, fail=fail)
    var source = value.value
    var negative = source.negative and Bool(exponent._word(0) & 1)
    if source.kind == 3:
        return _float_special(3, False, 0, target, fail)
    if source.kind == 0 or source.kind == 2:
        var reciprocal = exponent.sign() < 0
        var infinity = (source.kind == 2) != reciprocal
        return _float_special(
            2 if infinity else 0,
            negative,
            8 if source.kind == 0 and reciprocal else 0,
            target,
            fail,
        )
    var a = source.numerator
    var b = source.denominator
    var scale = source.scale
    if exponent.sign() < 0:
        var saved = a
        a = b
        b = saved
        scale = -scale
    var count = abs(exponent)
    if target.format()._is_exact():
        # Exact working values: the power itself. A negative exponent of a
        # value that is not a power of two leaves an odd denominator, which
        # rounding to the exact format reports as an inexact step.
        var n = Int(count)
        var numerator = a ** n
        return _round_ratio(
            -numerator if negative else numerator, b ** n, target,
            scale=scale * Int128(n), fail=fail,
        )
    if _is_power_of_two(a) and _is_power_of_two(b):
        var exact_scale = (
            _wide_integer(scale)
            + a.magnitude_bit_length()
            - b.magnitude_bit_length()
        ) * count
        return _round_scaled_ratio(
            Integer(1), Integer(1), exact_scale, negative, target, fail
        )
    if (
        count <= 64 and initial_work == 0
        and (a.magnitude_bit_length() + b.magnitude_bit_length()) * Int(count) <= 16384
        and scale < Int128(1) << 64 and scale > -(Int128(1) << 64)
    ):
        # Small exponents: the exact power is short enough to compute and
        # round once, with no bounds to refine.
        var n = Int(count)
        if b._is_one():
            # A binary base: the power is a product, rounded once by the
            # product rounding (native up to 128-bit significands).
            return _binary_product(a ** (n - 1), a, negative, scale * Int128(n), target, fail)
        var numerator = a ** n
        return _round_ratio(
            -numerator if negative else numerator, b ** n, target,
            scale=scale * Int128(n), fail=fail,
        )
    var work = (
        Int128(target.format().precision())
        + Int128(count.magnitude_bit_length())
        + 16
    )
    # Private qualification hook forces ambiguous bounds and refinement.
    if initial_work > 0:
        work = Int128(initial_work)
    var quiet = ArithmeticContext(
        format=target.format(), rounding=target.rounding()
    )
    while work <= Int128(FloatFormat.MAX_PRECISION):
        var lo, hi = _power_bounds(a, b, scale, count, Int(work), target)
        var lower = _round_scaled_ratio(
            lo.magnitude, Integer(1), lo.scale, negative, quiet, False
        )
        var upper = _round_scaled_ratio(
            hi.magnitude, Integer(1), hi.scale, negative, quiet, False
        )
        if _same_round(lower, upper):
            return _finish_round(lower, target, fail)
        work *= 2
    raise Error(
        "Cannot certify power rounding within addressable precision; choose a"
        " smaller output precision or exponent. The destination is"
        " unchanged."
    )


def _float_integer(value: _FloatInput, mode: Int) raises -> Integer:
    # The caller supplies a stored Float, whose denominator is one.
    # Modes: truncation=0, floor=1, ceiling=2, exact conversion=3, nearest with
    # ties to even=4.
    if value.kind >= 2:
        raise Error(
            "Cannot convert a nonfinite Float to Integer; check is_finite()"
            " before floor(), ceil(), trunc(), or exact conversion. The"
            " destination is unchanged."
        )
    if value.kind == 0:
        return Integer(0)
    var magnitude = value.numerator
    var remainder = False
    # For ties to even: the fraction against one half, as -1, 0 or 1.
    var half = -1
    if value.scale >= 0:
        if value.scale > Int128(Int.MAX):
            raise Error(
                "Cannot convert Float to Integer: the result exceeds"
                " addressable storage; keep the value as Float or scale it"
                " down. The destination is unchanged."
            )
        magnitude <<= Int(value.scale)
    elif -value.scale >= Int128(magnitude.magnitude_bit_length()):
        magnitude = Integer(0)
        remainder = True
        if mode == 4:
            half = _compare_scaled(value.numerator, Integer(1), -value.scale - 1)
    else:
        magnitude >>= Int(-value.scale)
        remainder = (
            _compare_scaled(value.numerator, magnitude, -value.scale) != 0
        )
        if mode == 4 and remainder:
            half = _compare_scaled(value.numerator, (magnitude << 1) + 1, -value.scale - 1)
    if remainder:
        if mode == 3:
            raise Error(
                "Cannot convert Float exactly to Integer: the value has a"
                " fractional part; keep it as Float or explicitly choose"
                " floor(), ceil(), or trunc(). The destination is unchanged."
            )
        if (mode == 1 and value.negative) or (mode == 2 and not value.negative):
            magnitude += 1
        elif mode == 4 and (half > 0 or (half == 0 and (magnitude._word(0) & 1) != 0)):
            magnitude += 1
    return -magnitude if value.negative else magnitude
