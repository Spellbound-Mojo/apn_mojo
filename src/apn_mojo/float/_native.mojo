"""Direct IEEE binary output, including gradual underflow, using integer work."""

from std.memory import bitcast
from std.sys import bit_width_of
from ..integer.value import Integer
from .context import ArithmeticContext, FloatFormat, RoundingMode
from .status import NumericStatus
from ._input import _FloatInput
from ._rounding import (
    _round_ratio,
    _compare_scaled,
    _round_away,
    _rounded_direction,
)


@fieldwise_init
struct _NativeFloatResult[dtype: DType](ImplicitlyCopyable):
    var value: SIMD[Self.dtype, 1]
    var status: NumericStatus


def _native_bits[dtype: DType](bits: UInt64) -> SIMD[dtype, 1]:
    comptime if bit_width_of[dtype]() == 16:
        return bitcast[dtype](UInt16(bits))
    elif bit_width_of[dtype]() == 32:
        return bitcast[dtype](UInt32(bits))
    else:
        return bitcast[dtype](bits)


def _float_native[
    dtype: DType
](
    source: _FloatInput,
    rounding: RoundingMode,
    *,
    fail: Bool = False,
) raises -> _NativeFloatResult[dtype]:
    # The caller supplies a stored Float, whose denominator is one.
    comptime assert (
        dtype == DType.float16
        or dtype == DType.bfloat16
        or dtype == DType.float32
        or dtype == DType.float64
    ), (
        "Float.to_native supports Float16, BFloat16, Float32 and Float64; use"
        " to_native_exact for a checked integral conversion"
    )
    rounding._validate()
    comptime fraction_bits = 52 if dtype == DType.float64 else 23 if dtype == DType.float32 else 10 if dtype == DType.float16 else 7
    comptime exponent_bits = bit_width_of[dtype]() - fraction_bits - 1
    comptime exponent_mask = (1 << exponent_bits) - 1
    comptime bias = (1 << (exponent_bits - 1)) - 1
    comptime emin = 2 - bias
    comptime emax = bias + 1
    comptime quantum = emin - 1 - fraction_bits
    var bits = UInt64(source.negative) << UInt64(bit_width_of[dtype]() - 1)
    var status = NumericStatus()
    if source.kind == 3:
        bits = (UInt64(exponent_mask) << UInt64(fraction_bits)) | (
            UInt64(1) << UInt64(fraction_bits - 1)
        )
        status = NumericStatus._make(0, 2)
    elif source.kind == 2:
        bits |= UInt64(exponent_mask) << UInt64(fraction_bits)
    elif source.kind == 1:
        var e = source.scale + Int128(source.numerator.magnitude_bit_length())
        if e < Int128(emin):
            var shift = source.scale - Int128(quantum)
            var q = Integer(0)
            var inexact = False
            var up = False
            if shift >= 0:
                q = source.numerator << Int(shift)
            else:
                var cut = -shift
                if cut < Int128(source.numerator.magnitude_bit_length()):
                    q = source.numerator >> Int(cut)
                inexact = (
                    not q or _compare_scaled(source.numerator, q, cut) != 0
                )
                if inexact:
                    up = _round_away(rounding, source.negative)
                    if rounding == RoundingMode.nearest_even:
                        var order = _compare_scaled(
                            source.numerator, q * 2 + 1, cut - 1
                        )
                        up = order > 0 or (order == 0 and Bool(q._word(0) & 1))
                    if up:
                        q += 1
            bits |= q.to_native_exact[DType.uint64]()
            status = NumericStatus._make(
                3 if inexact else 0,
                _rounded_direction(source.negative, up) if inexact else 0,
            )
        else:
            var rounded = _round_ratio(
                -source.numerator if source.negative else source.numerator,
                Integer(1),
                ArithmeticContext(
                    format=FloatFormat(fraction_bits + 1, emin=emin, emax=emax),
                    rounding=rounding,
                ),
                scale=source.scale,
            )
            status = rounded.status
            if rounded.kind == 2:
                bits |= UInt64(exponent_mask) << UInt64(fraction_bits)
            else:
                bits |= UInt64(rounded.exponent - 1 + bias) << UInt64(
                    fraction_bits
                )
                bits |= rounded.significand.to_native_exact[DType.uint64]() - (
                    UInt64(1) << UInt64(fraction_bits)
                )
    if fail:
        raise Error(
            "Injected native conversion failure before publication; retry the"
            " conversion. The destination is unchanged."
        )
    return _NativeFloatResult[dtype](_native_bits[dtype](bits), status)


def _ratio_native[dtype: DType](
    negative: Bool, numerator: Integer, denominator: Integer, rounding: RoundingMode,
) raises -> SIMD[dtype, 1]:
    """`numerator / denominator`, positive magnitudes, signed and rounded once to
    a native floating-point type, subnormals included. A quotient of 66 or more
    bits with the remainder folded into its lowest bit rounds as the ratio does
    at every native precision, so `_float_native` finishes it."""
    var shift = 66 + denominator.magnitude_bit_length() - numerator.magnitude_bit_length()
    var a = numerator << shift if shift > 0 else numerator
    var b = denominator << -shift if shift < 0 else denominator
    var parts = a._div_rem_trunc(b)
    var sticky = (parts[0] << 1) + (Integer(1) if parts[1] else Integer(0))
    return _float_native[dtype](_FloatInput(1, negative, sticky, Integer(1), Int128(-shift - 1)), rounding).value
