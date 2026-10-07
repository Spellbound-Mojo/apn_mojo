"""Exact scalar input decoding, before any requested Float rounding."""

from std.sys import bit_width_of
from ..integer.value import Integer
from ..rational.value import Rational


@fieldwise_init
struct _FloatInput(ImplicitlyCopyable):
    var kind: Int
    var negative: Bool
    var numerator: Integer
    var denominator: Integer
    var scale: Int128


def _float_input[T: Copyable](value: T) -> _FloatInput:
    comptime if T == Integer:
        return _float_input(rebind[Integer](value))
    elif T == Rational:
        return _float_input(rebind[Rational](value))
    elif T == Int:
        return _float_input(rebind[Int](value))
    elif T == Bool:
        return _float_input(rebind[Bool](value))
    elif T == Int8:
        return _float_input(rebind[Int8](value))
    elif T == UInt8:
        return _float_input(rebind[UInt8](value))
    elif T == Int16:
        return _float_input(rebind[Int16](value))
    elif T == UInt16:
        return _float_input(rebind[UInt16](value))
    elif T == Int32:
        return _float_input(rebind[Int32](value))
    elif T == UInt32:
        return _float_input(rebind[UInt32](value))
    elif T == Int64:
        return _float_input(rebind[Int64](value))
    elif T == UInt64:
        return _float_input(rebind[UInt64](value))
    elif T == Float16:
        return _float_input(rebind[Float16](value))
    elif T == BFloat16:
        return _float_input(rebind[BFloat16](value))
    elif T == Float32:
        return _float_input(rebind[Float32](value))
    elif T == Float64:
        return _float_input(rebind[Float64](value))
    else:
        comptime assert T == Integer, (
            "Float expects Integer, Rational, or a supported native numeric"
            " value; Bool and text are not accepted"
        )
        return _FloatInput(0, False, Integer(0), Integer(1), 0)


def _float_input(value: Integer) -> _FloatInput:
    return _FloatInput(
        Int(value != 0), value.sign() < 0, abs(value), Integer(1), 0
    )


def _float_input(value: Rational) -> _FloatInput:
    return _FloatInput(
        Int(value.sign() != 0),
        value.sign() < 0,
        abs(value.numerator()),
        value.denominator(),
        0,
    )


def _float_input(value: Int) -> _FloatInput:
    return _float_input(Integer(value))


def _float_input(value: IntLiteral) -> _FloatInput:
    return _float_input(Integer(value))


def _float_input(value: Bool) -> _FloatInput:
    comptime assert (
        False
    ), "Float does not accept Bool; choose Integer(0) or Integer(1) explicitly"


def _float_input(value: FloatLiteral) -> _FloatInput:
    comptime assert False, (
        "Exact decimal Float literals are not qualified yet; use"
        ' Float("0.1") for an exact decimal source, or'
        " Float(Float64(...)) for a native stored value"
    )


def _float_input[dtype: DType](value: SIMD[dtype, 1]) -> _FloatInput:
    comptime if dtype.is_integral() and dtype != DType.bool:
        return _float_input(Integer(value))
    else:
        comptime assert (
            dtype == DType.float16
            or dtype == DType.bfloat16
            or dtype == DType.float32
            or dtype == DType.float64
        ), (
            "Float accepts native integers through 64 bits and Float16,"
            " BFloat16, Float32 or Float64; use Integer/Rational for an exact"
            " wider source"
        )
        comptime fraction_bits = 52 if dtype == DType.float64 else 23 if dtype == DType.float32 else 10 if dtype == DType.float16 else 7
        comptime exponent_bits = bit_width_of[dtype]() - fraction_bits - 1
        comptime exponent_mask = (1 << exponent_bits) - 1
        comptime bias = (1 << (exponent_bits - 1)) - 1
        var bits = value.to_bits[DType.uint64]()
        var negative = Bool(bits >> UInt64(bit_width_of[dtype]() - 1))
        var exponent = Int(
            (bits >> UInt64(fraction_bits)) & UInt64(exponent_mask)
        )
        var fraction = bits & ((UInt64(1) << UInt64(fraction_bits)) - 1)
        if exponent == exponent_mask:
            return _FloatInput(
                3 if fraction else 2,
                negative and not fraction,
                Integer(0),
                Integer(1),
                0,
            )
        var magnitude = (
            fraction
            | (UInt64(1) << UInt64(fraction_bits)) if exponent else fraction
        )
        return _FloatInput(
            Int(magnitude != 0),
            negative,
            Integer(magnitude),
            Integer(1),
            Int128((exponent if exponent else 1) - bias - fraction_bits),
        )
