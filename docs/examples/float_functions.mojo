"""Square roots, powers and other rounded functions."""

from apn_mojo import (
    Float,
    Rational,
    FloatFormat,
    ArithmeticContext,
    square,
    sqrt,
    pow_int,
    ldexp,
    fma,
)


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(53))
    var root = sqrt(Rational(2), context=context)
    print("native root:", root.to_native[DType.float64]())
    print("square:", square(Float(3)))
    print("reciprocal power:", pow_int(Float(2), -3))
    print("scaled:", ldexp(Float(3), -2))
    var fused = fma(
        Rational(9, 8), Rational(9, 8), Rational(-5, 4), context=context
    )
    print("fused residual:", fused)
    var value = Float(Rational(-7, 4))
    print("floor, ceil, trunc:", value.floor(), value.ceil(), value.trunc())
    print(
        "checked native integer:", value.floor().to_native_exact[DType.int64]()
    )
    var saved = value
    value **= -2
    print("power kept precision:", value.precision() == saved.precision())
