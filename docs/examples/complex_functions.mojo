"""Magnitudes and square roots."""

from apn_mojo import (
    Complex,
    Float,
    FloatFormat,
    ArithmeticContext,
    abs,
    norm_sqr,
    sqrt,
)


def main() raises:
    var z = Complex(3, -4)
    print(norm_sqr(z))
    print(abs(z))
    print(sqrt(z))
    print(sqrt(Complex(-4, Float.zero())))
    print(sqrt(Complex(-4, Float.zero(negative=True))))
    var result = sqrt(
        Complex(1, 2),
        context=ArithmeticContext(format=FloatFormat(8)),
    )
    print(result)
