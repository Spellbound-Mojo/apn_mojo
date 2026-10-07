"""Complex text and JSON."""

from apn_mojo import (
    Complex,
    ComplexContext,
    ArithmeticContext,
    FloatFormat,
    ConversionLimits,
)


def main() raises:
    var z = Complex("3-4j")
    print(z)
    print(z.to_string(2))
    var settings = ComplexContext(
        real=ArithmeticContext(format=FloatFormat(8)),
        imag=ArithmeticContext(format=FloatFormat(11)),
    )
    var decimal = Complex.parse("0.1+0.3i", context=settings)
    var restored = Complex.from_json(decimal.to_json())
    print(restored == decimal)
    print(
        restored.real_format().precision(), restored.imag_format().precision()
    )
    print(Complex.from_json(Complex("-0-0j").to_json()))
    print(
        Complex("12+34j", limits=ConversionLimits(max_digits=4, max_values=1))
    )
