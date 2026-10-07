"""Complex values with independent component formats."""

from apn_mojo import (
    Complex,
    ComplexContext,
    FloatFormat,
    ArithmeticContext,
    Rational,
    RoundingMode,
)


def main() raises:
    var z = Complex(3, -4)
    print(z)
    print(z.conjugate())
    var part = z.imag()
    part += 1
    print(z.imag(), part)
    var settings = ComplexContext(
        real=ArithmeticContext(format=FloatFormat(3)),
        imag=ArithmeticContext(
            format=FloatFormat(5), rounding=RoundingMode.toward_positive
        ),
    )
    var fraction = Complex(
        Rational(1, 3), Rational(-1, 3), context=settings
    )
    print(fraction)
    var saved = fraction
    fraction = Complex(fraction, context=ArithmeticContext(format=FloatFormat(256)))
    print(fraction == saved, fraction.real_format().precision())
