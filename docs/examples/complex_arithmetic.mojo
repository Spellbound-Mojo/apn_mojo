"""Complex arithmetic rounded per component."""

from apn_mojo import (
    Complex,
    Rational,
    ArithmeticContext,
    FloatFormat,
    divide,
)


def main() raises:
    var z = Complex(3, 4)
    print(z * Complex(1, -2))
    print(25 / z)
    print(z + Rational(1, 2))
    var saved = z
    z *= z
    print(z)
    print(saved)
    var narrow = ArithmeticContext(format=FloatFormat(8))
    var result = divide(saved, 3, context=narrow)
    print(result)
