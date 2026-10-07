"""Exact complex fractions, square roots and conversion."""

from apn_mojo import ArithmeticContext, ComplexContext, ExactComplex, FloatFormat, Rational
from apn_mojo.exact_complex import sqrt_exact


def main() raises:
    var z = ExactComplex(Rational(1, 3), Rational(2, 3))
    print("square:", z * z)
    print("quotient:", z / z)
    print("JSON round trip:", ExactComplex.from_json(z.to_json()) == z)
    var root = sqrt_exact(ExactComplex(3, 4))
    if root:
        print("exact root:", root.value())
    print("sqrt(i) has rational parts:", Bool(sqrt_exact(ExactComplex(0, 1))))
    var context = ComplexContext(ArithmeticContext(format=FloatFormat.binary64()))
    print("rounded components:", z.to_complex(context=context))
