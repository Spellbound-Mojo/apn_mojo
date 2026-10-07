"""Exact integers and fractions, Floats of any precision and Complex numbers."""

from apn_mojo import ArithmeticContext, Complex, Float, FloatFormat, Integer, Rational, sqrt


def main() raises:
    # Integers and Rationals are exact: they grow instead of overflowing or rounding.
    print("2 ** 100 =", Integer(2) ** 100)
    print("1/3 + 1/6 =", Rational(1, 3) + Rational(1, 6))
    # A Float has the precision you choose and rounds each result once.
    var precise = ArithmeticContext(format=FloatFormat(256))
    var root = sqrt(Float(2, context=precise))
    print("sqrt(2) =", root.to_string(10, digits=60))
    # Decimal text is exact; a native Float64 brings its binary rounding error along.
    print("0.1 from text =", Float("0.1", context=precise).to_string(10, digits=60))
    print("0.1 as Float64 =", Float(Float64(0.1), context=precise).to_string(10, digits=60))
    # Complex numbers round each component once.
    var z = Complex("1+2j") * Complex("3-4j")
    print("(1+2j)(3-4j) =", String(z.real().to_integer_exact()) + "+" + String(z.imag().to_integer_exact()) + "j")
    var w = sqrt(Complex(-4))
    print("sqrt(-4) =", String(w.real().to_integer_exact()) + "+" + String(w.imag().to_integer_exact()) + "j")
