"""Float values, formats and exact conversion."""

from apn_mojo import Float, Rational, FloatFormat, ArithmeticContext


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(3))
    var value = Float(Rational(1, 3), context=context)
    print("stored value:", value)
    var saved = value
    value = value.to_format(FloatFormat(173))
    print("precisions:", saved.precision(), value.precision())
    print("same stored value:", value == saved)
    print("below exact third:", value < Rational(1, 3))
    print("native half:", Float.from_native(Float64(0.5)))
    var zero = Float.zero(negative=True)
    print("signed zero:", zero, zero == Float())
    var nan = Float.nan()
    print("NaN:", nan, nan == nan)
