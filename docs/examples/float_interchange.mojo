"""Float JSON that keeps every bit."""

from apn_mojo import Float, Rational, FloatFormat, ArithmeticContext, ConversionLimits


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(3, emin=-10, emax=10))
    var value = Float(Rational(-1, 3), context=context)
    print("hex:", value.to_string(16))
    print("binary:", value.to_string(2))
    var record = value.to_json()
    print(record)
    var restored = Float.from_json(record)
    print("same value and format:", restored == value, restored.format() == value.format())
    print("bounded output:", value.to_string(16, limits=ConversionLimits(max_output_bytes=7)))
    var zero = Float.from_json(Float.zero(negative=True, context=context).to_json())
    print("negative zero:", zero, zero.signbit())
