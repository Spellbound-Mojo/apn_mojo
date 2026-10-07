"""Complex batch JSON."""

from apn_mojo import (
    Batch,
    Complex,
    Float,
    FloatFormat,
    ArithmeticContext,
    ComplexContext,
    ConversionLimits,
)


def main() raises:
    var settings = ComplexContext(
        real=ArithmeticContext(format=FloatFormat(65)),
        imag=ArithmeticContext(format=FloatFormat(257)),
    )
    var values = Batch[Complex](
        [
            Complex("1.5-2j", context=settings),
            Complex(Float.zero(negative=True), Float.infinity()),
        ]
    )
    var saved = values[::-1]
    var limits = ConversionLimits(
        max_values=2,
        max_digits=1000,
        max_input_bytes=4096,
        max_output_bytes=4096,
        max_allocated_bytes=65536,
    )
    var text = saved.to_json(limits=limits)
    values[0] = Complex(99)
    var restored = Batch[Complex].from_json(text, limits=limits)
    print(restored)
    print(
        restored[1].real_format().precision(),
        restored[1].imag_format().precision(),
    )
    print(restored.to_json() == text)
    print(saved.to_json() == text)
