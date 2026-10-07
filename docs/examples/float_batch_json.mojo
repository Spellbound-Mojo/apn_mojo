"""Float batch JSON."""

from apn_mojo import (
    Batch,
    Float,
    FloatFormat,
    ArithmeticContext,
    ConversionLimits,
)


def main() raises:
    var small = ArithmeticContext(format=FloatFormat(3))
    var wide = ArithmeticContext(format=FloatFormat(80))
    var values = Batch[Float](
        [
            Float("1.5", context=small),
            Float.zero(negative=True, context=wide),
            Float.infinity(context=small),
        ]
    )
    var saved = values[::-1]
    var limits = ConversionLimits(
        max_values=3,
        max_digits=100,
        max_input_bytes=4096,
        max_output_bytes=4096,
        max_allocated_bytes=65536,
    )
    var text = saved.to_json(limits=limits)
    values[0] = Float(99)
    var restored = Batch[Float].from_json(text, limits=limits)
    print(restored)
    print(
        restored[0].precision(),
        restored[1].precision(),
        restored[2].precision(),
    )
    print(restored.to_json() == text)
    print(saved.to_json() == text)
