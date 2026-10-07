"""Decimal text, JSON and conversion limits for Integers."""

from apn_mojo import Batch, Integer, ConversionLimits


def main() raises:
    var limits = ConversionLimits(
        max_input_bytes=4096,
        max_output_bytes=4096,
        max_digits=1000,
        max_values=100,
        max_allocated_bytes=65536,
    )
    var value = Integer(
        " -0xFF_FF ", base=0, allow_whitespace=True,
        allow_underscores=True, limits=limits,
    )
    print("decimal:", value)
    print("hex:", value.to_string(16, prefix=True, uppercase=True, limits=limits))
    var document = value.to_json(limits=limits)
    print("JSON:", document)
    print("round trip:", Integer.from_json(document, limits=limits) == value)
    var batch = Batch[Integer]([value, Integer(2) ** 80])
    var restored = Batch[Integer].from_json(batch[::-1].to_json(), limits=limits)
    print("restored:", restored[0], restored[1])
