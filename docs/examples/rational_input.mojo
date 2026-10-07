"""Exact decimal input into Rationals."""

from apn_mojo import Rational, ConversionLimits


def main() raises:
    var rate = Rational("0.125")
    var amount = Rational("19.20")
    print("Exact charge:", amount * rate)
    print("Scientific notation:", Rational("1.25e-3"))
    print("Reduced fraction:", Rational("6/-8"))

    var limits = ConversionLimits(
        max_input_bytes=256, max_output_bytes=256,
        max_digits=64, max_values=1, max_allocated_bytes=16384,
    )
    var value = Rational("-7/3", limits=limits)
    var record = value.to_json(limits=limits)
    print("JSON:", record)
    print("Restored:", Rational.from_json(record, limits=limits))
    print("Exact text:", value.to_string(limits=limits))
