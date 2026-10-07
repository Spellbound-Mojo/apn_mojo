"""Saving and loading exact batches as JSON."""

from apn_mojo import Batch, Rational, ConversionLimits


def main() raises:
    var rates = Batch[Rational]([Rational(1, 8), Rational(-7, 3), Rational(0)])
    var saved = rates[::-2]
    var policy = ConversionLimits(
        max_values=2,
        max_digits=4,
        max_input_bytes=512,
        max_output_bytes=512,
        max_allocated_bytes=16384,
    )
    var wire = saved.to_json(limits=policy)
    print(wire)
    var restored = Batch[Rational].from_json(wire, limits=policy)
    rates += 1
    print("Restored:", restored)
    print("Updated:", rates)
    try:
        restored = Batch[Rational].from_json(
            '{"version":1,"family":"rational-batch","values":[{"numerator":"2","denominator":"4"}]}',
            limits=policy,
        )
        print("Imported:", restored)
    except:
        print("Noncanonical input rejected; kept:", restored)
    restored = Batch[Rational].from_json(wire, limits=policy)
    print("Retry:", restored)
