"""Complex batch arithmetic."""

from apn_mojo import (
    Batch,
    Complex,
    Integer,
    ArithmeticContext,
    FloatFormat,
    complex,
    vmap,
)


def main() raises:
    var samples = Batch[Complex]([Complex(1, 2), Complex(3, 4)])
    var weights = Batch[Integer].from_native([2, 3])
    var weighted = samples * weights
    print(weighted[0] == Complex(2, 4), weighted[1] == Complex(9, 12))

    var saved = samples[::-1]
    var shifted = 2 - saved
    print(shifted[0] == Complex(-1, -4))
    print((samples != Complex(0)).all(), (samples == samples[:]).all())

    var precise = vmap[complex.multiply]()(
        samples, saved, context=ArithmeticContext(format=FloatFormat(256))
    )
    print(precise[0] == Complex(-5, 10), precise[0].real_format().precision())
    print(samples[0] == Complex(1, 2), saved[0] == Complex(3, 4))
