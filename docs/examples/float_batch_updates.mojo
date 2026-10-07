"""Float compound updates."""

from apn_mojo import Batch, Float, Integer, ArithmeticContext, FloatFormat


def main() raises:
    var values = Batch[Float](
        [
            Float(1, context=ArithmeticContext(format=FloatFormat(3))),
            Float(1, context=ArithmeticContext(format=FloatFormat(8))),
        ]
    )
    var saved = values[:]
    var increment = Float(
        "0.125", context=ArithmeticContext(format=FloatFormat(16))
    )
    # Each lane rounds to its own destination precision, not the RHS precision.
    values += increment
    print(values)
    print(values[0].precision(), values[1].precision())
    print(saved)
    values /= 2
    values **= Batch[Integer]([-1, -2])
    print(values)
