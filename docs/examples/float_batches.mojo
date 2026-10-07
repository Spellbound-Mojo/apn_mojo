"""Float batches and their formats."""

from apn_mojo import Batch, Float, FloatFormat, ArithmeticContext, Mask


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(64))
    var values = Batch[Float](
        [
            Float("1.5", context=context),
            Float("-0"),
            Float(7),
        ]
    )
    print(values)
    var saved = values[::-1]
    values[1:] = values[:-1]
    print(values)
    print(saved)
    print(values[1].precision())
    print(saved[Mask([True, False, True])])
