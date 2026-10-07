"""A Float sum rounded once."""

from apn_mojo import (
    Batch,
    Float,
    Integer,
    FloatFormat,
    ArithmeticContext,
    sum,
)


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(53))
    var large = Float(Integer(1) << 60, context=context)
    var values = Batch[Float]([large, Float(1, context=context), -large])
    print(sum(values))
    print(sum(values[::-1]))

    var measurements = Batch[Float]([Float("1.25"), Float("0.125")])
    var compact = ArithmeticContext(format=FloatFormat(2))
    print(sum(measurements, context=compact))
