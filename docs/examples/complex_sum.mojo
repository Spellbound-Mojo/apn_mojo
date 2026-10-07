"""A Complex sum rounded once per component."""

from apn_mojo import (
    Batch,
    Complex,
    Integer,
    FloatFormat,
    ArithmeticContext,
    ComplexContext,
    sum,
)


def main() raises:
    var context = ComplexContext(
        real=ArithmeticContext(format=FloatFormat(53)),
        imag=ArithmeticContext(format=FloatFormat(65)),
    )
    var large = Integer(1) << 100
    var values = Batch[Complex](
        [
            Complex(large, -large, context=context),
            Complex(1, 3, context=context),
            Complex(-large, large, context=context),
        ]
    )
    var saved = values[::-1]
    var total = sum(saved)
    print(total)
    print(total.real_format().precision(), total.imag_format().precision())

    var sequential = Complex(0, context=context)
    for value in values:
        sequential += value
    print(sequential)

    values[1] = Complex(99)
    print(sum(saved) == total)
    print(sum(saved[:0], context=context))
