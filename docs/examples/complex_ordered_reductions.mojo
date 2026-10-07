"""Ordered Complex reductions."""

from apn_mojo import (
    Batch,
    Complex,
    ArithmeticContext,
    FloatFormat,
    sum,
    sum_sequential,
    sum_tree,
    dot_sequential,
)


def main() raises:
    var values = Batch[Complex](
        [
            Complex(8, 8),
            Complex(1, 1),
            Complex(1, 1),
            Complex(-8, -8),
        ]
    )
    var context = ArithmeticContext(format=FloatFormat(3))
    print(sum(values, context=context))
    print(sum_sequential(values, context=context))
    print(sum_tree(values, context=context))
    var signal = Batch[Complex]([Complex(1, 2), Complex(3, 4)])
    print(dot_sequential(signal, signal))
