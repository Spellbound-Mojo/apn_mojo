"""Sequential and tree reductions."""

from apn_mojo import (
    Batch,
    Float,
    Integer,
    FloatFormat,
    ArithmeticContext,
    sum,
    sum_sequential,
    sum_tree,
    dot,
    dot_sequential,
)


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(3))
    var values = Batch[Float]([Float(8), Float(1), Float(1), Float(-8)])
    print("round_once:", sum(values, context=context))
    print("sequential:", sum_sequential(values, context=context))
    print("tree:", sum_tree(values, context=context))

    var a = Batch[Float]([Float("1.25"), Float("-1.5")])
    var b = Batch[Float]([Float("1.25"), Float(1)])
    print("dot round_once:", dot(a, b, context=context))
    print("dot sequential:", dot_sequential(a, b, context=context))

    var exact = Batch[Integer].from_native([8, 1, 1, -8])
    print("exact:", sum_sequential(exact), sum_tree(exact))
