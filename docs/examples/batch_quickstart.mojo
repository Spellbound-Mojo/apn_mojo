"""A first batch calculation with an explicit precision and a reduction."""

from apn_mojo import ArithmeticContext, Batch, FloatFormat, Integer, batch


def main() raises:
    var values = Batch[Integer]([1, 4, 9, 16])
    var context = ArithmeticContext(format=FloatFormat(256))
    var roots = batch.sqrt(values, context=context)
    print("roots:", roots)
    print("sum:", batch.sum(roots, context=context))
