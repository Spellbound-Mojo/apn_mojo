"""NumPy-style functions of batches: apn_mojo.batch."""

from apn_mojo import ArithmeticContext, Batch, Float, FloatFormat, Integer, batch


def main() raises:
    var grid = batch.arange[Integer](6).reshape([2, 3])
    print("grid + row:", batch.add(grid, Batch[Integer]([10, 20, 30])))
    print("halves:", batch.divide(grid, 2))
    print("large kept:", batch.where(grid > 2, grid, 0))
    print("running sums:", batch.cumsum(grid, axis=1))
    print("stacked shape:", batch.stack([grid, grid]).shape())
    var c = ArithmeticContext(format=FloatFormat.binary64())
    var xs = batch.linspace[Float](0, 1, 5, context=c)
    print("xs:", xs)
    print("exp:", batch.exp(xs))
    print("atan2(1, xs):", batch.atan2(1, xs))
    print("exp(5), rounded once:", batch.exp(grid, context=c)[1, 2])
