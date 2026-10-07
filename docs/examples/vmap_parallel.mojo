"""Parallel mapping of a long batch."""

from apn_mojo import Batch, Float, FloatFormat, ArithmeticContext, vmap
from apn_mojo import float


def hypotenuse(x: Float, y: Float) raises -> Float:
    return float.sqrt(x * x + y * y)


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(64))
    var xs = Batch[Float]([Float(i, context=context) for i in range(1, 2001)])
    var ys = Batch[Float]([Float(2 * i, context=context) for i in range(1, 2001)])
    # vmap[f]() returns the mapped function; long batches use worker threads.
    var lengths_of = vmap[hypotenuse]()
    var lengths = lengths_of(xs, ys)
    print(lengths[0], lengths[1999])
    # Library functions work the same way, keyword-only context included.
    var sums = vmap[float.add]()(xs, ys, context=context)
    print(sums[0], sums[1999])
    # Every element equals the scalar call.
    print(hypotenuse(xs[1999], ys[1999]) == lengths[1999])
    print(lengths_of(ys, xs)[0] == lengths[0])
