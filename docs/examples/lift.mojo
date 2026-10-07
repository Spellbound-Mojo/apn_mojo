"""Lifting a binary function into reduce, accumulate and outer."""

from apn_mojo import Batch, Integer, Float, FloatFormat, ArithmeticContext, lift, sum, sum_sequential


def keep_first(x: Integer, y: Integer) raises -> Integer:
    # Associative but not commutative: keeps the earlier operand.
    return x


def add_into(mut total: Integer, value: Integer) raises:
    total += value


def main() raises:
    var grid = Batch[Integer]([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    # Lift a lambda; identity and associativity are stated once.
    var add = lift[lambda (x: Integer, y: Integer) raises -> Integer: x + y](
        identity=Integer(0), associative=True,
    )
    print(add.reduce(grid, axis=1))               # np.add.reduce: row sums
    print(add.reduce(grid))                       # axis=0 by default
    print(add.reduce(grid, axis=None))            # every element: a scalar
    print(add.reduce(grid, axis=[0, 1], keepdims=True))
    print(add.accumulate(grid, axis=1))           # running sums, np.cumsum
    print(add.outer(Batch[Integer]([10, 20]), Batch[Integer]([1, 2, 3])))
    print(add(grid, Batch[Integer]([100, 200, 300])))  # broadcasting call
    # Without associative=True, each result is a strict left fold.
    var subtract = lift[lambda (x: Integer, y: Integer) raises -> Integer: x - y]()
    print(subtract.reduce(Batch[Integer]([10, 1, 2, 3]), axis=None))
    print(subtract.reduce(grid, axis=0, initial=Integer(100)))
    # Without an identity, an empty reduction needs initial= ...
    var empty = Batch[Integer](List[Integer]()).reshape([0, 2])
    print(subtract.reduce(empty, axis=0, initial=Integer(7)))
    # ... and raises otherwise.
    try:
        _ = subtract.reduce(empty, axis=0)
    except error:
        print(error)
    # Associative functions keep operand order when split across threads.
    print(lift[keep_first](associative=True).reduce(grid, axis=None))
    # An in-place update reuses the accumulator's storage.
    print(lift[add_into](identity=Integer(0)).reduce(grid, axis=1))
    # Float results are computed exactly and rounded once, like sum ...
    var c = ArithmeticContext(format=FloatFormat(53))
    var xs = Batch[Float]([Float(1, context=c), Float("1e-16", context=c), Float("1e-16", context=c)])
    var fadd = lift[lambda (x: Float, y: Float) raises -> Float: x + y](associative=True)
    print(fadd.reduce(xs, axis=None), fadd.reduce(xs, axis=None) == sum(xs))
    # ... unless exact=False asks for every step to round, like sum_sequential.
    var stepwise = lift[lambda (x: Float, y: Float) raises -> Float: x + y](exact=False)
    print(stepwise.reduce(xs, axis=None), stepwise.reduce(xs, axis=None) == sum_sequential(xs))
    # Any Integer, Rational, Float or Complex function lifts the same way.
    var hypot = lift[lambda (x: Float, y: Float) raises -> Float: x * x + y * y]()
    print(hypot.outer(Batch[Float]([Float(3)]), Batch[Float]([Float(4), Float(0)])))
