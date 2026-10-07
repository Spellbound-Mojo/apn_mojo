"""Mapping a scalar function over batches."""

from apn_mojo import Batch, Integer, vmap


def polynomial(x: Integer) raises -> Integer:
    return x * x + 2 * x + 1


def difference(x: Integer, y: Integer) raises -> Integer:
    return x - y


def positive(x: Integer) raises -> Bool:
    return x > 0


def shifted(values: Batch[Integer]) raises -> Batch[Integer]:
    # A function can map over its own batch parameters; scalars are shared.
    return vmap[difference]()(values, 3)


def main() raises:
    var values = Batch[Integer]([-2, -1, 0, 1, 2])
    var evaluate = vmap[polynomial]()
    print(evaluate(values))
    print(evaluate(values[::-2]))
    print(vmap[difference](in_axes=(0, None))(values, Integer(3)))
    print(shifted(values))
    print(values[vmap[positive]()(values)])
