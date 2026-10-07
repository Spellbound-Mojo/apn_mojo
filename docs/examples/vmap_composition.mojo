"""Composing mapped functions."""

from apn_mojo import Batch, Integer, vmap


def combine(x: Integer, y: Integer) raises -> Tuple[Integer, Integer]:
    return (x + y, x * y)


def tuple_sum(pair: Tuple[Integer, Integer]) raises -> Integer:
    return pair[0] + pair[1]


def main() raises:
    var matrix = Batch[Integer].from_iterable(range(1, 7), shape=[2, 3])
    var shared = Batch[Integer]([10, 20, 30])
    var result = vmap[combine]().vmap(in_axes=(0, None), out_axes=(0, -1))(matrix, shared)
    print(result[0])
    print(result[1])
    print(vmap[tuple_sum]().vmap(in_axes=(0, None))((matrix, shared)))
    var empty = matrix.slice(0, stop=0)
    var empty_result = vmap[combine]().vmap(out_shape=([3], [3]))(empty, empty)
    print(empty_result[0].shape())
    print(empty_result[1].shape())
