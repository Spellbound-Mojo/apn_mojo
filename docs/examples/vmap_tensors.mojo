"""Mapping over rows and columns, with tensor results."""

from apn_mojo import Batch, Integer, sum_sequential, vmap


def total(row: Batch[Integer]) raises -> Integer:
    return sum_sequential(row)


def reverse(row: Batch[Integer]) raises -> Batch[Integer]:
    return row[::-1]


def shift(row: Batch[Integer], offset: Integer) raises -> Batch[Integer]:
    return row + offset


def main() raises:
    var matrix = Batch[Integer].from_iterable([1, 2, 3, 4, 5, 6], shape=[2, 3])
    print(vmap[total]()(matrix))
    print(vmap[total](in_axes=1)(matrix))
    print(vmap[reverse](out_axes=-1)(matrix))
    print(vmap[shift](in_axes=(0, None))(matrix, Integer(10)))
    var empty = Batch[Integer].from_iterable(List[Int](), shape=[0, 3])
    print(vmap[reverse](out_shape=[3])(empty).shape())
