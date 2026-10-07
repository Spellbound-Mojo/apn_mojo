"""Arithmetic with broadcasting across ranks."""

from apn_mojo import Batch, Integer


def main() raises:
    var matrix = Batch[Integer]([1, 2, 3, 4, 5, 6], shape=[2, 3])
    var row = Batch[Integer]([10, 20, 30])
    print(matrix + row)
    print(matrix / 2)
    print(matrix > 3)
    print((matrix + row)[1, 2])
    print(matrix.slice(1, step=-1) * 2)
