"""Shapes and axis views."""

from apn_mojo import Batch, Integer


def main() raises:
    var matrix = Batch[Integer]([1, 2, 3, 4, 5, 6], shape=[2, 3])
    print(matrix.shape(), matrix.ndim(), len(matrix))
    var columns = matrix.transpose()
    matrix[0, 1] = 20
    print(matrix)
    print(columns)
    print(matrix.at(1, -1))
    print(matrix.slice(1, step=-1))
    for i in range(matrix.shape()[0]):
        print(matrix.at(0, i))
    print(matrix.reshape([6]))
    var scalar = Batch[Integer]([42], shape=[])
    print(scalar.item())
