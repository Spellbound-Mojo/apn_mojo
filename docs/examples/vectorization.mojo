"""Map a scalar calculation, then combine or pair its results."""

from apn_mojo import Batch, Integer, batch, integer, lift, vmap


def squared_distance(value: Integer, center: Integer) raises -> Integer:
    var difference = value - center
    return difference * difference


def row_total(row: Batch[Integer]) raises -> Integer:
    return batch.sum(row)


def main() raises:
    var positions = Batch[Integer]([1, 2, 4])
    var distances = vmap[squared_distance](in_axes=(0, None))(positions, Integer(2))
    print("squared distances from 2:", distances)

    var add = lift[integer.add](identity=Integer(0), associative=True)
    print("total:", add.reduce(distances, axis=None))
    print("running totals:", add.accumulate(distances))
    var centers = Batch[Integer]([0, 2])
    print("all pairs:", lift[squared_distance]().outer(positions, centers))

    var grid = Batch[Integer]([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    print("row totals:", vmap[row_total](in_axes=0)(grid))
    print("column totals:", vmap[row_total](in_axes=1)(grid))
