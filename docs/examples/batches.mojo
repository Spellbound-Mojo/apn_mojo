"""Batch snapshots, selections and updates."""

from apn_mojo import Batch, Integer, sum


def main() raises:
    var values = Batch[Integer].from_native([4, -1, 0, -3, 8])
    var original = values[:]
    var negative = values < 0
    var selected = values[negative]
    values[negative] = 0
    print("negative count:", negative.count())
    print("selected values:", selected)
    print("totals:", sum(original), sum(values))
    values[1:] = values[:-1]
    print("overlap:", values)
    values[0] = 18446744073709551616
    print("wide assignment:", values)
    print("original snapshot:", original)
    var sequence = Batch[Integer].from_iterable(range(4))
    for item in sequence[::-1]:
        print("reverse:", item)
    print("all nonnegative:", (values >= 0).all())
