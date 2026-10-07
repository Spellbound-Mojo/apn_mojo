"""Counters and weighted totals with sum, dot and axpy."""

from apn_mojo import Batch, Integer, sum, dot, axpy


def main() raises:
    var counters = Batch[Integer]([
        Integer(2) ** 80, Integer(-3), Integer(5),
    ])
    var original = counters[:]
    counters[counters < 0] = 0
    var weights = Batch[Integer].from_native([2, 3, 4])
    var increments = Batch[Integer].from_native([1, 2, 3])
    print("clean total:", sum(counters))
    print("weighted total:", dot(counters, weights))
    var updated = axpy(2, counters, increments)
    print("scaled update:", updated[0], updated[1], updated[2])
    print("original correction:", original[1])
    print("exact checkpoint:", updated.to_json())
