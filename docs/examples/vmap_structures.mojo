"""Mapped functions that return tuples."""

from apn_mojo import Batch, Integer, vmap


def square(x: Integer) raises -> Integer:
    return x * x


def describe(x: Integer) raises -> Tuple[Integer, Integer, Bool]:
    return (x * x, x * x * x, x > 2)


def affine_value(value: Integer, scale: Integer, offset: Integer) raises -> Integer:
    return value * scale + offset


def affine(parts: Tuple[Integer, Integer, Integer]) raises -> Integer:
    return affine_value(parts[0], parts[1], parts[2])


def main() raises:
    var values = Batch[Integer]([1, 2, 3])
    var result = vmap[describe]()(values)
    print(result[0])
    print(result[1])
    print(result[2].count())
    print(vmap[affine](in_axes=(0, None, None))((values, Integer(2), Integer(10))))
    print(vmap[affine]()((values, values, values)))
    var matrix = Batch[Integer].from_iterable(range(1, 7), shape=[2, 3])
    print(vmap[square]().vmap()(matrix))
    print(vmap[affine](in_axes=(0, None, None)).vmap(in_axes=(0, None, None))((matrix, Integer(2), Integer(10))))
    print(vmap[describe]().vmap()(matrix)[2])
