"""Rounding to integers and signed powers on batches."""

from apn_mojo import Batch, Integer, Rational, pow_rational, rational, vmap


def main() raises:
    var portions = Batch[Rational]([Rational(7, 3), Rational(-7, 3), 2])
    print("Magnitudes:", vmap[rational.abs]()(portions))
    print("Signs:", portions.sign())
    print("Floor:", portions.floor())
    print("Ceiling:", portions.ceil())
    print("Truncation:", portions.trunc())

    var growth = Batch[Rational]([Rational(3, 2), Rational(5, 4)])
    var periods = Batch[Integer]([2, -2])
    print("Growth factors:", growth ** periods)
    print("Reverse growth:", growth[::-1] ** -1)
    var whole = Batch[Integer]([2, 4, 8])
    print("Integer reciprocals:", vmap[pow_rational]()(whole, -1))
