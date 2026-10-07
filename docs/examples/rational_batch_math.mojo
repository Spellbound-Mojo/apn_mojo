"""Exact elementwise batch calculations."""

from apn_mojo import Batch, Integer, Rational


def main() raises:
    var counts = Batch[Integer]([1, 2, 3])
    var portions = counts / 4
    print("portions:", portions)
    var rates = Batch[Rational]([1, Rational(1, 2), Rational(1, 3)])
    print("weighted:", counts * rates)
    print("above half:", portions[portions > Rational(1, 2)])
    print("at least half:", Rational(1, 2) <= portions)
    var saved = portions[:]
    portions = portions + Rational(1, 4)
    print("adjusted:", portions)
    print("saved:", saved)
    portions[portions > Rational(1, 2)] = (
        portions[portions > Rational(1, 2)] / 2
    )
    print("selected halves:", portions)
