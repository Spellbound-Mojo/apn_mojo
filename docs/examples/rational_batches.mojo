"""Batches of fractions and masked selections."""

from apn_mojo import Batch, Rational, Mask


def main() raises:
    print("mixed:", Batch[Rational]([1, Rational(1, 2)]))
    var shares = Batch[Rational](
        [Rational(1, 2), Rational(2, 3), Rational(3, 4)]
    )
    var saved = shares[:]
    shares[1:] = shares[:-1]
    print("updated:", shares)
    print("saved:", saved)
    shares[Mask([True, False, True])] = Rational(5, 6)
    print("selected fill:", shares)
    print("reverse:", shares[::-1])
    shares[0] = shares[0] + Rational(1, 6)
    print("scalar update:", shares)
    var total = Rational()
    for share in shares:
        total += share
    print("total:", total)
