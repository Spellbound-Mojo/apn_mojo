"""Transactional compound updates of exact quantities."""

from apn_mojo import Batch, Integer, Rational


def main() raises:
    var quantities = Batch[Rational]([Rational(1, 2), Rational(3, 4), 2])
    var original = quantities[:]
    quantities *= Rational(3, 2)
    quantities += Rational(1, 4)
    print(quantities)
    print(original)

    var factors = Batch[Integer]([2, 0, 4])
    try:
        quantities /= factors
    except error:
        print(error)
    print(quantities)

    factors[1] = 2
    quantities /= factors
    print(quantities)
    quantities **= -1
    print(quantities)
