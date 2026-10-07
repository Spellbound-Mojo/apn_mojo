"""Exact fractions: construction, arithmetic and powers."""

from apn_mojo import Integer, Rational, pow_rational


def main() raises:
    var flour = Rational(3, 4)
    var servings = Rational(10, 6)
    var scaled = flour * servings
    print("Scaled cups:", scaled)
    print("Components:", scaled.numerator(), scaled.denominator())
    print("Floor, ceil, trunc:", scaled.floor(), scaled.ceil(), scaled.trunc())

    var share = Rational(2, 3)
    var saved = share
    share += Rational(1, 6)
    print("Saved and updated:", saved, share)
    print("Reciprocal:", Rational(-2, 3) ** -1)
    print("Exact native value:", Int(Rational(12, 3)))
    print("Integer division:", Integer(7) / 3)
    print("Signed power:", pow_rational(2, -3))
    print("Mixed comparison:", Integer(2) < Rational(7, 3))

    try:
        share /= 0
    except error:
        print(error)
    print("After failed division:", share)
