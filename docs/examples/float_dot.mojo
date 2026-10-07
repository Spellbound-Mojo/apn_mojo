"""A Float dot product rounded once."""

from apn_mojo import Batch, Float, Integer, Rational, dot


def main() raises:
    var measurements = Batch[Float]([Float(1), Float(2), Float(4)])
    var weights = Batch[Rational]([Rational(1, 3), Rational(1, 3), Rational(1, 3)])
    var average = dot(measurements, weights)
    print(average.to_string(10, digits=6))
    print(dot(weights[::-1], measurements[::-1]) == average)

    var large = Integer(1) << 120
    var coefficients = Batch[Integer]([large, 1, -large])
    var ones = Batch[Float]([Float(1), Float(1), Float(1)])
    print(dot(ones, coefficients))
