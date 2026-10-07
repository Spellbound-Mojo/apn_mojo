"""Complex dot and vdot."""

from apn_mojo import Batch, Complex, Integer, Rational, dot, vdot


def main() raises:
    var values = Batch[Complex]([Complex(1, 2), Complex(3, -4)])
    print(dot(values, values))
    print(vdot(values, values))

    var weights = Batch[Rational]([Rational(1, 3), Rational(2, 3)])
    print(dot(values, weights) == dot(weights, values))
    print(vdot(weights, values) == dot(weights, values))

    var large = Integer(1) << 1024
    var terms = Batch[Complex](
        [Complex(large, large), Complex(1, -3), Complex(-large, -large)]
    )
    var ones = Batch[Integer]([1, 1, 1])
    var saved = terms[::-1]
    terms[0] = Complex(0)
    print(dot(saved, ones))
    print(vdot(saved, ones))
