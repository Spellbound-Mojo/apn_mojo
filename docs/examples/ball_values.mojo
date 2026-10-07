"""Carry measurement uncertainty through a calculation."""

from apn_mojo import ArithmeticContext, Ball, BallContext, FloatFormat, Rational, ball


def main() raises:
    var length = Ball(3, Rational(1, 8))
    var area = ball.square(length, context=BallContext(64))
    print("length bounds:", length.lower(), length.upper())
    print("area midpoint and radius:", area.midpoint(), area.radius())
    print("area contains both endpoints:",
          ball.contains(area, Rational(529, 64)),
          ball.contains(area, Rational(625, 64)))
    print("area compared with 8:", ball.compare(area, Ball(8)))
    print("overlapping measurements:", ball.compare(length, Ball(3, Rational(1, 4))))
    print("x - x is exact:", (length - length).is_exact())
    print("division across zero:", (length / Ball(0, 1)).is_indeterminate())

    var root = ball.sqrt(2, context=BallContext(128))
    var rounded = ball.to_float_if_certain(
        root, context=ArithmeticContext(format=FloatFormat.binary64())
    )
    if rounded:
        print("certified sqrt(2):", rounded.value())
    else:
        raise Error("The enclosure did not determine one rounded value.")
