"""Rectangular uncertainty and a certified complex result."""

from apn_mojo import (
    ArithmeticContext, Ball, BallContext, ComplexBall, ComplexContext,
    FloatFormat, Rational, ball, complex_ball,
)


def main() raises:
    var measured = ComplexBall(Ball(3, Rational(1, 8)), Ball(4, Rational(1, 8)))
    var magnitude = complex_ball.abs(measured, context=BallContext(64))
    print("magnitude contains 5:", ball.contains(magnitude, 5))
    print("magnitude is exact:", magnitude.is_exact())

    var root = complex_ball.sqrt(ComplexBall(3, 4), context=BallContext(128))
    print("root contains 2 + i:", complex_ball.contains(root, ComplexBall(2, 1)))
    var rounded = complex_ball.to_complex_if_certain(
        root, context=ComplexContext(ArithmeticContext(format=FloatFormat.binary64()))
    )
    if rounded:
        print("certified root:", rounded.value())
    else:
        raise Error("The rectangle did not determine one rounded complex value.")

    var across_cut = ComplexBall(Ball(-2), Ball(0, Rational(1, 8)))
    var logarithm = complex_ball.log(across_cut, context=BallContext(64))
    print("log across the cut spans both signs:", ball.contains_zero(logarithm.imag()))
