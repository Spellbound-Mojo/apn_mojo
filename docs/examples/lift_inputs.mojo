"""Exact numeric inputs and Ball folds use the scalar function's rules."""

from apn_mojo import Batch, Integer, Rational, FloatFormat, ArithmeticContext, BallContext, lift
from apn_mojo import rational, float, ball


def main() raises:
    var integers = Batch[Integer]([1, 2, 3])
    print(lift[rational.add]()(integers, Rational(1, 3)))
    print(lift[rational.add]().outer(10, integers))

    var thirds = Batch[Rational]([Rational(1, 3), Rational(2, 3)])
    var format = ArithmeticContext(format=FloatFormat(24))
    # The operands stay exact; only the returned total rounds.
    print(lift[float.add]().reduce(thirds, axis=None, context=format))

    var add = lift[ball.add]()
    var prefixes = add.accumulate(thirds, context=BallContext(precision=24))
    print(ball.contains(prefixes[0], Rational(1, 3)))
    print(prefixes[1].is_exact(), ball.contains(prefixes[1], 1))
