"""Float batch arithmetic."""

from apn_mojo import Batch, Float, FloatFormat, ArithmeticContext, Rational
from apn_mojo import float, vmap


def main() raises:
    var values = Batch[Float]([Float("1.5"), Float("-2.25"), Float("0.5")])
    print(values + Float("0.5"))
    print(values * Rational(2, 3))
    print(values[values > Float(0)])
    var context = ArithmeticContext(format=FloatFormat(3))
    print(vmap[float.add]()(values[::-1], Float("0.25"), context=context))
