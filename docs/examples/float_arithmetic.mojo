"""Correctly rounded arithmetic with an explicit context."""

from apn_mojo import (
    Float,
    Integer,
    Rational,
    FloatFormat,
    ArithmeticContext,
    add,
    divide,
)


def main() raises:
    var value = Float(
        Rational(5, 4), context=ArithmeticContext(format=FloatFormat(3))
    )
    var increment = Rational(1, 64)
    print("ordinary:", value + increment)
    print("native result precision:", (Integer(2) + Float64(0.5)).precision())
    var target = ArithmeticContext(format=FloatFormat(2))
    var result = add(value, increment, context=target)
    print("direct rounding:", result)
    var saved = value
    value += Float(1)
    print("updated:", value, "precision:", value.precision())
    print("saved:", saved)
    var quotient = divide(Float(1), Float.zero(negative=True))
    print("zero divisor:", quotient)
