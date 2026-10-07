"""Decimal and hexadecimal text."""

from apn_mojo import (
    Float,
    FloatFormat,
    ArithmeticContext,
    RoundingMode,
)


def main() raises:
    var price = Float("19.95")
    print(price.to_string(10, digits=4))
    print(Float("0x1.8p2").to_string(10, digits=3))
    var value = Float(
        "1.125", context=ArithmeticContext(format=FloatFormat(3))
    )
    print(value)
    print(
        Float("1.25").to_string(
            10, digits=2, rounding=RoundingMode.toward_positive
        )
    )
    var restored = Float.parse(
        price.to_string(), context=ArithmeticContext(format=price.format())
    )
    print(restored == price)
