"""Formats, rounding modes and traps."""

from apn_mojo import ArithmeticContext, FloatFormat, RoundingMode


def main() raises:
    var format = FloatFormat(173, emin=-1000, emax=1000)
    var context = ArithmeticContext(
        format=format,
        rounding=RoundingMode.toward_positive,
        trap_overflow=True,
    )
    print(format)
    print("precision:", context.format().precision())
    print("rounding:", context.rounding())
    print("trap overflow:", context.traps_overflow())
    print("default precision:", FloatFormat().precision())
