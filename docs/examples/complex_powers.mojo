"""Integral powers."""

from apn_mojo import Complex, Integer, pow_int


def main() raises:
    var z = Complex(2, 3)
    print(z**2)
    print(z**3)
    print(pow_int(Complex(1, 1), -2))
    var turns = (Integer(1) << 256) + 1
    print(Complex(0, 1) ** turns)
    var saved = z
    z **= 2
    print(z)
    print(saved)
