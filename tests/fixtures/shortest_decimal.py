"""Write tests/fixtures/shortest_decimal.txt from Python's repr, the oracle.

Each line is a binary64 value as repr prints it, then the digits and decimal
exponent of its shortest round-trip decimal. Python's repr is the shortest
decimal that rounds back to the double, closest on a tie of lengths. The values
are the classic hard cases (powers of two, the neighbours of powers of ten, the
extremes) and seeded random doubles, all normal and above 2**-1022, the
smallest value of apn_mojo's binary64 format, whose interval differs from
IEEE's because the format has no subnormals.

Run with any Python 3: python3 tests/fixtures/shortest_decimal.py
"""

import math
import random
import struct
from pathlib import Path

OUT = Path(__file__).with_suffix(".txt")


def shortest(value):
    mantissa, _, exponent = repr(abs(value)).partition("e")
    exponent = int(exponent) if exponent else 0
    whole, _, fraction = mantissa.partition(".")
    if fraction == "0":
        fraction = ""
    digits = (whole + fraction).lstrip("0")
    exponent -= len(fraction)
    stripped = digits.rstrip("0")
    return stripped, exponent + len(digits) - len(stripped)


def main():
    smallest = 2.0 ** -1022
    values = [math.pi, 0.1, 0.3, 1.0, 2.0, 1e23, 5e-324 * 2**60, 9007199254740993.0,
              1.7976931348623157e308, -1.7976931348623157e308, 123456.789, -0.001]
    values += [2.0 ** k for k in range(-1021, 1024)]
    for k in range(-307, 309):
        power = float(f"1e{k}")
        values += [power, math.nextafter(power, 0.0), math.nextafter(power, math.inf)]
    rng = random.Random(20261004)
    while len(values) < 6000:
        value = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if math.isfinite(value) and abs(value) > smallest:
            values.append(value)
    lines = []
    for value in values:
        if math.isfinite(value) and abs(value) > smallest:
            digits, exponent = shortest(value)
            lines.append(f"{value!r} {digits} {exponent}\n")
    OUT.write_text("".join(lines))
    print(len(lines), "values")


if __name__ == "__main__":
    main()
