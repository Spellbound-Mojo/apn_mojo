"""Complex compound updates."""

from apn_mojo import Batch, Complex, Integer


def main() raises:
    var values = Batch[Complex]([Complex(3, 4), Complex(1, -2)])
    var saved = values[:]
    values += Batch[Integer].from_native([1, 2])
    values -= 1
    values *= Complex(0, 1)
    values /= Complex(0, 1)
    values **= 2
    print(values[0] == Complex(-7, 24))
    print(values[1] == Complex(0, -8))
    print(saved[0] == Complex(3, 4))
    print(saved[1] == Complex(1, -2))
