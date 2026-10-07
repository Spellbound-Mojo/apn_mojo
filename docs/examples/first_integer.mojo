"""The smallest complete program: one exact Integer calculation."""

from apn_mojo import Integer


def main() raises:
    var value = Integer(2) ** 100
    var saved = value
    value += 1
    print("2 ** 100:", saved)
    print("plus one:", value)
    print("unchanged copy:", saved == Integer(2) ** 100)
