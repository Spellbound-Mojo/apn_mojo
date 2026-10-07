"""Reading an error's remedy and keeping the destination unchanged."""

from apn_mojo import Batch, Integer


def main() raises:
    var values = Batch[Integer].from_native([20, 30, 40])
    var before = values[:]
    var divisors = Batch[Integer].from_native([2, 0, 5])
    try:
        values //= divisors
    except error:
        print(error)
    print("whole destination unchanged:", (values == before).all())
    divisors[divisors == 0] = 1
    values //= divisors
    print("after retry:", values[0], values[1], values[2])
    try:
        _ = Integer("12x")
    except error:
        print(error)
