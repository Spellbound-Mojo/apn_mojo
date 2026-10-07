"""Complex batches and masks."""

from apn_mojo import Batch, Complex, Mask


def main() raises:
    var values = Batch[Complex]([Complex(1, 2), Complex(3, 4), Complex(5, 6)])
    var saved = values[::-1]
    values[:] = saved
    print(values[0] == Complex(5, 6))

    values[Mask([True, False, True])] = Complex(0, -1)
    print(values[0] == Complex(0, -1), values[1] == Complex(3, 4))
    print(saved[0] == Complex(5, 6), saved[2] == Complex(1, 2))

    var real_inputs = Batch[Complex].from_native([4, -1, 0])
    var count = 0
    for value in real_inputs:
        if value.imag().is_zero():
            count += 1
    print(count)
    print(len(Batch[Complex](saved[::2])))
