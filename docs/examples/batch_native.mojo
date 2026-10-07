"""Export row-major native values, keeping shape separately."""

from std.collections import Array
from apn_mojo import Ball, Batch, Complex, Rational, batch


def main() raises:
    var values = Batch[Rational]([Rational(1, 3), Rational(2, 3), Rational(3), Rational(4)]).reshape([2, 2])
    var shape = values.shape()
    var native = values.to_native[DType.float64]()
    print("shape:", shape)
    print("native values:", native)
    # Array's length is a compile-time parameter; check a runtime batch's size.
    if len(native) != 4:
        raise Error("This export needs exactly four elements.")
    var fixed = Array[Float64, 4](fill=0)
    for i in range(4):
        fixed[i] = native[i]
    print("fixed array, first and last:", fixed[0], fixed[3])
    print("integer floors:", batch.floor(values).to_native[DType.int64]())

    var complex_values = Batch[Complex]([Complex(1, 2), Complex(3, -4)])
    print("native real parts:", batch.real(complex_values).to_native[DType.float64]())
    print("native imaginary parts:", batch.imag(complex_values).to_native[DType.float64]())
    var uncertain = Batch[Ball]([Ball(3, Rational(1, 8))])
    print("midpoints only, uncertainty discarded:", uncertain.to_native[DType.float64]())
