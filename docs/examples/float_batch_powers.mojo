"""Float batch unary operations and powers."""

from apn_mojo import (
    Batch,
    Float,
    Integer,
    float,
    vmap,
    ArithmeticContext,
    FloatFormat,
)


def main() raises:
    var values = Batch[Float](
        [Float("1.5"), Float(-2), Float.zero(negative=True), Float.nan()]
    )
    print(-values)
    print(vmap[float.abs]()(values))
    print(values[~values.is_nan()].sign())
    var bases = Batch[Float]([Float("1.5"), Float(-2), Float("0.5")])
    print(bases ** Batch[Integer]([2, 3, -1]))
    print(Float(2) ** Batch[Integer]([-2, -1, 0, 1, 2]))
    var narrow = ArithmeticContext(format=FloatFormat(3))
    print(vmap[float.pow_int]()(bases, -1, context=narrow))
