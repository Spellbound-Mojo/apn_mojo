"""Strict Float batch functions and integral conversion."""

from apn_mojo import (
    Batch,
    Float,
    Integer,
    FloatFormat,
    ArithmeticContext,
    square,
    ldexp,
    fma,
    float,
    vmap,
)


def main() raises:
    var values = Batch[Float]([Float("1.5"), Float(2), Float(3)])
    print(vmap[float.sqrt]()(vmap[square]()(values)))
    print(vmap[ldexp]()(values, Batch[Integer]([-1, 0, 2])))
    print(values.floor())
    print(values.ceil())
    var context = ArithmeticContext(format=FloatFormat(3))
    var x = Batch[Float]([Float("1.5")])
    print(vmap[fma]()(x, Float("1.5"), -2, context=context))
    print(vmap[square]()(x, context=context) - Float(2))
