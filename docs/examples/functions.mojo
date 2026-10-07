"""Exact integer functions such as gcd, lcm and isqrt, on scalars and batches."""

from apn_mojo import Batch, Integer, Mask, gcd, lcm, isqrt, integer, vmap
from apn_mojo import sum, prod, min, max, dot, axpy
from apn_mojo import iroot, pow_mod, inverse_mod, div_exact


def main() raises:
    print("number theory:", gcd(-36, 48), lcm(-36, 48), isqrt(145))
    print("roots:", iroot(1000, 3), iroot(-1001, 3))
    print(
        "modular:",
        pow_mod(7, 100, 1000),
        inverse_mod(3, 11),
        pow_mod(7, -5, 1000),
    )
    print("exact division:", div_exact(-84, 7))
    var exponents = Batch[Integer].from_native([-1, 0, 1, 2])
    print("modular powers:", vmap[pow_mod]()(3, exponents, 11))
    var values = Batch[Integer].from_native([1, 2, 3, 4])
    print(
        "reductions:",
        sum(values),
        prod(values),
        min(values),
        max(values),
    )
    print("dot:", dot(values, values[::-1]))
    var shifted = axpy(3, values, values[::-1])
    print("axpy:", shifted[0], shifted[1], shifted[2], shifted[3])
    print("empty:", sum(Batch[Integer]()), prod(Batch[Integer]()))
    var signs = Batch[Integer].from_native([-1, 0, 1])
    print("unary:", vmap[integer.abs]()(signs)[0], (-signs)[2])
    var mask = Mask([True, False, True])
    print("mask:", mask.count(), mask.all(), mask.any(), (~mask).count())
