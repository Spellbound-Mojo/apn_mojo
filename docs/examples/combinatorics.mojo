"""Factorials and binomial coefficients without overflow."""

from apn_mojo import (
    Batch,
    Integer,
    factorial,
    factorial2,
    comb,
    div_exact,
    vmap,
)
from apn_mojo import integer


def main() raises:
    print("30!:", factorial(30))
    print("100 choose 50:", comb(100, 50))
    print("empty choice:", comb(0, 0))
    print("pairings of 10 objects:", factorial2(9))
    print("multisets of 2 from 3 kinds:", comb(3, 2, repetition=True))
    print("20! / 18!:", div_exact(factorial(20), factorial(18)))
    var sizes = Batch[Integer].from_native([0, 1, 5, 10])
    print("factorials:", vmap[factorial]()(sizes))
    print("pairs:", vmap[integer.comb]()(sizes, 2))
