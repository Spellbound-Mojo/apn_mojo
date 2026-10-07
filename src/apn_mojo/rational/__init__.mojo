"""Exact fractions and their functions, one declaration per name.

Every result is in lowest terms with a positive denominator. Apply a function
to batches with `vmap`, as in `vmap[apn_mojo.rational.add]()(xs, ys)`.
"""

from .value import Rational
from .math import (
    add, subtract, multiply, divide, abs, pow_rational, gcd, lcm, root_exact, stable_hash,
    floor, ceil, trunc, round, reciprocal, maximum, minimum, clip,
)
