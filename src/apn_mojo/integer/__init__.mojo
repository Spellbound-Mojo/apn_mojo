"""Exact Integers and their functions, one declaration per name.

Every function is scalar and exact; apply one to batches with `vmap`, as in
`vmap[apn_mojo.integer.gcd]()(values, 12)`. Arguments accept Integers, native
integers of up to 64 bits and integer literals of any width. Functions raise on
a domain error, naming the problem and a remedy.
"""

from .value import Integer, div_rem_floor, div_rem_trunc, div_rem_euclid
from .math import (
    add,
    subtract,
    multiply,
    divide,
    abs,
    floor,
    ceil,
    trunc,
    round,
    reciprocal,
    maximum,
    minimum,
    clip,
    factorial,
    comb,
    factorial2,
    iroot,
    pow_mod,
    inverse_mod,
    div_exact,
    stable_hash,
)
from .number_theory import gcd, lcm, isqrt, remove_factor, trial_division, TrialDivision, jacobi
from .powers import iroot_exact, perfect_power
from .primality import is_prime
from .factorization import factor, FactorBudget, Factorization, FactorTerm, primes_below, next_prime
