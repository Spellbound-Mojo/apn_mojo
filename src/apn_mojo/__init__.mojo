"""Arbitrary-precision numbers for Mojo: exact Integers and Rationals, Floats
of any precision rounded once, Complex numbers, and batches of any rank.

Import everything from here: `from apn_mojo import Integer, Float, Batch, sum`.
Each family also has its own package (`apn_mojo.integer`, `apn_mojo.float`, ...)
declaring one function per name, which is what `vmap` and `lift` take. The
names listed here are the ones only this package declares: functions that pick
a family from their arguments, the reductions, `lift` and `axpy`.
"""

from .integer.value import Integer
from .rational.value import Rational
from .float.value import Float
from .complex.value import Complex
from .complex.context import ComplexContext
from .exact_complex.value import ExactComplex
from .complex_ball.value import ComplexBall
from .complex.math import norm_sqr
from .common.functions import add, subtract, multiply, divide, sqrt, pow_int, abs, stable_hash
from .common.functions import reciprocal, conjugate, real, imag, floor, ceil, trunc, round, maximum, minimum, clip, comb
from .common.functions import (
    exp,
    expm1,
    exp2,
    log,
    log1p,
    log2,
    log10,
    sin,
    cos,
    tan,
    sin_cos,
    atan,
    asin,
    acos,
    atan2,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    pow,
    rootn,
    angle,
    gamma,
    gammaln,
    digamma,
    erf,
    erfc,
    erfi,
    expi,
    sici,
    shichi,
    fresnel,
    lambertw,
    ndtr,
    log_ndtr,
    beta,
    betaln,
    poch,
    erfinv,
    ndtri,
    zeta,
    polygamma,
    hyp1f1,
    gammainc,
    gammaincc,
    hyp2f1,
    betainc,
)
from .common.keys import FloatKey, ComplexKey, BallKey
from .ball.value import Ball, BallOrder
from .ball.context import BallContext
from .ball.math import round_midpoint, trim, add_error
from .ball.sets import (
    compare,
    contains,
    contains_zero,
    contains_ball,
    overlaps,
    contains_integer,
    union,
    intersection,
    split,
    to_float_if_certain,
    canonical,
    floor_if_certain,
    ceil_if_certain,
    round_half_even_if_certain,
    simplest_rational_in,
)
from .float.math import square, ldexp, fma
from .float.neighbors import nextafter, spacing, ulp_distance, equal_at_precision
from .float.shortest import shortest_decimal, ShortestDecimal
from .float.constants import pi, euler_e, ln2, log2_10, euler_gamma, catalan
from .ball.constants import pi_ball, euler_e_ball, ln2_ball, log2_10_ball, euler_gamma_ball, catalan_ball
from .ball.significance import bits_to_digits, digits_to_bits, radius_for_relative_digits, propagation_bound
from .rational.math import pow_rational, root_exact
from .integer.division import div_rem_floor, div_rem_trunc, div_rem_euclid
from .batch.value import Batch
from .batch.mask import Mask
from .batch.mapping import vmap
from .batch.lift import lift
from .common.threads import set_num_threads, get_num_threads
from .common.conversion import ConversionLimits
from .float.context import FloatFormat, ArithmeticContext, RoundingMode
from .integer.number_theory import gcd, lcm, isqrt, remove_factor, trial_division, TrialDivision, jacobi
from .integer.powers import iroot_exact, perfect_power
from .integer.primality import is_prime
from .integer.factorization import factor, FactorBudget, Factorization, FactorTerm, primes_below, next_prime
from .integer.math import (
    factorial,
    factorial2,
    iroot,
    pow_mod,
    inverse_mod,
    div_exact,
)
from .batch.reductions import sum, prod, min, max, argmin, argmax, dot, vdot
from .batch.reductions import sum_sequential, sum_tree, dot_sequential
from .integer.compound import axpy
