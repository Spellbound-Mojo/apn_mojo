"""Batches of any rank, Boolean masks, `vmap`, and NumPy-style functions of
batches.

A `Batch[T]` holds Integers, Rationals, Floats, Complex numbers or balls with
value semantics: selections share values, and updates publish new storage, so
no batch sees another's later update. Long operations and mappings run on a
worker pool with the same results and errors as a sequential loop.

`from apn_mojo import batch` gives the NumPy-style namespace: `batch.exp(xs)`,
`batch.atan2(ys, xs)`, `batch.where(xs > 0, xs, zero)`, `batch.zeros[Float](n)`,
`batch.cumsum(xs)`. Each elementwise function maps the scalar function of the
same name over every element, with broadcasting; the scalar functions stay at
the package root.
"""

from .value import Batch
from .mask import Mask
from .mapping import vmap
from .functions import (
    add,
    subtract,
    multiply,
    divide,
    reciprocal,
    maximum,
    minimum,
    clip,
    abs,
    sqrt,
    pow_int,
    floor,
    ceil,
    trunc,
    round,
    conjugate,
    real,
    imag,
    angle,
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
    gamma,
    gammaln,
    digamma,
    erf,
    erfc,
    erfi,
    erfinv,
    expi,
    ndtr,
    log_ndtr,
    ndtri,
    sici,
    shichi,
    fresnel,
    lambertw,
    beta,
    betaln,
    poch,
    zeta,
    polygamma,
    gammainc,
    gammaincc,
    hyp1f1,
    betainc,
    hyp2f1,
    comb,
    where,
)
from .creation import zeros, ones, full, arange, linspace
from .joining import concatenate, stack
from .reductions import sum, prod, min, max, argmin, argmax, dot, vdot, cumsum, cumprod
