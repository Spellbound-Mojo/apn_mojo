"""Floats of any precision, their formats and contexts, and correctly rounded
functions, one declaration per name.

Every function computes its exact result and rounds it once. Without a
`context`, operands merge their formats; operands that are all exact need a
context, so exact arithmetic is never silently rounded. A context's traps make
a call raise when its result sets the condition; nothing else reports it.
Apply a function to batches with `vmap`, as in
`vmap[apn_mojo.float.sqrt]()(values, context=c)`.
"""

from .value import Float
from .context import ArithmeticContext, FloatFormat, RoundingMode
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
    square,
    sqrt,
    pow_int,
    ldexp,
    fma,
    stable_hash,
)
from .neighbors import nextafter, spacing, ulp_distance, equal_at_precision
from .shortest import shortest_decimal, ShortestDecimal
from .constants import pi, euler_e, ln2, log2_10, euler_gamma, catalan
from .special import (
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
from .elementary import (
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
)
