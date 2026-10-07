"""Kernels of pow and rootn.

`pow(x, y) = exp(y log x)` for a positive `x`: `log x` is taken at `w` plus
the bit length of `y log x` plus 8 bits, so the exponent's absolute error stays
below `2**-(w + 8)`, and `exp` of that ball widens by its radius. `rootn`
rounds exactly in both directions, as `sqrt` does, so its kernel is their hull.
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._functions import _rootn_float
from .context import BallContext
from .value import Ball, _BallArgument
from ._arithmetic import _product, _hull, _directed
from ._exp import _exp_ball
from ._log import _kernel_log


def _kernel_pow(x: Float, y: Float, w: Int) raises -> Ball:
    """`x**y` for a positive Float `x != 1` and a finite nonzero Float `y`."""
    var rough = _product(_BallArgument(_kernel_log(x, 32)), _BallArgument(y), Optional[BallContext](BallContext(64)))
    var extra = 0
    if rough.is_finite() and not rough._midpoint.is_zero():
        extra = max(0, rough._midpoint._exponent)
    var bits = w + extra + 8
    var t = _product(_BallArgument(_kernel_log(x, bits)), _BallArgument(y), Optional[BallContext](BallContext(bits + 8)))
    return _exp_ball(t, w)


def _kernel_rootn(x: Float, n: Int, w: Int) raises -> Ball:
    """The n-th root of a Float, rounded down and up at `w` bits; `c = 1`."""
    var low = Float(_rounded=_rootn_float(x, n, _directed(w, False)))
    var high = Float(_rounded=_rootn_float(x, n, _directed(w, True)))
    return _hull(low, high, w)
