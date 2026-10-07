"""Correct rounding by Ziv's strategy: enclose, round both ends, retry wider.

A correctly rounded function evaluates a kernel, an enclosure of its exact
value at the exact argument, at a working precision `w`. When both ends of the
enclosure round to the same Float, with the same status, every point of the
ball rounds there too, the exact value among them. Otherwise the guard bits
`w - p` double. The first working precision is `p + 32 + bit_length(p)`, and the
context's budget bounds the last: past it the function raises an error that
names the function, the argument and the budget.

The loop terminates for every argument that is not an exact case: the
functions' values at binary fractions are transcendental there
(Lindemann-Weierstrass, Gelfond-Schneider), while every rounding boundary is a
binary fraction. Exact cases are found before the loop.

`_round_near` decides a value within a tiny distance of a Float of known sign,
such as `sin(x) = x - x**3/6 + ...` for a tiny `x`, without any kernel: no
rounding boundary lies between them, so rounding a point between them gives the
answer and its status.
"""

from std.bit import count_leading_zeros
from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float.status import NumericStatus
from ..float._arithmetic import _float_operation
from ..float._rounding import _RoundedBinary, _finish_round
from ..float._functions import _same_round
from .value import Ball


trait _Enclosure(ImplicitlyCopyable):
    """A function at a fixed exact argument, enclosed at any precision."""

    def enclosure(self, precision: Int) raises -> Ball:
        """A ball containing the exact value, with about `precision` bits of
        relative accuracy."""
        ...

    def describe(self) raises -> String:
        """The function and argument, for the budget error."""
        ...


def _bit_length(n: Int) -> Int:
    return 64 - Int(count_leading_zeros(UInt64(n)))


def _initial_precision(p: Int, guard: Int) -> Int:
    """`p + 32 + bit_length(p)`, or `p + guard` for a positive test override."""
    return p + (guard if guard > 0 else 32 + _bit_length(p))


def _next_precision(p: Int, w: Int) -> Int:
    """The working precision after a failed step: the guard bits double."""
    return p + 2 * max(1, w - p)


def _budget_error(description: String, budget: Int) -> Error:
    return Error(String(
        "precision budget exceeded: ", description, " needed more than ", budget,
        " bits; pass a larger max_precision in the context.",
    ))


def _certify(y: Ball, target: ArithmeticContext) raises -> Optional[_RoundedBinary]:
    """Z1: the rounding of every point of `y` in the target format and mode,
    when both ends round alike, status included; None otherwise. Traps are not
    applied."""
    if not y.is_finite():
        return None
    var quiet = target._quiet()
    if y._radius.is_zero():
        return _float_operation(y._midpoint, Integer(0), 0, quiet)
    var radius = y._radius.to_float()
    var low = _float_operation(y._midpoint, radius, 1, quiet)
    var high = _float_operation(y._midpoint, radius, 0, quiet)
    if _same_round(low, high):
        return low^
    return None


def _round_certified[E: _Enclosure](f: E, target: ArithmeticContext, guard: Int = 0) raises -> _RoundedBinary:
    """Z2: the correctly rounded value of `f` in the target's format and mode.

    Args:
        f: The function at its argument, which must not be an exact case.
        target: The output format, rounding mode, traps and budget.
        guard: A positive count of initial guard bits, for tests of the
            retry path; 0 for the default.

    Returns:
        The rounded value, its status checked against the traps.

    Raises:
        When the working precision would exceed the budget, or on a trap.
    """
    var p = target.format().precision()
    var budget = target._budget()
    var w = _initial_precision(p, guard)
    while True:
        if w > budget:
            raise _budget_error(f.describe(), budget)
        var rounded = _certify(f.enclosure(w), target)
        if rounded:
            return _finish_round(rounded.take(), target, False)
        w = _next_precision(p, w)


def _round_near(
    base: Float, upward: Bool, tiny: Int, target: ArithmeticContext,
) raises -> Optional[_RoundedBinary]:
    """The rounding of `v = base + d`, where `0 < |d| < 2**tiny` and `d > 0`
    exactly when `upward`, if `tiny` is small enough to decide it.

    With `E` the exponent of `base` (`2**(E-1) <= |base| < 2**E`), every
    rounding boundary and every Float of the target precision `p` other than
    `base` is at least `2**(E - max(p + 1, q) - 2)` from `base`, where `q` is
    `base`'s precision. So when `tiny <= E - max(p + 1, q) - 3`, the open
    interval from `base` to `base +/- 2**tiny` holds no boundary, and `v`
    rounds as its interior point `base +/- 2**(tiny - 1)` does, with the same
    status.

    Args:
        base: A finite nonzero Float.
        upward: Whether `v > base`.
        tiny: An exponent with `|v - base| < 2**tiny`.
        target: The output format, rounding mode and traps.

    Returns:
        The rounded value, its status checked against the traps; None when
        `tiny` is too large to decide.

    Raises:
        On a trap.
    """
    var p = target.format().precision()
    if tiny > base._exponent - max(p + 1, base.precision()) - 3:
        return None
    var step = Float(_rounded=_RoundedBinary(
        1, False, Integer(1), tiny, FloatFormat(1), NumericStatus(),
    ))
    var record = _float_operation(base, step, 0 if upward else 1, target._quiet())
    return _finish_round(record^, target, False)


trait _PairEnclosure(ImplicitlyCopyable):
    """A complex function at a fixed exact argument, enclosed part by part at
    any precision."""

    def enclosure(self, precision: Int) raises -> Tuple[Ball, Ball]:
        """Balls containing the real and imaginary parts of the exact value."""
        ...

    def describe(self) raises -> String:
        """The function and argument, for the budget error."""
        ...


def _round_certified_pair[E: _PairEnclosure](
    f: E,
    real_target: ArithmeticContext,
    imag_target: ArithmeticContext,
    guard: Int = 0,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    """Z2 for a complex value: each part correctly rounded in its own context,
    with the precision schedule and budget of `_round_certified`. A part that
    certifies is kept while the other needs more precision.

    Past the budget, a part whose enclosure still contains 0 points to a
    structural zero the function's rules missed, and the error says so."""
    var p = max(real_target.format().precision(), imag_target.format().precision())
    var budget = min(real_target._budget(), imag_target._budget())
    var w = _initial_precision(p, guard)
    var real = Optional[_RoundedBinary]()
    var imag = Optional[_RoundedBinary]()
    var zero_part = False
    while True:
        if w > budget:
            var description = f.describe()
            if zero_part:
                description += " (a part's enclosure still contains 0: a structural zero rule is missing)"
            raise _budget_error(description, budget)
        var pair = f.enclosure(w)
        if not real:
            real = _certify(pair[0], real_target)
        if not imag:
            imag = _certify(pair[1], imag_target)
        if real and imag:
            return (_finish_round(real.take(), real_target, False), _finish_round(imag.take(), imag_target, False))
        zero_part = (not real and _encloses_zero(pair[0])) or (not imag and _encloses_zero(pair[1]))
        w = _next_precision(p, w)


def _encloses_zero(x: Ball) raises -> Bool:
    if not x.is_finite():
        return False
    var low = x._exact_lower()
    var high = x._exact_upper()
    return (low.is_zero() or low._negative) and (high.is_zero() or not high._negative)
