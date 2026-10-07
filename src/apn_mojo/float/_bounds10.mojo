"""Directed bounds of `n * 2**s * 10**k`, for decimal conversions with large
exponents.

An exact power of ten is as long as its exponent: printing `2**(10**12)` or
reading `1e1000000000000` exactly would build integers of about 10**12 bits.
These conversions only need a short quotient, so they bound it instead: the
power of ten is formed by squaring with every product rounded down for a lower
bound and up for an upper bound, at a working precision. When the bounds
decide the result it is exact; otherwise the caller raises the precision and
tries again (Ziv's strategy). Callers use this only where an exact tie is
impossible, which guarantees that a high enough precision decides; the
derivations are in the conversion chapter.
"""

from ..integer.value import Integer
from ..integer._word_math import _trailing_zero_bits
from ..common.conversion import _ConversionBudget

# Exact conversions handle exponents whose powers stay below about this many
# bits; beyond it, the bounds below take over.
comptime _EXACT_POWER_BITS = 65536


@fieldwise_init
struct _Bound(ImplicitlyCopyable):
    """The nonnegative number `mantissa * 2**exponent`."""

    var mantissa: Integer
    var exponent: Int


def _log10_pow2(n: Int) -> Int:
    """floor(n * log10(2)), or one less: within one of it for every Int n.

    The constant is log10(2) rounded down to 18 digits, 2.2e-19 below it, so
    the product's error stays below one at every 64-bit n."""
    comptime unit = Int128(1000000000000000000)
    var product = Int128(n) * Int128(301029995663981195)
    if product >= 0:
        return Int(product // unit)
    return Int(-((-product + unit - 1) // unit)) - 1


def _shortened(value: Integer, exponent: Int, precision: Int, up: Bool) raises -> _Bound:
    """`value * 2**exponent` (value >= 0) rounded down or up to `precision` bits."""
    var extra = value.magnitude_bit_length() - precision
    if extra <= 0:
        return _Bound(value, exponent)
    var kept = value >> extra
    if up and _trailing_zero_bits(value) < extra:
        kept += 1
    return _Bound(kept^, exponent + extra)


def _power10_bound(k: Int, precision: Int, up: Bool, mut budget: _ConversionBudget) raises -> _Bound:
    """A lower (`up` False) or upper bound of `10**k` for k >= 0, of at most
    `precision` bits: `5**k` by squaring, each product rounded the same way,
    times `2**k`. Every factor is positive, so rounding each product down
    (up) bounds the power from below (above)."""
    var result = _Bound(Integer(1), 0)
    var base = _Bound(Integer(5), 0)
    var n = k
    while n:
        if n & 1:
            budget.allocate(2 * (precision // 32 + 2), 4)
            result = _shortened(result.mantissa * base.mantissa, result.exponent + base.exponent, precision, up)
        n >>= 1
        if n:
            budget.allocate(2 * (precision // 32 + 2), 4)
            base = _shortened(base.mantissa * base.mantissa, 2 * base.exponent, precision, up)
    result.exponent += k
    return result


def _scaled_bounds(
    n: Integer, s: Int, k: Int, precision: Int, mut budget: _ConversionBudget
) raises -> Tuple[_Bound, _Bound]:
    """Lower and upper bounds of `n * 2**s * 10**k` for n > 0, each with at
    least `precision` significant bits."""
    if k >= 0:
        var low = _power10_bound(k, precision, False, budget)
        var high = _power10_bound(k, precision, True, budget)
        budget.allocate(2 * ((n.magnitude_bit_length() + precision) // 32 + 2), 4)
        return (
            _Bound(n * low.mantissa, s + low.exponent),
            _Bound(n * high.mantissa, s + high.exponent),
        )
    # n * 2**s / 10**-k: divide by the upper bound for the lower bound, and by
    # the lower bound, rounding the quotient up, for the upper bound.
    var over = _power10_bound(-k, precision, True, budget)
    var under = _power10_bound(-k, precision, False, budget)
    return (_quotient(n, s, over, precision, False, budget), _quotient(n, s, under, precision, True, budget))


def _quotient(
    n: Integer, s: Int, d: _Bound, precision: Int, up: Bool, mut budget: _ConversionBudget
) raises -> _Bound:
    """`n * 2**s / (d.mantissa * 2**d.exponent)` rounded down or up, with at
    least `precision` bits."""
    var t = precision + 2 + d.mantissa.magnitude_bit_length() - n.magnitude_bit_length()
    budget.allocate(2 * ((n.magnitude_bit_length() + max(t, 0) + precision) // 32 + 2), 4)
    var a = n << t if t > 0 else n
    var b = d.mantissa << -t if t < 0 else d.mantissa
    var parts = a._div_rem_trunc(b)
    var q = parts[0]
    if up and parts[1]:
        q += 1
    return _Bound(q^, s - t - d.exponent)


def _floor(bound: _Bound) raises -> Tuple[Integer, Bool]:
    """`floor(bound)`, and whether the bound is an integer."""
    if bound.exponent >= 0:
        return (bound.mantissa << bound.exponent, True)
    var shift = -bound.exponent
    return (bound.mantissa >> shift, _trailing_zero_bits(bound.mantissa) >= shift or not bound.mantissa)


def _floor_scaled(
    n: Integer, s: Int, k: Int, precision: Int, mut budget: _ConversionBudget
) raises -> Optional[Integer]:
    """`floor(n * 2**s * 10**k)` for n > 0 when the bounds at `precision` bits
    decide it and show that the value is not an integer; None otherwise."""
    var bounds = _scaled_bounds(n, s, k, precision, budget)
    var low = _floor(bounds[0])
    var high = _floor(bounds[1])
    if low[1] or low[0] != high[0]:
        return None
    return low[0]
