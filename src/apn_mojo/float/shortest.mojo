"""The shortest decimal that reads back as the same Float."""

from ..integer.value import Integer
from ..common.conversion import ConversionLimits, _ConversionBudget
from .value import Float
from ._decimal import _layout_decimal, _check_notation
from ._bounds10 import _EXACT_POWER_BITS, _floor_scaled, _log10_pow2


@fieldwise_init
struct ShortestDecimal(ImplicitlyCopyable, Writable):
    """A decimal `(-1)**negative * digits * 10**exponent10`; see
    `shortest_decimal`."""

    var _negative: Bool
    var _digits: Integer
    var _exponent10: Int

    def negative(self) -> Bool:
        """The sign, including that of a negative zero.

        Returns:
            True for a negative value or `-0`.
        """
        return self._negative

    def digits(self) -> String:
        """The decimal digits, without leading or trailing zeros.

        Returns:
            The digits; `"0"` for a zero.
        """
        return String(self._digits)

    def exponent10(self) -> Int:
        """The power of ten that scales the digits.

        Returns:
            The exponent; 0 for a zero.
        """
        return self._exponent10

    def write_to(self, mut writer: Some[Writer]):
        """Write the value as `digits` and a decimal exponent, such as
        `3141592653589793e-15`, which reads back as the same Float.

        Args:
            writer: The destination.
        """
        if self._negative:
            writer.write("-")
        writer.write(self._digits, "e", self._exponent10)


struct _Interval:
    """The values that round (nearest-even) to one Float, scaled to
    `[low, high] * 2**scale`, with each end included or not."""

    var low: Integer
    var high: Integer
    var middle: Integer
    var scale: Int
    var low_included: Bool
    var high_included: Bool

    def __init__(out self, x: Float) raises:
        var format = x._format
        var precision = format.precision()
        var significand = x._significand
        # Four times the significand leaves room for quarter-unit ends.
        self.middle = significand << 2
        self.scale = x._exponent - precision - 2
        var power_of_two = significand == Integer(1) << (precision - 1)
        var even = (significand._word(0) & 1) == 0
        var below = Integer(2)
        if power_of_two and x._exponent > format.emin():
            # The neighbour below lies in the binade below, half as far.
            below = Integer(1)
        elif power_of_two:
            # Below the smallest value, values round to it down to half of
            # it; the halfway point itself rounds to zero.
            below = significand << 1
        self.low = self.middle - below
        self.high = self.middle + 2
        self.low_included = even and not (power_of_two and x._exponent == format.emin())
        self.high_included = even

    def candidates(
        self, exponent10: Int, mut budget: _ConversionBudget
    ) raises -> Tuple[Integer, Integer, Integer, Integer]:
        """The least and greatest d with d * 10**exponent10 inside, and the
        scaling `(a, b)` with x / 10**exponent10 = middle * a / b."""
        # The scale and the power of ten, and the products and quotients below.
        budget.allocate(3 * ((abs(self.scale) + 4 * abs(exponent10) + self.middle.magnitude_bit_length()) // 32 + 4), 4)
        var a = Integer(1)
        var b = Integer(1)
        if self.scale >= 0:
            a = a << self.scale
        else:
            b = b << -self.scale
        if exponent10 >= 0:
            b = b * Integer(10) ** exponent10
        else:
            a = a * Integer(10) ** (-exponent10)
        var lower = (self.low * a)._div_rem_trunc(b)
        var least = lower[0] + Integer(1 if lower[1] or not self.low_included else 0)
        var upper = (self.high * a)._div_rem_trunc(b)
        var greatest = upper[0] - Integer(1 if not upper[1] and not self.high_included else 0)
        return (least, greatest, a, b)

    def fits(self, exponent10: Int, mut budget: _ConversionBudget) raises -> Bool:
        var found = self.candidates(exponent10, budget)
        return found[0] <= found[1]

    def large(self) -> Bool:
        """Whether the exact powers would be long: then bounds decide, and no
        candidate can lie on an end of the interval (see _bounds10)."""
        return abs(self.scale) > _EXACT_POWER_BITS + 8 * self.middle.magnitude_bit_length()

    def bounded_range(self, exponent10: Int, precision: Int, mut budget: _ConversionBudget) raises -> Optional[Tuple[Integer, Integer]]:
        """For a large interval, the least and greatest d with d * 10**exponent10
        inside, from bounds at `precision` bits; None when they do not decide.
        No end is a multiple of 10**exponent10, so the least is above the floor
        of the low end."""
        var low = _floor_scaled(self.low, self.scale, -exponent10, precision, budget)
        var high = _floor_scaled(self.high, self.scale, -exponent10, precision, budget)
        if not low or not high:
            return None
        return (low.value() + 1, high.value())


def shortest_decimal(x: Float, *, limits: Optional[ConversionLimits] = None) raises -> ShortestDecimal:
    """The decimal with the fewest significant digits that reads back as `x`.

    Reading back rounds to nearest-even in `x`'s format. Among the decimals
    with that many digits, the result is the one closest to `x`, and on a tie
    the one with an even last digit. For binary64 values this is the decimal
    Python's `repr` prints, except at the smallest value: a format has no
    subnormals, so every value from half the smallest value up to it reads back
    as it, and `2e-308` is the shortest decimal of `2**-1022`.

    The search is exact: the values that round to `x` form an interval, and a
    binary search finds the largest power of ten whose multiples meet it.

    Args:
        x: A finite Float.
        limits: Optional per-call conversion limits; see `ConversionLimits`.

    Returns:
        The sign, the digits and the decimal exponent.

    Raises:
        For infinity and NaN, or when the digits exceed `limits`.
    """
    var budget = _ConversionBudget(limits)
    budget.values(1)
    return _shortest_decimal(x, budget)


def _shortest_decimal(x: Float, mut budget: _ConversionBudget) raises -> ShortestDecimal:
    """`shortest_decimal` against a caller's budget, which counts no value."""
    if not x.is_finite():
        raise Error(
            "Cannot write infinity or NaN as a decimal; check is_finite()"
            " first. The destination is unchanged."
        )
    if x.is_zero():
        budget.digits(1)
        return ShortestDecimal(x.signbit(), Integer(0), 0)
    var interval = _Interval(x)
    if interval.large():
        # Bounds at p + 64 bits nearly always decide; each retry doubles them.
        var precision = interval.middle.magnitude_bit_length() + 64
        for _ in range(12):
            var found = _bounded_shortest(interval, precision, budget)
            if found:
                var digits = found.value()[0]
                budget.digits(String(digits).byte_length())
                return ShortestDecimal(x._negative, digits^, found.value()[1])
            precision *= 2
    # 10**fitting meets the interval, 10**failing does not; log10(2) is just
    # above 0.30103, and the margins absorb that estimate's error.
    var high_bits = interval.high.magnitude_bit_length() + interval.scale
    var fitting = Int(Float64(interval.scale) * 0.30103) - 2
    var failing = Int(Float64(high_bits) * 0.30103) + 2
    while not interval.fits(fitting, budget):
        fitting -= 1
    while interval.fits(failing, budget):
        failing += 1
    while failing - fitting > 1:
        var middle = fitting + (failing - fitting) // 2
        if interval.fits(middle, budget):
            fitting = middle
        else:
            failing = middle
    var found = interval.candidates(fitting, budget)
    # The candidate nearest x, ties to even, kept inside the interval.
    var nearest = (interval.middle * found[2])._div_rem_trunc(found[3])
    var digits = nearest[0]
    var twice = nearest[1] << 1
    if twice > found[3] or (twice == found[3] and (digits._word(0) & 1) != 0):
        digits += 1
    if digits < found[0]:
        digits = found[0]
    elif digits > found[1]:
        digits = found[1]
    budget.digits(String(digits).byte_length())
    return ShortestDecimal(x._negative, digits^, fitting)


def _bounded_shortest(
    interval: _Interval, precision: Int, mut budget: _ConversionBudget
) raises -> Optional[Tuple[Integer, Int]]:
    """The search of `_shortest_decimal` on bounds at `precision` bits: the
    digits and exponent, or None when the bounds do not decide a step."""
    # Exact estimates: at these exponents 0.30103 would be off by billions.
    var high_bits = interval.high.magnitude_bit_length() + interval.scale
    var fitting = _log10_pow2(interval.scale) - 2
    var failing = _log10_pow2(high_bits) + 3
    while True:
        var span = interval.bounded_range(fitting, precision, budget)
        if not span:
            return None
        if span.value()[0] <= span.value()[1]:
            break
        fitting -= 1
    while True:
        var span = interval.bounded_range(failing, precision, budget)
        if not span:
            return None
        if span.value()[0] > span.value()[1]:
            break
        failing += 1
    while failing - fitting > 1:
        var middle = fitting + (failing - fitting) // 2
        var span = interval.bounded_range(middle, precision, budget)
        if not span:
            return None
        if span.value()[0] <= span.value()[1]:
            fitting = middle
        else:
            failing = middle
    var span = interval.bounded_range(fitting, precision, budget)
    # Twice x / 10**fitting is not an integer either, so the nearest candidate
    # is floor(2v + 1) // 2 without a tie.
    var twice = _floor_scaled(interval.middle << 1, interval.scale, -fitting, precision, budget)
    if not span or not twice:
        return None
    var digits = (twice.value() + 1) >> 1
    if digits < span.value()[0]:
        digits = span.value()[0]
    elif digits > span.value()[1]:
        digits = span.value()[1]
    return (digits^, fitting)


def _format_float_shortest(x: Float, notation: StaticString, mut budget: _ConversionBudget) raises -> String:
    """`x` as its shortest round-trip decimal in `notation`: `nan`, `inf` and
    `-inf` as such, zero as `0.0` or `-0.0`. The caller counts the value."""
    _check_notation(notation)
    if x.is_nan():
        budget.output(3)
        return String("nan")
    if x.is_infinite():
        budget.output(3 + Int(x.signbit()))
        return String("-inf" if x.signbit() else "inf")
    var decimal = _shortest_decimal(x, budget)
    var digits = decimal._digits._format_with_budget(10, False, False, budget)
    var exponent = decimal._exponent10 + digits.byte_length() - 1
    return _layout_decimal(decimal._negative, digits, exponent, notation, budget)
