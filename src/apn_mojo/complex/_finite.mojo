"""Two-term exact products and sparse quotient rounding, without dense gaps."""

from std.bit import count_leading_zeros, count_trailing_zeros
from ..integer.value import Integer
from ..integer._limits import _MAX_RESULT_BITS
from ..float._input import _FloatInput
from ..float._accumulator import _ExactAccumulator, _SparseMagnitude
from ..float._rounding import _RoundedBinary, _round_ratio, _exponent_add
from ..float.context import ArithmeticContext
from ..common._sizes import _checked_sum


def _negated(var value: _FloatInput) -> _FloatInput:
    if value.kind != 3:
        value.negative = not value.negative
    return value


def _finite_product(a: _FloatInput, b: _FloatInput) raises -> _FloatInput:
    return _FloatInput(
        0 if a.kind == 0 or b.kind == 0 else 1,
        a.negative != b.negative,
        a.numerator * b.numerator,
        a.denominator * b.denominator,
        _exponent_add(a.scale, b.scale),
    )


struct _ExactPair(ImplicitlyCopyable):
    var a: Integer
    var b: Integer
    var a_scale: Int128
    var b_scale: Int128
    var denominator: Integer
    var negative_zero: Bool

    def __init__(
        out self, a: _FloatInput, b: _FloatInput, negative_zero: Bool
    ) raises:
        self.a = (-(a.numerator) if a.negative else a.numerator) * b.denominator
        self.b = (-(b.numerator) if b.negative else b.numerator) * a.denominator
        self.a_scale = a.scale
        self.b_scale = b.scale
        self.denominator = a.denominator * b.denominator
        self.negative_zero = negative_zero

    def accumulate(
        self,
        mut total: _ExactAccumulator,
        factor: Integer = 1,
        scale: Int128 = 0,
    ) raises:
        total.add(self.a * factor, _exponent_add(self.a_scale, scale))
        total.add(self.b * factor, _exponent_add(self.b_scale, scale))

    def magnitude(self) raises -> _SparseMagnitude:
        var total = _ExactAccumulator()
        self.accumulate(total)
        return total.magnitude(self.negative_zero)

    def compact(self) raises -> Optional[Tuple[Integer, Int128]]:
        var value: Integer
        var scale: Int128
        if not self.a:
            value, scale = self.b, self.b_scale
        elif not self.b:
            value, scale = self.a, self.a_scale
        else:
            scale = min(self.a_scale, self.b_scale)
            var high = max(self.a_scale, self.b_scale)
            # Unsigned subtraction also handles opposite-sign Int128 scales.
            var gap = UInt128(high) - UInt128(scale)
            var bits = max(self.a.magnitude_bit_length(), self.b.magnitude_bit_length())
            if gap > UInt128(bits) or UInt128(bits) + gap + 1 > UInt128(_MAX_RESULT_BITS):
                return None
            # Bound temporary width by twice the existing coefficient width.
            # Larger exponent gaps stay sparse, independent of output precision.
            var a = self.a << Int(gap) if self.a_scale > scale else self.a
            var b = self.b << Int(gap) if self.b_scale > scale else self.b
            value = a + b
        if value:
            var words = 0
            while not value._word(words):
                words += 1
            var zeros = words * 32 + Int(count_trailing_zeros(value._word(words)))
            value >>= zeros
            scale = _exponent_add(scale, Int128(zeros))
        return (value, scale)

    def rounded(
        self, context: ArithmeticContext, fail: Bool = False
    ) raises -> _RoundedBinary:
        # Nearby terms add into one Integer and round as a ratio; only terms
        # far apart need the sparse accumulator.
        var compact = self.compact()
        if compact:
            ref pair = compact.value()
            return _round_ratio(
                pair[0], self.denominator, context, scale=pair[1],
                negative_zero=self.negative_zero, fail=fail,
            )
        var magnitude = self.magnitude()
        return magnitude.rounded(context, fail, denominator=self.denominator)


def _magnitude_top(value: _SparseMagnitude) -> Int128:
    var last = value.runs[len(value.runs) - 1]
    return Int128(last.high) * 32 + Int128(
        32 - Int(count_leading_zeros(last.digit))
    )


def _quotient_compare(
    numerator: _ExactPair,
    denominator: _ExactPair,
    negative: Bool,
    threshold: Integer,
    scale: Int128,
) raises -> Int:
    # Cross-multiply the four original terms, never the gaps between them.
    var difference = _ExactAccumulator()
    numerator.accumulate(
        difference,
        -denominator.denominator if negative else denominator.denominator,
    )
    denominator.accumulate(
        difference, -(numerator.denominator * threshold), scale
    )
    var magnitude = difference.magnitude()
    return (-1 if magnitude.negative else 1) if len(magnitude.runs) else 0


def _pair_quotient(
    numerator: _ExactPair,
    denominator: _ExactPair,
    context: ArithmeticContext,
    fail: Bool = False,
) raises -> _RoundedBinary:
    var compact_n = numerator.compact()
    if compact_n:
        var n, ns = compact_n.value()
        if not n:
            return _round_ratio(
                Integer(0), Integer(1), context,
                negative_zero=numerator.negative_zero, fail=fail,
            )
        var compact_d = denominator.compact()
        if compact_d:
            var d, ds = compact_d.value()
            return _round_ratio(
                n * denominator.denominator, d * numerator.denominator,
                context, scale=_exponent_add(ns, -ds), fail=fail,
            )
    var n = numerator.magnitude()
    if not len(n.runs):
        return _round_ratio(
            Integer(0), Integer(1), context, negative_zero=n.negative, fail=fail
        )
    # Absorb ordinary rational denominators into the two sparse magnitudes.
    var n_scaled = _ExactAccumulator()
    numerator.accumulate(n_scaled, denominator.denominator)
    n = n_scaled.magnitude()
    var d_scaled = _ExactAccumulator()
    denominator.accumulate(d_scaled, numerator.denominator)
    var d = d_scaled.magnitude()
    var nt = _magnitude_top(n)
    var dt = _magnitude_top(d)
    var exponent = nt - dt
    exponent += Int128(
        _quotient_compare(
            numerator, denominator, n.negative, Integer(1), exponent
        )
        >= 0
    )
    var format = context.format()
    if exponent < Int128(format.emin()):
        var order = _quotient_compare(
            numerator,
            denominator,
            n.negative,
            Integer(1),
            Int128(format.emin()) - 2,
        )
        var witness = Integer(1 if order < 0 else 2 if order == 0 else 3)
        return _round_ratio(
            -witness if n.negative else witness,
            Integer(1),
            context,
            scale=Int128(format.emin()) - 3,
            fail=fail,
        )
    if exponent > Int128(format.emax()):
        return _round_ratio(
            Integer(-1 if n.negative else 1),
            Integer(1),
            context,
            scale=exponent - 1,
            fail=fail,
        )
    var bits = _checked_sum(format.precision(), 4)
    var a = abs(n.prefix(nt, bits))
    var b = abs(d.prefix(dt, bits))
    var scale = exponent - Int128(format.precision())
    var shift = Int(nt - dt - scale)
    var q = (a << shift) // b
    # p+4-bit sticky prefixes place this candidate within one integer of floor.
    var order = _quotient_compare(numerator, denominator, n.negative, q, scale)
    if order < 0:
        q -= 1
        order = _quotient_compare(numerator, denominator, n.negative, q, scale)
    elif (
        _quotient_compare(numerator, denominator, n.negative, q + 1, scale) >= 0
    ):
        q += 1
        order = _quotient_compare(numerator, denominator, n.negative, q, scale)
    if (
        order < 0
        or _quotient_compare(numerator, denominator, n.negative, q + 1, scale)
        >= 0
    ):
        raise Error(
            "Cannot certify Complex division prefix; retain the inputs and"
            " report this internal arithmetic failure. The destination is"
            " unchanged."
        )
    if order == 0:
        return _round_ratio(
            -q if n.negative else q, Integer(1), context, scale=scale, fail=fail
        )
    var middle = _quotient_compare(
        numerator, denominator, n.negative, q * 2 + 1, scale - 1
    )
    var witness = q * 4 + (1 if middle < 0 else 2 if middle == 0 else 3)
    return _round_ratio(
        -witness if n.negative else witness,
        Integer(1),
        context,
        scale=scale - 2,
        fail=fail,
    )
