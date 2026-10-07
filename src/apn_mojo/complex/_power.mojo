"""Certified integral powers with bounded significands and unbounded scales."""

from ..integer.value import Integer
from ..integer.division import div_rem_trunc
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._arithmetic import _float_special
from ..float._rounding import _RoundedBinary, _round_ratio, _finish_round, _inexact_error, _is_power_of_two
from ..float._functions import _wide_integer, _round_scaled_ratio, _same_round
from ..float._accumulator import _ExactAccumulator
from .context import _ComplexContextArgument
from ._input import _ComplexArgument
from ..float._input import _FloatInput
from ._arithmetic import _target


@fieldwise_init
struct _Bound(ImplicitlyCopyable):
    var coefficient: Integer
    var scale: Integer
    var strict: Bool

    def negated(self) raises -> Self:
        return Self(-self.coefficient, self.scale, self.strict)

    def rounded(
        self, context: ArithmeticContext, upper: Bool
    ) raises -> _RoundedBinary:
        var m = self.coefficient
        var s = self.scale
        if self.strict:
            # A dyadic endpoint's one-sided limit, below every output boundary.
            if m == 0:
                m = Integer(-1 if upper else 1)
                s = Integer(context.format().emin()) - 3
            else:
                var extra = max(
                    4,
                    context.format().precision() - m.magnitude_bit_length() + 4,
                )
                m = (m << extra) + (-1 if upper else 1)
                s -= extra
        if m == 0:
            return _float_special(0, False, 0, context, False)
        return _round_scaled_ratio(
            abs(m), Integer(1), s, m.sign() < 0, context, False
        )


def _order(a: _Bound, b: _Bound) raises -> Int:
    var sa = a.coefficient.sign()
    var sb = b.coefficient.sign()
    if sa != sb:
        return -1 if sa < sb else 1
    if sa == 0:
        return 0
    var am = abs(a.coefficient)
    var bm = abs(b.coefficient)
    var at = a.scale + am.magnitude_bit_length()
    var bt = b.scale + bm.magnitude_bit_length()
    if at != bt:
        return sa * (-1 if at < bt else 1)
    var shift = am.magnitude_bit_length() - bm.magnitude_bit_length()
    if shift >= 0:
        bm <<= shift
    else:
        am <<= -shift
    return sa * (-1 if am < bm else 1 if am > bm else 0)


def _extreme(a: _Bound, b: _Bound, upper: Bool) raises -> _Bound:
    var order = _order(a, b)
    if order == 0:
        return _Bound(a.coefficient, a.scale, a.strict and b.strict)
    return a if (order > 0) == upper else b


def _quantize(a: _Bound, scale: Integer, upper: Bool) raises -> _Bound:
    var shift = scale - a.scale
    if a.coefficient == 0:
        return _Bound(Integer(0), scale, a.strict)
    var m = abs(a.coefficient)
    var q = m
    var lost = False
    if shift > 0:
        if shift >= m.magnitude_bit_length():
            q = Integer(0)
            lost = True
        else:
            var count = Int(shift)
            q = m >> count
            lost = (q << count) != m
        if lost and upper != (a.coefficient.sign() < 0):
            q += 1
    else:
        q <<= Int(-shift)
    return _Bound(
        -q if a.coefficient.sign() < 0 else q, scale, a.strict or lost
    )


def _trim(a: _Bound, work: Int, upper: Bool) raises -> _Bound:
    var discard = a.coefficient.magnitude_bit_length() - work
    return _quantize(a, a.scale + discard, upper) if discard > 0 else a


def _sum_bound(a: _Bound, b: _Bound, work: Int, upper: Bool) raises -> _Bound:
    if a.coefficient == 0 and not a.strict:
        return b
    if b.coefficient == 0 and not b.strict:
        return a
    var top = max(
        a.scale + a.coefficient.magnitude_bit_length(),
        b.scale + b.coefficient.magnitude_bit_length(),
    )
    var scale = max(min(a.scale, b.scale), top - work)
    var x = _quantize(a, scale, upper)
    var y = _quantize(b, scale, upper)
    return _trim(
        _Bound(x.coefficient + y.coefficient, scale, x.strict or y.strict),
        work,
        upper,
    )


def _product_bound(a: _Bound, b: _Bound) raises -> _Bound:
    return _Bound(
        a.coefficient * b.coefficient,
        a.scale + b.scale,
        (a.strict and b.coefficient != 0)
        or (b.strict and a.coefficient != 0),
    )


def _reciprocal_bound(a: _Bound, work: Int, upper: Bool) raises -> _Bound:
    var m = abs(a.coefficient)
    var bits = work + m.magnitude_bit_length()
    var qr = div_rem_trunc(Integer(1) << bits, m)
    var lost = qr[1] != 0
    var q = qr[0]
    if lost and upper != (a.coefficient.sign() < 0):
        q += 1
    return _Bound(
        -q if a.coefficient.sign() < 0 else q, -a.scale - bits, a.strict or lost
    )


@fieldwise_init
struct _Interval(ImplicitlyCopyable):
    var lo: _Bound
    var hi: _Bound

    @staticmethod
    def exact(m: Integer, scale: Integer) -> Self:
        var bound = _Bound(m, scale, False)
        return Self(bound, bound)

    def negated(self) raises -> Self:
        return Self(self.hi.negated(), self.lo.negated())

    def plus(self, b: Self, work: Int) raises -> Self:
        return Self(
            _sum_bound(self.lo, b.lo, work, False),
            _sum_bound(self.hi, b.hi, work, True),
        )

    def times(self, b: Self, work: Int) raises -> Self:
        var product = _product_bound(self.lo, b.lo)
        var lo = _trim(product, work, False)
        var hi = _trim(product, work, True)
        for i in range(1, 4):
            var x = self.hi if i & 2 else self.lo
            var y = b.hi if i & 1 else b.lo
            product = _product_bound(x, y)
            lo = _extreme(lo, _trim(product, work, False), False)
            hi = _extreme(hi, _trim(product, work, True), True)
        return Self(lo, hi)

    def squared(self, work: Int) raises -> Self:
        var left = _product_bound(self.lo, self.lo)
        var right = _product_bound(self.hi, self.hi)
        var hi = _extreme(
            _trim(left, work, True),
            _trim(right, work, True),
            True,
        )
        var lo = _Bound(Integer(0), Integer(0), False)
        if self.lo.coefficient.sign() > 0 or self.hi.coefficient.sign() < 0:
            lo = _extreme(
                _trim(left, work, False),
                _trim(right, work, False),
                False,
            )
        return Self(lo, hi)

    def reciprocal(self, work: Int) raises -> Self:
        return Self(
            _reciprocal_bound(self.hi, work, False),
            _reciprocal_bound(self.lo, work, True),
        )

    def scaled(self, shift: Integer) raises -> Self:
        return Self(
            _Bound(self.lo.coefficient, self.lo.scale + shift, self.lo.strict),
            _Bound(self.hi.coefficient, self.hi.scale + shift, self.hi.strict),
        )


def _power_intervals(
    a: _Interval, b: _Interval, count: Integer, work: Int
) raises -> Tuple[_Interval, _Interval]:
    var real = _Interval.exact(Integer(1), Integer(0))
    var imag = _Interval.exact(Integer(0), Integer(0))
    for bit in range(count.magnitude_bit_length() - 1, -1, -1):
        var r = real.squared(work).plus(imag.squared(work).negated(), work)
        imag = real.times(imag, work).scaled(Integer(1))
        real = r
        if count._word(bit // 32) & (UInt32(1) << UInt32(bit % 32)):
            r = real.times(a, work).plus(imag.times(b, work).negated(), work)
            imag = real.times(b, work).plus(imag.times(a, work), work)
            real = r
    return real, imag


def _exact_component(
    zero: Bool, negative: Bool, numerator: Integer, denominator: Integer, scale: Int128,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    if zero:
        return _float_special(0, negative, 0, context, fail)
    return _round_ratio(-numerator if negative else numerator, denominator, context, scale=scale, fail=fail)


def _exact_pow_complex(
    a: _FloatInput, b: _FloatInput, exponent: Integer, count: Integer, inverse: Bool, axis: Bool, diagonal: Bool,
    real_zero: Bool, imag_zero: Bool, real_negative: Bool, imag_negative: Bool,
    rc: ArithmeticContext, ic: ArithmeticContext, fail_component: Int,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    """A power of finite exact working components, exactly: each component a
    finite binary fraction, or an inexact error. Zeros and signs on the axes
    and diagonals come from the same rules as rounded powers."""
    if not (a.denominator._is_one() and b.denominator._is_one()):
        raise _inexact_error("pow")
    var n = Int(count)
    if axis or diagonal:
        # One magnitude |c|**n, times 2**(exponent // 2) on a diagonal,
        # placed on the components by the sign rules.
        var magnitude = (b.numerator if a.kind == 0 else a.numerator) ** n
        var scale = (b.scale if a.kind == 0 else a.scale) * Int128(n)
        var shift = Int128(Int(exponent // 2)) if diagonal else Int128(0)
        var top = Integer(1) if inverse else magnitude
        var bottom = magnitude if inverse else Integer(1)
        var at = (-scale if inverse else scale) + shift
        return (
            _exact_component(real_zero, real_negative, top, bottom, at, rc, fail_component == 0),
            _exact_component(imag_zero, imag_negative, top, bottom, at, ic, fail_component == 1),
        )
    # (x + yi)**n on a common scale, by repeated squaring of Gaussian integers.
    var s = min(a.scale, b.scale)
    var x = a.numerator << Int(a.scale - s)
    var y = b.numerator << Int(b.scale - s)
    if a.negative:
        x = -x
    if b.negative:
        y = -y
    var re = Integer(1)
    var im = Integer(0)
    var k = n
    while k:
        if k & 1:
            var t = re * x - im * y
            im = re * y + im * x
            re = t^
        k >>= 1
        if k:
            var t = x * x - y * y
            y = 2 * x * y
            x = t^
    var scale = s * Int128(n)
    if not inverse:
        return (
            _round_ratio(re, Integer(1), rc, scale=scale, fail=fail_component == 0),
            _round_ratio(im, Integer(1), ic, scale=scale, fail=fail_component == 1),
        )
    # 1 / (re + im i) = (re - im i) / (re**2 + im**2).
    var norm = re * re + im * im
    return (
        _round_ratio(re, norm, rc, scale=-scale, fail=fail_component == 0),
        _round_ratio(-im, norm, ic, scale=-scale, fail=fail_component == 1),
    )


def _pow_complex(
    value: _ComplexArgument,
    exponent: Integer,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *,
    fail_component: Int = -1,
    initial_work: Int = 0,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var target = _target(value, value, context)
    var rc = target.real()
    var ic = target.imag()
    var a = value.real.value
    var b = value.imag.value
    if exponent == 0:
        var negative_zero = False
        if a.kind != 0 or b.kind != 0:
            var order = 0
            if a.kind == 2 or b.kind == 2:
                order = 1
            elif a.kind != 3 and b.kind != 3:
                var difference = _ExactAccumulator()
                difference.add(a.numerator * a.numerator, 2 * a.scale)
                difference.add(b.numerator * b.numerator, 2 * b.scale)
                difference.add(Integer(-1), 0)
                var magnitude = difference.magnitude()
                order = (-1 if magnitude.negative else 1) if len(
                    magnitude.runs
                ) else 0
            negative_zero = (
                order < 0
                or (order == 0 and b.negative)
                or ic.rounding() == RoundingMode.toward_negative
            )
        return (
            _round_ratio(Integer(1), Integer(1), rc, fail=fail_component == 0),
            _float_special(
                0,
                negative_zero,
                0,
                ic,
                fail_component == 1,
            ),
        )
    var inverse = exponent.sign() < 0
    if a.kind == 2 or b.kind == 2 or (a.kind == 0 and b.kind == 0):
        var singular = (a.kind == 0 and b.kind == 0) == inverse
        var flags = 8 if inverse and a.kind == 0 and b.kind == 0 else 0
        return (
            _float_special(
                2 if singular else 0, False, flags, rc, fail_component == 0
            ),
            _float_special(
                3 if singular else 0, False, flags, ic, fail_component == 1
            ),
        )
    if a.kind == 3 or b.kind == 3:
        return (
            _float_special(3, False, 0, rc, fail_component == 0),
            _float_special(3, False, 0, ic, fail_component == 1),
        )
    var count = abs(exponent)
    var phase = Int(count._word(0) & 7)
    var axis = a.kind == 0 or b.kind == 0
    var am = _Bound(a.numerator, _wide_integer(a.scale), False)
    var bm = _Bound(b.numerator, _wide_integer(b.scale), False)
    var diagonal = not axis and _order(am, bm) == 0
    var real_zero = False
    var imag_zero = False
    var real_negative = False
    var imag_negative = False
    if b.kind == 0:
        real_negative = a.negative and Bool(phase & 1)
        imag_zero = True
        var unit_order = _order(am, _Bound(Integer(1), Integer(0), False))
        if a.negative:
            imag_negative = (
                real_negative if unit_order
                > 0 else (not real_negative) if unit_order
                < 0 else True
            )
        else:
            imag_negative = False if unit_order > 0 else (
                b.negative != inverse
            ) or (
                unit_order == 0
                and ic.rounding() == RoundingMode.toward_negative
            )
    elif a.kind == 0:
        real_zero = Bool(phase & 1)
        imag_zero = not real_zero
        real_negative = (a.negative != Bool(phase & 2)) if real_zero else Bool(
            phase & 2
        )
        # MPC fixes the sign of a real power of an imaginary number from the
        # exponent mod 4, starting from the computed zero: +0, except an exact
        # power with a negative exponent, which computes -0. Only a power of
        # two has an exact negative power.
        imag_negative = (
            ((a.negative != b.negative) != (not Bool(phase & 2)))
            != (inverse and _is_power_of_two(b.numerator))
        ) if imag_zero else ((b.negative != Bool(phase & 2)) != inverse)
    elif diagonal:
        var angle = (3 if a.negative else 1) * (-1 if b.negative else 1)
        var turn = ((-phase if inverse else phase) * angle) & 7
        real_zero = turn == 2 or turn == 6
        imag_zero = turn == 0 or turn == 4
        real_negative = turn > 2 and turn < 6
        imag_negative = turn > 4
        if imag_zero:
            # An inexact real power gives +0, as MPC's exp(y log x) does; exact
            # ones follow the exponent's phase, which can differ from MPC's
            # squaring sequence beyond the eighth power.
            imag_negative = (
                (a.negative != b.negative) != (phase == 0)
            ) != inverse if not inverse or _is_power_of_two(a.numerator) else False
    if rc.format()._is_exact() or ic.format()._is_exact():
        return _exact_pow_complex(
            a, b, exponent, count, inverse, axis, diagonal,
            real_zero, imag_zero, real_negative, imag_negative, rc, ic, fail_component,
        )
    var work = max(rc.format().precision(), ic.format().precision())
    if Int128(work) + Int128(count.magnitude_bit_length()) + 32 > Int128(
        FloatFormat.MAX_PRECISION
    ):
        raise Error(
            "Cannot certify Complex power within addressable precision; choose"
            " a smaller output precision or exponent. The destination is"
            " unchanged."
        )
    work += count.magnitude_bit_length() + 32
    if initial_work:
        work = initial_work
    var quiet_real = ArithmeticContext(
        format=rc.format(), rounding=rc.rounding()
    )
    var quiet_imag = ArithmeticContext(
        format=ic.format(), rounding=ic.rounding()
    )
    while work <= FloatFormat.MAX_PRECISION:
        var x = _Interval.exact(
            -a.numerator if a.negative else a.numerator, am.scale
        )
        var y = _Interval.exact(
            -b.numerator if b.negative else b.numerator, bm.scale
        )
        if axis or diagonal:
            x = _Interval.exact(
                b.numerator if a.kind == 0 else a.numerator,
                bm.scale if a.kind == 0 else am.scale,
            )
            y = _Interval.exact(Integer(0), Integer(0))
        var result = _power_intervals(x, y, count, work)
        x = result[0]
        y = result[1]
        var ready = True
        if inverse:
            if axis or diagonal:
                ready = x.lo.coefficient.sign() > 0
                if ready:
                    x = x.reciprocal(work)
            else:
                var norm = x.squared(work).plus(y.squared(work), work)
                ready = norm.lo.coefficient.sign() > 0
                if ready:
                    var reciprocal = norm.reciprocal(work)
                    x = x.times(reciprocal, work)
                    y = y.negated().times(reciprocal, work)
        if ready:
            if diagonal:
                x = x.scaled(exponent // 2)
            if axis or diagonal:
                y = x.negated() if imag_negative else x
                if real_negative:
                    x = x.negated()
            var rlo = _float_special(
                0, real_negative, 0, quiet_real, False
            ) if real_zero else x.lo.rounded(quiet_real, False)
            var rhi = rlo if real_zero else x.hi.rounded(quiet_real, True)
            var ilo = _float_special(
                0, imag_negative, 0, quiet_imag, False
            ) if imag_zero else y.lo.rounded(quiet_imag, False)
            var ihi = ilo if imag_zero else y.hi.rounded(quiet_imag, True)
            if _same_round(rlo, rhi) and _same_round(ilo, ihi):
                return (
                    _finish_round(rlo, rc, fail_component == 0),
                    _finish_round(ilo, ic, fail_component == 1),
                )
        if work > FloatFormat.MAX_PRECISION // 2:
            break
        work *= 2
    raise Error(
        "Cannot certify Complex power within addressable precision; choose a"
        " smaller output precision or exponent. The destination is"
        " unchanged."
    )
