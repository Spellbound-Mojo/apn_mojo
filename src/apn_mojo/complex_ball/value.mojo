"""Complex balls: a rectangle of two real balls."""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..exact_complex.value import ExactComplex
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._arithmetic import _sum, _product, _quotient, _square, _negate, _working, _from_record, _norm, _reaches_zero
from ..ball._radius import _Radius
from ..float.context import ArithmeticContext, FloatFormat
from ..complex.context import ComplexContext, _ComplexContextArgument
from ..complex._input import _ComplexArgument
from ..complex._arithmetic import _complex_operation
from ..common._traits import _BatchElement
from ..common._stable_hash import _StableHash
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..ball._json import _format_complex_ball_json, _read_complex_ball_json


struct ComplexBall(ImplicitlyCopyable, Writable, _BatchElement):
    """A complex ball: the rectangle `real x imag` of two real balls.

    A complex ball function returns a ball that contains the exact result for
    every point of its input rectangle. On a branch cut, a point takes the
    value of counter-clockwise continuity: the side
    a counter-clockwise circuit around the cut's finite branch point leaves.
    A rectangle that crosses a cut gets a result covering both sides. A
    function undefined somewhere in its input (`log 0`, division by a
    rectangle containing 0) is indeterminate.

    | Operation | Contract |
    |---|---|
    | `z + w`, `z - w`, `z * w`, `z / w`, `-z` | Encloses every result |
    | Division by a rectangle containing 0 | Indeterminate |

    Limitations:
        A complex ball has no signed zero and no comparison operators; there
        is no implicit conversion into it.
    """

    var _real: Ball
    var _imag: Ball

    def __init__(out self, *, _real: Ball, _imag: Ball):
        self._real = _real
        self._imag = _imag

    def __init__(out self, real: _BallArgument, imag: _BallArgument = 0) raises:
        """The rectangle of two real balls, or of exact numbers.

        Args:
            real: The real part.
            imag: The imaginary part.

        Raises:
            Only on a checked size error.
        """
        self._real = real.ball()
        self._imag = imag.ball()

    def __init__(out self, value: Complex, *, precision: Optional[Int] = None) raises:
        """The exact ball of a Complex, or its parts rounded to `precision`.

        Args:
            value: The Complex; infinite or NaN parts raise.
            precision: The midpoint precision, in bits.

        Raises:
            For an infinite or NaN part.
        """
        self._real = Ball(value.real(), precision=precision)
        self._imag = Ball(value.imag(), precision=precision)

    def __init__(out self, value: ExactComplex, *, precision: Optional[Int] = None) raises:
        """The ball of an ExactComplex: exact for binary fractions, otherwise
        rounded with a radius covering the error.

        Args:
            value: The ExactComplex.
            precision: The midpoint precision, in bits.

        Raises:
            Only on a checked size error.
        """
        self._real = Ball(value.real(), precision=precision)
        self._imag = Ball(value.imag(), precision=precision)

    def __init__(out self, *, copy: Self):
        self._real = copy._real
        self._imag = copy._imag

    @staticmethod
    def _placeholder() -> Self:
        return Self(_real=Ball._placeholder(), _imag=Ball._placeholder())

    def real(self) -> Ball:
        """The real part.

        Returns:
            The real ball.
        """
        return self._real

    def imag(self) -> Ball:
        """The imaginary part.

        Returns:
            The imaginary ball.
        """
        return self._imag

    def precision(self) -> Int:
        """The larger midpoint precision of the two parts.

        Returns:
            The precision in bits.
        """
        return max(self._real.precision(), self._imag.precision())

    def is_exact(self) -> Bool:
        """Whether both parts are exact.

        Returns:
            True for an exact ball.
        """
        return self._real.is_exact() and self._imag.is_exact()

    def is_finite(self) -> Bool:
        """Whether both parts are finite.

        Returns:
            True for a bounded rectangle.
        """
        return self._real.is_finite() and self._imag.is_finite()

    def is_indeterminate(self) -> Bool:
        """Whether either part is indeterminate.

        Returns:
            True for a ball without information.
        """
        return self._real.is_indeterminate() or self._imag.is_indeterminate()

    def same_representation(self, other: Self) -> Bool:
        """Whether both parts have the same representation.

        Args:
            other: The other ball.

        Returns:
            True when the midpoints, radii and kinds agree.
        """
        return self._real.same_representation(other._real) and self._imag.same_representation(other._imag)

    def _hash_into(self, mut hash: _StableHash) raises:
        self._real._hash_into(hash)
        self._imag._hash_into(hash)

    def __neg__(self) raises -> Self:
        return Self(_real=_negate(self._real), _imag=_negate(self._imag))

    def __add__(self, rhs: Self) raises -> Self:
        return _add(self, rhs, None)

    def __sub__(self, rhs: Self) raises -> Self:
        return _sub(self, rhs, None)

    def __mul__(self, rhs: Self) raises -> Self:
        return _mul(self, rhs, None)

    def __truediv__(self, rhs: Self) raises -> Self:
        return _div(self, rhs, None)

    def to_json(self, *, limits: Optional[ConversionLimits] = None) raises -> String:
        """Write the version-1 JSON record: two complete Ball records.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        return _format_complex_ball_json(self._real._record(), self._imag._record(), budget)

    @staticmethod
    def from_json(text: String, *, limits: Optional[ConversionLimits] = None) raises -> Self:
        """Read a ComplexBall from its version-1 JSON record, without rounding.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The ComplexBall the record holds.

        Raises:
            When the text is not exactly that schema in canonical form.
        """
        var budget = _ConversionBudget(limits)
        var parts = _read_complex_ball_json(text, budget)
        return Self(_real=Ball._from_record(parts[0]), _imag=Ball._from_record(parts[1]))

    def write_to(self, mut writer: Some[Writer]):
        """Write `ComplexBall(real, imag)` with both balls in midpoint-radius notation.

        Args:
            writer: The destination.
        """
        writer.write("ComplexBall(", self._real, ", ", self._imag, ")")


def _context(a: ComplexBall, context: Optional[BallContext], b: Int = 0) -> Optional[BallContext]:
    if context:
        return context
    try:
        return Optional[BallContext](BallContext(max(a.precision(), b)))
    except:
        return None


def _add(a: ComplexBall, b: ComplexBall, context: Optional[BallContext]) raises -> ComplexBall:
    var c = _context(a, context, b.precision())
    return ComplexBall(_real=_sum(_BallArgument(a._real), _BallArgument(b._real), False, c), _imag=_sum(_BallArgument(a._imag), _BallArgument(b._imag), False, c))


def _sub(a: ComplexBall, b: ComplexBall, context: Optional[BallContext]) raises -> ComplexBall:
    var c = _context(a, context, b.precision())
    return ComplexBall(_real=_sum(_BallArgument(a._real), _BallArgument(b._real), True, c), _imag=_sum(_BallArgument(a._imag), _BallArgument(b._imag), True, c))


def _product_of_finite(a: ComplexBall, b: ComplexBall, bits: Int) raises -> ComplexBall:
    """The product of two finite rectangles: the Complex product of the
    midpoints, each part rounded once, and each part's radius in radius
    arithmetic. For `(x + iy)(u + iv)` the real part moves by at most
    `|x| r_u + r_x |u| + r_x r_u + |y| r_v + r_y |v| + r_y r_v`, taken as
    `(|x| + r_x) r_u + r_x |u| + (|y| + r_y) r_v + r_y |v|`, and the imaginary
    part by the same with u and v exchanged."""
    var target = ArithmeticContext(format=FloatFormat(bits))
    var parts = _complex_operation(
        _ComplexArgument(real=a._real._midpoint, imag=a._imag._midpoint),
        _ComplexArgument(real=b._real._midpoint, imag=b._imag._midpoint),
        2,
        _ComplexContextArgument(ComplexContext(real=target, imag=target)),
    )
    var rx = a._real._radius
    var ry = a._imag._radius
    var ru = b._real._radius
    var rv = b._imag._radius
    var real = _Radius.zero()
    var imag = _Radius.zero()
    if not (rx.is_zero() and ry.is_zero() and ru.is_zero() and rv.is_zero()):
        var x = _Radius.upper(a._real._midpoint).add(rx)
        var y = _Radius.upper(a._imag._midpoint).add(ry)
        var u = _Radius.upper(b._real._midpoint)
        var v = _Radius.upper(b._imag._midpoint)
        real = x.multiply(ru).add(rx.multiply(u)).add(y.multiply(rv)).add(ry.multiply(v))
        imag = x.multiply(rv).add(rx.multiply(v)).add(y.multiply(ru)).add(ry.multiply(u))
    return ComplexBall(_real=_from_record(parts[0], real, bits), _imag=_from_record(parts[1], imag, bits))


def _quotient_of_short(a: ComplexBall, b: ComplexBall, bits: Int) raises -> Optional[ComplexBall]:
    """`a / b` for finite rectangles of at most 63 bits, `b` away from 0: the
    Complex quotient of the midpoints, each part rounded once on the Complex
    family's native path, widened by
    `(R_a |m_b| + |m_a| R_b) / (|m_b| (|m_b| - R_b))` for `R = r_x + r_y`, a
    bound of `|a/b - m_a/m_b| = |(a - m_a) m_b - m_a (b - m_b)| / (|b| |m_b|)`.
    None where `|m_b| <= R_b` may hold. Wider parts keep the composition: past
    63 bits the exact Complex quotient is slower than it (24% at 1024 bits),
    and as tight for a narrow divisor."""
    ref u = b._real._midpoint
    ref v = b._imag._midpoint
    if bits > 63 or a.precision() > 63 or b.precision() > 63 or (u.is_zero() and v.is_zero()):
        return None
    var ra = a._real._radius.add(a._imag._radius)
    var rb = b._real._radius.add(b._imag._radius)
    var bound = _Radius.zero()
    if not (ra.is_zero() and rb.is_zero()):
        var lu = _Radius.lower(u)
        var lv = _Radius.lower(v)
        var low = lu.multiply_down(lu).add_down(lv.multiply_down(lv)).sqrt_down()
        var gap = low.sub_down(rb)
        if gap.is_zero():
            return None
        var high_a = _Radius.upper(a._real._midpoint).add(_Radius.upper(a._imag._midpoint))
        var high_b = _Radius.upper(u).add(_Radius.upper(v))
        bound = ra.multiply(high_b).add(high_a.multiply(rb)).divide(low.multiply_down(gap))
    var target = ArithmeticContext(format=FloatFormat(bits))
    var parts = _complex_operation(
        _ComplexArgument(real=a._real._midpoint, imag=a._imag._midpoint),
        _ComplexArgument(real=u, imag=v),
        3,
        _ComplexContextArgument(ComplexContext(real=target, imag=target)),
    )
    return ComplexBall(_real=_from_record(parts[0], bound, bits), _imag=_from_record(parts[1], bound, bits))


def _mul(a: ComplexBall, b: ComplexBall, context: Optional[BallContext]) raises -> ComplexBall:
    var c = _context(a, context, b.precision())
    if a.is_finite() and b.is_finite():
        return _product_of_finite(a, b, _working(c, a.precision(), b.precision()))
    var ac = _product(_BallArgument(a._real), _BallArgument(b._real), c)
    var bd = _product(_BallArgument(a._imag), _BallArgument(b._imag), c)
    var ad = _product(_BallArgument(a._real), _BallArgument(b._imag), c)
    var bc = _product(_BallArgument(a._imag), _BallArgument(b._real), c)
    return ComplexBall(_real=_sum(_BallArgument(ac), _BallArgument(bd), True, c), _imag=_sum(_BallArgument(ad), _BallArgument(bc), False, c))


def _div(a: ComplexBall, b: ComplexBall, context: Optional[BallContext]) raises -> ComplexBall:
    """`a / b`: for finite rectangles of at most 63 bits with `b` away from
    0, from the Complex quotient of the midpoints; otherwise
    `a conj(b) / |b|**2`, its norm from b's magnitude bounds where the
    squares' norm seems to reach 0; indeterminate when the rectangle `b` may
    hold 0."""
    var c = _context(a, context, b.precision())
    if a.is_finite() and b.is_finite():
        var short = _quotient_of_short(a, b, _working(c, a.precision(), b.precision()))
        if short:
            return short.value()
    var norm = _sum(_BallArgument(_square(_BallArgument(b._real), c)), _BallArgument(_square(_BallArgument(b._imag), c)), False, c)
    if b.is_finite() and norm.is_finite() and _reaches_zero(_BallArgument(norm)):
        # The squares reached below 0 or the sum's radius passed a small
        # term: the magnitude bounds decide whether b may be 0.
        norm = _norm(b._real, b._imag, _working(c, b.precision()))
    var numerator = _mul(a, ComplexBall(_real=b._real, _imag=_negate(b._imag)), c)
    return ComplexBall(
        _real=_quotient(_BallArgument(numerator._real), _BallArgument(norm), c),
        _imag=_quotient(_BallArgument(numerator._imag), _BallArgument(norm), c),
    )
