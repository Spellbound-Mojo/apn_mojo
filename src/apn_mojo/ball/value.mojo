"""Balls: a Float midpoint and a radius that enclose every value they stand for."""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float._arithmetic import _FloatArgument, _float_argument, _float_operation
from ..float._rounding import _round_exact_binary
from ..float._input import _FloatInput
from ..common._traits import _BatchElement, _MapAdapter, _MapArgument
from ..common._stable_hash import _StableHash
from ..integer._word_math import _trailing_zero_bits
from .context import BallContext
from ._radius import _Radius
from ._arithmetic import _sum, _product, _quotient, _negate, _from_record
from ._json import _BallRecord, _format_ball_json, _read_ball_json
from ..common.conversion import ConversionLimits, _ConversionBudget

comptime _FINITE = 0
comptime _UNBOUNDED = 1
comptime _INDETERMINATE = 2


struct BallOrder(Equatable, ImplicitlyCopyable, Writable):
    """How two balls compare: one of five named constants.

    `less` and `greater` hold for every pair of points of the two balls,
    `equal` only for two exact, equal balls, `overlap` when no order is
    certain, and `undefined` when either ball is indeterminate.
    """

    var _code: Int

    def __init__(out self, *, _code: Int):
        self._code = _code

    comptime less = Self(_code=0)
    comptime equal = Self(_code=1)
    comptime greater = Self(_code=2)
    comptime overlap = Self(_code=3)
    comptime undefined = Self(_code=4)

    def __eq__(self, other: Self) -> Bool:
        return self._code == other._code

    def __ne__(self, other: Self) -> Bool:
        return self._code != other._code

    def write_to(self, mut writer: Some[Writer]):
        """Write the name, as `print` does.

        Args:
            writer: The destination.
        """
        var names: List[StaticString] = ["less", "equal", "greater", "overlap", "undefined"]
        writer.write("BallOrder.", names[self._code])


struct Ball(ImplicitlyCopyable, Writable, _BatchElement):
    """A real ball `[m +/- r]`: every real number within `r` of `m`.

    A ball function returns a ball that contains the exact result for every
    point of its input balls; it may be wider than necessary, but never
    excludes the true value. A ball is finite, unbounded (any real number), or
    indeterminate (no information, as where a function is undefined). The
    midpoint is a Float of any precision; the radius has 30 bits and is
    rounded up.

    | Operation | Contract |
    |---|---|
    | `x + y`, `x - y`, `x * y`, `x / y`, `-x` | Encloses every result; Integer, Rational and Float operands are exact |
    | Division by a ball containing 0 | Indeterminate |
    | Comparisons | `compare` and the `certainly_*` predicates; no `<` or `==` |

    Limitations:
        A ball has no signed zero. There are no comparison operators, since a
        three-valued comparison must not pass for a Bool.
    """

    var _midpoint: Float
    var _radius: _Radius
    var _kind: Int

    def __init__(out self, *, _midpoint: Float, _radius: _Radius, _kind: Int):
        self._midpoint = _midpoint
        self._radius = _radius
        self._kind = _kind

    @staticmethod
    def _placeholder() -> Self:
        return Self(_midpoint=Float._placeholder(), _radius=_Radius.zero(), _kind=_FINITE)

    def __init__(out self, value: _FloatArgument, *, precision: Optional[Int] = None) raises:
        """The ball of an exact number.

        An Integer, or a Rational that is a binary fraction, gives an exact
        ball with at least 128 bits, so arithmetic on small exact balls stays
        exact. A Float gives an exact ball in its own precision. Another
        Rational is rounded to 128 bits, with a radius covering the error. With
        `precision`, the number is rounded to that many bits instead.

        Args:
            value: An Integer, Rational, Float or native number.
            precision: The midpoint precision, in bits.

        Raises:
            For infinity and NaN; use `Ball.unbounded()` or
            `Ball.indeterminate()`.
        """
        if value.value.kind >= 2:
            raise Error(
                "Cannot make a finite Ball from infinity or NaN; use"
                " Ball.unbounded() or Ball.indeterminate()."
            )
        var bits = 128
        if precision:
            bits = precision.value()
        elif value.format:
            bits = value.format.value().precision()
        elif value.native_precision:
            bits = value.native_precision
        elif _is_power_of_two(value.value.denominator):
            bits = max(128, value.value.numerator.magnitude_bit_length())
        self = _rounded(value, _Radius.zero(), bits)

    def __init__(out self, midpoint: _FloatArgument, radius: _FloatArgument, *, precision: Optional[Int] = None) raises:
        """The ball `[midpoint +/- radius]`.

        The radius rounds up to 30 bits; a midpoint that is not exact at the
        precision is rounded, and its error added to the radius.

        Args:
            midpoint: The center, an exact number.
            radius: A nonnegative exact number; infinity gives an unbounded
                ball.
            precision: The midpoint precision, in bits; by default as for
                `Ball(midpoint)`.

        Raises:
            For a negative or NaN radius, or an infinite or NaN midpoint.
        """
        if radius.value.kind == 3 or (radius.value.negative and radius.value.kind != 0):
            raise Error(
                "Cannot make a Ball with a negative or NaN radius; use a"
                " nonnegative radius."
            )
        var center = Ball(midpoint, precision=precision)
        if radius.value.kind == 2:
            self = Ball.unbounded(center.precision())
            return
        self = Ball(
            _midpoint=center._midpoint,
            _radius=center._radius.add(_Radius.upper_input(radius.value)),
            _kind=_FINITE,
        )

    def __init__(out self, text: String, *, precision: Optional[Int] = None) raises:
        """Parse a ball: a number, or the midpoint-radius form `[m +/- r]`.

        A decimal or fraction (`"3.14"`, `"1/3"`) is exact and rounds once to
        the precision, 128 bits by default, with a radius covering the error;
        hexadecimal and binary text (`"0x1.8p0"`) are exact binary fractions.
        In `[m +/- r]`, `m` rounds to nearest and `r` rounds up. `[+/- inf]` is
        unbounded and `[nan +/- inf]` indeterminate.

        Args:
            text: The text.
            precision: The midpoint precision, in bits; 128 by default.

        Raises:
            When the text is not a number or a ball.
        """
        var bits = precision.value() if precision else 128
        var source = String(text.strip())
        if source.startswith("[") and source.endswith("]"):
            var inner = String(source.removeprefix("[").removesuffix("]"))
            var parts = inner.split("+/-")
            if len(parts) != 2:
                raise Error(String("Cannot parse Ball from '", text, "'; write [m +/- r]."))
            var center = String(parts[0].strip())
            var radius = String(parts[1].strip())
            if radius == "inf":
                if center == "nan":
                    self = Ball.indeterminate(bits)
                    return
                self = Ball.unbounded(bits)
                return
            self = Ball(_number(center if center.byte_length() else "0"), _number(radius), precision=bits)
            return
        self = Ball(_number(source), precision=bits)

    @staticmethod
    def from_interval(low: _FloatArgument, high: _FloatArgument, *, precision: Optional[Int] = None) raises -> Self:
        """The least ball at the precision that contains `[low, high]`.

        Args:
            low: The lower end, an exact number.
            high: The upper end, at least `low`.
            precision: The midpoint precision, in bits; 128 by default.

        Returns:
            A ball with midpoint `(low + high) / 2` rounded to the precision.

        Raises:
            When `low > high`, or an end is infinite or NaN.
        """
        if low.value.kind >= 2 or high.value.kind >= 2:
            raise Error("Cannot make a Ball from an infinite or NaN interval end.")
        var bits = precision.value() if precision else 128
        var lo = _rational_of(low.value)
        var hi = _rational_of(high.value)
        if lo > hi:
            raise Error("Cannot make a Ball from an interval with low > high; swap the ends.")
        var middle = Float((lo + hi) / 2, context=ArithmeticContext(format=FloatFormat(bits)))
        var center = middle.to_rational_exact()
        # The radius covers both ends from the rounded midpoint.
        var reach = hi - center if hi - center >= center - lo else center - lo
        return Ball(_midpoint=middle, _radius=_Radius.upper_input(_float_argument(reach).value), _kind=_FINITE)

    @staticmethod
    def unbounded(precision: Int = 128) raises -> Self:
        """The ball of every real number.

        Args:
            precision: The midpoint precision, in bits.

        Returns:
            An unbounded ball with midpoint 0.

        Raises:
            When the precision is invalid.
        """
        return Ball(_midpoint=Float.zero(context=ArithmeticContext(format=FloatFormat(precision))), _radius=_Radius.infinity(), _kind=_UNBOUNDED)

    @staticmethod
    def indeterminate(precision: Int = 128) raises -> Self:
        """The ball that carries no information, as where a function is undefined.

        Args:
            precision: The midpoint precision, in bits.

        Returns:
            An indeterminate ball, with a NaN midpoint.

        Raises:
            When the precision is invalid.
        """
        return Ball(_midpoint=Float.nan(context=ArithmeticContext(format=FloatFormat(precision))), _radius=_Radius.infinity(), _kind=_INDETERMINATE)

    def midpoint(self) -> Float:
        """The midpoint; NaN for an indeterminate ball.

        Returns:
            The midpoint.
        """
        return self._midpoint

    def radius(self) raises -> Float:
        """The radius, exactly, as a 30-bit Float; infinity unless finite.

        Returns:
            The radius.

        Raises:
            Only on a checked size error.
        """
        return self._radius.to_float()

    def precision(self) -> Int:
        """The midpoint precision.

        Returns:
            The precision in bits.
        """
        return self._midpoint.precision()

    def is_finite(self) -> Bool:
        """Whether the ball is finite.

        Returns:
            True unless unbounded or indeterminate.
        """
        return self._kind == _FINITE

    def is_exact(self) -> Bool:
        """Whether the ball is finite with radius 0.

        Returns:
            True for an exact ball.
        """
        return self._kind == _FINITE and self._radius.is_zero()

    def is_unbounded(self) -> Bool:
        """Whether the ball is unbounded.

        Returns:
            True for the ball of every real number.
        """
        return self._kind == _UNBOUNDED

    def is_indeterminate(self) -> Bool:
        """Whether the ball is indeterminate.

        Returns:
            True for the ball without information.
        """
        return self._kind == _INDETERMINATE

    def _require_finite(self) raises:
        if self._kind != _FINITE:
            raise Error(
                "This needs a finite ball; check is_finite() first. The"
                " destination is unchanged."
            )

    def _end(self, upper: Bool, context: ArithmeticContext) raises -> Float:
        return Float(_rounded=_float_operation(self._midpoint, self._radius.to_float(), 0 if upper else 1, context))

    def _exact_lower(self) raises -> Float:
        return self._end(False, _exact_context())

    def _exact_upper(self) raises -> Float:
        return self._end(True, _exact_context())

    def lower(self) raises -> Float:
        """The lower end, rounded down at the midpoint's precision.

        Returns:
            A Float at or below every point of the ball.

        Raises:
            Unless the ball is finite.
        """
        self._require_finite()
        return self._end(False, ArithmeticContext(format=FloatFormat(self.precision()), rounding=RoundingMode.toward_negative))

    def upper(self) raises -> Float:
        """The upper end, rounded up at the midpoint's precision.

        Returns:
            A Float at or above every point of the ball.

        Raises:
            Unless the ball is finite.
        """
        self._require_finite()
        return self._end(True, ArithmeticContext(format=FloatFormat(self.precision()), rounding=RoundingMode.toward_positive))

    def lower_rational(self) raises -> Rational:
        """The lower end, exactly.

        Returns:
            `midpoint - radius` as a Rational.

        Raises:
            Unless the ball is finite.
        """
        self._require_finite()
        return self._midpoint.to_rational_exact() - self._radius.to_float().to_rational_exact()

    def upper_rational(self) raises -> Rational:
        """The upper end, exactly.

        Returns:
            `midpoint + radius` as a Rational.

        Raises:
            Unless the ball is finite.
        """
        self._require_finite()
        return self._midpoint.to_rational_exact() + self._radius.to_float().to_rational_exact()

    def midpoint_rational(self) raises -> Rational:
        """The midpoint, exactly.

        Returns:
            The midpoint as a Rational.

        Raises:
            For an indeterminate ball.
        """
        if self._kind == _INDETERMINATE:
            raise Error("An indeterminate ball has no midpoint value; check is_indeterminate() first.")
        return self._midpoint.to_rational_exact()

    def magnitude_upper(self) raises -> Float:
        """An upper bound of `abs(x)` over the ball, as a 30-bit Float.

        Returns:
            The bound; infinity unless the ball is finite.

        Raises:
            Only on a checked size error.
        """
        if self._kind != _FINITE:
            return _Radius.infinity().to_float()
        return _Radius.upper(self._midpoint).add(self._radius).to_float()

    def magnitude_lower(self) raises -> Float:
        """A lower bound of `abs(x)` over the ball, as a 30-bit Float.

        Returns:
            The bound; 0 when the ball contains 0 or is not finite.

        Raises:
            Only on a checked size error.
        """
        if self._kind != _FINITE:
            return _Radius.zero().to_float()
        var gap = Float(_rounded=_float_operation(
            _abs(self._midpoint), self._radius.to_float(), 1,
            ArithmeticContext(format=FloatFormat(30), rounding=RoundingMode.toward_negative),
        ))
        if gap.signbit() or gap.is_zero():
            return _Radius.zero().to_float()
        return gap

    def relative_accuracy_bits(self) -> Int:
        """The relative accuracy, `-log2(radius / abs(midpoint))`, from the
        exponents and at most one bit pessimistic.

        Returns:
            `Int.MAX` for an exact ball, at most 0 when the radius reaches the
            midpoint, and `Int.MIN` unless the ball is finite.
        """
        if self._kind != _FINITE:
            return Int.MIN
        if self._radius.is_zero():
            return Int.MAX
        if self._midpoint.is_zero():
            return 0
        return self._midpoint._exponent - self._radius.exponent - 1

    def accuracy_bits(self) -> Int:
        """The absolute accuracy, `floor(-log2(radius))`.

        Returns:
            `Int.MAX` for an exact ball and `Int.MIN` unless the ball is finite.
        """
        if self._kind != _FINITE:
            return Int.MIN
        if self._radius.is_zero():
            return Int.MAX
        return -self._radius.exponent + Int(self._radius.mantissa == UInt64(1) << 29)

    def certainly_positive(self) raises -> Bool:
        """Whether every point is above 0.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        # m - r > 0, decided exactly by comparing the radius with |m|.
        return (
            self._kind == _FINITE and not self._midpoint._negative and not self._midpoint.is_zero()
            and not self._radius.covers_float(self._midpoint)
        )

    def certainly_negative(self) raises -> Bool:
        """Whether every point is below 0.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return self._kind == _FINITE and self._midpoint._negative and not self._radius.covers_float(self._midpoint)

    def certainly_nonnegative(self) raises -> Bool:
        """Whether every point is at least 0.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return self._kind == _FINITE and self._exact_lower() >= 0

    def certainly_nonpositive(self) raises -> Bool:
        """Whether every point is at most 0.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return self._kind == _FINITE and self._exact_upper() <= 0

    def certainly_nonzero(self) raises -> Bool:
        """Whether 0 is outside the ball; `not certainly_nonzero()` is the
        test for a possible zero.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return self.certainly_positive() or self.certainly_negative()

    def sign_if_certain(self) raises -> Optional[Int]:
        """The sign, when the whole ball has one.

        Returns:
            -1 or 1 when certain, 0 for the exact ball 0, otherwise None.

        Raises:
            Only on a checked size error.
        """
        if self.certainly_positive():
            return 1
        if self.certainly_negative():
            return -1
        if self.is_exact() and self._midpoint.is_zero():
            return 0
        return None

    def certainly_lt(self, other: _BallArgument) raises -> Bool:
        """Whether every point is below every point of `other`.

        Args:
            other: A ball or an exact number.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return _order(self, other.ball()) == BallOrder.less

    def certainly_le(self, other: _BallArgument) raises -> Bool:
        """Whether every point is at most every point of `other`.

        Args:
            other: A ball or an exact number.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        var b = other.ball()
        if self._kind != _FINITE or b._kind != _FINITE:
            return False
        return self._exact_upper() <= b._exact_lower()

    def certainly_gt(self, other: _BallArgument) raises -> Bool:
        """Whether every point is above every point of `other`.

        Args:
            other: A ball or an exact number.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        return _order(self, other.ball()) == BallOrder.greater

    def certainly_ge(self, other: _BallArgument) raises -> Bool:
        """Whether every point is at least every point of `other`.

        Args:
            other: A ball or an exact number.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        var b = other.ball()
        if self._kind != _FINITE or b._kind != _FINITE:
            return False
        return self._exact_lower() >= b._exact_upper()

    def certainly_eq(self, other: _BallArgument) raises -> Bool:
        """Whether both are the same exact value; an inexact ball is never
        certainly equal to anything.

        Args:
            other: A ball or an exact number.

        Returns:
            True only for two exact, equal balls.

        Raises:
            Only on a checked size error.
        """
        return _order(self, other.ball()) == BallOrder.equal

    def certainly_ne(self, other: _BallArgument) raises -> Bool:
        """Whether no point is shared with `other`.

        Args:
            other: A ball or an exact number.

        Returns:
            True only when it is certain.

        Raises:
            Only on a checked size error.
        """
        var order = _order(self, other.ball())
        return order == BallOrder.less or order == BallOrder.greater

    def same_representation(self, other: Self) -> Bool:
        """Whether kind, midpoint representation and radius all match.

        Args:
            other: The other ball.

        Returns:
            True when the representations match.
        """
        return self._kind == other._kind and self._midpoint.same_representation(other._midpoint) and self._radius.same(other._radius)

    def representation_cmp(self, other: Self) -> Int:
        """Compare representations in a strict total order: the kind, then the
        midpoint by `Float.representation_cmp`, then the radius.

        Args:
            other: The other ball.

        Returns:
            -1, 0 or 1; 0 exactly when `same_representation` holds.
        """
        if self._kind != other._kind:
            return -1 if self._kind < other._kind else 1
        var order = self._midpoint.representation_cmp(other._midpoint)
        return order if order else self._radius.compare(other._radius)

    def _hash_into(self, mut hash: _StableHash) raises:
        hash.word(UInt64(self._kind))
        self._midpoint._hash_into(hash)
        self._radius.to_float()._hash_into(hash)

    def __neg__(self) raises -> Self:
        return _negate(self)

    def __add__(self, other: _BallArgument) raises -> Self:
        return _sum(_BallArgument(self), other, False, None)

    def __radd__(self, other: _BallArgument) raises -> Self:
        return _sum(other, _BallArgument(self), False, None)

    def __sub__(self, other: _BallArgument) raises -> Self:
        return _sum(_BallArgument(self), other, True, None)

    def __rsub__(self, other: _BallArgument) raises -> Self:
        return _sum(other, _BallArgument(self), True, None)

    def __mul__(self, other: _BallArgument) raises -> Self:
        return _product(_BallArgument(self), other, None)

    def __rmul__(self, other: _BallArgument) raises -> Self:
        return _product(other, _BallArgument(self), None)

    def __truediv__(self, other: _BallArgument) raises -> Self:
        return _quotient(_BallArgument(self), other, None)

    def __rtruediv__(self, other: _BallArgument) raises -> Self:
        return _quotient(other, _BallArgument(self), None)

    def to_json(self, *, limits: Optional[ConversionLimits] = None) raises -> String:
        """Write the version-1 JSON record: the kind, and the midpoint and the
        radius as complete Float records. Reading it back gives the same
        representation.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        return _format_ball_json(self._record(), budget)

    @staticmethod
    def from_json(text: String, *, limits: Optional[ConversionLimits] = None) raises -> Self:
        """Read a Ball from its version-1 JSON record, without rounding.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The Ball the record holds.

        Raises:
            When the text is not exactly that schema in canonical form.
        """
        var budget = _ConversionBudget(limits)
        return Self._from_record(_read_ball_json(text, budget))

    def _record(self) raises -> _BallRecord:
        return _BallRecord(self._kind, self._midpoint._interchange_record(), self._radius.to_float()._interchange_record())

    @staticmethod
    def _from_record(record: _BallRecord) raises -> Self:
        var r = Float(_rounded=record.radius)
        var radius = _Radius.zero() if r.is_zero() else (_Radius.infinity() if r.is_infinite() else _Radius.upper(r))
        return Self(_midpoint=Float(_rounded=record.midpoint), _radius=radius, _kind=record.kind)

    def to_string(self, digits: Optional[Int] = None) raises -> String:
        """Write the ball in midpoint-radius form, `[midpoint +/- radius]`.

        Without `digits`, the midpoint has as many significant digits as the
        radius certifies; with them, that many. The printed radius has three
        significant digits, rounded up, and covers the rounding of the printed
        midpoint, so the printed ball contains this one. An exact ball whose
        midpoint prints exactly shows radius 0; `[+/- inf]` is unbounded and
        `[nan +/- inf]` indeterminate.

        Args:
            digits: The midpoint's significant digits, at least 1.

        Returns:
            The text.

        Raises:
            When `digits` is below 1.
        """
        if self._kind == _INDETERMINATE:
            return "[nan +/- inf]"
        if self._kind == _UNBOUNDED:
            return "[+/- inf]"
        var count: Int
        if digits:
            count = digits.value()
        elif self._radius.is_zero():
            count = max(1, Int(Float64(self.precision()) * 0.30103) + 2)
        else:
            count = max(1, Int(Float64(max(0, self.relative_accuracy_bits())) * 0.30103))
        var center = self._midpoint.to_string(10, digits=count)
        var error = abs(self._midpoint.to_rational_exact() - Rational(center)) + self._radius.to_float().to_rational_exact()
        if not error:
            return String("[", center, " +/- 0]")
        var bound = Float(error, context=ArithmeticContext(format=FloatFormat(30), rounding=RoundingMode.toward_positive))
        return String("[", center, " +/- ", bound.to_string(10, digits=3, rounding=RoundingMode.toward_positive), "]")

    def write_to(self, mut writer: Some[Writer]):
        """Write `to_string()`, as `print` does.

        Args:
            writer: The destination.
        """
        try:
            writer.write(self.to_string())
        except:
            writer.write("[? +/- ?]")


def _rational_of(x: _FloatInput) raises -> Rational:
    """An exact number as a Rational."""
    var value = Rational(x.numerator, x.denominator)
    if x.scale > 0:
        value = value * Rational(Integer(1) << Int(x.scale))
    elif x.scale < 0:
        value = value / Rational(Integer(1) << Int(-x.scale))
    return -value if x.negative else value


def _is_power_of_two(n: Integer) -> Bool:
    return _trailing_zero_bits(n) == n.magnitude_bit_length() - 1


def _abs(x: Float) raises -> Float:
    return -x if x.signbit() else x


def _number(text: String) raises -> _FloatArgument:
    """An exact number from text: hexadecimal and binary as binary fractions,
    anything else through Rational's exact decimal grammar."""
    var body = String(text.strip())
    var unsigned = String(body.removeprefix("-").removeprefix("+"))
    if unsigned.startswith("0x") or unsigned.startswith("0X") or unsigned.startswith("0b") or unsigned.startswith("0B"):
        # Each digit carries at most four bits, so this precision holds it exactly.
        return Float(body, context=ArithmeticContext(format=FloatFormat(4 * body.byte_length() + 8)))
    return Rational(body)


def _rounded(value: _FloatArgument, radius: _Radius, precision: Int) raises -> Ball:
    """The ball of an exact number rounded once to `precision` bits, with
    `radius` and the rounding error."""
    var context = ArithmeticContext(format=FloatFormat(precision))
    var a = value.value
    if a.kind == 1 and a.denominator._is_one():
        # A Float of this precision, or any value it holds exactly.
        var exact = _round_exact_binary(a.numerator, a.negative, a.scale, context, False)
        if exact.kind >= 0:
            return _from_record(exact, radius, precision)
    var record = _float_operation(value, Integer(0), 0, context)
    return _from_record(record, radius, precision)


def _order(a: Ball, b: Ball) raises -> BallOrder:
    """The three-valued order, decided on the exact ends."""
    if a._kind == _INDETERMINATE or b._kind == _INDETERMINATE:
        return BallOrder.undefined
    if a._kind == _UNBOUNDED or b._kind == _UNBOUNDED:
        return BallOrder.overlap
    if a.is_exact() and b.is_exact() and a._midpoint == b._midpoint:
        return BallOrder.equal
    if a._exact_upper() < b._exact_lower():
        return BallOrder.less
    if a._exact_lower() > b._exact_upper():
        return BallOrder.greater
    return BallOrder.overlap


struct _BallArgument(ImplicitlyCopyable, _MapAdapter, _MapArgument):
    """A ball operand: a Ball, or an exact number as an exact ball whose
    precision counts only for a Float."""

    var midpoint: _FloatArgument
    var radius: _Radius
    var kind: Int
    var precision: Int

    @staticmethod
    def _adapt[V: ImplicitlyCopyable & Deinitable](value: V) raises -> Self:
        comptime assert V == Ball or V == Integer or V == Rational or V == Float, (
            "A ball function takes Ball, Integer, Rational or Float values."
        )
        comptime if V == Ball:
            return Self(rebind[Ball](value))
        else:
            return Self(_number=_FloatArgument._adapt(value))

    @implicit
    def __init__(out self, value: Ball):
        self.midpoint = _FloatArgument(value._midpoint)
        self.radius = value._radius
        self.kind = value._kind
        self.precision = value.precision()

    @implicit
    def __init__(out self, value: Integer):
        self = Self(_number=_float_argument(value))

    @implicit
    def __init__(out self, value: Rational):
        self = Self(_number=_float_argument(value))

    @implicit
    def __init__(out self, value: Float):
        self = Self(_number=_float_argument(value))

    @implicit
    def __init__(out self, value: IntLiteral):
        self = Self(_number=_float_argument(Integer(value)))

    @implicit
    def __init__(out self, value: Int):
        self = Self(_number=_float_argument(Integer(value)))

    def __init__(out self, *, _number: _FloatArgument):
        self.midpoint = _number
        self.radius = _Radius.zero()
        self.kind = _FINITE
        if _number.value.kind == 2:
            self.kind = _UNBOUNDED
        elif _number.value.kind == 3:
            self.kind = _INDETERMINATE
        self.precision = _number.format.value().precision() if _number.format else _number.native_precision

    def ball(self) raises -> Ball:
        """This operand as a ball."""
        var bits = self.precision if self.precision else 128
        if self.kind == _INDETERMINATE:
            return Ball.indeterminate(bits)
        if self.kind == _UNBOUNDED:
            return Ball.unbounded(bits)
        if self.radius.is_zero():
            return Ball(self.midpoint)
        # A ball operand's midpoint is a Float of this precision: exact.
        var midpoint = Float(_rounded=_float_operation(self.midpoint, Integer(0), 0, ArithmeticContext(format=FloatFormat(bits))))
        return Ball(_midpoint=midpoint, _radius=self.radius, _kind=_FINITE)
