"""Scalar binary Float values with explicit numerical choices and exact comparisons."""

from std.builtin.bool import Boolable
from std.math import Absable
from std.sys import bit_width_of
from ..integer.value import Integer
from ..rational.value import Rational
from .context import ArithmeticContext, FloatFormat, RoundingMode
from .status import NumericStatus
from ._rounding import (
    _RoundedBinary,
    _round_ratio,
    _compare_scaled,
    _shifted_word,
    _finish_round,
    _exponent_add,
)
from ._multiplication import _binary_product
from ._format import _merge_float_formats
from ._input import _FloatInput, _float_input
from ._comparison import _FloatComparison, _float_relation
from ..complex._comparison import _ComplexComparison
from ..complex.value import Complex
from ._arithmetic import (
    _FloatSource,
    _FloatArgument,
    _float_operation,
    _binary_float_sum,
    _binary_quotient,
)
from ._batch_protocol import _FloatBatchArithmetic, _FloatBatchOperand
from ..batch.mask import Mask
from ._functions import _pow_float, _float_integer
from ._native import _float_native
from ..common.conversion import ConversionLimits, _ConversionBudget
from ._text import _format_float_exact, _write_float_exact
from ._json import _format_float_json, _read_float_json
from ._parse import _parse_float
from ._decimal import _format_float_decimal
from .shortest import _format_float_shortest
from ..common._traits import _BatchElement
from ..common._stable_hash import _StableHash
from ..integer._word_math import _trailing_zero_bits


struct Float(
    Absable,
    Boolable,
    Equatable,
    ImplicitlyCopyable,
    Writable,
    _FloatComparison,
    _FloatSource,
    _BatchElement,
):
    """A binary floating-point number of any precision.

    A finite nonzero value is `(-1)**s * m * 2**(e - p)`: a significand `m` of
    exactly `p` bits and an exponent `e` within its format's bounds. Every
    operation computes the exact result and rounds it once. Zeros and infinities are
    signed; NaN is canonical and signless.

    | Operation | Contract |
    |---|---|
    | `x + y`, `x - y`, `x * y`, `x / y` | One nearest-even rounding of the exact result |
    | `x ** n` | Signed integral power, nearest-even, in `x`'s format |
    | `+x`, `-x`, `abs(x)` | Exact, same format |
    | `x += y` and the other compound forms | Round into `x`'s format; `x` unchanged on error |
    | `==`, `!=`, `<`, `<=`, `>`, `>=` | Exact comparison of the stored values; NaN is unordered |
    | `Bool(x)` | False only for either zero |

    Without a context, operands merge their formats: exact operands contribute no
    precision, a native float its precision (Float64: 53 bits) and a Float its
    precision and exponent bounds, which must match. Quiet NaN propagates without
    signaling; `0/0`, `inf/inf`, `0 * inf` and `inf - inf` are invalid; a finite
    nonzero value over zero is a signed infinity with divide-by-zero. An exact
    cancellation is `+0`, or `-0` when rounding toward negative.

    Printing shows exact hexadecimal, such as `0x3p0` for 3 and `0x1p-1` for one
    half, without the format; JSON keeps both value and format.

    Limitations:
        Bare decimal literals are rejected: write `Float("0.1")` for the exact
        decimal or `Float(Float64(0.1))` for the binary64 value. A native number
        cannot be the left operand of a comparison. Float is not hashable; use
        `FloatKey` for dictionary keys. `Int(x)` and `Float64(x)` are
        unavailable; use the named conversions.
    """

    var _significand: Integer
    var _format: FloatFormat
    var _exponent: Int
    var _kind: Int
    var _negative: Bool

    def __add__[B: _FloatBatchArithmetic](self, rhs: B) raises -> B.FloatBatch:
        return rhs._float_reverse(_FloatArgument(self), 0)

    def __sub__[B: _FloatBatchArithmetic](self, rhs: B) raises -> B.FloatBatch:
        return rhs._float_reverse(_FloatArgument(self), 1)

    def __mul__[B: _FloatBatchArithmetic](self, rhs: B) raises -> B.FloatBatch:
        return rhs._float_reverse(_FloatArgument(self), 2)

    def __truediv__[B: _FloatBatchArithmetic](self, rhs: B) raises -> B.FloatBatch:
        return rhs._float_reverse(_FloatArgument(self), 3)

    def __eq__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 0)

    def __ne__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 1)

    def __lt__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 4)

    def __le__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 5)

    def __gt__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 2)

    def __ge__[B: _FloatBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_float(_FloatArgument(self), 3)

    def __init__(
        out self, *, context: Optional[ArithmeticContext] = None
    ) raises:
        """Positive zero; in the context's format, or 128 bits by default.

        Args:
            context: The format of the zero.

        Raises:
            Only on an invalid context.
        """
        self = Self(Int(0), context=context)

    def __init__[
        T: Copyable
    ](
        out self,
        value: T,
        *,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        """Round an exact number to a Float.

        An Integer, Rational, Float or Complex-free exact value is rounded once,
        directly to the format. A Float keeps its own format unless a context is
        given.

        Parameters:
            T: The source type.

        Args:
            value: The value to convert.
            context: The output format, rounding mode and traps; by default 128 bits,
                or the Float's own format.

        Raises:
            When a trapped condition occurs or the source type is not numeric.
        """
        var result: _RoundedBinary
        comptime if T == Self:
            var source = rebind[Self](value)
            var target = context.value() if context else ArithmeticContext(
                _format_of=source.format()
            )
            result = source._converted(target)
        else:
            result = Self._rounded_input(
                _float_input(value),
                context.value() if context else ArithmeticContext(),
            )
        self = Self(_rounded=result)

    def __init__(
        out self,
        value: IntLiteral,
        *,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        """Round an integer literal of any width once to the format."""
        var result = Self._rounded_input(
            _float_input(value),
            context.value() if context else ArithmeticContext(),
        )
        self = Self(_rounded=result)

    def __init__[
        dtype: DType
    ](
        out self,
        value: SIMD[dtype, 1],
        *,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        """Convert a typed native number exactly, then round once; Float64 subnormals and signed zeros keep their values."""
        var result = Self._rounded_input(
            _float_input(value),
            context.value() if context else ArithmeticContext(),
        )
        self = Self(_rounded=result)

    def __init__(
        out self,
        value: FloatLiteral,
        *,
        context: Optional[ArithmeticContext] = None,
    ) raises:
        """Rejects bare decimal literals, explaining the exact and binary64 alternatives."""
        var result = Self._rounded_input(
            _float_input(value),
            context.value() if context else ArithmeticContext(),
        )
        self = Self(_rounded=result)

    def _exactly(self, rounding: RoundingMode = RoundingMode.nearest_even) -> Self:
        """This value in an exact working format: arithmetic on it keeps every
        bit, and `rounding` decides the sign of zeros from exact cancellation."""
        var result = self
        result._format = FloatFormat._exact_format(rounding, self.precision())
        return result^

    @staticmethod
    def _placeholder() -> Self:
        var format = FloatFormat(_validated=(128, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX))
        return Self(_rounded=_RoundedBinary(0, False, Integer(0), 0, format, NumericStatus()))

    def __init__(out self, *, copy: Self):
        self._significand = copy._significand
        self._format = copy._format
        self._exponent = copy._exponent
        self._kind = copy._kind
        self._negative = copy._negative

    def __init__(
        out self,
        text: String,
        *,
        context: Optional[ArithmeticContext] = None,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises:
        """Parse exact decimal, hexadecimal or binary text and round it once.

        Decimal text takes an optional `e` exponent (`1.25e-3`); `0x` and `0b` text
        requires a `p` binary exponent (`0x1.8p2`). `inf` and `nan` are accepted in any
        case. The whole source is rounded once, with no native intermediate.

        Args:
            text: The number.
            context: The output format, rounding mode and traps; 128 bits by default.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Raises:
            When the text is not a valid number, naming the byte offset; on a trapped
            condition; or when it exceeds `limits`.
        """
        self = Self.parse(
            text,
            context=context,
            allow_whitespace=allow_whitespace,
            allow_underscores=allow_underscores,
            limits=limits,
        )

    @staticmethod
    def parse(
        text: String,
        *,
        context: Optional[ArithmeticContext] = None,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises -> Self:
        """Parse text; the same contract as the text constructor.

        Args:
            text: The number.
            context: The output format, rounding mode and traps.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The parsed Float.

        Raises:
            When the text is not a valid number, on a trapped condition, or when it
            exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        var rounded = _parse_float(
            text,
            context.value() if context else ArithmeticContext(),
            allow_whitespace,
            allow_underscores,
            budget,
        )
        return Self(_rounded=rounded)

    def _as_float_input(self) -> _FloatInput:
        return _FloatInput(
            self._kind,
            self._negative,
            self._significand,
            Integer(1),
            Int128(self._exponent) - Int128(self.precision()),
        )

    def _interchange_record(self) -> _RoundedBinary:
        return _RoundedBinary(
            self._kind,
            self._negative,
            self._significand,
            self._exponent,
            self._format,
            NumericStatus(),
        )

    def to_string(
        self,
        base: Int = 10,
        *,
        notation: StaticString = "auto",
        digits: Optional[Int] = None,
        rounding: RoundingMode = RoundingMode.nearest_even,
        limits: Optional[ConversionLimits] = None,
    ) raises -> String:
        """Write the value as text, by default as `print` does.

        Base 10 writes the shortest decimal that reads back as this Float, or
        with `digits` significant digits rounded with `rounding`. `notation`
        lays it out: `auto` (the default) is positional from 1e-6 up to but
        excluding 1e21 (`0.00001`, `3.0`) and scientific beyond (`1e+21`);
        `positional` and `scientific` force one. `hexadecimal`, or base 16,
        writes the exact value (`0x3p0`); base 2 writes it in binary.

        Args:
            base: 10 for decimal, 16 or 2 for the exact value.
            notation: `auto`, `positional`, `scientific` or `hexadecimal`.
            digits: The significant digits of a decimal, at least one; by
                default the fewest that read back as this Float.
            rounding: The rounding mode with `digits`.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The text.

        Raises:
            When the base, notation or options are invalid, or the output
            exceeds `limits`.
        """
        rounding._validate()
        var exact = notation == "hexadecimal" or base != 10
        if exact:
            if notation != "hexadecimal" and notation != "auto":
                raise Error(
                    "Notations other than hexadecimal apply to decimal output; use"
                    " base=10 for them. The destination is unchanged."
                )
            if notation == "hexadecimal" and base != 10 and base != 16:
                raise Error(
                    "Hexadecimal notation writes base 16; omit base or use"
                    " base=16. The destination is unchanged."
                )
            if digits or rounding != RoundingMode.nearest_even:
                raise Error(
                    "Float digits and rounding options apply only to decimal"
                    " output; use base=10 or omit those options for exact"
                    " binary/hexadecimal output. The destination is unchanged."
                )
            var budget = _ConversionBudget(limits)
            return _format_float_exact(self._interchange_record(), 16 if notation == "hexadecimal" else base, budget)
        if digits:
            var budget = _ConversionBudget(limits)
            return _format_float_decimal(self._interchange_record(), digits.value(), rounding, budget, notation)
        if rounding != RoundingMode.nearest_even:
            raise Error(
                "Rounding applies with digits; the shortest decimal reads back"
                " exactly. The destination is unchanged."
            )
        var budget = _ConversionBudget(limits)
        budget.values(1)
        return _format_float_shortest(self, notation, budget)

    def to_json(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """Write the version-1 JSON record: the value and its complete format.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON; see the conversion chapter for the schema.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        return _format_float_json(self._interchange_record(), budget)

    @staticmethod
    def from_json(
        text: String, *, limits: Optional[ConversionLimits] = None
    ) raises -> Self:
        """Read a Float from its version-1 JSON record, without rounding.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The Float the record holds, in the record's format.

        Raises:
            When the text is not exactly that schema or is not canonical.
        """
        var budget = _ConversionBudget(limits)
        return Self(_rounded=_read_float_json(text, budget))

    @staticmethod
    def _product(
        left: Self, right: Self, context: Optional[ArithmeticContext] = None,
        *, fail: Bool = False,
    ) raises -> _RoundedBinary:
        # Canonical finite Floats already have binary magnitudes. Borrow them
        # directly instead of constructing owning exact-source argument records.
        if left._kind != 1 or right._kind != 1:
            return _float_operation(left, right, 2, context, fail=fail)
        var target = context.value() if context else ArithmeticContext(
            _format_of=left._format if left._format == right._format else _merge_float_formats(left._format, right._format)
        )
        var scale = _exponent_add(
            Int128(left._exponent) - Int128(left.precision()),
            Int128(right._exponent) - Int128(right.precision()),
        )
        return _binary_product(
            left._significand, right._significand,
            left._negative != right._negative, scale, target, fail,
        )

    @staticmethod
    def _quotient(
        left: Self, right: Self, context: Optional[ArithmeticContext] = None,
        *, fail: Bool = False,
    ) raises -> _RoundedBinary:
        # As _product: borrow both binary significands.
        if left._kind != 1 or right._kind != 1:
            return _float_operation(left, right, 3, context, fail=fail)
        var target = context.value() if context else ArithmeticContext(
            _format_of=left._format if left._format == right._format else _merge_float_formats(left._format, right._format)
        )
        var scale = _exponent_add(
            Int128(left._exponent) - Int128(left.precision()),
            Int128(right.precision()) - Int128(right._exponent),
        )
        return _binary_quotient(
            left._significand, right._significand,
            left._negative != right._negative, scale, target, fail,
        )

    @staticmethod
    def _sum(
        left: Self, right: Self, negate_right: Bool,
        context: Optional[ArithmeticContext] = None, *, fail: Bool = False,
    ) raises -> _RoundedBinary:
        # As in _product: finite Floats lend their binary significands, so
        # neither operand is copied into an owning argument record.
        if left._kind != 1 or right._kind != 1:
            return _float_operation(
                left, right, 1 if negate_right else 0, context, fail=fail
            )
        var target = context.value() if context else ArithmeticContext(
            _format_of=left._format if left._format == right._format else _merge_float_formats(left._format, right._format)
        )
        return _binary_float_sum(
            left._significand,
            left._negative,
            Int128(left._exponent) - Int128(left.precision()),
            right._significand,
            right._negative != negate_right,
            Int128(right._exponent) - Int128(right.precision()),
            target,
            fail,
        )

    @staticmethod
    def _from_arguments(
        left: _FloatArgument,
        right: _FloatArgument,
        operation: Int,
        context: Optional[ArithmeticContext] = None,
        *,
        fail: Bool = False,
    ) raises -> Self:
        return Self(
            _rounded=_float_operation(
                left, right, operation, context, fail=fail
            )
        )

    def _assign(
        mut self, rhs: _FloatArgument, operation: Int, *, fail: Bool = False
    ) raises:
        var result = Self._from_arguments(
            self,
            rhs,
            operation,
            ArithmeticContext(_format_of=self.format()),
            fail=fail,
        )
        self = result

    def __pos__(self) -> Self:
        return self

    def __pow__(self, exponent: Integer) raises -> Self:
        return Self(_rounded=_pow_float(self, exponent))

    def __pow__[
        B: _FloatBatchOperand
    ](self, exponent: B) raises -> B.FloatBatch:
        return exponent._float_power_reverse(self)

    def __ipow__(mut self, exponent: Integer) raises:
        var result = Self(
            _rounded=_pow_float(
                self, exponent, ArithmeticContext(_format_of=self.format())
            )
        )
        self = result

    def floor(self) raises -> Integer:
        """Round toward negative infinity, to an Integer.

        Returns:
            The largest Integer not above the value.

        Raises:
            For infinity and NaN.
        """
        return _float_integer(self._as_float_input(), 1)

    def ceil(self) raises -> Integer:
        """Round toward positive infinity, to an Integer.

        Returns:
            The smallest Integer not below the value.

        Raises:
            For infinity and NaN.
        """
        return _float_integer(self._as_float_input(), 2)

    def trunc(self) raises -> Integer:
        """Round toward zero, to an Integer.

        Returns:
            The integer part.

        Raises:
            For infinity and NaN.
        """
        return _float_integer(self._as_float_input(), 0)

    def is_integer(self) -> Bool:
        """Whether the value is a finite integer (Python's `float.is_integer`).

        Returns:
            True for both zeros and for finite values without a fractional
            part; False for infinities and NaN.
        """
        if self._kind == 0:
            return True
        if self._kind != 1:
            return False
        var fraction = self.precision() - self._exponent
        if fraction <= 0:
            return True
        if fraction >= self.precision():
            return False
        return _trailing_zero_bits(self._significand) >= fraction

    def round(self) raises -> Integer:
        """Round to the nearest Integer, and a half to the even neighbour.

        Returns:
            The nearest Integer: 2.5 gives 2, 3.5 gives 4, -2.5 gives -2 and
            0.5 gives 0.

        Raises:
            For infinity and NaN.
        """
        return _float_integer(self._as_float_input(), 4)

    def to_rational_exact(self) raises -> Rational:
        """The exact value as a Rational, in lowest terms.

        A finite Float is a binary fraction, so the conversion never rounds;
        both zeros give 0. `Rational(x)` is the same conversion.

        Returns:
            The Rational with the same value.

        Raises:
            For infinity and NaN, or when the value would exceed addressable
            storage.
        """
        if not self.is_finite():
            raise Error(
                "Cannot convert infinity or NaN to Rational; check is_finite()"
                " first. The destination is unchanged."
            )
        if self._kind == 0:
            return Rational(0)
        var zeros = _trailing_zero_bits(self._significand)
        var odd = self._significand >> zeros
        var scale = self._exponent - self.precision() + zeros
        if self._negative:
            odd = -odd
        if scale >= 0:
            return Rational(_numerator=odd << scale, _denominator=Integer(1))
        return Rational(_numerator=odd, _denominator=Integer(1) << -scale)

    def to_integer_exact(self) raises -> Integer:
        """Convert an integral value to Integer.

        Returns:
            The equal Integer.

        Raises:
            When the value has a fractional part, or is infinite or NaN.
        """
        return _float_integer(self._as_float_input(), 3)

    def to_native[
        dtype: DType
    ](
        self,
        *,
        rounding: RoundingMode = RoundingMode.nearest_even,
    ) raises -> SIMD[dtype, 1]:
        """Round to a native floating-point type.

        Rounds directly, subnormals included, with no intermediate Float64.

        Parameters:
            dtype: `float16`, `bfloat16`, `float32` or `float64`.

        Args:
            rounding: The rounding mode.

        Returns:
            The rounded native value.

        Raises:
            When `dtype` is not a supported floating-point type.
        """
        return _float_native[dtype](self._as_float_input(), rounding).value

    def to_native_exact[dtype: DType](self) raises -> SIMD[dtype, 1]:
        """Convert to a native type without loss.

        Floating-point targets keep signed zero, infinity and NaN; integral targets
        require a finite integral value in range.

        Parameters:
            dtype: A native floating-point or integral type.

        Returns:
            The same value in the native type.

        Raises:
            When the conversion would round or the value does not fit.
        """
        comptime if dtype.is_integral():
            if self._kind == 1 and self._exponent > bit_width_of[dtype]():
                raise Error(
                    "Cannot convert Float exactly to the requested native"
                    " integer: the value is out of range; retain Float or use"
                    " to_integer_exact() when an arbitrary-precision integer"
                    " is intended. The destination is unchanged."
                )
            return self.to_integer_exact().to_native_exact[dtype]()
        else:
            var result = _float_native[dtype](
                self._as_float_input(), RoundingMode.nearest_even
            )
            if result.status.inexact():
                raise Error(
                    "Cannot convert Float exactly to the requested native"
                    " format; retain Float or use to_native with an explicit"
                    " rounding choice. The destination is unchanged."
                )
            return result.value

    def __add__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 0)

    def __sub__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 1)

    def __mul__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 2)

    def __truediv__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 3)

    def __neg__(self) -> Self:
        var result = self
        if not result.is_nan():
            result._negative = not result._negative
        return result

    def __abs__(self) -> Self:
        var result = self
        result._negative = False
        return result

    # Keep literals out of the pinned compiler's native-SIMD coercion path.
    def __add__(self, rhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(self, rhs, 0)

    def __radd__(self, lhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(lhs, self, 0)

    def __iadd__(mut self, rhs: FloatLiteral) raises:
        self._assign(rhs, 0)

    def __sub__(self, rhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(self, rhs, 1)

    def __rsub__(self, lhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(lhs, self, 1)

    def __isub__(mut self, rhs: FloatLiteral) raises:
        self._assign(rhs, 1)

    def __mul__(self, rhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(self, rhs, 2)

    def __rmul__(self, lhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(lhs, self, 2)

    def __imul__(mut self, rhs: FloatLiteral) raises:
        self._assign(rhs, 2)

    def __truediv__(self, rhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(self, rhs, 3)

    def __rtruediv__(self, lhs: FloatLiteral) raises -> Self:
        return Float._from_arguments(lhs, self, 3)

    def __itruediv__(mut self, rhs: FloatLiteral) raises:
        self._assign(rhs, 3)

    def __add__(self, rhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(self, rhs, 0)

    def __add__(self, rhs: Self) raises -> Self:
        return Self(_rounded=Self._sum(self, rhs, False))

    def __sub__(self, rhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(self, rhs, 1)

    def __sub__(self, rhs: Self) raises -> Self:
        return Self(_rounded=Self._sum(self, rhs, True))

    def __mul__(self, rhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(self, rhs, 2)

    def __mul__(self, rhs: Self) raises -> Self:
        return Self(_rounded=Self._product(self, rhs))

    def __truediv__(self, rhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(self, rhs, 3)

    def __truediv__(self, rhs: Self) raises -> Self:
        return Self(_rounded=Self._quotient(self, rhs))

    def __radd__(self, lhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(lhs, self, 0)

    def __rsub__(self, lhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(lhs, self, 1)

    def __rmul__(self, lhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(lhs, self, 2)

    def __rtruediv__(self, lhs: _FloatArgument) raises -> Self:
        return Self._from_arguments(lhs, self, 3)

    def __iadd__(mut self, rhs: _FloatArgument) raises:
        self._assign(rhs, 0)

    def __iadd__(mut self, var rhs: Self) raises:
        var rounded = Self._sum(self, rhs, False, ArithmeticContext(_format_of=self.format()))
        self = Self(_rounded=rounded^)

    def __isub__(mut self, rhs: _FloatArgument) raises:
        self._assign(rhs, 1)

    def __isub__(mut self, var rhs: Self) raises:
        var rounded = Self._sum(self, rhs, True, ArithmeticContext(_format_of=self.format()))
        self = Self(_rounded=rounded^)

    def __imul__(mut self, rhs: _FloatArgument) raises:
        self._assign(rhs, 2)

    def __imul__(mut self, var rhs: Self) raises:
        var rounded = Self._product(self, rhs, ArithmeticContext(_format_of=self.format()))
        self = Self(_rounded=rounded^)

    def __itruediv__(mut self, rhs: _FloatArgument) raises:
        self._assign(rhs, 3)

    @staticmethod
    def _from_rounded(mut rounded: _RoundedBinary) -> Self:
        """Take rounded's significand without a refcount pair; rounded keeps zero."""
        var result = Self(_rounded=_RoundedBinary(
            rounded.kind, rounded.negative, Integer(0), rounded.exponent,
            rounded.format, rounded.status,
        ))
        swap(result._significand, rounded.significand)
        return result^

    def __init__(out self, *, var _rounded: _RoundedBinary):
        self._format = _rounded.format
        self._exponent = _rounded.exponent
        self._kind = _rounded.kind
        self._negative = _rounded.negative if _rounded.kind != 3 else False
        self._significand = Integer(0)
        swap(self._significand, _rounded.significand)

    @staticmethod
    def _from_argument(argument: _FloatArgument) raises -> Optional[Self]:
        """The Float an argument record was made from, if it was made from one.

        Such a record holds the Float's exact value and its format, so rounding
        it to that format rebuilds the same Float.
        """
        if not argument.format or argument.native_precision:
            return None
        return Self(_rounded=Self._rounded_input(
            argument.value, ArithmeticContext(_format_of=argument.format.value())
        ))

    @staticmethod
    def _rounded_input(
        source: _FloatInput, context: ArithmeticContext, *, fail: Bool = False
    ) raises -> _RoundedBinary:
        if source.kind >= 2:
            var status = NumericStatus._make(0, 2 if source.kind == 3 else 0)
            return _finish_round(
                _RoundedBinary(
                    source.kind,
                    source.negative,
                    Integer(0),
                    0,
                    context.format(),
                    status,
                ),
                context,
                fail,
            )
        return _round_ratio(
            -source.numerator if source.negative else source.numerator,
            source.denominator,
            context,
            scale=source.scale,
            negative_zero=source.negative,
            fail=fail,
        )

    def _converted(
        self, context: ArithmeticContext, *, fail: Bool = False
    ) raises -> _RoundedBinary:
        if context.format() == self.format():
            return _finish_round(
                _RoundedBinary(
                    self._kind,
                    self._negative,
                    self._significand,
                    self._exponent,
                    self._format,
                    NumericStatus._make(0, 2 if self.is_nan() else 0),
                ),
                context,
                fail,
            )
        return Self._rounded_input(
            _FloatInput(
                self._kind,
                self._negative,
                self._significand,
                Integer(1),
                Int128(self._exponent) - Int128(self.precision()),
            ),
            context,
            fail=fail,
        )

    @staticmethod
    def from_native[
        dtype: DType
    ](
        value: SIMD[dtype, 1], *, context: Optional[ArithmeticContext] = None
    ) raises -> Self:
        """Convert a typed native number exactly, then round once.

        Parameters:
            dtype: The native type.

        Args:
            value: The native value.
            context: The output format, rounding mode and traps.

        Returns:
            The Float.

        Raises:
            On a trapped condition or an unsupported type.
        """
        return Self(value, context=context)

    @staticmethod
    def from_native(
        value: Int, *, context: Optional[ArithmeticContext] = None
    ) raises -> Self:
        """Convert a native `Int` exactly, then round once."""
        return Self(value, context=context)

    def to_format(
        self,
        format: FloatFormat,
        *,
        rounding: RoundingMode = RoundingMode.nearest_even,
    ) raises -> Self:
        """Round to another format.

        Increasing precision keeps the value; it cannot recover information lost
        earlier. Different exponent bounds may underflow or overflow.

        Args:
            format: The new format.
            rounding: The rounding mode.

        Returns:
            A new Float in `format`.

        Raises:
            Only on an invalid rounding mode.
        """
        return Self(
            self, context=ArithmeticContext(format=format, rounding=rounding)
        )

    @staticmethod
    def zero(
        *, negative: Bool = False, context: Optional[ArithmeticContext] = None
    ) raises -> Self:
        """A signed zero.

        Args:
            negative: Whether the zero is negative.
            context: Its format; 128 bits by default.

        Returns:
            The zero.

        Raises:
            Only on an invalid context.
        """
        var target = context.value() if context else ArithmeticContext()
        return Self(
            _rounded=_RoundedBinary(
                0, negative, Integer(0), 0, target.format(), NumericStatus()
            )
        )

    @staticmethod
    def infinity(
        *, negative: Bool = False, context: Optional[ArithmeticContext] = None
    ) raises -> Self:
        """A signed infinity.

        Args:
            negative: Whether the infinity is negative.
            context: Its format; 128 bits by default.

        Returns:
            The infinity.

        Raises:
            Only on an invalid context.
        """
        var target = context.value() if context else ArithmeticContext()
        return Self(
            _rounded=_RoundedBinary(
                2, negative, Integer(0), 0, target.format(), NumericStatus()
            )
        )

    @staticmethod
    def nan(*, context: Optional[ArithmeticContext] = None) raises -> Self:
        """The canonical quiet NaN.

        Args:
            context: Its format; 128 bits by default.

        Returns:
            NaN.

        Raises:
            Only on an invalid context.
        """
        var target = context.value() if context else ArithmeticContext()
        return Self(
            _rounded=_RoundedBinary(
                3,
                False,
                Integer(0),
                0,
                target.format(),
                NumericStatus._make(0, 2),
            )
        )

    def format(self) -> FloatFormat:
        """The stored format.

        Returns:
            The precision and exponent bounds.
        """
        return self._format

    def precision(self) -> Int:
        """The significand precision in bits.

        Returns:
            The format's precision.
        """
        return self._format.precision()

    def is_zero(self) -> Bool:
        """Whether the value is either zero.

        Returns:
            True for `+0` and `-0`.
        """
        return self._kind == 0

    def is_finite(self) -> Bool:
        """Whether the value is zero or finite.

        Returns:
            False for infinities and NaN.
        """
        return self._kind <= 1

    def is_infinite(self) -> Bool:
        """Whether the value is an infinity.

        Returns:
            True for `+inf` and `-inf`.
        """
        return self._kind == 2

    def is_nan(self) -> Bool:
        """Whether the value is NaN.

        Returns:
            True for NaN.
        """
        return self._kind == 3

    def __bool__(self) -> Bool:
        return not self.is_zero()

    def signbit(self) -> Bool:
        """Whether the sign is negative, including negative zero.

        Returns:
            The sign bit; false for NaN.
        """
        return self._negative

    def sign(self) raises -> Int:
        """The sign: -1, 0 or 1.

        Returns:
            -1 for negative values, 0 for zeros, 1 for positive values.

        Raises:
            For NaN.
        """
        if self.is_nan():
            raise Error(
                "Cannot take the sign of NaN; check is_nan() before requesting"
                " a numerical sign."
            )
        return 0 if self.is_zero() else -1 if self._negative else 1

    def significand(self) raises -> Integer:
        """The significand `m` of a finite value.

        Returns:
            A nonnegative Integer with exactly `precision()` bits; zero for zeros.

        Raises:
            For infinity and NaN.
        """
        if not self.is_finite():
            raise Error(
                "Infinity and NaN have no finite significand; check is_finite()"
                " before inspecting components."
            )
        return self._significand

    def exponent(self) raises -> Int:
        """The normalized exponent `e` of a finite nonzero value.

        Returns:
            The exponent, with `value = significand * 2**(exponent - precision)`.

        Raises:
            For zero, infinity and NaN.
        """
        if self._kind != 1:
            raise Error(
                "Only a nonzero finite Float has a normalized exponent; check"
                " is_finite() and is_zero() first."
            )
        return self._exponent

    def _compare_float(self, rhs: Self) -> Int:
        if self.is_nan() or rhs.is_nan():
            return 2
        if self.is_zero() and rhs.is_zero():
            return 0
        if self._negative != rhs._negative:
            return -1 if self._negative else 1
        var order = 0
        if self._kind != rhs._kind:
            order = -1 if self._kind < rhs._kind else 1
        elif self._kind == 1:
            if self._exponent != rhs._exponent:
                order = -1 if self._exponent < rhs._exponent else 1
            elif self.precision() == rhs.precision():
                # Aligned significands: compare them as integers.
                if self._significand != rhs._significand:
                    order = -1 if self._significand < rhs._significand else 1
            else:
                var precision = max(self.precision(), rhs.precision())
                var count = (precision - 1) // 32 + 1
                for i in range(count - 1, -1, -1):
                    var left = _shifted_word(
                        self._significand, i, precision - self.precision()
                    )
                    var right = _shifted_word(
                        rhs._significand, i, precision - rhs.precision()
                    )
                    if left != right:
                        order = -1 if left < right else 1
                        break
        return -order if self._negative else order

    def _compare_input(self, rhs: _FloatInput) raises -> Int:
        if self.is_nan() or rhs.kind == 3:
            return 2
        if self.is_zero() and rhs.kind == 0:
            return 0
        if self._negative != rhs.negative:
            return -1 if self._negative else 1
        var order: Int
        if self._kind != rhs.kind:
            order = -1 if self._kind < rhs.kind else 1
        elif self._kind != 1:
            order = 0
        else:
            var scale = (
                Int128(self._exponent) - Int128(self.precision()) - rhs.scale
            )
            var upper_bits = (
                Int128(self._exponent)
                + Int128(rhs.denominator.magnitude_bit_length())
                - rhs.scale
            )
            var right_bits = Int128(rhs.numerator.magnitude_bit_length())
            if upper_bits - 1 > right_bits:
                order = 1
            elif upper_bits < right_bits:
                order = -1
            else:
                var product = self._significand * rhs.denominator
                order = -_compare_scaled(rhs.numerator, product, scale)
        return -order if self._negative else order

    def _compare_integer(self, rhs: Integer, operation: Int) raises -> Bool:
        return _float_relation(
            self._compare_input(_float_input(rhs)), operation
        )

    def _compare_rational(self, rhs: Rational, operation: Int) raises -> Bool:
        return _float_relation(
            self._compare_input(_float_input(rhs)), operation
        )

    def __eq__(self, rhs: Self) -> Bool:
        return self._compare_float(rhs) == 0

    def __ne__(self, rhs: Self) -> Bool:
        return self._compare_float(rhs) != 0

    def __lt__(self, rhs: Self) -> Bool:
        return _float_relation(self._compare_float(rhs), 2)

    def __le__(self, rhs: Self) -> Bool:
        return _float_relation(self._compare_float(rhs), 3)

    def __gt__(self, rhs: Self) -> Bool:
        return _float_relation(self._compare_float(rhs), 4)

    def __ge__(self, rhs: Self) -> Bool:
        return _float_relation(self._compare_float(rhs), 5)

    def __eq__[T: Copyable](self, rhs: T) raises -> Bool:
        comptime if conforms_to(T, _ComplexComparison):
            return rhs._equals_real(self)
        else:
            return _float_relation(self._compare_input(_float_input(rhs)), 0)

    def __ne__[T: Copyable](self, rhs: T) raises -> Bool:
        comptime if conforms_to(T, _ComplexComparison):
            return not rhs._equals_real(self)
        else:
            return _float_relation(self._compare_input(_float_input(rhs)), 1)

    def __lt__[T: Copyable](self, rhs: T) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 2)

    def __le__[T: Copyable](self, rhs: T) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 3)

    def __gt__[T: Copyable](self, rhs: T) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 4)

    def __ge__[T: Copyable](self, rhs: T) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 5)

    def __eq__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 0)

    def __ne__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 1)

    def __lt__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 2)

    def __le__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 3)

    def __gt__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 4)

    def __ge__(self, rhs: IntLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 5)

    def __eq__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 0)

    def __ne__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 1)

    def __lt__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 2)

    def __le__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 3)

    def __gt__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 4)

    def __ge__[dtype: DType](self, rhs: SIMD[dtype, 1]) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 5)

    def __eq__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 0)

    def __ne__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 1)

    def __lt__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 2)

    def __le__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 3)

    def __gt__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 4)

    def __ge__(self, rhs: FloatLiteral) raises -> Bool:
        return _float_relation(self._compare_input(_float_input(rhs)), 5)

    def total_cmp(self, rhs: Self) -> Int:
        """Compare in the total order.

        The order is negative infinity, negative finite values, `-0`, `+0`, positive
        finite values, positive infinity, NaN. Equal values in different formats tie.

        Args:
            rhs: The other Float.

        Returns:
            -1, 0 or 1.
        """
        if self.is_nan() or rhs.is_nan():
            return Int(self.is_nan()) - Int(rhs.is_nan())
        if self.is_zero() and rhs.is_zero():
            return Int(rhs.signbit()) - Int(self.signbit())
        return self._compare_float(rhs)

    def same_representation(self, other: Self) -> Bool:
        """Whether two Floats are the same representation, not just equal values.

        `==` compares values, so `0.0 == -0.0`, NaN equals nothing, and 1 at 53
        and 128 bits are equal. Representations differ in each of those cases:
        they match when class, sign, precision, exponent bounds, exponent and
        significand all do. NaN is canonical, so the NaNs of one format match.

        Args:
            other: The other Float.

        Returns:
            True when every part of the representation matches.
        """
        if self._kind != other._kind or not self._same_format(other):
            return False
        if self._kind == 3:
            return True
        if self._negative != other._negative:
            return False
        if self._kind != 1:
            return True
        return self._exponent == other._exponent and self._significand == other._significand

    def representation_cmp(self, other: Self) -> Int:
        """Compare representations in a strict total order.

        Values come first, in the `total_cmp` order; equal values then order by
        precision, then by the lower and the upper exponent bound, each
        ascending. Sorting by this order gives the same result from any input
        order.

        Args:
            other: The other Float.

        Returns:
            -1, 0 or 1; 0 exactly when `same_representation` holds.
        """
        var order = self.total_cmp(other)
        if order:
            return order
        var a = self._format
        var b = other._format
        if a.precision() != b.precision():
            return -1 if a.precision() < b.precision() else 1
        if a.emin() != b.emin():
            return -1 if a.emin() < b.emin() else 1
        if a.emax() != b.emax():
            return -1 if a.emax() < b.emax() else 1
        return 0

    def _same_format(self, other: Self) -> Bool:
        return (
            self._format.precision() == other._format.precision()
            and self._format.emin() == other._format.emin()
            and self._format.emax() == other._format.emax()
        )

    def _hash_into(self, mut hash: _StableHash):
        """Feed this representation to a stable hash (APNH-64)."""
        hash.float(
            self._format.precision(),
            self._format.emin(),
            self._format.emax(),
            self._kind,
            self._negative,
            self._exponent,
            self._significand,
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the shortest decimal that reads back as this Float, as `print`
        does: positional from 1e-6 up to but excluding 1e21, scientific beyond
        (see `to_string`). A value too long for the default conversion limits
        is written in exact hexadecimal.

        Args:
            writer: The destination.
        """
        try:
            var budget = _ConversionBudget(None)
            writer.write(_format_float_shortest(self, "auto", budget))
        except:
            _write_float_exact(self._interchange_record(), 16, writer)
