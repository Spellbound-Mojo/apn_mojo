"""Independent Complex components with explicit mathematical settings."""

from std.builtin.bool import Boolable
from ..integer.value import Integer
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..float.value import Float
from ..float.context import FloatFormat, ArithmeticContext
from ..float.status import NumericStatus
from ..float._arithmetic import _FloatArgument
from ..float._rounding import _RoundedBinary, _finish_round
from .context import ComplexContext, _ComplexContextArgument
from .status import ComplexStatus
from ._format import _ComplexFormats
from ._comparison import _ComplexComparison
from ._input import _ComplexSource, _ComplexArgument
from ._batch_protocol import _ComplexBatchArithmetic, _ComplexBatchOperand
from ..batch.mask import Mask
from ._arithmetic import _complex_operation, _target
from ._functions import _complex_real_function, _sqrt_complex
from ._power import _pow_complex
from ._text import _parse_complex, _format_complex_exact
from ..float.shortest import _format_float_shortest
from ..common._sizes import _checked_sum
from ._json import _read_complex_json, _format_complex_json
from ..common._traits import _BatchElement
from ..common._stable_hash import _StableHash


def _component_round(
    source: _FloatArgument, context: ArithmeticContext, fail: Bool
) raises -> _RoundedBinary:
    if source.format and source.format.value() == context.format():
        var value = source.value
        return _finish_round(
            _RoundedBinary(
                value.kind,
                value.negative,
                value.numerator,
                Int(
                    value.scale + Int128(context.format().precision())
                ) if value.kind
                == 1 else 0,
                context.format(),
                NumericStatus._make(0, 2 if value.kind == 3 else 0),
            ),
            context,
            fail,
        )
    return Float._rounded_input(source.value, context, fail=fail)


struct Complex(
    Boolable,
    Equatable,
    ImplicitlyCopyable,
    Writable,
    _ComplexComparison,
    _ComplexSource,
    _BatchElement,
):
    """A complex number with two Float components.

    The components have independent precisions and common exponent bounds. Every
    operation rounds each component of the exact result once: a product rounds
    `ac - bd` and `ad + bc`, never the products first.

    | Operation | Contract |
    |---|---|
    | `z + w`, `z - w`, `z * w`, `z / w` | Each component rounded once; either side may be real or exact |
    | `z ** n` | Signed integral power, each component rounded once |
    | `+z`, `-z` | Exact, same formats |
    | `z += w` and the other compound forms | Keep the destination's formats; unchanged on error |
    | `z == w`, `z != w` | Exact equality; signed zeros are equal, NaN never is |
    | `Bool(z)` | False only for complex zero |

    Special values are decided by each operation as a whole, not by a chain of
    real operations.
    Printing shows `Complex(real, imag)` with exact hexadecimal components, which
    the text parser accepts.

    Limitations:
        There is no ordering, no hashing and no implicit conversion to a native
        number. A native number cannot be the left operand of `==`.
    """

    var _real: Float
    var _imag: Float

    @staticmethod
    def _placeholder() -> Self:
        return Self(_real=Float._placeholder(), _imag=Float._placeholder())

    def __init__(out self, *, copy: Self):
        self._real = copy._real
        self._imag = copy._imag

    def __init__(
        out self,
        *,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        """Complex zero: `+0` in both components.

        Args:
            context: The component formats.

        Raises:
            Only on an invalid context.
        """
        self = Self(Int(0), Int(0), context=context)

    def __init__(
        out self,
        real: _FloatArgument,
        *,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        """A real value with imaginary `+0`.

        Args:
            real: The real part; any exact or Float value.
            context: The component formats; by default the real part's format.

        Raises:
            On a trapped condition.
        """
        self = Self(real, Int(0), context=context)

    def __init__(
        out self,
        real: _FloatArgument,
        imag: _FloatArgument,
        *,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        """Construct from both components, each rounded once.

        A component without a Float format takes its partner's; with neither, both use
        128 bits. The exponent bounds must agree unless a context is given.

        Args:
            real: The real part.
            imag: The imaginary part.
            context: The component formats.

        Raises:
            When the bounds differ without a context, or on a trapped condition.
        """
        self = Self._from_components(real, imag, context)[0]

    def __init__(
        out self,
        value: Self,
        *,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises:
        """A copy, rounded to the context when one is given."""
        self = Self(value._real, value._imag, context=context)

    def __init__(
        out self,
        text: String,
        *,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises:
        """Parse real, imaginary, algebraic or paired text.

        Accepts `3`, `4j`, `-j`, `3+4j`, `3-4i`, `(3+4j)`, `(3,4)` and the display form
        `Complex(3, 4)`. Each component uses Float's decimal, hexadecimal and binary
        grammar; both are validated before either rounds.

        Args:
            text: The number.
            context: The component formats.
            allow_whitespace: Accept surrounding and component-boundary whitespace.
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
        context: _ComplexContextArgument = _ComplexContextArgument(),
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises -> Self:
        """Parse text; the same contract as the text constructor.

        Args:
            text: The number.
            context: The component formats.
            allow_whitespace: Accept surrounding and component-boundary whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The parsed Complex.

        Raises:
            When the text is not a valid number, on a trapped condition, or when it
            exceeds `limits`.
        """
        return Self._parse(
            text, context, allow_whitespace, allow_underscores, limits
        )[0]

    @staticmethod
    def _parse(
        text: String,
        context: _ComplexContextArgument,
        whitespace: Bool,
        underscores: Bool,
        limits: Optional[ConversionLimits],
        *,
        fail_component: Int = -1,
    ) raises -> Tuple[Self, ComplexStatus]:
        var budget = _ConversionBudget(limits)
        var parts = _parse_complex(
            text,
            context,
            whitespace,
            underscores,
            budget,
            fail_component=fail_component,
        )
        return (
            Self(
                _real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1])
            ),
            ComplexStatus._make(parts[0].status, parts[1].status),
        )

    def to_string(
        self, base: Int = 10, *, notation: StaticString = "auto", limits: Optional[ConversionLimits] = None,
    ) raises -> String:
        """Write `Complex(real, imag)`, by default as `print` does.

        Each part is written as `Float.to_string` writes it: the shortest
        decimal that reads back, laid out by `notation` (`auto`, `positional`
        or `scientific`), or the exact value with `hexadecimal`, base 16 or
        base 2.

        Args:
            base: 10 for decimal, 16 or 2 for the exact value.
            notation: `auto`, `positional`, `scientific` or `hexadecimal`.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The text.

        Raises:
            When the base or notation is invalid or the output exceeds `limits`.
        """
        if notation == "hexadecimal" or base != 10:
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
            var budget = _ConversionBudget(limits)
            return _format_complex_exact(
                self._real._interchange_record(),
                self._imag._interchange_record(),
                16 if notation == "hexadecimal" else base,
                budget,
            )
        # One value, and one budget for both parts and the framing.
        var budget = _ConversionBudget(limits)
        budget.values(1)
        budget.output(11)
        var a = _format_float_shortest(self._real, notation, budget)
        var b = _format_float_shortest(self._imag, notation, budget)
        var count = _checked_sum(11, _checked_sum(a.byte_length(), b.byte_length()))
        var result = String()
        budget.grow_string(result, count)
        result.reserve_bytes(count)
        result.write("Complex(", a, ", ", b, ")")
        return result

    def to_json(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """Write the version-1 JSON record: two complete Float records.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Compact canonical JSON.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        return _format_complex_json(
            self._real._interchange_record(),
            self._imag._interchange_record(),
            budget,
        )

    @staticmethod
    def from_json(
        text: String, *, limits: Optional[ConversionLimits] = None
    ) raises -> Self:
        """Read a Complex from its version-1 JSON record, without rounding.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The Complex the record holds.

        Raises:
            When the text is not exactly that schema, or the component bounds differ.
        """
        var budget = _ConversionBudget(limits)
        var parts = _read_complex_json(text, budget)
        return Self(
            _real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1])
        )

    def __init__(out self, *, var _real: Float, var _imag: Float):
        # Callers have validated common bounds and completed all fallible work.
        self._real = _real^
        self._imag = _imag^

    @staticmethod
    def _from_components(
        real: _FloatArgument,
        imag: _FloatArgument,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        *,
        fail_component: Int = -1,
    ) raises -> Tuple[Self, ComplexStatus]:
        var target: ComplexContext
        if context:
            target = context.value()
        else:
            var default = (
                real.format.value() if real.format else imag.format.value() if imag.format else FloatFormat()
            )
            var formats = _ComplexFormats(
                real.format.value() if real.format else default,
                imag.format.value() if imag.format else default,
            )
            target = ComplexContext(
                real=ArithmeticContext(format=formats.real()),
                imag=ArithmeticContext(format=formats.imag()),
            )
        var a = _component_round(real, target.real(), fail_component == 0)
        var b = _component_round(imag, target.imag(), fail_component == 1)
        return (
            Self(_real=Float(_rounded=a), _imag=Float(_rounded=b)),
            ComplexStatus._make(a.status, b.status),
        )

    def real(self) -> Float:
        """The real part.

        Returns:
            An independent Float.
        """
        return self._real

    def _real_function(
        self,
        operation: Int,
        context: Optional[ArithmeticContext] = None,
    ) raises -> _RoundedBinary:
        return _complex_real_function(self, operation, context)

    def _sqrt(
        self,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        *,
        initial_work: Int = 0,
    ) raises -> Tuple[Self, ComplexStatus]:
        var parts = _sqrt_complex(self, context, initial_work=initial_work)
        return (
            Self(
                _real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1])
            ),
            ComplexStatus._make(parts[0].status, parts[1].status),
        )

    def _pow(
        self,
        exponent: Integer,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        *,
        fail_component: Int = -1,
        initial_work: Int = 0,
    ) raises -> Tuple[Self, ComplexStatus]:
        var parts = _pow_complex(
            self,
            exponent,
            context,
            fail_component=fail_component,
            initial_work=initial_work,
        )
        return (
            Self(
                _real=Float(_rounded=parts[0]), _imag=Float(_rounded=parts[1])
            ),
            ComplexStatus._make(parts[0].status, parts[1].status),
        )

    def __pow__(self, exponent: Integer) raises -> Self:
        return self._pow(exponent)[0]

    def __pow__[
        B: _ComplexBatchOperand
    ](self, exponent: B) raises -> B.ComplexBatch:
        return exponent._complex_power_reverse(_ComplexArgument(self))

    def __ipow__(mut self, exponent: Integer) raises:
        var result = self._pow(exponent)
        self = result[0]

    def _complex_real(self) -> _FloatArgument:
        return _FloatArgument(self._real)

    def _complex_imag(self) -> _FloatArgument:
        return _FloatArgument(self._imag)

    @staticmethod
    def _calculate(
        left: _ComplexArgument,
        right: _ComplexArgument,
        operation: Int,
        context: _ComplexContextArgument = _ComplexContextArgument(),
        *,
        fail_component: Int = -1,
    ) raises -> Self:
        var parts = _complex_operation(
            left, right, operation, context, fail_component=fail_component
        )
        # Swap each rounded significand out of the tuple instead of copying it.
        return Self(
            _real=Float._from_rounded(parts[0]), _imag=Float._from_rounded(parts[1])
        )

    @staticmethod
    def _calculate_status(
        left: _ComplexArgument,
        right: _ComplexArgument,
        operation: Int,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises -> ComplexStatus:
        var parts = _complex_operation(left, right, operation, context)
        return ComplexStatus._make(parts[0].status, parts[1].status)

    @staticmethod
    def _sum(
        left: Self, right: Self, negate_right: Bool,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises -> Self:
        # Component sums on borrowed Floats: no owning argument records. The
        # result formats are those the general path selects.
        var target: ComplexContext
        if context:
            target = context.value()
        else:
            ref lr = left._real._format
            ref li = left._imag._format
            if (
                lr == right._real._format
                and li == right._imag._format
                and lr.emin() == li.emin()
                and lr.emax() == li.emax()
            ):
                target = ComplexContext(
                    _validated=(
                        ArithmeticContext(format=lr),
                        ArithmeticContext(format=li),
                    )
                )
            else:
                target = _target(left, right, _ComplexContextArgument())
        return Self(
            _real=Float(
                _rounded=Float._sum(
                    left._real, right._real, negate_right, target.real()
                )
            ),
            _imag=Float(
                _rounded=Float._sum(
                    left._imag, right._imag, negate_right, target.imag()
                )
            ),
        )

    def _assign(
        mut self,
        rhs: _ComplexArgument,
        operation: Int,
        *,
        fail_component: Int = -1,
    ) raises:
        var context = ComplexContext(
            real=ArithmeticContext(format=self.real_format()),
            imag=ArithmeticContext(format=self.imag_format()),
        )
        self = Self._calculate(
            self, rhs, operation, context, fail_component=fail_component
        )

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return Self(_real=-self._real, _imag=-self._imag)

    def __add__[B: _ComplexBatchArithmetic](self, rhs: B) raises -> B.ComplexBatch:
        return rhs._complex_reverse(_ComplexArgument(self), 0)

    def __sub__[B: _ComplexBatchArithmetic](self, rhs: B) raises -> B.ComplexBatch:
        return rhs._complex_reverse(_ComplexArgument(self), 1)

    def __mul__[B: _ComplexBatchArithmetic](self, rhs: B) raises -> B.ComplexBatch:
        return rhs._complex_reverse(_ComplexArgument(self), 2)

    def __truediv__[
        B: _ComplexBatchArithmetic
    ](self, rhs: B) raises -> B.ComplexBatch:
        return rhs._complex_reverse(_ComplexArgument(self), 3)

    def __eq__[B: _ComplexBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_complex(_ComplexArgument(self), 0)

    def __ne__[B: _ComplexBatchArithmetic](self, rhs: B) raises -> Mask:
        return rhs._compare_complex(_ComplexArgument(self), 1)

    def __add__(self, rhs: _ComplexArgument) raises -> Self:
        return Self._calculate(self, rhs, 0)

    def __add__(self, rhs: Self) raises -> Self:
        return Self._sum(self, rhs, False)

    def __radd__(self, lhs: _ComplexArgument) raises -> Self:
        return Self._calculate(lhs, self, 0)

    def __iadd__(mut self, rhs: _ComplexArgument) raises:
        self._assign(rhs, 0)

    def __iadd__(mut self, var rhs: Self) raises:
        var context = ComplexContext(
            real=ArithmeticContext(format=self.real_format()),
            imag=ArithmeticContext(format=self.imag_format()),
        )
        self = Self._sum(self, rhs, False, context)

    def __sub__(self, rhs: _ComplexArgument) raises -> Self:
        return Self._calculate(self, rhs, 1)

    def __sub__(self, rhs: Self) raises -> Self:
        return Self._sum(self, rhs, True)

    def __rsub__(self, lhs: _ComplexArgument) raises -> Self:
        return Self._calculate(lhs, self, 1)

    def __isub__(mut self, rhs: _ComplexArgument) raises:
        self._assign(rhs, 1)

    def __isub__(mut self, var rhs: Self) raises:
        var context = ComplexContext(
            real=ArithmeticContext(format=self.real_format()),
            imag=ArithmeticContext(format=self.imag_format()),
        )
        self = Self._sum(self, rhs, True, context)

    def __mul__(self, rhs: _ComplexArgument) raises -> Self:
        return Self._calculate(self, rhs, 2)

    def __rmul__(self, lhs: _ComplexArgument) raises -> Self:
        return Self._calculate(lhs, self, 2)

    def __imul__(mut self, rhs: _ComplexArgument) raises:
        self._assign(rhs, 2)

    def __truediv__(self, rhs: _ComplexArgument) raises -> Self:
        return Self._calculate(self, rhs, 3)

    def __rtruediv__(self, lhs: _ComplexArgument) raises -> Self:
        return Self._calculate(lhs, self, 3)

    def __itruediv__(mut self, rhs: _ComplexArgument) raises:
        self._assign(rhs, 3)

    def imag(self) -> Float:
        """The imaginary part.

        Returns:
            An independent Float.
        """
        return self._imag

    def real_format(self) -> FloatFormat:
        """The real part's format.

        Returns:
            The format.
        """
        return self._real.format()

    def imag_format(self) -> FloatFormat:
        """The imaginary part's format.

        Returns:
            The format.
        """
        return self._imag.format()

    def same_representation(self, other: Self) -> Bool:
        """Whether both components are the same representation.

        See `Float.same_representation`: signed zeros, formats and NaN count.

        Args:
            other: The other Complex.

        Returns:
            True when both components match.
        """
        return self._real.same_representation(other._real) and self._imag.same_representation(other._imag)

    def representation_cmp(self, other: Self) -> Int:
        """Compare representations in a strict total order: the real components
        by `Float.representation_cmp`, then the imaginary ones.

        Args:
            other: The other Complex.

        Returns:
            -1, 0 or 1; 0 exactly when `same_representation` holds.
        """
        var order = self._real.representation_cmp(other._real)
        return order if order else self._imag.representation_cmp(other._imag)

    def _hash_into(self, mut hash: _StableHash):
        """Feed both components to a stable hash (APNH-64)."""
        self._real._hash_into(hash)
        self._imag._hash_into(hash)

    def is_zero(self) -> Bool:
        """Whether both components are zero, of either sign.

        Returns:
            True for complex zero.
        """
        return self._real.is_zero() and self._imag.is_zero()

    def is_finite(self) -> Bool:
        """Whether both components are finite.

        Returns:
            True when neither is infinite or NaN.
        """
        return self._real.is_finite() and self._imag.is_finite()

    def is_infinite(self) -> Bool:
        """Whether either component is infinite.

        Returns:
            True when either is an infinity; `is_nan` may also be true.
        """
        return self._real.is_infinite() or self._imag.is_infinite()

    def is_nan(self) -> Bool:
        """Whether either component is NaN.

        Returns:
            True when either is NaN.
        """
        return self._real.is_nan() or self._imag.is_nan()

    def __bool__(self) -> Bool:
        return not self.is_zero()

    def conjugate(self) -> Self:
        """The complex conjugate.

        Returns:
            The same real part and the negated imaginary part, formats kept.
        """
        return Self(_real=self._real, _imag=-self._imag)

    def __eq__(self, rhs: Self) -> Bool:
        return self._real == rhs._real and self._imag == rhs._imag

    def __ne__(self, rhs: Self) -> Bool:
        return not self.__eq__(rhs)

    def _equals_real(self, rhs: _FloatArgument) raises -> Bool:
        return (
            self._imag.is_zero() and self._real._compare_input(rhs.value) == 0
        )

    def __eq__(self, rhs: _FloatArgument) raises -> Bool:
        return self._equals_real(rhs)

    def __ne__(self, rhs: _FloatArgument) raises -> Bool:
        return not self._equals_real(rhs)

    def write_to(self, mut writer: Some[Writer]):
        """Write `Complex(real, imag)` with each part as a Float prints.

        Args:
            writer: The destination.
        """
        writer.write("Complex(", self._real, ", ", self._imag, ")")
