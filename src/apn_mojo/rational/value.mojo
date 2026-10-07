"""Canonical exact fractions built on the shared Integer engine."""

from std.math import Absable
from std.builtin.int import IntableRaising
from std.builtin.bool import Boolable
from std.hashlib import Hasher

from ..integer.value import Integer, _unsigned_magnitude
from ..integer._gcd import _native_gcd
from ..batch._comparison import _ExactBatchComparison
from ..float._comparison import _FloatComparison
from ..complex._comparison import _ComplexComparison
from ..complex.value import Complex
from ..float.value import Float
from ..batch.mask import Mask
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..common._sizes import _checked_count, _checked_sum
from ._text import _rational_text
from ._json import _read_rational_json
from ..integer._conversion import (
    _conversion_sign,
    _conversion_gcd,
    _conversion_div_rem,
    _conversion_multiply,
    _conversion_power10,
)
from ..common._traits import _BatchElement
from ..integer.number_theory import gcd


def _coprime(a: Integer, b: Integer, mut budget: _ConversionBudget) raises -> Bool:
    """Whether a JSON fraction is in lowest terms.

    A counted budget charges the allocations of its own Euclid steps; without
    one, the library gcd gives the same answer much faster.
    """
    if budget.bounded_allocation():
        return _conversion_gcd(a, b, budget) == 1
    return gcd(a, b)._is_one()


struct Rational(
    Absable,
    Boolable,
    Equatable,
    Hashable,
    ImplicitlyCopyable,
    IntableRaising,
    Writable,
    _BatchElement,
):
    """An exact fraction, always in lowest terms with a positive denominator.

    Zero is `0/1`, and every result is canonical, so there is nothing to
    normalize. `Rational(8, -12)` is `-2/3`, and `Rational("1.25e-3")` is exactly
    `1/800`: text never passes through floating point.

    | Operation | Contract |
    |---|---|
    | `x + y`, `x - y`, `x * y`, `x / y` | Exact; division by zero raises |
    | `x ** n` | Signed integral exponent; a negative power takes the reciprocal |
    | `-x`, `abs(x)` | Exact |
    | `==`, `!=`, `<`, `<=`, `>`, `>=` | Exact comparison returning `Bool` |
    | `Bool(x)` | False only for zero |

    Integer, native integral and literal operands mix exactly in either order;
    Float operands give `Float`. `+=`, `-=`, `*=`, `/=` and `**=` leave the
    destination unchanged when they raise. Equal values hash equally, and an
    integral Rational hashes like the corresponding Integer.

    Limitations:
        A native integer cannot be the left operand of a comparison; write
        `value > native` or `Rational(native) < value`. Ordered comparisons can
        raise checked size errors, so `Rational` is not `Comparable`.
    """

    var _numerator: Integer
    var _denominator: Integer

    @implicit
    def __init__(out self, value: Int = 0):
        """A Rational from a native `Int`; `Rational()` is zero.

        Args:
            value: The value.
        """
        self._numerator = Integer(value)
        self._denominator = Integer(1)

    @implicit
    def __init__(out self, value: Integer):
        """A Rational equal to an Integer."""
        self._numerator = value
        self._denominator = Integer(1)

    @implicit
    def __init__(out self, value: IntLiteral):
        """A Rational from an integer literal of any width."""
        self._numerator = Integer(value)
        self._denominator = Integer(1)

    @implicit
    def __init__[dtype: DType](out self, value: SIMD[dtype, 1]):
        """A Rational from a native integral scalar of at most 64 bits."""
        self._numerator = Integer(value)
        self._denominator = Integer(1)

    def __init__(
        out self,
        text: String,
        *,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises:
        """Parse an exact fraction or decimal.

        Accepts integers, fractions such as `-12/+30`, and decimals such as `.125`,
        `1.` or `1.25e-3`. Decimal points and exponents cannot be combined with `/`.

        Args:
            text: The number.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Raises:
            When the text is not a valid number, naming the byte offset; when a
            denominator is zero; or when it exceeds `limits`.
        """
        self = Self.parse(
            text,
            allow_whitespace=allow_whitespace,
            allow_underscores=allow_underscores,
            limits=limits,
        )

    def __init__(
        out self, var numerator: Integer, var denominator: Integer
    ) raises:
        """The fraction `numerator / denominator`, reduced.

        Args:
            numerator: The numerator.
            denominator: The denominator, nonzero.

        Raises:
            When the denominator is zero.
        """
        if not denominator:
            raise Error(
                "Cannot construct Rational: denominator is 0; use a nonzero"
                " denominator. The destination is unchanged."
            )
        if not numerator:
            self = Self()
            return
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        var common = numerator._gcd(denominator)
        if common._is_one():
            # Already in lowest terms: x // 1 would only copy x.
            self._numerator = numerator^
            self._denominator = denominator^
            return
        self._numerator = numerator // common
        self._denominator = denominator // common

    @staticmethod
    def _placeholder() -> Self:
        return Self(0)

    def __init__(out self, *, copy: Self):
        self._numerator = copy._numerator
        self._denominator = copy._denominator

    def __init__(out self, value: Float) raises:
        """The exact value of a finite Float; see `Float.to_rational_exact`.

        Args:
            value: A finite Float; both zeros give 0.

        Raises:
            For infinity and NaN.
        """
        var exact = value.to_rational_exact()
        self._numerator = exact._numerator
        self._denominator = exact._denominator

    def __init__(out self, *, var _numerator: Integer, var _denominator: Integer):
        # Internal callers prove coprimality and a positive denominator. The
        # parts are moved in: a borrowed part's copy would cost an atomic
        # increment for a heap value, and the temporary's release a decrement.
        self._numerator = _numerator^
        self._denominator = _denominator^

    def numerator(self) -> Integer:
        """The numerator in lowest terms.

        Returns:
            An independent Integer carrying the sign.
        """
        return self._numerator

    def denominator(self) -> Integer:
        """The denominator in lowest terms.

        Returns:
            An independent positive Integer.
        """
        return self._denominator

    @staticmethod
    def parse(
        text: String,
        *,
        allow_whitespace: Bool = False,
        allow_underscores: Bool = False,
        limits: Optional[ConversionLimits] = None,
    ) raises -> Self:
        """Parse an exact fraction or decimal; the same contract as the text constructor.

        Args:
            text: The number.
            allow_whitespace: Accept surrounding ASCII whitespace.
            allow_underscores: Accept single underscores between digits.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The parsed Rational.

        Raises:
            When the text is not a valid number, or exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        budget.input(text.byte_length())
        budget.values(1)
        var form = _rational_text(
            text, allow_whitespace, allow_underscores, budget
        )
        var scale = form.scale(text)
        if form.zero:
            return Self()
        var numerator = Integer._parse_words(
            text,
            form.start,
            form.end,
            10,
            False,
            budget,
            skip_decimal_point=True,
        )
        var denominator = Integer(1)
        if form.denominator_start >= 0:
            denominator = Integer._parse_words(
                text,
                form.denominator_start,
                form.denominator_end,
                10,
                False,
                budget,
            )
        elif scale:
            var power = _conversion_power10(abs(scale), budget)
            if scale > 0:
                numerator = _conversion_multiply(numerator, power, budget)
            else:
                denominator = power
        var common = _conversion_gcd(numerator, denominator, budget)
        if common != 1:
            numerator, _ = _conversion_div_rem(numerator, common, budget)
            denominator, _ = _conversion_div_rem(denominator, common, budget)
        return Self(
            _numerator=_conversion_sign(numerator, form.negative, budget),
            _denominator=denominator,
        )

    def to_string(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """The canonical text `numerator/denominator`, omitting `/1`.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            Exact text, not a rounded decimal.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        budget.values(1)
        if not self._denominator._is_one():
            budget.output(1)
        var numerator = self._numerator._format_with_budget(
            10, False, False, budget
        )
        if self._denominator._is_one():
            return numerator
        var denominator = self._denominator._format_with_budget(
            10, False, False, budget
        )
        var count = _checked_count(
            _checked_sum(
                _checked_sum(
                    numerator.byte_length(), denominator.byte_length()
                ),
                1,
            ),
            1,
        )
        budget.string_allocation(count)
        return String(numerator, "/", denominator)

    def to_json(
        self, *, limits: Optional[ConversionLimits] = None
    ) raises -> String:
        """Write the version-1 JSON record of this Rational.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            `{"version":1,"family":"rational","numerator":"-7","denominator":"3"}`
            style compact JSON.

        Raises:
            When the output exceeds `limits`.
        """
        var budget = _ConversionBudget(limits)
        budget.values(1)
        comptime prefix: StaticString = (
            '{"version":1,"family":"rational","numerator":"'
        )
        comptime middle: StaticString = '","denominator":"'
        var framing = prefix.byte_length() + middle.byte_length() + 2
        budget.output(framing)
        var numerator = self._numerator._format_with_budget(
            10, False, False, budget
        )
        var denominator = self._denominator._format_with_budget(
            10, False, False, budget
        )
        var count = _checked_count(
            _checked_sum(
                _checked_sum(
                    numerator.byte_length(), denominator.byte_length()
                ),
                framing,
            ),
            1,
        )
        budget.string_allocation(count)
        return String(prefix, numerator, middle, denominator, '"}')

    @staticmethod
    def from_json(
        text: String, *, limits: Optional[ConversionLimits] = None
    ) raises -> Self:
        """Read a Rational from its version-1 JSON record.

        The components must already be canonical: coprime, a positive denominator,
        and zero as `0/1`.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The Rational the record holds.

        Raises:
            When the text is not exactly that schema, naming the byte offset.
        """
        var budget = _ConversionBudget(limits)
        var numerator_text, denominator_text = _read_rational_json(text, budget)
        return Self._from_json_components(
            numerator_text, denominator_text, budget
        )

    @staticmethod
    def _from_json_components(
        numerator_text: String,
        denominator_text: String,
        mut budget: _ConversionBudget,
    ) raises -> Self:
        # Syntax and denominator positivity are checked by the JSON reader.
        var negative = numerator_text.as_bytes()[0] == 45
        var numerator = Integer._parse_words(
            numerator_text,
            Int(negative),
            numerator_text.byte_length(),
            10,
            False,
            budget,
        )
        var denominator = Integer._parse_json_digits(denominator_text, budget)
        if (not numerator and not denominator._is_one()) or not _coprime(
            numerator, denominator, budget
        ):
            raise Error(
                "Invalid rational interchange: noncanonical fraction; use"
                " coprime numerator and denominator, write zero as 0/1, or use"
                " Rational(text) to normalize ordinary fraction text. The"
                " destination is unchanged."
            )
        return Self(
            _numerator=_conversion_sign(numerator, negative, budget),
            _denominator=denominator,
        )

    def __bool__(self) -> Bool:
        return self._numerator != 0

    def sign(self) -> Int:
        """The sign: -1, 0 or 1.

        Returns:
            The sign of the numerator.
        """
        return self._numerator.sign()

    def __add__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 0)

    def __sub__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 1)

    def __mul__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 2)

    def __truediv__(self, rhs: Complex) raises -> Complex:
        return Complex._calculate(self, rhs, 3)

    def __neg__(self) -> Self:
        return Self(_numerator=-self._numerator, _denominator=self._denominator)

    def __abs__(self) -> Self:
        return -self if self._numerator < 0 else self

    def __eq__[C: _ComplexComparison](self, rhs: C) raises -> Bool:
        return rhs._equals_real(self)

    def __ne__[C: _ComplexComparison](self, rhs: C) raises -> Bool:
        return not rhs._equals_real(self)

    def __eq__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 0)

    def __ne__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 1)

    def __lt__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 4)

    def __le__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 5)

    def __gt__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 2)

    def __ge__[F: _FloatComparison](self, rhs: F) raises -> Bool:
        return rhs._compare_rational(self, 3)

    def __eq__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 0)

    def __ne__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 1)

    def __lt__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 4)

    def __le__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 5)

    def __gt__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 2)

    def __ge__[B: _ExactBatchComparison](self, rhs: B) raises -> Mask:
        return rhs._compare_rational(self, 3)

    def __eq__(self, rhs: Self) -> Bool:
        return (
            self._numerator == rhs._numerator
            and self._denominator == rhs._denominator
        )

    def __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    def __hash__[H: Hasher](self, mut hasher: H):
        self._numerator.__hash__(hasher)
        if not self._denominator._is_one():
            self._denominator.__hash__(hasher)

    def _compare(self, rhs: Self) raises -> Int:
        if _inline_parts(self, rhs):
            # Cross products of inline parts stay below 2**126.
            var left = Int128(self._numerator._storage[Int64]) * Int128(rhs._denominator._storage[Int64])
            var right = Int128(rhs._numerator._storage[Int64]) * Int128(self._denominator._storage[Int64])
            return -1 if left < right else Int(left > right)
        if self.sign() != rhs.sign():
            return -1 if self.sign() < rhs.sign() else 1
        if self._denominator == rhs._denominator:
            return self._numerator._compare(rhs._numerator)
        var a = abs(self._numerator)
        var b = self._denominator
        var c = abs(rhs._numerator)
        var d = rhs._denominator
        var direction = -1 if self.sign() < 0 else 1
        # Continued fractions avoid potentially much larger cross-products.
        while True:
            var left, left_rem = a._div_rem_trunc(b)
            var right, right_rem = c._div_rem_trunc(d)
            if left != right:
                return direction * left._compare(right)
            if not left_rem or not right_rem:
                return direction * (Int(left_rem != 0) - Int(right_rem != 0))
            a = b
            b = left_rem
            c = d
            d = right_rem
            direction = -direction

    def __lt__(self, rhs: Self) raises -> Bool:
        return self._compare(rhs) < 0

    def __le__(self, rhs: Self) raises -> Bool:
        return self._compare(rhs) <= 0

    def __gt__(self, rhs: Self) raises -> Bool:
        return self._compare(rhs) > 0

    def __ge__(self, rhs: Self) raises -> Bool:
        return self._compare(rhs) >= 0

    def _add(self, rhs: Self, subtract: Bool) raises -> Self:
        if _inline_parts(self, rhs):
            return _native_sum(self, rhs, subtract)
        if self._denominator == rhs._denominator:
            # Statements, not a conditional expression, which copies its
            # heap result once more.
            var numerator: Integer
            if subtract:
                numerator = self._numerator - rhs._numerator
            else:
                numerator = self._numerator + rhs._numerator
            return Self(numerator^, self._denominator)
        var common = self._denominator._gcd(rhs._denominator)
        var coprime = common._is_one()
        var left_scale = rhs._denominator if coprime else rhs._denominator // common
        var right_scale = self._denominator if coprime else self._denominator // common
        var right = rhs._numerator * right_scale
        var numerator = self._numerator * left_scale
        # Statements, not a conditional expression, which copies its heap
        # result once more. (In place, the exact-size product would regrow.)
        if subtract:
            numerator = numerator - right
        else:
            numerator = numerator + right
        if not numerator:
            return Self()
        if coprime:
            # Coprime denominators leave the sum in lowest terms: a prime
            # dividing one denominator divides neither the other nor its own
            # numerator, so it cannot divide the sum's numerator.
            return Self(_numerator=numerator^, _denominator=right_scale * rhs._denominator)
        # Any remaining common factor divides the original denominator gcd.
        var remaining = numerator._gcd(common)
        return Self(
            _numerator=numerator // remaining,
            _denominator=right_scale * (rhs._denominator // remaining),
        )

    def __add__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 0)

    def __add__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self + Self(rhs)

    def __radd__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 0)

    def __radd__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) + self

    def __sub__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 1)

    def __sub__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self - Self(rhs)

    def __rsub__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 1)

    def __rsub__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) - self

    def __mul__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 2)

    def __mul__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self * Self(rhs)

    def __rmul__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 2)

    def __rmul__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) * self

    def __truediv__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(self, rhs, 3)

    def __truediv__[
        dtype: DType
    ](
        self, rhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return self / Self(rhs)

    def __rtruediv__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Float where dtype.is_floating_point():
        return Float._from_arguments(lhs, self, 3)

    def __rtruediv__[
        dtype: DType
    ](
        self, lhs: SIMD[dtype, 1]
    ) raises -> Self where not dtype.is_floating_point():
        return Self(lhs) / self

    # Keep literals out of the pinned compiler's native-SIMD coercion path.
    def __add__(self, rhs: IntLiteral) raises -> Self:
        return self + Self(rhs)

    def __add__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 0)

    def __radd__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) + self

    def __radd__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 0)

    def __sub__(self, rhs: IntLiteral) raises -> Self:
        return self - Self(rhs)

    def __sub__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 1)

    def __rsub__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) - self

    def __rsub__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 1)

    def __mul__(self, rhs: IntLiteral) raises -> Self:
        return self * Self(rhs)

    def __mul__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 2)

    def __rmul__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) * self

    def __rmul__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 2)

    def __truediv__(self, rhs: IntLiteral) raises -> Self:
        return self / Self(rhs)

    def __truediv__(self, rhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(self, rhs, 3)

    def __rtruediv__(self, lhs: IntLiteral) raises -> Self:
        return Self(lhs) / self

    def __rtruediv__(self, lhs: FloatLiteral) raises -> Float:
        return Float._from_arguments(lhs, self, 3)

    def __add__(self, rhs: Self) raises -> Self:
        return self._add(rhs, False)

    def __sub__(self, rhs: Self) raises -> Self:
        return self._add(rhs, True)

    def __mul__(self, rhs: Self) raises -> Self:
        if not self or not rhs:
            return Self()
        if _inline_parts(self, rhs):
            return _native_product(self, rhs)
        if self._numerator == rhs._numerator and self._denominator == rhs._denominator:
            # A square: the parts of a fraction in lowest terms are coprime,
            # and so are their squares.
            return Self(
                _numerator=self._numerator * self._numerator,
                _denominator=self._denominator * self._denominator,
            )
        var left_cancel = self._numerator._gcd(rhs._denominator)
        var right_cancel = rhs._numerator._gcd(self._denominator)
        if left_cancel._is_one() and right_cancel._is_one():
            # Coprime cross pairs, the usual case: x // 1 would only copy x
            # (an atomic increment for a heap part, and a decrement after).
            return Self(
                _numerator=self._numerator * rhs._numerator,
                _denominator=self._denominator * rhs._denominator,
            )
        return Self(
            _numerator=(self._numerator // left_cancel)
            * (rhs._numerator // right_cancel),
            _denominator=(self._denominator // right_cancel)
            * (rhs._denominator // left_cancel),
        )

    def __truediv__(self, rhs: Self) raises -> Self:
        if not rhs:
            raise Error(
                "Cannot divide Rational: divisor is 0; use a nonzero divisor."
                " The destination is unchanged."
            )
        if not self:
            return Self()
        if _inline_parts(self, rhs):
            return _native_quotient(self, rhs)
        var numerators = self._numerator._gcd(rhs._numerator)
        var denominators = self._denominator._gcd(rhs._denominator)
        var numerator: Integer
        var denominator: Integer
        if numerators._is_one() and denominators._is_one():
            # Coprime pairs, the usual case: no divisions by one, which only
            # copy their dividends.
            numerator = self._numerator * rhs._denominator
            denominator = self._denominator * rhs._numerator
        else:
            numerator = (self._numerator // numerators) * (
                rhs._denominator // denominators
            )
            denominator = (self._denominator // denominators) * (
                rhs._numerator // numerators
            )
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return Self(_numerator=numerator^, _denominator=denominator^)

    def __radd__(self, lhs: Integer) raises -> Self:
        return Self(lhs) + self

    def __rsub__(self, lhs: Integer) raises -> Self:
        return Self(lhs) - self

    def __rmul__(self, lhs: Integer) raises -> Self:
        return Self(lhs) * self

    def __rtruediv__(self, lhs: Integer) raises -> Self:
        return Self(lhs) / self

    def __iadd__(mut self, var rhs: Self) raises:
        self = self + rhs

    def __isub__(mut self, var rhs: Self) raises:
        self = self - rhs

    def __imul__(mut self, var rhs: Self) raises:
        self = self * rhs

    def __itruediv__(mut self, var rhs: Self) raises:
        self = self / rhs

    def __pow__(self, exponent: Integer) raises -> Self:
        if exponent < 0 and not self:
            raise Error(
                "Cannot raise zero Rational to a negative power; use a nonzero"
                " base or a nonnegative exponent. The destination is unchanged."
            )
        var magnitude = abs(exponent)
        var numerator = self._numerator**magnitude
        var denominator = self._denominator**magnitude
        if exponent < 0:
            var previous = numerator
            numerator = denominator
            denominator = previous
            if denominator < 0:
                numerator = -numerator
                denominator = -denominator
        return Self(_numerator=numerator, _denominator=denominator)

    def __ipow__(mut self, exponent: Integer) raises:
        self = self**exponent

    def floor(self) raises -> Integer:
        """Round toward negative infinity.

        Returns:
            The largest Integer not above the value; `Rational(-7, 3).floor()` is -3.

        Raises:
            Only on a checked size error.
        """
        return self._numerator // self._denominator

    def ceil(self) raises -> Integer:
        """Round toward positive infinity.

        Returns:
            The smallest Integer not below the value.

        Raises:
            Only on a checked size error.
        """
        return -((-self._numerator) // self._denominator)

    def trunc(self) raises -> Integer:
        """Round toward zero.

        Returns:
            The integer part.

        Raises:
            Only on a checked size error.
        """
        var quotient, _ = self._numerator._div_rem_trunc(self._denominator)
        return quotient

    def is_integer(self) -> Bool:
        """Whether the value is an integer (Python's `Fraction.is_integer`).

        Returns:
            True when the denominator is 1.
        """
        return self._denominator._is_one()

    def round(self) raises -> Integer:
        """Round to the nearest Integer, and a half to the even neighbour.

        Returns:
            The nearest Integer; `Rational(5, 2)` and `Rational(3, 2)` both give
            2, and `Rational(-5, 2)` gives -2.

        Raises:
            Only on a checked size error.
        """
        var whole = self._numerator // self._denominator
        var twice = (self._numerator % self._denominator) << 1
        if twice > self._denominator or (
            twice == self._denominator and (whole._word(0) & 1) != 0
        ):
            return whole + 1
        return whole

    def to_integer_exact(self) raises -> Integer:
        """Convert an integral value to Integer.

        Returns:
            The Integer equal to this value.

        Raises:
            When the value has a fractional part; choose `floor`, `ceil` or `trunc`.
        """
        if not self._denominator._is_one():
            raise Error(
                "Cannot convert Rational exactly to an integer: value has a"
                " fractional part; keep it as Rational or explicitly choose"
                " floor(), ceil(), or trunc()."
            )
        return self._numerator

    def to_native_exact[dtype: DType](self) raises -> SIMD[dtype, 1]:
        """Convert an integral value to a native integer type, exactly.

        Parameters:
            dtype: The target: `DType.int` or a signed or unsigned 8- to 64-bit integer.

        Returns:
            The same value in the native type.

        Raises:
            When the value has a fractional part or does not fit the type.
        """
        return self.to_integer_exact().to_native_exact[dtype]()

    def __int__(self) raises -> Int:
        return self.to_native_exact[DType.int]()

    def write_to(self, mut writer: Some[Writer]):
        """Write the canonical text, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write(self._numerator)
        if not self._denominator._is_one():
            writer.write("/", self._denominator)


def pow_rational(base: Rational, exponent: Integer) raises -> Rational:
    """An exact power for a signed integral exponent.

    A negative exponent takes the exact reciprocal; every zero exponent, including
    `0 ** 0`, gives one. Integer and native bases widen exactly.

    Args:
        base: The base.
        exponent: The exponent, of any size.

    Returns:
        `base ** exponent` as a Rational, even when integral.

    Raises:
        When `base` is zero and `exponent` negative.
    """
    return base**exponent


@always_inline
def _inline_parts(a: Rational, b: Rational) -> Bool:
    """Whether all four parts are inline Int64 values, so that sums, products
    and quotients fit 128-bit integers."""
    return (
        a._numerator._storage.isa[Int64]()
        and a._denominator._storage.isa[Int64]()
        and b._numerator._storage.isa[Int64]()
        and b._denominator._storage.isa[Int64]()
    )


def _signed(magnitude: UInt128, negative: Bool) raises -> Integer:
    """An Integer from a magnitude below 2**127 and a sign."""
    var value = Int128(magnitude)
    return Integer._from_int128(-value if negative else value)


def _native_sum(a: Rational, b: Rational, subtract: Bool) raises -> Rational:
    """a + b or a - b for inline parts, in 128-bit integers, with the general
    path's reductions: by the denominators' gcd, then by what the sum's
    numerator shares with it. The numerator stays below 2**127, the
    denominator below 2**126. A gcd of one skips its divisions, each a
    hardware divide."""
    var d1 = UInt64(a._denominator._storage[Int64])
    var d2 = UInt64(b._denominator._storage[Int64])
    var n2 = Int128(b._numerator._storage[Int64])
    var common = _native_gcd(d1, d2)
    var left = d2
    var right = d1
    if common != 1:
        left //= common
        right //= common
    var numerator = Int128(a._numerator._storage[Int64]) * Int128(left) + (
        -n2 if subtract else n2
    ) * Int128(right)
    if not numerator:
        return Rational()
    var denominator = UInt128(right) * UInt128(d2)
    var negative = numerator < 0
    var magnitude = UInt128(-numerator if negative else numerator)
    if common != 1:
        var remaining = UInt128(_native_gcd(UInt64(magnitude % UInt128(common)), common))
        if remaining != 1:
            magnitude //= remaining
            denominator //= remaining
    return Rational(
        _numerator=_signed(magnitude, negative), _denominator=_signed(denominator, False)
    )


def _native_product(a: Rational, b: Rational) raises -> Rational:
    """a * b for nonzero inline parts: each numerator cancels against the other
    denominator, as in the general path; a square needs no cancelling, since
    the parts of a fraction in lowest terms are coprime."""
    var n1 = a._numerator._storage[Int64]
    var n2 = b._numerator._storage[Int64]
    var d1 = UInt64(a._denominator._storage[Int64])
    var d2 = UInt64(b._denominator._storage[Int64])
    var m1 = _unsigned_magnitude(n1)
    var m2 = _unsigned_magnitude(n2)
    if n1 != n2 or d1 != d2:
        var left = _native_gcd(m1, d2)
        var right = _native_gcd(m2, d1)
        if left != 1:
            m1 //= left
            d2 //= left
        if right != 1:
            m2 //= right
            d1 //= right
    return Rational(
        _numerator=_signed(UInt128(m1) * UInt128(m2), (n1 < 0) != (n2 < 0)),
        _denominator=_signed(UInt128(d1) * UInt128(d2), False),
    )


def _native_quotient(a: Rational, b: Rational) raises -> Rational:
    """a / b for nonzero inline parts: the numerators cancel against each
    other and so do the denominators, as in the general path; the sign goes
    to the numerator."""
    var n1 = a._numerator._storage[Int64]
    var n2 = b._numerator._storage[Int64]
    var d1 = UInt64(a._denominator._storage[Int64])
    var d2 = UInt64(b._denominator._storage[Int64])
    var m1 = _unsigned_magnitude(n1)
    var m2 = _unsigned_magnitude(n2)
    var numerators = _native_gcd(m1, m2)
    var denominators = _native_gcd(d1, d2)
    if numerators != 1:
        m1 //= numerators
        m2 //= numerators
    if denominators != 1:
        d1 //= denominators
        d2 //= denominators
    return Rational(
        _numerator=_signed(UInt128(m1) * UInt128(d2), (n1 < 0) != (n2 < 0)),
        _denominator=_signed(UInt128(d1) * UInt128(m2), False),
    )
