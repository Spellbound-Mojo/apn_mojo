"""Exact complex numbers: two Rational parts, exact arithmetic, text and JSON."""

from std.hashlib import Hasher
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext
from ..complex.value import Complex
from ..complex.context import ComplexContext
from ..common.conversion import ConversionLimits, _ConversionBudget
from ..common._json import _JSONReader
from ..common._sizes import _checked_sum
from ..common._stable_hash import _StableHash
from ..rational._json import _read_rational_json_record


struct ExactComplex(Equatable, Hashable, ImplicitlyCopyable, Writable):
    """A complex number with exact Rational parts, `real + imag i`.

    Arithmetic is exact: `+`, `-`, `*` and `/` never round, and division by
    zero raises. A Rational, Integer or integer literal operand on either side
    is exact too. There is no implicit conversion from those types, so a
    package-level function never becomes ambiguous; write `ExactComplex(1, 2)`.

    | Operation | Contract |
    |---|---|
    | `z + w`, `z - w`, `z * w` | Exact |
    | `z / w` | Exact; raises for `w = 0` |
    | `-z`, `+z` | Exact |
    | `==`, `!=`, `hash` | Equal values compare and hash equal |

    Printing writes `ExactComplex(1/2, 3)`, which the text constructor reads
    back; JSON holds two canonical Rational records.
    """

    var _real: Rational
    var _imag: Rational

    def __init__(out self, real: Rational = Rational(0), imag: Rational = Rational(0)):
        """The number `real + imag i`.

        Args:
            real: The real part.
            imag: The imaginary part.
        """
        self._real = real
        self._imag = imag

    def __init__(out self, text: String) raises:
        """Parse `ExactComplex(re, im)` as printing writes it; each part is
        Rational text such as `-3/4`.

        Args:
            text: The text.

        Raises:
            When the text is not in that form or a part is not a fraction.
        """
        var source = String(text.strip())
        if not source.startswith("ExactComplex(") or not source.endswith(")"):
            raise Error(String("Cannot parse ExactComplex from '", text, "'; write ExactComplex(re, im)."))
        var inner = String(source.removeprefix("ExactComplex(").removesuffix(")"))
        var parts = inner.split(",")
        if len(parts) != 2:
            raise Error(String("Cannot parse ExactComplex from '", text, "'; write ExactComplex(re, im)."))
        self._real = Rational(String(parts[0].strip()))
        self._imag = Rational(String(parts[1].strip()))

    def __init__(out self, *, copy: Self):
        self._real = copy._real
        self._imag = copy._imag

    def real(self) -> Rational:
        """The real part.

        Returns:
            The real part.
        """
        return self._real

    def imag(self) -> Rational:
        """The imaginary part.

        Returns:
            The imaginary part.
        """
        return self._imag

    def is_real(self) -> Bool:
        """Whether the imaginary part is zero.

        Returns:
            True for a real number.
        """
        return self._imag.sign() == 0

    def __eq__(self, rhs: Self) -> Bool:
        return self._real == rhs._real and self._imag == rhs._imag

    def __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    def __hash__[H: Hasher](self, mut hasher: H):
        self._real.__hash__(hasher)
        self._imag.__hash__(hasher)

    def _hash_into(self, mut hash: _StableHash):
        hash.integer(self._real.numerator())
        hash.integer(self._real.denominator())
        hash.integer(self._imag.numerator())
        hash.integer(self._imag.denominator())

    def __neg__(self) raises -> Self:
        return Self(-self._real, -self._imag)

    def __pos__(self) -> Self:
        return self

    def __add__(self, rhs: Self) raises -> Self:
        return Self(self._real + rhs._real, self._imag + rhs._imag)

    def __add__(self, rhs: Rational) raises -> Self:
        return Self(self._real + rhs, self._imag)

    def __radd__(self, lhs: Rational) raises -> Self:
        return Self(lhs + self._real, self._imag)

    def __sub__(self, rhs: Self) raises -> Self:
        return Self(self._real - rhs._real, self._imag - rhs._imag)

    def __sub__(self, rhs: Rational) raises -> Self:
        return Self(self._real - rhs, self._imag)

    def __rsub__(self, lhs: Rational) raises -> Self:
        return Self(lhs - self._real, -self._imag)

    def __mul__(self, rhs: Self) raises -> Self:
        return Self(
            self._real * rhs._real - self._imag * rhs._imag,
            self._real * rhs._imag + self._imag * rhs._real,
        )

    def __mul__(self, rhs: Rational) raises -> Self:
        return Self(self._real * rhs, self._imag * rhs)

    def __rmul__(self, lhs: Rational) raises -> Self:
        return Self(lhs * self._real, lhs * self._imag)

    def __truediv__(self, rhs: Self) raises -> Self:
        var norm = rhs._real * rhs._real + rhs._imag * rhs._imag
        if norm.sign() == 0:
            raise Error("division by zero: an ExactComplex divisor must be nonzero.")
        return Self(
            (self._real * rhs._real + self._imag * rhs._imag) / norm,
            (self._imag * rhs._real - self._real * rhs._imag) / norm,
        )

    def __truediv__(self, rhs: Rational) raises -> Self:
        if rhs.sign() == 0:
            raise Error("division by zero: an ExactComplex divisor must be nonzero.")
        return Self(self._real / rhs, self._imag / rhs)

    def __rtruediv__(self, lhs: Rational) raises -> Self:
        return Self(lhs, Rational(0)) / self

    def to_complex(self, *, context: Optional[ComplexContext] = None) raises -> Complex:
        """Each part rounded once to a Complex.

        Args:
            context: The component formats, rounding modes and traps; by
                default 128 bits each, to nearest-even.

        Returns:
            The rounded Complex.

        Raises:
            On a trapped condition.
        """
        var c = context.value() if context else ComplexContext(ArithmeticContext())
        return Complex(Float(self._real, context=c.real()), Float(self._imag, context=c.imag()))

    def write_to(self, mut writer: Some[Writer]):
        """Write `ExactComplex(re, im)`, which the text constructor reads.

        Args:
            writer: The destination.
        """
        writer.write("ExactComplex(", self._real, ", ", self._imag, ")")

    def to_json(self, *, limits: Optional[ConversionLimits] = None) raises -> String:
        """Write the version-1 JSON record: two canonical Rational records.

        Args:
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            `{"version":1,"family":"exact_complex","real":{...},"imag":{...}}`.

        Raises:
            When the output exceeds `limits`.
        """
        var real = self._real.to_json(limits=limits)
        var imag = self._imag.to_json(limits=limits)
        var budget = _ConversionBudget(limits)
        comptime prefix: StaticString = '{"version":1,"family":"exact_complex","real":'
        comptime middle: StaticString = ',"imag":'
        budget.output(_checked_sum(_checked_sum(prefix.byte_length() + middle.byte_length() + 1, real.byte_length()), imag.byte_length()))
        return String(prefix, real, middle, imag, "}")

    @staticmethod
    def from_json(text: String, *, limits: Optional[ConversionLimits] = None) raises -> Self:
        """Read an ExactComplex from its version-1 JSON record.

        Args:
            text: The JSON record.
            limits: Optional per-call conversion limits; see `ConversionLimits`.

        Returns:
            The ExactComplex the record holds.

        Raises:
            When the text is not exactly that schema, naming the byte offset,
            or a part is not a canonical fraction.
        """
        var budget = _ConversionBudget(limits)
        budget.input(text.byte_length())
        budget.values(1)
        var reader = _JSONReader(text, budget, "Invalid exact_complex interchange at byte ")
        reader.expect(123, "provide an object produced by ExactComplex.to_json()")
        var seen = 0
        var parts = List[Tuple[String, String]]()
        var flags = List[Int]()
        while True:
            var key = reader.field_name()
            var flag = 0
            if key == "version":
                flag = 1
            elif key == "family":
                flag = 2
            elif key == "real":
                flag = 4
            elif key == "imag":
                flag = 8
            else:
                reader.fail("unknown field", "use version, family, real and imag from ExactComplex.to_json()")
            reader.mark_field(seen, flag)
            if flag == 1:
                reader.version_one("use ExactComplex interchange version 1")
            elif flag == 2:
                reader.family("exact_complex", "use family exact_complex with ExactComplex.from_json()")
            else:
                parts.append(_read_rational_json_record(reader))
                flags.append(flag)
            if reader.end_object("separate fields with commas and close the object with }"):
                break
        reader.require_fields(seen, 15, "include version, family, real and imag")
        reader.finish()
        budget = reader.budget
        var first = Rational._from_json_components(parts[0][0], parts[0][1], budget)
        var second = Rational._from_json_components(parts[1][0], parts[1][1], budget)
        if flags[0] == 4:
            return Self(first, second)
        return Self(second, first)

