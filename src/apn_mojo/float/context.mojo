"""Explicit numerical choices. No global state or execution controls."""

from ..integer._limits import _MAX_RESULT_BITS
from .status import NumericStatus


struct RoundingMode(Equatable, ImplicitlyCopyable, Writable):
    """How an inexact result rounds: one of five named constants.

    `nearest_even` (the default) rounds to the nearest value, and a tie to the even
    one; `toward_zero`, `toward_positive`, `toward_negative` and `away_from_zero`
    are directed. At the boundary between zero and the smallest value, a tie
    chooses zero.
    """

    var _code: Int

    def __init__(out self, *, _code: Int):
        self._code = _code

    comptime nearest_even = Self(_code=0)
    comptime toward_zero = Self(_code=1)
    comptime toward_positive = Self(_code=2)
    comptime toward_negative = Self(_code=3)
    comptime away_from_zero = Self(_code=4)

    def __eq__(self, other: Self) -> Bool:
        return self._code == other._code

    def _validate(self) raises:
        if self._code < 0 or self._code > 4:
            raise Error(
                "Invalid rounding mode; use a named RoundingMode constant. The"
                " destination is unchanged."
            )

    def write_to(self, mut writer: Some[Writer]):
        """Write the constant's name, such as `nearest_even`.

        Args:
            writer: The destination.
        """
        if self._code == 0:
            writer.write("nearest_even")
        elif self._code == 1:
            writer.write("toward_zero")
        elif self._code == 2:
            writer.write("toward_positive")
        elif self._code == 3:
            writer.write("toward_negative")
        elif self._code == 4:
            writer.write("away_from_zero")
        else:
            writer.write("invalid_rounding_mode")


struct FloatFormat(Equatable, ImplicitlyCopyable, Writable):
    """A binary precision and inclusive exponent bounds.

    A finite nonzero value in the format is `(-1)**s * m * 2**(e - p)` with `m` of
    exactly `p` bits and `emin <= e <= emax`; the smallest positive value is
    `2**(emin - 1)`. There are no subnormals. Making a format allocates nothing.
    """

    comptime MAX_PRECISION = _MAX_RESULT_BITS - 2
    comptime DEFAULT_EMIN = -(Int(1) << 62)
    comptime DEFAULT_EMAX = (Int(1) << 62) - 1
    var _precision: Int
    var _emin: Int
    var _emax: Int
    # Internal exact formats: 0 for an ordinary format; otherwise results keep
    # every bit, and the value is 1 + the rounding mode applied at the end
    # (it decides the sign of zeros from exact cancellation).
    var _exact: Int

    def __init__(
        out self,
        precision: Int = 128,
        *,
        emin: Int = Self.DEFAULT_EMIN,
        emax: Int = Self.DEFAULT_EMAX,
    ) raises:
        """A format with `precision` bits and the given exponent bounds.

        Args:
            precision: Significand bits, from 1 to `FloatFormat.MAX_PRECISION`.
            emin: The smallest normalized exponent.
            emax: The largest normalized exponent, at least `emin`.

        Raises:
            When the precision or bounds are out of range.
        """
        if precision < 1 or precision > Self.MAX_PRECISION:
            raise Error(
                String(
                    "Cannot create FloatFormat with precision ",
                    precision,
                    "; choose 1 through ",
                    Self.MAX_PRECISION,
                    (
                        " significand bits within supported addressable"
                        " storage. The destination is unchanged."
                    ),
                )
            )
        if emin > emax:
            raise Error(
                String(
                    "Cannot create FloatFormat with emin ",
                    emin,
                    " greater than emax ",
                    emax,
                    "; choose emin <= emax. The destination is unchanged.",
                )
            )
        self._precision = precision
        self._emin = emin
        self._emax = emax
        self._exact = 0

    @staticmethod
    def binary64() raises -> Self:
        """IEEE 754 binary64 (double): 53 bits, exponents -1021 through 1024.

        It covers the normal range of a `Float64`; there are no subnormals, so
        the smallest positive value is `2**-1022`.

        Returns:
            The format.

        Raises:
            Never in practice; the bounds are valid.
        """
        return Self(53, emin=-1021, emax=1024)

    @staticmethod
    def binary32() raises -> Self:
        """IEEE 754 binary32 (single): 24 bits, exponents -125 through 128.

        It covers the normal range of a `Float32`; there are no subnormals, so
        the smallest positive value is `2**-126`.

        Returns:
            The format.

        Raises:
            Never in practice; the bounds are valid.
        """
        return Self(24, emin=-125, emax=128)

    def precision(self) -> Int:
        """The significand precision in bits.

        Returns:
            The precision.
        """
        return self._precision

    def __init__(out self, *, _validated: Tuple[Int, Int, Int]):
        # Packed storage contains formats already validated at construction.
        self._precision = _validated[0]
        self._emin = _validated[1]
        self._emax = _validated[2]
        self._exact = 0

    @staticmethod
    def _exact_format(rounding: RoundingMode, precision: Int = 1) -> Self:
        """An exact working format: arithmetic keeps every bit of each result
        and raises when a result is not a finite binary fraction."""
        var result = Self(_validated=(precision, Self.DEFAULT_EMIN, Self.DEFAULT_EMAX))
        result._exact = rounding._code + 1
        return result

    def _is_exact(self) -> Bool:
        return self._exact != 0

    def _exact_rounding(self) -> RoundingMode:
        return RoundingMode(_code=self._exact - 1)

    def _with_precision(self, precision: Int) -> Self:
        var result = self
        result._precision = precision
        return result

    def emin(self) -> Int:
        """The smallest normalized exponent.

        Returns:
            `emin`.
        """
        return self._emin

    def emax(self) -> Int:
        """The largest normalized exponent.

        Returns:
            `emax`.
        """
        return self._emax

    def __eq__(self, other: Self) -> Bool:
        return (
            self._precision == other._precision
            and self._emin == other._emin
            and self._emax == other._emax
            and self._exact == other._exact
        )

    def write_to(self, mut writer: Some[Writer]):
        """Write the precision and bounds.

        Args:
            writer: The destination.
        """
        writer.write(
            "FloatFormat(precision=",
            self._precision,
            ", emin=",
            self._emin,
            ", emax=",
            self._emax,
            ")",
        )


struct ArithmeticContext(ImplicitlyCopyable, Writable):
    """The numerical choices for a rounded operation: format, rounding mode,
    traps, and the budget of functions whose cost depends on their input.

    A context is a value passed to each call; there is no global or thread-local
    context, and making one changes nothing elsewhere. A trapped condition makes the
    call raise, naming the condition, and nothing is published. Math functions
    report conditions only through traps: there is no status argument or sticky
    flag. NaN results and exact special results are not inexact, and division by
    zero signals only divide-by-zero.

    A correctly rounded transcendental function such as `exp` works at a higher
    precision until its result is certain; `max_precision` bounds that working
    precision, and a function that would need more raises an error naming it.
    """

    var _format: FloatFormat
    var _rounding: RoundingMode
    var _traps: Int
    var _max_precision: Optional[Int]

    def __init__(out self, *, _placeholder: FloatFormat):
        # A context nothing reads, for an absent slot: built without the
        # validation that makes the public constructor raise.
        self._format = _placeholder
        self._rounding = RoundingMode.nearest_even
        self._traps = 0
        self._max_precision = None

    def __init__(out self, *, _format_of: FloatFormat):
        # `ArithmeticContext(format=f)` for a valid f, such as an operand's:
        # the same context without the Optional argument and the checks of
        # rounding and max_precision, which cost a function called without a
        # context about 100 instructions.
        self._format = _format_of
        self._rounding = _format_of._exact_rounding() if _format_of._is_exact() else RoundingMode.nearest_even
        self._traps = 0
        self._max_precision = None

    def __init__(
        out self,
        *,
        format: Optional[FloatFormat] = None,
        rounding: RoundingMode = RoundingMode.nearest_even,
        trap_inexact: Bool = False,
        trap_underflow: Bool = False,
        trap_overflow: Bool = False,
        trap_divide_by_zero: Bool = False,
        trap_invalid: Bool = False,
        max_precision: Optional[Int] = None,
    ) raises:
        """A context; every argument is optional.

        Args:
            format: The output format; `FloatFormat()` (128 bits) by default.
            rounding: The rounding mode.
            trap_inexact: Raise when a result is rounded.
            trap_underflow: Raise when a result is tiny and inexact.
            trap_overflow: Raise when a result exceeds the format's range.
            trap_divide_by_zero: Raise when a finite value is divided by zero.
            trap_invalid: Raise when a result is an invalid NaN.
            max_precision: The largest working precision, in bits, that a
                correctly rounded function may use; by default
                `max(8 * p, p + 4096)` for an output precision `p`.

        Raises:
            When `rounding` is not one of the named constants, or
            `max_precision` is below 1.
        """
        rounding._validate()
        if max_precision and max_precision.value() < 1:
            raise Error(
                "Cannot create ArithmeticContext with max_precision below 1;"
                " choose a positive number of bits. The destination is unchanged."
            )
        self._max_precision = max_precision
        self._format = format.value() if format else FloatFormat()
        # An exact working format carries the rounding mode of its final result.
        self._rounding = self._format._exact_rounding() if self._format._is_exact() else rounding
        self._traps = (
            Int(trap_inexact)
            | (Int(trap_underflow) << 1)
            | (Int(trap_overflow) << 2)
            | (Int(trap_divide_by_zero) << 3)
            | (Int(trap_invalid) << 4)
        )

    def format(self) -> FloatFormat:
        """The output format.

        Returns:
            The format.
        """
        return self._format

    def rounding(self) -> RoundingMode:
        """The rounding mode.

        Returns:
            The mode.
        """
        return self._rounding

    def max_precision(self) -> Optional[Int]:
        """The budget of correctly rounded functions, if one is set.

        Returns:
            The largest working precision in bits, or None for the default.
        """
        return self._max_precision

    def _budget(self) -> Int:
        """The largest working precision: the budget, or by default
        `max(8 * p, p + 4096)`."""
        if self._max_precision:
            return self._max_precision.value()
        var p = self._format.precision()
        return max(8 * p, p + 4096)

    def _with_format(self, format: FloatFormat) -> Self:
        """These settings with another output format."""
        var result = self
        result._format = format
        return result^

    def _quiet(self) -> Self:
        """These settings without traps, for trial roundings."""
        var result = self
        result._traps = 0
        return result^

    def traps_inexact(self) -> Bool:
        """Whether inexact results raise.

        Returns:
            The trap setting.
        """
        return Bool(self._traps & 1)

    def traps_underflow(self) -> Bool:
        """Whether underflow raises.

        Returns:
            The trap setting.
        """
        return Bool(self._traps & 2)

    def traps_overflow(self) -> Bool:
        """Whether overflow raises.

        Returns:
            The trap setting.
        """
        return Bool(self._traps & 4)

    def traps_divide_by_zero(self) -> Bool:
        """Whether division by zero raises.

        Returns:
            The trap setting.
        """
        return Bool(self._traps & 8)

    def traps_invalid(self) -> Bool:
        """Whether invalid operations raise.

        Returns:
            The trap setting.
        """
        return Bool(self._traps & 16)

    def _check_status(self, status: NumericStatus) raises:
        var trapped = self._traps & status._flags
        if trapped:
            var name = (
                "invalid" if trapped
                & 16 else "divide_by_zero" if trapped
                & 8 else "overflow" if trapped
                & 4 else "underflow" if trapped
                & 2 else "inexact"
            )
            raise Error(
                String(
                    "Floating-point operation trapped ",
                    name,
                    (
                        "; correct the input or explicitly disable that trap in"
                        " ArithmeticContext."
                    ),
                    (
                        " For rounding or range traps, a suitable output format"
                        " may avoid the condition."
                    ),
                    " The destination is unchanged.",
                )
            )

    def write_to(self, mut writer: Some[Writer]):
        """Write the format, rounding mode and traps.

        Args:
            writer: The destination.
        """
        writer.write(
            "ArithmeticContext(format=",
            self._format,
            ", rounding=",
            self._rounding,
            ", trap_inexact=",
            self.traps_inexact(),
            ", trap_underflow=",
            self.traps_underflow(),
            ", trap_overflow=",
            self.traps_overflow(),
            ", trap_divide_by_zero=",
            self.traps_divide_by_zero(),
            ", trap_invalid=",
            self.traps_invalid(),
            ", max_precision=",
        )
        if self._max_precision:
            writer.write(self._max_precision.value())
        else:
            writer.write("None")
        writer.write(")")


def _exact_context() -> ArithmeticContext:
    """The context of exact working arithmetic: every bit of each result is
    kept, and a result that is not a finite binary fraction raises."""
    return ArithmeticContext(_format_of=FloatFormat._exact_format(RoundingMode.nearest_even))
