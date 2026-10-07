"""Value-independent real-format selection for scalar Float adapters."""

from std.os import abort
from .context import FloatFormat, ArithmeticContext


struct _FormatSlot(ImplicitlyCopyable):
    """An optional FloatFormat as a plain pair, for argument records. With an
    `Optional[FloatFormat]`, building a record stored the payload a byte at a
    time, and reading it back as words stalled store forwarding; the slot
    stores four words and a flag. It converts from a format, an `Optional`
    and `None`, and tests and reads as an `Optional` does."""

    var _format: FloatFormat
    var _present: Bool

    @always_inline
    def __init__(out self, format: FloatFormat, present: Bool):
        self._format = format
        self._present = present

    @implicit
    @always_inline
    def __init__(out self, format: FloatFormat):
        self = Self(format, True)

    @implicit
    @always_inline
    def __init__(out self, none: type_of(None)):
        self = _absent_format()

    @implicit
    @always_inline
    def __init__(out self, format: Optional[FloatFormat]):
        self = Self(format.value(), True) if format else _absent_format()

    @always_inline
    def __bool__(self) -> Bool:
        return self._present

    @always_inline
    def value(ref self) -> ref[self._format] FloatFormat:
        return self._format


@always_inline
def _absent_format() -> _FormatSlot:
    # The placeholder of an absent format; constant arguments cannot raise.
    try:
        return _FormatSlot(FloatFormat(1), False)
    except:
        abort("unreachable")


def _merge_float_formats(
    left: _FormatSlot,
    right: _FormatSlot,
    *,
    left_native_precision: Int = 0,
    right_native_precision: Int = 0,
    context: Optional[ArithmeticContext] = None,
) raises -> FloatFormat:
    # An absent library format and zero native precision denote an exact operand.
    if (
        left_native_precision < 0
        or right_native_precision < 0
        or (left and left_native_precision != 0)
        or (right and right_native_precision != 0)
    ):
        raise Error(
            "Cannot select a floating-point output format: each operand must"
            " have one numeric family. Supply valid exact, native or Float"
            " operands."
        )
    if context:
        return context.value().format()
    # Exact working values make every result exact; bounds come at the end.
    if left and left.value()._is_exact():
        return left.value()
    if right and right.value()._is_exact():
        return right.value()
    if left and right and left.value() == right.value():
        return left.value()
    if (
        left
        and right
        and (
            left.value().emin() != right.value().emin()
            or left.value().emax() != right.value().emax()
        )
    ):
        raise Error(
            String(
                "FormatMismatch: cannot combine exponent bounds [",
                left.value().emin(),
                ", ",
                left.value().emax(),
                "] and [",
                right.value().emin(),
                ", ",
                right.value().emax(),
                (
                    "]; supply an ArithmeticContext with the desired output"
                    " FloatFormat. The destination is unchanged."
                ),
            )
        )
    var precision = max(
        left.value().precision() if left else 0,
        right.value().precision() if right else 0,
        left_native_precision,
        right_native_precision,
    )
    if not precision:
        raise Error(
            "Exact operands do not select a Float format; keep the exact result"
            " or supply an ArithmeticContext for explicit rounding."
        )
    var format = (
        left.value() if left else right.value() if right else FloatFormat()
    )
    return FloatFormat(precision, emin=format.emin(), emax=format.emax())
