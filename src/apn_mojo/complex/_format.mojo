"""Complex component-format selection without numerical conversion."""

from ..float.context import FloatFormat, ArithmeticContext
from ..float._format import _merge_float_formats, _FormatSlot


struct _ComplexFormats(Equatable, ImplicitlyCopyable):
    var _real: FloatFormat
    var _imag: FloatFormat

    def __init__(out self, real: FloatFormat, imag: FloatFormat) raises:
        if real.emin() != imag.emin() or real.emax() != imag.emax():
            raise Error(
                String(
                    "FormatMismatch: Complex components have bounds [",
                    real.emin(),
                    ", ",
                    real.emax(),
                    "] and [",
                    imag.emin(),
                    ", ",
                    imag.emax(),
                    (
                        "]; choose output component formats with common"
                        " exponent bounds. The destination is unchanged."
                    ),
                )
            )
        self._real = real
        self._imag = imag

    def real(self) -> FloatFormat:
        return self._real

    def imag(self) -> FloatFormat:
        return self._imag

    def real_result_format(self) raises -> FloatFormat:
        return FloatFormat(
            max(self._real.precision(), self._imag.precision()),
            emin=self._real.emin(),
            emax=self._real.emax(),
        )

    def __eq__(self, other: Self) -> Bool:
        return self._real == other._real and self._imag == other._imag


def _merge_complex_formats(
    left: Optional[_ComplexFormats],
    right: Optional[_ComplexFormats],
    *,
    left_real: _FormatSlot = None,
    right_real: _FormatSlot = None,
    left_native_precision: Int = 0,
    right_native_precision: Int = 0,
    destination: Optional[_ComplexFormats] = None,
) raises -> _ComplexFormats:
    # Absent format and zero native precision denote an exact real operand.
    for side in range(2):
        var pair = left if side == 0 else right
        var real = left_real if side == 0 else right_real
        var native = (
            left_native_precision if side == 0 else right_native_precision
        )
        if native < 0 or (
            (1 if pair else 0) + (1 if real else 0) + Int(native != 0) > 1
        ):
            raise Error(
                "Cannot select Complex formats: each operand must have one"
                " numeric family. Supply valid exact, native, Float or Complex"
                " operands. The destination is unchanged."
            )
    if not left and not right:
        raise Error(
            "Complex arithmetic requires a Complex operand; retain the real"
            " result or explicitly construct a Complex value."
        )
    var lr = left_real
    var li = left_real
    var rr = right_real
    var ri = right_real
    if left:
        lr = left.value().real()
        li = left.value().imag()
    if right:
        rr = right.value().real()
        ri = right.value().imag()
    var real_context: Optional[ArithmeticContext] = None
    var imag_context: Optional[ArithmeticContext] = None
    if destination:
        real_context = ArithmeticContext(format=destination.value().real())
        imag_context = ArithmeticContext(format=destination.value().imag())
    return _ComplexFormats(
        _merge_float_formats(
            lr,
            rr,
            left_native_precision=left_native_precision,
            right_native_precision=right_native_precision,
            context=real_context,
        ),
        _merge_float_formats(
            li,
            ri,
            left_native_precision=left_native_precision,
            right_native_precision=right_native_precision,
            context=imag_context,
        ),
    )
