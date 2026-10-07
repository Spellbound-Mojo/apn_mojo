"""Operation-local component statuses; no ordering of a complex result."""

from ..float.status import NumericStatus


struct ComplexStatus(Equatable, ImplicitlyCopyable, Writable):
    var _real: NumericStatus
    var _imag: NumericStatus

    def __init__(out self):
        self._real = NumericStatus()
        self._imag = NumericStatus()

    @staticmethod
    def _make(real: NumericStatus, imag: NumericStatus) -> Self:
        var result = Self()
        result._real = real
        result._imag = imag
        return result

    def real(self) -> NumericStatus:
        return self._real

    def imag(self) -> NumericStatus:
        return self._imag

    def inexact(self) -> Bool:
        return self._real.inexact() or self._imag.inexact()

    def underflow(self) -> Bool:
        return self._real.underflow() or self._imag.underflow()

    def overflow(self) -> Bool:
        return self._real.overflow() or self._imag.overflow()

    def divide_by_zero(self) -> Bool:
        return self._real.divide_by_zero() or self._imag.divide_by_zero()

    def invalid(self) -> Bool:
        return self._real.invalid() or self._imag.invalid()

    def clear(mut self):
        self = Self()

    def __eq__(self, rhs: Self) -> Bool:
        return self._real == rhs._real and self._imag == rhs._imag

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ComplexStatus(real=", self._real, ", imag=", self._imag, ")"
        )
