"""Operation-local numerical status, never ambient sticky state."""


struct NumericStatus(Equatable, ImplicitlyCopyable, Writable):
    """Flags and rounding direction for one committed scalar operation.

    Capture replaces the prior status, rather than accumulating sticky flags.
    Named ordered reductions union their step flags; direction is final-step only.
    """

    var _flags: Int
    var _direction: Int

    def __init__(out self):
        self._flags = 0
        self._direction = 0

    @staticmethod
    def _make(flags: Int, direction: Int) -> Self:
        var result = Self()
        result._flags = flags
        result._direction = direction
        return result

    def inexact(self) -> Bool:
        return Bool(self._flags & 1)

    def underflow(self) -> Bool:
        return Bool(self._flags & 2)

    def overflow(self) -> Bool:
        return Bool(self._flags & 4)

    def divide_by_zero(self) -> Bool:
        return Bool(self._flags & 8)

    def invalid(self) -> Bool:
        return Bool(self._flags & 16)

    def direction(self) -> StaticString:
        if self._direction == -1:
            return "below"
        if self._direction == 0:
            return "exact"
        if self._direction == 1:
            return "above"
        return "unordered"

    def clear(mut self):
        self = Self()

    def __eq__(self, other: Self) -> Bool:
        return (
            self._flags == other._flags and self._direction == other._direction
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "NumericStatus(inexact=",
            self.inexact(),
            ", underflow=",
            self.underflow(),
            ", overflow=",
            self.overflow(),
            ", divide_by_zero=",
            self.divide_by_zero(),
            ", invalid=",
            self.invalid(),
            ", direction=",
            self.direction(),
            ")",
        )
