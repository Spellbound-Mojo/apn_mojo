"""Immutable literal words in program data, with no runtime allocation."""

from std.builtin.globals import global_constant
from ._operand import _LiteralOperand, _IntegerOperand


def _literal_data[
    value: IntLiteral
]() -> Array[UInt32, _LiteralOperand[value]()._word_count() + 3]:
    comptime count = _LiteralOperand[value]()._word_count()
    var result = Array[UInt32, count + 3](fill=0)
    result[0] = UInt32(UInt64(count) & 0xFFFFFFFF)
    result[1] = UInt32(UInt64(count) >> 32)
    result[2] = UInt32(value < 0)
    comptime for i in range(count):
        result[i + 3] = _LiteralOperand[value]()._word(i)
    return result^


struct _StaticInteger(ImplicitlyCopyable, _IntegerOperand):
    var data: Pointer[UInt32, ImmStaticOrigin]

    def __init__(out self, value: IntLiteral):
        comptime words = _literal_data[type_of(value)()]()
        self.data = Pointer(to=global_constant[words]()[0])

    def _negative(self) -> Bool:
        return Bool(self.data[unsafe_offset=2])

    def _word_count(self) -> Int:
        return Int(
            UInt64(self.data[unsafe_offset=0])
            | (UInt64(self.data[unsafe_offset=1]) << 32)
        )

    def _word(self, index: Int) -> UInt32:
        if index < 0 or index >= self._word_count():
            return 0
        return self.data[unsafe_offset=index + 3]
