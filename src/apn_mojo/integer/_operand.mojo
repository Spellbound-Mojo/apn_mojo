"""Private non-owning word access for integer update operands."""

from std.sys import bit_width_of
from ._limits import _count_error


trait _IntegerOperand:
    def _negative(self) -> Bool:
        ...

    def _word_count(self) -> Int:
        ...

    def _word(self, index: Int) -> UInt32:
        ...


struct _NativeOperand(ImplicitlyCopyable, _IntegerOperand):
    var magnitude: UInt64
    var negative: Bool

    def __init__[dtype: DType](out self, value: SIMD[dtype, 1]):
        comptime assert dtype.is_integral() and bit_width_of[dtype]() <= 64, (
            "Integer accepts native integral types up to 64 bits; use text or"
            " an integer literal for wider values"
        )
        comptime if dtype.is_unsigned():
            self.negative = False
            self.magnitude = UInt64(value)
        else:
            var signed = Int64(value)
            self.negative = signed < 0
            self.magnitude = UInt64(
                -(signed + 1)
            ) + 1 if signed < 0 else UInt64(signed)

    def _negative(self) -> Bool:
        return self.negative

    def _word_count(self) -> Int:
        return 2 if self.magnitude >> 32 else Int(self.magnitude != 0)

    def _word(self, index: Int) -> UInt32:
        if index < 0 or index >= 2:
            return 0
        return UInt32((self.magnitude >> UInt64(index * 32)) & 0xFFFFFFFF)


def _literal_count(value: IntLiteral) -> Int:
    comptime magnitude = type_of(value)()
    comptime if magnitude == 0:
        return 0
    else:
        return 1 + _literal_count(magnitude >> 32)


def _literal_word(value: IntLiteral, index: Int) -> UInt32:
    comptime magnitude = type_of(value)()
    comptime if magnitude == 0:
        return 0
    else:
        if index == 0:
            return UInt32(magnitude & 0xFFFFFFFF)
        return _literal_word(magnitude >> 32, index - 1)


struct _LiteralOperand[value: IntLiteral](ImplicitlyCopyable, _IntegerOperand):
    def __init__(out self):
        pass

    def _negative(self) -> Bool:
        return Self.value < 0

    def _word_count(self) -> Int:
        comptime if Self.value < 0:
            return _literal_count(-Self.value)
        else:
            return _literal_count(Self.value)

    def _word(self, index: Int) -> UInt32:
        comptime if Self.value < 0:
            return _literal_word(-Self.value, index)
        else:
            return _literal_word(Self.value, index)


def _operand_at_least[R: _IntegerOperand](rhs: R, limit: Int) -> Bool:
    if rhs._word_count() > 2:
        return True
    return (UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32)) >= UInt64(
        limit
    )


def _operand_count[
    R: _IntegerOperand
](rhs: R, limit: Int, operation: Int) raises -> Int:
    if rhs._word_count() > 2:
        _count_error(operation, too_large=True)
    var value = UInt64(rhs._word(0)) | (UInt64(rhs._word(1)) << 32)
    if value > UInt64(limit):
        _count_error(operation, too_large=True)
    return Int(value)


