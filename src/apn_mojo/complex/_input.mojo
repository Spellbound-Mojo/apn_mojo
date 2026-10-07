"""Retained exact operands; a real input is not an artificial complex zero."""

from ..float._arithmetic import _FloatArgument
from ..common._traits import _MapAdapter, _MapArgument
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from .value import Complex


trait _ComplexSource(Copyable):
    def _complex_real(self) -> _FloatArgument:
        ...

    def _complex_imag(self) -> _FloatArgument:
        ...


struct _ComplexArgument(ImplicitlyCopyable, _MapAdapter, _MapArgument):
    var real: _FloatArgument
    var imag: _FloatArgument
    var complex: Bool

    @staticmethod
    def _adapt[V: ImplicitlyCopyable & Deinitable](value: V) raises -> Self:
        comptime assert V == Integer or V == Rational or V == Float or V == Complex, (
            "A Complex function takes Integer, Rational, Float or Complex values."
        )
        return Self(value)

    @implicit
    def __init__[T: Copyable](out self, value: T):
        comptime if conforms_to(T, _ComplexSource):
            self.real = value._complex_real()
            self.imag = value._complex_imag()
            self.complex = True
        else:
            self.real = _FloatArgument(value)
            self.imag = _FloatArgument(Int(0))
            self.complex = False

    @implicit
    def __init__(out self, value: _FloatArgument):
        self.real = value
        self.imag = _FloatArgument(Int(0))
        self.complex = False

    def __init__(out self, *, real: _FloatArgument, imag: _FloatArgument):
        """A Complex operand from its two parts, without building a Complex,
        whose constructor rounds and merges the parts' formats."""
        self.real = real
        self.imag = imag
        self.complex = True

    @implicit
    def __init__(out self, value: IntLiteral):
        self = Self(_FloatArgument(value))

    @implicit
    def __init__(out self, value: FloatLiteral):
        self = Self(_FloatArgument(value))
