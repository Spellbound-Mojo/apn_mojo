"""Scalar Complex dispatch without importing public batch owners."""

from ._input import _ComplexArgument
from ..batch.mask import Mask


trait _ComplexBatchArithmetic(ImplicitlyCopyable):
    comptime ComplexBatch: ImplicitlyCopyable

    def _complex_reverse(
        self, lhs: _ComplexArgument, operation: Int
    ) raises -> Self.ComplexBatch:
        ...

    def _compare_complex(
        self, rhs: _ComplexArgument, operation: Int
    ) raises -> Mask:
        ...


trait _ComplexBatchOperand(_ComplexBatchArithmetic):
    def _complex_power_reverse(
        self, base: _ComplexArgument
    ) raises -> Self.ComplexBatch:
        ...
