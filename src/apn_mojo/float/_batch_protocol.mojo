"""Scalar-left Float dispatch without importing public batch owners."""

from ._arithmetic import _FloatArgument
from ..batch.mask import Mask


trait _FloatBatchArithmetic(ImplicitlyCopyable):
    comptime FloatBatch: ImplicitlyCopyable

    def _float_reverse(
        self, lhs: _FloatArgument, operation: Int
    ) raises -> Self.FloatBatch:
        ...

    def _compare_float(
        self, rhs: _FloatArgument, operation: Int
    ) raises -> Mask:
        ...

trait _FloatBatchOperand(_FloatBatchArithmetic):
    def _float_power_reverse(
        self, base: _FloatArgument
    ) raises -> Self.FloatBatch:
        ...
