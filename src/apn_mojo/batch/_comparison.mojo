"""Scalar-left exact comparison without a dependency on public batch types."""

from ..integer.value import Integer
from ..rational.value import Rational
from .mask import Mask


trait _ExactBatchComparison:
    def _compare_integer(self, rhs: Integer, operation: Int) raises -> Mask:
        ...

    def _compare_rational(self, rhs: Rational, operation: Int) raises -> Mask:
        ...
