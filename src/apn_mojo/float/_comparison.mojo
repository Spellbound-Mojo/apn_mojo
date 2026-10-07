"""Exact-family comparison adapters without importing the public Float type."""

from ..integer.value import Integer
from ..rational.value import Rational


trait _FloatComparison:
    def _compare_integer(self, rhs: Integer, operation: Int) raises -> Bool:
        ...

    def _compare_rational(self, rhs: Rational, operation: Int) raises -> Bool:
        ...


def _float_relation(order: Int, operation: Int) -> Bool:
    # Unordered is 2, distinct from the three ordinary comparison results.
    if operation == 0:
        return order == 0
    if operation == 1:
        return order != 0
    if order == 2:
        return False
    if operation == 2:
        return order < 0
    if operation == 3:
        return order <= 0
    if operation == 4:
        return order > 0
    return order >= 0
