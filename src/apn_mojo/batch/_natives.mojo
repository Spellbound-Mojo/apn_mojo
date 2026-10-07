"""Batch elements as native numbers: `Batch.to_native`."""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import RoundingMode
from ..float._input import _float_input
from ..float._native import _float_native, _ratio_native
from ..ball.value import Ball


def _native_element[dtype: DType, T: ImplicitlyCopyable & Deinitable](
    value: T, rounding: RoundingMode,
) raises -> SIMD[dtype, 1]:
    """One element as a native number: rounded once to a floating-point type,
    exactly to an integer type, or raising."""
    comptime assert T != Ball or dtype.is_floating_point(), (
        "Ball batches convert to floating-point types only, through their midpoints."
    )
    comptime if dtype.is_floating_point():
        comptime if T == Float:
            return rebind[Float](value).to_native[dtype](rounding=rounding)
        elif T == Ball:
            return rebind[Ball](value).midpoint().to_native[dtype](rounding=rounding)
        elif T == Integer:
            return _float_native[dtype](_float_input(rebind[Integer](value)), rounding).value
        else:
            comptime assert T == Rational, (
                "to_native converts Integer, Rational, Float and Ball batches; take"
                " batch.real and batch.imag of a Complex batch first."
            )
            ref q = rebind[Rational](value)
            if q.sign() == 0:
                return SIMD[dtype, 1](0)
            var n = q.numerator()
            return _ratio_native[dtype](n.sign() < 0, abs(n), q.denominator(), rounding)
    else:
        comptime if T == Integer:
            return rebind[Integer](value).to_native_exact[dtype]()
        elif T == Rational:
            return rebind[Rational](value).to_native_exact[dtype]()
        else:
            comptime assert T == Float, (
                "to_native converts Integer, Rational, Float and Ball batches; take"
                " batch.real and batch.imag of a Complex batch first."
            )
            return rebind[Float](value).to_native_exact[dtype]()
