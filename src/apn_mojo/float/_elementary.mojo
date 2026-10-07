"""The shared steps of correctly rounded Float functions.

For each function, in order:

1. The target: the context, or the operand's format rounded to nearest-even;
   exact operands need a context, as for `sqrt`.
2. NaN propagates without a flag; zeros, infinities and points outside the
   domain take the special values of the requirements' Appendix A.
3. Exact cases (Appendix D.1) are computed exactly and rounded once.
4. Results certainly beyond the format's range overflow or underflow, decided
   from exact comparisons, before any kernel runs.
5. A value within a tiny distance of a Float of known sign, such as
   `sin(x) = x - x**3/6 + ...` for a tiny `x`, rounds as `_round_near` decides.
6. Otherwise the Ziv driver runs over the function's kernel at the exact
   argument. An exact Rational that is not a binary fraction has no Float
   point, so its kernel is the ball function of an enclosure of it at each
   working precision.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from .value import Float
from .context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from .status import NumericStatus
from ._arithmetic import _FloatArgument, _float_special, _float_operation, _call_context
from ._rounding import _RoundedBinary, _round_ratio, _inexact_error, _finish_round, _overflow_result, _is_power_of_two
from ._functions import _pow_float, _round_scaled_ratio, _rootn_float
from ._format import _merge_float_formats
from ..integer.powers import iroot_exact
from ..rational.math import root_exact
from ._input import _FloatInput
from ..integer._word_math import _trailing_zero_bits
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._certified import _Enclosure, _round_certified, _round_near
from ..ball._kernels import (
    _RealKernel, _integral, _power_of_two, _power_of_ten,
    _SQRT, _EXP, _EXPM1, _EXP2, _LOG, _LOG1P, _LOG2, _LOG10, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS,
    _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH, _POW, _ATAN2, _ROOTN,
    _PI, _HALF_PI, _QUARTER_PI, _THREE_QUARTER_PI, _function_name,
)
from ..ball._functions import _ball_function, _ball_pow, _ball_atan2


def _constant_rounded(code: Int, context: Optional[ArithmeticContext], guard: Int = 0) raises -> _RoundedBinary:
    """A constant, correctly rounded in the context, by default 128 bits to
    nearest-even. An exact working format cannot hold it."""
    var target = context.value() if context else ArithmeticContext()
    if target.format()._is_exact():
        raise _inexact_error("a constant")
    return _round_certified(_RealKernel.constant(code), target, guard)


def _exact_float(x: _FloatArgument) raises -> Optional[Float]:
    """The exact Float of a finite nonzero binary fraction; None for another
    Rational."""
    if not _is_power_of_two(x.value.denominator):
        return None
    return Float(_rounded=_float_operation(x, Integer(0), 0, _exact_context()))


def _unit(negative: Bool) raises -> Float:
    """+1 or -1 in a 1-bit format, so a shortcut near it applies as early as
    the target precision allows."""
    return Float(-1 if negative else 1, context=ArithmeticContext(_format_of=FloatFormat(1)))


def _special(kind: Int, negative: Bool, flags: Int, target: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(kind, negative, flags, target, False)


def _invalid(target: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(3, False, 16, target, False)


def _exactly(value: Integer, target: ArithmeticContext) raises -> _RoundedBinary:
    return _round_ratio(value, Integer(1), target)


def _signed_constant(code: Int, negative: Bool, target: ArithmeticContext) raises -> _RoundedBinary:
    return _round_certified(_RealKernel.constant(code, negative), target)


def _overflow(negative: Bool, target: ArithmeticContext) raises -> _RoundedBinary:
    return _finish_round(_overflow_result(negative, target), target, False)


def _underflow(negative: Bool, target: ArithmeticContext) raises -> _RoundedBinary:
    """A value certainly below `2**(emin - 2)` in magnitude, rounded: zero or the
    smallest Float by the mode, with underflow."""
    return _round_ratio(Integer(-1 if negative else 1), Integer(1), target, scale=Int128(target.format().emin()) - 3)


def _times_ln2(n: Int, upper: Bool) raises -> Rational:
    """A bound of `n ln 2`: above it when `upper` (for n >= 0), else below."""
    var c = Rational(693148, 1000000) if upper else Rational(693147, 1000000)
    return Rational(Integer(n)) * c


def _beyond(x: Float, bound: Rational) raises -> Bool:
    """`x > bound` for a finite x, cheaply for a huge x."""
    if x._exponent > 80:
        return not x._negative
    return x.to_rational_exact() > bound


def _below(x: Float, bound: Rational) raises -> Bool:
    """`x < bound` for a finite x, cheaply for a huge x."""
    if x._exponent > 80:
        return x._negative
    return x.to_rational_exact() < bound


def _scaled_down(x: Float, factor_numerator: Int, factor_denominator: Int) raises -> Int:
    """`floor(|x| * a / b)` clamped to `2**62`, for exponents of tiny bounds."""
    if x._exponent > 62:
        return Int(1) << 62
    var value = (abs(x).to_rational_exact() * Rational(factor_numerator, factor_denominator)).floor()
    if value > Integer(Int(1) << 62):
        return Int(1) << 62
    return Int(value)


@fieldwise_init
struct _BallPathKernel(_Enclosure):
    """A real function at exact arguments that are not binary fractions:
    the ball function of an enclosure of the arguments at each precision."""

    var code: Int
    var x: _FloatArgument
    var y: _FloatArgument
    var n: Int

    def enclosure(self, precision: Int) raises -> Ball:
        var c = Optional[BallContext](BallContext(precision + 8))
        var a = Ball(self.x, precision=precision + 32)
        if self.code == _POW:
            return _ball_pow(_BallArgument(a), _BallArgument(Ball(self.y, precision=precision + 32)), c)
        if self.code == _ATAN2:
            return _ball_atan2(_BallArgument(a), _BallArgument(Ball(self.y, precision=precision + 32)), c)
        return _ball_function(self.code, _BallArgument(a), self.n, c)

    def describe(self) raises -> String:
        return String(_function_name(self.code), " at an exact rational")


def _tiny(code: Int, x: Float) -> Tuple[Bool, Int, Bool]:
    """For the functions near `x` or near 1 at a tiny `x`: whether the base is
    1 (else x), an exponent `t` with `|f(x) - base| < 2**t`, and whether
    `f(x) > base`."""
    var e = x._exponent
    var negative = x._negative
    if code == _EXP:
        return (True, e + 1, not negative)
    if code == _EXP2:
        return (True, e, not negative)
    if code == _COS:
        return (True, 2 * e - 1, False)
    if code == _COSH:
        return (True, 2 * e, True)
    if code == _EXPM1:
        return (False, 2 * e, True)
    if code == _LOG1P:
        return (False, 2 * e, False)
    if code == _SIN or code == _ATAN or code == _TANH or code == _ASINH:
        return (False, 3 * e - 1, negative)
    # tan, asin, sinh, atanh grow faster than x.
    return (False, 3 * e - 1, not negative)


def _has_tiny_form(code: Int) -> Bool:
    return code == _EXP or code == _EXP2 or code == _COS or code == _COSH or code == _EXPM1 or code == _LOG1P or code == _SIN or code == _ATAN or code == _TANH or code == _ASINH or code == _TAN or code == _ASIN or code == _SINH or code == _ATANH


def _real_rounded(
    code: Int, value: _FloatArgument, context: Optional[ArithmeticContext], guard: Int = 0,
) raises -> _RoundedBinary:
    """A unary real function, correctly rounded."""
    var target = _call_context(value, context)
    var v = value.value
    if v.kind == 3:
        return _special(3, False, 0, target)
    var negative = v.negative
    # Zeros.
    if v.kind == 0:
        if code == _EXP or code == _EXP2 or code == _COS or code == _COSH:
            return _exactly(Integer(1), target)
        if code == _LOG or code == _LOG2 or code == _LOG10:
            return _special(2, True, 8, target)
        if code == _ACOS:
            return _signed_constant(_HALF_PI, False, target)
        if code == _ACOSH:
            return _invalid(target)
        return _special(0, negative, 0, target)
    # Infinities.
    if v.kind == 2:
        if code == _EXP or code == _EXP2:
            return _special(2, False, 0, target) if not negative else _special(0, False, 0, target)
        if code == _EXPM1:
            return _special(2, False, 0, target) if not negative else _exactly(Integer(-1), target)
        if code == _LOG or code == _LOG2 or code == _LOG10 or code == _LOG1P or code == _ACOSH:
            return _special(2, False, 0, target) if not negative else _invalid(target)
        if code == _SINH or code == _ASINH:
            return _special(2, negative, 0, target)
        if code == _COSH:
            return _special(2, False, 0, target)
        if code == _TANH:
            return _exactly(Integer(-1 if negative else 1), target)
        if code == _ATAN:
            return _signed_constant(_HALF_PI, negative, target)
        return _invalid(target)
    var exact = _exact_float(value)
    if not exact:
        if target.format()._is_exact():
            raise _inexact_error(_function_name(code))
        return _round_certified(_BallPathKernel(code, value, value, 0), target, guard)
    var x = exact.take()
    var one = Float(1)
    # Domains and exact points.
    if code == _LOG or code == _LOG2 or code == _LOG10:
        if negative:
            return _invalid(target)
        if x == one:
            return _special(0, False, 0, target)
        if code == _LOG2:
            var k = _power_of_two(x)
            if k:
                return _exactly(Integer(k.value()), target)
        if code == _LOG10:
            var k = _power_of_ten(x)
            if k:
                return _exactly(Integer(k.value()), target)
    elif code == _LOG1P:
        if x == Float(-1):
            return _special(2, True, 8, target)
        if x < Float(-1):
            return _invalid(target)
    elif code == _ASIN or code == _ACOS:
        var magnitude = abs(x)
        if magnitude > one:
            return _invalid(target)
        if magnitude == one:
            if code == _ASIN:
                return _signed_constant(_HALF_PI, negative, target)
            if negative:
                return _signed_constant(_PI, False, target)
            return _special(0, False, 0, target)
    elif code == _ACOSH:
        if x < one:
            return _invalid(target)
        if x == one:
            return _special(0, False, 0, target)
    elif code == _ATANH:
        var magnitude = abs(x)
        if magnitude > one:
            return _invalid(target)
        if magnitude == one:
            return _special(2, negative, 8, target)
    elif code == _ATAN:
        if abs(x) == one:
            return _signed_constant(_QUARTER_PI, negative, target)
    elif code == _EXP2:
        var whole = _integral(x)
        if whole:
            return _round_scaled_ratio(Integer(1), Integer(1), whole.value(), False, target, False)
    var emax = target.format().emax()
    var emin = target.format().emin()
    # Range: results certainly beyond the format.
    if code == _EXP or code == _EXPM1:
        if _beyond(x, _times_ln2(emax, True)):
            return _overflow(False, target)
        if code == _EXP and _below(x, _times_ln2(emin - 2, True)):
            return _underflow(False, target)
        if code == _EXPM1 and x._negative and x._exponent > 0:
            # expm1(x) = -1 + exp(x) with exp(x) < 2**t, t = -floor(|x| 1.4426).
            var t = -_scaled_down(x, 14426, 10000)
            var near = _round_near(_unit(True), True, t, target)
            if near:
                return near.take()
    elif code == _EXP2:
        if _beyond(x, Rational(Integer(emax))):
            return _overflow(False, target)
        if _below(x, Rational(Integer(emin - 2))):
            return _underflow(False, target)
    elif code == _SINH or code == _COSH:
        if _beyond(abs(x), _times_ln2(emax + 1, True)):
            return _overflow(negative and code == _SINH, target)
    elif code == _TANH:
        if x._exponent > 0:
            # 1 - |tanh x| = 2/(exp(2|x|) + 1) < 2**(1 - floor(2|x| 1.4426)).
            var t = 1 - _scaled_down(x, 28852, 10000)
            var near = _round_near(_unit(negative), negative, t, target)
            if near:
                return near.take()
    # Tiny arguments.
    if _has_tiny_form(code) and x._exponent < 0:
        var form = _tiny(code, x)
        var near = _round_near(_unit(False) if form[0] else x, form[2], form[1], target)
        if near:
            return near.take()
    if code == _LOG:
        var d = Float(_rounded=_float_operation(x, Integer(1), 1, _exact_context()))
        if d._exponent < 0:
            var near = _round_near(d, False, 2 * d._exponent, target)
            if near:
                return near.take()
    if target.format()._is_exact():
        raise _inexact_error(_function_name(code))
    return _round_certified(_RealKernel.unary(code, x, target._budget()), target, guard)


def _atan2_rounded(
    y: _FloatArgument, x: _FloatArgument, context: Optional[ArithmeticContext], guard: Int = 0,
) raises -> _RoundedBinary:
    """atan2(y, x), correctly rounded, with the special values of C99."""
    var target = _call_context(y, x, context)
    var a = y.value
    var b = x.value
    if a.kind == 3 or b.kind == 3:
        return _special(3, False, 0, target)
    if a.kind == 0:
        if b.negative:
            return _signed_constant(_PI, a.negative, target)
        return _special(0, a.negative, 0, target)
    if b.kind == 0:
        return _signed_constant(_HALF_PI, a.negative, target)
    if a.kind == 2:
        if b.kind == 2:
            return _signed_constant(_THREE_QUARTER_PI if b.negative else _QUARTER_PI, a.negative, target)
        return _signed_constant(_HALF_PI, a.negative, target)
    if b.kind == 2:
        if b.negative:
            return _signed_constant(_PI, a.negative, target)
        return _special(0, a.negative, 0, target)
    var fy = _exact_float(y)
    var fx = _exact_float(x)
    if target.format()._is_exact():
        raise _inexact_error("atan2")
    if not fy or not fx:
        return _round_certified(_BallPathKernel(_ATAN2, y, x, 0), target, guard)
    return _round_certified(_RealKernel.binary(_ATAN2, fy.take(), fx.take()), target, guard)


def _pow_rounded(
    base: _FloatArgument, exponent: _FloatArgument, context: Optional[ArithmeticContext], guard: Int = 0,
) raises -> _RoundedBinary:
    """`base**exponent`, correctly rounded, with the special values of C99
    F.9.4.4 and the exact cases of binary-fraction exponents."""
    var target = _call_context(base, exponent, context)
    var a = base.value
    var b = exponent.value
    if b.kind == 0:
        return _exactly(Integer(1), target)
    var base_float = _exact_float(base) if a.kind == 1 else None
    if base_float and base_float.value() == Float(1):
        return _exactly(Integer(1), target)
    if a.kind == 3 or b.kind == 3:
        return _special(3, False, 0, target)
    # An integral exponent is an integer power.
    if b.kind == 1 and _is_power_of_two(b.denominator):
        var y = _exact_float(exponent).value()
        var whole = _integral(y)
        if whole:
            return _pow_float(base, whole.value(), target)
    if b.kind == 2:
        if a.kind == 2:
            return _special(0, False, 0, target) if b.negative else _special(2, False, 0, target)
        if a.kind == 0:
            # 0**-inf = +inf and 0**+inf = 0 raise no flag (C99 Annex F permits
            # divide-by-zero for the first).
            return _special(2, False, 0, target) if b.negative else _special(0, False, 0, target)
        var magnitude = abs(base_float.value()) if base_float else Float(Rational(1))
        if not base_float:
            var r = Rational(a.numerator, a.denominator)
            if r == 1:
                return _exactly(Integer(1), target)
            var small = r < 1
            return _special(2, False, 0, target) if small == b.negative else _special(0, False, 0, target)
        if magnitude == Float(1):
            return _exactly(Integer(1), target)
        var small = magnitude < Float(1)
        return _special(2, False, 0, target) if small == b.negative else _special(0, False, 0, target)
    # A finite non-integral exponent.
    if a.kind == 0:
        return _special(2, False, 8, target) if b.negative else _special(0, False, 0, target)
    if a.kind == 2:
        if a.negative:
            return _special(0, False, 0, target) if b.negative else _special(2, False, 0, target)
        return _special(0, False, 0, target) if b.negative else _special(2, False, 0, target)
    if a.negative:
        return _invalid(target)
    if target.format()._is_exact():
        raise _inexact_error("pow")
    if not base_float or not _is_power_of_two(b.denominator):
        var exact_root = _rational_exponent_exact(base, exponent, target)
        if exact_root:
            return exact_root.take()
        return _round_certified(_BallPathKernel(_POW, base, exponent, 0), target, guard)
    var x = base_float.take()
    var y = _exact_float(exponent).value()
    var dyadic = _dyadic_power(x, y, target)
    if dyadic:
        return dyadic.take()
    # Range: y log2 x from a low-precision ball.
    var estimate = _ball_function(_LOG2, _BallArgument(Ball(x)), 0, Optional[BallContext](BallContext(48)))
    var t = Ball(y) * estimate
    if t.is_finite():
        if t.certainly_gt(Ball(target.format().emax())):
            return _overflow(False, target)
        if t.certainly_lt(Ball(target.format().emin() - 2)):
            return _underflow(False, target)
    return _round_certified(_RealKernel.binary(_POW, x, y, target._budget()), target, guard)


def _dyadic_power(x: Float, y: Float, target: ArithmeticContext) raises -> Optional[_RoundedBinary]:
    """D.1: for a positive x and a non-integral binary fraction `y = c / 2**k`
    (c odd), `x**y` is a binary fraction exactly when `x = a 2**b` (a odd) has
    `a = a1**(2**k)`, `2**k` divides `b`, and `a1 = 1` for a negative c. It must
    be computed when it may be a rounding boundary: a power of two, or an
    `a1**c` of at most `p + 2` bits."""
    var zeros_y = _trailing_zero_bits(y._significand)
    var c = y._significand >> zeros_y
    if y._negative:
        c = -c
    var k = y.precision() - y._exponent - zeros_y
    if k < 1 or k > 62:
        return None
    var degree = Integer(1) << k
    var zeros_x = _trailing_zero_bits(x._significand)
    var a = x._significand >> zeros_x
    var b = Integer(x._exponent - x.precision() + zeros_x)
    if b % degree:
        return None
    var root: Integer
    if a == 1:
        root = Integer(1)
    else:
        if c.sign() < 0 or degree > Integer(a.magnitude_bit_length()):
            return None
        var found = iroot_exact(a, degree)
        if not found:
            return None
        root = found.take()
    var shift = c * (b // degree)
    if root == 1:
        return _round_scaled_ratio(Integer(1), Integer(1), shift, False, target, False)
    if c * Integer(root.magnitude_bit_length() - 1) > Integer(target.format().precision() + 2):
        return None
    return _round_scaled_ratio(root ** Int(c), Integer(1), shift, False, target, False)


def _rational_value(v: _FloatInput) raises -> Rational:
    """The exact value of a finite argument as a Rational."""
    var numerator = -v.numerator if v.negative else v.numerator
    if v.scale >= 0:
        return Rational(numerator << Int(v.scale), v.denominator)
    return Rational(numerator, v.denominator << Int(-v.scale))


def _rational_exponent_exact(
    base: _FloatArgument, exponent: _FloatArgument, target: ArithmeticContext,
) raises -> Optional[_RoundedBinary]:
    """`x**(c/d)` for an exact positive x and an exact exponent `c/d` in
    lowest terms: it is rational exactly when `x**(1/d)` is (from Bezout's
    identity), and then the exact rational `r**c` rounds once. None when the
    root is irrational, or too large to compute."""
    var y = _rational_value(exponent.value)
    var d = y.denominator()
    var c = y.numerator()
    if d > Integer(1) << 20 or c.magnitude_bit_length() > 20:
        return None
    var root = root_exact(_rational_value(base.value), d)
    if not root:
        return None
    var r = root.take()
    var bits = max(r.numerator().magnitude_bit_length(), r.denominator().magnitude_bit_length())
    if Integer(bits) * abs(c) > Integer(1) << 24:
        return None
    var n = Int(abs(c))
    var value = Rational(r.numerator() ** n, r.denominator() ** n)
    if c.sign() < 0:
        value = Rational(1) / value
    return _round_ratio(value.numerator(), value.denominator(), target)
