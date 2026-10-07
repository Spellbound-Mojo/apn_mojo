"""Point kernels: enclosures of a function at exact arguments.

A kernel `_kernel_f(x, w)` returns a ball containing `f(x)` with at least
`w - c_f` bits of relative accuracy, for a documented `c_f <= 8`, whenever
`f(x) != 0`; zeros of `f` are exact cases and never reach a kernel. Kernels
are the bottom layer of every function: a correctly rounded Float function runs
the Ziv driver over its kernel at the exact argument, and a ball function
evaluates the kernel at its midpoint or at its ends.

`_RealKernel` names a function by code, so the driver compiles once for every
real function and constant. Its `budget` bounds the precision of argument
reduction, which only `sin`, `cos` and `tan` need beyond the working
precision; past it their kernels return an indeterminate ball, and the driver
then reaches its own budget error.
"""

from ..integer.value import Integer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._functions import _sqrt_float
from ..integer._word_math import _trailing_zero_bits
from .context import BallContext
from .value import Ball, _BallArgument
from ._arithmetic import _hull, _directed, _scale2, _product, _negate
from ._certified import _Enclosure
from ._constants import _pi, _e, _ln2, _log2_10, _euler_gamma, _catalan
from ._exp import _kernel_exp, _kernel_expm1, _kernel_exp2, _kernel_sinh, _kernel_cosh, _kernel_tanh
from ._log import _kernel_log, _kernel_log1p, _kernel_log2, _kernel_log10, _kernel_asinh, _kernel_acosh, _kernel_atanh
from ._trig import _kernel_sin_cos, _kernel_tan, _kernel_atan, _kernel_asin, _kernel_acos, _kernel_atan2
from ._power import _kernel_pow, _kernel_rootn

comptime _SQRT = 0
comptime _EXP = 1
comptime _EXPM1 = 2
comptime _EXP2 = 3
comptime _LOG = 4
comptime _LOG1P = 5
comptime _LOG2 = 6
comptime _LOG10 = 7
comptime _SIN = 8
comptime _COS = 9
comptime _TAN = 10
comptime _ATAN = 11
comptime _ASIN = 12
comptime _ACOS = 13
comptime _SINH = 14
comptime _COSH = 15
comptime _TANH = 16
comptime _ASINH = 17
comptime _ACOSH = 18
comptime _ATANH = 19
comptime _POW = 20
comptime _ATAN2 = 21
comptime _ROOTN = 22
comptime _PI = 30
comptime _EULER_E = 31
comptime _LN2 = 32
comptime _LOG2_10 = 33
comptime _EULER_GAMMA = 34
comptime _CATALAN = 35
comptime _HALF_PI = 36
comptime _QUARTER_PI = 37
comptime _THREE_QUARTER_PI = 38


def _function_name(code: Int) -> StaticString:
    var names: List[StaticString] = [
        "sqrt", "exp", "expm1", "exp2", "log", "log1p", "log2", "log10", "sin", "cos", "tan",
        "atan", "asin", "acos", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "pow", "atan2", "rootn",
    ]
    if code >= 0 and code < len(names):
        return names[code]
    var constants: List[StaticString] = [
        "pi", "euler_e", "ln2", "log2_10", "euler_gamma", "catalan", "pi/2", "pi/4", "3pi/4",
    ]
    if code >= _PI and code <= _THREE_QUARTER_PI:
        return constants[code - _PI]
    return "function"


def _is_constant(code: Int) -> Bool:
    return code >= _PI and code <= _THREE_QUARTER_PI


def _constant_kernel(code: Int, w: Int) raises -> Ball:
    if code == _PI:
        return _pi(w)
    if code == _EULER_E:
        return _e(w)
    if code == _LN2:
        return _ln2(w)
    if code == _LOG2_10:
        return _log2_10(w)
    if code == _EULER_GAMMA:
        return _euler_gamma(w)
    if code == _CATALAN:
        return _catalan(w)
    var c = Optional[BallContext](BallContext(w + 16))
    var pi = _pi(w)
    if code == _HALF_PI:
        return _scale2(_BallArgument(pi), Integer(-1), c)
    if code == _QUARTER_PI:
        return _scale2(_BallArgument(pi), Integer(-2), c)
    return _scale2(_BallArgument(_product(_BallArgument(pi), _BallArgument(Integer(3)), c)), Integer(-2), c)


def _kernel_sqrt(x: Float, w: Int) raises -> Ball:
    """The square root of a positive Float: both directed roundings at `w`
    bits, so `c = 1`."""
    var low = Float(_rounded=_sqrt_float(x, _directed(w, False)))
    var high = Float(_rounded=_sqrt_float(x, _directed(w, True)))
    return _hull(low, high, w)


def _point_kernel(code: Int, x: Float, y: Float, n: Int, w: Int, budget: Int) raises -> Ball:
    """The kernel of a real function at exact arguments: `x` is the argument
    (the first of two), `y` the second, `n` an integer parameter."""
    if code == _SQRT:
        return _kernel_sqrt(x, w)
    if code == _EXP:
        return _kernel_exp(x, w)
    if code == _EXPM1:
        return _kernel_expm1(x, w)
    if code == _EXP2:
        return _kernel_exp2(x, w)
    if code == _LOG:
        return _kernel_log(x, w)
    if code == _LOG1P:
        return _kernel_log1p(x, w)
    if code == _LOG2:
        return _kernel_log2(x, w)
    if code == _LOG10:
        return _kernel_log10(x, w)
    if code == _SIN:
        return _kernel_sin_cos(x, w, budget)[0]
    if code == _COS:
        return _kernel_sin_cos(x, w, budget)[1]
    if code == _TAN:
        return _kernel_tan(x, w, budget)
    if code == _ATAN:
        return _kernel_atan(x, w)
    if code == _ASIN:
        return _kernel_asin(x, w)
    if code == _ACOS:
        return _kernel_acos(x, w)
    if code == _SINH:
        return _kernel_sinh(x, w)
    if code == _COSH:
        return _kernel_cosh(x, w)
    if code == _TANH:
        return _kernel_tanh(x, w)
    if code == _ASINH:
        return _kernel_asinh(x, w)
    if code == _ACOSH:
        return _kernel_acosh(x, w)
    if code == _ATANH:
        return _kernel_atanh(x, w)
    if code == _POW:
        return _kernel_pow(x, y, w)
    if code == _ATAN2:
        return _kernel_atan2(x, y, w)
    if code == _ROOTN:
        return _kernel_rootn(x, n, w)
    if _is_constant(code):
        return _constant_kernel(code, w)
    raise Error(String("No kernel for function code ", code))


@fieldwise_init
struct _RealKernel(_Enclosure):
    """A real function, by code, at exact arguments; `negate` flips the result."""

    var code: Int
    var x: Float
    var y: Float
    var n: Int
    var budget: Int
    var negate: Bool

    @staticmethod
    def unary(code: Int, x: Float, budget: Int = Int.MAX) raises -> Self:
        return Self(code, x, x, 0, budget, False)

    @staticmethod
    def binary(code: Int, x: Float, y: Float, budget: Int = Int.MAX) raises -> Self:
        return Self(code, x, y, 0, budget, False)

    @staticmethod
    def constant(code: Int, negate: Bool = False) raises -> Self:
        var zero = Float._placeholder()
        return Self(code, zero, zero, 0, Int.MAX, negate)

    def enclosure(self, precision: Int) raises -> Ball:
        var result = _point_kernel(self.code, self.x, self.y, self.n, precision, self.budget)
        return _negate(result) if self.negate else result^

    def describe(self) raises -> String:
        if _is_constant(self.code):
            return String(_function_name(self.code))
        if self.code == _POW or self.code == _ATAN2:
            return String(_function_name(self.code), " at (", self.x, ", ", self.y, ")")
        return String(_function_name(self.code), " at ", self.x)


def _integral(t: Float) raises -> Optional[Integer]:
    """The integer value of a finite Float, when it has no fraction."""
    if t.is_zero():
        return Integer(0)
    var p = t.precision()
    if t._exponent < p and _trailing_zero_bits(t._significand) < p - t._exponent:
        return None
    return t.trunc()


def _power_of_two(t: Float) raises -> Optional[Int]:
    """k when a finite positive Float is exactly `2**k`."""
    if t.is_zero() or t._negative or t._kind != 1:
        return None
    if t._significand != Integer(1) << (t.precision() - 1):
        return None
    return t._exponent - 1


def _power_of_ten(t: Float) raises -> Optional[Int]:
    """k when a finite Float is exactly `10**k` for an integer `k >= 0`."""
    if t._negative or t._kind != 1 or t._exponent < 1:
        return None
    var n = _integral(t)
    if not n:
        return None
    var value = n.value()
    var k = _trailing_zero_bits(value)
    if Integer(10) ** k == value:
        return k
    return None


def _exact_ball(value: Integer, w: Int) raises -> Ball:
    return Ball(value, precision=max(w + 8, value.magnitude_bit_length()))


def _point(code: Int, t: Float, n: Int, w: Int, budget: Int) raises -> Ball:
    """A real function at an exact point: exact values exactly (zeros of the
    function, `exp(0) = 1`, `log(1) = 0`, `log2(2**k) = k`, ...), otherwise
    its kernel. Points outside the domain are indeterminate."""
    if t.is_zero():
        if code == _EXP or code == _EXP2 or code == _COS or code == _COSH:
            return _exact_ball(Integer(1), w)
        if code == _ACOS:
            return _constant_kernel(_HALF_PI, w)
        if code == _LOG or code == _LOG2 or code == _LOG10 or code == _ACOSH:
            return Ball.indeterminate(w + 8)
        if code == _ROOTN or code == _SQRT:
            return _exact_ball(Integer(0), w)
        return _exact_ball(Integer(0), w)
    var one = t == Float(1)
    if one and (code == _LOG or code == _LOG2 or code == _LOG10 or code == _ACOS or code == _ACOSH):
        return _exact_ball(Integer(0), w)
    if code == _EXP2:
        var whole = _integral(t)
        if whole:
            return _scale2(_BallArgument(_exact_ball(Integer(1), w)), whole.value(), Optional[BallContext](BallContext(w + 8)))
    if code == _LOG2:
        var k = _power_of_two(t)
        if k:
            return _exact_ball(Integer(k.value()), w)
    if code == _LOG10:
        var k = _power_of_ten(t)
        if k:
            return _exact_ball(Integer(k.value()), w)
    return _point_kernel(code, t, t, n, w, budget)
