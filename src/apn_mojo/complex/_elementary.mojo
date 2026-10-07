"""The shared steps of correctly rounded Complex functions.

Each part is correctly rounded in its own context. In order:

1. Special values: NaN and infinite parts follow C99 Annex G, with the
   requirements' Appendix B decisions for the cases it leaves open.
2. Structural parts: on the real or imaginary axis one part is an exact
   signed zero and the other a real function of one part, so both come from
   the correctly rounded real functions. The sign of a zero part follows the
   function's formula, `cosh(x) sin(+-0)` and the like.
3. A point on a branch cut takes the side its signed zero selects: the
   counter-clockwise continuous side directly, the other through
   `f(z) = conj(f(conj z))` (real-axis cuts) or `f(z) = -conj(f(-conj z))`
   (imaginary-axis cuts).
4. Otherwise the two-part Ziv driver runs over the complex ball function at
   the exact point.

`sin`, `cos`, `tan`, `asin` and `atan` follow from the hyperbolic functions:
`sin z = -i sinh(iz)`, `cos z = cosh(iz)`, `tan z = -i tanh(iz)`,
`asin z = -i asinh(iz)`, `atan z = -i atanh(iz)`. Multiplying by `-i` moves the
imaginary part to the real part and negates the real part into the imaginary
part, so the hyperbolic function runs with the parts' contexts swapped, the
negated part's directed rounding flipped.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode, _exact_context
from ..float.status import NumericStatus
from ..float._arithmetic import _float_special, _float_operation
from ..float._rounding import _RoundedBinary, _round_ratio, _inexact_error, _finish_round
from ..float._elementary import _real_rounded, _atan2_rounded, _pow_rounded, _scaled_down, _underflow, _unit
from ..complex.value import Complex
from ..complex.context import ComplexContext, _ComplexContextArgument
from ..complex._input import _ComplexArgument
from ..complex._power import _pow_complex
from ..exact_complex.value import ExactComplex
from ..exact_complex.math import sqrt_exact as _sqrt_exact, pow_int as _exact_pow_int
from ..integer._word_math import _trailing_zero_bits
from ..ball._arithmetic import _product
from ..ball._exp import _exp_ball
from ..ball._constants import _pi
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._certified import _PairEnclosure, _round_certified_pair, _Enclosure, _round_certified, _round_near, _budget_error
from ..ball._functions import _ball_function
from ..ball.math import divide as _ball_divide
from ..ball._kernels import (
    _RealKernel, _integral, _constant_kernel,
    _EXP, _LOG, _SIN, _COS, _TAN, _ATAN, _ASIN, _ACOS, _SINH, _COSH, _TANH, _ASINH, _ACOSH, _ATANH, _POW,
    _SQRT, _PI, _HALF_PI, _QUARTER_PI, _THREE_QUARTER_PI, _function_name,
)
from ..complex_ball.value import ComplexBall
from ..complex_ball.elementary import (
    exp as _ball_exp, log as _ball_log, pow as _ball_pow, sinh as _ball_sinh, cosh as _ball_cosh,
    tanh as _ball_tanh, asinh as _ball_asinh, acosh as _ball_acosh, atanh as _ball_atanh, acos as _ball_acos,
)

comptime _Pair = Tuple[_RoundedBinary, _RoundedBinary]


def _targets(z: Complex, context: _ComplexContextArgument) raises -> ComplexContext:
    if context:
        return context.value()
    return ComplexContext(
        real=ArithmeticContext(format=z._real.format()), imag=ArithmeticContext(format=z._imag.format()),
    )


def _nan(t: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(3, False, 0, t, False)


def _invalid(t: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(3, False, 16, t, False)


def _inf(negative: Bool, t: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(2, negative, 0, t, False)


def _zero(negative: Bool, t: ArithmeticContext) raises -> _RoundedBinary:
    return _float_special(0, negative, 0, t, False)


def _one(t: ArithmeticContext) raises -> _RoundedBinary:
    return _round_ratio(Integer(1), Integer(1), t)


def _const(code: Int, negative: Bool, t: ArithmeticContext) raises -> _RoundedBinary:
    return _round_certified(_RealKernel.constant(code, negative), t)


def _flipped(t: ArithmeticContext) -> ArithmeticContext:
    """The context whose rounding of `-v` negates this context's of `v`."""
    var result = t
    if t._rounding == RoundingMode.toward_positive:
        result._rounding = RoundingMode.toward_negative
    elif t._rounding == RoundingMode.toward_negative:
        result._rounding = RoundingMode.toward_positive
    return result^


def _negated(var r: _RoundedBinary) -> _RoundedBinary:
    if r.kind != 3:
        r.negative = not r.negative
    var direction = r.status._direction
    if direction == 1 or direction == -1:
        r.status = NumericStatus._make(r.status._flags, -direction)
    return r^


def _sign_of(code: Int, y: Float, cr: ArithmeticContext, ci: ArithmeticContext) raises -> Bool:
    """Whether `cos y` (or `sin y`) of a finite nonzero y is negative; neither
    vanishes at a nonzero binary fraction. Raises past the contexts' budget."""
    var budget = min(cr._budget(), ci._budget())
    var sign = _sign_beyond(code, y, budget, budget)
    if sign == 0:
        raise _budget_error(String("the sign of ", _function_name(code), " at ", y), budget)
    return sign < 0


def _halved(var r: _RoundedBinary, t: ArithmeticContext) raises -> Optional[_RoundedBinary]:
    """A correctly rounded value halved exactly, when the result stays in range."""
    if r.kind != 1:
        return r^
    if r.exponent - 1 < t.format().emin():
        return None
    r.exponent -= 1
    return r^


@fieldwise_init
struct _ComplexKernel(_PairEnclosure):
    """A complex function, by code, at exact arguments `x + iy` (and `a + ib`
    for pow)."""

    var code: Int
    var x: Float
    var y: Float
    var a: Float
    var b: Float

    def enclosure(self, precision: Int) raises -> Tuple[Ball, Ball]:
        var z = ComplexBall(_real=Ball(self.x), _imag=Ball(self.y))
        var c = Optional[BallContext](BallContext(precision + 8))
        var result: ComplexBall
        if self.code == _EXP:
            result = _ball_exp(z, context=c)
        elif self.code == _LOG:
            result = _ball_log(z, context=c)
        elif self.code == _POW:
            result = _ball_pow(z, ComplexBall(_real=Ball(self.a), _imag=Ball(self.b)), context=c)
        elif self.code == _SINH:
            result = _ball_sinh(z, context=c)
        elif self.code == _COSH:
            result = _ball_cosh(z, context=c)
        elif self.code == _TANH:
            result = _ball_tanh(z, context=c)
        elif self.code == _ASINH:
            result = _ball_asinh(z, context=c)
        elif self.code == _ACOSH:
            result = _ball_acosh(z, context=c)
        elif self.code == _ATANH:
            result = _ball_atanh(z, context=c)
        else:
            result = _ball_acos(z, context=c)
        return (result.real(), result.imag())

    def describe(self) raises -> String:
        return String("complex ", _function_name(self.code), " at Complex(", self.x, ", ", self.y, ")")


@fieldwise_init
struct _ComplexPart(_Enclosure):
    """One part of a complex function's enclosure, for a part that rounds on
    its own once a rule has decided the other."""

    var f: _ComplexKernel
    var imaginary: Bool

    def enclosure(self, precision: Int) raises -> Ball:
        var pair = self.f.enclosure(precision)
        return pair[1] if self.imaginary else pair[0]

    def describe(self) raises -> String:
        return self.f.describe()


def _ziv(code: Int, x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    if cr.format()._is_exact() or ci.format()._is_exact():
        raise _inexact_error(_function_name(code))
    return _round_certified_pair(_ComplexKernel(code, x, y, x, y), cr, ci, guard)


def _conjugated(r: _Pair) -> _Pair:
    return (r[0], _negated(r[1]))


def _neg_float(x: Float) -> Float:
    return -x


# ------------------------------------------------------------------- exp


def _cexp(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    var kx = x._kind
    var ky = y._kind
    if kx == 3 or ky == 3:
        if kx == 3 and ky == 0:
            return (_nan(cr), _zero(y._negative, ci))
        if ky == 3 and kx == 2:
            if x._negative:
                return (_zero(False, cr), _zero(False, ci))
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 2:
        if kx == 2:
            if x._negative:
                return (_zero(False, cr), _zero(False, ci))
            return (_inf(False, cr), _nan(ci))
        return (_invalid(cr), _invalid(ci))
    if kx == 2:
        if ky == 0:
            if x._negative:
                return (_zero(False, cr), _zero(y._negative, ci))
            return (_inf(False, cr), _zero(y._negative, ci))
        var cos_negative = _sign_of(_COS, y, cr, ci)
        var sin_negative = _sign_of(_SIN, y, cr, ci)
        if x._negative:
            return (_zero(cos_negative, cr), _zero(sin_negative, ci))
        return (_inf(cos_negative, cr), _inf(sin_negative, ci))
    if ky == 0:
        if kx == 0:
            return (_one(cr), _zero(y._negative, ci))
        return (_real_rounded(_EXP, x, cr), _zero(y._negative, ci))
    if kx == 0:
        # exp(+-0 + iy) = cos y + i sin y.
        return (_real_rounded(_COS, y, cr), _real_rounded(_SIN, y, ci))
    return _ziv(_EXP, x, y, cr, ci, guard)


# ------------------------------------------------------------------- log


def _clog(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    var kx = x._kind
    var ky = y._kind
    if kx == 3 or ky == 3:
        if kx == 2 or ky == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if kx == 0 and ky == 0:
        # A pole: -inf with divide-by-zero, and the angle of the signed zeros.
        return (_float_special(2, True, 8, cr, False), _atan2_rounded(y, x, ci))
    if ky == 2:
        if kx == 2:
            return (_inf(False, cr), _const(_THREE_QUARTER_PI if x._negative else _QUARTER_PI, y._negative, ci))
        return (_inf(False, cr), _const(_HALF_PI, y._negative, ci))
    if kx == 2:
        if x._negative:
            return (_inf(False, cr), _const(_PI, y._negative, ci))
        return (_inf(False, cr), _zero(y._negative, ci))
    if ky == 0:
        if not x._negative:
            return (_real_rounded(_LOG, x, cr), _zero(y._negative, ci))
        return (_real_rounded(_LOG, -x, cr), _const(_PI, y._negative, ci))
    if kx == 0:
        return (_real_rounded(_LOG, abs(y), cr), _const(_HALF_PI, y._negative, ci))
    # log|z| = log(x**2 + y**2) / 2, the square sum exact; arg z = atan2(y, x).
    var square = Float(_rounded=_float_operation(
        Float(_rounded=_float_operation(x, x, 2, _exact_context())),
        Float(_rounded=_float_operation(y, y, 2, _exact_context())), 0, _exact_context(),
    ))
    var real = _halved(_real_rounded(_LOG, square, cr), cr)
    var imag = _atan2_rounded(y, x, ci)
    if real:
        return (real.take(), imag)
    return (_ziv(_LOG, x, y, cr, ci, guard)[0], imag)


# ------------------------------------------------------------------- pow


def _cpow(x: Float, y: Float, a: Float, b: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    """`(x + iy)**(a + ib)` with its special values and the exact cases of
    the requirements' Appendix D.2."""
    if a.is_zero() and b.is_zero():
        return (_one(cr), _zero(False, ci))
    if x._kind == 1 and x == Float(1) and y.is_zero():
        return (_one(cr), _zero(_real_power_zero(y, a, ci), ci))
    if x.is_nan() or y.is_nan() or a.is_nan() or b.is_nan():
        return (_nan(cr), _nan(ci))
    if x.is_infinite() or y.is_infinite() or a.is_infinite() or b.is_infinite():
        return (_invalid(cr), _invalid(ci))
    if x.is_zero() and y.is_zero():
        if a._negative or a.is_zero():
            return (_invalid(cr), _invalid(ci))
        return (_zero(False, cr), _zero(False, ci))
    if b.is_zero():
        var whole = _integral(a)
        if whole:
            return _pow_complex(
                _ComplexArgument(Complex(_real=x, _imag=y)), whole.value(),
                _ComplexContextArgument(ComplexContext(_validated=(cr, ci))),
            )
        if y.is_zero():
            if not x._negative:
                # A positive real to a real power: real, with a structural zero.
                return (_pow_rounded(x, a, cr), _zero(a._negative != y._negative, ci))
            var twice = _integral(Float(_rounded=_float_operation(a, a, 0, _exact_context())))
            if twice:
                # (-r)**(c/2) for an odd c: +0 + i s r**(c/2) with
                # s = (-1)**((c-1)/2), negated on the -0 side.
                var c = twice.value()
                var negative = (((c - 1) // 2) & 1).__bool__() != y._negative
                var magnitude = _pow_rounded(-x, a, _flipped(ci) if negative else ci)
                return (_zero(False, cr), _negated(magnitude) if negative else magnitude)
    var exact = _exact_power(x, y, a, b, cr, ci)
    if exact:
        return exact.take()
    return _round_certified_pair(_ComplexKernel(_POW, x, y, a, b), cr, ci, guard)


def _real_power_zero(y: Float, a: Float, ci: ArithmeticContext) -> Bool:
    """Whether the imaginary zero of a real `z**w` with `|z| = 1` is -0:
    when rounding toward -inf, or when the signs of Im z and Re w differ."""
    return ci.rounding() == RoundingMode.toward_negative or y._negative != a._negative


@fieldwise_init
struct _ExpOfPi(_Enclosure):
    """`e**(s pi)` for an exact s."""

    var s: Float

    def enclosure(self, precision: Int) raises -> Ball:
        var c = Optional[BallContext](BallContext(precision + 8))
        return _exp_ball(_product(_BallArgument(Ball(self.s)), _BallArgument(_pi(precision + 16)), c), precision)

    def describe(self) raises -> String:
        return String("exp at ", self.s, " pi")


def _exact_power(x: Float, y: Float, a: Float, b: Float, cr: ArithmeticContext, ci: ArithmeticContext) raises -> Optional[_Pair]:
    """D.2's exact and structural powers of `z = x + iy`.

    For a real exponent `c / 2**k`, z**w is exact when `k` principal square
    roots of z are; then the exact power rounds once. For `z` in `{-1, i, -i}`
    and `w = u + iv`, `z**w = e**(-v t) (cos(u t) + i sin(u t))` with `t` the
    argument. The power is real when `u t` is a multiple of pi,
    with the imaginary zero of `_real_power_zero`, and imaginary when it is an
    odd multiple of pi/2, with a real part of +0; a part whose cosine or sine vanishes exactly is a structural
    `+0`, and the other part is `+-e**(-v t)`, correctly rounded."""
    if b.is_zero() and not a.is_zero():
        var zeros = _trailing_zero_bits(a._significand)
        var c = a._significand >> zeros
        if a._negative:
            c = -c
        var k = a.precision() - a._exponent - zeros
        if k >= 1 and k <= 20 and c.magnitude_bit_length() <= 20:
            var conjugate = y.is_zero() and y._negative
            var z = ExactComplex(x.to_rational_exact(), (-y if conjugate else y).to_rational_exact())
            var found = True
            for _ in range(k):
                var root = _sqrt_exact(z)
                if not root:
                    found = False
                    break
                z = root.take()
            if found:
                var power = _exact_pow_int(z, c)
                var re = power.real()
                var im = power.imag()
                if conjugate:
                    im = -im
                var real_part = _zero(False, cr) if re.sign() == 0 else _round_ratio(re.numerator(), re.denominator(), cr)
                var imag_part = _zero(y._negative, ci) if im.sign() == 0 else _round_ratio(im.numerator(), im.denominator(), ci)
                return (real_part, imag_part)
    if b.is_zero():
        return None
    # z in {-1, i, -i}: t = +-pi for -1 (by y's zero), +-pi/2 for +-i.
    var t: Rational
    if x == Float(-1) and y.is_zero():
        t = Rational(-1) if y._negative else Rational(1)
    elif x.is_zero() and abs(y) == Float(1):
        t = Rational(-1, 2) if y._negative else Rational(1, 2)
    else:
        return None
    var u = a.to_rational_exact()
    var phase = u * t
    # e**(-v t pi), exactly as a multiple of pi.
    var s = Float(-b.to_rational_exact() * t, context=_exact_context())
    if phase.denominator() == Integer(1):
        # cos(phase pi) = (-1)**phase and sin(phase pi) = 0.
        var odd = (phase.numerator() & 1).__bool__()
        var size = _round_certified(_ExpOfPi(s), _flipped(cr) if odd else cr)
        return (_negated(size) if odd else size, _zero(_real_power_zero(y, a, ci), ci))
    var shifted = phase - Rational(1, 2)
    if shifted.denominator() == Integer(1):
        # cos(phase pi) = 0 and sin(phase pi) = (-1)**(phase - 1/2).
        var negative = (shifted.numerator() & 1).__bool__()
        var magnitude = _round_certified(_ExpOfPi(s), _flipped(ci) if negative else ci)
        return (_zero(False, cr), _negated(magnitude) if negative else magnitude)
    return None


# --------------------------------------------------- hyperbolic functions


def _csinh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    # Odd and conjugate-symmetric: work with a positive-signed real part.
    if x._negative and not x.is_nan():
        var r = _csinh(-x, -y, _flipped(cr), _flipped(ci), guard)
        return (_negated(r[0]), _negated(r[1]))
    var kx = x._kind
    var ky = y._kind
    if kx == 3:
        if ky == 0:
            return (_nan(cr), _zero(y._negative, ci))
        return (_nan(cr), _nan(ci))
    if ky == 3:
        if kx == 0:
            return (_zero(False, cr), _nan(ci))
        if kx == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 2:
        if kx == 0:
            return (_zero(False, cr), _invalid(ci))
        if kx == 2:
            return (_inf(False, cr), _invalid(ci))
        return (_invalid(cr), _invalid(ci))
    if kx == 2:
        if ky == 0:
            return (_inf(False, cr), _zero(y._negative, ci))
        return (_inf(_sign_of(_COS, y, cr, ci), cr), _inf(_sign_of(_SIN, y, cr, ci), ci))
    if ky == 0:
        # sinh x + i cosh(x) sin(+-0).
        return (_real_rounded(_SINH, x, cr) if kx != 0 else _zero(False, cr), _zero(y._negative, ci))
    if kx == 0:
        # sinh(+0) cos y + i sin y.
        return (_zero(_sign_of(_COS, y, cr, ci), cr), _real_rounded(_SIN, y, ci))
    return _ziv(_SINH, x, y, cr, ci, guard)


def _ccosh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    # The NaN cases come first: the even reflection below would lose the
    # zeros' signs. cosh(NaN +- 0i) = NaN -+ 0i and cosh(+-0 + NaN i) = NaN +- 0i.
    if x.is_nan() or y.is_nan():
        if x.is_nan() and y.is_zero():
            return (_nan(cr), _zero(not y._negative, ci))
        if y.is_nan() and x.is_zero():
            return (_nan(cr), _zero(x._negative, ci))
        if y.is_nan() and x.is_infinite():
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    # Even and conjugate-symmetric: cosh(-z) = cosh(z).
    if x._negative:
        return _ccosh(-x, -y, cr, ci, guard)
    var kx = x._kind
    var ky = y._kind
    if ky == 2:
        # The zero or the infinity takes y's sign.
        if kx == 0:
            return (_invalid(cr), _zero(y._negative, ci))
        if kx == 2:
            return (_inf(y._negative, cr), _invalid(ci))
        return (_invalid(cr), _invalid(ci))
    if kx == 2:
        if ky == 0:
            return (_inf(False, cr), _zero(y._negative, ci))
        return (_inf(_sign_of(_COS, y, cr, ci), cr), _inf(_sign_of(_SIN, y, cr, ci), ci))
    if ky == 0:
        # cosh x + i sinh(x) sin(+-0).
        if kx == 0:
            return (_one(cr), _zero(y._negative, ci))
        return (_real_rounded(_COSH, x, cr), _zero(y._negative, ci))
    if kx == 0:
        # cos y + i sinh(+0) sin y.
        return (_real_rounded(_COS, y, cr), _zero(_sign_of(_SIN, y, cr, ci), ci))
    return _ziv(_COSH, x, y, cr, ci, guard)


def _sign_beyond(code: Int, t: Float, margin: Int, budget: Int) raises -> Int:
    """The sign of `sin t` or `cos t` at an exact nonzero t, counting a value
    only once it is beyond `2**-margin` in magnitude: -1 or 1, or 0 when the
    budget ends first."""
    var bound = Float(_rounded=_RoundedBinary(1, False, Integer(1), 1 - margin, FloatFormat(1), NumericStatus()))
    var w = 64
    while w <= budget:
        var value = _ball_function(code, _BallArgument(Ball(t)), 0, Optional[BallContext](BallContext(w)))
        if value.is_finite():
            if value._exact_lower() > bound:
                return 1
            if value._exact_upper() < -bound:
                return -1
        w *= 2
    return 0


def _ctanh_far(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> Optional[_Pair]:
    """tanh(x + iy) for x >= 1 whose real part is within `2**-(p+3)` of 1.

    With `E = e**-2x`, the real part is `1 - d`, `d = (E + cos 2y) / (cosh 2x
    + cos 2y)`, and the imaginary part is `sin 2y / (cosh 2x + cos 2y)`. For
    x >= 1 the denominator is at least `e**2x / 2 - 1 >= 0.364 e**2x`, so both
    `|d|` and the imaginary part are below `3.12 E < 2**(2 - T)` with `T =
    floor(2.8852 x) <= 2x log2(e)`. The real part rounds as `1 -/+ 2**(1 -
    T)` once the sign of `E + cos 2y` is known (`_round_near`); the imaginary
    part, of the sign of `sin 2y`, underflows below `2**(emin - 2)` and
    otherwise rounds by its own iteration, where it has full relative
    accuracy. An enclosure of the real part always contains 1 until its width
    is below `d`, so no iteration could decide it."""
    var t = _scaled_down(x, 28852, 10000)
    var tiny = 2 - t
    if tiny > -cr.format().precision() - 3:
        return None
    var doubled = Float(_rounded=_float_operation(y, y, 0, _exact_context()))
    var budget = min(cr._budget(), ci._budget())
    # cos 2y > 0, or cos 2y < -2**-T < -E, fixes the sign of E + cos 2y.
    var cosine = _sign_beyond(_COS, doubled, t, budget)
    if cosine == 0:
        return None
    var real_part = _round_near(_unit(False), cosine < 0, tiny, cr)
    if not real_part:
        return None
    if tiny <= ci.format().emin() - 2:
        var sine = _sign_beyond(_SIN, doubled, budget, budget)
        if sine == 0:
            return None
        return (real_part.take(), _underflow(sine < 0, ci))
    return (real_part.take(), _round_certified(_ComplexPart(_ComplexKernel(_TANH, x, y, x, y), True), ci, guard))


def _ctanh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    # The NaN cases come first: the odd reflection below would flip their
    # zeros. tanh(NaN +- 0i) = NaN +- 0i and tanh(+-inf + NaN i) = +-1 + 0i.
    if x.is_nan() or y.is_nan():
        if x.is_nan() and y.is_zero():
            return (_nan(cr), _zero(y._negative, ci))
        if y.is_nan() and x.is_infinite():
            var one = _one(cr)
            return (_negated(one) if x._negative else one, _zero(False, ci))
        return (_nan(cr), _nan(ci))
    if x._negative:
        var r = _ctanh(-x, -y, _flipped(cr), _flipped(ci), guard)
        return (_negated(r[0]), _negated(r[1]))
    var kx = x._kind
    var ky = y._kind
    if kx == 2:
        # 1 + i 0 sin(2y): the zero takes the sign of sin(2y), or y's for an infinite or NaN y.
        if ky == 1:
            var doubled = Float(_rounded=_float_operation(y, y, 0, _exact_context()))
            return (_one(cr), _zero(_sign_of(_SIN, doubled, cr, ci), ci))
        return (_one(cr), _zero(y._negative, ci))
    if ky == 2:
        return (_invalid(cr), _invalid(ci))
    if ky == 0:
        return (_real_rounded(_TANH, x, cr) if kx != 0 else _zero(False, cr), _zero(y._negative, ci))
    if kx == 0:
        # tanh(+0 + iy) = +0 + i tan y.
        return (_zero(False, cr), _real_rounded(_TAN, y, ci))
    if x._exponent > 0 and not (cr.format()._is_exact() or ci.format()._is_exact()):
        var far = _ctanh_far(x, y, cr, ci, guard)
        if far:
            return far.take()
    return _ziv(_TANH, x, y, cr, ci, guard)


def _casinh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    if x._negative and not x.is_nan():
        var r = _casinh(-x, -y, _flipped(cr), _flipped(ci), guard)
        return (_negated(r[0]), _negated(r[1]))
    var kx = x._kind
    var ky = y._kind
    if kx == 3:
        if ky == 0:
            return (_nan(cr), _zero(y._negative, ci))
        if ky == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 3:
        if kx == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 2:
        return (_inf(False, cr), _const(_QUARTER_PI if kx == 2 else _HALF_PI, y._negative, ci))
    if kx == 2:
        return (_inf(False, cr), _zero(y._negative, ci))
    if ky == 0:
        return (_real_rounded(_ASINH, x, cr) if kx != 0 else _zero(False, cr), _zero(y._negative, ci))
    if kx == 0:
        var magnitude = abs(y)
        if magnitude <= Float(1):
            return (_zero(False, cr), _real_rounded(_ASIN, y, ci))
        # On a cut: x = +0 selects the right side, the continuous one from
        # the right for (i, i inf) and from the left for (-i inf, -i).
        return (_real_rounded(_ACOSH, magnitude, cr), _const(_HALF_PI, y._negative, ci))
    return _ziv(_ASINH, x, y, cr, ci, guard)


def _cacosh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    # Conjugate-symmetric: work with a positive-signed imaginary part.
    if y._negative and not y.is_nan():
        return _conjugated(_cacosh(x, -y, cr, _flipped(ci), guard))
    var kx = x._kind
    var ky = y._kind
    if kx == 3:
        if ky == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 3:
        if kx == 2:
            return (_inf(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 2:
        if kx == 2:
            return (_inf(False, cr), _const(_THREE_QUARTER_PI if x._negative else _QUARTER_PI, False, ci))
        return (_inf(False, cr), _const(_HALF_PI, False, ci))
    if kx == 2:
        if x._negative:
            return (_inf(False, cr), _const(_PI, False, ci))
        return (_inf(False, cr), _zero(False, ci))
    if ky == 0:
        if kx == 0:
            return (_zero(False, cr), _const(_HALF_PI, False, ci))
        if not x._negative and x >= Float(1):
            return (_real_rounded(_ACOSH, x, cr), _zero(False, ci))
        if x > Float(-1):
            return (_zero(False, cr), _real_rounded(_ACOS, x, ci))
        return (_real_rounded(_ACOSH, -x, cr), _const(_PI, False, ci))
    return _ziv(_ACOSH, x, y, cr, ci, guard)


def _catanh(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    if x._negative and not x.is_nan():
        var r = _catanh(-x, -y, _flipped(cr), _flipped(ci), guard)
        return (_negated(r[0]), _negated(r[1]))
    var kx = x._kind
    var ky = y._kind
    if kx == 3:
        if ky == 2:
            return (_zero(False, cr), _const(_HALF_PI, y._negative, ci))
        return (_nan(cr), _nan(ci))
    if ky == 3:
        if kx == 0:
            return (_zero(False, cr), _nan(ci))
        if kx == 2:
            return (_zero(False, cr), _nan(ci))
        return (_nan(cr), _nan(ci))
    if ky == 2 or kx == 2:
        return (_zero(False, cr), _const(_HALF_PI, y._negative, ci))
    if ky == 0:
        if kx == 0:
            return (_zero(False, cr), _zero(y._negative, ci))
        if x < Float(1):
            return (_real_rounded(_ATANH, x, cr), _zero(y._negative, ci))
        if x == Float(1):
            return (_float_special(2, False, 8, cr, False), _zero(y._negative, ci))
        # On the cut (1, inf): atanh(1/x) + i (+-pi/2), the side by y's sign.
        return (_cut_atanh(x, cr), _const(_HALF_PI, y._negative, ci))
    if kx == 0:
        return (_zero(False, cr), _real_rounded(_ATAN, y, ci))
    return _ziv(_ATANH, x, y, cr, ci, guard)


def _cut_atanh(x: Float, cr: ArithmeticContext) raises -> _RoundedBinary:
    """`atanh(1/x) = log((x + 1)/(x - 1)) / 2` for an x above 1, the real part
    of atanh on its cut, correctly rounded."""
    return _round_certified(_CutAtanh(x), cr)


@fieldwise_init
struct _CutAtanh(_Enclosure):
    var x: Float

    def enclosure(self, precision: Int) raises -> Ball:
        # 1/x at the working precision: the enclosure narrows as it rises, and
        # atanh keeps the relative accuracy of a small 1/x that the logarithm
        # of a ratio near 1 would cancel.
        var c = Optional[BallContext](BallContext(precision + 8))
        var inverse = _ball_divide(_BallArgument(Ball(1)), _BallArgument(Ball(self.x)), context=c)
        return _ball_function(_ATANH, _BallArgument(inverse), 0, c)

    def describe(self) raises -> String:
        return String("atanh at Complex(", self.x, ", 0)")


def _cacos(x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    # Conjugate-symmetric: cacos(conj z) = conj(cacos z).
    if y._negative and not y.is_nan():
        return _conjugated(_cacos(x, -y, cr, _flipped(ci), guard))
    var kx = x._kind
    var ky = y._kind
    if kx == 3:
        if ky == 2:
            return (_nan(cr), _inf(True, ci))
        return (_nan(cr), _nan(ci))
    if ky == 3:
        if kx == 0:
            return (_const(_HALF_PI, False, cr), _nan(ci))
        if kx == 2:
            return (_nan(cr), _inf(True, ci))
        return (_nan(cr), _nan(ci))
    if ky == 2:
        if kx == 2:
            return (_const(_THREE_QUARTER_PI if x._negative else _QUARTER_PI, False, cr), _inf(True, ci))
        return (_const(_HALF_PI, False, cr), _inf(True, ci))
    if kx == 2:
        if x._negative:
            return (_const(_PI, False, cr), _inf(True, ci))
        return (_zero(False, cr), _inf(True, ci))
    if ky == 0:
        var magnitude = abs(x)
        if magnitude <= Float(1):
            return (_real_rounded(_ACOS, x, cr), _zero(True, ci))
        # On a cut, y = +0 selects the side from above: acos(x + i0) is
        # 0 - i acosh(x) for x > 1 and pi - i acosh(-x) for x < -1.
        var real = _zero(False, cr) if not x._negative else _const(_PI, False, cr)
        return (real, _negated(_real_rounded(_ACOSH, magnitude, _flipped(ci))))
    return _ziv(_ACOS, x, y, cr, ci, guard)


# --------------------------------------------------- the trigonometric forms


def _times_i_form(function: Int, x: Float, y: Float, cr: ArithmeticContext, ci: ArithmeticContext, guard: Int) raises -> _Pair:
    """`f(z) = -i g(iz)` for g = sinh, tanh, asinh or atanh: g runs at
    `iz = -y + ix` with its real part rounded as f's imaginary part negated,
    and its imaginary part as f's real part."""
    if y.is_nan() and x.is_infinite():
        # asin(+-inf + NaN i) = NaN + i inf and atan(+-inf + NaN i) = +-pi/2 + 0i.
        if function == _ASIN:
            return (_nan(cr), _inf(False, ci))
        if function == _ATAN:
            return (_const(_HALF_PI, x._negative, cr), _zero(False, ci))
    var r: _Pair
    var flipped = _flipped(ci)
    if function == _SIN:
        r = _csinh(-y, x, flipped, cr, guard)
    elif function == _TAN:
        r = _ctanh(-y, x, flipped, cr, guard)
    elif function == _ASIN:
        r = _casinh(-y, x, flipped, cr, guard)
    else:
        r = _catanh(-y, x, flipped, cr, guard)
    return (r[1], _negated(r[0]))


def _complex_function(code: Int, z: Complex, context: _ComplexContextArgument, guard: Int = 0) raises -> _Pair:
    """A unary complex function, each part correctly rounded."""
    var target = _targets(z, context)
    var cr = target.real()
    var ci = target.imag()
    var x = z._real
    var y = z._imag
    if code == _EXP:
        return _cexp(x, y, cr, ci, guard)
    if code == _LOG:
        return _clog(x, y, cr, ci, guard)
    if code == _SINH:
        return _csinh(x, y, cr, ci, guard)
    if code == _COSH:
        return _ccosh(x, y, cr, ci, guard)
    if code == _TANH:
        return _ctanh(x, y, cr, ci, guard)
    if code == _ASINH:
        return _casinh(x, y, cr, ci, guard)
    if code == _ACOSH:
        return _cacosh(x, y, cr, ci, guard)
    if code == _ATANH:
        return _catanh(x, y, cr, ci, guard)
    if code == _ACOS:
        return _cacos(x, y, cr, ci, guard)
    if code == _COS:
        # These NaN cases keep the zero's sign, unlike cosh(iz)'s.
        if x.is_nan() or y.is_nan():
            if x.is_nan() and y.is_zero():
                return (_nan(cr), _zero(y._negative, ci))
            if y.is_nan() and x.is_zero():
                return (_nan(cr), _zero(x._negative, ci))
            if x.is_nan() and y.is_infinite():
                return (_inf(False, cr), _nan(ci))
            return (_nan(cr), _nan(ci))
        # cos z = cosh(iz) = cosh(-y + ix).
        return _ccosh(-y, x, cr, ci, guard)
    return _times_i_form(code, x, y, cr, ci, guard)


def _complex_pow(base: Complex, exponent: Complex, context: _ComplexContextArgument, guard: Int = 0) raises -> _Pair:
    var target = _targets(base, context)
    return _cpow(base._real, base._imag, exponent._real, exponent._imag, target.real(), target.imag(), guard)
