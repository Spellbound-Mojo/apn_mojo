"""The Hurwitz and Riemann zeta functions and the polygamma functions as
balls (scipy's zeta and polygamma).

For real s > 1 and q > 0, `zeta(s, q) = sum_{k>=0} (q+k)**-s`. While its terms
shrink fast the sum is direct, with the tail below its integral,
`(q+K)**(1-s) / (s-1)`. Otherwise the Euler-Maclaurin formula at `a = q + N`,

    sum_{k<N} (q+k)**-s + a**(1-s)/(s-1) + a**-s/2
        + sum_{j=1}^{M} B_2j/(2j)! (s)_{2j-1} a**(-s-2j+1) + R,

has `|R| <= 4 |(s)_2M| (2 pi)**-2M a**(1-s-2M) / (s+2M-1)` for real s with
`s + 2M - 1 > 0`, from `|B~_2M(x)| <= 4 (2M)! / (2 pi)**2M` (DLMF 24.9.4)
and `int_N^inf (q+x)**(-s-2M) dx`. With `a >= |s| + 2M` each correction is
below the last by about `(2 pi)**2`, so `M = (w + 10)/5` terms reach w bits.
The formula continues zeta(s, q) to every s other than 1, so it also gives
the Riemann zeta function, q = 1, on `-1 < s < 1`; below, the functional
equation `zeta(s) = 2**s pi**(s-1) sin(pi s/2) Gamma(1-s) zeta(1-s)`, with
`s/2` reduced exactly in the sine. Riemann's values at the non-positive
integers are rational, `zeta(-n) = -B_{n+1}/(n+1)` for n >= 1, 0 for even
n, and -1/2 at 0. A negative non-integer q, for an integer s, takes
`zeta(s, q) = sum_{k<m} (q+k)**-s + zeta(s, q+m)` with `q + m > 0`.

`polygamma(n, x) = (-1)**(n+1) n! zeta(n+1, x)` for n >= 1.

The domains are scipy's: zeta(s, q) is defined for s > 1 (every s other
than 1 when q = 1), is infinite at s = 1 and at the non-positive integers q,
and takes a negative q only for an integer s.
"""

from ..integer.value import Integer
from ..integer.math import factorial
from ..rational.value import Rational
from ..float.value import Float
from ..float.math import _exact_rational
from .context import BallContext
from .value import Ball, _BallArgument, _FINITE
from ._arithmetic import _working, _rounded_ball, _power, _negate
from ._kernels import _EXP, _LOG, _SIN
from ._constants import _pi
from ._special import (
    _ball_special, _GAMMA, _DIGAMMA, _add, _sub, _mul, _div, _fn, _c, _one, _scaled, _magnitude, _below, _tangent_numbers, _bernoulli, _sin_pi,
)

comptime _NEITHER = 0
comptime _RATIONAL = 1
comptime _INFINITE = 2
comptime _INVALID = 3


@fieldwise_init
struct _ZetaCase(ImplicitlyCopyable):
    """zeta(s, q) decided without the kernels: a rational value, infinite,
    outside the domain, or neither."""

    var kind: Int
    var value: Rational


def _int(n: Integer) raises -> Int:
    """A count of terms, which the budget keeps far below 2**62."""
    if n.magnitude_bit_length() > 62:
        raise Error("Cannot sum zeta: the argument needs more than 2**62 terms; use a smaller argument.")
    var v = Int(n._low_magnitude())
    return -v if n.sign() < 0 else v


def _case(kind: Int) -> _ZetaCase:
    return _ZetaCase(kind, Rational(0))


def _bernoulli_rational(n: Int) raises -> Rational:
    """`B_n` exactly, for n >= 0: 1, -1/2, then `B_2k = (-1)**(k-1) 2k T_k /
    (4**k (4**k - 1))` from the tangent numbers, and 0 at the other odd n."""
    if n == 0:
        return Rational(1)
    if n == 1:
        return Rational(-1, 2)
    if n % 2 == 1:
        return Rational(0)
    var k = n // 2
    var t = _tangent_numbers(k)[k - 1]
    var four = Integer(1) << (2 * k)
    var value = Rational(Integer(2 * k) * t, four * (four - 1))
    return value if k % 2 == 1 else -value


def _zeta_case(s: Rational, q: Rational) raises -> _ZetaCase:
    """The poles, the domain, and Riemann's rational values."""
    if s == Rational(1):
        return _case(_INFINITE)
    if q == Rational(1):
        if s.sign() == 0:
            return _ZetaCase(_RATIONAL, Rational(-1, 2))
        if s.is_integer() and s.sign() < 0:
            var n = Int((-s).numerator()._low_magnitude())
            return _ZetaCase(_RATIONAL, -_bernoulli_rational(n + 1) / Rational(n + 1))
        return _case(_NEITHER)
    if s < Rational(1):
        return _case(_INVALID)
    if q.is_integer() and q.sign() <= 0:
        return _case(_INFINITE)
    if q.sign() < 0 and not s.is_integer():
        return _case(_INVALID)
    return _case(_NEITHER)


def _negative_power(a: Ball, s: Ball, w: Int) raises -> Ball:
    """`a**-s` for a ball a > 0: an integer power for an exact integer s,
    else `exp(-s log a)`."""
    if s.is_exact() and s._midpoint.is_integer():
        return _power(_BallArgument(a), -s._midpoint.round(), _c(w))
    return _fn(_EXP, _negate(_mul(s, _fn(_LOG, a, w), w)), w)


def _hurwitz_positive(s: Ball, q: Ball, w: Int) raises -> Ball:
    """zeta(s, q) for q > 0 and `s > -1` other than 1 (the continuation for
    s < 1), by direct summation or Euler-Maclaurin."""
    var work = w + 16
    var m = (w + 10) // 5 + 4
    var s_top = s._exact_upper()
    var span = _int(s_top.ceil()) if s_top > Float(0) else _int((-s._exact_lower()).ceil())
    var reach = span + 2 * m + 1
    var total = Ball(Integer(0), precision=work)
    var a = q
    var above_one = s._exact_lower() > Float(1)
    for k in range(reach):
        if not (a._exact_lower() < Float(reach)):
            break
        var term = _negative_power(a, s, work)
        total = _add(total, term, work)
        a = _add(a, _one(work), work)
        if above_one and k >= 2 and not total._midpoint.is_zero() and _below(term, total._midpoint._exponent - w - 12):
            # The tail is below its integral: term (q+k+1) / (s-1) bounds it.
            var tail = _div(_mul(term, a, work), _sub(s, _one(work), work), work)
            return Ball(_midpoint=total._midpoint, _radius=total._radius.add(_magnitude(tail)), _kind=_FINITE)
    # Euler-Maclaurin at a = q + N.
    var power = _negative_power(a, s, work)
    var s_minus_one = _sub(s, _one(work), work)
    total = _add(total, _div(_mul(power, a, work), s_minus_one, work), work)
    total = _add(total, _scaled(power, -1, work), work)
    var inverse = _div(_one(work), a, work)
    var inverse_square = _mul(inverse, inverse, work)
    var tangents = _tangent_numbers(m)
    # c_j = (s)_{2j-1} a**(1-2j) / (2j)!, from c_1 = s / (2a).
    var c = _scaled(_mul(s, inverse, work), -1, work)
    var corrections = Ball(Integer(0), precision=work)
    for j in range(1, m + 1):
        if j > 1:
            var rise = _mul(_add(s, Ball(Integer(2 * j - 3), precision=work), work), _add(s, Ball(Integer(2 * j - 2), precision=work), work), work)
            c = _div(_mul(_mul(c, rise, work), inverse_square, work), Ball(Integer((2 * j - 1) * (2 * j)), precision=work), work)
        corrections = _add(corrections, _mul(_bernoulli(j, tangents[j - 1], work), c, work), work)
    total = _add(total, _mul(power, corrections, work), work)
    # |R| <= 4 |(s)_2M| (2 pi)**-2M a**(1-s-2M) / (s+2M-1).
    var rising = _one(work)
    for i in range(2 * m):
        rising = _mul(rising, _add(s, Ball(Integer(i), precision=work), work), work)
    var two_pi = _scaled(_pi(work), 1, work)
    var bound = _scaled(_mul(rising, _power(_BallArgument(_mul(two_pi, a, work)), Integer(-2 * m), _c(work)), work), 2, work)
    bound = _div(_mul(_mul(bound, power, work), a, work), _add(s, Ball(Integer(2 * m - 1), precision=work), work), work)
    return Ball(_midpoint=total._midpoint, _radius=total._radius.add(_magnitude(bound)), _kind=_FINITE)


def _hurwitz(s: Ball, q: Ball, w: Int) raises -> Ball:
    """zeta(s, q) for s > 1 and q > 0, or an integer s with a negative
    non-integer q, by shifting q above 0."""
    if q._exact_lower() > Float(0):
        return _hurwitz_positive(s, q, w)
    var work = w + 16
    var shift = _int((-q._exact_lower()).floor()) + 1
    var total = Ball(Integer(0), precision=work)
    var a = q
    for _ in range(shift):
        total = _add(total, _negative_power(a, s, work), work)
        a = _add(a, _one(work), work)
    return _add(total, _hurwitz_positive(s, a, work), work)


def _riemann(s: Ball, w: Int) raises -> Ball:
    """zeta(s) for s other than 1: Euler-Maclaurin above -1, the functional
    equation below, with `sin(pi s/2)` reduced exactly for an exact s."""
    if s._exact_lower() > Float(-1):
        return _hurwitz_positive(s, _one(w + 16), w)
    var work = w + 16
    var reflected = _sub(_one(work), s, work)
    var sine: Ball
    if s.is_exact():
        sine = _sin_pi(_scaled(s, -1, work)._midpoint, work)
    else:
        sine = _fn(_SIN, _mul(_pi(work), _scaled(s, -1, work), work), work)
    var two = Ball(Integer(2), precision=work)
    var factor = _mul(
        _fn(_EXP, _mul(s, _fn(_LOG, two, work), work), work),
        _fn(_EXP, _mul(_sub(s, _one(work), work), _fn(_LOG, _pi(work), work), work), work),
        work,
    )
    var gamma = _ball_special(_GAMMA, _BallArgument(reflected), 0, _c(work))
    return _mul(_mul(_mul(factor, sine, work), gamma, work), _hurwitz_positive(reflected, _one(work), work), work)


def _zeta_ball(x: _BallArgument, q: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """zeta(s, q) of two balls: a ball containing it at every pair of points,
    indeterminate at the poles and outside the domain."""
    var w = _working(context, x.precision, q.precision)
    if x.kind != _FINITE or q.kind != _FINITE:
        return Ball.indeterminate(w)
    var exact_s = x.kind == _FINITE and x.radius.is_zero()
    var exact_q = q.kind == _FINITE and q.radius.is_zero()
    if exact_s and exact_q:
        var decided = _zeta_case(_rational_value(x), _rational_value(q))
        if decided.kind == _RATIONAL:
            return Ball(decided.value, precision=w)
        if decided.kind != _NEITHER:
            return Ball.indeterminate(w)
    var s = _rounded_ball(x, w + 24)
    var a = _rounded_ball(q, w + 24)
    var one = Float(1)
    if exact_q and _rational_value(q) == Rational(1):
        if not (s._exact_lower() > one or s._exact_upper() < one):
            return Ball.indeterminate(w)
        return _rounded_ball(_BallArgument(_riemann(s, w)), w)
    if not (s._exact_lower() > one):
        return Ball.indeterminate(w)
    if not (a._exact_lower() > Float(0)):
        # A negative q needs an integer s and no pole in the ball of q.
        if not (s.is_exact() and s._midpoint.is_integer()) or a._exact_upper() >= Float(0):
            return Ball.indeterminate(w)
        if a._exact_lower().floor() != a._exact_upper().floor() or a._exact_lower().is_integer():
            return Ball.indeterminate(w)
    return _rounded_ball(_BallArgument(_hurwitz(s, a, w)), w)


def _rational_value(x: _BallArgument) raises -> Rational:
    if x.midpoint.value.kind == 0:
        return Rational(0)
    return _exact_rational(x.midpoint.value)


def _polygamma_ball(n: Integer, x: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """polygamma(n, x) of a ball: digamma for n = 0, else
    `(-1)**(n+1) n! zeta(n+1, x)`; indeterminate for n < 0 and at the poles."""
    var w = _working(context, x.precision)
    if n.sign() < 0 or x.kind != _FINITE:
        return Ball.indeterminate(w)
    if n.sign() == 0:
        return _ball_special(_DIGAMMA, x, 0, context)
    var zeta = _zeta_ball(_BallArgument(Ball(n + 1, precision=w + 16)), x, _c(w + 16))
    if not zeta.is_finite():
        return Ball.indeterminate(w)
    var scaled = _mul(zeta, Ball(factorial(n), precision=w + 16), w + 16)
    if not (n & 1).__bool__():
        scaled = _negate(scaled)
    return _rounded_ball(_BallArgument(scaled), w)
