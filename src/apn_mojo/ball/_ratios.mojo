"""Gamma ratios as balls: Euler's beta function, its logarithm and the
Pochhammer symbol (scipy's beta, betaln and poch).

`B(a, b) = Gamma(a) Gamma(b) / Gamma(a + b)` and `(z)_m = Gamma(z + m) /
Gamma(z)` come from the gamma kernels in ball arithmetic, and
`log |B(a, b)|` from log |Gamma|, with as many more bits as its terms'
magnitude cancels. Rational values are computed exactly, so that an exactly
representable one rounds correctly in every mode: `B(a, b)` when both
arguments are integers or one is a positive integer k,
`(k-1)! / (b (b+1) ... (b+k-1))`; `(z)_m` for an integer m,
`z (z+1) ... (z+m-1)`, or `1 / ((z-1) (z-2) ... (z+m))` for m < 0.

At the poles they take scipy's values. `B(a, b)` for a non-positive integer
a is the limit `(-1)**b B(1-a-b, b)` when b is an integer with
`1 - a - b > 0`, and infinite otherwise; it is 0 where `a + b` alone is a
non-positive integer. `(z)_m` is infinite where a factor of its divisor is 0
or, for a non-integer m, where `z + m` alone is a non-positive integer, and
0 where a factor of its product is 0 or z alone is a non-positive integer.
"""

from ..integer.value import Integer
from ..integer.math import factorial
from ..rational.value import Rational
from ..float.math import _exact_rational
from .context import BallContext
from .value import Ball, _BallArgument, _FINITE
from ._arithmetic import _working, _rounded_ball
from ._kernels import _LOG
from ._special import _ball_special, _GAMMA, _LOG_GAMMA, _BETA, _LOG_BETA, _POCH, _add, _sub, _mul, _div, _fn, _c
from ._fixed_series import _bit_count

comptime _EXACT_BITS = 1 << 20
"""The largest exact product a rational value is computed as, in bits."""

comptime _NEITHER = 0
comptime _RATIONAL = 1
comptime _INFINITE = 2


@fieldwise_init
struct _Decided(ImplicitlyCopyable):
    """A gamma ratio decided without the kernels: rational, infinite, or
    neither."""

    var kind: Int
    var value: Rational


def _neither() -> _Decided:
    return _Decided(_NEITHER, Rational(0))


def _infinite() -> _Decided:
    return _Decided(_INFINITE, Rational(0))


def _nonpositive_integer(q: Rational) raises -> Bool:
    return q.is_integer() and q.sign() <= 0


def _fits(count: Integer, q: Rational) raises -> Bool:
    """Whether `count` factors of about q's size stay within the exact
    product's bound."""
    if count > Integer(1 << 20):
        return False
    var bits = q.numerator().magnitude_bit_length() + q.denominator().magnitude_bit_length() + 21
    return Int(count._low_magnitude()) * bits <= _EXACT_BITS


def _beta_by_integer(k: Rational, other: Rational) raises -> _Decided:
    """`B(k, other) = (k-1)! / (other (other+1) ... (other+k-1))` for a
    positive integer k."""
    var n = k.numerator()
    if not _fits(n, other):
        return _neither()
    var divisor = Rational(1)
    for i in range(Int(n._low_magnitude())):
        divisor = divisor * (other + Rational(i))
    return _Decided(_RATIONAL, Rational(factorial(n - 1)) / divisor)


def _beta_decided(a: Rational, b: Rational) raises -> _Decided:
    """B(a, b) where it is rational this way or infinite."""
    if a.is_integer() and b.is_integer():
        if a.sign() <= 0 and b.sign() <= 0:
            return _infinite()
        if a.sign() <= 0 or b.sign() <= 0:
            var x = a if a.sign() <= 0 else b
            var y = b if a.sign() <= 0 else a
            if x + y >= Rational(1):
                return _infinite()
            var inner = _beta_decided(Rational(1) - x - y, y)
            if inner.kind != _RATIONAL:
                return inner
            var odd = (y.numerator() & 1).__bool__()
            return _Decided(_RATIONAL, -inner.value if odd else inner.value)
        return _beta_by_integer(a, b) if a <= b else _beta_by_integer(b, a)
    if a.is_integer() and a.sign() > 0:
        return _beta_by_integer(a, b)
    if b.is_integer() and b.sign() > 0:
        return _beta_by_integer(b, a)
    if _nonpositive_integer(a) or _nonpositive_integer(b):
        return _infinite()
    if _nonpositive_integer(a + b):
        return _Decided(_RATIONAL, Rational(0))
    return _neither()


def _poch_decided(z: Rational, m: Rational) raises -> _Decided:
    """(z)_m where it is rational this way, zero or infinite."""
    if not m.is_integer():
        if _nonpositive_integer(z + m) and not _nonpositive_integer(z):
            return _infinite()
        if _nonpositive_integer(z):
            return _Decided(_RATIONAL, Rational(0))
        return _neither()
    var n = m.numerator()
    if n.sign() == 0:
        return _Decided(_RATIONAL, Rational(1))
    if z.is_integer():
        # A factor z + i of the product is 0 for z <= 0 < z + m; one of the
        # divisor's, z - i, for z + m <= 0 < z.
        if n.sign() > 0 and z.sign() <= 0 and (z + m).sign() > 0:
            return _Decided(_RATIONAL, Rational(0))
        if n.sign() < 0 and z.sign() > 0 and (z + m).sign() <= 0:
            return _infinite()
    var count = abs(n)
    if not _fits(count, z):
        return _neither()
    var product = Rational(1)
    var steps = Int(count._low_magnitude())
    if n.sign() > 0:
        for i in range(steps):
            product = product * (z + Rational(i))
        return _Decided(_RATIONAL, product^)
    for i in range(1, steps + 1):
        product = product * (z - Rational(i))
    return _Decided(_RATIONAL, Rational(1) / product)


def _rational_of(x: _BallArgument) raises -> Optional[Rational]:
    """The value of an exact finite operand."""
    if x.kind != _FINITE or not x.radius.is_zero():
        return None
    if x.midpoint.value.kind == 0:
        return Rational(0)
    return _exact_rational(x.midpoint.value)


def _decided(code: Int, a: Rational, b: Rational) raises -> _Decided:
    return _poch_decided(a, b) if code == _POCH else _beta_decided(a, b)


def _ratio_ball(code: Int, x: _BallArgument, y: _BallArgument, context: Optional[BallContext]) raises -> Ball:
    """beta, betaln or poch of two balls: a ball containing its value at
    every pair of points, indeterminate where one is infinite."""
    var w = _working(context, x.precision, y.precision)
    if x.kind != _FINITE or y.kind != _FINITE:
        return Ball.indeterminate(w)
    var a = _rational_of(x)
    var b = _rational_of(y)
    if a and b:
        var decided = _decided(code, a.value(), b.value())
        if decided.kind == _INFINITE:
            return Ball.indeterminate(w)
        if decided.kind == _RATIONAL:
            if code != _LOG_BETA:
                return Ball(decided.value, precision=w)
            var size = abs(decided.value)
            if size.sign() == 0:
                return Ball.indeterminate(w)
            if size == Rational(1):
                return Ball(Integer(0), precision=w)
            return _fn(_LOG, Ball(size, precision=w + 16), w)
    var work = w + 16
    var xb = _rounded_ball(x, work + 8)
    var yb = _rounded_ball(y, work + 8)
    if code == _LOG_BETA:
        # log |Gamma| of the larger argument has about exponent + log2(exponent) bits before the point.
        var e = max(xb._midpoint._exponent, yb._midpoint._exponent, 0)
        work += e + _bit_count(e + 1)
        xb = _rounded_ball(x, work + 8)
        yb = _rounded_ball(y, work + 8)
    var sum = _add(xb, yb, work + 8)
    if code == _POCH:
        var top = _ball_special(_GAMMA, _BallArgument(sum), 0, _c(work))
        var bottom = _ball_special(_GAMMA, _BallArgument(xb), 0, _c(work))
        return _rounded_ball(_BallArgument(_div(top, bottom, work)), w)
    if code == _BETA:
        var top = _mul(_ball_special(_GAMMA, _BallArgument(xb), 0, _c(work)), _ball_special(_GAMMA, _BallArgument(yb), 0, _c(work)), work)
        return _rounded_ball(_BallArgument(_div(top, _ball_special(_GAMMA, _BallArgument(sum), 0, _c(work)), work)), w)
    var terms = _add(_ball_special(_LOG_GAMMA, _BallArgument(xb), 0, _c(work)), _ball_special(_LOG_GAMMA, _BallArgument(yb), 0, _c(work)), work)
    return _rounded_ball(_BallArgument(_sub(terms, _ball_special(_LOG_GAMMA, _BallArgument(sum), 0, _c(work)), work)), w)
