"""Every elementwise function of apn_mojo.batch against its scalar function:
each in its Float form, and Ball, Complex and ComplexBall forms through each
shared dispatcher. Kept apart from test_batch_functions so each build stays
well under the compiler memory cap."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    Integer, Rational, Float, Complex, Ball, ComplexBall, Batch, FloatFormat, ArithmeticContext,
    BallContext, ComplexContext, RoundingMode,
)
from apn_mojo import (
    add, subtract, multiply, divide, reciprocal, maximum, minimum, clip, abs, sqrt, pow_int, floor, ceil,
    trunc, round, conjugate, real, imag, angle, exp, expm1, exp2, log, log1p, log2, log10, sin, cos, tan,
    sin_cos, atan, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh, pow, rootn, gamma, gammaln,
    digamma, erf, erfc, erfi, erfinv, expi, ndtr, log_ndtr, ndtri, sici, shichi, fresnel, lambertw, beta,
    betaln, poch, zeta, polygamma, gammainc, gammaincc, hyp1f1, betainc, hyp2f1, comb,
)
from apn_mojo import batch


def _c() raises -> ArithmeticContext:
    return ArithmeticContext(format=FloatFormat.binary64())


def _f(text: String) raises -> Float:
    return Float(text, context=_c())


def _json[T: ImplicitlyCopyable & Deinitable](x: T) raises -> String:
    comptime if T == Float:
        return rebind[Float](x).to_json()
    elif T == Complex:
        return rebind[Complex](x).to_json()
    elif T == Ball:
        return rebind[Ball](x).to_json()
    elif T == ComplexBall:
        return rebind[ComplexBall](x).to_json()
    elif T == Rational:
        return String(rebind[Rational](x))
    else:
        return String(rebind[Integer](x))


def _agree[T: ImplicitlyCopyable & Deinitable](got: Batch[T], expected: List[T]) raises:
    """A vector equal, element by element and format by format, to the scalar results."""
    var items = got.to_list()
    assert_equal(len(items), len(expected))
    for i in range(len(expected)):
        assert_equal(_json(items[i]), _json(expected[i]))


def _floats(values: List[Int], shape: List[Int]) raises -> Batch[Float]:
    var items = List[Float]()
    for v in values:
        items.append(Float(v, context=_c()))
    return Batch[Float](items, shape=shape)


# --------------------------------------------------------- Float forms


def test_real_functions() raises:
    var x0 = _f("0.25")
    var x1 = _f("0.5")
    var xs = Batch[Float]([x0, x1])
    _agree(batch.sqrt(xs), [sqrt(x0), sqrt(x1)])
    _agree(batch.exp(xs), [exp(x0), exp(x1)])
    _agree(batch.expm1(xs), [expm1(x0), expm1(x1)])
    _agree(batch.exp2(xs), [exp2(x0), exp2(x1)])
    _agree(batch.log(xs), [log(x0), log(x1)])
    _agree(batch.log1p(xs), [log1p(x0), log1p(x1)])
    _agree(batch.log2(xs), [log2(x0), log2(x1)])
    _agree(batch.log10(xs), [log10(x0), log10(x1)])
    _agree(batch.sin(xs), [sin(x0), sin(x1)])
    _agree(batch.cos(xs), [cos(x0), cos(x1)])
    _agree(batch.tan(xs), [tan(x0), tan(x1)])
    _agree(batch.atan(xs), [atan(x0), atan(x1)])
    _agree(batch.asin(xs), [asin(x0), asin(x1)])
    _agree(batch.acos(xs), [acos(x0), acos(x1)])
    _agree(batch.sinh(xs), [sinh(x0), sinh(x1)])
    _agree(batch.cosh(xs), [cosh(x0), cosh(x1)])
    _agree(batch.tanh(xs), [tanh(x0), tanh(x1)])
    _agree(batch.asinh(xs), [asinh(x0), asinh(x1)])
    _agree(batch.atanh(xs), [atanh(x0), atanh(x1)])
    _agree(batch.gamma(xs), [gamma(x0), gamma(x1)])
    _agree(batch.gammaln(xs), [gammaln(x0), gammaln(x1)])
    _agree(batch.digamma(xs), [digamma(x0), digamma(x1)])
    _agree(batch.erf(xs), [erf(x0), erf(x1)])
    _agree(batch.erfc(xs), [erfc(x0), erfc(x1)])
    _agree(batch.erfi(xs), [erfi(x0), erfi(x1)])
    _agree(batch.erfinv(xs), [erfinv(x0), erfinv(x1)])
    _agree(batch.expi(xs), [expi(x0), expi(x1)])
    _agree(batch.ndtr(xs), [ndtr(x0), ndtr(x1)])
    _agree(batch.log_ndtr(xs), [log_ndtr(x0), log_ndtr(x1)])
    _agree(batch.ndtri(xs), [ndtri(x0), ndtri(x1)])
    _agree(batch.zeta(xs), [zeta(x0), zeta(x1)])
    _agree(batch.lambertw(xs), [lambertw(x0), lambertw(x1)])
    _agree(batch.rootn(xs, 3), [rootn(x0, 3), rootn(x1, 3)])
    var big = Batch[Float]([_f("1.25"), _f("2")])
    _agree(batch.acosh(big), [acosh(_f("1.25")), acosh(_f("2"))])


def test_pairs() raises:
    var x0 = _f("0.25")
    var x1 = _f("0.5")
    var xs = Batch[Float]([x0, x1]).reshape([2, 1])
    var s, k = batch.sin_cos(xs)
    assert_equal(s.shape(), [2, 1])
    assert_equal(_json(s[1, 0]), _json(sin_cos(x1)[0]))
    assert_equal(_json(k[1, 0]), _json(sin_cos(x1)[1]))
    var si, ci = batch.sici(xs)
    assert_equal(_json(si[0, 0]), _json(sici(x0)[0]))
    assert_equal(_json(ci[0, 0]), _json(sici(x0)[1]))
    var shi, chi = batch.shichi(xs)
    assert_equal(_json(chi[1, 0]), _json(shichi(x1)[1]))
    assert_equal(_json(shi[1, 0]), _json(shichi(x1)[0]))
    var fs, fc = batch.fresnel(xs)
    assert_equal(_json(fs[0, 0]), _json(fresnel(x0)[0]))
    assert_equal(_json(fc[0, 0]), _json(fresnel(x0)[1]))


def test_two_argument_functions() raises:
    var a0 = _f("1.5")
    var a1 = _f("2.5")
    var x0 = _f("0.25")
    var x1 = _f("0.75")
    var a = Batch[Float]([a0, a1])
    var x = Batch[Float]([x0, x1])
    _agree(batch.atan2(a, x), [atan2(a0, x0), atan2(a1, x1)])
    _agree(batch.beta(a, x), [beta(a0, x0), beta(a1, x1)])
    _agree(batch.betaln(a, x), [betaln(a0, x0), betaln(a1, x1)])
    _agree(batch.poch(a, x), [poch(a0, x0), poch(a1, x1)])
    _agree(batch.gammainc(a, x), [gammainc(a0, x0), gammainc(a1, x1)])
    _agree(batch.gammaincc(a, x), [gammaincc(a0, x0), gammaincc(a1, x1)])
    _agree(batch.zeta(a, x), [zeta(a0, x0), zeta(a1, x1)])
    _agree(batch.pow(a, x), [pow(a0, x0), pow(a1, x1)])
    _agree(batch.polygamma(Batch[Integer]([1, 2]), x), [polygamma(Integer(1), x0), polygamma(Integer(2), x1)])
    _agree(batch.pow_int(x, 3), [pow_int(x0, Integer(3)), pow_int(x1, Integer(3))])


def test_hypergeometric() raises:
    var a = _f("0.5")
    var b = _f("1.5")
    var c = _f("2.5")
    var x0 = _f("0.25")
    var x1 = _f("0.75")
    var xs = Batch[Float]([x0, x1])
    _agree(batch.hyp1f1(a, b, xs), [hyp1f1(a, b, x0), hyp1f1(a, b, x1)])
    _agree(batch.betainc(a, b, xs), [betainc(a, b, x0), betainc(a, b, x1)])
    _agree(batch.hyp2f1(a, b, c, xs), [hyp2f1(a, b, c, x0), hyp2f1(a, b, c, x1)])


# ----------------------------------------------------- the other families


def test_ball_functions() raises:
    var x0 = Ball(_f("0.25"))
    var x1 = Ball(_f("0.5"))
    var xs = Batch[Ball]([x0, x1])
    var c = BallContext(80)
    _agree(batch.exp(xs, context=c), [exp(x0, context=c), exp(x1, context=c)])
    _agree(batch.gamma(xs), [gamma(x0), gamma(x1)])
    _agree(batch.zeta(xs), [zeta(x0), zeta(x1)])
    _agree(batch.lambertw(xs), [lambertw(x0), lambertw(x1)])
    _agree(batch.rootn(xs, 3), [rootn(x0, 3), rootn(x1, 3)])
    _agree(batch.pow_int(xs, 3), [pow_int(x0, Integer(3)), pow_int(x1, Integer(3))])
    _agree(batch.atan2(xs, xs), [atan2(x0, x0), atan2(x1, x1)])
    # A ball and a Float give balls.
    _agree(batch.atan2(xs, _f("2")), [atan2(x0, Ball(_f("2"))), atan2(x1, Ball(_f("2")))])
    var s, k = batch.sin_cos(xs)
    assert_equal(_json(s[1]) + _json(k[1]), _json(sin_cos(x1)[0]) + _json(sin_cos(x1)[1]))
    var a = Ball(_f("0.5"))
    var b = Ball(_f("1.5"))
    _agree(batch.hyp1f1(a, b, xs), [hyp1f1(a, b, x0), hyp1f1(a, b, x1)])


def test_complex_functions() raises:
    var z0 = Complex(_f("0.25"), _f("0.5"))
    var z1 = Complex(_f("-0.5"), _f("0.25"))
    var zs = Batch[Complex]([z0, z1])
    _agree(batch.sqrt(zs), [sqrt(z0), sqrt(z1)])
    _agree(batch.log(zs), [log(z0), log(z1)])
    var c = ComplexContext(real=_c(), imag=ArithmeticContext(format=FloatFormat.binary64(), rounding=RoundingMode.toward_zero))
    _agree(batch.exp(zs, context=c), [exp(z0, context=c), exp(z1, context=c)])
    _agree(batch.pow(zs, z1), [pow(z0, z1), pow(z1, z1)])
    _agree(batch.pow_int(zs, 3), [pow_int(z0, Integer(3)), pow_int(z1, Integer(3))])
    var w0 = ComplexBall(z0)
    var w1 = ComplexBall(z1)
    var ws = Batch[ComplexBall]([w0, w1])
    _agree(batch.exp(ws), [exp(w0), exp(w1)])
    _agree(batch.atanh(ws), [atanh(w0), atanh(w1)])
    _agree(batch.pow(ws, ws), [pow(w0, w0), pow(w1, w1)])
    _agree(batch.pow_int(ws, 3), [pow_int(w0, Integer(3)), pow_int(w1, Integer(3))])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
