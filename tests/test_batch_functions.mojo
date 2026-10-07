"""NumPy-style batch functions: shapes, broadcasting, contexts, exact
arithmetic in every family, selection, constructors, joining and running
folds. test_batch_math checks each elementwise function against its scalar."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import (
    Integer, Rational, Float, Complex, Ball, ComplexBall, Batch, FloatFormat, ArithmeticContext,
    BallContext, ComplexContext,
)
from apn_mojo import (
    add, subtract, multiply, divide, reciprocal, maximum, minimum, clip, abs, sqrt, pow_int, floor, ceil,
    trunc, round, conjugate, real, imag, angle, exp, expm1, exp2, log, log1p, log2, log10, sin, cos, tan,
    sin_cos, atan, asin, acos, atan2, sinh, cosh, tanh, asinh, acosh, atanh, pow, rootn, gamma, gammaln,
    digamma, erf, erfc, erfi, erfinv, expi, ndtr, log_ndtr, ndtri, sici, shichi, fresnel, lambertw, beta,
    betaln, poch, zeta, polygamma, gammainc, gammaincc, hyp1f1, betainc, hyp2f1, comb,
)
from apn_mojo import batch, ball, complex_ball


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


# -------------------------------------------------------------- shapes


def test_shapes() raises:
    var xs = _floats([1, -2, 3, 0, 5, -7], [2, 3])
    var ys = batch.exp(xs)
    assert_equal(ys.shape(), [2, 3])
    var a = xs.to_list()
    var b = ys.to_list()
    for i in range(len(a)):
        assert_equal(_json(b[i]), _json(exp(a[i])))
    # A strided view, a transpose, a rank-zero batch and an empty one.
    _agree(batch.exp(xs.reshape([6])[1::2]), [exp(a[1]), exp(a[3]), exp(a[5])])
    var t = batch.exp(xs.transpose([1, 0]))
    assert_equal(t.shape(), [3, 2])
    assert_equal(_json(t[0, 1]), _json(exp(a[3])))
    var one = batch.exp(Batch[Float](a[2]))
    assert_equal(one.ndim(), 0)
    assert_equal(_json(one.item()), _json(exp(a[2])))
    assert_equal(batch.exp(Batch[Float]()).shape(), [0])


def test_exact_elements() raises:
    var c = _c()
    var ints = Batch[Integer]([0, 1, 2])
    _agree(batch.exp(ints, context=c), [exp(Integer(0), context=c), exp(Integer(1), context=c), exp(Integer(2), context=c)])
    var halves = Batch[Rational]([Rational(1, 2)])
    _agree(batch.sqrt(halves, context=c), [sqrt(Rational(1, 2), context=c)])
    with assert_raises():
        _ = batch.exp(ints)


# ------------------------------------------------- several arguments


def test_broadcasting() raises:
    var c = _c()
    var ys = _floats([1, -2, 3, 4, 5, -6], [2, 3])
    var xs = _floats([2, -1, 7], [3])
    var r = batch.atan2(ys, xs)
    assert_equal(r.shape(), [2, 3])
    assert_equal(_json(r[1, 2]), _json(atan2(Float(-6, context=c), Float(7, context=c))))
    var column = _floats([1, 2], [2, 1])
    var outer = batch.atan2(column, xs)
    assert_equal(outer.shape(), [2, 3])
    assert_equal(_json(outer[1, 0]), _json(atan2(Float(2, context=c), Float(2, context=c))))
    var s = batch.atan2(ys, Float(1, context=c))
    assert_equal(_json(s[0, 1]), _json(atan2(Float(-2, context=c), Float(1, context=c))))
    var lit = batch.atan2(1, xs, context=c)
    assert_equal(_json(lit[1]), _json(atan2(Float(1, context=c), Float(-1, context=c))))
    with assert_raises():
        _ = batch.atan2(ys, _floats([1, 2], [2]))


def test_exact_arithmetic() raises:
    var i = Batch[Integer]([7, -3])
    var j = Batch[Integer]([2, 5])
    _agree(batch.add(i, j), [Integer(9), Integer(2)])
    _agree(batch.subtract(i, j), [Integer(5), Integer(-8)])
    _agree(batch.multiply(i, j), [Integer(14), Integer(-15)])
    _agree(batch.divide(i, j), [Rational(7, 2), Rational(-3, 5)])
    _agree(batch.add(i, 1), [Integer(8), Integer(-2)])
    # A vector of length one broadcasts, as in NumPy, unlike the operators.
    _agree(batch.subtract(i, Batch[Integer]([1])), [Integer(6), Integer(-4)])
    _agree(batch.divide(1, j), [Rational(1, 2), Rational(1, 5)])
    var q = Batch[Rational]([Rational(1, 3), Rational(-1, 2)])
    _agree(batch.add(q, i), [Rational(22, 3), Rational(-7, 2)])
    _agree(batch.subtract(q, q), [Rational(0), Rational(0)])
    _agree(batch.multiply(q, j), [Rational(2, 3), Rational(-5, 2)])
    _agree(batch.divide(q, j), [Rational(1, 6), Rational(-1, 10)])
    with assert_raises():
        _ = batch.divide(i, 0)


def test_rounded_arithmetic() raises:
    var c = _c()
    var f0 = _f("0.1")
    var f1 = _f("3")
    var fs = Batch[Float]([f0, f1])
    var i = Batch[Integer]([1, 2])
    _agree(batch.add(fs, fs), [add(f0, f0), add(f1, f1)])
    _agree(batch.subtract(fs, i), [subtract(f0, Integer(1)), subtract(f1, Integer(2))])
    _agree(batch.multiply(fs, fs), [multiply(f0, f0), multiply(f1, f1)])
    _agree(batch.divide(i, fs), [divide(Integer(1), f0), divide(Integer(2), f1)])
    # A context rounds exact operands.
    _agree(batch.divide(i, 3, context=c), [divide(Integer(1), Integer(3), context=c), divide(Integer(2), Integer(3), context=c)])
    _agree(batch.add(i, i, context=c), [add(Integer(1), Integer(1), context=c), add(Integer(2), Integer(2), context=c)])
    _agree(batch.subtract(fs, fs, context=c), [subtract(f0, f0, context=c), subtract(f1, f1, context=c)])
    _agree(batch.multiply(fs, i, context=c), [multiply(f0, Integer(1), context=c), multiply(f1, Integer(2), context=c)])
    var z0 = Complex(f0, f1)
    var z1 = Complex(f1, f0)
    var zs = Batch[Complex]([z0, z1])
    _agree(batch.add(zs, zs), [add(z0, z0), add(z1, z1)])
    _agree(batch.subtract(zs, fs), [subtract(z0, f0), subtract(z1, f1)])
    _agree(batch.multiply(zs, zs), [multiply(z0, z0), multiply(z1, z1)])
    _agree(batch.divide(zs, zs), [divide(z0, z0), divide(z1, z1)])
    var cc = ComplexContext(c)
    _agree(batch.multiply(zs, zs, context=cc), [multiply(z0, z0, context=cc), multiply(z1, z1, context=cc)])
    _agree(batch.add(zs, i, context=c), [add(z0, Complex(Integer(1), Integer(0), context=c), context=c), add(z1, Complex(Integer(2), Integer(0), context=c), context=c)])


def test_ball_arithmetic() raises:
    var b0 = Ball(_f("0.1"))
    var b1 = Ball(_f("3"))
    var bs = Batch[Ball]([b0, b1])
    var i = Batch[Integer]([1, 2])
    _agree(batch.add(bs, bs), [add(b0, b0), add(b1, b1)])
    _agree(batch.subtract(bs, i), [subtract(b0, Integer(1)), subtract(b1, Integer(2))])
    _agree(batch.multiply(bs, bs), [multiply(b0, b0), multiply(b1, b1)])
    _agree(batch.divide(i, bs), [divide(Integer(1), b0), divide(Integer(2), b1)])
    var c = BallContext(40)
    _agree(batch.divide(bs, bs, context=c), [divide(b0, b0, context=c), divide(b1, b1, context=c)])
    _agree(batch.divide(i, 3, context=c), [ball.divide(Integer(1), Integer(3), context=c), ball.divide(Integer(2), Integer(3), context=c)])
    var w0 = ComplexBall(Complex(_f("0.25"), _f("0.5")))
    var w1 = ComplexBall(Complex(_f("-0.5"), _f("0.25")))
    var ws = Batch[ComplexBall]([w0, w1])
    _agree(batch.add(ws, ws), [add(w0, w0), add(w1, w1)])
    _agree(batch.subtract(ws, ws), [subtract(w0, w0), subtract(w1, w1)])
    _agree(batch.multiply(ws, ws), [multiply(w0, w0), multiply(w1, w1)])
    _agree(batch.divide(ws, ws, context=c), [divide(w0, w0, context=c), divide(w1, w1, context=c)])
    # The root functions' ComplexBall forms are the family declarations.
    assert_equal(_json(add(w0, w1)), _json(complex_ball.add(w0, w1)))
    assert_equal(_json(subtract(w0, w1)), _json(complex_ball.subtract(w0, w1)))
    assert_equal(_json(multiply(w0, w1)), _json(complex_ball.multiply(w0, w1)))
    assert_equal(_json(divide(w0, w1, context=c)), _json(complex_ball.divide(w0, w1, context=c)))
    assert_equal(_json(pow_int(w0, Integer(3))), _json(complex_ball.pow_int(w0, Integer(3))))


def test_reciprocal_and_abs() raises:
    var c = _c()
    _agree(batch.reciprocal(Batch[Integer]([2, -3])), [Rational(1, 2), Rational(-1, 3)])
    _agree(batch.reciprocal(Batch[Rational]([Rational(2, 3)])), [Rational(3, 2)])
    _agree(batch.reciprocal(Batch[Integer]([3]), context=c), [reciprocal(Integer(3), context=c)])
    var f = _f("3")
    _agree(batch.reciprocal(Batch[Float]([f])), [reciprocal(f)])
    var z = Complex(_f("0.25"), _f("0.5"))
    _agree(batch.reciprocal(Batch[Complex]([z])), [reciprocal(z)])
    var b = Ball(f)
    _agree(batch.reciprocal(Batch[Ball]([b])), [reciprocal(b)])
    _agree(batch.reciprocal(Batch[Ball]([b]), context=BallContext(30)), [reciprocal(b, context=BallContext(30))])
    var w = ComplexBall(z)
    _agree(batch.reciprocal(Batch[ComplexBall]([w])), [reciprocal(w)])
    _agree(batch.abs(Batch[Integer]([-2, 3])), [Integer(2), Integer(3)])
    _agree(batch.abs(Batch[Rational]([Rational(-1, 3)])), [Rational(1, 3)])
    _agree(batch.abs(Batch[Float]([_f("-0.5")])), [abs(_f("-0.5"))])
    _agree(batch.abs(Batch[Complex]([z])), [abs(z)])
    _agree(batch.abs(Batch[Complex]([z]), context=c), [abs(z, context=c)])
    _agree(batch.abs(Batch[Ball]([Ball(_f("-0.5"))])), [abs(Ball(_f("-0.5")))])
    _agree(batch.abs(Batch[ComplexBall]([w])), [abs(w)])


def test_ordering() raises:
    var c = _c()
    var i = Batch[Integer]([7, -3])
    _agree(batch.maximum(i, 0), [Integer(7), Integer(0)])
    _agree(batch.minimum(i, 0), [Integer(0), Integer(-3)])
    _agree(batch.clip(i, -1, 5), [Integer(5), Integer(-1)])
    var q = Batch[Rational]([Rational(1, 3), Rational(-1, 2)])
    _agree(batch.maximum(q, i), [Rational(7), Rational(-1, 2)])
    _agree(batch.minimum(q, i), [Rational(1, 3), Rational(-3)])
    _agree(batch.clip(q, 0, Rational(1, 4)), [Rational(1, 4), Rational(0)])
    var f0 = _f("0.1")
    var f1 = _f("-3")
    var fs = Batch[Float]([f0, f1])
    var zero = _f("0")
    _agree(batch.maximum(fs, zero), [maximum(f0, zero), maximum(f1, zero)])
    _agree(batch.minimum(fs, zero), [minimum(f0, zero), minimum(f1, zero)])
    _agree(batch.clip(fs, zero, _f("1")), [clip(f0, zero, _f("1")), clip(f1, zero, _f("1"))])
    _agree(batch.maximum(i, q, context=c), [maximum(Integer(7), Rational(1, 3), context=c), maximum(Integer(-3), Rational(-1, 2), context=c)])
    _agree(batch.minimum(fs, i, context=c), [minimum(f0, Integer(7), context=c), minimum(f1, Integer(-3), context=c)])
    _agree(batch.clip(i, 0, 5, context=c), [clip(Integer(7), Integer(0), Integer(5), context=c), clip(Integer(-3), Integer(0), Integer(5), context=c)])
    var bs = Batch[Ball]([Ball(f0), Ball(f1)])
    _agree(batch.maximum(bs, Ball(zero)), [maximum(Ball(f0), Ball(zero)), maximum(Ball(f1), Ball(zero))])
    _agree(batch.minimum(bs, Ball(zero)), [minimum(Ball(f0), Ball(zero)), minimum(Ball(f1), Ball(zero))])
    _agree(batch.clip(bs, Ball(zero), Ball(_f("1"))), [clip(Ball(f0), Ball(zero), Ball(_f("1"))), clip(Ball(f1), Ball(zero), Ball(_f("1")))])
    var bc = BallContext(30)
    _agree(batch.maximum(bs, Ball(zero), context=bc), [maximum(Ball(f0), Ball(zero), context=bc), maximum(Ball(f1), Ball(zero), context=bc)])
    _agree(batch.clip(bs, Ball(zero), Ball(_f("1")), context=bc), [clip(Ball(f0), Ball(zero), Ball(_f("1")), context=bc), clip(Ball(f1), Ball(zero), Ball(_f("1")), context=bc)])


def test_integers_and_parts() raises:
    var q = Batch[Rational]([Rational(5, 2), Rational(-5, 2)])
    _agree(batch.floor(q), [Integer(2), Integer(-3)])
    _agree(batch.ceil(q), [Integer(3), Integer(-2)])
    _agree(batch.trunc(q), [Integer(2), Integer(-2)])
    _agree(batch.round(q), [Integer(2), Integer(-2)])
    var fs = Batch[Float]([_f("2.5"), _f("-3.5")])
    _agree(batch.floor(fs), [Integer(2), Integer(-4)])
    _agree(batch.ceil(fs), [Integer(3), Integer(-3)])
    _agree(batch.trunc(fs), [Integer(2), Integer(-3)])
    _agree(batch.round(fs), [Integer(2), Integer(-4)])
    var i = Batch[Integer]([4])
    _agree(batch.floor(i), [Integer(4)])
    _agree(batch.ceil(i), [Integer(4)])
    _agree(batch.trunc(i), [Integer(4)])
    _agree(batch.round(i), [Integer(4)])
    var z = Complex(_f("0.25"), _f("-0.5"))
    var zs = Batch[Complex]([z])
    _agree(batch.conjugate(zs), [conjugate(z)])
    _agree(batch.real(zs), [real(z)])
    _agree(batch.imag(zs), [imag(z)])
    _agree(batch.angle(zs), [angle(z)])
    var w = ComplexBall(z)
    var ws = Batch[ComplexBall]([w])
    _agree(batch.conjugate(ws), [conjugate(w)])
    _agree(batch.real(ws), [real(w)])
    _agree(batch.imag(ws), [imag(w)])
    _agree(batch.angle(ws), [angle(w)])
    _agree(batch.comb(Batch[Integer]([5, 6]), 2), [Integer(10), Integer(15)])
    _agree(batch.comb(5, Batch[Integer]([2, 3]), repetition=True), [comb(Integer(5), Integer(2), repetition=True), comb(Integer(5), Integer(3), repetition=True)])


# -------------------------------------------------------------- selection


def test_where() raises:
    var c = _c()
    var xs = _floats([1, -2, 3, -4], [2, 2])
    var zero = Float(0, context=c)
    var r = batch.where(xs > 0, xs, zero)
    assert_equal(r.shape(), [2, 2])
    assert_equal(_json(r[0, 0]), _json(xs[0, 0]))
    assert_equal(_json(r[0, 1]), _json(zero))
    var row = _floats([10, 20], [2])
    var q = batch.where(xs > 0, zero, row)
    assert_equal(_json(q[1, 1]), _json(row[1]))
    assert_equal(_json(q[1, 0]), _json(zero))
    _agree(batch.where(Batch[Integer]([1, -1]) > 0, Batch[Integer]([5, 6]), 0), [Integer(5), Integer(0)])
    with assert_raises():
        _ = batch.where(_floats([1, 2, 3], [3]) > 0, xs, zero)


# ----------------------------------------------------------- constructors


def test_constructors() raises:
    var c = _c()
    var z = batch.zeros[Integer]([2, 3])
    assert_equal(z.shape(), [2, 3])
    assert_equal(z.to_list(), List[Integer](length=6, fill=Integer(0)))
    assert_equal(batch.ones[Rational](3).to_list(), List[Rational](length=3, fill=Rational(1)))
    var f = batch.zeros[Float](2, context=c)
    assert_equal(f[1].format().precision(), 53)
    assert_equal(_json(batch.ones[Float]([1], context=c)[0]), _json(Float(1, context=c)))
    assert_equal(_json(batch.ones[Complex](1)[0]), _json(Complex(Integer(1), Integer(0))))
    assert_equal(_json(batch.zeros[Ball](1)[0]), _json(Ball(Integer(0))))
    assert_equal(_json(batch.ones[ComplexBall](1)[0]), _json(ComplexBall(Ball(Integer(1)))))
    assert_equal(batch.zeros[Integer]([]).ndim(), 0)
    with assert_raises():
        _ = batch.zeros[Integer]([-1])
    _agree(batch.full([2], Rational(1, 3)), [Rational(1, 3), Rational(1, 3)])
    _agree(batch.full(2, 7), [Integer(7), Integer(7)])
    _agree(batch.arange[Integer](4), [Integer(0), Integer(1), Integer(2), Integer(3)])
    _agree(batch.arange[Integer](5, 0, -2), [Integer(5), Integer(3), Integer(1)])
    _agree(batch.arange[Rational](0, 1, Rational(1, 3)), [Rational(0), Rational(1, 3), Rational(2, 3)])
    assert_equal(batch.arange[Integer](3, 3).shape(), [0])
    # Each Float element is start + k * step rounded once: the third is 0.3, not 0.1 + 0.1 + 0.1.
    var tenths = batch.arange[Float](0, Rational(1, 2), Rational(1, 10), context=c)
    assert_equal(_json(tenths[3]), _json(Float(Rational(3, 10), context=c)))
    with assert_raises():
        _ = batch.arange[Integer](0, 1, Rational(1, 2))
    with assert_raises():
        _ = batch.arange[Integer](0, 1, 0)
    _agree(batch.linspace[Rational](0, 1, 4), [Rational(0), Rational(1, 3), Rational(2, 3), Rational(1)])
    _agree(batch.linspace[Rational](0, 1, 4, endpoint=False), [Rational(0), Rational(1, 4), Rational(1, 2), Rational(3, 4)])
    var thirds = batch.linspace[Float](0, 1, 4, context=c)
    assert_equal(_json(thirds[1]), _json(Float(Rational(1, 3), context=c)))
    assert_equal(_json(thirds[3]), _json(Float(1, context=c)))
    assert_equal(batch.linspace[Integer](0, 4, 1).to_list(), [Integer(0)])


def test_joining() raises:
    var a = Batch[Integer]([1, 2, 3, 4]).reshape([2, 2])
    var b = Batch[Integer]([5, 6]).reshape([1, 2])
    var rows = batch.concatenate([a, b])
    assert_equal(rows.shape(), [3, 2])
    assert_equal(rows.to_list(), [Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6)])
    var columns = batch.concatenate([a, b.reshape([2, 1])], axis=1)
    assert_equal(columns.shape(), [2, 3])
    assert_equal(columns.to_list(), [Integer(1), Integer(2), Integer(5), Integer(3), Integer(4), Integer(6)])
    var s = batch.stack([Batch[Integer]([1, 2]), Batch[Integer]([3, 4])])
    assert_equal(s.shape(), [2, 2])
    assert_equal(s.to_list(), [Integer(1), Integer(2), Integer(3), Integer(4)])
    var t = batch.stack([Batch[Integer]([1, 2]), Batch[Integer]([3, 4])], axis=-1)
    assert_equal(t.shape(), [2, 2])
    assert_equal(t.to_list(), [Integer(1), Integer(3), Integer(2), Integer(4)])
    with assert_raises():
        _ = batch.concatenate([a, Batch[Integer]([1, 2, 3]).reshape([1, 3])])
    with assert_raises():
        _ = batch.stack([Batch[Integer]([1]), Batch[Integer]([1, 2])])


# ---------------------------------------------------------- running folds


def test_running() raises:
    var c = _c()
    _agree(batch.cumsum(Batch[Integer]([1, 2, 3, 4])), [Integer(1), Integer(3), Integer(6), Integer(10)])
    var m = Batch[Integer]([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    assert_equal(batch.cumsum(m, axis=1).to_list(), [Integer(1), Integer(3), Integer(6), Integer(4), Integer(9), Integer(15)])
    assert_equal(batch.cumsum(m, axis=0).to_list(), [Integer(1), Integer(2), Integer(3), Integer(5), Integer(7), Integer(9)])
    assert_equal(batch.cumsum(m).shape(), [6])
    assert_equal(batch.cumprod(m, axis=1).to_list(), [Integer(1), Integer(2), Integer(6), Integer(4), Integer(20), Integer(120)])
    _agree(batch.cumprod(Batch[Rational]([Rational(1, 2), Rational(2, 3)])), [Rational(1, 2), Rational(1, 3)])
    # Each Float running sum is exact, rounded once: 1e16 + 1 - 1e16 is 1.
    var fs = Batch[Float]([Float(10**16, context=c), Float(1, context=c), Float(-(10**16), context=c)])
    assert_equal(_json(batch.cumsum(fs, context=c)[2]), _json(Float(1, context=c)))
    var z = Complex(_f("0.5"), _f("2"))
    _agree(batch.cumprod(Batch[Complex]([z, z])), [z, multiply(z, z)])
    var b = Ball(_f("0.5"))
    _agree(batch.cumsum(Batch[Ball]([b, b])), [b, add(b, b)])
    var w = ComplexBall(z)
    assert_equal(len(batch.cumsum(Batch[ComplexBall]([w, w]))), 2)
    with assert_raises():
        _ = batch.cumsum(Batch[Integer]([1]), context=c)


def test_reductions_exported() raises:
    var i = Batch[Integer]([3, 1, 2])
    assert_equal(batch.sum(i), 6)
    assert_equal(batch.prod(i), 6)
    assert_equal(batch.max(i), 3)
    assert_equal(batch.min(i), 1)
    assert_equal(batch.argmax(i), 0)
    assert_equal(batch.argmin(i), 1)
    assert_equal(batch.dot(i, i), 14)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
