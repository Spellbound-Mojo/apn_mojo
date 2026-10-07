"""Kernels of the constants: pi, e, ln 2, log2(10), Euler's gamma and Catalan's G.

Each returns a ball at `w + 16` bits containing the constant. The series run on
the shared binary-splitting engine; each truncated series adds a bound of its
tail, and the few ball operations that combine the series add their own
rounding errors. Nothing is cached: a constant is computed for each call, and
at medium precision that costs a few products.

| Constant | Formula | Tail after `n` terms |
|---|---|---|
| pi | Chudnovsky: `426880 sqrt(10005) / S` | `2**30 (n + 1) 2**(-47 n)`, the next term of an alternating series |
| e | `sum 1/k!` | `2 / n!` |
| ln 2 | `18 atanh(1/26) - 2 atanh(1/4801) + 8 atanh(1/8749)` | `2 m**-(2n+1)` for `atanh(1/m)` |
| log2(10) | `3 + 2 atanh(1/9) / ln 2` | as for `atanh` |
| gamma | Brent-McMillan: `A/B - log n`, minus `K0(2n)/I0(2n)`, which lies in `(0, pi e**(-4n))` | `2 t_K H_K` and `(4/3) t_K` beyond `K > 2n` |
| Catalan | Ramanujan: `(pi/8) log(2 + sqrt 3) + (3/8) sum (k!)**2 / ((2k)! (2k + 1)**2)` | `2 * 4**-n`; `log(2 + sqrt 3) = (2/sqrt 3) sum 3**-k / (2k + 1)` with tail `2 * 3**-n` |
"""

from std.bit import count_leading_zeros
from ..integer.value import Integer
from ..float.context import ArithmeticContext, FloatFormat
from ..float._input import _FloatInput
from ..float._rounding import _round_ratio
from ..common._binary_splitting import _SeriesTerms, _Split, _binary_split
from .context import BallContext
from ._radius import _Radius, _up_integer
from .value import Ball, _BallArgument, _FINITE
from ..float.value import Float
from ..float.status import NumericStatus
from ..float._rounding import _RoundedBinary
from ._arithmetic import _from_record, _sum, _product, _quotient, _sqrt, _scale2
from ._fixed import _Fix, _check_terms, _ulp
from std.collections import Array
from ._tables import _LN2_VALUE, _QUARTER_PI_VALUE, _EULER_GAMMA_VALUE, _LONG_LIMBS
from ._medium import _read_limbs


def _bits_of(n: Int) -> Int:
    return 64 - Int(count_leading_zeros(UInt64(n)))


def _series_ball(split: _Split, tail: _Radius, bits: Int) raises -> Ball:
    """`T / (B Q)` rounded to nearest at `bits` bits, widened by `tail`."""
    var record = _round_ratio(split.t, split.b * split.q, ArithmeticContext(format=FloatFormat(bits)))
    return _from_record(record^, tail, bits)


def _ratio_bound(numerator: Integer, denominator: Integer) raises -> _Radius:
    """An upper bound of a positive ratio."""
    return _Radius.upper_input(_FloatInput(1, False, numerator, denominator, 0))


struct _Chudnovsky(_SeriesTerms):
    def __init__(out self):
        pass

    def p(self, k: Int) raises -> Integer:
        if k == 0:
            return Integer(1)
        return -(Integer(6 * k - 5) * Integer(2 * k - 1) * Integer(6 * k - 1))

    def q(self, k: Int) raises -> Integer:
        if k == 0:
            return Integer(1)
        var kk = Integer(k)
        return kk * kk * kk * Integer(10939058860032000)

    def a(self, k: Int) raises -> Integer:
        return Integer(13591409) + Integer(545140134) * Integer(k)

    def b(self, k: Int) raises -> Integer:
        return Integer(1)


comptime _TABLE_BITS = 64 * _LONG_LIMBS
"""The width of the ln 2, pi/4 and Euler gamma tables (`_tables.mojo`), certified by
interval arithmetic (mpmath, a development tool) in
scripts/generate_function_tables.py and checked against the series in
tests/test_constants.mojo."""


def _table_ball(table: StaticString, exponent: Int, bits: Int) raises -> Ball:
    """A constant c in `[2**(exponent-1), 2**exponent)` from the table of
    `c 2**-exponent` (ln 2, Euler's gamma, or pi/4 for pi), for
    `bits <= 4608`: the top `bits`
    bits as the midpoint and one unit of their last place as the radius, since
    c lies above the truncation by less than that unit."""
    var significand = _table_bits(table, bits)
    return Ball(
        _midpoint=Float(_rounded=_RoundedBinary(1, False, significand, exponent, FloatFormat(bits), NumericStatus())),
        _radius=_Radius.power_of_two(exponent - bits),
        _kind=_FINITE,
    )


def _table_bits(table: StaticString, bits: Int) raises -> Integer:
    """The top `bits <= 4608` bits of a constant's table: `floor(f 2**bits)`."""
    var count = (bits + 63) // 64
    var limbs = Array[UInt64, _LONG_LIMBS](uninitialized=True)
    var p = Span(limbs).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _read_limbs(table, 0, _LONG_LIMBS, count, p)
    var words = Span(unsafe_ptr=p.unsafe_bitcast[UInt32]().as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=2 * count)
    var value = Integer._from_span(words, False) >> (64 * count - bits)
    _ = limbs^
    return value^


def _table_fix(table: StaticString, exponent: Int, scale: Int) raises -> _Fix:
    """The table's constant c at a fixed-point scale with
    `scale + exponent <= 4608`: `floor(c 2**scale)`, its top bits, within one
    unit below c."""
    return _Fix(_table_bits(table, scale + exponent), _ulp(), scale)


def _pi(w: Int) raises -> Ball:
    """pi; `c = 2`. From the table up to 4608 bits, else by the series."""
    var bits = w + 16
    if bits <= _TABLE_BITS:
        return _table_ball(_QUARTER_PI_VALUE, 2, bits)
    return _pi_series(bits)


def _pi_series(bits: Int) raises -> Ball:
    """pi by Chudnovsky's series, at `bits` bits."""
    # The sum is about 2**23.7 and each term adds more than 47 bits.
    var n = (bits + 40) // 47 + 1
    var split = _binary_split(_Chudnovsky(), 0, n)
    var tail = _Radius.power_of_two(30 - 47 * n).multiply(_up_integer(Integer(n + 1), 0))
    var c = Optional[BallContext](BallContext(bits))
    var root = _sqrt(_BallArgument(Integer(10005)), c)
    var numerator = _product(_BallArgument(Integer(426880)), _BallArgument(root), c)
    return _quotient(_BallArgument(numerator), _BallArgument(_series_ball(split, tail, bits)), c)


struct _ReciprocalFactorials(_SeriesTerms):
    def __init__(out self):
        pass

    def p(self, k: Int) raises -> Integer:
        return Integer(1)

    def q(self, k: Int) raises -> Integer:
        return Integer(max(k, 1))

    def a(self, k: Int) raises -> Integer:
        return Integer(1)

    def b(self, k: Int) raises -> Integer:
        return Integer(1)


def _e(w: Int) raises -> Ball:
    """e; `c = 1`."""
    var bits = w + 16
    # Sum floor(log2 k) for a lower bound of log2(n!) past 2**(bits + 4).
    var n = 2
    var have = 1
    while have < bits + 4:
        n += 1
        have += _bits_of(n) - 1
    var split = _binary_split(_ReciprocalFactorials(), 0, n)
    # Q = (n - 1)!, and sum over k >= n of 1/k! <= 2 / n!.
    return _series_ball(split, _ratio_bound(Integer(2), split.q * Integer(n)), bits)


struct _OddSeries(_SeriesTerms):
    """`sum r**-k / (2k + 1)` for an integer `r >= 2`; after `n` terms the tail
    is at most `r**-n / ((2n + 1)(1 - 1/r)) <= 2 r**-n`."""

    var r: Integer

    def __init__(out self, r: Integer):
        self.r = r

    def p(self, k: Int) raises -> Integer:
        return Integer(1)

    def q(self, k: Int) raises -> Integer:
        return Integer(1) if k == 0 else self.r

    def a(self, k: Int) raises -> Integer:
        return Integer(1)

    def b(self, k: Int) raises -> Integer:
        return Integer(2 * k + 1)


def _atanh_reciprocal(m: Int, bits: Int) raises -> Ball:
    """`atanh(1/m) = (1/m) sum (m**2)**-k / (2k + 1)` for an integer `m >= 2`."""
    var n = bits // (2 * (_bits_of(m) - 1)) + 2
    var split = _binary_split(_OddSeries(Integer(m) * Integer(m)), 0, n)
    split.q = split.q * Integer(m)
    # The tail, divided by m, is at most 2 m**-(2n+1).
    return _series_ball(split, _ratio_bound(Integer(2), Integer(m) ** (2 * n + 1)), bits)


def _ln2(w: Int) raises -> Ball:
    """ln 2; `c = 2`. From the table up to 4608 bits, else by the series."""
    var bits = w + 16
    if bits <= _TABLE_BITS:
        return _table_ball(_LN2_VALUE, 0, bits)
    return _ln2_series(bits)


def _ln2_series(bits: Int) raises -> Ball:
    """ln 2 = 18 atanh(1/26) - 2 atanh(1/4801) + 8 atanh(1/8749), at `bits` bits."""
    var c = Optional[BallContext](BallContext(bits))
    var a = _product(_BallArgument(Integer(18)), _BallArgument(_atanh_reciprocal(26, bits)), c)
    var b = _product(_BallArgument(Integer(2)), _BallArgument(_atanh_reciprocal(4801, bits)), c)
    var d = _product(_BallArgument(Integer(8)), _BallArgument(_atanh_reciprocal(8749, bits)), c)
    return _sum(_BallArgument(_sum(_BallArgument(a), _BallArgument(b), True, c)), _BallArgument(d), False, c)


def _ln10(w: Int) raises -> Ball:
    """ln 10 = 3 ln 2 + 2 atanh(1/9); `c = 2`."""
    var bits = w + 16
    var c = Optional[BallContext](BallContext(bits))
    var three = _product(_BallArgument(Integer(3)), _BallArgument(_ln2(bits)), c)
    var quarter = _scale2(_BallArgument(_atanh_reciprocal(9, bits)), Integer(1), c)
    return _sum(_BallArgument(three), _BallArgument(quarter), False, c)


def _log2_10(w: Int) raises -> Ball:
    """log2(10) = 3 + 2 atanh(1/9) / ln 2; `c = 2`."""
    var bits = w + 16
    var c = Optional[BallContext](BallContext(bits))
    var ratio = _quotient(_BallArgument(_atanh_reciprocal(9, bits)), _BallArgument(_ln2(bits)), c)
    return _sum(_BallArgument(Integer(3)), _BallArgument(_scale2(_BallArgument(ratio), Integer(1), c)), False, c)


def _euler_gamma(w: Int) raises -> Ball:
    """Euler's gamma; `c = 2`. From the table up to 4608 bits, else by the
    series."""
    var bits = w + 16
    if bits <= _TABLE_BITS:
        return _table_ball(_EULER_GAMMA_VALUE, 0, bits)
    return _euler_gamma_series(bits)


def _euler_gamma_series(bits: Int) raises -> Ball:
    """Euler's gamma at `bits` bits by Brent-McMillan.

    For `n >= 1`, `gamma = A/B - log n - K0(2n)/I0(2n)` with
    `B = sum t_k`, `A = sum t_k H_k`, `t_k = (n**k / k!)**2` and harmonic
    numbers `H_k`. The Bessel ratio is positive and below `pi e**(-4n)`, from
    `K0(x) < sqrt(pi / 2x) e**-x` and `I0(x) > e**x / sqrt(2 pi x)` for `x >= 1`.
    `n` is a power of two, so `log n = j ln 2`.
    """
    var scale = bits + 32
    # pi e**(-4n) < 2**(2 - 5.77 n) must fall below 2**-(bits + 8).
    var j = 0
    while (Integer(1) << j) * 577 < Integer(bits + 10) * 100:
        j += 1
    var n = Integer(1) << j
    var n2 = n * n
    var t = _Fix.exact(Integer(1), scale)
    var a = _Fix.exact(Integer(0), scale)
    var total_t = t
    var total_a = a
    var k = 1
    while True:
        # a_k = (a_{k-1} + t_{k-1} / k) n**2 / k**2 and t_k = t_{k-1} n**2 / k**2.
        var kk = Integer(k)
        var k2 = kk * kk
        a = a.add(t.div_integer(kk)).mul_integer(n2).div_integer(k2)
        t = t.mul_integer(n2).div_integer(k2)
        total_t = total_t.add(t)
        total_a = total_a.add(a)
        k += 1
        if Integer(k) > n * 2 and not t.mid and not a.mid:
            break
        _check_terms(k, 4 * Int(n) + scale, "Euler gamma")
    # Beyond K > 2n the ratio of t is at most 1/4, and of t H at most 1/2:
    # tails (4/3) t_K and 2 t_K H_K, with H_K <= bit_length(K) and t_K,
    # a_K below their errors.
    var tail = a.err.add(_Radius.power_of_two(0)).scale2(1).add(
        t.err.add(_Radius.power_of_two(0)).multiply(_up_integer(Integer(2 * _bits_of(k)), 0))
    )
    total_a = total_a.add_error(tail)
    total_t = total_t.add_error(tail)
    var log_n = _Fix.of_ball(_ln2(scale), scale).mul_integer(Integer(j))
    var gamma = total_a.div(total_t).sub(log_n)
    # Subtract the Bessel ratio's midpoint bound and widen by it.
    var bessel = _Radius.power_of_two(2 - (577 * Int(n)) // 100).scale2(scale)
    var half = bessel.scale2(-1)
    gamma = gamma.sub(_Fix(half.ceiling(), _Radius.zero(), scale)).add_error(half.add(_Radius.power_of_two(0)))
    return gamma.to_ball(bits)


struct _CentralBinomial(_SeriesTerms):
    """`sum (k!)**2 / ((2k)! (2k + 1)**2)`."""

    def __init__(out self):
        pass

    def p(self, k: Int) raises -> Integer:
        return Integer(max(k, 1))

    def q(self, k: Int) raises -> Integer:
        return Integer(1 if k == 0 else 2 * (2 * k - 1))

    def a(self, k: Int) raises -> Integer:
        return Integer(1)

    def b(self, k: Int) raises -> Integer:
        var odd = Integer(2 * k + 1)
        return odd * odd


def _catalan(w: Int) raises -> Ball:
    """Catalan's G by Ramanujan's series; `c = 3`."""
    var bits = w + 16
    var c = Optional[BallContext](BallContext(bits))
    # sum (k!)**2 / ((2k)! (2k + 1)**2): each term below 4**-k / (2k + 1).
    var n = bits // 2 + 2
    var series = _series_ball(_binary_split(_CentralBinomial(), 0, n), _ratio_bound(Integer(2), Integer(4) ** n), bits)
    # log(2 + sqrt 3) = 2 atanh(1/sqrt 3) = (2/sqrt 3) sum 3**-k / (2k + 1).
    var m = (bits * 2) // 3 + 2
    var odd = _series_ball(_binary_split(_OddSeries(Integer(3)), 0, m), _ratio_bound(Integer(2), Integer(3) ** m), bits)
    var root = _sqrt(_BallArgument(Integer(3)), c)
    var log_term = _quotient(_BallArgument(_scale2(_BallArgument(odd), Integer(1), c)), _BallArgument(root), c)
    var first = _scale2(_BallArgument(_product(_BallArgument(_pi(bits)), _BallArgument(log_term), c)), Integer(-3), c)
    var second = _scale2(_BallArgument(_product(_BallArgument(series), _BallArgument(Integer(3)), c)), Integer(-3), c)
    return _sum(_BallArgument(first), _BallArgument(second), False, c)
