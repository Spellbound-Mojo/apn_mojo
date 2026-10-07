"""Constants (T-CONST): correct rounding in every mode against MPFR
(tests/fixtures/constants.txt, written by gmpy2), the Appendix H.1 canonical
balls, canonical balls from wider ones, kernel accuracy, budgets and traps."""

from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises
from apn_mojo import (
    Integer,
    Rational,
    Float,
    Ball,
    FloatFormat,
    ArithmeticContext,
    RoundingMode,
    pi,
    euler_e,
    ln2,
    log2_10,
    euler_gamma,
    catalan,
    pi_ball,
    euler_e_ball,
    ln2_ball,
    log2_10_ball,
    euler_gamma_ball,
    catalan_ball,
    canonical,
)
from apn_mojo.float.status import NumericStatus
from apn_mojo.float._rounding import _RoundedBinary
from apn_mojo.ball._fixed import _Fix
from apn_mojo.ball._radius import _Radius
from std.collections import Array
from apn_mojo.ball._medium import _read_limbs
from apn_mojo.ball._exp import _exp_core
from apn_mojo.ball._log import _log_core
from apn_mojo.ball._trig import _sin_cos_core, _atan_core
from apn_mojo.ball._tables import (
    _EXP_SHORT, _EXP_LONG_COARSE, _EXP_LONG_FINE, _LOG_SHORT_COARSE, _LOG_SHORT_FINE, _LOG_LONG_COARSE, _LOG_LONG_FINE, _ATAN_SHORT,
    _ATAN_LONG_COARSE, _ATAN_LONG_FINE, _SIN_COS_SHORT, _SIN_COS_LONG_COARSE, _SIN_COS_LONG_FINE, _LN2_VALUE, _QUARTER_PI_VALUE, _HALF_PI_MINUS_ONE_VALUE, _EULER_GAMMA_VALUE,
)
from apn_mojo.ball._constants import _pi, _e, _ln2, _log2_10, _euler_gamma, _catalan, _ln2_series, _pi_series, _euler_gamma_series, _table_ball


def _float(m: Integer, exponent: Int, precision: Int) raises -> Float:
    return Float(_rounded=_RoundedBinary(1, False, m, exponent, FloatFormat(precision), NumericStatus()))


def _constant(name: String, context: ArithmeticContext) raises -> Float:
    if name == "pi":
        return pi(context=context)
    if name == "euler_e":
        return euler_e(context=context)
    if name == "ln2":
        return ln2(context=context)
    if name == "log2_10":
        return log2_10(context=context)
    if name == "euler_gamma":
        return euler_gamma(context=context)
    return catalan(context=context)


def _ball(name: String, precision: Int) raises -> Ball:
    if name == "pi":
        return pi_ball(precision)
    if name == "euler_e":
        return euler_e_ball(precision)
    if name == "ln2":
        return ln2_ball(precision)
    if name == "log2_10":
        return log2_10_ball(precision)
    if name == "euler_gamma":
        return euler_gamma_ball(precision)
    return catalan_ball(precision)


def _lines() raises -> List[String]:
    var text: String
    with open("tests/fixtures/constants.txt", "r") as source:
        text = source.read()
    var result = List[String]()
    for line in text.splitlines():
        if line.byte_length():
            result.append(String(line))
    return result^


def test_correct_rounding_against_mpfr() raises:
    var modes: List[RoundingMode] = [
        RoundingMode.nearest_even, RoundingMode.toward_zero, RoundingMode.toward_positive,
        RoundingMode.toward_negative, RoundingMode.away_from_zero,
    ]
    var checked = 0
    for line in _lines():
        var fields = line.split(" ")
        if String(fields[0]) == "near":
            continue
        var name = String(fields[0])
        var p = Int(fields[1])
        var exponent = Int(fields[2])
        var down_m = Integer.parse(String(fields[3]), base=16)
        var down = _float(down_m, exponent, p)
        var up_m = down_m + 1
        var up = _float(up_m, exponent, p) if up_m.magnitude_bit_length() == p else _float(Integer(1) << (p - 1), exponent + 1, p)
        var nearest = up if String(fields[4]) == "1" else down
        var expected: List[Float] = [nearest, down, up, down, up]
        for i in range(5):
            var result = _constant(name, ArithmeticContext(format=FloatFormat(p), rounding=modes[i]))
            assert_true(result == expected[i] and result.precision() == p, String(name, " at ", p, " bits, mode ", modes[i]))
            checked += 1
    assert_true(checked > 3000)


def test_high_precision() raises:
    for line in _lines():
        var fields = line.split(" ")
        if String(fields[0]) != "near":
            continue
        var p = Int(fields[2])
        var expected = _float(Integer.parse(String(fields[4]), base=16), Int(fields[3]), p)
        assert_true(_constant(String(fields[1]), ArithmeticContext(format=FloatFormat(p))) == expected, String(fields[1], " at ", p))


def test_canonical_balls() raises:
    # Appendix H.1: the ends of the canonical balls at 64 and 128 bits.
    var names: List[String] = ["pi", "pi", "euler_e", "euler_e", "ln2", "ln2"]
    var precisions: List[Int] = [64, 128, 64, 128, 64, 128]
    var exponents: List[Int] = [2, 2, 2, 2, 0, 0]
    var lows: List[String] = [
        "c90fdaa22168c234", "c90fdaa22168c234c4c6628b80dc1cd1",
        "adf85458a2bb4a9a", "adf85458a2bb4a9aafdc5620273d3cf1",
        "b17217f7d1cf79ab", "b17217f7d1cf79abc9e3b39803f2f6af",
    ]
    for i in range(6):
        var p = precisions[i]
        var ball = _ball(names[i], p)
        var low = _float(Integer.parse(lows[i], base=16), exponents[i], p)
        var high = _float(Integer.parse(lows[i], base=16) + 1, exponents[i], p)
        assert_true(ball.lower_rational() == low.to_rational_exact(), String(names[i], " low at ", p))
        assert_true(ball.upper_rational() == high.to_rational_exact(), String(names[i], " high at ", p))
        assert_equal(ball.precision(), p + 1)
    # A wider ball serves narrower requests with the same canonical ball.
    var all_names: List[String] = ["pi", "euler_e", "ln2", "log2_10", "euler_gamma", "catalan"]
    for name in all_names:
        var wide = _ball(name, 300)
        for p in [2, 17, 53, 64, 200]:
            var narrow = canonical(wide, p)
            assert_true(Bool(narrow) and narrow.value().same_representation(_ball(name, p)), String(name, " canonical at ", p))
        assert_false(Bool(canonical(wide, 299)), String(name, " too narrow a source"))
    # An exact ball's canonical ball is exact; a wide one decides nothing.
    var exact = canonical(Ball(Rational(3, 8)), 10)
    assert_true(exact.value().is_exact() and exact.value().midpoint() == Float(Rational(3, 8), context=ArithmeticContext(format=FloatFormat(8))))
    assert_false(Bool(canonical(Ball(3, Rational(1, 4)), 10)))
    assert_false(Bool(canonical(Ball(0, Rational(1, 2**40)), 10)))


def test_kernel_accuracy() raises:
    # Each kernel's ball has at least w - c bits of relative accuracy, and
    # contains the constant (checked against its canonical ball at w + 40).
    for w in [10, 53, 100, 333, 1000]:
        var balls: List[Ball] = [_pi(w), _e(w), _ln2(w), _log2_10(w), _euler_gamma(w), _catalan(w)]
        var names: List[String] = ["pi", "euler_e", "ln2", "log2_10", "euler_gamma", "catalan"]
        for i in range(6):
            assert_true(balls[i].relative_accuracy_bits() >= w + 8, String(names[i], " accuracy at ", w))
            var truth = _ball(names[i], w + 40)
            assert_true(
                balls[i].lower_rational() <= truth.lower_rational() and truth.upper_rational() <= balls[i].upper_rational(),
                String(names[i], " inclusion at ", w),
            )


def test_budget_traps_and_formats() raises:
    with assert_raises(contains="precision budget exceeded: pi needed more than 60 bits"):
        _ = pi(context=ArithmeticContext(format=FloatFormat(53), max_precision=60))
    with assert_raises(contains="inexact"):
        _ = euler_e(context=ArithmeticContext(format=FloatFormat(53), trap_inexact=True))
    with assert_raises(contains="exactly"):
        _ = ln2(context=ArithmeticContext(format=FloatFormat._exact_format(RoundingMode.nearest_even)))
    # The default is 128 bits, to nearest-even; results are deterministic.
    var first = pi()
    assert_equal(first.precision(), 128)
    assert_true(first == pi() and first == pi_ball(128).midpoint() - Float(Rational(1, 2**127), context=ArithmeticContext(format=FloatFormat(200))))


def _floor_scaled(b: Ball, shift: Int) raises -> Integer:
    var scale = Rational(Integer(1) << shift)
    var low = (b.lower_rational() * scale).floor()
    assert_true(low == (b.upper_rational() * scale).floor(), "the series ball decides the floor")
    return low


def test_constant_tables() raises:
    # The ln 2, pi/4 and Euler gamma tables are the series' floors at their
    # full width, and a table ball at any width contains the series value.
    var ln2 = _ln2_series(4800)
    var pi = _pi_series(4800)
    var euler = _euler_gamma_series(4800)
    assert_true(_floor_scaled(ln2, 4608) == (_table_ball(_LN2_VALUE, 0, 4608).midpoint_rational() * Rational(Integer(1) << 4608)).floor())
    assert_true(_floor_scaled(pi, 4606) == (_table_ball(_QUARTER_PI_VALUE, 2, 4608).midpoint_rational() * Rational(Integer(1) << 4606)).floor())
    assert_true(_floor_scaled(euler, 4608) == (_table_ball(_EULER_GAMMA_VALUE, 0, 4608).midpoint_rational() * Rational(Integer(1) << 4608)).floor())
    for bits in [2, 53, 64, 100, 1000, 4608]:
        var a = _table_ball(_LN2_VALUE, 0, bits)
        var b = _table_ball(_QUARTER_PI_VALUE, 2, bits)
        var g = _table_ball(_EULER_GAMMA_VALUE, 0, bits)
        assert_true(a.lower_rational() <= ln2.lower_rational() and ln2.upper_rational() <= a.upper_rational())
        assert_true(b.lower_rational() <= pi.lower_rational() and pi.upper_rational() <= b.upper_rational())
        assert_true(g.lower_rational() <= euler.lower_rational() and euler.upper_rational() <= g.upper_rational())
        assert_equal(a.midpoint().precision(), bits)


def _entry_top(table: StaticString, entry: Int, limbs: Int) raises -> Integer:
    """An entry's top two limbs: `floor(f 2**128)`."""
    var a = Array[UInt64, 2](uninitialized=True)
    var p = Span(a).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    _read_limbs(table, entry, limbs, 2, p)
    var value = (Integer(p.unsafe_offset(1)[]) << 64) + Integer(p.unsafe_offset(0)[])
    _ = a^
    return value^


def _near(entry: Integer, value: _Fix) raises -> Bool:
    return abs(entry - value.mid) <= value.err.ceiling() + 2


def test_medium_tables() raises:
    # The top 128 bits of every entry of the 512-bit tables and of the
    # 4608-bit ones, against the general series kernels at that scale.
    comptime s = 128
    var zero = _Radius.zero()
    for i in range(178):
        assert_true(_near(_entry_top(_EXP_SHORT, i, 8) * 2, _exp_core(_Fix(Integer(i) << (s - 8), zero, s))), String("exp entry ", i))
    for i in range(23):
        assert_true(_near(_entry_top(_EXP_LONG_COARSE, i, 72) * 2, _exp_core(_Fix(Integer(i) << (s - 5), zero, s))), String("exp entry ", i))
    for i in range(32):
        assert_true(_near(_entry_top(_EXP_LONG_FINE, i, 72) * 2, _exp_core(_Fix(Integer(i) << (s - 10), zero, s))), String("exp entry ", i))
    for i in range(1, 128):
        assert_true(_near(_entry_top(_LOG_SHORT_COARSE, i, 8), _log_core(_Fix(Integer(128 + i) << (s - 7), zero, s))), String("log entry ", i))
        assert_true(_near(_entry_top(_LOG_SHORT_FINE, i, 8), _log_core(_Fix(Integer(16384 + i) << (s - 14), zero, s))), String("log entry ", i))
    for i in range(1, 32):
        assert_true(_near(_entry_top(_LOG_LONG_COARSE, i, 72), _log_core(_Fix(Integer(32 + i) << (s - 5), zero, s))), String("log entry ", i))
        assert_true(_near(_entry_top(_LOG_LONG_FINE, i, 72), _log_core(_Fix(Integer(1024 + i) << (s - 10), zero, s))), String("log entry ", i))
    for i in range(256):
        assert_true(_near(_entry_top(_ATAN_SHORT, i, 8), _atan_core(_Fix(Integer(i) << (s - 8), zero, s))), String("atan entry ", i))
    for i in range(32):
        assert_true(_near(_entry_top(_ATAN_LONG_COARSE, i, 72), _atan_core(_Fix(Integer(i) << (s - 5), zero, s))), String("atan entry ", i))
        assert_true(_near(_entry_top(_ATAN_LONG_FINE, i, 72), _atan_core(_Fix(Integer(i) << (s - 10), zero, s))), String("atan entry ", i))
    for i in range(1, 203):
        var pair = _sin_cos_core(_Fix(Integer(i) << (s - 8), zero, s))
        assert_true(_near(_entry_top(_SIN_COS_SHORT, 2 * i, 8), pair[0]) and _near(_entry_top(_SIN_COS_SHORT, 2 * i + 1, 8), pair[1]), String("sin and cos entry ", i))
    for i in range(1, 26):
        var pair = _sin_cos_core(_Fix(Integer(i) << (s - 5), zero, s))
        assert_true(_near(_entry_top(_SIN_COS_LONG_COARSE, 2 * i, 72), pair[0]) and _near(_entry_top(_SIN_COS_LONG_COARSE, 2 * i + 1, 72), pair[1]), String("sin and cos entry ", i))
    for i in range(1, 32):
        var pair = _sin_cos_core(_Fix(Integer(i) << (s - 10), zero, s))
        assert_true(_near(_entry_top(_SIN_COS_LONG_FINE, 2 * i, 72), pair[0]) and _near(_entry_top(_SIN_COS_LONG_FINE, 2 * i + 1, 72), pair[1]), String("sin and cos entry ", i))
    # ln 2, pi/4 and pi/2 - 1 against binary splitting.
    assert_true(_near(_entry_top(_LN2_VALUE, 0, 72), _Fix.of_ball(_ln2_series(200), s)))
    assert_true(_near(_entry_top(_QUARTER_PI_VALUE, 0, 72) * 4, _Fix.of_ball(_pi_series(200), s)))
    assert_true(_near(_entry_top(_HALF_PI_MINUS_ONE_VALUE, 0, 72) * 2 + (Integer(2) << s), _Fix.of_ball(_pi_series(200), s)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
