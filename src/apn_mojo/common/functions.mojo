"""Package-level math functions: one overload per number family.

add(x, y) picks the Integer, Rational, Float or Complex version from its
arguments. Each family's own version is a single declaration, for example
apn_mojo.complex.add, and that is what vmap maps over batches: Mojo 1.1
cannot pass an overload set such as this one as a function value. Rational
and Float forms have lower priority, so integer-only calls stay exact, and a
context= argument selects the rounded Float form.
"""

from std.math import Absable, abs as _stdlib_abs
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..float.context import ArithmeticContext
from ..float._arithmetic import _FloatArgument
from ..complex._input import _ComplexArgument
from ..complex.context import _ComplexContextArgument
from ..integer.math import add as _integer_add, subtract as _integer_subtract, multiply as _integer_multiply, divide as _integer_divide
from ..rational.math import add as _rational_add, subtract as _rational_subtract, multiply as _rational_multiply, divide as _rational_divide
from ..float.math import add as _float_add, subtract as _float_subtract, multiply as _float_multiply, divide as _float_divide
from ..complex.math import add as _complex_add, subtract as _complex_subtract, multiply as _complex_multiply, divide as _complex_divide
from ..float.math import sqrt as _float_sqrt, pow_int as _float_pow_int
from ..complex.math import sqrt as _complex_sqrt, pow_int as _complex_pow_int
from ..integer.math import stable_hash as _integer_stable_hash
from ..rational.math import stable_hash as _rational_stable_hash
from ..float.math import stable_hash as _float_stable_hash
from ..complex.math import stable_hash as _complex_stable_hash
from ..ball.value import Ball, _BallArgument
from ..exact_complex.value import ExactComplex
from ..exact_complex.math import stable_hash as _exact_complex_stable_hash
from ..exact_complex.math import conjugate as _exact_complex_conjugate, real as _exact_complex_real, imag as _exact_complex_imag, reciprocal as _exact_complex_reciprocal
from ..complex.elementary import exp as _complex_exp, log as _complex_log, sin as _complex_sin, cos as _complex_cos, tan as _complex_tan, sinh as _complex_sinh, cosh as _complex_cosh, tanh as _complex_tanh, asin as _complex_asin, acos as _complex_acos, atan as _complex_atan, asinh as _complex_asinh, acosh as _complex_acosh, atanh as _complex_atanh, pow as _complex_pow, angle as _complex_angle
from ..complex_ball.value import ComplexBall
from ..complex_ball.elementary import exp as _complex_ball_exp, log as _complex_ball_log, sin as _complex_ball_sin, cos as _complex_ball_cos, tan as _complex_ball_tan, sinh as _complex_ball_sinh, cosh as _complex_ball_cosh, tanh as _complex_ball_tanh, asin as _complex_ball_asin, acos as _complex_ball_acos, atan as _complex_ball_atan, asinh as _complex_ball_asinh, acosh as _complex_ball_acosh, atanh as _complex_ball_atanh, sqrt as _complex_ball_sqrt, pow as _complex_ball_pow
from ..complex_ball.math import angle as _complex_ball_angle, stable_hash as _complex_ball_stable_hash
from ..complex_ball.math import add as _complex_ball_add, subtract as _complex_ball_subtract, multiply as _complex_ball_multiply, divide as _complex_ball_divide, pow_int as _complex_ball_pow_int
from ..complex_ball.math import abs as _complex_ball_abs, conjugate as _complex_ball_conjugate, real as _complex_ball_real, imag as _complex_ball_imag, reciprocal as _complex_ball_reciprocal
from ..ball.context import BallContext
from ..ball.math import add as _ball_add, subtract as _ball_subtract, multiply as _ball_multiply, divide as _ball_divide
from ..ball.math import sqrt as _ball_sqrt, abs as _ball_abs, pow_int as _ball_pow_int, stable_hash as _ball_stable_hash
from ..ball.math import reciprocal as _ball_reciprocal, maximum as _ball_maximum, minimum as _ball_minimum, clip as _ball_clip
from ..integer.math import comb as _integer_comb
from ..integer.math import floor as _integer_floor, ceil as _integer_ceil, trunc as _integer_trunc, round as _integer_round, reciprocal as _integer_reciprocal, maximum as _integer_maximum, minimum as _integer_minimum, clip as _integer_clip
from ..rational.math import floor as _rational_floor, ceil as _rational_ceil, trunc as _rational_trunc, round as _rational_round, reciprocal as _rational_reciprocal, maximum as _rational_maximum, minimum as _rational_minimum, clip as _rational_clip
from ..float.math import floor as _float_floor, ceil as _float_ceil, trunc as _float_trunc, round as _float_round, reciprocal as _float_reciprocal, maximum as _float_maximum, minimum as _float_minimum, clip as _float_clip
from ..complex.math import abs as _complex_abs, conjugate as _complex_conjugate, real as _complex_real, imag as _complex_imag, reciprocal as _complex_reciprocal
from ..float.elementary import exp as _float_exp, expm1 as _float_expm1, exp2 as _float_exp2, log as _float_log, log1p as _float_log1p, log2 as _float_log2, log10 as _float_log10, sin as _float_sin, cos as _float_cos, tan as _float_tan, atan as _float_atan, asin as _float_asin, acos as _float_acos, sinh as _float_sinh, cosh as _float_cosh, tanh as _float_tanh, asinh as _float_asinh, acosh as _float_acosh, atanh as _float_atanh, sin_cos as _float_sin_cos, atan2 as _float_atan2, pow as _float_pow, rootn as _float_rootn
from ..ball.elementary import exp as _ball_exp, expm1 as _ball_expm1, exp2 as _ball_exp2, log as _ball_log, log1p as _ball_log1p, log2 as _ball_log2, log10 as _ball_log10, sin as _ball_sin, cos as _ball_cos, tan as _ball_tan, atan as _ball_atan, asin as _ball_asin, acos as _ball_acos, sinh as _ball_sinh, cosh as _ball_cosh, tanh as _ball_tanh, asinh as _ball_asinh, acosh as _ball_acosh, atanh as _ball_atanh, sin_cos as _ball_sin_cos, atan2 as _ball_atan2, pow as _ball_pow, rootn as _ball_rootn
from ..float.special import gamma as _float_gamma, gammaln as _float_gammaln, digamma as _float_digamma, erf as _float_erf, erfc as _float_erfc, erfi as _float_erfi, expi as _float_expi, sici as _float_sici, shichi as _float_shichi, fresnel as _float_fresnel, lambertw as _float_lambertw, ndtr as _float_ndtr, log_ndtr as _float_log_ndtr, beta as _float_beta, betaln as _float_betaln, poch as _float_poch, erfinv as _float_erfinv, ndtri as _float_ndtri, zeta as _float_zeta, polygamma as _float_polygamma, hyp1f1 as _float_hyp1f1, gammainc as _float_gammainc, gammaincc as _float_gammaincc, hyp2f1 as _float_hyp2f1, betainc as _float_betainc
from ..ball.special import gamma as _ball_gamma, gammaln as _ball_gammaln, digamma as _ball_digamma, erf as _ball_erf, erfc as _ball_erfc, erfi as _ball_erfi, expi as _ball_expi, sici as _ball_sici, shichi as _ball_shichi, fresnel as _ball_fresnel, lambertw as _ball_lambertw, ndtr as _ball_ndtr, log_ndtr as _ball_log_ndtr, beta as _ball_beta, betaln as _ball_betaln, poch as _ball_poch, erfinv as _ball_erfinv, ndtri as _ball_ndtri, zeta as _ball_zeta, polygamma as _ball_polygamma, hyp1f1 as _ball_hyp1f1, gammainc as _ball_gammainc, gammaincc as _ball_gammaincc, hyp2f1 as _ball_hyp2f1, betainc as _ball_betainc


def add(left: Integer, right: Integer) raises -> Integer:
    """`left + right`, choosing the family from the arguments.

    The package-level name has one overload per family: two exact operands give
    an exact Integer or Rational, a `context=` or a Float operand gives a Float rounded once,
    a Complex operand gives a Complex, and balls or complex balls give an
    enclosing ball of their family. To pass the operation to `vmap`, name a
    family's single declaration, such as `apn_mojo.float.add`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        An exact Integer or Rational.

    Raises:
        Only on a checked size error.
    """
    return _integer_add(left, right)


def add(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The exact Rational `left + right`."""
    return _rational_add(left, right)


def add(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The Float `left + right`, rounded once; a context requests a Float even for exact operands."""
    return _float_add(left, right, context=context)


def add(
    left: Complex,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The Complex `left + right`, each component rounded once."""
    return _complex_add(left, right, context=context)


def add(
    left: Complex,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left + right` with a real or exact right operand."""
    return _complex_add(left, right, context=context)


def add(
    left: _ComplexArgument,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left + right` with a real or exact left operand."""
    return _complex_add(left, right, context=context)

def subtract(left: Integer, right: Integer) raises -> Integer:
    """`left - right`, choosing the family from the arguments.

    The package-level name has one overload per family: two exact operands give
    an exact Integer or Rational, a `context=` or a Float operand gives a Float rounded once,
    a Complex operand gives a Complex, and balls or complex balls give an
    enclosing ball of their family. To pass the operation to `vmap`, name a
    family's single declaration, such as `apn_mojo.float.subtract`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        An exact Integer or Rational.

    Raises:
        Only on a checked size error.
    """
    return _integer_subtract(left, right)


def subtract(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The exact Rational `left - right`."""
    return _rational_subtract(left, right)


def subtract(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The Float `left - right`, rounded once; a context requests a Float even for exact operands."""
    return _float_subtract(left, right, context=context)


def subtract(
    left: Complex,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The Complex `left - right`, each component rounded once."""
    return _complex_subtract(left, right, context=context)


def subtract(
    left: Complex,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left - right` with a real or exact right operand."""
    return _complex_subtract(left, right, context=context)


def subtract(
    left: _ComplexArgument,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left - right` with a real or exact left operand."""
    return _complex_subtract(left, right, context=context)

def multiply(left: Integer, right: Integer) raises -> Integer:
    """`left * right`, choosing the family from the arguments.

    The package-level name has one overload per family: two exact operands give
    an exact Integer or Rational, a `context=` or a Float operand gives a Float rounded once,
    a Complex operand gives a Complex, and balls or complex balls give an
    enclosing ball of their family. To pass the operation to `vmap`, name a
    family's single declaration, such as `apn_mojo.float.multiply`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        An exact Integer or Rational.

    Raises:
        Only on a checked size error.
    """
    return _integer_multiply(left, right)


def multiply(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The exact Rational `left * right`."""
    return _rational_multiply(left, right)


def multiply(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The Float `left * right`, rounded once; a context requests a Float even for exact operands."""
    return _float_multiply(left, right, context=context)


def multiply(
    left: Complex,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The Complex `left * right`, each component rounded once."""
    return _complex_multiply(left, right, context=context)


def multiply(
    left: Complex,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left * right` with a real or exact right operand."""
    return _complex_multiply(left, right, context=context)


def multiply(
    left: _ComplexArgument,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left * right` with a real or exact left operand."""
    return _complex_multiply(left, right, context=context)

def divide(left: Integer, right: Integer) raises -> Rational:
    """`left / right`, choosing the family from the arguments.

    The package-level name has one overload per family: two exact operands give
    an exact Rational, a `context=` or a Float operand gives a Float rounded once,
    a Complex operand gives a Complex, and balls or complex balls give an
    enclosing ball of their family. To pass the operation to `vmap`, name a
    family's single declaration, such as `apn_mojo.float.divide`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        An exact Rational.

    Raises:
        When `right` is zero.
    """
    return _integer_divide(left, right)


def divide(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The exact Rational `left / right`."""
    return _rational_divide(left, right)


def divide(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The Float `left / right`, rounded once; a context requests a Float even for exact operands."""
    return _float_divide(left, right, context=context)


def divide(
    left: Complex,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The Complex `left / right`, each component rounded once."""
    return _complex_divide(left, right, context=context)


def divide(
    left: Complex,
    right: _ComplexArgument,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left / right` with a real or exact right operand."""
    return _complex_divide(left, right, context=context)


def divide(
    left: _ComplexArgument,
    right: Complex,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """A Complex `left / right` with a real or exact left operand."""
    return _complex_divide(left, right, context=context)


def sqrt(
    value: _FloatArgument, *, context: Optional[ArithmeticContext] = None
) raises -> Float:
    """The square root, choosing the family from the argument.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode and traps.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, or an exact value without a context.
    """
    return _float_sqrt(value, context=context)


def sqrt(
    value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()
) raises -> Complex:
    """The principal square root of a Complex, each component rounded once."""
    return _complex_sqrt(value, context=context)


def pow_int(
    value: _FloatArgument,
    exponent: Integer,
    *,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The power for a signed integral exponent, choosing the family from the base.

    Args:
        value: A Float or exact base; exact bases need a context.
        exponent: The exponent, of any size.
        context: The output format, rounding mode and traps.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, or an exact base without a context.
    """
    return _float_pow_int(value, exponent, context=context)


def pow_int(
    value: Complex,
    exponent: Integer,
    *,
    context: _ComplexContextArgument = _ComplexContextArgument(),
) raises -> Complex:
    """The power of a Complex for a signed integral exponent, each component rounded once."""
    return _complex_pow_int(value, exponent, context=context)


def abs(value: IntLiteral) -> Integer:
    """The absolute value of an integer literal, as an exact Integer.

    Args:
        value: The literal, of any width.

    Returns:
        Its magnitude.
    """
    return _stdlib_abs(Integer(value))


def abs[T: Absable](value: T) -> T:
    """The absolute value of an Integer, Rational, Float or native number, in the same type."""
    return _stdlib_abs(value)


def abs(value: Complex, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The magnitude of a Complex, `sqrt(real**2 + imag**2)`, rounded once (numpy's `abs`)."""
    return _complex_abs(value, context=context)


def abs(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the magnitudes of a complex ball."""
    return _complex_ball_abs(value, context=context)


def stable_hash(x: Integer) -> UInt64:
    """A 64-bit hash stable across processes and releases, choosing the family
    from the argument.

    The algorithm is APNH-64, with one encoding per family; see the family
    versions, such as `apn_mojo.float.stable_hash`. Floats and Complex numbers
    hash their representation, as `FloatKey` and `ComplexKey` compare them.

    Args:
        x: The number.

    Returns:
        The hash.
    """
    return _integer_stable_hash(x)


def stable_hash(x: Rational, *__disambiguate: NoneType) -> UInt64:
    """The stable hash of a Rational."""
    return _rational_stable_hash(x)


def stable_hash(x: Float, *__disambiguate: NoneType) -> UInt64:
    """The stable hash of a Float's representation."""
    return _float_stable_hash(x)


def stable_hash(x: Complex, *__disambiguate: NoneType) -> UInt64:
    """The stable hash of a Complex number's representation."""
    return _complex_stable_hash(x)


def add(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left + right`, containing every result."""
    return _ball_add(left, right, context=context)


def add(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left + right` with an exact right operand."""
    return _ball_add(left, right, context=context)


def add(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left + right` with an exact left operand."""
    return _ball_add(left, right, context=context)


def add(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `left + right`, containing every result."""
    return _complex_ball_add(left, right, context=context)


def subtract(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left - right`, containing every result."""
    return _ball_subtract(left, right, context=context)


def subtract(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left - right` with an exact right operand."""
    return _ball_subtract(left, right, context=context)


def subtract(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left - right` with an exact left operand."""
    return _ball_subtract(left, right, context=context)


def subtract(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `left - right`, containing every result."""
    return _complex_ball_subtract(left, right, context=context)


def multiply(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left * right`, containing every result."""
    return _ball_multiply(left, right, context=context)


def multiply(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left * right` with an exact right operand."""
    return _ball_multiply(left, right, context=context)


def multiply(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left * right` with an exact left operand."""
    return _ball_multiply(left, right, context=context)


def multiply(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `left * right`, containing every result."""
    return _complex_ball_multiply(left, right, context=context)


def divide(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left / right`, containing every result."""
    return _ball_divide(left, right, context=context)


def divide(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left / right` with an exact right operand."""
    return _ball_divide(left, right, context=context)


def divide(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `left / right` with an exact left operand."""
    return _ball_divide(left, right, context=context)


def divide(left: ComplexBall, right: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `left / right`, containing every result."""
    return _complex_ball_divide(left, right, context=context)


def sqrt(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the square root; indeterminate when it reaches below 0."""
    return _ball_sqrt(value, context=context)


def pow_int(value: Ball, exponent: Integer, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of an integer power."""
    return _ball_pow_int(value, exponent, context=context)


def pow_int(value: ComplexBall, exponent: Integer, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of an integer power."""
    return _complex_ball_pow_int(value, exponent, context=context)


def abs(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the absolute value."""
    return _ball_abs(value, context=context)


def stable_hash(x: ExactComplex, *__disambiguate: NoneType) raises -> UInt64:
    """The APNH-64 hash of an ExactComplex: tag 5 and its four integers."""
    return _exact_complex_stable_hash(x)


def stable_hash(x: Ball, *__disambiguate: NoneType) raises -> UInt64:
    """The stable hash of a Ball's representation."""
    return _ball_stable_hash(x)


def exp(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`exp`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.exp`; a Ball gives the enclosing ball of
    `apn_mojo.ball.exp`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_exp(value, context=context)


def exp(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `exp`, enclosing its value at every point."""
    return _ball_exp(value, context=context)


def expm1(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`expm1`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.expm1`; a Ball gives the enclosing ball of
    `apn_mojo.ball.expm1`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_expm1(value, context=context)


def expm1(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `expm1`, enclosing its value at every point."""
    return _ball_expm1(value, context=context)


def exp2(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`exp2`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.exp2`; a Ball gives the enclosing ball of
    `apn_mojo.ball.exp2`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_exp2(value, context=context)


def exp2(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `exp2`, enclosing its value at every point."""
    return _ball_exp2(value, context=context)


def log(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`log`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.log`; a Ball gives the enclosing ball of
    `apn_mojo.ball.log`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_log(value, context=context)


def log(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log`, enclosing its value at every point."""
    return _ball_log(value, context=context)


def log1p(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`log1p`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.log1p`; a Ball gives the enclosing ball of
    `apn_mojo.ball.log1p`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_log1p(value, context=context)


def log1p(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log1p`, enclosing its value at every point."""
    return _ball_log1p(value, context=context)


def log2(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`log2`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.log2`; a Ball gives the enclosing ball of
    `apn_mojo.ball.log2`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_log2(value, context=context)


def log2(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log2`, enclosing its value at every point."""
    return _ball_log2(value, context=context)


def log10(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`log10`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.log10`; a Ball gives the enclosing ball of
    `apn_mojo.ball.log10`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_log10(value, context=context)


def log10(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log10`, enclosing its value at every point."""
    return _ball_log10(value, context=context)


def sin(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`sin`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.sin`; a Ball gives the enclosing ball of
    `apn_mojo.ball.sin`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_sin(value, context=context)


def sin(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `sin`, enclosing its value at every point."""
    return _ball_sin(value, context=context)


def cos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`cos`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.cos`; a Ball gives the enclosing ball of
    `apn_mojo.ball.cos`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_cos(value, context=context)


def cos(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `cos`, enclosing its value at every point."""
    return _ball_cos(value, context=context)


def tan(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`tan`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.tan`; a Ball gives the enclosing ball of
    `apn_mojo.ball.tan`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_tan(value, context=context)


def tan(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `tan`, enclosing its value at every point."""
    return _ball_tan(value, context=context)


def atan(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`atan`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.atan`; a Ball gives the enclosing ball of
    `apn_mojo.ball.atan`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_atan(value, context=context)


def atan(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `atan`, enclosing its value at every point."""
    return _ball_atan(value, context=context)


def asin(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`asin`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.asin`; a Ball gives the enclosing ball of
    `apn_mojo.ball.asin`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_asin(value, context=context)


def asin(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `asin`, enclosing its value at every point."""
    return _ball_asin(value, context=context)


def acos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`acos`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.acos`; a Ball gives the enclosing ball of
    `apn_mojo.ball.acos`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_acos(value, context=context)


def acos(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `acos`, enclosing its value at every point."""
    return _ball_acos(value, context=context)


def sinh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`sinh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.sinh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.sinh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_sinh(value, context=context)


def sinh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `sinh`, enclosing its value at every point."""
    return _ball_sinh(value, context=context)


def cosh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`cosh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.cosh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.cosh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_cosh(value, context=context)


def cosh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `cosh`, enclosing its value at every point."""
    return _ball_cosh(value, context=context)


def tanh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`tanh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.tanh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.tanh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_tanh(value, context=context)


def tanh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `tanh`, enclosing its value at every point."""
    return _ball_tanh(value, context=context)


def asinh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`asinh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.asinh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.asinh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_asinh(value, context=context)


def asinh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `asinh`, enclosing its value at every point."""
    return _ball_asinh(value, context=context)


def acosh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`acosh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.acosh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.acosh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_acosh(value, context=context)


def acosh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `acosh`, enclosing its value at every point."""
    return _ball_acosh(value, context=context)


def atanh(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`atanh`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.atanh`; a Ball gives the enclosing ball of
    `apn_mojo.ball.atanh`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_atanh(value, context=context)


def atanh(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `atanh`, enclosing its value at every point."""
    return _ball_atanh(value, context=context)


def sin_cos(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """The sine and cosine, choosing the family from the argument.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        Both, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_sin_cos(value, context=context)


def sin_cos(value: Ball, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of the sine and cosine."""
    return _ball_sin_cos(value, context=context)


def atan2(y: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The angle of `(x, y)`, choosing the family from the arguments.

    Args:
        y: The ordinate.
        x: The abscissa.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float in `[-pi, pi]`, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without a
        context.
    """
    return _float_atan2(y, x, context=context)


def atan2(y: Ball, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the angle of `x + i y`."""
    return _ball_atan2(y, x, context=context)


def atan2(y: Ball, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the angle of `x + i y`, for an exact abscissa."""
    return _ball_atan2(y, x, context=context)


def atan2(y: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the angle of `x + i y`, for an exact ordinate."""
    return _ball_atan2(y, x, context=context)


def pow(base: _FloatArgument, exponent: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The power `base**exponent`, choosing the family from the arguments.

    Args:
        base: The base.
        exponent: The exponent.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without a
        context.
    """
    return _float_pow(base, exponent, context=context)


def pow(base: Ball, exponent: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `base**exponent`."""
    return _ball_pow(base, exponent, context=context)


def pow(base: Ball, exponent: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `base**exponent` for an exact exponent."""
    return _ball_pow(base, exponent, context=context)


def pow(base: _BallArgument, exponent: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `base**exponent` for an exact base."""
    return _ball_pow(base, exponent, context=context)


def rootn(value: _FloatArgument, n: Int, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The real `n`-th root, choosing the family from the argument.

    Args:
        value: A Float or exact value; exact values need a context.
        n: The degree, at least 1.
        context: The output format, rounding mode and traps.

    Returns:
        A Float, rounded once.

    Raises:
        When `n` is below 1, on a trapped condition, or for an exact value
        without a context.
    """
    return _float_rootn(value, n, context=context)


def rootn(value: Ball, n: Int, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the real `n`-th root."""
    return _ball_rootn(value, n, context=context)


def exp(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `exp`, each part correctly rounded."""
    return _complex_exp(value, context=context)


def exp(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `exp`, enclosing its value at every point."""
    return _complex_ball_exp(value, context=context)


def log(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `log`, each part correctly rounded."""
    return _complex_log(value, context=context)


def log(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `log`, enclosing its value at every point."""
    return _complex_ball_log(value, context=context)


def sin(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `sin`, each part correctly rounded."""
    return _complex_sin(value, context=context)


def sin(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `sin`, enclosing its value at every point."""
    return _complex_ball_sin(value, context=context)


def cos(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `cos`, each part correctly rounded."""
    return _complex_cos(value, context=context)


def cos(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `cos`, enclosing its value at every point."""
    return _complex_ball_cos(value, context=context)


def tan(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `tan`, each part correctly rounded."""
    return _complex_tan(value, context=context)


def tan(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `tan`, enclosing its value at every point."""
    return _complex_ball_tan(value, context=context)


def sinh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `sinh`, each part correctly rounded."""
    return _complex_sinh(value, context=context)


def sinh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `sinh`, enclosing its value at every point."""
    return _complex_ball_sinh(value, context=context)


def cosh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `cosh`, each part correctly rounded."""
    return _complex_cosh(value, context=context)


def cosh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `cosh`, enclosing its value at every point."""
    return _complex_ball_cosh(value, context=context)


def tanh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `tanh`, each part correctly rounded."""
    return _complex_tanh(value, context=context)


def tanh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `tanh`, enclosing its value at every point."""
    return _complex_ball_tanh(value, context=context)


def asin(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `asin`, each part correctly rounded."""
    return _complex_asin(value, context=context)


def asin(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `asin`, enclosing its value at every point."""
    return _complex_ball_asin(value, context=context)


def acos(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `acos`, each part correctly rounded."""
    return _complex_acos(value, context=context)


def acos(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `acos`, enclosing its value at every point."""
    return _complex_ball_acos(value, context=context)


def atan(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `atan`, each part correctly rounded."""
    return _complex_atan(value, context=context)


def atan(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `atan`, enclosing its value at every point."""
    return _complex_ball_atan(value, context=context)


def asinh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `asinh`, each part correctly rounded."""
    return _complex_asinh(value, context=context)


def asinh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `asinh`, enclosing its value at every point."""
    return _complex_ball_asinh(value, context=context)


def acosh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `acosh`, each part correctly rounded."""
    return _complex_acosh(value, context=context)


def acosh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `acosh`, enclosing its value at every point."""
    return _complex_ball_acosh(value, context=context)


def atanh(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `atanh`, each part correctly rounded."""
    return _complex_atanh(value, context=context)


def atanh(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `atanh`, enclosing its value at every point."""
    return _complex_ball_atanh(value, context=context)


def sqrt(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of the principal square root."""
    return _complex_ball_sqrt(value, context=context)


def pow(base: Complex, exponent: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex principal power, each part correctly rounded."""
    return _complex_pow(base, exponent, context=context)


def pow(base: ComplexBall, exponent: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of the principal power."""
    return _complex_ball_pow(base, exponent, context=context)


def angle(value: Complex, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The angle of a Complex, choosing the family from the argument (numpy's
    `angle`).

    Args:
        value: A Complex.
        context: The output format, rounding mode, traps and budget.

    Returns:
        `atan2(imag, real)`, correctly rounded, in `[-pi, pi]`.

    Raises:
        On a trapped condition, or past the budget.
    """
    return _complex_angle(value, context=context)


def angle(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the angles of a complex ball, `pi` on the negative real axis."""
    return _complex_ball_angle(value, context=context)


def stable_hash(x: ComplexBall, *__disambiguate: NoneType) raises -> UInt64:
    """The APNH-64 hash of a complex ball: tag 7 and its two balls."""
    return _complex_ball_stable_hash(x)


def gamma(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`gamma`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.gamma`; a Ball gives the enclosing ball of
    `apn_mojo.ball.gamma`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_gamma(value, context=context)


def gamma(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gamma`, enclosing its value at every point."""
    return _ball_gamma(value, context=context)


def gammaln(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`gammaln`, `log |Gamma(x)|`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.gammaln`; a Ball gives the enclosing ball of
    `apn_mojo.ball.gammaln`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_gammaln(value, context=context)


def gammaln(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammaln`, enclosing its value at every point."""
    return _ball_gammaln(value, context=context)


def digamma(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`digamma`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.digamma`; a Ball gives the enclosing ball of
    `apn_mojo.ball.digamma`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_digamma(value, context=context)


def digamma(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `digamma`, enclosing its value at every point."""
    return _ball_digamma(value, context=context)


def erf(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`erf`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.erf`; a Ball gives the enclosing ball of
    `apn_mojo.ball.erf`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_erf(value, context=context)


def erf(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `erf`, enclosing its value at every point."""
    return _ball_erf(value, context=context)


def erfc(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`erfc`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.erfc`; a Ball gives the enclosing ball of
    `apn_mojo.ball.erfc`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_erfc(value, context=context)


def erfc(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `erfc`, enclosing its value at every point."""
    return _ball_erfc(value, context=context)


def erfi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`erfi`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.erfi`; a Ball gives the enclosing ball of
    `apn_mojo.ball.erfi`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_erfi(value, context=context)


def erfi(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `erfi`, enclosing its value at every point."""
    return _ball_erfi(value, context=context)


def expi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`expi`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.expi`; a Ball gives the enclosing ball of
    `apn_mojo.ball.expi`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_expi(value, context=context)


def expi(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `expi`, enclosing its value at every point."""
    return _ball_expi(value, context=context)


def sici(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """`(Si(x), Ci(x))`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Floats of
    `apn_mojo.float.sici`; a Ball gives the enclosing balls of
    `apn_mojo.ball.sici`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        Two Floats, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_sici(value, context=context)


def sici(value: Ball, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of `sici`, enclosing their values at every point."""
    return _ball_sici(value, context=context)


def shichi(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """`(Shi(x), Chi(x))`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Floats of
    `apn_mojo.float.shichi`; a Ball gives the enclosing balls of
    `apn_mojo.ball.shichi`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        Two Floats, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_shichi(value, context=context)


def shichi(value: Ball, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of `shichi`, enclosing their values at every point."""
    return _ball_shichi(value, context=context)


def fresnel(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Tuple[Float, Float]:
    """Fresnel's `(S(x), C(x))`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Floats of
    `apn_mojo.float.fresnel`; a Ball gives the enclosing balls of
    `apn_mojo.ball.fresnel`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        Two Floats, each rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_fresnel(value, context=context)


def fresnel(value: Ball, *, context: Optional[BallContext] = None) raises -> Tuple[Ball, Ball]:
    """The balls of `fresnel`, enclosing their values at every point."""
    return _ball_fresnel(value, context=context)


def lambertw(value: _FloatArgument, *, k: Int = 0, context: Optional[ArithmeticContext] = None) raises -> Float:
    """`lambertw`, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.lambertw`; a Ball gives the enclosing ball of
    `apn_mojo.ball.lambertw`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        k: The branch, 0 or -1.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_lambertw(value, k=k, context=context)


def lambertw(value: Ball, *, k: Int = 0, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `lambertw`, enclosing its value at every point."""
    return _ball_lambertw(value, k=k, context=context)


def floor(value: Integer) raises -> Integer:
    """`floor`, choosing the family from the argument: round toward negative infinity, to an
    Integer (numpy's `floor`). An Integer is its own floor; to pass the
    function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.floor`.

    Args:
        value: An Integer, Rational or Float.

    Returns:
        The Integer.

    Raises:
        For an infinite or NaN Float.
    """
    return _integer_floor(value)


def floor(value: Rational, *__disambiguate: NoneType) raises -> Integer:
    """`floor` of a Rational, exactly."""
    return _rational_floor(value)


def floor(value: Float) raises -> Integer:
    """`floor` of a Float, exactly."""
    return _float_floor(value)


def ceil(value: Integer) raises -> Integer:
    """`ceil`, choosing the family from the argument: round toward positive infinity, to an
    Integer (numpy's `ceil`). An Integer is its own ceil; to pass the
    function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.ceil`.

    Args:
        value: An Integer, Rational or Float.

    Returns:
        The Integer.

    Raises:
        For an infinite or NaN Float.
    """
    return _integer_ceil(value)


def ceil(value: Rational, *__disambiguate: NoneType) raises -> Integer:
    """`ceil` of a Rational, exactly."""
    return _rational_ceil(value)


def ceil(value: Float) raises -> Integer:
    """`ceil` of a Float, exactly."""
    return _float_ceil(value)


def trunc(value: Integer) raises -> Integer:
    """`trunc`, choosing the family from the argument: round toward zero, to an
    Integer (numpy's `trunc`). An Integer is its own trunc; to pass the
    function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.trunc`.

    Args:
        value: An Integer, Rational or Float.

    Returns:
        The Integer.

    Raises:
        For an infinite or NaN Float.
    """
    return _integer_trunc(value)


def trunc(value: Rational, *__disambiguate: NoneType) raises -> Integer:
    """`trunc` of a Rational, exactly."""
    return _rational_trunc(value)


def trunc(value: Float) raises -> Integer:
    """`trunc` of a Float, exactly."""
    return _float_trunc(value)


def round(value: Integer) raises -> Integer:
    """`round`, choosing the family from the argument: round to nearest, a half to the even neighbour, to an
    Integer (numpy's `round`). An Integer is its own round; to pass the
    function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.round`.

    Args:
        value: An Integer, Rational or Float.

    Returns:
        The Integer.

    Raises:
        For an infinite or NaN Float.
    """
    return _integer_round(value)


def round(value: Rational, *__disambiguate: NoneType) raises -> Integer:
    """`round` of a Rational, exactly."""
    return _rational_round(value)


def round(value: Float) raises -> Integer:
    """`round` of a Float, exactly."""
    return _float_round(value)


def maximum(left: Integer, right: Integer) raises -> Integer:
    """The larger of two numbers, choosing the family from the arguments (numpy's `maximum`).

    Two exact operands give an exact Integer or Rational, a `context=` or a
    Float operand gives a Float compared exactly and rounded once, and a Ball
    gives the enclosing ball. A NaN Float propagates. To pass the function to
    `vmap`, name a family's declaration, such as `apn_mojo.float.maximum`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        The chosen operand; the first when they are equal.

    Raises:
        Only on a checked size error.
    """
    return _integer_maximum(left, right)


def maximum(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The larger of two numbers of two exact Rationals."""
    return _rational_maximum(left, right)


def maximum(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The larger of two numbers, compared exactly and rounded once."""
    return _float_maximum(left, right, context=context)


def maximum(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the larger of two numbers over all points of two balls."""
    return _ball_maximum(left, right, context=context)


def maximum(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the larger of two numbers for an exact right operand."""
    return _ball_maximum(left, right, context=context)


def maximum(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the larger of two numbers for an exact left operand."""
    return _ball_maximum(left, right, context=context)


def minimum(left: Integer, right: Integer) raises -> Integer:
    """The smaller of two numbers, choosing the family from the arguments (numpy's `minimum`).

    Two exact operands give an exact Integer or Rational, a `context=` or a
    Float operand gives a Float compared exactly and rounded once, and a Ball
    gives the enclosing ball. A NaN Float propagates. To pass the function to
    `vmap`, name a family's declaration, such as `apn_mojo.float.minimum`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        The chosen operand; the first when they are equal.

    Raises:
        Only on a checked size error.
    """
    return _integer_minimum(left, right)


def minimum(left: Rational, right: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The smaller of two numbers of two exact Rationals."""
    return _rational_minimum(left, right)


def minimum(
    left: _FloatArgument,
    right: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """The smaller of two numbers, compared exactly and rounded once."""
    return _float_minimum(left, right, context=context)


def minimum(left: Ball, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the smaller of two numbers over all points of two balls."""
    return _ball_minimum(left, right, context=context)


def minimum(left: Ball, right: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the smaller of two numbers for an exact right operand."""
    return _ball_minimum(left, right, context=context)


def minimum(left: _BallArgument, right: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the smaller of two numbers for an exact left operand."""
    return _ball_minimum(left, right, context=context)


def clip(value: Integer, a_min: Integer, a_max: Integer) raises -> Integer:
    """`value` limited to `[a_min, a_max]`, `minimum(maximum(value, a_min),
    a_max)`, choosing the family from the arguments (numpy's `clip`).

    Exact operands give an exact result, a `context=` or a Float operand a
    Float chosen exactly and rounded once, and a Ball value the enclosing
    ball. To pass the function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.clip`.

    Args:
        value: The operand.
        a_min: The lower limit.
        a_max: The upper limit.

    Returns:
        The limited value; `a_max` when `a_min > a_max`.

    Raises:
        Only on a checked size error.
    """
    return _integer_clip(value, a_min, a_max)


def clip(value: Rational, a_min: Rational, a_max: Rational, *__disambiguate: NoneType) raises -> Rational:
    """`value` limited to `[a_min, a_max]`, exactly."""
    return _rational_clip(value, a_min, a_max)


def clip(
    value: _FloatArgument,
    a_min: _FloatArgument,
    a_max: _FloatArgument,
    *__disambiguate: NoneType,
    context: Optional[ArithmeticContext] = None,
) raises -> Float:
    """`value` limited to `[a_min, a_max]`, chosen exactly and rounded once."""
    return _float_clip(value, a_min, a_max, context=context)


def clip(value: Ball, a_min: _BallArgument, a_max: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `value` limited to `[a_min, a_max]`."""
    return _ball_clip(value, a_min, a_max, context=context)


def reciprocal(value: Integer) raises -> Rational:
    """`1 / value`, choosing the family from the argument (numpy's
    `reciprocal`).

    Exact operands give an exact Rational (or ExactComplex), a `context=` or a
    Float operand a Float rounded once, and a Complex or ball its own family.
    To pass the function to `vmap`, name a family's declaration, such as
    `apn_mojo.float.reciprocal`.

    Args:
        value: The nonzero operand.

    Returns:
        The exact reciprocal.

    Raises:
        When the operand is exactly zero.
    """
    return _integer_reciprocal(value)


def reciprocal(value: Rational, *__disambiguate: NoneType) raises -> Rational:
    """The exact reciprocal of a Rational."""
    return _rational_reciprocal(value)


def reciprocal(value: _FloatArgument, *__disambiguate: NoneType, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Float `1 / value`, rounded once."""
    return _float_reciprocal(value, context=context)


def reciprocal(value: Complex, *, context: _ComplexContextArgument = _ComplexContextArgument()) raises -> Complex:
    """The Complex `1 / value`, each component rounded once."""
    return _complex_reciprocal(value, context=context)


def reciprocal(value: ExactComplex) raises -> ExactComplex:
    """The exact reciprocal of an ExactComplex."""
    return _exact_complex_reciprocal(value)


def reciprocal(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `1 / value`."""
    return _ball_reciprocal(value, context=context)


def reciprocal(value: ComplexBall, *, context: Optional[BallContext] = None) raises -> ComplexBall:
    """The complex ball of `1 / value`."""
    return _complex_ball_reciprocal(value, context=context)


def conjugate(value: Complex) raises -> Complex:
    """The conjugate, choosing the family from the argument (numpy's
    `conjugate`).

    Args:
        value: A Complex, ExactComplex or ComplexBall.

    Returns:
        `real - imag i`, exactly.

    Raises:
        Only on a checked size error.
    """
    return _complex_conjugate(value)


def conjugate(value: ExactComplex) raises -> ExactComplex:
    """The conjugate of an ExactComplex."""
    return _exact_complex_conjugate(value)


def conjugate(value: ComplexBall) raises -> ComplexBall:
    """The conjugate of a complex ball."""
    return _complex_ball_conjugate(value)


def real(value: Complex) raises -> Float:
    """The real part, choosing the family from the argument (numpy's `real`).

    Args:
        value: A Complex, ExactComplex or ComplexBall.

    Returns:
        The real part, exactly.

    Raises:
        Only on a checked size error.
    """
    return _complex_real(value)


def real(value: ExactComplex) raises -> Rational:
    """The real part of an ExactComplex."""
    return _exact_complex_real(value)


def real(value: ComplexBall) raises -> Ball:
    """The real ball of a complex ball."""
    return _complex_ball_real(value)


def imag(value: Complex) raises -> Float:
    """The imaginary part, choosing the family from the argument (numpy's
    `imag`).

    Args:
        value: A Complex, ExactComplex or ComplexBall.

    Returns:
        The imaginary part, exactly.

    Raises:
        Only on a checked size error.
    """
    return _complex_imag(value)


def imag(value: ExactComplex) raises -> Rational:
    """The imaginary part of an ExactComplex."""
    return _exact_complex_imag(value)


def imag(value: ComplexBall) raises -> Ball:
    """The imaginary ball of a complex ball."""
    return _complex_ball_imag(value)


def comb(N: Integer, k: Integer, *, repetition: Bool = False) raises -> Integer:
    """The number of combinations of `N` things taken `k` at a time, exactly
    (scipy's `comb` with `exact=True`).

    With `repetition`, the number of multisets of `k` things from `N` kinds,
    `comb(N + k - 1, k)`. As in scipy, the result is 0 when `k > N`, `N < 0`
    or `k < 0`. To pass the function to `vmap`, name the family declaration
    `apn_mojo.integer.comb`, which has no `repetition`.

    Args:
        N: The number of things.
        k: The number taken.
        repetition: Whether a thing may be taken more than once.

    Returns:
        The exact count.

    Raises:
        When the result exceeds the addressable size.
    """
    return _integer_comb(N + k - 1, k) if repetition else _integer_comb(N, k)


def ndtr(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The standard normal distribution function, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.ndtr`; a Ball gives the enclosing ball of
    `apn_mojo.ball.ndtr`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_ndtr(value, context=context)


def ndtr(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `ndtr`, enclosing its value at every point."""
    return _ball_ndtr(value, context=context)


def log_ndtr(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The logarithm of the standard normal distribution function, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.log_ndtr`; a Ball gives the enclosing ball of
    `apn_mojo.ball.log_ndtr`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_log_ndtr(value, context=context)


def log_ndtr(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `log_ndtr`, enclosing its value at every point."""
    return _ball_log_ndtr(value, context=context)


def beta(a: _FloatArgument, b: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Euler's beta function, choosing the family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.beta`; a Ball gives the enclosing ball of
    `apn_mojo.ball.beta`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        b: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_beta(a, b, context=context)


def beta(a: Ball, b: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `beta`, enclosing its value at every pair of points."""
    return _ball_beta(a, b, context=context)


def beta(a: Ball, b: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `beta` for an exact second argument."""
    return _ball_beta(a, b, context=context)


def beta(a: _BallArgument, b: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `beta` for an exact first argument."""
    return _ball_beta(a, b, context=context)


def betaln(a: _FloatArgument, b: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The logarithm of the absolute value of the beta function, choosing the family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.betaln`; a Ball gives the enclosing ball of
    `apn_mojo.ball.betaln`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        b: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_betaln(a, b, context=context)


def betaln(a: Ball, b: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `betaln`, enclosing its value at every pair of points."""
    return _ball_betaln(a, b, context=context)


def betaln(a: Ball, b: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `betaln` for an exact second argument."""
    return _ball_betaln(a, b, context=context)


def betaln(a: _BallArgument, b: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `betaln` for an exact first argument."""
    return _ball_betaln(a, b, context=context)


def poch(z: _FloatArgument, m: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Pochhammer symbol, choosing the family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.poch`; a Ball gives the enclosing ball of
    `apn_mojo.ball.poch`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        z: A Float or exact value; exact values need a context.
        m: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_poch(z, m, context=context)


def poch(z: Ball, m: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `poch`, enclosing its value at every pair of points."""
    return _ball_poch(z, m, context=context)


def poch(z: Ball, m: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `poch` for an exact second argument."""
    return _ball_poch(z, m, context=context)


def poch(z: _BallArgument, m: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `poch` for an exact first argument."""
    return _ball_poch(z, m, context=context)


def erfinv(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The inverse error function, choosing the family from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.erfinv`; a Ball gives the enclosing ball of
    `apn_mojo.ball.erfinv`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_erfinv(value, context=context)


def erfinv(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `erfinv`, enclosing its value at every point."""
    return _ball_erfinv(value, context=context)


def ndtri(value: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The inverse standard normal distribution function, choosing the family
    from the argument.

    A Float or exact operand gives the correctly rounded Float of
    `apn_mojo.float.ndtri`; a Ball gives the enclosing ball of
    `apn_mojo.ball.ndtri`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        value: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_ndtri(value, context=context)


def ndtri(value: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `ndtri`, enclosing its value at every point."""
    return _ball_ndtri(value, context=context)


def zeta(x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Riemann zeta function, choosing the family from the argument
    (scipy's `zeta` without q): `zeta(x, 1)`.

    Args:
        x: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_zeta(x, Integer(1), context=context)


def zeta(x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of the Riemann zeta function."""
    return _ball_zeta(x, Integer(1), context=context)


def zeta(x: _FloatArgument, q: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The Hurwitz zeta function `zeta(x, q)`, choosing the family from the
    arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.zeta`; a Ball gives the enclosing ball of
    `apn_mojo.ball.zeta`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        x: A Float or exact value; exact values need a context.
        q: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_zeta(x, q, context=context)


def zeta(x: Ball, q: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `zeta`, enclosing its value at every pair of points."""
    return _ball_zeta(x, q, context=context)


def zeta(x: Ball, q: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `zeta` for an exact shift."""
    return _ball_zeta(x, q, context=context)


def zeta(x: _BallArgument, q: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `zeta` for an exact exponent."""
    return _ball_zeta(x, q, context=context)


def polygamma(n: Integer, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The polygamma function of order n, choosing the family from the
    argument.

    A Float or exact argument gives the correctly rounded Float of
    `apn_mojo.float.polygamma`; a Ball gives the enclosing ball of
    `apn_mojo.ball.polygamma`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        n: The order, a non-negative integer.
        x: A Float or exact value; exact values need a context.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for an exact value without
        a context.
    """
    return _float_polygamma(n, x, context=context)


def polygamma(n: Integer, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `polygamma`, enclosing its value at every point."""
    return _ball_polygamma(n, x, context=context)


def hyp1f1(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Kummer's confluent hypergeometric function `M(a, b, x)`, choosing the
    family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.hyp1f1`; a Ball x gives the enclosing ball of
    `apn_mojo.ball.hyp1f1`, with the parameters as balls or exact numbers. To
    pass the function to `vmap`, name a family's declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        b: A Float or exact value.
        x: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_hyp1f1(a, b, x, context=context)


def hyp1f1(a: _BallArgument, b: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `hyp1f1`, enclosing its value at every triple of points."""
    return _ball_hyp1f1(a, b, x, context=context)


def gammainc(a: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized incomplete gamma function `P(a, x)`, choosing the
    family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.gammainc`; a Ball gives the enclosing ball of
    `apn_mojo.ball.gammainc`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        x: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_gammainc(a, x, context=context)


def gammainc(a: Ball, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammainc`, enclosing its value at every pair of points."""
    return _ball_gammainc(a, x, context=context)


def gammainc(a: Ball, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammainc` for an exact argument."""
    return _ball_gammainc(a, x, context=context)


def gammainc(a: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammainc` for an exact shape."""
    return _ball_gammainc(a, x, context=context)


def gammaincc(a: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized incomplete gamma function `Q(a, x)`, choosing the
    family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.gammaincc`; a Ball gives the enclosing ball of
    `apn_mojo.ball.gammaincc`. To pass the function to `vmap`, name a family's
    declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        x: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_gammaincc(a, x, context=context)


def gammaincc(a: Ball, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammaincc`, enclosing its value at every pair of points."""
    return _ball_gammaincc(a, x, context=context)


def gammaincc(a: Ball, x: _BallArgument, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammaincc` for an exact argument."""
    return _ball_gammaincc(a, x, context=context)


def gammaincc(a: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `gammaincc` for an exact shape."""
    return _ball_gammaincc(a, x, context=context)


def hyp2f1(a: _FloatArgument, b: _FloatArgument, c: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """Gauss's hypergeometric function `F(a, b; c; x)`, choosing the family
    from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.hyp2f1`; a Ball x gives the enclosing ball of
    `apn_mojo.ball.hyp2f1`, with the parameters as balls or exact numbers. To
    pass the function to `vmap`, name a family's declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        b: A Float or exact value.
        c: A Float or exact value.
        x: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_hyp2f1(a, b, c, x, context=context)


def hyp2f1(a: _BallArgument, b: _BallArgument, c: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `hyp2f1`, enclosing its value at every point of the balls."""
    return _ball_hyp2f1(a, b, c, x, context=context)


def betainc(a: _FloatArgument, b: _FloatArgument, x: _FloatArgument, *, context: Optional[ArithmeticContext] = None) raises -> Float:
    """The regularized incomplete beta function `I_x(a, b)`, choosing the
    family from the arguments.

    Floats or exact arguments give the correctly rounded Float of
    `apn_mojo.float.betainc`; a Ball x gives the enclosing ball of
    `apn_mojo.ball.betainc`, with the shapes as balls or exact numbers. To
    pass the function to `vmap`, name a family's declaration.

    Args:
        a: A Float or exact value; exact values need a context.
        b: A Float or exact value.
        x: A Float or exact value.
        context: The output format, rounding mode, traps and budget.

    Returns:
        A Float, rounded once.

    Raises:
        On a trapped condition, past the budget, or for exact values without
        a context.
    """
    return _float_betainc(a, b, x, context=context)


def betainc(a: _BallArgument, b: _BallArgument, x: Ball, *, context: Optional[BallContext] = None) raises -> Ball:
    """The ball of `betainc`, enclosing its value at every point of the balls."""
    return _ball_betainc(a, b, x, context=context)

