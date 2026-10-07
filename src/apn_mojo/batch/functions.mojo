"""NumPy-style functions of batches: `apn_mojo.batch.exp(xs)` is `numpy.exp`.

Each function maps, through `vmap`, the family declarations that the scalar
`apn_mojo` function of the same name dispatches to, so the mathematics is the
scalar's: Integer and Rational results are exact, each Float is rounded once,
and balls enclose. Batch arguments broadcast as the arithmetic operators do,
a scalar argument is shared by every element, and the result has the
broadcast shape. Each function needs a batch argument; the scalar functions
stay at the package root.

A function takes the families that declare it. Exact elements give Floats
under a `context=`; the arithmetic functions keep them exact without one, as
the scalar functions do.
"""

from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..ball.value import Ball, _BallArgument
from ..float._arithmetic import _FloatArgument
from ..complex_ball.value import ComplexBall
from ..float.context import ArithmeticContext
from ..ball.context import BallContext
from ..complex.context import ComplexContext, _ComplexContextArgument
from ..common._traits import _MapArgument
from ..common.functions import zeta as _root_zeta, comb as _root_comb
from ..integer.math import (
    reciprocal as _integer_reciprocal, abs as _integer_abs,
    floor as _integer_floor, ceil as _integer_ceil, trunc as _integer_trunc, round as _integer_round,
    maximum as _integer_maximum, minimum as _integer_minimum, clip as _integer_clip, comb as _integer_comb,
)
from ..rational.math import (
    reciprocal as _rational_reciprocal, abs as _rational_abs,
    floor as _rational_floor, ceil as _rational_ceil, trunc as _rational_trunc, round as _rational_round,
    maximum as _rational_maximum, minimum as _rational_minimum, clip as _rational_clip,
)
from ..float.math import (
    add as _float_add, subtract as _float_subtract, multiply as _float_multiply, divide as _float_divide,
    reciprocal as _float_reciprocal, abs as _float_abs, sqrt as _float_sqrt, pow_int as _float_pow_int,
    floor as _float_floor, ceil as _float_ceil, trunc as _float_trunc, round as _float_round,
    maximum as _float_maximum, minimum as _float_minimum, clip as _float_clip,
)
from ..complex.math import (
    add as _complex_add, subtract as _complex_subtract, multiply as _complex_multiply,
    divide as _complex_divide, reciprocal as _complex_reciprocal, abs as _complex_abs,
    sqrt as _complex_sqrt, pow_int as _complex_pow_int, conjugate as _complex_conjugate,
    real as _complex_real, imag as _complex_imag,
)
from ..ball.math import (
    add as _ball_add, subtract as _ball_subtract, multiply as _ball_multiply, divide as _ball_divide,
    reciprocal as _ball_reciprocal, abs as _ball_abs, sqrt as _ball_sqrt, pow_int as _ball_pow_int,
    maximum as _ball_maximum, minimum as _ball_minimum, clip as _ball_clip,
)
from ..complex_ball.math import (
    add as _complex_ball_add, subtract as _complex_ball_subtract, multiply as _complex_ball_multiply,
    divide as _complex_ball_divide, reciprocal as _complex_ball_reciprocal, abs as _complex_ball_abs,
    pow_int as _complex_ball_pow_int, conjugate as _complex_ball_conjugate, real as _complex_ball_real,
    imag as _complex_ball_imag, angle as _complex_ball_angle,
)
from ..float.elementary import (
    exp as _float_exp, expm1 as _float_expm1, exp2 as _float_exp2, log as _float_log,
    log1p as _float_log1p, log2 as _float_log2, log10 as _float_log10, sin as _float_sin, cos as _float_cos,
    tan as _float_tan, atan as _float_atan, asin as _float_asin, acos as _float_acos, sinh as _float_sinh,
    cosh as _float_cosh, tanh as _float_tanh, asinh as _float_asinh, acosh as _float_acosh,
    atanh as _float_atanh, sin_cos as _float_sin_cos, atan2 as _float_atan2, pow as _float_pow,
    rootn as _float_rootn,
)
from ..ball.elementary import (
    exp as _ball_exp, expm1 as _ball_expm1, exp2 as _ball_exp2, log as _ball_log, log1p as _ball_log1p,
    log2 as _ball_log2, log10 as _ball_log10, sin as _ball_sin, cos as _ball_cos, tan as _ball_tan,
    atan as _ball_atan, asin as _ball_asin, acos as _ball_acos, sinh as _ball_sinh, cosh as _ball_cosh,
    tanh as _ball_tanh, asinh as _ball_asinh, acosh as _ball_acosh, atanh as _ball_atanh,
    sin_cos as _ball_sin_cos, atan2 as _ball_atan2, pow as _ball_pow, rootn as _ball_rootn,
)
from ..complex.elementary import (
    exp as _complex_exp, log as _complex_log, sin as _complex_sin, cos as _complex_cos, tan as _complex_tan,
    sinh as _complex_sinh, cosh as _complex_cosh, tanh as _complex_tanh, asin as _complex_asin,
    acos as _complex_acos, atan as _complex_atan, asinh as _complex_asinh, acosh as _complex_acosh,
    atanh as _complex_atanh, pow as _complex_pow, angle as _complex_angle,
)
from ..complex_ball.elementary import (
    exp as _complex_ball_exp, log as _complex_ball_log, sin as _complex_ball_sin, cos as _complex_ball_cos,
    tan as _complex_ball_tan, sinh as _complex_ball_sinh, cosh as _complex_ball_cosh,
    tanh as _complex_ball_tanh, asin as _complex_ball_asin, acos as _complex_ball_acos,
    atan as _complex_ball_atan, asinh as _complex_ball_asinh, acosh as _complex_ball_acosh,
    atanh as _complex_ball_atanh, sqrt as _complex_ball_sqrt, pow as _complex_ball_pow,
)
from ..float.special import (
    gamma as _float_gamma, gammaln as _float_gammaln, digamma as _float_digamma, erf as _float_erf,
    erfc as _float_erfc, erfi as _float_erfi, expi as _float_expi, sici as _float_sici,
    shichi as _float_shichi, fresnel as _float_fresnel, lambertw as _float_lambertw, ndtr as _float_ndtr,
    log_ndtr as _float_log_ndtr, beta as _float_beta, betaln as _float_betaln, poch as _float_poch,
    erfinv as _float_erfinv, ndtri as _float_ndtri, zeta as _float_zeta, polygamma as _float_polygamma,
    hyp1f1 as _float_hyp1f1, gammainc as _float_gammainc, gammaincc as _float_gammaincc,
    hyp2f1 as _float_hyp2f1, betainc as _float_betainc,
)
from ..ball.special import (
    gamma as _ball_gamma, gammaln as _ball_gammaln, digamma as _ball_digamma, erf as _ball_erf,
    erfc as _ball_erfc, erfi as _ball_erfi, expi as _ball_expi, sici as _ball_sici, shichi as _ball_shichi,
    fresnel as _ball_fresnel, lambertw as _ball_lambertw, ndtr as _ball_ndtr, log_ndtr as _ball_log_ndtr,
    beta as _ball_beta, betaln as _ball_betaln, poch as _ball_poch, erfinv as _ball_erfinv,
    ndtri as _ball_ndtri, zeta as _ball_zeta, polygamma as _ball_polygamma, hyp1f1 as _ball_hyp1f1,
    gammainc as _ball_gammainc, gammaincc as _ball_gammaincc, hyp2f1 as _ball_hyp2f1,
    betainc as _ball_betainc,
)
from .value import Batch, _BatchShape, _ElementOf, _Sum, _Quotient, _batch_from
from .mask import Mask
from .mapping import vmap, _AxisLevels, _Axes, _Context, _Outputs, _WholeLiteral
from .lift import _input_layout, _broadcast_shape, _no_context
from .reductions import _uniform_reduction_context, _complex_reduction_context


# ---------------------------------------------------------------- families


# The family of an argument: a batch's elements, or a scalar's own family.
comptime _FamilyOf[V: ImplicitlyCopyable & Deinitable] = Integer if conforms_to(V, _WholeLiteral) else _ElementOf[V]
comptime _Batched[V: AnyType] = conforms_to(V, _BatchShape)
comptime _Exact[T: ImplicitlyCopyable & Deinitable] = T == Integer or T == Rational
comptime _Real[T: ImplicitlyCopyable & Deinitable] = _Exact[T] or T == Float
comptime _Balls[T: ImplicitlyCopyable & Deinitable] = T == Ball or T == ComplexBall
# What a function of these elements returns: balls and Complex numbers keep
# their family, and exact values give correctly rounded Floats.
comptime _Mapped[T: ImplicitlyCopyable & Deinitable] = (
    Ball if T == Ball else ComplexBall if T == ComplexBall else Complex if T == Complex else Float
)
# The context that family's functions take.
comptime _ContextOf[T: ImplicitlyCopyable & Deinitable] = (
    Optional[BallContext] if _Balls[T] else _ComplexContextArgument if T == Complex else Optional[ArithmeticContext]
)
# The mapped family of two values together, as the scalar adapters join them.
comptime _Joint[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable] = (
    ComplexBall if A == ComplexBall or B == ComplexBall or (
        (A == Ball or B == Ball) and (A == Complex or B == Complex)
    ) else Ball if A == Ball or B == Ball else Complex if A == Complex or B == Complex else Float
)
comptime _Pair[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable] = _Joint[
    _FamilyOf[A], _FamilyOf[B]
]
comptime _Triple[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable] = (
    _Joint[_Pair[A, B], _FamilyOf[C]]
)
comptime _Quadruple[
    A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, D: ImplicitlyCopyable & Deinitable,
] = _Joint[_Pair[A, B], _Pair[C, D]]
# Arithmetic keeps exact families exact: I is the Integer form's result,
# Rational for division. _ExactPair joins families already found, since the
# compiler expands each use of a parameter: _FamilyOf of a joined family would
# repeat the whole join at every use.
comptime _ExactPair[X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable, I: ImplicitlyCopyable & Deinitable] = (
    _Joint[X, Y] if _Balls[X] or _Balls[Y] else _Quotient[X, Y] if I == Rational else _Sum[X, Y]
)
comptime _Arithmetic[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, I: ImplicitlyCopyable & Deinitable] = (
    _ExactPair[_FamilyOf[A], _FamilyOf[B], I]
)
comptime _BallContextType[C: AnyType] = C == BallContext or C == Optional[BallContext]
comptime _ComplexContextType[C: AnyType] = C == ComplexContext or C == Optional[ComplexContext] or C == _ComplexContextArgument
# The family of values F under an explicit context of type C.
comptime _WithContext[F: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable] = (
    (ComplexBall if F == Complex or F == ComplexBall else Ball) if _BallContextType[C] else
    Complex if F == Complex or _ComplexContextType[C] else Float
)
# Magnitudes and angles: Complex gives Float, balls give Ball, others keep their family.
comptime _Magnitude[T: ImplicitlyCopyable & Deinitable] = Ball if _Balls[T] else Float if T == Complex else T


struct _NoContext(Defaultable, ImplicitlyCopyable):
    """The context of a function that takes none for these elements: there is none to give."""

    def __init__(out self):
        pass


comptime _MagnitudeContext[T: ImplicitlyCopyable & Deinitable] = (
    Optional[BallContext] if _Balls[T] else Optional[ArithmeticContext] if T == Complex else _NoContext
)


def _ball_context[C: ImplicitlyCopyable](context: C) -> Optional[BallContext]:
    comptime if C == BallContext:
        return rebind[BallContext](context)
    else:
        comptime assert C == Optional[BallContext], "Ball values take a BallContext."
        return rebind[Optional[BallContext]](context)


# ------------------------------------------------------------- the mapping


struct _Elementwise(Movable):
    """How a call maps: the result's broadcast shape, and the shape vmap maps,
    which has an axis even for a rank-zero result."""

    var shape: List[Int]
    var mapped: List[Int]

    def __init__(out self, var shape: List[Int]):
        self.mapped = shape.copy() if len(shape) else [1]
        self.shape = shape^

    def levels(self, arguments: List[Bool]) -> _AxisLevels:
        """vmap levels mapping axis 0 of each batch argument at every level, so
        the function meets single elements; scalar arguments are shared."""
        var levels = _AxisLevels()
        levels.levels = List[_Axes](capacity=len(self.mapped))
        for _ in range(len(self.mapped)):
            var axes = _Axes()
            axes.uniform = False
            axes.axis = None
            for mapped in arguments:
                axes.values.append(Optional[Int](0) if mapped else None)
            levels.levels.append(axes^)
        levels.listed = True
        return levels^

    def spread[V: _MapArgument & ImplicitlyCopyable & Deinitable](self, value: V) raises -> V:
        """A batch viewed at the mapped shape, with stride zero along the axes
        it lacks or has once; a scalar stays as it is."""
        comptime if _Batched[V]:
            if value.shape() == self.mapped:
                return value
            return rebind_var[V](_batch_from[_ElementOf[V]](value._array().broadcast_to(self.mapped)))
        else:
            return value

    def result[R: ImplicitlyCopyable & Deinitable](self, var result: Batch[R]) raises -> Batch[R]:
        return result^ if len(self.shape) else result.reshape(self.shape)


def _shape[V: _MapArgument & ImplicitlyCopyable & Deinitable](value: V) -> List[Int]:
    return _input_layout(value).shape.copy()


def _map1[
    V: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable, //,
    function: def(V, /, *, context: C) raises thin -> R,
](values: Batch[E], context: C) raises -> Batch[R] where _Context[C]:
    var plan = _Elementwise(values.shape())
    var result = vmap[function](in_axes=plan.levels([True]))(plan.spread(values), context=context)
    return plan.result(rebind_var[Batch[R]](result^))


def _map1_pair[
    V: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: type_of(Tuple), E: ImplicitlyCopyable & Deinitable, //,
    function: def(V, /, *, context: C) raises thin -> R, X: ImplicitlyCopyable & Deinitable,
](values: Batch[E], context: C) raises -> Tuple[Batch[X], Batch[X]] where _Context[C]:
    """A function with two results of family X."""
    var plan = _Elementwise(values.shape())
    var first, second = rebind_var[Tuple[Batch[X], Batch[X]]](
        vmap[function](in_axes=plan.levels([True]))(plan.spread(values), context=context)
    )
    return (plan.result(first^), plan.result(second^))


def _map2[
    V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable,
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //,
    function: def(V, W, /, *, context: C) raises thin -> R,
](a: A, b: B, context: C) raises -> Batch[R] where _Context[C]:
    comptime assert _Batched[A] or _Batched[B], (
        "apn_mojo.batch functions need a batch argument; use the apn_mojo function of the same name for scalars."
    )
    var plan = _Elementwise(_broadcast_shape(_shape(a), _shape(b)))
    var result = vmap[function](in_axes=plan.levels([_Batched[A], _Batched[B]]))(
        plan.spread(a), plan.spread(b), context=context,
    )
    return plan.result(rebind_var[Batch[R]](result^))


def _map3[
    V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable,
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, //,
    function: def(V, W, X, /, *, context: C) raises thin -> R,
](a: A, b: B, d: D, context: C) raises -> Batch[R] where _Context[C]:
    comptime assert _Batched[A] or _Batched[B] or _Batched[D], (
        "apn_mojo.batch functions need a batch argument; use the apn_mojo function of the same name for scalars."
    )
    var plan = _Elementwise(_broadcast_shape(_broadcast_shape(_shape(a), _shape(b)), _shape(d)))
    var result = vmap[function](in_axes=plan.levels([_Batched[A], _Batched[B], _Batched[D]]))(
        plan.spread(a), plan.spread(b), plan.spread(d), context=context,
    )
    return plan.result(rebind_var[Batch[R]](result^))


def _map4[
    V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    Y: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: ImplicitlyCopyable & Deinitable, A: _MapArgument & ImplicitlyCopyable & Deinitable,
    B: _MapArgument & ImplicitlyCopyable & Deinitable, D: _MapArgument & ImplicitlyCopyable & Deinitable,
    E: _MapArgument & ImplicitlyCopyable & Deinitable, //,
    function: def(V, W, X, Y, /, *, context: C) raises thin -> R,
](a: A, b: B, d: D, e: E, context: C) raises -> Batch[R] where _Context[C]:
    comptime assert _Batched[A] or _Batched[B] or _Batched[D] or _Batched[E], (
        "apn_mojo.batch functions need a batch argument; use the apn_mojo function of the same name for scalars."
    )
    var plan = _Elementwise(_broadcast_shape(
        _broadcast_shape(_shape(a), _shape(b)), _broadcast_shape(_shape(d), _shape(e)),
    ))
    var result = vmap[function](in_axes=plan.levels([_Batched[A], _Batched[B], _Batched[D], _Batched[E]]))(
        plan.spread(a), plan.spread(b), plan.spread(d), plan.spread(e), context=context,
    )
    return plan.result(rebind_var[Batch[R]](result^))


# Exact family declarations take no context; these give them the mapped form.


def _bare1[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //, function: def(T) raises thin -> R,
](x: T, /, *, context: _ComplexContextArgument) raises -> R:
    return function(x)


def _bare3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    R: ImplicitlyCopyable & Deinitable, //, function: def(T, U, W) raises thin -> R,
](x: T, y: U, z: W, /, *, context: _ComplexContextArgument) raises -> R:
    return function(x, y, z)


# ----------------------------------------------------------- dispatch by shape


# Each family's declaration is a parameter, so one body serves every function
# of the same shape; the families a function lacks have shapes of their own.


def _unary[
    T: ImplicitlyCopyable & Deinitable, RV: ImplicitlyCopyable & Deinitable, BV: ImplicitlyCopyable & Deinitable,
    CV: ImplicitlyCopyable & Deinitable, KV: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, /, *, context: Optional[BallContext]) raises thin -> Ball,
    complex: def(CV, /, *, context: _ComplexContextArgument) raises thin -> Complex,
    complex_ball: def(KV, /, *, context: Optional[BallContext]) raises thin -> ComplexBall,
](values: Batch[T], context: _ContextOf[T]) raises -> Batch[_Mapped[T]]:
    comptime if T == ComplexBall:
        return rebind_var[Batch[_Mapped[T]]](_map1[complex_ball](values, rebind[Optional[BallContext]](context)))
    elif T == Complex:
        return rebind_var[Batch[_Mapped[T]]](_map1[complex](values, rebind[_ComplexContextArgument](context)))
    else:
        return _real_unary[real, ball](values, context)


def _real_unary[
    T: ImplicitlyCopyable & Deinitable, RV: ImplicitlyCopyable & Deinitable, BV: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, /, *, context: Optional[BallContext]) raises thin -> Ball,
](values: Batch[T], context: _ContextOf[T]) raises -> Batch[_Mapped[T]]:
    comptime if T == Ball:
        return rebind_var[Batch[_Mapped[T]]](_map1[ball](values, rebind[Optional[BallContext]](context)))
    else:
        comptime assert _Real[T], "This function takes Integer, Rational, Float or Ball elements."
        return rebind_var[Batch[_Mapped[T]]](_map1[real](values, rebind[Optional[ArithmeticContext]](context)))


def _real_unary_pair[
    T: ImplicitlyCopyable & Deinitable, RV: ImplicitlyCopyable & Deinitable, BV: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, /, *, context: Optional[ArithmeticContext]) raises thin -> Tuple[Float, Float],
    ball: def(BV, /, *, context: Optional[BallContext]) raises thin -> Tuple[Ball, Ball],
](values: Batch[T], context: _ContextOf[T]) raises -> Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]:
    comptime Pair = Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]
    comptime if T == Ball:
        return rebind_var[Pair](_map1_pair[ball, Ball](values, rebind[Optional[BallContext]](context)))
    else:
        comptime assert _Real[T], "This function takes Integer, Rational, Float or Ball elements."
        return rebind_var[Pair](_map1_pair[real, Float](values, rebind[Optional[ArithmeticContext]](context)))


def _binary[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable,
    CV: ImplicitlyCopyable & Deinitable, CW: ImplicitlyCopyable & Deinitable,
    KV: ImplicitlyCopyable & Deinitable, KW: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
    complex: def(CV, CW, /, *, context: _ComplexContextArgument) raises thin -> Complex,
    complex_ball: def(KV, KW, /, *, context: Optional[BallContext]) raises thin -> ComplexBall,
](a: A, b: B, context: _ContextOf[_Pair[A, B]]) raises -> Batch[_Pair[A, B]]:
    comptime if _Pair[A, B] == ComplexBall:
        return rebind_var[Batch[_Pair[A, B]]](_map2[complex_ball](a, b, rebind[Optional[BallContext]](context)))
    elif _Pair[A, B] == Complex:
        return rebind_var[Batch[_Pair[A, B]]](_map2[complex](a, b, rebind[_ComplexContextArgument](context)))
    else:
        return _real_binary[real, ball](a, b, context)


def _real_binary[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
](a: A, b: B, context: _ContextOf[_Pair[A, B]]) raises -> Batch[_Pair[A, B]]:
    comptime if _Pair[A, B] == Ball:
        return rebind_var[Batch[_Pair[A, B]]](_map2[ball](a, b, rebind[Optional[BallContext]](context)))
    else:
        comptime assert _Pair[A, B] == Float, "This function takes Integer, Rational, Float or Ball values."
        return rebind_var[Batch[_Pair[A, B]]](_map2[real](a, b, rebind[Optional[ArithmeticContext]](context)))


def _real_ternary[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable,
    RU: ImplicitlyCopyable & Deinitable, RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    BU: ImplicitlyCopyable & Deinitable, BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable, //,
    real: def(RU, RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BU, BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
](a: A, b: B, d: D, context: _ContextOf[_Triple[A, B, D]]) raises -> Batch[_Triple[A, B, D]]:
    comptime F = _Triple[A, B, D]
    comptime if F == Ball:
        return rebind_var[Batch[F]](_map3[ball](a, b, d, rebind[Optional[BallContext]](context)))
    else:
        comptime assert F == Float, "This function takes Integer, Rational, Float or Ball values."
        return rebind_var[Batch[F]](_map3[real](a, b, d, rebind[Optional[ArithmeticContext]](context)))


def _real_quaternary[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, E: _MapArgument & ImplicitlyCopyable & Deinitable,
    RU: ImplicitlyCopyable & Deinitable, RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    RX: ImplicitlyCopyable & Deinitable, BU: ImplicitlyCopyable & Deinitable, BV: ImplicitlyCopyable & Deinitable,
    BW: ImplicitlyCopyable & Deinitable, BX: ImplicitlyCopyable & Deinitable, //,
    real: def(RU, RV, RW, RX, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BU, BV, BW, BX, /, *, context: Optional[BallContext]) raises thin -> Ball,
](a: A, b: B, d: D, e: E, context: _ContextOf[_Quadruple[A, B, D, E]]) raises -> Batch[_Quadruple[A, B, D, E]]:
    comptime F = _Quadruple[A, B, D, E]
    comptime if F == Ball:
        return rebind_var[Batch[F]](_map4[ball](a, b, d, e, rebind[Optional[BallContext]](context)))
    else:
        comptime assert F == Float, "This function takes Integer, Rational, Float or Ball values."
        return rebind_var[Batch[F]](_map4[real](a, b, d, e, rebind[Optional[ArithmeticContext]](context)))


comptime _ADD = 0
comptime _SUBTRACT = 1
comptime _MULTIPLY = 2
comptime _DIVIDE = 3


def _operator[
    operation: Int, R: ImplicitlyCopyable & Deinitable,
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
](a: A, b: B) raises -> Batch[R]:
    """A batch operator, which applies the scalar arithmetic on the batch
    executor, several times faster than mapping the scalar function."""
    comptime assert _Batched[A] or _Batched[B], (
        "apn_mojo.batch functions need a batch argument; use the apn_mojo function of the same name for scalars."
    )
    # The generic operators take numbers; an integer literal of any width becomes an Integer first.
    comptime if conforms_to(A, _WholeLiteral):
        return _operator[operation, R](a._exact(), b)
    elif conforms_to(B, _WholeLiteral):
        return _operator[operation, R](a, b._exact())
    var left = a
    var right = b
    comptime if _Batched[A] and _Batched[B]:
        # Operators keep two vectors to equal lengths; NumPy broadcasts a length of one.
        var shapes = (_shape(a), _shape(b))
        if len(shapes[0]) == 1 and len(shapes[1]) == 1 and shapes[0] != shapes[1]:
            var plan = _Elementwise(_broadcast_shape(shapes[0], shapes[1]))
            left = plan.spread(a)
            right = plan.spread(b)
    comptime if _Batched[A]:
        ref x = rebind[Batch[_ElementOf[A]]](left)
        comptime if operation == _ADD:
            return rebind_var[Batch[R]](x + right)
        elif operation == _SUBTRACT:
            return rebind_var[Batch[R]](x - right)
        elif operation == _MULTIPLY:
            return rebind_var[Batch[R]](x * right)
        else:
            return rebind_var[Batch[R]](x / right)
    else:
        ref y = rebind[Batch[_ElementOf[B]]](right)
        comptime if operation == _ADD:
            return rebind_var[Batch[R]](y.__radd__(left))
        elif operation == _SUBTRACT:
            return rebind_var[Batch[R]](y.__rsub__(left))
        elif operation == _MULTIPLY:
            return rebind_var[Batch[R]](y.__rmul__(left))
        else:
            return rebind_var[Batch[R]](y.__rtruediv__(left))


def _arithmetic[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable,
    KV: ImplicitlyCopyable & Deinitable, KW: ImplicitlyCopyable & Deinitable, //,
    I: ImplicitlyCopyable & Deinitable, operation: Int,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
    complex_ball: def(KV, KW, /, *, context: Optional[BallContext]) raises thin -> ComplexBall,
](a: A, b: B) raises -> Batch[_Arithmetic[A, B, I]]:
    """Arithmetic without a context, exact for exact families: the batch
    operators, which have no ball forms, or the mapped ball functions."""
    comptime R = _Arithmetic[A, B, I]
    comptime if R == ComplexBall:
        return rebind_var[Batch[R]](_map2[complex_ball](a, b, Optional[BallContext]()))
    elif R == Ball:
        return rebind_var[Batch[R]](_map2[ball](a, b, Optional[BallContext]()))
    else:
        return _operator[operation, R](a, b)


def _rounded_arithmetic[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    CV: ImplicitlyCopyable & Deinitable, CW: ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable,
    KV: ImplicitlyCopyable & Deinitable, KW: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    complex: def(CV, CW, /, *, context: _ComplexContextArgument) raises thin -> Complex,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
    complex_ball: def(KV, KW, /, *, context: Optional[BallContext]) raises thin -> ComplexBall,
](a: A, b: B, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """Arithmetic rounded with the context's family: Float, Complex or a ball."""
    comptime F = _Arithmetic[A, B, Integer]
    comptime R = _WithContext[F, C]
    comptime assert not _Balls[F] or _BallContextType[C], "Ball values take a BallContext."
    comptime if R == ComplexBall:
        return rebind_var[Batch[R]](_map2[complex_ball](a, b, _ball_context(context)))
    elif R == Ball:
        return rebind_var[Batch[R]](_map2[ball](a, b, _ball_context(context)))
    elif R == Complex:
        return rebind_var[Batch[R]](_map2[complex](a, b, _complex_reduction_context(context)))
    else:
        return rebind_var[Batch[R]](_map2[real](a, b, _uniform_reduction_context(context)))


def _ordered[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable, //,
    integer: def(Integer, Integer) raises thin -> Integer,
    rational: def(Rational, Rational) raises thin -> Rational,
    real: def(RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
](a: A, b: B) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """maximum and minimum: exact without a context; real families only."""
    comptime R = _Arithmetic[A, B, Integer]
    comptime assert R != Complex and R != ComplexBall, "Complex numbers have no order."
    comptime if R == Integer:
        return rebind_var[Batch[R]](_map2[_no_context[integer]](a, b, _ComplexContextArgument()))
    elif R == Rational:
        return rebind_var[Batch[R]](_map2[_no_context[rational]](a, b, _ComplexContextArgument()))
    elif R == Ball:
        return rebind_var[Batch[R]](_map2[ball](a, b, Optional[BallContext]()))
    else:
        return rebind_var[Batch[R]](_map2[real](a, b, Optional[ArithmeticContext]()))


def _ordered_with[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, RV: ImplicitlyCopyable & Deinitable, RW: ImplicitlyCopyable & Deinitable,
    BV: ImplicitlyCopyable & Deinitable, BW: ImplicitlyCopyable & Deinitable, //,
    real: def(RV, RW, /, *, context: Optional[ArithmeticContext]) raises thin -> Float,
    ball: def(BV, BW, /, *, context: Optional[BallContext]) raises thin -> Ball,
](a: A, b: B, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    comptime F = _Arithmetic[A, B, Integer]
    comptime R = _WithContext[F, C]
    comptime assert R == Float or R == Ball, "maximum, minimum and clip take real values with an ArithmeticContext or a BallContext."
    comptime assert not _Balls[F] or _BallContextType[C], "Ball values take a BallContext."
    comptime if R == Ball:
        return rebind_var[Batch[R]](_map2[ball](a, b, _ball_context(context)))
    else:
        return rebind_var[Batch[R]](_map2[real](a, b, _uniform_reduction_context(context)))


# ------------------------------------------------------------- arithmetic


def add[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """`a + b` elementwise, like `numpy.add`.

    Integers and Rationals stay exact; a Float value gives Floats rounded once
    to the operands' format, a Complex value Complex numbers, a ball balls.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _arithmetic[Integer, _ADD, _ball_add, _complex_ball_add](a, b)


def add[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """`a + b` elementwise, each sum rounded once with `context`: an
    ArithmeticContext gives Floats (Complex for complex values), a
    ComplexContext Complex numbers, a BallContext balls."""
    return _rounded_arithmetic[_float_add, _complex_add, _ball_add, _complex_ball_add](a, b, context)


def subtract[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """`a - b` elementwise, like `numpy.subtract`; families as for `add`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _arithmetic[Integer, _SUBTRACT, _ball_subtract, _complex_ball_subtract](a, b)


def subtract[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """`a - b` elementwise, each difference rounded once with `context`, as for `add`."""
    return _rounded_arithmetic[_float_subtract, _complex_subtract, _ball_subtract, _complex_ball_subtract](
        a, b, context
    )


def multiply[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """`a * b` elementwise, like `numpy.multiply`; families as for `add`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _arithmetic[Integer, _MULTIPLY, _ball_multiply, _complex_ball_multiply](a, b)


def multiply[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """`a * b` elementwise, each product rounded once with `context`, as for `add`."""
    return _rounded_arithmetic[_float_multiply, _complex_multiply, _ball_multiply, _complex_ball_multiply](
        a, b, context
    )


def divide[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Rational]]:
    """`a / b` elementwise, like `numpy.divide`: Integers give exact
    Rationals; other families as for `add`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        On division by zero of exact values, when the shapes do not broadcast,
        or as the scalar function does.
    """
    return _arithmetic[Rational, _DIVIDE, _ball_divide, _complex_ball_divide](a, b)


def divide[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """`a / b` elementwise, each quotient rounded once with `context`, as for `add`."""
    return _rounded_arithmetic[_float_divide, _complex_divide, _ball_divide, _complex_ball_divide](a, b, context)


def reciprocal[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[
    Rational if T == Integer else T
]:
    """`1 / x` elementwise, like `numpy.reciprocal`: Integers give exact
    Rationals, Rationals stay exact, the other families keep theirs.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    comptime R = Rational if T == Integer else T
    comptime if T == Integer:
        return rebind_var[Batch[R]](_map1[_bare1[_integer_reciprocal]](values, _ComplexContextArgument()))
    elif T == Rational:
        return rebind_var[Batch[R]](_map1[_bare1[_rational_reciprocal]](values, _ComplexContextArgument()))
    elif T == Float:
        return rebind_var[Batch[R]](_map1[_float_reciprocal](values, Optional[ArithmeticContext]()))
    elif T == Complex:
        return rebind_var[Batch[R]](_map1[_complex_reciprocal](values, _ComplexContextArgument()))
    elif T == Ball:
        return rebind_var[Batch[R]](_map1[_ball_reciprocal](values, Optional[BallContext]()))
    else:
        comptime assert T == ComplexBall, "reciprocal takes Integer, Rational, Float, Complex or ball elements."
        return rebind_var[Batch[R]](_map1[_complex_ball_reciprocal](values, Optional[BallContext]()))


def reciprocal[T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable, //](
    values: Batch[T], *, context: C,
) raises -> Batch[_WithContext[T, C]]:
    """`1 / x` elementwise, each rounded once with `context`, as for `add`."""
    comptime R = _WithContext[T, C]
    comptime assert not _Balls[T] or _BallContextType[C], "Ball values take a BallContext."
    comptime if R == ComplexBall:
        return rebind_var[Batch[R]](_map1[_complex_ball_reciprocal](values, _ball_context(context)))
    elif R == Ball:
        return rebind_var[Batch[R]](_map1[_ball_reciprocal](values, _ball_context(context)))
    elif R == Complex:
        return rebind_var[Batch[R]](_map1[_complex_reciprocal](values, _complex_reduction_context(context)))
    else:
        return rebind_var[Batch[R]](_map1[_float_reciprocal](values, _uniform_reduction_context(context)))


def maximum[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """The larger of each pair, like `numpy.maximum`; a NaN wins, as in numpy.

    Exact families stay exact. Real families and balls only.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _ordered[_integer_maximum, _rational_maximum, _float_maximum, _ball_maximum](a, b)


def maximum[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """The larger of each pair, rounded once with `context`."""
    return _ordered_with[_float_maximum, _ball_maximum](a, b, context)


def minimum[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B,
) raises -> Batch[_Arithmetic[A, B, Integer]]:
    """The smaller of each pair, like `numpy.minimum`; as for `maximum`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _ordered[_integer_minimum, _rational_minimum, _float_minimum, _ball_minimum](a, b)


def minimum[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable, //,
](a: A, b: B, *, context: C) raises -> Batch[_WithContext[_Arithmetic[A, B, Integer], C]]:
    """The smaller of each pair, rounded once with `context`."""
    return _ordered_with[_float_minimum, _ball_minimum](a, b, context)


comptime _Clipped[
    A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, D: ImplicitlyCopyable & Deinitable,
] = _ExactPair[_Arithmetic[A, B, Integer], _FamilyOf[D], Integer]


def clip[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, //,
](values: A, a_min: B, a_max: D) raises -> Batch[_Clipped[A, B, D]]:
    """Each value limited to `[a_min, a_max]`, like `numpy.clip`.

    Exact families stay exact. Real families and balls only.

    Parameters:
        A: The type of `values`, inferred.
        B: The type of `a_min`, inferred.
        D: The type of `a_max`, inferred.

    Args:
        values: A batch or a scalar.
        a_min: The lower bounds: a batch or a scalar.
        a_max: The upper bounds: a batch or a scalar; at least one argument is a batch.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    comptime R = _Clipped[A, B, D]
    comptime assert R != Complex and R != ComplexBall, "Complex numbers have no order."
    comptime if R == Integer:
        return rebind_var[Batch[R]](_map3[_bare3[_integer_clip]](values, a_min, a_max, _ComplexContextArgument()))
    elif R == Rational:
        return rebind_var[Batch[R]](_map3[_bare3[_rational_clip]](values, a_min, a_max, _ComplexContextArgument()))
    elif R == Ball:
        return rebind_var[Batch[R]](_map3[_ball_clip](values, a_min, a_max, Optional[BallContext]()))
    else:
        return rebind_var[Batch[R]](_map3[_float_clip](values, a_min, a_max, Optional[ArithmeticContext]()))


def clip[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable, //,
](values: A, a_min: B, a_max: D, *, context: C) raises -> Batch[_WithContext[_Clipped[A, B, D], C]]:
    """Each value limited to `[a_min, a_max]`, rounded once with `context`."""
    comptime F = _Clipped[A, B, D]
    comptime R = _WithContext[F, C]
    comptime assert R == Float or R == Ball, "clip takes real values with an ArithmeticContext or a BallContext."
    comptime assert not _Balls[F] or _BallContextType[C], "Ball values take a BallContext."
    comptime if R == Ball:
        return rebind_var[Batch[R]](_map3[_ball_clip](values, a_min, a_max, _ball_context(context)))
    else:
        return rebind_var[Batch[R]](_map3[_float_clip](values, a_min, a_max, _uniform_reduction_context(context)))


def abs[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _MagnitudeContext[T] = _MagnitudeContext[T](),
) raises -> Batch[_Magnitude[T]]:
    """The absolute value of every element, like `numpy.abs`.

    Integers, Rationals and Floats keep their family exactly; Complex numbers
    give their magnitudes as Floats rounded once, and balls give balls.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The rounding of Complex magnitudes, or the ball context; none
            for the other families.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    comptime R = _Magnitude[T]
    comptime if T == Integer:
        return rebind_var[Batch[R]](_map1[_bare1[_integer_abs]](values, _ComplexContextArgument()))
    elif T == Rational:
        return rebind_var[Batch[R]](_map1[_bare1[_rational_abs]](values, _ComplexContextArgument()))
    elif T == Float:
        return rebind_var[Batch[R]](_map1[_bare1[_float_abs]](values, _ComplexContextArgument()))
    elif T == Complex:
        return rebind_var[Batch[R]](_map1[_complex_abs](values, rebind[Optional[ArithmeticContext]](context)))
    elif T == Ball:
        return rebind_var[Batch[R]](_map1[_ball_abs](values, rebind[Optional[BallContext]](context)))
    else:
        comptime assert T == ComplexBall, "abs takes Integer, Rational, Float, Complex or ball elements."
        return rebind_var[Batch[R]](_map1[_complex_ball_abs](values, rebind[Optional[BallContext]](context)))


def sqrt[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The square root of every element, like `numpy.sqrt`.

    Integer, Rational and Float elements give correctly rounded Floats; Complex
    elements give principal roots; balls give enclosing balls.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_sqrt, _ball_sqrt, _complex_sqrt, _complex_ball_sqrt](values, context)


def pow_int[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    values: A, exponent: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """Each value to an Integer power, rounded once.

    Exact values give Floats under a context; Complex values, Complex numbers;
    balls, balls.

    Parameters:
        A: The type of `values`, inferred.
        B: The type of `exponent`, inferred.

    Args:
        values: A batch or a scalar of any family.
        exponent: Integers: a batch or a scalar.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _binary[_float_pow_int, _ball_pow_int, _complex_pow_int, _complex_ball_pow_int](values, exponent, context)


# ---------------------------------------------------------- rounding to integers


def _integral[
    T: ImplicitlyCopyable & Deinitable, //,
    integer: def(Integer) raises thin -> Integer,
    rational: def(Rational) raises thin -> Integer,
    real: def(Float) raises thin -> Integer,
](values: Batch[T]) raises -> Batch[Integer]:
    comptime if T == Integer:
        return _map1[_bare1[integer]](values, _ComplexContextArgument())
    elif T == Rational:
        return _map1[_bare1[rational]](values, _ComplexContextArgument())
    else:
        comptime assert T == Float, "This function takes Integer, Rational or Float elements."
        return _map1[_bare1[real]](values, _ComplexContextArgument())


def floor[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[Integer]:
    """The largest Integer at most each element, exactly, like `numpy.floor`
    but with Integer results.

    Parameters:
        T: The element type, inferred: Integer, Rational or Float.

    Args:
        values: The batch.

    Returns:
        Integers, in a batch of the same shape.

    Raises:
        For a NaN or an infinity among Float elements.
    """
    return _integral[_integer_floor, _rational_floor, _float_floor](values)


def ceil[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[Integer]:
    """The smallest Integer at least each element, exactly, like `numpy.ceil`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        Integers, in a batch of the same shape.

    Raises:
        For a NaN or an infinity among Float elements.
    """
    return _integral[_integer_ceil, _rational_ceil, _float_ceil](values)


def trunc[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[Integer]:
    """Each element rounded toward zero to an Integer, like `numpy.trunc`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        Integers, in a batch of the same shape.

    Raises:
        For a NaN or an infinity among Float elements.
    """
    return _integral[_integer_trunc, _rational_trunc, _float_trunc](values)


def round[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[Integer]:
    """Each element rounded to the nearest Integer, ties to even, like `numpy.round`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        Integers, in a batch of the same shape.

    Raises:
        For a NaN or an infinity among Float elements.
    """
    return _integral[_integer_round, _rational_round, _float_round](values)


# ------------------------------------------------------------ complex parts


def conjugate[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[T]:
    """The complex conjugate of every element, like `numpy.conjugate`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        A batch of the same shape.

    Raises:
        Only on a checked size error.
    """
    comptime if T == Complex:
        return rebind_var[Batch[T]](_map1[_bare1[_complex_conjugate]](values, _ComplexContextArgument()))
    else:
        comptime assert T == ComplexBall, "conjugate takes Complex or ComplexBall elements."
        return rebind_var[Batch[T]](_map1[_bare1[_complex_ball_conjugate]](values, _ComplexContextArgument()))


def real[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[_Magnitude[T]]:
    """The real part of every element, like `numpy.real`: Floats or balls.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        A batch of the same shape.

    Raises:
        Only on a checked size error.
    """
    comptime if T == Complex:
        return rebind_var[Batch[_Magnitude[T]]](_map1[_bare1[_complex_real]](values, _ComplexContextArgument()))
    else:
        comptime assert T == ComplexBall, "real takes Complex or ComplexBall elements."
        return rebind_var[Batch[_Magnitude[T]]](_map1[_bare1[_complex_ball_real]](values, _ComplexContextArgument()))


def imag[T: ImplicitlyCopyable & Deinitable, //](values: Batch[T]) raises -> Batch[_Magnitude[T]]:
    """The imaginary part of every element, like `numpy.imag`: Floats or balls.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.

    Returns:
        A batch of the same shape.

    Raises:
        Only on a checked size error.
    """
    comptime if T == Complex:
        return rebind_var[Batch[_Magnitude[T]]](_map1[_bare1[_complex_imag]](values, _ComplexContextArgument()))
    else:
        comptime assert T == ComplexBall, "imag takes Complex or ComplexBall elements."
        return rebind_var[Batch[_Magnitude[T]]](_map1[_bare1[_complex_ball_imag]](values, _ComplexContextArgument()))


def angle[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _MagnitudeContext[T] = _MagnitudeContext[T](),
) raises -> Batch[_Magnitude[T]]:
    """The argument of every element in `(-pi, pi]`, like `numpy.angle`:
    Floats rounded once, or balls.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    comptime if T == Complex:
        return rebind_var[Batch[_Magnitude[T]]](_map1[_complex_angle](values, rebind[Optional[ArithmeticContext]](context)))
    else:
        comptime assert T == ComplexBall, "angle takes Complex or ComplexBall elements."
        return rebind_var[Batch[_Magnitude[T]]](_map1[_complex_ball_angle](values, rebind[Optional[BallContext]](context)))


# ------------------------------------------------------------- elementary


def exp[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """`apn_mojo.exp` of every element, like `numpy.exp`.

    Integer, Rational and Float elements give correctly rounded Floats; Complex
    elements give Complex numbers; balls give balls enclosing the function's
    values. The other one-argument functions follow the same rules.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_exp, _ball_exp, _complex_exp, _complex_ball_exp](values, context)


def expm1[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """`exp(x) - 1` of every real element or ball, like `numpy.expm1`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_expm1, _ball_expm1](values, context)


def exp2[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """`2**x` of every real element or ball, like `numpy.exp2`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_exp2, _ball_exp2](values, context)


def log[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The natural logarithm of every element, like `numpy.log`; Complex
    elements take the principal branch.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_log, _ball_log, _complex_log, _complex_ball_log](values, context)


def log1p[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """`log(1 + x)` of every real element or ball, like `numpy.log1p`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_log1p, _ball_log1p](values, context)


def log2[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The base-2 logarithm of every real element or ball, like `numpy.log2`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_log2, _ball_log2](values, context)


def log10[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The base-10 logarithm of every real element or ball, like `numpy.log10`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_log10, _ball_log10](values, context)


def sin[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The sine of every element, like `numpy.sin`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_sin, _ball_sin, _complex_sin, _complex_ball_sin](values, context)


def cos[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The cosine of every element, like `numpy.cos`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_cos, _ball_cos, _complex_cos, _complex_ball_cos](values, context)


def tan[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The tangent of every element, like `numpy.tan`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_tan, _ball_tan, _complex_tan, _complex_ball_tan](values, context)


def sin_cos[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]:
    """The sines and the cosines of every real element or ball, computed together.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of each result, both of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary_pair[_float_sin_cos, _ball_sin_cos](values, context)


def atan[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The arctangent of every element, like `numpy.arctan`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_atan, _ball_atan, _complex_atan, _complex_ball_atan](values, context)


def asin[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The arcsine of every element, like `numpy.arcsin`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_asin, _ball_asin, _complex_asin, _complex_ball_asin](values, context)


def acos[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The arccosine of every element, like `numpy.arccos`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_acos, _ball_acos, _complex_acos, _complex_ball_acos](values, context)


def atan2[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    y: A, x: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The angle of each point `(x, y)`, like `numpy.arctan2`.

    Two-argument functions broadcast their arguments; either may be a scalar.

    Parameters:
        A: The type of `y`, inferred.
        B: The type of `x`, inferred.

    Args:
        y: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape: Floats, or balls if either argument holds balls.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_atan2, _ball_atan2](y, x, context)


def sinh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The hyperbolic sine of every element, like `numpy.sinh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_sinh, _ball_sinh, _complex_sinh, _complex_ball_sinh](values, context)


def cosh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The hyperbolic cosine of every element, like `numpy.cosh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_cosh, _ball_cosh, _complex_cosh, _complex_ball_cosh](values, context)


def tanh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The hyperbolic tangent of every element, like `numpy.tanh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_tanh, _ball_tanh, _complex_tanh, _complex_ball_tanh](values, context)


def asinh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The inverse hyperbolic sine of every element, like `numpy.arcsinh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_asinh, _ball_asinh, _complex_asinh, _complex_ball_asinh](values, context)


def acosh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The inverse hyperbolic cosine of every element, like `numpy.arccosh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_acosh, _ball_acosh, _complex_acosh, _complex_ball_acosh](values, context)


def atanh[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The inverse hyperbolic tangent of every element, like `numpy.arctanh`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _unary[_float_atanh, _ball_atanh, _complex_atanh, _complex_ball_atanh](values, context)


def pow[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    base: A, exponent: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """`base ** exponent` elementwise, like `numpy.power`, for real, Complex and ball values.

    Parameters:
        A: The type of `base`, inferred.
        B: The type of `exponent`, inferred.

    Args:
        base: A batch or a scalar.
        exponent: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _binary[_float_pow, _ball_pow, _complex_pow, _complex_ball_pow](base, exponent, context)


def _float_root(value: _FloatArgument, n: Integer, /, *, context: Optional[ArithmeticContext]) raises -> Float:
    return _float_rootn(value, Int(n), context=context)


def _ball_root(value: _BallArgument, n: Integer, /, *, context: Optional[BallContext]) raises -> Ball:
    return _ball_rootn(value, Int(n), context=context)


def rootn[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], n: Int, *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The real `n`th root of every real element or ball.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        n: The degree of the root.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    comptime R = _Mapped[T]
    var result = _real_binary[_float_root, _ball_root](values, Integer(n), rebind[_ContextOf[_Pair[Batch[T], Integer]]](context))
    return rebind_var[Batch[R]](result^)


# ---------------------------------------------------------------- special


def gamma[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The gamma function of every real element or ball, like `scipy.special.gamma`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_gamma, _ball_gamma](values, context)


def gammaln[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """`log|gamma(x)|` of every real element or ball, like `scipy.special.gammaln`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_gammaln, _ball_gammaln](values, context)


def digamma[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The digamma function of every real element or ball, like `scipy.special.digamma`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_digamma, _ball_digamma](values, context)


def erf[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The error function of every real element or ball, like `scipy.special.erf`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_erf, _ball_erf](values, context)


def erfc[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The complementary error function of every real element or ball, like `scipy.special.erfc`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_erfc, _ball_erfc](values, context)


def erfi[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The imaginary error function of every real element or ball, like `scipy.special.erfi`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_erfi, _ball_erfi](values, context)


def erfinv[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The inverse error function of every real element or ball, like `scipy.special.erfinv`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_erfinv, _ball_erfinv](values, context)


def expi[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The exponential integral Ei of every real element or ball, like `scipy.special.expi`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_expi, _ball_expi](values, context)


def ndtr[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The standard normal distribution function of every real element or ball, like `scipy.special.ndtr`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_ndtr, _ball_ndtr](values, context)


def log_ndtr[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The logarithm of `ndtr` of every real element or ball, like `scipy.special.log_ndtr`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_log_ndtr, _ball_log_ndtr](values, context)


def ndtri[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The inverse of `ndtr` of every real element or ball, like `scipy.special.ndtri`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary[_float_ndtri, _ball_ndtri](values, context)


def sici[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]:
    """The sine and cosine integrals of every real element or ball, like `scipy.special.sici`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of each result, both of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary_pair[_float_sici, _ball_sici](values, context)


def shichi[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]:
    """The hyperbolic sine and cosine integrals of every real element or ball, like `scipy.special.shichi`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of each result, both of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary_pair[_float_shichi, _ball_shichi](values, context)


def fresnel[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Tuple[Batch[_Mapped[T]], Batch[_Mapped[T]]]:
    """The Fresnel integrals `(S, C)` of every real element or ball, like `scipy.special.fresnel`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of each result, both of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    return _real_unary_pair[_float_fresnel, _ball_fresnel](values, context)


def _float_branch(value: _FloatArgument, k: Integer, /, *, context: Optional[ArithmeticContext]) raises -> Float:
    return _float_lambertw(value, k=Int(k), context=context)


def _ball_branch(value: _BallArgument, k: Integer, /, *, context: Optional[BallContext]) raises -> Ball:
    return _ball_lambertw(value, k=Int(k), context=context)


def lambertw[T: ImplicitlyCopyable & Deinitable, //](
    values: Batch[T], *, k: Int = 0, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """Branch `k` of the Lambert W function of every real element or ball, like `scipy.special.lambertw`.

    Parameters:
        T: The element type, inferred.

    Args:
        values: The batch.
        k: The branch.
        context: The family's context, as for the scalar function; exact
            elements need one.

    Returns:
        A batch of the same shape.

    Raises:
        As the scalar function does, with the failing element's position.
    """
    var result = _real_binary[_float_branch, _ball_branch](
        values, Integer(k), rebind[_ContextOf[_Pair[Batch[T], Integer]]](context)
    )
    return rebind_var[Batch[_Mapped[T]]](result^)


def beta[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The beta function of each pair, like `scipy.special.beta`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_beta, _ball_beta](a, b, context)


def betaln[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, b: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """`log|beta(a, b)|` of each pair, like `scipy.special.betaln`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_betaln, _ball_betaln](a, b, context)


def poch[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    z: A, m: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The Pochhammer symbol `(z)_m` of each pair, like `scipy.special.poch`.

    Parameters:
        A: The type of `z`, inferred.
        B: The type of `m`, inferred.

    Args:
        z: A batch or a scalar.
        m: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_poch, _ball_poch](z, m, context)


def _float_riemann(x: _FloatArgument, /, *, context: Optional[ArithmeticContext]) raises -> Float:
    return _root_zeta(x, context=context)


def _ball_riemann(x: Ball, /, *, context: Optional[BallContext]) raises -> Ball:
    return _root_zeta(x, context=context)


def zeta[T: ImplicitlyCopyable & Deinitable, //](
    x: Batch[T], *, context: _ContextOf[T] = _ContextOf[T](),
) raises -> Batch[_Mapped[T]]:
    """The Riemann zeta function of every real element or ball, like `scipy.special.zeta` without `q`."""
    return _real_unary[_float_riemann, _ball_riemann](x, context)


def zeta[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    x: A, q: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The Hurwitz zeta function `zeta(x, q)` of each pair, like `scipy.special.zeta`.

    Parameters:
        A: The type of `x`, inferred.
        B: The type of `q`, inferred.

    Args:
        x: A batch or a scalar.
        q: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_zeta, _ball_zeta](x, q, context)


def polygamma[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    n: A, x: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The `n`th derivative of digamma at each `x`, like `scipy.special.polygamma`; `n` is Integers.

    Parameters:
        A: The type of `n`, inferred.
        B: The type of `x`, inferred.

    Args:
        n: The orders, Integers: a batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_polygamma, _ball_polygamma](n, x, context)


def gammainc[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, x: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The regularized lower incomplete gamma function, like `scipy.special.gammainc`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `x`, inferred.

    Args:
        a: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_gammainc, _ball_gammainc](a, x, context)


def gammaincc[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    a: A, x: B, *, context: _ContextOf[_Pair[A, B]] = _ContextOf[_Pair[A, B]](),
) raises -> Batch[_Pair[A, B]]:
    """The regularized upper incomplete gamma function, like `scipy.special.gammaincc`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `x`, inferred.

    Args:
        a: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_binary[_float_gammaincc, _ball_gammaincc](a, x, context)


def hyp1f1[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, //,
](
    a: A, b: B, x: D, *, context: _ContextOf[_Triple[A, B, D]] = _ContextOf[_Triple[A, B, D]](),
) raises -> Batch[_Triple[A, B, D]]:
    """Kummer's confluent hypergeometric function, like `scipy.special.hyp1f1`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.
        D: The type of `x`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_ternary[_float_hyp1f1, _ball_hyp1f1](a, b, x, context)


def betainc[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, //,
](
    a: A, b: B, x: D, *, context: _ContextOf[_Triple[A, B, D]] = _ContextOf[_Triple[A, B, D]](),
) raises -> Batch[_Triple[A, B, D]]:
    """The regularized incomplete beta function, like `scipy.special.betainc`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.
        D: The type of `x`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_ternary[_float_betainc, _ball_betainc](a, b, x, context)


def hyp2f1[
    A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
    D: _MapArgument & ImplicitlyCopyable & Deinitable, E: _MapArgument & ImplicitlyCopyable & Deinitable, //,
](
    a: A, b: B, c: D, x: E, *, context: _ContextOf[_Quadruple[A, B, D, E]] = _ContextOf[_Quadruple[A, B, D, E]](),
) raises -> Batch[_Quadruple[A, B, D, E]]:
    """The Gauss hypergeometric function, like `scipy.special.hyp2f1`.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.
        D: The type of `c`, inferred.
        E: The type of `x`, inferred.

    Args:
        a: A batch or a scalar.
        b: A batch or a scalar.
        c: A batch or a scalar.
        x: A batch or a scalar; at least one argument is a batch.
        context: The family's context; exact values need one.

    Returns:
        A batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or as the scalar function does.
    """
    return _real_quaternary[_float_hyp2f1, _ball_hyp2f1](a, b, c, x, context)


# ---------------------------------------------------------------- integers


def _comb_repeated(N: Integer, k: Integer) raises -> Integer:
    return _root_comb(N, k, repetition=True)


def comb[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    N: A, k: B, *, repetition: Bool = False,
) raises -> Batch[Integer]:
    """The number of ways to choose `k` of `N` things, exactly, like
    `scipy.special.comb` with `exact=True`; 0 when `k > N`, `N < 0` or `k < 0`.

    Parameters:
        A: The type of `N`, inferred.
        B: The type of `k`, inferred.

    Args:
        N: The numbers of things: Integers, a batch or a scalar.
        k: The numbers taken: Integers, a batch or a scalar; at least one
            argument is a batch.
        repetition: Whether a thing may be taken more than once.

    Returns:
        Integers, in a batch of the broadcast shape.

    Raises:
        When the shapes do not broadcast, or a count exceeds the addressable size.
    """
    if repetition:
        return _map2[_no_context[_comb_repeated]](N, k, _ComplexContextArgument())
    return _map2[_no_context[_integer_comb]](N, k, _ComplexContextArgument())


# ---------------------------------------------------------------- selection


comptime _Chosen[A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable] = (
    _ElementOf[A] if _Batched[A] else _ElementOf[B]
)


def _select[E: ImplicitlyCopyable & Deinitable](condition: Bool, a: E, b: E, /) raises -> E:
    return a if condition else b


def where[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, //](
    condition: Mask, a: A, b: B,
) raises -> Batch[_Chosen[A, B]]:
    """Elements of `a` where `condition` holds and of `b` elsewhere, like `numpy.where`.

    `a` and `b` broadcast to the mask's shape; masks do not broadcast. A
    scalar converts exactly to the batch's family, as `vmap` converts it.

    Parameters:
        A: The type of `a`, inferred.
        B: The type of `b`, inferred.

    Args:
        condition: The mask.
        a: A batch or a scalar.
        b: A batch or a scalar; at least one of `a` and `b` is a batch.

    Returns:
        A batch of the mask's shape.

    Raises:
        When `a` or `b` does not broadcast to the mask's shape.
    """
    comptime assert _Batched[A] or _Batched[B], "where needs a batch for a or b."
    comptime if _Batched[A] and _Batched[B]:
        comptime assert _ElementOf[A] == _ElementOf[B], "where takes batches of one family; convert one first."
    var plan = _Elementwise(condition.shape())
    var mask = condition if len(plan.shape) else Mask(condition.to_list(), shape=plan.mapped.copy())
    var result = vmap[_select[_Chosen[A, B]]](in_axes=plan.levels([True, _Batched[A], _Batched[B]]))(
        mask, plan.spread(a), plan.spread(b),
    )
    return plan.result(rebind_var[Batch[_Chosen[A, B]]](result^))
