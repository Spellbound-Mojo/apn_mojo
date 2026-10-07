"""Complex balls as sets: containment, overlap, hulls, and the certain
rounding of both parts."""

from ..float.value import Float
from ..float.context import ArithmeticContext
from ..float._rounding import _finish_round
from ..complex.value import Complex
from ..complex.context import ComplexContext
from ..ball.value import Ball
from ..ball.sets import contains_ball as _contains_ball, contains_zero as _contains_zero, overlaps as _overlaps
from ..ball.sets import union as _union, intersection as _intersection
from ..ball._certified import _certify
from .value import ComplexBall


def contains(ball: ComplexBall, value: ComplexBall) raises -> Bool:
    """Whether `ball`'s rectangle contains `value`'s.

    An exact point is a complex ball too: `ComplexBall(z)` of a Complex or an
    ExactComplex.

    Args:
        ball: The containing ball.
        value: The contained ball.

    Returns:
        True when both parts contain the other's.

    Raises:
        Only on a checked size error.
    """
    return _contains_ball(ball.real(), value.real()) and _contains_ball(ball.imag(), value.imag())


def contains_zero(ball: ComplexBall) raises -> Bool:
    """Whether the rectangle contains 0.

    Args:
        ball: The ball.

    Returns:
        True when both parts contain 0.

    Raises:
        Only on a checked size error.
    """
    return _contains_zero(ball.real()) and _contains_zero(ball.imag())


def overlaps(a: ComplexBall, b: ComplexBall) raises -> Bool:
    """Whether two rectangles share a point.

    Args:
        a: The first ball.
        b: The second ball.

    Returns:
        True when both pairs of parts overlap.

    Raises:
        Only on a checked size error.
    """
    return _overlaps(a.real(), b.real()) and _overlaps(a.imag(), b.imag())


def union(a: ComplexBall, b: ComplexBall) raises -> ComplexBall:
    """The least complex ball containing both rectangles.

    Args:
        a: The first ball.
        b: The second ball.

    Returns:
        The hull, part by part.

    Raises:
        Only on a checked size error.
    """
    return ComplexBall(_real=_union(a.real(), b.real()), _imag=_union(a.imag(), b.imag()))


def intersection(a: ComplexBall, b: ComplexBall) raises -> Optional[ComplexBall]:
    """The least complex ball containing the common points, if any.

    Args:
        a: The first ball.
        b: The second ball.

    Returns:
        The intersection, part by part, or None when the rectangles are
        disjoint.

    Raises:
        Only on a checked size error.
    """
    var re = _intersection(a.real(), b.real())
    var im = _intersection(a.imag(), b.imag())
    if not re or not im:
        return None
    return ComplexBall(_real=re.take(), _imag=im.take())


def to_complex_if_certain(ball: ComplexBall, *, context: Optional[ComplexContext] = None) raises -> Optional[Complex]:
    """The Complex that every point of the rectangle rounds to, when there is
    one: each part rounded in its own context.

    Args:
        ball: The ball.
        context: The component formats, rounding modes and traps; by default
            each part's midpoint format, rounded to nearest-even.

    Returns:
        The Complex, or None when either part's ends round differently.

    Raises:
        On a trapped condition of either part.
    """
    var real_target = context.value().real() if context else ArithmeticContext(format=ball.real()._midpoint.format())
    var imag_target = context.value().imag() if context else ArithmeticContext(format=ball.imag()._midpoint.format())
    var re = _certify(ball.real(), real_target)
    var im = _certify(ball.imag(), imag_target)
    if not re or not im:
        return None
    return Complex(_real=Float(_rounded=_finish_round(re.take(), real_target, False)), _imag=Float(_rounded=_finish_round(im.take(), imag_target, False)))
