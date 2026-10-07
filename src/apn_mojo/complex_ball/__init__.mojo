"""Complex balls: rectangles of two real balls, one declaration per name.

Every function returns a complex ball that contains the exact result for
every point of its input rectangles. On a branch cut a point takes its
counter-clockwise continuous value; a rectangle
that crosses a cut gets a result covering both sides. A `context=`
(`BallContext`) sets the result precision.
"""

from .value import ComplexBall
from .math import add, subtract, multiply, divide, conjugate, real, imag, reciprocal, abs, angle, pow_int, stable_hash
from .elementary import exp, log, sqrt, pow, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh, acosh, atanh
from .sets import contains, contains_zero, overlaps, union, intersection, to_complex_if_certain
