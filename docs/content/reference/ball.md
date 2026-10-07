# apn_mojo.ball

Use a `Ball` to carry guaranteed bounds through a real-valued calculation.
It stores an interval as a midpoint and a nonnegative radius. Every operation
includes all possible results for the input intervals, even after rounding;
the bounds may be wider than the smallest possible interval.

Start with [calculations with guaranteed bounds](../tutorial/balls.md) for a
measurement example, comparisons, and certified conversion to Float.

<!-- api: ball -->

## Kinds, precision and exactness

A finite ball represents `[m - r, m + r]`. The midpoint is a `Float` with a
chosen precision; the radius has a private 30-bit significand and rounds
upward. `midpoint()` and `radius()` return the stored values as Floats.
A radius of zero represents one exact point.

An unbounded ball contains every real number. An indeterminate ball has no
usable enclosure, usually because part of the input is outside the function's
domain. Division by an interval containing zero and a square root of an
interval reaching below zero both return an indeterminate ball. A square
root can accept a lower bound of exactly zero.

Pass `context=BallContext(precision)` to choose midpoint precision. Without
one, operations use the largest midpoint precision of their Ball and Float
operands. Exact integer and rational operands contribute no precision. Balls
constructed from integers or binary fractions have at least 128 midpoint bits,
so small exact inputs are not given an unnecessarily narrow working format.
Only actual midpoint rounding adds a rounding error to the radius.

## Batches of balls

`Batch[Ball]` supports shapes, indexing, views, masks, assignment, iteration,
and printing. Apply scalar functions from `apn_mojo.ball` with `vmap`, or use
`lift` for broadcasting binary calls, outer products, and folds with
`BallContext`. NumPy-style functions such as `batch.add` and `batch.exp`
also work on balls. `min` and `max` enclose the extreme over all points;
`cumsum` and `cumprod` carry the enclosure through each prefix. Use `lift`
for other folds. Ball batches have no arithmetic operators or batch JSON.

Mapping can also take exact-number batches as input. For example,
`vmap[ball.sqrt]()(integers, context=BallContext(64))` produces balls from
integers. A mapped predicate returns a `Mask`. Supported Optional-returning
functions produce a value batch and a mask showing which results exist.

## Comparisons and sets

Balls have no ordinary `<` or `==`. `compare` returns one of five
`BallOrder` values: `less`, `equal`, `greater`, `overlap`, or `undefined`.
The `certainly_*` predicates are true only when the relation holds for every
point in the intervals. Overlap alone does not establish equality.

`contains`, `contains_ball`, and `overlaps` test exact interval endpoints.
`union` returns an enclosing interval, including any gap between disjoint
inputs. `intersection`, `split`, `floor_if_certain`, `ceil_if_certain`,
`round_half_even_if_certain`, and `simplest_rational_in` provide set and
rounding operations with their own documented result types.

## Text, keys and scope

Balls support text output and parsing, representation comparisons,
`stable_hash`, and `BallKey` for dictionary keys. Ordinary interval comparison
and representation identity answer different questions; use the operation
that matches the caller's purpose.

JSON is a version-1 record with the kind (`finite`, `unbounded` or
`indeterminate`) and two complete Float records, the midpoint and the radius:

```json
{"version":1,"family":"ball","kind":"finite",
 "midpoint":{"version":1,"family":"float","precision":"53","emin":"-4611686018427387904","emax":"4611686018427387903","class":"finite","sign":"+","significand":"4503599627370496","exponent":"1"},
 "radius":{"version":1,"family":"float","precision":"30","emin":"-4611686018427387904","emax":"4611686018427387903","class":"finite","sign":"+","significand":"536870912","exponent":"-29"}}
```

This record represents `[1 +/- 2**-30]`. JSON preserves the representation
exactly: `Ball.from_json(x.to_json())` has the same stored representation as
`x`. The reader accepts only canonical records. The radius must have 30 bits,
the default bounds, and sign `+`. An unbounded ball has a finite midpoint
and an infinite radius; an indeterminate ball has a NaN midpoint and an
infinite radius.

Printed text preserves an enclosure, not necessarily the stored representation.
The printer expands the displayed radius to cover midpoint rounding, and parsing
that text can widen the interval again. Use representation comparisons or
`BallKey` when you need to distinguish stored representations.

The family is real-valued; `ComplexBall` in
[apn_mojo.complex_ball](complex_ball.md) pairs two balls into a rectangle.
[Ball arithmetic](../architecture/balls.md) explains the enclosure formulas and
the internal radius representation.

## Elementary functions

The elementary functions of `apn_mojo.float` have ball versions of the same
names, which return a ball containing the function's value at every point of
the input ball. Monotone functions evaluate a correctly rounded kernel at the
two exact ends of the ball; `sin` and `cos` widen their midpoint value by a
bound of the derivative and stay within `[-1, 1]`. A function undefined
somewhere in its input ball gives an indeterminate ball: `log` of a ball
reaching 0, `asin` of a ball reaching past 1, or `tan` of a ball that may
contain a pole. `atan2` gives `[0 +/- pi]` for a rectangle that meets the
negative real axis, where the angle jumps.

A ball function does not raise when it reaches the `BallContext`'s
`max_precision` budget. At that point, `sin` and `cos` return `[0 +/- 1]`,
and `tan` returns an indeterminate ball.

## Constants and canonical balls

`pi_ball(p)`, `euler_e_ball`, `ln2_ball`, `log2_10_ball`, `euler_gamma_ball`
and `catalan_ball` return the canonical ball of the constant at precision `p`:
`[round_down(c, p), round_up(c, p)]`, with a midpoint of `p + 1` bits and the
exact half-width as radius. It depends only on the constant and `p`.

`canonical(ball, p)` returns the canonical ball of the enclosed value, or
None if the input is too wide to determine both ends. You can use this to
cache the highest precision computed for a constant and serve later requests
with `canonical(cached, p)`. For pi, if that returns None, compute
`pi_ball(2 * p)` and keep the new result. Each answer depends only on the
constant and requested precision, regardless of the cache's history.

`to_float_if_certain(ball, context=c)` rounds both ends of a ball to the
context's format and returns the Float when they agree: then every point of
the ball rounds to it. It is the step that correctly rounded functions repeat
at higher precision.

## Special functions

The special functions of `apn_mojo.float` have ball versions of the same
names. The monotone ones evaluate their kernel at the two ends of the ball:
erf, erfc, erfi, Shi, Chi, digamma between its poles, Ei on each side of 0,
and each branch of Lambert's W. Gamma and log Gamma use their monotone pieces
on each side of the minimum at 1.4616...; the others widen the midpoint value
by a bound of the derivative: 1 for Si and Fresnel's integrals, `1/x` for Ci,
and the end values of `|Gamma| |psi|` for Gamma between negative poles. A
ball containing a pole, 0 or a negative integer for Gamma and digamma and 0
for Ei, or reaching outside the domain is indeterminate.

## Significance arithmetic

For choosing a precision and certifying displayed digits, see
[precision and accuracy](../guides/precision.md).

A precision tracked as in Mathematica is a ball radius on a logarithmic
scale: `d` digits of relative precision mean a radius of `|m| 10**-d`.
`radius_for_relative_digits(m, d)` returns that radius rounded up to the
30-bit radius format; a zero midpoint has no relative precision and raises.
`bits_to_digits(b)` and `digits_to_bits(d)` convert precisions as balls,
since `log2 10` is irrational. `propagation_bound["exp"](m, r)` returns the
term that the ball function adds to its kernel's radius at the midpoint, for
`exp`, `expm1`, `exp2`, `log`, `log2`, `log10`, `log1p`, `sin`, `cos`, `atan`,
`tanh`, `asinh` and `atanh`, so that a caller bounds its own errors the same
way; it raises for a ball reaching outside the function's domain.
