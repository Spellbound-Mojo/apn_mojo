# Ball arithmetic

Ball arithmetic carries guaranteed bounds through calculations. Real balls
support arithmetic, comparisons, set operations, text, hashes, mapping with
`vmap`, elementary functions, and constants. Complex balls provide arithmetic,
set operations, and elementary functions. These bounds also let Float and
Complex functions certify their final rounding; see
[Correct rounding](certified.md). The public APIs are in
[apn_mojo.ball](../reference/ball.md) and
[apn_mojo.complex_ball](../reference/complex_ball.md).

## What a ball is

A finite ball `[m +/- r]` represents the real interval `[m - r, m + r]`.
The midpoint is a Float and the radius is nonnegative. Each operation returns
an enclosure for every result possible from its input intervals. An unbounded
ball contains all reals; an indeterminate ball carries no usable enclosure.

These bounds let a caller determine whether a value is certainly positive,
nonzero, or below another interval, when a point approximation alone would
not settle the question.
A `ComplexBall` pairs two real balls to describe a rectangle in the complex
plane.

## Decisions

### Interval arithmetic is in scope

`Ball` is a separate family with the same scalar-function and mapping design
as the existing numbers. `Float` remains a point type. Internal error bounds
used to certify a Float result do not turn that result into an interval.

### Transcendental functions are correctly rounded

Exponentials, logarithms, trigonometric and hyperbolic functions and the
constants return correctly rounded Floats; a Complex result rounds each part
correctly in its own mode. Special functions follow the same contract. A ball
version may be available before its point-valued counterpart can certify the
last bit.

### No public status

Rounding status stays internal. The API uses exact comparisons, directed
rounding, traps, and interval bounds to answer numerical questions. A parallel
API returning status beside every value would duplicate the function surface.

### Budgets and ball precision live in contexts

`BallContext` holds the midpoint precision and working-precision budget.
Functions document which budget fields they use; those fields do not impose
a general time or memory cap. Keeping these settings in `context=` lets
mapping pass them through the same interface.

The transcendental functions use the context's budget to limit adaptive
precision. A point-valued function raises when it cannot certify a result
within the budget; a ball function returns a wider valid enclosure instead.

### Values, operations and conversions

- Constructors into Ball are explicit. Mixed arithmetic uses argument adapters
  instead of adding implicit conversions that could make root calls ambiguous.
- Mathematical operations are free functions. Value inspection, such as
  `midpoint()` and `certainly_positive()`, uses methods.
- A ball constructed from an Integer or binary-fraction Rational has at least
  128 midpoint bits. Small exact inputs therefore retain useful working
  precision even when the point itself needs only a few bits.
- The radius uses a private 30-bit mantissa and an integer exponent, rounded
  upward. `radius()` returns its exact value as a Float.
- `Batch[Ball]` and `Batch[ComplexBall]` support containers, `vmap`, and `lift`,
  including ordered folds with `BallContext`. Ball batch operators, built-in
  reductions, and batch JSON are not implemented.

## Arithmetic

Each operation rounds the midpoint to nearest at its working precision and
rounds the radius outward. For input centers `m_x`, `m_y` and radii `r_x`,
`r_y`, the bounds include:

- Addition: `r_x + r_y`.
- Multiplication: `abs(m_x) * r_y + abs(m_y) * r_x + r_x * r_y`.
- Division, when the denominator excludes zero:
  `(abs(m_x) * r_y + abs(m_y) * r_x) / (abs(m_y) * (abs(m_y) - r_y))`.

Midpoint rounding adds half a unit in the last place only when inexact.
Exact inputs can therefore produce a zero radius. Radius products and sums
round up, while denominator lower bounds round down. The radius terms stay
in radius arithmetic (as in Johansson, *Arb: efficient arbitrary-precision midpoint-radius interval arithmetic*, IEEE Transactions on Computers 66(8), 2017): whether a denominator contains zero is
decided exactly from the midpoint's top bits against the radius's 30-bit
mantissa, and `abs(m_y) - r_y` is bounded below from the midpoint's top 64
bits, so only the midpoint takes a Float operation.

**Kernel operations.** The special-function kernels do most of their
arithmetic on two Balls at a known precision, so they skip the public path's
argument records and context objects. Their results are the same as the
public path's:
- `_ball_sum`, `_ball_product`, `_ball_quotient` and `_ball_scale2` call the
  Float binary cores that the general path reaches for a finite binary
  midpoint. `_ball_product_word` and `_ball_quotient_word` take a word-sized
  integer operand, whose radius term is `|n| r` or `r / |n|`.
- A product's or quotient's radius terms add in one word before a single
  upward rounding (`_products_up`). Two 30-bit mantissas multiply to a word
  in `[2**58, 2**60)`; in units of half the largest term's unit, each term is
  below `2**61` and three are below `2**63`. A term shifted down adds one
  unit for the bits it loses. The bound is never wider than the chained
  roundings were, and it is tighter in most cases.
- `_ball_mul_assign` and `_ball_add_assign` write the result's significand
  into the destination's own block when the destination is its only owner,
  much as an output argument does in Arb. A loop step past 64 bits then
  allocates nothing.
- `_ball_dot` (`ball/_dot.mojo`) computes `initial + sum x_i y_i` with one
  rounding, as Arb's `arb_dot` does (Johansson, "Faster arbitrary-precision
  dot product and matrix multiplication", ARITH 26, 2019). The products of
  midpoints of at most 128 bits are exact in 256 bits. They add exactly in a
  fixed-point accumulator whose unit `2**U` lies 64 bits below the precision
  under the largest term. A term's bits below the unit add one unit to the
  radius, and each product adds its E.3 bound.

Division by a ball containing zero returns an indeterminate ball. So does a
square root whose interval reaches below zero; a lower endpoint of zero is
allowed. Set operations compare exact endpoints. `compare` has five outcomes:
less, equal, greater, overlap, and undefined.

Text output accounts for decimal midpoint rounding and rounds the displayed
radius upward, preserving an enclosure of the stored ball. Reading the text
back can widen that interval again. To preserve the exact representation,
JSON stores the midpoint and radius as complete Float records; a ComplexBall
record contains two Ball records.

## Transcendental functions

The implementation has three layers:

1. A private point kernel encloses a function value at a requested working
   precision.
2. A ball function extends those bounds over an input interval: monotone
   functions through the kernel at the two exact ends, `sin` and `cos` through
   a derivative bound.
3. A shared rounding driver retries at higher precision until every point in
   the enclosure rounds to the same Float or Complex result, within the budget.

Exact cases need separate treatment: an interval around a rounding boundary
cannot certify one side merely by becoming narrower. Exact results, such as
`log2(8) = 3`, results within a tiny distance of a Float, such as `exp` of a
tiny argument or `tanh` of a large one, and structural zeros of Complex parts
are decided before the driver runs. [Correct rounding](certified.md) gives the
rules and their proofs.

## Complex balls

A complex ball is a rectangle: a real ball for each part. Arithmetic follows
the real formulas part by part, and `abs` bounds the modulus over the
rectangle. The elementary functions combine real ball functions of the parts:
`exp(x + iy) = e**x (cos y + i sin y)`, `log z = log|z| + i arg z`, and the
inverse functions through logarithms and square roots. On a branch cut a point
takes the counter-clockwise continuous value, as the Complex functions do. A
rectangle that crosses a cut covers the values on both sides, since the value
jumps there, and a rectangle containing a singularity, such as 0 for `log`,
gives an indeterminate ball.

### Wide rectangles

A ball must contain every value, and it should not give up while a finite
enclosure exists. A differential check against Arb's balls (37,300 random
cases with radii up to twice their midpoints) found 68 where the result was
indeterminate and Arb's was finite. None contradicted Arb; each came from a
bound that midpoint-radius arithmetic loses for a wide ball:

- **Squares reach below 0.** `(m +- r)**2 = m**2 +- (2|m| r + r**2)` reaches
  below 0 once `r > 0.41 |m|`, so a norm `x**2 + y**2` seemed to reach 0 and a
  quotient by it, or its logarithm, was lost. Norms now come from the parts'
  exact least and greatest magnitudes (`_norm` in `ball/_arithmetic.mojo`),
  shared by division, `log` and `abs`.
- **Hulls cross 0.** A hull `[low, high]` centred on its midpoint misses each
  end by the rounding of its radius, a `2**-29` share of the width: for
  `[1e-9, 50]` that is past 0. A hull of a nonnegative range (a norm, a
  magnitude, `(|z| +- x)/2`) keeps its lower end within one unit of its
  midpoint's last place below `low`, and its midpoint takes enough bits for
  that unit to stay below half of `low` (`_hull_from`).
- **Division.** Up to 63 bits the quotient is the Complex quotient of the
  midpoints widened by `(R_a |m_b| + |m_a| R_b) / (|m_b| (|m_b| - R_b))`;
  beyond, `a conj(b) / |b|**2`, whose norm now comes from the magnitude bounds
  where the squares' norm seems to reach 0, so it stays finite wherever the
  divisor's rectangle excludes 0 and costs nothing for a narrow divisor.
  The midpoint quotient at every precision was measured too: 7% slower at
  256 bits and 24% at 1024, and no tighter for a narrow divisor, so the
  composition stays.
- **`log` at `|m| = 1`.** The medium kernel declines at and next to 1; the
  certified real logarithm of the exact `|m|**2` takes over.
- **`sqrt` near the cut.** The real part's `(|z| + x)/2` comes from exact
  bounds of `|z|` and `x`, and the imaginary part is the narrower of `y /
  (2 Re)` and `[-u, u]`.
- **`tan` and `tanh` with a wide hyperbolic part.** `cosh` varies over orders
  of magnitude across such a rectangle, and the ratio of two such balls is
  lost. Divided through by cosh, the parts are `sin * sech` and `tanh` over
  `1 + cos * sech`, with `sech` in `[0, 1/cosh(min |v|)]` and `tanh` bounded:
  `tan(-1 + [-84, -12] i)` is `[-3e-11 +/- 4e-11] + [-1 +/- 4e-6] i`, where
  Arb gives `[-1 +/- 9.4] + [-0.75 +/- 8.4] i`.

Three cases remain indeterminate, each a `gammainc` or `betainc` parameter
ball reaching 0 and below, outside the domain these functions keep (scipy's,
`a, b > 0`). A sampling check with mpmath confirmed that every finite ball of
the run, 5,869 of them, contains the function at its sampled points.
