# Calculations with guaranteed bounds

Use `Ball` when you need to know how much uncertainty a result carries.
It represents every real value between a midpoint minus a radius and that
midpoint plus the radius. Arithmetic includes both the input uncertainty
and any rounding error in the returned bounds.

## From a measurement to a result

Suppose a length is 3, with an uncertainty of one eighth in either direction.
`Ball(3, Rational(1, 8))` represents that measurement. Squaring it gives an
enclosure of every possible area. `BallContext(64)` chooses 64 bits for the
result's midpoint; it does not promise 64 accurate bits in the measurement.

<!-- example: docs/examples/ball_values.mojo -->

`midpoint()` and `radius()` expose the stored description. `lower()` and
`upper()` give outward-rounded endpoints; `lower_rational()` and
`upper_rational()` give the exact endpoints as fractions. The area contains
the squares of both extreme lengths, and may extend farther to keep the
enclosure guarantee.

An exact input can also need a radius: `Ball(Rational(1, 3))` encloses one
third, which has no finite binary representation. `Ball("0.1")` encloses
exactly one tenth. Constructing from an already rounded `Float` instead
encloses that stored value; it cannot recover earlier rounding error.

## Decide what the bounds establish

Balls have no ordinary comparison operators. `ball.compare(a, b)` returns
`less`, `equal`, `greater`, `overlap`, or `undefined`. In the example, every
possible area exceeds 8, so the result is `BallOrder.greater`. Overlapping
measurements do not establish equality. Use `contains` to test a number,
`contains_ball` to test an entire interval, and `overlaps` to test whether
two intervals meet.

An indeterminate ball has no usable enclosure. For example, division by
an interval containing zero includes an undefined calculation. Check
`is_indeterminate()` before trying to inspect finite endpoints. An unbounded
ball instead represents every real number; `is_finite()` distinguishes both
cases from a finite interval.

## Understand why bounds widen

Each operation considers its operand intervals independently. The library
does not remember that two occurrences came from the same measurement.
For an uncertain `x`, `x - x` therefore need not be the exact zero, even
though the algebraic expression is zero. Simplifying the expression before
evaluating it can give much tighter bounds.

Increasing midpoint precision reduces rounding error. It cannot remove
measurement uncertainty or restore relationships lost between intermediate
intervals. To benefit from a higher precision, recompute from the original
exact inputs or measurements. Merely widening the format of an existing
result does not narrow its uncertainty.

## Extract a result when the rounding is certain

`ball.to_float_if_certain` returns an Optional Float. It succeeds only when
every point in the enclosure rounds to the same result with the requested
context and status. In the example, a 128-bit enclosure of `sqrt(2)` is
narrow enough to determine its 53-bit rounded value. `None` means the
enclosure does not establish one result under those rules.

If the uncertainty comes from rounding, retry the whole calculation with
more working bits and a finite retry budget. If it comes from the input,
you may need a better measurement or a less demanding output precision.
The [precision guide](../guides/precision.md) demonstrates a bounded retry
and a separate check for reliable decimal digits.

## Rectangles in the complex plane

`ComplexBall` pairs real and imaginary balls into a rectangle. Use it for
uncertainty in complex arithmetic; `complex_ball.abs` returns a real Ball
for the magnitude. Both component intervals participate in the calculation.

<!-- example: docs/examples/complex_ball_values.mojo -->

`to_complex_if_certain` requires a definite rounded value for both parts.
Functions use their principal branches. When a rectangle crosses a branch
cut, its result covers both sides: the logarithm around a negative real
number can have an imaginary interval spanning from negative to positive
pi. Increasing precision does not remove that jump. A rectangle containing
a singularity, such as zero for `log`, gives an indeterminate result.

## Work with many intervals

`Batch[Ball]` and `Batch[ComplexBall]` support shapes, selections, and mapping.
Use functions such as `batch.add` and `batch.exp`, or pass a family function
to `vmap` or `lift`. Ball batches have no arithmetic operators or batch JSON.
Real Ball batches support `min` and `max`; both families support `cumsum`
and `cumprod`. Use `lift` for other folds, preserving operand order unless
your function is associative.

See the [Ball reference](../reference/ball.md) and
[ComplexBall reference](../reference/complex_ball.md) for set operations,
special functions, serialization, and the full contracts.
