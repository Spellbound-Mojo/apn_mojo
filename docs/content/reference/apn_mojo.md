# apn_mojo

Import common number types, arithmetic functions, reductions, `lift`, and
integer `axpy` from the root package. For a function to pass to `vmap` or
`lift`, use its single declaration in a family package.

<!-- api: apn_mojo -->

## Lifting binary functions

For a step-by-step introduction, start with
[vectorization with vmap and lift](../tutorial/vectorization.md).

Use `lift[f]()` to turn a binary function into `reduce`, `accumulate`,
`outer`, and a broadcasting call. Functions can work with
Integer, Rational, Float, Complex, Ball, and ComplexBall values; Bool functions
can work with masks. Numeric inputs use the same adapters as `vmap`.

<!-- example: docs/examples/lift.mojo -->

A named family function keeps its scalar input rules. For example,
`lift[ball.add]()` accepts Ball, Float, Integer, and Rational operands;
`lift[float.add]()` accepts Float, Integer, and Rational operands. Exact inputs
stay exact until the scalar function uses them. A callback declared with
concrete `Float` parameters still needs Float inputs. Integer literals keep
their full width; wrap native variables and decimal literals in the intended
library type, just as with `vmap`.

<!-- example: docs/examples/lift_inputs.mojo -->

Two options control how the fold runs:

- `identity=` supplies the empty result. Without an identity or a call's
  `initial=`, reducing an empty selection raises.
- `associative=True` promises `f(f(a, b), c) == f(a, f(b, c))`. It permits
  a long reduction to split into chunks whose results combine in order.
  Commutativity is not required. The default, `False`, keeps a left fold in
  row-major order; independent output lanes can still run in parallel.

For mixed inputs, the associativity promise also covers calls on two source
elements: a chunk can start without `initial`. Keep the default order when
an initial value is needed to make those scalar calls valid.

### Rounded once

Float and Complex `reduce` and `accumulate` keep working values exact by
default and round each returned total or prefix once. Accumulation carries
the exact prefix into the next step, preserving information that its rounded
output may lose. Lifted addition therefore follows the same rule as `sum`.

The function must be able to evaluate each intermediate exactly in the
internal binary representation. Addition and multiplication of binary inputs
can do so within storage limits; results such as `1/3` or `sqrt(2)` cannot be
stored exactly. Exact powers and
square roots work when their results are finite binary fractions. An
unrepresentable intermediate raises instead of silently rounding.

Use `lift[f](exact=False)` for rounding at each step. Do not claim
associativity for an operation whose rounded results depend on grouping.

| Call | Result |
|---|---|
| `reduce(xs, axis=0, keepdims=False, initial=None, context=None)` | Fold along selected axes; negative axes count from the end. `keepdims=True` retains reduced dimensions of size one. |
| `reduce(xs, axis=None, initial=None, context=None)` | Fold all elements into a scalar. |
| `accumulate(xs, axis=0, context=None)` | Prefix folds along one axis, with the input shape. |
| `outer(a, b, context=None)` | All pairs, with shape `a.shape() + b.shape()`. |
| `f_lifted(a, b, context=None)` | Elementwise calls over broadcast-compatible shapes. |

A lifted broadcasting call aligns trailing dimensions and accepts size-one
dimensions, including a one-element vector. Scalars have shape `[]` and
broadcast across a batch. In `outer`, a scalar contributes no axes. Two scalar
inputs produce a rank-zero batch, or a rank-zero mask for a Bool result.
Batch operators have a stricter
rule for two rank-one vectors: their lengths must match.

Folds require a `(T, T) -> T` callback; the scalar function's numeric argument
adapters can stand in for `T`. For example,
`lift[rational.add]().reduce(integers, axis=None)` returns a Rational, and
`lift[ball.add]().accumulate(rationals)` returns a Ball batch. Outer products
and broadcasting can use supported `(T, U) -> R` signatures.
Use `vmap` for callbacks that return tuples or batches. Functions must be
pure, except for the dedicated accumulator argument in the in-place form
below. An element failure reports a logical index and discards partial output.

### Contexts

For exact Float and Complex folds, `context=` controls the final rounding of
each result. Without it, formats merge from the inputs. A `ComplexContext`
sets the two output components separately. Integer and Rational inputs supply
no Float format: give a context when the inputs and `initial` are all exact.
Scalar domain rules still apply. Complex arithmetic needs a Complex operand;
for a reduction of real inputs, a Complex `initial` supplies one. A context
alone does not turn real operands into Complex operands.

The first operand of a mixed fold stays exact until the first callback. A
singleton reduction converts it directly to the result family. An accumulation
does the same for its first output, but keeps the original operand for the next
step. Thus adding Rational inputs `1/3` and `2/3` produces exactly `1` before
the final rounding. The restrictions on subsequent exact intermediates still
apply. An empty reduction returns `initial` or the declared identity under the
same rules as a fold over the result family.

For `exact=False`, `outer`, and broadcasting, a context goes to every function
call. The function must accept that keyword. Passing an unused context to a
plain function raises an error.

Ball functions receive `BallContext` on each call. Ball folds run in operand
order by default, carrying the enclosure from one scalar call to the next;
`exact` does not change their behavior. A singleton Ball input, and the first
output of a Ball accumulation, stay unchanged because no callback runs there.
For a mixed input, that first output is enclosed at the context's precision,
or the scalar Ball arithmetic default. The unconverted input remains the seed
for the next call. Ball arithmetic is generally sensitive to grouping, so
leave `associative=False` unless the particular callback guarantees otherwise.

ComplexBall functions also receive `BallContext` and carry each enclosure
forward. Their inputs must be ComplexBall values, as with scalar calls; there
is no implicit conversion from real numbers or real balls.

### Masks

A Bool result from `outer` or broadcasting becomes a `Mask`. A function of two
Bools can reduce or accumulate a mask, with the same axis, identity, and
initial-value choices. Mask folds do not accept a numerical context.

### In-place updates

`lift[update]()` also accepts a function such as `def(mut acc: T, x: T)`.
Updating a reusable integer accumulator can reduce temporary allocations.
Use a named function, or a one-expression lambda calling a method such as
`acc.__iadd__(x)`. Actual gains depend on the values and operation; compare
the same workload using the [benchmark protocol](../contributing/benchmarks.md).

### One reduction driver

Built-in reductions and `lift` share the lane driver in `batch/_lanes.mojo`.
Each family supplies its arithmetic and rounding rules. Separate result lanes
can run in parallel; an associative reduction can also split a long lane.
The [architecture chapter](../architecture/reductions.md) describes the kernels.

## Reductions

Reduction names follow NumPy. Integer and Rational `sum`, `prod` and `dot`
are exact. Float and Complex `sum` and `dot` round the exact total once;
the named ordered modes round at each step.

`min`, `max`, `argmin` and `argmax` compare Integer, Rational and Float values
exactly. The first extreme wins, and a Float NaN propagates (`argmax` gives
the first NaN's index). Without `axis`, `argmin` and `argmax` return the
row-major flat index. `min` and `max` also reduce Ball batches to an enclosure
of the extreme over all points, as Arb's `arb_min` and `arb_max`. Complex
values have no order.

Single-family reductions keep the input family. Mixed integer/rational dot
products return a Rational. Vector dot products need equal lengths. Higher-
rank reductions accept `axis=` and `keepdims=`; see each declaration for the
supported arguments. Inputs remain unchanged.

For custom folds, see [`lift`](#lifting-binary-functions). A practical example
is [Exact totals and weights](../tutorial/rationals.md#exact-totals-and-weights).

## Elementwise extremes and NumPy names

`maximum(x, y)` and `minimum(x, y)` are two-argument functions, one per family,
so `vmap` maps them with broadcasting. They return an operand: the first when
the two are equal, so `maximum(-0, +0)` is `-0` as in NumPy, and a NaN when
either Float is NaN. Exact operands compare exactly; a Float result is the
chosen operand rounded once into the context's format or the merged operand
format, as for `add`. A Ball result contains the extreme over all points.
`clip(x, a_min, a_max)` is `minimum(maximum(x, a_min), a_max)` with one
rounding.

`floor`, `ceil`, `trunc` and `round` (ties to even, as NumPy's) return
Integers; `reciprocal` is `1 / x` in every family, exact for Integer, Rational
and ExactComplex. Complex values have `abs` (the magnitude), `angle`,
`conjugate`, `real` and `imag`. Special functions take scipy.special's names:
`gammaln`, `expi`, `sici`, `shichi`, `fresnel` (the last three return a pair),
`lambertw(x, k=0)`, `ndtr`, `log_ndtr`, `erfinv`, `ndtri`, `beta`, `betaln`,
`poch`, `zeta` (`zeta(x)` is Riemann's, `zeta(x, q)` Hurwitz's), `polygamma`,
`hyp1f1(a, b, x)`, `hyp2f1(a, b, c, x)` and `betainc(a, b, x)` (a Ball x
selects the ball family), `gammainc`, `gammaincc`
and, for Integers, `comb(N, k, repetition=False)` and `factorial2`.

## Scaled addition

`axpy(a, x, y)` computes `a * x[i] + y[i]` into a new integer batch.
Supply a scalar coefficient and two vectors of equal length. The operation
leaves both inputs unchanged and avoids creating a separate scaled batch.

## Float sum

<!-- example: docs/examples/float_sum.mojo -->

`sum(values)` accumulates selected values exactly before rounding. Large
opposing terms can cancel without first overflowing, and small residuals are
retained. The final value does not depend on the order of finite terms.

Without a context, the destination uses the largest selected precision and
requires matching exponent bounds. A context supplies its own format, mode,
and traps. An empty sum uses positive zero and the container's default format,
unless a context overrides it.

NaNs propagate; opposite infinities produce NaN and invalid. A sum of only
negative zeros stays negative. Mixed zero signs or exact cancellation produce
negative zero only when rounding toward negative infinity. Traps apply to the
final result, including its special-value conditions.

## Float dot

<!-- example: docs/examples/float_dot.mojo -->

`dot(a, b)` adds exact pairwise products and rounds once. For a Float result,
one operand is a Float batch and the other can contain Float, Integer, or
Rational values. Exact fractional weights do not round before multiplication.

Vectors must have equal lengths. Slices and masks select the values as usual.
No intermediate product is rounded to the destination range; the final result
sets rounding conditions and traps.

## Complex sum

<!-- example: docs/examples/complex_sum.mojo -->

Complex sums keep real and imaginary totals separately and round each once.
Without a context, each output precision is the largest selected precision
for that component. An `ArithmeticContext` applies to both, while a
`ComplexContext` supplies separate settings. Zero signs and nonfinite values
follow the Float sum rules per component.

## Complex dot products

<!-- example: docs/examples/complex_dot.mojo -->

`dot(a, b)` sums unconjugated products. `vdot(a, b)` conjugates the first
operand, so `vdot(values, values)` sums squared magnitudes. Vectors need equal
lengths. Cross terms remain exact until each final component rounds.
Nonfinite products follow the Complex multiplication rules. `vdot` also accepts
real batches, where conjugation has no effect and it returns the same result
as `dot`.

## Ordered reductions

<!-- example: docs/examples/ordered_reductions.mojo -->

<!-- example: docs/examples/complex_ordered_reductions.mojo -->

Use an ordered mode when the sequence of rounding steps is part of the
calculation:

- `sum_sequential` adds in logical order, rounding each step.
- `sum_tree` uses a fixed adjacent-pair tree and rounds each internal node.
- `dot_sequential` rounds each product, then each addition to the running sum.

Rounding at each step can introduce intermediate overflow or rounding error
that an exact total avoids. A trap fires at the first prescribed step that
raises its condition.
