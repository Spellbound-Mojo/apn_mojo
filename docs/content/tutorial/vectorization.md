# Vectorization with vmap and lift

Write a calculation for one value, then apply it to many values with `vmap`.
Use `lift` when a function combines two values and you want a reduction,
running accumulation, outer product, or broadcasting call. Both reuse the
scalar function's arithmetic and error handling.

## Choose the interface for the task

| Task | Start with |
|---|---|
| Apply a built-in function, such as `exp`, to a batch | `batch.exp(values)` |
| Apply your own function independently to elements, rows, or columns | `vmap[f](...)` |
| Combine values with a binary function or evaluate all pairs | `lift[f](...)` |
| Compute a standard total, product, or dot product | `batch.sum`, `batch.prod`, or `batch.dot` |

Here vectorization means applying a scalar or slice calculation across
arrays. Eligible work may use CPU threads; arbitrary-precision numbers do
not become native SIMD lanes. The scalar numerical guarantees still apply.

## Start with one scalar calculation

This example computes squared distances from a center. The same function
handles one position, a vector of positions, or every position-center pair.
There is no separate batch implementation of the formula.

<!-- example: docs/examples/vectorization.mojo -->

In `vmap[squared_distance](in_axes=(0, None))`, brackets select the function
at compile time. The first call configures its mapping; the next supplies
the data. Axis `0` visits the positions, while `None` shares the center
across every call. A shared argument can also be a whole batch.

Pass changing settings as explicit arguments rather than capturing local
variables in a closure. For native variables, construct the intended APN
type, such as `Integer(center)`. Arbitrary-width integer literals can be
passed directly and remain exact.

For library functions, select a family declaration such as `integer.add`
or `float.sqrt`. Root functions such as `apn_mojo.add` are overloaded
dispatchers and cannot serve as one compile-time function value. A mapped
`context=` is forwarded to the scalar function when its signature accepts it.

## Map rows or columns

Mapping an axis removes that axis from each input seen by the function.
For a `[2, 3]` matrix, `in_axes=0` passes two rows of shape `[3]`, and
`in_axes=1` passes three columns of shape `[2]`. That is why `row_total`
accepts a `Batch[Integer]`, even though each call returns one Integer.

A function taking a scalar Integer cannot consume a whole row. Nest mappings
to reach individual entries of a higher-rank batch; the
[composition example](../reference/batch.md#composing-mappings) shows how.
For built-in elementwise work across any rank, `batch.*` already handles it.

`out_axes` chooses where the new mapped axis goes in the output. A Bool
result becomes a `Mask`; a tuple produces a corresponding tuple of outputs.
Batch results must have a consistent shape. Empty mappings with batch
results need `out_shape` when no call can establish that shape. See
[rows, columns, and tensor results](../reference/batch.md#rows-columns-and-tensor-results)
for a runnable example of these options.

## Combine values with lift

The lifted addition in the example supports these calls:

| Operation | Meaning |
|---|---|
| `reduce(values, axis=None)` | Combine every element into one scalar |
| `reduce(values, axis=0)` | Combine along one axis; this is the default axis |
| `accumulate(values, axis=0)` | Return every running prefix along an axis |
| `outer(a, b)` | Evaluate every pair, with shape `a.shape() + b.shape()` |
| A direct call on two arguments | Apply the binary function over broadcast-compatible shapes |

Use `keepdims=True` to retain reduced dimensions with size one. An empty
reduction needs an `identity` configured on the lifted function or an
`initial` value supplied to the reduction. For nonempty input, `initial`
is combined before the selected elements.

`associative=True` promises that regrouping the function's calls leaves
the result unchanged. Exact integer addition satisfies this promise, so
long reductions can combine partial totals in parallel. It does not require
commutativity: operand order is preserved. Leave the default `False` for
subtraction and other order-sensitive functions. A fold combines values of
one result family; outer products and broadcasting also accept supported
functions with different input and result types.

## Keep the shape rules separate

`vmap` pairs entries along mapped axes, whose extents must match. It does
not repeat a one-element vector to match a longer mapped axis. Pass a scalar
to share one value instead. `batch.*` elementwise functions and lifted
binary calls align trailing dimensions and allow size-one dimensions.
Batch operators have an additional restriction on two rank-one vectors:
their lengths must match.

<!-- example: docs/examples/batch_broadcasting.mojo -->

Masks used for selection must match the selected shape. `batch.where`
broadcasts its two value arguments to the mask's shape; the mask itself
does not broadcast.

## Choose how a fold rounds

Integer and Rational folds remain exact. Float and Complex reductions and
accumulations keep intermediate values exact by default, then round each
returned total or prefix once. `context=` controls that final rounding.
This is useful for sums and products, but an intermediate such as `1/3`
cannot be held exactly in a finite binary representation and raises.

Family functions can also take mixed exact inputs. In this example, the
Rational inputs `1/3` and `2/3` reach the first addition exactly, so their
sum is `1` before it is rounded to a Float. Ball accumulation instead carries
an enclosure from one step to the next.

<!-- example: docs/examples/lift_inputs.mojo -->

Use `lift[f](exact=False)` when rounding after each step is the calculation
you intend. A rounded addition is generally not associative, so do not
promise associativity for that mode. Ball folds carry an enclosure through
each call and receive `BallContext`; `exact` does not change their behavior.

The [lifting reference](../reference/apn_mojo.md#lifting-binary-functions)
includes a runnable comparison of exact and stepwise rounding, as well as
empty reductions and in-place accumulators.

## Write functions that can run independently

Mapped functions must be pure: avoid printing, shared mutation, random state,
and other effects whose outcome depends on call order. A failed element may
be called again to obtain its error message. The library discards partial
output and reports the lowest failing logical index.

Eligible long mappings can use the worker pool; short runs and some
signatures stay on the calling thread. Set the thread limit as shown in
[thread controls](../reference/batch.md#threads). Parallel execution does
not relax numerical guarantees or the associativity promise. The
[mapping reference](../reference/batch.md#pinned-compiler-limits) lists
signature limits and supported result types.
