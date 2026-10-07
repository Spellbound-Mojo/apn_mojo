# Batches and selections

## Construct a batch

Use [`Batch[T]`](../reference/batch.md) to work with many integers, rationals,
floats, complex numbers, or balls at once. Start with a vector of integers:
`Batch[Integer]()` is empty, `Batch[Integer](items)` accepts a `List[Integer]`, and
`Batch[Integer].from_native([1, 2, 3])` converts native integers.
`from_iterable(range(4))` consumes a finite iterable.

Integer batches print as lists of decimal values, with `[]` for an empty
batch. Arithmetic operators work element by element. A scalar such as `3`
broadcasts in `values + 3`; two one-dimensional batches must have equal
lengths, even if one has just one element.

## Snapshots and slicing

Indexing returns an independent value. Slices such as `values[:]`,
`values[::2]`, and `values[::-1]` share the stored elements but keep their
values when the original changes. A slice is itself a batch that you can
update; updating it does not change the original.

To change the original, assign through its slice: `values[::2] = 0`.

<!-- example: docs/examples/batches.mojo -->

## Conditional selection and masks

`values < 0` produces a `Mask`. Combine conditions with parentheses and
bitwise operators, such as `(values >= 0) & (values < 10)`. Use `.any()`,
`.all()`, or `.count()` to summarize a mask. Converting a mask to `Bool`
raises because it does not specify which of those decisions you want.

`values[mask]` collects the selected elements in logical order.
`values[mask] = 0` updates those elements. A batch used as the replacement
must have as many elements as the mask selects. Masks keep their bits when
the original numbers change; they do not rerun the comparison.

## Assignment and iteration

Index, slice, and mask assignments read the source before changing the
destination. This lets overlapping assignments such as
`values[1:] = values[:-1]` use the original values throughout. Assignments
keep the batch's shape.

You can update a whole batch with `values += 1`, or a selection with
`values[mask] += 1` or `values[1:3] *= 2`. A failed update leaves the
destination unchanged.

Iteration visits values in logical order using a snapshot of the sequence.
You can update the batch variable during iteration without changing that
snapshot. This guarantee does not cover concurrent mutation from multiple
threads. Use `to_list()` to collect APN elements in a Mojo list. For a list
of native numbers, use `to_native[dtype]()`; see [native output](#native-output).

## More dimensions and other number types

Rank and shape are runtime properties of `Batch[T]`. Higher-rank arithmetic
aligns trailing dimensions, which must match or have size one. Masks must
match the selected shape and do not broadcast. The
[batch reference](../reference/batch.md) covers reshaping, axes, and mapping
rows or columns with `vmap`.

`vmap[f]` applies scalar functions to every batch element family, and `lift[f]`
adds folds and outer products to binary functions. The
[vectorization tutorial](vectorization.md) walks through custom functions,
axes, shapes, and rounding choices.

## NumPy-style functions

If you know NumPy, `from apn_mojo import batch` gives you familiar calls such
as `batch.exp(xs)`, `batch.atan2(ys, xs)`, `batch.where(xs > 0, xs, zero)`,
`batch.zeros[Integer]([2, 3])`, and `batch.cumsum(xs, axis=1)`. The functions
reuse scalar arithmetic: Integer results stay exact, Float results round
once, and balls enclose their results. Elementwise arguments broadcast over
matching or size-one trailing dimensions, including one-element vectors.
A scalar is shared by every element. Unlike these functions, operators on
two vectors require equal lengths. Scalar functions remain available at the
package root.

<!-- example: docs/examples/batch_functions.mojo -->

The [batch reference](../reference/batch.md#numpy-style-functions) lists the
functions and their rules. Integer, rational, float, and complex batches also
provide arithmetic operators, built-in reductions, and JSON. For ball and
complex ball batches, use the NumPy-style functions or map functions from
`apn_mojo.ball` and `apn_mojo.complex_ball` with `vmap` or `lift`. See
[Ball](../reference/ball.md) and [ComplexBall](../reference/complex_ball.md).

## Native output

`values.to_native[DType.float64]()` returns a flat `List[Float64]` in
row-major order. Keep `values.shape()` alongside it to retain dimensions.
Floating-point output rounds each element once; integer output requires
exact whole numbers in range. `to_list()` instead keeps the APN types and
their precision. The [native conversion example](../reference/batch.md#native-values)
also covers fixed-size Mojo arrays, complex components, and Ball midpoints.

## Choose the thread count

Import `set_num_threads` and `get_num_threads` from `apn_mojo`, or set
`APN_MOJO_NUM_THREADS` before starting the program. The count includes the
calling thread; use `1` to keep all work on it. Short operations may run on
the caller even with a larger configured count. See the
[thread example and defaults](../reference/batch.md#threads).
