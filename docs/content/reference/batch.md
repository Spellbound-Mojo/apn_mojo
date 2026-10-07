# apn_mojo.batch

This package provides multidimensional batches, boolean masks, `vmap`, and
NumPy-style array functions. Copies and views keep their values across
updates. Start with the
[batch tutorial](../tutorial/batches.md) and the
[vectorization tutorial](../tutorial/vectorization.md); the full `lift`
contract is described in [apn_mojo](apn_mojo.md).

<!-- api: batch -->

## NumPy-style functions

Import `batch` with `from apn_mojo import batch` for calls familiar from
`numpy` or `jax.numpy`: `batch.exp(xs)`, `batch.atan2(ys, xs)`, and
`batch.where(mask, a, b)`. Scalar functions stay at the package root, keeping
`apn_mojo.exp(x)` and `batch.exp(xs)` unambiguous.

<!-- example: docs/examples/batch_functions.mojo -->

| Group | Functions |
|---|---|
| Arithmetic | `add`, `subtract`, `multiply`, `divide`, `reciprocal`, `maximum`, `minimum`, `clip`, `abs` |
| Powers and roots | `sqrt`, `pow`, `pow_int`, `rootn` |
| Elementary | `exp`, `expm1`, `exp2`, `log`, `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`, `sin_cos`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Special | `gamma`, `gammaln`, `digamma`, `polygamma`, `beta`, `betaln`, `poch`, `erf`, `erfc`, `erfi`, `erfinv`, `ndtr`, `log_ndtr`, `ndtri`, `expi`, `sici`, `shichi`, `fresnel`, `lambertw`, `zeta`, `gammainc`, `gammaincc`, `betainc`, `hyp1f1`, `hyp2f1` |
| Integers and parts | `floor`, `ceil`, `trunc`, `round`, `comb`, `conjugate`, `real`, `imag`, `angle` |
| Selection | `where` |
| Constructors | `zeros`, `ones`, `full`, `arange`, `linspace` |
| Joining | `concatenate`, `stack` |
| Reductions | `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `dot`, `vdot`, `cumsum`, `cumprod` |

Each elementwise function uses `vmap` to call the same family declarations as
its root counterpart. Every element therefore follows the scalar function's
rules:

- **Families.** A function takes the families that declare it. Integer and
  Rational elements give correctly rounded Floats under a `context=`, Complex
  elements give Complex numbers, and Ball and ComplexBall elements give
  enclosing balls. The arithmetic functions keep exact families exact without
  a context, as the operators do: `batch.divide` of Integer batches gives
  Rationals.
- **Broadcasting.** Elementwise arguments align trailing dimensions, which
  must match or have size one. This includes one-element vectors, unlike
  operators on two vectors. A scalar argument, including an integer literal
  of any width, is shared by every element; pass a native `Int` as
  `Integer(n)`, as for `vmap`. Each
  elementwise function needs at least one batch argument. `where` broadcasts
  its value arguments to the mask's shape; the mask itself does not broadcast.
- **Shapes.** Results have the broadcast shape; a function with two results,
  such as `sin_cos`, returns a batch of each.
- **Errors.** A failing element raises the scalar function's error with that
  element's position.

Choose a constructor's element type with a parameter, much as you would use
`dtype` in NumPy:
`batch.zeros[Integer]([2, 3])`, `batch.ones[Float](4, context=c)`.
`arange` and `linspace` compute each element exactly; a Float element is the
exact value rounded once, so no rounding error accumulates along a range.
`cumsum` and `cumprod` keep each running Float or Complex value exact and round
it once, as `sum` does.

## Shapes and axis views

`Batch[T]` holds Integer, Rational, Float, Complex, Ball, or ComplexBall values.
Rank and shape are runtime properties: one type serves rank-zero scalars, vectors,
matrices, and higher-rank arrays. Ball batches have container operations,
mapping through `vmap`, binary operations through `lift`, and the NumPy-style
functions, such as `batch.add`. Real Ball batches support `min` and `max`;
both ball families support `cumsum` and `cumprod`. Use `lift` for other
folds. Ball batches have no arithmetic operators or batch JSON.

<!-- example: docs/examples/ranked_batches.mojo -->

Dimensions can have size zero. Printing uses nested brackets for the shape.
Slices, axis views, transpositions, and suitable reshapes share element
storage. Some operations on noncontiguous views gather elements into a new
buffer. All returned batches keep their values when the source is updated.

## Higher-rank arithmetic

Integer, rational, float, and complex batch operators use the corresponding
scalar arithmetic and broadcast compatible shapes.

<!-- example: docs/examples/ranked_arithmetic.mojo -->

Broadcasting aligns trailing dimensions, which must match or have size one.
For example, `[columns]` can broadcast over `[rows, columns]`. Two rank-one
vectors are a special case: they must have equal lengths, even if one length
is one. A scalar operand broadcasts over every element.

Result families follow scalar rules. Dividing integer batches gives rationals;
mixing with floats or complex numbers gives those rounded families. Comparisons
return a mask. In-place updates can broadcast the right operand but cannot
expand the destination shape.

Masks must match shape and do not broadcast. Selection returns values in
row-major order. Reductions accept axes and optional `keepdims`; see their
individual declarations.

| Interface | Two vectors of lengths `n` and `1`, where `n > 1` |
|---|---|
| Arithmetic operators | Shape error |
| Elementwise `batch.*`, such as `batch.add` or `batch.atan2` | Broadcast to length `n` |
| A direct lifted call | Broadcast to length `n` |
| `vmap` over both vector axes | Extent mismatch; share a scalar instead |

The [vectorization tutorial](../tutorial/vectorization.md#keep-the-shape-rules-separate)
demonstrates these differences.

## Mapping functions

Create a mapped function with `vmap[f]()` and call it with batch or scalar
arguments. The function is a compile-time parameter. Use a family declaration
such as `apn_mojo.float.add`, rather than the overloaded `apn_mojo.add`.

<!-- example: docs/examples/vmap.mojo -->

`in_axes` chooses the mapped axis of each argument, defaulting to zero.
A scalar is shared across calls; `in_axes=None` shares an entire batch.
Scalars are library numbers, masks, Bools and literals. Convert a native
number first, such as `Integer(n)` for an `Int`: if natives were accepted, a
wide integer literal could bind as an `Int` and lose its value.
Mapped axes must have equal extents. A `context=` argument is forwarded to
functions that accept it.

Functions can be named or lambdas, must use supported signatures, and must
be pure. Captured local closures are not supported. Pass runtime values as
arguments instead.

### Rows, columns and tensor results

<!-- example: docs/examples/vmap_tensors.mojo -->

A batch parameter receives the dimensions left after mapping removes an axis,
such as a row or column of a matrix. Scalar parameters receive elements.
`out_axes` positions the new mapped axis, defaulting to zero. Batch results
must all have the same shape. Empty mappings need `out_shape` when no call can
supply the shape.

### Tuple results

<!-- example: docs/examples/vmap_structures.mojo -->

Functions can accept or return flat tuples. Each leaf can have its own axis
setting. A failure discards all partial fields. Nested tuple trees are not
supported by the current mapping interface.

### Optional results

Supported one- or two-argument functions returning an Optional number produce
a value batch and a mask. Missing values use the family's zero as a placeholder;
select with the mask before treating them as results. For example,
`var roots, found = vmap[integer.iroot_exact]()(xs, 2)` gives roots and their
presence mask. This Optional form takes no context keyword; use a plain
function with fixed numerical settings when needed.

### Composing mappings

<!-- example: docs/examples/vmap_composition.mojo -->

Nest mappings with `.vmap(...)` or a list of axes. For example,
`vmap[f](in_axes=[0, 1])` corresponds to
`vmap[f](in_axes=0).vmap(in_axes=1)`. Eligible nested mappings use one flat
execution plan; other signatures use the general mapping path.

### Parallel mapping

<!-- example: docs/examples/vmap_parallel.mojo -->

Long eligible mappings use the worker pool. Number and Bool results, including
flat tuples of them, can use this path with numeric or batch arguments, across
strided, reversed, transposed, and nested layouts.

Short runs and single-core environments stay on the caller thread. So do
batch-returning functions, functions taking Bools or Masks, tuple arguments
with more than three leaves, and mappings with `axis_size` or `out_shape`.
See [execution](../architecture/batches.md#flat-and-general-execution).

Parallel evaluation gives the same results and lowest failing logical index
as sequential evaluation. The executor may call a failed element again to
obtain its error message, so mapped functions must be pure.

### Pinned-compiler limits

The interface supports up to three separate positional arguments, or one flat
tuple, with the documented context forms. Use a tuple adapter for a wider
argument list. A function must accept and return types that mapping supports;
a custom result struct is not automatically a new batch element type.

### Boolean mappings

Bool outputs become shaped masks. Masks support shape inspection, reshaping,
transposition, indexing, `.any()`, `.all()`, and `.count()`.

## Construction and iteration

Construct from a list of family values, or use a family's native-input helpers,
such as `Batch[Integer].from_native([1, 2, 3])`. Iteration reads a snapshot in
logical order and returns independent element values. Container operations
are shared across all supported families; conversion helpers vary by family.

## Printing

`print(values)` and `String(values)` use each family's text form, nested to
match the shape. Integer vectors print like `[4, -1, 0]`, Float vectors like
`[0.25, 1.0]`. Empty vectors print as `[]`.

## Native values

Use `to_list()` to keep the APN element types, or `to_native[dtype]()` to
convert to native numbers. Both produce a flat list; preserve `shape()`
separately for a multidimensional result.

`values.to_native[DType.float64]()` returns the elements as a
`List[Float64]` in row-major order, the counterpart of `from_native`.
To `float64`, `float32`, `float16` or `bfloat16` each
value is rounded once, subnormals included, with `rounding=` (nearest-even by
default); beyond the range it becomes an infinity. Balls convert through their
midpoints. To an integer type each value converts exactly, and a value that
is not a whole number in range raises; round first with `batch.round` or
`batch.floor`. Integer output is available for Integer, Rational, and Float
batches; Ball batches export only floating-point midpoints. Take `batch.real`
and `batch.imag` of a Complex batch first.

<!-- example: docs/examples/batch_native.mojo -->

The returned `List` has a runtime length. Mojo's `Array[T, N]` requires `N`
at compile time, so the example checks the length before filling a fixed-size
array. Converting a Ball's midpoint discards its uncertainty; keep the balls
or export endpoints separately when the receiving calculation needs bounds.
For a ComplexBall batch, extract the real and imaginary Ball batches first.

## Threads

Long operations and mappings run on a pool of POSIX threads, with the results
and errors of a sequential loop. `set_num_threads(n)` sets how many threads
take part, the calling thread included, and `get_num_threads()` reads it;
`n` may exceed the core count, and `1` runs everything on the calling thread.
The default is the `APN_MOJO_NUM_THREADS` environment variable when it is a
positive integer, else one thread per usable physical core. The call never
waits: made while an operation runs, even from inside a mapped function, it
takes effect from the next operation.

<!-- example: docs/examples/batch_threads.mojo -->

To choose the default at launch, set the variable before starting your own
program:

```sh
APN_MOJO_NUM_THREADS=4 pixi run --locked mojo run -I src your_program.mojo
```

The environment is read when the pool is first initialized; use
`set_num_threads` to change the count during a run. The setting is
process-wide. Changes wait for a running operation to finish, and growing
the pool restarts its workers. Configure it outside mapped callbacks.

The count is the available participation limit, not a promise that every
call uses that many threads. Short work, unsupported mapping signatures,
and calls made while the pool is busy can run on the caller. On platforms
without POSIX-thread support, `get_num_threads()` returns `1`. On Linux,
the default also respects the process's CPU affinity.

## Operators and shapes

Operators depend on the family: integer batches have bitwise operations and
integer division, while rational, float, and complex batches expose their own
arithmetic. Integer `/` produces a rational batch. Read the generated
declarations for mixed operands and return types. Ball batches use mapped
ball functions instead of operators.

## Indexing and assignment

Use the source forms documented for the destination family. A scalar fills
the selection; a replacement sequence must match it. For overlapping
assignments, the batch reads the old source values before changing the
destination. Compound selection updates such as `values[mask] += 1` follow
the same rule.

## Sharing and copies

Views share element storage but have their own layout. Updating either batch
leaves the other unchanged. A retained view can keep a large backing buffer
alive. To detach a small integer selection, for example, construct
`Batch[Integer](values.to_list())`.

## Empty inputs and failures

An empty mapping makes no scalar calls, so there is no scalar domain check.
Shape, axis, and configuration checks can still fail. An element failure
leaves update destinations unchanged and reports the earliest failing logical
index.

## Rational batches

`Batch[Rational]` provides exact fractional arithmetic, comparisons, and
updates. JSON stores canonical fractions, using version 1 for vectors and
version 2 with a shape for other ranks.

## Float batches

<!-- example: docs/examples/float_batches.mojo -->

<!-- example: docs/examples/float_batch_arithmetic.mojo -->

Each Float element retains its format. Integer and rational operands keep
their exact value until the operation's final rounding.

### Unary operations and powers

<!-- example: docs/examples/float_batch_powers.mojo -->

Unary operations preserve formats. Powers take a scalar integer exponent or
an integer batch and round each element once.

### Strict functions and integral conversion

<!-- example: docs/examples/float_batch_functions.mojo -->

Use `batch.sqrt` and the other NumPy-style functions, or map family functions
such as `fma` with `vmap`.
Batch `.floor()`, `.ceil()`, and `.trunc()` return integer batches.

### Float compound updates

<!-- example: docs/examples/float_batch_updates.mojo -->

Compound operators keep destination formats and use nearest-even rounding.

### Float batch JSON

<!-- example: docs/examples/float_batch_json.mojo -->

JSON preserves each value's format and the default format for empty results.

## Complex batches

<!-- example: docs/examples/complex_batches.mojo -->

<!-- example: docs/examples/complex_batch_arithmetic.mojo -->

<!-- example: docs/examples/complex_batch_functions.mojo -->

Complex batch operations follow the scalar rules and round each output
component once. Multiplication and similar operations use both input parts
together to produce each component.

### Complex compound updates

<!-- example: docs/examples/complex_batch_updates.mojo -->

Compound operators keep the destination component formats. The destination
changes only after the whole update succeeds.

### Complex batch JSON

<!-- example: docs/examples/complex_batch_json.mojo -->

JSON preserves both component values, their formats, and container defaults.

### Function and arithmetic rules

Vectors need equal lengths, scalars broadcast, and higher-rank arrays follow
the shape rules above. Use family functions for mapping and root functions
for reductions.

## Masks

A `Mask` stores boolean values and a shape. It can come from a comparison,
a mapped predicate, or boolean input.

### Iteration

Masks iterate in logical order. They retain their bits when numerical inputs
change, rather than reevaluating a predicate.

### Boolean operations

Combine same-shaped masks with `&`, `|`, `^`, and `~`. A direct conversion to
Bool raises; use `.any()` or `.all()` for a condition.

### Selection

`batch[mask]` collects matching values. `batch[mask] = source` updates those
positions while preserving the destination shape.
