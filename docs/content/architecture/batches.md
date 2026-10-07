# Batches and execution

`Batch[T]` adds shape and layout to scalar values. Views share their storage,
and common executors handle mapping and reductions. This keeps the numerical
rules in scalar functions while the batch layer manages traversal and work
distribution. The [Batch reference](../reference/batch.md) defines public
behavior; private layouts and scheduling thresholds may change.

## Batch records

A batch holds a shared list of scalar elements, a shared layout descriptor
(shape, strides and offsets), and default format metadata where needed. Rank
and shape are runtime properties, so the same type represents rank-0 values,
vectors, matrices and higher-rank arrays.

Copying a batch shares its storage. An update installs a new value for the
destination without changing earlier copies or views. Each stored element is
a scalar handle: `Integer`, `Rational`, `Float`, `Complex` or `Ball`. Mapped
functions can borrow these handles directly and move results into output lists.
Float and Complex elements retain their own formats and special values.

Slicing, reversal, transposition, broadcasting and contiguous reshaping share
element storage and describe the new layout. Creating a view may allocate a
layout descriptor; it does not copy every scalar. Operations walk logical
indices through that layout, gathering values only where their implementation
needs a contiguous buffer. Internal interfaces that erase the element type use
`_ArrayData`, which carries a family tag, element storage, layout and formats.

### Construction from iterators

`batch/_iterator.mojo` consumes an iterator once. For each element, it calls
the family's converter, checks the next list size, runs any failure checkpoint,
and moves the converted value into the list. The tensor becomes available
only after the iterator finishes. An exception destroys the partial list.

Converters keep their own accepted types and diagnostics. Same-family values
move into storage; native integers keep their full width. Constructors still
handle batch shapes and default formats, including the unread part of a batch
iterator. Ball uses the same driver with a same-family converter.

### Element families

`batch/_families.mojo` lists the element families once. That list drives
the buffer variants in `_ArrayData`, runtime tag dispatch, and the types that
`vmap` can collect. Construction, views, indexing, mask selection, assignment,
iteration and printing share one implementation across the families.

Arithmetic operators, comparisons, JSON and conversions between families have
separate support requirements. Integer, Rational, Float and Complex provide
these operations; Ball provides container, mapping and lifting support.
Use scalar Ball functions through `vmap` or `lift` for ball arithmetic.

A scalar family declares `_BatchElement` and supplies a zero that needs no
heap storage. The mapper uses that zero for missing Optional results and for
slots whose tuple fields have been moved out. Functions accepting several
numeric families use an argument type implementing `_MapAdapter`; this keeps
the element conversion with the scalar function's interface.

To add a container family, add it to the family list, implement `_BatchElement`
and its zero, and register any new context type with `vmap`. Its function
argument type, if needed, implements `_MapAdapter`. Compile-time checks keep
the marker trait and family list consistent.

A `Mask` stores packed bits and a runtime shape. Masks require matching shapes,
never broadcast, and select values in logical row-major order. Views and owned
batches use the same public type, so selections such as `x[mask] += 1` and
`x[1:3] *= 2` use the same assignment rules.

Masks and numeric batches use the same checked shape product, transpose
calculation, and axis-selection plan. A Mask transpose gathers values from the
permuted layout. Axis selection uses the shared plan with a direct block gather,
avoiding temporary full layouts. Both operations produce fresh packed storage.
Mask-specific shape and permutation diagnostics stay at the Mask boundary.

The axis-selection plan accepts a validated shape. Selecting an empty axis
fails before computing a size. Otherwise, any empty axis remains in the result;
a nonempty result's dimension product cannot exceed the source product. This
lets the plan reuse the source's addressability guarantee.

## Operator rules

Elementwise arithmetic, comparisons and updates call scalar kernels through
`batch/_parallel.mojo`. Two rank-one batches must have equal lengths. Higher-rank
operations align trailing dimensions, which must be equal or one; broadcasting
uses views over the input storage.

`_prepare_binary_shape` in `batch/value.mojo` checks shaped operands once
and selects a shared shape or a broadcast target before entering a family
kernel. It carries
the result shape through arithmetic and comparison paths. Rank-one operators
keep their equal-length rule. `lift` uses the same trailing-dimension and
stride helpers, but permits singleton broadcasting for vectors too; `vmap`
requires equal mapped extents.

Operator result types come from `_Sum` and `_Quotient`, the aliases used by the
operators themselves. Tests check the types of public arithmetic expressions;
there is no separate promotion model maintained just for the tests.

Mixed operations borrow their operands without first constructing a converted
batch. A shared scalar operand is prepared once and borrowed by each call.
Results follow logical row-major order, even when workers evaluate calls in
parallel. Operators do not fuse: `a * b + c` creates an intermediate product
batch. Mapping a scalar function that computes the whole expression avoids
that intermediate batch, while still allocating the output and any scalar
arithmetic temporaries.

## Mapping functions

`vmap[f]` maps a pure function with a supported signature. `f` is a compile-time
parameter; Mojo 1.1 cannot pass runtime function values across worker threads.
The [mapping reference](../reference/batch.md#mapping-functions) gives the
supported argument and result types.

- `in_axes` chooses an axis for each argument, or `None` to share it.
  `out_axes` chooses where to place the result's new dimension.
- Numeric results form batches, Bool results form masks, and batch results
  stack into a larger batch. The first result supplies any inferred shape.
- Flat tuples map field by field, with independent axis settings.
- `.vmap()` adds another mapping level.
- Empty mappings do not call the function on invented inputs. Supply
  `out_shape` when a returned batch's shape cannot otherwise be inferred.

Mapped functions borrow stored inputs and move results into output lists.
Measure mapping overhead against a scalar loop using the same values;
the result depends on the function, layout and batch size. See the
[benchmark guide](../contributing/benchmarks.md).

The signature wrappers share nesting logic and execution runners. Calls with
one, two or three arguments use one runner per arity, whether they return a
single leaf or a tuple. Four-argument calls, such as `hyp2f1`, have a runner
on the general path only, since each call costs far more than the flat loops
save. A small adapter supplies the unused context for plain
callbacks. Functions taking a tuple use one runner for both result forms.
Typed readers and specialized flat loops still borrow ordinary arguments;
only a callback that takes a tuple needs its argument tuple assembled.

`_MappingResults` describes result handling at compile time. `_LeafResults`
collects one leaf; `_TupleResults` collects each field in order. The runners
share output validation, error handling and traversal, then ask the adapter
to finish the outputs. Both adapters reuse the existing collectors and flat
output builders, including their ownership records and shape rules. There
is no runtime result-type dispatch.

A callback returning an empty tuple still runs at every mapped position and
can fail. Empty mappings keep their existing shape-hint requirements. If a
later tuple field fails validation, earlier fields are discarded with it;
the caller receives a complete result or an error.

### Flat and general execution

The flat path handles numeric or Bool results, including flat tuples, when the
arguments and axes can be described by one plan. It supports transposed,
strided and reversed inputs, shared arguments, nonzero `in_axes`, and reordered
`out_axes`. The planner in `batch/mapping.mojo` resolves extents and strides at
each mapping level.

A flat run:

- reads a single strided run directly, or advances a cursor through a more
  general layout;
- passes a row or column argument as a view, using a remaining-axis layout
  prepared once per chunk;
- expands a flat tuple argument of up to three fields into argument slots;
- moves numeric results into lists and packs Bool results into masks;
- collects supported Optional results as a value batch and a validity mask,
  filling missing values with the family's zero;
- places output axes using views and sends sufficiently long work to the pool.

The general path runs on the calling thread. It handles batch results, Bool
or Mask arguments, tuple arguments with more than three fields, and mappings
using `axis_size` or `out_shape`. It also defines validation diagnostics: the
planner sends unsupported flat plans to this path, and both paths use the same
messages for equivalent failures.

## Lifting binary functions

`lift[f]` supplies broadcasting calls, `outer`, `reduce` and `accumulate` for
supported binary functions on every batch element family. It also supports Bool
functions on masks. Calls and outer products use the same `_FlatInput` readers
as flat `vmap` runs: matching elements are borrowed, and mixed operands go
through `_scalar_argument`. Scalar inputs have rank zero. Broadcasting and
outer products supply their own storage positions to those readers.

An `identity` supplies the result of an empty reduction when no `initial`
value is given. `associative=True` permits chunked reductions whose partials
combine in operand order; it does not require commutativity. The caller must
ensure that this regrouping preserves the function's result.

Float and Complex folds use exact working intermediates by default. A reduction
rounds its final result once. An accumulation rounds each output prefix once,
without feeding that rounded output into the next step. Intermediates must be
finite binary fractions: a result such as `1/3` or `sqrt(2)` raises because
it cannot be represented exactly. Pass
`exact=False` to round each step with the caller's context instead.

Same-family reductions retain their existing kernels. Mixed reductions and
accumulations share `_InputPart`, which holds either an untouched source
operand or a callback result. The first source operand stays exact until the
first callback, including when it is the only element of a parallel chunk.
Only a final singleton result or an emitted prefix converts that operand to
the result family. This prevents early rounding of Rational inputs. Float and
Complex adapters select exact working formats for callbacks, while the fold
tracks source formats separately for output rounding.

Ball folds call the scalar function at each step with `BallContext`. The
resulting enclosure becomes the next accumulator; there is no exact-working
Ball mode. `exact` affects only Float and Complex results. As with any lifted
function, callers must justify `associative=True` before allowing regrouping.

Functions with a context parameter receive the appropriate context through an
adapter; plain functions are wrapped. Bool results become masks. Mask folds use
the same lane machinery with internal zero and one values.

An in-place accumulator such as `lift[update]`, where `update` calls
`acc.__iadd__(x)`, can reuse Integer storage across steps. It may still need to
allocate when the result outgrows that storage.

## Shared elementwise execution

Flat `vmap` calls, lifted calls and outer products, and the Integer, Rational,
Float and Complex arithmetic runners use `_execute_values` in
`batch/_parallel.mojo`.
The executor checks the output allocation size, chooses the execution path,
stages results, and handles cleanup and failure replay. Existing indexed
kernels reach it through `_map_tensor` or `_map_values`.
Packed comparisons use the same executor with one mask tile per result.

Each frontend supplies two small contracts:

- An `_ElementKernel` owns one chunk's readers. It starts at a logical index
  and computes successive elements from a borrowed job. Flat mapping keeps
  its cursors and lent Float or Complex argument records here; lifted calls
  supply broadcasting or outer-product positions. An indexed arithmetic
  kernel needs no reader state.
- An `_ElementErrors` policy formats the failing index and the error raised
  by the scalar call. This preserves nested `vmap` coordinates, lifted
  coordinates, and each arithmetic family's messages.

Reader state belongs to a chunk, never to the shared job. The job outlives
every chunk and any replay. The pool receives an immutable pointer whose
origin keeps that borrow tied to the calling scope. Jobs need not be copyable,
and dispatch does not clone their plans or retained inputs. A failed parallel
run releases all written values
before the executor constructs a fresh kernel at the lowest failing index and
repeats that call. Both reader setup and scalar failures pass through the
frontend's error policy. A successful run transfers the complete output and
its ownership record to the caller.

Error policies borrow their shape metadata on the calling thread and build
coordinates only after a failure. They do not add a shape copy to successful
calls or travel to worker threads.

Each adapter owns shape checks, exact input conversion, output formats, and
scalar arithmetic. Complex format checks and the speculative stored-pair path
keep their error precedence. The executor leaves promotion and rounding to
these family rules.

This contract covers runs with one stored result per call, including flat tuple
results. General `vmap` result collectors still validate dynamic result shapes
on the calling thread. Reductions and accumulations keep their lane kernels:
their ordering and rounding rules differ from independent elementwise calls.
They share the worker pool and storage machinery below the executor.

## Parallel execution

`common/_threads.mojo` and `batch/_parallel.mojo` provide a persistent worker
pool whose default size follows usable physical cores, or a positive
`APN_MOJO_NUM_THREADS` setting. `set_num_threads` changes the participation
count, including the caller; growing the pool restarts the workers between
jobs. See [thread controls](../reference/batch.md#threads). Workers take chunks
through atomic counters and write results into separate slots. If several
calls fail, the executor reports the lowest failing logical index.

The failure counter has its own shared allocation, retained until all workers
finish. With an inline atomic field in the mutable slot record, a Mojo 1.1 run
could lose a worker's recorded failure: a lane wrote 997 of 1,000 outputs,
recorded its error, and the caller later read the no-error sentinel. Keeping
the atomic outside that record prevents an incomplete result from being
published and lets cleanup destroy only the outputs that were written.

Most runners consider the pool only for at least 64 elements; flat `vmap`
keeps its initial probe for shorter runs too. For eligible runs, the calling
thread times the first 16 elements. It sends chunks to workers when the
remaining work is estimated to take at least 100 µs. If the pool is busy,
including during a nested call, the calling thread continues the work.

Elementwise operations, reductions, flat mappings and unlimited batch JSON
writes can use the pool. Unlimited JSON reads parallelize numeric conversion
for Integer, Rational and sufficiently wide Float or Complex values. The
general mapping path and conversions with limits stay on the calling thread.

### Owner-affine memory release

Freeing an allocation on a different CPU can incur cache and allocator costs.
Parallel runs record which worker owns each chunk. When the last reference to
a large result (at least 1,024 elements) is released, its chunks return to
their owning workers for destruction. Multi-CPU measurements are included
in the full report linked from [Benchmark results](benchmark-results.md).

## Sharing and lifetime

Views retain the underlying element storage and a layout descriptor. Updating a
parent batch leaves existing views intact. Reversed, strided and transposed
views expose logical indices, which also determine serialization order and error
indices. Mask selections gather the selected handles into a new contiguous list.

These ownership rules protect saved values during ordinary updates. They do not
make concurrent mutation of the same batch variable safe.

Element readers use `_borrow_values` in `batch/_values.mojo` to obtain an
immutable `Span`. Storage, strided views, mapping readers and reduction lanes
share this access path. The span does not copy elements or acquire another
storage reference. Its origin is the reader that already retains the storage;
even the pointer used by a linear lifted fold is read-only and carries that
origin.

Mojo 1.1 cannot infer this relationship through the retained owner and list,
so the helper contains an explicit origin cast. Each caller must keep the
list alive and leave its storage and borrowed elements unchanged until the
borrow ends. Mapping readers may advance their cursor or replace a converted
argument only after the callback returns. Their return types expose an
immutable borrow even though preparing the next argument mutates the reader.

A Float input can hold a Float run, a Complex component run, or a broadcast
scalar. Its `element` method borrows each representation through one private
helper that ties the reference to the input. This helper may only receive
values retained by that input; it handles the compiler's origin restriction
on optional payloads and component fields. Format inspection uses the same
borrow. The `value` method makes an owning copy when one is needed.

Float dot products use `_borrow_values` for native runs. Their pointers are
immutable and cannot outlive the input. The starting position includes the
selection offset, and the step combines the tensor and selection strides,
including negative strides. Callers must establish a nonempty native run
before requesting its starting pointer.

The separate mechanism that lends Float and Complex significands to argument
records still clears those records before destruction.

## Updates and transactional safety

An update checks shapes and indices, reads the original inputs, and builds
a private result. Overlapping assignments such as `x += x[::-1]` therefore
use the original values throughout. Scalar kernels check domains and normalize
results as they run. The wrapper replaces the destination only after every
step succeeds, preserving its shape and default formats.

A recoverable error discards the private result. The destination, earlier copies
and views keep their previous values. General arithmetic has no allocation quota;
conversion limits apply only to conversion calls.

## Reductions

Built-in reductions and `lift` share `batch/_lanes.mojo`. It traverses selected
axes and preserves dimensions when `keepdims=True`. The
[reductions chapter](reductions.md) explains exact accumulation, ordered
rounding and the arithmetic adapters.

## Resource boundaries

Live elements, spare capacity, layouts and retained views all consume memory.
`ConversionLimits` bounds newly requested storage within one conversion; it
does not cap retained batches, later arithmetic or execution time.
