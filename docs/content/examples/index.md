# Examples

Choose an example below and run it from the directory containing `pixi.toml`.
Each is a standalone program in `docs/examples/`:

```sh
pixi run --locked mojo run -I src docs/examples/<name>.mojo
```

`docs-test` compiles the examples with `--Werror` and checks their output
against `docs/examples.json`.

## Getting started

| Example | What it shows | Page |
|---|---|---|
| `quickstart.mojo` | Exact integers and fractions, Floats of any precision and Complex numbers | [Arbitrary-precision numbers for Mojo: First program](../index.md#first-program) |
| `first_integer.mojo` | The smallest complete program: one exact Integer calculation | [Installation and first run: Work from a checkout](../start/installation.md#work-from-a-checkout) |
| `batch_quickstart.mojo` | Elementwise square roots at a chosen precision, followed by a sum | [Batches, mapping and reductions](../index.md#batches-mapping-and-reductions) |
| `integer_values.mojo` | Integer arithmetic and the three division conventions | [Integer values: Arithmetic and division](../tutorial/integers.md#arithmetic-and-division) |
| `integer_tour.mojo` | A tour of Integer: text, JSON, division conventions, functions and batch updates | [apn_mojo.integer: A tour of Integer](../reference/integer.md#a-tour-of-integer) |
| `recovery.mojo` | Reading an error's remedy and keeping the destination unchanged | [Errors and recovery: Read the remedy](../tutorial/errors.md#read-the-remedy) |
| `conversion.mojo` | Decimal text, JSON and conversion limits for Integers | [Text and JSON: Keep every digit in storage](../tutorial/conversion.md#keep-every-digit-in-storage) |

## Integer functions

| Example | What it shows | Page |
|---|---|---|
| `functions.mojo` | Exact integer functions such as gcd, lcm and isqrt, on scalars and batches | [apn_mojo.integer: Functions on batches](../reference/integer.md#functions-on-batches) |
| `combinatorics.mojo` | Factorials and binomial coefficients without overflow | [Exact combinatorics](../guides/combinatorics.md) |
| `number_theory.mojo` | Modular arithmetic, primality tests and prime search | [apn_mojo.integer](../reference/integer.md) |

## Rationals

| Example | What it shows | Page |
|---|---|---|
| `rationals.mojo` | Exact fractions: construction, arithmetic and powers | [Exact fractions](../tutorial/rationals.md) |
| `rational_input.mojo` | Exact decimal input into Rationals | [Exact fractions: Exact decimal input](../tutorial/rationals.md#exact-decimal-input) |
| `rational_batches.mojo` | Batches of fractions and masked selections | [Exact fractions: Batches of fractions](../tutorial/rationals.md#batches-of-fractions) |
| `rational_batch_math.mojo` | Exact elementwise batch calculations | [Exact fractions: Exact batch calculations](../tutorial/rationals.md#exact-batch-calculations) |
| `rational_reductions.mojo` | Exact totals and weighted sums | [Exact fractions: Exact totals and weights](../tutorial/rationals.md#exact-totals-and-weights) |
| `rational_batch_powers.mojo` | Rounding to integers and signed powers on batches | [Exact fractions: Rounding and signed powers](../tutorial/rationals.md#rounding-and-signed-powers) |
| `rational_batch_updates.mojo` | Transactional compound updates of exact quantities | [Exact fractions: Updating exact quantities](../tutorial/rationals.md#updating-exact-quantities) |
| `rational_batch_json.mojo` | Saving and loading exact batches as JSON | [Exact fractions: Saving exact batches](../tutorial/rationals.md#saving-exact-batches) |

## Floats

| Example | What it shows | Page |
|---|---|---|
| `float_values.mojo` | Float values, formats and exact conversion | [Floats of any precision: Choose a format](../tutorial/floats.md#choose-a-format) |
| `float_formats.mojo` | Formats, rounding modes and traps | [apn_mojo.float: Formats, rounding and traps](../reference/float.md#formats-rounding-and-traps) |
| `float_arithmetic.mojo` | Correctly rounded arithmetic with an explicit context | [Floats of any precision: Arithmetic rounds once](../tutorial/floats.md#arithmetic-rounds-once) |
| `float_functions.mojo` | Square roots, powers and other rounded functions | [Floats of any precision: Functions](../tutorial/floats.md#functions) |
| `float_interchange.mojo` | Float JSON that keeps every bit | [apn_mojo.float: Text and JSON](../reference/float.md#text-and-json) |
| `float_text.mojo` | Decimal and hexadecimal text | [Floats of any precision: Text input and formatting](../tutorial/floats.md#text-input-and-formatting) |
| `float_elementary.mojo` | Constants, exponentials, logarithms, and trigonometric functions | [Constants and elementary functions](../tutorial/floats.md#constants-and-elementary-functions) |
| `special_functions.mojo` | Gamma and beta functions, normal quantiles, and upper-tail probabilities | [Special functions](../tutorial/floats.md#special-functions) |
| `precision.mojo` | Cancellation, exact totals, and certification of 50 decimal digits | [Precision and accuracy](../guides/precision.md) |

## Complex numbers

| Example | What it shows | Page |
|---|---|---|
| `complex_values.mojo` | Complex values with independent component formats | [Complex numbers: Values and components](../tutorial/complex.md#values-and-components) |
| `complex_arithmetic.mojo` | Complex arithmetic rounded per component | [Complex numbers: Arithmetic](../tutorial/complex.md#arithmetic) |
| `complex_functions.mojo` | Magnitudes and square roots | [Complex numbers: Magnitudes and square roots](../tutorial/complex.md#magnitudes-and-square-roots) |
| `complex_powers.mojo` | Integral powers | [Complex numbers: Powers and phase cycling](../tutorial/complex.md#powers-and-phase-cycling) |
| `complex_interchange.mojo` | Complex text and JSON | [apn_mojo.complex: Text and JSON](../reference/complex.md#text-and-json) |
| `complex_elementary.mojo` | Complex exponentials and the principal logarithm | [Complex elementary functions](../tutorial/complex.md#elementary-functions) |
| `exact_complex_values.mojo` | Exact complex fractions, square roots, JSON, and conversion | [Keep complex fractions exact](../tutorial/complex.md#keep-complex-fractions-exact) |

## Interval bounds

| Example | What it shows | Page |
|---|---|---|
| `ball_values.mojo` | Measurement uncertainty, comparisons, and a certified rounded result | [Calculations with guaranteed bounds](../tutorial/balls.md) |
| `complex_ball_values.mojo` | Complex rectangles, magnitude bounds, and branch cuts | [Rectangles in the complex plane](../tutorial/balls.md#rectangles-in-the-complex-plane) |

## Batches and mapping

| Example | What it shows | Page |
|---|---|---|
| `batches.mojo` | Batch snapshots, selections and updates | [Batches and selections: Snapshots and slicing](../tutorial/batches.md#snapshots-and-slicing) |
| `batch_functions.mojo` | NumPy-style construction, broadcasting, selection, and elementary functions | [NumPy-style functions](../tutorial/batches.md#numpy-style-functions) |
| `vectorization.mojo` | A custom scalar function mapped, accumulated, and evaluated over all pairs | [Vectorization with vmap and lift](../tutorial/vectorization.md) |
| `batch_broadcasting.mojo` | Vector shape rules for operators, `batch.*`, `vmap`, and `lift` | [Keep the shape rules separate](../tutorial/vectorization.md#keep-the-shape-rules-separate) |
| `batch_native.mojo` | Native lists and fixed-size arrays, shape, complex components, and Ball midpoints | [Native values](../reference/batch.md#native-values) |
| `batch_threads.mojo` | Thread counts and numerical results with one or two participants | [Threads](../reference/batch.md#threads) |
| `ranked_batches.mojo` | Shapes and axis views | [apn_mojo.batch: Shapes and axis views](../reference/batch.md#shapes-and-axis-views) |
| `ranked_arithmetic.mojo` | Arithmetic with broadcasting across ranks | [apn_mojo.batch: Higher-rank arithmetic](../reference/batch.md#higher-rank-arithmetic) |
| `vmap.mojo` | Mapping a scalar function over batches | [apn_mojo.batch: Mapping functions](../reference/batch.md#mapping-functions) |
| `vmap_tensors.mojo` | Mapping over rows and columns, with tensor results | [apn_mojo.batch: Rows, columns and tensor results](../reference/batch.md#rows-columns-and-tensor-results) |
| `vmap_structures.mojo` | Mapped functions that return tuples | [apn_mojo.batch: Tuple results](../reference/batch.md#tuple-results) |
| `vmap_composition.mojo` | Composing mapped functions | [apn_mojo.batch: Composing mappings](../reference/batch.md#composing-mappings) |
| `vmap_parallel.mojo` | Parallel mapping of a long batch | [apn_mojo.batch: Parallel mapping](../reference/batch.md#parallel-mapping) |
| `lift.mojo` | Lifting a binary function into reduce, accumulate and outer | [apn_mojo: Lifting binary functions](../reference/apn_mojo.md#lifting-binary-functions) |
| `lift_inputs.mojo` | Mixed exact inputs and Ball folds | [apn_mojo: Lifting binary functions](../reference/apn_mojo.md#lifting-binary-functions) |
| `float_batches.mojo` | Float batches and their formats | [apn_mojo.batch: Float batches](../reference/batch.md#float-batches) |
| `float_batch_arithmetic.mojo` | Float batch arithmetic | [apn_mojo.batch: Float batches](../reference/batch.md#float-batches) |
| `float_batch_powers.mojo` | Float batch unary operations and powers | [apn_mojo.batch: Unary operations and powers](../reference/batch.md#unary-operations-and-powers) |
| `float_batch_functions.mojo` | Strict Float batch functions and integral conversion | [apn_mojo.batch: Strict functions and integral conversion](../reference/batch.md#strict-functions-and-integral-conversion) |
| `float_batch_updates.mojo` | Float compound updates | [apn_mojo.batch: Float compound updates](../reference/batch.md#float-compound-updates) |
| `float_batch_json.mojo` | Float batch JSON | [apn_mojo.batch: Float batch JSON](../reference/batch.md#float-batch-json) |
| `complex_batches.mojo` | Complex batches and masks | [apn_mojo.batch: Complex batches](../reference/batch.md#complex-batches) |
| `complex_batch_arithmetic.mojo` | Complex batch arithmetic | [apn_mojo.batch: Complex batches](../reference/batch.md#complex-batches) |
| `complex_batch_functions.mojo` | Complex batch magnitudes and square roots | [apn_mojo.batch: Complex batches](../reference/batch.md#complex-batches) |
| `complex_batch_updates.mojo` | Complex compound updates | [apn_mojo.batch: Complex compound updates](../reference/batch.md#complex-compound-updates) |
| `complex_batch_json.mojo` | Complex batch JSON | [apn_mojo.batch: Complex batch JSON](../reference/batch.md#complex-batch-json) |

## Reductions

| Example | What it shows | Page |
|---|---|---|
| `totals.mojo` | Counters and weighted totals with sum, dot and axpy | [Counters and weighted totals](../guides/totals.md) |
| `float_sum.mojo` | A Float sum rounded once | [apn_mojo: Float sum](../reference/apn_mojo.md#float-sum) |
| `float_dot.mojo` | A Float dot product rounded once | [apn_mojo: Float dot](../reference/apn_mojo.md#float-dot) |
| `complex_sum.mojo` | A Complex sum rounded once per component | [apn_mojo: Complex sum](../reference/apn_mojo.md#complex-sum) |
| `complex_dot.mojo` | Complex dot and vdot | [apn_mojo: Complex dot products](../reference/apn_mojo.md#complex-dot-products) |
| `ordered_reductions.mojo` | Sequential and tree reductions | [apn_mojo: Ordered reductions](../reference/apn_mojo.md#ordered-reductions) |
| `complex_ordered_reductions.mojo` | Ordered Complex reductions | [apn_mojo: Ordered reductions](../reference/apn_mojo.md#ordered-reductions) |
