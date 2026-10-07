# Overview

APN Mojo builds arithmetic around scalar values: exact integers and fractions,
correctly rounded real and complex calculations, and interval bounds.
`Batch[T]` stores supported number families and reuses their scalar functions
through mapping. The [design principles](principles.md) explain the guarantees
this structure must preserve.

## Value-centric design

Callers calculate with values, save copies, take slices, and convert between
formats. The library handles storage sharing and execution. Optimizations
must preserve both the numerical results and the independence of saved copies.

| Contract | Implementation |
|---|---|
| **Exact integer and rational math** | Representations grow dynamically; operations never silently wrap or truncate. |
| **Floats round once** | Exact arithmetic or certified bounds establish the result rounded once to its format ([Float](float.md)). |
| **Copies keep their values** | Copy-on-write lets buffers share storage while updates leave existing snapshots and dictionary keys unchanged. |
| **Shared slices and views** | Views retain element storage and describe selections without copying every scalar. Layout descriptors may allocate. |
| **Updates succeed as a whole** | The library prepares and validates the result before changing the destination. A failed calculation leaves the original value intact. |
| **Errors explain the remedy** | Messages identify the failed operation, what went wrong, and how to address it. |
| **Batches reuse scalar arithmetic** | A shared execution and reduction driver applies scalar functions across batches. |

## System layers

The implementation separates numerical rules from storage and execution:

| Layer | Components | Details |
|---|---|---|
| **Scalars** | `Integer`, `Rational`, `Float`, `Complex`, `Ball`, their contexts, and scalar functions. | [Integer](integer.md), [Rational](rational.md), [Float](float.md), [Complex](complex.md), [Ball](balls.md) |
| **Batches** | Multi-dimensional `Batch[T]`, strided slicing, broadcasting, masks, and transactional updates. | [Batches and execution](batches.md) |
| **Vectorization** | `vmap[f]` and `lift[f]` mapping scalar functions sequentially or across the thread pool. | [Batches and execution](batches.md#mapping-functions) |
| **Reductions** | Exact single-rounding reductions (`sum`, `dot`), extrema, and ordered tree/sequential folds. | [Reductions](reductions.md) |
| **Conversion** | Text parsing, canonical JSON codecs, and resource-bounded `ConversionLimits`. | [Conversion and interchange](conversion.md) |

## Module map

The root package re-exports commonly used types and functions. Family packages
provide additional functions and the single declarations needed by `vmap` and
`lift`. The generated reference lists each export.

Source files are grouped by number family and shared operation:

```text
src/apn_mojo/
  __init__.mojo             public API exports and top-level dispatch
  integer/                 exact Integer values, arithmetic, and native batch execution
  rational/                canonical fractions and exact rational adapters
  float/                   Float values, precision formats, rounding, and certified math
  complex/                 Complex values with Float components and multi-component math
  ball/                    real intervals, outward bounds, and set operations
  batch/                   batches, masks, shared mapping executor, and reductions
  common/                  text/JSON parsers, conversion limits, and thread pool primitives
```

Use this map to locate an implementation within `src/apn_mojo/`:

| Source Path | Responsibility |
|---|---|
| `integer/value.mojo` | Scalar construction, arithmetic operators, comparisons, hashing, native/text/JSON conversion, shared GCD, and true division. |
| `integer/math.mojo`, `integer/number_theory.mojo`, `integer/division.mojo`, `integer/compound.mojo` | Scalar integer functions, named arithmetic, division pairs, number theory, and batch `axpy`. |
| `integer/_operand.mojo`, `integer/_static.mojo`, `integer/_scalar_update.mojo` | Private scalar readers, literal storage, and staged in-place updates. |
| `integer/_batch_native.mojo` | Native integer batch execution, checkpointed writers, and batch diagnostics. |
| `integer/_division.mojo`, `integer/_gcd.mojo`, `integer/_root.mojo`, `integer/_multiplication.mojo`, `integer/_magnitude.mojo` | Multi-precision division, greatest common divisors, k-th roots, multiplication kernels, and magnitude helpers. |
| `integer/_limbs.mojo` | Arithmetic on 64-bit limbs behind raw pointers, shared by the limb-level kernels. |
| `integer/_tiles.mojo` | SIMD lane aliases shared between mask bits and floating-point metadata records. |
| `integer/_text.mojo`, `integer/_json.mojo`, `integer/_conversion.mojo`, `integer/_limits.mojo` | Text and JSON grammars, conversion support, and checked numeric bounds. |
| `rational/value.mojo`, `rational/math.mojo` | Canonical fractions, GCD reduction, comparisons, signed powers, `pow_rational`, and named arithmetic. |
| `rational/_batch_storage.mojo`, `rational/_batch_ops.mojo`, `rational/_reductions.mojo` | Native rational inputs, exact batch adapters, and reductions. |
| `rational/_text.mojo`, `rational/_json.mojo`, `rational/_batch_json.mojo` | Scalar and batch text parsing and JSON codecs. |
| `float/value.mojo`, `float/math.mojo` | Scalar floating-point values, arithmetic, functions, and native hardware float conversions. |
| `float/context.mojo`, `float/status.mojo` | Formats, rounding modes, arithmetic contexts, and internal condition trap status. |
| `float/_input.mojo`, `float/_format.mojo`, `float/_comparison.mojo` | Input descriptors, format selection, and exact comparison logic. |
| `float/_rounding.mojo`, `float/_arithmetic.mojo`, `float/_functions.mojo`, `float/_native.mojo` | Exact single-rounding core, intermediate formats, certified functions, and native outputs. |
| `float/_text.mojo`, `float/_json.mojo` | Exact hexadecimal output, rounded decimal output, and format-preserving JSON codecs. |
| `float/_parse.mojo`, `float/_decimal.mojo` | Text parsing, range classification, budgeted final rounding, and decimal formatting. |
| `float/_batch_storage.mojo` | Float batch inputs (runs, Complex components, broadcast scalars) and conversion of iterated values. |
| `complex/value.mojo`, `complex/math.mojo`, `complex/context.mojo` | Complex values, arithmetic, functions, and paired component contexts. |
| `complex/_arithmetic.mojo`, `complex/_finite.mojo`, `complex/_functions.mojo`, `complex/_power.mojo` | Component-level single rounding, exact two-term sums, magnitudes, square roots, and integer powers. |
| `complex/_batch_storage.mojo`, `complex/_batch_ops.mojo`, `complex/_reductions.mojo`, `complex/_text.mojo`, `complex/_json.mojo` | Complex batches, reductions, text parsing, and JSON serialization. |
| `ball/value.mojo`, `ball/_radius.mojo`, `ball/context.mojo` | Balls: a Float midpoint and a one-word radius, their kinds, text and hashes, and the precision context. |
| `ball/_arithmetic.mojo`, `ball/math.mojo`, `ball/sets.mojo` | Ball arithmetic whose radii bound every rounding, and balls as sets: five-way comparison, containment, hulls and certain floors. |
| `ball/_certified.mojo`, `ball/_fixed.mojo` | The Ziv driver for one value or a complex pair, the rule for values near a Float, and the kernels' fixed-point balls on Integers or native words, with the series sum by rectangular splitting. |
| `ball/_kernels.mojo`, `ball/_exp.mojo`, `ball/_log.mojo`, `ball/_trig.mojo`, `ball/_power.mojo`, `ball/_constants.mojo`, `common/_binary_splitting.mojo` | Point kernels of the elementary functions, the constants by binary splitting, and the exact points. |
| `ball/_medium.mojo`, `ball/_tables.mojo` | Medium-precision kernels (Johansson, ARITH 22, 2015) for exp, expm1, log, atan, sin and cos up to 4608 bits, and their tables, written by `scripts/generate_function_tables.py`. |
| `ball/_functions.mojo`, `ball/elementary.mojo`, `ball/constants.mojo` | Ball versions of the elementary functions, with domains and budgets, and canonical constant balls. |
| `ball/_special.mojo`, `ball/special.mojo` | Kernels and ball versions of Gamma, log Gamma, digamma, the error functions, the exponential, sine, cosine and hyperbolic integrals, Fresnel's integrals and Lambert's W. |
| `ball/_json.mojo`, `ball/significance.mojo` | Ball and ComplexBall JSON, and significance arithmetic: precision conversions, relative-precision radii and propagation terms. |
| `float/elementary.mojo`, `float/constants.mojo`, `float/special.mojo`, `float/_elementary.mojo`, `float/_special.mojo` | Correctly rounded elementary and special functions and constants: special values, exact cases, and the rules decided before the Ziv driver. |
| `complex/elementary.mojo`, `complex/_elementary.mojo` | Complex elementary functions, each part correctly rounded, with MPC's special values and structural zeros. |
| `complex_ball/value.mojo`, `complex_ball/math.mojo`, `complex_ball/elementary.mojo`, `complex_ball/sets.mojo` | Complex balls: rectangles of two balls, their arithmetic, functions with branch cuts, and set operations. |
| `exact_complex/value.mojo`, `exact_complex/math.mojo` | Exact complex numbers with Rational parts, exact square roots and the stable hash. |
| `integer/factorization.mojo` | Factorization within a budget (Pollard-Brent rho), the prime sieve and `next_prime`. |
| `batch/value.mojo`, `batch/mask.mojo`, `batch/_slices.mojo` | Batches, views, iteration, indexing, shape propagation, broadcasting, updates, and selection masks. |
| `batch/_families.mojo`, `batch/_array.mojo`, `batch/_layout.mojo`, `batch/_tensor.mojo` | The list of element families, untyped array descriptors, runtime memory layouts, and 1-D contiguous vectors. |
| `batch/_parallel.mojo`, `batch/mapping.mojo`, `batch/lift.mojo` | Shared elementwise execution, public `vmap` and `lift`, and transactional results. |
| `batch/_lanes.mojo`, `batch/lift.mojo`, `batch/_exact_reduce.mojo`, `batch/_float_folds.mojo` | Reduction execution driver, lifted function decorators, and built-in reduction kernels. |
| `batch/_parallel.mojo`, `batch/_values.mojo`, `common/_threads.mojo` | Multi-threaded execution, owner-affine thread releases, and persistent worker pool management. |
| `batch/_comparison.mojo`, `batch/reductions.mojo`, `batch/_reduce_exec.mojo`, `batch/_exact_reduce.mojo` | Exact comparison contracts, whole-batch and axis reductions, and shared reduction execution. |
| `common/keys.mojo`, `common/_stable_hash.mojo` | Representation keys and the APNH-64 stable hash. |
| `integer/powers.mojo`, `integer/primality.mojo` | Exact roots, perfect powers, and deterministic or probable-prime tests. |
| `float/neighbors.mojo`, `float/shortest.mojo` | Adjacent values, spacing, and shortest decimal output. |
| `common/functions.mojo` | Package-level root dispatchers (`add`, `subtract`, `multiply`, `divide`, `sqrt`, `pow_int`, `abs`, and the elementary and special functions) delegating to each family's declaration. |
| `common/conversion.mojo`, `common/_sizes.mojo`, `common/_memory.mojo` | Conversion policies and quotas, checked addressable sizes, and diagnostic memory accounting. |
| `common/_text.mojo`, `common/_json.mojo` | ASCII parsing, JSON tokenization, and schema validation shared across numeric families. |

Shared text and JSON parsing utilities operate independently of specific numeric
types. Individual family modules define their own grammars, schemas, and
domain diagnostics, while the shared reader handles tokenization and enforces
resource budgets.

## Runtime and development dependencies

All runtime arithmetic in `apn_mojo` is implemented in pure Mojo using the standard
library. Production code never delegates calculations to Python, GMP, MPFR, or
other external libraries.

Heap storage uses standard Mojo FFI wrappers around `malloc` and `free`.
The persistent worker pool uses POSIX threads on Linux and macOS, with
single-threaded execution elsewhere. The codebase contains no C sources and
needs no external compilation steps.

GMP, MPFR, MPC, Arb, and mpmath are development references. AddressSanitizer checks
memory use on exercised paths. MkDocs and Pygments build the documentation;
see [Verification](verification.md) for the separate checks.
