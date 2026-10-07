# Verification

Verification combines tests of documented behavior with mathematical
arguments for algorithms and rounding decisions. The tests use Mojo 1.1.0
and establish results for the paths and inputs they exercise.

## Layers

| Layer | Location | What it checks |
|---|---|---|
| Functional arithmetic | `tests/test_apn.mojo` | Values, formats, rounding, failures, ownership, and batch behavior |
| Shared batch contracts | `tests/test_batch_refinements.mojo` | Failure injection, borrowing and retained views, linear and strided reductions, broadcasting rules, and budgeted JSON publication |
| Iterator construction | `tests/test_batch_construction.mojo` | Native widths, empty shapes, owned values, and cleanup after partial construction failures |
| Batch JSON | `tests/test_batch_json.mojo` | Family schemas, version and shape rules, bounded output, and failure publication |
| Mask layouts | `tests/test_mask_layout.mojo` | Permutations, axis selection, materialization, scalar and empty shapes, and overflow diagnostics |
| Operator result types | `tests/test_batch_promotions.mojo` | All numeric family pairs, native operands, and wide literals through public expressions |
| Mapping result collection | `tests/test_vmap_results.mojo` | Borrowed arguments for tuple and Optional results, cleanup after callback and later-field failures, and nested Ball outputs |
| Lifted numeric inputs | `tests/test_lift_inputs.mojo` | Exact adapters, wide literals, scalar broadcasting, real and complex ball enclosures, mixed fold seeds, contexts, axes, and repeated failures after the parallel probe |
| Balls | `tests/test_ball.mojo` | Enclosure of exact sample results, stored Arb radius references, kinds, five-way comparison, sets, text enclosures, keys, and batches |
| Float utilities | `tests/test_float_utilities.mojo` | Formats, conversions, rounding, neighbors, and shortest-decimal fixtures |
| Representation keys | `tests/test_interning.mojo` | Representation identity, ordering, dictionary keys, and stable hash vectors |
| Number theory | `tests/test_number_theory.mojo` | Roots, factors, primality, and Jacobi symbols against fixtures and a sieve |
| Factorization | `tests/test_factorization.mojo` | `factor` within its budget, `primes_below` and `next_prime` |
| Correct rounding | `tests/test_certified.mojo`, `tests/test_constants.mojo` | The Ziv driver, budgets and retries; constants and canonical balls against MPFR |
| Elementary functions | `tests/test_elementary.mojo` | Values and flags in every mode against MPFR fixtures, ball inclusion, Arb tightness and determinism |
| Complex functions | `tests/test_exact_complex.mojo`, `tests/test_complex_functions.mojo` | ExactComplex; each part in every mode against MPC fixtures, special values, branch cuts and complex balls |
| Special functions | `tests/test_special.mojo` | Values and flags against MPFR fixtures, values against Arb fixtures, exact rational values of the gamma ratios (`special_ratios.txt`), the Riemann zeta function against MPFR and the Hurwitz zeta function, polygamma, erfinv and ndtri against Arb (`special_zeta.txt`), ball inclusion and poles |
| Hypergeometric functions | `tests/test_hypergeometric.mojo` | hyp1f1, gammainc, gammaincc, hyp2f1 and betainc in every mode against Arb, and their exact values against exact fractions (`hypergeometric.txt`); scipy's poles and conventions, the limits at infinite arguments, the decisions near 1 and at underflow, ball inclusion and vmap with four arguments |
| Serialization and significance | `tests/test_significance.mojo` | Ball and ComplexBall JSON, precision conversions and propagation terms |
| AddressSanitizer | Functional suites with `--asan` | Memory errors detected by ASan on the exercised paths |
| Numerical comparison | `tests/benchmarks/report.py` | Results against Rug (GMP, MPFR, MPC) and python-flint (FLINT, Arb), checked once per case |
| Examples | `docs/examples/` and its manifest | Compilation with `--Werror` and expected output |
| Documentation | `scripts/check_docs.py` | Content inventories, rendered links, anchors, assets, and search entries |

The functional runner compiles each `tests/test_*.mojo` suite into its own
executable, one at a time. Select a suite with `--suite NAME`, or inspect
scenario names with `--list`, which does not compile or run tests. See
[development](../contributing/development.md#verification-gates) for commands.

Borrowing tests check that element reads leave significand owner counts
unchanged, explicit copies survive their readers, and run pointers are
immutable. They also exercise combined selection and tensor strides, retained
views after parent updates, and failures in both components of a Complex dot
product. Sanitizer runs check the same paths for memory errors.

## Independent references

Rational identities compare rounded arithmetic with exact source values.
Rug supplies independent Integer, real, and complex results through GMP,
MPFR, and MPC. python-flint supplies Integer, rational, and ball results
through FLINT and Arb. The [benchmark report](benchmark-results.md) checks
every case against these references before timing it.

Fixtures from gmpy2 and Python's decimal formatting cover additional cases,
including MPFR elementary and special functions with their flags and MPC
complex functions. For special functions MPFR lacks, an Arb fixture counts
only when both ends of its certified enclosure round to the same Float. Arb
also provides the reference for ball tightness.

Reference inputs and destination formats matter. A comparison must avoid
rounding inputs before the tested operation and account for APN's lack of
subnormals and its underflow rules. See the
[benchmark guide](../contributing/benchmarks.md).
