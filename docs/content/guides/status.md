# Support and limitations

## Supported platforms

APN Mojo supports **Linux and macOS with Mojo 1.1.0**. You can work with exact
integers and fractions, binary floats, exact and rounded complex numbers,
and real and complex balls. The
[verification chapter](../architecture/verification.md) explains how these
features are checked and what the results establish.

The worker pool uses POSIX threads on both platforms. See the
[installation guide](../start/installation.md) for system requirements and
setup instructions.

## Known boundaries

| Area | Current behavior |
|---|---|
| Number types | `Integer`, `Rational`, and `ExactComplex` stay exact. `Float` and `Complex` round each operation once. `Ball` and `ComplexBall` carry interval bounds. All remain subject to storage limits. |
| Float formats | Precision and exponent bounds are configurable. There are five rounding modes and optional traps. Formats have no subnormal values; native output conversions can produce native subnormals. |
| Ball arithmetic | Real and complex balls support arithmetic, set operations, elementary functions, text, and hashes; see [Ball arithmetic](../architecture/balls.md). |
| Batches | Integer, Rational, Float, Complex, Ball, and ComplexBall support containers, views, selection, assignment, iteration, printing, `vmap`, and `lift`. Arithmetic operators and batch JSON apply to Integer, Rational, Float, and Complex. Ball supports `min` and `max`; both ball families support `cumsum` and `cumprod`, with `lift` for other folds. ExactComplex is scalar-only. |
| Shapes | Rank and shape are runtime properties. Higher-rank arithmetic broadcasts compatible trailing dimensions. Operators on two vectors need equal lengths; elementwise `batch.*` and lifted calls also accept length one. `vmap` requires equal mapped extents. Masks never broadcast. |
| Mapping | `vmap` supports scalar and batch arguments, masks, flat tuples, and several result forms. Use a family function such as `apn_mojo.float.add`, not an overloaded root dispatcher. The [batch reference](../reference/batch.md) lists signature and parallel-execution limits. |
| Interval comparison | `Ball` has no ordinary `<` or `==`. Use `compare`, containment tests, or the `certainly_*` predicates. |
| Integer division and powers | `/` returns a `Rational`; integer `**` requires a nonnegative exponent. Use `pow_rational` for reciprocal powers. |
| Assignment | A variable keeps its type. An `Integer` cannot receive a fractional result through `/=`; start with a `Rational` when needed. |
| Native inputs | Integer constructors accept native integral scalars up to 64 bits, plus arbitrary-width integer literals and text. Put library values on the left of comparisons with typed native numbers, or convert the native number explicitly. |
| Division pairs | Use `div_rem_floor`, `div_rem_trunc`, or `div_rem_euclid`; standard `divmod` is not supported. |
| Primality | `is_prime` is deterministic below `2**64`. Above that it uses Baillie–PSW, a probable-prime test rather than a proof. |
| JSON | Integer, Rational, Float, and Complex values and batches have versioned formats. ExactComplex, Ball, and ComplexBall have scalar JSON records. Ball batches do not yet have JSON. There is no promised raw binary ABI. |
| Native output | `Batch.to_native[dtype]()` returns a flat native list, rounding floating-point output once and requiring exact integer output. Ball conversion keeps only the midpoint; complex batches need component extraction first. See [native values](../reference/batch.md#native-values). |
| Threads | `set_num_threads` sets the process-wide count including the caller; `APN_MOJO_NUM_THREADS` chooses the startup default. Small work and unsupported signatures can stay serial. See [thread controls](../reference/batch.md#threads). |
| Decimal conversion | Large decimal inputs or outputs can require substantial working memory. Conversion limits bound one call, not later arithmetic or process memory. |
| Execution | GPU execution and concurrent mutation of a shared value are not supported. |
| Memory exhaustion | Checked limits catch some oversized requests, but physical out-of-memory can terminate the Mojo process. |
| Cryptographic use | Arithmetic and hashes are not designed for constant-time execution or side-channel resistance. |

## Reference disagreements

The benchmark report checks results before timing and returns failure for
disagreements with its references. Investigate the inputs, destination
format, and rounding mode before attributing a mismatch to either library.
See [Verification](../architecture/verification.md#independent-references)
for the reference backends and comparison rules.

For common compiler and usage errors, see [Troubleshooting](troubleshooting.md).
