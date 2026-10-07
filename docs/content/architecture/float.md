# Float

A `Float` stores a binary value and its precision format. Its central promise
is correct rounding: each operation returns the mathematical result rounded
once to the target format. Exact arithmetic and certified bounds establish
that result; native fast paths preserve the same rules. Public calls are in
the [Float reference](../reference/float.md).

## Values and formats

A finite nonzero `Float` has the normalized form:

$$
(-1)^s \times m \times 2^{e - p}
$$

where $s \in \{0, 1\}$ is the sign, $m$ is an [`Integer`](integer.md) significand
of exactly $p$ bits, and $e$ is an integer exponent normalized such that
$2^{p-1} \le m < 2^p$. Special values (zero, infinity, and NaN) are handled as
distinct classes. Zeros and infinities retain their signs, while `NaN` is
signless and canonical. Copies share immutable significand storage.

- **`FloatFormat(precision, emin=..., emax=...)`**: Defines the significand bit-width $p$
  and inclusive exponent boundaries. Subnormal numbers are intentionally omitted:
  values strictly below $2^{emin - 1}$ round either to zero or to the smallest
  normal value. `MAX_PRECISION` denotes the addressability boundary of `Integer`
  minus two bits reserved for rounding carries.
- **`ArithmeticContext`**: Holds a `FloatFormat`, one of five
  rounding modes (`nearest_even`, `toward_zero`, `toward_positive`, `toward_negative`,
  `away_from_zero`), and five condition trap flags (disabled by default). Contexts
  are passed explicitly; there is no global ambient state.

### Format selection

When an operation is called without an explicit context, it merges the formats
of its operands:

- Exact inputs (`Integer`, `Rational`, integer literals) contribute no precision bits.
- Native hardware floats contribute their standard bit-widths (e.g. 53 bits for
  `Float64`) but impose no exponent boundaries.
- Library `Float` instances contribute their configured precision and bounds (and
  their exponent bounds must match).

Operations containing only exact inputs require an explicit context, ensuring
exact mathematics never silently converts to floating point.

### Exact intermediate formats

An internal "exact format" flag marks working values that must accumulate
without rounding, for example during a vector reduction. It preserves every
bit of precision. A ratio is representable exactly only if the denominator's
odd factor divides the numerator completely; otherwise, the operation raises.
`lift` uses these formats for intermediate terms and rounds the final
reduction result once.

## One rounding

The general rounding kernel converts an exact rational input
$(n / d) \times 2^{scale}$ to the destination format. Native fast paths use the
same rounding rules without building all the general-path intermediates:

1. **Virtual exponent estimation**: For $a = |n|$ and $b = |d|$, bit lengths and
   virtual shifted comparisons determine an integer $k$ satisfying
   $2^{k-1} \le a/b < 2^k$. This estimation inspects existing limb buffers without
   heap allocations.
2. **Range classification**: The tentative exponent $e = k + scale$ is evaluated
   in checked 128-bit integers. Values far outside the format's exponent range
   are classified immediately, avoiding buffers proportional to a huge exponent
   gap. Producing the final value can still require precision-sized storage.
3. **Underflow handling**: If the value falls below $q_{min} = 2^{emin - 1}$, it
   rounds to zero or $q_{min}$ (nearest-even selects $q_{min}$ only when strictly
   above $q_{min}/2$). Both underflow and inexact conditions are recorded.
4. **Significand extraction and division**: When within range, the exact quotient
   $q = \lfloor a \cdot 2^{p - k} / b \rfloor$ yields $p$ bits. A single multi-precision
   division computes $q$ while categorizing the remainder $r$ relative to $b/2$:
   zero, below half, exactly half, or above half. Nearest-even rounds up when
   $r > b/2$, or when $r = b/2$ and $q$ is odd. Directed modes round away from
   zero whenever $r > 0$ in the specified direction. Power-of-two divisors bypass
   division entirely.
5. **Carry normalization**: If rounding causes a carry that turns $q$ into $2^p$,
   a 1-bit shift and an increment to $e$ restore the significand to $p$ bits.
   Exponents exceeding $emax$ are projected to infinity or the maximum finite
   value according to rounding mode and sign.
6. **Trap dispatch**: Condition traps are checked in standard precedence order:
   invalid, divide-by-zero, overflow, underflow, and inexact. Trapped conditions
   raise before any destination data is committed.

Operands of at most 256 bits, with a precision of at most 128, skip this
kernel: long division on stack limbs gives a quotient of $p + 2$ or $p + 3$
bits, its remainder a sticky bit below it, and the native rounder used for
short sums and products rounds that. A power-of-two denominator, the binary
value of `x + 0` or of a fixed-point kernel result, goes to the native rounder
without the division. Exponents outside the format's range return to the
general kernel.

A binary value whose numerator already has exactly the format's precision,
with its exponent in range, is that format's record: building a ball from a
Float of its own precision, or scaling one by $2^k$, keeps the significand and
sets the exponent (`_round_exact_binary`), with nothing to round.

For an exact rational input, this kernel decides rounding from the quotient
and remainder. Functions such as large powers have separate adaptive paths
that certify an enclosure before returning a result.

## Arithmetic

- **Products and quotients**: Form exact cross-products $(a_1 \cdot a_2) / (b_1 \cdot b_2)$
  and $(a_1 \cdot b_2) / (b_1 \cdot a_2)$, add binary scales in checked 128-bit
  integers, and round once through the core finalizer. Special values (zeros,
  infinities, NaNs) are classified upfront.
- **Sums**: Sums of exact ratios take the form $(\pm A \cdot 2^s \pm B \cdot 2^t) / D$
  with $s \ge t$. Setting $c = \text{bit\_length}(D) + p + 4$ and $gap = s - t$:
  - If $gap \le \text{bit\_length}(B) + c$, the shifted sum is formed exactly. Its
    bit length is bounded by operand lengths and $p$, avoiding unbounded allocation
    while covering all cases of significant cancellation.
  - If $gap > \text{bit\_length}(B) + c$, the remote minor term satisfies
    $0 < B \cdot 2^{-gap} < 2^{-c}$ and is replaced by a private rounding witness
    $\pm 2^{-c}$.

The cutoff accounts for the denominator and output precision: both perturbations
are smaller than the distance to the next distinct rounding boundary. If the
dominant term itself lies on a boundary, the true tail and its replacement move
it to the same side. Thus both sums have the same rounding, direction and
inexactness in all five modes. Exact working formats retain the actual tail;
they cannot substitute a rounding witness.

### Native fast paths

Returning large structs across non-inlined calls costs 15–20 ns in the
recorded measurements. Common paths for small precisions avoid that cost
by inlining arithmetic on native integer types:

| Format Constraint | Operation | Native Evaluation Strategy |
|---|---|---|
| $p \le 63$, inline limbs | `+`, `-`, `*` | Native 128-bit alignment or product, followed by native integer rounding. |
| $p \le 63$, inline limbs | `/` | 128-bit quotient providing at least $p + 1$ bits, tracking the remainder via a sticky bit. |
| $p \le 128$, $\le$ 128-bit limbs | `+`, `-`, `*` | Native 256-bit alignment or multiplication. |
| $p \le 128$ | `sqrt` | Native 256-bit square root with its remainder. |

When terms lie far apart, minor terms are collapsed to sticky bits only when
provably safe (the minor term is strictly below one unit of the shifted major term,
and the rounding boundary is at least two bits higher). If safety conditions are
not met, execution uses the general path.

## Functions

- **Square root**: For positive $x$, the root's exponent is $\lceil e(x) / 2 \rceil$.
  After scaling, the general case seeks a $p$-bit integer
  $q = \lfloor\sqrt{A/B}\rfloor$. It is exact when $q^2 B = A$.
  Nearest rounding compares $4A$ with $B(2q+1)^2$, using $q$'s parity for a tie.
  When $B=1$, the integer square-root kernel returns $q$ and the remainder
  $r=A-q^2$ together. A tie is impossible because $(q+1/2)^2$ is not an integer;
  rounding goes up exactly when $r>q$. This binary-input path applies scaling
  as it loads limbs, avoiding a separate shifted copy of $A$. See
  `float/_functions.mojo` and the [Integer chapter](integer.md).
- **Integral powers**: A zero exponent evaluates to $1$. Negative powers swap
  numerator and denominator, evaluating powers directly without pre-rounding
  reciprocals. When a bit-length estimate bounds the exact powers by
  $16,384$ bits, they evaluate with exact integer arithmetic before rounding once. Larger powers use outward
  dyadic interval arithmetic: the base is bounded by outward dyadic brackets,
  squarings are rounded outward, and results are accepted only when both interval
  endpoints round to identical values and flags. If bounds diverge, working
  precision doubles iteratively.
- **Fused multiply-add (`fma`)**: Passes the unrounded product $a \times b$ and
  addend $c$ directly into the bounded sum kernel, resolving cancellation and
  exponent scaling before applying a single final rounding.
- **`ldexp`**: Adds an integer scaling factor directly to the exponent, classifying
  range boundaries via bit lengths without forming intermediate powers of two.

## Comparison and conversion

Comparisons read normalized exponents and virtually aligned words to compare
stored values exactly, including against `Integer` and `Rational` operands.
They need no shared rounded format. Text parsing sends decimal, hexadecimal,
and binary sources through a budgeted rounding kernel, rounding the exact
input once.

## Neighbours, conversions and shortest decimals

- **Machine formats.** `FloatFormat.binary64()` is 53 bits with exponents
  $-1021$ through $1024$, and `binary32()` 24 bits with $-125$ through $128$:
  the normal ranges of `Float64` and `Float32`. A format has no subnormals, so
  the smallest positive value is $2^{e_{min} - 1}$.
- **Exact conversions.** `to_rational_exact` (also `Rational(x)`) returns the
  binary fraction in lowest terms, and `round` rounds to the nearest
  Integer with ties to even. Both share the integer conversion of `floor`,
  `ceil` and `trunc`, which compares the fraction with one half by bit length
  before any shift.
- **Neighbours.** A format's values are numbered from zero, with each infinity
  one step beyond the largest finite value. `nextafter(x, toward)` moves one
  step toward any number, compared exactly; `ulp_distance` counts steps after rounding both values to the coarser
  precision; `equal_at_precision` rounds both to a given precision. `spacing(x)` is
  $\pm 2^{e - p}$, the spacing at $x$ with its sign, as NumPy's. Without subnormals the spacing in the lowest
  $p - 1$ binades is below the smallest positive value, so `spacing` raises there,
  `spacing` of a zero is the gap to the smallest positive value, and of an
  infinity or NaN it is NaN.
- **Shortest decimals.** `shortest_decimal` finds the decimal with the fewest
  significant digits that reads back (nearest-even) as the same Float, the
  closest such decimal, with ties to an even last digit. The values that round
  to $x = m \cdot 2^{e - p}$ form an interval whose ends are the midpoints to the
  neighbours, included when $m$ is even; at a power of two the lower neighbour
  is half as far, and at the smallest value the interval reaches down to half
  of it, exclusive, since that halfway point rounds to zero. A binary search
  over the decimal exponent finds the largest power of ten with a multiple in
  the interval; exact Integer arithmetic decides each step. Such a multiple is
  never a multiple of the next power of ten, so all candidates have the same
  number of digits. For binary64 this is the decimal Python's `repr` prints
  (all 6000 cases of the test fixture agree), except at the smallest value,
  where `2e-308` already reads back as $2^{-1022}$.

## Representation identity and hashing

`Float` cannot be a hash-table key directly: numeric equality merges zero
signs and equal values in different formats, while NaN equals nothing.
A hash of its stored representation would not match that equality. For
caches that distinguish representations, use these tools:

- **Identity and order.** `same_representation` holds when class, sign,
  precision, exponent bounds, exponent and significand all match; the NaNs of
  one format match, since NaN is canonical. `representation_cmp` orders values
  by `total_cmp` and breaks ties by precision, then by the lower and the upper
  exponent bound, so it returns 0 exactly when the representations match.
  `Complex` compares its real components first.
- **Keys.** `FloatKey` and `ComplexKey` wrap a value and compare, order and hash
  by representation, so a `Dict[FloatKey, V]` keeps each signed zero, each
  format and NaN as its own key, and sorted keys come out the same in every run.
- **Stable hashes.** `stable_hash` computes APNH-64, a fixed algorithm, where
  Mojo's `Hasher` may change between releases. A value is a family tag followed
  by 64-bit words, each mixed with SplitMix64's finalizer, and the word count
  closes the hash. An Integer is its sign, its number of 64-bit magnitude words
  and the words; a Rational is its numerator and denominator; a Float is its
  precision, exponent bounds, class (zero, finite, infinite, NaN) and sign, then
  for a finite nonzero value its exponent and significand; a Complex is its two
  components. Changing an encoding is a breaking change. The keys hash this
  value through Mojo's `Hasher`.

## Verification

The functional suites check rounding against exact Rational values and test
formats, special values, conversions, neighbors, and decimal fixtures. The
comparison harness adds independent MPFR results through Rug. See
[Verification](verification.md) for commands, coverage, and known disagreements.
