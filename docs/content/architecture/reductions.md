# Reductions

Combining elements requires a choice of order, an identity, and a rule for
when rounding happens. Reductions share one driver while their arithmetic
kernels supply those choices. This chapter explains exact accumulation and
ordered rounding; the [Batch reference](../reference/apn_mojo.md#float-sum)
covers their use.

## Kinds of reduction

| Reduction | Families | Result |
|---|---|---|
| `sum`, `prod`, `dot` | Integer, Rational | Exact |
| `min`, `max`, `argmin`, `argmax` | Integer, Rational, Float; `min` and `max` also Ball | The first extreme by exact comparison, a Float NaN propagating; for Ball, the ball of the extreme over all points |
| `sum`, `dot` | Float, Complex (and exact partners in `dot`) | The exact total rounded once, per component for Complex |
| `vdot` | The same families as `dot` | Conjugates the first operand; equals `dot` for real inputs |
| `sum_sequential`, `sum_tree`, `dot_sequential` | Integer, Rational, Float, Complex | A stated order; Float and Complex round at each step |
| `lift[f].reduce`, `.accumulate` | Integer, Rational, Float, Complex, Ball; Bool functions on masks | A fold of a supported `f`; Float and Complex keep exact intermediates by default, while Ball carries each scalar enclosure forward |

Built-in reductions and `lift.reduce` take an axis or a list of axes and
`keepdims`; `lift.accumulate` takes one axis and retains each prefix result.
Empty sums are zero and empty products one; empty extrema raise. For Ball
reductions, lift a scalar Ball function; built-in Ball reductions are not
implemented yet.

## One driver

`batch/_lanes.mojo` runs lifted folds, built-in sums, exact products and
extrema, and Integer dot products. It splits a batch
into lanes: the reduced axes, walked in row-major order, for each position of
the kept axes. A kernel implements three steps:

- `fold` folds a run of one lane into a partial result;
- `combine` combines consecutive partial results, in order;
- `finish` turns the last partial result into the output.

Separate lanes run in parallel. When a reduction is associative and has fewer
than 64 lanes, each long lane also splits into chunks of at least 64 elements,
about 256 in all, that fold on separate threads and combine in order. Exact
accumulation is associative, so a long Float sum splits this way and still
rounds once.

Private failure hooks for Integer and Rational folds run these same kernels
and the same lane scheduler. Checkpoints follow logical elements inside each
chunk; they do not substitute a sequential arithmetic implementation. Normal
calls select a specialization with the checkpoint branches compiled out.
Tests inject failures at chunk boundaries and on reversed selections, checking
that inputs and an existing destination survive the failure. Ordered modes
retain their own step-based hooks because their grouping is part of the API.

## Round-once sums

A Float `sum` adds the stored values exactly and rounds once. Reversing or
permuting the values gives the same result and flags. Without a context, the
output precision is the largest selected precision and all exponent bounds
must agree; with one, its format, rounding and traps apply directly. Format
checks cover every selected element, including NaN and infinity, before any
accumulation, and a mismatch names its logical index.

Special values and zero signs:

| Selected values | Result |
|---|---|
| Empty | `+0` |
| Only negative zeros | `-0` |
| Mixed zero signs, or exact cancellation | `+0`, or `-0` when rounding toward negative |
| Both signs of infinity | NaN, invalid |
| A quiet NaN, without conflicting infinities | NaN |
| One sign of infinity, no NaN | That infinity |
| Otherwise | The exact total rounded once |

Large terms may cancel without overflowing first: only the final result
produces flags. Complex `sum` does this independently per component, real
first. `dot` sums exact pairwise products and rounds once. At least one operand
is Float, or Complex for a Complex result, and the other may be Integer or
Rational. Zero times infinity is invalid, and no product is rounded or
range-limited on its own. Complex `dot` uses the cross terms `ac - bd` and
`ad + bc`; `vdot` conjugates the first input.

## Exact accumulation

The accumulator in `float/_accumulator.mojo` stores an exact signed sum of
scaled integers in radix-2**32 digits, called bins. A dense window handles
nearby values; a sparse map handles values far apart. Both represent the
same exact sum.

### Sparse map and the rounding window

1. Set `B = 2**32`. Each input limb is placed at its absolute bit scale, split
   over at most two bins when unaligned. Bit scales are Int128; bin indices are
   checked before narrowing to native Int. A map coefficient satisfies
   `-(B-1) <= d < B`. Adding one signed word gives `abs(total) <= 2*(B-1)`,
   safely within Int64. Truncating division produces a remainder of magnitude
   below B and a carry in `{-1,0,1}`. Replacing `total` by remainder plus
   `carry*B` preserves the exact sum. Carry traverses occupied bins but stops at
   the first empty bin, rather than propagating a negative borrow across a gap.
2. Sort nonzero bin indices with the standard-library sort. The sign of the
   highest nonzero coefficient is the sign of the sum: the sum of all lower
   coefficient magnitudes is strictly less than one unit of that highest bin.
   No remaining nonzero coefficient can represent exact zero. Zero coefficients
   may remain in the map until the call ends; they do not affect normalization.
3. Multiply coefficients by the overall sign and sweep upward. Canonicalization
   needs only carry `0` or borrow `-1`. A borrow over an empty interval produces
   an all-ones digit run, represented by its two endpoints instead of materialized
   limbs. At the highest coefficient the borrow is discharged, yielding an exact
   positive magnitude. There are at most two runs per nonzero map entry.
4. Obtain the exact top bit position from the highest run. In the normal range,
   extract the leading `p+2` bits into a precision-sized Integer. Jam any nonzero
   omitted suffix into its least bit, using the first occupied run to test the
   suffix exactly. Two low bits distinguish an exact p-bit point, below midpoint,
   exact midpoint and above midpoint; jamming preserves all four distinctions,
   including midpoint parity and carry into the next exponent. It also preserves
   whether any information is lost, hence inexact and the signed direction.
5. Feed this sufficient rounding representation to the existing exact-ratio
   finalizer with its absolute scale. For a magnitude already outside normal
   exponent bounds, three leading bits with suffix jamming suffice: underflow
   compares against the power-of-two midpoint `qmin/2`, while overflow selection
   depends only on mode/sign. This avoids allocating enormous precision for a
   result that rounds to zero. Saturation or a minimum nonzero result may still
   require the requested output precision. Range flags, ties, signed direction
   and traps use the existing finalizer without double rounding.

Storage is proportional to the visited bins and the output precision, not to
the distance between exponents.

### Dense windows

Most sums keep their exponents within a few thousand bits of each other. For
them the accumulator starts with a dense window instead of the map: Int64
counters for consecutive bins `[low, low + n)`. Adding an input adds one
signed shifted word, below `B`, to each of at most `count + 1` consecutive
counters, without carrying. That is one pass over the input's limbs, with no
hashing.

- **Bounds.** After `k` additions since the last carry pass, every counter
  satisfies `abs(d) < (k + 1) * B`. A carry pass every `2**30` additions
  keeps that below `2**63`. The pass leaves every counter in `[0, B)` except a
  signed top one, and the window grows by one bin to hold its carry.
- **Merges.** Partial sums add counter by counter when the window can hold
  both and their addition counts together stay under the bound; otherwise
  they add bin by bin.
- **Finalization.** Floor-carry a copy of the window. Every bin then lies in
  `[0, B)`, so the sign of the final carry is the sign of the sum. A negative
  sum carries the negated counters instead, giving its magnitude. Each nonzero
  bin becomes a one-bin run, and steps 4 and 5 apply unchanged.
- **Fallback.** A window spans at most 4096 bins (131,072 bits). An input
  that would need a wider window, or bins beyond `2**60`, moves the whole sum
  to the sparse map, each counter as at most two words, and the sum continues
  there.

Results do not depend on the representation: both hold the exact sum.

### Products and Rational denominators

Dot products reuse retained Float and exact operand descriptors. They check
metadata across both selections first, then skip numeric extraction for zero
pairs or a known nonfinite result. Only selected nonzero finite Rational
denominators contribute to a common positive least common multiple `D`.
Because at least one family is Float, each
term has at most one nonunit denominator. A second pass multiplies exact stored
significands/numerators, scales by `D / d`, and adds signed limbs at the sum of the
two Int128 binary scales. Thus bins represent exactly `D` times the dot result.
There is no product batch or batch-wide scalar list; extracted operands and
products are scoped to one pair. The common denominator can grow with input
denominators; this is exact rational growth, not a dense exponent-gap allocation.

For `D != 1`, let `d = bit_length(D)` and `k` be the sparse numerator's top bit
position. The result exponent is either `k-d` or `k-d+1`. Retaining `p+d+2`
leading numerator bits puts the output quantum at least two bits above the
extraction cutoff. Representable points and midpoints, multiplied by `D`, are
therefore even integers at that cutoff. Replacing a nonzero omitted suffix by
the low sticky bit cannot cross or land on a boundary: an even prefix becomes
odd, and an odd prefix stays odd. Exact boundaries keep a zero suffix. This
preserves rounding, midpoint parity, exactness and signed direction before the
existing ratio finalizer divides by `D` once.

Outside the normal range, `d+3` leading bits suffice for the power-of-two range
boundaries and the underflow midpoint. A one-bit exponent ambiguity at a range
boundary is resolved with that small prefix before requesting output precision.
Checked storage depends on visited bins, exact products/common denominator and
output precision, not the distance between exponents. This implementation uses
existing exact Integer multiplication/division and scalar sparse-bin arithmetic;
it adds no SIMD or speedup claim. Profiling remains deferred.

## Exact Rational reductions

Rational reductions read selected elements directly and reduce each fraction
as they go. They need no intermediate product batch.

- **Sum** divides out the denominator gcd before scaling, then cancels the
  remaining common factor. A prefix of `k` fractions has a reduced
  denominator dividing the product of their denominators. With common
  denominator `D`, write each term as `N_i / D`. The sum's numerator has at most
  `max(bit_length(abs(N_i))) + ceil(log2(k))` bits before reduction, for `k > 0`.
  This bound applies to the scaled numerators, not the original numerators.
- **Product** cross-cancels before multiplying; its bit lengths are bounded by
  the sums of the inputs'. A selected zero is detected before any extraction.
- **Dot** cross-cancels each product and adds it canonically.
- **Extrema** compare with continued fractions, holding one winner and one
  candidate.

These are numerical growth bounds, not memory quotas.

## Ordered reductions

`sum_sequential`, `sum_tree` and `dot_sequential` prescribe a traversal for
users who need a reproducible sequence of rounded steps. `sum_sequential`
rounds each addition in logical order; `dot_sequential` rounds each product,
then each addition; `sum_tree` uses the fixed adjacent-pair grouping. They use
one private executor (`batch/_reduce_exec.mojo`) across all four families.

### Separation of responsibilities

`batch/_reduce_exec.mojo` imports no numeric type, storage descriptor, context or
status class. Mojo traits bind an adapter at compile time, without runtime boxing.

| Contract or hook | Responsibility |
| --- | --- |
| `_ReductionInputs` | Retain read-only sources and expose logical length |
| `_ReductionOperation.Inputs`, `.Value`, `.Result` | Specify input, working and result types; working/result values may be distinct and move-only |
| `validate`, `prepare` | Check shapes first, then select formats and prepare call-local state |
| `identity`, `element` | Supply the operation's identity and a selected value or pair product |
| `combine` | Perform arithmetic and any prescribed rounding at one node |
| `singleton` | Finalize a one-leaf tree without an artificial addition |
| `finish` | Return an unpublished result with the rounding status that decides traps |

`batch/_exact_reduce.mojo` adapts retained native Integer/Rational inputs
and existing exact arithmetic. `float/_ordered_reductions.mojo` selects the target
format, reuses existing scalar rounding primitives and records each step's flags.
Dot is a fold whose element hook supplies a product; it needs no second loop.
Ordinary Integer folds and dot products use the lane kernels instead.
Product checks for zero before multiplying; extrema reject empty lanes and
keep the first winner when values compare equal.

### Traversal and memory

Sequential execution starts with the adapter's identity, zero for a sum or
one for a product, and visits logical indices in order. Tree execution uses
the fixed adjacent-pair grouping, padded with positive zeros to the next
power of two. Leaves are not pre-rounded. A left-to-right depth-first
postorder traversal keeps only logarithmic-depth partial results alive.
Padding-only subtrees return the identity without touching source storage;
adapters must make identity-plus-identity equivalent in value, flags and traps.

Neither traversal creates a batch-wide product or partial-result list. Numeric
values themselves and arithmetic scratch can still grow; logarithmic depth is
not a byte limit. Reversed and strided sources retain their original owners and
are read only at selected logical indices. No implicit promotion batch is built.

### Errors and publication

The adapter checks formats before arithmetic and reports a failure at the
prescribed step where it occurs. Storage failures remain errors rather than
numerical flags. Recoverable errors release temporary values; the wrapper
returns a new result only after the whole operation succeeds. Inputs and
saved snapshots keep their values.

For Float, a trap fires at the first prescribed step whose rounding sets the
trapped condition. See [ordered reduction semantics and examples](../reference/apn_mojo.md#ordered-reductions).

### Adding a numeric family

A new family supplies input access, arithmetic and result hooks, then
connects its public overloads to the existing executor. Sequential and tree
traversal stay in the executor. Adapter tests cover separate component buffers,
selected indices, per-value metadata, move-only working values and a distinct
result type.

## Measured performance

The [benchmark results](benchmark-results.md) include sums and dot products
over batches of 1000, with individual timings in the full report. Reduction
cost also depends on operand precision and exponent spread: a dense cluster
of exponents and widely separated terms exercise different accumulator
paths. Preserve those input properties when comparing implementations.
