# Exact fractions

Use [`Rational`](../reference/rational.md) when a calculation must keep
fractions exact. Scaling three quarters of a cup from six servings to ten,
for example, gives exactly five quarters (`5/4`).

<!-- example: docs/examples/rationals.mojo -->

`Rational(numerator, denominator)` takes care of reducing the fraction and
keeping its denominator positive. Arithmetic returns a `Rational` even when
the answer is a whole number. Copies and values returned by `numerator()`
and `denominator()` keep their values if you later update the original.

To convert to a native `Int`, use `Int(value)`. The fraction must be a whole
number that fits in `Int`; otherwise, the conversion raises an error. To
round a fraction to an `Integer`, choose the direction: `floor()` rounds
toward negative infinity, `ceil()` toward positive infinity, and `trunc()`
toward zero.

Dividing integers with `/` also returns a `Rational`. For signed powers, use
`pow_rational`; integer `**` accepts only nonnegative exponents. Because Mojo
variables have fixed types, `integer /= 2` cannot turn an integer variable into
a fraction. Start with a `Rational` or assign the result to a new variable.

## Exact decimal input

Text lets you supply a decimal value without first rounding it to a native
float. `Rational("0.1")` is exactly `1/10`. The constructor also accepts
fractions such as `"3/4"` and scientific notation such as `"1.5e-3"`.

<!-- example: docs/examples/rational_input.mojo -->

JSON saves the reduced numerator and positive denominator as decimal strings.
To limit the input, digit counts, or allocation requests for a conversion,
pass `limits=ConversionLimits(...)`. These limits apply to that call; later
arithmetic has no such budget.

## Batches of fractions

[`Batch[Rational]`](../reference/batch.md) uses the same indexing, slicing,
and selection rules as other batches. Saved copies and slices keep their
values when the original changes.

<!-- example: docs/examples/rational_batches.mojo -->

## Exact batch calculations

Dividing integer batches produces a `Batch[Rational]`. Mixing integer counts
with rational rates also keeps the result exact. Comparisons return masks,
and scalar operands broadcast across the batch. Two vectors must have equal
lengths; higher-rank operations broadcast compatible trailing dimensions.

<!-- example: docs/examples/rational_batch_math.mojo -->

Updates such as `values += adjustment` change only the destination batch.

## Exact totals and weights

`sum` returns an exact total, and `dot` computes an exact weighted sum with
integer or rational weights. The result remains a `Rational` when a rational
batch participates. Divide by a nonzero total weight to obtain a weighted mean.

<!-- example: docs/examples/rational_reductions.mojo -->

The two vectors in a dot product must have equal lengths. `dot` computes the
total without creating an intermediate batch of products. Empty sums return
zero and empty products return one; `min` and `max` need at least one value.
If your values are in a Mojo list, convert it to `Batch[Rational]` first.

## Rounding and signed powers

For a rational batch, `.floor()`, `.ceil()`, and `.trunc()` return a
`Batch[Integer]`. `vmap[rational.abs]()` takes absolute values while keeping
rational elements.

A power can use one scalar exponent or an integer batch of exponents.
`vmap[pow_rational]()` also computes exact reciprocal powers of integer batches.

<!-- example: docs/examples/rational_batch_powers.mojo -->

Zero to a negative power raises an error identifying the element. Zero to
zero is one. `values **= exponent` applies the signed-power rules in place.

## Updating exact quantities

`+=`, `-=`, `*=`, `/=`, and integral `**=` update rational batches. If any
element fails, the whole destination keeps its previous value.

<!-- example: docs/examples/rational_batch_updates.mojo -->

Use `Batch[Rational]` when updates may produce fractions. Masked updates such
as `values[mask] *= rate` affect only the selected elements.

## Saving exact batches

JSON preserves exact fractions across languages. Saving a slice writes only
its selected elements, and decoding restores a new batch. Rank-one batches
use schema version 1; other ranks use version 2 with an explicit shape.

<!-- example: docs/examples/rational_batch_json.mojo -->

One `ConversionLimits` budget covers the whole call, including digits in
both numerators and denominators. JSON input must already be canonical; use
the text constructor to reduce a fraction such as `"2/4"`. Reusing a limits
object starts a new budget for each call.

See the [Rational reference](../reference/rational.md) for the full API.
