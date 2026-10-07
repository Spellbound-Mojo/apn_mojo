# Rational

`Rational` stores a fraction in lowest terms. Cancelling factors before
multiplication keeps intermediate values smaller without changing the answer.
See the [Rational reference](../reference/rational.md) for public methods.

## Canonical form

A `Rational` stores two [`Integer`](integer.md) values: the numerator and
denominator. Construction divides out their greatest common divisor (GCD)
and moves the sign to the numerator, leaving a strictly positive denominator.
Zero is always `0/1`.

Every public operation preserves this form, so equality can compare the two
components directly without reducing them again. Copies follow `Integer`'s
copy-on-write rules, and accessors return independent values.

The two families share their scalar implementations: true division on
`Integer` (`/`) returns an exact `Rational`, and `Rational` uses `Integer`
components and the GCD implementation in the `Integer` package
(see [Integer](integer.md#arithmetic)).

## Arithmetic

Arithmetic cancels factors before forming products to keep intermediate
values small:

- **Addition and subtraction**: Calculate $\gcd(d_1, d_2)$ upfront to determine
  the least common denominator, multiplying only by required cofactors and
  performing a single final reduction on any remaining common factors. When
  the denominators are coprime the sum is already in lowest terms: a prime
  dividing one denominator divides neither the other nor its own numerator, so
  it cannot divide $n_1 d_2 + n_2 d_1$. Both reductions are then skipped.
- **Multiplication and division**: Cross-cancel before forming products.
  Multiplication cancels numerator/denominator factors across the operands;
  division applies the corresponding cancellation after taking the reciprocal
  of the divisor. Equal operands multiply without cancelling: the parts of a
  fraction in lowest terms are coprime, and so are their squares.
- **Ordered comparisons**: Use continued fractions to compare the operands
  without forming the full cross-products ($n_1 d_2 \text{ vs. } n_2 d_1$).
  The comparison can stop as soon as their order is known. The quotient
  and remainder calculations can still allocate temporaries.
- **Inline parts**: When all four parts fit an `Int64`, sums, products,
  quotients and comparisons run in 128-bit integers with the same reductions:
  numerators stay below $2^{127}$, denominators and cross products below
  $2^{126}$. A gcd of one skips its divisions, each a hardware divide, and the
  results become Integers without an allocation when they fit inline.
- **Signed powers**: Evaluate component-wise integer powers ($n^k / d^k$), swapping
  numerator and denominator when exponents are negative. Large exponents never
  narrow to native integers, and constant-value shortcuts (such as $1^k$ or $0^k$)
  return immediately.

`floor`, `ceil`, and `trunc` return exact `Integer` values. Compound updates
compute the new fraction before assigning it back. If division by zero or
another checked error occurs, the destination and all existing copies keep
their values.

## Batches

`Batch[Rational]` stores reduced fractions in the shared layout described in
[Batches and execution](batches.md). Arithmetic, comparisons, unary functions,
and updates call the scalar kernels for each element and write into temporary
destination buffers.

Mixed operations read integer operands with a virtual denominator of 1,
avoiding an intermediate converted batch. Shapes are checked before arithmetic.
Element failures name the logical index; shape or configuration errors can
occur before any element is read.

Reductions (`sum`, `prod`, `min`, `max`, and mixed `dot`) read elements directly
and reduce fractions at each step. A dot product needs no intermediate batch
of products; see [Reductions](reductions.md).

## Text and JSON

Text parsing accepts exact decimal strings (such as `"0.1"`, which parses as
$1/10$) as well as standard fractional notation (`"numerator/denominator"`).
JSON stores both components as canonical decimal strings and rejects records
that are not already reduced.

With `ConversionLimits`, one allocation budget covers JSON decoding, decimal
powers, GCD reductions, and string formatting. A failed conversion discards
its partial buffers and leaves the destination unchanged
(see [Conversion and interchange](conversion.md)).

## Measured performance

Rational arithmetic depends heavily on GCD and division. The
[benchmark results](benchmark-results.md) compare Rational arithmetic with
GMP and FLINT; the full report gives individual operand sizes. Costs depend
on the fractions and the cancellation they allow.
