# Errors and recovery

## Read the remedy

When a checked operation fails, it raises a Mojo `Error` naming the operation
and suggesting a remedy. For a batch element, the message includes its
zero-based logical index, even in a strided or reversed view. Text and JSON
readers report UTF-8 byte offsets; an offset can point just after a token
whose decoded value failed validation. Shape or configuration errors can
arise before the calculation reaches any element.

Not every exceptional numerical input raises. Floating-point functions return
special values unless the relevant condition is trapped. Ball functions return
an indeterminate ball when an input interval includes points where the operation
is undefined. See [Float](../reference/float.md#special-values) and
[Ball](../reference/ball.md) for those rules.

Catch an error where the application can choose how to recover. This example
handles a zero divisor and retries with a replacement chosen by the caller:

<!-- example: docs/examples/recovery.mojo -->

## Transactional updates

An update takes effect only after the whole calculation succeeds. If division
fails on the last element of a batch, the destination keeps all its old
values, as do copies and slices. This applies to checked compound operators
and slice or mask assignments. If a parsing call fails, a variable you were
assigning its result to also keeps its previous value.

Correct the input or settings before retrying. Error text is written for
people and is not a stable error-code API.

## Common situations and remedies

| Situation | Remedy |
|---|---|
| Zero divisor in exact arithmetic | Handle zero in the application or supply a nonzero divisor. |
| Negative shift | Use a nonnegative shift count, or shift in the other direction. |
| Negative integer exponent | Use `pow_rational` for an exact reciprocal power. |
| Negative `isqrt` argument | Supply a nonnegative integer. |
| Batch shape mismatch | Operators on two vectors and `vmap`'s mapped axes require equal extents; elementwise `batch.*` and lifted calls also broadcast size-one dimensions. |
| Mask used as a condition | Call `.any()` or `.all()`. |
| Invalid text | Correct the syntax or enable the parser options you intend to accept. |
| Native conversion out of range | Choose a wider native type or keep the library value. |
| Conversion limit exceeded | Reduce the input or raise the relevant `ConversionLimits` setting. |
| Function working-precision budget exceeded | Inspect the argument and `max_precision` in the error, then raise the working ceiling if appropriate. See [precision and accuracy](../guides/precision.md). |
| A ball is too wide to certify a rounded result | Recompute from the original inputs with more precision, or address input uncertainty; see the [interval tutorial](balls.md#extract-a-result-when-the-rounding-is-certain). |

## Unrecoverable system limits

The pinned Mojo runtime can terminate the process if it runs out of physical
memory, even when you use checked size limits and conversion budgets.
Conversion limits also do not cap later arithmetic or execution time. See
[support and limitations](../guides/status.md).
