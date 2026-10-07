# How to read the reference

Choose a package below to find its overview and API declarations. The
declarations come from source docstrings and describe signatures,
compile-time parameters, arguments, return values, and errors.

## Packages

| Package | Contents |
|---|---|
| [`apn_mojo`](apn_mojo.md) | Arithmetic dispatch, reductions, `lift`, `axpy`, and commonly used re-exports |
| [`apn_mojo.integer`](integer.md) | Exact integers, division, number theory, and combinatorics |
| [`apn_mojo.rational`](rational.md) | Exact fractions and rational arithmetic |
| [`apn_mojo.float`](float.md) | Binary floats, formats, contexts, rounding, and float utilities |
| [`apn_mojo.complex`](complex.md) | Complex values, component contexts, and arithmetic |
| [`apn_mojo.exact_complex`](exact_complex.md) | Exact complex fractions, rational square roots, and conversion |
| [`apn_mojo.ball`](ball.md) | Real intervals, arithmetic bounds, and set operations |
| [`apn_mojo.complex_ball`](complex_ball.md) | Complex rectangles, enclosing arithmetic, and elementary functions |
| [`apn_mojo.batch`](batch.md) | Multidimensional batches, masks, `vmap`, and NumPy-style functions |
| [`apn_mojo.common`](conversion.md) | Conversion limits, representation keys, and thread controls |

Import common names from `apn_mojo`, or use a family package for its complete
API. For `vmap` and `lift`, choose a family function such as
`apn_mojo.float.add`: it has the single declaration these tools need.
Overloaded root functions such as `apn_mojo.add` cannot be passed as one
compile-time function value.

## Reading a declaration

- Brackets `[...]` hold compile-time parameters. `//` separates inferred
  parameters from explicit ones.
- Arguments after `*`, such as `context=` and `limits=`, are keyword-only.
- A `where` clause limits which types a declaration accepts.
- Internal argument adapters such as `_FloatArgument` let one declaration
  accept several number types while preserving exact inputs. Pass the
  documented values, not manually constructed adapters.
- Source links open the implementation file included with the documentation.

## Common contracts

Integer and rational arithmetic stays exact within storage limits. Each
Float or Complex operation rounds once to its chosen format. Ball operations
return enclosures for their inputs, with an indeterminate result when the
operation is undefined for part of an interval.

Functions that can raise declare `raises`. Checked failures leave update
destinations unchanged. Map scalar functions with `vmap` for batch work;
see the [batch page](batch.md) for supported signatures and shapes.

The [support guide](../guides/status.md) describes platform and feature limits.
