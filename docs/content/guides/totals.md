# Counters and weighted totals

Integer batches let counts and totals grow beyond native integer limits.
This example reads values, replaces negative readings, saves a snapshot, and
computes an exact weighted total.

<!-- example: docs/examples/totals.mojo -->

## Choose the operation that matches the calculation

- `sum(counters)` returns the exact integer total.
- `dot(counters, weights)` adds pairwise products without building a product
  batch. The two vectors must have equal lengths.
- `axpy(a, x, y)` computes `a * x + y` for a scalar integer coefficient and
  matching integer vectors, without building a separate scaled batch.
- `to_json()` records the values in a versioned format for later decoding.

`dot` and `axpy` leave their inputs alone and accept strided or reversed
selections. Replacing negative readings is a choice made by this example;
keep them when they represent valid balances or offsets in your application.

A snapshot keeps its backing storage alive. Release snapshots you no longer
need, especially when processing a long-running stream of updates.
