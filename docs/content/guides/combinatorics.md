# Exact combinatorics

Factorials and binomial coefficients quickly outgrow native integers: `20!`
fits in a signed 64-bit integer, but `21!` does not. APN's combinatorial
functions return exact [`Integer`](../reference/integer.md) values and accept
native integers and arbitrary-width integer literals.

- `factorial(n)` computes `n!`, the number of permutations of `n` distinct items.
- `comb(N, k)` counts ways to choose `k` items from `N` (scipy's `comb`), and
  `comb(N, k, repetition=True)` the multisets of `k` items from `N` kinds.
- `factorial2(n)` multiplies every second positive integer down to 1 or 2.
  For example, `factorial2(9)` counts pairings of ten distinct items.

<!-- example: docs/examples/combinatorics.mojo -->

## Exact division and boundaries

Use `div_exact(a, b)` when the quotient must be an integer. It raises if the
divisor is zero or leaves a remainder, which helps catch a broken assumption
in a formula. Use `//` for floor division, or a `div_rem_*` function when you
need a quotient and remainder under a chosen sign convention.

Both factorial functions require nonnegative inputs and return one for zero.
As in scipy, `comb(N, k)` is zero unless `0 <= k <= N`.

Use `vmap` to apply a combinatorial function across a batch, sharing any
scalar arguments between calls. Results can grow beyond a fixed integer
width, though large calculations still take time and memory.

For a larger example, try `comb(200, 100)`. You can also check the symmetry
`comb(n, k) == comb(n, n-k)` for `0 <= k <= n`. The
[Integer reference](../reference/integer.md) lists the related functions.
