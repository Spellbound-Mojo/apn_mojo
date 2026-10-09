# apn_mojo.integer

The integer family provides exact signed integers, three division
conventions, roots, number theory, combinatorics, and modular arithmetic. See the
[Integer tutorial](../tutorial/integers.md) for construction and basic use.

<!-- api: integer -->

## Mixing families

Arithmetic with native integers up to 64 bits and arbitrary-width integer
literals stays exact. An integer combined with a `Rational` produces a
`Rational`; with a `Float` or typed native float, it produces a rounded
`Float`. Result types follow the operation, not the size of the answer.

Comparisons between library values use their exact mathematical values.
Put the library value on the left of a comparison with a typed native value,
or explicitly construct a suitable library value first. Native integer
constructors are limited to 64 bits; use text or integer literals for wider
initial values.

An `Integer` variable keeps its type. `/=` cannot store a fraction in it, and
`+=` cannot silently replace it with a `Rational`. Use a rational destination
when results may be fractional, or call `to_integer_exact()` on a fraction
that must be integral.

Integer `**` requires nonnegative exponents. Use
[`pow_rational`](rational.md#pow_rational) for reciprocal powers. For a
quotient and remainder, choose `div_rem_floor`, `div_rem_trunc`, or
`div_rem_euclid`; standard `divmod` is not supported. Checked size guards
reject results that exceed addressable storage.

## Text and bytes

`to_string(base)` and `Integer(text, base=...)` accept bases 2 through 36.
Power-of-two bases convert in time linear in the length; other bases in
quadratic time, many digits per pass (see
[Exact text](../architecture/conversion.md#exact-text)).
For binary interchange, `to_bytes()` returns the magnitude in base 256, least
significant byte first unless `big_endian=True`, and
`Integer.from_bytes(bytes, negative=...)` rebuilds the value; the sign is
`sign()`, and zero has no bytes.

## Functions on batches

Use [`vmap`](batch.md#mapping-functions) to apply a scalar function to batch
elements. For example, `vmap[gcd]()(values, 12)` shares the scalar divisor,
and `vmap[div_rem_floor]()(a, b)` returns quotient and remainder batches.
Mapped axes must have equal extents; a one-element batch is not a scalar.
An element failure discards partial results and reports its logical index.

<!-- example: docs/examples/functions.mojo -->

See [Exact combinatorics](../guides/combinatorics.md) and
[Counters and weighted totals](../guides/totals.md) for applications.
Arithmetic is not constant-time and makes no cryptographic guarantee.

## Number theory

The API includes factor removal, trial division, exact roots, perfect powers,
primality tests, and the Jacobi symbol. `is_prime` is deterministic below
`2**64` and uses the Baillie–PSW probable-prime test above that range. See
[Integer architecture](../architecture/integer.md#number-theory) for the
algorithms. Rational `gcd`, `lcm`, and `root_exact` live in
[apn_mojo.rational](rational.md).

`factor(n)` returns the sign and prime powers of `n` in ascending order. It
tries division below `FactorBudget.trial_bound`, perfect-power reduction,
and then Pollard and Brent's rho method. Rho shares at most `rho_iterations`
steps across all parts, with a default budget of `2**24`. In the recorded
tests, that budget handled products of four 40-bit primes; a 256-bit composite
that rho could not split took about 8 seconds to exhaust it.

An unsplit part is reported with `is_prime = False`, making `is_complete()`
false; a composite is never reported as prime.
For example, `factor(2**64 + 1)` is `274177 * 67280421310721`.
`primes_below(n)` lists the primes below `n` by a segmented sieve, and
`next_prime(n)` returns the least prime above `n`.

<!-- example: docs/examples/number_theory.mojo -->

## A tour of Integer

This example covers text and JSON, conversion limits, division rules,
number-theoretic functions, and batch updates.

<!-- example: docs/examples/integer_tour.mojo -->
