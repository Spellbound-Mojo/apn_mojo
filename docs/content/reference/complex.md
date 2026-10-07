# apn_mojo.complex

`Complex` lets you configure its real and imaginary Floats separately and
rounds each arithmetic result once per component. See the
[Complex tutorial](../tutorial/complex.md) for examples,
[Batch](batch.md#complex-batches) for arrays, and
[reductions](apn_mojo.md) for totals and dot products.

<!-- api: complex -->

## Formats and contexts

Float components retain their formats when no context overrides them.
An exact component adopts its partner's format; two exact components default
to 128 bits. Typed native floats contribute their own precision. Exponent
bounds must agree. Operators merge precision separately for the two parts.

An `ArithmeticContext` applies the same settings to both components.
A `ComplexContext` selects them separately, with shared exponent bounds.
Parsing validates both component inputs before rounding either one.

Components accept integers, rationals, Floats, integer literals, native
integers up to 64 bits, and typed native floats. Decimal text or a `Rational`
preserves the exact source until the component's final rounding; the resulting
binary Float can still be an approximation.

## Rules and special values

Real operands affect only the parts that the operation calls for. Adding a
real positive zero to `(1, -0)` preserves the imaginary negative zero;
adding a complex `(+0, +0)` can change it. For nonfinite values, arithmetic
follows the MPC rules described in
[Complex architecture](../architecture/complex.md#special-values).

`abs` returns the magnitude, as NumPy's does, and `norm_sqr` returns its
square. Both return a Float and use the larger component precision unless
you supply a context. The family also provides `angle`, `conjugate`, `real`,
and `imag`. `sqrt` returns the principal square root, respecting the imaginary
zero's sign on a branch cut. Integer powers accept exponents of either sign
and arbitrary width. A part of a power that is exactly zero takes MPC's sign
for a base on an axis, such as `(3i)**-2`, and for an inexact power of a
diagonal base; an exact power of a diagonal base, such as `(1 + i)**12`,
follows the exponent's phase, which can differ from MPC's sign there.

Equality compares stored values across formats. Zero signs compare equal,
a NaN component makes equality false, and comparison with a real value
requires a zero imaginary part. Typed native values belong on the right.
For representation-based dictionary keys, use `ComplexKey`.

## Elementary functions

`exp`, `log`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
`asin`, `acos`, `atan`, `asinh`, `acosh` and `atanh` round each part correctly
in its own mode: each part is the exact part rounded once, as MPC gives it,
and `angle` returns the argument as a correctly rounded Float. Special values
follow C99 Annex G, and a part that is zero for every argument of its kind,
such as the imaginary part of `exp(x + 0i)`, is an exact zero with the sign
the annex gives.

On a branch cut the sign of a zero part selects the side: `log(-1 + 0i)` is
`pi i` and `log(-1 - 0i)` is `-pi i`; without a signed zero, a point on a cut
takes the counter-clockwise continuous value. `pow(z, w)` is
`exp(w log z)`, and exact when it can be: `pow(-4, 1/4)` is `1 + i`.

Like the Float functions, these functions certify their rounding within the
context's `max_precision` and raise if that budget is insufficient. See
[Correct rounding](../architecture/certified.md).

## Text and JSON

Accepted text includes `3`, `4j`, `-j`, `3+4j`, `3-4i`, `3+j`, `(3+4j)`,
`(3,4)`, and `Complex(3, 4)`. Components follow Float grammar rules;
`1e+2-3e-1j`, for example, describes `100 - 0.3j` before rounding.
Suffixes are lowercase `i` or `j`.

<!-- example: docs/examples/complex_interchange.mojo -->

`to_string()` writes each component as a Float does, by default the shortest
decimal; `to_string(16)` writes them exactly, without format metadata. JSON
preserves the two values and formats. A Complex value counts as one logical
value under `ConversionLimits`; one budget covers both components.
