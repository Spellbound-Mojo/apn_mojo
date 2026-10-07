# apn_mojo.exact_complex

Use `ExactComplex` when a calculation needs both exact fractions and `i`.
It stores two Rational parts, so arithmetic on Gaussian rationals stays
exact. Converting to `Complex` rounds each part to a Float.

The [complex tutorial](../tutorial/complex.md#keep-complex-fractions-exact)
includes a runnable example of exact arithmetic, optional square roots,
JSON, and conversion. `ExactComplex` is currently scalar-only: it is not
a `Batch` element type or a supported `vmap` result.

<!-- api: exact_complex -->

## Arithmetic and powers

`+`, `-`, `*` and `/` are exact on ExactComplex values, and on exact Rational,
Integer or literal operands on either side; division by zero raises. There is
no implicit conversion into `ExactComplex`, so write `ExactComplex(1, 2)`.
`pow_int(z, n)` takes an exponent of any sign: `pow_int(z, 0)` is 1, even for
`z = 0`, and a negative exponent takes the reciprocal.

## Square roots

`sqrt_exact(z)` returns the principal square root when its parts are
rational, and None otherwise: `sqrt_exact(2i)` is `1 + i`,
`sqrt_exact(3 + 4i)` is `2 + i` and `sqrt_exact(-4)` is `2i`, while
`sqrt_exact(-2)` and `sqrt_exact(i)` are None. The principal root has a
nonnegative real part, and a nonnegative imaginary part when its real part is
0; either both parts of the root are rational or neither is.

## Text, JSON and hashing

The text form, such as `ExactComplex(1/2, 3)`, can be read back by the constructor.
JSON is a version-1 record with two canonical Rational records, `real` and
`imag`. Equal values have equal hashes, so you can use an ExactComplex as a
`Dict` key. `stable_hash` gives the APNH-64 hash, tag 5, which never changes
between releases. `to_complex` rounds each part once to a `Complex`.
