# apn_mojo.float

`Float` provides binary floating-point arithmetic with a precision, format,
rounding mode, and traps you can choose. The
[Float tutorial](../tutorial/floats.md) introduces the main
choices; [Batch](batch.md#float-batches) covers arrays of floats.
For working precision, cancellation, and certified digits, see
[precision and accuracy](../guides/precision.md).

<!-- api: float -->

## Format selection

If you omit the context, arithmetic combines the operand formats as follows:

- Integers, rationals, and integer literals contribute no precision bits.
- Native floats contribute their significand precision: 11 bits for Float16,
  8 for BFloat16, 24 for Float32, and 53 for Float64. They impose no exponent bounds.
- Library Floats contribute their stored precision and exponent bounds.
  Bounds must agree when no context overrides them.

An `ArithmeticContext` sets the output format, rounding mode, and traps.
To keep a value's format while choosing other settings, use
`format=value.format()`. Compound assignments keep their destination format.
Top-level arithmetic on exact inputs stays exact unless a context requests a
Float result.

Bare decimal literals are not accepted as floating-point inputs. Use
`Float("0.1")` to round the decimal value directly, or `Float(Float64(0.1))`
to import the native float's existing binary approximation.

## Special values

Operators use nearest-even rounding without condition traps. Named functions
accept contexts that can trap inexact, underflow, overflow, divide-by-zero,
and invalid conditions. Status is internal; there is no mutable global flag set.

Quiet NaNs propagate. `0/0`, `inf/inf`, `0 * inf`, and `inf - inf` produce NaN
and the invalid condition. A finite nonzero value divided by signed zero gives
a signed infinity and divide-by-zero. Multiplication and division combine signs
with XOR. Like-signed zeros add to that sign; opposite zeros and exact
cancellation give negative zero only when rounding toward negative infinity.

`sqrt(-0)` is negative zero. A negative nonzero square-root input produces
NaN and invalid. Any value to the zeroth power is one, including NaN and
infinity. Odd powers preserve the sign of a negative zero or infinity;
even powers clear it. Integral conversions reject NaNs and infinities.

## Comparisons

Comparisons use the exact stored values, without rounding them to a common
format. Their exponent bounds need not match. Integer and rational comparisons
are exact. The two zero signs compare equal; NaN compares unequal to every
value, including itself. `total_cmp` provides a total ordering of values.

`Float` has no hash under this equality. To cache values by representation,
use `FloatKey`. It distinguishes formats and zero signs, and treats a
canonical NaN in one format as the same key. For direct representation
comparisons, use `same_representation` and `representation_cmp`. See
[representation identity](../architecture/float.md#representation-identity-and-hashing).

With typed native numbers, put the library value on the left of the comparison
or explicitly convert the native value to an appropriate library type.

## Native output

`to_native[dtype]()` rounds directly to Float16, BFloat16, Float32, or Float64,
including native subnormals, with a selected rounding mode.
`to_native_exact[dtype]()` raises when an exact conversion is impossible.
`to_rational_exact()` returns the stored finite binary fraction exactly.

`floor`, `ceil`, and `trunc` produce integers in the named direction.
`round` rounds to the nearest integer with ties to even, as NumPy's does, and `is_integer()` tests for an integral value.

## Text and JSON

Floats print the shortest decimal that reads back as the same value in their
format: `0.1`, `3.0`, `0.00001`. From 1e-6 up to but excluding 1e21 the
notation is positional, beyond it scientific, as JavaScript prints numbers:
`1e-07`, `1.1805916207174113e+21`. Zero, infinity, and NaN print as `0.0`,
`-0.0`, `inf`, `-inf`, or `nan`. `to_string()` writes the same text and takes
options: `notation="positional"`, `"scientific"` or `"hexadecimal"`, and
`digits=n` with a rounding mode. Hexadecimal text, also `to_string(16)`,
shows the exact value: `0x3p0` is 3, and `0x1p-1` is one half. Text omits
precision and exponent bounds; JSON keeps the format as well as the value.

<!-- example: docs/examples/float_interchange.mojo -->

Decimal text accepts `e` exponents. Hexadecimal (`0x`) and binary (`0b`)
text use `p` exponents for powers of two. A decimal printed by a Float reads
back exactly with that Float's format: `Float(text,
context=ArithmeticContext(format=x.format()))`. `shortest_decimal` finds the shortest decimal that rounds back to the same
value in the target format under nearest-even rounding. Consult its declaration
for the result object and limits.

Large decimal conversions can require large working buffers. Use
`ConversionLimits` where input size is not controlled. JSON details are in
[Conversion and interchange](../architecture/conversion.md#wire-formats).

## Formats, rounding and traps

A finite nonzero value is `(-1)**s * m * 2**(e - p)`, with a `p`-bit
significand and `emin <= e <= emax`. Formats have no subnormals: values below
`2**(emin - 1)` round to zero or the smallest positive value. Nearest-even is
the default; a tie halfway between zero and that smallest value goes to zero.

<!-- example: docs/examples/float_formats.mojo -->

The five modes are `nearest_even`, `toward_zero`, `toward_positive`,
`toward_negative`, and `away_from_zero`. The last always rounds an inexact
magnitude upward; it is not a ties-away nearest mode.

`FloatFormat.binary32()` and `.binary64()` use the corresponding native
normal ranges and precisions, but still have no subnormals. `nextafter`,
`spacing` (signed, as NumPy's), `ulp_distance`, and `equal_at_precision` inspect spacing
and neighboring values; see their declarations for zero, range, and NaN rules.

A trapped condition raises before the destination changes. Ordered reductions
can trap at an intermediate step, while exact `sum` and `dot` apply traps to
their final result. See [ordered reductions](apn_mojo.md#ordered-reductions).

## Elementary functions

`exp`, `expm1`, `exp2`, `log`, `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`,
`sin_cos`, `atan`, `asin`, `acos`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
`acosh`, `atanh`, `pow` and `rootn` round correctly in every mode: each result
is the exact value rounded once, as MPFR gives it. Special values follow C99
Annex F: NaN propagates without a flag, poles such as `log(0)` and `atanh(1)`
are infinities with divide-by-zero, and arguments outside the domain give NaN,
which is invalid. Only trivial results are exact, such as `exp(0) = 1`,
`log2(2**k) = k`, `rootn(-27, 3) = -3` and `pow(16, 0.75) = 8`; exact Rational
arguments such as `1/3` are evaluated exactly, never rounded first.

A function increases its working precision until it can certify the rounding.
The context's `max_precision` sets the ceiling, by default
`max(8 * p, p + 4096)`. If a call needs more, it raises an error naming the
function, argument, and budget. Huge arguments to `sin`, `cos` and `tan` are
the cases that approach this ceiling: argument reduction needs about as many
bits of pi as the argument's exponent. See
[Correct rounding](../architecture/certified.md).

## Constants

`pi`, `euler_e`, `ln2`, `log2_10`, `euler_gamma` and `catalan` return the
constant correctly rounded to the context, using 128 bits and nearest-even
rounding by default. Up to 4096 bits, ln 2 and pi come from compiled tables;
the other constants and wider precisions are computed on each call. The
library does not cache them at run time. For enclosing balls, see
[apn_mojo.ball](ball.md#constants-and-canonical-balls).

## Special functions

`gamma`, `gammaln`, `digamma`, `erf`, `erfc`, `erfi`, `expi`, `sici`,
`shichi`, `fresnel`, `lambertw`, `ndtr`, `log_ndtr`, `erfinv` and `ndtri`,
the two-argument `beta`, `betaln`, `poch`, `zeta`, `polygamma`, `gammainc`
and `gammaincc`, the three-argument `hyp1f1` and `betainc`, and the
four-argument `hyp2f1` use scipy.special's names. Like the elementary
functions, they round correctly in every mode and accept exact Integer and
Rational arguments. `sici`, `shichi` and `fresnel` return pairs.

Special values follow MPFR where it has the function (`gamma`,
`lgamma`, `digamma`, `erf`, `erfc` and `eint`), scipy at the poles of the
gamma ratios, and the limits otherwise: `gamma(n)` is `(n-1)!` rounded once,
`gamma(+-0)` is `+-inf` with divide-by-zero, and a negative integer gives
NaN. `gammaln` is `log |Gamma|`, `+inf` at 0 and the negative integers. `Ci`
and `Chi` are defined for x > 0, so a negative argument gives NaN. Fresnel's
integrals are normalized as `S(x) = int_0^x sin(pi t**2 / 2) dt`.
`lambertw(x, k=0)` is the principal branch, defined for `x >= -1/e`; `k=-1`
is the lower branch on `[-1/e, 0)`. `ndtr(x) = (1 + erf(x/sqrt 2))/2` is
exactly 1/2 at 0, and `log_ndtr` keeps its relative accuracy where `ndtr` is
near 0 or 1.

`beta(a, b)`, `betaln(a, b) = log |B(a, b)|` and `poch(z, m) =
Gamma(z+m)/Gamma(z)` are exact before their one rounding where the value is
rational: `B(a, b)` when both arguments are integers or one is a positive
integer, `(z)_m` for an integer m. At the poles they follow scipy: for a
non-positive integer a, `B(a, b)` is the limit `(-1)**b B(1-a-b, b)` when b is
an integer with `1 - a - b > 0` and `+inf` with divide-by-zero otherwise, and
0 where only `a + b` is a non-positive integer; `(z)_m` is `+inf` where a
factor of its divisor is 0 or, for a non-integer m, where only `z + m` is a
non-positive integer, and 0 where only z is.

`zeta(x, q)` is the Hurwitz zeta function on scipy's domain, x > 1, with a
negative q only for an integer x, and `+inf` at x = 1 and at the non-positive
integers q; `zeta(x, 1)` (and the package-level `zeta(x)`) is the Riemann zeta
function at every x other than 1, exact at the non-positive integers.
`polygamma(n, x) = (-1)**(n+1) n! zeta(n+1, x)` takes an Integer order, is
digamma for n = 0, and is `(-1)**(n+1) inf` at the poles, as scipy's.
`hyp1f1(a, b, x)` is Kummer's `M(a, b, x)`, scipy's unregularized function:
`+inf` at the non-positive integers b, except that a negative integer
`a >= b` ends the series before the pole (so `hyp1f1(-n, -n, x)` is the
truncated exponential), NaN for an infinite a, 1 for an infinite b, and the
limit at an infinite x. Its rational values, such as the polynomials of a
non-positive integer a and `hyp1f1(2, 4, 2) = 3`, are exact.
`gammainc(a, x)` and `gammaincc(a, x)` are the regularized incomplete gamma
functions P and Q = 1 - P, each without cancellation. They are NaN for a < 0
or x < 0 and at a = x = 0, with scipy's values at a = 0, x = 0 and the
infinities.
`hyp2f1(a, b, c, x)` is Gauss's F for real `x <= 1`, and a polynomial at
every x when a or b is a non-positive integer. It is `+inf` at the
non-positive integers c that the polynomial does not end before, and at
x = 1 with `c - a - b <= 0`, as scipy's. For x > 1 otherwise it is NaN,
which is invalid; scipy returns `+inf` there.
`betainc(a, b, x)` is the regularized incomplete beta function, with
scipy's values at the edges, and exact for integer shapes.
`erfinv` and `ndtri` invert erf and ndtr on `(-1, 1)` and `(0, 1)`, with
`+-inf` at the ends and NaN beyond.

The methods and their error bounds are in
[Correct rounding](../architecture/certified.md#special-functions).
