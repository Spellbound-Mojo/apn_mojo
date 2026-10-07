# Floats of any precision

With [`Float`](../reference/float.md), you choose how many bits of precision
your calculation needs. Each arithmetic operation returns the exact
mathematical result rounded once to its destination format. That guarantee
applies to each operation; a longer calculation can still accumulate rounding
error.

## Choose a format

Start with a `FloatFormat` to set the significand precision and exponent
bounds. An `ArithmeticContext` adds a rounding mode and traps for conditions
such as division by zero. The `Float` remembers its format, while rounding
modes and traps apply to the call that uses them.

<!-- example: docs/examples/float_values.mojo -->

Floats print the shortest decimal that reads back as the same value in their
format. In the example, the stored value is `0.3`: in a 3-bit format no
shorter decimal reads back as `5 * 2**-4`, the nearest approximation to one
third, and `to_string(16)` shows that exact value as `0x5p-4`.
Increasing precision preserves that stored value when it fits the new exponent
range; it does not make the original approximation more accurate.

Comparisons with integers and fractions use the exact stored value. Positive
and negative zero compare equal, and `NaN` compares unequal to every value,
including itself.

## Arithmetic rounds once

Operators derive a result format from their operands and use nearest-even
rounding without condition traps. Named functions such as `add`, `subtract`,
`multiply`, and `divide` accept `context=` for an explicit destination format,
rounding mode, or traps. Compound assignments keep the destination's format.

<!-- example: docs/examples/float_arithmetic.mojo -->

Exact integer and rational operands keep their full value until the final
rounding. A typed native float contributes the binary approximation it already
holds. Dividing a finite nonzero value by zero returns a signed infinity by
default; `trap_divide_by_zero=True` makes the call raise. `0 / 0` is an invalid
operation and returns `NaN` unless the invalid condition is trapped.

Adding an `Integer` and a `Rational` without a context returns an exact
`Rational`. Passing an arithmetic context to the top-level `add` instead asks
for a `Float` rounded to that format.

## Functions

`sqrt`, `square`, `pow_int`, `ldexp`, and `fma` follow the same rounding rule.
`fma(a, b, c)` computes `a * b + c` with no rounding between the product and
the sum. This can preserve a small residual that separate multiplication and
addition would lose; the final result still rounds if necessary.

<!-- example: docs/examples/float_functions.mojo -->

`floor`, `ceil`, and `trunc` return an `Integer` after the requested rounding.
`to_integer_exact()` raises if a finite value is not integral. For native
floating-point output, `to_native[dtype]()` rounds directly to the requested
type, while `to_native_exact[dtype]()` requires an exact conversion.

Use `sum` and `dot` for batch totals that round only once. To carry interval
bounds through a calculation, use [`Ball`](balls.md). The
[precision guide](../guides/precision.md) shows how to choose working bits,
avoid cancellation, and certify the digits you print.

## Constants and elementary functions

`pi`, `euler_e`, `ln2`, and the other constants use the requested context.
So do exponentials, logarithms, trigonometric and hyperbolic functions, and
their inverses. Trigonometric arguments are in radians. Exact Integer and
Rational arguments retain their value until the function's final rounding.

<!-- example: docs/examples/float_elementary.mojo -->

Use `sin_cos` when you need both results. For a small increment, `log1p(x)`
computes `log(1 + x)` without rounding away `x` in an intermediate addition;
`expm1(x)` similarly avoids the cancellation in `exp(x) - 1`. Each returned
value is correctly rounded, even when the example prints fewer digits.

## Special functions

The library includes gamma and beta functions, error functions, normal
probabilities and quantiles, zeta, Lambert W, and hypergeometric functions.
Names follow `scipy.special`; the [reference](../reference/float.md#special-functions)
states their domains, branches, normalizations, and special values.

<!-- example: docs/examples/special_functions.mojo -->

Here `ndtr` is the standard normal cumulative probability, and `ndtri`
finds its quantile. `gammaincc(a, x)` gives the regularized upper incomplete
gamma function, also the upper-tail probability for a gamma distribution
with shape `a` and scale 1. Use it directly instead of subtracting
`gammainc(a, x)` from 1 when a small upper tail matters. Likewise,
`log_ndtr` computes a log probability without first rounding the probability.

The real-valued functions also have [Ball versions](../reference/ball.md#special-functions)
for enclosing uncertain inputs. Complex elementary functions are covered in
the [next tutorial](complex.md#elementary-functions); special-function
availability differs by family.

## Text input and formatting

How you supply a number affects the result. `Float("19.95")` parses the
decimal value and rounds it once to the chosen binary format.
`Float(Float64(19.95))` starts from the approximation already stored in the
native float. In either case, a non-binary fraction still needs rounding to
fit a finite binary format.

Printing and `to_string()` write positional decimals, like `0.00001` and
`3.0`, from 1e-6 up to 1e21, and scientific notation beyond, as JavaScript
does; `notation="positional"` or `notation="scientific"` forces one.
`to_string(digits=n)` writes `n` significant digits with the requested
rounding mode. `notation="hexadecimal"`, or `to_string(16)`, writes the exact
value, which suits tests and benchmarks. Neither form keeps the format; use
JSON when you also need to save precision and exponent bounds.

<!-- example: docs/examples/float_text.mojo -->

The [Float reference](../reference/float.md) also covers neighboring values,
shortest decimal output, representation comparisons, and keys for caching.
Continue with [complex numbers](complex.md) for arithmetic on pairs of floats.
