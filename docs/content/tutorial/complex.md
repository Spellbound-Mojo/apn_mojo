# Complex numbers

Choose `ExactComplex` for arithmetic on exact rational components, or
`Complex` for rounded components and elementary functions. For uncertain
components with guaranteed bounds, use
[ComplexBall](balls.md#rectangles-in-the-complex-plane).

Use [`Complex`](../reference/complex.md) for complex arithmetic with a
precision you choose for each [`Float`](../reference/float.md) component.
Arithmetic rounds each component once. Multiplication, for example, rounds
`ac - bd` and `ad + bc` without first rounding the four products.

## Values and components

Construct a complex number from real numeric components or from text such as
`"3-4j"`. A `ComplexContext` sets separate precisions, rounding modes, and
traps for the two parts. An `ArithmeticContext` applies the same settings to
both. The component formats must share exponent bounds.

<!-- example: docs/examples/complex_values.mojo -->

Components print as Floats do, in decimal: `Complex(3.0, -4.0)`. Use `real()`
and `imag()` to read the components. Each returns an
independent value that you can change without affecting the complex number.

## Arithmetic

You can mix complex numbers with integers, rationals, and floats in either
operand order. Real operands retain their meaning: adding a real zero to a
complex number preserves the imaginary component, including its zero sign.

<!-- example: docs/examples/complex_arithmetic.mojo -->

When the result fits the destination formats, `(3+4j) * (1-2j)` is exactly
`11-2j`, and `25 / (3+4j)` is `3-4j`. Division rounds each quotient component
once, including the denominator's contribution. `z *= z` keeps the formats
of `z`; a named function with `context=` can choose other formats.

## Magnitudes and square roots

`abs(z)` returns the magnitude, and `norm_sqr(z)` returns its square. Both
return a `Float`. Use `sqrt(z)` for the principal square root.

<!-- example: docs/examples/complex_functions.mojo -->

These functions account for both components before rounding. On the negative
real axis, the imaginary zero's sign selects the side of the branch cut:
`sqrt(-4+0j)` is `2j`, and `sqrt(-4-0j)` is `-2j`.

## Powers and phase cycling

`z ** n` and `pow_int(z, n)` accept positive, zero, and negative integer
exponents, including exponents too large for a native integer.

<!-- example: docs/examples/complex_powers.mojo -->

For example, `(1+1j) ** -2` is exactly `-0.5j` when the format can hold it.
Powers of the imaginary unit repeat every four steps, so even a huge exponent
can be inexpensive. Other powers may need more work and storage.

## Elementary functions

`exp`, `log`, trigonometric and hyperbolic functions, their inverses, and
general `pow` accept Complex inputs. Each result component rounds once.
Use a context to choose the destination precision, just as for arithmetic.

<!-- example: docs/examples/complex_elementary.mojo -->

The logarithm in the example is the principal value: `log(Complex(-2))`
has real part `log(2)` and imaginary part `pi`. Branch cuts matter when
moving between these functions and their inverses; see the
[branch conventions](../reference/complex.md#elementary-functions).
Real Float functions keep their real domain: explicitly construct a Complex
when a calculation needs a complex result.

## Keep complex fractions exact

`ExactComplex` stores two Rationals. Arithmetic and integer powers stay exact;
division by zero raises. `sqrt_exact` returns an Optional value, succeeding
only when both components of the principal square root are rational.

<!-- example: docs/examples/exact_complex_values.mojo -->

Call `to_complex(context=...)` when you are ready to round the components
or use elementary functions. Text, JSON, and dictionary keys preserve exact
values. `ExactComplex` currently has no `Batch` or `vmap` support; convert
the elements to `Complex` or `ComplexBall` before constructing those batches.
See the [ExactComplex reference](../reference/exact_complex.md) for the full API.

See the [Complex reference](../reference/complex.md) for special values and
conversion rules, and the [architecture chapter](../architecture/complex.md)
for the algorithms that establish correct rounding.
