# Arbitrary-precision numbers for Mojo

<div class="hero" markdown="1">

APN Mojo brings arbitrary-precision arithmetic and array programming together
in one pure Mojo library. It includes exact integers and fractions, correctly
rounded floating-point and complex arithmetic, interval arithmetic with
guaranteed bounds, and multidimensional arrays of these number types. Write
scalar functions once, then reuse them across arrays and build reductions,
accumulations, and outer products with its vectorization tools.

All of this runs in Mojo, without a runtime dependency on GMP, MPFR, MPC, or
Python. APN Mojo supports Linux and macOS with Mojo 1.1.0.

<div class="hero-actions" markdown="1">
[Get started](start/installation.md){ .button .primary }
[Tutorial](tutorial/integers.md){ .button }
[API reference](reference/index.md){ .button }
</div>

</div>

<div class="cards" markdown="1">
<div class="card" markdown="1">
[Integers](tutorial/integers.md)

Exact whole numbers, with bit operations, integer roots, modular arithmetic,
primality tests, and combinatorics.
</div>
<div class="card" markdown="1">
[Rationals](tutorial/rationals.md)

Fractions in lowest terms, including exact decimal input such as `"0.1"`.
</div>
<div class="card" markdown="1">
[Floats](tutorial/floats.md)

Choose a binary precision, exponent range, rounding mode, and error traps.
Each arithmetic operation rounds once to the chosen format.
</div>
<div class="card" markdown="1">
[Complex numbers](tutorial/complex.md)

Choose exact rational components with `ExactComplex`, or independently
rounded Float components with `Complex`, including elementary functions.
</div>
<div class="card" markdown="1">
[Ball intervals](tutorial/balls.md)

Carry real intervals or complex rectangles through calculations, and find
out what their guaranteed bounds establish.
</div>
<div class="card" markdown="1">
[Batches and vmap](tutorial/batches.md)

Store multidimensional arrays, select elements with masks, and map scalar
functions across their values.
</div>
<div class="card" markdown="1">
[Reductions](reference/apn_mojo.md#float-sum)

`sum` and `dot` keep intermediates exact and round once. `lift` builds
reductions, accumulations, and outer products from binary functions.
</div>
</div>

## First program

After [installing the package or a checkout](start/installation.md), run this
program from the directory containing `pixi.toml`. From a checkout:

```sh
pixi run --locked mojo run -I src docs/examples/quickstart.mojo
```

With the `apn_mojo` package installed, save it as `quickstart.mojo` and run
`pixi run mojo run quickstart.mojo`, without `-I src`.

<!-- example: docs/examples/quickstart.mojo -->

The integers and fractions in this example stay exact as their storage grows.
The Float calculation uses 256 bits of precision, so `sqrt(2)` returns the
correctly rounded 256-bit approximation. Each operation has this rounding
guarantee; a sequence of operations can still accumulate error.

Supply decimal values as text to avoid rounding them through a native float
first. `Rational("0.1")` is exactly one tenth; `Float("0.1")` rounds one tenth
to a binary approximation. Importing `Float64(0.1)` starts with the
approximation already stored in the native float.

## Number families

| Family | What it holds | Rounding |
|---|---|---|
| `Integer` | Signed whole numbers | Exact, within storage limits |
| `Rational` | Fractions in lowest terms | Exact, within storage limits |
| `ExactComplex` | A real and an imaginary `Rational` | Exact, within storage limits |
| `Float` | Binary floating-point values with a chosen precision and exponent range | Once per operation |
| `Complex` | A real and an imaginary `Float` | Once per component |
| `Ball` | A real interval described by a midpoint and radius | The bounds include input uncertainty and rounding error |
| `ComplexBall` | A rectangle described by two `Ball` components | Each component encloses all possible results |

Integer and rational arithmetic stays exact: `Integer + Rational` returns a
`Rational`. Mixing with a `Float` produces a rounded result. Named functions
such as `add` and `sqrt` accept a context when the selected number family
needs one. See the family references for accepted operands and return types.

## Batches, mapping and reductions

`Batch[T]` holds Integer, Rational, Float, Complex, Ball, or ComplexBall values.
Copies and slices keep their values when the original changes. Integer,
rational, float, and complex batches provide arithmetic operators,
comparisons, reductions, and JSON.
Ball batches support NumPy-style functions such as `batch.add` and `batch.exp`,
and mapping with `vmap` or `lift`. Real Ball batches support `min` and `max`;
both ball families support `cumsum` and `cumprod`, but have no arithmetic
operators or batch JSON. `ExactComplex` is a scalar type, with no batch support.

This example applies a scalar function to a batch, then reduces its results:

<!-- example: docs/examples/batch_quickstart.mojo -->

With `vmap[f]`, you can apply library functions or your own scalar functions
across batch arguments. `lift[f]` builds on binary functions for integers,
rationals, floats, complex numbers, and balls. Large supported operations can
use a worker pool; short runs and some mapping signatures run on the caller
thread. The [batch reference](reference/batch.md) describes those rules.

For floating-point and complex totals, `sum` and `dot` keep intermediates
exact and round only the final result. The
[reduction reference](reference/apn_mojo.md) also covers explicit rounding
orders, axes, contexts, and custom folds.

## Find your starting point

| I want to... | Read |
|---|---|
| Do exact integer arithmetic | [Integer values](tutorial/integers.md) |
| Keep fractions exact | [Exact fractions](tutorial/rationals.md) |
| Compute with a chosen precision | [Floats of any precision](tutorial/floats.md) |
| Work with exact or rounded complex numbers | [Complex numbers](tutorial/complex.md) |
| Carry guaranteed interval bounds | [Calculations with bounds](tutorial/balls.md) |
| Choose precision and check reliable digits | [Precision and accuracy](guides/precision.md) |
| Work on many values at once | [Batches and selections](tutorial/batches.md) |
| Map a custom function or build a fold | [Vectorization with vmap and lift](tutorial/vectorization.md) |
| Choose a thread count | [Thread controls](reference/batch.md#threads) |
| Export native arrays | [Native values](reference/batch.md#native-values) |
| Handle bad input or a failed update | [Errors and recovery](tutorial/errors.md) |
| Save values without losing digits | [Text and JSON](tutorial/conversion.md) |
| Look up a function or type | [API reference](reference/index.md) |
| Browse runnable examples | [Examples](examples/index.md) |
| Understand the implementation | [Architecture](architecture/overview.md) |

See [support and limitations](guides/status.md) for supported platforms and
library boundaries.
