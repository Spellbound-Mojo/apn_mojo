# APN Mojo

APN Mojo brings arbitrary-precision arithmetic and array programming together
in one pure Mojo library. It includes exact integers and fractions, correctly
rounded floating-point and complex arithmetic, interval arithmetic with
guaranteed bounds, and multidimensional arrays of these number types. Write
scalar functions once, then reuse them across arrays and build reductions,
accumulations, and outer products with its vectorization tools.

All of this runs in Mojo, without a runtime dependency on GMP, MPFR, MPC, or
Python.

![Mojo 1.1.0](https://img.shields.io/badge/Mojo-1.1.0-orange)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-blue)

APN Mojo supports **Linux and macOS with Mojo 1.1.0**. See the
[installation guide](docs/content/start/installation.md) for system requirements
and [support and limitations](docs/content/guides/status.md) for library boundaries.

## Getting started

APN Mojo is published as `apn_mojo` in the
[Modular community channel](https://github.com/modular/modular-community),
available once [modular/modular-community#392](https://github.com/modular/modular-community/pull/392) is merged. Add the channel to your
[Pixi](https://pixi.sh) project and install the package. It is verified with
Mojo 1.1.0 on Linux x86-64 and macOS arm64:

```toml
# pixi.toml
[workspace]
channels = ["https://conda.modular.com/max", "https://repo.prefix.dev/modular-community", "conda-forge"]
```

```sh
pixi add apn_mojo "mojo==1.1.0"
```

Save [first_integer.mojo](docs/examples/first_integer.mojo) beside your
project's `pixi.toml`, then run it; the installed package needs no `-I` flag:

```sh
pixi run mojo run first_integer.mojo
```

For a new project, the [installation guide](docs/content/start/installation.md)
walks through creating the environment and running a first program.

To work from a source checkout instead, install [Pixi](https://pixi.sh/),
then get the source and its dependencies:

```sh
git clone https://github.com/Spellbound-Mojo/apn_mojo.git
cd apn_mojo
pixi install --locked
pixi run --locked mojo --version
```

Run the remaining commands from the checkout root, which contains
`pixi.toml`. `--locked` keeps the compiler and dependency versions fixed to
`pixi.lock`.

Try a calculation beyond the range of a native integer:

```mojo
from apn_mojo import Integer


def main() raises:
    var value = Integer(2) ** 100
    var saved = value
    value += 1
    print("2 ** 100:", saved)
    print("plus one:", value)
    print("unchanged copy:", saved == Integer(2) ** 100)
```

Run this [example](docs/examples/first_integer.mojo) with:

```sh
pixi run --locked mojo run -I src --Werror docs/examples/first_integer.mojo
```

It prints:

```text
2 ** 100: 1267650600228229401496703205376
plus one: 1267650600228229401496703205377
unchanged copy: True
```

Starting with `Integer(2)` keeps the power exact. Updating `value` leaves
the saved copy unchanged.

From a checkout, include `-I src` when running your own programs so Mojo can
find the library; from another project, pass the absolute path to this
checkout's `src` directory with `-I` and use the supported Mojo compiler. With
the package installed, no `-I` flag is needed.
Next, try `pixi run --locked example` for an Integer tour, or
[quickstart.mojo](docs/examples/quickstart.mojo) for fractions, floats, and
complex numbers too.

## Choosing a number type

| Type | When to use it |
|---|---|
| [`Integer`](docs/content/tutorial/integers.md) | Whole-number calculations that stay exact beyond native integer limits. |
| [`Rational`](docs/content/tutorial/rationals.md) | Fractions and decimal quantities that must stay exact. |
| [`Float`](docs/content/tutorial/floats.md) | Binary floating-point arithmetic with a precision and rounding mode you choose. |
| [`ExactComplex`](docs/content/tutorial/complex.md#keep-complex-fractions-exact) | Exact complex arithmetic with rational real and imaginary parts. |
| [`Complex`](docs/content/tutorial/complex.md) | Complex arithmetic with a separate precision for each part. |
| [`Ball`](docs/content/tutorial/balls.md) | Guaranteed bounds on real calculations, including input uncertainty and rounding error. |
| [`ComplexBall`](docs/content/tutorial/balls.md#rectangles-in-the-complex-plane) | Guaranteed bounds on complex calculations, using a ball for each part. |

Integers and rationals grow as needed, within memory and library size limits.
Division with `/` keeps the fraction: `Integer(7) / 3` is the exact Rational
`7/3`, while `Integer(7) // 3` is `2`. The integer API also provides bit
operations, roots, modular arithmetic, primality tests, factorials, and
binomial coefficients.

For numerical work, the library includes constants such as `pi`, exponentials,
logarithms, trigonometric and hyperbolic functions, gamma and beta functions,
normal probabilities and quantiles, zeta, Lambert W, and hypergeometric
functions. Start with the [elementary functions](docs/content/tutorial/floats.md#constants-and-elementary-functions)
and [special functions](docs/content/tutorial/floats.md#special-functions)
examples; the references describe availability and domains for each family.

### Precision and rounding

Choose the precision in bits and exponent range with `FloatFormat`.
An `ArithmeticContext` adds a rounding mode and traps for conditions such as
division by zero. For example, `ArithmeticContext(format=FloatFormat(256))`
selects 256-bit precision. Contexts apply to individual calls, so each part
of your program can use the settings it needs. Use `ComplexContext` for
separate real and imaginary settings.

Each floating-point operation rounds its mathematical result once to the
chosen format. A series of operations can still accumulate rounding error,
but `sum` and `dot` keep intermediates exact and round only the final total.
If you need rounding at each step in a defined order, use `sum_sequential`
or `sum_tree`.

Supply decimals as text to avoid an initial rounding through a native float.
`Rational("0.1")` is exactly one tenth; `Float("0.1")` rounds one tenth to a
binary approximation. `Float(Float64(0.1))` starts with the approximation
already stored in the native float. Extra precision cannot recover digits
lost before conversion.

The [precision and accuracy guide](docs/content/guides/precision.md) connects
bits to decimal digits, explains cancellation, and shows how to certify a
rounded result and its displayed digits.

### Interval bounds with Ball

A `Ball` stores a midpoint `m` and a nonnegative radius `r`, representing
the interval `[m - r, m + r]`. Arithmetic carries these bounds through the
calculation, including rounding error, so the result contains every possible
answer for the input intervals. Choose the midpoint precision with
`BallContext`.

These bounds let you test whether a ball contains a value, whether two balls
overlap, or whether a comparison holds throughout their intervals. An
operation undefined for part of its input, such as division by an interval
containing zero, returns an indeterminate ball. See the
[interval tutorial](docs/content/tutorial/balls.md) for a measurement example
and certified conversion, or the [Ball reference](docs/content/reference/ball.md)
for domain rules and set operations.

## Working with batches

`Batch[T]` holds multidimensional arrays of `Integer`, `Rational`, `Float`,
`Complex`, `Ball`, or `ComplexBall` values. You can index, slice, reshape,
and select elements with a `Mask`. Use the NumPy-style `batch.*` functions
for elementwise calculations, array construction, and reductions:

```mojo
from apn_mojo import ArithmeticContext, Batch, FloatFormat, Integer, batch


def main() raises:
    var values = Batch[Integer]([1, 4, 9, 16])
    var context = ArithmeticContext(format=FloatFormat(256))
    var roots = batch.sqrt(values, context=context)
    print("roots:", roots)
    print("sum:", batch.sum(roots, context=context))
```

Run [batch_quickstart.mojo](docs/examples/batch_quickstart.mojo) with:

```sh
pixi run --locked mojo run -I src --Werror docs/examples/batch_quickstart.mojo
```

```text
roots: [1.0, 2.0, 3.0, 4.0]
sum: 10.0
```

Each root uses the 256-bit output format; `sum` rounds the exact total of
the stored roots once. Elementwise `batch.*` calls align trailing dimensions
and allow size-one dimensions, including singleton vectors. Integer,
rational, float, and complex batch operators also broadcast across ranks,
but two vectors must have equal lengths. Comparisons produce masks for
selecting or updating matching elements.

Copies and slices keep their values across updates: after
`var saved = values[:]`, changing `values` leaves `saved` unchanged. To update
the original batch, assign through its index, slice, or mask.

With `vmap[f]`, you can apply library functions or your own scalar functions
across batch elements. Choose library functions from their number package,
such as `apn_mojo.float.sqrt`. For real and complex ball batches, map
functions from `apn_mojo.ball` or `apn_mojo.complex_ball`, or use the supported
`batch.*` functions directly. These batches have no arithmetic operators.

`lift[f]` turns a binary function into reductions, accumulations, outer
products, and a broadcasting call for the same six number types. The
[vectorization tutorial](docs/content/tutorial/vectorization.md) shows how
to reuse a custom function, map rows or columns, and choose a fold's rounding
behavior. `ExactComplex` is scalar-only; convert its elements to `Complex`
or `ComplexBall` before building a batch.

### Threads and native output

Large supported operations can use multiple CPU cores with the same
numerical results as sequential execution. Import `set_num_threads` and
`get_num_threads` from `apn_mojo`, or set `APN_MOJO_NUM_THREADS` before launch.
The count includes the caller; `1` keeps all work on that thread. Short work
may stay serial. See the [thread controls and example](docs/content/reference/batch.md#threads).

`values.to_native[DType.float64]()` exports a flat `List[Float64]` in
row-major order; keep `values.shape()` separately. Floating-point output
rounds once per element, while integer output requires exact whole numbers
in range. Extract complex components before conversion; Ball conversion
exports midpoints and discards bounds. The [native output example](docs/content/reference/batch.md#native-values)
also shows how to fill a fixed-size Mojo `Array`.

## Saving and restoring values

Save scalar values or Integer, Rational, Float, and Complex batches as
versioned JSON to restore their stored values and formats exactly. Large
numeric fields use strings, so readers in other languages can preserve every
digit. Ball and ComplexBall support scalar JSON only.

Pass `ConversionLimits` to supported parsing and formatting calls to bound
input size, digit counts, and allocation requests. Each conversion gets its
own budget; later arithmetic is unaffected. See the
[conversion guide](docs/content/tutorial/conversion.md) for text, JSON, and
the available limits.

## Documentation

The documentation is published at
[spellbound-mojo.github.io/apn_mojo](https://spellbound-mojo.github.io/apn_mojo/):
tutorials, guides, the architecture, an API reference generated from the
source, runnable examples and benchmark results. Its sources are in
[`docs/content`](docs/content/index.md); to browse them locally, start the
website:

```sh
pixi run --locked -e docs docs-serve
```

Pixi installs the documentation tools as needed. Open the address MkDocs
prints. The links below open Markdown sources on GitHub; the website also
includes the generated declarations, example code, and recorded output.

- [Tutorials](docs/content/tutorial/integers.md) walk through the number types
  and common operations.
- [Vectorization](docs/content/tutorial/vectorization.md) introduces `vmap`,
  `lift`, axes, broadcasting, and folds.
- [Interval bounds](docs/content/tutorial/balls.md) carry uncertainty through
  real and complex calculations.
- [Precision and accuracy](docs/content/guides/precision.md) helps you choose
  working precision and check reliable digits.
- [API reference](docs/content/reference/index.md) describes each type and
  function, including arguments and errors.
- [Example programs](docs/content/examples/index.md) cover arithmetic,
  conversions, batches, and reductions.
- [Architecture](docs/content/architecture/overview.md) explains how the
  arithmetic, storage, and execution work.
- [Support and limitations](docs/content/guides/status.md) describes supported
  platforms and library boundaries.

## Development

Run the functional tests on Linux, allowing time for the first compilation:

```sh
pixi run --locked test --timeout 1500
```

The runner builds one suite at a time under a 12 GB memory cap. It reuses
binaries when the sources and compiler are unchanged, but runs the tests
every time.
Run only the suites affected by your change. For ball arithmetic, use
`pixi run --locked test --suite ball --timeout 1500`.
Keep Mojo compilations sequential, including examples and test suites.

For documentation changes, run `pixi run --locked -e docs docs-check` on
either platform to build the site and check its links. After changing a
runnable example, also run `pixi run --locked docs-test` on Linux to compile
the examples under the same memory guards and check their output.

The [development guide](docs/content/contributing/development.md) explains
which checks arithmetic and memory-management changes need. For prose,
examples, and API descriptions, follow the
[documentation guide](docs/content/contributing/documentation.md).

## License

APN Mojo is distributed under the [Apache License 2.0](../LICENSE).
