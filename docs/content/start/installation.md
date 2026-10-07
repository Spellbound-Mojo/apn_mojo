# Installation and first run

## Supported platforms

APN Mojo supports **Linux and macOS with Mojo 1.1.0**.
[Pixi](https://pixi.sh/) installs the pinned compiler and dependencies for
Linux x86-64 and macOS on Apple silicon. Check the
[Mojo system requirements](https://docs.modular.com/mojo/requirements/)
for operating-system and toolchain prerequisites.

## Work from a checkout

After installing Pixi, clone the repository and enter the project directory:

```sh
git clone https://github.com/Spellbound-Mojo/apn_mojo.git
cd apn_mojo
pixi install --locked
pixi run --locked mojo --version
pixi run --locked mojo run -I src --Werror docs/examples/first_integer.mojo
```

Run the documentation's commands from the checkout root, which contains
`pixi.toml`. `--locked` keeps dependency versions
fixed to `pixi.lock`, and `-I src` tells Mojo where to find the library. The
last command runs this program:

<!-- example: docs/examples/first_integer.mojo -->

## Run your own program

Import the types you need, such as `Integer` or `Float`, from `apn_mojo`.
Arithmetic and conversions can raise errors, so declare a `def main() raises`
entry point when you want those errors to propagate:

```sh
pixi run --locked mojo run -I src --Werror path/to/your_program.mojo
```

Mojo compiles the library along with your program, so you can run it without
a separate library build or Python runtime. If you invoke Mojo from another
directory, pass an absolute path to this checkout's `src` directory.

## Tests and examples

Run the Integer tour on either platform:

```sh
pixi run --locked example                # Integer tour
```

The test and example-verification runners use Linux-specific compiler
memory guards. On Linux, run:

```sh
pixi run --locked test --timeout 1500    # Functional suites, compiled one at a time
pixi run --locked docs-test              # Compile examples and check their output
```

Run only one Mojo compilation at a time. The functional runner caps compiler
memory at 12 GB and caches each suite's binary. Use `--suite ball`, for
example, to run just the ball suite. AddressSanitizer and numerical comparisons
need additional time or dependencies; see the
[development guide](../contributing/development.md) and
[benchmark guide](../contributing/benchmarks.md).

To build or serve the documentation on either platform, use the separate
Pixi `docs` environment:

```sh
pixi run --locked -e docs docs-check
pixi run --locked -e docs docs-serve
```

Continue with the [Integer tutorial](../tutorial/integers.md), or compare the
[number families](../index.md#number-families).
