# apn_mojo.complex_ball

`ComplexBall` pairs two real [balls](ball.md) to describe a rectangle in the
complex plane. Each operation encloses its result for every point in the
input rectangles. The
[Ball arithmetic](../architecture/balls.md#complex-balls) chapter explains the
design.

The [interval tutorial](../tutorial/balls.md#rectangles-in-the-complex-plane)
shows construction, magnitude bounds, branch cuts, and certified conversion
to Complex in a runnable example.

<!-- api: complex_ball -->

## Values and arithmetic

`ComplexBall(real, imag)` takes two balls or exact values, and the imaginary
part defaults to 0. `ComplexBall(z)` of a finite `Complex` is exact; with
`precision=p` each part is rounded to `p` bits, with a radius covering the
error. `ComplexBall` of an `ExactComplex` is exact when both parts are binary
fractions and otherwise encloses each part the same way. There is no implicit
conversion into `ComplexBall`.

`+`, `-`, `*` and `/` follow the real ball formulas part by part; `add`,
`subtract`, `multiply` and `divide` take a `BallContext` for the result
precision. Dividing by a rectangle that contains 0 gives an indeterminate
ball. `conjugate` negates the imaginary part exactly, and `pow_int` takes an
Integer exponent of either sign. `abs` returns a real ball containing the
modulus of every point, and `angle` one containing the argument: `[0 +/- pi]`
for a rectangle that meets the negative real axis, where the argument jumps,
and indeterminate for one that contains 0.

## Elementary functions

`exp`, `log`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`,
`asin`, `acos`, `atan`, `asinh`, `acosh` and `atanh` return a rectangle that
contains the function's value at every point of the input. The `context`
sets the result precision, by default the input's.

A rectangle on a branch cut takes the counter-clockwise continuous value
there, as the [Complex](complex.md#elementary-functions) functions do: `log`
of an exact `-2` has imaginary part `pi`, and `sqrt(-2)` is `i sqrt 2`. A
rectangle that crosses a cut covers the values on both sides, since the
function jumps there: `log` of a rectangle around `-2` has an imaginary part
from `-pi` to `pi`. A rectangle that contains a singularity gives an
indeterminate ball: 0 for `log`, `1` and `-1` for `atanh`, `i` and `-i` for
`atan`, and a pole for `tan`.

## Sets and conversion

`contains`, `contains_zero` and `overlaps` compare the exact endpoints of both
parts. `union` returns the smallest rectangle containing both inputs.
`intersection` returns their common rectangle, or None if they are disjoint.
`to_complex_if_certain(ball, context=c)` returns the `Complex` that every
point of the rectangle rounds to, part by part, or None when the parts' ends
round differently.

## Text, hashing and batches

The text form is `ComplexBall(real, imag)`, with each part in the notation of
[apn_mojo.ball](ball.md). `same_representation` compares stored midpoints and
radii, and `stable_hash` gives the APNH-64 hash, tag 7, which never changes
between releases. `Batch[ComplexBall]` holds complex balls, and `vmap` maps
the functions over it. `lift` adds broadcasting, outer products, and ordered
folds to binary functions, forwarding `BallContext` to each call. Scalar
JSON uses a version-1 record containing two complete Ball records, `real`
and `imag`. Saving and restoring it preserves the representation exactly.

ComplexBall batches also support NumPy-style elementwise functions and
`cumsum` and `cumprod`. Use `lift` for other folds; arithmetic operators and
batch JSON are not available.
