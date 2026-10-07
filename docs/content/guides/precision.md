# Choosing precision and checking accuracy

Choose precision for the result you need, then check whether the calculation
supports that accuracy. Correct rounding of each operation does not guarantee
the same accuracy for a whole expression: input approximations and
cancellation can dominate the final error.

## Bits, digits, and working budgets

A `FloatFormat` sets binary significand bits. As a starting point, storing
$d$ significant decimal digits takes about $\lceil d\log_2(10)\rceil$ bits:
50 digits need about 167 bits. Using 192 bits leaves some extra precision,
but those extra bits alone do not prove that the final 50 digits are correct.

| Setting | What it controls |
|---|---|
| `FloatFormat(192)` | The precision of a stored Float and each result rounded to that format |
| `ArithmeticContext(max_precision=8192)` | The internal working-precision ceiling for functions that refine an enclosure to establish rounding |
| `BallContext(256)` | Midpoint precision for interval calculations; the radius still records uncertainty |
| `to_string(digits=50)` | The number of significant decimal digits displayed, without changing the stored value |

Pass a context to named functions when you need explicit settings. Increasing
`max_precision` lets a function do more work to resolve the *same* output
format; increasing `FloatFormat` changes the requested output. The default
working ceiling for correctly rounded Float functions is
`max(8 * p, p + 4096)` bits for a result precision `p`.

Keep exact inputs exact for as long as possible. Use `Rational("0.1")` or
decimal text at the chosen Float precision, rather than importing a native
float that has already lost digits. To retry at higher precision, rebuild
rounded inputs from their original text or exact values.

## Avoid losing a small result

Subtracting nearby large numbers can expose rounding error accumulated
earlier. In the example below, the separate addition loses the `1` before
the large terms cancel. `sum` retains the exact intermediate total and
rounds once, so it returns `1`.

<!-- example: docs/examples/precision.mojo -->

The same principle guides other choices: use `dot` for a sum of products,
`fma` for one product plus an addend, `log1p(x)` for `log(1 + x)`, and
`expm1(x)` for `exp(x) - 1` when `x` is small. These functions avoid an
intermediate rounding that could lose the quantity you want.

## Certify the requested result

The second half of the example recomputes an enclosure of `sqrt(2)` with
increasing working precision. It stops only after two checks succeed:

1. `to_float_if_certain` establishes a single correctly rounded 192-bit Float.
2. The outward endpoints print identically at 50 significant decimal digits.
   Every value between them therefore has that same rounded decimal text.

These checks answer different questions. A certified binary Float does not,
by itself, prove the last digit of a separately rounded decimal display.
The endpoint check establishes the displayed digits directly.

The retry loop has a 1,536-bit ceiling and reports failure if it cannot
certify the answer. For a larger expression, recompute every dependent
intermediate inside the loop. Comparing two higher-precision Float runs is
a useful diagnostic, but agreement alone is not a proof of accuracy.

## When more precision does not help

An interval may remain wide because the input is uncertain, because repeated
uses of an input lost their correlation, or because the function changes
rapidly near that input. Near zero, an absolute error bound is often more
useful than a count of relative digits. Inspect the radius and endpoints
instead of treating midpoint precision as a measure of accuracy.

A Float elementary or special function that cannot certify its rounding
within `max_precision` raises. Check the function, argument, and budget in
the message before raising the ceiling. Huge trigonometric arguments can
need many extra bits for argument reduction even when the output precision
is modest. A larger budget permits more work; it is not a time or memory cap.

Ball functions may instead return a wider enclosure or an indeterminate
result; the response depends on the function. For example, `sin` and `cos`
can fall back to `[-1, 1]`. Check domain and uncertainty before retrying.
See [calculations with bounds](../tutorial/balls.md) and the
[Float function contracts](../reference/float.md#elementary-functions).
