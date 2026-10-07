# Complex

A `Complex` stores two [`Float`](float.md) components and rounds each result
once per component. Exact expressions or certified bounds let multiplication,
division, roots, and powers preserve this guarantee without rounding their
intermediates prematurely. Elementary functions use the methods in
[Correct rounding](certified.md). Public calls are in the
[Complex reference](../reference/complex.md).

## Components and formats

The real and imaginary parts have independent precisions and shared exponent
bounds. Format selection follows the `Float` rules:

| Operand Combination | Resulting Component Precisions |
|---|---|
| **Complex with exact Integer/Rational** | Preserves existing Complex component precisions. |
| **Complex with native or library Float** | Sets component precisions to $\max(\text{precision}_{\text{real}}, \text{precision}_{\text{component}})$. |
| **Complex with another Complex** | Computes componentwise maxima: $\max(\text{real}_1, \text{real}_2)$ and $\max(\text{imag}_1, \text{imag}_2)$. |

Operand exponent bounds must match unless overridden by an explicit context. A
`ComplexContext` pairs two separate `ArithmeticContext` instances (one per
component), while an `ArithmeticContext` applies uniform settings across both
components. Real-valued functions (such as `abs`) default to the larger
component's precision.

Real operands keep their real meaning: adding a real `+0` to $(1, -0)$
preserves the imaginary `-0`, whereas adding complex $(+0, +0)$ may change
the zero signs. Mixed real operations need no intermediate complex values.

## One final rounding

Complex multiplication rounds $(ac - bd)$ and $(ad + bc)$ once per component,
never rounding intermediate partial products. Complex division rounds
$(ac + bd) / (c^2 + d^2)$ and $(bc - ad) / (c^2 + d^2)$ once per component
without preliminary denominator rounding.

- **Short components**: When all four parts are finite, nonzero, and have significands
  fitting within 63 bits, components evaluate as two-term sums of native 128-bit
  products, aligned and rounded using 256-bit native integers.
- **Binary components**: Finite binary parts of arbitrary precision evaluate
  using exact `Integer` products and the binary `Float` sum kernel, requiring only
  the buffers needed for exact products and final component values.
- **General case**: Two-term expressions are represented as signed integers with
  separate binary exponents and rational denominators. Adjacent terms accumulate
  into an `Integer` and round as a ratio; distant terms route to the
  [exact accumulator](reductions.md#exact-accumulation), ensuring large exponent
  gaps never instantiate dense limb arrays.

Complex division with short parts, significands within 63 bits, forms both
numerators and the denominator exactly in 256-bit native integers when each
pair of terms lies within 127 bits, and rounds each component with the Float
kernel's native quotient (see [Float](float.md)); a zero numerator takes the
signed zero of the general path. Otherwise, complex division estimates quotients
using $(p + 4)$-bit prefixes of numerator
and denominator, obtaining an initial floor estimate within one unit. Exact
cross-multiplied comparisons against original operands certify interval brackets
and midpoints. Underflow comparisons check against half the minimum magnitude,
and overflow is classified before output precision buffers are allocated.

## Functions

- **Magnitude (`abs`)**: Evaluates $|z| = \sqrt{x^2 + y^2}$ without
  forming an unrounded sum of squares. The algorithm retains a sticky $(2p + 4)$-bit
  prefix of the exact squared magnitude and computes its `Float` square root. If
  the top exponent of $x^2 + y^2$ is $T$, the root's exponent is $E = \lceil T/2 \rceil$.
  The squares of the root's grid points and midpoints are multiples of
  $2^{2E - 2p - 2}$ (at least four prefix units), so this prefix preserves
  all rounding decisions.
- **Square root**: Both components come from one estimate: the larger,
  $\sqrt{(\sqrt{a^2 + b^2} + |a|) / 2}$, once, and the smaller as
  $|b| / (2 \cdot \text{larger})$, preventing catastrophic cancellation along the
  branch cut.
  - The estimate runs in fixed point with $f = w + 2$ fractional bits, for
    $w = p + 32$ and $p$ the wider component's precision, after scaling by an even
    power $2^{-k}$ that puts $\max(|a|, |b|)$ in $[1/2, 2)$: the truncated root $H$
    of $\lfloor (a^2 + b^2) 4^f \rfloor$, plus $\lfloor |a| 2^f \rfloor$, then the
    truncated root $M_f$ of that sum times $2^{f - 1}$. With $h = \sqrt{a^2 + b^2}$
    and $M = \sqrt{(h + |a|) / 2}$: $h 2^f - 2 < H \le h 2^f$, the sum is low by less
    than 3 units, and $M 2^f (1 - 3 / ((h + |a|) 2^f)) - 1 < M_f \le M 2^f$. Since
    $h + |a| \ge 1/2$ and $M \ge 1/2$, $M_f$ is low by less than
    $8 \cdot 2^{-f} = 2^{1 - w}$, relatively, and has at least $w + 1$ bits. The
    smaller component, a floor quotient of at least $w + 2$ bits, is within
    $2^{2 - w}$.
  - Certification reads the estimate's bits. The root lies within $2^s$ units of
    the estimate $m$'s last bit, for $s = \text{bits}(m) - w + 4$, a margin over
    those bounds. Rounding to $p$ bits changes at boundaries every $2^g$ units:
    $g = t - 1$ for nearest (midpoints and representable values) and $g = t$
    otherwise, for $t = \text{bits}(m) - p$. With $T$ the low $t$ bits of $m$,
    $(T + 2^s) \bmod 2^g \ge 2^{s + 2}$ puts every boundary more than $2^s$ units
    away, so the root rounds as $m$ does, inexact in its direction. The test reads
    the whole tail, however long: near-ties need all of it.
  - Up to $w = 120$ (parts of up to 88 bits) with significands of at most 128
    bits, the estimate and its certification run in 128- and 256-bit native
    integers, the same arithmetic as above: the scaled terms are below
    $2^{f + 1}$, the sum of squares below $2^{2f + 3}$, both roots below
    $2^{f + 2}$ and the minor root's shifted dividend below $2^{2f + 2}$, within
    256 bits for $f \le 122$.
  - A component within the error bound of a midpoint gets one retry at
    $2p + 32$ bits, which settles ties down to about $2^{-p - 28}$ ulp; the
    benchmark corpus's, about $2^{-p}$ ulp from a midpoint, need about $2p$ bits.
    One within the bound of a representable value may be exact, and goes straight
    to the exact certification, as does anything near the exponent range's ends.
  - The exact certification: for a dyadic trial value $t$, the signs of
    $2t^2 - a$ and of the exact sparse form $b^2 + 4at^2 - 4t^4$ decide whether the
    real component $x = \sqrt{(\sqrt{a^2 + b^2} + a) / 2}$ exceeds $t$, certifying
    exponents and midpoints without evaluating nested irrationals.
- **Integral powers**: Evaluates $z^n$ using binary exponentiation by squaring
  over signed dyadic intervals. Endpoint coefficients remain bounded by working
  precision while binary scales use arbitrary-precision integers. Results are
  certified only when both interval endpoints round to identical values, signs,
  and condition flags; otherwise, working precision doubles. Diagonal and
  on-axis inputs use exact phase cycles modulo 4 or 8.

## Special values

Special values follow GNU MPC 1.4.1, with the library tracking condition
flags. Quiet `NaN` values propagate without setting the invalid condition.

| Operation | Special Value Rule |
|---|---|
| **Complex `+`, `-`** | Componentwise `Float` addition and subtraction. |
| **Mixed real `+`, `-`** | Modifies the real component; copies or negates the imaginary component. |
| **Real $\times$ Complex, Complex / real** | Evaluates independent real operations across both components. |
| **Finite Complex `*`** | Exact two-product sums; zero results evaluate to `-0` when rounding downward or when both product terms are negative. |
| **`*` with NaN (no infinity)** | Evaluates both components to `NaN`. |
| **`*` with infinity** | Projects to signed infinite directions or `NaN`, recovering infinite results that would otherwise collapse to double `NaN`. |
| **Division by complex zero** | Multiplies numerator components by infinity using the denominator's real-zero sign; finite nonzero numerators signal divide-by-zero, while zero numerators signal invalid. |
| **Finite / infinite** | Evaluates to signed zeros without condition flags. |
| **Infinite / infinite** | Evaluates both components to `NaN` with an invalid signal. |

Square root checks special values in MPC's precedence order:

| `sqrt(a, b)` Condition (in precedence order) | Result |
|---|---|
| `b` is infinite | $(+\infty, \text{sign}(b) \times \infty)$ |
| `a = +inf`, `b` finite | $(+\infty, \text{sign}(b) \times 0)$ |
| `a = +inf`, `b` NaN | $(+\infty, \text{NaN})$ |
| `a = -inf`, `b` finite | $(+0, \text{sign}(b) \times \infty)$ |
| `a = -inf`, `b` NaN | $(\text{NaN}, +\infty)$ |
| Other input containing NaN | $(\text{NaN}, \text{NaN})$ |
| `b` signed zero, `a >= 0` | $(\sqrt{a}, \text{sign}(b) \times 0)$ |
| `b` signed zero, `a < 0` | $(+0, \text{sign}(b) \times \sqrt{-a})$ |
| General finite case | Positive real part; imaginary sign matches $b$ |

For example, $\sqrt{-4 + 0j} = 0 + 2j$, while $\sqrt{-4 - 0j} = 0 - 2j$.

## Batches, text and JSON

Complex batches use contiguous list storage and retain each element's formats,
plus defaults for empty containers. The shared mapping executor runs
arithmetic, functions, and updates. It checks shapes and formats in logical
order before computing the real and imaginary components.
Reductions like `sum`, `dot`, and `vdot` round once per component (see
[Reductions](reductions.md#round-once-sums)).

The text parser locates real and imaginary spans without allocating substrings,
then passes each to the `Float` parser. JSON stores two complete `Float`
records, with one `ConversionLimits` budget for the whole conversion.
