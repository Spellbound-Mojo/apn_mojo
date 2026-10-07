# Correct rounding

Float transcendental functions and constants make the same promise as `+`
and `sqrt`: the exact value rounded once, in every rounding mode. Their Ball
counterparts enclose the function's value at every point of the input. Both
use the same point kernels; one rounding driver turns their certified bounds
into Float results. The algorithms and proofs below explain why this works.

## Three layers

Every function `f` is built in three layers:

1. **Point kernel.** `_kernel_f(x, w)` returns a ball containing `f(x)` at an
   exact argument `x`, with at least `w - c_f` bits of relative accuracy
   whenever `f(x) != 0`. Every kernel here has `c_f <= 1`; zeros of `f` are
   exact cases and never reach a kernel.
2. **Ball function.** `apn_mojo.ball.f` evaluates the kernel at the ends of its
   input ball (the endpoint method, for monotone functions) or at its midpoint,
   widened by a bound of the derivative (`sin` and `cos`).
3. **Correctly rounded function.** `apn_mojo.float.f` runs the Ziv driver over
   the kernel at the exact argument, never over the ball function, which would
   add the radius growth of a whole ball.

A few kernels compose others through ball arithmetic: `pow` is
`exp(y log x)`, `tan` divides the balls of `sin` and `cos`, and the inverse
hyperbolic functions take logarithms of exact expressions. Ball arithmetic is
rigorous, so the compositions are too.

## Fixed-point kernels

Kernels compute in fixed point. `_Fix` represents the interval
`(mid +/- err) * 2**-scale`, with an Integer midpoint and an error counted in
units of the last place. The error uses Ball's 30-bit radius type. Each
operation rounds its midpoint once and adds the rounding error to the
propagated input errors:

| Operation | Error bound in units |
|---|---|
| `x + y`, `x - y` | `e_x + e_y` |
| `x * y` | `(abs(X) e_y + abs(Y) e_x + e_x e_y) / 2**scale + 2`: the product keeps only the 64-bit digit diagonals from two below the scale up (`Integer._high_product`), at most one unit below the floor |
| `x / y` | `(abs(X) e_y + abs(Y) e_x) 2**scale / (abs(Y) (abs(Y) - e_y)) + 1`, infinite unless `abs(Y) > e_y` |
| `sqrt(x)` | `e 2**scale / sqrt(X) + 1` in scaled integers, at most twice the bound `e 2**scale / (sqrt(X - e) + sqrt(X))` and one integer square root cheaper; infinite unless `X > e` |
| `x * k`, `x / k` for an integer `k` | `e abs(k)`, `e / abs(k) + 1`, the quotient truncated in the midpoint's own words |

These bounds account for arithmetic error, leaving the kernel to bound the
tails of truncated series. Reducing arguments to magnitudes near 1 keeps
midpoints at about `scale` bits. For a small result, such as `sin` near zero
or `log` near 1, the kernel adjusts the scale by the result's binary exponent
to preserve relative accuracy.

The series cores are written once, over the `_FixedPoint` trait, and run on
one of two representations with the same bounds. `_Fix` has an Integer
midpoint, at any scale. `_Wide[n]` keeps the midpoint in `n` native words, two's
complement, and the error as a count of units in one word. The count saturates
at `2**64 - 1`, which stands for an unbounded error, so no bound is lost. A core
runs on `_Wide[4]` while its working scale stays below 256 bits, on `_Wide[8]`
below 512, and on `_Fix` above that or when its argument's error reaches
`2**32` units. On `_Wide`, a product is a word-by-word multiplication with no
allocation, and its error bound takes a few word operations. The few divisions
and square roots of a core go through `_Fix`.

Each core sums its series with `_sum_series`. It bounds the terms `c_k z**k`
from a bound of `abs(z)` in radius arithmetic, and takes terms up to the first
whose bound is below half a unit. Twice that bound covers the tail, since each
later term is at most half the one before; the function checks this
condition. The sum runs by rectangular splitting (Smith). It forms the powers
`z**i` for `i <= m`, with `m = ceil(sqrt(N))`, then applies Horner's rule in
`z**m` over blocks of `m` terms. `N` terms cost about `2 sqrt(N)` full
products, and each term one division by a small integer and one sum. Since
terms are cheap, the cores reduce their arguments less than a term-by-term sum
would. At 1024 bits (scale 1105), `exp` squares 22 times instead of 33, `sin`
and `cos` double 8 times instead of 16, and `atan` and `log` take 6 halvings or
square roots instead of 16 and 11.

| Function | Reduction | Core |
|---|---|---|
| `exp`, `exp2` | `x = k ln 2 + r` (`exp2`: `x = k + f`, `r = f ln 2`); below `abs(x) = 2**40`, `k` from x's top 64 bits over ln 2's, within `2**-20` of `x / ln 2` | Taylor series of `exp(r / 2**s)`, `s = 2 isqrt(scale) / 3`, then `s` squarings |
| `expm1` | below `abs(x) = 1/2`: none | `y (1 + y/2! + y**2/3! + ...)` for `y = x / 2**s`, `s <= isqrt(scale) / 3`, then `expm1(2a) = expm1(a)(expm1(a) + 2)`, which keeps relative accuracy |
| `log` | `x = 2**k m`, `m` in `[1/sqrt 2, sqrt 2)` | `r <= isqrt(scale) / 5` square roots of `m`, then `log m_r = 2 atanh((m_r - 1)/(m_r + 1))` |
| `sin`, `cos` | `x = k pi/2 + r`, quadrant `k mod 4`, with more bits of pi while `r` lacks accuracy | series of `sin(r / 2**s)`, `s <= isqrt(scale) / 4 + E(r) + 2`, `cos = sqrt(1 - sin**2)`, then `sin 2a = 2 sin a cos a`, `cos 2a = 1 - 2 sin**2 a` |
| `atan` | `atan x = pi/2 - atan(1/x)` above 1 | `s <= isqrt(scale) / 5 + E(y) + 2` halvings `atan y = 2 atan(y / (1 + sqrt(1 + y**2)))`, then the alternating series |

The general kernels support every precision, including the `_Fix` arguments
used by `asin`, `acos`, `atan2`, `log1p`, and inverse hyperbolic functions.
For Float inputs below 4608 bits, `exp`, `expm1`, `log`, `atan`, `sin`, and
`cos` first try the medium-precision kernels.

### Medium-precision kernels

`ball/_medium.mojo` implements the method of F. Johansson, "Efficient
implementation of elementary functions in the medium-precision range"
(ARITH 22, 2015), up to 4608 bits:

1. **Fixed point on limbs.** Values are fractions of `wn` whole 64-bit limbs in
   one stack block, at the target precision plus 4 to 8 bits (plus the
   argument's `-E(x)` where a small result needs relative accuracy). The
   limb operations are Integer's word-span kernels (`integer/_limbs.mojo`).
2. **Reduction against stored constants.** `exp` reduces modulo ln 2 and
   `sin`/`cos` modulo pi/4 by one long division of the argument's limbs by a
   4608-bit table of the constant, within 2 or 3 units; `log` reads `2m - 1`
   for `x = m 2**e`; `atan` takes `1/x` above 1.
3. **Table steps.** Up to 512 bits one table step on the top 8 bits (7 for
   `log`), up to 4608 bits two steps on 5 bits each. `exp`, `sin` and `cos`
   subtract the index bits and multiply by, or combine with, `exp(i/2**b)/2`,
   `sin(i/2**b)` and `cos(i/2**b)`; `atan` takes
   `atan w = atan(p/q) + atan((q w - p)/(q + p w))`, one division; `log`
   divides by the word `p + q`, then fuses the second step with
   `log(1 + v) = 2 atanh(v/(2 + v))`, one division. The remainder is below
   `2**-8` or `2**-10`.
4. **Taylor sums.** By rectangular splitting (Paterson and Stockmeyer, 1973;
   Smith, 1989) with the coefficients `1/k!` and `1/(2k+1)` as integers over
   denominators shared by groups of terms: a term is one word multiply-add, a
   word division falls between groups, and the sum is within 2 units (3 for
   the two-term `atan` sum). One evaluator, `_series_sum`, takes the
   exponential's, the sine's and cosine's (over powers of `x**2`, alternating)
   and the arctangent's sums. Long `exp` and `sin` sums give way
   to `cosh = sqrt(1 + sinh**2)` and `cos = sqrt(1 - sin**2)`.
5. **One error count.** The error is a count of units in the last limb,
   raised by the bound each step adds, derived at that step: 3 times the
   reduction's for `exp` (since `exp' < 3`), 4 or 6 for a table product, 1 per
   table value added, `2 e_left + 2 e_right + 3` for the `sin`/`cos` addition
   formulas, plus the series' truncation. The midpoint is rounded once to the
   precision, and the count, scaled, is the radius.

The midpoint is rounded once, to nearest at the target precision, straight
from the limbs: the top bits, then the half
bit and the sticky bits below; the rounding goes into the radius. `expm1`
subtracts the one on the limbs before that rounding. Products of up to 16
limbs take rows of word multiply-adds (`_multiply_limbs`), unrolled for equal
sizes up to 4 limbs, since the word kernels' dispatch costs more than such a
product.

A ball function of a narrow ball takes the same kernels at the ball's own
precision (`_medium_ball`): the
kernel at the midpoint, rounded once, widened by the function's derivative
over the radius `r`, in radius arithmetic: `e**m r (1 + r)` for `exp` and
`expm1`, `(r/m)(1 + 2**-16)` for `log`, `r/(1 + d**2)` with
`d = |m|(1 - 2**-17)` for `atan`, and, inside the kernel, by Taylor's
theorem, `min(r, A r + r**2/2)` for `sin` and `cos`, with `A`
above the other function's absolute value from that value's top limb and
the error count. `sin` and `cos` round only the result asked for; `tan`
takes both at 4 more bits and divides the two balls, which is indeterminate
when the cosine ball reaches 0. A ball is narrow when its radius
is below `2**-17` of its midpoint; wider balls, exact points such as 0, and
what a kernel declines take the endpoint and midpoint methods above.

The general kernels handle tiny and huge arguments, `|x| = 1` in `atan`,
and `x` within `2**-(prec/2)` of 1 in `log`. They also take over when a sine
or cosine of an `x` above 1 falls below `2**-10`. In that case, reduction
preserves absolute accuracy, but correct rounding needs relative accuracy.

### Tables

`scripts/generate_function_tables.py` writes `ball/_tables.mojo`: the
function tables (`exp(i/2**b)/2`, `log(1 + i/2**b)`, `atan(i/2**b)`, `sin` and
`cos(i/2**b)`, at 512 and 4608 bits), ln 2, pi/4 and pi/2 - 1 at 4608 bits,
and the coefficient tables. Each entry is `floor(f 2**(64 L))`, kept only
when every point of an interval enclosure (mpmath's interval arithmetic,
a development tool, with outward rounding) has the same floor. mpmath has no
interval atan, so the script halves the argument with
`atan x = 2 atan(x / (1 + sqrt(1 + x**2)))` to below `2**-16` and sums the
alternating series, whose next term bounds the rest. Generated this way, the
tables matched those of an earlier generator, which used another
interval-arithmetic library, byte for byte. The tables are named by width and
level: `_SHORT` up to 512 bits, `_LONG_COARSE` and `_LONG_FINE` the two
lookups up to 4608 bits. `test_medium_tables` checks every entry against the
general series kernels, and `test_constant_tables` ln 2 and pi against binary
splitting. An entry is hex digits, most significant limb first, in one string
literal per table: static data, read in place, 16 digits to a limb. String
literals encode bytes above `0x7f` as UTF-8, so raw bytes would not survive; a
comptime SIMD table, the other option, was rebuilt on every read (about 1.1 µs
for a 37 KB table) and took 1.4 GB to compile. The constants ln 2, pi and Euler's
gamma come from the same tables for every precision up to 4608 bits. Brent
and McMillan's series for gamma had been 22% of `hyp2f1`'s digamma path at
53 bits.

`scripts/generate_bernoulli_table.py` writes `ball/_bernoulli_table.mojo`:
the tangent numbers `T_1 ... T_256`, from which Stirling's series takes
`B_2k = (-1)**(k-1) 2k T_k / (4**k (4**k - 1))`. Its coefficients are then
integers: `B_2k / (2k (2k-1))` is `(-1)**(k-1) T_k / (4**k (4**k - 1) (2k-1))`.
The script computes the numbers by Brent and Harvey's recurrence ("Fast
computation of Bernoulli, tangent and secant numbers", 2011), the one the
library runs past the table's 1,920 bits of working precision. It checks
them against the exact Bernoulli recurrence for k <= 64, and against the
rounding of `2 (2k)! zeta(2k) / (2 pi)**2k` for every k. Recomputing the
numbers on every call had been a third of `gammainc`'s time at 53 bits.

For a ball without a pole, Gamma, log |Gamma| and digamma take the
midpoint's value and widen it over the radius r. With `M = max |psi|` over
the ball, psi being log |Gamma|'s derivative,
`|log |Gamma(t)| - log |Gamma(m)|| <= r M` and
`|Gamma(t) - Gamma(m)| <= r M |Gamma(m)| e**(r M)`, where `e**(r M) < 65/64`.
Above 0, `0 < psi'(t) < 1/t + 1/t**2` bounds digamma's change, so
`M <= |psi(m)| + r (1/low + 1/low**2)`. Here `low <= m - r` comes from radius
arithmetic, so a ball above 0 needs neither of its exact ends, and the
bounds are radius operations. Gamma and psi at m come from one pass of the
Taylor method below. Below 0, psi increases between the poles, so M is at an
end, both computed at 32 bits. One evaluation replaces the two ends, or five
evaluations below 0, while `r M < 2**-8`; a wider ball keeps the ends, which
are tighter.

Measured by the report against its HEAD baseline (pinned, case by case,
drift 0.9%), the time over the old one at 53, 256 and 1024 bits was:
- **Float gamma, gammaln, digamma:** 0.23, 0.38, 0.26 at 53 bits; 0.37, 0.41,
  0.49 at 256; 0.28, 0.30, 0.42 at 1024;
- **Ball versions:** 0.29, 0.33, 0.15 at 53 bits; 0.25, 0.24, 0.28 at 256;
  0.16, 0.18, 0.21 at 1024.

The Float gamma went from 28 to 6.6 times MPFR's time at 53 bits, and the
ball gamma from 271 to 58 times Arb's.

`scripts/generate_rgamma_table.py` writes `ball/_rgamma_table.mojo`: the
Taylor coefficients of `R(u) = 1/Gamma(1 + u) = sum_n b_n u**n`, `b_0 ...
b_273`, each `|b_n| 2**1280` rounded toward zero (85 KB of hex). It computes
them by the recurrence `(n-1) a_n = gamma a_{n-1} - zeta(2) a_{n-2} + ... +
(-1)**n zeta(n-1) a_1` (Wrench 1968), `b_n = a_{n+1}`, at two working
precisions far above the recurrence's cancellation, and the entries must
agree between them. It also checks the truncated series against `rgamma` at
sample points. Johansson ("Arbitrary-precision computation of the gamma
function", 2021, section 5.1) gives the method; its Theorem 5.3 bounds the
truncation: for complex `|z| <= 20` and `N <= 1000`,
`|R(z) - sum_{n<N} b_n z**n| <= 8 max(1/2, |z|) |b_N| |z|**N` when that is
below `2**-8`.

`ball/_rgamma.mojo` evaluates R and `R'(u) = sum n b_n u**(n-1)` by Horner's
rule in fixed point, at `F = 64 wn >= w + 24` fraction bits, for
`x = 1 + u + m`, `|u| <= 1/2`. Then:
- Gamma is `(1+u) ... (m+u) / R(u)`;
- log Gamma is its logarithm, or that of `Gamma(1 + x)/x` below 1/2;
- digamma is `-R'(u)/R(u) + sum_{k<m} 1/(1+u+k)`.

The product and the sum are exact in integers for a short dyadic x. The
bounds:
- **Coefficients and products.** Each coefficient is the entry's top `wn`
  limbs, below the true one by less than a unit (n units for `n b_n`). Each
  product is truncated, below the true one by less than a unit, and later
  steps scale both by `|u| <= 1/2`. So R is within `sum_n 2 |u|**n <= 4`
  units and R' within `sum_n (n+1) 2**(1-n) <= 6`.
- **A u with bits below the fixed point** is truncated, which moves R by at
  most 3 units and R' by at most 6. On `t = 1 + u` in [1/2, 3/2],
  `R' = -psi(t)/Gamma(t)` and `R'' = (psi(t)**2 - psi'(t))/Gamma(t)`. psi
  rises from `psi(1/2) > -1.97` to `psi(3/2) < 1`, psi' falls from
  `pi**2/2 < 4.94` to above 0, and Gamma stays above 0.88.
- **R's tail** at `|u| <= 1/2` is at most `4 |b_N| |u|**N`, by Theorem 5.3.
- **R''s tail.** Theorem 5.3 bounds the series' absolute terms, so it holds
  on the circle of radius 1/8 about u, where `|z| <= rho = |u| + 1/8`.
  Cauchy's estimate then bounds the tail by
  `8 (5/8) |b_N| rho**N / (1/8) < 2**6 |b_N| rho**N`.

N is the first index where both bounds fall below `2**-(F+2)`, with
`|b_N| < 2**(bits_N - 1280)` from the entries' lengths. The table serves
working precisions up to 1256 bits, which a 1024-bit Float reaches with its
guard bits, a reflection and a Ziv retry. Stirling's series serves larger x
and higher precisions.

A u with zero low limbs, such as the 1/2 of a half-integer x, skips them in
each product, so its step is one row of word products.

**Shorter steps.** A step whose result later steps scale by `|u|**n` (for R)
or `|u|**(n-1)` (for R') ignores the limbs of the partial sum and of u below
`2**(-n log2 |u| - 66)` units, and cuts those of its result. That is less
than `2**-64` units a step and under one in all, so R is within 5 units and
R' within 7. Products also skip a partial sum's leading zero limbs: late in
Horner's rule the sum is near its coefficient `|b_n|`. The generator writes
each entry's bit length beside the table, so the term count reads one value
a term.

**One pass for Gamma and psi.** With `x = 1 + u + m`, the product
`P = (1+u) ... (m+u)` and its derivative `P' = P sum_k 1/(k+u)` come from one
fixed-point pass (`_shift_factors`). Then `Gamma = P / R` and
`psi = P'/P - R'/R`, and the exact Integer product and reciprocal sum serve
only past 8 limbs or for a u inexact at the fixed point. The bounds:
- **The factors** `f_k = (k + u) 2**F` are exact integers.
- **P** is an integer of `n = wn + 2` limbs times `2**e`. Each step multiplies
  by `f_k` and keeps the top n limbs, cutting the rest. After a cut,
  `P >= 2**(64 (n-1))`, so each cut loses less than `eps = 2**-(64 (n-1))` of
  P. The true P lies within a factor `(1 + eps)**m` above the kept one, and a
  radius of `2 m 2**(64 + e)` covers it.
- **P'** steps as `P' f_k + P 2**F` at the same scale and cut. Every term is
  positive, and `P'/P = sum 1/(j+u)` stays between 2/3 and 16 for
  `m <= 4096`, so P' loses less than `1.5 eps` of itself a step and fits
  `n + 1` limbs. A radius of `4 m 2**(128 + e)` covers it.

**Balls take few guard bits.** A ball's radius carries the kernel's error,
so these passes run at `F = 64 wn >= w + 8` for a ball, not a Float's
`w + 24`, which keeps a Ziv step's error small. The joint pass runs 8 bits past the
ball's precision.

**One word.** Where `w <= 62` (`F = 64 >= w + 2`), `_gamma_word` runs the
pass in native 128-bit integers. It serves narrow balls and exact non-integer
points; integers keep their exact values:
- **x splits exactly** from its significand: `x 2**64` is an integer below
  `2**84` for `x < 2**20`, so n is it rounded and `u = x - n`.
- **R and R'** take Horner's rule as magnitudes and signs. The partial sums
  stay below 2 (`|b_1| < 0.58`, `|b_2| < 0.66`, and
  `sum_{j>=3} |b_j| 2**(3-j) < 0.25`), so `s u < 2**128`. The error bounds are
  those of the limb version.
- **P and P'** are 124-bit mantissas at a shared exponent, cut each step,
  within a factor `(1 + 2**-122)**m` above. P is normalized to 124 bits
  before the division, exactly.
- **Gamma** is P's top 64 bits over R, one 128-by-65-bit division. Its
  relative error adds P's, R's (`(4 + tail)/R`), the cut of P (below
  `2**-63`) and the quotient's cut (below `2**-62`), each bounded from above.
- **psi** only bounds the radius for Gamma and log Gamma, so doubles serve,
  with margins far above their roundings. For digamma itself, psi is
  `floor(P' 2**64 / P) - floor(R' 2**64 / R)` in units of `2**-64`, within
  those cuts and the relative errors of P', P, R' and R.

The result is rounded once into the result's format, and only log Gamma takes
a ball operation after it. Past one word, Gamma and log Gamma take the
value-only pass. psi's bound there comes from the one-word pass at the
midpoint rounded to 53 bits, its slope term widened by the rounding's
distance. Digamma takes the derivative in full.

Ball containment held for 1,397 random balls at 20 to 200 bits and for 17,415
sample points (each ball's ends and midpoint) aimed at the one-word path,
with x near 1/2, at and near integers, at `u = +-1/2` and at the top of its
range, against Floats at twice the precision. Ball gamma, gammaln and digamma
took, pinned, in µs:

| | 53 bits: before → after | Arb | 256 bits: before → after | Arb |
|---|---|---|---|---|
| gamma | 8.8 → 0.85 | 0.57 | 11.0 → 5.8 | 3.14 |
| gammaln | 8.5 → 1.3 | 1.00 | 11.2 → 6.6 | 5.2 |
| digamma | 6.0 → 1.1 | 1.63 | 9.3 → 8.2 | 13.1 |

At 53 bits the radii were 0.03 to 0.75 of Arb's on five inputs.

Against MPFR, 468 values of Gamma, log Gamma and digamma at 24 to 1240 bits
were all correctly rounded, besides the suites. With the narrow-ball rule
above, the report against its dad8583 baseline (pinned, case by case, drift
3.5%) gave, as the time over the Stirling version at 53, 256 and 1024 bits:
- **Float gamma, gammaln, digamma:** 0.19, 0.53, 0.59 at 53 bits; 0.075,
  0.39, 0.37 at 256; 0.11, 0.24, 0.20 at 1024;
- **Ball versions:** 0.24, 0.53, 0.44 at 53 bits; 0.20, 0.47, 0.13 at 256;
  0.12, 0.078, 0.20 at 1024.

The Float gamma took 1.26 times MPFR's time at 53 bits and 0.40 and 0.39 at
256 and 1024. The ball gamma took 15.6, 7.8 and 2.8 times Arb's time; at 53
bits Arb's takes 0.6 µs, and the remaining cost is the ball operations
around the two kernel calls.

The remaining kernels derive `log1p`, `log2`, `log10`, `asin`, `acos`,
`atan2`, and the hyperbolic functions and their inverses from these functions.
Each kernel documents how it avoids cancellation. Like `sqrt`, `rootn`
rounds from an exact integer root: `q = floor(abs(x)**(1/n) 2**t)`
has at least `p + 2` bits, and `2q + 1`, when the root is inexact, lies
strictly between the same two rounding boundaries as the exact root.

## The Ziv driver

The driver evaluates the kernel at working precision `w`, then rounds both
ends of its enclosure to the target format. If both give the same Float and
status, monotonicity ensures that every point between them, including the
exact value, rounds the same way. Otherwise, it doubles the guard bits
`w - p` and tries again. The first `w` is `p + 32 + bit_length(p)`.

The loop terminates for every argument that is not an exact case. Every
rounding boundary is a binary fraction, a Float of the target precision or the
midpoint of two neighbours. By the Lindemann-Weierstrass and Gelfond-Schneider
theorems, the functions' values at binary fractions other than the exact cases
of Appendix D.1 of the requirements are transcendental, so the exact value is
never a boundary, and a narrow enough enclosure excludes every boundary.

The same step is public as `apn_mojo.ball.to_float_if_certain`. Requiring
equal statuses as well as equal values costs at most a little precision when
the enclosure contains the rounded value itself, and it gives the correct
rounding direction.

**Values near a Float.** For a small `x`, `sin(x) = x - x**3/6 + ...` lies
within `2**(3E - 1)` of `x`, where `E` is `x`'s exponent; a directed rounding
then hinges on that tiny difference, and Ziv would need about `3E` bits to
see it. Instead, `_round_near` decides such a value directly. Every rounding
boundary and every Float of precision `p`, other than the base `x` itself, lies
at least `2**(E - max(p + 1, q) - 2)` from a base of precision `q`, so when the
distance bound is at most `2**(E - max(p + 1, q) - 3)`, no boundary separates
the value from a point between it and the base, and rounding that point gives
the answer and its status. The same rule serves `exp`, `exp2`, `cos` and
`cosh` near 1, `expm1` and `log1p` near `x`, `log` near 1 through
`x - 1`, `expm1` near -1 and `tanh` near +-1 for large arguments.

**Exact cases** are found before the driver and rounded once: zeros of the
functions, `exp(0) = 1`, `log(1) = 0`, `exp2` of an integer, `log2(2**k) = k`,
`log10(10**k) = k`, `rootn` of a perfect power, and `pow(x, y)` for a
binary-fraction `y = c / 2**k` when `x = a 2**b` (odd `a`) has
`a = a1**(2**k)`, `2**k` divides `b`, and `a1 = 1` for a negative `c`. A
Rational exponent `c/d` gives an exact power exactly when the `d`-th root of
`x` is a binary fraction. An exact Rational argument that is not a binary
fraction has no Float point, so its kernel encloses the ball function of a
tight ball around it at each working precision.

## Complex functions

A Complex function rounds each part in its own context. The pair driver
encloses both parts, keeps either result once certified, and raises precision
for the other. Some parts are always zero for a class of arguments: the
imaginary part of `exp(x + 0i)` or of `sqrt` on the positive real axis, for
example. An enclosure cannot decide that zero's sign, so the function handles
it before calling the driver. If the budget runs out while a part's enclosure
still contains 0, the error reports a missing structural rule.

Special values follow C99 Annex G as MPC implements it, including MPC's
choices where the annex leaves a sign unspecified: `cosh(NaN + 0i)` is
`NaN - 0i`, while `cos(+0 + NaN i)` is `NaN + 0i`. The trigonometric functions
are their hyperbolic counterparts at `iz`, but MPC's special values of
`asin`, `atan` and `cos` are not always the hyperbolic ones', so those cases
have their own rules. Powers of 1, -1, i and -i with a complex exponent
`w = u + iv` follow MPC's `pow`: the power is real when `u t` is an integer
for the argument `t pi`, with an imaginary zero that is -0 when rounding
toward -inf or when the signs of `Im z` and `Re w` differ, and imaginary,
with a real part of +0, when `u t` is an odd multiple of 1/2.

`tanh(x + iy)` for x >= 1 has a real part `1 - d` with
`|d| <= (1 + e**-2) / (e**(2x)/2 - 1) < 3.12 e**(-2x)`. An enclosure of it
contains 1 until it is narrower than `d`, which for `x = 10**8` would take
about `3 * 10**8` bits, so `_round_near` decides it from the sign of `d`, that
of `cos 2y + e**(-2x)`; the imaginary part, of the size of `e**(-2x)`, keeps
its relative accuracy and rounds by its own iteration. `tan` inherits the rule.

## Special functions

Special functions use the same three layers. Their series kernels account
for each rounding in the radius and add a remainder bound whenever they
truncate a series:

| Function | Method | Remainder |
|---|---|---|
| erf | `(2/sqrt pi) e**(-x**2) sum 2**n x**(2n+1) / (2n+1)!!`, positive terms | once a term ratio is at most 1/2, the tail is below the last term |
| erfc | `1 + erf|x|` for x < 0; `1 - erf x` with `1.443 x**2` extra bits; for large x, `e**(-x**2)/(x sqrt pi) sum (-1)**k (2k-1)!!/(2x**2)**k` | the first omitted term (DLMF 7.12(i)) |
| erfi | `(2/sqrt pi) sum x**(2n+1) / (n! (2n+1))`; for large x, `(2/sqrt pi) e**(x**2) D(x)` | below |
| Ei | `gamma + log|x| + sum x**n / (n n!)`; for large x < 0, `-E1(-x)` by its asymptotic series; for large x > 0, `e**x sum_{k<n} k!/x**(k+1)` | ratio bound; the first omitted term (DLMF 6.12(ii)); below |
| Si, Ci | their series with `1.443 |x|` extra bits for the cancellation; for large x, `pi/2 - f cos x - g sin x` and `f sin x - g cos x` | ratio bound; the first omitted terms of f and g (DLMF 6.12(ii)) |
| Shi, Chi | positive series; for large x, `(Ei(x) +- E1(x))/2` with `0 < E1(x) < e**(-x)/x` | ratio bound; as Ei |
| Fresnel S, C | series in `u = pi x**2 / 2` with `1.443 u` extra bits; for large x, `1/2 - f cos u - g sin u` and `1/2 + f sin u - g cos u`, with `x**2/2` reduced exactly modulo 2 | ratio bound; the first omitted terms (DLMF 7.12(ii)) |
| Gamma, log Gamma, digamma | below `w/2 + 10` and up to 1256 bits, the tabulated Taylor series of `1/Gamma(1 + u)` and its derivative at `x = 1 + u + m`, `|u| <= 1/2` (below), with `P = (1+u) ... (m+u)` and `P'` from one fixed-point pass: Gamma `P/R`, log Gamma its logarithm (of `Gamma(1 + x)/x` below 1/2), digamma `P'/P - R'/R`; in native 128-bit integers where `w + 8 <= 64`; otherwise Stirling's series at `z = x + N >= w/2 + 10`, its coefficients in integers from the tabulated tangent numbers (below), back to x by `Gamma(x) = Gamma(z) / (x (x+1) ... (x+N-1))`, log Gamma by the product's logarithm, digamma by `sum 1/(x+k)`; the products and sums exact in integers for a short dyadic x; reflection below 1/2 | Johansson's Theorem 5.3 and Cauchy's estimate (below); the first omitted term, for a real z (DLMF 5.11(ii)) |
| Lambert W | Halley's iteration, then the signs of `t e**t - x` at both ends of `t (1 -+ 2**-w)` | none: the root is bracketed |
| log Gamma below 0 | `log pi - log |sin pi x| - log Gamma(1 - x)`, x reduced exactly in `sin pi x` | as log Gamma |
| ndtr, log ndtr | `erfc(-x/sqrt 2)/2` from the erfc ball at `x/sqrt 2`, with `2 exponent(x)` more bits for erfc's sensitivity; its logarithm, as `log1p(-erfc(x/sqrt 2)/2)` for x > 0 | as erfc |
| Hurwitz zeta | direct summation while the terms shrink fast, the tail below its integral; otherwise Euler-Maclaurin at `a = q + N >= |s| + 2M`, `M = (w + 10)/5`; Riemann's below -1 by the functional equation | `4 |(s)_2M| (2 pi)**-2M a**(1-s-2M) / (s+2M-1)` (DLMF 24.9.4); the integral |
| polygamma | `(-1)**(n+1) n! zeta(n+1, x)`, a negative x shifted above 0 | as zeta |
| erfinv, ndtri | Newton's iteration on erf for `|x| <= 1/2` and on erfc in the tails, from Giles's approximation or `t = sqrt(L - log(t sqrt pi))`, each step at twice the bits the last one showed right | none: the root is bracketed by the signs at `t (1 -+ 2**-(w+8))` |
| hyp1f1 | the series at `|x|` after Kummer's transformation for x < 0, or, where the series would be longer than about the precision, DLMF 13.2.41's real part, `Gamma(b) [e**x x**(a-b) S_1 / Gamma(a) + cos(pi a) x**-a S_2 / Gamma(b-a)]` with U's expansions; rational values exactly ([Hypergeometric functions](hypergeometric.md)) | the ratio bound `|T_N| / (1 - D)` (Johansson 2019); Olver's bounds (DLMF 13.7.5) |
| gammainc, gammaincc | the smaller of P and Q directly, the other as 1 minus it: P by `x**a e**-x / Gamma(a+1) M(1, a+1, x)` for `x < a + 1`, Q by its asymptotic expansion past it, or `1 - P` with `-log2 Q` more bits | the ratio bound; the first omitted term (DLMF 8.11.3) |
| hyp2f1 | Pfaff's transformation below -1/2, the series up to 1/2, the connection with `1 - x` (DLMF 15.8.4) above, its digamma limit for an integer `c - a - b` (15.8.10); Gauss's value at 1 | the ratio bound; for the digamma series, `|L_k| <= |log t| + |a+m-1| h + |b-1| h` from `psi' < 1/t + 1/t**2` |
| betainc | `x**a (1-x)**b / (a B(a, b)) F(a+b, 1; a+1; x)` for `x <= (a+1)/(a+b+2)`, positive shrinking terms, else `1 - I_{1-x}(b, a)` | the ratio bound |
| beta, betaln, poch | `Gamma(a) Gamma(b) / Gamma(a+b)`, `log |Gamma(a)| + log |Gamma(b)| - log |Gamma(a+b)|` with the magnitude of the larger term in extra bits, `Gamma(z+m) / Gamma(z)`; rational values exactly (`ball/_ratios.mojo`) | as Gamma |

Every series is used only where its cancellation and its length stay within a
few times the working precision; an asymptotic series is used once its
smallest term is below the precision, and otherwise the convergent one.

**Dawson's integral.** `D(x) = e**(-x**2) int_0^x e**(t**2) dt
= (x/2) int_0^1 e**(-x**2 v) (1 - v)**(-1/2) dv`. Expanding
`(1 - v)**(-1/2) = sum c_k v**k`, with `c_k = (2k-1)!!/(2**k k!)` decreasing,
leaves a remainder between 0 and `v**n (1 - v)**(-1/2)`. Integrating, with
`int_0^1 = int_0^inf - int_1^inf` for the polynomial part and a split at
`v = 1/2` for the remainder, gives
`-n e**(-x**2)/x <= D(x) - sum_{k<n} T_k <= 2 sqrt(2n) T_n + (x/sqrt 2) e**(-x**2/2)`
for `T_k = (2k-1)!!/(2**(k+1) x**(2k+1))` and `n <= x**2/2`, using
`c_n >= 1/(2 sqrt n)`. It reaches about `0.72 x**2` bits.

**Ei at large x > 0.** `e**(-x) Ei(x) = PV int_0^inf e**(-t)/(x - t) dt`.
On `[0, x/2]` the geometric expansion of `1/(1 - t/x)` gives
`sum_{k<n} k!/x**(k+1)` less at most `4 e**(-x/2)/x`, plus a remainder in
`[0, 2 n!/x**(n+1)]`, for `n <= x/4`. The rest is
`e**(-x) (2 Shi(x/2) - E1(x/2))`, between `-e**(-3x/2)` and
`8 e**(-x/2)/x` since `Shi(a) <= 2 e**a/a` for `a >= 8`. So
`|e**(-x) Ei(x) - sum| <= 2 n!/x**(n+1) + 8 e**(-x/2)/x` for `x >= 16`.

**Values near a Float** decided by `_round_near`:

| Function | Argument | Base | Distance below | Side |
|---|---|---|---|---|
| erf | `|x| >= 1` | `+-1` | `e**(-x**2)` | toward 0 |
| erfc | `x <= -1`; `|x| < 1/2` | 2; 1 | `e**(-x**2)`; `1.13 |x|` | below; above for x < 0 |
| Si, Shi | `|x| < 1/2` | x | `|x|**3 / 16` | toward 0 for Si, away for Shi |
| Fresnel C | `|x| < 1/2` | x | `|x|**5 / 4` | toward 0 |
| Fresnel S, C | `|x| >= 2**(p+4)` | `+-1/2` | `1/|x|` | the sign of `-(f cos u + g sin u)`, `f sin u - g cos u` |
| Lambert W_0 | `|x| < 1/4` | x | `4 x**2` | below |
| Gamma, digamma | `x = +-2**k <= 1/4` | `1/x`, `-1/x` | 2 | below |

Gamma's overflow uses `log Gamma(x) > (x - 1/2) log x - x`, Binet's remainder
being positive, and its underflow for x < 0 uses
`|Gamma(x)| <= pi / (2 d Gamma(1 - x))` with d the distance from x to the
nearest integer. erfc, erfi, Ei, Shi and Chi decide overflow and underflow from
`e**(+-x**2)` and `e**(+-x)` bounds before their kernels, whose exponents
would leave the 64-bit range.

## Budgets

Adaptive calculations are bounded by `max_precision` in `ArithmeticContext`
and `BallContext`. It defaults to `max(8 p, p + 4096)` for output precision
`p`. If the driver's next precision would exceed this budget, a Float
function raises an error naming the function, argument, and limit:

```text
precision budget exceeded: sin at 9.900656229295898e+301029 needed more than 4149 bits; pass a larger max_precision in the context.
```

Argument reduction for `sin`, `cos` and `tan` needs about as many bits of pi
as the argument's exponent, and stops at the budget: `sin(2**(10**6))` at 53
bits raises. A ball function never raises for its budget; it returns a
trivially valid enclosure instead, `[0 +/- 1]` for `sin` and `cos` and an
indeterminate ball for `tan`. A tight budget, such as `p + 8`, makes every
non-exact call raise at once; exact cases need none.

## Constants

Constants share one binary-splitting engine (Haible and Papanikolaou). It
merges terms like a binary counter on an explicit stack, keeping the depth
logarithmic. Each truncated series adds a bound for its tail:

| Constant | Formula | Tail after `n` terms |
|---|---|---|
| pi | Chudnovsky, `426880 sqrt(10005) / S` | `2**30 (n + 1) 2**(-47 n)`, the next term of an alternating series |
| e | `sum 1/k!` | `2 / n!` |
| ln 2 | `18 atanh(1/26) - 2 atanh(1/4801) + 8 atanh(1/8749)` | `2 m**-(2n+1)` for `atanh(1/m)` |
| log2(10) | `3 + 2 atanh(1/9) / ln 2` | as for `atanh` |
| gamma | the table up to 4608 bits (as ln 2 and pi); beyond, Brent-McMillan: `A/B - log n - K0(2n)/I0(2n)` for a power of two `n` | the table's unit; the Bessel ratio lies in `(0, pi e**(-4n))`; tails `2 t_K H_K` and `(4/3) t_K` |
| Catalan | Ramanujan: `(pi/8) log(2 + sqrt 3) + (3/8) sum (k!)**2 / ((2k)! (2k + 1)**2)` | `2 * 4**-n` |

The Bessel bound follows from `K0(x) < sqrt(pi/2x) e**-x` and
`I0(x) > e**x / sqrt(2 pi x)` for `x >= 1`.

ln 2 and pi, which every `exp`, `log` and argument reduction needs, come from
tables up to 4096 bits: `floor(c 2**(4096 - E))` for the constant's exponent
`E`, compiled in as 64 words. A request for `b` bits reads the top `b` bits as
the midpoint, and one unit of their last place is the radius, since the
constant exceeds its truncation by less than that unit. The tables are the
series' own floors at 4300 bits, which the tests check, and agree with MPFR's.
Recomputing ln 2 was 42% of a 64-bit `exp`.

A constant's canonical ball at precision `p` is
`[round_down(c, p), round_up(c, p)]`, computed by two directed runs of the
driver; `canonical(ball, p)` gives the same ball from any wider enclosure, or
None when it cannot decide both ends. A caller that caches the widest ball it
has computed therefore serves every narrower request exactly as a fresh
computation would.

## Determinism

Calls share no mutable cache or numerical flags. A result depends only on
its arguments and context, so repeated calls give the same bits, including
through `vmap` and at different thread counts. The tests check this property.
