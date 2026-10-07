# apn_mojo.rational

`Rational` keeps fractions exact, in lowest terms with positive denominators.
See the [Rational tutorial](../tutorial/rationals.md) for examples and
[Batch](batch.md#rational-batches) for collections of fractions.

<!-- api: rational -->

## Exact text

Supply text to parse a fraction exactly: `"1.25e-3"` becomes `1/800`, and
`"0.1"` becomes `1/10`.

```text
integer  := [+-]? digits
fraction := integer "/" integer
decimal  := [+-]? (digits ("." digits?)? | "." digits)
            ([eE] [+-]? digits)?
```

Digits must be ASCII. Leading zeros and signed zero are accepted and reduced
to canonical values. A fraction needs two integers and a nonzero denominator;
its components cannot contain decimal points or exponents. Prefixes such as
`0x`, infinities, NaNs, and trailing characters are rejected.

Zero can be parsed without expanding a huge decimal exponent. A nonzero value
whose expansion exceeds storage limits raises a checked error.

## JSON and limits

JSON records must already be reduced. The denominator is positive, the two
components are coprime, and zero has numerator `"0"` and denominator `"1"`.
Canonical decimal fields permit a minus sign on a negative numerator, but no
plus sign, padding zeros, or negative zero. Use the text constructor when
input such as `"2/4"` needs normalization.

One rational counts as one logical value under `ConversionLimits`.
Input digits include written significand and exponent digits; zeros implied
by an exponent do not count. Text output omits a denominator of one, while
JSON writes both components. The allocation budget covers numeric conversion,
normalization, and output storage. See [conversion limits](conversion.md#what-limits-count).

## Mixing families

Integer and rational operands produce exact fractions in either arithmetic
order. A Float or typed native float gives a Float result, rounded from the
exact fraction and the other operand's stored value. Library comparisons
use exact values; typed native numbers should appear on the right.

`Int(x)` requires an integral value that fits a native `Int`.
`to_integer_exact()` requires an integral value but returns an arbitrary-
precision integer. Use `floor`, `ceil`, or `trunc` for directed rounding.
`pow_rational` accepts signed integer exponents.

The family also provides `gcd`, `lcm`, and `root_exact`. If an exact root is
absent, the fraction is not the requested rational power; the generated
declarations give the domain rules. Use the family names explicitly, since
the root package exports the integer versions of `gcd` and `lcm`.
