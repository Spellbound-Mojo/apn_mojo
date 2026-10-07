# Conversion and interchange

Conversion must preserve the meaning of a value while keeping resource use
under control. JSON saves Integer, Rational, Float and Complex values and
formats, including batches. Integer and Rational text parses exactly; Float
and Complex text rounds to a chosen format. `ConversionLimits` bounds the
work of a call. This chapter explains the readers, writers, and budgets; the
[conversion reference](../reference/conversion.md) gives the public grammar
and schemas. Real and complex balls have their own text and scalar JSON formats.

## One reader, family schemas

`common/_json.mojo` tokenizes JSON and validates record fields; `common/_text.mojo`
handles ASCII digits and trimming. Neither depends on a numeric family. Each
family keeps its own grammar, schema, numeric validation and diagnostics.

The JSON reader is schema-specific and non-recursive: it accepts exactly the
documented records, in any field order and with JSON whitespace and escapes,
and rejects duplicates (including escaped-equivalent names), unknown fields,
comments and trailing data. Batch imports hold parsed records and converted
values until the entire input is valid.

Syntax errors report a zero-based UTF-8 byte offset and, inside an array, the
logical element. An offset may point just after a string whose decoded value
failed validation.

Rank-one batches write version 1; any other rank writes version 2, which adds
an explicit shape ahead of the row-major values. Records carry values and
formats only, never representation, capacity, pointers, hashes or workspaces,
so the format does not depend on the memory layout or byte order. Encoders
emit compact ASCII in a fixed field order, so version-1 bytes are canonical.

All four numeric batch families share the envelope driver in `batch/_json.mojo`.
It validates fields, traverses the values array, and checks the version and shape
schema. Only Float and Complex accept and require `default_format`. Each codec
owns its record parser, temporary conversion state, budget charges, and error
boundaries. The shared reader has no dependency on Float's deferred significands.

Integer and Rational first collect validated decimal records, then convert them
into scalar values. Float and Complex restore their records during the parse,
with large significands deferred when allocation limits permit. These paths
retain their existing validation order and element diagnostics.

Limited output uses a shared append loop and closing frame. Family callbacks
write records directly into the output buffer and charge for their bytes in
the established order. Integer and Rational records need no extra temporary
string. Unlimited writers use their existing sequential and parallel paths.

## Wire formats

Integer and Rational records hold canonical decimal strings; their schemas are
in [the reference](../reference/conversion.md#json). A Float record
keeps its format with its value:

```json
{"version":1,"family":"float","precision":"3","emin":"-10","emax":"10","class":"finite","sign":"-","significand":"5","exponent":"-1"}
```

This is `-5 * 2**(-1-3)` with a three-bit significand. Precision, bounds,
significand and exponent are canonical decimal strings, even when they would
fit a JSON number. `class` is `finite`, `zero`, `infinity` or `nan`; `sign` is
`"+"` or `"-"`, and NaN is always `"+"`. Only finite records carry
`significand` and `exponent`, and the significand must have exactly
`precision` bits. Import rejects noncanonical payloads instead of normalizing or
rounding them, and takes no context: records restore exactly.

A Float batch adds a default format, kept even when the batch is empty:

```json
{"version":1,"family":"float-batch","default_format":{"precision":"3","emin":"-10","emax":"10"},"values":[]}
```

A Complex record embeds two complete Float records, whose bounds must agree:

```json
{"version":1,"family":"complex","real":{"version":1,"family":"float","precision":"3","emin":"-10","emax":"10","class":"zero","sign":"+"},"imag":{"version":1,"family":"float","precision":"3","emin":"-10","emax":"10","class":"zero","sign":"+"}}
```

A Complex batch keeps a pair of default formats in a `complex-batch` envelope.
Arithmetic state, traps and sharing are never serialized.

## Exact text

**Integers** parse decimal, or with `base=0`, prefixed binary, octal and
hexadecimal, optionally with single underscores between digits. Digits are
validated before any magnitude is built.

**Rationals** parse `n/d` and exact decimals, so `"0.1"` is `1/10`. Decimal
powers, multiplication, division and the gcd reduction run on budgeted buffers
through the allocation-free Integer kernels.

**Floats** parse decimal (`1.25e-3`), hexadecimal (`0x1.8p2`) and binary
(`0b1.01p-2`) sources and round the complete exact source once, with no native
floating-point intermediate. `Float("0.1")` starts from exact `1/10`;
`Float(Float64(0.1))` starts from the already rounded native value. At the
default precision, these produce different results. The parser validates the whole grammar before any zero
or range shortcut. Exponent accumulation saturates at `2**80`: no addressable
significand can bring a larger magnitude back into range. Coarse integer
logarithm bounds classify remote decimal exponents before any power is formed.

Float output is the shortest decimal that reads back in the value's format
by default (`shortest_decimal`), laid out positionally from 1e-6 up to 1e21 and
in scientific notation beyond. Exact hexadecimal output (`to_string(16)`)
reads existing significand words without exponent-sized buffers. Decimal
output with a digit count (`to_string(digits=N)`)
rounds the exact stored value with any of the five modes: it finds the exact
decimal order by correcting an integer estimate, divides at the requested
decimal unit, and rounds with the remainder and parity. Parsing exact output
with the value's own format reconstructs it.

**Complex** text accepts `3`, `4j`, `j`, `3+4j`, `3-4i`, `(3+4j)`, `(3,4)` and
the display spelling `Complex(3, 4)`. Omitted components are `+0`, and an
omitted imaginary coefficient is one. Decimal exponent signs do not split
components: `1e+2-3e-1j` is `100 - 0.3j`. The parser splits the two spans
without allocating substrings and delegates each to Float. Both are validated
before either rounds, and neither is published until both succeed.

Ball text represents an enclosure. Printing rounds the midpoint to decimal
and expands the printed radius to cover that rounding, so parsing the output
can produce a wider ball. See [Ball text](../reference/ball.md#text-keys-and-scope).

### Large exponents

The default exponent range is about $\pm 2^{62}$, so exact decimal conversion
can need powers of ten as long as the exponent: printing $2^{10^{12}}$ or
reading `1e1000000000000` exactly would build integers of about $10^{12}$
bits. Beyond a threshold, the shortest decimal, decimal output with a digit
count, and decimal parsing into a finite format bound the result instead
(`float/_bounds10.mojo`):

- $10^k$ is formed as $5^k \cdot 2^k$ by squaring, every product rounded down
  for a lower bound and up for an upper bound, at a working precision $P$.
  Every factor is positive, so the bounds hold, and bounds of
  $n \cdot 2^s \cdot 10^k$ follow with one more product or quotient.
- Output needs floors of such values: of $x / 10^j$ for the decimal order and
  the digits, and of twice it to round to nearest. When both bounds have the
  same floor and the lower one is not an integer, the floor is decided.
- Parsing rounds both bounds to the target format with its mode and traps
  off; when they agree, with the same status, that is the result, and the
  traps apply to it once.
- Otherwise $P$ doubles (Ziv's strategy). $P$ starts 64 bits above the
  format's precision (plus about 3.3 bits per requested digit), and one try
  nearly always decides.

Doubling ends because no value decided here lies on a boundary. Take a
significand $n < 2^{p+4}$ and a scale $s$. For $s > 0$, $n \cdot 2^s / 10^j$
is an integer only if $5^j$ divides $n$, so only if $j < (p + 4) / \log_2 5$.
The threshold, $|s| > 65536 + 8p$ for the shortest decimal and
$65536 + 8p + 16N$ for $N$ digits, puts $j$ near $0.3\,s$, far above that.
For $s < 0$, $n \cdot 10^{|j|} / 2^{|s|}$ is an integer only if
$2^{|s| - |j|}$ divides $n \cdot 5^{|j|}$, so only if
$|s| - |j| \le p + 4$, while $|s| - |j|$ is near $0.7\,|s|$. Twice the
value, for rounding to nearest, adds one bit to either margin. In parsing,
$N \cdot 10^S$ with $N$ of $b$ bits takes the bounded path when
$3.3\,|S| > 65536 + 4(b + p)$. For $S > 0$ its odd part has at least
$2.32\,S > p + 1$ bits, so it is neither a $p$-bit value nor a midpoint. For
$S < 0$ it is a binary fraction only if $5^{|S|}$ divides $N$, but
$5^{|S|} > 2^b > N$. Below the thresholds the exact algorithms run, on powers
of at most about $65536 + 8p$ bits.

The decimal order comes from $\lfloor n \log_{10} 2 \rfloor$ in exact
128-bit arithmetic with an 18-digit constant, within one at every 64-bit $n$.
The 0.30103 that suffices for small exponents is off by about $10^{10}$ at
$2^{61}$; a search starting there would step that many times.

In an exact format, a decimal with a negative exponent whose power of five
exceeds its digits cannot be a binary fraction. It raises at once, without
forming the power. `shortest_decimal` also counts its exact work against
`max_allocated_bytes`.

`tests/fixtures/large_exponent_decimal.txt` holds 1043 cases, from exponents
just below the threshold to $\pm 2^{61}$, in every rounding mode. They come
from Arb's balls, whose exponents are unbounded, and are decided only by
certain comparisons. Within MPFR's default exponent range ($2^{30}$ through
gmpy2), all 350 of the cases it can check agree with MPFR.

## Budgets

`ConversionLimits` counts input bytes, output bytes, digits, values, and
requested allocation bytes for one call. Everything shares that budget:
nested JSON decoding, numeric construction, formatting, and the returned
string. The same allowance covers all batch elements.

- **Before work.** The input-byte limit is checked before scanning. Digit limits
  are checked during lexical validation, before a magnitude is built. Value
  limits are checked before the next value is decoded.
- **Before allocation.** Every allocation's entire requested capacity, including
  spare capacity and owner headers, is charged before the allocation happens.
  Reallocation charges the whole replacement; freed temporaries are not
  refunded. The charges follow the pinned `List` and `String` growth policies,
  not payload estimates.
- **Output.** A cheap digit-count lower bound is checked before magnitudes are
  copied, then exact counts as digits are produced; the estimate never rejects
  output that fits.
- **Diagnostics.** Building an error message has its own 8 KiB allowance.

The rounding kernel has a compile-time budgeted variant that charges its
temporary limbs and owners; ordinary arithmetic uses the unbudgeted variant.
Errors name the limit, its setting, the requested count, the logical element
when there is one, and a remedy. A failed conversion returns nothing and leaves
a reassigned destination unchanged.

Limits bound one conversion only. They do not bound later arithmetic, retained
snapshots, RSS or time; physical out-of-memory remains fatal.
