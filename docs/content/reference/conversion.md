# apn_mojo.common

Use `ConversionLimits` to bound a conversion's resource use, and `FloatKey`,
`ComplexKey`, or `BallKey` for dictionary keys that distinguish stored
representations. This page also covers the shared text and JSON rules. For
an introduction, see [Text and JSON](../tutorial/conversion.md).

<!-- api: common -->

## Integer text

Integer parsing accepts ASCII digits and, for bases above ten, letters without
case distinctions. A sign precedes any prefix. Leading zeros are accepted,
but the whole input must match the grammar. `base=0` detects binary, octal,
and hexadecimal prefixes, with decimal as the fallback.

With an explicit base and prefixes disabled, prefix-like characters are just
digits when that base permits them: `Integer.parse("0b10", 16)` is 2832.
Whitespace and underscores require their respective options; internal
whitespace, repeated underscores, and trailing characters are invalid.
Formatting produces unpadded digits; printing an Integer uses decimal.

Other grammars are documented under [Rational](rational.md#exact-text),
[Float](float.md#text-and-json), [Complex](complex.md#text-and-json),
[ExactComplex](exact_complex.md#text-json-and-hashing),
[Ball](ball.md#text-keys-and-scope), and
[ComplexBall](complex_ball.md#text-hashing-and-batches).

## JSON

An integer record stores its value as a canonical decimal string:

```json
{"version":1,"family":"integer","value":"18446744073709551616"}
```

A rank-one batch uses version 1:

```json
{"version":1,"family":"integer-batch","values":["18446744073709551616","-7"]}
```

Other ranks use version 2 with a shape and flat row-major values:

```json
{"version":2,"family":"integer-batch","shape":["2","3"],"values":["1","2","3","4","5","6"]}
```

Shape dimensions must multiply to the number of values. Rational, Float, and
Complex batches use the corresponding family records; see
[wire formats](../architecture/conversion.md#wire-formats). Float and Complex
batches also save default formats for empty containers. Real and complex
balls have scalar JSON records; ball batches do not yet have JSON.
ExactComplex has a scalar record with two canonical Rational records;
there is no ExactComplex batch type.

- `version` is a JSON integer, not a string.
- `family` and the allowed fields must match the target type.
- Numeric payload strings use canonical decimal integers: `0` or an optional
  minus followed by nonzero-leading digits. Fields such as Float `sign` and
  `class` have their own grammar.
- Decoders accept field reordering, JSON whitespace, and valid escapes, but
  reject duplicate or unknown fields, comments, and trailing commas.
- Encoders write compact ASCII in a fixed field order.

Parse errors report byte offsets and, where applicable, logical element indices.

## What limits count

| Limit | Counted resource |
|---|---|
| `max_input_bytes` / `max_output_bytes` | Raw input before trimming or unescaping, or returned output bytes |
| `max_digits` | Numeric payload digits consumed or written, including written exponents; signs and prefixes are excluded |
| `max_values` | Logical numeric values, including batch elements |
| `max_allocated_bytes` | Cumulative allocation requests made by the conversion |

Float JSON counts significand and value-exponent digits, but not format metadata
such as precision or exponent bounds. Batch shape metadata does not count toward
the digit limit either; byte and allocation limits still cover it.

One budget covers the whole call, including nested components and final
output. Freeing temporary storage does not restore the allocation budget.
Error messages have a separate 8 KiB allowance, and each new call starts
with a fresh budget.

These limits do not cover memory used to receive a payload, later arithmetic,
or elapsed time. Apply input-size limits before buffering large records.

## Representation keys

`FloatKey`, `ComplexKey`, and `BallKey` compare and hash stored representations.
This preserves distinctions such as zero signs, formats, and special-value
classes. It is useful for caches that must distinguish values ordinary numeric
equality would merge. The
[representation chapter](../architecture/float.md#representation-identity-and-hashing)
explains identity, ordering, and `stable_hash`.
