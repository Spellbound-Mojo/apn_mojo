# Text and JSON

## Deliberate text parsing

Start with `Integer("123")` to read a decimal integer. By default, the parser
accepts a sign and leading zeros, but rejects surrounding whitespace,
underscores, prefixes, and trailing characters. You can choose the same
parsing options in the constructor and in `Integer.parse`.

`base=0` detects `0b`, `0o`, and `0x` prefixes, using decimal when none is
present. With an explicit base, `allow_prefix=True` permits its matching
prefix. `allow_whitespace=True` accepts surrounding ASCII whitespace, and
`allow_underscores=True` permits separators between digits.

## Keep every digit in storage

Integers print in decimal, and rationals as reduced fractions or whole
numbers. Floats and complex components print the shortest decimal that reads
back as the same value in its format, such as `0.1` or `3.0`.
`to_string()` offers the formatting options for each type: for Floats,
positional or scientific notation, a digit count, and the exact hexadecimal
form for tests and interchange.

JSON preserves both values and format metadata for integers, rationals,
floats, complex numbers, and their batches. Large numeric fields are strings,
so a reader can preserve them even if its native number type is too small.
Decode with the matching `from_json()` method. Real and complex balls also
have scalar JSON formats; ball batches currently support text output only.
`ExactComplex` saves its two rational parts exactly; see the
[complex-fraction example](complex.md#keep-complex-fractions-exact).

<!-- example: docs/examples/conversion.mojo -->

For native Mojo numbers instead of serialized text, see
[native batch output](../reference/batch.md#native-values). That conversion
can round values or discard interval bounds; it serves a different purpose
from saving an APN value and its metadata.

## Bounding conversions

To bound the work a conversion can do, pass `limits=ConversionLimits(...)`.
You can reuse the limits object: each call gets a fresh budget, and the
resulting value does not retain it. An unset field (`None`) is unlimited;
zero allows none of that resource.

| Setting | What it bounds |
|---|---|
| `max_input_bytes` | Raw input bytes before trimming or unescaping |
| `max_output_bytes` | Bytes in the returned text |
| `max_digits` | Numeric digits parsed or written, as defined by the format |
| `max_values` | Logical values, including batch elements |
| `max_allocated_bytes` | Cumulative allocation requests during conversion |

Apply transport or file-size limits before buffering large payloads, since
conversion limits cannot recover memory already used to receive the input.
Reject an oversized record; truncating it could change its value. The policy
does not cap later arithmetic, retained snapshots, or process execution time.

See the [conversion reference](../reference/conversion.md) for grammars,
JSON records, and exactly what each limit counts.
