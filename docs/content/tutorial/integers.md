# Integer values

## Begin with an Integer

Use [`Integer`](../reference/integer.md) for exact whole-number calculations
that can grow beyond native integer limits. `Integer()` creates zero; you
can also supply a native integer of up to 64 bits, an integer literal of any
width, or text. For example, `Integer(2) ** 100` computes the exact power.

Convert native variables before a calculation that might exceed their range:
once a native calculation overflows, wrapping its result in `Integer` cannot
recover the lost bits. Integer literals keep their full value through the
constructor's separate `IntLiteral` path.

Assigning `var saved = value` keeps a copy of the number. Changing either
variable leaves the other alone. Large values can share storage internally
until an update needs its own buffer.

## Arithmetic and division

`+`, `-`, `*`, `**`, `//`, and `%` return exact integers, within the library's
storage limits. Integer exponents must be nonnegative; `0 ** 0` is `1`.

Division with `/` returns an exact [`Rational`](../reference/rational.md):
`Integer(7) / 3` is `7/3`, while `Integer(7) // 3` is `2`. Use `pow_rational`
for a negative exponent that should produce an exact fraction.

With negative operands, choose the division rule that your calculation needs:

| Call | Quotient rule | Remainder rule |
|---|---|---|
| `div_rem_floor(a, b)` or `a // b`, `a % b` | Round toward negative infinity | Zero or the sign of `b` |
| `div_rem_trunc(a, b)` | Round toward zero | Zero or the sign of `a` |
| `div_rem_euclid(a, b)` | Choose a nonnegative remainder | `0 <= r < abs(b)` |

All three satisfy `a == q * b + r` and raise on a zero divisor. If you need
both results, call the corresponding `div_rem_*` function. Standard `divmod`
is not supported.

<!-- example: docs/examples/integer_values.mojo -->

## Signed bits and comparisons

Bit operations (`&`, `|`, `^`, and `~`) behave as if integers had an unlimited
sign-extended two's-complement representation. Right shifts round toward
negative infinity: `Integer(-7) >> 1` is `-4`. Shift counts must be nonnegative.
`magnitude_bit_length()` counts the bits in the magnitude; zero has length zero.

Comparisons return a `Bool`, and `Bool(value)` is false only for zero.
With typed native operands, put the library value on the left or convert the
native value explicitly: `value > native` or `Integer(native) < value`.
Mojo's native comparison overloads do not accept the library value on the right.

## Return to native code

`Int(value)` and `value.to_native_exact[DType.uint8]()` check that the number
fits the target type. They raise for an out-of-range value. Choose a wider
native type or keep the `Integer` when it does not fit.

You can use integers as dictionary and set keys; equal values have equal
ordinary hashes. For a reproducible APNH-64 hash, use `stable_hash`. These
hashes and arithmetic routines provide no cryptographic guarantees.

See the [Integer reference](../reference/integer.md) for roots, number theory,
combinatorics, and modular arithmetic. The [batch tutorial](batches.md)
explains how to work with collections of integers.
