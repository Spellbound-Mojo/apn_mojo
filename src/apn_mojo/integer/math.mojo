"""Exact combinatorics, general roots, and modular integer arithmetic."""

from .value import Integer
from ..common._stable_hash import _StableHash
from ..rational.value import Rational
from .number_theory import isqrt
from std.bit import count_leading_zeros, pop_count
from std.collections import Array
from ._division import _store_words
from ._limbs import _multiply_by_limb, _multiply_limbs, _shift_left_limbs
from ._limits import _MAX_RESULT_BITS
from ._root import _integer_root


comptime _SMALL_FACTORIALS = SIMD[DType.uint64, 32](
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800,
    479001600, 6227020800, 87178291200, 1307674368000, 20922789888000,
    355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
"""n! for n <= 20, the largest that fits an Int64; zeros pad the vector."""

comptime _SMALL_DOUBLE_FACTORIALS = SIMD[DType.uint64, 64](
    1, 1, 2, 3, 8, 15, 48, 105, 384, 945, 3840, 10395, 46080, 135135, 645120,
    2027025, 10321920, 34459425, 185794560, 654729075, 3715891200, 13749310575,
    81749606400, 316234143225, 1961990553600, 7905853580625, 51011754393600,
    213458046676875, 1428329123020800, 6190283353629375, 42849873690624000,
    191898783962510625, 1371195958099968000, 6332659870762850625,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)
"""n!! for n <= 33, the largest that fits an Int64; zeros pad the vector."""


@always_inline
def _table_value[size: Int, table: SIMD[DType.uint64, size]](index: Int) -> UInt64:
    """`table[index]`, for 0 <= index < size. LLVM turns these constant cases
    into a chain of compares and conditional moves; indexing the vector at a
    runtime index copied all of it to the stack first, 8 or 16 AVX stores."""
    comptime for i in range(size):
        if index == i:
            return table[i]
    return 0


def _math_count(value: Integer, limit: Int, operation: String) raises -> Int:
    if value._storage.isa[Int64]() and Int(value._storage[Int64]) <= limit:
        return Int(value._storage[Int64])
    if value > limit:
        raise Error(
            String(
                "Cannot compute ",
                operation,
                ": the result exceeds supported addressable storage; use a",
                (
                    " smaller input or selection count. The destination is"
                    " unchanged."
                ),
            )
        )
    return Int(value)


comptime _CHAIN_LIMBS = 96
"""Products of fewer limbs multiply into one accumulator a limb at a time; a
balanced split costs as many limb products below Karatsuba's threshold (48
limbs a side), plus a call each. Longer ones split in halves, so that the
products past that threshold meet at the top."""

comptime _FACTORIAL_STACK_LIMBS = 1024
"""Scratch limbs on the stack: odd parts of factorials up to about 1,500!."""


def _odd_words(n: Int, every_level: Bool, words: Pointer[mut=True, UInt64, _]) -> Int:
    """Store in words the odd numbers 3, 5, ... up to n (with every_level, then
    up to n / 2, n / 4 and so on), packed into words of at most 64 bits;
    return their count. In a level whose terms have at most b bits, any
    64 / b of them fit a word, so words fill in groups of that many with no
    comparison; checking each product against a bound instead cost 11
    instructions a term. The words' multiplications do not wait on one
    another."""
    var count = 0
    var top = n
    while top >= 3:
        var group = 64 // (64 - Int(count_leading_zeros(UInt64(top))))
        var last = UInt64(top)
        var term = UInt64(3)
        while term <= last:
            var word = term
            var end = min(term + UInt64(2 * (group - 1)), last)
            term += 2
            while term <= end:
                word *= term
                term += 2
            words.unsafe_offset(count)[] = word
            count += 1
        if not every_level:
            break
        top >>= 1
    return count


def _limb_product(
    words: Pointer[mut=True, UInt64, _], k: Int, product: Pointer[mut=True, UInt64, _],
    work: Pointer[mut=True, UInt64, _],
) raises -> Int:
    """product = words[0] words[1] ... words[k - 1], for k >= 1 nonzero words:
    product has room for k limbs, work for 2 k plus the recursion's depth.
    Returns the product's length."""
    if k <= _CHAIN_LIMBS:
        product.unsafe_offset(0)[] = words.unsafe_offset(0)[]
        var n = 1
        for i in range(1, k):
            var carry = _multiply_by_limb(product, n, words.unsafe_offset(i)[])
            if carry:
                product.unsafe_offset(n)[] = carry
                n += 1
        return n
    # The halves' products go to work[0, k); their own scratch follows, in
    # disjoint slices of one block.
    var half = k // 2
    var block = work.unsafe_origin_cast[MutUntrackedOrigin]()
    var left = _limb_product(words, half, block, block.unsafe_offset(k))
    var right = _limb_product(words.unsafe_offset(half), k - half, block.unsafe_offset(half), block.unsafe_offset(k))
    return _multiply_limbs(block, left, block.unsafe_offset(half), right, product)


def _odd_product(n: Int, every_level: Bool, shift: Int) raises -> Integer:
    """The product of the odd numbers up to n, times 2**shift; with
    every_level, of those up to n, n / 2, n / 4 and so on: the odd part of n!,
    whose odd numbers j come in once for each a with j 2**a <= n.

    The odd numbers pack into words, which multiply on limbs in one scratch
    block with one
    allocation for the result. A level whose terms have b bits packs at least
    64 / b - 1 of them a word, so the words number at most
    n log2(n) / (64 - log2(n)), plus one partial word a level."""
    var bits = 64 - Int(count_leading_zeros(UInt64(n)))
    var levels = bits if every_level else 1
    # For n below 2**16 a constant divisor, which compiles to a multiplication.
    var full = n * bits // 48 if bits <= 16 else n * bits // (64 - bits)
    var capacity = full + levels + 6
    var limbs = capacity + (capacity + 1) + (2 * capacity + 64)
    var stack = Array[UInt64, _FACTORIAL_STACK_LIMBS](uninitialized=True)
    var heap = List[UInt64]()
    var scratch: Pointer[UInt64, MutUntrackedOrigin]
    if limbs <= _FACTORIAL_STACK_LIMBS:
        scratch = Span(stack).unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    else:
        heap = List[UInt64](unsafe_uninit_length=limbs)
        scratch = heap.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var words = scratch
    var product = words.unsafe_offset(capacity)
    var work = product.unsafe_offset(capacity + 1)
    var count = _odd_words(n, every_level, words)
    var used = 1
    if count:
        used = _limb_product(words, count, product, work)
    else:
        product.unsafe_offset(0)[] = 1
    used = _shift_left_limbs(product, used, shift & 63)
    var whole = shift >> 6
    var owner = Integer._Shared.uninitialized(2 * (used + whole), False)
    var target = owner[].words.unsafe_ptr()
    for i in range(2 * whole):
        target.unsafe_offset(i)[] = 0
    var stored = _store_words(product, used, 0, target.unsafe_offset(2 * whole))
    _ = heap^
    _ = stack^
    return Integer._from_product[False](owner^, 2 * whole + stored)


def factorial(n: Integer) raises -> Integer:
    """`n!`, exactly.

    Args:
        n: A nonnegative integer.

    Returns:
        The product `1 * 2 * ... * n`; `factorial(0)` is one.

    Raises:
        When `n` is negative or the result exceeds the addressable size.
    """
    if n._negative():
        raise Error("Cannot compute factorial: use a nonnegative integer n.")
    var count = _math_count(n, _MAX_RESULT_BITS, "factorial")
    if count <= 20:
        return Integer(Int64(_table_value[32, _SMALL_FACTORIALS](count)))
    # n! is its odd part times 2**(n - popcount(n)).
    return _odd_product(count, True, count - Int(pop_count(UInt64(count))))


def factorial2(n: Integer) raises -> Integer:
    """`n!!`, exactly: `n * (n - 2) * ...`.

    Args:
        n: A nonnegative integer.

    Returns:
        The product of the integers down to 1 or 2 with the parity of `n`; zero and
        one both give one.

    Raises:
        When `n` is negative or the result exceeds the addressable size.
    """
    if n._negative():
        raise Error(
            "Cannot compute factorial2: use a nonnegative integer n."
        )
    var value = _math_count(n, 2 * _MAX_RESULT_BITS + 1, "factorial2")
    if value <= 33:
        return Integer(Int64(_table_value[64, _SMALL_DOUBLE_FACTORIALS](value)))
    if value % 2:
        return _odd_product(value, False, 0)
    # (2m)!! = 2**m m!, and m! is its odd part times 2**(m - popcount(m)).
    var half = value // 2
    return _odd_product(half, True, 2 * half - Int(pop_count(UInt64(half))))


def div_exact(a: Integer, b: Integer) raises -> Integer:
    """The quotient of an exact division, checked.

    Args:
        a: The dividend.
        b: The nonzero divisor.

    Returns:
        `a / b` when `b` divides `a`.

    Raises:
        When `b` is zero or the remainder is not zero.
    """
    if not b:
        raise Error("Cannot compute div_exact: use a nonzero divisor.")
    var pair = a._div_rem_trunc(b)
    if pair[1]:
        raise Error(
            "Cannot compute div_exact: division has a nonzero remainder; use"
            " // for floor division or div_rem_trunc for a quotient and"
            " remainder. The destination is unchanged."
        )
    return pair[0]


def comb(N: Integer, k: Integer) raises -> Integer:
    """The number of combinations of `N` things taken `k` at a time, exactly
    (scipy's `comb` with `exact=True`); `apn_mojo.comb` adds `repetition`.

    As in scipy, the result is 0 when `k > N`, `N < 0` or `k < 0`.

    Args:
        N: The number of things.
        k: The number taken.

    Returns:
        The exact count.

    Raises:
        When the result exceeds the addressable size.
    """
    var upper = N
    if k._negative() or upper._negative() or k > upper:
        return Integer(0)
    var count = k
    if count > upper - count:
        count = upper - count
    if not count:
        return Integer(1)
    if count == 1:
        return upper
    var steps = _math_count(count, _MAX_RESULT_BITS, "comb")
    var result = Integer(1)
    for i in range(1, steps + 1):
        result = div_exact(result * (upper - (i - 1)), Integer(i))
    return result^


def iroot(x: Integer, n: Integer) raises -> Integer:
    """The integer `n`-th root, truncated toward zero.

    `iroot(-1001, 3)` is `-10`: negative roots truncate toward zero, while
    nonnegative roots round down.

    Args:
        x: The radicand.
        n: The degree, positive.

    Returns:
        The root; degree one returns `x`.

    Raises:
        When `n` is not positive, or `x` is negative and `n` even.
    """
    if n._negative() or not n._word_count():
        raise Error("Cannot compute iroot: use a positive integer degree n.")
    if x._negative() and not (n._word(0) & 1):
        raise Error(
            "Cannot compute iroot of a negative value with an even degree;"
            " use a nonnegative value or an odd positive degree."
        )
    # The degree natively: one past 2**62 exceeds every bit length.
    var degree = Int.MAX
    if n._storage.isa[Int64]():
        degree = Int(n._storage[Int64])
    elif n.magnitude_bit_length() <= 62:
        degree = Int(n)
    if degree == 1:
        return x
    var bits = x.magnitude_bit_length()
    if bits <= 1:
        return x
    if degree >= bits:
        return Integer(-1 if x._negative() else 1)
    if degree == 2:
        return isqrt(x)
    # The root kernel reads the magnitude.
    var root = _integer_root(x, degree)[0]
    return -root if x._negative() else root


def _math_modulus(modulus: Integer, operation: String) raises -> Integer:
    if not modulus:
        raise Error(
            String("Cannot compute ", operation, ": use a nonzero modulus.")
        )
    return abs(modulus)


def inverse_mod(x: Integer, modulus: Integer) raises -> Integer:
    """The multiplicative inverse of `x` modulo `modulus`.

    Args:
        x: The value to invert.
        modulus: The nonzero modulus.

    Returns:
        `y` in `0 <= y < abs(modulus)` with `x * y` congruent to 1.

    Raises:
        When the modulus is zero or `gcd(x, modulus) != 1`.
    """
    var m = _math_modulus(modulus, "inverse_mod")
    if m == 1:
        return Integer(0)
    var previous_remainder = m
    var remainder = x % m
    var previous_coefficient = Integer(0)
    var coefficient = Integer(1)
    while remainder:
        var pair = previous_remainder._div_rem_trunc(remainder)
        var next_coefficient = previous_coefficient - pair[0] * coefficient
        previous_remainder = remainder
        remainder = pair[1]
        previous_coefficient = coefficient
        coefficient = next_coefficient
    if previous_remainder != 1:
        raise Error(
            "Cannot compute inverse_mod: no inverse exists; use a value and"
            " modulus whose greatest common divisor is 1."
        )
    return previous_coefficient % m


def pow_mod(
    base: Integer, exponent: Integer, modulus: Integer
) raises -> Integer:
    """`base ** exponent` modulo `modulus`, without forming the full power.

    A negative exponent first takes the modular inverse of `base`. Not constant
    time: do not use it on secret operands.

    Args:
        base: The base.
        exponent: The exponent; negative requires an inverse.
        modulus: The nonzero modulus.

    Returns:
        The result in `0 <= r < abs(modulus)`; modulus `1` or `-1` gives zero.

    Raises:
        When the modulus is zero, or the exponent is negative and `base` has no
        inverse.
    """
    var m = _math_modulus(modulus, "pow_mod")
    if m == 1:
        return Integer(0)
    if not exponent:
        return Integer(1)
    var factor: Integer
    if exponent._negative():
        try:
            factor = inverse_mod(base, m)
        except error:
            raise Error(
                String(
                    "Cannot compute pow_mod with a negative exponent: ", error
                )
            )
    else:
        factor = base % m
    if factor <= 1:
        return factor
    var count = exponent.magnitude_bit_length()
    var result = Integer(1)
    for i in range(count):
        if (exponent._word(i // 32) >> UInt32(i % 32)) & 1:
            result = (result * factor) % m
        if i + 1 < count:
            factor = (factor * factor) % m
    return result


def _math_scalar[
    operation: Int
](a: Integer, b: Integer, c: Integer) raises -> Integer:
    return _math_scalar(operation, a, b, c)


# Scalar replay dispatches at run time, so it compiles once for all operations.
def _math_scalar(operation: Int, a: Integer, b: Integer, c: Integer) raises -> Integer:
    if operation == 0:
        return factorial(a)
    elif operation == 1:
        return factorial2(a)
    elif operation == 2:
        return comb(a, b)
    elif operation == 3:
        return iroot(a, b)
    elif operation == 4:
        return inverse_mod(a, b)
    elif operation == 5:
        return div_exact(a, b)
    else:
        return pow_mod(a, b, c)


def add(left: Integer, right: Integer) raises -> Integer:
    """The exact sum; the same as `left + right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left + right`.

    Raises:
        Only on a checked size error.
    """
    return left + right


def subtract(left: Integer, right: Integer) raises -> Integer:
    """The exact difference; the same as `left - right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left - right`.

    Raises:
        Only on a checked size error.
    """
    return left - right


def multiply(left: Integer, right: Integer) raises -> Integer:
    """The exact product; the same as `left * right`.

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left * right`.

    Raises:
        Only on a checked size error.
    """
    return left * right


def divide(left: Integer, right: Integer) raises -> Rational:
    """The exact quotient as a Rational; the same as `left / right`.

    Args:
        left: The dividend.
        right: The divisor.

    Returns:
        `left / right` in lowest terms, even when integral.

    Raises:
        When `right` is zero.
    """
    return left / right


def abs(value: Integer) raises -> Integer:
    """The absolute value.

    Args:
        value: The integer.

    Returns:
        The nonnegative magnitude.

    Raises:
        Only on a checked size error.
    """
    return value.__abs__()


def stable_hash(x: Integer) -> UInt64:
    """A 64-bit hash of the value, stable across processes and releases.

    The algorithm is APNH-64: tag 1, then the sign, the number of 64-bit
    magnitude words and the words. Unlike Mojo's `Hasher`, its output never
    changes, so stored hashes stay valid. Equal Integers hash equal.

    Args:
        x: The integer.

    Returns:
        The hash.
    """
    var hash = _StableHash(1)
    hash.integer(x)
    return hash.finish()


def floor(value: Integer) raises -> Integer:
    """An Integer is its own floor (numpy's `floor`).

    Args:
        value: The operand.

    Returns:
        The value.

    Raises:
        Only on a checked size error.
    """
    return value


def ceil(value: Integer) raises -> Integer:
    """An Integer is its own ceil (numpy's `ceil`).

    Args:
        value: The operand.

    Returns:
        The value.

    Raises:
        Only on a checked size error.
    """
    return value


def trunc(value: Integer) raises -> Integer:
    """An Integer is its own trunc (numpy's `trunc`).

    Args:
        value: The operand.

    Returns:
        The value.

    Raises:
        Only on a checked size error.
    """
    return value


def round(value: Integer) raises -> Integer:
    """An Integer is its own round (numpy's `round`).

    Args:
        value: The operand.

    Returns:
        The value.

    Raises:
        Only on a checked size error.
    """
    return value


def reciprocal(value: Integer) raises -> Rational:
    """`1 / value`, exactly (numpy's `reciprocal`).

    Args:
        value: A nonzero Integer.

    Returns:
        The exact Rational reciprocal.

    Raises:
        When `value` is zero.
    """
    return divide(Integer(1), value)


def maximum(left: Integer, right: Integer) raises -> Integer:
    """The larger of two Integers, the first when they are equal (numpy's
    `maximum`).

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left` if `left >= right`, else `right`.

    Raises:
        Only on a checked size error.
    """
    return left if left >= right else right


def minimum(left: Integer, right: Integer) raises -> Integer:
    """The smaller of two Integers, the first when they are equal (numpy's
    `minimum`).

    Args:
        left: The first operand.
        right: The second operand.

    Returns:
        `left` if `left <= right`, else `right`.

    Raises:
        Only on a checked size error.
    """
    return left if left <= right else right


def clip(value: Integer, a_min: Integer, a_max: Integer) raises -> Integer:
    """`value` limited to `[a_min, a_max]`: `minimum(maximum(value, a_min),
    a_max)` (numpy's `clip`), so `a_max` when `a_min > a_max`.

    Args:
        value: The operand.
        a_min: The lower limit.
        a_max: The upper limit.

    Returns:
        The limited value.

    Raises:
        Only on a checked size error.
    """
    return minimum(maximum(value, a_min), a_max)
