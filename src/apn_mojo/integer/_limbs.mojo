"""Arithmetic on little-endian 64-bit limbs behind raw pointers, shared by
the limb-level kernels: greatest common divisors, square roots and k-th
roots. Callers own the buffers and their sizes; nothing here allocates."""

from std.bit import count_leading_zeros
from ._division import _divide_limbs, _divide_by_limb_into
from ._multiplication import _multiply_with_scratch, _product_scratch_size


@always_inline
def _add_limbs(x: Pointer[mut=True, UInt64, _], y: Pointer[mut=False, UInt64, _], n: Int) -> UInt64:
    """x[0, n) += y[0, n); the carry out."""
    var carry = UInt64(0)
    for i in range(n):
        var total = UInt128(x.unsafe_offset(i)[]) + UInt128(y.unsafe_offset(i)[]) + UInt128(carry)
        x.unsafe_offset(i)[] = UInt64(total)
        carry = UInt64(total >> 64)
    return carry


@always_inline
def _subtract_limbs(x: Pointer[mut=True, UInt64, _], y: Pointer[mut=False, UInt64, _], n: Int) -> UInt64:
    """x[0, n) -= y[0, n); the borrow out."""
    var borrow = UInt64(0)
    for i in range(n):
        var total = UInt128(x.unsafe_offset(i)[]) - UInt128(y.unsafe_offset(i)[]) - UInt128(borrow)
        x.unsafe_offset(i)[] = UInt64(total)
        borrow = UInt64(total >> 127)
    return borrow


@always_inline
def _add_one(x: Pointer[mut=True, UInt64, _], n: Int, value: UInt64) -> UInt64:
    """x[0, n) += value; the carry out."""
    var carry = value
    for i in range(n):
        if not carry:
            return 0
        var word = x.unsafe_offset(i)[] + carry
        carry = UInt64(word < carry)
        x.unsafe_offset(i)[] = word
    return carry


@always_inline
def _subtract_one(x: Pointer[mut=True, UInt64, _], n: Int, value: UInt64) -> UInt64:
    """x[0, n) -= value; the borrow out."""
    var borrow = value
    for i in range(n):
        if not borrow:
            return 0
        var word = x.unsafe_offset(i)[]
        x.unsafe_offset(i)[] = word - borrow
        borrow = UInt64(word < borrow)
    return borrow


@always_inline
def _add_mul_limbs(x: Pointer[mut=True, UInt64, _], y: Pointer[mut=False, UInt64, _], n: Int, factor: UInt64) -> UInt64:
    """x[0, n) += y[0, n) * factor; the carry out."""
    var carry = UInt64(0)
    for i in range(n):
        var total = UInt128(y.unsafe_offset(i)[]) * UInt128(factor) + UInt128(x.unsafe_offset(i)[]) + UInt128(carry)
        x.unsafe_offset(i)[] = UInt64(total)
        carry = UInt64(total >> 64)
    return carry


@always_inline
def _multiply_limb_into(x: Pointer[mut=True, UInt64, _], y: Pointer[mut=False, UInt64, _], n: Int, factor: UInt64) -> UInt64:
    """x[0, n) = y[0, n) * factor; the carry out."""
    var carry = UInt64(0)
    for i in range(n):
        var total = UInt128(y.unsafe_offset(i)[]) * UInt128(factor) + UInt128(carry)
        x.unsafe_offset(i)[] = UInt64(total)
        carry = UInt64(total >> 64)
    return carry


@always_inline
def _product_rows[n: Int](a: Pointer[mut=False, UInt64, _], b: Pointer[mut=False, UInt64, _], r: Pointer[mut=True, UInt64, _]):
    """r[0, 2n) = a[0, n) * b[0, n) for a small fixed n, unrolled: the first
    row writes, the others add."""
    var carry = UInt64(0)
    var factor = UInt128(b.unsafe_offset(0)[])
    comptime for i in range(n):
        var total = UInt128(a.unsafe_offset(i)[]) * factor + UInt128(carry)
        r.unsafe_offset(i)[] = UInt64(total)
        carry = UInt64(total >> 64)
    r.unsafe_offset(n)[] = carry
    comptime for j in range(1, n):
        carry = 0
        factor = UInt128(b.unsafe_offset(j)[])
        comptime for i in range(n):
            var total = UInt128(a.unsafe_offset(i)[]) * factor + UInt128(r.unsafe_offset(i + j)[]) + UInt128(carry)
            r.unsafe_offset(i + j)[] = UInt64(total)
            carry = UInt64(total >> 64)
        r.unsafe_offset(j + n)[] = carry


@always_inline
def _subtract_mul_limbs(x: Pointer[mut=True, UInt64, _], y: Pointer[mut=False, UInt64, _], n: Int, factor: UInt64) -> UInt64:
    """x[0, n) -= y[0, n) * factor; the borrow out."""
    var borrow = UInt64(0)
    for i in range(n):
        var product = UInt128(y.unsafe_offset(i)[]) * UInt128(factor) + UInt128(borrow)
        var low = UInt64(product)
        var limb = x.unsafe_offset(i)[]
        x.unsafe_offset(i)[] = limb - low
        borrow = UInt64(product >> 64) + UInt64(limb < low)
    return borrow


def _divide_limbs_by_limb(x: Pointer[mut=True, UInt64, _], n: Int, d: UInt64) -> UInt64:
    """x[0, n) = floor(x[0, n) / d) in place for a nonzero d; the remainder.
    The word loop of Integer's division by a word, on the limbs as words."""
    if not n:
        return 0
    var words = x.unsafe_bitcast[UInt32]().unsafe_origin_cast[MutUntrackedOrigin]()
    return _divide_by_limb_into[True](
        Span(unsafe_ptr=words.as_imm().unsafe_origin_cast[ImmutAnyOrigin](), length=2 * n), d, words
    )


@always_inline
def _used(limbs: Pointer[mut=True, UInt64, _], var n: Int) -> Int:
    while n and not limbs.unsafe_offset(n - 1)[]:
        n -= 1
    return n


@always_inline
def _compare(a: Pointer[mut=True, UInt64, _], b: Pointer[mut=True, UInt64, _], n: Int) -> Int:
    for i in range(n - 1, -1, -1):
        var x = a.unsafe_offset(i)[]
        var y = b.unsafe_offset(i)[]
        if x != y:
            return -1 if x < y else 1
    return 0


@always_inline
def _compare_used(a: Pointer[mut=False, UInt64, _], an: Int, b: Pointer[mut=False, UInt64, _], bn: Int) -> Int:
    """The sign of a - b for a[0, an) and b[0, bn) without leading zero limbs."""
    if an != bn:
        return -1 if an < bn else 1
    for i in range(an - 1, -1, -1):
        var x = a.unsafe_offset(i)[]
        var y = b.unsafe_offset(i)[]
        if x != y:
            return -1 if x < y else 1
    return 0


@always_inline
def _multiply_by_limb(x: Pointer[mut=True, UInt64, _], n: Int, factor: UInt64) -> UInt64:
    """x[0, n) *= factor; the carry out."""
    var carry = UInt64(0)
    for i in range(n):
        var total = UInt128(x.unsafe_offset(i)[]) * UInt128(factor) + UInt128(carry)
        x.unsafe_offset(i)[] = UInt64(total)
        carry = UInt64(total >> 64)
    return carry


def _shift_right_limbs(
    source: Pointer[mut=False, UInt64, _], n: Int, shift: Int, target: Pointer[mut=True, UInt64, _], m: Int,
):
    """target[0, m) = source[0, n) >> shift, zero past the source's end."""
    var whole = shift >> 6
    var bits = UInt64(shift & 63)
    for i in range(m):
        var j = i + whole
        var limb = source.unsafe_offset(j)[] if j < n else 0
        if bits:
            var above = source.unsafe_offset(j + 1)[] if j + 1 < n else 0
            limb = (limb >> bits) | (above << (64 - bits))
        target.unsafe_offset(i)[] = limb


def _shift_left_limbs(x: Pointer[mut=True, UInt64, _], n: Int, shift: Int) -> Int:
    """x[0, n) <<= shift in place, x having room for n + shift / 64 + 1
    limbs; the new length without leading zero limbs."""
    if not n:
        return 0
    var whole = shift >> 6
    var bits = UInt64(shift & 63)
    var top = n + whole
    if bits:
        x.unsafe_offset(top)[] = x.unsafe_offset(n - 1)[] >> (64 - bits)
        for i in range(n - 1, 0, -1):
            x.unsafe_offset(i + whole)[] = (x.unsafe_offset(i)[] << bits) | (x.unsafe_offset(i - 1)[] >> (64 - bits))
        x.unsafe_offset(whole)[] = x.unsafe_offset(0)[] << bits
        top += 1
    else:
        for i in range(n - 1, -1, -1):
            x.unsafe_offset(i + whole)[] = x.unsafe_offset(i)[]
    for i in range(whole):
        x.unsafe_offset(i)[] = 0
    return _used(x, top)


comptime _BASECASE_LIMBS = 16
"""Up to this many limbs in the shorter operand, a product takes rows of
word multiply-adds on the limbs themselves, unrolled for equal sizes up to 4
limbs: the word kernels' operand scans,
scratch sizing and dispatch cost more than such a product."""


def _multiply_limbs(
    a: Pointer[mut=False, UInt64, _], an: Int, b: Pointer[mut=False, UInt64, _], bn: Int,
    result: Pointer[mut=True, UInt64, _], square: Bool = False,
) raises -> Int:
    """result[0, an + bn) = a[0, an) * b[0, bn) through the word kernels (a
    square, with square, when a and b are one operand); the length without
    leading zero limbs. result must not overlap either operand."""
    if min(an, bn) <= _BASECASE_LIMBS:
        var r = result.unsafe_origin_cast[MutUntrackedOrigin]()
        if an == bn and an <= 4:
            if an == 1:
                _product_rows[1](a, b, r)
            elif an == 2:
                _product_rows[2](a, b, r)
            elif an == 3:
                _product_rows[3](a, b, r)
            else:
                _product_rows[4](a, b, r)
        else:
            r.unsafe_offset(an)[] = _multiply_limb_into(r, a, an, b.unsafe_offset(0)[])
            for j in range(1, bn):
                r.unsafe_offset(j + an)[] = _add_mul_limbs(r.unsafe_offset(j), a, an, b.unsafe_offset(j)[])
        return _used(r, an + bn)
    var words = _multiply_with_scratch(
        Span(unsafe_ptr=a.unsafe_bitcast[UInt32](), length=2 * an),
        Span(unsafe_ptr=b.unsafe_bitcast[UInt32](), length=2 * bn),
        Span(unsafe_ptr=result.unsafe_bitcast[UInt32](), length=2 * (an + bn)),
        square,
        _product_scratch_size(2 * an, 2 * bn, square),
    )
    return (words + 1) // 2


def _power_limbs(
    base: Pointer[mut=True, UInt64, _], n: Int, exponent: Int,
    result: Pointer[mut=True, UInt64, _], other: Pointer[mut=True, UInt64, _],
) raises -> Int:
    """result = base[0, n) ** exponent for exponent >= 1, by squarings and
    multiplications from the exponent's top bit, alternating between result
    and other so that the last product lands in result; the length without
    leading zero limbs. Both buffers need room for the power's limbs plus n,
    and neither may overlap base, which is left unchanged."""
    var top = 63 - Int(count_leading_zeros(UInt64(exponent)))
    var steps = top
    for bit in range(top):
        steps += (exponent >> bit) & 1
    if not steps:
        for i in range(n):
            result.unsafe_offset(i)[] = base.unsafe_offset(i)[]
        return n
    # Disjoint buffers of the caller's scratch.
    var first = base.unsafe_origin_cast[MutUntrackedOrigin]()
    var into = result.unsafe_origin_cast[MutUntrackedOrigin]()
    var spare = other.unsafe_origin_cast[MutUntrackedOrigin]()
    var into_result = steps % 2 == 1
    var current = first
    var length = n
    for bit in range(top - 1, -1, -1):
        var target = into if into_result else spare
        length = _multiply_limbs(current, length, current, length, target, square=True)
        current = target
        into_result = not into_result
        if (exponent >> bit) & 1:
            target = into if into_result else spare
            length = _multiply_limbs(current, length, first, n, target)
            current = target
            into_result = not into_result
    return length


def _divide_normalized(
    num: Pointer[mut=False, UInt64, _], nn: Int, den: Pointer[mut=False, UInt64, _], dn: Int,
    q: Pointer[mut=True, UInt64, _], work: Pointer[mut=True, UInt64, _],
) -> UInt64:
    """Long division of num[0, nn) by den[0, dn), nn >= dn and den's top limb
    nonzero, through _divide_limbs on copies shifted left until den's top
    bit is set: work holds the divisor's dn limbs, then the dividend's
    nn + 2. Writes the nn + 2 - dn quotient limbs to q and leaves the
    remainder, shifted left by the returned count, in work[dn, 2 dn)."""
    var norm = UInt64(count_leading_zeros(den.unsafe_offset(dn - 1)[]))
    var d = work.unsafe_origin_cast[MutUntrackedOrigin]()
    var u = d.unsafe_offset(dn)
    for i in range(dn):
        var limb = den.unsafe_offset(i)[] << norm
        if norm and i:
            limb |= den.unsafe_offset(i - 1)[] >> (64 - norm)
        d.unsafe_offset(i)[] = limb
    var below = UInt64(0)
    for i in range(nn):
        var limb = num.unsafe_offset(i)[]
        u.unsafe_offset(i)[] = (limb << norm) | below if norm else limb
        below = limb >> (64 - norm) if norm else 0
    u.unsafe_offset(nn)[] = below
    u.unsafe_offset(nn + 1)[] = 0
    _divide_limbs(u, nn + 2, d, dn, q)
    return norm
