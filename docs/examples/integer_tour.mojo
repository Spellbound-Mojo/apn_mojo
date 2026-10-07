"""A tour of Integer: text, JSON, division conventions, functions and batch updates."""

from apn_mojo import (
    Batch,
    Integer,
    div_rem_floor,
    div_rem_trunc,
    div_rem_euclid,
    gcd,
    lcm,
    isqrt,
    ConversionLimits,
    sum,
    prod,
    min,
    max,
    dot,
    axpy,
    integer,
    vmap,
)


def main() raises:
    var value = Integer("340282366920938463463374607431768211455")
    var saved = value
    value += value
    print("original:", saved)
    print("doubled: ", value)
    var restored = Integer.from_json(value.to_json())
    var labels = Dict[Integer, String]()
    labels[value] = "exact value"
    print("restored key:", labels[restored])
    var limits = ConversionLimits(
        max_input_bytes=4096,
        max_output_bytes=4096,
        max_digits=1000,
        max_values=100,
    )
    print(
        "bounded round trip:",
        Integer.from_json(saved.to_json(limits=limits), limits=limits),
    )
    var code = Integer(
        " -0xFF_FF ", base=0, allow_whitespace=True, allow_underscores=True
    )
    print("hexadecimal:", code.to_string(16, prefix=True, uppercase=True))

    var quotient, remainder = div_rem_floor(-7, 3)
    print("floor:", quotient, remainder)
    quotient, remainder = div_rem_trunc(-7, 3)
    print("truncating:", quotient, remainder)
    quotient, remainder = div_rem_euclid(-7, -3)
    print("Euclidean:", quotient, remainder)
    print("native:", Int(quotient))
    print("unsigned:", Integer(UInt64.MAX).to_native_exact[DType.uint64]())
    print("power:", Integer(3) ** 100)
    print("gcd, lcm, square root:", gcd(-36, 48), lcm(-36, 48), isqrt(145))
    print("signed bits:", Integer(-7) >> 1, ~Integer(7))
    print(
        "sign and magnitude bits:", value.sign(), value.magnitude_bit_length()
    )
    try:
        value //= 0
    except error:
        print(error)

    var values: List[Integer] = [saved, value, Integer(-7)]
    var batch = Batch[Integer](values)
    print(
        "negation and absolute value:",
        (-batch)[2],
        vmap[integer.abs]()(batch[::-1])[0],
    )
    var results = (batch + 3) * batch
    for result in results:
        print(result)
    var sequence = Batch[Integer].from_iterable(range(5))
    print("sum and product:", sum(sequence), prod(sequence[1:]))
    print("minimum and maximum:", min(sequence), max(sequence[::-1]))
    print("dot product:", dot(sequence, sequence[::-1]))
    print("scaled addition:", axpy(2, sequence, sequence[::-1]).to_json())
    for value in sequence:
        print("from range:", value)

    var batch_q, batch_r = vmap[div_rem_euclid]()(batch, -7)
    print("batch division:", batch_q[0], batch_r[0])
    var preserved = batch[:]
    var divisors = Batch[Integer].from_native([3, 5, 0])
    try:
        batch //= divisors
    except error:
        print(error)
    print("unchanged after error:", (batch == preserved).all())
    batch //= 3
    batch %= -7
    batch **= 2
    print(
        "batch square root and gcd:",
        vmap[isqrt]()(batch)[0],
        vmap[gcd]()(batch, 12)[0],
    )
    print(
        "batch sign and magnitude bits:",
        batch.sign()[0],
        batch.magnitude_bit_length()[0],
    )

    var positive = batch > 0
    print("positive values:", positive.count())
    print("all positive:", positive.all())
    var snapshot = batch[:]
    batch[1:] = batch[:-1]
    batch[::2] = -1
    print("updated:", batch[0], batch[1], batch[2])
    print("original snapshot:", snapshot[0], snapshot[1], snapshot[2])
    var negative = batch < 0
    var selected = batch[negative]
    batch[negative] = 0
    print("selected snapshot:", selected.to_json())
    print("negative values replaced:", batch.to_json())
    var document = snapshot[::-1].to_json()
    var restored_batch = Batch[Integer].from_json(document)
    print("saved reversed snapshot:", document)
    print("restored first value:", restored_batch[0])

    try:
        _ = Integer("12x")
    except error:
        print(error)
