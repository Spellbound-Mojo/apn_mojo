"""Exact totals and weighted sums."""

from apn_mojo import (
    Batch,
    Integer,
    Rational,
    sum,
    prod,
    min,
    max,
    dot,
)


def main() raises:
    var portions = Batch[Rational](
        [Rational(1, 2), Rational(3, 4), Rational(2, 5)]
    )
    var counts = Batch[Integer].from_native([4, 2, 5])
    var total = dot(portions, counts)
    print("Total:", total)
    print("Weighted mean:", total / sum(counts))
    print("Sum:", sum(portions))
    print("Product:", prod(portions))
    print("Range:", min(portions), max(portions))
    print("Selected sum:", sum(portions[counts > 2]))
    print("Reversed pairs:", dot(counts[::-1], portions[::-1]))
    var empty = Batch[Rational]()
    print("Empty identities:", sum(empty), prod(empty))
