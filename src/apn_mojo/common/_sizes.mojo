"""Private checked size arithmetic; physical allocator OOM remains fatal."""


def _checked_count(count: Int, item_bytes: Int) raises -> Int:
    # Leave room for the standard containers' geometric growth and metadata.
    if count < 0 or count > (Int.MAX // 2 - 64) // item_bytes:
        raise Error(
            "Cannot allocate numeric storage: size exceeds the addressable"
            " range; use a smaller value or batch."
        )
    return count


def _checked_sum(left: Int, right: Int) raises -> Int:
    if left < 0 or right < 0 or left > Int.MAX - right:
        raise Error(
            "Cannot grow numeric storage: size exceeds the addressable range;"
            " use a smaller value or batch."
        )
    return left + right
