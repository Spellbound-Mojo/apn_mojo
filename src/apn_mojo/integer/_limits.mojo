"""Shared checked bounds and actionable errors for powers and shifts."""

# Keep bit lengths native and leave spare rows within the packed allocation bound.
comptime _MAX_RESULT_BITS = (Int.MAX // 64 - 4) * 32


def _count_error(
    operation: Int, index: Int = -1, too_large: Bool = False
) raises:
    var message = String(
        "Cannot ",
        "raise Integer to a power" if operation == 6 else "shift Integer",
    )
    if index >= 0:
        message += String(" at logical element ", index, " (zero-based)")
    if too_large:
        message += (
            ": estimated result exceeds supported addressable storage; use a"
            " smaller value or count."
        )
    else:
        message += String(
            ": negative ",
            "exponent" if operation == 6 else "shift count",
            "; use a nonnegative ",
            "exponent" if operation == 6 else "shift count",
            ".",
        )
    raise Error(message + " The destination is unchanged.")
