"""ASCII helpers for numeric text, independent of numeric representations."""

from ._sizes import _checked_count


def _text_digit(byte: UInt8) -> Int:
    var code = Int(byte)
    if 48 <= code and code <= 57:
        return code - 48
    if 65 <= code and code <= 90:
        return code - 55
    if 97 <= code and code <= 122:
        return code - 87
    return -1


def _text_space(byte: UInt8) -> Bool:
    return byte == 32 or (byte >= 9 and byte <= 13)


def _text_bounds(
    text: String, allow_whitespace: Bool
) raises -> Tuple[Int, Int]:
    var end = _checked_count(text.byte_length(), 1)
    var start = 0
    if allow_whitespace:
        var bytes = text.as_bytes()
        while start < end and _text_space(bytes[start]):
            start += 1
        while end > start and _text_space(bytes[end - 1]):
            end -= 1
    return start, end
