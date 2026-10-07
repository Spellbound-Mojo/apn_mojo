"""Non-recursive ASCII numeric-interchange reader, not a general JSON DOM.

Schemas supply their field sets, diagnostics and numeric validation. Helpers
consume tokens only when called, preserving each schema's validation order.
"""

from ._sizes import _checked_count, _checked_sum
from .conversion import _ConversionBudget


struct _JSONReader(Movable):
    var text: String
    var offset: Int
    var element: Int
    var budget: _ConversionBudget
    var diagnostic_prefix: StaticString

    def __init__(
        out self,
        text: String,
        budget: _ConversionBudget,
        diagnostic_prefix: StaticString,
    ) raises:
        _ = _checked_count(text.byte_length(), 1)
        self.text = text
        self.offset = 0
        self.element = -1
        self.budget = budget
        self.diagnostic_prefix = diagnostic_prefix

    def fail(self, reason: String, remedy: String) raises:
        var location = String(self.diagnostic_prefix, self.offset)
        if self.element >= 0:
            location += String(" (element ", self.element, ")")
        raise Error(String(location, ": ", reason, "; ", remedy, "."))

    def peek(self) -> Int:
        if self.offset == self.text.byte_length():
            return -1
        return Int(self.text.as_bytes()[self.offset])

    def whitespace(mut self):
        # JSON excludes vertical tab and form feed, unlike ordinary numeric text.
        while True:
            var byte = self.peek()
            if byte != 32 and byte != 9 and byte != 10 and byte != 13:
                return
            self.offset += 1

    def expect(mut self, byte: Int, remedy: String) raises:
        self.whitespace()
        if self.peek() != byte:
            self.fail("unexpected token or end of input", remedy)
        self.offset += 1

    def string(mut self, number: Bool = False) raises -> String:
        self.expect(34, "use a double-quoted JSON string")
        # Without digit or allocation limits nothing needs counting per byte:
        # a plain string is found by one scan and copied once. Escapes, invalid
        # bytes, a missing quote and limited conversions take the loop below,
        # which reports every error.
        if self.budget.limits._digits < 0 and not self.budget.bounded_allocation():
            var text = self.text.as_bytes()
            var end = self.offset
            while end < len(text):
                var byte = text[end]
                if byte == 34:
                    var start = self.offset
                    self.offset = end + 1
                    # The scan admitted printable ASCII only: valid UTF-8.
                    return String(unsafe_from_utf8=text[start:end])
                if byte == 92 or byte < 32 or byte > 126:
                    break
                end += 1
        var bytes = List[UInt8]()
        while True:
            var byte = self.peek()
            if byte == -1:
                self.fail("unterminated string", "add the closing double quote")
            self.offset += 1
            if byte == 34:
                self.budget.string_allocation(len(bytes), self.element)
                return String(from_utf8=Span(bytes))
            if byte == 92:
                byte = self.peek()
                if byte == -1:
                    self.fail(
                        "incomplete escape", "complete the JSON string escape"
                    )
                self.offset += 1
                if byte == 117:
                    byte = 0
                    for _ in range(4):
                        var digit = self.peek()
                        if 48 <= digit and digit <= 57:
                            digit -= 48
                        elif 65 <= digit and digit <= 70:
                            digit -= 55
                        elif 97 <= digit and digit <= 102:
                            digit -= 87
                        else:
                            self.fail(
                                "invalid Unicode escape",
                                "use four hexadecimal digits after \\u",
                            )
                        self.offset += 1
                        byte = byte * 16 + digit
                    if byte > 127:
                        self.fail(
                            "non-ASCII schema string",
                            (
                                "use ASCII field names, family tags, and"
                                " decimal digits"
                            ),
                        )
                elif byte == 98:
                    byte = 8
                elif byte == 102:
                    byte = 12
                elif byte == 110:
                    byte = 10
                elif byte == 114:
                    byte = 13
                elif byte == 116:
                    byte = 9
                elif byte != 34 and byte != 92 and byte != 47:
                    self.fail(
                        "invalid escape", "use a valid JSON string escape"
                    )
            elif byte < 32 or byte > 126:
                self.fail(
                    "invalid schema string character",
                    "use ASCII text and escape any control characters",
                )
            if number and byte >= 48 and byte <= 57:
                self.budget.digits(1, self.element)
            self.budget.append(bytes, UInt8(byte), self.element)

    def decimal(
        mut self, count_value: Bool = True, *, count_digits: Bool = True
    ) raises -> String:
        if count_value:
            self.budget.values(1, self.element)
        var result = self.string(number=count_digits)
        var bytes = result.as_bytes()
        var start = 0
        if len(bytes) and bytes[0] == 45:
            start = 1
        if start == len(bytes):
            self.fail(
                "empty integer or sign without digits",
                'provide a canonical decimal string such as "0" or "-12"',
            )
        if bytes[start] == 48 and (start != 0 or len(bytes) != 1):
            self.fail(
                "noncanonical zero or leading zero",
                'write zero as "0" and omit leading zeros',
            )
        for i in range(start, len(bytes)):
            if bytes[i] < 48 or bytes[i] > 57:
                self.fail(
                    "noncanonical integer",
                    (
                        "use decimal digits with an optional minus sign,"
                        " without plus signs, spaces, or prefixes"
                    ),
                )
        return result

    def field_name(mut self) raises -> String:
        var key = self.string()
        self.expect(58, "place a colon after the field name")
        return key

    def mark_field(self, mut seen: Int, flag: Int) raises:
        if seen & flag:
            self.fail("duplicate field", "include each field exactly once")
        seen |= flag

    def version_one(mut self, remedy: StaticString) raises:
        self.whitespace()
        if self.peek() != 49:
            self.fail("unsupported version", String(remedy))
        self.offset += 1

    def version(mut self, remedy: StaticString) raises -> Int:
        """Batch records: version 1 for vectors, version 2 with an explicit shape."""
        self.whitespace()
        var byte = self.peek()
        if byte != 49 and byte != 50:
            self.fail("unsupported version", String(remedy))
        self.offset += 1
        return byte - 48

    def shape(mut self) raises -> List[Int]:
        """A version-2 shape: an array of canonical nonnegative decimal strings."""
        var result = List[Int]()
        self.expect(91, "provide shape as an array of decimal strings")
        self.whitespace()
        if self.peek() != 93:
            while True:
                var text = self.decimal(False, count_digits=False)
                if text.as_bytes()[0] == 45 or text.byte_length() > 18:
                    self.fail("invalid dimension", "use nonnegative dimensions within the addressable range")
                result.append(atol(text))
                self.whitespace()
                if self.peek() == 93:
                    break
                self.expect(44, "separate shape dimensions with commas and close the array with ]")
        self.offset += 1
        return result^

    def check_shape_field(self, version: Int, has_shape: Bool) raises:
        if version == 2 and not has_shape:
            self.fail("missing shape", "include shape in version 2 batch records")
        if version == 1 and has_shape:
            self.fail("unknown field", "use shape only in version 2 batch records")

    def family(mut self, expected: StaticString, remedy: StaticString) raises:
        if self.string() != expected:
            self.fail("wrong number family", String(remedy))

    def end_object(mut self, remedy: StaticString) raises -> Bool:
        self.whitespace()
        if self.peek() == 125:
            self.offset += 1
            return True
        self.expect(44, String(remedy))
        return False

    def require_fields(
        self,
        seen: Int,
        required: Int,
        remedy: StaticString,
        *,
        reason: StaticString = "missing required field",
    ) raises:
        if seen != required:
            self.fail(String(reason), String(remedy))

    def finish(mut self) raises:
        self.whitespace()
        if self.peek() != -1:
            self.fail("trailing content", "provide exactly one JSON object")


def _join_records(var head: String, records: List[String]) raises -> String:
    """The record opening, the element records separated by commas, and the
    closing; the result is sized once."""
    var total = _checked_sum(head.byte_length(), 2 + max(0, len(records) - 1))
    for record in records:
        total = _checked_sum(total, record.byte_length())
    _ = _checked_count(total, 1)
    head.reserve_bytes(total)
    for i in range(len(records)):
        if i:
            head += ","
        head += records[i]
    head += "]}"
    return head^


def _batch_json_header(family: StaticString, shape: Optional[List[Int]]) -> String:
    """The record opening: version 1 for vectors, version 2 with a shape otherwise."""
    if not shape:
        return String('{"version":1,"family":"', family, '"')
    var result = String('{"version":2,"family":"', family, '","shape":[')
    var dims = shape.value().copy()
    for i in range(len(dims)):
        if i:
            result += ","
        result.write('"', dims[i], '"')
    return result + "]"
