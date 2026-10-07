"""Optional per-call conversion counts and cumulative allocation quotas."""

from std.sys import size_of

from ._sizes import _checked_sum


def _limit_setting(value: Optional[Int], name: StaticString) raises -> Int:
    if not value:
        return -1
    if value.value() < 0:
        raise Error(
            String(
                "Cannot set ",
                name,
                " to ",
                value.value(),
                "; use a nonnegative count or None for no limit.",
            )
        )
    return value.value()


struct ConversionLimits(ImplicitlyCopyable):
    """Limits for one text or JSON conversion.

    Pass it as `limits=` to a parser, `to_string`, `to_json` or `from_json`. Each
    limit is inclusive, and `None` means unlimited; zero is a real limit. Counters
    start fresh on every call and are shared by everything the call does, so a
    batch's elements share one allowance. Limits are never stored in the result.

    Limitations:
        Limits bound one conversion only, not later arithmetic, retained values,
        memory use or time.
    """

    var _input: Int
    var _output: Int
    var _digits: Int
    var _values: Int
    var _allocated: Int

    def __init__(
        out self,
        *,
        max_input_bytes: Optional[Int] = None,
        max_output_bytes: Optional[Int] = None,
        max_digits: Optional[Int] = None,
        max_values: Optional[Int] = None,
        max_allocated_bytes: Optional[Int] = None,
    ) raises:
        """Limits for a conversion; omit a setting for no limit.

        Args:
            max_input_bytes: The whole input string, before trimming or unescaping.
            max_output_bytes: The whole returned string, including JSON framing.
            max_digits: Numeric digits across the whole call.
            max_values: One for a scalar, the logical length for a batch.
            max_allocated_bytes: Bytes newly requested during the call, cumulatively;
                freed temporaries are not refunded.

        Raises:
            When a setting is negative.
        """
        self._input = _limit_setting(max_input_bytes, "max_input_bytes")
        self._output = _limit_setting(max_output_bytes, "max_output_bytes")
        self._digits = _limit_setting(max_digits, "max_digits")
        self._values = _limit_setting(max_values, "max_values")
        self._allocated = _limit_setting(
            max_allocated_bytes, "max_allocated_bytes"
        )


def _conversion_limit(
    name: StaticString, limit: Int, requested: Int, element: Int
) raises:
    if limit >= 0 and requested > limit:
        var location = String()
        if element >= 0:
            location = String(" at logical element ", element, " (zero-based)")
        raise Error(
            String(
                "Cannot convert numeric text/JSON",
                location,
                ": ",
                name,
                " limit ",
                limit,
                " would be exceeded (needs at least ",
                requested,
                "); use a smaller input/value or raise ",
                name,
                " in ConversionLimits. The destination is unchanged.",
            )
        )


struct _ConversionBudget(ImplicitlyCopyable):
    var limits: ConversionLimits
    var digits_used: Int
    var output_used: Int
    var values_used: Int
    var allocated_used: Int

    def __init__(out self, limits: Optional[ConversionLimits]) raises:
        self.limits = limits.value() if limits else ConversionLimits()
        self.digits_used = 0
        self.output_used = 0
        self.values_used = 0
        self.allocated_used = 0

    def bounded_allocation(self) -> Bool:
        return self.limits._allocated >= 0

    def allocate(
        mut self, count: Int, width: Int = 1, element: Int = -1
    ) raises:
        if not self.bounded_allocation():
            return
        if count < 0 or width <= 0 or count > Int.MAX // width:
            raise Error(
                "Cannot convert numeric text/JSON: allocation byte count"
                " exceeds the addressable range; use a smaller input/value."
                " The destination is unchanged."
            )
        var requested = _checked_sum(self.allocated_used, count * width)
        _conversion_limit(
            "max_allocated_bytes", self.limits._allocated, requested, element
        )
        self.allocated_used = requested

    def append[
        T: ImplicitlyCopyable & Deinitable
    ](mut self, mut values: List[T], var value: T, element: Int = -1) raises:
        if self.bounded_allocation() and len(values) == values.capacity():
            var capacity = max(
                1, _checked_sum(values.capacity(), values.capacity())
            )
            self.allocate(capacity, size_of[T](), element)
        values.append(value^)

    def string_allocation(mut self, length: Int, element: Int = -1) raises:
        if self.bounded_allocation() and length > String.INLINE_CAPACITY:
            # The pinned String rounds capacity to eight bytes and prepends
            # its reference count. String copies only share this allocation.
            var capacity = _checked_sum(length, 7) // 8 * 8
            self.allocate(
                _checked_sum(capacity, String.REF_COUNT_SIZE), 1, element
            )

    def grow_string(
        mut self, mut value: String, extra: Int, element: Int = -1
    ) raises:
        if not self.bounded_allocation():
            return
        var required = _checked_sum(value.byte_length(), extra)
        if required > value.capacity_bytes():
            var capacity = max(
                required,
                _checked_sum(value.capacity_bytes(), value.capacity_bytes()),
            )
            self.string_allocation(capacity, element)
            value.reserve_bytes(required)

    def input(self, count: Int) raises:
        _conversion_limit("max_input_bytes", self.limits._input, count, -1)

    def values(mut self, count: Int, element: Int = -1) raises:
        if self.limits._values >= 0:
            var requested = _checked_sum(self.values_used, count)
            _conversion_limit(
                "max_values", self.limits._values, requested, element
            )
            self.values_used = requested

    def digits(mut self, count: Int, element: Int = -1) raises:
        if self.limits._digits >= 0:
            var requested = _checked_sum(self.digits_used, count)
            _conversion_limit(
                "max_digits", self.limits._digits, requested, element
            )
            self.digits_used = requested

    def output(mut self, count: Int, element: Int = -1) raises:
        if self.limits._output >= 0:
            var requested = _checked_sum(self.output_used, count)
            _conversion_limit(
                "max_output_bytes", self.limits._output, requested, element
            )
            self.output_used = requested

    def output_digit(mut self, element: Int = -1) raises:
        self.digits(1, element)
        self.output(1, element)

    def bounded_output(self) -> Bool:
        return (
            self.limits._digits >= 0
            or self.limits._output >= 0
            or self.bounded_allocation()
        )

    def preflight(
        self, bits: Int, base: Int, overhead: Int, element: Int = -1
    ) raises:
        # A lower bound can reject early, but never rejects a result that fits.
        var width = 1
        while (1 << width) < base:
            width += 1
        var minimum = (bits - 1) // width + 1 if bits else 1
        if self.limits._digits >= 0:
            _conversion_limit(
                "max_digits",
                self.limits._digits,
                _checked_sum(self.digits_used, minimum),
                element,
            )
        if self.limits._output >= 0:
            _conversion_limit(
                "max_output_bytes",
                self.limits._output,
                _checked_sum(self.output_used, _checked_sum(minimum, overhead)),
                element,
            )
