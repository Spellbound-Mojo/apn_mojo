"""Round-once reductions reading immutable selected Float limbs directly."""

from ._batch_storage import _FloatInput
from ._accumulator import _ExactAccumulator
from ._format import _merge_float_formats
from ._arithmetic import _float_special
from ._rounding import _RoundedBinary
from .context import FloatFormat, ArithmeticContext, RoundingMode
from ..integer.value import Integer
from ..integer.number_theory import lcm
from ._batch_ops import _FloatOperand
from std.memory import ArcPointer
from ..batch._lanes import _Fold, _Lanes, _reduce_positions
from .value import Float


@fieldwise_init
struct _SumFlags(ImplicitlyCopyable):
    """The merged format and value classes of part of a sum."""
    var format: Optional[FloatFormat]
    var nan: Bool
    var positive_infinity: Bool
    var negative_infinity: Bool
    var positive_zero: Bool
    var negative_zero: Bool
    var finite: Bool

    def __init__(out self):
        self = Self(None, False, False, False, False, False, False)

    def merge(mut self, other: Self, context: Optional[ArithmeticContext]) raises:
        if other.format:
            self.format = _merge_float_formats(self.format, other.format, context=context)
        self.nan |= other.nan
        self.positive_infinity |= other.positive_infinity
        self.negative_infinity |= other.negative_infinity
        self.positive_zero |= other.positive_zero
        self.negative_zero |= other.negative_zero
        self.finite |= other.finite


def _finish_float_sum(
    flags: _SumFlags, context: Optional[ArithmeticContext], default: FloatFormat,
    total: _ExactAccumulator,
) raises -> _RoundedBinary:
    """Round an exact Float total once: special classes first, then the sign
    of an exact zero, then the magnitude."""
    var target = context.value() if context else ArithmeticContext(
        _format_of=flags.format.value() if flags.format else default
    )
    if flags.positive_infinity and flags.negative_infinity:
        return _float_special(3, False, 16, target, False)
    if flags.nan:
        return _float_special(3, False, 0, target, False)
    if flags.positive_infinity or flags.negative_infinity:
        return _float_special(2, flags.negative_infinity, 0, target, False)
    var zero_sign = (
        target.rounding() == RoundingMode.toward_negative
    ) if flags.finite or (
        flags.negative_zero and flags.positive_zero
    ) else flags.negative_zero
    return total.magnitude(zero_sign).rounded(target, False)


@fieldwise_init
struct _DotMetadata(ImplicitlyCopyable):
    var kind: Int
    var negative: Bool
    var format: Optional[FloatFormat]


def _dot_metadata(input: _FloatOperand, index: Int) -> _DotMetadata:
    if input.floating:
        ref value = input.floating.value().element(index)
        return _DotMetadata(value._kind, value.signbit(), value._format)
    var sign = input.exact.value().sign(index)
    return _DotMetadata(Int(sign != 0), sign < 0, None)


struct _DotFlags(ImplicitlyCopyable):
    var nan: Bool
    var invalid: Bool
    var positive_infinity: Bool
    var negative_infinity: Bool
    var positive_zero: Bool
    var negative_zero: Bool
    var finite: Bool

    def __init__(out self):
        self.nan = False
        self.invalid = False
        self.positive_infinity = False
        self.negative_infinity = False
        self.positive_zero = False
        self.negative_zero = False
        self.finite = False

    def merge(mut self, other: Self):
        self.nan |= other.nan
        self.invalid |= other.invalid
        self.positive_infinity |= other.positive_infinity
        self.negative_infinity |= other.negative_infinity
        self.positive_zero |= other.positive_zero
        self.negative_zero |= other.negative_zero
        self.finite |= other.finite

    def product(
        mut self, a: _DotMetadata, b: _DotMetadata, negate: Bool = False
    ):
        self.product_kinds(a.kind, a.negative, b.kind, b.negative, negate)

    @always_inline
    def product_kinds(
        mut self, a_kind: Int, a_negative: Bool, b_kind: Int, b_negative: Bool, negate: Bool = False,
    ):
        var negative = (a_negative != b_negative) != negate
        self.nan |= a_kind == 3 or b_kind == 3
        self.invalid |= (a_kind == 0 and b_kind == 2) or (
            a_kind == 2 and b_kind == 0
        )
        if a_kind == 3 or b_kind == 3:
            return
        if a_kind == 2 or b_kind == 2:
            self.positive_infinity |= not negative
            self.negative_infinity |= negative
        elif a_kind == 0 or b_kind == 0:
            self.positive_zero |= not negative
            self.negative_zero |= negative
        else:
            self.finite = True

    def special(self) -> Bool:
        return (
            self.invalid
            or self.nan
            or self.positive_infinity
            or self.negative_infinity
        )


trait _DotSource(ImplicitlyCopyable, Deinitable):
    def length(self) -> Int:
        ...

    def format(
        self, index: Int, context: Optional[ArithmeticContext]
    ) raises -> FloatFormat:
        ...

    def classify(self, index: Int) -> _DotFlags:
        ...

    def denominator(self, index: Int) raises -> Integer:
        ...

    def accumulate(
        self, index: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        ...

    def part(self, start: Int, end: Int, context: Optional[ArithmeticContext]) raises -> _DotPart:
        """Preflight elements [start, end): _dot_part, or a faster equivalent."""
        ...

    def accumulate_run(
        self, start: Int, end: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        """Accumulate elements [start, end) of a plan without special values:
        _accumulate_run, or a faster equivalent."""
        ...


def _accumulate_run[S: _DotSource](
    source: S, start: Int, end: Int, denominator: Integer, mut total: _ExactAccumulator
) raises:
    # Each source skips products with a zero factor itself.
    for index in range(start, end):
        source.accumulate(index, denominator, total)


@fieldwise_init
struct _DotPlan(ImplicitlyCopyable):
    var target: ArithmeticContext
    var flags: _DotFlags
    # The common denominator of finite products (one for Float operands).
    var denominator: Integer


@fieldwise_init
struct _DotPart(ImplicitlyCopyable):
    """Part of a dot product's preflight: formats, classes, denominators."""
    var format: Optional[FloatFormat]
    var flags: _DotFlags
    var denominator: Integer

    def merge(mut self, other: Self, context: Optional[ArithmeticContext]) raises:
        if other.format:
            self.format = _merge_float_formats(self.format, other.format, context=context)
        self.flags.merge(other.flags)
        if not other.denominator._is_one():
            self.denominator = lcm(self.denominator, other.denominator)


def _dot_part[S: _DotSource](
    source: S, start: Int, end: Int, context: Optional[ArithmeticContext],
) raises -> _DotPart:
    var part = _DotPart(None, _DotFlags(), Integer(1))
    for index in range(start, end):
        try:
            var pair_format = source.format(index, context)
            part.format = _merge_float_formats(part.format, pair_format, context=context)
        except error:
            raise Error(String("Float dot element ", index, ": ", error))
        var flags = source.classify(index)
        part.flags.merge(flags)
        if flags.finite:
            var current = source.denominator(index)
            if not current._is_one():
                part.denominator = lcm(part.denominator, current)
    return part^


@fieldwise_init
struct _DotPreflight[S: _DotSource](_Fold):
    """Dot preflight as a lane kernel: formats, classes and the denominator."""

    comptime Element = Float
    comptime Partial = _DotPart
    comptime Result = _DotPart
    var source: Self.S
    var context: Optional[ArithmeticContext]

    def fold(self, lanes: _Lanes[Float], j: Int, start: Int, end: Int, opening: Bool) raises -> _DotPart:
        return self.source.part(start, end, self.context)

    def combine(self, left: _DotPart, right: _DotPart, lanes: _Lanes[Float], j: Int, k: Int) raises -> _DotPart:
        var merged = left
        try:
            merged.merge(right, self.context)
        except:
            # Name the first conflicting element, as the sequential pass does.
            _ = _dot_part(self.source, 0, self.source.length(), self.context)
            raise Error("Float dot formats could not be merged.")
        return merged^

    def finish(self, var partial: _DotPart) raises -> _DotPart:
        return partial^


def _prepare_dot[
    S: _DotSource
](source: S, context: Optional[ArithmeticContext] = None) raises -> _DotPlan:
    # Metadata-only preflight completes before numeric extraction.
    var part = _reduce_positions(_DotPreflight[S](source, context), source.length(), True)
    var format = part.format
    var flags = part.flags
    if not format:
        format = source.format(-1, context)
    var target = context.value() if context else ArithmeticContext(
        _format_of=format.value()
    )
    flags.invalid |= flags.positive_infinity and flags.negative_infinity
    return _DotPlan(target, flags, part.denominator)


@fieldwise_init
struct _DotTotal[S: _DotSource](_Fold):
    """Exact dot accumulation as a lane kernel; runs merge their totals."""

    comptime Element = Float
    comptime Partial = ArcPointer[_ExactAccumulator]
    comptime Result = ArcPointer[_ExactAccumulator]
    var source: Self.S
    var denominator: Integer

    def fold(
        self, lanes: _Lanes[Float], j: Int, start: Int, end: Int, opening: Bool,
    ) raises -> ArcPointer[_ExactAccumulator]:
        var total = _ExactAccumulator()
        self.source.accumulate_run(start, end, self.denominator, total)
        return ArcPointer(total^)

    def combine(
        self, left: ArcPointer[_ExactAccumulator], right: ArcPointer[_ExactAccumulator],
        lanes: _Lanes[Float], j: Int, k: Int,
    ) raises -> ArcPointer[_ExactAccumulator]:
        var total = _ExactAccumulator()
        total.merge(left[])
        total.merge(right[])
        return ArcPointer(total^)

    def finish(self, var partial: ArcPointer[_ExactAccumulator]) raises -> ArcPointer[_ExactAccumulator]:
        return partial^


def _execute_dot[
    S: _DotSource
](
    source: S,
    plan: _DotPlan,
    *,
    fail_after_element: Int = -1,
    fail: Bool = False,
) raises -> _RoundedBinary:
    var flags = plan.flags
    var target = plan.target
    var special = flags.special()
    var denominator = plan.denominator
    var accumulator = _ExactAccumulator()
    var count = source.length()
    if not special and fail_after_element < 0:
        var total = _reduce_positions(_DotTotal[S](source, denominator), count, True)
        accumulator.merge(total[])
        count = 0
    for index in range(count):
        if not special and source.classify(index).finite:
            source.accumulate(index, denominator, accumulator)
        if index == fail_after_element:
            raise Error(
                String(
                    "Injected Float dot failure after logical element ",
                    index,
                    (
                        "; retry the operation. The destination is"
                        " unchanged."
                    ),
                )
            )
    if flags.invalid:
        return _float_special(3, False, 16, target, fail)
    if flags.nan:
        return _float_special(3, False, 0, target, fail)
    if flags.positive_infinity or flags.negative_infinity:
        return _float_special(2, flags.negative_infinity, 0, target, fail)
    var zero_sign = (
        target.rounding() == RoundingMode.toward_negative
    ) if flags.finite or (
        flags.positive_zero and flags.negative_zero
    ) else flags.negative_zero
    var magnitude = accumulator.magnitude(zero_sign)
    return magnitude.rounded(target, fail, denominator=denominator)


def _dot_denominator(
    left: _FloatOperand, right: _FloatOperand, index: Int
) raises -> Integer:
    # Approximate dot pairs have at most one nonunit exact denominator.
    if left.exact:
        return left.exact.value().component(True, index)
    if right.exact:
        return right.exact.value().component(True, index)
    return Integer(1)


@always_inline
def _add_float_product(a: Float, b: Float, negate: Bool, mut total: _ExactAccumulator) raises:
    """Add the exact product of two finite Floats (others are skipped); short
    significands multiply natively."""
    if a._kind != 1 or b._kind != 1:
        return
    var scale = (Int128(a._exponent) - Int128(a.precision())) + (Int128(b._exponent) - Int128(b.precision()))
    var negative = (a.signbit() != b.signbit()) != negate
    if a._significand._storage.isa[Int64]() and b._significand._storage.isa[Int64]():
        var product = UInt128(UInt64(a._significand._storage[Int64])) * UInt128(UInt64(b._significand._storage[Int64]))
        var words = Array[UInt32, 4](fill=0)
        for i in range(4):
            words[i] = UInt32((product >> UInt128(32 * i)) & 0xFFFFFFFF)
        var used = 4
        while used and not words[used - 1]:
            used -= 1
        total.add_magnitude(Span(words)[:used], scale, negative)
    else:
        var product = a._significand * b._significand
        var small = product._inline_words()
        total.add_magnitude(product._words_span(small), scale, negative)


def _accumulate_dot_product(
    left: _FloatOperand,
    right: _FloatOperand,
    index: Int,
    denominator: Integer,
    mut total: _ExactAccumulator,
    negate: Bool = False,
) raises:
    if left.floating and right.floating and denominator._is_one():
        # Float pairs: borrow both stored values.
        _add_float_product(
            left.floating.value().element(index), right.floating.value().element(index), negate, total
        )
        return
    if (
        _dot_metadata(left, index).kind != 1
        or _dot_metadata(right, index).kind != 1
    ):
        return
    var a = left.value(index).value
    var b = right.value(index).value
    var term = a.numerator * b.numerator
    var divisor = a.denominator * b.denominator
    if denominator != divisor:
        term *= denominator // divisor
    if (a.negative != b.negative) != negate:
        term = -term
    total.add(term, a.scale + b.scale)


@fieldwise_init
struct _FloatDotSource(_DotSource):
    var left: _FloatOperand
    var right: _FloatOperand

    def length(self) -> Int:
        return self.left.length

    def format(
        self, index: Int, context: Optional[ArithmeticContext]
    ) raises -> FloatFormat:
        return _merge_float_formats(
            self.left.format() if index
            < 0 else _dot_metadata(self.left, index).format,
            self.right.format() if index
            < 0 else _dot_metadata(self.right, index).format,
            context=context,
        )

    def classify(self, index: Int) -> _DotFlags:
        var flags = _DotFlags()
        flags.product(
            _dot_metadata(self.left, index), _dot_metadata(self.right, index)
        )
        return flags

    def denominator(self, index: Int) raises -> Integer:
        return _dot_denominator(self.left, self.right, index)

    def accumulate(
        self, index: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        _accumulate_dot_product(
            self.left, self.right, index, denominator, total
        )

    def _runs(self) -> Optional[Tuple[Int, Int]]:
        """Storage steps of both operands when each is one run of stored Floats."""
        if not self.left.floating or not self.right.floating or self.left.scalar or self.right.scalar:
            return None
        if self.left.broadcast or self.right.broadcast:
            return None
        var a = self.left.floating.value().run_step()
        var b = self.right.floating.value().run_step()
        if not a or not b:
            return None
        return (a.value(), b.value())

    def part(self, start: Int, end: Int, context: Optional[ArithmeticContext]) raises -> _DotPart:
        var steps = self._runs()
        if not steps or start >= end:
            return _dot_part(self, start, end, context)
        # Float runs: read both stored values in place; formats merge only
        # when one differs from the formats merged so far.
        var a_step, b_step = steps.value()
        var a_first = self.left.floating.value().run_start()
        var b_first = self.right.floating.value().run_start()
        var part = _DotPart(None, _DotFlags(), Integer(1))
        if context:
            part.format = context.value().format()
        for index in range(start, end):
            ref a = a_first.unsafe_offset(index * a_step)[]
            ref b = b_first.unsafe_offset(index * b_step)[]
            if not context and not (
                part.format and a._format == part.format.value() and b._format == part.format.value()
            ):
                try:
                    var pair_format = _merge_float_formats(a.format(), b.format(), context=context)
                    part.format = _merge_float_formats(part.format, pair_format, context=context)
                except error:
                    raise Error(String("Float dot element ", index, ": ", error))
            part.flags.product_kinds(a._kind, a.signbit(), b._kind, b.signbit())
        return part^

    def accumulate_run(
        self, start: Int, end: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        var steps = self._runs()
        if not steps or start >= end or not denominator._is_one():
            _accumulate_run(self, start, end, denominator, total)
            return
        var a_step, b_step = steps.value()
        var a_first = self.left.floating.value().run_start()
        var b_first = self.right.floating.value().run_start()
        for index in range(start, end):
            _add_float_product(
                a_first.unsafe_offset(index * a_step)[], b_first.unsafe_offset(index * b_step)[], False, total
            )


def _dot_shape(left: Int, right: Int, name: String) raises:
    if left != right:
        raise Error(
            String(
                "Cannot compute ",
                name,
                " for lengths ",
                left,
                " and ",
                right,
                "; use equal-length batches.",
                " No truncation or broadcasting is performed.",
                " The destination is unchanged.",
            )
        )


def _dot_float(
    left: _FloatOperand,
    right: _FloatOperand,
    context: Optional[ArithmeticContext] = None,
    *,
    fail_after_element: Int = -1,
    fail: Bool = False,
) raises -> _RoundedBinary:
    _dot_shape(left.length, right.length, "Float dot product")
    var source = _FloatDotSource(left, right)
    return _execute_dot(
        source,
        _prepare_dot(source, context),
        fail_after_element=fail_after_element,
        fail=fail,
    )
