"""Float and Complex sums as kernels of the lane driver.

Each run of a lane merges its formats, records which value classes it saw and
adds every finite significand into an exact accumulator; runs combine by
merging, and the finish rounds the exact total once. Results, traps and
error messages are those of a sequential sum.
"""

from std.memory import ArcPointer
from ..float.value import Float
from ..float.context import ArithmeticContext, FloatFormat
from ..float._format import _merge_float_formats
from ..float._accumulator import _ExactAccumulator
from ..float._reductions import _SumFlags, _finish_float_sum
from ..complex.value import Complex
from ..complex.context import _ComplexContextArgument
from ._lanes import _Fold, _Lanes


@fieldwise_init
struct _SumState(ImplicitlyCopyable):
    """Part of a Float sum: formats and classes seen, and the exact total."""

    var flags: _SumFlags
    var total: ArcPointer[_ExactAccumulator]


def _sum_into(
    value: Float, k: Int, context: Optional[ArithmeticContext],
    mut flags: _SumFlags, mut total: _ExactAccumulator,
) raises:
    if context:
        # A context sets the output format; the merge would return it.
        flags.format = context.value().format()
    elif not flags.format or flags.format.value() != value._format:
        try:
            flags.format = _merge_float_formats(flags.format, value.format(), context=context)
        except error:
            raise Error(String("Float sum element ", k, ": ", error))
    var kind = value._kind
    var negative = value.signbit()
    flags.nan |= kind == 3
    flags.positive_infinity |= kind == 2 and not negative
    flags.negative_infinity |= kind == 2 and negative
    flags.positive_zero |= kind == 0 and not negative
    flags.negative_zero |= kind == 0 and negative
    flags.finite |= kind == 1
    if kind == 1:
        var scale = Int128(value._exponent) - Int128(value.precision())
        var small = value._significand._inline_words()
        total.add_magnitude(value._significand._words_span(small), scale, negative)


def _merged(left: _SumState, right: _SumState, context: Optional[ArithmeticContext]) raises -> _SumState:
    var flags = left.flags
    flags.merge(right.flags, context)
    var total = _ExactAccumulator()
    total.merge(left.total[])
    total.merge(right.total[])
    return _SumState(flags, ArcPointer(total^))


def _first_mismatch(
    flags: _SumFlags, lanes: _Lanes[Float], j: Int, start: Int, context: Optional[ArithmeticContext],
) raises:
    """Raise for the first element from `start` whose format conflicts with `flags`."""
    var format = flags.format
    var base = lanes.base(j)
    for k in range(start, lanes.length()):
        try:
            format = _merge_float_formats(format, lanes.at(base, k).format(), context=context)
        except error:
            raise Error(String("Float sum element ", k, ": ", error))


@fieldwise_init
struct _FloatSumFold(_Fold):
    comptime Element = Float
    comptime Partial = _SumState
    comptime Result = Float
    var context: Optional[ArithmeticContext]
    # The batch's format, for an empty sum.
    var default: FloatFormat

    def fold(self, lanes: _Lanes[Float], j: Int, start: Int, end: Int, opening: Bool) raises -> _SumState:
        var flags = _SumFlags()
        var total = _ExactAccumulator()
        var base = lanes.base(j)
        for k in range(start, end):
            _sum_into(lanes.at(base, k), k, self.context, flags, total)
        return _SumState(flags, ArcPointer(total^))

    def combine(self, left: _SumState, right: _SumState, lanes: _Lanes[Float], j: Int, k: Int) raises -> _SumState:
        try:
            return _merged(left, right, self.context)
        except:
            # Name the first element that conflicts, as a sequential sum does.
            _first_mismatch(left.flags, lanes, j, k, self.context)
            raise Error("Float sum formats could not be merged.")

    def finish(self, var partial: _SumState) raises -> Float:
        return Float(_rounded=_finish_float_sum(partial.flags, self.context, self.default, partial.total[]))


@fieldwise_init
struct _ComplexSumState(ImplicitlyCopyable):
    var real: _SumState
    var imag: _SumState


@fieldwise_init
struct _ComplexSumFold(_Fold):
    comptime Element = Complex
    comptime Partial = _ComplexSumState
    comptime Result = Complex
    var context: _ComplexContextArgument
    var real_default: FloatFormat
    var imag_default: FloatFormat

    def _real(self) -> Optional[ArithmeticContext]:
        return self.context.value().real() if self.context else None

    def _imag(self) -> Optional[ArithmeticContext]:
        return self.context.value().imag() if self.context else None

    def fold(self, lanes: _Lanes[Complex], j: Int, start: Int, end: Int, opening: Bool) raises -> _ComplexSumState:
        var base = lanes.base(j)
        # Each component finishes its formats before the other is read, as a
        # sequential sum checks the real component first.
        var real_flags = _SumFlags()
        var real_total = _ExactAccumulator()
        try:
            for k in range(start, end):
                _sum_into(lanes.at(base, k)._real, k, self._real(), real_flags, real_total)
        except error:
            raise Error(String("Complex sum real component: ", error))
        var imag_flags = _SumFlags()
        var imag_total = _ExactAccumulator()
        try:
            for k in range(start, end):
                _sum_into(lanes.at(base, k)._imag, k, self._imag(), imag_flags, imag_total)
        except error:
            raise Error(String("Complex sum imaginary component: ", error))
        return _ComplexSumState(
            _SumState(real_flags, ArcPointer(real_total^)), _SumState(imag_flags, ArcPointer(imag_total^)),
        )

    def combine(
        self, left: _ComplexSumState, right: _ComplexSumState, lanes: _Lanes[Complex], j: Int, k: Int,
    ) raises -> _ComplexSumState:
        var real: _SumState
        var imag: _SumState
        try:
            real = _merged(left.real, right.real, self._real())
        except error:
            raise Error(String("Complex sum real component: ", error))
        try:
            imag = _merged(left.imag, right.imag, self._imag())
        except error:
            raise Error(String("Complex sum imaginary component: ", error))
        return _ComplexSumState(real, imag)

    def finish(self, var partial: _ComplexSumState) raises -> Complex:
        var real: Float
        var imag: Float
        try:
            real = Float(_rounded=_finish_float_sum(partial.real.flags, self._real(), self.real_default, partial.real.total[]))
        except error:
            raise Error(String("Complex sum real component: ", error))
        try:
            imag = Float(_rounded=_finish_float_sum(partial.imag.flags, self._imag(), self.imag_default, partial.imag.total[]))
        except error:
            raise Error(String("Complex sum imaginary component: ", error))
        return Complex(_real=real, _imag=imag)
