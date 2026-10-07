"""Exact lane kernels and adapters for explicitly ordered reductions."""

from ._reduce_exec import _ReductionInputs, _ReductionOperation
from ..integer.value import Integer
from ..rational.value import Rational
from ..rational._batch_storage import _RationalInput
from ._lanes import _Fold, _Lanes, _reduce_lanes
from .value import Batch
from ..float.value import Float
from ..ball.value import Ball
from ..ball.math import maximum as _ball_maximum, minimum as _ball_minimum


@always_inline
def _reduction_checkpoint[T: ImplicitlyCopyable & Deinitable, checked: Bool](
    position: Int, fail: Int,
) raises:
    # The unchecked specialization has no branch in the arithmetic loop.
    comptime if checked:
        if position == fail:
            raise Error(String(
                "Injected ", "integer" if T == Integer else "Rational",
                " reduction failure after logical element ", position,
                "; the destination is unchanged. Retry the operation.",
            ))


@fieldwise_init
struct _ExactReductionInputs(_ReductionInputs, Copyable):
    var left: _RationalInput
    var right: Optional[_RationalInput]

    def length(self) -> Int:
        return self.left.selection.count


struct _ExactReduction[rational: Bool, dot: Bool](_ReductionOperation):
    comptime Inputs = _ExactReductionInputs
    comptime Value: ImplicitlyCopyable & Deinitable = Rational if Self.rational else Integer
    comptime Result = Self.Value
    var fail_after_step: Int
    var steps: Int128

    def __init__(out self, fail_after_step: Int = -1):
        self.fail_after_step = fail_after_step
        self.steps = 0

    def validate(self, inputs: Self.Inputs) raises:
        comptime if Self.dot:
            if inputs.length() != inputs.right.value().selection.count:
                raise Error(
                    String(
                        "Cannot compute dot_sequential for lengths ",
                        inputs.length(),
                        " and ",
                        inputs.right.value().selection.count,
                        (
                            "; use equal-length batches. No"
                            " broadcasting is performed. The destination is"
                            " unchanged."
                        ),
                    )
                )

    def prepare(mut self, inputs: Self.Inputs) raises:
        pass

    def identity(self) raises -> Self.Value:
        comptime if Self.rational:
            return rebind[Self.Value](Rational(0))
        else:
            return rebind[Self.Value](Integer(0))

    def element(mut self, inputs: Self.Inputs, index: Int) raises -> Self.Value:
        if inputs.left.is_zero(index):
            return self.identity()
        comptime if Self.dot:
            if inputs.right.value().is_zero(index):
                return self.identity()
        comptime if Self.rational:
            var result = inputs.left.value(index)
            comptime if Self.dot:
                result *= inputs.right.value().value(index)
            return rebind[Self.Value](result)
        else:
            var result = inputs.left.component(False, index)
            comptime if Self.dot:
                result *= inputs.right.value().component(False, index)
            return rebind[Self.Value](result)

    def combine(
        mut self, left: Self.Value, right: Self.Value, start: Int, end: Int
    ) raises -> Self.Value:
        if self.steps == Int128(self.fail_after_step):
            raise Error(
                "Injected exact reduction failure; retry the operation. The"
                " destination is unchanged."
            )
        self.steps += 1
        comptime if Self.rational:
            return rebind[Self.Value](
                rebind[Rational](left) + rebind[Rational](right)
            )
        else:
            return rebind[Self.Value](
                rebind[Integer](left) + rebind[Integer](right)
            )

    def singleton(mut self, value: Self.Value) raises -> Self.Value:
        return value

    def finish(mut self, var value: Self.Value) raises -> Self.Result:
        return value^


# ----------------------------------------------- kernels for the lane driver
# The built-in exact reductions are kernels of the same driver as lift: each
# folds runs of a lane and combines partial results in order. Exact results
# do not depend on grouping, so every kernel may be split into chunks.


def _empty_extreme[T: ImplicitlyCopyable & Deinitable](maximum: Bool, index: Bool = False) -> Error:
    return Error(String(
        "Cannot compute ", "arg" if index else "", "max" if maximum else "min", " of an empty ",
        "integer" if T == Integer else "Rational" if T == Rational else "Float" if T == Float else "Ball",
        " selection; check len(values) and choose a fallback, or supply at least one value.",
        "" if T == Integer else " The destination is unchanged.",
    ))


@fieldwise_init
struct _IntegerSumFold[checked: Bool](_Fold):
    """Integer sums: values up to 64 bits add natively, wider ones as Integers.

    Int128 cannot overflow here: a run that fits in memory has fewer than
    2**63 terms, each below 2**64 in magnitude.
    """

    comptime Element = Integer
    comptime Partial = Integer
    comptime Result = Integer
    var fail: Int

    def fold(self, lanes: _Lanes[Integer], j: Int, start: Int, end: Int, opening: Bool) raises -> Integer:
        var base = lanes.base(j)
        var small = Int128(0)
        var large = Integer()
        for k in range(start, end):
            ref value = lanes.at(base, k)
            if value._storage.isa[Int64]():
                small += Int128(value._storage[Int64])
            elif value._storage.isa[UInt64]():
                small += Int128(value._storage[UInt64])
            elif value._storage.isa[Integer._Shared]():
                ref stored = value._storage[Integer._Shared].ptr()[]
                var words = stored.words.span()
                if len(words) <= 2:
                    var magnitude = Int128(0)
                    for word in range(len(words)):
                        magnitude |= Int128(words[word]) << Int128(32 * word)
                    small += -magnitude if stored.negative else magnitude
                else:
                    large = large + value
            else:
                large = large + value
            _reduction_checkpoint[Integer, Self.checked](k, self.fail)
        return large + Integer._from_native_product(small)

    def combine(self, left: Integer, right: Integer, lanes: _Lanes[Integer], j: Int, k: Int) raises -> Integer:
        return left + right

    def finish(self, var partial: Integer) raises -> Integer:
        return partial^


@fieldwise_init
struct _RationalSumFold[checked: Bool](_Fold):
    comptime Element = Rational
    comptime Partial = Rational
    comptime Result = Rational
    var fail: Int

    def fold(self, lanes: _Lanes[Rational], j: Int, start: Int, end: Int, opening: Bool) raises -> Rational:
        var base = lanes.base(j)
        var result = Rational()
        for k in range(start, end):
            ref value = lanes.at(base, k)
            if value.__bool__():
                result = result + value if result.__bool__() else value
            _reduction_checkpoint[Rational, Self.checked](k, self.fail)
        return result^

    def combine(self, left: Rational, right: Rational, lanes: _Lanes[Rational], j: Int, k: Int) raises -> Rational:
        return left + right

    def finish(self, var partial: Rational) raises -> Rational:
        return partial^


@fieldwise_init
struct _ProductFold[T: ImplicitlyCopyable & Deinitable, checked: Bool](_Fold):
    """Exact products; a zero in a run ends it without further growth."""

    comptime Element = Self.T
    comptime Partial = Self.T
    comptime Result = Self.T
    var fail: Int

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.T:
        comptime assert Self.T == Integer or Self.T == Rational
        var base = lanes.base(j)
        for k in range(start, end):
            comptime if Self.T == Integer:
                if not rebind[Integer](lanes.at(base, k)).__bool__():
                    return rebind[Self.T](Integer())
            else:
                if not rebind[Rational](lanes.at(base, k)).__bool__():
                    comptime if Self.checked:
                        for index in range(start, end):
                            _reduction_checkpoint[Rational, True](index, self.fail)
                    return rebind[Self.T](Rational())
        var result: Self.T
        comptime if Self.T == Integer:
            result = rebind[Self.T](Integer(1))
        else:
            result = rebind[Self.T](Rational(1))
        for k in range(start, end):
            comptime if Self.T == Integer:
                result = rebind[Self.T](rebind[Integer](result) * rebind[Integer](lanes.at(base, k)))
            else:
                result = rebind[Self.T](rebind[Rational](result) * rebind[Rational](lanes.at(base, k)))
            _reduction_checkpoint[Self.T, Self.checked](k, self.fail)
        return result^

    def combine(self, left: Self.T, right: Self.T, lanes: _Lanes[Self.T], j: Int, k: Int) raises -> Self.T:
        comptime if Self.T == Integer:
            return rebind[Self.T](rebind[Integer](left) * rebind[Integer](right))
        else:
            return rebind[Self.T](rebind[Rational](left) * rebind[Rational](right))

    def finish(self, var partial: Self.T) raises -> Self.T:
        return partial^


@always_inline
def _before[T: ImplicitlyCopyable & Deinitable, maximum: Bool](candidate: T, best: T) raises -> Bool:
    """Whether `candidate` replaces `best`; ties keep the earlier value, and a
    Float NaN, once reached, stays (numpy's `maximum` folded in order)."""
    comptime if T == Integer:
        ref a = rebind[Integer](candidate)
        ref b = rebind[Integer](best)
        return a > b if maximum else a < b
    elif T == Float:
        ref a = rebind[Float](candidate)
        ref b = rebind[Float](best)
        if b.is_nan():
            return False
        if a.is_nan():
            return True
        return a > b if maximum else a < b
    else:
        var order = rebind[Rational](candidate)._compare(rebind[Rational](best))
        return order > 0 if maximum else order < 0


@fieldwise_init
struct _ExtremeFold[T: ImplicitlyCopyable & Deinitable, maximum: Bool, checked: Bool](_Fold):
    """The earliest least (or greatest) value."""

    comptime Element = Self.T
    comptime Partial = Self.T
    comptime Result = Self.T
    var fail: Int

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.T:
        if start == end:
            raise _empty_extreme[Self.T](Self.maximum)
        var base = lanes.base(j)
        var best = lanes.at(base, start)
        _reduction_checkpoint[Self.T, Self.checked](start, self.fail)
        for k in range(start + 1, end):
            ref candidate = lanes.at(base, k)
            if _before[Self.T, Self.maximum](candidate, best):
                best = candidate
            _reduction_checkpoint[Self.T, Self.checked](k, self.fail)
        return best^

    def combine(self, left: Self.T, right: Self.T, lanes: _Lanes[Self.T], j: Int, k: Int) raises -> Self.T:
        return right if _before[Self.T, Self.maximum](right, left) else left

    def finish(self, var partial: Self.T) raises -> Self.T:
        return partial^


@fieldwise_init
struct _BallExtremeFold[maximum: Bool](_Fold):
    """The ball of the greatest (or least) value over all points of a lane's
    balls, folded with `maximum` (or `minimum`) in order."""

    comptime Element = Ball
    comptime Partial = Ball
    comptime Result = Ball

    def fold(self, lanes: _Lanes[Ball], j: Int, start: Int, end: Int, opening: Bool) raises -> Ball:
        if start == end:
            raise _empty_extreme[Ball](Self.maximum)
        var base = lanes.base(j)
        var best = lanes.at(base, start)
        for k in range(start + 1, end):
            best = self.pair(best, lanes.at(base, k))
        return best^

    def combine(self, left: Ball, right: Ball, lanes: _Lanes[Ball], j: Int, k: Int) raises -> Ball:
        return self.pair(left, right)

    def finish(self, var partial: Ball) raises -> Ball:
        return partial^

    def pair(self, left: Ball, right: Ball) raises -> Ball:
        comptime if Self.maximum:
            return _ball_maximum(left, right)
        else:
            return _ball_minimum(left, right)


@fieldwise_init
struct _Indexed[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable, Deinitable):
    """A value and its position in its lane."""

    var value: Self.T
    var index: Int


@fieldwise_init
struct _ArgExtremeFold[T: ImplicitlyCopyable & Deinitable, maximum: Bool](_Fold):
    """The position of the earliest greatest (or least) value of each lane."""

    comptime Element = Self.T
    comptime Partial = _Indexed[Self.T]
    comptime Result = Integer

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> _Indexed[Self.T]:
        if start == end:
            raise _empty_extreme[Self.T](Self.maximum, True)
        var base = lanes.base(j)
        var best = _Indexed[Self.T](lanes.at(base, start), start)
        for k in range(start + 1, end):
            ref candidate = lanes.at(base, k)
            if _before[Self.T, Self.maximum](candidate, best.value):
                best = _Indexed[Self.T](candidate, k)
        return best^

    def combine(self, left: _Indexed[Self.T], right: _Indexed[Self.T], lanes: _Lanes[Self.T], j: Int, k: Int) raises -> _Indexed[Self.T]:
        return right if _before[Self.T, Self.maximum](right.value, left.value) else left

    def finish(self, var partial: _Indexed[Self.T]) raises -> Integer:
        return Integer(partial.index)


@fieldwise_init
struct _IntegerDotFold[checked: Bool](_Fold):
    """Exact Integer dot products: the right operand is a second set of lanes."""

    comptime Element = Integer
    comptime Partial = Integer
    comptime Result = Integer
    var fail: Int
    var right: _Lanes[Integer]

    def fold(self, lanes: _Lanes[Integer], j: Int, start: Int, end: Int, opening: Bool) raises -> Integer:
        var base = lanes.base(j)
        var other = self.right.base(j)
        var result = Integer()
        for k in range(start, end):
            ref a = lanes.at(base, k)
            ref b = self.right.at(other, k)
            if a.__bool__() and b.__bool__():
                result = result + a * b
            _reduction_checkpoint[Integer, Self.checked](k, self.fail)
        return result^

    def combine(self, left: Integer, right: Integer, lanes: _Lanes[Integer], j: Int, k: Int) raises -> Integer:
        return left + right

    def finish(self, var partial: Integer) raises -> Integer:
        return partial^


def _exact_fold[operation: Int, T: ImplicitlyCopyable & Deinitable](
    values: Batch[T], axes: List[Int], every: Bool, keepdims: Bool,
    fail: Int = -1,
) raises -> Batch[T]:
    """Sum (0), product (1), minimum (2) or maximum (3) of an exact batch.

    Failure injection uses the same lane scheduler and arithmetic kernels.
    Only the checked specialization includes element checkpoints.
    """
    if fail < 0:
        return _run_exact_fold[operation, False](values, axes, every, keepdims, fail)
    return _run_exact_fold[operation, True](values, axes, every, keepdims, fail)


def _run_exact_fold[operation: Int, checked: Bool, T: ImplicitlyCopyable & Deinitable](
    values: Batch[T], axes: List[Int], every: Bool, keepdims: Bool, fail: Int,
) raises -> Batch[T]:
    comptime assert T == Integer or T == Rational
    comptime if operation == 0:
        comptime if T == Integer:
            return rebind[Batch[T]](_reduce_lanes(_IntegerSumFold[checked](fail), rebind[Batch[Integer]](values), axes, every, keepdims, True))
        else:
            return rebind[Batch[T]](_reduce_lanes(_RationalSumFold[checked](fail), rebind[Batch[Rational]](values), axes, every, keepdims, True))
    elif operation == 1:
        return _reduce_lanes(_ProductFold[T, checked](fail), values, axes, every, keepdims, True)
    else:
        return _reduce_lanes(_ExtremeFold[T, operation == 3, checked](fail), values, axes, every, keepdims, True)


def _integer_dot(left: Batch[Integer], right: Batch[Integer], fail: Int = -1) raises -> Integer:
    """Check lengths once, then read both operands in logical order."""
    if left.size() != right.size():
        raise Error(String(
            "Cannot compute integer dot product for lengths ", left.size(),
            " and ", right.size(), "; use equal-length batches.",
            " No truncation or broadcasting is performed.",
            " The destination is unchanged.",
        ))
    if fail < 0:
        return _run_integer_dot[False](left, right, fail)
    return _run_integer_dot[True](left, right, fail)


def _run_integer_dot[checked: Bool](left: Batch[Integer], right: Batch[Integer], fail: Int) raises -> Integer:
    var flat_right = right if right.ndim() == 1 else right._flat()
    var flat_left = left if left.ndim() == 1 else left._flat()
    var kernel = _IntegerDotFold[checked](fail, _Lanes[Integer](flat_right, List[Int](), [0]))
    return _reduce_lanes(kernel, flat_left, List[Int](), True, False, True).item()
