"""NumPy-style reductions, accumulations and outer products of a binary function.

`lift[f]` turns a scalar binary function, named or lambda, into a batch
function with the methods of a NumPy ufunc made by `np.frompyfunc`:
`reduce`, `accumulate`, `outer` and a broadcasting call. Lifting also states
what a generic fold cannot know about `f`:

- `identity` is the result of reducing an empty selection. Without one, an
  empty reduction raises, as NumPy does for functions without an identity.
- `associative=True` promises f(f(a, b), c) == f(a, f(b, c)), so one long
  reduction may be split into chunks folded on separate threads and combined
  in order. Operand order is kept, so commutativity is not required. Without
  it, every result is a strict left fold in row-major order; separate results
  still run in parallel.

Internally every lifted function takes a `context` keyword: functions that take
one (such as `apn_mojo.float.add`) receive the caller's context on each call,
and plain functions are wrapped by `_no_context`.
"""

from std.memory import ArcPointer
from std.sys import size_of
from ..integer.value import Integer
from ..float.value import Float
from ..complex.value import Complex
from ._lanes import _Fold, _Lanes, _ReduceAxes, _failure, _reduce_lanes, _start, _unravel
from ._layout import _FlatLayout, _axis, _broadcast_shapes
from ._parallel import _Slots, _parallel_map, _parallel_ready, _ElementKernel, _ElementErrors, _execute_values
from ._tensor import _Tensor
from ..common._traits import _BatchElement, _MapArgument
from ..common._sizes import _checked_count
from .value import Batch, _BatchShape
from .mask import Mask
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from ..float._format import _merge_float_formats
from ..complex.context import _ComplexContextArgument, ComplexContext
from ..float._arithmetic import _FloatArgument
from ..complex._input import _ComplexArgument
from .mapping import _scalar_argument, _FlatInput, _Element
from ..ball.value import Ball, _BallArgument
from ..ball.context import BallContext
from ..ball._arithmetic import _rounded_ball, _working as _ball_precision

# Functions returning Bool give Masks; Bool functions fold Masks.
comptime _LiftOutput[R: ImplicitlyCopyable & Deinitable] = Mask if R == Bool else Batch[R]
comptime _LiftContext[C: AnyType] = C == Optional[ArithmeticContext] or C == _ComplexContextArgument or C == Optional[BallContext]
comptime _CallContext[C: AnyType] = Optional[BallContext] if C == Optional[BallContext] else _ComplexContextArgument
comptime _Numeric[T: AnyType] = conforms_to(T, _BatchElement)
comptime _Closed[T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable] = _LiftElement[T] == R and _LiftElement[U] == R


def _no_context[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U) raises thin -> R,
](x: T, y: U, /, *, context: _ComplexContextArgument) raises -> R:
    """A plain function in the lifted form; it has no use for the context."""
    return function(x, y)


# Fold closure is a property of the callback family, not of its input batch.
# Same-family fold kernels adapt their borrowed elements at the scalar call.
comptime _LiftElement[T: ImplicitlyCopyable & Deinitable] = (
    Float if T == _FloatArgument else Complex if T == _ComplexArgument else Ball if T == _BallArgument else T
)


def _adapted[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R,
](x: _LiftElement[T], y: _LiftElement[U], /, *, context: C) raises -> R:
    # Concrete parameters already match: preserve the lane's borrow. Only
    # adapter parameters need owning argument records at this boundary.
    comptime if T == _LiftElement[T]:
        comptime if U == _LiftElement[U]:
            return function(rebind[T](x), rebind[U](y), context=context)
        else:
            return function(rebind[T](x), _scalar_argument[U](y), context=context)
    else:
        comptime if U == _LiftElement[U]:
            return function(_scalar_argument[T](x), rebind[U](y), context=context)
        else:
            return function(_scalar_argument[T](x), _scalar_argument[U](y), context=context)


def _uniform(context: _ComplexContextArgument) raises -> ArithmeticContext:
    """The context of real results: an ArithmeticContext, not a ComplexContext."""
    if not context._uniform:
        raise Error(
            "A ComplexContext rounds Complex results; pass an ArithmeticContext"
            " for Float results."
        )
    return context.value().real()


def _step_context[C: ImplicitlyCopyable & Deinitable & Defaultable](context: _CallContext[C]) raises -> C:
    """The context each call of a context-taking function receives."""
    comptime if C == _ComplexContextArgument or C == Optional[BallContext]:
        return rebind[C](context)
    else:
        comptime assert C == Optional[ArithmeticContext], "lift passes ArithmeticContext or ComplexContext."
        if not context:
            return C()
        return rebind[C](Optional[ArithmeticContext](_uniform(rebind[_ComplexContextArgument](context))))


# ------------------------------------------------------------------- reduce


@fieldwise_init
struct _LiftFold[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    function: def(T, T, /, *, context: C) raises thin -> T,
](_Fold):
    """A lifted function as a reduction kernel: a left fold with `function`."""

    comptime Element = Self.T
    comptime Partial = Self.T
    comptime Result = Self.T
    var initial: Optional[Self.T]
    var identity: Optional[Self.T]
    var context: Self.C

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.T:
        var base = lanes.base(j)
        var k = start
        var result: Self.T
        if opening and self.initial:
            result = self.initial.value()
        elif start < end:
            result = lanes.at(base, start)
            k = start + 1
        elif self.identity:
            return self.identity.value()
        else:
            raise Error(
                "Cannot reduce an empty selection: the lifted function has no"
                " identity. Lift it with identity= or pass initial=."
            )
        try:
            if lanes.linear:
                # The common case: one pointer and a fixed step.
                var step = lanes.step
                var element = lanes.run(base).unsafe_offset(k * step)
                while k < end:
                    result = Self.function(result, element[], context=self.context)
                    element = element.unsafe_offset(step)
                    k += 1
            else:
                while k < end:
                    result = Self.function(result, lanes.at(base, k), context=self.context)
                    k += 1
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return result^

    def combine(self, left: Self.T, right: Self.T, lanes: _Lanes[Self.T], j: Int, k: Int) raises -> Self.T:
        try:
            return Self.function(left, right, context=self.context)
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)

    def finish(self, var partial: Self.T) raises -> Self.T:
        return partial^


@fieldwise_init
struct _UpdateFold[T: ImplicitlyCopyable & Deinitable, //, update: def(mut T, T) raises thin -> None](_Fold):
    """An in-place update as a reduction kernel: the accumulator is updated
    with each element, reusing its storage where the update can."""

    comptime Element = Self.T
    comptime Partial = Self.T
    comptime Result = Self.T
    var initial: Optional[Self.T]
    var identity: Optional[Self.T]

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.T:
        var base = lanes.base(j)
        var k = start
        var result: Self.T
        if opening and self.initial:
            result = self.initial.value()
        elif start < end:
            result = lanes.at(base, start)
            k = start + 1
        elif self.identity:
            return self.identity.value()
        else:
            raise Error(
                "Cannot reduce an empty selection: the lifted function has no"
                " identity. Lift it with identity= or pass initial=."
            )
        try:
            while k < end:
                Self.update(result, lanes.at(base, k))
                k += 1
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return result^

    def combine(self, left: Self.T, right: Self.T, lanes: _Lanes[Self.T], j: Int, k: Int) raises -> Self.T:
        var result = left
        try:
            Self.update(result, right)
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return result^

    def finish(self, var partial: Self.T) raises -> Self.T:
        return partial^


def _updated[T: ImplicitlyCopyable & Deinitable, //, update: def(mut T, T) raises thin -> None](x: T, y: T) raises -> T:
    """An in-place update as a function of two values."""
    var result = x
    update(result, y)
    return result^


# ------------------------------------------------------------ exact folds
# Float and Complex arithmetic rounds every result. An exact fold works on
# exact working values instead (arithmetic keeps every bit, and raises if a
# result is not a finite binary fraction) and rounds once, at the end, to the
# merged formats of the inputs it read.

comptime _Rounded[T: ImplicitlyCopyable & Deinitable] = T == Float or T == Complex


@fieldwise_init
struct _Formats(ImplicitlyCopyable):
    """Merged input formats: the real part's and, for Complex, the imaginary part's."""

    var real: Optional[FloatFormat]
    var imag: Optional[FloatFormat]

    def __init__(out self):
        self.real = None
        self.imag = None

    def take[T: ImplicitlyCopyable & Deinitable](mut self, value: T) raises:
        comptime if T == Float:
            self.real = _merge_float_formats(self.real, rebind[Float](value).format())
        else:
            ref number = rebind[Complex](value)
            self.real = _merge_float_formats(self.real, number._real.format())
            self.imag = _merge_float_formats(self.imag, number._imag.format())

    def take(mut self, other: Self) raises:
        if other.real:
            self.real = _merge_float_formats(self.real, other.real)
        if other.imag:
            self.imag = _merge_float_formats(self.imag, other.imag)


@always_inline
def _working[T: ImplicitlyCopyable & Deinitable](value: T) -> T:
    """The exact working form of a Float or Complex value."""
    comptime if T == Float:
        return rebind[T](rebind[Float](value)._exactly())
    else:
        ref number = rebind[Complex](value)
        return rebind[T](Complex(_real=number._real._exactly(), _imag=number._imag._exactly()))


def _rounded_once[T: ImplicitlyCopyable & Deinitable](
    value: T, formats: _Formats, context: _ComplexContextArgument,
) raises -> T:
    """Round an exact working value once: with the context when one is given,
    else to the merged input formats."""
    comptime if T == Float:
        var target = _uniform(context) if context else ArithmeticContext(format=(formats.real.value() if formats.real else _merge_float_formats(None, None)))
        return rebind[T](Float(rebind[Float](value), context=target))
    else:
        ref number = rebind[Complex](value)
        if context:
            return rebind[T](Complex(number, context=context))
        return rebind[T](Complex(
            _real=Float(number._real, context=ArithmeticContext(format=(formats.real.value() if formats.real else _merge_float_formats(None, None)))),
            _imag=Float(number._imag, context=ArithmeticContext(format=(formats.imag.value() if formats.imag else _merge_float_formats(None, None)))),
        ))


@fieldwise_init
struct _ExactPart[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    """A partial exact fold: the working value and the formats it read."""

    var value: Self.T
    var formats: _Formats
    # An empty lane's result is the identity itself, not rounded.
    var identity: Bool


@fieldwise_init
struct _ExactFold[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    function: def(T, T, /, *, context: C) raises thin -> T,
](_Fold):
    """A lifted Float or Complex function folded exactly and rounded once.

    The function runs on exact working values with its default context, so it
    keeps every bit; `rounding` decides the one final rounding."""

    comptime Element = Self.T
    comptime Partial = _ExactPart[Self.T]
    comptime Result = Self.T
    var initial: Optional[Self.T]
    var identity: Optional[Self.T]
    var rounding: _ComplexContextArgument

    def fold(self, lanes: _Lanes[Self.T], j: Int, start: Int, end: Int, opening: Bool) raises -> _ExactPart[Self.T]:
        var base = lanes.base(j)
        var k = start
        var formats = _Formats()
        var result: Self.T
        if opening and self.initial:
            formats.take(self.initial.value())
            result = _working(self.initial.value())
        elif start < end:
            ref first = lanes.at(base, start)
            try:
                formats.take(first)
            except error:
                raise _failure("reduce", lanes.coordinates(j, start), error)
            result = _working(first)
            k = start + 1
        elif self.identity:
            return _ExactPart[Self.T](self.identity.value(), formats, True)
        else:
            raise Error(
                "Cannot reduce an empty selection: the lifted function has no"
                " identity. Lift it with identity= or pass initial=."
            )
        try:
            while k < end:
                ref element = lanes.at(base, k)
                formats.take(element)
                result = Self.function(result, _working(element), context=Self.C())
                k += 1
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return _ExactPart[Self.T](result^, formats, False)

    def combine(
        self, left: _ExactPart[Self.T], right: _ExactPart[Self.T], lanes: _Lanes[Self.T], j: Int, k: Int,
    ) raises -> _ExactPart[Self.T]:
        var formats = left.formats
        try:
            formats.take(right.formats)
            return _ExactPart[Self.T](Self.function(left.value, right.value, context=Self.C()), formats, False)
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)

    def finish(self, var partial: _ExactPart[Self.T]) raises -> Self.T:
        if partial.identity:
            return partial.value
        return _rounded_once(partial.value, partial.formats, self.rounding)


# --------------------------------------------------------- mixed-input folds


@fieldwise_init
struct _InputPart[E: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    """Keep an uncombined first operand exact, even across reduction chunks."""

    var seed: Optional[Self.E]
    var value: Optional[Self.R]
    var formats: _Formats
    var identity: Bool

    def __init__(out self):
        self.seed = None
        self.value = None
        self.formats = _Formats()
        self.identity = False


def _input_formats[R: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable](
    mut formats: _Formats, value: E,
) raises:
    comptime if _Rounded[E]:
        formats.take(value)
        comptime if R == Complex and E == Float:
            formats.imag = _merge_float_formats(formats.imag, rebind[Float](value).format())


def _working_argument[T: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable, exact: Bool](value: E) raises -> T:
    var result = _scalar_argument[T](value)
    comptime if exact:
        comptime if T == _FloatArgument:
            ref arg = rebind[_FloatArgument](result)
            arg.format = FloatFormat._exact_format(RoundingMode.nearest_even)
        elif T == _ComplexArgument:
            ref arg = rebind[_ComplexArgument](result)
            arg.real.format = FloatFormat._exact_format(RoundingMode.nearest_even)
            arg.imag.format = FloatFormat._exact_format(RoundingMode.nearest_even)
        else:
            return _working(result)
    return result^


def _part_argument[T: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, exact: Bool](
    part: _InputPart[E, R],
) raises -> T:
    if part.value:
        return _working_argument[T, R, exact](part.value.value())
    return _working_argument[T, E, exact](part.seed.value())


def _input_step[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](mut part: _InputPart[E, R], value: E, context: C) raises:
    comptime if exact:
        _input_formats[R](part.formats, value)
    if not part.value and not part.seed:
        part.seed = value
    else:
        part.value = function(_part_argument[T, E, R, exact](part), _working_argument[U, E, exact](value), context=context)
        part.seed = None


def _input_finish[E: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, exact: Bool](
    part: _InputPart[E, R], context: _CallContext[C],
) raises -> R:
    if part.value:
        comptime if exact:
            if not part.identity:
                return _rounded_once(part.value.value(), part.formats, rebind[_ComplexContextArgument](context))
        return part.value.value()
    # No callback runs for a singleton. Convert only the emitted result; a
    # running fold retains its exact seed for the next call.
    comptime if R == Float:
        var arg = _scalar_argument[_FloatArgument](part.seed.value())
        var target = _uniform(rebind[_ComplexContextArgument](context)) if context else ArithmeticContext(format=_merge_float_formats(arg.format, None))
        return rebind[R](Float(_rounded=Float._rounded_input(arg.value, target)))
    elif R == Complex:
        var arg = _scalar_argument[_ComplexArgument](part.seed.value())
        if context:
            return rebind[R](Complex(arg.real, arg.imag, context=rebind[_ComplexContextArgument](context)))
        var real = _merge_float_formats(arg.real.format, arg.imag.format)
        var imag = _merge_float_formats(arg.imag.format, arg.real.format)
        return rebind[R](Complex(arg.real, arg.imag, context=ComplexContext(
            real=ArithmeticContext(format=real), imag=ArithmeticContext(format=imag),
        )))
    elif R == Ball:
        var arg = _scalar_argument[_BallArgument](part.seed.value())
        comptime if C == Optional[BallContext]:
            return rebind[R](_rounded_ball(arg, _ball_precision(rebind[Optional[BallContext]](context), arg.precision)))
        else:
            return rebind[R](_rounded_ball(arg, _ball_precision(None, arg.precision)))
    else:
        return _scalar_argument[R](part.seed.value())


@fieldwise_init
struct _InputFold[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    E: ImplicitlyCopyable & Deinitable, function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](_Fold):
    comptime Element = Self.E
    comptime Partial = _InputPart[Self.E, Self.R]
    comptime Result = Self.R
    var initial: Optional[Self.R]
    var identity: Optional[Self.R]
    var step: Self.C
    var rounding: _CallContext[Self.C]

    def fold(self, lanes: _Lanes[Self.E], j: Int, start: Int, end: Int, opening: Bool) raises -> Self.Partial:
        var part = Self.Partial()
        if opening and self.initial:
            part.value = self.initial
            comptime if Self.exact:
                part.formats.take(self.initial.value())
        elif start == end:
            if not self.identity:
                raise Error("Cannot reduce an empty selection: the lifted function has no identity. Lift it with identity= or pass initial=.")
            part.value = self.identity
            part.identity = True
        var base = lanes.base(j)
        var k = start
        try:
            while k < end:
                _input_step[Self.function, Self.exact](part, lanes.at(base, k), self.step)
                k += 1
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return part^

    def combine(self, left: Self.Partial, right: Self.Partial, lanes: _Lanes[Self.E], j: Int, k: Int) raises -> Self.Partial:
        var part = Self.Partial()
        try:
            part.formats = left.formats
            part.formats.take(right.formats)
            part.value = Self.function(
                _part_argument[Self.T, Self.E, Self.R, Self.exact](left),
                _part_argument[Self.U, Self.E, Self.R, Self.exact](right), context=self.step,
            )
        except error:
            raise _failure("reduce", lanes.coordinates(j, k), error)
        return part^

    def finish(self, var partial: Self.Partial) raises -> Self.R:
        return _input_finish[Self.E, Self.R, Self.C, Self.exact](partial, self.rounding)


# --------------------------------------------------------------- accumulate


@fieldwise_init
struct _AccumulateJob[T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable](
    Copyable, Movable,
):
    var lanes: _Lanes[Self.T]
    # The context each call receives, and the one rounding of exact outputs.
    var step: Self.C
    var rounding: _CallContext[Self.C]


def _accumulate_into[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, E: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](job: _AccumulateJob[E, C], j: Int, mut slots: _Slots[R], mut written: Int) raises:
    """One lane's running folds, written in order; exact folds round each
    output once from the exact running value."""
    ref lanes = job.lanes
    var length = lanes.length()
    var base = lanes.base(j)
    comptime if E != R:
        var part = _InputPart[E, R]()
        for k in range(length):
            _input_step[function, exact](part, lanes.at(base, k), job.step)
            slots.put(j * length + k, _input_finish[E, R, C, exact](part, job.rounding))
            written += 1
    elif exact:
        var formats = _Formats()
        ref first = lanes.at(base, 0)
        formats.take(first)
        var result = _working(rebind[R](first))
        slots.put(j * length, rebind[R](first) if not job.rounding else _rounded_once(result, formats, rebind[_ComplexContextArgument](job.rounding)))
        written += 1
        for k in range(1, length):
            ref element = lanes.at(base, k)
            formats.take(element)
            result = function(rebind[T](result), rebind[U](_working(element)), context=C())
            slots.put(j * length + k, _rounded_once(result, formats, rebind[_ComplexContextArgument](job.rounding)))
            written += 1
    else:
        var result = rebind[R](lanes.at(base, 0))
        slots.put(j * length, result)
        written += 1
        for k in range(1, length):
            result = function(rebind[T](result), rebind[U](lanes.at(base, k)), context=job.step)
            slots.put(j * length + k, result)
            written += 1


def _accumulate_chunk[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    E: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](job: _AccumulateJob[E, C], begin: Int, end: Int, mut slots: _Slots[R]) -> Int:
    """Lanes [begin, end), each written in order to its own run of the result."""
    var written = 0
    for j in range(begin, end):
        if slots.stopped(j):
            return written
        if not job.lanes.length():
            continue
        try:
            _accumulate_into[function, exact](job, j, slots, written)
        except:
            slots.fail(j)
            return written
    return written


def _accumulate_lane[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, E: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](job: _AccumulateJob[E, C], j: Int) raises:
    """Run one lane's folds again, raising with coordinates at the first failure."""
    ref lanes = job.lanes
    var base = lanes.base(j)
    comptime if E != R:
        var part = _InputPart[E, R]()
        for k in range(lanes.length()):
            try:
                _input_step[function, exact](part, lanes.at(base, k), job.step)
                _ = _input_finish[E, R, C, exact](part, job.rounding)
            except error:
                raise _failure("accumulate", lanes.coordinates(j, k), error)
        return
    comptime if E == R:
        var formats = _Formats()
        var result: R
        comptime if exact:
            formats.take(lanes.at(base, 0))
            result = _working(rebind[R](lanes.at(base, 0)))
            if job.rounding:
                try:
                    _ = _rounded_once(result, formats, rebind[_ComplexContextArgument](job.rounding))
                except error:
                    raise _failure("accumulate", lanes.coordinates(j, 0), error)
        else:
            result = rebind[R](lanes.at(base, 0))
        for k in range(1, lanes.length()):
            try:
                comptime if exact:
                    formats.take(lanes.at(base, k))
                    result = function(rebind[T](result), rebind[U](_working(lanes.at(base, k))), context=C())
                    _ = _rounded_once(result, formats, rebind[_ComplexContextArgument](job.rounding))
                else:
                    result = function(rebind[T](result), rebind[U](lanes.at(base, k)), context=job.step)
            except error:
                raise _failure("accumulate", lanes.coordinates(j, k), error)


def _accumulate[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, E: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, exact: Bool,
](values: Batch[E], axis: Int, step: C, rounding: _CallContext[C]) raises -> Batch[R]:
    ref layout = values._layout[]
    var rank = len(layout.shape)
    var a = _axis(axis, rank)
    var kept = List[Int]()
    for i in range(rank):
        if i != a:
            kept.append(i)
    var job = _AccumulateJob[E, C](_Lanes[E](values, kept.copy(), [a]), step, rounding)
    var length = job.lanes.length()
    _ = _checked_count(values.size(), max(1, size_of[R]()))
    var result = _parallel_map[_accumulate_chunk[E, function, exact]](job.copy(), job.lanes.count(), length)
    if result.failed >= 0:
        # Lanes are pure: the lowest failing lane fails again here, with its message.
        _accumulate_lane[function, exact](job, result.failed)
        raise Error(String(
            "Batch lane ", result.failed, " failed on a worker thread but not"
            " when repeated; lifted functions must be pure.",
        ))
    # Each lane is one contiguous run, so the result is stored lane by lane:
    # the axis has stride 1 and the other axes step over whole lanes.
    var shape = layout.shape.copy()
    var result_layout = _FlatLayout(shape.copy())
    var step_size = length
    for j in range(len(kept)):
        var i = kept[len(kept) - 1 - j]
        result_layout.strides[i] = step_size
        step_size *= shape[i]
    result_layout.strides[a] = 1
    result_layout.contiguous = result_layout._is_contiguous()
    return Batch[R](_owner=ArcPointer(result^.take()), _layout=result_layout^)


# ---------------------------------------------------------- outer and call


@fieldwise_init
struct _PairJob[
    A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable,
](Copyable, Movable):
    var a: Self.A
    var left: _FlatLayout
    var b: Self.B
    var right: _FlatLayout
    var context: Self.C


def _input_layout[V: _MapArgument & ImplicitlyCopyable & Deinitable](value: V) -> _FlatLayout:
    comptime if conforms_to(V, _BatchShape):
        return rebind[Batch[_Element[V]]](value)._layout[]
    else:
        return _FlatLayout()


@always_inline
def _pair_value[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, outer: Bool,
](job: _PairJob[A, B, C], mut a: _FlatInput[T, A], mut b: _FlatInput[U, B], index: Int) raises -> R:
    comptime if outer:
        var i = index // job.right.size
        var j = index % job.right.size
        return function(a.at(job.left.position(i)), b.at(job.right.position(j)), context=job.context)
    else:
        return function(a.at(job.left.position(index)), b.at(job.right.position(index)), context=job.context)


struct _PairKernel[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> R, outer: Bool,
](_ElementKernel):
    comptime Element = Self.R
    comptime Job = _PairJob[Self.A, Self.B, Self.C]
    var a: _FlatInput[Self.T, Self.A]
    var b: _FlatInput[Self.U, Self.B]

    def __init__(out self, job: Self.Job, begin: Int) raises:
        self.a = _FlatInput[Self.T, Self.A](job.a)
        self.b = _FlatInput[Self.U, Self.B](job.b)

    @always_inline
    def apply(mut self, job: Self.Job, index: Int) raises -> Self.R:
        return _pair_value[Self.function, Self.outer](job, self.a, self.b, index)


@fieldwise_init
struct _PairErrors[origin: Origin[mut=False], //, outer: Bool](_ElementErrors):
    var shape: Pointer[List[Int], Self.origin]

    def failure(self, index: Int, var error: Error) -> Error:
        comptime operation: StaticString = "outer" if Self.outer else "call"
        return _failure(operation, _unravel(index, self.shape[]), error)

    def repeated(self, index: Int) -> Error:
        return Error("A lifted call failed on a worker thread but not when repeated; lifted functions must be pure.")


def _pairs[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, outer: Bool,
](job: _PairJob[A, B, C], shape: List[Int]) raises -> _LiftOutput[R]:
    var count = _FlatLayout(shape.copy()).size
    var values = _execute_values[_PairKernel[T, U, R, C, A, B, function, outer]](
        job, count, _PairErrors[outer](Pointer(to=shape)),
    )
    return _shaped_output[R](_Tensor[R](values^, [count]), shape)


def _shape_text(shape: List[Int]) -> String:
    var text = String("[")
    for i in range(len(shape)):
        if i:
            text += ", "
        text += String(shape[i])
    return text + "]"


def _broadcast_shape(a: List[Int], b: List[Int]) raises -> List[Int]:
    """NumPy broadcasting: align trailing axes; each pair must match or contain 1."""
    try:
        return _broadcast_shapes(a, b)
    except:
        raise Error(String(
            "Cannot broadcast shapes ", _shape_text(a), " and ", _shape_text(b),
            "; trailing axes must be equal or 1.",
        ))


def _broadcast_walk(layout: _FlatLayout, shape: List[Int]) raises -> _FlatLayout:
    """Use the shared layout rules for broadcast strides and checked sizes."""
    var result = layout.broadcast_to(shape)
    result.offset = _start(layout)
    return result^


# --------------------------------------------------------------- the lift


# Masks fold as batches of 0 and 1: the lane machinery serves number batches.


def _bit(value: Bool) -> Integer:
    return Integer(1) if value else Integer(0)


def _bits_of(mask: Mask) raises -> Batch[Integer]:
    var bits = mask.to_list()
    var values = List[Integer](capacity=len(bits))
    for bit in bits:
        values.append(_bit(bit))
    return Batch[Integer](values).reshape(mask.shape())


def _mask_of(values: Batch[Integer]) raises -> Mask:
    var integers = values.to_list()
    var bits = List[Bool](capacity=len(integers))
    for value in integers:
        bits.append(value.sign() != 0)
    return Mask(bits, shape=values.shape())


def _on_bits[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    function: def(T, T, /, *, context: C) raises thin -> T,
](x: Integer, y: Integer, /, *, context: C) raises -> Integer:
    """A Bool function (T is Bool) on 0 and 1."""
    var a = x.sign() != 0
    var b = y.sign() != 0
    return _bit(rebind[Bool](function(rebind[T](a), rebind[T](b), context=context)))


def _shaped_output[R: ImplicitlyCopyable & Deinitable](var values: _Tensor[R], shape: List[Int]) raises -> _LiftOutput[R]:
    """Mapped results with their shape: a Mask of Bools, else a batch."""
    comptime if R == Bool:
        return rebind_var[_LiftOutput[R]](Mask(rebind[List[Bool]](values._owner[].list), shape=shape))
    else:
        return rebind_var[_LiftOutput[R]](Batch[R](_tensor=values).reshape(shape))


struct _Lifted[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, //,
    function: def(T, U, /, *, context: C) raises thin -> R, takes_context: Bool,
](Copyable, Movable):
    """A binary function lifted to NumPy-style batch operations; see `lift`."""

    var identity: Optional[Self.R]
    var associative: Bool
    var exact: Bool

    def __init__(out self, identity: Optional[Self.R], associative: Bool, exact: Bool):
        self.identity = identity
        self.associative = associative
        self.exact = exact

    def _step(self, context: _CallContext[Self.C]) raises -> Self.C:
        """The context each call receives. A given context must be used: by a
        function that takes one, or by the rounding of an exact fold."""
        comptime if Self.takes_context:
            return _step_context[Self.C](context)
        else:
            if context:
                raise Error(
                    "This lifted function takes no context: lift a function with a"
                    " context keyword, such as apn_mojo.float.add, or fold Float or"
                    " Complex values exactly, where context= decides the final rounding."
                )
            return Self.C()

    def __call__[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, a: A, b: B, *, context: _CallContext[Self.C] = _CallContext[Self.C](),
    ) raises -> _LiftOutput[Self.R]:
        """Apply the function elementwise with NumPy broadcasting.

        Inputs follow vmap's numeric conversion rules. A scalar has shape []
        and broadcasts over every batch position. Bool results give a Mask.
        """
        var left = _input_layout(a)
        var right = _input_layout(b)
        var shape = _broadcast_shape(left.shape, right.shape)
        var job = _PairJob[A, B, Self.C](
            a, _broadcast_walk(left, shape), b, _broadcast_walk(right, shape), self._step(context),
        )
        return _pairs[Self.function, False](job, shape)

    def outer[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, a: A, b: B, *, context: _CallContext[Self.C] = _CallContext[Self.C](),
    ) raises -> _LiftOutput[Self.R]:
        """Apply the function to every pair. The result shape concatenates
        the input shapes; a scalar contributes no axes. Bool results give a Mask."""
        var left = _input_layout(a)
        var right = _input_layout(b)
        var shape = left.shape.copy()
        for extent in right.shape:
            shape.append(extent)
        var job = _PairJob[A, B, Self.C](a, left^, b, right^, self._step(context))
        return _pairs[Self.function, True](job, shape)

    def _reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], axes: List[Int], every: Bool, keepdims: Bool, initial: Optional[Self.R],
        context: _CallContext[Self.C],
    ) raises -> Batch[Self.R] where _Closed[Self.T, Self.U, Self.R]:
        comptime if E == Self.R:
            comptime same = (
                rebind[def(Self.R, Self.R, /, *, context: Self.C) raises thin -> Self.R](Self.function)
                if Self.T == Self.R and Self.U == Self.R else
                rebind[def(Self.R, Self.R, /, *, context: Self.C) raises thin -> Self.R](_adapted[Self.function])
            )
            comptime if _Rounded[Self.R]:
                if self.exact:
                    return _reduce_lanes(
                        _ExactFold[same](initial, self.identity, rebind[_ComplexContextArgument](context)), rebind[Batch[Self.R]](values), axes, every, keepdims, self.associative,
                    )
            return _reduce_lanes(
                _LiftFold[same](initial, self.identity, self._step(context)), rebind[Batch[Self.R]](values), axes, every, keepdims, self.associative,
            )
        else:
            comptime if _Rounded[Self.R]:
                if self.exact:
                    return _reduce_lanes(
                        _InputFold[E, Self.function, True](initial, self.identity, Self.C(), context),
                        values, axes, every, keepdims, self.associative,
                    )
            return _reduce_lanes(
                _InputFold[E, Self.function, False](initial, self.identity, self._step(context), context),
                values, axes, every, keepdims, self.associative,
            )

    def reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, var axis: _ReduceAxes = 0, keepdims: Bool = False,
        initial: Optional[Self.R] = None, context: _CallContext[Self.C] = _CallContext[Self.C](),
    ) raises -> Batch[Self.R] where _Closed[Self.T, Self.U, Self.R]:
        """Fold along one axis or several, like `np.add.reduce(values, axis=axis)`.

        `axis` is an Int or a list of axes; negative axes count from the end.
        Each result folds its elements in row-major order. `initial`, when
        given, is folded in first. An empty selection gives `initial`, else
        the identity; without either it raises. Float and Complex results are
        exact folds rounded once, with `context` when given, unless lifted with
        `exact=False`; then `context` goes to every call.
        """
        return self._reduce(values, axis.axes, False, keepdims, initial, context)

    def reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, axis: NoneType, initial: Optional[Self.R] = None,
        context: _CallContext[Self.C] = _CallContext[Self.C](),
    ) raises -> Self.R where _Closed[Self.T, Self.U, Self.R]:
        """Fold every element in row-major order to one value."""
        return self._reduce(values, List[Int](), True, False, initial, context).item()

    def _bits(self) -> _Lifted[
        _on_bits[rebind[def(Self.T, Self.T, /, *, context: Self.C) raises thin -> Self.T](Self.function)],
        Self.takes_context,
    ] where Self.T == Bool and Self.U == Bool and Self.R == Bool:
        """This Bool function on 0 and 1, for folding Masks."""
        var identity = Optional[Integer]()
        if self.identity:
            identity = _bit(rebind[Bool](self.identity.value()))
        return {identity, self.associative, False}

    def reduce(
        self, mask: Mask, *, var axis: _ReduceAxes = 0, keepdims: Bool = False, initial: Optional[Bool] = None,
    ) raises -> Mask where Self.T == Bool and Self.U == Bool and Self.R == Bool:
        """Fold a Mask along one axis or several with a Bool function."""
        var start = Optional[Integer]()
        if initial:
            start = _bit(initial.value())
        return _mask_of(self._bits()._reduce(_bits_of(mask), axis.axes, False, keepdims, start, _CallContext[Self.C]()))

    def reduce(
        self, mask: Mask, *, axis: NoneType, initial: Optional[Bool] = None,
    ) raises -> Bool where Self.T == Bool and Self.U == Bool and Self.R == Bool:
        """Fold every entry of a Mask in row-major order."""
        var start = Optional[Integer]()
        if initial:
            start = _bit(initial.value())
        return self._bits()._reduce(_bits_of(mask), List[Int](), True, False, start, _CallContext[Self.C]()).item().sign() != 0

    def accumulate[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, axis: Int = 0, context: _CallContext[Self.C] = _CallContext[Self.C](),
    ) raises -> Batch[Self.R] where _Closed[Self.T, Self.U, Self.R]:
        """Running folds along one axis, like `np.add.accumulate` (`np.cumsum` for addition).

        Element k along the axis is f(...f(f(x0, x1), x2)..., xk); for Float
        and Complex each is computed exactly and rounded once, with `context`
        when given, unless lifted with `exact=False`. Separate lanes run in
        parallel; each lane runs in order.
        """
        comptime if E == Self.R:
            comptime same = (
                rebind[def(Self.R, Self.R, /, *, context: Self.C) raises thin -> Self.R](Self.function)
                if Self.T == Self.R and Self.U == Self.R else
                rebind[def(Self.R, Self.R, /, *, context: Self.C) raises thin -> Self.R](_adapted[Self.function])
            )
            comptime if _Rounded[Self.R]:
                if self.exact:
                    return _accumulate[same, True](values, axis, Self.C(), context)
            return _accumulate[same, False](values, axis, self._step(context), context)
        else:
            comptime if _Rounded[Self.R]:
                if self.exact:
                    return _accumulate[Self.function, True](values, axis, Self.C(), context)
            return _accumulate[Self.function, False](values, axis, self._step(context), context)

    def accumulate(
        self, mask: Mask, *, axis: Int = 0,
    ) raises -> Mask where Self.T == Bool and Self.U == Bool and Self.R == Bool:
        """Running folds of a Mask along one axis with a Bool function."""
        return _mask_of(self._bits().accumulate(_bits_of(mask), axis=axis))


struct _LiftedUpdate[T: ImplicitlyCopyable & Deinitable, //, update: def(mut T, T) raises thin -> None](Copyable, Movable):
    """An in-place update lifted to NumPy-style batch operations; see `lift`."""

    var identity: Optional[Self.T]
    var associative: Bool
    var exact: Bool

    def __init__(out self, identity: Optional[Self.T], associative: Bool, exact: Bool):
        self.identity = identity
        self.associative = associative
        self.exact = exact

    def _values(self) -> _Lifted[_no_context[_updated[Self.update]], False]:
        return _Lifted[_no_context[_updated[Self.update]], False](self.identity, self.associative, self.exact)

    def __call__[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable](self, a: A, b: B) raises -> Batch[Self.T]:
        return rebind_var[Batch[Self.T]](self._values()(a, b))

    def outer[A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable](self, a: A, b: B) raises -> Batch[Self.T]:
        return rebind_var[Batch[Self.T]](self._values().outer(a, b))

    def _reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], axes: List[Int], every: Bool, keepdims: Bool, initial: Optional[Self.T],
        context: _ComplexContextArgument,
    ) raises -> Batch[Self.T]:
        comptime assert _Closed[Self.T, Self.T, Self.T]
        comptime if _Rounded[Self.T] or E != Self.T:
            # Exact folds work on exact copies, so in-place updates gain nothing.
            return self._values()._reduce(values, axes, every, keepdims, initial, context)
        else:
            _ = self._values()._step(context)
            return _reduce_lanes(
                _UpdateFold[Self.update](initial, self.identity), rebind[Batch[Self.T]](values), axes, every, keepdims, self.associative,
            )

    def reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, var axis: _ReduceAxes = 0, keepdims: Bool = False,
        initial: Optional[Self.T] = None, context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises -> Batch[Self.T]:
        """As for a lifted function; the accumulator is updated in place."""
        return self._reduce(values, axis.axes, False, keepdims, initial, context)

    def reduce[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, axis: NoneType, initial: Optional[Self.T] = None,
        context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises -> Self.T:
        return self._reduce(values, List[Int](), True, False, initial, context).item()

    def accumulate[E: ImplicitlyCopyable & Deinitable](
        self, values: Batch[E], *, axis: Int = 0, context: _ComplexContextArgument = _ComplexContextArgument(),
    ) raises -> Batch[Self.T]:
        comptime assert _Closed[Self.T, Self.T, Self.T]
        # Every running value is kept, so each step copies; updates gain nothing here.
        return self._values().accumulate(values, axis=axis, context=context)


def lift[
    T: ImplicitlyCopyable & Deinitable, //, update: def(mut T, T) raises thin -> None,
](*, identity: Optional[T] = None, associative: Bool = False, exact: Bool = True) -> _LiftedUpdate[update]:
    """Lift an in-place update `def(mut acc: T, x: T)`, such as `acc += x`, so reductions reuse one accumulator."""
    comptime assert _Numeric[T], (
        "lift takes and returns apn_mojo numbers (Integer, Rational, Float or"
        " Complex or Ball); use vmap for Bool, Mask or batch results."
    )
    return _LiftedUpdate[update](identity, associative, exact)


def lift[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U) raises thin -> R,
](
    *__disambiguate: NoneType, identity: Optional[R] = None, associative: Bool = False, exact: Bool = True,
) -> _Lifted[_no_context[function], False]:
    """Lift a binary function to NumPy-style batch operations.

    `lift[f]()` has `reduce` (like `np.add.reduce`), `accumulate` (like
    `np.cumsum` for addition), `outer` and a broadcasting call. For Float and
    Complex values, `reduce` and `accumulate` fold exact working copies and round
    each result once to the merged input formats, or with `context=` when given,
    so a lifted `+` returns what `sum` returns. Ball folds carry each scalar
    enclosure forward in order; `exact` does not affect them. Numeric inputs
    follow vmap's exact adaptation rules; calls and outer products also take
    scalars, with shape []. A function returning Bool gives
    Masks from `outer` and the call; a function of two Bools folds Masks.

    Parameters:
        T: The first argument type, inferred.
        U: The second argument type, inferred.
        R: The result type, inferred.
        function: The binary function: a named `def` or a lambda that captures
            nothing.

    Args:
        __disambiguate: Unused; it makes the arguments after it keyword-only.
        identity: The result of reducing an empty selection; without it, an empty
            reduction raises unless the call passes `initial=`.
        associative: Promise that `f(f(a, b), c) == f(a, f(b, c))`, so one long
            reduction can fold chunks on separate threads and combine them in order.
        exact: Fold Float and Complex values exactly and round once; `False`
            rounds every step, as a scalar loop would.

    Returns:
        The lifted function.
    """
    comptime assert (_Numeric[_LiftElement[T]] and _Numeric[_LiftElement[U]] and (_Numeric[R] or R == Bool)) or (
        T == Bool and U == Bool and R == Bool
    ), (
        "lift takes apn_mojo numbers (Integer, Rational, Float, Complex or Ball) and"
        " returns a number or Bool, or folds Bools; use vmap for batch results."
    )
    return _Lifted[_no_context[function], False](identity, associative, exact)


def lift[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R,
](
    *, identity: Optional[R] = None, associative: Bool = False, exact: Bool = True,
) -> _Lifted[function, True] where (_LiftContext[C]):
    """Lift a binary function with a `context` keyword, such as `apn_mojo.float.add`.

    As for a plain function; the calls of `outer`, the broadcasting call and
    folds with `exact=False` receive the caller's `context=`, and exact Float
    and Complex folds round once with it. Ball functions receive `BallContext`
    at every step. Argument adapters accept the same numeric inputs as vmap,
    including mixed exact operands and full-width integer literals.

    Parameters:
        T: The first argument type, inferred; adapters retain exact inputs
            until the scalar call.
        U: The second argument type, inferred.
        C: The context type: `Optional[ArithmeticContext]`, the Complex form,
            or `Optional[BallContext]`.
        R: The result type, inferred.
        function: The binary function with a `context` keyword.

    Args:
        identity: The result of reducing an empty selection; without it, an empty
            reduction raises unless the call passes `initial=`.
        associative: Promise that `f(f(a, b), c) == f(a, f(b, c))`, so one long
            reduction can fold chunks on separate threads and combine them in order.
        exact: Fold Float and Complex values exactly and round once; `False`
            rounds every step, passing the context to each call.

    Returns:
        The lifted function.
    """
    comptime assert _Numeric[_LiftElement[T]] and _Numeric[_LiftElement[U]] and (_Numeric[R] or R == Bool), (
        "lift takes apn_mojo numbers (Integer, Rational, Float, Complex or Ball) and"
        " returns a number or Bool; use vmap for batch results."
    )
    return _Lifted[function, True](identity, associative, exact)
