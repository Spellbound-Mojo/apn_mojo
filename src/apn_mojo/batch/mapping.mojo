"""Lift typed Mojo functions over batch axes of any rank.

A mapping has one type per callback. Nesting vmap adds a runtime level, not
a new type, and a nested mapping runs as one flat loop over every mapped
axis. Inputs and results travel as untyped arrays (see _array.mojo); only
the callback call and per-leaf argument and result adapters are typed.
"""

from std.collections import Array
from std.builtin.rebind import downcast
from std.memory import ArcPointer
from std.sys import size_of
from .value import Batch, _BatchShape, _batch_from, _layout_of, _result_layout, _set_formats
from .mask import Mask
from ._array import _ArrayData, _ArrayBuilder
from ._families import _family_of
from ._layout import _FlatLayout, _axis
from ._tensor import _Tensor
from ._values import _Ownership, _Values, _borrow_values
from ._parallel import _ElementKernel, _ElementErrors, _execute_values
from ..common._sizes import _checked_count
from ..common._traits import _BatchElement, _MapAdapter, _MapArgument
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..float._arithmetic import _FloatArgument
from ..float._input import _FloatInput
from ..complex.value import Complex
from ..complex._input import _ComplexArgument
from ..ball.context import BallContext
from ..complex.context import _ComplexContextArgument
from ..float.context import ArithmeticContext, FloatFormat


trait _TupleArgument:
    pass


__extension Tuple(_TupleArgument):
    pass


# Mapped calls accept library values, batches, masks, Bool and literals; see
# _MapArgument in common/_traits.mojo. Literals also convert exactly here.
trait _WholeLiteral:
    def _exact(self) -> Integer: ...


trait _DecimalLiteral:
    pass


__extension IntLiteral(_WholeLiteral):
    def _exact(self) -> Integer:
        return Integer(type_of(self)())


__extension FloatLiteral(_DecimalLiteral):
    pass


# Leaf kinds, known at compile time from argument and result types.
comptime _NUMBER = 0
comptime _BATCH = 1
comptime _BOOL = 2
comptime _MASK = 3

# Source kinds, known at run time from the inputs.
comptime _ARRAY_SOURCE = 0
comptime _MASK_SOURCE = 1
comptime _VALUE_SOURCE = 2


comptime _MappingLeaf[T: ImplicitlyCopyable & Deinitable] = (
    conforms_to(T, _BatchShape) or conforms_to(T, _BatchElement) or T == Bool or T == Mask
)
# Scalar functions may take an argument adapter that accepts any exact number.
# Function types match by arity alone, so only these types mark a context
# keyword; any other trailing argument is an ordinary mapped argument.
comptime _Context[C: AnyType] = (
    C == Optional[ArithmeticContext] or C == _ComplexContextArgument or C == Optional[BallContext]
)
comptime _Adapter[T: AnyType] = conforms_to(T, _MapAdapter)
comptime _ArgumentLeaf[T: ImplicitlyCopyable & Deinitable] = _MappingLeaf[T] or _Adapter[T]
comptime _Element[T: ImplicitlyCopyable & Deinitable] = T.Element if conforms_to(T, _BatchShape) else T
comptime _LeafOutput[T: ImplicitlyCopyable & Deinitable] = Mask if T == Bool or T == Mask else Batch[_Element[T]]
comptime _TupleLeaf[T: Movable] = downcast[T, ImplicitlyCopyable & Deinitable]
comptime _TupleOutput[T: Movable] = _LeafOutput[_TupleLeaf[T]]
comptime _Outputs[R: type_of(Tuple)] = Tuple[*R.Ts.map[ToTrait=ImplicitlyCopyable & Deinitable, Mapper=_TupleOutput]()]


struct _Result[R: ImplicitlyCopyable & Deinitable]:
    comptime leaf = _MappingLeaf[Self.R]
    comptime Output = _LeafOutput[Self.R]


def _leaf_kind[T: ImplicitlyCopyable & Deinitable]() -> Int:
    comptime assert _ArgumentLeaf[T], "vmap leaves must be apn_mojo numbers, numeric batches, Mask or Bool; nested tuple trees remain pending."
    comptime if T == Bool:
        return _BOOL
    elif T == Mask:
        return _MASK
    elif conforms_to(T, _BatchShape):
        return _BATCH
    else:
        return _NUMBER


# ---------------------------------------------------------------- options


trait _ShapeDimensions:
    def _dimensions(self) -> Optional[List[Int]]: ...


__extension Array(_ShapeDimensions):
    def _dimensions(self) -> Optional[List[Int]]:
        comptime assert Self.T == Int, "out_shape dimensions must be Int values."
        ref values = rebind[Array[Int, Self.length]](self)
        var result = List[Int](capacity=Self.length)
        for value in values:
            result.append(value)
        return result^


__extension Optional(_ShapeDimensions):
    def _dimensions(self) -> Optional[List[Int]]:
        comptime assert conforms_to(Self.T, Copyable & Deinitable), "out_shape needs copyable Int dimensions."
        if self:
            return _capture_dimensions(rebind[downcast[Self.T, Copyable & Deinitable]](self.value()))
        return None


def _capture_dimensions[S: Copyable & Deinitable](shape: S) -> Optional[List[Int]]:
    comptime if S == NoneType or S == type_of(None):
        return None
    elif S == List[Int]:
        return rebind[List[Int]](shape).copy()
    elif conforms_to(S, _ShapeDimensions):
        return rebind[downcast[S, _ShapeDimensions]](shape)._dimensions()
    else:
        comptime assert False, "out_shape must be None or an Int list/array with one dimension per returned batch axis; omit the mapped axis."


def _optional_axis[A: ImplicitlyCopyable](axis: A) -> Optional[Int]:
    comptime if A == NoneType or A == type_of(None):
        return None
    elif A == Int:
        return rebind[Int](axis)
    else:
        comptime assert A == Optional[Int], "in_axes entries must be Int or None."
        return rebind[Optional[Int]](axis)


struct _Axes(Copyable, Defaultable):
    """One axis shared by every leaf, or one axis (or None) per leaf."""
    var values: List[Optional[Int]]
    var uniform: Bool
    var axis: Optional[Int]

    def __init__(out self):
        self.values = List[Optional[Int]]()
        self.uniform = True
        self.axis = 0

    @implicit
    def __init__[A: ImplicitlyCopyable](out self, axis: A) where (A == Int or A == Optional[Int] or A == NoneType or A == type_of(None)):
        self.values = List[Optional[Int]]()
        self.uniform = True
        self.axis = _optional_axis(axis)

    @implicit
    def __init__[*As: ImplicitlyCopyable](out self, axes: Tuple[*As]):
        self.values = List[Optional[Int]](capacity=len(As))
        self.uniform = False
        self.axis = None
        comptime for i in range(len(As)):
            self.values.append(_optional_axis(axes[i]))

    def resolve(self, count: Int, message: StaticString) raises -> List[Optional[Int]]:
        if self.uniform:
            return List[Optional[Int]](length=count, fill=self.axis)
        if len(self.values) != count:
            raise Error(message)
        return self.values.copy()


struct _Shapes(Copyable, Defaultable):
    """out_shape: one optional shape shared by a single result, or one per result."""
    var values: List[Optional[List[Int]]]
    var per_result: Bool

    def __init__(out self):
        self.values = List[Optional[List[Int]]]()
        self.values.append(None)
        self.per_result = False

    @implicit
    def __init__(out self, none: NoneType):
        self = Self()

    @implicit
    def __init__(out self, var shape: List[Int]):
        self = Self()
        self.values[0] = shape^

    def __init__(out self, var *dimensions: Int, __list_literal__: NoneType):
        self = Self()
        var shape = List[Int](capacity=len(dimensions))
        for i in range(len(dimensions)):
            shape.append(dimensions[i])
        self.values[0] = shape^

    @implicit
    def __init__[N: Int](out self, shape: Array[Int, N]):
        self = Self()
        self.values[0] = shape._dimensions()

    @implicit
    def __init__[S: Copyable & Deinitable](out self, shape: Optional[S]):
        self = Self()
        self.values[0] = shape._dimensions()

    @implicit
    def __init__[*Ss: Copyable & Deinitable](out self, shapes: Tuple[*Ss]):
        self.values = List[Optional[List[Int]]](capacity=len(Ss))
        self.per_result = True
        comptime for i in range(len(Ss)):
            self.values.append(_capture_dimensions(shapes[i]))


struct _Level(Copyable):
    """One vmap level: how inputs are split and where results are stacked."""
    var in_axes: _Axes
    var out_axes: _Axes
    var axis_size: Optional[Int]
    var out_shapes: List[Optional[List[Int]]]
    var tuple_shapes: Bool
    # Options that cannot apply, reported when the mapped function is called.
    var problem: String

    def __init__(out self, var in_axes: _Axes, var out_axes: _Axes, axis_size: Optional[Int], out_shape: _Shapes):
        self.in_axes = in_axes^
        self.out_axes = out_axes^
        self.axis_size = axis_size
        self.tuple_shapes = out_shape.per_result
        self.out_shapes = out_shape.values.copy()
        self.problem = ""

    def out_axis(self, result: Int, count: Int) raises -> Int:
        var axes = self.out_axes.resolve(count, "out_axes needs one Int per tuple result, or one Int shared by all results.")
        if not axes[result]:
            raise Error("out_axes entries must be integers.")
        return axes[result].value()

    def out_shape(self, result: Int, count: Int) raises -> Optional[List[Int]]:
        if not self.tuple_shapes:
            if count != 1 and self.out_shapes[0]:
                raise Error("Tuple results need one out_shape entry per result; use None for scalar leaves.")
            return self.out_shapes[0].copy() if count == 1 else None
        if len(self.out_shapes) != count:
            raise Error("out_shape needs one entry per result; use None for scalar leaves.")
        return self.out_shapes[result].copy()


def _level(var in_axes: _Axes, var out_axes: _Axes, axis_size: Optional[Int], var out_shape: _Shapes) -> _Level:
    return _Level(in_axes^, out_axes^, axis_size, out_shape^)


struct _AxisLevels(Copyable, Defaultable):
    """in_axes or out_axes: one setting, or a list with one per level, innermost first."""
    var levels: List[_Axes]
    var listed: Bool

    def __init__(out self):
        self.levels = [_Axes()]
        self.listed = False

    @implicit
    def __init__[A: ImplicitlyCopyable](out self, axis: A) where (A == Int or A == Optional[Int] or A == NoneType or A == type_of(None)):
        self.levels = [_Axes(axis)]
        self.listed = False

    @implicit
    def __init__[*As: ImplicitlyCopyable](out self, axes: Tuple[*As]):
        self.levels = [_Axes(axes)]
        self.listed = False

    def __init__(out self, var *levels: _Axes, __list_literal__: NoneType):
        self.levels = List[_Axes](capacity=len(levels))
        for i in range(len(levels)):
            self.levels.append(levels[i].copy())
        self.listed = True


def _levels(
    var in_axes: _AxisLevels, var out_axes: _AxisLevels, axis_size: Optional[Int], var out_shape: _Shapes,
) -> List[_Level]:
    """Mapping levels, outermost first, from options listing them innermost first.

    A single out_axes applies to every level; axis_size and out_shape describe
    the innermost level (add outer levels with .vmap(...) to set theirs).
    """
    var count = len(in_axes.levels)
    var problem = String("")
    if count == 0:
        count = 1
        in_axes.levels.append(_Axes())
        problem = "in_axes=[...] needs at least one level."
    if out_axes.listed and len(out_axes.levels) != count:
        problem = String(
            "out_axes=[...] has ", len(out_axes.levels), " levels but in_axes has ", count,
            "; give one out_axes entry per level, or one value for all levels.",
        )
    var levels = List[_Level](capacity=count)
    for k in range(count - 1, -1, -1):
        var out = out_axes.levels[k].copy() if out_axes.listed and k < len(out_axes.levels) else out_axes.levels[0].copy()
        var level = _Level(
            in_axes.levels[k].copy(), out^, axis_size if k == 0 else None,
            out_shape.copy() if k == 0 else _Shapes(),
        )
        level.problem = problem
        levels.append(level^)
    return levels^


# ---------------------------------------------------------------- inputs


struct _Source(Movable):
    """One argument leaf: an array, a mask or a value passed through as is."""
    var kind: Int
    var array: Optional[_ArrayData]
    var bits: Optional[ArcPointer[List[Bool]]]
    var layout: _FlatLayout
    var steps: List[Int]

    def __init__(out self, kind: Int):
        self.kind = kind
        self.array = None
        self.bits = None
        self.layout = _FlatLayout()
        self.steps = List[Int]()

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.array = move.array^
        self.bits = move.bits^
        self.layout = move.layout^
        self.steps = move.steps^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def rank(self) -> Int:
        return len(self.layout.shape)


def _source_of[V: ImplicitlyCopyable & Deinitable](value: V) raises -> _Source:
    comptime if conforms_to(V, _BatchShape):
        var data = value._array()
        var result = _Source(_ARRAY_SOURCE)
        result.layout = data.layout
        result.array = data^
        return result^
    elif V == Mask:
        ref mask = rebind[Mask](value)
        var result = _Source(_MASK_SOURCE)
        result.layout = _FlatLayout(mask.shape())
        result.bits = ArcPointer(mask.to_list())
        return result^
    else:
        return _Source(_VALUE_SOURCE)


struct _Execution(Movable):
    """The flat loop over every level's mapped axis, outermost first."""
    var counts: List[Int]
    var sources: List[_Source]
    var index: List[Int]
    var offsets: List[Int]
    var total: Int

    def __init__(out self, var sources: List[_Source]):
        self.counts = List[Int]()
        self.sources = sources^
        self.index = List[Int]()
        self.offsets = List[Int]()
        self.total = 1

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.counts = move.counts^
        self.sources = move.sources^
        self.index = move.index^
        self.offsets = move.offsets^
        self.total = move.total

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def advance(mut self):
        var level = len(self.counts) - 1
        while level >= 0:
            self.index[level] += 1
            for i in range(len(self.sources)):
                if self.sources[i].steps:
                    self.offsets[i] += self.sources[i].steps[level]
            if self.index[level] < self.counts[level]:
                return
            for i in range(len(self.sources)):
                if self.sources[i].steps:
                    self.offsets[i] -= self.sources[i].steps[level] * self.counts[level]
            self.index[level] = 0
            level -= 1

    def failure(self, error: Error) -> Error:
        var message = String(error)
        for j in range(len(self.index)):
            var level = len(self.index) - 1 - j
            message = String("vmap failed at mapped index ", self.index[level], ": ", message)
        return Error(message)


def _flat_count(lengths: List[Optional[Int]], size: Optional[Int]) raises -> Int:
    var inferred = Optional[Int]()
    for length in lengths:
        if length:
            if inferred and inferred.value() != length.value():
                raise Error("Mapped axes have different lengths; vmap requires equal axis lengths, not broadcasting.")
            inferred = length
    if size:
        if size.value() < 0:
            raise Error("vmap axis_size must be nonnegative; use zero for an empty result.")
        if inferred and inferred.value() != size.value():
            raise Error("vmap axis_size does not match the mapped input; use its axis length.")
        return size.value()
    if not inferred:
        raise Error("vmap needs axis_size when all arguments are unmapped; specify the result length.")
    return inferred.value()


def _settle(mut layout: _FlatLayout):
    """Size and contiguity of a layout whose mapped axes were removed."""
    var size = 1
    for dim in layout.shape:
        size *= dim
    layout.size = size
    layout.contiguous = layout._is_contiguous()


def _prepare(var sources: List[_Source], levels: List[_Level], kinds: List[Int]) raises -> _Execution:
    """Split every source along its mapped axes; axes are checked before lengths."""
    var execution = _Execution(sources^)
    for level in levels:
        if level.problem:
            raise Error(level.problem)
        var count = len(execution.sources)
        var axes = level.in_axes.resolve(count, "in_axes needs one Int or None per tuple input; match the function argument tuple.")
        var lengths = List[Optional[Int]](length=count, fill=None)
        var selected = List[Int](length=count, fill=-1)
        for i in range(count):
            if not axes[i]:
                continue
            if execution.sources[i].kind == _VALUE_SOURCE:
                # A shared axis applies to batches; scalars are always shared.
                if level.in_axes.uniform:
                    continue
                raise Error("A scalar argument has no mapped axis; use in_axes=None for this argument.")
            selected[i] = _axis(axes[i].value(), execution.sources[i].rank())
            lengths[i] = execution.sources[i].layout.shape[selected[i]]
        execution.counts.append(_flat_count(lengths, level.axis_size))
        for i in range(count):
            if execution.sources[i].kind == _VALUE_SOURCE:
                continue
            if selected[i] < 0:
                execution.sources[i].steps.append(0)
                continue
            var a = selected[i]
            execution.sources[i].steps.append(execution.sources[i].layout.strides[a])
            _ = execution.sources[i].layout.shape.pop(a)
            _ = execution.sources[i].layout.strides.pop(a)
    for i in range(len(execution.sources)):
        if kinds[i] == _NUMBER and execution.sources[i].kind != _VALUE_SOURCE and execution.sources[i].rank() != 0:
            raise Error("A mapped argument must have one fewer axis than its input; change the function argument rank or use in_axes=None.")
        if kinds[i] == _BOOL and execution.sources[i].kind == _MASK_SOURCE and execution.sources[i].rank() != 0:
            raise Error("A Bool argument maps a vector mask; nest vmap to map additional axes, or use a Mask argument.")
        if execution.sources[i].kind != _VALUE_SOURCE:
            _settle(execution.sources[i].layout)
            execution.offsets.append(execution.sources[i].layout.offset)
        else:
            execution.offsets.append(0)
    for count in execution.counts:
        execution.total *= count
        execution.index.append(0)
    return execution^


def _argument[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    source: _Source, offset: Int, value: V,
) raises -> T:
    """Materialize one callback argument; only this adapter is typed per leaf."""
    comptime kind = _leaf_kind[T]()
    comptime if kind == _NUMBER:
        comptime if conforms_to(V, _BatchShape):
            comptime if V.Element == T:
                return source.array.value().owner[T]()[][offset]
            else:
                # Elements convert exactly, as unmapped scalars do.
                comptime assert V.Element != Complex or T == _ComplexArgument, (
                    "A real function cannot take Complex elements; map the"
                    " Complex function instead, for example apn_mojo.complex.sqrt."
                )
                return _scalar_argument[T](source.array.value().owner[V.Element]()[][offset])
        else:
            return _scalar_argument[T](value)
    elif kind == _BATCH:
        comptime assert conforms_to(V, _BatchShape), "A batch argument needs a batch input."
        comptime assert _Element[V] == _Element[T], "vmap argument and input must have the same numeric element type."
        var layout = source.layout
        layout.offset = offset
        var data = source.array.value().with_layout(layout^)
        return rebind_var[T](_batch_from[_Element[T]](data))
    elif kind == _BOOL:
        comptime if V == Bool:
            return rebind[T](value)
        else:
            comptime assert V == Mask, "A mask input needs a Bool or Mask argument."
            return rebind[T](source.bits.value()[][offset])
    else:
        comptime assert V == Mask, "A mask input needs a Bool or Mask argument."
        var layout = source.layout
        layout.offset = offset
        var bits = List[Bool](capacity=layout.size)
        ref values = source.bits.value()[]
        var cursor = layout.cursor()
        for _ in range(layout.size):
            bits.append(values[cursor.position])
            cursor.advance()
        return rebind_var[T](Mask(bits, shape=layout.shape.copy()))


struct _Reader[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable]:
    """One callback argument, prepared once per call.

    Mapped elements of the argument's own type are borrowed in place and an
    unmapped scalar converts once, so neither copies per element. Elements of
    another family (an Integer batch for a Rational or Float parameter) convert
    exactly from their borrowed storage; batch, Bool and Mask arguments build a
    value at each index. A converted value lives in a one-element list: an
    Optional would rebuild its Variant at every element.
    """
    var elements: Optional[ArcPointer[_Values[_Element[Self.V]]]]
    var held: List[Self.T]

    def __init__(out self, source: _Source, value: Self.V) raises:
        self.elements = None
        self.held = List[Self.T](capacity=1)
        comptime if _leaf_kind[Self.T]() == _NUMBER:
            comptime if conforms_to(Self.V, _BatchShape):
                self.elements = source.array.value().owner[_Element[Self.V]]()
            else:
                self._hold(_scalar_argument[Self.T](value))

    @always_inline
    def _hold(mut self, var value: Self.T):
        if len(self.held):
            self.held[0] = value^
        else:
            self.held.append(value^)

    @always_inline
    def _held(self) -> ref[self] Self.T:
        return _borrow_values[origin_of(self)](self.held)[0]

    @always_inline
    def read(mut self, source: _Source, offset: Int, value: Self.V) raises -> ref[ImmOrigin(origin_of(self))] Self.T:
        # The reader owns its storage and outlives the callback invocation, so
        # each borrow is tied to the reader itself (as _Tensor._read does).
        comptime if _leaf_kind[Self.T]() == _NUMBER and not conforms_to(Self.V, _BatchShape):
            return self._held()
        elif _leaf_kind[Self.T]() == _NUMBER and _Element[Self.V] == Self.T:
            return rebind[Self.T](_borrow_values[ImmOrigin(origin_of(self))](self.elements.value()[].list)[offset])
        elif _leaf_kind[Self.T]() == _NUMBER:
            comptime assert _Element[Self.V] != Complex or Self.T == _ComplexArgument, (
                "A real function cannot take Complex elements; map the"
                " Complex function instead, for example apn_mojo.complex.sqrt."
            )
            self._hold(_scalar_argument[Self.T](self.elements.value()[].list[offset]))
            return self._held()
        else:
            self._hold(_argument[Self.T](source, offset, value))
            return self._held()


def _scalar_argument[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](value: V) raises -> T:
    """An unmapped scalar or element as the function's argument type, exactly."""
    comptime if conforms_to(V, _WholeLiteral):
        var whole = value._exact()
        comptime if T == Integer:
            return rebind_var[T](whole^)
        elif T == Rational:
            return rebind_var[T](Rational(whole))
        elif conforms_to(T, _MapAdapter):
            return rebind_var[T](downcast[T, _MapAdapter]._adapt(whole))
        else:
            comptime assert False, (
                "An integer literal converts exactly only to Integer or Rational"
                " arguments or to a function taking any number; pass Float(...)"
                " or Complex(...) explicitly."
            )
    elif conforms_to(V, _DecimalLiteral):
        comptime assert False, (
            'Decimal literals are not exact; use Float("0.1") for an exact'
            " decimal or Float(Float64(...)) for a native value."
        )
    elif V == T:
        return rebind[T](value)
    elif T == Rational and V == Integer:
        return rebind_var[T](Rational(rebind[Integer](value)))
    elif conforms_to(T, _MapAdapter):
        # A family's argument union converts each value itself.
        return rebind_var[T](downcast[T, _MapAdapter]._adapt(value))
    else:
        comptime assert False, "An unmapped scalar must match the function argument type."


# ---------------------------------------------------------------- results


struct _Collector(Movable):
    """Stack one result leaf across every mapped index."""
    var kind: Int
    var builder: Optional[_ArrayBuilder]
    var bits: List[Bool]
    var shape: Optional[List[Int]]
    var formats: Bool

    def __init__(out self, kind: Int, family: Int):
        self.kind = kind
        self.builder = None
        if kind == _NUMBER or kind == _BATCH:
            self.builder = _ArrayBuilder(family)
        self.bits = List[Bool]()
        self.shape = None
        if kind == _NUMBER or kind == _BOOL:
            self.shape = List[Int]()
        self.formats = False

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.builder = move.builder^
        self.bits = move.bits^
        self.shape = move.shape^
        self.formats = move.formats

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def check_shape(mut self, shape: List[Int]) raises:
        if self.shape:
            if self.shape.value() != shape:
                if self.kind == _MASK:
                    raise Error("Mapped function returned inconsistent mask shapes; return the same shape at every mapped index.")
                raise Error("Mapped function returned inconsistent shapes; return the same tensor shape at every mapped index.")
        else:
            self.shape = shape.copy()

    def append_array(mut self, data: _ArrayData) raises:
        self.check_shape(data.shape())
        if not self.formats:
            self.builder.value().real_format = data.real_format
            self.builder.value().imag_format = data.imag_format
            self.formats = True
        self.builder.value().append_array(data)


def _collector[R: ImplicitlyCopyable & Deinitable]() -> _Collector:
    comptime kind = _leaf_kind[R]()
    comptime if kind == _NUMBER or kind == _BATCH:
        return _Collector(kind, _family_of[_Element[R]]())
    else:
        return _Collector(kind, 0)


def _collect[R: ImplicitlyCopyable & Deinitable](mut collector: _Collector, var value: R) raises:
    comptime if R == Bool:
        collector.bits.append(rebind[Bool](value))
    elif R == Mask:
        ref mask = rebind[Mask](value)
        collector.check_shape(mask.shape())
        collector.bits.extend(mask.to_list())
    elif conforms_to(R, _BatchShape):
        collector.append_array(value._array())
    else:
        collector.builder.value().append(value^)


def _apply_hints(mut collector: _Collector, levels: List[_Level], counts: List[Int], result: Int, results: Int) raises:
    """Fix a result's leaf shape from out_shape; returned values must then match it."""
    # An out_shape at level L describes that level's per-index result: the
    # inner levels' axes placed by their out_axes around the leaf shape.
    for j in range(len(levels)):
        var level = len(levels) - 1 - j
        var hint = levels[level].out_shape(result, results)
        if not hint:
            continue
        var expected = hint.value().copy()
        var inner_levels = len(levels) - 1 - level
        var leaf_rank = len(expected) - inner_levels
        if leaf_rank < 0:
            raise Error("out_shape must contain one dimension for each returned batch axis; omit the mapped axis.")
        var full = List[Int](capacity=level + 1 + len(expected))
        for k in range(level + 1):
            full.append(counts[k])
        full.extend(expected.copy())
        _ = _checked_count(_FlatLayout(full^).size, 1)
        var labels = List[Int]()
        for k in range(leaf_rank):
            labels.append(-1 - k)
        for inner in range(len(levels) - 1, level, -1):
            labels.insert(_axis(levels[inner].out_axis(result, results), len(labels) + 1), inner)
        var leaf = List[Int]()
        for k in range(len(labels)):
            if labels[k] < 0:
                leaf.append(expected[k])
            elif expected[k] != counts[labels[k]]:
                raise Error("Mapped function returned inconsistent shapes; return the same tensor shape at every mapped index.")
        collector.check_shape(leaf)


def _stacked_layout(collector: _Collector, levels: List[_Level], counts: List[Int], result: Int, results: Int) raises -> Tuple[List[Int], List[Int]]:
    """The stacked shape (levels outermost first) and the out_axes permutation."""
    if not collector.shape:
        if collector.kind == _MASK:
            raise Error("Empty mask mapping needs out_shape; provide the returned mask's dimensions.")
        raise Error("Empty tensor mapping needs out_shape; provide the returned tensor's dimensions.")
    var leaf = collector.shape.value().copy()
    var shape = counts.copy()
    shape.extend(leaf.copy())
    _ = _checked_count(_FlatLayout(shape.copy()).size, 1)
    # Insert each level's axis from the innermost outwards; labels name the
    # stacked axes, whose order is levels outermost first, then leaf axes.
    var labels = List[Int]()
    for k in range(len(leaf)):
        labels.append(len(levels) + k)
    for j in range(len(levels)):
        var level = len(levels) - 1 - j
        labels.insert(_axis(levels[level].out_axis(result, results), len(labels) + 1), level)
    return (shape^, labels^)


def _check_outputs(mut collectors: List[_Collector], levels: List[_Level], counts: List[Int]) raises:
    """Validate out_shape and out_axes before the function runs; reserve scalar results."""
    var total = 1
    for count in counts:
        total *= count
    for j in range(len(collectors)):
        _apply_hints(collectors[j], levels, counts, j, len(collectors))
        if collectors[j].shape:
            _ = _stacked_layout(collectors[j], levels, counts, j, len(collectors))
        if collectors[j].kind == _NUMBER:
            collectors[j].builder.value().reserve(total)
        elif collectors[j].kind == _BOOL:
            _ = _checked_count(total, 1)
            collectors[j].bits.reserve(total)


def _finish[R: ImplicitlyCopyable & Deinitable](
    var collector: _Collector, levels: List[_Level], counts: List[Int], result: Int, results: Int,
) raises -> _LeafOutput[R]:
    var stacked = _stacked_layout(collector, levels, counts, result, results)
    var shape = stacked[0].copy()
    var axes = stacked[1].copy()
    var ordered = True
    for i in range(len(axes)):
        ordered = ordered and axes[i] == i
    comptime kind = _leaf_kind[R]()
    comptime if kind == _BOOL or kind == _MASK:
        var mask = Mask(collector.bits, shape=shape^)
        if ordered:
            return rebind_var[_LeafOutput[R]](mask^)
        return rebind_var[_LeafOutput[R]](mask.transpose(axes))
    else:
        var builder = collector.builder.take()
        var data = builder^.finish(shape^)
        if not ordered:
            data = data.transposed(axes)
        return rebind_var[_LeafOutput[R]](_batch_from[_Element[R]](data))


def _check_leaves[Rs: type_of(Tuple)]():
    comptime for j in range(Rs.Ts.length):
        comptime assert _MappingLeaf[_TupleLeaf[Rs.Ts[j]]], "Tuple results must contain apn_mojo numbers, numeric batches, Mask or Bool; nested tuple trees remain pending."


def _collectors[Rs: type_of(Tuple)]() -> List[_Collector]:
    var result = List[_Collector](capacity=Rs.Ts.length)
    comptime for j in range(Rs.Ts.length):
        result.append(_collector[_TupleLeaf[Rs.Ts[j]]]())
    return result^


def _collect_tuple[Rs: type_of(Tuple)](
    mut collectors: List[_Collector], value: downcast[Rs, ImplicitlyCopyable & Deinitable],
) raises:
    ref leaves = rebind[Tuple[*Rs.Ts]](value)
    comptime for j in range(Rs.Ts.length):
        _collect(collectors[j], rebind[_TupleLeaf[Rs.Ts[j]]](leaves[j]))


struct _CollectedLeaves(Movable):
    """The general path's collectors, finished one tuple field at a time."""
    var collectors: List[_Collector]
    var levels: List[_Level]
    var counts: List[Int]

    def __init__(out self, var collectors: List[_Collector], levels: List[_Level], counts: List[Int]):
        self.collectors = collectors^
        self.levels = levels.copy()
        self.counts = counts.copy()

    def leaf[R: ImplicitlyCopyable & Deinitable, j: Int, n: Int](mut self) raises -> _LeafOutput[R]:
        # Fields finish left to right, so the next collector is always field j.
        return _finish[R](self.collectors.pop(0), self.levels, self.counts, j, n)


def _tuple_leaf[Rs: type_of(Tuple), j: Int, S: Movable](mut source: S) raises -> _TupleOutput[Rs.Ts[j]]:
    """Field j of a tuple result, from collectors or from a flat run."""
    comptime R = _TupleLeaf[Rs.Ts[j]]
    comptime if S == _CollectedLeaves:
        ref collected = rebind[_CollectedLeaves](source)
        return rebind_var[_TupleOutput[Rs.Ts[j]]](collected.leaf[R, j, Rs.Ts.length]())
    else:
        ref flat = rebind[_FlatLeaves[Rs]](source)
        return rebind_var[_TupleOutput[Rs.Ts[j]]](flat.leaf[R, j]())


def _tuple_outputs[Rs: type_of(Tuple), S: Movable](mut source: S) raises -> _Outputs[Rs]:
    comptime n = Rs.Ts.length
    comptime assert n <= 8, "vmap collects at most eight tuple results."
    # Fields finish left to right; each arity builds the tuple directly.
    comptime if n == 0:
        return rebind_var[_Outputs[Rs]](Tuple())
    elif n == 1:
        return rebind_var[_Outputs[Rs]](Tuple(_tuple_leaf[Rs, 0](source)))
    elif n == 2:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
        ))
    elif n == 3:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
        ))
    elif n == 4:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
            _tuple_leaf[Rs, 3](source),
        ))
    elif n == 5:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
            _tuple_leaf[Rs, 3](source),
            _tuple_leaf[Rs, 4](source),
        ))
    elif n == 6:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
            _tuple_leaf[Rs, 3](source),
            _tuple_leaf[Rs, 4](source),
            _tuple_leaf[Rs, 5](source),
        ))
    elif n == 7:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
            _tuple_leaf[Rs, 3](source),
            _tuple_leaf[Rs, 4](source),
            _tuple_leaf[Rs, 5](source),
            _tuple_leaf[Rs, 6](source),
        ))
    else:
        return rebind_var[_Outputs[Rs]]((
            _tuple_leaf[Rs, 0](source),
            _tuple_leaf[Rs, 1](source),
            _tuple_leaf[Rs, 2](source),
            _tuple_leaf[Rs, 3](source),
            _tuple_leaf[Rs, 4](source),
            _tuple_leaf[Rs, 5](source),
            _tuple_leaf[Rs, 6](source),
            _tuple_leaf[Rs, 7](source),
        ))


def _finish_tuple[Rs: type_of(Tuple)](
    var collectors: List[_Collector], levels: List[_Level], counts: List[Int],
) raises -> _Outputs[Rs]:
    var source = _CollectedLeaves(collectors^, levels, counts)
    return _tuple_outputs[Rs](source)


trait _MappingResults:
    """Compile-time result handling for the shared mapping runners.

    Readers and traversal do not depend on whether a call returns one leaf or
    a tuple. The policy supplies collectors and finishes outputs in leaf order.
    """
    comptime Value: ImplicitlyCopyable & Deinitable
    comptime Output: ImplicitlyCopyable & Deinitable
    comptime count: Int
    comptime flat: Bool

    @staticmethod
    def collectors() -> List[_Collector]: ...

    @staticmethod
    def collect(mut collectors: List[_Collector], var value: Self.Value) raises: ...

    @staticmethod
    def finish(var collectors: List[_Collector], levels: List[_Level], counts: List[Int]) raises -> Self.Output: ...

    @staticmethod
    def finish_flat(var values: _Values[Self.Value], like: Optional[ArcPointer[_FlatLayout]], plan: _FlatPlan) raises -> Self.Output: ...


struct _LeafResults[R: ImplicitlyCopyable & Deinitable](_MappingResults):
    comptime Value = Self.R
    comptime Output = _LeafOutput[Self.R]
    comptime count = 1
    comptime flat = _flat_result[Self.R]()

    @staticmethod
    def collectors() -> List[_Collector]:
        return _collectors[Tuple[Self.R]]()

    @staticmethod
    def collect(mut collectors: List[_Collector], var value: Self.Value) raises:
        _collect(collectors[0], value^)

    @staticmethod
    def finish(var collectors: List[_Collector], levels: List[_Level], counts: List[Int]) raises -> Self.Output:
        return _finish[Self.R](collectors.pop(0), levels, counts, 0, 1)

    @staticmethod
    def finish_flat(var values: _Values[Self.Value], like: Optional[ArcPointer[_FlatLayout]], plan: _FlatPlan) raises -> Self.Output:
        return _flat_leaf[Self.R](values^, like, plan, 0)


struct _TupleResults[Rs: type_of(Tuple)](_MappingResults):
    comptime Value = _Leaves[Self.Rs]
    comptime Output = _Outputs[Self.Rs]
    comptime count = Self.Rs.Ts.length
    comptime flat = _flat_results[Self.Rs]()

    @staticmethod
    def collectors() -> List[_Collector]:
        return _collectors[Self.Rs]()

    @staticmethod
    def collect(mut collectors: List[_Collector], var value: Self.Value) raises:
        _collect_tuple[Self.Rs](collectors, value)

    @staticmethod
    def finish(var collectors: List[_Collector], levels: List[_Level], counts: List[Int]) raises -> Self.Output:
        return _finish_tuple[Self.Rs](collectors^, levels, counts)

    @staticmethod
    def finish_flat(var values: _Values[Self.Value], like: Optional[ArcPointer[_FlatLayout]], plan: _FlatPlan) raises -> Self.Output:
        var leaves = _FlatLeaves[Self.Rs](values^, like, plan)
        return _tuple_outputs[Self.Rs](leaves)


def _tuple_arguments[As: type_of(Tuple), *Vs: ImplicitlyCopyable & Deinitable](
    execution: _Execution, values: Tuple[*Vs],
) raises -> Tuple[*As.Ts]:
    comptime n = As.Ts.length
    comptime assert len(Vs) == n, "Pass one tuple input per function argument leaf."
    comptime assert n <= 8, "vmap maps at most eight tuple argument leaves."
    ref s = execution.sources
    ref o = execution.offsets
    # Common arities build the argument tuple directly.
    comptime if n == 0:
        return rebind_var[Tuple[*As.Ts]](Tuple())
    elif n == 1:
        return rebind_var[Tuple[*As.Ts]](Tuple(_argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0])))
    elif n == 2:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
        ))
    elif n == 3:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
        ))
    elif n == 4:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
            _argument[_TupleLeaf[As.Ts[3]]](s[3], o[3], values[3]),
        ))
    elif n == 5:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
            _argument[_TupleLeaf[As.Ts[3]]](s[3], o[3], values[3]),
            _argument[_TupleLeaf[As.Ts[4]]](s[4], o[4], values[4]),
        ))
    elif n == 6:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
            _argument[_TupleLeaf[As.Ts[3]]](s[3], o[3], values[3]),
            _argument[_TupleLeaf[As.Ts[4]]](s[4], o[4], values[4]),
            _argument[_TupleLeaf[As.Ts[5]]](s[5], o[5], values[5]),
        ))
    elif n == 7:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
            _argument[_TupleLeaf[As.Ts[3]]](s[3], o[3], values[3]),
            _argument[_TupleLeaf[As.Ts[4]]](s[4], o[4], values[4]),
            _argument[_TupleLeaf[As.Ts[5]]](s[5], o[5], values[5]),
            _argument[_TupleLeaf[As.Ts[6]]](s[6], o[6], values[6]),
        ))
    else:
        return rebind_var[Tuple[*As.Ts]]((
            _argument[_TupleLeaf[As.Ts[0]]](s[0], o[0], values[0]),
            _argument[_TupleLeaf[As.Ts[1]]](s[1], o[1], values[1]),
            _argument[_TupleLeaf[As.Ts[2]]](s[2], o[2], values[2]),
            _argument[_TupleLeaf[As.Ts[3]]](s[3], o[3], values[3]),
            _argument[_TupleLeaf[As.Ts[4]]](s[4], o[4], values[4]),
            _argument[_TupleLeaf[As.Ts[5]]](s[5], o[5], values[5]),
            _argument[_TupleLeaf[As.Ts[6]]](s[6], o[6], values[6]),
            _argument[_TupleLeaf[As.Ts[7]]](s[7], o[7], values[7]),
        ))


def _tuple_sources[As: type_of(Tuple), *Vs: ImplicitlyCopyable & Deinitable](
    values: Tuple[*Vs], mut kinds: List[Int],
) raises -> List[_Source]:
    comptime n = As.Ts.length
    comptime assert len(Vs) == n, "Pass one tuple input per function argument leaf."
    var sources = List[_Source](capacity=n)
    comptime for i in range(n):
        sources.append(_source_of(values[i]))
        kinds.append(_leaf_kind[_TupleLeaf[As.Ts[i]]]())
    return sources^


def _single_source[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    value: V, mut kinds: List[Int],
) raises -> List[_Source]:
    var sources = List[_Source](capacity=1)
    sources.append(_source_of(value))
    kinds.append(_leaf_kind[T]())
    return sources^


def _pair_sources[T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable](
    first: V, second: W, mut kinds: List[Int],
) raises -> List[_Source]:
    var sources = List[_Source](capacity=2)
    sources.append(_source_of(first))
    sources.append(_source_of(second))
    kinds.append(_leaf_kind[T]())
    kinds.append(_leaf_kind[U]())
    return sources^


def _triple_sources[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, D: ImplicitlyCopyable & Deinitable,
](first: A, second: B, third: D, mut kinds: List[Int]) raises -> List[_Source]:
    var sources = List[_Source](capacity=3)
    sources.append(_source_of(first))
    sources.append(_source_of(second))
    sources.append(_source_of(third))
    kinds.append(_leaf_kind[T]())
    kinds.append(_leaf_kind[U]())
    kinds.append(_leaf_kind[W]())
    return sources^


# ---------------------------------------------------------------- flat fast path
# Scalar arguments and results are the common case. When every level maps an
# axis of each batch argument down to its elements (or shares the argument),
# a call is one run over the mapped extents in row-major order: each argument
# is read in place through one stride per level, and results move into one
# reserved list, stacked by out_axes without copying. Anything else takes the
# general path, which also owns every validation error.


def _lend_integer(mut target: Integer, source: Integer):
    """Copy source's handle bits into target without touching any refcount.

    Target's previous bits are an inline value or another lent handle, so they
    are overwritten without release. Only _FlatInput lends: the batch it reads
    is immutable and outlives the call, and _FlatInput restores an inline zero
    before the argument is destroyed. A copy the mapped function makes counts
    normally.
    """
    comptime assert size_of[Integer]() == 16, "Integer is two words"
    Pointer(to=target).unsafe_bitcast[SIMD[DType.uint64, 2]]()[] = (
        Pointer(to=source).unsafe_bitcast[SIMD[DType.uint64, 2]]()[]
    )


def _lend_float(mut target: _FloatArgument, source: Float):
    """Make target the argument record of source, lending its significand."""
    target.value.kind = source._kind
    target.value.negative = source._negative
    target.value.scale = Int128(source._exponent) - Int128(source.precision())
    target.format = source._format
    _lend_integer(target.value.numerator, source._significand)


def _lent_float() -> _FloatArgument:
    """An argument record whose significand slot holds an inline zero."""
    return _FloatArgument(
        _FloatInput(0, False, Integer(0), Integer(1), Int128(0)), None, 0
    )


struct _Cursor(Copyable, Movable):
    """Where a flat run finds one argument's elements.

    Index i of the run is a coordinate per level, innermost fastest; the
    element's position is offset plus each coordinate times that level's
    stride. Levels whose strides chain like a contiguous run merge into the
    innermost. An input steps along the innermost level itself and calls
    _carry at the end of each inner run; _seek places it at any index.
    """
    var offset: Int
    var stride: Int
    var inner: Int
    # Unmerged outer levels, outermost first; empty for a single run.
    var outer_shape: List[Int]
    var outer_strides: List[Int]
    var coords: List[Int]
    var base: Int
    var column: Int

    def __init__(out self, plan: _FlatPlan, argument: Int):
        var level = plan.depth - 1
        self.offset = plan.offset(argument)
        self.stride = plan.stride(argument, level)
        self.inner = plan.extent(level)
        while level > 0 and plan.stride(argument, level - 1) == self.stride * self.inner:
            level -= 1
            self.inner *= plan.extent(level)
        # A single run, the common case, allocates nothing.
        self.outer_shape = List[Int]()
        self.outer_strides = List[Int]()
        self.coords = List[Int]()
        for k in range(level):
            self.outer_shape.append(plan.extent(k))
            self.outer_strides.append(plan.stride(argument, k))
            self.coords.append(0)
        self.base = self.offset
        self.column = 0

    @no_inline
    def _carry(mut self):
        """Move base to the start of the next inner run."""
        self.column = 0
        var k = len(self.outer_shape) - 1
        while k >= 0:
            self.coords[k] += 1
            self.base += self.outer_strides[k]
            if self.coords[k] < self.outer_shape[k]:
                return
            self.base -= self.outer_strides[k] * self.outer_shape[k]
            self.coords[k] = 0
            k -= 1

    @no_inline
    def _seek(mut self, index: Int):
        var rest = index // self.inner
        self.column = index - rest * self.inner
        self.base = self.offset
        for j in range(len(self.outer_shape)):
            var k = len(self.outer_shape) - 1 - j
            var extent = self.outer_shape[k]
            var quotient = rest // extent
            self.coords[k] = rest - quotient * extent
            self.base += self.coords[k] * self.outer_strides[k]
            rest = quotient


struct _FlatPlan(Copyable, Movable):
    """How a call runs flat, or count < 0 for the general path.

    One table holds each level's extent (outermost first), then each
    argument's first position, then each argument's stride per level, zero
    where the level shares it, then each argument's mapped axes as a bitmask
    (a batch parameter receives the others). axes holds each result's stacked-axis order,
    depth entries per result, and is empty when every result stays in level
    order, as default out_axes leave it.
    """
    var count: Int
    var depth: Int
    var arguments: Int
    var table: List[Int]
    var axes: List[Int]

    def __init__(out self):
        self.count = -2
        self.depth = 0
        self.arguments = 0
        self.table = List[Int]()
        self.axes = List[Int]()

    @always_inline
    def extent(self, level: Int) -> Int:
        return self.table[level]

    @always_inline
    def offset(self, argument: Int) -> Int:
        return self.table[self.depth + argument]

    @always_inline
    def stride(self, argument: Int, level: Int) -> Int:
        return self.table[self.depth + self.arguments + argument * self.depth + level]

    def used(self, argument: Int) -> Int:
        return self.table[self.depth + self.arguments * (1 + self.depth) + argument]

    def shape(self) -> List[Int]:
        var shape = List[Int](capacity=self.depth)
        for level in range(self.depth):
            shape.append(self.extent(level))
        return shape^

    def order(self, result: Int) -> List[Int]:
        """Result's stacked axes as a transpose permutation; empty when in level order."""
        var order = List[Int]()
        if len(self.axes):
            var identity = True
            for level in range(self.depth):
                order.append(self.axes[result * self.depth + level])
                identity = identity and order[level] == level
            if identity:
                order.clear()
        return order^

    def cursor(self, argument: Int) -> _Cursor:
        return _Cursor(self, argument)


@always_inline
def _unused_axis(used: Int, index: Int) -> Int:
    """The physical axis of the index-th axis not yet in the used bitmask."""
    var axis = 0
    var seen = 0
    while True:
        if not (used >> axis) & 1:
            if seen == index:
                return axis
            seen += 1
        axis += 1


@no_inline
def _plan(
    levels: List[_Level], layouts: List[Optional[ArcPointer[_FlatLayout]]], results: Int, batches: Int,
) -> _FlatPlan:
    """The flat plan for number or batch arguments and scalar results, when one applies.

    Axes resolve as in _prepare: each level maps an axis of what the outer
    levels left. A number argument must have no axis left; an argument whose
    bit is set in batches is a batch parameter, which receives the rest. Whatever the
    general path would reject, or handles specially (axis_size, out_shape),
    gives the general plan, so that path raises its own errors.
    """
    var count = len(layouts)
    var depth = len(levels)
    for level in levels:
        if level.problem or level.axis_size:
            return _FlatPlan()
        if not level.in_axes.uniform and len(level.in_axes.values) != count:
            return _FlatPlan()
        for shape in level.out_shapes:
            if shape:
                return _FlatPlan()
    var plan = _FlatPlan()
    plan.depth = depth
    plan.arguments = count
    plan.table = List[Int](length=depth + count * (2 + depth), fill=0)
    for l in range(depth):
        plan.table[l] = -1
    for k in range(count):
        if not layouts[k]:
            # A scalar is shared; a per-argument axis for it is an error.
            for level in levels:
                if not level.in_axes.uniform and level.in_axes.values[k]:
                    return _FlatPlan()
            continue
        ref layout = layouts[k].value()[]
        var left = len(layout.shape)
        if layout.scalar or left > 62:
            return _FlatPlan()
        # Axes the outer levels took, as a bitmask over the layout's axes.
        var used = 0
        for l in range(depth):
            ref axes = levels[l].in_axes
            var axis = axes.axis if axes.uniform else axes.values[k]
            if not axis:
                continue
            var a = axis.value()
            if a < -left or a >= left:
                return _FlatPlan()
            var physical = _unused_axis(used, a + left if a < 0 else a)
            used |= 1 << physical
            left -= 1
            var extent = layout.shape[physical]
            if plan.table[l] >= 0 and plan.table[l] != extent:
                return _FlatPlan()
            plan.table[l] = extent
            plan.table[depth + count + k * depth + l] = layout.strides[physical]
        if left and not (batches >> k) & 1:
            return _FlatPlan()
        plan.table[depth + k] = layout.offset
        plan.table[depth + count * (1 + depth) + k] = used
    var total = 1
    for l in range(depth):
        var extent = plan.table[l]
        if extent < 0 or (extent and total > Int.MAX // extent):
            return _FlatPlan()
        total *= extent
    # Stack each result like _stacked_layout does for a scalar leaf: every
    # level placing its axis first keeps level order.
    var ordered = True
    for j in range(results):
        for l in range(depth):
            ref axes = levels[l].out_axes
            var axis = axes.axis if axes.uniform else (axes.values[j] if len(axes.values) == results else None)
            if not axis:
                return _FlatPlan()
            var a = axis.value()
            var rank = depth - l
            if a < -rank or a >= rank:
                return _FlatPlan()
            ordered = ordered and (a == 0 or a == -rank)
    if not ordered:
        for j in range(results):
            var labels = List[Int](capacity=depth)
            for i in range(depth):
                var l = depth - 1 - i
                ref axes = levels[l].out_axes
                var a = (axes.axis if axes.uniform else axes.values[j]).value()
                labels.insert(a + i + 1 if a < 0 else a, l)
            plan.axes.extend(labels^)
    plan.count = total
    return plan^


def _flat_argument[T: ImplicitlyCopyable & Deinitable]() -> Bool:
    """Whether a flat run can pass this parameter: a number, or a batch view."""
    comptime kind = _leaf_kind[T]()
    return kind == _NUMBER or kind == _BATCH


def _batch_parameter[T: ImplicitlyCopyable & Deinitable]() -> Int:
    return 1 if _leaf_kind[T]() == _BATCH else 0


@fieldwise_init
struct _View(Copyable, Movable):
    """What a batch parameter's view keeps of its source: the axes the levels
    leave, and the source's default formats. Each call moves the offset."""
    var layout: _FlatLayout
    var real_format: FloatFormat
    var imag_format: FloatFormat


@fieldwise_init
struct _Run(ImplicitlyCopyable, Movable):
    """A flat input's positions as one run, copied into a chunk's loop.

    Locals stay in registers; the input itself does not, because its next
    element is returned by reference across the mapped function's call.
    """
    var single: Bool
    var offset: Int
    var stride: Int


struct _FlatInput[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable]:
    """One number argument of a flat run; each chunk reads through its own.

    A single run, the common case, reads offset + index * stride. Other
    layouts step: from start(index), step() returns each position in turn and
    moves on along the innermost level, then to the next inner run.
    """
    var elements: Optional[ArcPointer[_Values[_Element[Self.V]]]]
    var single: Bool
    var offset: Int
    var stride: Int
    var position: Int
    var column: Int
    var inner: Int
    # Outer levels of a layout that is not a single run.
    var cursor: List[_Cursor]
    # A batch parameter's view, kept off the loop like the cursor.
    var view: List[_View]
    var held: List[Self.T]
    # Float or Complex elements are lent to one reused argument record.
    comptime _lends = (
        (_Element[Self.V] == Float and Self.T == _FloatArgument)
        or (_Element[Self.V] == Complex and Self.T == _ComplexArgument)
    )

    def __init__(out self, value: Self.V) raises:
        """Own the input for reads by storage position, also used by lift."""
        self.elements = None
        self.single = True
        self.offset = 0
        self.stride = 0
        self.position = 0
        self.column = 0
        self.inner = 0
        self.cursor = List[_Cursor]()
        self.view = List[_View]()
        self.held = List[Self.T]()
        comptime if conforms_to(Self.V, _BatchShape):
            ref batch = rebind[Batch[_Element[Self.V]]](value)
            self.elements = batch._owner
        else:
            comptime assert _leaf_kind[Self.T]() != _BATCH, "A batch argument needs a batch input."
            self.held.append(_scalar_argument[Self.T](value))

    def __init__(out self, value: Self.V, plan: _FlatPlan, argument: Int) raises:
        self = Self(value)
        comptime if conforms_to(Self.V, _BatchShape):
            ref batch = rebind[Batch[_Element[Self.V]]](value)
            comptime if _leaf_kind[Self.T]() == _BATCH:
                comptime assert _Element[Self.V] == _Element[Self.T], "vmap argument and input must have the same numeric element type."
                # The axes the levels did not map, as the general path keeps them.
                var layout = batch._layout[]
                var used = plan.used(argument)
                var rank = len(layout.shape)
                for j in range(rank):
                    # Last axis first, so popping keeps lower indices in place.
                    var axis = rank - 1 - j
                    if (used >> axis) & 1:
                        _ = layout.shape.pop(axis)
                        _ = layout.strides.pop(axis)
                _settle(layout)
                self.view.append(_View(layout^, batch._real_format, batch._imag_format))
            var cursor = plan.cursor(argument)
            self.offset = cursor.offset
            self.stride = cursor.stride
            # An empty run reads nothing, and its extents may be zero.
            if len(cursor.outer_shape) and plan.count > 0:
                self.single = False
                self.inner = cursor.inner
                self.cursor.append(cursor^)
    def __deinit__(deinit self):
        # Return lent significand slots to inline zeros before they are destroyed.
        comptime if conforms_to(Self.V, _BatchShape) and Self._lends:
            if len(self.held):
                comptime if Self.T == _FloatArgument:
                    ref record = Pointer(to=self.held[0]).unsafe_bitcast[_FloatArgument]()[]
                    _lend_integer(record.value.numerator, Integer(0))
                else:
                    ref record = Pointer(to=self.held[0]).unsafe_bitcast[_ComplexArgument]()[]
                    _lend_integer(record.real.value.numerator, Integer(0))
                    _lend_integer(record.imag.value.numerator, Integer(0))

    def run(self) -> _Run:
        return _Run(self.single, self.offset, self.stride)

    def start(mut self, index: Int):
        """Make step() begin at index; a single run needs no state."""
        if not self.single:
            ref cursor = self.cursor[0]
            cursor._seek(index)
            self.column = cursor.column
            self.position = cursor.base + cursor.column * self.stride

    @always_inline
    def step(mut self) -> Int:
        var position = self.position
        self.position += self.stride
        self.column += 1
        if self.column == self.inner:
            ref cursor = self.cursor[0]
            cursor._carry()
            self.column = 0
            self.position = cursor.base
        return position

    @always_inline
    def get(mut self, run: _Run, index: Int) raises -> ref[ImmOrigin(origin_of(self))] Self.T:
        """The element at index, when chunks read indices in order from start."""
        return self.at(run.offset + index * run.stride if run.single else self.step())

    def read(mut self, index: Int) raises -> ref[ImmOrigin(origin_of(self))] Self.T:
        """The element at index, out of order: for repeating one call."""
        self.start(index)
        return self.get(self.run(), index)

    @always_inline
    def at(mut self, position: Int) raises -> ref[ImmOrigin(origin_of(self))] Self.T:
        """The argument for the element at a storage position: the element
        itself, a converted copy, or for a batch parameter a view starting there."""
        comptime if not conforms_to(Self.V, _BatchShape):
            return _borrow_values[ImmOrigin(origin_of(self))](self.held)[0]
        elif _leaf_kind[Self.T]() == _BATCH:
            ref view = self.view[0]
            var layout = view.layout
            layout.offset = position
            var owner = rebind[ArcPointer[_Values[_Element[Self.T]]]](self.elements.value())
            var batch = Batch[_Element[Self.T]](_owner=owner^, _layout=layout^)
            _set_formats(batch, view.real_format, view.imag_format)
            if len(self.held):
                self.held[0] = rebind_var[Self.T](batch^)
            else:
                self.held.append(rebind_var[Self.T](batch^))
            return _borrow_values[ImmOrigin(origin_of(self))](self.held)[0]
        else:
            comptime if _Element[Self.V] == Self.T:
                return rebind[Self.T](_borrow_values[ImmOrigin(origin_of(self))](self.elements.value()[].list)[position])
            else:
                comptime assert _Element[Self.V] != Complex or Self.T == _ComplexArgument, (
                    "A real function cannot take Complex elements; map the"
                    " Complex function instead, for example apn_mojo.complex.sqrt."
                )
                comptime if Self._lends:
                    ref element = self.elements.value()[].list[position]
                    comptime if Self.T == _FloatArgument:
                        if not len(self.held):
                            self.held.append(rebind_var[Self.T](_lent_float()))
                        ref record = Pointer(to=self.held[0]).unsafe_bitcast[_FloatArgument]()[]
                        _lend_float(record, rebind[Float](element))
                    else:
                        if not len(self.held):
                            self.held.append(rebind_var[Self.T](_ComplexArgument(_lent_float())))
                        ref record = Pointer(to=self.held[0]).unsafe_bitcast[_ComplexArgument]()[]
                        ref value = rebind[Complex](element)
                        _lend_float(record.real, value._real)
                        _lend_float(record.imag, value._imag)
                        record.complex = True
                    return _borrow_values[ImmOrigin(origin_of(self))](self.held)[0]
                var converted = _scalar_argument[Self.T](self.elements.value()[].list[position])
                if len(self.held):
                    self.held[0] = converted^
                else:
                    self.held.append(converted^)
                return _borrow_values[ImmOrigin(origin_of(self))](self.held)[0]


def _flat_failure(index: Int, error: Error, shape: List[Int] = List[Int]()) -> Error:
    """The general path's message: one mapped index per level, outermost first."""
    if len(shape) <= 1:
        return Error(String("vmap failed at mapped index ", index, ": ", error))
    var message = String(error)
    var rest = index
    for j in range(len(shape)):
        var axis = len(shape) - 1 - j
        message = String("vmap failed at mapped index ", rest % shape[axis], ": ", message)
        rest //= shape[axis]
    return Error(message)


# Shared orchestration; the wrappers below only adapt callback signatures.


def _nest_levels(
    levels: List[_Level], var in_axes: _AxisLevels, var out_axes: _AxisLevels,
    axis_size: Optional[Int], var out_shape: _Shapes,
) -> List[_Level]:
    var outer = _levels(in_axes^, out_axes^, axis_size, out_shape^)
    for level in levels:
        outer.append(level.copy())
    return outer^


def _run_map1[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, V: _MapArgument & ImplicitlyCopyable & Deinitable,
    function: def(T, /, *, context: C) raises thin -> Results.Value,
](levels: List[_Level], value: V, context: C) raises -> Results.Output:
    comptime if _flat_argument[T]() and Results.flat:
        var plan = _flat_plan1[T, V](levels, value, Results.count)
        if plan.count >= 0:
            return _flat_run1[T, C, Results, V, function](value, plan, context)
    var kinds = List[Int]()
    var execution = _prepare(_single_source[T](value, kinds), levels, kinds)
    var collectors = Results.collectors()
    _check_outputs(collectors, levels, execution.counts)
    var a = _Reader[T, V](execution.sources[0], value)
    for _ in range(execution.total):
        try:
            Results.collect(collectors, function(
                a.read(execution.sources[0], execution.offsets[0], value), context=context,
            ))
        except error:
            raise execution.failure(error)
        execution.advance()
    return Results.finish(collectors^, levels, execution.counts)


def _run_map2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, V: _MapArgument & ImplicitlyCopyable & Deinitable, W: _MapArgument & ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> Results.Value,
](levels: List[_Level], first: V, second: W, context: C) raises -> Results.Output:
    comptime if _flat_argument[T]() and _flat_argument[U]() and Results.flat:
        var plan = _flat_plan2[T, U](levels, first, second, Results.count)
        if plan.count >= 0:
            return _flat_run2[T, U, C, Results, V, W, function](first, second, plan, context)
    var kinds = List[Int]()
    var execution = _prepare(_pair_sources[T, U](first, second, kinds), levels, kinds)
    var collectors = Results.collectors()
    _check_outputs(collectors, levels, execution.counts)
    var a = _Reader[T, V](execution.sources[0], first)
    var b = _Reader[U, W](execution.sources[1], second)
    for _ in range(execution.total):
        try:
            Results.collect(collectors, function(
                a.read(execution.sources[0], execution.offsets[0], first),
                b.read(execution.sources[1], execution.offsets[1], second),
                context=context,
            ))
        except error:
            raise execution.failure(error)
        execution.advance()
    return Results.finish(collectors^, levels, execution.counts)


def _run_map3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable, D: _MapArgument & ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, /, *, context: C) raises thin -> Results.Value,
](levels: List[_Level], first: A, second: B, third: D, context: C) raises -> Results.Output:
    comptime if _flat_argument[T]() and _flat_argument[U]() and _flat_argument[W]() and Results.flat:
        var plan = _flat_plan3[T, U, W](levels, first, second, third, Results.count)
        if plan.count >= 0:
            return _flat_run3[T, U, W, C, Results, A, B, D, function](first, second, third, plan, context)
    var kinds = List[Int]()
    var execution = _prepare(_triple_sources[T, U, W](first, second, third, kinds), levels, kinds)
    var collectors = Results.collectors()
    _check_outputs(collectors, levels, execution.counts)
    var a = _Reader[T, A](execution.sources[0], first)
    var b = _Reader[U, B](execution.sources[1], second)
    var c = _Reader[W, D](execution.sources[2], third)
    for _ in range(execution.total):
        try:
            Results.collect(collectors, function(
                a.read(execution.sources[0], execution.offsets[0], first),
                b.read(execution.sources[1], execution.offsets[1], second),
                c.read(execution.sources[2], execution.offsets[2], third),
                context=context,
            ))
        except error:
            raise execution.failure(error)
        execution.advance()
    return Results.finish(collectors^, levels, execution.counts)


def _quad_sources[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable, D: ImplicitlyCopyable & Deinitable, E: ImplicitlyCopyable & Deinitable,
](first: A, second: B, third: D, fourth: E, mut kinds: List[Int]) raises -> List[_Source]:
    var sources = List[_Source](capacity=4)
    sources.append(_source_of(first))
    sources.append(_source_of(second))
    sources.append(_source_of(third))
    sources.append(_source_of(fourth))
    kinds.append(_leaf_kind[T]())
    kinds.append(_leaf_kind[U]())
    kinds.append(_leaf_kind[W]())
    kinds.append(_leaf_kind[X]())
    return sources^


def _run_map4[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, A: _MapArgument & ImplicitlyCopyable & Deinitable,
    B: _MapArgument & ImplicitlyCopyable & Deinitable, D: _MapArgument & ImplicitlyCopyable & Deinitable, E: _MapArgument & ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, X, /, *, context: C) raises thin -> Results.Value,
](levels: List[_Level], first: A, second: B, third: D, fourth: E, context: C) raises -> Results.Output:
    """Four arguments take the general path only: such functions, like
    hyp2f1, cost far more per call than the flat loops save."""
    var kinds = List[Int]()
    var execution = _prepare(_quad_sources[T, U, W, X](first, second, third, fourth, kinds), levels, kinds)
    var collectors = Results.collectors()
    _check_outputs(collectors, levels, execution.counts)
    var a = _Reader[T, A](execution.sources[0], first)
    var b = _Reader[U, B](execution.sources[1], second)
    var c = _Reader[W, D](execution.sources[2], third)
    var d = _Reader[X, E](execution.sources[3], fourth)
    for _ in range(execution.total):
        try:
            Results.collect(collectors, function(
                a.read(execution.sources[0], execution.offsets[0], first),
                b.read(execution.sources[1], execution.offsets[1], second),
                c.read(execution.sources[2], execution.offsets[2], third),
                d.read(execution.sources[3], execution.offsets[3], fourth),
                context=context,
            ))
        except error:
            raise execution.failure(error)
        execution.advance()
    return Results.finish(collectors^, levels, execution.counts)


def _run_tuple_map[
    T: _TupleArgument & ImplicitlyCopyable & Deinitable, Results: _MappingResults, As: type_of(Tuple),
    function: def(T, /, *, context: Bool) raises thin -> Results.Value, *Vs: ImplicitlyCopyable & Deinitable,
](levels: List[_Level], values: Tuple[*Vs]) raises -> Results.Output:
    comptime n = As.Ts.length
    comptime if len(Vs) == n and _flat_tuple_leaves[As]() and Results.flat:
        comptime A = _TupleLeaf[As.Ts[0]]
        comptime if n == 1:
            var plan = _flat_plan1[A, Vs[0]](levels, values[0], Results.count)
            if plan.count >= 0:
                return _flat_run1[A, Bool, Results, Vs[0], _untuple1[T, A, Results.Value, function]](values[0], plan, False)
        elif n == 2:
            comptime B = _TupleLeaf[As.Ts[1]]
            var plan = _flat_plan2[A, B](levels, values[0], values[1], Results.count)
            if plan.count >= 0:
                return _flat_run2[A, B, Bool, Results, Vs[0], Vs[1], _untuple2[T, A, B, Results.Value, function]](values[0], values[1], plan, False)
        else:
            comptime B = _TupleLeaf[As.Ts[1]]
            comptime C = _TupleLeaf[As.Ts[2]]
            var plan = _flat_plan3[A, B, C](levels, values[0], values[1], values[2], Results.count)
            if plan.count >= 0:
                return _flat_run3[A, B, C, Bool, Results, Vs[0], Vs[1], Vs[2], _untuple3[T, A, B, C, Results.Value, function]](values[0], values[1], values[2], plan, False)
    var kinds = List[Int]()
    var execution = _prepare(_tuple_sources[As](values, kinds), levels, kinds)
    var collectors = Results.collectors()
    _check_outputs(collectors, levels, execution.counts)
    for _ in range(execution.total):
        try:
            Results.collect(collectors, function(rebind_var[T](_tuple_arguments[As](execution, values)), context=False))
        except error:
            raise execution.failure(error)
        execution.advance()
    return Results.finish(collectors^, levels, execution.counts)


# ---------------------------------------------------------------- mappings
# One struct per callback signature and result form; nesting adds levels.


struct _Map[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    function: def(T) raises thin -> R,
](Copyable, Movable):
    """A one-argument callback returning one leaf."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable](self, value: V) raises -> Self.Output:
        return _run_map1[Self.T, Bool, _LeafResults[Self.R], V, _without_context[Self.T, Self.R, Self.function]](self.levels, value, False)


struct _TupleResultMap[
    T: ImplicitlyCopyable & Deinitable, R: type_of(Tuple),
    function: def(T) raises thin -> R,
](Copyable, Movable):
    """A one-argument callback returning a tuple of leaves."""
    comptime Output = _Outputs[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable](self, value: V) raises -> Self.Output:
        return _run_map1[Self.T, Bool, _TupleResults[Self.R], V, _tuple_without_context[Self.T, Self.R, Self.function]](self.levels, value, False)


struct _TupleArgumentMap[
    T: _TupleArgument & ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    function: def(T) raises thin -> R, As: type_of(Tuple),
](Copyable, Movable):
    """A callback taking a flat tuple (leaves As) and returning one leaf."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[*Vs: ImplicitlyCopyable & Deinitable](self, values: Tuple[*Vs]) raises -> Self.Output:
        return _run_tuple_map[Self.T, _LeafResults[Self.R], Self.As, _without_context[Self.T, Self.R, Self.function]](self.levels, values)


struct _TupleArgumentResultMap[
    T: _TupleArgument & ImplicitlyCopyable & Deinitable, R: type_of(Tuple),
    function: def(T) raises thin -> R, As: type_of(Tuple),
](Copyable, Movable):
    """A callback taking a flat tuple (leaves As) and returning a tuple of leaves."""
    comptime Output = _Outputs[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[*Vs: ImplicitlyCopyable & Deinitable](self, values: Tuple[*Vs]) raises -> Self.Output:
        return _run_tuple_map[Self.T, _TupleResults[Self.R], Self.As, _tuple_without_context[Self.T, Self.R, Self.function]](self.levels, values)


struct _Map2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable,
    R: ImplicitlyCopyable & Deinitable, function: def(T, U) raises thin -> R,
](Copyable, Movable):
    """A two-argument callback returning one leaf; each argument may be mapped or shared."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable, W: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, first: V, second: W,
    ) raises -> Self.Output:
        return _run_map2[Self.T, Self.U, Bool, _LeafResults[Self.R], V, W, _without_context2[Self.T, Self.U, Self.R, Self.function]](self.levels, first, second, False)


struct _TupleResultMap2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable,
    R: type_of(Tuple), function: def(T, U) raises thin -> R,
](Copyable, Movable):
    """A two-argument callback returning a tuple of leaves."""
    comptime Output = _Outputs[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable, W: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, first: V, second: W,
    ) raises -> Self.Output:
        return _run_map2[Self.T, Self.U, Bool, _TupleResults[Self.R], V, W, _tuple_without_context2[Self.T, Self.U, Self.R, Self.function]](self.levels, first, second, False)


struct _MapC[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: ImplicitlyCopyable & Deinitable, function: def(T, /, *, context: C) raises thin -> R,
](Copyable, Movable):
    """A one-argument callback with a context keyword, passed on every call."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, value: V, *, context: Self.C = Self.C(),
    ) raises -> Self.Output:
        return _run_map1[Self.T, Self.C, _LeafResults[Self.R], V, Self.function](self.levels, value, context)


struct _TupleResultMapC[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable,
    R: type_of(Tuple), function: def(T, /, *, context: C) raises thin -> R,
](Copyable, Movable):
    """A one-argument callback with a context keyword that returns a tuple of leaves."""
    comptime Output = _Outputs[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, value: V, *, context: Self.C = Self.C(),
    ) raises -> Self.Output:
        return _run_map1[Self.T, Self.C, _TupleResults[Self.R], V, Self.function](self.levels, value, context)


struct _Map2C[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> R,
](Copyable, Movable):
    """A two-argument callback with a context keyword, passed on every call."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[V: _MapArgument & ImplicitlyCopyable & Deinitable, W: _MapArgument & ImplicitlyCopyable & Deinitable](
        self, first: V, second: W, *, context: Self.C = Self.C(),
    ) raises -> Self.Output:
        return _run_map2[Self.T, Self.U, Self.C, _LeafResults[Self.R], V, W, Self.function](self.levels, first, second, context)


struct _Map3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    R: ImplicitlyCopyable & Deinitable, function: def(T, U, W, /) raises thin -> R,
](Copyable, Movable):
    """A three-argument callback returning one leaf."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[
        A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
        D: _MapArgument & ImplicitlyCopyable & Deinitable,
    ](self, first: A, second: B, third: D) raises -> Self.Output:
        return _run_map3[Self.T, Self.U, Self.W, Bool, _LeafResults[Self.R], A, B, D, _without_context3[Self.T, Self.U, Self.W, Self.R, Self.function]](self.levels, first, second, third, False)


struct _Map3C[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, /, *, context: C) raises thin -> R,
](Copyable, Movable):
    """A three-argument callback with a context keyword, passed on every call."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[
        A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
        D: _MapArgument & ImplicitlyCopyable & Deinitable,
    ](self, first: A, second: B, third: D, *, context: Self.C = Self.C()) raises -> Self.Output:
        return _run_map3[Self.T, Self.U, Self.W, Self.C, _LeafResults[Self.R], A, B, D, Self.function](self.levels, first, second, third, context)


struct _Map4[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    X: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, function: def(T, U, W, X, /) raises thin -> R,
](Copyable, Movable):
    """A four-argument callback returning one leaf."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[
        A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
        D: _MapArgument & ImplicitlyCopyable & Deinitable, E: _MapArgument & ImplicitlyCopyable & Deinitable,
    ](self, first: A, second: B, third: D, fourth: E) raises -> Self.Output:
        return _run_map4[Self.T, Self.U, Self.W, Self.X, Bool, _LeafResults[Self.R], A, B, D, E, _without_context4[Self.T, Self.U, Self.W, Self.X, Self.R, Self.function]](self.levels, first, second, third, fourth, False)


struct _Map4C[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    X: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, X, /, *, context: C) raises thin -> R,
](Copyable, Movable):
    """A four-argument callback with a context keyword, passed on every call."""
    comptime Output = _LeafOutput[Self.R]
    var levels: List[_Level]

    def __init__(out self, var levels: List[_Level]):
        self.levels = levels^

    def vmap(
        var self, *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
        axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
    ) -> Self:
        """Map this mapped function over one more outer axis (or several, listed innermost first)."""
        self.levels = _nest_levels(self.levels, in_axes^, out_axes^, axis_size, out_shape^)
        return self^

    def __call__[
        A: _MapArgument & ImplicitlyCopyable & Deinitable, B: _MapArgument & ImplicitlyCopyable & Deinitable,
        D: _MapArgument & ImplicitlyCopyable & Deinitable, E: _MapArgument & ImplicitlyCopyable & Deinitable,
    ](self, first: A, second: B, third: D, fourth: E, *, context: Self.C = Self.C()) raises -> Self.Output:
        return _run_map4[Self.T, Self.U, Self.W, Self.X, Self.C, _LeafResults[Self.R], A, B, D, E, Self.function](self.levels, first, second, third, fourth, context)


# ---------------------------------------------------------------- vmap


# ------------------------------------------------- parallel flat runs
# A compile-time function can run on worker threads (Mojo 1.1 cannot call a
# runtime function value from another thread). Flat calls - one level, rank-1
# batches and shared scalars, number results - split long runs over the pool.


@fieldwise_init
struct _ParallelArgs[
    V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable,
](Copyable, Movable):
    var first: Self.V
    var second: Self.W
    var third: Self.X
    var context: Self.C
    var plan: _FlatPlan


def _without_context[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, function: def(T, /) raises thin -> R,
](x: T, /, *, context: Bool) raises -> R:
    return function(x)


def _without_context2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, function: def(T, U, /) raises thin -> R,
](x: T, y: U, /, *, context: Bool) raises -> R:
    return function(x, y)


def _without_context4[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    R: ImplicitlyCopyable & Deinitable, function: def(T, U, W, X, /) raises thin -> R,
](x: T, y: U, z: W, v: X, /, *, context: Bool) raises -> R:
    return function(x, y, z, v)


def _without_context3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, function: def(T, U, W, /) raises thin -> R,
](x: T, y: U, z: W, /, *, context: Bool) raises -> R:
    return function(x, y, z)


struct _FlatKernel1[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable,
    function: def(T, /, *, context: C) raises thin -> R,
](_ElementKernel):
    comptime Element = Self.R
    comptime Job = _ParallelArgs[Self.V, Bool, Bool, Self.C]

    var a: _FlatInput[Self.T, Self.V]
    var ra: _Run

    def __init__(out self, job: Self.Job, begin: Int) raises:
        self.a = _FlatInput[Self.T, Self.V](job.first, job.plan, 0)
        self.a.start(begin)
        self.ra = self.a.run()

    @always_inline
    def apply(mut self, job: Self.Job, index: Int) raises -> Self.R:
        return Self.function(self.a.get(self.ra, index), context=job.context)


struct _FlatKernel2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> R,
](_ElementKernel):
    comptime Element = Self.R
    comptime Job = _ParallelArgs[Self.V, Self.X, Bool, Self.C]

    var a: _FlatInput[Self.T, Self.V]
    var ra: _Run
    var b: _FlatInput[Self.U, Self.X]
    var rb: _Run

    def __init__(out self, job: Self.Job, begin: Int) raises:
        self.a = _FlatInput[Self.T, Self.V](job.first, job.plan, 0)
        self.b = _FlatInput[Self.U, Self.X](job.second, job.plan, 1)
        self.a.start(begin)
        self.ra = self.a.run()
        self.b.start(begin)
        self.rb = self.b.run()

    @always_inline
    def apply(mut self, job: Self.Job, index: Int) raises -> Self.R:
        return Self.function(self.a.get(self.ra, index), self.b.get(self.rb, index), context=job.context)


struct _FlatKernel3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, /, *, context: C) raises thin -> R,
](_ElementKernel):
    comptime Element = Self.R
    comptime Job = _ParallelArgs[Self.V, Self.X, Self.Y, Self.C]

    var a: _FlatInput[Self.T, Self.V]
    var ra: _Run
    var b: _FlatInput[Self.U, Self.X]
    var rb: _Run
    var c: _FlatInput[Self.W, Self.Y]
    var rc: _Run

    def __init__(out self, job: Self.Job, begin: Int) raises:
        self.a = _FlatInput[Self.T, Self.V](job.first, job.plan, 0)
        self.b = _FlatInput[Self.U, Self.X](job.second, job.plan, 1)
        self.c = _FlatInput[Self.W, Self.Y](job.third, job.plan, 2)
        self.a.start(begin)
        self.ra = self.a.run()
        self.b.start(begin)
        self.rb = self.b.run()
        self.c.start(begin)
        self.rc = self.c.run()

    @always_inline
    def apply(mut self, job: Self.Job, index: Int) raises -> Self.R:
        return Self.function(self.a.get(self.ra, index), self.b.get(self.rb, index), self.c.get(self.rc, index), context=job.context)


@fieldwise_init
struct _FlatErrors[origin: Origin[mut=False]](_ElementErrors):
    var plan: Pointer[_FlatPlan, Self.origin]

    def failure(self, index: Int, var error: Error) -> Error:
        return _flat_failure(index, error, self.plan[].shape())

    def repeated(self, index: Int) -> Error:
        return Error(String(
            "vmap element ", index, " failed on a worker thread but not when"
            " repeated; mapped functions must be pure.",
        ))


def _flat_plan1[T: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable](
    levels: List[_Level], first: V, results: Int = 1,
) raises -> _FlatPlan:
    var plan = _plan(levels, [_layout_of(first)], results, _batch_parameter[T]())
    if plan.count >= 0:
        # Scalar arguments convert once, before any call, as before.
        _ = _FlatInput[T, V](first, plan, 0)
    return plan^


def _flat_plan2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable,
    V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
](levels: List[_Level], first: V, second: W, results: Int = 1) raises -> _FlatPlan:
    var plan = _plan(levels, [_layout_of(first), _layout_of(second)], results, _batch_parameter[T]() | _batch_parameter[U]() << 1)
    if plan.count >= 0:
        _ = _FlatInput[T, V](first, plan, 0)
        _ = _FlatInput[U, W](second, plan, 1)
    return plan^


def _flat_plan3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    V: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable,
](levels: List[_Level], first: V, second: X, third: Y, results: Int = 1) raises -> _FlatPlan:
    var batches = _batch_parameter[T]() | _batch_parameter[U]() << 1 | _batch_parameter[W]() << 2
    var plan = _plan(levels, [_layout_of(first), _layout_of(second), _layout_of(third)], results, batches)
    if plan.count >= 0:
        _ = _FlatInput[T, V](first, plan, 0)
        _ = _FlatInput[U, X](second, plan, 1)
        _ = _FlatInput[W, Y](third, plan, 2)
    return plan^


def _flat_values1[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable,
    function: def(T, /, *, context: C) raises thin -> R,
](first: V, plan: _FlatPlan, context: C) raises -> _Values[R]:
    """Every call's result in order, or the error the sequential loop raises first."""
    return _execute_values[_FlatKernel1[T, C, R, V, function]](
        _ParallelArgs[V, Bool, Bool, C](first, False, False, context, plan.copy()),
        plan.count, _FlatErrors(Pointer(to=plan)), probe=True,
    )


def _flat_values2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> R,
](first: V, second: W, plan: _FlatPlan, context: C) raises -> _Values[R]:
    return _execute_values[_FlatKernel2[T, U, C, R, V, W, function]](
        _ParallelArgs[V, W, Bool, C](first, second, False, context, plan.copy()),
        plan.count, _FlatErrors(Pointer(to=plan)), probe=True,
    )


def _flat_values3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, V: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, /, *, context: C) raises thin -> R,
](first: V, second: X, third: Y, plan: _FlatPlan, context: C) raises -> _Values[R]:
    return _execute_values[_FlatKernel3[T, U, W, C, R, V, X, Y, function]](
        _ParallelArgs[V, X, Y, C](first, second, third, context, plan.copy()),
        plan.count, _FlatErrors(Pointer(to=plan)), probe=True,
    )


def _flat_leaf[R: ImplicitlyCopyable & Deinitable](
    var values: _Values[R], like: Optional[ArcPointer[_FlatLayout]], plan: _FlatPlan, result: Int,
) raises -> _LeafOutput[R]:
    """A flat run's results as one output, a batch of numbers or a Mask of
    Bools: one axis per level, in level order, then ordered by out_axes."""
    var axes = plan.order(result)
    comptime if R == Bool:
        var bits = rebind_var[List[Bool]](values^.take_list())
        var mask = Mask(bits, shape=plan.shape()) if plan.depth > 1 else Mask(bits)
        if len(axes):
            return rebind_var[_LeafOutput[R]](mask.transpose(axes))
        return rebind_var[_LeafOutput[R]](mask^)
    else:
        var count = len(values)
        var batch = Batch[R](_tensor=_Tensor[R](values^, [count]), _like=like)
        if plan.depth > 1:
            batch = batch.reshape(plan.shape())
        if len(axes):
            return rebind_var[_LeafOutput[R]](batch.transpose(axes))
        return rebind_var[_LeafOutput[R]](batch^)


def _flat_run1[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, V: ImplicitlyCopyable & Deinitable,
    function: def(T, /, *, context: C) raises thin -> Results.Value,
](first: V, plan: _FlatPlan, context: C) raises -> Results.Output:
    return Results.finish_flat(_flat_values1[T, C, Results.Value, V, function](first, plan, context), _layout_of(first), plan)


def _flat_run2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, V: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    function: def(T, U, /, *, context: C) raises thin -> Results.Value,
](first: V, second: W, plan: _FlatPlan, context: C) raises -> Results.Output:
    var values = _flat_values2[T, U, C, Results.Value, V, W, function](first, second, plan, context)
    return Results.finish_flat(values^, _result_layout(first, second), plan)


def _flat_run3[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable, Results: _MappingResults, V: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, Y: ImplicitlyCopyable & Deinitable,
    function: def(T, U, W, /, *, context: C) raises thin -> Results.Value,
](first: V, second: X, third: Y, plan: _FlatPlan, context: C) raises -> Results.Output:
    var values = _flat_values3[T, U, W, C, Results.Value, V, X, Y, function](first, second, third, plan, context)
    return Results.finish_flat(values^, _result_layout(first, second) or _layout_of(third), plan)


# Tuple results run flat as tuples, then move each field into its own output.

comptime _Leaves[Rs: type_of(Tuple)] = downcast[Rs, ImplicitlyCopyable & Deinitable]


def _flat_result[R: ImplicitlyCopyable & Deinitable]() -> Bool:
    """Whether a flat run can collect a result: a number, or a Bool into a Mask."""
    return _leaf_kind[R]() == _NUMBER or R == Bool


def _flat_results[Rs: type_of(Tuple)]() -> Bool:
    """Whether every leaf of a tuple result can be collected by a flat run."""
    comptime if Rs.Ts.length == 0:
        return False
    comptime for j in range(Rs.Ts.length):
        comptime if _leaf_kind[_TupleLeaf[Rs.Ts[j]]]() != _NUMBER and _TupleLeaf[Rs.Ts[j]] != Bool:
            return False
    return True


# An Optional result maps to two leaves: its value, or the family's zero
# where there is none, and a Bool that the Mask of present values collects.

comptime _Present[X: ImplicitlyCopyable & Deinitable] = conforms_to(X, _BatchElement)


def _present[
    T: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    function: def(T) raises thin -> Optional[X],
](x: T) raises -> Tuple[X, Bool]:
    var result = function(x)
    if result:
        return (result.take(), True)
    return (_placeholder[X](), False)


def _present2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable,
    function: def(T, U) raises thin -> Optional[X],
](x: T, y: U) raises -> Tuple[X, Bool]:
    var result = function(x, y)
    if result:
        return (result.take(), True)
    return (_placeholder[X](), False)


def _tuple_without_context[
    T: ImplicitlyCopyable & Deinitable, Rs: type_of(Tuple), function: def(T) raises thin -> Rs,
](x: T, /, *, context: Bool) raises -> _Leaves[Rs]:
    return rebind_var[_Leaves[Rs]](function(x))


def _tuple_without_context2[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, Rs: type_of(Tuple), function: def(T, U) raises thin -> Rs,
](x: T, y: U, /, *, context: Bool) raises -> _Leaves[Rs]:
    return rebind_var[_Leaves[Rs]](function(x, y))


def _placeholder[R: ImplicitlyCopyable & Deinitable]() -> R:
    """A value with no heap storage, left behind in a tuple when its field
    moves out, or standing for an absent Optional result."""
    comptime if R == Bool:
        return rebind_var[R](False)
    else:
        comptime assert conforms_to(R, _BatchElement), "Flat tuple results hold numbers and Bools."
        return rebind_var[R](downcast[R, _BatchElement]._placeholder())


struct _FlatLeaves[Rs: type_of(Tuple)](Movable):
    """A flat run's tuples, split one field at a time into separate outputs."""
    var tuples: List[_Leaves[Self.Rs]]
    var ownership: Optional[_Ownership]
    var like: Optional[ArcPointer[_FlatLayout]]
    var plan: _FlatPlan

    def __init__(out self, var values: _Values[_Leaves[Self.Rs]], like: Optional[ArcPointer[_FlatLayout]], plan: _FlatPlan):
        # Every field keeps the run's record of who computed each element.
        self.ownership = values.ownership.copy()
        self.tuples = values^.take_list()
        self.like = like
        self.plan = plan.copy()

    def leaf[R: ImplicitlyCopyable & Deinitable, j: Int](mut self) raises -> _LeafOutput[R]:
        # Moving, not copying, leaves reference counts alone; the placeholders
        # left behind have no storage to release with the tuples.
        var blank = _placeholder[R]()
        var values = List[R](capacity=len(self.tuples))
        for i in range(len(self.tuples)):
            ref leaves = rebind[Tuple[*Self.Rs.Ts]](self.tuples[i])
            var value = blank
            swap(value, rebind[R](leaves[j]))
            values.append(value^)
        return _flat_leaf[R](_Values[R](values^, self.ownership.copy()), self.like, self.plan, j)


# A function of a flat tuple with up to three leaves runs flat as a function of
# its leaves; each call rebuilds the tuple from copies, as _tuple_arguments does.


def _flat_tuple_leaves[As: type_of(Tuple)]() -> Bool:
    """Whether a tuple argument's leaves can run flat: one to three numbers or batches."""
    comptime n = As.Ts.length
    comptime if n == 0 or n > 3:
        return False
    comptime for j in range(n):
        comptime if not _flat_argument[_TupleLeaf[As.Ts[j]]]():
            return False
    return True


def _untuple1[
    T: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable,
    function: def(T, /, *, context: Bool) raises thin -> R,
](a: A, /, *, context: Bool) raises -> R:
    return function(rebind_var[T](Tuple(a)), context=context)


def _untuple2[
    T: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable,
    R: ImplicitlyCopyable & Deinitable, function: def(T, /, *, context: Bool) raises thin -> R,
](a: A, b: B, /, *, context: Bool) raises -> R:
    return function(rebind_var[T]((a, b)), context=context)


def _untuple3[
    T: ImplicitlyCopyable & Deinitable, A: ImplicitlyCopyable & Deinitable, B: ImplicitlyCopyable & Deinitable,
    C: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, function: def(T, /, *, context: Bool) raises thin -> R,
](a: A, b: B, c: C, /, *, context: Bool) raises -> R:
    return function(rebind_var[T]((a, b, c)), context=context)


# ------------------------------------------------- public vmap
# vmap[f](...) returns the mapped function. Options describe its level; a list
# such as in_axes=[0, 1] gives one entry per level, innermost first, and
# mapped.vmap(...) adds outer levels.


def vmap[
    T: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //, function: def(T) raises thin -> R,
](
    *__disambiguate: NoneType, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map[T, R, function]:
    """Map a scalar function over batches.

    `vmap[f]()` returns a mapped function: `var mapped = vmap[f](); mapped(xs)`.
    Each argument is a scalar or a batch. A mapped argument passes one slice along
    its mapped axis to each call, and a scalar is shared by every call; numeric
    results stack into a batch, Boolean results into a `Mask`, and batch results
    into a batch with one more axis. A function of one or two arguments that
    returns an Optional number gives two results: the values, with
    0 where there is none, and a `Mask` of where there is one. Name a family's single declaration, such as
    `vmap[apn_mojo.float.add]`; the package-level `add` is an overload set. Mapped
    lengths must agree; a length-one batch is a sequence, not a scalar. Long
    mappings with number or Bool results, tuples of them included, run on the
    worker pool with the same results and errors.

    Parameters:
        T: The function's argument type, inferred.
        R: The function's result type, inferred.
        function: The scalar function: a named `def` or a lambda that captures
            nothing.

    Args:
        __disambiguate: Unused; it makes the arguments after it keyword-only.
        in_axes: The mapped axis of each argument, or `None` to share it; a list
            sets one entry per nesting level, innermost first.
        out_axes: Where the new axis goes in each result.
        axis_size: The mapped length when no argument is mapped.
        out_shape: The per-call result shape, for empty mappings of batch results.

    Returns:
        The mapped function. `.vmap(...)` on it adds an outer layer.

    Limitations:
        The function cannot capture local variables; pass shared values as
        arguments or `context=`. Functions take one to four positional
        arguments, optionally with a keyword-only `context`, or one flat tuple.
    """
    comptime assert _ArgumentLeaf[T], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map[T, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, R: type_of(Tuple), //, function: def(T) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleResultMap[T, R, function] where (_MappingLeaf[T]):
    """Map a unary function that returns a tuple; each leaf collects separately."""
    comptime assert _ArgumentLeaf[T], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    _check_leaves[R]()
    return _TupleResultMap[T, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, //,
    function: def(T) raises thin -> Optional[X],
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleResultMap[T, Tuple[X, Bool], _present[T, X, function]] where (_MappingLeaf[T] and _Present[X]):
    """Map a unary function that returns an Optional number: the values (0 where None) and a Mask of where they exist."""
    return _TupleResultMap[T, Tuple[X, Bool], _present[T, X, function]](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: _TupleArgument & ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T) raises thin -> R, *Ts: ImplicitlyCopyable & Deinitable,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleArgumentMap[T, R, function, Tuple[*Ts]] where (T == Tuple[*Ts] and _MappingLeaf[R]):
    """Map a function of one flat tuple, with one `in_axes` entry per leaf."""
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _TupleArgumentMap[T, R, function, Tuple[*Ts]](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: _TupleArgument & ImplicitlyCopyable & Deinitable, R: type_of(Tuple), //,
    function: def(T) raises thin -> R, *Ts: ImplicitlyCopyable & Deinitable,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleArgumentResultMap[T, R, function, Tuple[*Ts]] where (T == Tuple[*Ts]):
    """Map a function of one flat tuple that returns a tuple."""
    _check_leaves[R]()
    return _TupleArgumentResultMap[T, R, function, Tuple[*Ts]](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //, function: def(T, U) raises thin -> R,
](
    *__disambiguate: NoneType, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map2[T, U, R, function]:
    """Map a binary function."""
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map2[T, U, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, R: type_of(Tuple), //, function: def(T, U) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleResultMap2[T, U, R, function]:
    """Map a binary function that returns a tuple."""
    _check_leaves[R]()
    return _TupleResultMap2[T, U, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, X: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U) raises thin -> Optional[X],
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleResultMap2[T, U, Tuple[X, Bool], _present2[T, U, X, function]] where (_Present[X]):
    """Map a binary function that returns an Optional number: the values (0 where None) and a Mask of where they exist."""
    return _TupleResultMap2[T, U, Tuple[X, Bool], _present2[T, U, X, function]](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, /, *, context: C) raises thin -> R,
](
    *__disambiguate: NoneType, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _MapC[T, C, R, function] where (_Context[C]):
    """Map a unary function with a keyword-only `context`, passed to every call."""
    comptime assert _ArgumentLeaf[T], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _MapC[T, C, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: type_of(Tuple), //,
    function: def(T, /, *, context: C) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _TupleResultMapC[T, C, R, function] where (_Context[C]):
    """Map a unary function with a keyword-only `context` that returns a
    tuple, such as `sici`; each leaf collects separately."""
    comptime assert _ArgumentLeaf[T], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    _check_leaves[R]()
    return _TupleResultMapC[T, C, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, /, *, context: C) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map2C[T, U, C, R, function] where (_Context[C]):
    """Map a binary function with a keyword-only `context`, passed to every call."""
    comptime assert _ArgumentLeaf[T] and _ArgumentLeaf[U], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map2C[T, U, C, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, W, /) raises thin -> R,
](
    *__disambiguate: NoneType, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map3[T, U, W, R, function]:
    """Map a ternary function, such as `fma`."""
    comptime assert _ArgumentLeaf[T] and _ArgumentLeaf[U] and _ArgumentLeaf[W], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map3[T, U, W, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, W, /, *, context: C) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map3C[T, U, W, C, R, function] where (_Context[C]):
    """Map a ternary function with a keyword-only `context`, passed to every call."""
    comptime assert _ArgumentLeaf[T] and _ArgumentLeaf[U] and _ArgumentLeaf[W], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map3C[T, U, W, C, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    X: ImplicitlyCopyable & Deinitable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, W, X, /) raises thin -> R,
](
    *__disambiguate: NoneType, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map4[T, U, W, X, R, function]:
    """Map a four-argument function."""
    comptime assert _ArgumentLeaf[T] and _ArgumentLeaf[U] and _ArgumentLeaf[W] and _ArgumentLeaf[X], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map4[T, U, W, X, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))


def vmap[
    T: ImplicitlyCopyable & Deinitable, U: ImplicitlyCopyable & Deinitable, W: ImplicitlyCopyable & Deinitable,
    X: ImplicitlyCopyable & Deinitable, C: ImplicitlyCopyable & Deinitable & Defaultable, R: ImplicitlyCopyable & Deinitable, //,
    function: def(T, U, W, X, /, *, context: C) raises thin -> R,
](
    *, var in_axes: _AxisLevels = {}, var out_axes: _AxisLevels = {},
    axis_size: Optional[Int] = None, var out_shape: _Shapes = {},
) -> _Map4C[T, U, W, X, C, R, function] where (_Context[C]):
    """Map a four-argument function, such as hyp2f1, with a keyword-only
    `context`, passed to every call."""
    comptime assert _ArgumentLeaf[T] and _ArgumentLeaf[U] and _ArgumentLeaf[W] and _ArgumentLeaf[X], "vmap arguments must be apn_mojo numbers, numeric batches, Mask or Bool."
    comptime assert _MappingLeaf[R], "vmap must return an apn_mojo number, numeric batch, Mask or Bool."
    return _Map4C[T, U, W, X, C, R, function](_levels(in_axes^, out_axes^, axis_size, out_shape^))
