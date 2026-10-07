"""Private scalar-native 1-D value runs; no numeric-family dependencies."""

from std.builtin.builtin_slice import StridedSlice
from std.collections import Array
from std.memory import ArcPointer
from std.sys import size_of
from ..common._sizes import _checked_count
from ._layout import _Layout, _Ints, _index
from ._values import _Values, _borrow_values


trait _TensorSequence(ImplicitlyCopyable):
    comptime Element: ImplicitlyCopyable & Deinitable

    def _tensor_view(self) raises -> _Tensor[Self.Element]: ...




struct _Tensor[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable):
    """A shared immutable value list read through one strided run.

    Batch shapes are runtime metadata (see _array.mojo); storage is always 1-D.
    """
    var _owner: ArcPointer[_Values[Self.T]]
    var _layout: _Layout

    def __init__(out self, var values: List[Self.T], shape: Array[Int, 1]) raises:
        var layout = _Layout(shape)
        _ = _checked_count(layout.size, max(1, size_of[Self.T]()))
        if layout.size != len(values):
            raise Error("Batch shape does not match its values; use dimensions whose product equals the value count.")
        self._owner = ArcPointer(_Values[Self.T](values^))
        self._layout = layout

    def __init__(out self, var values: _Values[Self.T], shape: Array[Int, 1]) raises:
        """A run that remembers which threads computed its values."""
        var layout = _Layout(shape)
        _ = _checked_count(layout.size, max(1, size_of[Self.T]()))
        if layout.size != len(values):
            raise Error("Batch shape does not match its values; use dimensions whose product equals the value count.")
        self._owner = ArcPointer(values^)
        self._layout = layout

    def __init__(out self, owner: ArcPointer[_Values[Self.T]], layout: _Layout):
        self._owner = owner
        self._layout = layout

    @always_inline
    def _read(self, logical: Int) -> ref[self] Self.T:
        return _borrow_values[origin_of(self)](self._owner[].list)[self._layout.position(logical)]

    def item(self, logical: Int) raises -> Self.T:
        return self._read(_index(logical, self._layout.size))

    def slice(self, axis: Int, selection: StridedSlice) raises -> Self:
        return Self(self._owner, self._layout.sliced(axis, selection))

    def to_list(self) -> List[Self.T]:
        var result = List[Self.T](capacity=self._layout.size)
        for i in range(self._layout.size):
            result.append(self._read(i))
        return result^


# Element walkers take rank-erased layouts: one specialization per element type.


def _write_axis[T: ImplicitlyCopyable & Deinitable](
    owner: ArcPointer[_Values[T]], shape: _Ints, strides: _Ints,
    axis: Int, position: Int, mut writer: Some[Writer],
) where conforms_to(T, Writable):
    if axis == len(shape):
        writer.write(owner[][position])
        return
    ref values = owner[]
    var innermost = axis + 1 == len(shape)
    writer.write("[")
    for i in range(shape[axis]):
        if i:
            writer.write(", ")
        var at = position + i * strides[axis]
        # The last axis writes in this loop: a call per element cost 2 ns.
        if innermost:
            writer.write(values[at])
        else:
            _write_axis(owner, shape, strides, axis + 1, at, writer)
    writer.write("]")
