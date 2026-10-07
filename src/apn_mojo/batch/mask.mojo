"""Elementwise comparison results with explicit truth reductions."""

from std.memory import ArcPointer
from std.iter import (
    Iterable,
    IterableOwned,
    Iterator,
    StopIteration,
    iter,
    next,
)
from std.builtin.builtin_slice import ContiguousSlice, StridedSlice

from ..common._sizes import _checked_count, _checked_sum
from ._slices import _Selection
from ._layout import (
    _index, _AxisSelection, _select_axis, _FlatLayout, _row_major,
    _Ints, _ints, _list_ints, _list_mut_ints, _transpose_into,
)
from ..integer._tiles import _Mask, _WIDTH
from ..common._traits import _MapArgument


def _mask_shape_size(shape: List[Int]) raises -> Int:
    var strides = List[Int]()
    return _checked_count(_row_major(
        _list_ints(shape), _list_mut_ints(strides),
        negative="Mask dimensions must be nonnegative; use zero for an empty axis.",
        overflow="Mask shape exceeds addressable storage; use smaller dimensions.",
    ), 1)


struct _MaskStorage(Movable):
    var length: Int
    var tiles: List[_Mask]

    def __init__(out self, length: Int, var tiles: List[_Mask]):
        self.length = length
        self.tiles = tiles^


@fieldwise_init
struct _MaskIterator(ImplicitlyCopyable, Iterable, IterableOwned, Iterator):
    comptime Element = Bool
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime IteratorOwnedType: Iterator = Self

    var _storage: ArcPointer[_MaskStorage]
    var _position: Int

    def __iter__(ref self) -> Self:
        return self

    def __iter__(var self) -> Self:
        return self^

    def __next__(mut self) raises StopIteration -> Bool:
        if self._position == self._storage.ptr()[].length:
            raise StopIteration()
        var position = self._position
        self._position += 1
        return self._storage.ptr()[].tiles[position // _WIDTH][
            position % _WIDTH
        ]

    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = self._storage.ptr()[].length - self._position
        return remaining, Optional(remaining)


def _pack_mask_iterator[
    I: Iterator
](var cursor: I, fail_after_element: Int = -1) raises -> _MaskStorage:
    comptime assert conforms_to(
        I.Element, ImplicitlyCopyable
    ), "Masks require Bool elements; compare numeric values explicitly first."
    comptime assert (
        I.Element == Bool
    ), "Masks require Bool elements; compare numeric values explicitly first."
    var tiles = List[_Mask]()
    var length = 0
    while True:
        var value: Bool
        try:
            value = rebind_var[Bool](next(cursor))
        except StopIteration:
            break
        var new_length = _checked_sum(length, 1)
        if length % _WIDTH == 0:
            _ = _checked_count(_checked_sum(len(tiles), 1), 8)
            tiles.append(_Mask(fill=False))
        tiles[len(tiles) - 1][length % _WIDTH] = value
        if length == fail_after_element:
            raise Error(
                String(
                    "Injected mask construction failure at element ",
                    length,
                    (
                        "; the destination is unchanged. Retry with a fresh"
                        " iterable."
                    ),
                )
            )
        length = new_length
    return _MaskStorage(length, tiles^)


struct Mask(ImplicitlyCopyable, Iterable, IterableOwned, Sized, Writable, _MapArgument):
    """An immutable batch of Booleans, returned by batch comparisons.

    A mask has a shape like a batch. `batch[mask]` needs a mask of exactly the
    batch's shape and selects the true positions in row-major order. `&`, `|`, `^`
    and `~` combine masks of equal shape; nothing broadcasts. A comparison mask keeps
    its values when the numbers change.

    Limitations:
        `Bool(mask)` always raises as ambiguous: ask `any()` or `all()`. Use `&`
        and `|`, not `and` and `or`, to combine masks.
    """

    var _storage: ArcPointer[_MaskStorage]
    var _shape: Optional[ArcPointer[List[Int]]]

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self._storage = move._storage^
        self._shape = move._shape^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _MaskIterator
    comptime IteratorOwnedType: Iterator = _MaskIterator

    def __iter__(ref self) -> _MaskIterator:
        return _MaskIterator(self._storage, 0)

    def __iter__(var self) -> _MaskIterator:
        return _MaskIterator(self._storage, 0)

    def __init__[I: Iterable](out self, values: I) raises:
        """A mask from a finite iterable of Booleans.

        Parameters:
            I: The iterable type.

        Args:
            values: The Booleans.

        Raises:
            When an item is not a Boolean.
        """
        self._storage = ArcPointer(_pack_mask_iterator(iter(values)))
        self._shape = None

    def __init__(out self, var values: Some[IterableOwned]) raises:
        """A mask from an owned finite iterable of Booleans."""
        self._storage = ArcPointer(_pack_mask_iterator(iter(values^)))
        self._shape = None

    @staticmethod
    def from_iterable[I: Iterable](values: I) raises -> Self:
        """A mask from a finite iterable of Booleans.

        Parameters:
            I: The iterable type.

        Args:
            values: The Booleans.

        Returns:
            The mask.

        Raises:
            When an item is not a Boolean.
        """
        return Self(values)

    @staticmethod
    def from_iterable(var values: Some[IterableOwned]) raises -> Self:
        """A mask from an owned finite iterable of Booleans."""
        return Self(values^)

    def __init__(out self):
        """An empty mask."""
        self._storage = ArcPointer(_MaskStorage(0, List[_Mask]()))
        self._shape = None

    def __init__(out self, values: List[Bool]) raises:
        """A vector mask from a list of Booleans."""
        var length = len(values)
        var whole = length // _WIDTH
        var tiles = List[_Mask](capacity=_checked_count(whole + Int(length % _WIDTH != 0), 8))
        # A Bool is one byte, so each tile is one load of _WIDTH bytes.
        var lanes = values.unsafe_ptr().unsafe_bitcast[SIMD[DType.uint8, _WIDTH]]()
        for tile in range(whole):
            tiles.append(lanes.unsafe_offset(tile)[].cast[DType.bool]())
        if length % _WIDTH:
            var last = _Mask(fill=False)
            for i in range(whole * _WIDTH, length):
                last[i % _WIDTH] = values[i]
            tiles.append(last)
        self._storage = ArcPointer(_MaskStorage(length, tiles^))
        self._shape = None

    def __init__(out self, values: List[Bool], *, shape: List[Int]) raises:
        """A mask with a shape, from row-major Booleans."""
        if _mask_shape_size(shape) != len(values):
            raise Error("Mask shape does not match its values; use dimensions whose product equals the Boolean count.")
        self = Self(values)
        if len(shape) != 1:
            self._shape = ArcPointer(shape.copy())

    def __init__(out self, *, copy: Self):
        self._storage = copy._storage
        self._shape = copy._shape

    def __init__(out self, *, length: Int, var _tiles: List[_Mask]):
        self._storage = ArcPointer(_MaskStorage(length, _tiles^))
        self._shape = None

    def __len__(self) -> Int:
        return self._storage.ptr()[].length

    def shape(self) -> List[Int]:
        """The dimensions.

        Returns:
            A copy of the shape.
        """
        if self._shape:
            return self._shape.value()[].copy()
        return [len(self)]

    def ndim(self) -> Int:
        """The rank.

        Returns:
            The number of axes.
        """
        return len(self._shape.value()[]) if self._shape else 1

    def _dim(self, axis: Int) -> Int:
        return self._shape.value()[][axis] if self._shape else len(self)

    def size(self) -> Int:
        """The number of entries.

        Returns:
            The product of the dimensions.
        """
        return len(self)

    def item(self) raises -> Bool:
        """The only entry.

        Returns:
            The entry of a mask of size one.

        Raises:
            When the size is not one.
        """
        if len(self) != 1:
            raise Error("item() requires exactly one Boolean; select a coordinate or singleton mask first.")
        return self[0]

    def reshape(self, shape: List[Int]) raises -> Self:
        """The same entries with another shape.

        Args:
            shape: The new dimensions; the size must not change.

        Returns:
            The reshaped mask.

        Raises:
            When the sizes differ.
        """
        if _mask_shape_size(shape) != len(self):
            raise Error("Mask reshape must preserve the Boolean count; use dimensions with the same product.")
        var result = self
        result._shape = None
        if len(shape) != 1:
            result._shape = ArcPointer(shape.copy())
        return result

    def transpose(self, axes: List[Int]) raises -> Self:
        """Permute the axes.

        Args:
            axes: A permutation of the axes.

        Returns:
            The transposed mask.

        Raises:
            When `axes` is not a permutation.
        """
        if len(axes) != self.ndim():
            raise Error("Mask transpose needs one axis per dimension; supply a permutation of all axes.")
        if self._shape:
            return self._transpose_shape(_list_ints(self._shape.value()[]), axes)
        var shape = [len(self)]
        return self._transpose_shape(_ints(shape), axes)

    def _transpose_shape(self, shape: _Ints, axes: List[Int]) raises -> Self:
        var layout = _FlatLayout()
        layout.scalar = False
        layout.shape = List[Int](length=len(shape), fill=0)
        layout.strides = List[Int](length=len(shape), fill=0)
        var strides = List[Int](length=len(shape), fill=0)
        layout.size = _row_major(shape, _list_mut_ints(strides))
        _transpose_into(
            shape, _list_ints(strides), _list_ints(axes),
            _list_mut_ints(layout.shape), _list_mut_ints(layout.strides),
            repeated="Mask transpose repeats an axis; supply each axis once.",
        )
        layout.contiguous = layout._is_contiguous()
        return self._gather_layout(layout)

    def transpose(self) raises -> Self:
        """Reverse the axes."""
        var axes = List[Int](capacity=self.ndim())
        for i in range(self.ndim() - 1, -1, -1):
            axes.append(i)
        return self.transpose(axes)

    def at(self, axis: Int, index: Int) raises -> Self:
        """Select one index along an axis, dropping that axis.

        Args:
            axis: The axis.
            index: The index along it.

        Returns:
            A mask of one rank lower.

        Raises:
            When the axis or index is out of range.
        """
        if self._shape:
            return self._gather_axis(_select_axis(_list_ints(self._shape.value()[]), axis, index))
        var shape = [len(self)]
        return self._gather_axis(_select_axis(_ints(shape), axis, index))

    def _gather_axis(self, var selection: _AxisSelection) raises -> Self:
        var block = 1
        if selection.size:
            for i in range(selection.axis + 1, self.ndim()):
                block *= self._dim(i)
        var values = List[Bool](capacity=selection.size)
        for i in range(selection.size):
            var source = (i // block * self._dim(selection.axis) + selection.index) * block + i % block
            values.append(self[source])
        return Self(values, shape=selection.shape)

    def _gather_layout(self, layout: _FlatLayout) raises -> Self:
        var values = List[Bool](capacity=layout.size)
        for i in range(layout.size):
            values.append(self[layout.position(i)])
        return Self(values, shape=layout.shape)

    def _check_length(self, length: Int) raises:
        if self.ndim() != 1:
            raise Error("A vector needs a one-dimensional mask; reshape the mask explicitly to the vector's shape.")
        if len(self) != length:
            raise Error(
                String(
                    "Cannot use a mask of length ",
                    len(self),
                    " with a sequence of length ",
                    length,
                    (
                        "; use one boolean for each indexed value."
                        " The destination is unchanged."
                    ),
                )
            )

    def write_to(self, mut writer: Some[Writer]):
        """Write the entries as nested lists of `True` and `False`.

        Args:
            writer: The destination.
        """
        var position = 0
        self._write(0, position, writer)

    def _write(self, axis: Int, mut position: Int, mut writer: Some[Writer]):
        if axis == self.ndim():
            writer.write(Bool(self._storage[].tiles[position // _WIDTH][position % _WIDTH]))
            position += 1
            return
        writer.write("[")
        for i in range(self._dim(axis)):
            if i:
                writer.write(", ")
            self._write(axis + 1, position, writer)
        writer.write("]")

    def __getitem__(self, index: Int) raises -> Bool:
        var adjusted = index + len(self) if index < 0 else index
        if adjusted < 0 or adjusted >= len(self):
            raise Error(
                String(
                    "Cannot access mask element ",
                    index,
                    " in a mask of length ",
                    len(self),
                    "; use an index within the mask bounds.",
                )
            )
        return self._storage.ptr()[].tiles[adjusted // _WIDTH][
            adjusted % _WIDTH
        ]

    def __getitem__(self, selection: StridedSlice) raises -> Self:
        var positions = _Selection(len(self), selection)
        var values = List[Bool](capacity=_checked_count(positions.count, 1))
        for i in range(positions.count):
            values.append(self[positions.index(i)])
        return Self(values)

    def __getitem__(self, selection: ContiguousSlice) raises -> Self:
        return self[StridedSlice(selection.start, selection.end, 1)]

    def __getitem__(self, *indices: Int) raises -> Bool:
        if len(indices) != self.ndim():
            raise Error("Supply one integer coordinate per mask axis, or a single flat logical index.")
        var position = 0
        for i in range(self.ndim()):
            position = position * self._dim(i) + _index(indices[i], self._dim(i))
        return self[position]

    def __getitem__(self, empty: Tuple[]) raises -> Bool:
        if self.ndim() != 0:
            raise Error("Empty coordinates require a rank-zero mask; supply one coordinate per axis.")
        return self.item()

    def __bool__(self) raises -> Bool:
        raise Error(
            "A comparison mask has no single truth value; use .all() or .any(),"
            " or select one element with an index."
        )

    def all(self) -> Bool:
        """Whether every entry is true.

        Returns:
            True for an empty mask.
        """
        var complete = len(self) // _WIDTH
        for i in range(complete):
            if self._storage.ptr()[].tiles[i] != _Mask(fill=True):
                return False
        for lane in range(len(self) % _WIDTH):
            if not self._storage.ptr()[].tiles[complete][lane]:
                return False
        return True

    def any(self) -> Bool:
        """Whether some entry is true.

        Returns:
            False for an empty mask.
        """
        for tile in self._storage.ptr()[].tiles:
            if tile != _Mask(fill=False):
                return True
        return False

    def count(self) -> Int:
        """The number of true entries.

        Returns:
            The count; the length of `batch[mask]`.
        """
        var result = 0
        for tile in self._storage.ptr()[].tiles:
            result += Int(tile.cast[DType.uint32]().reduce_add())
        return result

    def to_list(self) raises -> List[Bool]:
        """The entries, in row-major order.

        Returns:
            An independent list.

        Raises:
            Only on a checked size error.
        """
        var values = List[Bool](capacity=_checked_count(len(self), 1))
        for i in range(len(self)):
            values.append(self[i])
        return values^

    def _combine(self, rhs: Self, operation: Int) raises -> Self:
        if self.ndim() != rhs.ndim() or (self.ndim() != 1 and self.shape() != rhs.shape()):
            raise Error("Cannot combine masks of different shapes; use masks with the same shape. Masks do not broadcast.")
        if len(self) != len(rhs):
            raise Error(
                String(
                    "Cannot combine masks of lengths ",
                    len(self),
                    " and ",
                    len(rhs),
                    "; use masks of equal lengths.",
                )
            )
        var tiles = List[_Mask](capacity=len(self._storage.ptr()[].tiles))
        for i in range(len(self._storage.ptr()[].tiles)):
            var a = self._storage.ptr()[].tiles[i]
            var b = rhs._storage.ptr()[].tiles[i]
            tiles.append(
                a & b if operation == 0 else a | b if operation == 1 else a ^ b
            )
        var result = Self(length=len(self), _tiles=tiles^)
        result._shape = self._shape
        return result


    def __and__(self, rhs: Self) raises -> Self:
        return self._combine(rhs, 0)

    def __or__(self, rhs: Self) raises -> Self:
        return self._combine(rhs, 1)

    def __xor__(self, rhs: Self) raises -> Self:
        return self._combine(rhs, 2)

    def __invert__(self) raises -> Self:
        var tiles = List[_Mask](capacity=len(self._storage.ptr()[].tiles))
        for i in range(len(self._storage.ptr()[].tiles)):
            var tile = ~self._storage.ptr()[].tiles[i]
            for lane in range(min(_WIDTH, len(self) - i * _WIDTH), _WIDTH):
                tile[lane] = False
            tiles.append(tile)
        var result = Self(length=len(self), _tiles=tiles^)
        result._shape = self._shape
        return result


