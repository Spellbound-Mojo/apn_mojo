"""Joining batches, like NumPy's `concatenate` and `stack`."""

from ._layout import _axis
from .value import Batch


def concatenate[T: ImplicitlyCopyable & Deinitable](batches: List[Batch[T]], *, axis: Int = 0) raises -> Batch[T]:
    """The batches joined along an existing axis, like `numpy.concatenate`.

    Parameters:
        T: The element type, inferred.

    Args:
        batches: One or more batches of one rank, with equal dimensions off the axis.
        axis: The axis to join along; negative axes count from the end.

    Returns:
        A new batch holding the values in order.

    Raises:
        For no batches, rank-zero batches, or shapes that differ off the axis.
    """
    if len(batches) == 0:
        raise Error("concatenate needs at least one batch.")
    var shape = batches[0].shape()
    var rank = len(shape)
    if rank == 0:
        raise Error("Rank-zero batches have no axis to join along; use stack.")
    var a = _axis(axis, rank)
    var total = 0
    for batch in batches:
        var other = batch.shape()
        if len(other) != rank:
            raise Error("concatenate needs batches of one rank.")
        for i in range(rank):
            if i != a and other[i] != shape[i]:
                raise Error("concatenate needs equal dimensions off the joined axis.")
        total += other[a]
    var outer = 1
    for i in range(a):
        outer *= shape[i]
    var inner = 1
    for i in range(a + 1, rank):
        inner *= shape[i]
    # Row-major order: each outer index takes one block from every batch in turn.
    var flat = List[List[T]](capacity=len(batches))
    for batch in batches:
        flat.append(batch.to_list())
    var values = List[T](capacity=outer * total * inner)
    for o in range(outer):
        for i in range(len(batches)):
            var block = batches[i].shape()[a] * inner
            for j in range(o * block, (o + 1) * block):
                values.append(flat[i][j])
    shape[a] = total
    return Batch[T](values, shape=shape)


def stack[T: ImplicitlyCopyable & Deinitable](batches: List[Batch[T]], *, axis: Int = 0) raises -> Batch[T]:
    """The batches joined along a new axis, like `numpy.stack`.

    Parameters:
        T: The element type, inferred.

    Args:
        batches: One or more batches of one shape.
        axis: The position of the new axis in the result; negative axes count from the end.

    Returns:
        A new batch with one more axis.

    Raises:
        For no batches or different shapes.
    """
    if len(batches) == 0:
        raise Error("stack needs at least one batch.")
    var shape = batches[0].shape()
    var a = _axis(axis, len(shape) + 1)
    var expanded = shape.copy()
    expanded.insert(a, 1)
    var parts = List[Batch[T]](capacity=len(batches))
    for batch in batches:
        if batch.shape() != shape:
            raise Error("stack needs batches of one shape.")
        parts.append(batch.reshape(expanded))
    return concatenate(parts, axis=a)
