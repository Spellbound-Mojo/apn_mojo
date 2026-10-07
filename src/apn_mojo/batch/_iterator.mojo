"""Build a rank-one tensor from an iterator, transferring each converted value."""

from std.iter import Iterator, StopIteration, next
from std.sys import size_of
from ..common._sizes import _checked_count, _checked_sum
from ._tensor import _Tensor


def _transfer_checkpoint(index: Int, fail: Int) raises:
    pass


def _iterator_tensor[
    T: ImplicitlyCopyable & Deinitable,
    I: Iterator,
    convert: def(var I.Element) raises thin -> T,
    checkpoint: def(Int, Int) raises thin -> None = _transfer_checkpoint,
](var cursor: I, fail: Int = -1) raises -> _Tensor[T]:
    var values = List[T]()
    while True:
        var value: I.Element
        try:
            value = next(cursor)
        except StopIteration:
            break
        var element = convert(value^)
        var index = len(values)
        _ = _checked_count(_checked_sum(index, 1), size_of[T]())
        checkpoint(index, fail)
        values.append(element^)
    var count = len(values)
    return _Tensor[T](values^, [count])


def _same_element[T: ImplicitlyCopyable & Deinitable, I: Iterator](var value: I.Element) raises -> T:
    comptime assert I.Element == T, (
        "A batch of this family takes only elements of its own family; convert"
        " each value first."
    )
    return rebind_var[T](value^)
