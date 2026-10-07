"""Checked logical selections; never raw borrows into movable storage."""

from std.builtin.builtin_slice import StridedSlice


struct _Selection(ImplicitlyCopyable):
    var start: Int
    var count: Int
    var step: Int

    def __init__(out self, length: Int, selection: StridedSlice) raises:
        var start, end, step = selection.indices(length)
        if step == 0:
            raise Error(
                "Cannot use a slice with step zero; use a positive or"
                " negative nonzero step."
            )
        var count = 0
        if step > 0 and start < end:
            count = 1 + (end - start - 1) // step
        elif step < 0 and start > end:
            var magnitude = UInt64(-(step + 1)) + 1
            count = 1 + Int(UInt64(start - end - 1) // magnitude)
        self.start = start
        self.count = count
        self.step = step

    def __init__(out self, start: Int, count: Int, step: Int):
        self.start = start
        self.count = count
        self.step = step

    def index(self, position: Int) -> Int:
        # Called only for 0 <= position < count; the result is within bounds.
        return self.start + position * self.step

    def sliced(self, selection: StridedSlice) raises -> Self:
        var inner = Self(self.count, selection)
        if inner.count == 0:
            return Self(0, 0, 1)
        # A singleton's stride is unobservable; avoid overflowing its product.
        var step = self.step * inner.step if inner.count > 1 else 1
        return Self(self.index(inner.start), inner.count, step)
