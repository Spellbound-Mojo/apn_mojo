"""Borrowed scalar Integer execution with unpublished, transactional outputs."""

from ..batch._tensor import _Tensor
from ..batch._parallel import _map_tensor, _parallel_mask
from ..batch.mask import Mask
from .value import Integer
from ._limits import _MAX_RESULT_BITS, _count_error
from ._tiles import _WIDTH


def _division_zero(index: Int) raises:
    raise Error(
        String(
            "Cannot divide batch: divisor is 0 at logical element ",
            index,
            (
                " (zero-based); use a nonzero divisor. The destination is"
                " unchanged."
            ),
        )
    )


def _length_mismatch(family: Int, operation: Int, left: Int, right: Int) raises:
    """Report unequal sequence lengths for an Integer operation family."""
    var prefix: String
    if family == 4:
        prefix = "Cannot compare"
    elif family == 3:
        prefix = String(
            "Cannot compute ", "gcd" if operation == 0 else "lcm", " for"
        )
    else:
        var code = (
            operation if family == 0 else operation + 4 if family == 1 else 3
        )
        var name = (
            "add" if code
            == 0 else "subtract" if code
            == 1 else "multiply" if code
            == 2 else "divide" if code
            == 3 else "combine with &" if code
            == 4 else "combine with |" if code
            == 5 else "combine with ^" if code
            == 6 else "left-shift" if code
            == 8 else "right-shift" if code
            == 9 else "raise powers of"
        )
        prefix = String("Cannot ", name)
    raise Error(
        String(
            prefix,
            " batches of lengths ",
            left,
            " and ",
            right,
            "; use equal lengths or a scalar operand.",
            "" if family == 4 else " The destination is unchanged.",
        )
    )



def _stage_checkpoint(
    label: StringSlice, fail: Int, start: Int, active: Int
) raises:
    """Injected failure after an eight-element group, for rollback tests."""
    if fail == start // _WIDTH:
        raise Error(
            String(
                "Injected ",
                label,
                " failure after logical element ",
                start + active - 1,
                "; the destination is unchanged. Retry the operation.",
            )
        )


@fieldwise_init
struct _NativeIntegerInputs(ImplicitlyCopyable):
    var a: _Tensor[Integer]
    var b: _Tensor[Integer]
    var broadcast_a: Bool
    var broadcast_b: Bool

    def length(self) -> Int:
        return self.b._layout.size if self.broadcast_a else self.a._layout.size

    def left(self, index: Int) -> ref[self.a] Integer:
        return self.a._read(0 if self.broadcast_a else index)

    def right(self, index: Int) -> ref[self.b] Integer:
        return self.b._read(0 if self.broadcast_b else index)

    def validate(self, family: Int, operation: Int) raises:
        if family == 2:
            for i in range(self.length()):
                if not self.right(i):
                    _division_zero(i)
        elif family == 1 and 4 <= operation <= 6:
            for i in range(self.length()):
                if self.right(i)._negative():
                    _count_error(operation, i)


def _element_checkpoint(label: StaticString, fail: Int, count: Int, index: Int) raises:
    if fail >= 0 and ((index + 1) % _WIDTH == 0 or index + 1 == count):
        var start = index // _WIDTH * _WIDTH
        _stage_checkpoint(label, fail, start, index + 1 - start)


def _native_integer_bit(a: Integer, b: Integer, operation: Int, index: Int) raises -> Integer:
    # Keep indexed capacity errors and arbitrary-width identity shortcuts.
    if operation == 4 and a and b > Integer(_MAX_RESULT_BITS - a.magnitude_bit_length()):
        _count_error(operation, index, too_large=True)
    if operation == 6 and a and (a._word_count() != 1 or a._word(0) != 1) and b != 1:
        if b > Integer(_MAX_RESULT_BITS // a.magnitude_bit_length()):
            _count_error(operation, index, too_large=True)
    if operation == 3:
        return ~a
    if operation == 7:
        return Integer(a.sign())
    if operation == 8:
        return Integer(a.magnitude_bit_length())
    if operation == 9:
        return -a
    if operation == 10:
        return abs(a)
    return a._compound_value(b, operation + 4)


@fieldwise_init
struct _IntegerJob[origin: ImmOrigin](Copyable, Movable):
    # The synchronous executor borrows the operator's retained inputs.
    var inputs: Pointer[_NativeIntegerInputs, Self.origin]
    var family: Int
    var operation: Int
    var fail: Int
    var label: StaticString


@always_inline
def _integer_value(
    inputs: _NativeIntegerInputs, family: Int, operation: Int, index: Int
) raises -> Integer:
    ref a = inputs.left(index)
    ref b = inputs.right(index)
    if family == 1:
        return _native_integer_bit(a, b, operation, index)
    if family == 2:
        var q, r = a._div_rem(b, 0)
        return r^ if operation == 1 else q^
    return a._compound_value(b, operation)


@always_inline
def _integer_element[origin: ImmOrigin](job: _IntegerJob[origin], index: Int) raises -> Integer:
    var value = _integer_value(job.inputs[], job.family, job.operation, index)
    _element_checkpoint(job.label, job.fail, job.inputs[].length(), index)
    return value^


def _native_integer_arithmetic(
    inputs: _NativeIntegerInputs, family: Int, operation: Int,
    *, fail: Int = -1, label: StaticString = "staging",
) raises -> _Tensor[Integer]:
    inputs.validate(family, operation)
    return _map_tensor[_integer_element[origin_of(inputs)]](
        _IntegerJob(Pointer(to=inputs), family, operation, fail, label),
        inputs.length(),
    )


@fieldwise_init
struct _IntegerOrderJob(Copyable, Movable):
    var inputs: _NativeIntegerInputs
    var operation: Int


@always_inline
def _integer_order(job: _IntegerOrderJob, index: Int) raises -> Bool:
    ref a = job.inputs.left(index)
    ref b = job.inputs.right(index)
    var operation = job.operation
    if operation == 0:
        return a == b
    if operation == 1:
        return a != b
    if operation == 2:
        return a < b
    if operation == 3:
        return a <= b
    if operation == 4:
        return a > b
    return a >= b


def _native_integer_comparison(inputs: _NativeIntegerInputs, operation: Int) raises -> Mask:
    return _parallel_mask[_integer_order](_IntegerOrderJob(inputs, operation), inputs.length())
