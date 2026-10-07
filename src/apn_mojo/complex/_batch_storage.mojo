"""Complex batch inputs and the conversion of iterators to Complex runs."""

from std.iter import Iterator
from .value import Complex
from ..float._arithmetic import _float_argument
from ..float._batch_storage import _FloatInput
from ..batch._iterator import _iterator_tensor
from ..batch._slices import _Selection
from ..batch._tensor import _Tensor


@fieldwise_init
struct _ComplexInput(ImplicitlyCopyable):
    var real: _FloatInput
    var imag: _FloatInput

    @no_inline
    def __init__(out self, *, deinit move: Self):
        self.real = move.real^
        self.imag = move.imag^

    @no_inline
    def __deinit__(deinit self):
        # Out of line: -O0 otherwise expands field teardown at every exit.
        pass

    def value(self, index: Int) -> Complex:
        return Complex(
            _real=self.real.value(index), _imag=self.imag.value(index)
        )


def _complex_transfer_checkpoint(index: Int, fail: Int) raises:
    if index == fail:
        raise Error(
            String(
                "Injected Complex conversion failure at element ",
                index,
                "; no batch is built.",
            )
        )


def _complex_iterator_element[I: Iterator](var value: I.Element) raises -> Complex:
    comptime assert conforms_to(I.Element, ImplicitlyCopyable), (
        "Complex batches require numeric elements; convert each element to"
        " Complex explicitly."
    )
    comptime if I.Element == Complex:
        return rebind_var[Complex](value^)
    elif conforms_to(I.Element, ImplicitlyCopyable & Deinitable):
        return Complex(_float_argument(value))
    else:
        comptime assert (
            False
        ), "Complex batches require ordinary copyable numeric values."


def _native_complex_iterator[I: Iterator](var cursor: I, fail: Int = -1) raises -> _Tensor[Complex]:
    return _iterator_tensor[Complex, I, _complex_iterator_element[I], _complex_transfer_checkpoint](cursor^, fail)
