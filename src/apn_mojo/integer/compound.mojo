"""Exact scaled addition a*x + y over native Integer batches."""

from .value import Integer
from ..batch.value import Batch, _operand_shape, _shape_like
from ._batch_native import _element_checkpoint
from ..batch._parallel import _map_tensor
from ..batch._tensor import _Tensor


def _axpy_operand[T: ImplicitlyCopyable](values: T) -> Batch[Integer]:
    comptime assert T == Batch[Integer], (
        "Integer axpy requires integer batches for x and y; construct"
        " Batch[Integer](values) from an integer iterable first."
    )
    return rebind[Batch[Integer]](values)



def _axpy[
    X: ImplicitlyCopyable, Y: ImplicitlyCopyable
](
    a: Integer,
    x: X,
    y: Y,
    fail_after_tile: Int = -1,
) raises -> Batch[Integer]:
    var shape = _operand_shape(x, y, x)
    return _shape_like(_axpy_views(a, _axpy_operand(x), _axpy_operand(y), fail_after_tile), shape)


# Operand adapters stay generic; the kernel compiles once for all of them.
@fieldwise_init
struct _AxpyJob(Copyable, Movable):
    var a: Integer
    var u: _Tensor[Integer]
    var v: _Tensor[Integer]
    var fail: Int


@always_inline
def _axpy_value(job: _AxpyJob, index: Int) raises -> Integer:
    if not job.a:
        return job.v._read(index)
    return job.a * job.u._read(index) + job.v._read(index)


def _axpy_element(job: _AxpyJob, index: Int) raises -> Integer:
    var value = _axpy_value(job, index)
    _element_checkpoint("axpy", job.fail, job.v._layout.size, index)
    return value^


def _axpy_views(
    a: Integer,
    left: Batch[Integer],
    right: Batch[Integer],
    fail_after_tile: Int,
) raises -> Batch[Integer]:
    var length = len(left)
    if length != len(right):
        raise Error(
            String(
                "Cannot compute integer axpy for x and y lengths ",
                length,
                " and ",
                len(right),
                "; use equal-length batches.",
                (
                    " Only the scalar coefficient broadcasts. The destination"
                    " is unchanged."
                ),
            )
        )
    var v = right._tensor_view()
    var u = v
    if a:
        u = left._tensor_view()
    return Batch[Integer](_tensor=_map_tensor[_axpy_element](
        _AxpyJob(a, u, v, fail_after_tile), length,
    ))


def axpy[
    X: ImplicitlyCopyable, Y: ImplicitlyCopyable
](a: Integer, x: X, y: Y) raises -> Batch[Integer]:
    """Exact scaled addition: `a * x[i] + y[i]` for every element.

    Elements pair by logical position; only the coefficient broadcasts.

    Parameters:
        X: The type of `x`, an Integer batch.
        Y: The type of `y`, an Integer batch.

    Args:
        a: The coefficient.
        x: The scaled batch.
        y: The added batch, of the same length.

    Returns:
        A new Integer batch.

    Raises:
        When the lengths differ, even for a zero coefficient.
    """
    return _axpy(a, x, y)
