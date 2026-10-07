"""Contracts shared by batch orchestration and family-specific kernels."""

from std.testing import TestSuite, assert_equal, assert_raises
from std.memory import ArcPointer
from apn_mojo import Integer, Rational, Float, Complex, Batch, ConversionLimits, vmap, lift, sum, dot
from apn_mojo import ArithmeticContext
from apn_mojo.batch._exact_reduce import _exact_fold
from apn_mojo.batch._lanes import _Lanes
from apn_mojo.batch._tensor import _Tensor
from apn_mojo.batch._values import _Values
from apn_mojo.common._traits import _MapArgument
from apn_mojo.batch.reductions import _reduce, _reduce_exact, _dot
from apn_mojo.batch._slices import _Selection
from apn_mojo.batch.value import _float_operand, _complex_operand
from apn_mojo.float._batch_storage import _FloatInput
from apn_mojo.float._reductions import _dot_float
from apn_mojo.complex._reductions import _dot_complex


def _check_fold_failure[T: ImplicitlyCopyable & Deinitable, operation: Int](values: Batch[T]) raises:
    var source = values.to_json()
    var destination = values[:1]
    var saved = destination.to_json()
    # A long reversed selection crosses the lane driver's chunk boundaries.
    for index in range(values.size()):
        with assert_raises(contains=String("after logical element ", index)):
            destination = _exact_fold[operation](values, List[Int](), True, False, index)
        assert_equal(destination.to_json(), saved)
        assert_equal(values.to_json(), source)
    assert_equal(
        _exact_fold[operation](values, List[Int](), True, False, values.size()).to_json(),
        _exact_fold[operation](values, List[Int](), True, False).to_json(),
    )


def test_exact_failures_use_chunked_and_strided_kernels() raises:
    var wide = (Integer(1) << 200) + 3
    var integers = List[Integer]()
    var rationals = List[Rational]()
    for i in range(514):
        var value = wide if i % 128 == 1 else Integer(-1 if i % 3 == 0 else 1)
        integers.append(value)
        rationals.append(Rational(value, 3 if i % 128 == 1 else 1))
    var a = Batch[Integer](integers)[::-2]
    var b = Batch[Rational](rationals)[::-2]
    comptime for operation in range(4):
        _check_fold_failure[Integer, operation](a)
        _check_fold_failure[Rational, operation](b)
    var source = a.to_json()
    var result = wide
    for index in range(len(a)):
        with assert_raises(contains=String("after logical element ", index)):
            result = _dot(a, a[::-1], index)
        assert_equal(result, wide)
        assert_equal(a.to_json(), source)
    assert_equal(_dot(a, a, len(a)), _dot(a, a))
    # Enough wide work to exercise the scheduler's worker-eligible path.
    var long = Batch[Integer]([wide + i for i in range(20000)])
    var saved = long.to_json()
    with assert_raises(contains="after logical element 19999"):
        result = _reduce[0](long, 19999)
    assert_equal(result, wide)
    with assert_raises(contains="after logical element 19999"):
        result = _dot(long, long, 19999)
    assert_equal(result, wide)
    assert_equal(long.to_json(), saved)
    # Validation still precedes arithmetic, even with failure injection enabled.
    with assert_raises(contains="equal-length batches"):
        _ = _dot(a, a[:1], 0)
    with assert_raises(contains="empty integer selection"):
        _ = _reduce[2](a[:0], 0)
    with assert_raises(contains="empty Rational selection"):
        _ = _reduce_exact[3](b[:0], 0)
    assert_equal(_reduce[1](Batch[Integer]([wide, 0]), 0), Integer(0))
    with assert_raises(contains="after logical element 0"):
        _ = _reduce_exact[1](Batch[Rational]([Rational(wide), Rational(0)]), 0)


def _add(x: Integer, y: Integer) raises -> Integer:
    return x + y


def test_operators_mapping_and_lifting_keep_their_shape_rules() raises:
    var vector = Batch[Integer]([1, 2, 3])
    var singleton = Batch[Integer]([10])
    with assert_raises():
        _ = vector + singleton
    with assert_raises():
        _ = vmap[_add]()(vector, singleton)
    assert_equal(lift[_add]()(vector, singleton).to_list(), [Integer(11), Integer(12), Integer(13)])
    var column = vector.reshape([3, 1])
    var row = Batch[Integer]([10, 20]).reshape([1, 2])
    assert_equal((column + row).to_json(), lift[_add]()(column, row).to_json())
    assert_equal((column.slice(0, step=-1) + row).to_json(), lift[_add]()(column.slice(0, step=-1), row).to_json())
    assert_equal((column.slice(0, stop=0) + row).shape(), [0, 2])
    assert_equal(lift[_add]()(column.slice(0, stop=0), row).shape(), [0, 2])
    var saved = column.to_json()
    with assert_raises(contains="cannot expand the destination"):
        column += row
    assert_equal(column.to_json(), saved)


def _check_envelope[T: ImplicitlyCopyable & Deinitable](values: Batch[T]) raises:
    var encoded = values.to_json()
    var bounded = ConversionLimits(max_digits=100000, max_allocated_bytes=1000000)
    assert_equal(Batch[T].from_json(encoded, limits=bounded).to_json(), encoded)
    assert_equal(values.to_json(limits=bounded), encoded)
    var destination = values
    for counted in [False, True]:
        var limits: Optional[ConversionLimits] = bounded if counted else None
        with assert_raises(contains="trailing content"):
            destination = Batch[T].from_json(encoded + " true", limits=limits)
        assert_equal(destination.to_json(), encoded)
        with assert_raises(contains="duplicate field"):
            destination = Batch[T].from_json('{"version":1,' + String(unsafe_from_utf8=encoded.as_bytes()[1:]), limits=limits)
        assert_equal(destination.to_json(), encoded)
    with assert_raises():
        destination = Batch[T].from_json(encoded, limits=ConversionLimits(max_values=values.size() - 1))
    assert_equal(destination.to_json(), encoded)


def test_formatted_envelopes_keep_limits_and_failure_atomicity() raises:
    _check_envelope(Batch[Float]([Float(1), Float.zero(negative=True), Float.infinity()]).reshape([1, 3]))
    _check_envelope(Batch[Complex]([Complex(1, -2), Complex(3, 4)]).reshape([2, 1]))


def test_storage_borrows_do_not_retain_elements() raises:
    var wide = (Integer(1) << 200) + 3
    var owner = ArcPointer(_Values([wide]))
    var owners = wide._storage[Integer._Shared].count()
    ref stored = owner[][0]
    assert_equal(stored._storage[Integer._Shared].count(), owners)
    var tensor = _Tensor[Integer]([wide], [1])
    owners = wide._storage[Integer._Shared].count()
    ref element = tensor._read(0)
    assert_equal(element._storage[Integer._Shared].count(), owners)
    var lanes = _Lanes(Batch[Integer]([wide, wide + 1])[::-1], List[Int](), [0])
    owners = wide._storage[Integer._Shared].count()
    ref last = lanes.at(lanes.base(0), 1)
    assert_equal(last, wide)
    assert_equal(last._storage[Integer._Shared].count(), owners)
    var run = lanes.run(lanes.base(0))
    comptime assert not type_of(run).origin.mut
    assert_equal(run[], wide + 1)
    assert_equal(run.unsafe_offset(lanes.step)[], wide)
    assert_equal(wide._storage[Integer._Shared].count(), owners)
    # Keep all three owners live through the count above; Mojo releases locals
    # after their last use, rather than at the closing brace.
    assert_equal(stored, wide)
    assert_equal(element, wide)
    assert_equal(last, wide)


def _owners(value: Integer) -> Integer:
    return Integer(value._storage[Integer._Shared].count())


def test_mapping_borrows_without_retaining_each_element() raises:
    var inputs = Batch[Integer]([Integer(i + 1) << 200 for i in range(128)])
    var expected = [Integer(1) for _ in range(128)]
    assert_equal(vmap[_owners]()(inputs).to_list(), expected)
    assert_equal(vmap[_owners](axis_size=128)(inputs).to_list(), expected)


def _keep[T: ImplicitlyCopyable & Deinitable](value: T) -> T:
    return value


def _keep_or_fail[T: ImplicitlyCopyable & Deinitable & Equatable](value: T, marker: T) raises -> T:
    var retained = value
    if retained == marker:
        raise Error("borrowed callback failed")
    return retained^


def _check_retained_view[T: _MapArgument & ImplicitlyCopyable & Deinitable & Equatable & Writable](var parent: Batch[T]) raises:
    comptime assert T == Integer or T == Rational or T == Float or T == Complex
    var view = parent.reshape([16, 8]).slice(0, step=-1).transpose()
    var saved = view.to_json()
    var marker = parent[31]
    # Only the view remains; every reader must retain its backing storage.
    parent = Batch[T](List[T]())
    var flat = rebind_var[Batch[T]](vmap[_keep[T]]().vmap()(view))
    var general = rebind_var[Batch[T]](vmap[_keep[T]](axis_size=16).vmap(axis_size=8)(view))
    assert_equal(flat.to_list(), view.to_list())
    assert_equal(general.to_list(), view.to_list())
    # A callback copies an element before failing. Readers must unwind without
    # releasing borrowed values or publishing part of the mapped result.
    var destination = flat
    for forced in [False, True]:
        with assert_raises(contains="borrowed callback failed"):
            if forced:
                destination = rebind_var[Batch[T]](vmap[_keep_or_fail[T]](in_axes=(0, None), axis_size=16).vmap(in_axes=(0, None), axis_size=8)(view, marker))
            else:
                destination = rebind_var[Batch[T]](vmap[_keep_or_fail[T]](in_axes=(0, None)).vmap(in_axes=(0, None))(view, marker))
        assert_equal(destination.to_list(), flat.to_list())
        assert_equal(view.to_json(), saved)
    # Returned values outlive both the reader and the source view.
    view = Batch[T](List[T]())
    assert_equal(flat.to_list(), general.to_list())


def test_readers_keep_temporary_and_strided_sources_alive() raises:
    var wide = (Integer(1) << 100) + 3
    _check_retained_view(Batch[Integer]([wide + i for i in range(128)]))
    _check_retained_view(Batch[Rational]([Rational(wide + i, 3) for i in range(128)]))
    _check_retained_view(Batch[Float]([Float(wide + i) for i in range(128)]))
    _check_retained_view(Batch[Complex]([Complex(wide + i, -wide - i) for i in range(128)]))


def test_lifted_reads_keep_linear_and_strided_lanes_alive() raises:
    var wide = (Integer(1) << 200) + 3
    var matrix = Batch[Integer]([wide + i for i in range(128)]).reshape([16, 8])
    var view = matrix.slice(0, step=-1).transpose()
    matrix = Batch[Integer]([])
    var plus = lift[_add](identity=Integer(0))
    assert_equal(plus.reduce(view, axis=None), sum(view))
    assert_equal(plus.reduce(view, axis=0).to_list(), sum(view, axis=0).to_list())
    assert_equal(plus.reduce(view, axis=1).to_list(), sum(view, axis=1).to_list())
    assert_equal(plus.reduce(Batch[Integer]([])).item(), Integer(0))


def _copy_float_borrow(input: _FloatInput, index: Int) raises -> Float:
    var owners = input.element(index)._significand._storage[Integer._Shared].count()
    ref value = input.element(index)
    var pointer = Pointer(to=value)
    comptime assert not type_of(pointer).origin.mut
    assert_equal(pointer[]._significand._storage[Integer._Shared].count(), owners)
    assert_equal(input.format_at(index), value.format())
    var copy = input.value(index)
    assert_equal(value._significand._storage[Integer._Shared].count(), owners + 1)
    assert_equal(copy.to_json(), value.to_json())
    return copy^


def test_float_component_borrows_retain_only_explicit_copies() raises:
    var wide = (Integer(1) << 200) + 3
    var inputs = List[_FloatInput]()
    inputs.append(Batch[Float]([Float(wide), Float(wide + 1)])[::-1]._float_input())
    var parts = Batch[Complex]([Complex(wide, -wide), Complex(wide + 1, -wide - 1)])[::-1]._complex_input()
    inputs.append(parts.real)
    inputs.append(parts.imag)
    inputs.append(_FloatInput(Float(wide + 2), _Selection(0, 1, 1)))
    var copies = List[Float]()
    for input in inputs:
        copies.append(_copy_float_borrow(input, 0))
    assert_equal(Bool(inputs[0].run_step()), True)
    for i in range(1, len(inputs)):
        assert_equal(Bool(inputs[i].run_step()), False)
    # Drop every input owner before using the explicit copies again.
    inputs.clear()
    _ = parts^
    assert_equal(copies, [Float(wide + 1), Float(wide + 1), Float(-wide - 1), Float(wide + 2)])


def test_float_run_borrows_preserve_composed_strides() raises:
    var wide = (Integer(1) << 200) + 3
    var parent = Batch[Float]([Float(wide + i) for i in range(13)])
    # Selection stride and tensor stride both contribute to the run pointer.
    var input = _FloatInput(None, _Selection(1, 3, 2), native=parent[::-2]._tensor_view())
    parent = Batch[Float]([])
    assert_equal(input.run_step().value(), -4)
    var owners = input.element(0)._significand._storage[Integer._Shared].count()
    var first = input.run_start()
    comptime assert not type_of(first).origin.mut
    for i in range(3):
        ref value = first.unsafe_offset(i * input.run_step().value())[]
        assert_equal(value, Float(wide + 10 - 4 * i))
        assert_equal(Int(Pointer(to=value)), Int(Pointer(to=input.element(i))))
    assert_equal(first[]._significand._storage[Integer._Shared].count(), owners)
    assert_equal(input.element(0), Float(wide + 10))


def test_float_complex_dot_borrows_survive_failures_and_source_updates() raises:
    var wide = (Integer(1) << 100) + 3
    var parent = Batch[Float]([Float(wide + i) for i in range(130)])
    var complex_parent = Batch[Complex]([Complex(wide + i, -wide - i) for i in range(130)])
    var left = parent[::-2]
    var right = parent[1::2]
    var z = complex_parent[::-2]
    var w = complex_parent[1::2]
    parent += Float(1)
    complex_parent += Complex(1, 1)
    var exact = Integer(0)
    for i in range(65):
        exact += (wide + 129 - 2 * i) * (wide + 1 + 2 * i)
    var result = dot(left, right)
    var complex_result = dot(z, w)
    assert_equal(result, Float(exact, context=ArithmeticContext(format=result.format())))
    assert_equal(complex_result, Complex(Float(0), Float(-2 * exact, context=ArithmeticContext(format=complex_result._imag.format()))))
    var saved = left.to_json()
    var saved_complex = z.to_json()
    for index in [0, 32, 64]:
        with assert_raises(contains=String("after logical element ", index)):
            result = Float(_rounded=_dot_float(_float_operand(left), _float_operand(right), fail_after_element=index))
        assert_equal(result, dot(left, right))
        for component in range(2):
            with assert_raises(contains=String("after logical element ", index)):
                complex_result = _dot_complex(_complex_operand(z), _complex_operand(w), fail_after_element=index, fail_component=component)[0]
            assert_equal(complex_result, dot(z, w))
    assert_equal(left.to_json(), saved)
    assert_equal(z.to_json(), saved_complex)
    assert_equal(dot(left[:0], right[:0]), Float(0))
    assert_equal(dot(z[:0], w[:0]), Complex(0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
