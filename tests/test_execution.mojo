"""Shared elementwise execution: traversal, borrowing and failure cleanup."""

from std.testing import TestSuite, assert_equal, assert_raises
from std.time import perf_counter_ns
from apn_mojo import Batch, Integer, Float, Complex, FloatFormat, ArithmeticContext, lift, vmap, set_num_threads, get_num_threads
from apn_mojo import float
from apn_mojo.common._threads import _requested_threads
from std.os import setenv, unsetenv
from std.testing import assert_true
from apn_mojo.batch._parallel import _ElementKernel, _MapErrors, _execute_values


@fieldwise_init
struct _Job(Movable):
    # A run must borrow its job: even a worker dispatch cannot copy this type.
    var value: Integer
    var fail: Int
    var setup_failure: Int


struct _Kernel(_ElementKernel):
    comptime Element = Integer
    comptime Job = _Job
    var next: Int

    def __init__(out self, job: _Job, begin: Int) raises:
        self.next = begin
        if job.setup_failure >= 0 and begin >= job.setup_failure:
            raise Error("reader setup")

    def apply(mut self, job: _Job, index: Int) raises -> Integer:
        # Enough work for the probe to dispatch on a machine with a pool.
        var start = perf_counter_ns()
        while perf_counter_ns() - start < 20_000:
            pass
        assert_equal(index, self.next)
        self.next += 1
        if index == job.fail or index == job.fail + 73:
            raise Error(String("element ", index))
        return job.value


def test_executor_releases_partial_results_and_reports_first_failure() raises:
    var wide = Integer(1) << 600
    var errors = _MapErrors("test ", "")
    var owners = wide._storage[Integer._Shared].count()
    for probe in [False, True]:
        for fail in [0, 15, 16, 17, 63, 64, 256]:
            with assert_raises(contains=String("test ", fail, ": element ", fail)):
                _ = _execute_values[_Kernel](_Job(wide, fail, -1), 257, errors, probe=probe)
            assert_equal(wide._storage[Integer._Shared].count(), owners)
        with assert_raises(contains="test 0: reader setup"):
            _ = _execute_values[_Kernel](_Job(wide, -1000, 0), 257, errors, probe=probe)
        assert_equal(wide._storage[Integer._Shared].count(), owners)
    # Reader setup can also fail after the probe has written a full prefix.
    with assert_raises(contains="test 16: reader setup"):
        _ = _execute_values[_Kernel](_Job(wide, -1000, 16), 257, errors, probe=True)
    assert_equal(wide._storage[Integer._Shared].count(), owners)
    _ = wide.magnitude_bit_length()


def test_executor_preserves_order_empty_runs_and_result_lifetime() raises:
    var wide = Integer(1) << 600
    for count in [0, 1, 63, 64, 65, 257]:
        for probe in [False, True]:
            var values = _execute_values[_Kernel](_Job(wide, -1000, -1), count, _MapErrors("", ""), probe=probe)
            var kept = values^.take_list()
            assert_equal(len(kept), count)
            for value in kept:
                assert_equal(value, wide)
    for count in [-1, Int.MAX]:
        with assert_raises(contains="size exceeds the addressable range"):
            _ = _execute_values[_Kernel](_Job(wide, -1000, -1), count, _MapErrors("", ""))


def _borrow(x: Integer) raises -> Integer:
    assert_equal(x._storage[Integer._Shared].count(), UInt64(1))
    return x >> 200


def _borrow2(x: Integer, y: Integer) raises -> Integer:
    return _borrow(x) - _borrow(y)


def _borrow3(x: Integer, y: Integer, z: Integer) raises -> Integer:
    return _borrow(x) - _borrow(y) + _borrow(z)


def test_vmap_arities_keep_chunk_readers_borrowed_and_follow_nested_axes() raises:
    var values = Batch[Integer]([Integer(i + 1) << 200 for i in range(130)]).reshape([2, 65]).transpose()
    var reversed = values.slice(0, step=-1)
    var unary = vmap[_borrow]().vmap()(values)
    var binary = vmap[_borrow2]().vmap()(values, reversed)
    var ternary = vmap[_borrow3]().vmap()(values, reversed, values)
    for i in range(65):
        for j in range(2):
            var a = Integer(j * 65 + i + 1)
            var b = Integer(j * 65 + 65 - i)
            assert_equal(unary[i, j], a)
            assert_equal(binary[i, j], a - b)
            assert_equal(ternary[i, j], a - b + a)


def _retain_or_fail(x: Integer, y: Integer) raises -> Integer:
    if x == 17 or x == 109:
        raise Error("retained callback")
    return y


def test_frontends_preserve_coordinates_and_destination_on_failure() raises:
    var wide = Integer(1) << 600
    var xs = Batch[Integer]([Integer(i) for i in range(260)]).reshape([2, 130])
    var destination = Batch[Integer]([wide])
    var owners = wide._storage[Integer._Shared].count()
    for _ in range(3):
        with assert_raises(contains="vmap failed at mapped index 0: vmap failed at mapped index 17: retained callback"):
            destination = vmap[_retain_or_fail](in_axes=(0, None)).vmap(in_axes=(0, None))(xs, wide)
        with assert_raises(contains="call failed at index [0, 17]: retained callback"):
            destination = lift[_retain_or_fail]()(xs, wide)
        with assert_raises(contains="outer failed at index [0, 17]: retained callback"):
            destination = lift[_retain_or_fail]().outer(xs, wide)
        assert_equal(destination[0], wide)
        assert_equal(wide._storage[Integer._Shared].count(), owners)


def test_mixed_complex_comparisons_keep_mask_tail_bits_and_order() raises:
    for count in [0, 1, 63, 64, 65, 513]:
        var values = Batch[Complex]([Complex(i % 3, i % 2) for i in range(count)])
        var exact = Batch[Integer]([Integer(i % 3) for i in range(count)])
        assert_equal((values == exact).to_list(), [i % 2 == 0 for i in range(count)])
        assert_equal((values != exact).to_list(), [i % 2 != 0 for i in range(count)])
        assert_equal((values[::-1] == exact[::-1]).to_list(), [(count - 1 - i) % 2 == 0 for i in range(count)])


def test_thread_count() raises:
    # Any thread count gives the sequential results; more threads than cores restart the pool.
    var c = ArithmeticContext(format=FloatFormat.binary64())
    var values = List[Float]()
    for i in range(2048):
        values.append(Float(i + 1, context=c))
    var xs = Batch[Float](values)
    var default = get_num_threads()
    assert_true(default >= 1)
    set_num_threads(1)
    assert_equal(get_num_threads(), 1)
    var one = vmap[float.exp]()(xs, context=c).to_json()
    set_num_threads(2 * default + 1)
    assert_equal(get_num_threads(), 2 * default + 1)
    var many = vmap[float.exp]()(xs, context=c).to_json()
    set_num_threads(2)
    var two = vmap[float.exp]()(xs, context=c).to_json()
    set_num_threads(default)
    assert_equal(get_num_threads(), default)
    assert_equal(one, many)
    assert_equal(one, two)
    with assert_raises():
        set_num_threads(0)
    _ = setenv("APN_MOJO_NUM_THREADS", "3")
    assert_equal(_requested_threads(), 3)
    _ = setenv("APN_MOJO_NUM_THREADS", "none")
    assert_true(_requested_threads() >= 1)
    _ = unsetenv("APN_MOJO_NUM_THREADS")


def _resize_inside(x: Float) raises -> Float:
    set_num_threads(3)
    return float.exp(x, context=ArithmeticContext(format=FloatFormat(2000)))


def test_thread_count_inside_a_job() raises:
    # Set from inside a parallel job, the count waits for the next job instead
    # of waiting for the running one, which would never finish.
    var default = get_num_threads()
    var values = List[Float]()
    for i in range(512):
        values.append(Float(i))
    var xs = Batch[Float](values)
    var ys = vmap[_resize_inside]()(xs)
    assert_equal(len(ys), 512)
    assert_equal(get_num_threads(), 3)
    var c = ArithmeticContext(format=FloatFormat.binary64())
    _ = vmap[float.exp]()(xs, context=c)
    assert_equal(get_num_threads(), 3)
    set_num_threads(default)
    assert_equal(get_num_threads(), default)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
