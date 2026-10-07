"""Result collection preserves borrowing, cleanup and family-independent mapping."""

from std.testing import TestSuite, assert_equal, assert_true, assert_raises
from apn_mojo import Batch, Integer, Ball, Mask, vmap


def _owners(x: Integer) -> Tuple[Integer, Bool]:
    return (Integer(x._storage[Integer._Shared].count()), True)


def _pair_owners(x: Integer, y: Integer) -> Tuple[Integer, Integer]:
    return (Integer(x._storage[Integer._Shared].count()), Integer(y._storage[Integer._Shared].count()))


def _optional_owners(x: Integer) -> Optional[Integer]:
    return Integer(x._storage[Integer._Shared].count())


def test_tuple_and_optional_results_keep_arguments_borrowed() raises:
    var values = Batch[Integer]([Integer(i + 1) << 200 for i in range(63)])
    var ones = [Integer(1) for _ in range(63)]
    for forced in [False, True]:
        var size: Optional[Int] = 63 if forced else None
        var unary = vmap[_owners](axis_size=size)(values[::-1])
        var binary = vmap[_pair_owners](axis_size=size)(values, values[::-1])
        var optional = vmap[_optional_owners](axis_size=size)(values)
        assert_equal(unary[0].to_list(), ones)
        assert_true(unary[1].all())
        assert_equal(binary[0].to_list(), ones)
        assert_equal(binary[1].to_list(), ones)
        assert_equal(optional[0].to_list(), ones)
        assert_true(optional[1].all())


def _retained(x: Integer, marker: Integer) raises -> Tuple[Integer, Integer, Bool]:
    var copy = x
    if copy == marker:
        raise Error("tuple callback failed")
    return (copy, x, True)


def test_tuple_failures_release_partial_results_and_keep_destination() raises:
    var values = Batch[Integer]([Integer(i + 1) << 200 for i in range(17)])
    var destination = (Batch[Integer]([91]), Batch[Integer]([92]), Mask([False]))
    for forced in [False, True]:
        var size: Optional[Int] = 17 if forced else None
        for index in range(17):
            with assert_raises(contains=String("vmap failed at mapped index ", index, ": tuple callback failed")):
                destination = vmap[_retained](in_axes=(0, None), axis_size=size)(values, values[index])
            assert_equal(destination[0].to_list(), [Integer(91)])
            assert_equal(destination[1].to_list(), [Integer(92)])
            assert_equal(destination[2].to_list(), [False])
            assert_equal(vmap[_owners]()(values)[0].to_list(), [Integer(1) for _ in range(17)])
    var kept = vmap[_retained](in_axes=(0, None))(values[::-1], Integer(0))
    values = Batch[Integer]([])
    for i in range(17):
        assert_equal(kept[0][i], Integer(17 - i) << 200)
        assert_equal(kept[1][i], kept[0][i])
    assert_true(kept[2].all())


def _changing_mask(x: Integer) raises -> Tuple[Integer, Mask]:
    return (x, Mask([True, False]) if x == 2 else Mask([True]))


def test_failure_in_later_result_leaf_releases_earlier_leaves() raises:
    var wide = Integer(1) << 200
    var values = Batch[Integer]([wide, 2, 3])
    var owners = wide._storage[Integer._Shared].count()
    var destination = (Batch[Integer]([91]), Mask([False], shape=[1, 1]))
    with assert_raises(contains="vmap failed at mapped index 1: Mapped function returned inconsistent mask shapes"):
        destination = vmap[_changing_mask]()(values)
    assert_equal(destination[0].to_list(), [Integer(91)])
    assert_equal(destination[1].shape(), [1, 1])
    assert_equal(destination[1].to_list(), [False])
    assert_equal(wide._storage[Integer._Shared].count(), owners)
    assert_equal(values[0], wide)


def _ball_parts(x: Ball) raises -> Tuple[Ball, Integer, Bool]:
    return (x, x.midpoint().floor(), x.is_exact())


def _ball_tuple(parts: Tuple[Ball, Integer, Bool]) raises -> Tuple[Ball, Integer, Bool]:
    return (parts[0], parts[1], parts[2])


def test_ball_tuple_outputs_keep_nested_axes_and_empty_shapes() raises:
    var values = Batch[Ball]([Ball(i) for i in range(6)]).reshape([2, 3])
    var marks = Mask([True, False, True, False, True, False], shape=[2, 3])
    for forced in [False, True]:
        var size: Optional[Int] = 3 if forced else None
        var result = vmap[_ball_parts](axis_size=size).vmap(out_axes=(1, 0, 1))(values)
        assert_equal(result[0].shape(), [3, 2])
        assert_equal(result[1].shape(), [2, 3])
        assert_equal(result[2].shape(), [3, 2])
        for i in range(2):
            for j in range(3):
                assert_true(result[0][j, i].same_representation(values[i, j]))
                assert_equal(result[1][i, j], Integer(i * 3 + j))
        assert_true(result[2].all())
        var tuple_result = vmap[_ball_tuple](in_axes=(0, None, 0), axis_size=size).vmap(in_axes=(0, None, 0))((values, Integer(7), marks))
        assert_equal(tuple_result[1].to_list(), [Integer(7) for _ in range(6)])
        assert_equal(tuple_result[2].to_list(), marks.to_list())
        for i in range(2):
            for j in range(3):
                assert_true(tuple_result[0][i, j].same_representation(values[i, j]))
    var empty = vmap[_ball_parts]().vmap(out_shape=([3], [3], [3]))(values.slice(0, stop=0))
    assert_equal(empty[0].shape(), [0, 3])
    assert_equal(empty[1].shape(), [0, 3])
    assert_equal(empty[2].shape(), [0, 3])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
