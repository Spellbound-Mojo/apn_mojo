"""Masks and numeric layouts agree, while masks keep packed independent results."""

from std.testing import TestSuite, assert_equal, assert_raises
from apn_mojo import Mask, Batch, Integer


def _check_values(mask: Mask, numbers: Batch[Integer]) raises:
    assert_equal(mask.shape(), numbers.shape())
    assert_equal(len(mask), len(numbers))
    for i in range(len(mask)):
        assert_equal(mask[i], numbers[i] != 0)


def test_mask_permutations_and_axis_selections_match_numeric_layouts() raises:
    var values = [Bool(i % 3) for i in range(24)]
    var mask = Mask(values, shape=[2, 3, 4])
    var numbers = Batch[Integer]([Integer(Int(v)) for v in values], shape=[2, 3, 4])
    assert_equal(mask.at(1, 1).to_list(), [values[i] for i in [4, 5, 6, 7, 16, 17, 18, 19]])
    assert_equal(mask.at(-1, -1).to_list(), [values[i] for i in [3, 7, 11, 15, 19, 23]])
    assert_equal(mask.transpose([2, 0, 1]).to_list(), [values[i] for i in [
        0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
        2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23,
    ]])
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            var axes: List[Int] = [a - 3, b, 3 - a - b]
            var transposed = mask.transpose(axes)
            _check_values(transposed, numbers.transpose(axes))
            assert_equal(Int(transposed._storage.ptr()) == Int(mask._storage.ptr()), False)
    for axis in range(-3, 3):
        _check_values(mask.at(axis, -1), numbers.at(axis, -1))
    assert_equal(mask.to_list(), values)
    with assert_raises(contains="Mask transpose repeats an axis"):
        _ = mask.transpose([0, 1, -3])
    with assert_raises(contains="Mask transpose needs one axis per dimension"):
        _ = mask.transpose([0, 1])
    with assert_raises(contains="axis is out of range"):
        _ = mask.transpose([0, 1, 3])
    with assert_raises(contains="index is out of range"):
        _ = mask.at(0, 2)


def test_mask_scalar_empty_and_overflow_shapes() raises:
    var scalar = Mask([True], shape=[])
    assert_equal(scalar.transpose().shape(), [])
    assert_equal(scalar.transpose().item(), True)
    with assert_raises(contains="axis is out of range"):
        _ = scalar.at(0, 0)
    var empty = Mask(List[Bool](), shape=[Int.MAX, 0, Int.MAX])
    assert_equal(empty.transpose([2, 0, 1]).shape(), [Int.MAX, Int.MAX, 0])
    assert_equal(empty.at(0, -1).shape(), [0, Int.MAX])
    with assert_raises(contains="index is out of range"):
        _ = empty.at(1, 0)
    with assert_raises(contains="Mask dimensions must be nonnegative"):
        _ = Mask(List[Bool](), shape=[0, -1])
    with assert_raises(contains="Mask shape exceeds addressable storage"):
        _ = Mask(List[Bool](), shape=[Int.MAX, 2])
    with assert_raises(contains="Cannot allocate numeric storage"):
        _ = Mask(List[Bool](), shape=[Int.MAX])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
