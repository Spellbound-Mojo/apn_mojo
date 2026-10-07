"""Batch envelope schemas and conversion limits across all numeric families."""

from std.testing import TestSuite, assert_equal, assert_raises
from apn_mojo import Batch, Integer, Rational, Float, Complex, ConversionLimits


def _check_schema[T: ImplicitlyCopyable & Deinitable](values: Batch[T]) raises:
    var encoded = values.to_json()
    for counted in [False, True]:
        var limits: Optional[ConversionLimits] = None
        if counted:
            limits = ConversionLimits(max_digits=100000, max_allocated_bytes=1000000)
        assert_equal(Batch[T].from_json(encoded, limits=limits).to_json(), encoded)
        assert_equal(values.to_json(limits=limits), encoded)
        var destination = values
        for text in [
            encoded + " true",
            encoded.replace('"version":2', '"version":3'),
            encoded.replace('"version":2', '"version":1'),
            encoded.replace('"version":2,', ''),
            encoded.replace('"values":', '"value":'),
            encoded.replace('"values":', '"extra":0,"values":'),
            encoded.replace('"values":', '"values":[],"values":'),
            encoded.replace('"shape":["2","1"]', '"shape":["3","1"]'),
        ]:
            assert_equal(text == encoded, False)
            with assert_raises():
                destination = Batch[T].from_json(text, limits=limits)
            assert_equal(destination.to_json(), encoded)
    with assert_raises(contains="max_values"):
        _ = Batch[T].from_json(encoded, limits=ConversionLimits(max_values=1))
    with assert_raises(contains="max_output_bytes"):
        _ = values.to_json(limits=ConversionLimits(max_output_bytes=encoded.byte_length() - 1))
    assert_equal(values.to_json(limits=ConversionLimits(max_output_bytes=encoded.byte_length())), encoded)
    var empty = values[:0].reshape([0, 2])
    assert_equal(Batch[T].from_json(empty.to_json()).to_json(), empty.to_json())
    var scalar = values[:1].reshape([])
    assert_equal(Batch[T].from_json(scalar.to_json()).to_json(), scalar.to_json())


def test_shared_envelope_preserves_each_family_schema() raises:
    _check_schema(Batch[Integer]([1, -2]).reshape([2, 1]))
    _check_schema(Batch[Rational]([Rational(1, 3), Rational(-2, 7)]).reshape([2, 1]))
    _check_schema(Batch[Float]([Float(1), Float.zero(negative=True)]).reshape([2, 1]))
    _check_schema(Batch[Complex]([Complex(1, 2), Complex(-3, 4)]).reshape([2, 1]))


def test_exact_envelopes_reject_formats_and_scalar_schemas() raises:
    for text in [
        '{"version":1,"family":"integer-batch","values":[],"default_format":{}}',
        '{"version":1,"family":"integer","value":"1"}',
    ]:
        with assert_raises():
            _ = Batch[Integer].from_json(text)
    for text in [
        '{"version":1,"family":"rational-batch","values":[],"default_format":{}}',
        '{"version":1,"family":"rational","numerator":"1","denominator":"3"}',
    ]:
        with assert_raises():
            _ = Batch[Rational].from_json(text)
    with assert_raises():
        _ = Integer.from_json('{"version":1,"family":"integer-batch","values":["1"]}')


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
