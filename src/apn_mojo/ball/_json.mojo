"""Version-1 Ball and ComplexBall records.

A Ball record holds its kind and two complete Float records, the midpoint and
the radius; a ComplexBall record holds two Ball records. Only canonical
records are read: the radius has precision 30, the default bounds and sign
`+`; a finite ball has a finite midpoint and radius, an unbounded ball a
finite midpoint and an infinite radius, and an indeterminate ball a NaN
midpoint and an infinite radius. A record written by `to_json` reads back to
the same representation."""

from ..common._json import _JSONReader
from ..common._sizes import _checked_sum
from ..common.conversion import _ConversionBudget
from ..float._json import _read_float_json_record, _format_float_json
from ..float._rounding import _RoundedBinary
from ..float.context import FloatFormat
from .value import _FINITE, _UNBOUNDED, _INDETERMINATE


@fieldwise_init
struct _BallRecord(ImplicitlyCopyable):
    """A Ball's kind (0 finite, 1 unbounded, 2 indeterminate), midpoint and
    radius, as read or to be written."""

    var kind: Int
    var midpoint: _RoundedBinary
    var radius: _RoundedBinary


def _format_ball_json(
    value: _BallRecord, mut budget: _ConversionBudget, element: Int = -1, *, count_value: Bool = True,
) raises -> String:
    comptime separator: StaticString = ',"radius":'
    var kind: StaticString = "indeterminate"
    if value.kind == _FINITE:
        kind = "finite"
    elif value.kind == _UNBOUNDED:
        kind = "unbounded"
    var prefix = String('{"version":1,"family":"ball","kind":"', kind, '","midpoint":')
    var overhead = _checked_sum(prefix.byte_length(), separator.byte_length() + 1)
    if count_value:
        budget.values(1, element)
    budget.output(overhead, element)
    var midpoint = _format_float_json(value.midpoint, budget, element, count_value=False)
    var radius = _format_float_json(value.radius, budget, element, count_value=False)
    var count = _checked_sum(overhead, _checked_sum(midpoint.byte_length(), radius.byte_length()))
    var result = String()
    budget.grow_string(result, count, element)
    result.reserve_bytes(count)
    result.write(prefix, midpoint, separator, radius, "}")
    return result


def _read_ball_json_record(mut reader: _JSONReader) raises -> _BallRecord:
    var diagnostic = reader.diagnostic_prefix
    reader.expect(123, "provide an object produced by Ball.to_json()")
    var seen = 0
    var kind = -1
    var midpoint: Optional[_RoundedBinary] = None
    var radius: Optional[_RoundedBinary] = None
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "kind":
            flag = 4
        elif key == "midpoint":
            flag = 8
        elif key == "radius":
            flag = 16
        else:
            reader.fail("unknown field", "use version, family, kind, midpoint and radius from Ball.to_json()")
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one("use Ball interchange version 1")
        elif flag == 2:
            reader.family("ball", "use family ball with Ball.from_json()")
        elif flag == 4:
            var name = reader.string()
            if name == "finite":
                kind = _FINITE
            elif name == "unbounded":
                kind = _UNBOUNDED
            elif name == "indeterminate":
                kind = _INDETERMINATE
            else:
                reader.fail("unknown kind", "use kind finite, unbounded or indeterminate")
        else:
            reader.diagnostic_prefix = "Invalid Ball midpoint at byte " if flag == 8 else "Invalid Ball radius at byte "
            var record = _read_float_json_record(reader)
            if flag == 8:
                midpoint = record
            else:
                radius = record
            reader.diagnostic_prefix = diagnostic
        if reader.end_object("separate fields with commas and close the object with }"):
            break
    reader.require_fields(seen, 31, "include version, family, kind, midpoint and radius")
    var m = midpoint.value()
    var r = radius.value()
    if (
        r.format.precision() != 30 or r.format.emin() != FloatFormat.DEFAULT_EMIN
        or r.format.emax() != FloatFormat.DEFAULT_EMAX or r.negative or r.kind == 3
    ):
        reader.fail("non-canonical radius", "write the radius as Ball.to_json() does: 30 bits, the default bounds, sign +")
    var finite_midpoint = m.kind == 0 or m.kind == 1
    if kind == _FINITE and not (finite_midpoint and r.kind != 2):
        reader.fail("non-canonical finite ball", "give a finite ball a finite midpoint and radius")
    if kind == _UNBOUNDED and not (finite_midpoint and r.kind == 2):
        reader.fail("non-canonical unbounded ball", "give an unbounded ball a finite midpoint and an infinite radius")
    if kind == _INDETERMINATE and not (m.kind == 3 and r.kind == 2):
        reader.fail("non-canonical indeterminate ball", "give an indeterminate ball a NaN midpoint and an infinite radius")
    return _BallRecord(kind, m, r)


def _read_ball_json(text: String, mut budget: _ConversionBudget) raises -> _BallRecord:
    budget.input(text.byte_length())
    budget.values(1)
    var reader = _JSONReader(text, budget, "Invalid ball interchange at byte ")
    var result = _read_ball_json_record(reader)
    reader.finish()
    budget = reader.budget
    return result


def _format_complex_ball_json(
    real: _BallRecord, imag: _BallRecord, mut budget: _ConversionBudget, element: Int = -1,
) raises -> String:
    comptime prefix: StaticString = '{"version":1,"family":"complex_ball","real":'
    comptime separator: StaticString = ',"imag":'
    comptime overhead = prefix.byte_length() + separator.byte_length() + 1
    budget.values(1, element)
    budget.output(overhead, element)
    var a = _format_ball_json(real, budget, element, count_value=False)
    var b = _format_ball_json(imag, budget, element, count_value=False)
    var count = _checked_sum(overhead, _checked_sum(a.byte_length(), b.byte_length()))
    var result = String()
    budget.grow_string(result, count, element)
    result.reserve_bytes(count)
    result.write(prefix, a, separator, b, "}")
    return result


def _read_complex_ball_json(text: String, mut budget: _ConversionBudget) raises -> Tuple[_BallRecord, _BallRecord]:
    budget.input(text.byte_length())
    budget.values(1)
    var reader = _JSONReader(text, budget, "Invalid complex ball interchange at byte ")
    var diagnostic = reader.diagnostic_prefix
    reader.expect(123, "provide an object produced by ComplexBall.to_json()")
    var seen = 0
    var real: Optional[_BallRecord] = None
    var imag: Optional[_BallRecord] = None
    while True:
        var key = reader.field_name()
        var flag = 0
        if key == "version":
            flag = 1
        elif key == "family":
            flag = 2
        elif key == "real":
            flag = 4
        elif key == "imag":
            flag = 8
        else:
            reader.fail("unknown field", "use version, family, real and imag from ComplexBall.to_json()")
        reader.mark_field(seen, flag)
        if flag == 1:
            reader.version_one("use ComplexBall interchange version 1")
        elif flag == 2:
            reader.family("complex_ball", "use family complex_ball with ComplexBall.from_json()")
        else:
            reader.diagnostic_prefix = "Invalid ComplexBall real part at byte " if flag == 4 else "Invalid ComplexBall imaginary part at byte "
            var part = _read_ball_json_record(reader)
            if flag == 4:
                real = part
            else:
                imag = part
            reader.diagnostic_prefix = diagnostic
        if reader.end_object("separate fields with commas and close the object with }"):
            break
    reader.require_fields(seen, 15, "include version, family, real and imag")
    reader.finish()
    budget = reader.budget
    return (real.value(), imag.value())
