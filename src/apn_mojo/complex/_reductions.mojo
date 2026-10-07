"""Round-once component sums through the existing sparse Float reduction."""

from ..float._reductions import (
    _DotSource,
    _DotFlags,
    _DotMetadata,
    _DotPlan,
    _dot_metadata,
    _dot_denominator,
    _accumulate_dot_product,
    _prepare_dot,
    _execute_dot,
    _dot_shape,
    _DotPart,
    _dot_part,
    _accumulate_run,
)
from ..float._batch_ops import _FloatOperand
from ..float._input import _FloatInput
from ..float._accumulator import _ExactAccumulator
from ..integer.value import Integer
from ..float._rounding import _RoundedBinary
from ..float.context import ArithmeticContext, FloatFormat
from ..float.value import Float
from ._batch_storage import _ComplexInput
from ._format import _ComplexFormats
from .context import _ComplexContextArgument
from .status import ComplexStatus
from .value import Complex
from ._batch_ops import _ComplexOperand
from ._arithmetic import _target, _special_product_symbols


def _dot_component(input: _ComplexOperand, imag: Bool) -> _FloatOperand:
    if input.components:
        var pair = input.components.value()
        return _FloatOperand(
            pair.imag if imag else pair.real, None, None, input.length, False
        )
    return input.real.value()


def _dot_symbol(value: _DotMetadata, negate: Bool = False) -> _FloatInput:
    return _FloatInput(
        value.kind,
        value.negative != (negate and value.kind != 3),
        Integer(0),
        Integer(1),
        0,
    )


@fieldwise_init
struct _ComplexDotSource(_DotSource):
    var left: _ComplexOperand
    var right: _ComplexOperand
    var imag: Bool
    var conjugate: Bool
    var context: _ComplexContextArgument

    def length(self) -> Int:
        return self.left.length

    def format(
        self, index: Int, context: Optional[ArithmeticContext]
    ) raises -> FloatFormat:
        var target = _target(
            self.left.formats(index), self.right.formats(index), self.context
        )
        return target.imag().format() if self.imag else target.real().format()

    def first(self) -> Tuple[_FloatOperand, _FloatOperand]:
        return (
            _dot_component(self.left, self.imag and not self.right.components),
            _dot_component(self.right, self.imag),
        )

    def classify(self, index: Int) -> _DotFlags:
        var flags = _DotFlags()
        var first = self.first()
        if not self.left.components or not self.right.components:
            flags.product(
                _dot_metadata(first[0], index),
                _dot_metadata(first[1], index),
                self.conjugate and self.imag and Bool(self.left.components),
            )
            return flags
        var a = _dot_metadata(_dot_component(self.left, False), index)
        var b = _dot_metadata(_dot_component(self.left, True), index)
        var c = _dot_metadata(_dot_component(self.right, False), index)
        var d = _dot_metadata(_dot_component(self.right, True), index)
        if a.kind >= 2 or b.kind >= 2 or c.kind >= 2 or d.kind >= 2:
            var pair = _special_product_symbols(
                _dot_symbol(a),
                _dot_symbol(b, self.conjugate),
                _dot_symbol(c),
                _dot_symbol(d),
            )
            var symbol = pair[1] if self.imag else pair[0]
            flags.nan = symbol.kind == 3
            flags.invalid = symbol.invalid
            flags.positive_infinity = symbol.kind == 2 and not symbol.negative
            flags.negative_infinity = symbol.kind == 2 and symbol.negative
        else:
            flags.product(a, d if self.imag else c)
            flags.product(
                b, c if self.imag else d, self.conjugate != (not self.imag)
            )
        return flags

    def denominator(self, index: Int) raises -> Integer:
        var pair = self.first()
        return _dot_denominator(pair[0], pair[1], index)

    def part(self, start: Int, end: Int, context: Optional[ArithmeticContext]) raises -> _DotPart:
        return _dot_part(self, start, end, context)

    def accumulate_run(
        self, start: Int, end: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        _accumulate_run(self, start, end, denominator, total)

    def accumulate(
        self, index: Int, denominator: Integer, mut total: _ExactAccumulator
    ) raises:
        var pair = self.first()
        _accumulate_dot_product(
            pair[0],
            pair[1],
            index,
            denominator,
            total,
            self.conjugate
            and self.imag
            and Bool(self.left.components)
            and not self.right.components,
        )
        if self.left.components and self.right.components:
            _accumulate_dot_product(
                _dot_component(self.left, True),
                _dot_component(self.right, not self.imag),
                index,
                denominator,
                total,
                self.conjugate != (not self.imag),
            )


def _dot_complex(
    left: _ComplexOperand,
    right: _ComplexOperand,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *,
    conjugate: Bool = False,
    fail_after_element: Int = -1,
    fail_component: Int = 0,
    fail: Bool = False,
) raises -> Tuple[Complex, ComplexStatus]:
    var name = String("Complex vdot" if conjugate else "Complex dot")
    _dot_shape(left.length, right.length, name)
    var real_source = _ComplexDotSource(left, right, False, conjugate, context)
    var imag_source = _ComplexDotSource(left, right, True, conjugate, context)
    var real_context: Optional[ArithmeticContext] = None
    var imag_context: Optional[ArithmeticContext] = None
    if context:
        real_context = context.value().real()
        imag_context = context.value().imag()
    var real: _DotPlan
    var imag: _DotPlan
    try:
        real = _prepare_dot(real_source, real_context)
    except error:
        raise Error(String(name, " real component: ", error))
    try:
        imag = _prepare_dot(imag_source, imag_context)
    except error:
        raise Error(String(name, " imaginary component: ", error))
    _ = _ComplexFormats(real.target.format(), imag.target.format())
    var a: _RoundedBinary
    var b: _RoundedBinary
    try:
        a = _execute_dot(
            real_source,
            real,
            fail_after_element=fail_after_element if fail_component
            == 0 else -1,
            fail=fail and fail_component == 0,
        )
    except error:
        raise Error(String(name, " real component: ", error))
    try:
        b = _execute_dot(
            imag_source,
            imag,
            fail_after_element=fail_after_element if fail_component
            == 1 else -1,
            fail=fail and fail_component == 1,
        )
    except error:
        raise Error(String(name, " imaginary component: ", error))
    return (
        Complex(_real=Float(_rounded=a), _imag=Float(_rounded=b)),
        ComplexStatus._make(a.status, b.status),
    )


