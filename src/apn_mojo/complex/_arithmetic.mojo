"""Round-once scalar complex arithmetic, retaining exact mixed real inputs."""

from ..integer.value import Integer
from ..float._input import _FloatInput
from ..float._arithmetic import _float_input_operation, _float_special, _float_sum, _wide_sum, _signed_sum
from ..float._rounding import _RoundedBinary, _exponent_add, _round_wide, _round_ratio, _round_quotient_native
from ..float.status import NumericStatus
from ..float.context import ArithmeticContext, FloatFormat, RoundingMode
from .context import ComplexContext, _ComplexContextArgument
from ._format import _ComplexFormats, _merge_complex_formats
from ..float._format import _FormatSlot
from ._input import _ComplexArgument
from ._finite import _negated, _finite_product, _ExactPair, _pair_quotient


def _real_op(
    a: _FloatInput,
    b: _FloatInput,
    operation: Int,
    context: ArithmeticContext,
    fail: Bool = False,
) raises -> _RoundedBinary:
    return _float_input_operation(a, b, operation, context, fail)


def _target(
    left: _ComplexArgument,
    right: _ComplexArgument,
    context: _ComplexContextArgument,
) raises -> ComplexContext:
    if context and (left.complex or right.complex):
        # An explicit context decides both formats. ComplexContext has checked
        # that their bounds agree; Complex operands keep agreeing bounds and
        # argument records hold one family each, so nothing needs merging.
        return context.value()
    if not context and left.complex and right.complex:
        ref lr = left.real.format.value()
        ref li = left.imag.format.value()
        if (
            lr == right.real.format.value()
            and li == right.imag.format.value()
            and lr.emin() == li.emin()
            and lr.emax() == li.emax()
        ):
            # Matching operands: each component keeps its format.
            return ComplexContext(
                _validated=(
                    ArithmeticContext(format=lr), ArithmeticContext(format=li)
                )
            )
    var a: Optional[_ComplexFormats] = None
    var b: Optional[_ComplexFormats] = None
    var ar: _FormatSlot = None
    var br: _FormatSlot = None
    if left.complex:
        a = _ComplexFormats(left.real.format.value(), left.imag.format.value())
    else:
        ar = left.real.format
    if right.complex:
        b = _ComplexFormats(
            right.real.format.value(), right.imag.format.value()
        )
    else:
        br = right.real.format
    var destination: Optional[_ComplexFormats] = None
    if context:
        destination = _ComplexFormats(
            context.value().real().format(), context.value().imag().format()
        )
    var formats = _merge_complex_formats(
        a,
        b,
        left_real=ar,
        right_real=br,
        left_native_precision=left.real.native_precision,
        right_native_precision=right.real.native_precision,
        destination=destination,
    )
    return context.value() if context else ComplexContext(
        real=ArithmeticContext(format=formats.real()),
        imag=ArithmeticContext(format=formats.imag()),
    )


def _unit(value: _FloatInput, infinite_operand: Bool) -> _FloatInput:
    var nonzero = value.kind == 2 if infinite_operand else value.kind == 1
    return _FloatInput(
        Int(nonzero), value.negative, Integer(Int(nonzero)), Integer(1), 0
    )


@fieldwise_init
struct _Symbol(ImplicitlyCopyable):
    var kind: Int
    var negative: Bool
    var invalid: Bool


def _symbol_product(a: _FloatInput, b: _FloatInput) -> _Symbol:
    var invalid = (a.kind == 0 and b.kind == 2) or (a.kind == 2 and b.kind == 0)
    return _Symbol(
        3 if invalid
        or a.kind == 3
        or b.kind == 3 else 2 if a.kind == 2
        or b.kind == 2 else 0 if a.kind == 0
        or b.kind == 0 else 1,
        a.negative != b.negative,
        invalid,
    )


def _symbol_sum(a: _Symbol, b: _Symbol) -> _Symbol:
    var invalid = (
        a.invalid
        or b.invalid
        or (a.kind == 2 and b.kind == 2 and a.negative != b.negative)
    )
    return _Symbol(
        3 if invalid or a.kind == 3 or b.kind == 3 else 2,
        a.negative if a.kind == 2 else b.negative,
        invalid,
    )


def _signed_unit(value: _FloatInput, infinite_operand: Bool) -> Int:
    var unit = _unit(value, infinite_operand)
    return (-1 if unit.negative else 1) if unit.kind else 0


def _special_product_symbols(
    a: _FloatInput,
    b: _FloatInput,
    c: _FloatInput,
    d: _FloatInput,
) -> Tuple[_Symbol, _Symbol]:
    var r = _symbol_sum(_symbol_product(a, c), _symbol_product(_negated(b), d))
    var i = _symbol_sum(_symbol_product(a, d), _symbol_product(b, c))
    var ai = a.kind == 2 or b.kind == 2
    var bi = c.kind == 2 or d.kind == 2
    var first_real = a if ai else c
    var first_imag = b if ai else d
    var second_real = c if ai else a
    var second_imag = d if ai else b
    # Both real parts infinite and the first imaginary part finite and nonzero:
    # the imaginary part is the first cross term's infinity, where the sum of
    # the two cross terms would be undefined.
    if (
        first_real.kind == 2
        and second_real.kind == 2
        and first_imag.kind == 1
        and (second_imag.kind == 1 or second_imag.kind == 2)
    ):
        i = _Symbol(2, first_real.negative != second_imag.negative, False)
    if a.kind == 3 or b.kind == 3 or c.kind == 3 or d.kind == 3:
        r.kind = 3
        i.kind = 3
    if (ai or bi) and r.kind == 3 and i.kind == 3:
        var aa = _signed_unit(a, ai)
        var bb = _signed_unit(b, ai)
        var cc = _signed_unit(c, bi)
        var dd = _signed_unit(d, bi)
        var real = aa * cc - bb * dd
        var imag = aa * dd + bb * cc
        r = _Symbol(2 if real else 3, real < 0, not real)
        i = _Symbol(2 if imag else 3, imag < 0, not imag)
    return r, i


def _special_product(
    a: _FloatInput,
    b: _FloatInput,
    c: _FloatInput,
    d: _FloatInput,
    target: ComplexContext,
    fail: Int,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var symbols = _special_product_symbols(a, b, c, d)
    var r = symbols[0]
    var i = symbols[1]
    var real = _float_special(
        r.kind, r.negative, 16 if r.invalid else 0, target.real(), fail == 0
    )
    var imag = _float_special(
        i.kind, i.negative, 16 if i.invalid else 0, target.imag(), fail == 1
    )
    return (real, imag)


def _numerators(
    a: _FloatInput,
    b: _FloatInput,
    c: _FloatInput,
    d: _FloatInput,
    real_left: Bool = False,
) raises -> Tuple[_ExactPair, _ExactPair]:
    var ac = _finite_product(a, c)
    var bd = _finite_product(b, d)
    var bc = _finite_product(b, c)
    var ad = _negated(_finite_product(a, d))
    if real_left:
        bd.negative = ac.negative
        bc.negative = ad.negative
    return (
        _ExactPair(ac, bd, ac.negative and bd.negative),
        _ExactPair(bc, ad, bc.negative and ad.negative),
    )


def _direction_result(
    value: _ExactPair, kind: Int, context: ArithmeticContext, fail: Bool
) raises -> _RoundedBinary:
    var magnitude = value.magnitude()
    var zero = not len(magnitude.runs)
    return _float_special(
        3 if kind == 2 and zero else kind,
        magnitude.negative,
        16 if kind == 2 and zero else 0,
        context,
        fail,
    )


@always_inline
def _binary(value: _FloatInput) -> Bool:
    """A finite nonzero binary value: a Float, or an integer."""
    return value.kind == 1 and value.denominator._is_one()


@always_inline
def _short_binary(value: _FloatInput) -> Bool:
    """Zero, or a finite binary value with a significand under 2**63."""
    return value.kind == 0 or (
        value.kind == 1 and value.denominator._is_one() and value.numerator._storage.isa[Int64]()
    )


@always_inline
def _short_significand(value: _FloatInput) -> UInt128:
    return UInt128(UInt64(value.numerator._storage[Int64])) if value.kind else 0


@always_inline
def _native_pair(
    a: _FloatInput, c: _FloatInput, b: _FloatInput, d: _FloatInput, subtract: Bool,
    context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """Round a*c - b*d (or a*c + b*d) once from 128-bit native products."""
    var format = context.format()
    if format.precision() > 128 or format._is_exact():
        return _RoundedBinary(-1, False, Integer(), 0, format, NumericStatus())
    var first = _short_significand(a) * _short_significand(c)
    var second = _short_significand(b) * _short_significand(d)
    var first_negative = a.negative != c.negative
    var second_negative = (b.negative != d.negative) != subtract
    if not first or not second:
        # _wide_sum takes nonzero terms. One zero product leaves the other,
        # rounded once; two give the zero the exact path gives:
        # negative toward negative infinity, or when both products are.
        if first:
            return _round_wide(UInt256(first), first_negative, _exponent_add(a.scale, c.scale), context, fail)
        if second:
            return _round_wide(UInt256(second), second_negative, _exponent_add(b.scale, d.scale), context, fail)
        return _float_special(
            0,
            context.rounding() == RoundingMode.toward_negative
            or (first_negative and second_negative),
            0, context, fail,
        )
    return _wide_sum(
        first, first_negative, _exponent_add(a.scale, c.scale),
        second, second_negative, _exponent_add(b.scale, d.scale),
        context, fail,
    )


@always_inline
def _two_terms(
    x: UInt128, x_negative: Bool, x_scale: Int128,
    y: UInt128, y_negative: Bool, y_scale: Int128,
) -> Tuple[UInt256, Bool, Int128, Bool]:
    """x * 2**x_scale + y * 2**y_scale exactly, as a magnitude, a sign and a
    scale, and whether it fits: a zero term leaves the other, and nonzero
    terms may lie at most 127 bits apart."""
    if not x:
        return (UInt256(y), y_negative, y_scale, True)
    if not y:
        return (UInt256(x), x_negative, x_scale, True)
    var on_left = x_scale >= y_scale
    var gap = x_scale - y_scale if on_left else y_scale - x_scale
    if gap > 127:
        return (UInt256(0), False, Int128(0), False)
    var total, negative = _signed_sum(
        UInt256(x if on_left else y) << UInt256(gap), x_negative if on_left else y_negative,
        UInt256(y if on_left else x), y_negative if on_left else x_negative,
    )
    return (total, negative, y_scale if on_left else x_scale, True)


@always_inline
def _native_component(
    numerator: Tuple[UInt256, Bool, Int128, Bool], denominator: Tuple[UInt256, Bool, Int128, Bool],
    negative_zero: Bool, context: ArithmeticContext, fail: Bool,
) raises -> _RoundedBinary:
    """One component of a native complex quotient: its exact numerator over
    the shared denominator, rounded once; a zero numerator gives the general
    path's signed zero."""
    if not numerator[0]:
        return _round_ratio(Integer(0), Integer(1), context, negative_zero=negative_zero, fail=fail)
    return _round_quotient_native(
        numerator[0], denominator[0], numerator[1], _exponent_add(numerator[2], -denominator[2]),
        context, fail,
    )


def _native_quotients(
    a: _FloatInput, b: _FloatInput, c: _FloatInput, d: _FloatInput, real_left: Bool,
    real_context: ArithmeticContext, imag_context: ArithmeticContext, rf: Bool, inf: Bool,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    """(a + bi) / (c + di) for short binary parts and a nonzero divisor: the
    exact numerators ac + bd and bc - ad and the denominator c**2 + d**2 in
    256-bit native integers, each component rounded once. Kind -1 asks for the
    general path: a format past 128 bits or exact, terms more than 127 bits
    apart, or a result outside the exponent range."""
    real_context.rounding()._validate()
    imag_context.rounding()._validate()
    var unavailable = _RoundedBinary(-1, False, Integer(), 0, real_context.format(), NumericStatus())
    var real_format = real_context.format()
    var imag_format = imag_context.format()
    if (
        real_format.precision() > 128 or real_format._is_exact()
        or imag_format.precision() > 128 or imag_format._is_exact()
    ):
        return (unavailable, unavailable)
    var sa = _short_significand(a)
    var sb = _short_significand(b)
    var sc = _short_significand(c)
    var sd = _short_significand(d)
    var denominator = _two_terms(
        sc * sc, False, _exponent_add(c.scale, c.scale), sd * sd, False, _exponent_add(d.scale, d.scale)
    )
    # Signs of ac, bd, bc and -ad; a real dividend's zero imaginary part takes
    # its partners' signs, as in _numerators, which decides a zero's sign.
    var ac_negative = a.negative != c.negative
    var ad_negative = a.negative == d.negative
    var bd_negative = ac_negative if real_left else b.negative != d.negative
    var bc_negative = ad_negative if real_left else b.negative != c.negative
    var real = _two_terms(
        sa * sc, ac_negative, _exponent_add(a.scale, c.scale),
        sb * sd, bd_negative, _exponent_add(b.scale, d.scale),
    )
    var imag = _two_terms(
        sb * sc, bc_negative, _exponent_add(b.scale, c.scale),
        sa * sd, ad_negative, _exponent_add(a.scale, d.scale),
    )
    if not denominator[3] or not real[3] or not imag[3]:
        return (unavailable, unavailable)
    var real_part = _native_component(real, denominator, ac_negative and bd_negative, real_context, rf)
    if real_part.kind < 0:
        return (unavailable, unavailable)
    var imag_part = _native_component(imag, denominator, bc_negative and ad_negative, imag_context, inf)
    return (real_part^, imag_part^)


def _complex_operation(
    left: _ComplexArgument,
    right: _ComplexArgument,
    operation: Int,
    context: _ComplexContextArgument = _ComplexContextArgument(),
    *,
    fail_component: Int = -1,
) raises -> Tuple[_RoundedBinary, _RoundedBinary]:
    var target = _target(left, right, context)
    ref a = left.real.value
    ref b = left.imag.value
    ref c = right.real.value
    ref d = right.imag.value
    var rf = fail_component == 0
    var inf = fail_component == 1
    if operation < 2:
        var real = _real_op(a, c, operation, target.real(), rf)
        var imag: _RoundedBinary
        if left.complex and right.complex:
            imag = _real_op(b, d, operation, target.imag(), inf)
        else:
            var source = (
                b if left.complex else _negated(d) if operation == 1 else d
            )
            imag = _real_op(
                source,
                _unit(_FloatInput(1, False, Integer(1), Integer(1), 0), False),
                2,
                target.imag(),
                inf,
            )
        return (real, imag)
    if not right.complex or (operation == 2 and not left.complex):
        var real_operand = c if not right.complex else a
        var real_part = a if not right.complex else c
        var imag_part = b if not right.complex else d
        var real = _real_op(
            real_part, real_operand, operation, target.real(), rf
        )
        var imag = _real_op(
            imag_part, real_operand, operation, target.imag(), inf
        )
        return (real, imag)
    var finite_left = a.kind < 2 and b.kind < 2
    var finite_right = c.kind < 2 and d.kind < 2
    if operation == 2:
        if not finite_left or not finite_right:
            return _special_product(a, b, c, d, target, fail_component)
        if _short_binary(a) and _short_binary(b) and _short_binary(c) and _short_binary(d):
            # Four native products, each component one rounded two-term sum.
            var real = _native_pair(a, c, b, d, True, target.real(), rf)
            if real.kind >= 0:
                var imag = _native_pair(a, d, b, c, False, target.imag(), inf)
                if imag.kind >= 0:
                    return (real^, imag^)
        if _binary(a) and _binary(b) and _binary(c) and _binary(d):
            # Wider binary parts: exact Integer products, each component one
            # binary two-term sum.
            var real = _float_sum(
                _finite_product(a, c), _finite_product(b, d), target.real(), rf, negate_right=True
            )
            var imag = _float_sum(_finite_product(a, d), _finite_product(b, c), target.imag(), inf)
            return (real^, imag^)
        var ac = _finite_product(a, c)
        var bd = _negated(_finite_product(b, d))
        var ad = _finite_product(a, d)
        var bc = _finite_product(b, c)
        var real = _ExactPair(
            ac,
            bd,
            target.real().rounding() == RoundingMode.toward_negative
            or (ac.negative and bd.negative),
        ).rounded(target.real(), rf)
        var imag = _ExactPair(
            ad,
            bc,
            target.imag().rounding() == RoundingMode.toward_negative
            or (ad.negative and bc.negative),
        ).rounded(target.imag(), inf)
        return (real, imag)
    if c.kind == 0 and d.kind == 0:
        var infinity = _FloatInput(2, c.negative, Integer(0), Integer(1), 0)
        var real = _real_op(a, infinity, 2, target.real(), rf)
        var imag: _RoundedBinary
        # Finite nonzero / complex zero raises division-by-zero per component.
        if a.kind == 1:
            real = _float_special(
                2, a.negative != c.negative, 8, target.real(), rf
            )
        if not left.complex:
            imag = _float_special(3, False, 16, target.imag(), inf)
        elif b.kind == 1:
            imag = _float_special(
                2, b.negative != c.negative, 8, target.imag(), inf
            )
        else:
            imag = _real_op(b, infinity, 2, target.imag(), inf)
        return (real, imag)
    var left_inf = a.kind == 2 or b.kind == 2
    var right_inf = c.kind == 2 or d.kind == 2
    if left_inf and finite_right:
        var parts = _numerators(
            _unit(a, True), _unit(b, True), c, d, not left.complex
        )
        var real = _direction_result(parts[0], 2, target.real(), rf)
        var imag = _direction_result(parts[1], 2, target.imag(), inf)
        return (real, imag)
    if finite_left and right_inf:
        var parts = _numerators(
            a, b, _unit(c, True), _unit(d, True), not left.complex
        )
        var real = _direction_result(parts[0], 0, target.real(), rf)
        var imag = _direction_result(parts[1], 0, target.imag(), inf)
        return (real, imag)
    if not finite_left or not finite_right:
        var flags = 16 if left_inf and right_inf else 0
        var real = _float_special(3, False, flags, target.real(), rf)
        var imag = _float_special(3, False, flags, target.imag(), inf)
        return (real, imag)
    if _short_binary(a) and _short_binary(b) and _short_binary(c) and _short_binary(d):
        var native = _native_quotients(
            a, b, c, d, not left.complex, target.real(), target.imag(), rf, inf
        )
        if native[0].kind >= 0 and native[1].kind >= 0:
            return native^
    var parts = _numerators(a, b, c, d, not left.complex)
    var denominator = _ExactPair(
        _finite_product(c, c), _finite_product(d, d), False
    )
    var real = _pair_quotient(parts[0], denominator, target.real(), rf)
    var imag = _pair_quotient(parts[1], denominator, target.imag(), inf)
    return (real, imag)
