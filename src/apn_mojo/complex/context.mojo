"""Component numerical settings, reusing the ordinary Float context."""

from ..float.context import ArithmeticContext, FloatFormat
from ._format import _ComplexFormats
from std.builtin.bool import Boolable


struct ComplexContext(ImplicitlyCopyable, Writable):
    """A pair of arithmetic contexts, one per component.

    Use it for different precisions, rounding modes or traps in the real and
    imaginary parts; an `ArithmeticContext` applies to both. The two formats must
    have common exponent bounds.
    """

    var _real: ArithmeticContext
    var _imag: ArithmeticContext

    @implicit
    def __init__(out self, context: ArithmeticContext):
        """The same context for both components.

        Args:
            context: The context.
        """
        self._real = context
        self._imag = context

    def __init__(
        out self,
        *,
        real: Optional[ArithmeticContext] = None,
        imag: Optional[ArithmeticContext] = None,
    ) raises:
        """One context per component.

        Args:
            real: The real part's context; the defaults when omitted.
            imag: The imaginary part's context; the real one when omitted.

        Raises:
            When the two formats have different exponent bounds.
        """
        var a = real.value() if real else ArithmeticContext()
        var b = imag.value() if imag else a
        _ = _ComplexFormats(a.format(), b.format())
        self._real = a
        self._imag = b

    def __init__(
        out self, *, _validated: Tuple[ArithmeticContext, ArithmeticContext]
    ):
        # The caller has checked that the component bounds agree.
        self._real = _validated[0]
        self._imag = _validated[1]

    def real(self) -> ArithmeticContext:
        """The real part's context.

        Returns:
            The context.
        """
        return self._real

    def imag(self) -> ArithmeticContext:
        """The imaginary part's context.

        Returns:
            The context.
        """
        return self._imag

    def write_to(self, mut writer: Some[Writer]):
        """Write both contexts.

        Args:
            writer: The destination.
        """
        writer.write(
            "ComplexContext(real=", self._real, ", imag=", self._imag, ")"
        )


@always_inline
def _absent_complex_context() -> ComplexContext:
    # The placeholder of an absent context, never read; built unvalidated.
    var placeholder = ArithmeticContext(
        _placeholder=FloatFormat(_validated=(1, FloatFormat.DEFAULT_EMIN, FloatFormat.DEFAULT_EMAX))
    )
    return ComplexContext(_validated=(placeholder, placeholder))


struct _ComplexContextArgument(Boolable, Defaultable, ImplicitlyCopyable):
    # One implicit step accepts uniform or component contexts on the pinned compiler.
    # A plain slot, a context and a flag, as _FormatSlot is for formats: as an
    # Optional[ComplexContext], every Complex function call built it through an
    # out-of-line initializer of 285 instructions.
    var _context: ComplexContext
    var _present: Bool
    var _uniform: Bool

    @implicit
    def __init__(out self, value: ComplexContext):
        self._context = value
        self._present = True
        self._uniform = False

    @implicit
    def __init__(out self, value: ArithmeticContext):
        self._context = ComplexContext(value)
        self._present = True
        self._uniform = True

    @implicit
    def __init__(out self, value: Optional[ComplexContext]):
        self._context = value.value() if value else _absent_complex_context()
        self._present = Bool(value)
        self._uniform = False

    @implicit
    def __init__(out self, value: Optional[ArithmeticContext]):
        self._context = ComplexContext(value.value()) if value else _absent_complex_context()
        self._present = Bool(value)
        self._uniform = True

    @implicit
    def __init__(out self, value: NoneType):
        self._context = _absent_complex_context()
        self._present = False
        self._uniform = True

    def __init__(out self):
        self._context = _absent_complex_context()
        self._present = False
        self._uniform = True

    def __bool__(self) -> Bool:
        return self._present

    def value(self) -> ComplexContext:
        return self._context
