"""The working precision and budget of ball functions."""


struct BallContext(ImplicitlyCopyable, Writable):
    """The working precision of a ball result, and the budget of functions
    whose cost depends on their input.

    It is a ball function's only keyword, `context=`, so `vmap` and `lift`
    pass it through to scalar ball functions. Without a precision, a result takes the largest
    midpoint precision among its Ball and Float operands; exact Integer and
    Rational operands contribute none, and a result of exact operands alone
    has 128 bits. Rounding modes and traps do not apply to balls.
    """

    var _precision: Optional[Int]
    var _max_precision: Optional[Int]

    def __init__(out self, precision: Optional[Int] = None, *, max_precision: Optional[Int] = None) raises:
        """A context with an optional working precision and budget.

        Args:
            precision: The midpoint precision of results, in bits, at least 2.
            max_precision: The largest working precision a function may use;
                by default, as the function documents.

        Raises:
            When a precision is below 2.
        """
        if (precision and precision.value() < 2) or (max_precision and max_precision.value() < 2):
            raise Error(
                "Cannot create BallContext with a precision below 2 bits;"
                " choose at least 2. The destination is unchanged."
            )
        self._precision = precision
        self._max_precision = max_precision

    def precision(self) -> Optional[Int]:
        """The working precision, if one is set.

        Returns:
            The precision in bits, or None to follow the operands.
        """
        return self._precision

    def max_precision(self) -> Optional[Int]:
        """The budget, if one is set.

        Returns:
            The largest working precision in bits, or None for the default.
        """
        return self._max_precision

    def write_to(self, mut writer: Some[Writer]):
        """Write the settings, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("BallContext(precision=")
        if self._precision:
            writer.write(self._precision.value())
        else:
            writer.write("None")
        writer.write(", max_precision=")
        if self._max_precision:
            writer.write(self._max_precision.value())
        else:
            writer.write("None")
        writer.write(")")
