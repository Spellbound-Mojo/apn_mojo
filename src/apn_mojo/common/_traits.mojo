"""Marker traits shared by every numeric family; no numeric dependencies."""


trait _MapArgument:
    """A value mapping accepts as an argument: a library number, batch, mask, Bool or literal.

    Library types declare it directly: conformance added only through
    `__extension` is invisible to functions the compiler checks before the
    extended type is first used. Natives are deliberately absent: a generic
    parameter that native integers satisfy narrows a wide literal to Int, so
    literals must bind as themselves.
    """

    pass


trait _BatchElement(_MapArgument):
    """An element family of Batch, listed in batch/_families.mojo.

    The family's struct declares it, so overload constraints can test
    membership with `conforms_to`: a `where` clause cannot call the function
    that searches the list. A family's values are vmap arguments.
    """

    @staticmethod
    def _placeholder() -> Self:
        """Zero, without heap storage: it fills a batch position with no
        result, where a mapped function returned no value, and the slot a
        tuple result's field moved out of."""
        ...


trait _MapAdapter(ImplicitlyCopyable, Deinitable):
    """A function argument that accepts several element families, such as any
    real number: vmap and lift adapt each element or shared scalar to it.

    A family's argument union declares it (`_FloatArgument`, `_ComplexArgument`, `_BallArgument`),
    so vmap needs no list of adapters.
    """

    @staticmethod
    def _adapt[V: ImplicitlyCopyable & Deinitable](value: V) raises -> Self:
        """The argument for one element or scalar, converted exactly."""
        ...


# Extensions count only beside their trait, so the standard types the library
# cannot edit gain _MapArgument here; library types declare it directly.
__extension IntLiteral(_MapArgument):
    pass


__extension FloatLiteral(_MapArgument):
    pass


__extension Bool(_MapArgument):
    pass
