"""The element families of a Batch, listed once.

Every family-dependent dispatch in the batch layer reads this list: the
untyped array's buffer and tag (`_array.mojo`), the types vmap maps
(`mapping.mojo`), and the generic container paths of `Batch` (views,
selections, assignment, iteration and printing). A family that needs only a
container is one entry here. Arithmetic operators, JSON and conversions from
other families are written per family; `_Arithmetic` names the families that
have them.
"""

from std.builtin.rebind import downcast
from std.memory import ArcPointer
from std.utils import Variant
from ._values import _Values
from ..common._traits import _BatchElement
from ..integer.value import Integer
from ..rational.value import Rational
from ..float.value import Float
from ..complex.value import Complex
from ..ball.value import Ball
from ..complex_ball.value import ComplexBall

comptime _Families = Tuple[Integer, Rational, Float, Complex, Ball, ComplexBall]
comptime _FAMILY_COUNT = _Families.Ts.length
comptime _Family[index: Int] = downcast[_Families.Ts[index], ImplicitlyCopyable & Deinitable & Writable]
comptime _Owner[T: Movable] = ArcPointer[_Values[downcast[T, ImplicitlyCopyable & Deinitable]]]
# One shared value list of any family; the array's tag says which.
comptime _Buffer = Variant[*_Families.Ts.map[ToTrait=Copyable, Mapper=_Owner]()]
# The arithmetic families: batch operators, JSON, lift and reductions apply to
# them. Other families are containers whose functions run through vmap.
comptime _Arithmetic[T: AnyType] = T == Integer or T == Rational or T == Float or T == Complex


def _family_index[T: AnyType]() -> Int:
    """The position of `T` in `_Families`, or -1 for another type."""
    comptime for index in range(_FAMILY_COUNT):
        comptime if T == _Families.Ts[index]:
            return index
    return -1


def _family_of[T: AnyType]() -> Int:
    """The tag of a family's arrays: its position in `_Families`."""
    comptime index = _family_index[T]()
    comptime assert index >= 0, (
        "Batch elements must be one of the families listed in"
        " batch/_families.mojo; add a new family there."
    )
    comptime assert conforms_to(T, _BatchElement), (
        "A family listed in batch/_families.mojo declares _BatchElement."
    )
    return index

