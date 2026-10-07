"""Keys that compare and hash numbers by representation, for `Dict` and `Set`.

`Float` and `Complex` compare values with `==`: `0.0 == -0.0`, NaN equals
nothing, and 1 at 53 and 128 bits are equal. That makes them unsuitable as
dictionary keys, so they are not hashable. A key wraps one value and compares
its whole representation instead, as a computer algebra system needs when it
interns numbers: each signed zero, each format and NaN is its own key.
"""

from std.hashlib import Hasher
from ..float.value import Float
from ..complex.value import Complex
from ..float.math import stable_hash as _float_stable_hash
from ..complex.math import stable_hash as _complex_stable_hash
from ..ball.value import Ball


struct FloatKey(Comparable, Hashable, ImplicitlyCopyable, Writable):
    """A Float as a dictionary key, compared and hashed by its representation.

    Keys are equal when `Float.same_representation` holds, order by
    `Float.representation_cmp`, and hash by `stable_hash`, so a sorted list of
    keys is the same in every run.
    """

    var _value: Float

    def __init__(out self, value: Float):
        """Wrap a Float.

        Args:
            value: The Float.
        """
        self._value = value

    def value(self) -> Float:
        """The wrapped Float.

        Returns:
            The Float.
        """
        return self._value

    def __eq__(self, other: Self) -> Bool:
        return self._value.same_representation(other._value)

    def __ne__(self, other: Self) -> Bool:
        return not self._value.same_representation(other._value)

    def __lt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) < 0

    def __le__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) <= 0

    def __gt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) > 0

    def __ge__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) >= 0

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the representation, through its stable hash.

        Args:
            hasher: The hasher.
        """
        _float_stable_hash(self._value).__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        """Write `FloatKey(` and the Float, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("FloatKey(", self._value, ")")


struct ComplexKey(Comparable, Hashable, ImplicitlyCopyable, Writable):
    """A Complex as a dictionary key, compared and hashed by its representation.

    Keys are equal when `Complex.same_representation` holds, order by
    `Complex.representation_cmp`, and hash by `stable_hash`.
    """

    var _value: Complex

    def __init__(out self, value: Complex):
        """Wrap a Complex.

        Args:
            value: The Complex number.
        """
        self._value = value

    def value(self) -> Complex:
        """The wrapped Complex.

        Returns:
            The Complex number.
        """
        return self._value

    def __eq__(self, other: Self) -> Bool:
        return self._value.same_representation(other._value)

    def __ne__(self, other: Self) -> Bool:
        return not self._value.same_representation(other._value)

    def __lt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) < 0

    def __le__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) <= 0

    def __gt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) > 0

    def __ge__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) >= 0

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the representation, through its stable hash.

        Args:
            hasher: The hasher.
        """
        _complex_stable_hash(self._value).__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        """Write `ComplexKey(` and the Complex number, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("ComplexKey(", self._value, ")")


struct BallKey(Comparable, Hashable, ImplicitlyCopyable, Writable):
    """A Ball as a dictionary key, compared and hashed by its representation.

    Keys are equal when `Ball.same_representation` holds: the kind, the
    midpoint's representation and the radius all match. They order by
    `Ball.representation_cmp`.
    """

    var _value: Ball

    def __init__(out self, value: Ball):
        """Wrap a Ball.

        Args:
            value: The ball.
        """
        self._value = value

    def value(self) -> Ball:
        """The wrapped Ball.

        Returns:
            The ball.
        """
        return self._value

    def __eq__(self, other: Self) -> Bool:
        return self._value.same_representation(other._value)

    def __ne__(self, other: Self) -> Bool:
        return not self._value.same_representation(other._value)

    def __lt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) < 0

    def __le__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) <= 0

    def __gt__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) > 0

    def __ge__(self, other: Self) -> Bool:
        return self._value.representation_cmp(other._value) >= 0

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the kind, the midpoint's representation and the radius.

        Args:
            hasher: The hasher.
        """
        self._value._kind.__hash__(hasher)
        _float_stable_hash(self._value._midpoint).__hash__(hasher)
        self._value._radius.infinite.__hash__(hasher)
        self._value._radius.mantissa.__hash__(hasher)
        (self._value._radius.exponent if self._value._radius.mantissa else 0).__hash__(hasher)

    def write_to(self, mut writer: Some[Writer]):
        """Write `BallKey(` and the ball, as `print` does.

        Args:
            writer: The destination.
        """
        writer.write("BallKey(", self._value, ")")
