"""APNH-64: a fixed 64-bit hash of a number's representation.

Unlike Mojo's `Hasher`, whose output may change between releases, these hashes
stay the same across processes, platforms and versions, so they can be stored.
A value is a tag for its family followed by 64-bit words; each word is mixed
with SplitMix64's finalizer, and the word count closes the hash. Changing the
algorithm or an encoding is a breaking change.
"""

from ..integer.value import Integer


def _mix(var z: UInt64) -> UInt64:
    z ^= z >> 30
    z *= 0xBF58476D1CE4E5B9
    z ^= z >> 27
    z *= 0x94D049BB133111EB
    return z ^ (z >> 31)


struct _StableHash:
    """The running state of one APNH-64 hash."""

    var state: UInt64
    var count: UInt64

    def __init__(out self, tag: UInt64):
        self.state = tag
        self.count = 0

    def word(mut self, value: UInt64):
        self.state = _mix(self.state ^ value) + 0x9E3779B97F4A7C15
        self.count += 1

    def signed(mut self, value: Int):
        """A signed field, as its two's-complement 64-bit word."""
        self.word(Int64(value).cast[DType.uint64]())

    def integer(mut self, x: Integer):
        """The sign, the number of 64-bit magnitude words, then the words from
        the least significant; zero is [0, 0]."""
        var words = (x.magnitude_bit_length() + 63) // 64
        self.word(UInt64(Int(x._negative())))
        self.word(UInt64(words))
        for i in range(words):
            self.word(UInt64(x._word(2 * i)) | (UInt64(x._word(2 * i + 1)) << 32))

    def float(
        mut self,
        precision: Int,
        emin: Int,
        emax: Int,
        kind: Int,
        negative: Bool,
        exponent: Int,
        significand: Integer,
    ):
        """The format, the class (0 zero, 1 finite, 2 infinity, 3 NaN) and the
        sign, then the exponent and significand of a finite nonzero value."""
        self.signed(precision)
        self.signed(emin)
        self.signed(emax)
        self.word(UInt64(kind))
        self.word(UInt64(Int(negative and kind != 3)))
        if kind == 1:
            self.signed(exponent)
            self.integer(significand)

    def finish(self) -> UInt64:
        return _mix(self.state ^ self.count)
