"""Conversion limits shared by every family's text and JSON conversion.

Text and JSON conversion is exact unless a context asks for rounding. Parsers,
`to_string`, `to_json` and `from_json` accept an optional `limits=` that bounds
one call.
"""

from .conversion import ConversionLimits
from .keys import FloatKey, ComplexKey, BallKey
