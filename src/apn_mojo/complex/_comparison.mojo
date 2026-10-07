"""Real-left equality without importing the public Complex value."""

from ..float._arithmetic import _FloatArgument


trait _ComplexComparison(Copyable):
    def _equals_real(self, rhs: _FloatArgument) raises -> Bool:
        ...
