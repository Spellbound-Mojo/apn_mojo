"""Numeric-family-independent ordered folds and fixed padded binary trees."""


trait _ReductionInputs(Movable):
    def length(self) -> Int:
        ...


trait _ReductionOperation(Movable):
    comptime Inputs: _ReductionInputs
    comptime Value: Movable & Deinitable
    comptime Result: Movable & Deinitable

    def validate(self, inputs: Self.Inputs) raises:
        ...

    def prepare(mut self, inputs: Self.Inputs) raises:
        ...

    def identity(self) raises -> Self.Value:
        ...

    def element(mut self, inputs: Self.Inputs, index: Int) raises -> Self.Value:
        ...

    def combine(
        mut self, left: Self.Value, right: Self.Value, start: Int, end: Int
    ) raises -> Self.Value:
        ...

    def singleton(mut self, value: Self.Value) raises -> Self.Value:
        ...

    def finish(mut self, var value: Self.Value) raises -> Self.Result:
        ...


def _reduction_tree[
    O: _ReductionOperation
](
    mut operation: O,
    inputs: O.Inputs,
    start: Int128,
    width: Int128,
) raises -> O.Value:
    # Whole padding subtrees are identities. Adapters must make identity plus
    # identity observationally identical to identity, including status/traps.
    if start >= Int128(inputs.length()):
        return operation.identity()
    if width == 1:
        return operation.element(inputs, Int(start))
    var half = width // 2
    var left = _reduction_tree(operation, inputs, start, half)
    var right = _reduction_tree(operation, inputs, start + half, half)
    return operation.combine(
        left,
        right,
        Int(start),
        Int(min(start + width, Int128(inputs.length()))),
    )


def _execute_reduction[
    tree: Bool, O: _ReductionOperation
](mut operation: O, inputs: O.Inputs,) raises -> O.Result:
    operation.validate(inputs)
    var length = inputs.length()
    if length < 0:
        raise Error(
            "Cannot reduce a negative sequence length; supply a valid input"
            " descriptor. The destination is unchanged."
        )
    operation.prepare(inputs)
    if not length:
        return operation.finish(operation.identity())
    comptime if tree:
        if length == 1:
            var value = operation.element(inputs, 0)
            return operation.finish(operation.singleton(value))
        # Native sequence lengths fit Int; their padded width fits Int128.
        var width = Int128(1)
        while width < Int128(length):
            width *= 2
        var result = _reduction_tree(operation, inputs, 0, width)
        return operation.finish(result^)
    else:
        var result = operation.identity()
        for index in range(length):
            var value = operation.element(inputs, index)
            result = operation.combine(result, value, index, index + 1)
        return operation.finish(result^)
