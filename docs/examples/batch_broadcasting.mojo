"""The vector rules for operators, mapped functions and lift."""

from apn_mojo import ArithmeticContext, Batch, FloatFormat, Integer, batch, integer, lift, vmap


def main() raises:
    var values = Batch[Integer]([1, 2, 3])
    var singleton = Batch[Integer]([10])
    try:
        _ = values + singleton
    except:
        print("operator: vector lengths must match")
    print("batch.add:", batch.add(values, singleton))
    try:
        _ = vmap[integer.add]()(values, singleton)
    except:
        print("vmap: mapped extents must match")
    print("lift:", lift[integer.add]()(values, singleton))
    var context = ArithmeticContext(format=FloatFormat(24))
    print("batch.add with context:", batch.add(values, singleton, context=context))
    print("batch.atan2 shape:", batch.atan2(values, singleton, context=context).shape())
