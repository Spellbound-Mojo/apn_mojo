"""Cancellation, exact totals and certified decimal digits."""

from apn_mojo import ArithmeticContext, BallContext, Batch, Float, FloatFormat, ball, sum


def main() raises:
    var native_format = ArithmeticContext(format=FloatFormat.binary64())
    var large = Float("1e20", context=native_format)
    var one = Float(1, context=native_format)
    print("separate additions:", (large + one) - large)
    print("exact total rounded once:", sum(Batch[Float]([large, one, -large])))

    var target = ArithmeticContext(format=FloatFormat(192))
    var precision = 192
    while precision <= 1536:
        # Recompute from the exact input at each working precision.
        var enclosure = ball.sqrt(2, context=BallContext(precision))
        var rounded = ball.to_float_if_certain(enclosure, context=target)
        var lower_text = enclosure.lower().to_string(digits=50)
        var upper_text = enclosure.upper().to_string(digits=50)
        if rounded and lower_text == upper_text:
            print("certified binary precision:", rounded.value().precision())
            print("sqrt(2), 50 significant digits:", lower_text)
            return
        precision *= 2
    raise Error("The precision budget did not certify the requested result.")
