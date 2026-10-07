"""Constants, elementary functions and small increments."""

from apn_mojo import ArithmeticContext, FloatFormat, Rational, exp, log, log1p, pi, sin_cos


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(128))
    print("pi:", pi(context=context).to_string(digits=20))
    print("exp(1/10):", exp(Rational(1, 10), context=context).to_string(digits=12))
    print("log(2):", log(2, context=context).to_string(digits=12))
    var sine, cosine = sin_cos(Rational(1, 2), context=context)
    print("sin and cos of 1/2 radian:", sine.to_string(digits=12), cosine.to_string(digits=12))
    var increment = Rational("1e-20")
    print("log1p of a small increment:", log1p(increment, context=context).to_string(digits=12))
