"""Special functions for probabilities and exact fractional inputs."""

from apn_mojo import ArithmeticContext, FloatFormat, Rational, beta, gamma, gammaincc, ndtr, ndtri


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(128))
    print("Gamma(5):", gamma(5, context=context))
    print("B(2, 3):", beta(2, 3, context=context).to_string(digits=12))
    print("normal CDF at zero:", ndtr(0, context=context))
    print("normal 97.5% quantile:", ndtri(Rational(975, 1000), context=context).to_string(digits=12))
    print("gamma upper tail, shape 3 at 2:", gammaincc(3, 2, context=context).to_string(digits=12))
