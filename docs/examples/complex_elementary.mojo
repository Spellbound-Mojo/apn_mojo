"""Complex exponentials and the principal logarithm."""

from apn_mojo import ArithmeticContext, Complex, FloatFormat, exp, log


def main() raises:
    var context = ArithmeticContext(format=FloatFormat(128))
    var exponential = exp(Complex(1, 1), context=context)
    print("exp(1 + i), real and imaginary:",
          exponential.real().to_string(digits=12), exponential.imag().to_string(digits=12))
    var logarithm = log(Complex(-2), context=context)
    print("principal log(-2), real and imaginary:",
          logarithm.real().to_string(digits=12), logarithm.imag().to_string(digits=12))
