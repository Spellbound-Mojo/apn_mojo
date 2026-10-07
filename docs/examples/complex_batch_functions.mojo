"""Complex batch magnitudes and square roots."""

from apn_mojo import Batch, Complex, Integer, Float, norm_sqr
from apn_mojo import complex, vmap


def main() raises:
    var samples = Batch[Complex](
        [Complex(3, 4), Complex(-4, Float.zero(negative=True))]
    )
    var saved = samples[::-1]
    print(
        vmap[norm_sqr]()(samples)[0] == Float(25),
        vmap[complex.abs]()(samples)[0] == Float(5),
    )
    var roots = vmap[complex.sqrt]()
    print(roots(samples)[0] == Complex(2, 1), roots(saved)[0] == Complex(0, -2))
    print(samples.conjugate()[0] == Complex(3, -4))
    print(samples.real()[0] == Float(3), samples.imag()[0] == Float(4))
    print(samples.is_finite().all(), samples.is_zero().any())
    var exponents = Batch[Integer].from_native([2, -1])
    var powers = vmap[complex.pow_int]()(samples, exponents)
    print(powers[0] == Complex(-7, 24))
    var cycle = Complex(0, 1) ** exponents
    print(cycle[0] == Complex(-1, 0), cycle[1] == Complex(0, -1))
    print(samples[0] == Complex(3, 4))
