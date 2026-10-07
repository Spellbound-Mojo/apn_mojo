"""Integer arithmetic and the three division conventions."""

from apn_mojo import Integer, abs, div_rem_floor, div_rem_trunc, div_rem_euclid


def main() raises:
    var wide = Integer(18446744073709551616)
    print("wide literal:", wide)
    var q, r = div_rem_floor(-7, 3)
    print("floor:", q, r)
    q, r = div_rem_trunc(-7, 3)
    print("truncating:", q, r)
    q, r = div_rem_euclid(-7, -3)
    print("Euclidean:", q, r)
    print("operators:", Integer(-7) // 3, Integer(-7) % 3)
    print("signed bits:", Integer(-7) >> 1, ~Integer(7))
    print("queries:", wide.sign(), wide.magnitude_bit_length())
    print("absolute value:", abs(Integer(-42)))
    print("native result:", Integer(255).to_native_exact[DType.uint8]())
    var labels = Dict[Integer, String]()
    labels[wide] = "large counter"
    print("key lookup:", labels[Integer("18446744073709551616")])
