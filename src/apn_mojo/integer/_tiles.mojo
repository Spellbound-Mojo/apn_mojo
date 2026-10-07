"""SIMD lane aliases shared by mask bits and Float metadata records."""


comptime _WIDTH = 8
comptime _Digits = SIMD[DType.uint32, _WIDTH]
comptime _Mask = SIMD[DType.bool, _WIDTH]
comptime _Lengths = SIMD[DType.int64, _WIDTH]














