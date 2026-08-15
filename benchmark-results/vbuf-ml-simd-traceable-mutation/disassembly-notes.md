# SIMD Traceable Mutation Disassembly Notes

Binary compiled with `-O3 -march=znver3 -mavx2 -mfma`.

- AVX2 integer multiply (`vpmulld`) occurrences: 176
- Vector integer add (`vpaddd`) occurrences: 110
- Vector xor (`vpxor`) occurrences: 32
- Vector right shift (`vpsrld`) occurrences: 32
- Integer-to-FP32 conversion (`vcvtdq2ps`) occurrences: 24
- FMA mnemonic-prefix occurrences: 67
- YMM registers referenced: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
- Conservative YMM stack-spill patterns: 20

`generate_avx2_basic`, `generate_avx2_unrolled4`, and `direct_generate_dot` are noinline hot-loop symbols. The instruction inventory confirms AVX2/FMA execution rather than scalar fallback. Basic carries one state; unrolled carries four independent states to expose instruction-level parallelism. Stack-pattern counting is conservative and is not a dynamic spill count.

The whole-binary scan found 20 conservative YMM stack-traffic patterns. The unrolled implementation is therefore not claimed spill-free. Dynamic instructions/weight were not available; measured cycles/weight and the static instruction inventory are the bounded controls.
