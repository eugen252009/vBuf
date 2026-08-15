#ifndef CCC_C3_KERNEL_H
#define CCC_C3_KERNEL_H

#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Physical C3 packing: 8 weights in 3 bytes (24 bits)
// levels: array of 8 float reconstruction values
// x: input activation vector of size 'cols'
// y: output result vector of size 'rows'

// Variant A: Dense FP32 reference (Oracle path)
void c3_matvec_dense_ref(
    const float * W_dense,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Variant B: Unpacked byte C3 (1 code / byte, 8 bpw - isolates lookup+MAC without 3-bit unpack)
void c3_matvec_byte_unpacked(
    const uint8_t * byte_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Variant C.1: Scalar Direct Packed C3
void c3_matvec_scalar(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Variant C.2: BMI2 Bit-Extract Direct Packed C3
void c3_matvec_bmi2(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Variant C.3: AVX2 FMA Vectorized Direct Packed C3
void c3_matvec_avx2(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Variant C.4: AVX2 16-wide Unrolled Direct Packed C3
void c3_matvec_avx2_unrolled(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
);

// Standalone 3-bit Unpack Benchmark Primitive (PACK_ONLY)
void c3_unpack_only(
    const uint8_t * packed_codes,
    uint8_t * unpacked_bytes,
    size_t n_codes
);

// Packing helper: pack uint8 codes [0..7] into 24-bit little-endian bitstream
void c3_pack_codes(
    const uint8_t * codes,
    uint8_t * packed_bytes,
    size_t n_codes
);

// Unpacking helper: unpack 24-bit bitstream into uint8 codes [0..7]
void c3_unpack_codes(
    const uint8_t * packed_bytes,
    uint8_t * codes,
    size_t n_codes
);

#ifdef __cplusplus
}
#endif

#endif // CCC_C3_KERNEL_H
