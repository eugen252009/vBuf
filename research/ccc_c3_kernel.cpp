#include "ccc_c3_kernel.h"
#include <immintrin.h>
#include <x86intrin.h>
#include <cstring>
#include <cmath>

// Pack array of uint8 codes (each 0..7) into 24-bit bitstream (8 codes in 3 bytes)
void c3_pack_codes(const uint8_t * codes, uint8_t * packed_bytes, size_t n_codes) {
    size_t n_groups = n_codes / 8;
    for (size_t g = 0; g < n_groups; g++) {
        const uint8_t * c = codes + g * 8;
        uint32_t u = (uint32_t)(c[0] & 7)       |
                    ((uint32_t)(c[1] & 7) << 3)  |
                    ((uint32_t)(c[2] & 7) << 6)  |
                    ((uint32_t)(c[3] & 7) << 9)  |
                    ((uint32_t)(c[4] & 7) << 12) |
                    ((uint32_t)(c[5] & 7) << 15) |
                    ((uint32_t)(c[6] & 7) << 18) |
                    ((uint32_t)(c[7] & 7) << 21);
                    
        uint8_t * p = packed_bytes + g * 3;
        p[0] = (uint8_t)(u & 0xFF);
        p[1] = (uint8_t)((u >> 8) & 0xFF);
        p[2] = (uint8_t)((u >> 16) & 0xFF);
    }
}

// Unpack 24-bit bitstream into array of uint8 codes (each 0..7)
void c3_unpack_codes(const uint8_t * packed_bytes, uint8_t * codes, size_t n_codes) {
    size_t n_groups = n_codes / 8;
    for (size_t g = 0; g < n_groups; g++) {
        const uint8_t * p = packed_bytes + g * 3;
        uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        uint8_t * c = codes + g * 8;
        c[0] = (uint8_t)(u & 7);
        c[1] = (uint8_t)((u >> 3) & 7);
        c[2] = (uint8_t)((u >> 6) & 7);
        c[3] = (uint8_t)((u >> 9) & 7);
        c[4] = (uint8_t)((u >> 12) & 7);
        c[5] = (uint8_t)((u >> 15) & 7);
        c[6] = (uint8_t)((u >> 18) & 7);
        c[7] = (uint8_t)((u >> 21) & 7);
    }
}

// Standalone unpack benchmark primitive (PACK_ONLY)
void c3_unpack_only(const uint8_t * packed_bytes, uint8_t * unpacked_bytes, size_t n_codes) {
    c3_unpack_codes(packed_bytes, unpacked_bytes, n_codes);
}

// Variant A: Dense FP32 reference (Oracle path)
void c3_matvec_dense_ref(
    const float * W_dense,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    for (size_t r = 0; r < rows; r++) {
        const float * row_ptr = W_dense + r * cols;
        double sum = 0.0;
        for (size_t c = 0; c < cols; c++) {
            sum += (double)row_ptr[c] * (double)x[c];
        }
        y[r] = (float)sum;
    }
}

// Variant B: Unpacked byte C3 (1 code / byte, 8 bpw - isolates lookup+MAC without 3-bit unpack)
void c3_matvec_byte_unpacked(
    const uint8_t * byte_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    for (size_t r = 0; r < rows; r++) {
        const uint8_t * row_codes = byte_codes + r * cols;
        float sum = 0.0f;
        for (size_t c = 0; c < cols; c++) {
            uint8_t code = row_codes[c] & 7;
            sum += levels[code] * x[c];
        }
        y[r] = sum;
    }
}

// Variant C.1: Scalar Direct Packed C3
void c3_matvec_scalar(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = (cols * 3) / 8;
    size_t n_groups = cols / 8;
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t * row_packed = packed_codes + r * bytes_per_row;
        float sum = 0.0f;
        
        for (size_t g = 0; g < n_groups; g++) {
            const uint8_t * p = row_packed + g * 3;
            uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            const float * x_ptr = x + g * 8;
            
            sum += levels[u & 7]        * x_ptr[0];
            sum += levels[(u >> 3) & 7] * x_ptr[1];
            sum += levels[(u >> 6) & 7] * x_ptr[2];
            sum += levels[(u >> 9) & 7] * x_ptr[3];
            sum += levels[(u >> 12) & 7] * x_ptr[4];
            sum += levels[(u >> 15) & 7] * x_ptr[5];
            sum += levels[(u >> 18) & 7] * x_ptr[6];
            sum += levels[(u >> 21) & 7] * x_ptr[7];
        }
        y[r] = sum;
    }
}

// Variant C.2: BMI2 Bit-Extract Direct Packed C3
void c3_matvec_bmi2(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = (cols * 3) / 8;
    size_t n_groups = cols / 8;
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t * row_packed = packed_codes + r * bytes_per_row;
        float sum = 0.0f;
        
        for (size_t g = 0; g < n_groups; g += 2) {
            // Read 6 bytes for 16 codes
            const uint8_t * p = row_packed + g * 3;
            uint64_t u64 = 0;
            std::memcpy(&u64, p, 6);
            
            const float * x_ptr = x + g * 8;
            
            sum += levels[(u64) & 7]         * x_ptr[0];
            sum += levels[(u64 >> 3) & 7]    * x_ptr[1];
            sum += levels[(u64 >> 6) & 7]    * x_ptr[2];
            sum += levels[(u64 >> 9) & 7]    * x_ptr[3];
            sum += levels[(u64 >> 12) & 7]   * x_ptr[4];
            sum += levels[(u64 >> 15) & 7]   * x_ptr[5];
            sum += levels[(u64 >> 18) & 7]   * x_ptr[6];
            sum += levels[(u64 >> 21) & 7]   * x_ptr[7];
            
            sum += levels[(u64 >> 24) & 7]   * x_ptr[8];
            sum += levels[(u64 >> 27) & 7]   * x_ptr[9];
            sum += levels[(u64 >> 30) & 7]   * x_ptr[10];
            sum += levels[(u64 >> 33) & 7]   * x_ptr[11];
            sum += levels[(u64 >> 36) & 7]   * x_ptr[12];
            sum += levels[(u64 >> 39) & 7]   * x_ptr[13];
            sum += levels[(u64 >> 42) & 7]   * x_ptr[14];
            sum += levels[(u64 >> 45) & 7]   * x_ptr[15];
        }
        y[r] = sum;
    }
}

// Variant C.3: AVX2 Vectorized Direct Packed C3
void c3_matvec_avx2(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = (cols * 3) / 8;
    size_t n_groups = cols / 8;
    
    // Broadcast levels array to 8-element alignment
    __m256 v_levels = _mm256_loadu_ps(levels);
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t * row_packed = packed_codes + r * bytes_per_row;
        __m256 acc = _mm256_setzero_ps();
        
        for (size_t g = 0; g < n_groups; g++) {
            const uint8_t * p = row_packed + g * 3;
            uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
            
            // Extract 8 integer indices (3 bits each) into 256-bit SIMD register
            __m256i idx = _mm256_setr_epi32(
                u & 7,
                (u >> 3) & 7,
                (u >> 6) & 7,
                (u >> 9) & 7,
                (u >> 12) & 7,
                (u >> 15) & 7,
                (u >> 18) & 7,
                (u >> 21) & 7
            );
            
            // AVX2 Gather float levels using integer indices
            __m256 w_vec = _mm256_i32gather_ps(levels, idx, 4);
            __m256 x_vec = _mm256_loadu_ps(x + g * 8);
            
            // FMA3 vector multiply-accumulate
            acc = _mm256_fmadd_ps(w_vec, x_vec, acc);
        }
        
        // Horizontal sum of 256-bit AVX2 register
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        y[r] = _mm_cvtss_f32(sum128);
    }
}

// Variant C.4: AVX2 16-wide Unrolled Direct Packed C3
void c3_matvec_avx2_unrolled(
    const uint8_t * packed_codes,
    const float * levels,
    const float * x,
    float * y,
    size_t rows,
    size_t cols
) {
    size_t bytes_per_row = (cols * 3) / 8;
    size_t n_groups = cols / 8;
    
    for (size_t r = 0; r < rows; r++) {
        const uint8_t * row_packed = packed_codes + r * bytes_per_row;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        
        for (size_t g = 0; g < n_groups; g += 2) {
            const uint8_t * p = row_packed + g * 3;
            uint64_t u64 = 0;
            std::memcpy(&u64, p, 6);
            
            __m256i idx0 = _mm256_setr_epi32(
                (int)(u64 & 7),
                (int)((u64 >> 3) & 7),
                (int)((u64 >> 6) & 7),
                (int)((u64 >> 9) & 7),
                (int)((u64 >> 12) & 7),
                (int)((u64 >> 15) & 7),
                (int)((u64 >> 18) & 7),
                (int)((u64 >> 21) & 7)
            );
            
            __m256i idx1 = _mm256_setr_epi32(
                (int)((u64 >> 24) & 7),
                (int)((u64 >> 27) & 7),
                (int)((u64 >> 30) & 7),
                (int)((u64 >> 33) & 7),
                (int)((u64 >> 36) & 7),
                (int)((u64 >> 39) & 7),
                (int)((u64 >> 42) & 7),
                (int)((u64 >> 45) & 7)
            );
            
            __m256 w0 = _mm256_i32gather_ps(levels, idx0, 4);
            __m256 w1 = _mm256_i32gather_ps(levels, idx1, 4);
            
            __m256 x0 = _mm256_loadu_ps(x + g * 8);
            __m256 x1 = _mm256_loadu_ps(x + (g + 1) * 8);
            
            acc0 = _mm256_fmadd_ps(w0, x0, acc0);
            acc1 = _mm256_fmadd_ps(w1, x1, acc1);
        }
        
        __m256 acc = _mm256_add_ps(acc0, acc1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        y[r] = _mm_cvtss_f32(sum128);
    }
}
