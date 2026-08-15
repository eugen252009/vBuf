// Research-only standalone procedural weight generation microbenchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <x86intrin.h>

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}

static inline int popcount8(uint32_t x) { return __builtin_popcount(x & 255U); }

int main() {
    const int lengths[] = {8, 16, 32, 64, 128, 256};
    const char * families[] = {"G0_HASH_SIGN", "G1_HASH_AFFINE", "G2_CENTER_DENSE", "G3_RECURRENCE", "G4_TINY_BASIS"};
    std::printf("generator,length,generated_weights_per_second,cycles_per_weight,generated_GB_per_s,integer_ops_per_weight,fp_ops_per_weight,lut_accesses_per_weight,branches_per_weight,state_words,reconstruction_class,vectorization\n");
    volatile float sink = 0.0f;
    for (int family = 0; family < 5; ++family) for (int length : lengths) {
        const size_t count = (size_t(length) * 131072 + 63) / 64 * 64;
        std::vector<float> output(count);
        uint64_t best_cycles = UINT64_MAX;
        double best_seconds = 1e9;
        for (int repeat = 0; repeat < 5; ++repeat) {
            const auto begin_time = std::chrono::steady_clock::now();
            const uint64_t begin_cycles = __rdtsc();
            for (size_t segment = 0; segment < count / length; ++segment) {
                const uint32_t seed = uint32_t(segment * 17 + 11);
                const float scale = 0.0125f + float(seed & 7) * 0.0001f;
                const float bias = float(int(seed & 3) - 1) * 0.00005f;
                uint32_t state = mix32(seed + 0xa341316cU) | 1U;
                for (int i = 0; i < length; ++i) {
                    float raw = 0.0f;
                    if (family <= 2) {
                        const uint32_t h = mix32(seed * 0x9e3779b9U + uint32_t(i) * 0x85ebca6bU + 0xd1b54a35U);
                        if (family == 0) raw = (h >> 31) ? 1.0f : -1.0f;
                        else if (family == 1) raw = float(int16_t(h >> 16)) / 32768.0f;
                        else raw = float(popcount8(h) - 4) * 0.5f;
                    } else if (family == 3) {
                        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
                        raw = float(int16_t(state >> 16)) / 32768.0f;
                    } else {
                        int sum = 0;
                        for (int basis = 0; basis < 8; ++basis) {
                            const uint32_t h = mix32(uint32_t(0xb0 + basis) * 0x9e3779b9U + uint32_t(i) * 0x85ebca6bU + 0xd1b54a35U);
                            const int value = (h >> 31) ? 1 : -1;
                            sum += ((seed >> basis) & 1U) ? value : -value;
                        }
                        raw = float(sum) * 0.3535533905932738f;
                    }
                    output[segment * length + i] = raw * scale + (family == 1 ? bias : 0.0f);
                }
            }
            const uint64_t cycles = __rdtsc() - begin_cycles;
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin_time).count();
            best_cycles = std::min(best_cycles, cycles); best_seconds = std::min(best_seconds, seconds);
        }
        for (size_t i = 0; i < output.size(); i += 4096) sink += output[i];
        const double rate = count / best_seconds;
        const double iops[] = {9, 9, 13, 6, 79};
        const double fops[] = {1, 2, 1, 2, 9};
        // G4 conservatively regenerates its accounted 2 KiB basis instead of loading it.
        const double luts[] = {0, 0, 0, 0, 0};
        const double state[] = {0, 0, 0, 1, 0};
        std::printf("%s,%d,%.9f,%.9f,%.9f,%.1f,%.1f,%.1f,0,%.0f,SMALL_TILE_GENERATABLE,%s\n",
            families[family], length, rate, double(best_cycles) / count, rate * 4.0 / 1e9,
            iops[family], fops[family], luts[family], state[family], family == 3 ? "loop-carried state; SIMD across segments" : "SIMD-friendly across weights/segments");
    }
    return sink == 123456.0f ? 1 : 0;
}
