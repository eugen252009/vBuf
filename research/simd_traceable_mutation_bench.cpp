// Research-only scalar/AVX2 benchmark for traceable weight mutations.
#include <immintrin.h>
#include <x86intrin.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum Family { M0_LCG_JUMP, M1_LANE_AFFINE, M2_AFFINE_XOR, M3_FLOAT_AFFINE };
static const char * NAMES[] = {"M0_LCG_JUMP", "M1_LANE_AFFINE", "M2_AFFINE_XOR", "M3_FLOAT_AFFINE"};
static constexpr uint32_t LCG_A = 0x9e3779b1U;
static constexpr uint32_t LCG_B = 0x7f4a7c15U;
static constexpr uint32_t LANE_C = 0x6d2b79f5U;
static constexpr uint32_t POS_C = 0x85ebca77U;
static constexpr uint32_t TENSOR_C = 0x243f6a89U;
static constexpr uint32_t A[4] = {0x9e3779b1U, 0x85ebca77U, 0xc2b2ae3dU, 0x27d4eb2fU};
static constexpr uint32_t B[4] = {0x7f4a7c15U, 0x165667b1U, 0xd3a2646dU, 0xfd7046c5U};
static constexpr uint32_t C[4] = {0x94d049bbU, 0x369dea0fU, 0x7feb352dU, 0x846ca68bU};
static constexpr int K[4] = {7, 11, 9, 13};
static constexpr float FA[4] = {0.75f, -0.875f, 1.125f, 0.625f};
static constexpr float FB[4] = {0.0625f, -0.03125f, 0.046875f, -0.078125f};

static inline uint32_t position_key(uint32_t segment, bool position_aware) {
    return position_aware ? segment * POS_C + TENSOR_C : 0U;
}

static inline std::pair<uint32_t, uint32_t> affine_jump(uint32_t a, uint32_t b, uint32_t steps) {
    uint32_t acc_a = 1, acc_b = 0;
    while (steps) {
        if (steps & 1U) { acc_b = acc_b * a + b; acc_a *= a; }
        b = b * (a + 1U); a *= a; steps >>= 1U;
    }
    return {acc_a, acc_b};
}

struct JumpTables {
    alignas(32) uint32_t a[5][64];
    alignas(32) uint32_t b[5][64];
    JumpTables() {
        for (int rounds = 0; rounds <= 4; ++rounds) for (int lane = 0; lane < 64; ++lane) {
            const auto jump = affine_jump(LCG_A, LCG_B, uint32_t(lane + 1 + rounds));
            a[rounds][lane] = jump.first; b[rounds][lane] = jump.second;
        }
    }
};

static const JumpTables & jump_tables() { static const JumpTables tables; return tables; }

static inline float map_integer(uint32_t state) {
    return float(int32_t(state) >> 16) * (1.0f / 32768.0f);
}

static inline uint32_t scalar_integer_state(Family family, int rounds, uint32_t seed, uint32_t segment, int lane, bool position_aware) {
    const uint32_t pos = position_key(segment, position_aware);
    if (family == M0_LCG_JUMP) {
        const auto & tables = jump_tables();
        return tables.a[rounds][lane] * (seed + pos) + tables.b[rounds][lane];
    }
    uint32_t state = seed + pos + uint32_t(lane) * LANE_C;
    for (int round = 0; round < rounds; ++round) {
        state = state * A[round] + B[round] + uint32_t(lane) * (B[round] | 1U);
        if (family == M2_AFFINE_XOR) {
            state ^= state >> K[round];
            state *= C[round];
        }
    }
    return state;
}

static inline float scalar_float_state(int rounds, uint32_t seed, uint32_t segment, int lane, bool position_aware) {
    const float root = float(int32_t(seed)) * (1.0f / 2147483648.0f);
    const float lane_term = float((lane * 5) & 15) * (1.0f / 8.0f) - 0.9375f;
    const float pos_term = position_aware ? float(int(segment & 15U) - 7) * (1.0f / 64.0f) : 0.0f;
    float state = root + lane_term + pos_term;
    for (int round = 0; round < rounds; ++round) state = state * FA[round] + FB[round] + lane_term * 0.03125f;
    return state;
}

static void scalar_segment(Family family, int rounds, uint32_t seed, uint32_t segment, int length, float scale, float * output, bool position_aware = true) {
    for (int lane = 0; lane < length; ++lane) {
        const float raw = family == M3_FLOAT_AFFINE ? scalar_float_state(rounds, seed, segment, lane, position_aware)
                                                    : map_integer(scalar_integer_state(family, rounds, seed, segment, lane, position_aware));
        output[lane] = raw * scale;
    }
}

static inline __m256 avx_raw8(Family family, int rounds, uint32_t seed, uint32_t segment, int lane_base, bool position_aware) {
    const __m256i lanes = _mm256_setr_epi32(lane_base, lane_base + 1, lane_base + 2, lane_base + 3, lane_base + 4, lane_base + 5, lane_base + 6, lane_base + 7);
    if (family == M3_FLOAT_AFFINE) {
        const __m256 lane_f = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_mullo_epi32(lanes, _mm256_set1_epi32(5)), _mm256_set1_epi32(15)));
        const __m256 lane_term = _mm256_sub_ps(_mm256_mul_ps(lane_f, _mm256_set1_ps(1.0f / 8.0f)), _mm256_set1_ps(0.9375f));
        const float root = float(int32_t(seed)) * (1.0f / 2147483648.0f);
        const float pos = position_aware ? float(int(segment & 15U) - 7) * (1.0f / 64.0f) : 0.0f;
        __m256 state = _mm256_add_ps(_mm256_set1_ps(root + pos), lane_term);
        for (int round = 0; round < rounds; ++round) {
            state = _mm256_fmadd_ps(state, _mm256_set1_ps(FA[round]), _mm256_add_ps(_mm256_set1_ps(FB[round]), _mm256_mul_ps(lane_term, _mm256_set1_ps(0.03125f))));
        }
        return state;
    }
    __m256i state;
    const uint32_t pos = position_key(segment, position_aware);
    if (family == M0_LCG_JUMP) {
        const auto & tables = jump_tables();
        state = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_set1_epi32(int(seed + pos)), _mm256_loadu_si256((const __m256i *)(tables.a[rounds] + lane_base))), _mm256_loadu_si256((const __m256i *)(tables.b[rounds] + lane_base)));
    } else {
        state = _mm256_add_epi32(_mm256_set1_epi32(int(seed + pos)), _mm256_mullo_epi32(lanes, _mm256_set1_epi32(int(LANE_C))));
        for (int round = 0; round < rounds; ++round) {
            const __m256i add = _mm256_add_epi32(_mm256_set1_epi32(int(B[round])), _mm256_mullo_epi32(lanes, _mm256_set1_epi32(int(B[round] | 1U))));
            state = _mm256_add_epi32(_mm256_mullo_epi32(state, _mm256_set1_epi32(int(A[round]))), add);
            if (family == M2_AFFINE_XOR) {
                state = _mm256_xor_si256(state, _mm256_srli_epi32(state, K[round]));
                state = _mm256_mullo_epi32(state, _mm256_set1_epi32(int(C[round])));
            }
        }
    }
    return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srai_epi32(state, 16)), _mm256_set1_ps(1.0f / 32768.0f));
}

__attribute__((noinline)) static void generate_avx2_basic(Family family, int rounds, const uint32_t * seeds, const float * scales, size_t segments, int length, float * output) {
    for (size_t segment = 0; segment < segments; ++segment) for (int lane = 0; lane < length; lane += 8) {
        const __m256 raw = avx_raw8(family, rounds, seeds[segment], uint32_t(segment), lane, true);
        _mm256_storeu_ps(output + segment * length + lane, _mm256_mul_ps(raw, _mm256_set1_ps(scales[segment])));
    }
}

__attribute__((noinline)) static void generate_avx2_unrolled4(Family family, int rounds, const uint32_t * seeds, const float * scales, size_t segments, int length, float * output) {
    size_t segment = 0;
    for (; segment + 3 < segments; segment += 4) for (int lane = 0; lane < length; lane += 8) {
        const __m256 r0 = avx_raw8(family, rounds, seeds[segment], uint32_t(segment), lane, true);
        const __m256 r1 = avx_raw8(family, rounds, seeds[segment + 1], uint32_t(segment + 1), lane, true);
        const __m256 r2 = avx_raw8(family, rounds, seeds[segment + 2], uint32_t(segment + 2), lane, true);
        const __m256 r3 = avx_raw8(family, rounds, seeds[segment + 3], uint32_t(segment + 3), lane, true);
        _mm256_storeu_ps(output + segment * length + lane, _mm256_mul_ps(r0, _mm256_set1_ps(scales[segment])));
        _mm256_storeu_ps(output + (segment + 1) * length + lane, _mm256_mul_ps(r1, _mm256_set1_ps(scales[segment + 1])));
        _mm256_storeu_ps(output + (segment + 2) * length + lane, _mm256_mul_ps(r2, _mm256_set1_ps(scales[segment + 2])));
        _mm256_storeu_ps(output + (segment + 3) * length + lane, _mm256_mul_ps(r3, _mm256_set1_ps(scales[segment + 3])));
    }
    if (segment < segments) generate_avx2_basic(family, rounds, seeds + segment, scales + segment, segments - segment, length, output + segment * length);
}

static void generate_scalar(Family family, int rounds, const uint32_t * seeds, const float * scales, size_t segments, int length, float * output) {
    for (size_t segment = 0; segment < segments; ++segment) scalar_segment(family, rounds, seeds[segment], uint32_t(segment), length, scales[segment], output + segment * length);
}

struct Timing { double seconds; uint64_t cycles; };
template <class Function> static Timing best_time(Function function) {
    Timing best{1e9, UINT64_MAX};
    for (int repeat = 0; repeat < 5; ++repeat) {
        const auto wall = std::chrono::steady_clock::now(); const uint64_t ticks = __rdtsc();
        function();
        const uint64_t elapsed_ticks = __rdtsc() - ticks;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall).count();
        if (elapsed < best.seconds) best = {elapsed, elapsed_ticks};
    }
    return best;
}

static bool self_test() {
    alignas(32) float scalar[64], avx[64];
    const uint32_t seed = 0xd00dfeedU; const float scale = 0.03125f;
    for (int family = 0; family < 4; ++family) for (int rounds = 0; rounds <= 4; ++rounds) for (int length : {8, 16, 32, 64}) {
        scalar_segment(Family(family), rounds, seed, 37, length, scale, scalar);
        for (int lane = 0; lane < length; lane += 8) _mm256_store_ps(avx + lane, _mm256_mul_ps(avx_raw8(Family(family), rounds, seed, 37, lane, true), _mm256_set1_ps(scale)));
        for (int index = 0; index < length; ++index) {
            const float tolerance = family == M3_FLOAT_AFFINE ? 2e-7f : 0.0f;
            if (std::fabs(scalar[index] - avx[index]) > tolerance) {
                std::fprintf(stderr, "equivalence failure family=%d rounds=%d length=%d lane=%d %.9g %.9g\n", family, rounds, length, index, scalar[index], avx[index]);
                return false;
            }
        }
    }
    return true;
}

static int throughput() {
    std::puts("family,rounds,length,variant,states_in_flight,cycles_per_weight,gweights_per_second,speedup_vs_scalar,materialization,correctness");
    volatile float sink = 0;
    for (int family = 0; family < 4; ++family) for (int rounds = 0; rounds <= 4; ++rounds) for (int length : {8, 16, 32, 64}) {
        const size_t count = 4U << 20; const size_t segments = count / length;
        std::vector<uint32_t> seeds(segments); std::vector<float> scales(segments); std::vector<float> output(count);
        for (size_t i = 0; i < segments; ++i) { seeds[i] = uint32_t(i * 0x9e3779b1U + 17U); scales[i] = 0.01f + float(i & 7U) * 0.00025f; }
        const Timing scalar = best_time([&]{ generate_scalar(Family(family), rounds, seeds.data(), scales.data(), segments, length, output.data()); });
        const Timing basic = best_time([&]{ generate_avx2_basic(Family(family), rounds, seeds.data(), scales.data(), segments, length, output.data()); });
        const Timing unrolled = best_time([&]{ generate_avx2_unrolled4(Family(family), rounds, seeds.data(), scales.data(), segments, length, output.data()); });
        for (size_t i = 0; i < output.size(); i += 4096) sink += output[i];
        const Timing timings[] = {scalar, basic, unrolled}; const char * variants[] = {"scalar", "avx2_basic", "avx2_unrolled4"}; const int states[] = {1, 1, 4};
        for (int variant = 0; variant < 3; ++variant) {
            const double rate = count / timings[variant].seconds;
            std::printf("%s,%d,%d,%s,%d,%.9f,%.9f,%.6f,generate_to_buffer,PASS\n", NAMES[family], rounds, length, variants[variant], states[variant], double(timings[variant].cycles) / count, rate / 1e9, scalar.seconds / timings[variant].seconds);
        }
    }
    return sink == 123456.0f ? 1 : 0;
}

static inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value), high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high); sum = _mm_hadd_ps(sum, sum); sum = _mm_hadd_ps(sum, sum); return _mm_cvtss_f32(sum);
}

__attribute__((noinline)) static float direct_generate_dot(Family family, int rounds, const uint32_t * seeds, const float * scales, size_t segments, int length, const float * activation) {
    __m256 accumulator = _mm256_setzero_ps();
    for (size_t segment = 0; segment < segments; ++segment) for (int lane = 0; lane < length; lane += 8) {
        const __m256 weight = _mm256_mul_ps(avx_raw8(family, rounds, seeds[segment], uint32_t(segment), lane, true), _mm256_set1_ps(scales[segment]));
        accumulator = _mm256_fmadd_ps(weight, _mm256_loadu_ps(activation + segment * length + lane), accumulator);
    }
    return horizontal_sum(accumulator);
}

static int dot_benchmark() {
    std::puts("family,rounds,length,path,cycles_per_weight,gweights_per_second,generated_weight_memory_writes,relative_difference,correctness");
    volatile float sink = 0;
    const int length = 32; const size_t count = 5120; const size_t segments = count / length; const int repetitions = 32768;
    std::vector<uint32_t> seeds(segments); std::vector<float> scales(segments), activation(count), weights(count);
    for (size_t i = 0; i < segments; ++i) { seeds[i] = uint32_t(i * 0x9e3779b1U + 29U); scales[i] = 0.0125f; }
    for (size_t i = 0; i < count; ++i) activation[i] = std::sin(float(i) * 0.01f);
    for (int family = 0; family < 4; ++family) for (int rounds = 0; rounds <= 4; ++rounds) {
        generate_avx2_basic(Family(family), rounds, seeds.data(), scales.data(), segments, length, weights.data());
        __m256 reference_acc = _mm256_setzero_ps(); for (size_t i = 0; i < count; i += 8) reference_acc = _mm256_fmadd_ps(_mm256_loadu_ps(weights.data() + i), _mm256_loadu_ps(activation.data() + i), reference_acc);
        const float reference = horizontal_sum(reference_acc); const float direct = direct_generate_dot(Family(family), rounds, seeds.data(), scales.data(), segments, length, activation.data());
        const double difference = std::fabs(reference - direct) / std::max(std::fabs(reference), 1e-20f);
        const Timing buffered = best_time([&]{ for (int r = 0; r < repetitions; ++r) { generate_avx2_basic(Family(family), rounds, seeds.data(), scales.data(), segments, length, weights.data()); __m256 acc = _mm256_setzero_ps(); for (size_t i = 0; i < count; i += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(weights.data()+i), _mm256_loadu_ps(activation.data()+i), acc); sink += horizontal_sum(acc); }});
        const Timing fused = best_time([&]{ for (int r = 0; r < repetitions; ++r) sink += direct_generate_dot(Family(family), rounds, seeds.data(), scales.data(), segments, length, activation.data()); });
        const double total = double(count) * repetitions;
        std::printf("%s,%d,%d,GENERATE_TO_BUFFER_PLUS_DOT,%.9f,%.9f,%zu,%.9g,%s\n", NAMES[family], rounds, length, double(buffered.cycles)/total, total/buffered.seconds/1e9, count, difference, difference < 1e-5 ? "PASS" : "FAIL");
        std::printf("%s,%d,%d,DIRECT_GENERATE_AND_DOT,%.9f,%.9f,0,%.9g,%s\n", NAMES[family], rounds, length, double(fused.cycles)/total, total/fused.seconds/1e9, difference, difference < 1e-5 ? "PASS" : "FAIL");
    }
    return sink == 123456.0f ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (!self_test()) return 3;
    if (argc == 2 && std::strcmp(argv[1], "self-test") == 0) { std::puts("scalar_avx2_equivalence=PASS"); return 0; }
    if (argc == 2 && std::strcmp(argv[1], "throughput") == 0) return throughput();
    if (argc == 2 && std::strcmp(argv[1], "dot") == 0) return dot_benchmark();
    return 2;
}
