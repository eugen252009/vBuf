#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "softmax_compute_extent.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using vbuf_ggml::SoftmaxComputeExtent;

uint32_t cpu_softmax_granularity() {
    // Mirrors the compile-time branch order in the pinned GGML vec.cpp CPU
    // softmax implementation. Scalable-vector paths need runtime VL analysis.
    if (ggml_cpu_has_avx512()) return 16;
    if (ggml_cpu_has_avx2() && ggml_cpu_has_fma()) return 8;
    if (ggml_cpu_has_sse3() || ggml_cpu_has_neon()) return 4;
    if (ggml_cpu_has_sve() || ggml_cpu_has_riscv_v()) return 0;
    return 1;

}

struct Context {
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Context() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (backend) ggml_backend_free(backend);
        if (ctx) ggml_free(ctx);
    }
};

std::vector<float> run_pair(const std::vector<float> & scores, const SoftmaxComputeExtent & extent,
    std::vector<float> * padded_output, double * unpadded_us, double * padded_us) {
    const uint32_t compute = extent.compute;
    Context context;
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    context.ctx = ggml_init(params);
    context.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!context.ctx || !context.backend) throw std::runtime_error("GGML CPU init failed");
    ggml_backend_cpu_set_n_threads(context.backend, 1);

    const uint32_t logical = static_cast<uint32_t>(scores.size());
    ggml_tensor * unpadded_scores = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, logical, 1);
    ggml_tensor * unpadded_mask = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, logical, 1);
    ggml_tensor * unpadded = ggml_soft_max_ext(context.ctx, unpadded_scores, unpadded_mask, 1.0f, 0.0f);

    ggml_tensor * padded_scores = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, compute, 1);
    ggml_tensor * padded_mask = ggml_new_tensor_2d(context.ctx, GGML_TYPE_F32, compute, 1);
    ggml_tensor * padded = ggml_soft_max_ext(context.ctx, padded_scores, padded_mask, 1.0f, 0.0f);
    ggml_cgraph * unpadded_graph = ggml_new_graph(context.ctx);
    ggml_cgraph * padded_graph = ggml_new_graph(context.ctx);
    ggml_build_forward_expand(unpadded_graph, unpadded);
    ggml_build_forward_expand(padded_graph, padded);
    context.buffer = ggml_backend_alloc_ctx_tensors(context.ctx, context.backend);
    if (!context.buffer) throw std::runtime_error("GGML contract allocation failed");

    const std::vector<float> zero_mask(logical, 0.0f);
    ggml_backend_tensor_set(unpadded_scores, scores.data(), 0, scores.size() * sizeof(float));
    ggml_backend_tensor_set(unpadded_mask, zero_mask.data(), 0, zero_mask.size() * sizeof(float));
    std::vector<float> padded_input = extent.padded_scores(scores);
    std::vector<float> mask(compute, 0.0f);
    ggml_backend_tensor_set(padded_scores, padded_input.data(), 0, padded_input.size() * sizeof(float));
    ggml_backend_tensor_set(padded_mask, mask.data(), 0, mask.size() * sizeof(float));

    constexpr uint32_t repetitions = 100;
    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < repetitions; ++i) {
        if (ggml_backend_graph_compute(context.backend, unpadded_graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("GGML unpadded softmax graph failed");
        ggml_backend_synchronize(context.backend);
    }
    auto elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    *unpadded_us = elapsed / repetitions;
    start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < repetitions; ++i) {
        if (ggml_backend_graph_compute(context.backend, padded_graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("GGML padded softmax graph failed");
        ggml_backend_synchronize(context.backend);
    }
    elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
    *padded_us = elapsed / repetitions;
    std::vector<float> result(logical), padded_result(compute);
    ggml_backend_tensor_get(unpadded, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_tensor_get(padded, padded_result.data(), 0, padded_result.size() * sizeof(float));
    *padded_output = std::move(padded_result);
    return result;
}

std::vector<float> fp64_softmax(const std::vector<float> & scores) {
    const double maximum = *std::max_element(scores.begin(), scores.end());
    std::vector<double> values(scores.size());
    double sum = 0;
    for (size_t i = 0; i < scores.size(); ++i) {
        values[i] = std::exp(static_cast<double>(scores[i]) - maximum);
        sum += values[i];
    }
    std::vector<float> result(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) result[i] = static_cast<float>(values[i] / sum);
    return result;
}
}

int main() try {
    const uint32_t granularity = cpu_softmax_granularity();
    if (granularity == 0) {
        std::puts("softmax_extent_contract=SKIP scalable-vector kernel granularity is not qualified");
        return 77;
    }
    std::printf("softmax_extent_contract backend=GGML_CPU granularity=%u threads=1\n", granularity);
    for (uint32_t distribution = 0; distribution < 3; ++distribution)
    for (uint32_t logical = 1; logical <= 64; ++logical) {
        const auto extent = SoftmaxComputeExtent::make(logical, granularity);
        if (extent.compute < logical || extent.compute % granularity != 0 ||
            extent.padding() != extent.compute - logical)
            throw std::runtime_error("extent rounding invariant failed");
        std::vector<float> scores(logical);
        uint32_t random_state = 0x9e3779b9u ^ (logical * 0x85ebca6bu) ^ (distribution * 0xc2b2ae35u);
        for (uint32_t i = 0; i < logical; ++i) {
            random_state ^= random_state << 13;
            random_state ^= random_state >> 17;
            random_state ^= random_state << 5;
            if (distribution == 0)
                scores[i] = static_cast<float>(static_cast<int>((i * 17 + logical * 11) % 31) - 15) / 7.0f;
            else if (distribution == 1)
                scores[i] = (i % 2 == 0 ? 12.0f : -12.0f) + static_cast<float>(random_state % 101) / 100.0f;
            else
                scores[i] = i == (logical / 2) ? 16.0f : -static_cast<float>(random_state % 240) / 10.0f;
        }
        std::vector<float> padded;
        double unpadded_us = 0, padded_us = 0;
        const auto padded_score_input = extent.padded_scores(scores);
        if (!std::equal(scores.begin(), scores.end(), padded_score_input.begin()))
            throw std::runtime_error("logical raw scores changed during padding");
        for (uint32_t i = logical; i < extent.compute; ++i)
            if (!std::isinf(padded_score_input[i]) || padded_score_input[i] >= 0.0f)
                throw std::runtime_error("padded score was not negative infinity");
        const auto unpadded = run_pair(scores, extent, &padded, &unpadded_us, &padded_us);
        const auto oracle = fp64_softmax(scores);
        double probability_delta = 0, oracle_delta = 0, context_a = 0, context_b = 0;
        for (uint32_t i = 0; i < logical; ++i) {
            probability_delta = std::max(probability_delta,
                std::abs(static_cast<double>(unpadded[i]) - padded[i]));
            oracle_delta = std::max(oracle_delta,
                std::abs(static_cast<double>(padded[i]) - oracle[i]));
            const double v = static_cast<double>(i + 1) / logical;
            context_a += static_cast<double>(unpadded[i]) * v;
            context_b += static_cast<double>(padded[i]) * v;
        }
        const double context_delta = std::abs(context_a - context_b);
        uint64_t max_ulp = 0;
        for (uint32_t i = 0; i < logical; ++i) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &unpadded[i], sizeof(a));
            std::memcpy(&b, &padded[i], sizeof(b));
            const uint32_t oa = (a & 0x80000000u) ? ~a : (a | 0x80000000u);
            const uint32_t ob = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            max_ulp = std::max<uint64_t>(max_ulp, oa > ob ? oa - ob : ob - oa);
        }
        for (uint32_t i = logical; i < extent.compute; ++i)
            if (padded[i] != 0.0f) throw std::runtime_error("padded score received nonzero probability");
        if (oracle_delta > 1e-5) throw std::runtime_error("padded softmax differs from logical FP64 oracle");
        const bool boundary = logical % granularity == 1 || logical % granularity == 0 ||
            logical % granularity == granularity - 1;
        std::printf("extent_case distribution=%u logical=%u compute=%u padding=%u unpadded_vs_canonical_max_abs=%.9g ulp=%llu context_max_abs=%.9g fp64_max_abs=%.9g padded_probability=ZERO unpadded_us=%.3f padded_us=%.3f score_mask_probability_bytes=%zu/%zu boundary=%s\n",
            distribution, logical, extent.compute, extent.padding(), probability_delta,
            static_cast<unsigned long long>(max_ulp), context_delta, oracle_delta,
            unpadded_us, padded_us, static_cast<size_t>(3) * logical * sizeof(float),
            static_cast<size_t>(3) * extent.compute * sizeof(float), boundary ? "YES" : "NO");
    }
    bool rejected = false;
    try { (void)SoftmaxComputeExtent::make(1, 0); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("zero granularity was accepted");
    rejected = false;
    try { (void)SoftmaxComputeExtent::make(0, granularity); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("zero logical extent was accepted");
    rejected = false;
    try { (void)SoftmaxComputeExtent::make(std::numeric_limits<uint32_t>::max(), granularity); }
    catch (const std::length_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("overflowing compute extent was accepted");
    std::puts("softmax_extent_contract=PASS distributions=3 extents=1..64 logical_semantics=preserved");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "softmax_extent_contract=FAIL error=%s\n", error.what());
    return 1;
}
