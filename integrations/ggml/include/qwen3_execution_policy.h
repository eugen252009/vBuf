#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

enum class Qwen3ExecutionMode { Serial, Parallel };

struct Qwen3ExecutionPolicy {
    Qwen3ExecutionMode mode = Qwen3ExecutionMode::Parallel;
    uint32_t total_cpu_threads = 4;

    void validate() const {
        if (total_cpu_threads == 0 || total_cpu_threads > 128)
            throw std::invalid_argument("Qwen3 CPU thread budget must be in 1..128");
    }
};

// Q is the larger projection. Give it about half of the budget and split the
// remainder between K/V; with three threads or fewer use one per branch.
inline std::vector<int> qwen3_qkv_kernel_budgets(const Qwen3ExecutionPolicy & policy) {
    policy.validate();
    if (policy.mode == Qwen3ExecutionMode::Serial)
        return {static_cast<int>(policy.total_cpu_threads),
            static_cast<int>(policy.total_cpu_threads), static_cast<int>(policy.total_cpu_threads)};
    if (policy.total_cpu_threads <= 3) return {1, 1, 1};
    const uint32_t q = (policy.total_cpu_threads + 1) / 2;
    const uint32_t remainder = policy.total_cpu_threads - q;
    const uint32_t k = (remainder + 1) / 2;
    return {static_cast<int>(q), static_cast<int>(k), static_cast<int>(remainder - k)};
}

inline std::vector<int> qwen3_gate_up_kernel_budgets(const Qwen3ExecutionPolicy & policy) {
    policy.validate();
    if (policy.mode == Qwen3ExecutionMode::Serial)
        return {static_cast<int>(policy.total_cpu_threads),
            static_cast<int>(policy.total_cpu_threads)};
    if (policy.total_cpu_threads == 1) return {1, 1};
    return {static_cast<int>((policy.total_cpu_threads + 1) / 2),
        static_cast<int>(policy.total_cpu_threads / 2)};
}

// Return deterministic synchronous batches. This expresses vBuf's graph-level
// concurrency policy separately from each GGML kernel's thread count.
inline std::vector<std::vector<size_t>> qwen3_independent_task_batches(
    size_t task_count, const Qwen3ExecutionPolicy & policy) {
    policy.validate();
    std::vector<std::vector<size_t>> batches;
    if (task_count == 0) return batches;
    const size_t concurrency = policy.mode == Qwen3ExecutionMode::Parallel ?
        std::min<size_t>(task_count, policy.total_cpu_threads) : 1;
    for (size_t first = 0; first < task_count; first += concurrency) {
        std::vector<size_t> batch;
        const size_t end = std::min(task_count, first + concurrency);
        for (size_t i = first; i < end; ++i) batch.push_back(i);
        batches.push_back(std::move(batch));
    }
    return batches;
}

inline uint32_t qwen3_batch_kernel_budget(const std::vector<size_t> & batch,
    const std::vector<int> & budgets) {
    uint32_t total = 0;
    for (size_t task : batch) {
        if (task >= budgets.size() || budgets[task] < 1 ||
            static_cast<uint32_t>(budgets[task]) > UINT32_MAX - total)
            throw std::invalid_argument("invalid Qwen3 task kernel budget");
        total += static_cast<uint32_t>(budgets[task]);
    }
    return total;
}

} // namespace vbuf_ggml
