#include "qwen3_execution_policy.h"

#include <stdexcept>
#include <string>

using namespace vbuf_ggml;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    Qwen3ExecutionPolicy serial_one{Qwen3ExecutionMode::Serial, 1};
    require(qwen3_qkv_kernel_budgets(serial_one) == std::vector<int>({1, 1, 1}),
        "serial one-thread QKV budgets");
    require(qwen3_gate_up_kernel_budgets(serial_one) == std::vector<int>({1, 1}),
        "serial one-thread gate/up budgets");
    require(qwen3_independent_task_batches(3, serial_one) ==
        std::vector<std::vector<size_t>>({{0}, {1}, {2}}), "serial task batches");

    Qwen3ExecutionPolicy serial_eight{Qwen3ExecutionMode::Serial, 8};
    require(qwen3_qkv_kernel_budgets(serial_eight) == std::vector<int>({8, 8, 8}),
        "serial graph keeps the requested per-kernel budget");
    require(qwen3_independent_task_batches(2, serial_eight) ==
        std::vector<std::vector<size_t>>({{0}, {1}}), "serial graph does not overlap branches");

    Qwen3ExecutionPolicy parallel_one{Qwen3ExecutionMode::Parallel, 1};
    require(qwen3_independent_task_batches(3, parallel_one) ==
        std::vector<std::vector<size_t>>({{0}, {1}, {2}}), "one-thread parallel policy is serial");

    Qwen3ExecutionPolicy parallel_two{Qwen3ExecutionMode::Parallel, 2};
    const auto two_batches = qwen3_independent_task_batches(3, parallel_two);
    require(two_batches == std::vector<std::vector<size_t>>({{0, 1}, {2}}),
        "two-thread QKV executes a bounded pair then remaining branch");
    const auto two_qkv = qwen3_qkv_kernel_budgets(parallel_two);
    require(two_qkv == std::vector<int>({1, 1, 1}) &&
        qwen3_batch_kernel_budget(two_batches[0], two_qkv) == 2 &&
        qwen3_batch_kernel_budget(two_batches[1], two_qkv) == 1,
        "two-thread QKV never oversubscribes");
    require(qwen3_independent_task_batches(2, parallel_two) ==
        std::vector<std::vector<size_t>>({{0, 1}}), "two-thread gate/up overlap");

    for (uint32_t threads : {3u, 4u, 8u}) {
        Qwen3ExecutionPolicy policy{Qwen3ExecutionMode::Parallel, threads};
        const auto qkv = qwen3_qkv_kernel_budgets(policy);
        const auto batches = qwen3_independent_task_batches(qkv.size(), policy);
        require(batches.size() == 1 && qwen3_batch_kernel_budget(batches[0], qkv) <= threads,
            "QKV active kernel budget exceeds total");
        const auto gate_up = qwen3_gate_up_kernel_budgets(policy);
        const auto gate_batches = qwen3_independent_task_batches(gate_up.size(), policy);
        for (const auto & batch : gate_batches)
            require(qwen3_batch_kernel_budget(batch, gate_up) <= threads,
                "gate/up active kernel budget exceeds total");
    }
    require(qwen3_qkv_kernel_budgets({Qwen3ExecutionMode::Parallel, 4}) ==
        std::vector<int>({2, 1, 1}), "Q projection receives weighted share");
    require(qwen3_gate_up_kernel_budgets({Qwen3ExecutionMode::Parallel, 8}) ==
        std::vector<int>({4, 4}), "gate/up split evenly at eight threads");

    bool rejected = false;
    try { (void) qwen3_qkv_kernel_budgets({Qwen3ExecutionMode::Parallel, 0}); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "zero CPU budget accepted");
    return 0;
}
