#pragma once

#include "qwen3_cuda_core.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vbuf_ggml {

struct Qwen3GenerationExecution {
    std::vector<uint32_t> tokens;
    std::vector<float> final_logits;
    std::vector<float> final_hidden;
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t h2d_calls = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2h_calls = 0;
    uint64_t d2h_bytes = 0;
    uint64_t peak_vram_bytes = 0;
    uint64_t post_run_free_vram_bytes = 0;
    uint64_t completed_positions = 0;
    uint64_t boundary_handoffs = 0;
    uint64_t boundary_bytes = 0;
    uint64_t boundary_audits = 0;
    uint64_t boundary_audit_bytes = 0;
    bool boundary_bytes_equal = true;
    bool completed = false;
    bool cancelled = false;
};

// Session-scoped graph orchestration over the shared Qwen CUDA layer builder.
// Weights/backend remain runtime-owned; mutable KV and buffers remain session-owned.
class Qwen3MultiDeviceGenerationExecutor final {
public:
    Qwen3MultiDeviceGenerationExecutor(Qwen3Model & model,
        std::shared_ptr<QwenCudaRuntimeState> runtime,
        std::shared_ptr<QwenCudaSessionState> session);
    ~Qwen3MultiDeviceGenerationExecutor();
    Qwen3MultiDeviceGenerationExecutor(const Qwen3MultiDeviceGenerationExecutor &) = delete;
    Qwen3MultiDeviceGenerationExecutor & operator=(const Qwen3MultiDeviceGenerationExecutor &) = delete;

    Qwen3GenerationExecution run(const std::vector<uint32_t> & prompt,
        uint32_t max_new_tokens, std::optional<uint32_t> stop_token = std::nullopt,
        const std::function<bool(uint32_t, uint32_t)> & on_token = {},
        const std::function<bool()> & should_cancel = {},
        const std::function<void(uint32_t, const std::vector<float> &)> & on_progress = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Qwen3GenerationExecutor final {
public:
    Qwen3GenerationExecutor(Qwen3Model & model,
        std::shared_ptr<QwenCudaRuntimeState> runtime,
        std::shared_ptr<QwenCudaSessionState> session,
        bool capture_final_hidden = false);
    ~Qwen3GenerationExecutor();
    Qwen3GenerationExecutor(const Qwen3GenerationExecutor &) = delete;
    Qwen3GenerationExecutor & operator=(const Qwen3GenerationExecutor &) = delete;

    Qwen3GenerationExecution run(const std::vector<uint32_t> & prompt,
        uint32_t max_new_tokens, std::optional<uint32_t> stop_token = std::nullopt,
        const std::function<bool(uint32_t, uint32_t)> & on_token = {},
        const std::function<bool()> & should_cancel = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace vbuf_ggml
