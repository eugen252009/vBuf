#pragma once

#include "vbuf_runtime_mode.h"
#include "vbuf_residency.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vbuf_ggml {

struct VbufGenerationConfig {
    std::string semantic_model;
    std::string source_endpoint;
    uint32_t block_count = 2;
    uint32_t context_capacity = 1032;
    uint64_t residency_capacity = 268435456;
    uint32_t max_new_tokens = 4;
    // Qualification-only source fault injection. Zero leaves the source unchanged.
    uint32_t source_failure_requests = 0;
    std::optional<uint64_t> source_failure_after_successful_requests;
    RuntimeMode mode = RuntimeMode::NormalInference;
    bool detailed_trace = false;
    uint32_t expert_workers = 1;
    uint32_t expert_threads = 1;
    std::vector<uint32_t> prompt_tokens;
    std::optional<uint32_t> stop_token;
    std::function<bool(uint32_t, uint32_t)> on_token;
    std::function<bool()> should_cancel;
};

struct VbufGenerationResult {
    uint64_t parallel_expert_jobs = 0;
    uint64_t parallel_expert_waves = 0;
    uint64_t peak_expert_workers = 0;
    uint64_t peak_expert_wave_bytes = 0;
    uint64_t expert_serial_fallbacks = 0;
    bool completed = false;
    bool cancelled = false;
    std::string error;
    std::vector<uint32_t> tokens;
    std::vector<float> final_logits;
    uint64_t prompt_tokens = 0;
    uint64_t source_bytes = 0;
    uint64_t materialized_bytes = 0;
    uint64_t reload_bytes = 0;
    uint64_t resident_bytes_before = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t peak_active_bytes = 0;
    uint64_t resident_bytes_after = 0;
    uint32_t active_lease_count_after = 0;
    uint64_t active_lease_bytes_after = 0;
    uint64_t active_inflight_bytes_after = 0;
    uint64_t peak_vram_bytes = 0;
    uint64_t post_run_free_vram_bytes = 0;
    uint64_t evictions = 0;
    uint64_t reacquisitions = 0;
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t qwen_model_upload_bytes = 0;
    uint64_t qwen_session_h2d_calls = 0;
    uint64_t qwen_session_h2d_bytes = 0;
    uint64_t qwen_session_d2h_calls = 0;
    uint64_t qwen_session_d2h_bytes = 0;
    uint64_t source_successful_requests = 0;
    uint64_t source_successful_requests_before_failure = 0;
    uint32_t completed_layers = 0;
    uint64_t completed_positions = 0;
    bool source_failure_injected = false;
    uint64_t elapsed_ns = 0;
};

struct VbufGenerationSnapshot {
    uint64_t request_count = 0;
    uint64_t active_generations = 0;
    uint64_t resident_bytes = 0;
    uint64_t resident_count = 0;
    uint32_t active_lease_count = 0;
    uint64_t active_lease_bytes = 0;
    uint64_t active_inflight_bytes = 0;
    uint64_t source_requests = 0;
    uint64_t source_bytes = 0;
    uint64_t source_unique_bytes = 0;
    uint64_t source_connections = 0;
    uint64_t materializations = 0;
    uint64_t reacquisitions = 0;
    uint64_t eviction_events = 0;
    uint64_t qwen_model_upload_tensors = 0;
    uint64_t qwen_model_upload_bytes = 0;
};

bool validate_vbuf_generation_model(const std::string & semantic_model,
    uint32_t block_count, std::string * error = nullptr);

class VbufGenerationSession;

// Owns immutable parsed model state and reusable model-scoped source,
// materializer, residency, and execution resources. A runtime must be managed
// by shared_ptr before create_session(); executions sharing one runtime are
// currently serialized and are not thread-safe for concurrent run()/snapshot().
class VbufModelRuntime : public std::enable_shared_from_this<VbufModelRuntime> {
public:
    VbufModelRuntime(const std::string & semantic_model, uint32_t block_count);
    VbufModelRuntime(const std::string & semantic_model, uint32_t block_count,
        std::shared_ptr<TensorResidencyStore> shared_residency);
    ~VbufModelRuntime();
    VbufModelRuntime(const VbufModelRuntime &) = delete;
    VbufModelRuntime & operator=(const VbufModelRuntime &) = delete;

    std::unique_ptr<VbufGenerationSession> create_session(uint32_t context_capacity = 1032);
    // Eagerly initialize exact-admitted Qwen CUDA model residency for long-lived servers.
    void prepare_qwen3_cuda(const std::string & source_endpoint);
    // Compatibility convenience: create a session, run one request, destroy it.
    VbufGenerationResult run(const VbufGenerationConfig & config);
    VbufGenerationSnapshot snapshot() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class VbufGenerationSession;
};

// Owns mutable generation/KV state and retains the model runtime. Its run()
// remains a whole-request operation; it resets at request start and leaves
// successfully processed state inspectable until reset(), the next run, or
// destruction. One session must not be used concurrently.
class VbufGenerationSession {
public:
    VbufGenerationSession(const std::string & semantic_model, uint32_t block_count);
    VbufGenerationSession(const std::string & semantic_model, uint32_t block_count,
        std::shared_ptr<TensorResidencyStore> shared_residency);
    explicit VbufGenerationSession(std::shared_ptr<VbufModelRuntime> runtime,
        uint32_t context_capacity = 1032);
    ~VbufGenerationSession();
    VbufGenerationSession(const VbufGenerationSession &) = delete;
    VbufGenerationSession & operator=(const VbufGenerationSession &) = delete;

    VbufGenerationResult run(const VbufGenerationConfig & config) const;
    VbufGenerationSnapshot snapshot() const;
    void reset() const noexcept;
    uint32_t current_context_length() const noexcept;

private:
    struct SessionState;
    std::shared_ptr<VbufModelRuntime> runtime_;
    std::shared_ptr<VbufModelRuntime::Impl> impl_;
    mutable std::unique_ptr<SessionState> session_state_;
};

VbufGenerationResult run_vbuf_generation(const VbufGenerationConfig & config);

} // namespace vbuf_ggml
