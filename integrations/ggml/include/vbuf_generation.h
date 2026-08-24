#pragma once

#include "vbuf_runtime_mode.h"

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
    uint64_t residency_capacity = 268435456;
    uint32_t max_new_tokens = 4;
    // Qualification-only source fault injection. Zero leaves the source unchanged.
    uint32_t source_failure_requests = 0;
    std::optional<uint64_t> source_failure_after_successful_requests;
    RuntimeMode mode = RuntimeMode::NormalInference;
    std::vector<uint32_t> prompt_tokens;
    std::optional<uint32_t> stop_token;
    std::function<bool(uint32_t, uint32_t)> on_token;
    std::function<bool()> should_cancel;
};

struct VbufGenerationResult {
    bool completed = false;
    bool cancelled = false;
    std::string error;
    std::vector<uint32_t> tokens;
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
    uint64_t evictions = 0;
    uint64_t reacquisitions = 0;
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
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
};

bool validate_vbuf_generation_model(const std::string & semantic_model,
    uint32_t block_count, std::string * error = nullptr);

class VbufGenerationSession {
public:
    VbufGenerationSession(const std::string & semantic_model, uint32_t block_count);
    ~VbufGenerationSession();
    VbufGenerationSession(const VbufGenerationSession &) = delete;
    VbufGenerationSession & operator=(const VbufGenerationSession &) = delete;

    VbufGenerationResult run(const VbufGenerationConfig & config) const;
    VbufGenerationSnapshot snapshot() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

VbufGenerationResult run_vbuf_generation(const VbufGenerationConfig & config);

} // namespace vbuf_ggml
