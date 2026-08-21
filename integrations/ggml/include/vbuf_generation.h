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
    uint64_t peak_resident_bytes = 0;
    uint64_t peak_active_bytes = 0;
    uint64_t elapsed_ns = 0;
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

VbufGenerationResult run_vbuf_generation(const VbufGenerationConfig & config);

} // namespace vbuf_ggml
