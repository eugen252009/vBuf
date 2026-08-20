#pragma once

#include <cstdint>

namespace vbuf_ggml {

enum class RuntimeMode {
    NormalInference,
    Qualification,
};

inline bool runs_reference_control(RuntimeMode mode) {
    return mode == RuntimeMode::Qualification;
}

struct RuntimeTiming {
    uint64_t actual_embedding_ns = 0;
    uint64_t reference_embedding_ns = 0;
    uint64_t actual_attention_ns = 0;
    uint64_t reference_attention_ns = 0;
    uint64_t actual_ffn_ns = 0;
    uint64_t reference_ffn_ns = 0;
    uint64_t actual_router_moe_ns = 0;
    uint64_t reference_router_moe_ns = 0;
    uint64_t actual_output_head_ns = 0;
    uint64_t reference_output_head_ns = 0;
};

} // namespace vbuf_ggml
