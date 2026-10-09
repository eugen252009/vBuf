#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

namespace vbuf_ggml {

struct QwenExecutionPlan;
struct QwenExecutionCandidate;
enum class QwenExecutionPhase : uint8_t;

enum class Qwen3AttentionAVPath : uint8_t {
    PackedCanonical,
    NativeLayout,
};

struct Qwen3AttentionAVFacts {
    uint32_t sm_version = 0;
    uint32_t capacity = 0;
    uint32_t visible_context = 0;
    uint32_t query_rows = 0;
    uint32_t kv_heads = 0;
    uint32_t query_heads = 0;
    uint32_t head_dimension = 0;
    ggml_type value_type = GGML_TYPE_COUNT;
    ggml_type probability_type = GGML_TYPE_COUNT;
    ggml_type position_type = GGML_TYPE_COUNT;
    bool canonical_position_major_v = false;
};

struct Qwen3AttentionAVResolution {
    Qwen3AttentionAVPath path = Qwen3AttentionAVPath::PackedCanonical;
    bool native_requested = false;
    bool native_supported = false;
    const char * reason = "native AV not requested";
};

// Pure, fail-closed capability selection. Packed canonical AV remains the
// fallback for every unsupported geometry/device/type combination.
Qwen3AttentionAVResolution resolve_qwen3_attention_av(
    const Qwen3AttentionAVFacts & facts, bool request_native) noexcept;

std::string qwen3_native_attention_av_candidate_identity(
    const QwenExecutionPlan & plan, QwenExecutionPhase phase);
QwenExecutionCandidate make_qwen3_native_attention_av_candidate(
    const QwenExecutionPlan & plan, QwenExecutionPhase phase);

} // namespace vbuf_ggml
