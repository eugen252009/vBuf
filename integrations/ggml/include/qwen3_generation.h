#pragma once

#include "qwen3_cuda_core.h"
#include "qwen3_execution_plan.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace vbuf_ggml {

struct Qwen3DecodeProfile {
    uint64_t steps = 0;
    uint64_t plan_select_ns = 0;
    uint64_t optimizer_record_ns = 0;
    uint64_t control_prepare_ns = 0;
    uint64_t control_enqueue_cpu_ns = 0;
    uint64_t layer_resolve_cpu_ns = 0;
    uint64_t boundary_descriptor_cpu_ns = 0;
    uint64_t boundary_source_wait_d2h_ns = 0;
    uint64_t boundary_destination_h2d_wait_ns = 0;
    uint64_t control_h2d_bytes = 0;
    uint64_t control_h2d_calls = 0;
    uint64_t graph_submit_cpu_ns = 0;
    uint64_t candidate_setup_ns = 0;
    uint64_t candidate_dispatch_cpu_ns = 0;
    uint64_t boundary_wait_transfer_ns = 0;
    uint64_t final_device_wait_ns = 0;
    uint64_t final_output_readback_ns = 0;
    uint64_t final_sync_readback_ns = 0;
    uint64_t step_wall_ns = 0;
    std::vector<uint64_t> step_wall_samples_ns;
    std::vector<uint64_t> outer_token_wall_samples_ns;
};

#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
struct Qwen3LayerActivationCapture {
    uint32_t layer = 0;
    uint32_t device_id = 0;
    uint32_t position = 0;
    uint32_t rows = 0;
    bool prefill = false;
    std::vector<float> hidden;
};

// Qualification-only step/output snapshots, compiled into the diagnostic core.
struct Qwen3GenerationStepDiagnostic {
    uint32_t position = 0;
    uint32_t rows = 0;
    QwenExecutionPhase phase = QwenExecutionPhase::Decode;
    bool native_candidate_requested = false;
    bool native_candidate_selected = false;
    bool native_av_executed = false;
    uint32_t native_av_layer_graphs = 0;
    uint32_t diagnostic_native_av_interventions = 0;
};

struct Qwen3GenerationOutputCapture {
    uint32_t position = 0;
    uint32_t rows = 0;
    QwenExecutionPhase phase = QwenExecutionPhase::Decode;
    uint32_t input_token = 0;
    std::vector<float> hidden;
    std::vector<float> logits;
};

struct Qwen3AttentionAVBoundaryCapture {
    uint32_t layer = 0;
    uint32_t device_id = 0;
    bool prefill = false;
    bool native_intervention = false;
    uint32_t capacity = 0;
    uint32_t query_rows = 0;
    ggml_type value_type = GGML_TYPE_COUNT;
    ggml_type probability_type = GGML_TYPE_COUNT;
    ggml_type position_type = GGML_TYPE_COUNT;
    ggml_type score_type = GGML_TYPE_COUNT;
    ggml_type query_type = GGML_TYPE_COUNT;
    ggml_type key_type = GGML_TYPE_COUNT;
    std::array<int64_t, 4> value_ne{};
    std::array<size_t, 4> value_nb{};
    std::array<int64_t, 4> probability_ne{};
    std::array<size_t, 4> probability_nb{};
    std::array<int64_t, 4> output_ne{};
    std::array<size_t, 4> output_nb{};
    std::array<int64_t, 4> score_ne{};
    std::array<size_t, 4> score_nb{};
    std::array<int64_t, 4> query_ne{};
    std::array<size_t, 4> query_nb{};
    std::array<int64_t, 4> key_ne{};
    std::array<size_t, 4> key_nb{};
    std::vector<uint8_t> value_bytes;
    std::vector<uint8_t> key_bytes;
    std::vector<float> scores;
    std::vector<float> query;
    std::vector<float> probabilities;
    std::vector<int32_t> positions;
    std::vector<float> canonical_output;
    std::vector<float> native_output;
};
#endif

struct Qwen3GenerationExecution {
    std::vector<uint32_t> tokens;
    std::vector<float> final_logits;
    std::vector<float> final_hidden;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
    std::vector<Qwen3AttentionAVBoundaryCapture> attention_av_boundary_captures;
    std::vector<Qwen3LayerActivationCapture> layer_activation_captures;
    std::vector<Qwen3GenerationStepDiagnostic> attention_av_step_diagnostics;
    std::vector<Qwen3GenerationOutputCapture> attention_av_output_captures;
    uint64_t diagnostic_native_av_interventions = 0;
#endif
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t canonical_decode_steps = 0;
    uint64_t specialized_decode_steps = 0;
    uint64_t native_av_steps = 0;
    uint64_t native_av_layers = 0;
    uint64_t packed_v_copy_bytes_avoided = 0;
    Qwen3DecodeProfile decode_profile;
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
        std::shared_ptr<QwenCudaSessionState> session,
        bool capture_decode_profile = false,
        bool enable_native_attention_av_candidate = false
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
        , bool capture_attention_av_boundary = false
#endif
        );
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
    void configure_attention_av_diagnostic(uint32_t capture_position, bool capture_local_av,
        int32_t native_intervention_layer = -1, bool allow_native_prefill = true,
        bool allow_native_decode = true);
#endif
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
