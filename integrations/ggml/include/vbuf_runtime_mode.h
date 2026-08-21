#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vbuf_ggml {

enum class RuntimeMode {
    NormalInference,
    Qualification,
};

inline bool runs_reference_control(RuntimeMode mode) {
    return mode == RuntimeMode::Qualification;
}

enum class AttributionStage {
    Other,
    Embedding,
    Attention,
    DenseFfn,
    Router,
    RoutedExpert,
    SharedExpert,
    OutputHead,
};

inline size_t attribution_stage_index(AttributionStage stage) {
    return static_cast<size_t>(stage);
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

    uint64_t attribution_ready_ns = 0;
    uint64_t graph_build_ns = 0;
    uint64_t graph_allocation_ns = 0;
    uint64_t backend_compute_ns = 0;
    uint64_t result_handling_ns = 0;
    std::array<uint64_t, 8> stage_graph_build_ns{};
    std::array<uint64_t, 8> stage_graph_allocation_ns{};
    std::array<uint64_t, 8> stage_result_handling_ns{};
    uint64_t embedding_ready_ns = 0;
    uint64_t embedding_backend_compute_ns = 0;
    uint64_t attention_ready_ns = 0;
    uint64_t attention_backend_compute_ns = 0;
    uint64_t router_ready_ns = 0;
    uint64_t router_backend_compute_ns = 0;
    uint64_t router_selection_ns = 0;
    uint64_t dense_ffn_ready_ns = 0;
    uint64_t dense_ffn_backend_compute_ns = 0;
    uint64_t routed_expert_ready_ns = 0;
    uint64_t routed_expert_gate_up_compute_ns = 0;
    uint64_t routed_expert_activation_ns = 0;
    uint64_t routed_expert_down_compute_ns = 0;
    uint64_t routed_expert_other_compute_ns = 0;
    uint64_t routed_expert_accumulation_ns = 0;
    uint64_t shared_expert_ready_ns = 0;
    uint64_t shared_expert_backend_compute_ns = 0;
    uint64_t output_head_ready_ns = 0;
    uint64_t output_head_backend_compute_ns = 0;

    void record_attribution(AttributionStage stage, const char * operation,
        std::string_view phase, uint64_t elapsed_ns) {
        const std::string_view op = operation == nullptr ? std::string_view{} : operation;
        if (phase == "ready") {
            attribution_ready_ns += elapsed_ns;
            switch (stage) {
            case AttributionStage::Embedding: embedding_ready_ns += elapsed_ns; break;
            case AttributionStage::Attention: attention_ready_ns += elapsed_ns; break;
            case AttributionStage::Router: router_ready_ns += elapsed_ns; break;
            case AttributionStage::DenseFfn: dense_ffn_ready_ns += elapsed_ns; break;
            case AttributionStage::RoutedExpert: routed_expert_ready_ns += elapsed_ns; break;
            case AttributionStage::SharedExpert: shared_expert_ready_ns += elapsed_ns; break;
            default: break;
            }
            return;
        }
        if (phase == "graph_build") {
            graph_build_ns += elapsed_ns;
            stage_graph_build_ns[attribution_stage_index(stage)] += elapsed_ns;
            return;
        }
        if (phase == "graph_allocation") {
            graph_allocation_ns += elapsed_ns;
            stage_graph_allocation_ns[attribution_stage_index(stage)] += elapsed_ns;
            return;
        }
        if (phase == "result") {
            result_handling_ns += elapsed_ns;
            stage_result_handling_ns[attribution_stage_index(stage)] += elapsed_ns;
            return;
        }
        if (phase != "compute") return;

        backend_compute_ns += elapsed_ns;
        switch (stage) {
        case AttributionStage::Embedding:
            embedding_backend_compute_ns += elapsed_ns;
            break;
        case AttributionStage::Attention:
            attention_backend_compute_ns += elapsed_ns;
            break;
        case AttributionStage::Router:
            router_backend_compute_ns += elapsed_ns;
            break;
        case AttributionStage::DenseFfn:
            dense_ffn_backend_compute_ns += elapsed_ns;
            break;
        case AttributionStage::RoutedExpert:
            if (op == "expert_gate_matmul" || op == "expert_up_matmul")
                routed_expert_gate_up_compute_ns += elapsed_ns;
            else if (op == "expert_swiglu")
                routed_expert_activation_ns += elapsed_ns;
            else if (op == "expert_down_matmul")
                routed_expert_down_compute_ns += elapsed_ns;
            else
                routed_expert_other_compute_ns += elapsed_ns;
            break;
        case AttributionStage::SharedExpert:
            shared_expert_backend_compute_ns += elapsed_ns;
            break;
        case AttributionStage::OutputHead:
            output_head_backend_compute_ns += elapsed_ns;
            break;
        default:
            break;
        }
    }

    void accumulate(const RuntimeTiming & other, bool subtract = false) {
        const auto update = [subtract](uint64_t * destination, uint64_t value) {
            *destination = subtract ? *destination - value : *destination + value;
        };
#define VBUF_TIMING_FIELD(name) update(&name, other.name)
        VBUF_TIMING_FIELD(actual_embedding_ns);
        VBUF_TIMING_FIELD(reference_embedding_ns);
        VBUF_TIMING_FIELD(actual_attention_ns);
        VBUF_TIMING_FIELD(reference_attention_ns);
        VBUF_TIMING_FIELD(actual_ffn_ns);
        VBUF_TIMING_FIELD(reference_ffn_ns);
        VBUF_TIMING_FIELD(actual_router_moe_ns);
        VBUF_TIMING_FIELD(reference_router_moe_ns);
        VBUF_TIMING_FIELD(actual_output_head_ns);
        VBUF_TIMING_FIELD(reference_output_head_ns);
        VBUF_TIMING_FIELD(attribution_ready_ns);
        VBUF_TIMING_FIELD(graph_build_ns);
        VBUF_TIMING_FIELD(graph_allocation_ns);
        VBUF_TIMING_FIELD(backend_compute_ns);
        VBUF_TIMING_FIELD(result_handling_ns);
        VBUF_TIMING_FIELD(embedding_ready_ns);
        VBUF_TIMING_FIELD(embedding_backend_compute_ns);
        VBUF_TIMING_FIELD(attention_ready_ns);
        VBUF_TIMING_FIELD(attention_backend_compute_ns);
        VBUF_TIMING_FIELD(router_ready_ns);
        VBUF_TIMING_FIELD(router_backend_compute_ns);
        VBUF_TIMING_FIELD(router_selection_ns);
        VBUF_TIMING_FIELD(dense_ffn_ready_ns);
        VBUF_TIMING_FIELD(dense_ffn_backend_compute_ns);
        VBUF_TIMING_FIELD(routed_expert_ready_ns);
        VBUF_TIMING_FIELD(routed_expert_gate_up_compute_ns);
        VBUF_TIMING_FIELD(routed_expert_activation_ns);
        VBUF_TIMING_FIELD(routed_expert_down_compute_ns);
        VBUF_TIMING_FIELD(routed_expert_other_compute_ns);
        VBUF_TIMING_FIELD(routed_expert_accumulation_ns);
        VBUF_TIMING_FIELD(shared_expert_ready_ns);
        VBUF_TIMING_FIELD(shared_expert_backend_compute_ns);
        VBUF_TIMING_FIELD(output_head_ready_ns);
        VBUF_TIMING_FIELD(output_head_backend_compute_ns);
#undef VBUF_TIMING_FIELD
        for (size_t index = 0; index < stage_graph_build_ns.size(); ++index) {
            update(&stage_graph_build_ns[index], other.stage_graph_build_ns[index]);
            update(&stage_graph_allocation_ns[index], other.stage_graph_allocation_ns[index]);
            update(&stage_result_handling_ns[index], other.stage_result_handling_ns[index]);
        }
    }

    RuntimeTiming delta(const RuntimeTiming & before) const {
        RuntimeTiming result = *this;
        result.accumulate(before, true);
        return result;
    }
};

} // namespace vbuf_ggml
