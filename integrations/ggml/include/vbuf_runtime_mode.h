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

struct AttributionOperationSignature {
    std::array<uint64_t, 4> input_dims{};
    std::array<uint64_t, 4> weight_dims{};
    std::array<uint64_t, 4> output_dims{};
    uint8_t input_rank = 0;
    uint8_t weight_rank = 0;
    uint8_t output_rank = 0;
    uint8_t input_representation = 0;
    uint8_t weight_representation = 0;
    uint8_t output_representation = 0;
    uint64_t weight_payload_bytes = 0;
    uint64_t count = 0;
    uint64_t logical_macs = 0;

    bool matches(const AttributionOperationSignature & other) const {
        return input_dims == other.input_dims && weight_dims == other.weight_dims &&
            output_dims == other.output_dims && input_rank == other.input_rank &&
            weight_rank == other.weight_rank && output_rank == other.output_rank &&
            input_representation == other.input_representation &&
            weight_representation == other.weight_representation &&
            output_representation == other.output_representation &&
            weight_payload_bytes == other.weight_payload_bytes;
    }
};

struct RuntimeTiming {
    static constexpr size_t MAX_GATE_UP_COMPUTE_SAMPLES = 2048;
    static constexpr size_t MAX_DOWN_COMPUTE_SAMPLES = 1024;

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

    uint64_t descriptor_setup_ns = 0;
    uint64_t routed_moe_layer_count = 0;
    uint64_t routed_expert_invocation_count = 0;
    std::array<uint64_t, 64> routed_expert_id_counts{};
    uint64_t gate_up_logical_invocations = 0;
    uint64_t gate_up_backend_submissions = 0;
    uint64_t gate_matmul_calls = 0;
    uint64_t up_matmul_calls = 0;
    uint64_t gate_matmul_compute_ns = 0;
    uint64_t up_matmul_compute_ns = 0;
    uint64_t down_logical_invocations = 0;
    uint64_t down_backend_submissions = 0;
    uint64_t gate_up_ready_ns = 0;
    uint64_t gate_up_descriptor_setup_ns = 0;
    uint64_t gate_up_graph_build_ns = 0;
    uint64_t gate_up_backend_allocation_ns = 0;
    uint64_t gate_up_result_handling_ns = 0;
    uint64_t down_ready_ns = 0;
    uint64_t down_descriptor_setup_ns = 0;
    uint64_t down_graph_build_ns = 0;
    uint64_t down_backend_allocation_ns = 0;
    uint64_t down_result_handling_ns = 0;
    uint64_t indexed_expert_layer_count = 0;
    uint64_t indexed_expert_bank_submissions = 0;
    uint64_t indexed_expert_logical_slots = 0;
    uint64_t indexed_expert_compute_ns = 0;
    uint64_t indexed_expert_materialized_bytes = 0;
    uint64_t indexed_expert_repack_bytes = 0;
    std::array<AttributionOperationSignature, 4> gate_up_signatures{};
    std::array<AttributionOperationSignature, 4> down_signatures{};
    std::array<uint64_t, MAX_GATE_UP_COMPUTE_SAMPLES> gate_up_compute_samples{};
    std::array<uint64_t, MAX_DOWN_COMPUTE_SAMPLES> down_compute_samples{};
    uint32_t gate_up_compute_sample_count = 0;
    uint32_t down_compute_sample_count = 0;

    void record_routed_expert_invocation(uint32_t expert_id) {
        ++routed_expert_invocation_count;
        ++gate_up_logical_invocations;
        ++down_logical_invocations;
        if (expert_id < routed_expert_id_counts.size()) ++routed_expert_id_counts[expert_id];
    }

    void record_operation_signature(std::string_view operation,
        uint8_t input_rank, const std::array<uint64_t, 4> & input_dims,
        uint8_t input_representation, uint8_t weight_rank,
        const std::array<uint64_t, 4> & weight_dims, uint8_t weight_representation,
        uint64_t weight_payload_bytes, uint8_t output_rank,
        const std::array<uint64_t, 4> & output_dims, uint8_t output_representation) {
        const bool gate_up = operation == "expert_gate_matmul" || operation == "expert_up_matmul";
        const bool down = operation == "expert_down_matmul";
        if (!gate_up && !down) return;
        AttributionOperationSignature signature{
            input_dims, weight_dims, output_dims, input_rank, weight_rank, output_rank,
            input_representation, weight_representation, output_representation,
            weight_payload_bytes, 1, 0 };
        if (input_rank >= 2 && weight_rank >= 2) {
            signature.logical_macs = input_dims[1] * weight_dims[1] * input_dims[0];
        }
        auto & signatures = gate_up ? gate_up_signatures : down_signatures;
        for (auto & existing : signatures) {
            if (existing.count != 0 && existing.matches(signature)) {
                ++existing.count;
                existing.logical_macs += signature.logical_macs;
                return;
            }
        }
        for (auto & existing : signatures) {
            if (existing.count == 0) {
                existing = signature;
                return;
            }
        }
    }

    void record_attribution(AttributionStage stage, const char * operation,
        std::string_view phase, uint64_t elapsed_ns) {
        const std::string_view op = operation == nullptr ? std::string_view{} : operation;
        const bool gate_up = stage == AttributionStage::RoutedExpert &&
            (op == "expert_gate_matmul" || op == "expert_up_matmul");
        const bool down = stage == AttributionStage::RoutedExpert && op == "expert_down_matmul";
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
            if (gate_up) gate_up_ready_ns += elapsed_ns;
            if (down) down_ready_ns += elapsed_ns;
            return;
        }
        if (phase == "descriptor_setup") {
            descriptor_setup_ns += elapsed_ns;
            if (gate_up) gate_up_descriptor_setup_ns += elapsed_ns;
            if (down) down_descriptor_setup_ns += elapsed_ns;
            return;
        }
        if (phase == "graph_build") {
            graph_build_ns += elapsed_ns;
            stage_graph_build_ns[attribution_stage_index(stage)] += elapsed_ns;
            if (gate_up) gate_up_graph_build_ns += elapsed_ns;
            if (down) down_graph_build_ns += elapsed_ns;
            return;
        }
        if (phase == "graph_allocation") {
            graph_allocation_ns += elapsed_ns;
            stage_graph_allocation_ns[attribution_stage_index(stage)] += elapsed_ns;
            if (gate_up) gate_up_backend_allocation_ns += elapsed_ns;
            if (down) down_backend_allocation_ns += elapsed_ns;
            return;
        }
        if (phase == "result") {
            result_handling_ns += elapsed_ns;
            stage_result_handling_ns[attribution_stage_index(stage)] += elapsed_ns;
            if (gate_up) gate_up_result_handling_ns += elapsed_ns;
            if (down) down_result_handling_ns += elapsed_ns;
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
            if (gate_up) {
                routed_expert_gate_up_compute_ns += elapsed_ns;
                ++gate_up_backend_submissions;
                if (op == "expert_gate_matmul") ++gate_matmul_calls;
                if (op == "expert_up_matmul") ++up_matmul_calls;
                if (op == "expert_gate_matmul") gate_matmul_compute_ns += elapsed_ns;
                if (op == "expert_up_matmul") up_matmul_compute_ns += elapsed_ns;
                if (gate_up_compute_sample_count < gate_up_compute_samples.size())
                    gate_up_compute_samples[gate_up_compute_sample_count++] = elapsed_ns;
            } else if (op == "expert_swiglu")
                routed_expert_activation_ns += elapsed_ns;
            else if (down) {
                routed_expert_down_compute_ns += elapsed_ns;
                ++down_backend_submissions;
                if (down_compute_sample_count < down_compute_samples.size())
                    down_compute_samples[down_compute_sample_count++] = elapsed_ns;
            } else
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
        VBUF_TIMING_FIELD(descriptor_setup_ns);
        VBUF_TIMING_FIELD(routed_moe_layer_count);
        VBUF_TIMING_FIELD(routed_expert_invocation_count);
        VBUF_TIMING_FIELD(gate_up_logical_invocations);
        VBUF_TIMING_FIELD(gate_up_backend_submissions);
        VBUF_TIMING_FIELD(gate_matmul_calls);
        VBUF_TIMING_FIELD(up_matmul_calls);
        VBUF_TIMING_FIELD(gate_matmul_compute_ns);
        VBUF_TIMING_FIELD(up_matmul_compute_ns);
        VBUF_TIMING_FIELD(down_logical_invocations);
        VBUF_TIMING_FIELD(down_backend_submissions);
        VBUF_TIMING_FIELD(gate_up_ready_ns);
        VBUF_TIMING_FIELD(gate_up_descriptor_setup_ns);
        VBUF_TIMING_FIELD(gate_up_graph_build_ns);
        VBUF_TIMING_FIELD(gate_up_backend_allocation_ns);
        VBUF_TIMING_FIELD(gate_up_result_handling_ns);
        VBUF_TIMING_FIELD(down_ready_ns);
        VBUF_TIMING_FIELD(down_descriptor_setup_ns);
        VBUF_TIMING_FIELD(down_graph_build_ns);
        VBUF_TIMING_FIELD(down_backend_allocation_ns);
        VBUF_TIMING_FIELD(down_result_handling_ns);
        VBUF_TIMING_FIELD(indexed_expert_layer_count);
        VBUF_TIMING_FIELD(indexed_expert_bank_submissions);
        VBUF_TIMING_FIELD(indexed_expert_logical_slots);
        VBUF_TIMING_FIELD(indexed_expert_compute_ns);
        VBUF_TIMING_FIELD(indexed_expert_materialized_bytes);
        VBUF_TIMING_FIELD(indexed_expert_repack_bytes);
#undef VBUF_TIMING_FIELD
        for (size_t index = 0; index < routed_expert_id_counts.size(); ++index)
            update(&routed_expert_id_counts[index], other.routed_expert_id_counts[index]);
        for (size_t index = 0; index < stage_graph_build_ns.size(); ++index) {
            update(&stage_graph_build_ns[index], other.stage_graph_build_ns[index]);
            update(&stage_graph_allocation_ns[index], other.stage_graph_allocation_ns[index]);
            update(&stage_result_handling_ns[index], other.stage_result_handling_ns[index]);
        }
        const auto update_signatures = [subtract](auto * destination, const auto & source) {
            for (const auto & incoming : source) {
                if (incoming.count == 0) continue;
                auto found = destination->end();
                for (auto iterator = destination->begin(); iterator != destination->end(); ++iterator) {
                    if (iterator->count != 0 && iterator->matches(incoming)) {
                        found = iterator;
                        break;
                    }
                }
                if (found == destination->end()) {
                    for (auto iterator = destination->begin(); iterator != destination->end(); ++iterator) {
                        if (iterator->count == 0) {
                            if (!subtract) *iterator = incoming;
                            break;
                        }
                    }
                    continue;
                }
                if (found != destination->end()) {
                    if (subtract) {
                        found->count -= incoming.count;
                        found->logical_macs -= incoming.logical_macs;
                        if (found->count == 0) *found = {};
                    } else {
                        found->count += incoming.count;
                        found->logical_macs += incoming.logical_macs;
                    }
                }
            }
        };
        update_signatures(&gate_up_signatures, other.gate_up_signatures);
        update_signatures(&down_signatures, other.down_signatures);
        if (!subtract) {
            for (uint32_t index = 0; index < other.gate_up_compute_sample_count &&
                gate_up_compute_sample_count < gate_up_compute_samples.size(); ++index)
                gate_up_compute_samples[gate_up_compute_sample_count++] = other.gate_up_compute_samples[index];
            for (uint32_t index = 0; index < other.down_compute_sample_count &&
                down_compute_sample_count < down_compute_samples.size(); ++index)
                down_compute_samples[down_compute_sample_count++] = other.down_compute_samples[index];
        }
    }

    RuntimeTiming delta(const RuntimeTiming & before) const {
        RuntimeTiming result = *this;
        result.gate_up_compute_sample_count = gate_up_compute_sample_count >= before.gate_up_compute_sample_count
            ? gate_up_compute_sample_count - before.gate_up_compute_sample_count : 0;
        result.down_compute_sample_count = down_compute_sample_count >= before.down_compute_sample_count
            ? down_compute_sample_count - before.down_compute_sample_count : 0;
        for (uint32_t index = 0; index < result.gate_up_compute_sample_count; ++index)
            result.gate_up_compute_samples[index] = gate_up_compute_samples[before.gate_up_compute_sample_count + index];
        for (uint32_t index = 0; index < result.down_compute_sample_count; ++index)
            result.down_compute_samples[index] = down_compute_samples[before.down_compute_sample_count + index];
        result.accumulate(before, true);
        return result;
    }
};

} // namespace vbuf_ggml
