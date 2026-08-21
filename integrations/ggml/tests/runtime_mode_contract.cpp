#include "vbuf_runtime_mode.h"

#include <array>
#include <cassert>
#include <cstdint>

int main() {
    assert(!vbuf_ggml::runs_reference_control(vbuf_ggml::RuntimeMode::NormalInference));
    assert(vbuf_ggml::runs_reference_control(vbuf_ggml::RuntimeMode::Qualification));

    vbuf_ggml::RuntimeTiming timing;
    timing.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "ready", 3);
    timing.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "graph_build", 5);
    timing.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "graph_allocation", 7);
    timing.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "compute", 11);
    timing.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "result", 13);
    assert(timing.attribution_ready_ns == 3);
    assert(timing.routed_expert_ready_ns == 3);
    assert(timing.graph_build_ns == 5);
    assert(timing.graph_allocation_ns == 7);
    assert(timing.backend_compute_ns == 11);
    assert(timing.routed_expert_gate_up_compute_ns == 11);
    assert(timing.gate_matmul_compute_ns == 11);
    assert(timing.gate_up_compute_sample_count == 1);
    assert(timing.gate_up_compute_samples[0] == 11);
    assert(timing.result_handling_ns == 13);
    assert(timing.stage_graph_build_ns[static_cast<size_t>(vbuf_ggml::AttributionStage::RoutedExpert)] == 5);

    const vbuf_ggml::RuntimeTiming before = timing;
    timing.record_attribution(vbuf_ggml::AttributionStage::Attention,
        "attention_q", "compute", 17);
    const vbuf_ggml::RuntimeTiming delta = timing.delta(before);
    assert(delta.backend_compute_ns == 17);
    assert(delta.attention_backend_compute_ns == 17);
    assert(delta.routed_expert_gate_up_compute_ns == 0);

    vbuf_ggml::RuntimeTiming moe;
    moe.routed_moe_layer_count = 1;
    moe.record_routed_expert_invocation(3);
    const std::array<uint64_t, 4> input_dims{ 2048, 1, 0, 0 };
    const std::array<uint64_t, 4> gate_up_dims{ 2048, 2816, 0, 0 };
    const std::array<uint64_t, 4> gate_up_output{ 2816, 1, 0, 0 };
    const std::array<uint64_t, 4> down_dims{ 2816, 2048, 0, 0 };
    const std::array<uint64_t, 4> down_output{ 2048, 1, 0, 0 };
    moe.record_operation_signature("expert_gate_matmul", 2, input_dims, 0, 2,
        gate_up_dims, 10, 1486848, 2, gate_up_output, 0);
    moe.record_operation_signature("expert_up_matmul", 2, input_dims, 0, 2,
        gate_up_dims, 10, 1486848, 2, gate_up_output, 0);
    moe.record_operation_signature("expert_down_matmul", 2, input_dims, 0, 2,
        down_dims, 10, 1486848, 2, down_output, 0);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "ready", 1);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "descriptor_setup", 2);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "graph_build", 3);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "graph_allocation", 4);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "compute", 5);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_gate_matmul", "result", 6);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_up_matmul", "compute", 7);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_down_matmul", "graph_build", 8);
    moe.record_attribution(vbuf_ggml::AttributionStage::RoutedExpert,
        "expert_down_matmul", "compute", 9);
    assert(moe.routed_expert_invocation_count == 1);
    assert(moe.gate_up_logical_invocations == 1);
    assert(moe.down_logical_invocations == 1);
    assert(moe.gate_up_backend_submissions == 2);
    assert(moe.gate_matmul_calls == 1);
    assert(moe.up_matmul_calls == 1);
    assert(moe.gate_matmul_compute_ns == 5);
    assert(moe.up_matmul_compute_ns == 7);
    assert(moe.gate_up_compute_sample_count == 2);
    assert(moe.gate_up_compute_samples[0] == 5);
    assert(moe.gate_up_compute_samples[1] == 7);
    assert(moe.down_backend_submissions == 1);
    assert(moe.down_compute_sample_count == 1);
    assert(moe.down_compute_samples[0] == 9);
    assert(moe.gate_up_descriptor_setup_ns == 2);
    assert(moe.gate_up_graph_build_ns == 3);
    assert(moe.gate_up_backend_allocation_ns == 4);
    assert(moe.gate_up_result_handling_ns == 6);
    assert(moe.gate_up_signatures[0].count == 2);
    assert(moe.gate_up_signatures[0].logical_macs == 2 * 2048ULL * 2816ULL);
    assert(moe.down_signatures[0].count == 1);
    assert(moe.down_signatures[0].weight_dims[0] == 2816);
    vbuf_ggml::RuntimeTiming accumulated;
    accumulated.accumulate(moe);
    assert(accumulated.gate_up_signatures[0].count == 2);
    assert(accumulated.gate_up_compute_sample_count == 2);
    assert(accumulated.down_compute_sample_count == 1);
    const vbuf_ggml::RuntimeTiming signature_delta = accumulated.delta(moe);
    assert(signature_delta.gate_up_signatures[0].count == 0);
    assert(signature_delta.gate_up_compute_sample_count == 0);
    assert(signature_delta.down_compute_sample_count == 0);
    return 0;
}
