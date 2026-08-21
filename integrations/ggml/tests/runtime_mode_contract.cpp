#include "vbuf_runtime_mode.h"

#include <cassert>

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
    assert(timing.result_handling_ns == 13);
    assert(timing.stage_graph_build_ns[static_cast<size_t>(vbuf_ggml::AttributionStage::RoutedExpert)] == 5);

    const vbuf_ggml::RuntimeTiming before = timing;
    timing.record_attribution(vbuf_ggml::AttributionStage::Attention,
        "attention_q", "compute", 17);
    const vbuf_ggml::RuntimeTiming delta = timing.delta(before);
    assert(delta.backend_compute_ns == 17);
    assert(delta.attention_backend_compute_ns == 17);
    assert(delta.routed_expert_gate_up_compute_ns == 0);
    return 0;
}
