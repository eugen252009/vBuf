#include "vbuf_prefetch_planner.h"

#include <cassert>
#include <cstdint>
#include <string>

using namespace vbuf_ggml;

int main() {
    const uint64_t dimensions[] = { 1, 1 };
    const uint8_t payload[sizeof(float)] = {};
    const VbufTensorView view{ 0, 2, dimensions, payload, sizeof(payload) };
    TensorWaveGraphView graph;
    graph.input_value = 0;
    graph.external_output = 5;
    graph.value_names = { "input", "norm_out", "gate_out", "up_out", "mul_out", "output" };
    graph.persistent = {
        { 30, "norm", { 0, 2, dimensions, payload, 8 } },
        { 20, "gate", { 0, 2, dimensions, payload, 4 } },
        { 10, "up", { 0, 2, dimensions, payload, 6 } },
        { 40, "down", { 0, 2, dimensions, payload, 10 } },
    };
    graph.operations = {
        { "norm_op", TensorWaveOpKind::RmsNorm,
            { { TensorWaveRef::Kind::Value, 0 }, { TensorWaveRef::Kind::Persistent, 0 } }, 1 },
        { "gate_op", TensorWaveOpKind::MulMat,
            { { TensorWaveRef::Kind::Persistent, 1 }, { TensorWaveRef::Kind::Value, 1 } }, 2 },
        { "up_op", TensorWaveOpKind::MulMat,
            { { TensorWaveRef::Kind::Persistent, 2 }, { TensorWaveRef::Kind::Value, 1 } }, 3 },
        { "swiglu_op", TensorWaveOpKind::SwiGlu,
            { { TensorWaveRef::Kind::Value, 2 }, { TensorWaveRef::Kind::Value, 3 } }, 4 },
        { "down_op", TensorWaveOpKind::MulMat,
            { { TensorWaveRef::Kind::Persistent, 3 }, { TensorWaveRef::Kind::Value, 4 } }, 5 },
    };

    TensorWavePlannerState state;
    state.available_values = { 0 };
    const PrefetchPlanner planner;
    const PrefetchPlan initial = planner.plan(graph, state, 3, UINT64_MAX);
    assert(initial.visible_candidates.size() == 4);
    assert(initial.visible_candidates[0].tensor_name == "norm");
    assert(initial.visible_candidates[0].dependency_distance == 0);
    assert(initial.visible_candidates[1].tensor_name == "up");
    assert(initial.visible_candidates[2].tensor_name == "gate");
    assert(initial.visible_candidates[1].dependency_distance == 1);
    assert(initial.visible_candidates[2].dependency_distance == 1);
    assert(initial.visible_candidates[3].tensor_name == "down");
    assert(initial.visible_candidates[3].dependency_distance == 3);
    assert(initial.total_planned_bytes == 28);

    assert(planner.plan(graph, state, 3, 0).candidates.empty());
    assert(planner.plan(graph, state, 3, 7).total_planned_bytes == 6);
    assert(planner.plan(graph, state, 3, 8).total_planned_bytes == 8);
    assert(planner.plan(graph, state, 3, 12).total_planned_bytes == 12);
    assert(planner.plan(graph, state, 3, 28).total_planned_bytes == 28);

    const TensorWavePlannerState before = state;
    state.resident_persistent = { 0 };
    const PrefetchPlan resident = planner.plan(graph, state, 3, UINT64_MAX);
    assert(resident.candidates.size() == 3);
    assert(resident.candidates.front().tensor_name == "up");
    state = before;
    assert(state.available_values == before.available_values);
    assert(state.completed_operations == before.completed_operations);
    assert(view.payload_len == sizeof(float));
    return 0;
}
