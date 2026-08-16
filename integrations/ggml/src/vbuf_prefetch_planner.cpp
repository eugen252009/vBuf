#include "vbuf_prefetch_planner.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace vbuf_ggml {
namespace {

bool contains(const std::vector<uint32_t> & values, uint32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool candidate_order(const PrefetchCandidate & lhs, const PrefetchCandidate & rhs) {
    if (lhs.dependency_distance != rhs.dependency_distance) {
        return lhs.dependency_distance < rhs.dependency_distance;
    }
    if (lhs.tensor_id != rhs.tensor_id) return lhs.tensor_id < rhs.tensor_id;
    if (lhs.required_by_op != rhs.required_by_op) {
        return lhs.required_by_op < rhs.required_by_op;
    }
    return lhs.tensor_ref < rhs.tensor_ref;
}

} // namespace

PrefetchPlan PrefetchPlanner::plan(
    const TensorWaveGraphView & graph,
    const TensorWavePlannerState & state,
    uint64_t max_dependency_distance,
    uint64_t byte_budget) const {
    PrefetchPlan result;
    result.byte_budget = byte_budget;

    std::vector<bool> completed(graph.operations.size(), false);
    for (uint32_t index : state.completed_operations) {
        if (index < completed.size()) completed[index] = true;
    }
    std::vector<bool> available(graph.value_names.size(), false);
    for (uint32_t index : state.available_values) {
        if (index < available.size()) available[index] = true;
    }

    std::unordered_map<uint32_t, uint32_t> producer;
    for (uint32_t index = 0; index < graph.operations.size(); ++index) {
        producer[graph.operations[index].output_value] = index;
    }

    std::vector<int64_t> distances(graph.operations.size(), -1);
    std::vector<bool> resolving(graph.operations.size(), false);
    const auto distance_for = [&](auto && self, uint32_t operation_index) -> uint64_t {
        if (operation_index >= graph.operations.size()) return max_dependency_distance + 1;
        if (completed[operation_index]) return 0;
        if (distances[operation_index] >= 0) {
            return static_cast<uint64_t>(distances[operation_index]);
        }
        if (resolving[operation_index]) return max_dependency_distance + 1;
        resolving[operation_index] = true;
        uint64_t distance = 0;
        for (const TensorWaveRef & input : graph.operations[operation_index].inputs) {
            if (input.kind != TensorWaveRef::Kind::Value) continue;
            if (input.index >= available.size()) return max_dependency_distance + 1;
            if (available[input.index]) continue;
            const auto found = producer.find(input.index);
            if (found == producer.end()) {
                distance = max_dependency_distance + 1;
                continue;
            }
            distance = std::max(distance, self(self, found->second) + 1);
        }
        resolving[operation_index] = false;
        distances[operation_index] = static_cast<int64_t>(distance);
        return distance;
    };

    std::unordered_map<uint32_t, PrefetchCandidate> by_tensor;
    for (uint32_t operation_index = 0; operation_index < graph.operations.size(); ++operation_index) {
        if (completed[operation_index]) continue;
        const uint64_t distance = distance_for(distance_for, operation_index);
        if (distance > max_dependency_distance) continue;
        const TensorWaveOp & operation = graph.operations[operation_index];
        for (const TensorWaveRef & input : operation.inputs) {
            if (input.kind != TensorWaveRef::Kind::Persistent ||
                input.index >= graph.persistent.size()) continue;
            const PersistentTensorRef & tensor = graph.persistent[input.index];
            PrefetchCandidate candidate{
                input.index, tensor.tensor_id, tensor.name, operation.op_id,
                distance, tensor.view.payload_len,
                contains(state.resident_persistent, input.index) };
            const auto found = by_tensor.find(input.index);
            if (found == by_tensor.end() || candidate_order(candidate, found->second)) {
                by_tensor[input.index] = std::move(candidate);
            }
        }
    }

    for (auto & entry : by_tensor) result.visible_candidates.push_back(std::move(entry.second));
    std::sort(result.visible_candidates.begin(), result.visible_candidates.end(), candidate_order);
    uint64_t remaining_budget = byte_budget;
    for (const PrefetchCandidate & candidate : result.visible_candidates) {
        if (candidate.already_resident) continue;
        // Oversized candidates are skipped so later smaller candidates can fit.
        if (candidate.byte_size > remaining_budget) continue;
        result.candidates.push_back(candidate);
        remaining_budget -= candidate.byte_size;
        result.total_planned_bytes += candidate.byte_size;
    }
    return result;
}

} // namespace vbuf_ggml
