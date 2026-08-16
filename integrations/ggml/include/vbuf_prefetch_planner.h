#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vbuf_tensor_wave.h"

namespace vbuf_ggml {

struct PrefetchCandidate {
    uint32_t tensor_ref = UINT32_MAX;
    uint64_t tensor_id = 0;
    std::string tensor_name;
    std::string required_by_op;
    uint64_t dependency_distance = 0;
    uint64_t byte_size = 0;
    bool already_resident = false;
};

struct PrefetchPlan {
    std::vector<PrefetchCandidate> visible_candidates;
    std::vector<PrefetchCandidate> candidates;
    uint64_t byte_budget = 0;
    uint64_t total_planned_bytes = 0;
};

class PrefetchPlanner final {
public:
    PrefetchPlan plan(
        const TensorWaveGraphView & graph,
        const TensorWavePlannerState & state,
        uint64_t max_dependency_distance,
        uint64_t byte_budget) const;
};

} // namespace vbuf_ggml
