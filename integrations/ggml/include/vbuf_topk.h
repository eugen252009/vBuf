#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vbuf_ggml {

struct TopKSelection {
    std::vector<uint32_t> ids;
    std::vector<float> scores;
};

// Selects descending scores; equal scores are ordered by lower expert ID.
// Invalid dimensions, k, or non-finite scores fail closed.
bool deterministic_top_k(const std::vector<float> & scores, uint32_t expert_count,
    uint32_t k, TopKSelection * selection, std::string * error = nullptr);

} // namespace vbuf_ggml
