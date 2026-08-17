#include "vbuf_topk.h"

#include <algorithm>
#include <cmath>

namespace vbuf_ggml {

bool deterministic_top_k(const std::vector<float> & scores, uint32_t expert_count,
    uint32_t k, TopKSelection * selection, std::string * error) {
    if (selection == nullptr) {
        if (error != nullptr) *error = "selection output is null";
        return false;
    }
    selection->ids.clear();
    selection->scores.clear();
    if (scores.size() != expert_count) {
        if (error != nullptr) *error = "score-count and expert-count mismatch";
        return false;
    }
    if (k == 0 || k > expert_count) {
        if (error != nullptr) *error = "top-k is outside the expert-count bounds";
        return false;
    }
    for (float score : scores) {
        if (!std::isfinite(score)) {
            if (error != nullptr) *error = "non-finite router score";
            return false;
        }
    }

    std::vector<uint32_t> order(expert_count);
    for (uint32_t id = 0; id < expert_count; ++id) order[id] = id;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (scores[lhs] != scores[rhs]) return scores[lhs] > scores[rhs];
        return lhs < rhs;
    });
    selection->ids.assign(order.begin(), order.begin() + k);
    selection->scores.reserve(k);
    for (uint32_t id : selection->ids) selection->scores.push_back(scores[id]);
    return true;
}

} // namespace vbuf_ggml
