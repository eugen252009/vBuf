#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

// Backend/kernel policy supplies the qualified granularity. This value type
// keeps logical attention semantics separate from the physical softmax extent.
struct SoftmaxComputeExtent {
    uint32_t logical = 0;
    uint32_t compute = 0;
    uint32_t granularity = 0;

    static SoftmaxComputeExtent make(uint32_t logical_extent, uint32_t kernel_granularity) {
        if (logical_extent == 0 || kernel_granularity == 0)
            throw std::invalid_argument("softmax logical extent and granularity must be nonzero");
        const uint64_t rounded = ((static_cast<uint64_t>(logical_extent) + kernel_granularity - 1) /
            kernel_granularity) * kernel_granularity;
        if (rounded > std::numeric_limits<uint32_t>::max())
            throw std::length_error("softmax compute extent overflow");
        const uint32_t compute_extent = static_cast<uint32_t>(rounded);
        if (compute_extent < logical_extent || compute_extent % kernel_granularity != 0)
            throw std::logic_error("invalid rounded softmax compute extent");
        return {logical_extent, compute_extent, kernel_granularity};
    }

    uint32_t padding() const noexcept { return compute - logical; }

    // For one query, `real_scores` contains exactly the causally visible
    // logical scores. All synthetic score slots are semantically impossible.
    std::vector<float> padded_scores(const std::vector<float> & real_scores) const {
        if (real_scores.size() != logical)
            throw std::invalid_argument("softmax score count differs from logical extent");
        std::vector<float> result(compute, -std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < real_scores.size(); ++i) result[i] = real_scores[i];
        return result;
    }
};

} // namespace vbuf_ggml
