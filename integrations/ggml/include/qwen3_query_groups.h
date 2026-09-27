#pragma once

#include "softmax_compute_extent.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vbuf_ggml {

struct Qwen3QueryGroup {
    uint32_t first_query = 0;
    uint32_t query_count = 0;
    uint32_t compute_extent = 0;
};

// Consecutive causal query rows that share the same per-query canonical
// softmax extent. Query q has logical extent q+1; groups preserve input order.
inline std::vector<Qwen3QueryGroup> qwen3_query_groups(
    uint32_t sequence_length, uint32_t granularity) {
    if (sequence_length == 0 || granularity == 0)
        throw std::invalid_argument("Qwen3 query-group geometry must be nonzero");
    std::vector<Qwen3QueryGroup> groups;
    for (uint32_t query = 0; query < sequence_length; ++query) {
        const uint32_t extent = SoftmaxComputeExtent::make(query + 1, granularity).compute;
        if (groups.empty() || groups.back().compute_extent != extent)
            groups.push_back({query, 1, extent});
        else
            ++groups.back().query_count;
    }
    uint32_t next_query = 0;
    for (const auto & group : groups) {
        if (group.first_query != next_query || group.query_count == 0 ||
            group.first_query + group.query_count > sequence_length)
            throw std::logic_error("Qwen3 query groups do not form an ordered partition");
        for (uint32_t row = 0; row < group.query_count; ++row) {
            const uint32_t logical = group.first_query + row + 1;
            if (SoftmaxComputeExtent::make(logical, granularity).compute != group.compute_extent)
                throw std::logic_error("Qwen3 query assigned to incorrect compute extent");
        }
        next_query += group.query_count;
    }
    if (next_query != sequence_length)
        throw std::logic_error("Qwen3 query group lost rows");
    return groups;
}

} // namespace vbuf_ggml
