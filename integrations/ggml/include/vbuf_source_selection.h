#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vbuf_range_source.h"

namespace vbuf_ggml {

struct SourceDescriptor {
    std::string id;
    std::string kind;
    bool available = false;
    bool range_capable = false;
    uint64_t measured_throughput_bytes_per_second = 0;
    uint64_t fixed_latency_us = 0;
    std::shared_ptr<RangeSource> source;
};

struct SourceEstimate {
    uint32_t source_index = UINT32_MAX;
    std::string source_id;
    bool eligible = false;
    uint64_t estimated_time_ns = UINT64_MAX;
    std::string reason;
};

struct SourceSelectionResult {
    std::vector<SourceEstimate> estimates;
    std::optional<uint32_t> selected_source;
    std::optional<uint32_t> fallback_source;
    uint64_t selection_duration_ns = 0;
    std::string reason;
};

class SourceSelectionPolicy final {
public:
    SourceSelectionResult select(
        const std::vector<SourceDescriptor> & sources,
        uint64_t tensor_bytes) const;
};

} // namespace vbuf_ggml
