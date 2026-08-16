#include "vbuf_source_selection.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace vbuf_ggml {
namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t estimate_ns(const SourceDescriptor & source, uint64_t bytes) {
    if (source.measured_throughput_bytes_per_second == 0) return UINT64_MAX;
    const uint64_t transfer_ns = bytes > UINT64_MAX / 1000000000ULL
        ? UINT64_MAX
        : (bytes * 1000000000ULL) / source.measured_throughput_bytes_per_second;
    const uint64_t latency_ns = source.fixed_latency_us > UINT64_MAX / 1000ULL
        ? UINT64_MAX : source.fixed_latency_us * 1000ULL;
    if (transfer_ns > UINT64_MAX - latency_ns) return UINT64_MAX;
    return transfer_ns + latency_ns;
}

} // namespace

SourceSelectionResult SourceSelectionPolicy::select(
    const std::vector<SourceDescriptor> & sources, uint64_t tensor_bytes) const {
    SourceSelectionResult result;
    const uint64_t started = now_ns();
    for (uint32_t index = 0; index < sources.size(); ++index) {
        const SourceDescriptor & source = sources[index];
        SourceEstimate estimate;
        estimate.source_index = index;
        estimate.source_id = source.id;
        estimate.eligible = source.available && source.range_capable && source.source != nullptr;
        if (!estimate.eligible) {
            estimate.reason = "unavailable-or-no-exact-range-capability";
        } else if (source.measured_throughput_bytes_per_second == 0) {
            estimate.reason = "unknown-throughput-ranked-last";
        } else {
            estimate.reason = "fixed-latency-plus-bytes-over-throughput";
        }
        estimate.estimated_time_ns = estimate.eligible
            ? estimate_ns(source, tensor_bytes) : UINT64_MAX;
        result.estimates.push_back(std::move(estimate));
    }
    std::vector<uint32_t> eligible;
    for (uint32_t index = 0; index < result.estimates.size(); ++index) {
        if (result.estimates[index].eligible) eligible.push_back(index);
    }
    std::sort(eligible.begin(), eligible.end(), [&](uint32_t lhs, uint32_t rhs) {
        const SourceEstimate & left = result.estimates[lhs];
        const SourceEstimate & right = result.estimates[rhs];
        if (left.estimated_time_ns != right.estimated_time_ns) {
            return left.estimated_time_ns < right.estimated_time_ns;
        }
        if (left.source_id != right.source_id) return left.source_id < right.source_id;
        return left.source_index < right.source_index;
    });
    if (!eligible.empty()) {
        result.selected_source = eligible[0];
        if (eligible.size() > 1) result.fallback_source = eligible[1];
        result.reason = "estimated-completion-time-then-source-id";
    } else {
        result.reason = "no-eligible-source";
    }
    result.selection_duration_ns = now_ns() - started;
    return result;
}

} // namespace vbuf_ggml
