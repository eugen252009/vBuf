#include "vbuf_source_selection.h"

#include <cassert>
#include <memory>

namespace {

class StubSource final : public vbuf_ggml::RangeSource {
public:
    explicit StubSource(const char * name) : name_(name) {}

    bool read_range(uint64_t, uint64_t length, uint8_t * destination,
        vbuf_ggml::RangeReadResult * result) override {
        result->returned_bytes = length;
        result->status_code = 200;
        result->source_id = name_;
        return true;
    }

private:
    const char * name_;
};

} // namespace

int main() {
    using namespace vbuf_ggml;
    auto local = std::make_shared<StubSource>("local");
    auto remote = std::make_shared<StubSource>("remote");
    SourceSelectionPolicy policy;

    std::vector<SourceDescriptor> sources = {
        { "remote", "http", true, true, 100, 1000, remote },
        { "local", "file", true, true, 200, 10, local },
        { "unknown", "other", true, true, 0, 0, local },
        { "disabled", "file", false, true, 1000, 0, local },
    };
    SourceSelectionResult result = policy.select(sources, 1000);
    assert(result.selected_source && *result.selected_source == 1);
    assert(result.fallback_source && *result.fallback_source == 0);
    assert(result.estimates[2].estimated_time_ns == UINT64_MAX);
    assert(!result.estimates[3].eligible);

    sources[1].measured_throughput_bytes_per_second = 0;
    sources[0].measured_throughput_bytes_per_second = 0;
    result = policy.select(sources, 1000);
    assert(result.selected_source && *result.selected_source == 0);
    assert(result.fallback_source && *result.fallback_source == 2);

    sources[0].available = false;
    sources[2].available = false;
    result = policy.select(sources, 1000);
    assert(result.selected_source && *result.selected_source == 1);
    assert(!result.fallback_source);

    return 0;
}
