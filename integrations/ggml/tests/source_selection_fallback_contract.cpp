#include "vbuf_materializer.h"
#include "vbuf_source_selection.h"

#include <cassert>
#include <cstring>
#include <memory>
#include <vector>

namespace {

class CountingSource final : public vbuf_ggml::RangeSource {
public:
    CountingSource(const uint8_t * bytes, uint64_t size, const char * id, bool fail)
        : bytes_(bytes), size_(size), id_(id), fail_(fail) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        vbuf_ggml::RangeReadResult * result) override {
        ++reads;
        if (result != nullptr) {
            result->requested_offset = offset;
            result->requested_length = length;
            result->source_id = id_;
        }
        if (fail_ || offset > size_ || length > size_ - offset) return false;
        std::memcpy(destination, bytes_ + offset, length);
        if (result != nullptr) {
            result->returned_bytes = length;
            result->status_code = 200;
        }
        return true;
    }

    uint32_t reads = 0;

private:
    const uint8_t * bytes_;
    uint64_t size_;
    const char * id_;
    bool fail_;
};

} // namespace

int main() {
    using namespace vbuf_ggml;
    const uint8_t artifact[] = { 3, 1, 4, 1, 5, 9, 2, 6 };
    auto primary = std::make_shared<CountingSource>(artifact, sizeof(artifact), "primary", true);
    auto fallback = std::make_shared<CountingSource>(artifact, sizeof(artifact), "fallback", false);

    SourceSelectionPolicy policy;
    const std::vector<SourceDescriptor> tied = {
        { "z-source", "test", true, true, 100, 10, primary },
        { "a-source", "test", true, true, 100, 10, fallback },
    };
    const SourceSelectionResult selected = policy.select(tied, sizeof(artifact));
    assert(selected.selected_source && *selected.selected_source == 1);
    assert(selected.fallback_source && *selected.fallback_source == 0);

    const uint64_t dimensions[] = { sizeof(artifact), 1 };
    const VbufTensorView view{ 0, 2, dimensions, artifact, sizeof(artifact) };
    const PersistentTensorRef tensor{ 7, "fallback-test", view, 0 };
    LocalVbufRangeMaterializer materializer(primary, fallback);
    assert(materializer.request(0, tensor, sizeof(artifact)));
    assert(materializer.wait(0) == MaterializationState::Ready);
    const auto ready = materializer.obtain_ready_tensor(0);
    assert(ready.has_value());
    assert(std::memcmp(ready->view().payload, artifact, sizeof(artifact)) == 0);
    assert(primary->reads == 1);
    assert(fallback->reads == 1);
    materializer.release(0);
    assert(materializer.active_ready_bytes() == 0);
    return 0;
}
