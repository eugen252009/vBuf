#include "vbuf_residency.h"
#include <stdexcept>
#include <future>
#include <cstring>
#include <thread>
#include <chrono>

using namespace vbuf_ggml;
static void require(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
int main() {
    std::vector<uint8_t> bytes(128, 7);
    uint64_t shape[] = {16};
    PersistentTensorRef tensor{1, "trace", {0, 1, shape, bytes.data(), 64}, 0};
    auto source = std::make_shared<LocalVbufRangeSource>(bytes.data(), bytes.size());
    auto local = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<TensorResidencyStore>(64);
    ResidentTensorMaterializer materializer(local, residency);
    local->set_trace_enabled(false);
    residency->set_trace_enabled(false);
    for (int i = 0; i < 2; ++i) {
        require(materializer.request(1, tensor, 64), "request");
        require(materializer.wait(1) == MaterializationState::Ready, "ready");
        require(materializer.obtain_ready_tensor(1).has_value(), "view");
        materializer.release(1);
        require(residency->evict(1), "evict");
    }
    require(local->trace().empty() && residency->trace().empty(), "disabled tracing accumulated history");
    require(local->materialized_bytes() == 128 && local->reloaded_bytes() == 64, "byte counters depend on trace");
    require(residency->materialization_count() == 2 && residency->eviction_count() == 2 &&
        residency->reacquisition_count() == 1, "residency counters depend on trace");
    local->set_trace_enabled(true);
    residency->set_trace_enabled(true);
    tensor.source_offset = 64;
    require(materializer.request(2, tensor, 64), "traced request");
    require(materializer.wait(2) == MaterializationState::Ready, "traced ready");
    require(materializer.obtain_ready_tensor(2).has_value(), "traced view");
    materializer.release(2);
    require(!local->trace().empty() && !residency->trace().empty(), "qualification trace missing");
    require(local->materialized_bytes() == 192 && local->reloaded_bytes() == 64, "physical range totals");
    require(materializer.active_inflight_bytes() == 0 && residency->active_lease_count() == 0, "cleanup");

    class BlockedSource final : public RangeSource {
    public:
        std::promise<void> gate;
        std::shared_future<void> ready = gate.get_future().share();
        bool read_range(uint64_t, uint64_t length, uint8_t * data, RangeReadResult *) override {
            ready.wait(); std::memset(data, 0, length); return true;
        }
    };
    auto blocked = std::make_shared<BlockedSource>();
    LocalVbufRangeMaterializer pending(blocked);
    require(pending.request(1, tensor, UINT64_MAX), "pending request");
    auto enormous = tensor;
    enormous.view.payload_len = UINT64_MAX;
    const bool overflow_rejected = !pending.request(2, enormous, UINT64_MAX);
    blocked->gate.set_value();
    require(pending.wait(1) == MaterializationState::Ready, "drain pending request");
    pending.release(1);
    require(overflow_rejected && pending.active_inflight_bytes() == 0, "inflight budget addition overflowed");

    class SlowResultSource final : public RangeSource {
    public:
        bool read_range(uint64_t offset, uint64_t length, uint8_t * data, RangeReadResult * result) override {
            result->source_id = std::string(128, 's');
            result->requested_offset = offset;
            result->requested_length = length;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::memset(data, 0, length);
            result->returned_bytes = length;
            result->status_code = 206;
            return true;
        }
    };
    LocalVbufRangeMaterializer traced(std::make_shared<SlowResultSource>());
    for (uint32_t i = 0; i < 8; ++i) {
        auto item = tensor; item.source_offset = i * 64;
        require(traced.request(i, item, 512), "parallel traced request");
    }
    for (uint32_t i = 0; i < 8; ++i) {
        require(traced.wait(i) == MaterializationState::Ready, "parallel traced readiness");
        traced.release(i);
    }
    for (const auto & event : traced.trace())
        if (event.event == "STATE" && event.state == MaterializationState::Ready)
            require(event.source_id == std::string(128, 's') && event.returned_bytes == 64,
                "incomplete source-result snapshot published");
}
