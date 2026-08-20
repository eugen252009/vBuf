#include "vbuf_striped_materializer.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>

namespace {

class FakeRangeSource final : public vbuf_ggml::RangeSource {
public:
    FakeRangeSource(std::vector<uint8_t> payload, uint64_t base, bool fail = false)
        : payload_(std::move(payload)), base_(base), fail_(fail) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        vbuf_ggml::RangeReadResult * result) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        result->requested_offset = offset;
        result->requested_length = length;
        result->first_byte_timestamp_ns = 1;
        result->source_id = fail_ ? "failed" : "fake";
        if (fail_ || offset < base_ || offset - base_ + length > payload_.size()) {
            result->error = "controlled failure";
            return false;
        }
        std::memcpy(destination, payload_.data() + (offset - base_), length);
        result->returned_bytes = length;
        result->status_code = 206;
        return true;
    }

private:
    std::vector<uint8_t> payload_;
    uint64_t base_;
    bool fail_;
};

} // namespace

int main() {
    constexpr uint64_t base = 1000;
    constexpr uint64_t length = 1024;
    std::vector<uint8_t> payload(length);
    for (uint64_t index = 0; index < length; ++index) payload[index] = static_cast<uint8_t>(index);
    auto source_a = std::make_shared<FakeRangeSource>(payload, base);
    auto source_b = std::make_shared<FakeRangeSource>(payload, base);
    const auto plan = vbuf_ggml::RangeStripingPlan::two_way(base, length, source_a, source_b, 32);
    assert(plan.has_value());
    assert(plan->stripes[0].destination_offset == 0);
    assert(plan->stripes[0].length + plan->stripes[1].length == length);
    assert(plan->stripes[0].source_offset + plan->stripes[0].length ==
        plan->stripes[1].source_offset);
    assert(plan->validate());

    const uint64_t dimensions[] = { 4, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, nullptr, length };
    const vbuf_ggml::PersistentTensorRef tensor{ 7, "tensor", view, base };
    vbuf_ggml::ParallelRangeMaterializer materializer(*plan);
    assert(materializer.request(3, tensor, length));
    assert(materializer.wait(3) == vbuf_ggml::MaterializationState::Ready);
    const auto ready = materializer.obtain_ready_tensor(3);
    assert(ready.has_value());
    const vbuf_ggml::VbufTensorView ready_view = ready->view();
    assert(ready_view.rank == 2 && ready_view.dimensions[0] == 4 && ready_view.dimensions[1] == 1);
    assert(std::memcmp(ready_view.payload, payload.data(), length) == 0);
    assert(materializer.stripe_trace().size() == 2);
    const auto & a = materializer.stripe_trace()[0];
    const auto & b = materializer.stripe_trace()[1];
    assert(std::max(a.request_start_ns, b.request_start_ns) <
        std::min(a.complete_ns, b.complete_ns));
    assert(a.returned_bytes + b.returned_bytes == length);
    materializer.release(3);
    assert(materializer.active_inflight_bytes() == 0);
    assert(materializer.active_ready_bytes() == 0);

    auto failed_b = std::make_shared<FakeRangeSource>(payload, base, true);
    const auto failed_plan = vbuf_ggml::RangeStripingPlan::two_way(
        base, length, source_a, failed_b, 32);
    assert(failed_plan.has_value());
    vbuf_ggml::ParallelRangeMaterializer failed(*failed_plan);
    assert(failed.request(3, tensor, length));
    assert(failed.wait(3) == vbuf_ggml::MaterializationState::Failed);
    assert(!failed.obtain_ready_tensor(3).has_value());

    auto failed_a = std::make_shared<FakeRangeSource>(payload, base, true);
    const auto both_failed_plan = vbuf_ggml::RangeStripingPlan::two_way(
        base, length, failed_a, failed_b, 32);
    assert(both_failed_plan.has_value());
    vbuf_ggml::ParallelRangeMaterializer both_failed(*both_failed_plan);
    assert(both_failed.request(3, tensor, length));
    assert(both_failed.wait(3) == vbuf_ggml::MaterializationState::Failed);
    assert(!both_failed.obtain_ready_tensor(3).has_value());
    return 0;
}
