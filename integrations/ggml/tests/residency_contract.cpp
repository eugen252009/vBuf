#include "vbuf_residency.h"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <utility>
#include <vector>

#undef assert
#define assert(condition) do { if (!(condition)) std::abort(); } while (false)

namespace {

class FakeMaterializer final : public vbuf_ggml::TensorMaterializer {
public:
    explicit FakeMaterializer(vbuf_ggml::MaterializedTensor value) : value_(std::move(value)) {}

    bool request(uint32_t ref, const vbuf_ggml::PersistentTensorRef &, uint64_t) override {
        ++request_count;
        requested = ref;
        return true;
    }
    vbuf_ggml::MaterializationState state(uint32_t ref) const override {
        return ref == requested ? vbuf_ggml::MaterializationState::Ready
            : vbuf_ggml::MaterializationState::NotRequested;
    }
    vbuf_ggml::MaterializationState wait(uint32_t) override { return vbuf_ggml::MaterializationState::Ready; }
    std::optional<vbuf_ggml::MaterializedTensor> obtain_ready_tensor(uint32_t) override {
        ++obtain_count;
        return value_;
    }
    void release(uint32_t) override { ++release_count; }
    uint64_t active_inflight_bytes() const override { return 0; }
    uint64_t active_ready_bytes() const override { return 0; }
    std::vector<vbuf_ggml::MaterializationTraceEvent> trace() const override { return {}; }

    uint32_t request_count = 0;
    uint32_t obtain_count = 0;
    uint32_t release_count = 0;

private:
    vbuf_ggml::MaterializedTensor value_;
    uint32_t requested = UINT32_MAX;
};

vbuf_ggml::MaterializedTensor make_tensor(const uint8_t * bytes, size_t size) {
    auto owner = std::make_shared<std::vector<uint8_t>>(bytes, bytes + size);
    const uint64_t dimensions[] = { 4, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, owner->data(), size };
    vbuf_ggml::MaterializedTensor result{};
    assert(result.assign_view(view));
    result.storage = { owner->data(), size, 0, std::shared_ptr<const void>(owner, owner->data()) };
    result.bytes = size;
    return result;
}

void assert_geometry(const vbuf_ggml::MaterializedTensor & tensor,
    const uint8_t * payload, size_t payload_size) {
    const vbuf_ggml::VbufTensorView view = tensor.view();
    assert(view.representation == 0);
    assert(view.rank == 2 && view.dimensions[0] == 4 && view.dimensions[1] == 1);
    assert(view.payload == payload && view.payload_len == payload_size);
    assert(tensor.bytes == payload_size);
}

} // namespace

int main() {
    const uint8_t bytes[] = { 1, 2, 3, 4 };
    const uint64_t dimensions[] = { 4, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, bytes, sizeof(bytes) };
    const vbuf_ggml::PersistentTensorRef tensor{ 7, "blk.0.ffn_down.weight", view, 100 };
    auto fake = std::make_shared<FakeMaterializer>(make_tensor(bytes, sizeof(bytes)));
    auto store = std::make_shared<vbuf_ggml::TensorResidencyStore>(sizeof(bytes));
    vbuf_ggml::ResidentTensorMaterializer materializer(fake, store);

    assert(materializer.request(3, tensor, sizeof(bytes)));
    assert(materializer.wait(3) == vbuf_ggml::MaterializationState::Ready);
    const auto cold = materializer.obtain_ready_tensor(3);
    assert(cold.has_value() && std::memcmp(cold->view().payload, bytes, sizeof(bytes)) == 0);
    assert_geometry(*cold, cold->view().payload, sizeof(bytes));
    materializer.release(3);
    assert(store->resident_bytes() == sizeof(bytes));
    assert(fake->request_count == 1 && fake->obtain_count == 1);
    assert(store->materialization_count() == 1 && store->reacquisition_count() == 0);

    assert(materializer.state(3) == vbuf_ggml::MaterializationState::Ready);
    const auto warm = materializer.obtain_ready_tensor(3);
    assert(warm.has_value() && warm->view().payload == cold->view().payload);
    assert_geometry(*warm, warm->view().payload, sizeof(bytes));

    vbuf_ggml::MaterializedTensor copied = *warm;
    assert_geometry(copied, warm->view().payload, sizeof(bytes));
    vbuf_ggml::MaterializedTensor moved = std::move(copied);
    assert_geometry(moved, warm->view().payload, sizeof(bytes));
    std::vector<vbuf_ggml::MaterializedTensor> relocated;
    relocated.reserve(1);
    relocated.push_back(moved);
    relocated.push_back(*cold);
    assert_geometry(relocated[0], warm->view().payload, sizeof(bytes));
    assert_geometry(relocated[1], cold->view().payload, sizeof(bytes));
    materializer.release(3);
    assert(fake->request_count == 1 && fake->obtain_count == 1);
    assert(store->materialization_count() == 1 && store->reacquisition_count() == 0);

    auto duplicate = make_tensor(bytes, sizeof(bytes));
    assert(!store->insert(3, "blk.0.ffn_down.weight", duplicate));
    auto other = make_tensor(bytes, sizeof(bytes));
    assert(store->acquire(3));
    assert(!store->insert(4, "other", other));
    assert(store->release(3));
    assert(store->insert(4, "other", other));
    assert(store->resident_count() == 1);
    assert(store->resident_count() == 1 && store->resident_bytes() == sizeof(bytes));
    assert(store->acquire(4));
    assert(!store->evict(4));
    assert(store->release(4));
    assert(store->evict(4));
    assert(store->resident_bytes() == 0);
    assert(materializer.request(3, tensor, sizeof(bytes)));
    const auto reloaded = materializer.obtain_ready_tensor(3);
    assert(reloaded.has_value());
    materializer.release(3);
    assert(fake->request_count == 2 && fake->obtain_count == 2);
    assert(store->resident_bytes() == sizeof(bytes));
    assert(store->materialization_count() == 2 && store->reacquisition_count() == 1);
    store->clear();
    assert(store->resident_count() == 0);
    return 0;
}
