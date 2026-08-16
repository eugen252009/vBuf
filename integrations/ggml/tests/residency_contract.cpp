#include "vbuf_residency.h"

#include <cassert>
#include <cstring>

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
    const uint64_t * dimensions = new uint64_t[2]{ 4, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, owner->data(), size };
    return { view, { owner->data(), size, 0, std::shared_ptr<const void>(owner, owner->data()) }, size };
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
    assert(cold.has_value() && std::memcmp(cold->view.payload, bytes, sizeof(bytes)) == 0);
    materializer.release(3);
    assert(store->resident_bytes() == sizeof(bytes));
    assert(fake->request_count == 1 && fake->obtain_count == 1);

    assert(materializer.state(3) == vbuf_ggml::MaterializationState::Ready);
    const auto warm = materializer.obtain_ready_tensor(3);
    assert(warm.has_value() && warm->view.payload == cold->view.payload);
    materializer.release(3);
    assert(fake->request_count == 1 && fake->obtain_count == 1);

    auto duplicate = make_tensor(bytes, sizeof(bytes));
    assert(!store->insert(3, "blk.0.ffn_down.weight", duplicate));
    auto other = make_tensor(bytes, sizeof(bytes));
    assert(!store->insert(4, "other", other));
    assert(store->resident_count() == 1);
    assert(store->acquire(3));
    assert(!store->evict(3));
    assert(store->release(3));
    assert(store->insert(4, "other", other));
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
    store->clear();
    assert(store->resident_count() == 0);
    return 0;
}
