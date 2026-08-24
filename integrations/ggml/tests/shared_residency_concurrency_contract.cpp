#include "vbuf_residency.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#undef assert
#define assert(condition) do { if (!(condition)) { std::fprintf(stderr, "shared residency assertion failed at line %d: %s\n", __LINE__, #condition); std::abort(); } } while (false)

namespace {

class StaticMaterializer final : public vbuf_ggml::TensorMaterializer {
public:
    explicit StaticMaterializer(vbuf_ggml::MaterializedTensor value) : value_(std::move(value)) {}

    bool request(uint32_t ref, const vbuf_ggml::PersistentTensorRef &, uint64_t) override {
        requested_.store(ref);
        ++request_count;
        return true;
    }
    vbuf_ggml::MaterializationState state(uint32_t ref) const override {
        return requested_.load() == ref ? vbuf_ggml::MaterializationState::Ready :
            vbuf_ggml::MaterializationState::NotRequested;
    }
    vbuf_ggml::MaterializationState wait(uint32_t) override {
        return vbuf_ggml::MaterializationState::Ready;
    }
    std::optional<vbuf_ggml::MaterializedTensor> obtain_ready_tensor(uint32_t) override {
        ++obtain_count;
        return value_;
    }
    void release(uint32_t) override { ++release_count; }
    uint64_t active_inflight_bytes() const override { return 0; }
    uint64_t active_ready_bytes() const override { return 0; }
    std::vector<vbuf_ggml::MaterializationTraceEvent> trace() const override { return {}; }

    std::atomic<uint32_t> request_count{0};
    std::atomic<uint32_t> obtain_count{0};
    std::atomic<uint32_t> release_count{0};

private:
    vbuf_ggml::MaterializedTensor value_;
    std::atomic<uint32_t> requested_{UINT32_MAX};
};

vbuf_ggml::MaterializedTensor make_tensor(uint8_t value, size_t size = 4) {
    auto owner = std::make_shared<std::vector<uint8_t>>(size, value);
    const uint64_t dimensions[] = { size, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, owner->data(), size };
    vbuf_ggml::MaterializedTensor result{};
    assert(result.assign_view(view));
    result.storage = { owner->data(), size, 0, std::shared_ptr<const void>(owner, owner->data()) };
    result.bytes = size;
    return result;
}

vbuf_ggml::PersistentTensorRef make_ref(uint32_t id, const char * name, uint64_t offset) {
    static const uint64_t dimensions[] = { 4, 1 };
    static const uint8_t payload[] = { 0, 0, 0, 0 };
    return { id, name, { 0, 2, dimensions, payload, sizeof(payload) }, offset };
}

std::optional<vbuf_ggml::MaterializedTensor> obtain(
    vbuf_ggml::ResidentTensorMaterializer * materializer,
    uint32_t ref, const vbuf_ggml::PersistentTensorRef & tensor) {
    assert(materializer->request(ref, tensor, tensor.view.payload_len));
    assert(materializer->wait(ref) == vbuf_ggml::MaterializationState::Ready);
    return materializer->obtain_ready_tensor(ref);
}

} // namespace

int main() {
    const auto tensor = make_ref(7, "shared.tensor", 100);
    auto store = std::make_shared<vbuf_ggml::TensorResidencyStore>(8);
    auto backing_a = std::make_shared<StaticMaterializer>(make_tensor(0x2a));
    auto backing_b = std::make_shared<StaticMaterializer>(make_tensor(0x2a));
    vbuf_ggml::ResidentTensorMaterializer materializer_a(backing_a, store);
    vbuf_ggml::ResidentTensorMaterializer materializer_b(backing_b, store);

    const auto warm = obtain(&materializer_a, 7, tensor);
    assert(warm.has_value());
    const auto reader_b = obtain(&materializer_b, 7, tensor);
    assert(reader_b.has_value());
    assert(warm->payload == reader_b->payload);
    assert(store->active_lease_count() == 2);
    assert(std::memcmp(reader_b->payload, warm->payload, warm->payload_len) == 0);

    materializer_a.release(7);
    assert(store->active_lease_count() == 1);
    assert(std::memcmp(reader_b->payload, warm->payload, warm->payload_len) == 0);
    assert(!store->evict(7));

    const auto other = make_tensor(0x3b);
    assert(store->insert(8, "pressure.one", other));
    assert(store->contains(7));
    assert(store->active_lease_count() == 1);
    materializer_b.release(7);
    assert(store->active_lease_count() == 0);
    assert(store->evict(7));

    const auto rewarm = obtain(&materializer_a, 7, tensor);
    assert(rewarm.has_value());
    assert(std::memcmp(rewarm->payload, warm->payload, warm->payload_len) == 0);
    materializer_a.release(7);

    auto cold_store = std::make_shared<vbuf_ggml::TensorResidencyStore>(8);
    auto cold_backing_a = std::make_shared<StaticMaterializer>(make_tensor(0x4c));
    auto cold_backing_b = std::make_shared<StaticMaterializer>(make_tensor(0x4c));
    vbuf_ggml::ResidentTensorMaterializer cold_a(cold_backing_a, cold_store);
    vbuf_ggml::ResidentTensorMaterializer cold_b(cold_backing_b, cold_store);
    std::optional<vbuf_ggml::MaterializedTensor> cold_result_a;
    std::optional<vbuf_ggml::MaterializedTensor> cold_result_b;
    std::thread cold_thread_a([&] { cold_result_a = obtain(&cold_a, 9, make_ref(9, "cold", 200)); });
    std::thread cold_thread_b([&] { cold_result_b = obtain(&cold_b, 9, make_ref(9, "cold", 200)); });
    cold_thread_a.join();
    cold_thread_b.join();
    assert(cold_result_a.has_value() && cold_result_b.has_value());
    assert(cold_result_a->payload == cold_result_b->payload);
    assert(std::memcmp(cold_result_a->payload, cold_result_b->payload, cold_result_a->payload_len) == 0);
    assert(cold_store->active_lease_count() == 2);
    assert(cold_store->materialization_count() >= 1);
    assert(cold_store->materialization_count() <= 2);
    cold_a.release(9);
    assert(cold_store->active_lease_count() == 1);
    assert(std::memcmp(cold_result_b->payload, cold_result_a->payload, cold_result_a->payload_len) == 0);
    cold_b.release(9);
    assert(cold_store->active_lease_count() == 0);

    auto different_store = std::make_shared<vbuf_ggml::TensorResidencyStore>(8);
    auto different_backing_a = std::make_shared<StaticMaterializer>(make_tensor(0x5d));
    auto different_backing_b = std::make_shared<StaticMaterializer>(make_tensor(0x6e));
    vbuf_ggml::ResidentTensorMaterializer different_a(different_backing_a, different_store);
    vbuf_ggml::ResidentTensorMaterializer different_b(different_backing_b, different_store);
    std::optional<vbuf_ggml::MaterializedTensor> different_result_a;
    std::optional<vbuf_ggml::MaterializedTensor> different_result_b;
    std::thread different_thread_a([&] {
        different_result_a = obtain(&different_a, 11, make_ref(11, "different.a", 300));
    });
    std::thread different_thread_b([&] {
        different_result_b = obtain(&different_b, 12, make_ref(12, "different.b", 400));
    });
    different_thread_a.join();
    different_thread_b.join();
    assert(different_result_a.has_value() && different_result_b.has_value());
    assert(different_result_a->payload != different_result_b->payload);
    assert(different_result_a->payload[0] == 0x5d);
    assert(different_result_b->payload[0] == 0x6e);
    assert(different_store->active_lease_count() == 2);
    different_a.release(11);
    different_b.release(12);
    assert(different_store->active_lease_count() == 0);

    std::printf("VBUF_SHARED_RESIDENCY_CONCURRENCY_CONTRACT=PASS\n");
    std::printf("DUAL_READER_RELEASE_ORDER=PASS\n");
    std::printf("EVICTION_PRESSURE_WITH_READER=PASS\n");
    std::printf("COLD_SAME_KEY_DUPLICATE_BUT_CORRECT=PASS\n");
    std::printf("COLD_DIFFERENT_KEY_CONCURRENT=PASS\n");
    return 0;
}
