#include "vbuf_residency.h"

#include <cassert>
#include <memory>
#include <vector>

namespace {

vbuf_ggml::MaterializedTensor tensor(uint64_t bytes) {
    auto owner = std::make_shared<std::vector<uint8_t>>(bytes);
    const uint64_t * dimensions = new uint64_t[2]{ bytes, 1 };
    const vbuf_ggml::VbufTensorView view{ 0, 2, dimensions, owner->data(), bytes };
    return { view, { owner->data(), bytes, 0, std::shared_ptr<const void>(owner, owner->data()) }, bytes };
}

uint32_t run_trace() {
    vbuf_ggml::TensorResidencyStore store(10);
    const auto first = tensor(6), second = tensor(4), third = tensor(4);
    assert(store.insert(1, "first", first));
    assert(store.insert(2, "second", second));
    assert(store.lookup(1) != nullptr);
    assert(store.insert(3, "third", third));
    assert(store.peek(1) != nullptr && store.peek(2) == nullptr && store.peek(3) != nullptr);
    assert(store.resident_bytes() <= store.max_resident_bytes());
    return store.peek(2) == nullptr ? 2 : UINT32_MAX;
}

} // namespace

int main() {
    assert(run_trace() == 2);
    vbuf_ggml::TensorResidencyStore store(8);
    const auto value = tensor(8);
    assert(store.insert(7, "reloadable", value));
    assert(store.acquire(7));
    assert(!store.evict(7));
    assert(store.release(7));
    assert(store.evict(7));
    assert(store.insert(7, "reloadable", value));
    size_t inserts = 0, evictions = 0;
    for (const auto & event : store.trace()) {
        if (event.kind == vbuf_ggml::ResidencyEventKind::Insert) ++inserts;
        if (event.kind == vbuf_ggml::ResidencyEventKind::Evict) ++evictions;
    }
    assert(inserts == 2 && evictions == 1);
    assert(run_trace() == 2);
    return 0;
}
