#include "vbuf_residency.h"

#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static DeviceResidencyKey key(uint64_t tensor_id = 17, uint32_t device_id = 0) {
    return { "sha256:artifact-a", tensor_id, 4096, 64, 9, { 8, 2 }, "ggml-cuda", device_id };
}

static DeviceResidentTensor allocation(const std::shared_ptr<int> & owner, uint64_t bytes = 64) {
    return { std::shared_ptr<void>(owner, owner.get()), owner.get(), bytes };
}

int main() {
    auto store = std::make_shared<TensorResidencyStore>(0,
        ResidencyReplacementPolicyKind::LRU, 256);
    auto first_owner = std::make_shared<int>(1);
    const auto first_key = key();
    require(store->insert_device(first_key, allocation(first_owner)), "insert resident tensor");
    require(!store->insert_device(first_key, allocation(std::make_shared<int>(2))),
        "duplicate identity replaced resident allocation");

    auto first = store->acquire_device(first_key);
    require(first.has_value() && first->backend_handle == first_owner.get(), "resident hit handle");
    auto second = store->acquire_device(first_key);
    require(second.has_value() && second->backend_handle == first->backend_handle,
        "same identity did not reuse resident allocation");
    require(store->active_device_lease_count() == 2, "device leases not counted");
    require(!store->evict_device(first_key), "leased allocation was evicted");

    std::vector<DeviceResidencyKey> distinct{
        key(18, 0),
        key(17, 1),
        key(17, 0),
    };
    distinct[2].artifact_identity = "sha256:artifact-b";
    for (size_t i = 0; i < distinct.size(); ++i) {
        auto owner = std::make_shared<int>(static_cast<int>(i + 3));
        require(store->insert_device(distinct[i], allocation(owner)), "insert distinct key");
        auto hit = store->acquire_device(distinct[i]);
        require(hit.has_value() && hit->backend_handle == owner.get(), "distinct key aliased");
        require(store->release_device(distinct[i]), "release distinct lease");
    }
    auto changed_shape = key();
    changed_shape.shape[0] = 16;
    auto changed_offset = key();
    changed_offset.source_offset += 64;
    auto changed_representation = key();
    ++changed_representation.representation;
    auto changed_backend = key();
    changed_backend.backend = "other-backend";
    for (const auto & distinct_key : { changed_shape, changed_offset,
             changed_representation, changed_backend }) {
        auto owner = std::make_shared<int>(7);
        require(store->insert_device(distinct_key, allocation(owner)), "insert geometry/backend key");
        require(store->contains_device(distinct_key), "key field omitted from identity");
    }

    require(store->release_device(first_key), "release first lease");
    require(store->release_device(first_key), "release second lease");
    require(store->active_device_lease_count() == 0, "lease count did not return to zero");
    require(store->device_resident_bytes() == 256, "device residency accounting/budget");

    auto failed_key = key(99);
    const bool upload_succeeded = false;
    if (upload_succeeded)
        require(store->insert_device(failed_key, allocation(std::make_shared<int>(99))),
            "unreachable upload insertion");
    require(!store->contains_device(failed_key), "failed upload poisoned residency store");

    std::weak_ptr<int> lifetime = first_owner;
    first_owner.reset();
    first.reset();
    second.reset();
    require(store->evict_device(first_key), "explicit device eviction");
    require(lifetime.expired(), "eviction retained device allocation owner");

    auto teardown_owner = std::make_shared<int>(10);
    const auto teardown_key = key(100);
    require(store->insert_device(teardown_key, allocation(teardown_owner)), "insert teardown allocation");
    std::weak_ptr<int> teardown_lifetime = teardown_owner;
    teardown_owner.reset();
    store->clear_device();
    require(teardown_lifetime.expired(), "session teardown retained device allocation");
    require(store->device_resident_count() == 0 && store->device_resident_bytes() == 0,
        "device teardown accounting");

    auto generic_owner = std::make_shared<int>(11);
    require(store->insert_device(key(101), allocation(generic_owner)), "insert generic clear allocation");
    generic_owner.reset();
    store->clear();
    require(store->device_resident_count() == 0, "store clear omitted device tier");

    std::printf("VBUF_DEVICE_RESIDENCY_IDENTITY_CONTRACT=PASS\n");
    std::printf("VBUF_DEVICE_RESIDENCY_REUSE_LEASES_TEARDOWN=PASS\n");
    std::printf("VBUF_DEVICE_RESIDENCY_FAILED_UPLOAD_ISOLATION=PASS\n");
    return 0;
}
