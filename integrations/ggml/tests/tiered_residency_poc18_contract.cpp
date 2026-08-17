#include "vbuf_tiered_residency.h"

#include <cassert>

using namespace vbuf_ggml;

int main() {
    TieredResidencyStore store(8, 32, 2);
    assert(store.request(7, 4, 0));
    const TieredResidencyEntry * entry = store.peek(7);
    assert(entry != nullptr && entry->tier == ResidencyTier::Warm &&
        entry->generation == ResidencyGeneration::New);
    assert(store.acquire(7));
    assert(!store.promote(7, ResidencyTier::Hot, 0, "leased"));
    assert(store.release(7));
    assert(store.request(7, 4, 1));
    assert(store.peek(7)->generation == ResidencyGeneration::Young);
    assert(store.request(7, 4, 2));
    assert(store.peek(7)->generation == ResidencyGeneration::Old);
    assert(store.promote(7, ResidencyTier::Hot, 2, "old_reuse"));
    assert(store.peek(7)->tier == ResidencyTier::Hot);
    assert(store.tier_bytes(ResidencyTier::Warm) == 0);
    assert(store.tier_bytes(ResidencyTier::Hot) == 4);
    assert(store.hot_hits() == 0 && store.warm_hits() == 2 && store.backing_misses() == 1);
    assert(store.promotion_bytes() == 4 && store.demotion_bytes() == 0);
    assert(store.demote(7, ResidencyTier::Warm, 3, "hot_pressure"));
    assert(store.peek(7)->tier == ResidencyTier::Warm);
    assert(store.demotion_bytes() == 4);
    assert(store.drop(7, 4, "teardown"));
    assert(store.peek(7) == nullptr && store.tier_bytes(ResidencyTier::Hot) == 0 &&
        store.tier_bytes(ResidencyTier::Warm) == 0);
    bool saw_promotion = false, saw_demotion = false;
    for (const auto & event : store.trace()) {
        saw_promotion = saw_promotion || event.from == ResidencyTier::Warm &&
            event.to == ResidencyTier::Hot;
        saw_demotion = saw_demotion || event.from == ResidencyTier::Hot &&
            event.to == ResidencyTier::Warm;
    }
    assert(saw_promotion && saw_demotion);
    TieredResidencyStore failed_promotion(2, 8, 2);
    assert(failed_promotion.request(11, 4, 0));
    assert(failed_promotion.request(11, 4, 1));
    assert(failed_promotion.request(11, 4, 2));
    assert(!failed_promotion.promote(11, ResidencyTier::Hot, 2, "insufficient_hot_capacity"));
    assert(failed_promotion.peek(11) != nullptr &&
        failed_promotion.peek(11)->tier == ResidencyTier::Warm &&
        failed_promotion.tier_bytes(ResidencyTier::Warm) == 4);
    return 0;
}
