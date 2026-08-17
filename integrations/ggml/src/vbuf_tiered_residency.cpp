#include "vbuf_tiered_residency.h"

#include <algorithm>

namespace vbuf_ggml {

const char * residency_tier_name(ResidencyTier tier) {
    switch (tier) {
    case ResidencyTier::Backing: return "BACKING";
    case ResidencyTier::Warm: return "WARM";
    case ResidencyTier::Hot: return "HOT";
    }
    return "UNKNOWN";
}

const char * residency_generation_name(ResidencyGeneration generation) {
    switch (generation) {
    case ResidencyGeneration::New: return "NEW";
    case ResidencyGeneration::Young: return "YOUNG";
    case ResidencyGeneration::Old: return "OLD";
    }
    return "UNKNOWN";
}

TieredResidencyStore::TieredResidencyStore(uint64_t hot_capacity_bytes,
    uint64_t warm_capacity_bytes, uint32_t old_after_reuses)
    : hot_capacity_bytes_(hot_capacity_bytes), warm_capacity_bytes_(warm_capacity_bytes),
      old_after_reuses_(old_after_reuses) {}

uint64_t TieredResidencyStore::tier_bytes(ResidencyTier tier) const {
    return tier == ResidencyTier::Hot ? hot_bytes_ : tier == ResidencyTier::Warm ? warm_bytes_ : 0;
}

uint64_t TieredResidencyStore::capacity(ResidencyTier tier) const {
    return tier == ResidencyTier::Hot ? hot_capacity_bytes_ :
        tier == ResidencyTier::Warm ? warm_capacity_bytes_ : 0;
}

const TieredResidencyEntry * TieredResidencyStore::peek(uint32_t tensor_ref) const {
    const auto it = entries_.find(tensor_ref);
    return it == entries_.end() ? nullptr : &it->second;
}

void TieredResidencyStore::add_event(const TieredResidencyEntry & entry, ResidencyTier from,
    ResidencyTier to, uint64_t epoch, const char * reason) {
    trace_.push_back({ trace_.size(), entry.tensor_ref, entry.bytes, from, to,
        entry.generation, epoch, reason });
}

bool TieredResidencyStore::make_room(ResidencyTier tier, uint64_t bytes, uint64_t epoch) {
    while (tier_bytes(tier) + bytes > capacity(tier)) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.tier != tier || it->second.active_leases != 0) continue;
            if (victim == entries_.end() || it->second.last_use < victim->second.last_use ||
                (it->second.last_use == victim->second.last_use && it->first < victim->first))
                victim = it;
        }
        if (victim == entries_.end()) return false;
        const uint32_t ref = victim->first;
        if (tier == ResidencyTier::Hot && warm_capacity_bytes_ >= victim->second.bytes &&
            make_room(ResidencyTier::Warm, victim->second.bytes, epoch)) {
            if (!move(ref, ResidencyTier::Warm, epoch, "hot_capacity")) return false;
        } else {
            if (!drop(ref, epoch, tier == ResidencyTier::Hot ? "hot_capacity" : "warm_capacity")) return false;
        }
    }
    return true;
}

bool TieredResidencyStore::admit(uint32_t tensor_ref, uint64_t bytes, ResidencyTier target,
    uint64_t epoch, const char * reason) {
    if (target == ResidencyTier::Backing || bytes > capacity(target) || entries_.count(tensor_ref) != 0)
        return false;
    if (!make_room(target, bytes, epoch)) return false;
    TieredResidencyEntry entry{ tensor_ref, bytes, target, ResidencyGeneration::New, 0, 1,
        ++clock_, epoch, epoch };
    entries_.emplace(tensor_ref, entry);
    if (target == ResidencyTier::Hot) hot_bytes_ += bytes; else warm_bytes_ += bytes;
    add_event(entry, ResidencyTier::Backing, target, epoch, reason);
    return true;
}

bool TieredResidencyStore::move(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch,
    const char * reason) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases != 0 || target == ResidencyTier::Backing)
        return false;
    const ResidencyTier from = it->second.tier;
    if (from == target) return true;
    if (!make_room(target, it->second.bytes, epoch)) return false;
    const TieredResidencyEntry entry = it->second;
    if (from == ResidencyTier::Hot) hot_bytes_ -= entry.bytes; else warm_bytes_ -= entry.bytes;
    if (target == ResidencyTier::Hot) hot_bytes_ += entry.bytes; else warm_bytes_ += entry.bytes;
    it->second.tier = target;
    if (from == ResidencyTier::Warm && target == ResidencyTier::Hot) promotion_bytes_ += entry.bytes;
    if (from == ResidencyTier::Hot && target == ResidencyTier::Warm) demotion_bytes_ += entry.bytes;
    add_event(it->second, from, target, epoch, reason);
    return true;
}

bool TieredResidencyStore::promote(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch,
    const char * reason) {
    return target == ResidencyTier::Hot && move(tensor_ref, target, epoch, reason);
}

bool TieredResidencyStore::demote(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch,
    const char * reason) {
    return target == ResidencyTier::Warm && move(tensor_ref, target, epoch, reason);
}

bool TieredResidencyStore::drop(uint32_t tensor_ref, uint64_t epoch, const char * reason) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases != 0) return false;
    const TieredResidencyEntry entry = it->second;
    if (entry.tier == ResidencyTier::Hot) hot_bytes_ -= entry.bytes; else warm_bytes_ -= entry.bytes;
    add_event(entry, entry.tier, ResidencyTier::Backing, epoch, reason);
    entries_.erase(it);
    return true;
}

bool TieredResidencyStore::request(uint32_t tensor_ref, uint64_t bytes, uint64_t epoch) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) {
        ++backing_misses_;
        return admit(tensor_ref, bytes, ResidencyTier::Warm, epoch, "backing_miss");
    }
    TieredResidencyEntry & entry = it->second;
    ++entry.request_count;
    entry.last_use = ++clock_;
    entry.last_epoch = epoch;
    if (entry.request_count == 2) entry.generation = ResidencyGeneration::Young;
    if (entry.request_count >= old_after_reuses_ + 1) entry.generation = ResidencyGeneration::Old;
    if (entry.tier == ResidencyTier::Hot) ++hot_hits_; else ++warm_hits_;
    add_event(entry, entry.tier, entry.tier, epoch, entry.tier == ResidencyTier::Hot ? "hot_hit" : "warm_hit");
    return true;
}

bool TieredResidencyStore::acquire(uint32_t tensor_ref) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) return false;
    ++it->second.active_leases;
    return true;
}

bool TieredResidencyStore::release(uint32_t tensor_ref) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases == 0) return false;
    --it->second.active_leases;
    return true;
}

} // namespace vbuf_ggml
