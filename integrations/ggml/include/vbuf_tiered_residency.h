#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vbuf_ggml {

enum class ResidencyTier { Backing, Warm, Hot };
enum class ResidencyGeneration { New, Young, Old };

struct TieredResidencyEntry {
    uint32_t tensor_ref = UINT32_MAX;
    uint64_t bytes = 0;
    ResidencyTier tier = ResidencyTier::Backing;
    ResidencyGeneration generation = ResidencyGeneration::New;
    uint32_t active_leases = 0;
    uint32_t request_count = 0;
    uint64_t last_use = 0;
    uint64_t first_epoch = 0;
    uint64_t last_epoch = 0;
};

struct TieredResidencyEvent {
    uint64_t ordinal = 0;
    uint32_t tensor_ref = UINT32_MAX;
    uint64_t bytes = 0;
    ResidencyTier from = ResidencyTier::Backing;
    ResidencyTier to = ResidencyTier::Backing;
    ResidencyGeneration generation = ResidencyGeneration::New;
    uint64_t epoch = 0;
    const char * reason = "";
};

class TieredResidencyStore final {
public:
    TieredResidencyStore(uint64_t hot_capacity_bytes, uint64_t warm_capacity_bytes,
        uint32_t old_after_reuses = 2);

    bool request(uint32_t tensor_ref, uint64_t bytes, uint64_t epoch);
    bool acquire(uint32_t tensor_ref);
    bool release(uint32_t tensor_ref);
    bool promote(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch,
        const char * reason = "policy");
    bool demote(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch,
        const char * reason = "capacity");
    bool admit(uint32_t tensor_ref, uint64_t bytes, ResidencyTier target, uint64_t epoch,
        const char * reason = "backing");
    bool drop(uint32_t tensor_ref, uint64_t epoch, const char * reason = "capacity");

    const TieredResidencyEntry * peek(uint32_t tensor_ref) const;
    uint64_t tier_bytes(ResidencyTier tier) const;
    uint64_t capacity(ResidencyTier tier) const;
    uint64_t hot_hits() const { return hot_hits_; }
    uint64_t warm_hits() const { return warm_hits_; }
    uint64_t backing_misses() const { return backing_misses_; }
    uint64_t promotion_bytes() const { return promotion_bytes_; }
    uint64_t demotion_bytes() const { return demotion_bytes_; }
    uint64_t duplicate_payload_bytes() const { return duplicate_payload_bytes_; }
    const std::vector<TieredResidencyEvent> & trace() const { return trace_; }

private:
    bool move(uint32_t tensor_ref, ResidencyTier target, uint64_t epoch, const char * reason);
    bool make_room(ResidencyTier tier, uint64_t bytes, uint64_t epoch);
    void add_event(const TieredResidencyEntry & entry, ResidencyTier from,
        ResidencyTier to, uint64_t epoch, const char * reason);

    uint64_t hot_capacity_bytes_ = 0;
    uint64_t warm_capacity_bytes_ = 0;
    uint32_t old_after_reuses_ = 2;
    uint64_t hot_bytes_ = 0;
    uint64_t warm_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t hot_hits_ = 0;
    uint64_t warm_hits_ = 0;
    uint64_t backing_misses_ = 0;
    uint64_t promotion_bytes_ = 0;
    uint64_t demotion_bytes_ = 0;
    uint64_t duplicate_payload_bytes_ = 0;
    std::unordered_map<uint32_t, TieredResidencyEntry> entries_;
    std::vector<TieredResidencyEvent> trace_;
};

const char * residency_tier_name(ResidencyTier tier);
const char * residency_generation_name(ResidencyGeneration generation);

} // namespace vbuf_ggml
