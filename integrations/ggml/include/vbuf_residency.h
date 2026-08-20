#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vbuf_materializer.h"

namespace vbuf_ggml {

struct ResidentTensor {
    uint32_t tensor_ref = UINT32_MAX;
    MaterializedTensor materialized{};
    uint64_t bytes = 0;
    uint64_t last_use = 0;
    uint32_t active_leases = 0;
};

enum class ResidencyEventKind {
    Request,
    Miss,
    Materialize,
    Insert,
    Hit,
    LeaseAcquire,
    LeaseRelease,
    Evict,
    InsertRejected,
};

struct ResidencyTraceEvent {
    uint32_t tensor_ref = UINT32_MAX;
    std::string tensor_name;
    ResidencyEventKind kind = ResidencyEventKind::Miss;
    uint64_t resident_bytes_before = 0;
    uint64_t resident_bytes_after = 0;
    uint32_t active_leases = 0;
    uint64_t timestamp_ns = 0;
    std::string source_id;
};

const char * residency_event_name(ResidencyEventKind kind);

enum class ResidencyReplacementPolicyKind {
    LRU,
    CostAware,
};

struct ResidencyReplacementCandidate {
    uint32_t tensor_ref = UINT32_MAX;
    uint64_t bytes = 0;
    uint32_t active_leases = 0;
    uint64_t last_request_ordinal = 0;
    uint64_t observed_request_count = 0;
    uint64_t reacquire_cost_bytes = 0;
};

class ResidencyReplacementPolicy {
public:
    virtual ~ResidencyReplacementPolicy() = default;
    virtual uint32_t choose(const std::vector<ResidencyReplacementCandidate> & candidates,
        uint64_t current_request_ordinal) const = 0;
    virtual const char * name() const = 0;
};

std::shared_ptr<const ResidencyReplacementPolicy> make_residency_replacement_policy(
    ResidencyReplacementPolicyKind kind);

class TensorResidencyStore final {
public:
    explicit TensorResidencyStore(uint64_t max_resident_bytes,
        ResidencyReplacementPolicyKind policy_kind = ResidencyReplacementPolicyKind::LRU);
    ~TensorResidencyStore();

    const ResidentTensor * lookup(uint32_t tensor_ref, const std::string & tensor_name = {});
    const ResidentTensor * peek(uint32_t tensor_ref) const;
    void note_miss(uint32_t tensor_ref, const std::string & tensor_name);
    void note_request(uint32_t tensor_ref, const std::string & tensor_name);
    void note_materialize(uint32_t tensor_ref, const std::string & tensor_name,
        const std::string & source_id = {});
    bool insert(uint32_t tensor_ref, const std::string & tensor_name,
        const MaterializedTensor & materialized, const std::string & source_id = {});
    bool acquire(uint32_t tensor_ref, const std::string & tensor_name = {});
    bool release(uint32_t tensor_ref, const std::string & tensor_name = {});
    bool evict(uint32_t tensor_ref, const std::string & tensor_name = {});
    void clear();

    uint64_t max_resident_bytes() const { return max_resident_bytes_; }
    uint64_t resident_bytes() const { return resident_bytes_; }
    size_t resident_count() const { return entries_.size(); }
    uint64_t active_lease_bytes() const;
    uint32_t active_lease_count() const;
    uint64_t materialization_count() const { return materialization_count_; }
    uint64_t reacquisition_count() const { return reacquisition_count_; }
    const char * replacement_policy_name() const { return replacement_policy_->name(); }
    uint64_t policy_decisions() const { return policy_decisions_; }
    uint64_t policy_candidates_evaluated() const { return policy_candidates_evaluated_; }
    uint64_t policy_cpu_time_ns() const { return policy_cpu_time_ns_; }
    uint64_t policy_max_decision_ns() const { return policy_max_decision_ns_; }
    const std::vector<ResidencyTraceEvent> & trace() const { return trace_; }

private:
    bool evict_one(const std::string & reason);
    void add_event(uint32_t tensor_ref, const std::string & tensor_name,
        ResidencyEventKind kind, uint64_t before, uint32_t leases,
        const std::string & source_id = {});

    uint64_t max_resident_bytes_ = 0;
    uint64_t resident_bytes_ = 0;
    uint64_t clock_ = 0;
    uint64_t request_ordinal_ = 0;
    std::unordered_map<uint32_t, ResidentTensor> entries_;
    std::unordered_map<uint32_t, std::string> names_;
    std::unordered_map<uint32_t, uint64_t> last_request_ordinal_;
    std::unordered_map<uint32_t, uint64_t> observed_request_count_;
    std::unordered_set<uint32_t> materialized_tensors_;
    std::vector<ResidencyTraceEvent> trace_;
    std::shared_ptr<const ResidencyReplacementPolicy> replacement_policy_;
    uint64_t materialization_count_ = 0;
    uint64_t reacquisition_count_ = 0;
    uint64_t policy_decisions_ = 0;
    uint64_t policy_candidates_evaluated_ = 0;
    uint64_t policy_cpu_time_ns_ = 0;
    uint64_t policy_max_decision_ns_ = 0;
};

// Retains completed materialization buffers in TensorResidencyStore while
// preserving TensorMaterializer's execution-time request/wait/release contract.
class ResidentTensorMaterializer final : public TensorMaterializer {
public:
    ResidentTensorMaterializer(std::shared_ptr<TensorMaterializer> backing,
        std::shared_ptr<TensorResidencyStore> residency);
    ~ResidentTensorMaterializer() override = default;

    bool request(uint32_t tensor_ref, const PersistentTensorRef & tensor,
        uint64_t byte_budget) override;
    MaterializationState state(uint32_t tensor_ref) const override;
    MaterializationState wait(uint32_t tensor_ref) override;
    std::optional<MaterializedTensor> obtain_ready_tensor(
        uint32_t tensor_ref) override;
    void release(uint32_t tensor_ref) override;
    uint64_t active_inflight_bytes() const override;
    uint64_t active_ready_bytes() const override;
    std::vector<MaterializationTraceEvent> trace() const override;

    const std::shared_ptr<TensorResidencyStore> & residency() const { return residency_; }

private:
    struct RequestInfo {
        PersistentTensorRef tensor{};
        bool lease_acquired = false;
        bool retained = false;
    };

    std::shared_ptr<TensorMaterializer> backing_;
    std::shared_ptr<TensorResidencyStore> residency_;
    std::unordered_map<uint32_t, PersistentTensorRef> known_tensors_;
    std::unordered_map<uint32_t, RequestInfo> requests_;
};

} // namespace vbuf_ggml
