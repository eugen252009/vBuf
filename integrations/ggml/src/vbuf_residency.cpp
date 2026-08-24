#include "vbuf_residency.h"

#include <algorithm>
#include <chrono>
#include <limits>

namespace vbuf_ggml {
namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

namespace {

class LruReplacementPolicy final : public ResidencyReplacementPolicy {
public:
    uint32_t choose(const std::vector<ResidencyReplacementCandidate> & candidates,
        uint64_t) const override {
        return std::min_element(candidates.begin(), candidates.end(),
            [](const auto & lhs, const auto & rhs) {
                if (lhs.last_request_ordinal != rhs.last_request_ordinal)
                    return lhs.last_request_ordinal < rhs.last_request_ordinal;
                return lhs.tensor_ref < rhs.tensor_ref;
            })->tensor_ref;
    }

    const char * name() const override { return "LRU"; }
};

class CostAwareReplacementPolicy final : public ResidencyReplacementPolicy {
public:
    uint32_t choose(const std::vector<ResidencyReplacementCandidate> & candidates,
        uint64_t current_request_ordinal) const override {
        if (candidates.empty()) return UINT32_MAX;
        double max_density = 0.0;
        double max_recency = 0.0;
        for (const auto & candidate : candidates) {
            const double density = candidate.bytes == 0 ? 0.0 :
                static_cast<double>(candidate.observed_request_count) *
                static_cast<double>(candidate.reacquire_cost_bytes) /
                static_cast<double>(candidate.bytes);
            const uint64_t age = current_request_ordinal >= candidate.last_request_ordinal
                ? current_request_ordinal - candidate.last_request_ordinal : 0;
            max_density = std::max(max_density, density);
            max_recency = std::max(max_recency, 1.0 / static_cast<double>(age + 1));
        }
        uint32_t victim = UINT32_MAX;
        double victim_score = std::numeric_limits<double>::infinity();
        for (const auto & candidate : candidates) {
            const double density = candidate.bytes == 0 ? 0.0 :
                static_cast<double>(candidate.observed_request_count) *
                static_cast<double>(candidate.reacquire_cost_bytes) /
                static_cast<double>(candidate.bytes);
            const uint64_t age = current_request_ordinal >= candidate.last_request_ordinal
                ? current_request_ordinal - candidate.last_request_ordinal : 0;
            const double recency = 1.0 / static_cast<double>(age + 1);
            const double score = 0.9 * (max_density == 0.0 ? 0.0 : density / max_density) +
                0.1 * (max_recency == 0.0 ? 0.0 : recency / max_recency);
            if (score < victim_score || (score == victim_score && candidate.tensor_ref < victim)) {
                victim = candidate.tensor_ref;
                victim_score = score;
            }
        }
        return victim;
    }

    const char * name() const override { return "COST_AWARE"; }
};

} // namespace

std::shared_ptr<const ResidencyReplacementPolicy> make_residency_replacement_policy(
    ResidencyReplacementPolicyKind kind) {
    if (kind == ResidencyReplacementPolicyKind::CostAware)
        return std::make_shared<CostAwareReplacementPolicy>();
    return std::make_shared<LruReplacementPolicy>();
}

const char * residency_event_name(ResidencyEventKind kind) {
    switch (kind) {
    case ResidencyEventKind::Request: return "REQUEST";
    case ResidencyEventKind::Miss: return "MISS";
    case ResidencyEventKind::Materialize: return "MATERIALIZE";
    case ResidencyEventKind::Insert: return "INSERT";
    case ResidencyEventKind::Hit: return "HIT";
    case ResidencyEventKind::LeaseAcquire: return "LEASE_ACQUIRE";
    case ResidencyEventKind::LeaseRelease: return "LEASE_RELEASE";
    case ResidencyEventKind::Evict: return "EVICT";
    case ResidencyEventKind::InsertRejected: return "INSERT_REJECTED";
    }
    return "UNKNOWN";
}

TensorResidencyStore::TensorResidencyStore(uint64_t max_resident_bytes,
    ResidencyReplacementPolicyKind policy_kind)
    : max_resident_bytes_(max_resident_bytes),
      replacement_policy_(make_residency_replacement_policy(policy_kind)) {}

TensorResidencyStore::~TensorResidencyStore() {
    clear();
}

void TensorResidencyStore::add_event(uint32_t tensor_ref, const std::string & tensor_name,
    ResidencyEventKind kind, uint64_t before, uint32_t leases, const std::string & source_id) {
    trace_.push_back({ tensor_ref, tensor_name, kind, before, resident_bytes_, leases,
        now_ns(), source_id });
}

const ResidentTensor * TensorResidencyStore::lookup(uint32_t tensor_ref,
    const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) return nullptr;
    it->second.last_use = ++clock_;
    names_[tensor_ref] = tensor_name.empty() ? names_[tensor_ref] : tensor_name;
    add_event(tensor_ref, names_[tensor_ref], ResidencyEventKind::Hit,
        resident_bytes_, it->second.active_leases);
    return &it->second;
}

const ResidentTensor * TensorResidencyStore::peek(uint32_t tensor_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    return it == entries_.end() ? nullptr : &it->second;
}

bool TensorResidencyStore::contains(uint32_t tensor_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.find(tensor_ref) != entries_.end();
}

std::optional<MaterializedTensor> TensorResidencyStore::acquire_materialized(
    uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) return std::nullopt;
    const uint64_t before = resident_bytes_;
    ++it->second.active_leases;
    it->second.last_use = ++clock_;
    add_event(tensor_ref, tensor_name.empty() ? names_[tensor_ref] : tensor_name,
        ResidencyEventKind::LeaseAcquire, before, it->second.active_leases);
    return it->second.materialized;
}

void TensorResidencyStore::note_miss(uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Miss, resident_bytes_, 0);
}

void TensorResidencyStore::note_request(uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++request_ordinal_;
    ++observed_request_count_[tensor_ref];
    last_request_ordinal_[tensor_ref] = request_ordinal_;
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Request, resident_bytes_, 0);
}

void TensorResidencyStore::note_materialize(uint32_t tensor_ref, const std::string & tensor_name,
    const std::string & source_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++materialization_count_;
    if (!materialized_tensors_.insert(tensor_ref).second) ++reacquisition_count_;
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Materialize,
        resident_bytes_, 0, source_id);
}

bool TensorResidencyStore::evict_one(const std::string &) {
    std::vector<ResidencyReplacementCandidate> candidates;
    candidates.reserve(entries_.size());
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.active_leases != 0) continue;
        candidates.push_back({ it->first, it->second.bytes, it->second.active_leases,
            last_request_ordinal_[it->first], observed_request_count_[it->first], it->second.bytes });
    }
    if (candidates.empty()) return false;
    const auto started = std::chrono::steady_clock::now();
    const uint32_t ref = replacement_policy_->choose(candidates, request_ordinal_);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    ++policy_decisions_;
    policy_candidates_evaluated_ += candidates.size();
    policy_cpu_time_ns_ += static_cast<uint64_t>(elapsed);
    policy_max_decision_ns_ = std::max(policy_max_decision_ns_, static_cast<uint64_t>(elapsed));
    auto candidate = entries_.find(ref);
    if (candidate == entries_.end() || candidate->second.active_leases != 0) return false;
    const uint64_t before = resident_bytes_;
    const std::string name = names_[ref];
    resident_bytes_ -= candidate->second.bytes;
    entries_.erase(candidate);
    names_.erase(ref);
    ++eviction_count_;
    add_event(ref, name, ResidencyEventKind::Evict, before, 0);
    return true;
}

bool TensorResidencyStore::insert(uint32_t tensor_ref, const std::string & tensor_name,
    const MaterializedTensor & materialized, const std::string & source_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.count(tensor_ref) != 0) return false;
    const uint64_t bytes = materialized.bytes;
    if (bytes > max_resident_bytes_) {
        add_event(tensor_ref, tensor_name, ResidencyEventKind::InsertRejected,
            resident_bytes_, 0, source_id);
        return false;
    }
    while (resident_bytes_ + bytes > max_resident_bytes_ && evict_one("budget")) {}
    if (resident_bytes_ + bytes > max_resident_bytes_) {
        add_event(tensor_ref, tensor_name, ResidencyEventKind::InsertRejected,
            resident_bytes_, 0, source_id);
        return false;
    }
    const uint64_t before = resident_bytes_;
    entries_.emplace(tensor_ref, ResidentTensor{ tensor_ref, materialized, bytes, ++clock_, 0 });
    names_[tensor_ref] = tensor_name;
    resident_bytes_ += bytes;
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Insert, before, 0, source_id);
    return true;
}

bool TensorResidencyStore::acquire(uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) return false;
    const uint64_t before = resident_bytes_;
    ++it->second.active_leases;
    it->second.last_use = ++clock_;
    add_event(tensor_ref, tensor_name.empty() ? names_[tensor_ref] : tensor_name,
        ResidencyEventKind::LeaseAcquire, before, it->second.active_leases);
    return true;
}

bool TensorResidencyStore::release(uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases == 0) return false;
    const uint64_t before = resident_bytes_;
    --it->second.active_leases;
    add_event(tensor_ref, tensor_name.empty() ? names_[tensor_ref] : tensor_name,
        ResidencyEventKind::LeaseRelease, before, it->second.active_leases);
    return true;
}

bool TensorResidencyStore::evict(uint32_t tensor_ref, const std::string & tensor_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases != 0) return false;
    const uint64_t before = resident_bytes_;
    const std::string name = tensor_name.empty() ? names_[tensor_ref] : tensor_name;
    resident_bytes_ -= it->second.bytes;
    entries_.erase(it);
    names_.erase(tensor_ref);
    ++eviction_count_;
    add_event(tensor_ref, name, ResidencyEventKind::Evict, before, 0);
    return true;
}

void TensorResidencyStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.active_leases != 0) {
            ++it;
            continue;
        }
        resident_bytes_ -= it->second.bytes;
        names_.erase(it->first);
        it = entries_.erase(it);
    }
}

void TensorResidencyStore::clear_trace() {
    std::lock_guard<std::mutex> lock(mutex_);
    trace_.clear();
}

uint64_t TensorResidencyStore::resident_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return resident_bytes_;
}

size_t TensorResidencyStore::resident_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

uint64_t TensorResidencyStore::active_lease_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t bytes = 0;
    for (const auto & entry : entries_) if (entry.second.active_leases != 0) bytes += entry.second.bytes;
    return bytes;
}

uint32_t TensorResidencyStore::active_lease_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t count = 0;
    for (const auto & entry : entries_) count += entry.second.active_leases;
    return count;
}

uint64_t TensorResidencyStore::materialization_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return materialization_count_;
}

uint64_t TensorResidencyStore::reacquisition_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reacquisition_count_;
}

uint64_t TensorResidencyStore::eviction_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return eviction_count_;
}

uint64_t TensorResidencyStore::policy_decisions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_decisions_;
}

uint64_t TensorResidencyStore::policy_candidates_evaluated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_candidates_evaluated_;
}

uint64_t TensorResidencyStore::policy_cpu_time_ns() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_cpu_time_ns_;
}

uint64_t TensorResidencyStore::policy_max_decision_ns() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return policy_max_decision_ns_;
}

std::vector<ResidencyTraceEvent> TensorResidencyStore::trace() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trace_;
}

ResidentTensorMaterializer::ResidentTensorMaterializer(
    std::shared_ptr<TensorMaterializer> backing,
    std::shared_ptr<TensorResidencyStore> residency)
    : backing_(std::move(backing)), residency_(std::move(residency)) {}

bool ResidentTensorMaterializer::request(uint32_t tensor_ref,
    const PersistentTensorRef & tensor, uint64_t byte_budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    known_tensors_[tensor_ref] = tensor;
    const auto existing = requests_.find(tensor_ref);
    if (existing != requests_.end()) {
        if (existing->second.retained) return true;
        if (const auto resident = residency_->acquire_materialized(
                tensor_ref, tensor.name); resident.has_value()) {
            existing->second.retained = true;
            existing->second.lease_acquired = true;
            existing->second.retained_tensor = resident;
            return true;
        }
        if (backing_->state(tensor_ref) == MaterializationState::InFlight ||
            backing_->state(tensor_ref) == MaterializationState::Ready) {
            return true;
        }
    }
    requests_[tensor_ref] = { tensor, false, false, false, std::nullopt };
    residency_->note_request(tensor_ref, tensor.name);
    if (const auto resident = residency_->acquire_materialized(
            tensor_ref, tensor.name); resident.has_value()) {
        auto & request = requests_.at(tensor_ref);
        request.retained = true;
        request.lease_acquired = true;
        request.retained_tensor = resident;
        return true;
    }
    residency_->note_miss(tensor_ref, tensor.name);
    auto & request = requests_.at(tensor_ref);
    request.backing_requested = true;
    if (backing_->request(tensor_ref, tensor, byte_budget)) return true;
    const auto backing_state = backing_->state(tensor_ref);
    return backing_state == MaterializationState::InFlight || backing_state == MaterializationState::Ready;
}

MaterializationState ResidentTensorMaterializer::state(uint32_t tensor_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(tensor_ref);
    if (it != requests_.end() && it->second.retained) {
        return MaterializationState::Ready;
    }
    if (residency_->contains(tensor_ref)) {
        return MaterializationState::Ready;
    }
    if (backing_->state(tensor_ref) == MaterializationState::NotRequested &&
        it != requests_.end()) return MaterializationState::NotRequested;
    return backing_->state(tensor_ref);
}

MaterializationState ResidentTensorMaterializer::wait(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(tensor_ref);
    if (it != requests_.end() && it->second.retained) {
        return MaterializationState::Ready;
    }
    if (residency_->contains(tensor_ref)) {
        return MaterializationState::Ready;
    }
    return backing_->wait(tensor_ref);
}

std::optional<MaterializedTensor> ResidentTensorMaterializer::obtain_ready_tensor(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto request = requests_.find(tensor_ref);
    if (request == requests_.end()) {
        auto known = known_tensors_.find(tensor_ref);
        if (known == known_tensors_.end()) return std::nullopt;
        request = requests_.emplace(tensor_ref,
            RequestInfo{ known->second, false, false, false, std::nullopt }).first;
    }
    if (request->second.retained && request->second.retained_tensor.has_value()) {
        return request->second.retained_tensor;
    }
    if (const auto resident = residency_->acquire_materialized(
            tensor_ref, request->second.tensor.name); resident.has_value()) {
        request->second.lease_acquired = true;
        request->second.retained = true;
        return resident;
    }
    const auto ready = backing_->obtain_ready_tensor(tensor_ref);
    if (!ready.has_value()) return std::nullopt;
    std::string source_id;
    const auto materialization_trace = backing_->trace();
    for (auto event = materialization_trace.rbegin();
         event != materialization_trace.rend(); ++event) {
        if (event->tensor_ref == tensor_ref && !event->source_id.empty()) {
            source_id = event->source_id;
            break;
        }
    }
    residency_->note_materialize(tensor_ref, request->second.tensor.name, source_id);
    if (residency_->insert(tensor_ref, request->second.tensor.name, *ready, source_id)) {
        request->second.retained = true;
        request->second.lease_acquired = true;
        const auto inserted = residency_->acquire_materialized(
            tensor_ref, request->second.tensor.name);
        if (inserted.has_value()) {
            request->second.retained_tensor = inserted;
            return inserted;
        }
        request->second.retained = false;
        request->second.lease_acquired = false;
        return ready;
    }
    if (const auto existing = residency_->acquire_materialized(
            tensor_ref, request->second.tensor.name); existing.has_value()) {
        request->second.retained = true;
        request->second.lease_acquired = true;
        request->second.retained_tensor = existing;
        return existing;
    }
    request->second.retained = false;
    request->second.lease_acquired = false;
    return ready;
}

void ResidentTensorMaterializer::release(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requests_.find(tensor_ref);
    if (it == requests_.end()) return;
    if (it->second.retained) residency_->release(tensor_ref, it->second.tensor.name);
    if (it->second.backing_requested) backing_->release(tensor_ref);
    requests_.erase(it);
}

void ResidentTensorMaterializer::release_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & entry : requests_) {
        const RequestInfo & request = entry.second;
        if (request.retained) residency_->release(entry.first, request.tensor.name);
        if (request.backing_requested) backing_->release(entry.first);
    }
    requests_.clear();
}

uint64_t ResidentTensorMaterializer::active_inflight_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backing_->active_inflight_bytes();
}

uint64_t ResidentTensorMaterializer::active_ready_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return residency_->resident_bytes();
}

std::vector<MaterializationTraceEvent> ResidentTensorMaterializer::trace() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return backing_->trace();
}

void ResidentTensorMaterializer::clear_trace() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto local = std::dynamic_pointer_cast<LocalVbufRangeMaterializer>(backing_);
    if (local) local->clear_trace();
}

} // namespace vbuf_ggml
