#include "vbuf_residency.h"

#include <algorithm>
#include <chrono>

namespace vbuf_ggml {
namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

const char * residency_event_name(ResidencyEventKind kind) {
    switch (kind) {
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

TensorResidencyStore::TensorResidencyStore(uint64_t max_resident_bytes)
    : max_resident_bytes_(max_resident_bytes) {}

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
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end()) return nullptr;
    it->second.last_use = ++clock_;
    names_[tensor_ref] = tensor_name.empty() ? names_[tensor_ref] : tensor_name;
    add_event(tensor_ref, names_[tensor_ref], ResidencyEventKind::Hit,
        resident_bytes_, it->second.active_leases);
    return &it->second;
}

const ResidentTensor * TensorResidencyStore::peek(uint32_t tensor_ref) const {
    auto it = entries_.find(tensor_ref);
    return it == entries_.end() ? nullptr : &it->second;
}

void TensorResidencyStore::note_miss(uint32_t tensor_ref, const std::string & tensor_name) {
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Miss, resident_bytes_, 0);
}

void TensorResidencyStore::note_materialize(uint32_t tensor_ref, const std::string & tensor_name,
    const std::string & source_id) {
    add_event(tensor_ref, tensor_name, ResidencyEventKind::Materialize,
        resident_bytes_, 0, source_id);
}

bool TensorResidencyStore::evict_one(const std::string &) {
    auto candidate = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->second.active_leases != 0) continue;
        if (candidate == entries_.end() || it->second.last_use < candidate->second.last_use ||
            (it->second.last_use == candidate->second.last_use && it->first < candidate->first)) {
            candidate = it;
        }
    }
    if (candidate == entries_.end()) return false;
    const uint32_t ref = candidate->first;
    const uint64_t before = resident_bytes_;
    const std::string name = names_[ref];
    resident_bytes_ -= candidate->second.bytes;
    entries_.erase(candidate);
    names_.erase(ref);
    add_event(ref, name, ResidencyEventKind::Evict, before, 0);
    return true;
}

bool TensorResidencyStore::insert(uint32_t tensor_ref, const std::string & tensor_name,
    const MaterializedTensor & materialized, const std::string & source_id) {
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
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases == 0) return false;
    const uint64_t before = resident_bytes_;
    --it->second.active_leases;
    add_event(tensor_ref, tensor_name.empty() ? names_[tensor_ref] : tensor_name,
        ResidencyEventKind::LeaseRelease, before, it->second.active_leases);
    return true;
}

bool TensorResidencyStore::evict(uint32_t tensor_ref, const std::string & tensor_name) {
    auto it = entries_.find(tensor_ref);
    if (it == entries_.end() || it->second.active_leases != 0) return false;
    const uint64_t before = resident_bytes_;
    const std::string name = tensor_name.empty() ? names_[tensor_ref] : tensor_name;
    resident_bytes_ -= it->second.bytes;
    entries_.erase(it);
    names_.erase(tensor_ref);
    add_event(tensor_ref, name, ResidencyEventKind::Evict, before, 0);
    return true;
}

void TensorResidencyStore::clear() {
    entries_.clear();
    names_.clear();
    resident_bytes_ = 0;
}

uint64_t TensorResidencyStore::active_lease_bytes() const {
    uint64_t bytes = 0;
    for (const auto & entry : entries_) if (entry.second.active_leases != 0) bytes += entry.second.bytes;
    return bytes;
}

uint32_t TensorResidencyStore::active_lease_count() const {
    uint32_t count = 0;
    for (const auto & entry : entries_) if (entry.second.active_leases != 0) ++count;
    return count;
}

ResidentTensorMaterializer::ResidentTensorMaterializer(
    std::shared_ptr<TensorMaterializer> backing,
    std::shared_ptr<TensorResidencyStore> residency)
    : backing_(std::move(backing)), residency_(std::move(residency)) {}

bool ResidentTensorMaterializer::request(uint32_t tensor_ref,
    const PersistentTensorRef & tensor, uint64_t byte_budget) {
    known_tensors_[tensor_ref] = tensor;
    requests_[tensor_ref] = { tensor, false, false };
    if (residency_->lookup(tensor_ref, tensor.name) != nullptr) return true;
    residency_->note_miss(tensor_ref, tensor.name);
    return backing_->request(tensor_ref, tensor, byte_budget);
}

MaterializationState ResidentTensorMaterializer::state(uint32_t tensor_ref) const {
    auto it = requests_.find(tensor_ref);
    if (residency_->peek(tensor_ref) != nullptr) {
        return MaterializationState::Ready;
    }
    if (backing_->state(tensor_ref) == MaterializationState::NotRequested &&
        it != requests_.end()) return MaterializationState::NotRequested;
    return backing_->state(tensor_ref);
}

MaterializationState ResidentTensorMaterializer::wait(uint32_t tensor_ref) {
    auto it = requests_.find(tensor_ref);
    if (residency_->peek(tensor_ref) != nullptr) {
        return MaterializationState::Ready;
    }
    return backing_->wait(tensor_ref);
}

std::optional<MaterializedTensor> ResidentTensorMaterializer::obtain_ready_tensor(uint32_t tensor_ref) {
    auto request = requests_.find(tensor_ref);
    if (request == requests_.end()) {
        auto known = known_tensors_.find(tensor_ref);
        if (known == known_tensors_.end()) return std::nullopt;
        request = requests_.emplace(tensor_ref, RequestInfo{ known->second, false, false }).first;
    }
    const ResidentTensor * resident = residency_->lookup(tensor_ref, request->second.tensor.name);
    if (resident != nullptr) {
        if (!request->second.lease_acquired) {
            residency_->acquire(tensor_ref, request->second.tensor.name);
            request->second.lease_acquired = true;
        }
        request->second.retained = true;
        return resident->materialized;
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
        const ResidentTensor * inserted = residency_->lookup(tensor_ref, request->second.tensor.name);
        residency_->acquire(tensor_ref, request->second.tensor.name);
        request->second.lease_acquired = true;
        return inserted->materialized;
    }
    request->second.retained = false;
    request->second.lease_acquired = true;
    return ready;
}

void ResidentTensorMaterializer::release(uint32_t tensor_ref) {
    auto it = requests_.find(tensor_ref);
    if (it == requests_.end()) return;
    if (it->second.retained) residency_->release(tensor_ref, it->second.tensor.name);
    backing_->release(tensor_ref);
    requests_.erase(it);
}

uint64_t ResidentTensorMaterializer::active_inflight_bytes() const {
    return backing_->active_inflight_bytes();
}

uint64_t ResidentTensorMaterializer::active_ready_bytes() const {
    return residency_->resident_bytes();
}

std::vector<MaterializationTraceEvent> ResidentTensorMaterializer::trace() const {
    return backing_->trace();
}

} // namespace vbuf_ggml
