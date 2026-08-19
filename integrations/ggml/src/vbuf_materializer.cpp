#include "vbuf_materializer.h"
#include "vbuf_d0_1_diagnostics.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <fstream>
#include <sstream>
#include <thread>

namespace vbuf_ggml {
namespace {

struct OwnedBytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    ~OwnedBytes() { std::free(data); }
};

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t rss_kib() {
    std::ifstream input("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        fields >> key >> value;
        if (key == "Rss:") return value;
    }
    return 0;
}

uint64_t fnv1a(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

const char * materialization_state_name(MaterializationState state) {
    switch (state) {
    case MaterializationState::NotRequested: return "NOT_REQUESTED";
    case MaterializationState::InFlight: return "IN_FLIGHT";
    case MaterializationState::Ready: return "READY";
    case MaterializationState::Released: return "RELEASED";
    case MaterializationState::Failed: return "FAILED";
    }
    return "UNKNOWN";
}

struct LocalVbufRangeMaterializer::Impl {
    struct Request {
        uint32_t tensor_ref = UINT32_MAX;
        PersistentTensorRef tensor{};
        MaterializationState state = MaterializationState::NotRequested;
        std::shared_ptr<OwnedBytes> owner;
        MaterializedTensor ready{};
        RangeReadResult read_result{};
        uint64_t payload_hash = 0;
        std::thread worker;
    };

    mutable std::mutex mutex;
    std::map<uint32_t, std::unique_ptr<Request>> requests;
    std::vector<MaterializationTraceEvent> events;
    uint64_t inflight_bytes = 0;
    uint64_t ready_bytes = 0;
    std::shared_ptr<RangeSource> source;
    std::shared_ptr<RangeSource> fallback_source;

    void record(const Request & request, MaterializationState state,
        const char * event = "STATE") {
        uint64_t rss = 0;
        if (vbuf_d0_1_smaps_enabled()) {
            const uint64_t sample_start = now_ns();
            rss = rss_kib();
            vbuf_d0_1_smaps(now_ns() - sample_start);
        }
        events.push_back({ request.tensor_ref, request.tensor.name, event, state, now_ns(),
            request.tensor.view.payload_len, inflight_bytes, ready_bytes, rss,
            request.read_result.requested_offset, request.read_result.first_byte_timestamp_ns,
            request.read_result.returned_bytes,
            request.read_result.status_code, request.read_result.source_id,
            request.read_result.content_range, request.payload_hash,
            request.read_result.local_endpoint, request.read_result.remote_endpoint });
    }
};

LocalVbufRangeMaterializer::LocalVbufRangeMaterializer(std::shared_ptr<RangeSource> source,
    std::shared_ptr<RangeSource> fallback_source)
    : impl_(std::make_unique<Impl>()) {
    impl_->source = std::move(source);
    impl_->fallback_source = std::move(fallback_source);
}

LocalVbufRangeMaterializer::~LocalVbufRangeMaterializer() {
    std::vector<std::thread *> workers;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (auto & entry : impl_->requests) {
            if (entry.second->worker.joinable()) workers.push_back(&entry.second->worker);
        }
    }
    for (std::thread * worker : workers) worker->join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto & entry : impl_->requests) {
        Impl::Request & request = *entry.second;
        if (request.state == MaterializationState::Ready) {
            impl_->ready_bytes -= request.tensor.view.payload_len;
            request.state = MaterializationState::Released;
            request.owner.reset();
            request.ready = {};
            impl_->record(request, MaterializationState::Released);
        }
    }
}

bool LocalVbufRangeMaterializer::request(
    uint32_t tensor_ref, const PersistentTensorRef & tensor, uint64_t byte_budget) {
    const uint64_t request_start_ns = now_ns();
    std::thread stale_worker;
    std::unique_lock<std::mutex> lock(impl_->mutex);
    const auto existing = impl_->requests.find(tensor_ref);
    if (existing != impl_->requests.end() &&
        existing->second->state != MaterializationState::Released &&
        existing->second->state != MaterializationState::Failed) return false;
    if (existing != impl_->requests.end()) {
        stale_worker = std::move(existing->second->worker);
        impl_->requests.erase(existing);
        lock.unlock();
        if (stale_worker.joinable()) stale_worker.join();
        lock.lock();
        if (impl_->requests.find(tensor_ref) != impl_->requests.end()) return false;
    }
    if (
        tensor.view.payload_len > byte_budget ||
        impl_->inflight_bytes + impl_->ready_bytes + tensor.view.payload_len > byte_budget) {
        return false;
    }
    auto request = std::make_unique<Impl::Request>();
    request->tensor_ref = tensor_ref;
    request->tensor = tensor;
    request->state = MaterializationState::InFlight;
    request->read_result.requested_offset = tensor.source_offset;
    request->read_result.requested_length = tensor.view.payload_len;
    request->read_result.source_id = impl_->source ? "range-source" : "inline-vbuf";
    vbuf_d0_1_first_payload_request(request_start_ns);
    impl_->inflight_bytes += tensor.view.payload_len;
    impl_->record(*request, MaterializationState::InFlight);
    Impl::Request * raw = request.get();
    impl_->requests.emplace(tensor_ref, std::move(request));
    const uint64_t request_setup_end_ns = now_ns();
    vbuf_d0_1_request_setup(request_setup_end_ns - request_start_ns);
    raw->worker = std::thread([this, raw, request_start_ns]() {
        vbuf_d0_1_worker_start_delay(now_ns() - request_start_ns);
        std::shared_ptr<OwnedBytes> owner = std::make_shared<OwnedBytes>();
        owner->size = raw->tensor.view.payload_len;
        if (posix_memalign(reinterpret_cast<void **>(&owner->data), 64, owner->size) != 0) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->inflight_bytes -= raw->tensor.view.payload_len;
            raw->state = MaterializationState::Failed;
            impl_->record(*raw, MaterializationState::Failed);
            return;
        }
        bool read_ok = false;
        if (impl_->source) {
            read_ok = impl_->source->read_range(raw->tensor.source_offset, owner->size,
                owner->data, &raw->read_result);
            if (!read_ok && impl_->fallback_source) {
                {
                    std::lock_guard<std::mutex> lock(impl_->mutex);
                    impl_->record(*raw, MaterializationState::InFlight, "PRIMARY_SOURCE_FAILED");
                }
                read_ok = impl_->fallback_source->read_range(raw->tensor.source_offset,
                    owner->size, owner->data, &raw->read_result);
            }
        } else {
            raw->read_result = { raw->tensor.source_offset, owner->size, owner->size,
                now_ns(), 200, "inline-vbuf", {}, {} };
            std::memcpy(owner->data, raw->tensor.view.payload, owner->size);
            read_ok = true;
        }
        if (!read_ok) {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->inflight_bytes -= raw->tensor.view.payload_len;
            raw->state = MaterializationState::Failed;
            impl_->record(*raw, MaterializationState::Failed);
            return;
        }
        const uint64_t hash_start_ns = now_ns();
        raw->payload_hash = fnv1a(owner->data, owner->size);
        vbuf_d0_1_hash(owner->size, now_ns() - hash_start_ns);
        const uint64_t finalization_start_ns = now_ns();
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            raw->owner = owner;
            raw->ready.view = raw->tensor.view;
            raw->ready.view.payload = owner->data;
            raw->ready.view.payload_len = owner->size;
            raw->ready.storage = { owner->data, owner->size,
                0, std::shared_ptr<const void>(owner, owner->data) };
            raw->ready.bytes = owner->size;
            impl_->inflight_bytes -= owner->size;
            impl_->ready_bytes += owner->size;
            raw->state = MaterializationState::Ready;
            impl_->record(*raw, MaterializationState::Ready);
        }
        vbuf_d0_1_request_finalization(now_ns() - finalization_start_ns);
    });
    vbuf_d0_1_worker_create(now_ns() - request_setup_end_ns, 0);
    return true;
}

MaterializationState LocalVbufRangeMaterializer::state(uint32_t tensor_ref) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->requests.find(tensor_ref);
    return found == impl_->requests.end()
        ? MaterializationState::NotRequested : found->second->state;
}

MaterializationState LocalVbufRangeMaterializer::wait(uint32_t tensor_ref) {
    Impl::Request * request = nullptr;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const auto found = impl_->requests.find(tensor_ref);
        if (found == impl_->requests.end()) return MaterializationState::NotRequested;
        request = found->second.get();
    }
    if (request->worker.joinable()) {
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            impl_->record(*request, request->state, "WAIT_START");
        }
        const uint64_t wait_start_ns = now_ns();
        request->worker.join();
        vbuf_d0_1_worker_wait(now_ns() - wait_start_ns);
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->record(*request, request->state, "WAIT_END");
    }
    return state(tensor_ref);
}

std::optional<MaterializedTensor> LocalVbufRangeMaterializer::obtain_ready_tensor(
    uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->requests.find(tensor_ref);
    if (found == impl_->requests.end() || found->second->state != MaterializationState::Ready) {
        return std::nullopt;
    }
    return found->second->ready;
}

void LocalVbufRangeMaterializer::release(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->requests.find(tensor_ref);
    if (found == impl_->requests.end()) return;
    Impl::Request & request = *found->second;
    if (request.state == MaterializationState::Ready) {
        impl_->ready_bytes -= request.tensor.view.payload_len;
        request.owner.reset();
        request.ready = {};
        request.state = MaterializationState::Released;
        impl_->record(request, MaterializationState::Released);
    }
}

uint64_t LocalVbufRangeMaterializer::active_inflight_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->inflight_bytes;
}

uint64_t LocalVbufRangeMaterializer::active_ready_bytes() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->ready_bytes;
}

std::vector<MaterializationTraceEvent> LocalVbufRangeMaterializer::trace() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->events;
}

} // namespace vbuf_ggml
