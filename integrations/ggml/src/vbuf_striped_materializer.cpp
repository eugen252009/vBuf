#include "vbuf_striped_materializer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace vbuf_ggml {
namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
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

struct ParallelRangeMaterializer::OwnedBytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    ~OwnedBytes() { std::free(data); }
};

std::optional<RangeStripingPlan> RangeStripingPlan::two_way(
    uint64_t tensor_offset, uint64_t tensor_length,
    std::shared_ptr<RangeSource> source_a, std::shared_ptr<RangeSource> source_b,
    uint64_t alignment) {
    if (!source_a || !source_b || tensor_length < 2 || alignment == 0) return std::nullopt;
    uint64_t split = tensor_length / 2;
    split -= split % alignment;
    if (split == 0 || split >= tensor_length) return std::nullopt;
    RangeStripingPlan plan;
    plan.tensor_offset = tensor_offset;
    plan.tensor_length = tensor_length;
    plan.stripes = {
        { "A", std::move(source_a), tensor_offset, 0, split },
        { "B", std::move(source_b), tensor_offset + split, split, tensor_length - split },
    };
    return plan.validate() ? std::optional<RangeStripingPlan>(std::move(plan)) : std::nullopt;
}

bool RangeStripingPlan::validate(std::string * detail) const {
    if (stripes.size() != 2) {
        if (detail) *detail = "exactly two stripes required";
        return false;
    }
    uint64_t next_destination = 0;
    for (const auto & stripe : stripes) {
        if (!stripe.source || stripe.length == 0 || stripe.destination_offset != next_destination ||
            stripe.source_offset != tensor_offset + stripe.destination_offset) {
            if (detail) *detail = "stripe has a gap, overlap, invalid source, or wrong offset";
            return false;
        }
        next_destination += stripe.length;
    }
    if (next_destination != tensor_length) {
        if (detail) *detail = "stripe union does not equal tensor length";
        return false;
    }
    return true;
}

ParallelRangeMaterializer::ParallelRangeMaterializer(RangeStripingPlan plan)
    : plan_(std::move(plan)) {}

ParallelRangeMaterializer::~ParallelRangeMaterializer() {
    if (worker_.joinable()) worker_.join();
    release(tensor_ref_);
}

bool ParallelRangeMaterializer::request(uint32_t tensor_ref,
    const PersistentTensorRef & tensor, uint64_t byte_budget) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != MaterializationState::NotRequested || tensor.view.payload_len > byte_budget ||
        tensor.source_offset != plan_.tensor_offset || tensor.view.payload_len != plan_.tensor_length) {
        return false;
    }
    std::string detail;
    if (!plan_.validate(&detail)) return false;
    owner_ = std::shared_ptr<OwnedBytes>(new OwnedBytes());
    owner_->size = tensor.view.payload_len;
    if (posix_memalign(reinterpret_cast<void **>(&owner_->data), 64, owner_->size) != 0) {
        owner_.reset();
        return false;
    }
    tensor_ref_ = tensor_ref;
    tensor_ = tensor;
    stripe_trace_.assign(plan_.stripes.size(), {});
    state_ = MaterializationState::InFlight;
    inflight_bytes_ = tensor.view.payload_len;
    trace_.push_back({ tensor_ref, tensor.name, "STRIPED_IN_FLIGHT", MaterializationState::InFlight,
        now_ns(), tensor.view.payload_len, inflight_bytes_, ready_bytes_, 0,
        tensor.source_offset, 0, 0, 0, "striped", {}, 0 });
    worker_ = std::thread([this]() {
        std::atomic<int> ready_workers{ 0 };
        std::atomic<bool> go{ false };
        std::vector<uint8_t> successes(plan_.stripes.size(), 0);
        std::vector<std::thread> workers;
        for (size_t index = 0; index < plan_.stripes.size(); ++index) {
            workers.emplace_back([&, index]() {
                const auto & stripe = plan_.stripes[index];
                StripeTraceEvent event;
                event.stripe_id = stripe.id;
                event.requested_offset = stripe.source_offset;
                event.requested_length = stripe.length;
                ready_workers.fetch_add(1);
                while (!go.load()) std::this_thread::yield();
                event.request_start_ns = now_ns();
                RangeReadResult result{};
                const bool success = stripe.source->read_range(stripe.source_offset, stripe.length,
                    owner_->data + stripe.destination_offset, &result);
                event.first_byte_ns = result.first_byte_timestamp_ns;
                event.complete_ns = now_ns();
                event.returned_bytes = result.returned_bytes;
                event.status_code = result.status_code;
                event.source_id = result.source_id;
                event.local_endpoint = result.local_endpoint;
                event.remote_endpoint = result.remote_endpoint;
                event.error = result.error;
                successes[index] = success && result.returned_bytes == stripe.length ? 1 : 0;
                std::lock_guard<std::mutex> lock(mutex_);
                stripe_trace_[index] = std::move(event);
            });
        }
        while (ready_workers.load() != static_cast<int>(workers.size())) std::this_thread::yield();
        go.store(true);
        for (auto & worker : workers) worker.join();
        const bool complete = std::all_of(successes.begin(), successes.end(), [](bool value) { return value; });
        std::lock_guard<std::mutex> lock(mutex_);
        inflight_bytes_ = 0;
        if (!complete) {
            owner_.reset();
            state_ = MaterializationState::Failed;
            trace_.push_back({ tensor_ref_, tensor_.name, "STRIPED_FAILED", state_, now_ns(),
                tensor_.view.payload_len, 0, 0, 0, tensor_.source_offset, 0, 0, 0, "striped", {}, 0 });
            return;
        }
        ready_.view = tensor_.view;
        ready_.view.payload = owner_->data;
        ready_.storage = { owner_->data, owner_->size, 0,
            std::shared_ptr<const void>(owner_, owner_->data) };
        ready_.bytes = owner_->size;
        ready_bytes_ = owner_->size;
        state_ = MaterializationState::Ready;
        trace_.push_back({ tensor_ref_, tensor_.name, "STRIPED_READY", state_, now_ns(),
            tensor_.view.payload_len, 0, ready_bytes_, 0, tensor_.source_offset, 0,
            tensor_.view.payload_len, 200, "striped", {}, fnv1a(owner_->data, owner_->size) });
    });
    return true;
}

MaterializationState ParallelRangeMaterializer::state(uint32_t tensor_ref) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tensor_ref == tensor_ref_ ? state_ : MaterializationState::NotRequested;
}

MaterializationState ParallelRangeMaterializer::wait(uint32_t tensor_ref) {
    if (worker_.joinable()) worker_.join();
    return state(tensor_ref);
}

std::optional<MaterializedTensor> ParallelRangeMaterializer::obtain_ready_tensor(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor_ref != tensor_ref_ || state_ != MaterializationState::Ready) return std::nullopt;
    return ready_;
}

void ParallelRangeMaterializer::release(uint32_t tensor_ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (tensor_ref != tensor_ref_ || state_ != MaterializationState::Ready) return;
    ready_bytes_ = 0;
    ready_ = {};
    owner_.reset();
    state_ = MaterializationState::Released;
    trace_.push_back({ tensor_ref_, tensor_.name, "STRIPED_RELEASED", state_, now_ns(),
        tensor_.view.payload_len, 0, 0, 0, tensor_.source_offset, 0,
        tensor_.view.payload_len, 200, "striped", {}, 0 });
}

uint64_t ParallelRangeMaterializer::active_inflight_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return inflight_bytes_;
}

uint64_t ParallelRangeMaterializer::active_ready_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ready_bytes_;
}

std::vector<MaterializationTraceEvent> ParallelRangeMaterializer::trace() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return trace_;
}

} // namespace vbuf_ggml
