#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "vbuf_materializer.h"

namespace vbuf_ggml {

struct RangeStripe {
    std::string id;
    std::shared_ptr<RangeSource> source;
    uint64_t source_offset = 0;
    uint64_t destination_offset = 0;
    uint64_t length = 0;
};

struct RangeStripingPlan {
    uint64_t tensor_offset = 0;
    uint64_t tensor_length = 0;
    std::vector<RangeStripe> stripes;

    static std::optional<RangeStripingPlan> two_way(
        uint64_t tensor_offset, uint64_t tensor_length,
        std::shared_ptr<RangeSource> source_a,
        std::shared_ptr<RangeSource> source_b,
        uint64_t alignment = 1);

    bool validate(std::string * detail = nullptr) const;
};

struct StripeTraceEvent {
    std::string stripe_id;
    uint64_t request_start_ns = 0;
    uint64_t first_byte_ns = 0;
    uint64_t complete_ns = 0;
    uint64_t requested_offset = 0;
    uint64_t requested_length = 0;
    uint64_t returned_bytes = 0;
    int status_code = 0;
    std::string source_id;
    std::string local_endpoint;
    std::string remote_endpoint;
    std::string error;
};

class ParallelRangeMaterializer final : public TensorMaterializer {
public:
    explicit ParallelRangeMaterializer(RangeStripingPlan plan);
    ~ParallelRangeMaterializer() override;

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

    const RangeStripingPlan & plan() const { return plan_; }
    const std::vector<StripeTraceEvent> & stripe_trace() const { return stripe_trace_; }

private:
    struct OwnedBytes;
    RangeStripingPlan plan_;
    PersistentTensorRef tensor_{};
    uint32_t tensor_ref_ = UINT32_MAX;
    MaterializationState state_ = MaterializationState::NotRequested;
    std::shared_ptr<OwnedBytes> owner_;
    MaterializedTensor ready_{};
    uint64_t inflight_bytes_ = 0;
    uint64_t ready_bytes_ = 0;
    std::vector<StripeTraceEvent> stripe_trace_;
    std::vector<MaterializationTraceEvent> trace_;
    mutable std::mutex mutex_;
    std::thread worker_;
};

} // namespace vbuf_ggml
