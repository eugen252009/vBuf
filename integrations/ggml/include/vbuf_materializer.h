#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vbuf_tensor_wave.h"
#include "vbuf_range_source.h"

namespace vbuf_ggml {

enum class MaterializationState {
    NotRequested,
    InFlight,
    Ready,
    Released,
    Failed,
};

struct MaterializedTensor {
    // Geometry is value-owned; payload remains borrowed through `storage.lease`.
    uint8_t representation = 0;
    uint8_t rank = 0;
    std::array<uint64_t, GGML_MAX_DIMS> dimensions{};
    const uint8_t * payload = nullptr;
    uint64_t payload_len = 0;
    VbufBorrowedStorage storage{};
    uint64_t bytes = 0;

    bool assign_view(const VbufTensorView & source) {
        if (source.rank > dimensions.size() ||
            (source.rank != 0 && source.dimensions == nullptr)) return false;
        representation = source.representation;
        rank = source.rank;
        dimensions.fill(0);
        for (uint8_t index = 0; index < rank; ++index)
            dimensions[index] = source.dimensions[index];
        payload = source.payload;
        payload_len = source.payload_len;
        return true;
    }

    VbufTensorView view() const {
        return { representation, rank, dimensions.data(), payload, payload_len };
    }
};

struct MaterializationTraceEvent {
    uint32_t tensor_ref = UINT32_MAX;
    std::string tensor_name;
    std::string event;
    MaterializationState state = MaterializationState::NotRequested;
    uint64_t timestamp_ns = 0;
    uint64_t bytes = 0;
    uint64_t active_inflight_bytes = 0;
    uint64_t active_ready_bytes = 0;
    uint64_t rss_kib = 0;
    uint64_t requested_offset = 0;
    uint64_t first_byte_timestamp_ns = 0;
    uint64_t returned_bytes = 0;
    int status_code = 0;
    std::string source_id;
    std::string content_range;
    uint64_t payload_hash = 0;
    std::string local_endpoint;
    std::string remote_endpoint;
};

class TensorMaterializer {
public:
    virtual ~TensorMaterializer() = default;

    virtual bool request(
        uint32_t tensor_ref,
        const PersistentTensorRef & tensor,
        uint64_t byte_budget) = 0;
    virtual MaterializationState state(uint32_t tensor_ref) const = 0;
    virtual MaterializationState wait(uint32_t tensor_ref) = 0;
    virtual std::optional<MaterializedTensor> obtain_ready_tensor(
        uint32_t tensor_ref) = 0;
    virtual void release(uint32_t tensor_ref) = 0;
    virtual uint64_t active_inflight_bytes() const = 0;
    virtual uint64_t active_ready_bytes() const = 0;
    virtual std::vector<MaterializationTraceEvent> trace() const = 0;
};

// Qualification source: copies one validated mapped vBuf tensor range into
// owned RAM. It intentionally has one worker and no source-selection policy.
class LocalVbufRangeMaterializer final : public TensorMaterializer {
public:
    explicit LocalVbufRangeMaterializer(std::shared_ptr<RangeSource> source = {},
        std::shared_ptr<RangeSource> fallback_source = {});
    ~LocalVbufRangeMaterializer() override;

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
    void clear_trace();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char * materialization_state_name(MaterializationState state);

} // namespace vbuf_ggml
