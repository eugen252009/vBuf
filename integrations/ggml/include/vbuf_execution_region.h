#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "vbuf_region_executor.h"

namespace vbuf_ggml {

enum class RegionOpKind {
    RmsNorm,
    MulMat,
    SwiGluSplit,
};

struct RegionOperation {
    RegionOpKind kind;
    uint32_t output_slot;
    uint32_t input_a_slot;
    uint32_t input_b_slot;
    float parameter;
};

struct RegionWeight {
    uint64_t tensor_id;
    uint32_t slot;
    VbufTensorView view;
};

struct RegionExecutionReport {
    std::vector<uint64_t> requested_tensor_ids;
    std::vector<uint64_t> acquired_tensor_ids;
    uint64_t acquired_weight_bytes = 0;
    uint64_t selected_weight_bytes = 0;
    uint64_t graph_tensor_count = 0;
};

using RegionStorageProvider = std::function<VbufBorrowedStorage(
    const VbufTensorView & view)>;
using RegionPhaseObserver = std::function<void(
    const char * phase, uint64_t active_weight_bytes, uint64_t active_weight_leases)>;

class ExecutionRegion final {
public:
    uint32_t add_weight(uint64_t tensor_id, const VbufTensorView & view);
    uint32_t add_input();
    uint32_t add_temporary();
    uint32_t add_output();
    void add_operation(const RegionOperation & operation);

    AdapterError execute(
        const VbufTensorView & input,
        const RegionStorageProvider & storage_provider,
        std::vector<uint8_t> * output,
        std::vector<int64_t> * output_shape,
        RegionExecutionReport * report,
        std::string * detail = nullptr,
        const RegionPhaseObserver & observer = {}) const;

private:
    uint32_t next_slot_ = 0;
    uint32_t input_slot_ = UINT32_MAX;
    uint32_t output_slot_ = UINT32_MAX;
    std::vector<RegionWeight> weights_;
    std::vector<RegionOperation> operations_;
};

} // namespace vbuf_ggml
