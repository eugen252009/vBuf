#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vbuf_region_executor.h"

namespace vbuf_ggml {

class TensorMaterializer;

enum class TensorWaveOpKind {
    RmsNorm,
    MulMat,
    SwiGlu,
    Dequantize,
};

struct TensorWaveRef {
    enum class Kind { Persistent, Value };
    Kind kind;
    uint32_t index;
};

struct PersistentTensorRef {
    uint64_t tensor_id;
    std::string name;
    VbufTensorView view;
    uint64_t source_offset = 0;
};

struct TensorWaveOp {
    std::string op_id;
    TensorWaveOpKind kind;
    std::vector<TensorWaveRef> inputs;
    uint32_t output_value;
    float parameter = 0.0f;
};

struct TensorWaveGraphView {
    uint32_t input_value = UINT32_MAX;
    uint32_t external_output = UINT32_MAX;
    std::vector<PersistentTensorRef> persistent;
    std::vector<std::string> value_names;
    std::vector<TensorWaveOp> operations;
};

struct TensorWavePlannerState {
    std::vector<uint32_t> completed_operations;
    std::vector<uint32_t> available_values;
    std::vector<uint32_t> resident_persistent;
    uint64_t active_persistent_weight_bytes = 0;
    uint64_t active_persistent_tensor_count = 0;
};

using TensorWaveStorageProvider = std::function<VbufBorrowedStorage(
    const VbufTensorView & view)>;
using TensorWavePlanningObserver = std::function<void(
    const TensorWavePlannerState & state)>;
using TensorWaveExecutionObserver = std::function<void(
    const char * op_id, const char * phase)>;

struct TensorWaveTraceStep {
    std::string op_id;
    TensorWaveOpKind op_kind;
    std::vector<std::string> runnable_before;
    std::vector<std::string> persistent_acquired;
    std::vector<std::string> persistent_released;
    std::vector<std::string> input_values_consumed;
    std::vector<std::string> output_values_produced;
    std::vector<std::string> values_released;
    uint64_t active_persistent_weight_bytes_during = 0;
    uint64_t active_persistent_tensor_count_during = 0;
    uint64_t active_persistent_weight_bytes = 0;
    uint64_t active_persistent_tensor_count = 0;
    uint64_t active_transient_bytes = 0;
};

struct TensorWaveLifetime {
    uint64_t tensor_id = 0;
    std::string name;
    uint64_t bytes = 0;
    uint64_t consumer_count = 0;
    std::string first_consumer;
    std::string last_consumer;
    uint64_t acquire_step = 0;
    uint64_t release_step = 0;
};

struct TensorWaveReport {
    std::vector<TensorWaveTraceStep> trace;
    std::vector<TensorWaveLifetime> persistent_lifetimes;
    uint64_t total_ffn_weight_bytes = 0;
    uint64_t peak_active_weight_bytes = 0;
    uint64_t sum_acquired_weight_bytes = 0;
    uint64_t active_transient_bytes = 0;
    std::vector<uint8_t> captured_value;
    std::vector<int64_t> captured_value_shape;
    bool all_persistent_released = false;
    bool external_output_preserved = false;
};

class TensorDependencyExecutor final {
public:
    uint32_t add_persistent(const PersistentTensorRef & tensor);
    uint32_t add_input(const std::string & name);
    uint32_t add_value(const std::string & name, bool external_output = false);
    void add_operation(const TensorWaveOp & operation);
    void set_external_output(uint32_t value);
    void set_capture_value(uint32_t value);
    TensorWaveGraphView graph_view() const;

    AdapterError execute(
        const VbufTensorView & input,
        const TensorWaveStorageProvider & storage_provider,
        std::vector<uint8_t> * output,
        std::vector<int64_t> * output_shape,
        TensorWaveReport * report,
        std::string * detail = nullptr,
        const TensorWavePlanningObserver & planning_observer = {},
        TensorMaterializer * materializer = nullptr,
        const TensorWaveExecutionObserver & execution_observer = {}) const;

private:
    struct ValueDefinition {
        std::string name;
        bool input = false;
        bool external_output = false;
    };

    uint32_t input_value_ = UINT32_MAX;
    uint32_t external_output_ = UINT32_MAX;
    uint32_t capture_value_ = UINT32_MAX;
    std::vector<PersistentTensorRef> persistent_;
    std::vector<ValueDefinition> values_;
    std::vector<TensorWaveOp> operations_;
};

const char * tensor_wave_op_name(TensorWaveOpKind kind);

} // namespace vbuf_ggml
