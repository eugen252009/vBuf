#include "vbuf_tensor_wave.h"
#include "vbuf_materializer.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <unordered_map>

namespace vbuf_ggml {

namespace {
uint64_t audit_hash(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) { hash ^= data[i]; hash *= 1099511628211ULL; }
    return hash;
}
}
namespace {

void set_detail(std::string * detail, const char * message) {
    if (detail != nullptr) *detail = message;
}

struct RuntimeValue {
    VbufTensorView view{};
    std::shared_ptr<std::vector<uint8_t>> bytes;
    std::shared_ptr<std::vector<uint64_t>> dimensions;
};

struct ResidentTensor {
    VbufBorrowedStorage storage{};
    uint64_t bytes = 0;
};

bool has_value(const std::unordered_map<uint32_t, RuntimeValue> & values, uint32_t index) {
    return values.find(index) != values.end();
}

} // namespace

const char * tensor_wave_op_name(TensorWaveOpKind kind) {
    switch (kind) {
    case TensorWaveOpKind::RmsNorm: return "RMSNorm";
    case TensorWaveOpKind::MulMat: return "MatMul";
    case TensorWaveOpKind::SwiGlu: return "SwiGLU";
    case TensorWaveOpKind::Dequantize: return "Dequantize";
    }
    return "Unknown";
}

uint32_t TensorDependencyExecutor::add_persistent(const PersistentTensorRef & tensor) {
    persistent_.push_back(tensor);
    return static_cast<uint32_t>(persistent_.size() - 1);
}

uint32_t TensorDependencyExecutor::add_input(const std::string & name) {
    const uint32_t value = add_value(name);
    input_value_ = value;
    return value;
}

uint32_t TensorDependencyExecutor::add_value(const std::string & name, bool external_output) {
    values_.push_back({ name, false, external_output });
    return static_cast<uint32_t>(values_.size() - 1);
}

void TensorDependencyExecutor::add_operation(const TensorWaveOp & operation) {
    operations_.push_back(operation);
}

void TensorDependencyExecutor::set_external_output(uint32_t value) {
    external_output_ = value;
    if (value < values_.size()) values_[value].external_output = true;
}

void TensorDependencyExecutor::set_capture_value(uint32_t value) {
    capture_value_ = value;
}

TensorWaveGraphView TensorDependencyExecutor::graph_view() const {
    TensorWaveGraphView view;
    view.input_value = input_value_;
    view.external_output = external_output_;
    view.persistent = persistent_;
    view.operations = operations_;
    view.value_names.reserve(values_.size());
    for (const ValueDefinition & value : values_) view.value_names.push_back(value.name);
    return view;
}

AdapterError TensorDependencyExecutor::execute(
    const VbufTensorView & input,
    const TensorWaveStorageProvider & storage_provider,
    std::vector<uint8_t> * output,
    std::vector<int64_t> * output_shape,
    TensorWaveReport * report,
    std::string * error_detail,
    const TensorWavePlanningObserver & planning_observer,
    TensorMaterializer * materializer,
    const TensorWaveExecutionObserver & execution_observer) const {
    if (!storage_provider || output == nullptr || output_shape == nullptr || report == nullptr ||
        input_value_ == UINT32_MAX || external_output_ == UINT32_MAX || operations_.empty()) {
        set_detail(error_detail, "tensor dependency graph has incomplete execution contract");
        return AdapterError::InvalidArgument;
    }

    report->trace.clear();
    report->persistent_lifetimes.clear();
    report->total_ffn_weight_bytes = 0;
    report->peak_active_weight_bytes = 0;
    report->sum_acquired_weight_bytes = 0;
    report->active_transient_bytes = 0;
    report->captured_value.clear();
    report->captured_value_shape.clear();
    report->all_persistent_released = false;
    report->external_output_preserved = false;

    if (input_value_ >= values_.size() || external_output_ >= values_.size()) {
        set_detail(error_detail, "tensor dependency graph references an unknown value");
        return AdapterError::InvalidArgument;
    }

    std::vector<uint64_t> persistent_consumers(persistent_.size(), 0);
    std::vector<uint64_t> value_consumers(values_.size(), 0);
    std::vector<std::vector<std::string>> persistent_consumer_names(persistent_.size());
    for (const TensorWaveOp & operation : operations_) {
        if (operation.output_value >= values_.size()) {
            set_detail(error_detail, "operation produces an unknown value");
            return AdapterError::InvalidArgument;
        }
        for (const TensorWaveRef & input_ref : operation.inputs) {
            if (input_ref.kind == TensorWaveRef::Kind::Persistent) {
                if (input_ref.index >= persistent_.size()) {
                    set_detail(error_detail, "operation references an unknown persistent tensor");
                    return AdapterError::InvalidArgument;
                }
                ++persistent_consumers[input_ref.index];
                persistent_consumer_names[input_ref.index].push_back(operation.op_id);
            } else {
                if (input_ref.index >= values_.size()) {
                    set_detail(error_detail, "operation references an unknown value");
                    return AdapterError::InvalidArgument;
                }
                ++value_consumers[input_ref.index];
            }
        }
    }
    for (size_t index = 0; index < persistent_.size(); ++index) {
        if (persistent_consumers[index] == 0) {
            set_detail(error_detail, "persistent tensor has no real consumer");
            return AdapterError::InvalidArgument;
        }
        report->total_ffn_weight_bytes += persistent_[index].view.payload_len;
        report->persistent_lifetimes.push_back({
            persistent_[index].tensor_id, persistent_[index].name,
            persistent_[index].view.payload_len, persistent_consumers[index],
            persistent_consumer_names[index].front(),
            persistent_consumer_names[index].back(), 0, 0 });
    }

    std::unordered_map<uint32_t, RuntimeValue> values;
    values.emplace(input_value_, RuntimeValue{ input, {}, {} });
    std::unordered_map<uint32_t, ResidentTensor> resident;
    std::vector<uint64_t> remaining_persistent = persistent_consumers;
    std::vector<uint64_t> remaining_values = value_consumers;
    std::vector<bool> completed(operations_.size(), false);
    uint64_t active_weight_bytes = 0;
    uint64_t active_transient_bytes = input.payload_len;
    size_t completed_count = 0;

    const auto obtain_ready = [&](uint32_t tensor_ref, const PersistentTensorRef & persistent,
        MaterializedTensor * result) {
        if (materializer == nullptr || result == nullptr) return false;
        MaterializationState state = materializer->state(tensor_ref);
        if (state == MaterializationState::NotRequested || state == MaterializationState::Released) {
            if (!materializer->request(tensor_ref, persistent, persistent.view.payload_len)) {
                set_detail(error_detail, "persistent tensor materialization request failed");
                return false;
            }
            state = materializer->state(tensor_ref);
        }
        if (state == MaterializationState::InFlight) state = materializer->wait(tensor_ref);
        if (state != MaterializationState::Ready) {
            if (error_detail != nullptr) *error_detail = std::string("persistent tensor is not ready: ") +
                materialization_state_name(state);
            return false;
        }
        const auto ready = materializer->obtain_ready_tensor(tensor_ref);
        if (!ready.has_value()) {
            set_detail(error_detail, "persistent tensor reported ready without a payload");
            return false;
        }
        *result = *ready;
        return true;
    };

    const auto notify_planner = [&]() {
        if (!planning_observer) return;
        TensorWavePlannerState state;
        for (uint32_t index = 0; index < completed.size(); ++index) {
            if (completed[index]) state.completed_operations.push_back(index);
        }
        for (const auto & value : values) state.available_values.push_back(value.first);
        for (const auto & tensor : resident) state.resident_persistent.push_back(tensor.first);
        std::sort(state.available_values.begin(), state.available_values.end());
        std::sort(state.resident_persistent.begin(), state.resident_persistent.end());
        state.active_persistent_weight_bytes = active_weight_bytes;
        state.active_persistent_tensor_count = resident.size();
        planning_observer(state);
    };

    while (completed_count < operations_.size()) {
        notify_planner();
        size_t selected = operations_.size();
        std::vector<std::string> runnable_before;
        for (size_t index = 0; index < operations_.size(); ++index) {
            if (completed[index]) continue;
            bool runnable = true;
            for (const TensorWaveRef & input_ref : operations_[index].inputs) {
                if (input_ref.kind == TensorWaveRef::Kind::Value &&
                    !has_value(values, input_ref.index)) {
                    runnable = false;
                    break;
                }
            }
            if (runnable) {
                runnable_before.push_back(operations_[index].op_id);
                if (selected == operations_.size()) selected = index;
            }
        }
        if (selected == operations_.size()) {
            set_detail(error_detail, "tensor dependency graph has no runnable operation");
            return AdapterError::InvalidArgument;
        }

        const TensorWaveOp & operation = operations_[selected];
        TensorWaveTraceStep trace;
        trace.op_id = operation.op_id;
        trace.op_kind = operation.kind;
        trace.runnable_before = runnable_before;
        trace.active_persistent_weight_bytes = active_weight_bytes;
        trace.active_persistent_tensor_count = resident.size();
        trace.active_transient_bytes = active_transient_bytes;

        std::vector<MaterializedTensor> materialized_inputs;
        std::vector<uint32_t> materialized_refs;
        for (const TensorWaveRef & input_ref : operation.inputs) {
            if (input_ref.kind != TensorWaveRef::Kind::Persistent) continue;
            if (resident.find(input_ref.index) != resident.end()) continue;
            const PersistentTensorRef & persistent = persistent_[input_ref.index];
            VbufBorrowedStorage storage{};
            bool materialized = false;
            if (materializer != nullptr) {
                MaterializedTensor ready{};
                if (!obtain_ready(input_ref.index, persistent, &ready)) {
                    return AdapterError::InvalidArgument;
                }
                materialized_inputs.push_back(ready);
                materialized_refs.push_back(input_ref.index);
                storage = ready.storage;
                materialized = true;
            }
            if (!materialized) storage = storage_provider(persistent.view);
            ResidentTensor active{ storage, persistent.view.payload_len };
            resident.emplace(input_ref.index, active);
            trace.persistent_acquired.push_back(persistent.name);
            active_weight_bytes += persistent.view.payload_len;
            report->sum_acquired_weight_bytes += persistent.view.payload_len;
            TensorWaveLifetime & lifetime = report->persistent_lifetimes[input_ref.index];
            lifetime.acquire_step = report->trace.size() + 1;
        }
        trace.active_persistent_weight_bytes = active_weight_bytes;
        trace.active_persistent_tensor_count = resident.size();
        trace.active_persistent_weight_bytes_during = active_weight_bytes;
        trace.active_persistent_tensor_count_during = resident.size();
        report->peak_active_weight_bytes = std::max(report->peak_active_weight_bytes,
            active_weight_bytes);

        ggml_init_params params{};
        params.mem_size = 8 * 1024 * 1024;
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        ggml_context * context = ggml_init(params);
        if (context == nullptr) {
            set_detail(error_detail, "tensor wave graph context allocation failed");
            return AdapterError::ContextAllocationFailed;
        }

        AdapterError input_error = AdapterError::None;
        std::vector<std::unique_ptr<BorrowedGgmlTensor>> borrowed;
        std::unordered_map<uint32_t, ggml_tensor *> tensors;
        ggml_backend_t backend = nullptr;
        for (const TensorWaveRef & input_ref : operation.inputs) {
            const VbufTensorView * view = nullptr;
            VbufBorrowedStorage storage{};
            if (input_ref.kind == TensorWaveRef::Kind::Persistent) {
                view = &persistent_[input_ref.index].view;
                storage = resident.at(input_ref.index).storage;
                const auto found = std::find(materialized_refs.begin(), materialized_refs.end(),
                    input_ref.index);
                if (found != materialized_refs.end()) {
                    const MaterializedTensor & materialized = materialized_inputs[
                        static_cast<size_t>(found - materialized_refs.begin())];
                    view = &materialized.view;
                    storage = materialized.storage;
                } else if (materializer != nullptr) {
                    MaterializedTensor ready{};
                    if (!obtain_ready(input_ref.index, persistent_[input_ref.index], &ready)) {
                        ggml_free(context);
                        return AdapterError::InvalidArgument;
                    }
                    materialized_inputs.push_back(ready);
                    materialized_refs.push_back(input_ref.index);
                    const MaterializedTensor & materialized = materialized_inputs.back();
                    view = &materialized.view;
                    storage = materialized.storage;
                }
            } else {
                view = &values.at(input_ref.index).view;
                storage = storage_provider(*view);
                trace.input_values_consumed.push_back(values_[input_ref.index].name);
            }
            if (std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr &&
                operation.op_id == "ffn_rms_norm" && input_ref.kind == TensorWaveRef::Kind::Persistent) {
                const PersistentTensorRef & persistent = persistent_[input_ref.index];
                const uint8_t * actual = storage.base + storage.payload_offset;
                std::fprintf(stderr,
                    "FFN_NORM_WEIGHT_AUDIT name=%s tensor_id=%llu source_offset=%llu "
                    "representation=%u rank=%u elements=%llu bytes=%llu raw_hash=%016llx "
                    "bound_hash=%016llx bound_representation=%u bound_rank=%u\n",
                    persistent.name.c_str(), static_cast<unsigned long long>(persistent.tensor_id),
                    static_cast<unsigned long long>(persistent.source_offset), persistent.view.representation,
                    persistent.view.rank, static_cast<unsigned long long>(persistent.view.payload_len / sizeof(float)),
                    static_cast<unsigned long long>(persistent.view.payload_len),
                    static_cast<unsigned long long>(audit_hash(persistent.view.payload, persistent.view.payload_len)),
                    static_cast<unsigned long long>(audit_hash(actual, view->payload_len)),
                    view->representation, view->rank);
                if (view->payload_len >= 8 * sizeof(float)) {
                    const float * values = reinterpret_cast<const float *>(actual);
                    std::fprintf(stderr, "FFN_NORM_WEIGHT_AUDIT_FIRST8=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
                }
            }
            if (std::getenv("VBUF_AUDIT_DENSE_FFN") != nullptr &&
                (operation.op_id == "expert_up_matmul" || operation.op_id == "expert_gate_matmul" ||
                 operation.op_id == "expert_down_matmul") && input_ref.kind == TensorWaveRef::Kind::Persistent) {
                const PersistentTensorRef & persistent = persistent_[input_ref.index];
                const uint8_t * actual = storage.base + storage.payload_offset;
                std::fprintf(stderr,
                    "DENSE_FFN_WEIGHT_AUDIT op=%s name=%s tensor_id=%llu source_offset=%llu "
                    "representation=%u rank=%u dims=%llu,%llu bytes=%llu raw_hash=%016llx bound_hash=%016llx\n",
                    operation.op_id.c_str(), persistent.name.c_str(),
                    static_cast<unsigned long long>(persistent.tensor_id),
                    static_cast<unsigned long long>(persistent.source_offset), persistent.view.representation,
                    persistent.view.rank, static_cast<unsigned long long>(persistent.view.dimensions[0]),
                    static_cast<unsigned long long>(persistent.view.dimensions[1]),
                    static_cast<unsigned long long>(persistent.view.payload_len),
                    static_cast<unsigned long long>(audit_hash(persistent.view.payload, persistent.view.payload_len)),
                    static_cast<unsigned long long>(audit_hash(actual, view->payload_len)));
            }
            auto tensor = BorrowedGgmlTensor::create(*view, &input_error, error_detail);
            if (!tensor) {
                ggml_free(context);
                return input_error;
            }
            const bool diagnostic_repack = std::getenv("VBUF_AUDIT_REPACKED_DOWN") != nullptr &&
                operation.op_id == "expert_down_matmul" && input_ref.kind == TensorWaveRef::Kind::Persistent &&
                persistent_[input_ref.index].name == "blk.1.ffn_down.weight";
            const AdapterError binding_error = diagnostic_repack
                ? tensor->bind_cpu_repacked(storage, error_detail)
                : tensor->bind_cpu(storage, error_detail);
            if (binding_error != AdapterError::None) {
                ggml_free(context);
                return binding_error;
            }
            if (std::getenv("VBUF_REPACK_V3_GATE_ONLY") != nullptr && diagnostic_repack) {
                const ggml_tensor * repacked = tensor->tensor();
                std::fprintf(stderr,
                    "VBUF_REPACK_V3_POST_SET type=%s ne=[%lld,%lld,%lld,%lld] "
                    "nb=[%zu,%zu,%zu,%zu] nbytes=%zu data=%p buffer=%p\n",
                    ggml_type_name(repacked->type),
                    static_cast<long long>(repacked->ne[0]), static_cast<long long>(repacked->ne[1]),
                    static_cast<long long>(repacked->ne[2]), static_cast<long long>(repacked->ne[3]),
                    repacked->nb[0], repacked->nb[1], repacked->nb[2], repacked->nb[3],
                    ggml_nbytes(repacked), repacked->data, tensor->buffer());
                std::fprintf(stderr,
                    "VBUF_REPACK_V3_FIRST_FAILED_GATE=WORKSPACE_CONTRACT_PARITY "
                    "llama_workspace_bytes=UNKNOWN vbuf_workspace_bytes=UNKNOWN\n");
                ggml_free(context);
                set_detail(error_detail, "vBuf intervention 3 stopped before kernel: workspace contract unknown");
                return AdapterError::BackendAllocationFailed;
            }
            backend = tensor->backend();
            if (std::getenv("VBUF_AUDIT_LOCAL_KERNEL") != nullptr) {
                ggml_backend_cpu_set_n_threads(backend, 2);
            }
            tensors.emplace(static_cast<uint32_t>(&input_ref - operation.inputs.data()), tensor->tensor());
            borrowed.push_back(std::move(tensor));
        }
        if (std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr && operation.op_id == "ffn_rms_norm") {
            std::vector<float> input_values(ggml_nelements(tensors.at(0)));
            ggml_backend_tensor_get(tensors.at(0), input_values.data(), 0, input_values.size() * sizeof(float));
            std::fprintf(stderr, "FFN_NORM_INPUT_FIRST8=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                input_values[0], input_values[1], input_values[2], input_values[3], input_values[4], input_values[5], input_values[6], input_values[7]);
        }

        ggml_tensor * result = nullptr;
        if (operation.kind == TensorWaveOpKind::RmsNorm) {
            result = ggml_mul(context, ggml_rms_norm(context, tensors.at(0), operation.parameter),
                tensors.at(1));
        } else if (operation.kind == TensorWaveOpKind::MulMat) {
            result = ggml_mul_mat(context, tensors.at(0), tensors.at(1));
        } else if (operation.kind == TensorWaveOpKind::SwiGlu) {
            result = ggml_swiglu_split(context, tensors.at(0), tensors.at(1));
        } else if (operation.kind == TensorWaveOpKind::Dequantize) {
            result = ggml_cast(context, tensors.at(0), GGML_TYPE_F32);
        }
        if (result == nullptr) {
            ggml_free(context);
            set_detail(error_detail, "tensor wave operation construction failed");
            return AdapterError::TensorConstructionFailed;
        }
        ggml_cgraph * graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, result);
        ggml_backend_buffer_t compute = ggml_backend_alloc_ctx_tensors(context, backend);
        if (execution_observer) execution_observer(operation.op_id.c_str(), "start");
        const ggml_status compute_status = compute == nullptr
            ? GGML_STATUS_FAILED : ggml_backend_graph_compute(backend, graph);
        if (compute_status != GGML_STATUS_SUCCESS) {
            if (compute != nullptr) ggml_backend_buffer_free(compute);
            ggml_free(context);
            set_detail(error_detail, "tensor wave operation execution failed");
            return AdapterError::BackendAllocationFailed;
        }
        ggml_backend_synchronize(backend);
        if (std::getenv("VBUF_AUDIT_FFN_NORM") != nullptr && operation.op_id == "ffn_rms_norm") {
            ggml_tensor * unweighted = result->src[0];
            std::vector<float> values(ggml_nelements(unweighted));
            ggml_backend_tensor_get(unweighted, values.data(), 0, values.size() * sizeof(float));
            std::fprintf(stderr, "FFN_NORM_UNWEIGHTED_FIRST8=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
                values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
        }
        if (execution_observer) execution_observer(operation.op_id.c_str(), "end");

        RuntimeValue produced;
        produced.bytes = std::make_shared<std::vector<uint8_t>>(ggml_nbytes(result));
        ggml_backend_tensor_get(result, produced.bytes->data(), 0, produced.bytes->size());
        if (std::getenv("VBUF_AUDIT_DENSE_FFN") != nullptr &&
            (operation.op_id == "expert_up_matmul" || operation.op_id == "expert_gate_matmul" ||
             operation.op_id == "expert_swiglu" || operation.op_id == "expert_down_matmul")) {
            const size_t count = produced.bytes->size() / sizeof(float);
            const float * values = reinterpret_cast<const float *>(produced.bytes->data());
            bool finite = true;
            for (size_t index = 0; index < count; ++index) finite = finite && std::isfinite(values[index]);
            std::fprintf(stderr, "DENSE_FFN_BOUNDARY op=%s elements=%zu hash=%016llx first8=",
                operation.op_id.c_str(), count,
                static_cast<unsigned long long>(audit_hash(produced.bytes->data(), produced.bytes->size())));
            for (size_t i = 0; i < std::min<size_t>(8, count); ++i)
                std::fprintf(stderr, "%s%.9g", i ? "," : "", values[i]);
            std::fprintf(stderr, " finite=%s\n", finite ? "YES" : "NO");
            const char * file = operation.op_id == "expert_up_matmul" ? "/tmp/poc22-vbuf-ffn-up.f32" :
                operation.op_id == "expert_gate_matmul" ? "/tmp/poc22-vbuf-ffn-gate.f32" :
                operation.op_id == "expert_swiglu" ? "/tmp/poc22-vbuf-ffn-swiglu.f32" :
                "/tmp/poc22-vbuf-ffn-out.f32";
            std::ofstream(file, std::ios::binary).write(
                reinterpret_cast<const char *>(produced.bytes->data()), produced.bytes->size());
        }
        produced.dimensions = std::make_shared<std::vector<uint64_t>>();
        produced.dimensions->assign(result->ne, result->ne + GGML_MAX_DIMS);
        produced.view = { 0, 2, produced.dimensions->data(), produced.bytes->data(),
            produced.bytes->size() };
        values[operation.output_value] = produced;
        active_transient_bytes += produced.bytes->size();
        trace.output_values_produced.push_back(values_[operation.output_value].name);
        if (operation.output_value == external_output_) {
            *output = *produced.bytes;
            output_shape->assign(result->ne, result->ne + GGML_MAX_DIMS);
            report->external_output_preserved = true;
        }
        if (operation.output_value == capture_value_) {
            report->captured_value = *produced.bytes;
            report->captured_value_shape.assign(result->ne, result->ne + GGML_MAX_DIMS);
        }

        ggml_backend_buffer_free(compute);
        ggml_free(context);
        borrowed.clear();

        for (const TensorWaveRef & input_ref : operation.inputs) {
            if (input_ref.kind == TensorWaveRef::Kind::Persistent) {
                --remaining_persistent[input_ref.index];
                if (remaining_persistent[input_ref.index] == 0) {
                    const PersistentTensorRef & persistent = persistent_[input_ref.index];
                    trace.persistent_released.push_back(persistent.name);
                    active_weight_bytes -= persistent.view.payload_len;
                    resident.erase(input_ref.index);
                    if (materializer != nullptr &&
                        std::find(materialized_refs.begin(), materialized_refs.end(),
                            input_ref.index) != materialized_refs.end()) {
                        materializer->release(input_ref.index);
                    }
                    report->persistent_lifetimes[input_ref.index].release_step =
                        report->trace.size() + 1;
                }
            } else {
                --remaining_values[input_ref.index];
                if (remaining_values[input_ref.index] == 0 &&
                    input_ref.index != external_output_) {
                    trace.values_released.push_back(values_[input_ref.index].name);
                    active_transient_bytes -= values.at(input_ref.index).view.payload_len;
                    values.erase(input_ref.index);
                }
            }
        }
        trace.active_persistent_weight_bytes = active_weight_bytes;
        trace.active_persistent_tensor_count = resident.size();
        trace.active_transient_bytes = active_transient_bytes;
        report->active_transient_bytes = active_transient_bytes;
        report->trace.push_back(std::move(trace));
        completed[selected] = true;
        ++completed_count;
    }

    report->all_persistent_released = resident.empty() && active_weight_bytes == 0;
    report->active_transient_bytes = active_transient_bytes;
    return report->all_persistent_released && report->external_output_preserved
        ? AdapterError::None : AdapterError::BackendAllocationFailed;
}

} // namespace vbuf_ggml
