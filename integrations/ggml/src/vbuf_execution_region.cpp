#include "vbuf_execution_region.h"

#include <limits>
#include <string>
#include <unordered_map>

namespace vbuf_ggml {
namespace {

void detail(std::string * value, const char * message) {
    if (value != nullptr) *value = message;
}

} // namespace

uint32_t ExecutionRegion::add_weight(uint64_t tensor_id, const VbufTensorView & view) {
    const uint32_t slot = next_slot_++;
    weights_.push_back({ tensor_id, slot, view });
    return slot;
}

uint32_t ExecutionRegion::add_input() {
    input_slot_ = next_slot_++;
    return input_slot_;
}

uint32_t ExecutionRegion::add_temporary() {
    return next_slot_++;
}

uint32_t ExecutionRegion::add_output() {
    output_slot_ = next_slot_++;
    return output_slot_;
}

void ExecutionRegion::add_operation(const RegionOperation & operation) {
    operations_.push_back(operation);
}

AdapterError ExecutionRegion::execute(
    const VbufTensorView & input,
    const RegionStorageProvider & storage_provider,
    std::vector<uint8_t> * output,
    std::vector<int64_t> * output_shape,
    RegionExecutionReport * report,
    std::string * error_detail,
    const RegionPhaseObserver & observer) const {
    if (output == nullptr || output_shape == nullptr || report == nullptr ||
        !storage_provider || input_slot_ == UINT32_MAX || output_slot_ == UINT32_MAX ||
        operations_.empty()) {
        detail(error_detail, "region has incomplete execution contract");
        return AdapterError::InvalidArgument;
    }

    report->requested_tensor_ids.clear();
    report->acquired_tensor_ids.clear();
    report->acquired_weight_bytes = 0;
    report->selected_weight_bytes = 0;
    report->graph_tensor_count = 0;

    std::unordered_map<uint32_t, ggml_tensor *> tensors;
    std::vector<std::unique_ptr<BorrowedGgmlTensor>> borrowed;
    borrowed.reserve(weights_.size() + 1);
    for (const RegionWeight & weight : weights_) {
        AdapterError creation_error = AdapterError::None;
        auto tensor = BorrowedGgmlTensor::create(weight.view, &creation_error, error_detail);
        if (!tensor) return creation_error;
        const VbufBorrowedStorage storage = storage_provider(weight.view);
        const AdapterError binding_error = tensor->bind_cpu(storage, error_detail);
        if (binding_error != AdapterError::None) return binding_error;
        report->requested_tensor_ids.push_back(weight.tensor_id);
        report->acquired_tensor_ids.push_back(weight.tensor_id);
        report->selected_weight_bytes += weight.view.payload_len;
        report->acquired_weight_bytes += weight.view.payload_len;
        tensors.emplace(weight.slot, tensor->tensor());
        borrowed.push_back(std::move(tensor));
    }

    AdapterError input_error = AdapterError::None;
    auto input_tensor = BorrowedGgmlTensor::create(input, &input_error, error_detail);
    if (!input_tensor) return input_error;
    const AdapterError input_binding_error = input_tensor->bind_cpu(
        storage_provider(input), error_detail);
    if (input_binding_error != AdapterError::None) return input_binding_error;
    ggml_backend_t backend = input_tensor->backend();
    tensors.emplace(input_slot_, input_tensor->tensor());
    borrowed.push_back(std::move(input_tensor));
    if (observer) observer("weights_acquired", report->acquired_weight_bytes,
        static_cast<uint64_t>(weights_.size()));

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        detail(error_detail, "region graph context allocation failed");
        return AdapterError::ContextAllocationFailed;
    }
    std::unordered_map<uint32_t, ggml_tensor *> graph_tensors = tensors;
    for (const RegionOperation & operation : operations_) {
        auto a = graph_tensors.find(operation.input_a_slot);
        auto b = graph_tensors.find(operation.input_b_slot);
        if (a == graph_tensors.end() ||
            (operation.kind != RegionOpKind::RmsNorm && b == graph_tensors.end())) {
            ggml_free(context);
            detail(error_detail, "region operation references an unknown slot");
            return AdapterError::InvalidArgument;
        }
        ggml_tensor * result = nullptr;
        switch (operation.kind) {
        case RegionOpKind::RmsNorm:
            result = ggml_mul(context, ggml_rms_norm(context, a->second, operation.parameter), b->second);
            break;
        case RegionOpKind::MulMat:
            result = ggml_mul_mat(context, a->second, b->second);
            break;
        case RegionOpKind::SwiGluSplit:
            result = ggml_swiglu_split(context, a->second, b->second);
            break;
        }
        if (result == nullptr) {
            ggml_free(context);
            detail(error_detail, "region operation construction failed");
            return AdapterError::TensorConstructionFailed;
        }
        graph_tensors[operation.output_slot] = result;
    }
    auto final = graph_tensors.find(output_slot_);
    if (final == graph_tensors.end()) {
        ggml_free(context);
        detail(error_detail, "region output slot was not produced");
        return AdapterError::InvalidArgument;
    }
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, final->second);
    ggml_backend_buffer_t compute = ggml_backend_alloc_ctx_tensors(context, backend);
    if (compute == nullptr || ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        if (compute != nullptr) ggml_backend_buffer_free(compute);
        ggml_free(context);
        detail(error_detail, "region graph execution failed");
        return AdapterError::BackendAllocationFailed;
    }
    if (observer) observer("executing", report->acquired_weight_bytes,
        static_cast<uint64_t>(weights_.size()));
    ggml_backend_synchronize(backend);
    if (observer) observer("synchronized", report->acquired_weight_bytes,
        static_cast<uint64_t>(weights_.size()));
    const size_t bytes = ggml_nbytes(final->second);
    output->resize(bytes);
    ggml_backend_tensor_get(final->second, output->data(), 0, bytes);
    output_shape->assign(final->second->ne, final->second->ne + GGML_MAX_DIMS);
    report->graph_tensor_count = static_cast<uint64_t>(ggml_graph_n_nodes(graph));
    ggml_backend_buffer_free(compute);
    ggml_free(context);
    borrowed.clear();
    if (observer) observer("released", 0, 0);
    return AdapterError::None;
}

} // namespace vbuf_ggml
