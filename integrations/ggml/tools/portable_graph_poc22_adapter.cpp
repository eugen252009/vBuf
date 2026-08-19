#include "vbuf_portable_graph_adapter.h"

#include <cstring>
#include <unordered_map>

namespace vbuf_ggml {
namespace {

bool fail(std::string * error, const char * message) {
    if (error != nullptr) *error = message;
    return false;
}

bool find_value(const VbufGraphOperationDesc & operation, uint32_t * value,
    std::string * error) {
    if (operation.inputs == nullptr) return fail(error, "operation input is missing");
    for (uint32_t i = 0; i < operation.input_count; ++i) {
        if (operation.inputs[i].kind == 0) {
            if (value != nullptr) *value = operation.inputs[i].id;
            return true;
        }
    }
    return fail(error, "operation value reference is missing");
}

bool find_tensor(const VbufGraphOperationDesc & operation, uint32_t * tensor_id,
    std::string * error) {
    if (operation.inputs == nullptr) return fail(error, "operation input is missing");
    for (uint32_t i = 0; i < operation.input_count; ++i) {
        if (operation.inputs[i].kind == 1) {
            if (tensor_id != nullptr) *tensor_id = operation.inputs[i].id;
            return true;
        }
    }
    return fail(error, "operation tensor binding is missing");
}

bool floats(const std::vector<uint8_t> & bytes, std::vector<float> * output,
    std::string * error) {
    if (bytes.size() % sizeof(float) != 0) return fail(error, "backend output is not f32");
    output->resize(bytes.size() / sizeof(float));
    if (!bytes.empty()) std::memcpy(output->data(), bytes.data(), bytes.size());
    return true;
}

bool run_backend_op(const VbufGraphOperationDesc & operation, const PortableActivation & input,
    const PortableResolvedTensor & tensor, const std::shared_ptr<const void> & lease,
    PortableActivation * output, std::string * error) {
    if (tensor.ref.view.payload == nullptr || tensor.ref.view.payload_len == 0)
        return fail(error, "resolved tensor payload is not ready");
    TensorDependencyExecutor backend;
    const uint32_t input_value = backend.add_input("portable_input");
    const uint32_t persistent = backend.add_persistent(tensor.ref);
    const uint32_t output_value = backend.add_value("portable_output", true);
    backend.set_external_output(output_value);
    if (operation.kind == 1) {
        if (operation.has_epsilon == 0) return fail(error, "RmsNorm epsilon is missing");
        backend.add_operation({ "portable_rms_norm", TensorWaveOpKind::RmsNorm,
            { { TensorWaveRef::Kind::Value, input_value },
              { TensorWaveRef::Kind::Persistent, persistent } }, output_value,
            operation.epsilon });
    } else if (operation.kind == 2) {
        if (operation.matmul_weight_operand == 0 || operation.matmul_transpose_weight != 1)
            return fail(error, "MatMul semantics are incomplete for this backend");
        backend.add_operation({ "portable_matmul", TensorWaveOpKind::MulMat,
            { { TensorWaveRef::Kind::Persistent, persistent },
              { TensorWaveRef::Kind::Value, input_value } }, output_value });
    } else {
        return fail(error, "unsupported backend operation");
    }
    uint64_t storage_calls = 0;
    const auto provider = [lease, &storage_calls](const VbufTensorView & view) {
        ++storage_calls;
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{ reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), lease };
    };
    std::vector<uint8_t> raw;
    std::vector<int64_t> shape;
    TensorWaveReport report;
    std::string detail;
    const VbufTensorView input_view{ 0, 2, input.dimensions.data(),
        reinterpret_cast<const uint8_t *>(input.values.data()),
        input.values.size() * sizeof(float) };
    const AdapterError status = backend.execute(input_view, provider, &raw, &shape,
        &report, &detail);
    if (status != AdapterError::None) return fail(error, detail.c_str());
    if (!floats(raw, &output->values, error)) return false;
    output->dimensions.clear();
    for (int64_t dimension : shape) {
        if (dimension < 0) return fail(error, "backend returned a negative dimension");
        output->dimensions.push_back(static_cast<uint64_t>(dimension));
    }
    return true;
}

} // namespace

bool execute_portable_graph(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, PortableRouterPrefixResult * result,
    std::string * error) {
    if (graph == nullptr || result == nullptr || !resolver || !lease)
        return fail(error, "portable graph execution argument is invalid");
    if (vbuf_runtime_graph_abi_version() != 1)
        return fail(error, "portable graph ABI version is unsupported");
    uint32_t graph_input = 0, graph_output = 0;
    if (vbuf_runtime_graph_input_value(graph, &graph_input) != 0 ||
        vbuf_runtime_graph_output_value(graph, &graph_output) != 0)
        return fail(error, "portable graph boundary is invalid");
    std::unordered_map<uint32_t, PortableActivation> values;
    values.emplace(graph_input, input);
    const uint32_t count = vbuf_runtime_graph_operation_count(graph);
    for (uint32_t index = 0; index < count; ++index) {
        VbufGraphOperationDesc operation{};
        if (vbuf_runtime_graph_operation_desc(graph, index, &operation) != 0)
            return fail(error, "portable operation descriptor is invalid");
        uint32_t value_id = UINT32_MAX;
        if (!find_value(operation, &value_id, error)) return false;
        const auto value = values.find(value_id);
        if (value == values.end()) return fail(error, "portable value is unavailable");
        if (operation.kind == 1 || operation.kind == 2) {
            uint32_t tensor_id = 0;
            if (!find_tensor(operation, &tensor_id, error)) return false;
            PortableResolvedTensor resolved;
            if (!resolver(tensor_id, &resolved, error)) return false;
            PortableActivation output;
            if (!run_backend_op(operation, value->second, resolved, lease, &output, error)) return false;
            values[operation.output] = std::move(output);
            if (operation.kind == 1) result->normalized = values[operation.output].values;
            else result->logits = values[operation.output].values;
        } else if (operation.kind == 4) {
            if (operation.input_count != 1 || operation.top_k_order != 1 ||
                operation.top_k_tie_break != 1 || operation.has_top_k == 0 ||
                operation.output != graph_output)
                return fail(error, "portable TopK semantics are incomplete");
            std::string topk_error;
            if (!deterministic_top_k(value->second.values,
                    static_cast<uint32_t>(value->second.values.size()), operation.top_k,
                    &result->selection, &topk_error))
                return fail(error, topk_error.c_str());
        } else {
            return fail(error, "portable operation kind is unsupported");
        }
    }
    if (result->selection.ids.empty()) return fail(error, "portable TopK produced no selection");
    return true;
}

} // namespace vbuf_ggml
