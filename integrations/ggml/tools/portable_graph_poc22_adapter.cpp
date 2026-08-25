#include "vbuf_portable_graph_adapter.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace vbuf_ggml {

struct VbufPortableExecutionState {
    uint32_t binding_id = 0;
    uint64_t capacity = 0;
    uint64_t length = 0;
    uint64_t batch_size = 0;
    uint64_t kv_head_count = 0;
    uint64_t head_dim = 0;
    std::vector<float> key;
    std::vector<float> value;
};

extern "C" VbufPortableExecutionState * vbuf_portable_execution_state_create(
    uint32_t binding_id, uint64_t capacity) {
    if (capacity == 0) return nullptr;
    return new VbufPortableExecutionState{ binding_id, capacity };
}

extern "C" void vbuf_portable_execution_state_reset(VbufPortableExecutionState * state) {
    if (state == nullptr) return;
    state->length = 0;
    state->batch_size = 0;
    state->kv_head_count = 0;
    state->head_dim = 0;
    state->key.clear();
    state->value.clear();
}

extern "C" void vbuf_portable_execution_state_destroy(VbufPortableExecutionState * state) {
    delete state;
}

extern "C" uint64_t vbuf_portable_execution_state_length(
    const VbufPortableExecutionState * state) {
    return state == nullptr ? 0 : state->length;
}

extern "C" uint64_t vbuf_portable_execution_state_bytes(
    const VbufPortableExecutionState * state) {
    if (state == nullptr) return 0;
    if (state->key.size() > std::numeric_limits<uint64_t>::max() / sizeof(float) ||
        state->value.size() > std::numeric_limits<uint64_t>::max() / sizeof(float))
        return 0;
    const uint64_t key_bytes = static_cast<uint64_t>(state->key.size()) * sizeof(float);
    const uint64_t value_bytes = static_cast<uint64_t>(state->value.size()) * sizeof(float);
    if (key_bytes > std::numeric_limits<uint64_t>::max() - value_bytes) return 0;
    return key_bytes + value_bytes;
}

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

bool find_values(const VbufGraphOperationDesc & operation, std::vector<uint32_t> * values,
    std::string * error) {
    if (operation.inputs == nullptr) return fail(error, "operation input is missing");
    values->clear();
    for (uint32_t i = 0; i < operation.input_count; ++i) {
        if (operation.inputs[i].kind == 0) values->push_back(operation.inputs[i].id);
    }
    return true;
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
    } else if (operation.kind == 2 || operation.kind == 5) {
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

bool run_activation(const VbufGraphOperationDesc & operation, const PortableActivation & input,
    PortableActivation * output, std::string * error) {
    if (operation.reserved[0] != VBUF_ACTIVATION_SILU)
        return fail(error, "activation kind is unsupported");
    output->dimensions = input.dimensions;
    output->values.resize(input.values.size());
    for (size_t index = 0; index < input.values.size(); ++index) {
        const float value = input.values[index];
        output->values[index] = value / (1.0f + std::exp(-value));
    }
    return true;
}

bool run_residual(const PortableActivation & lhs, const PortableActivation & rhs,
    PortableActivation * output, std::string * error) {
    if (lhs.values.size() != rhs.values.size() || lhs.dimensions != rhs.dimensions)
        return fail(error, "residual operands have incompatible geometry");
    output->dimensions = lhs.dimensions;
    output->values.resize(lhs.values.size());
    for (size_t index = 0; index < lhs.values.size(); ++index)
        output->values[index] = lhs.values[index] + rhs.values[index];
    return true;
}

bool checked_elements(const std::vector<uint64_t> & dimensions, uint64_t * result,
    std::string * error) {
    uint64_t elements = 1;
    for (uint64_t dimension : dimensions) {
        if (dimension == 0 || elements > std::numeric_limits<uint64_t>::max() / dimension)
            return fail(error, "attention shape product overflows");
        elements *= dimension;
    }
    *result = elements;
    return true;
}

bool activation_from_tensor(const PersistentTensorRef & tensor, PortableActivation * output,
    std::string * error) {
    if (tensor.view.representation != 0 || tensor.view.rank != 4 ||
        tensor.view.dimensions == nullptr || tensor.view.payload == nullptr)
        return fail(error, "attention tensor input must be a rank-4 f32 payload");
    output->dimensions.assign(tensor.view.dimensions, tensor.view.dimensions + tensor.view.rank);
    uint64_t elements = 0;
    if (!checked_elements(output->dimensions, &elements, error) ||
        elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
        tensor.view.payload_len != elements * sizeof(float))
        return fail(error, "attention tensor input payload length is invalid");
    output->values.resize(static_cast<size_t>(elements));
    std::memcpy(output->values.data(), tensor.view.payload, tensor.view.payload_len);
    return true;
}

bool same_dimensions(const std::vector<uint64_t> & actual,
    std::initializer_list<uint64_t> expected, std::string * error) {
    if (actual.size() != expected.size() ||
        !std::equal(actual.begin(), actual.end(), expected.begin()))
        return fail(error, "attention tensor dimensions are invalid");
    return true;
}

size_t attention_index(uint64_t batch, uint64_t position, uint64_t head, uint64_t component,
    uint64_t sequence, uint64_t heads, uint64_t dimension) {
    return static_cast<size_t>((((batch * sequence + position) * heads + head) * dimension) + component);
}

size_t state_index(uint64_t batch, uint64_t position, uint64_t head, uint64_t component,
    const VbufPortableExecutionState & state) {
    return attention_index(batch, position, head, component, state.length,
        state.kv_head_count, state.head_dim);
}

bool run_attention(const VbufGraphOperationDescV2 & operation,
    const std::vector<PortableActivation> & inputs,
    VbufPortableExecutionState * state, PortableActivation * output,
    PortableRouterPrefixResult::AttentionReport * report, std::string * error) {
    if (state == nullptr) return fail(error, "attention execution state is missing");
    const VbufAttentionDesc & attributes = operation.attention;
    if (attributes.mask_kind != 0 && attributes.mask_kind != 1)
        return fail(error, "attention mask semantics are unsupported");
    if (attributes.position_kind != 1)
        return fail(error, "attention position semantics are unsupported");
    if (attributes.batch_size == 0 || attributes.query_head_count == 0 ||
        attributes.kv_head_count == 0 || attributes.head_dim == 0 ||
        attributes.query_length == 0 ||
        attributes.query_head_count % attributes.kv_head_count != 0 ||
        !std::isfinite(attributes.scale) || attributes.scale <= 0.0f)
        return fail(error, "attention geometry or scale is invalid");
    if (state->binding_id != attributes.state_id)
        return fail(error, "attention state binding does not match execution state");
    if (state->length > state->capacity ||
        attributes.query_length > state->capacity - state->length)
        return fail(error, "attention state capacity is exceeded");
    if (attributes.current_kv_length != state->length + attributes.query_length)
        return fail(error, "attention state length is inconsistent");

    if (inputs.size() != 3)
        return fail(error, "attention requires Q, K, and V operands");
    const PortableActivation & query = inputs[0];
    const PortableActivation & key = inputs[1];
    const PortableActivation & value = inputs[2];
    if (!same_dimensions(query.dimensions, { attributes.batch_size, attributes.query_length,
            attributes.query_head_count, attributes.head_dim }, error) ||
        !same_dimensions(key.dimensions, { attributes.batch_size, attributes.query_length,
            attributes.kv_head_count, attributes.head_dim }, error) ||
        !same_dimensions(value.dimensions, { attributes.batch_size, attributes.query_length,
            attributes.kv_head_count, attributes.head_dim }, error)) return false;
    uint64_t query_elements = 0;
    if (!checked_elements(query.dimensions, &query_elements, error) ||
        query_elements != query.values.size()) return fail(error, "Q payload length is invalid");
    uint64_t key_elements = 0;
    if (!checked_elements(key.dimensions, &key_elements, error) ||
        key_elements != key.values.size() || value.values.size() != key.values.size())
        return fail(error, "K/V payload length is invalid");
    if (state->length != 0 &&
        (state->batch_size != attributes.batch_size ||
         state->kv_head_count != attributes.kv_head_count ||
         state->head_dim != attributes.head_dim))
        return fail(error, "attention state geometry does not match operation");
    if (state->length != 0) {
        uint64_t state_elements = 0;
        if (!checked_elements({ attributes.batch_size, state->length,
                attributes.kv_head_count, attributes.head_dim }, &state_elements, error) ||
            state_elements != state->key.size() || state_elements != state->value.size())
            return fail(error, "attention state payload length is invalid");
    }

    const uint64_t visible_length = attributes.current_kv_length;
    uint64_t score_elements = 0;
    if (!checked_elements({ attributes.batch_size, attributes.query_length,
            attributes.query_head_count, visible_length }, &score_elements, error) ||
        score_elements > std::numeric_limits<size_t>::max() / sizeof(float))
        return fail(error, "attention score allocation overflows host size");
    output->dimensions = { attributes.batch_size, attributes.query_length,
        attributes.query_head_count, attributes.head_dim };
    output->values.assign(static_cast<size_t>(query_elements), 0.0f);
    if (report != nullptr) {
        report->q_bytes = query.values.size() * sizeof(float);
        report->k_bytes = key.values.size() * sizeof(float);
        report->v_bytes = value.values.size() * sizeof(float);
        report->score_bytes = score_elements * sizeof(float);
        report->probability_bytes = report->score_bytes;
        report->output_bytes = output->values.size() * sizeof(float);
    }

    std::vector<float> next_key = state->key;
    std::vector<float> next_value = state->value;
    if (state->length == 0) {
        next_key.clear();
        next_value.clear();
    }
    uint64_t next_elements = 0;
    if (!checked_elements({ attributes.batch_size, visible_length,
            attributes.kv_head_count, attributes.head_dim }, &next_elements, error) ||
        next_elements > std::numeric_limits<size_t>::max())
        return fail(error, "attention state allocation overflows host size");
    next_key.reserve(static_cast<size_t>(next_elements));
    next_value.reserve(next_key.capacity());
    next_key.insert(next_key.end(), key.values.begin(), key.values.end());
    next_value.insert(next_value.end(), value.values.begin(), value.values.end());

    const uint64_t group = attributes.query_head_count / attributes.kv_head_count;
    std::vector<float> scores(static_cast<size_t>(visible_length));
    std::vector<float> probabilities(static_cast<size_t>(visible_length));
    for (uint64_t batch = 0; batch < attributes.batch_size; ++batch) {
        for (uint64_t query_position = 0; query_position < attributes.query_length; ++query_position) {
            const uint64_t first_allowed = 0;
            const uint64_t last_allowed = attributes.mask_kind == 1
                ? state->length + query_position : visible_length - 1;
            for (uint64_t query_head = 0; query_head < attributes.query_head_count; ++query_head) {
                const uint64_t kv_head = query_head / group;
                float maximum = -std::numeric_limits<float>::infinity();
                for (uint64_t key_position = first_allowed; key_position <= last_allowed; ++key_position) {
                    float dot = 0.0f;
                    for (uint64_t component = 0; component < attributes.head_dim; ++component) {
                        const size_t q_index = attention_index(batch, query_position, query_head,
                            component, attributes.query_length, attributes.query_head_count,
                            attributes.head_dim);
                        const size_t k_index = key_position < state->length
                            ? state_index(batch, key_position, kv_head, component, *state)
                            : attention_index(batch, key_position - state->length, kv_head, component,
                                attributes.query_length, attributes.kv_head_count,
                                attributes.head_dim);
                        const float q = query.values[q_index];
                        const float k = key_position < state->length
                            ? state->key[k_index] : key.values[k_index];
                        if (!std::isfinite(q) || !std::isfinite(k))
                            return fail(error, "attention input contains a non-finite value");
                        dot += q * k;
                    }
                    scores[static_cast<size_t>(key_position)] = dot * attributes.scale;
                    maximum = std::max(maximum, scores[static_cast<size_t>(key_position)]);
                }
                float total = 0.0f;
                for (uint64_t key_position = first_allowed; key_position <= last_allowed; ++key_position) {
                    const float probability = std::exp(scores[static_cast<size_t>(key_position)] - maximum);
                    probabilities[static_cast<size_t>(key_position)] = probability;
                    total += probability;
                }
                if (!std::isfinite(total) || total <= 0.0f)
                    return fail(error, "attention softmax normalization failed");
                for (uint64_t key_position = first_allowed; key_position <= last_allowed; ++key_position)
                    probabilities[static_cast<size_t>(key_position)] /= total;
                for (uint64_t component = 0; component < attributes.head_dim; ++component) {
                    float accumulated = 0.0f;
                    for (uint64_t key_position = first_allowed; key_position <= last_allowed; ++key_position) {
                        const size_t v_index = key_position < state->length
                            ? state_index(batch, key_position, kv_head, component, *state)
                            : attention_index(batch, key_position - state->length, kv_head, component,
                                attributes.query_length, attributes.kv_head_count,
                                attributes.head_dim);
                        const float v = key_position < state->length
                            ? state->value[v_index] : value.values[v_index];
                        if (!std::isfinite(v)) return fail(error, "attention input contains a non-finite value");
                        accumulated += probabilities[static_cast<size_t>(key_position)] * v;
                    }
                    const size_t output_index = attention_index(batch, query_position, query_head,
                        component, attributes.query_length, attributes.query_head_count,
                        attributes.head_dim);
                    output->values[output_index] = accumulated;
                }
            }
        }
    }
    state->batch_size = attributes.batch_size;
    state->kv_head_count = attributes.kv_head_count;
    state->head_dim = attributes.head_dim;
    state->length += attributes.query_length;
    state->key = std::move(next_key);
    state->value = std::move(next_value);
    if (report != nullptr) {
        report->state_bytes = vbuf_portable_execution_state_bytes(state);
        report->scratch_peak_bytes = report->score_bytes + report->probability_bytes;
    }
    return true;
}

} // namespace

bool execute_portable_graph(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, PortableRouterPrefixResult * result,
    std::string * error) {
    return execute_portable_graph_with_state(graph, input, lease, resolver, nullptr, result, error);
}

bool execute_portable_graph_with_state(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, VbufPortableExecutionState * state,
    PortableRouterPrefixResult * result, std::string * error) {
    if (graph == nullptr || result == nullptr || !resolver || !lease)
        return fail(error, "portable graph execution argument is invalid");
    const uint32_t abi = vbuf_runtime_graph_abi_version();
    if (abi != VBUF_PORTABLE_EXEC_ABI_V1 && abi != VBUF_PORTABLE_EXEC_ABI_V2)
        return fail(error, "portable graph ABI version is unsupported");
    uint32_t graph_input = 0, graph_output = 0;
    if (vbuf_runtime_graph_input_value(graph, &graph_input) != 0 ||
        vbuf_runtime_graph_output_value(graph, &graph_output) != 0)
        return fail(error, "portable graph boundary is invalid");
    std::unordered_map<uint32_t, PortableActivation> values;
    values.emplace(graph_input, input);
    bool selection_required = false;
    const uint32_t count = vbuf_runtime_graph_operation_count(graph);
    for (uint32_t index = 0; index < count; ++index) {
        VbufGraphOperationDescV2 operation_v2{};
        const uint32_t descriptor_status = abi == VBUF_PORTABLE_EXEC_ABI_V2
            ? vbuf_runtime_graph_operation_desc_v2(graph, index, &operation_v2)
            : vbuf_runtime_graph_operation_desc(graph, index, &operation_v2.base);
        if (descriptor_status != 0)
            return fail(error, "portable operation descriptor is invalid");
        const VbufGraphOperationDesc & operation = operation_v2.base;
        const PortableActivation * value = nullptr;
        if (operation.kind != 6) {
            uint32_t value_id = UINT32_MAX;
            if (!find_value(operation, &value_id, error)) return false;
            const auto value_it = values.find(value_id);
            if (value_it == values.end()) return fail(error, "portable value is unavailable");
            value = &value_it->second;
        }
        if (operation.kind == 1 || operation.kind == 2 || operation.kind == 5) {
            uint32_t tensor_id = 0;
            if (!find_tensor(operation, &tensor_id, error)) return false;
            PortableResolvedTensor resolved;
            if (!resolver(tensor_id, &resolved, error)) return false;
            PortableActivation output;
            if (!run_backend_op(operation, *value, resolved, lease, &output, error)) return false;
            values[operation.output] = std::move(output);
            if (operation.kind == 1) result->normalized = values[operation.output].values;
            else result->logits = values[operation.output].values;
        } else if (operation.kind == 3) {
            PortableActivation output;
            if (!run_activation(operation, *value, &output, error)) return false;
            values[operation.output] = std::move(output);
        } else if (operation.kind == 7) {
            std::vector<uint32_t> value_ids;
            if (!find_values(operation, &value_ids, error) || value_ids.size() != 2)
                return fail(error, "residual requires two value operands");
            const auto lhs = values.find(value_ids[0]);
            const auto rhs = values.find(value_ids[1]);
            if (lhs == values.end() || rhs == values.end())
                return fail(error, "residual value is unavailable");
            PortableActivation output;
            if (!run_residual(lhs->second, rhs->second, &output, error)) return false;
            values[operation.output] = std::move(output);
        } else if (operation.kind == 4) {
            selection_required = true;
            if (operation.input_count != 1 || operation.top_k_order != 1 ||
                operation.top_k_tie_break != 1 || operation.has_top_k == 0 ||
                operation.output != graph_output)
                return fail(error, "portable TopK semantics are incomplete");
            std::string topk_error;
            if (!deterministic_top_k(value->values,
                    static_cast<uint32_t>(value->values.size()), operation.top_k,
                    &result->selection, &topk_error))
                return fail(error, topk_error.c_str());
        } else if (operation.kind == 6) {
            if (abi != VBUF_PORTABLE_EXEC_ABI_V2)
                return fail(error, "attention backend operation requires a sequence/state contract");
            std::vector<PortableActivation> attention_inputs;
            if (operation.input_count != 3 || operation.inputs == nullptr)
                return fail(error, "attention requires Q, K, and V operands");
            for (uint32_t input_index = 0; input_index < operation.input_count; ++input_index) {
                const VbufGraphInputDesc & input_ref = operation.inputs[input_index];
                if (input_ref.kind == 0) {
                    const auto value_it = values.find(input_ref.id);
                    if (value_it == values.end())
                        return fail(error, "attention value operand is unavailable");
                    attention_inputs.push_back(value_it->second);
                } else if (input_ref.kind == 1) {
                    PortableResolvedTensor resolved;
                    if (!resolver(input_ref.id, &resolved, error)) return false;
                    PortableActivation tensor_input;
                    if (!activation_from_tensor(resolved.ref, &tensor_input, error)) return false;
                    attention_inputs.push_back(std::move(tensor_input));
                } else {
                    return fail(error, "attention input reference kind is unsupported");
                }
            }
            PortableActivation attention_output;
            if (!run_attention(operation_v2, attention_inputs, state, &attention_output,
                    &result->attention, error)) return false;
            result->attention_output = attention_output.values;
            values[operation.output] = std::move(attention_output);
        } else {
            return fail(error, "portable operation kind is unsupported");
        }
    }
    if (selection_required && result->selection.ids.empty())
        return fail(error, "portable TopK produced no selection");
    return true;
}

} // namespace vbuf_ggml
