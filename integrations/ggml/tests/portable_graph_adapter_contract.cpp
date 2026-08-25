#include "vbuf_portable_graph_adapter.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <limits>

struct VbufRuntimeGraphHandle {};

namespace {

VbufGraphInputDesc topk_input{ 0, { 0, 0, 0 }, 0 };
VbufGraphInputDesc residual_inputs[] = {
    { 0, { 0, 0, 0 }, 0 },
    { 0, { 0, 0, 0 }, 0 },
};
VbufGraphInputDesc attention_inputs[] = {
    { 1, { 0, 0, 0 }, 10 },
    { 1, { 0, 0, 0 }, 11 },
    { 1, { 0, 0, 0 }, 12 },
};
bool invalid_topk = false;
uint32_t operation_mode = 0;
bool attention_v2 = false;
uint64_t attention_kv_heads = 1;

const std::vector<float> attention_q = {
    1.0f, 0.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f,
};
const std::vector<float> attention_k = {
    1.0f, 0.0f, 0.0f, 1.0f,
};
const std::vector<float> attention_v = {
    1.0f, 2.0f, 3.0f, 4.0f,
};
const std::vector<float> attention_mha_k = {
    1.0f, 0.0f, 0.0f, 1.0f,
    0.0f, 1.0f, 1.0f, 0.0f,
};
const std::vector<float> attention_mha_v = {
    1.0f, 2.0f, 2.0f, 1.0f,
    3.0f, 4.0f, 4.0f, 3.0f,
};
const uint64_t attention_q_dimensions[] = { 1, 2, 2, 2 };
const uint64_t attention_kv_dimensions[] = { 1, 2, 1, 2 };
const uint64_t attention_mha_dimensions[] = { 1, 2, 2, 2 };

size_t attention_test_index(uint64_t position, uint64_t head, uint64_t component,
    uint64_t sequence, uint64_t heads, uint64_t dimension) {
    (void) sequence;
    return static_cast<size_t>((((position * heads) + head) * dimension) + component);
}

std::vector<float> reference_attention(const std::vector<float> & query,
    const std::vector<float> & key, const std::vector<float> & value,
    uint64_t query_heads, uint64_t kv_heads, uint64_t dimension, float scale) {
    const uint64_t query_length = 2;
    std::vector<float> output(query.size(), 0.0f);
    const uint64_t group = query_heads / kv_heads;
    for (uint64_t query_position = 0; query_position < query_length; ++query_position) {
        for (uint64_t query_head = 0; query_head < query_heads; ++query_head) {
            const uint64_t kv_head = query_head / group;
            float scores[2] = {};
            for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
                float dot = 0.0f;
                for (uint64_t component = 0; component < dimension; ++component) {
                    const size_t q_index = attention_test_index(query_position, query_head,
                        component, query_length, query_heads, dimension);
                    const size_t k_index = attention_test_index(key_position, kv_head,
                        component, query_length, kv_heads, dimension);
                    dot += query[q_index] * key[k_index];
                }
                scores[key_position] = dot * scale;
            }
            const float maximum = *std::max_element(scores, scores + query_position + 1);
            float total = 0.0f;
            for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
                scores[key_position] = std::exp(scores[key_position] - maximum);
                total += scores[key_position];
            }
            for (uint64_t component = 0; component < dimension; ++component) {
                float accumulated = 0.0f;
                for (uint64_t key_position = 0; key_position <= query_position; ++key_position) {
                    const size_t v_index = attention_test_index(key_position, kv_head,
                        component, query_length, kv_heads, dimension);
                    accumulated += (scores[key_position] / total) * value[v_index];
                }
                const size_t output_index = attention_test_index(query_position, query_head,
                    component, query_length, query_heads, dimension);
                output[output_index] = accumulated;
            }
        }
    }
    return output;
}

} // namespace

extern "C" uint32_t vbuf_runtime_graph_abi_version() {
    return attention_v2 ? VBUF_PORTABLE_EXEC_ABI_V2 : VBUF_PORTABLE_EXEC_ABI_V1;
}
extern "C" uint32_t vbuf_runtime_graph_input_value(const VbufRuntimeGraphHandle *, uint32_t * value) {
    *value = 0;
    return VBUF_FFI_OK;
}
extern "C" uint32_t vbuf_runtime_graph_output_value(const VbufRuntimeGraphHandle *, uint32_t * value) {
    *value = 3;
    return VBUF_FFI_OK;
}
extern "C" uint32_t vbuf_runtime_graph_operation_count(const VbufRuntimeGraphHandle *) { return 1; }
extern "C" uint32_t vbuf_runtime_graph_operation_desc(const VbufRuntimeGraphHandle *, uint32_t,
    VbufGraphOperationDesc * output) {
    *output = {};
    if (operation_mode == 1) {
        output->kind = static_cast<uint32_t>(vbuf_ggml::PortableGraphOperationKind::Activation);
        output->inputs = &topk_input;
        output->input_count = 1;
        output->output = 3;
        output->reserved[0] = VBUF_ACTIVATION_SILU;
        return VBUF_FFI_OK;
    }
    if (operation_mode == 2) {
        output->kind = static_cast<uint32_t>(vbuf_ggml::PortableGraphOperationKind::ResidualAdd);
        output->inputs = residual_inputs;
        output->input_count = 2;
        output->output = 3;
        return VBUF_FFI_OK;
    }
    if (operation_mode == 3) {
        output->kind = static_cast<uint32_t>(vbuf_ggml::PortableGraphOperationKind::Attention);
        output->inputs = &topk_input;
        output->input_count = 1;
        output->output = 3;
        return VBUF_FFI_OK;
    }
    output->kind = 4;
    output->inputs = &topk_input;
    output->input_count = 1;
    output->output = 3;
    output->top_k_order = invalid_topk ? 0 : 1;
    output->top_k_tie_break = 1;
    output->has_top_k = 1;
    output->top_k = 2;
    return VBUF_FFI_OK;
}
extern "C" uint32_t vbuf_runtime_graph_operation_desc_v2(const VbufRuntimeGraphHandle * graph,
    uint32_t index, VbufGraphOperationDescV2 * output) {
    *output = {};
    vbuf_runtime_graph_operation_desc(graph, index, &output->base);
    if (operation_mode == 4) {
        output->base.kind = static_cast<uint32_t>(vbuf_ggml::PortableGraphOperationKind::Attention);
        output->base.inputs = attention_inputs;
        output->base.input_count = 3;
        output->base.output = 3;
        output->attention = {
            1, 2, attention_kv_heads, 2, 2, 2, 1.0f / std::sqrt(2.0f),
            1, 1, { 0, 0 }, 7,
        };
    }
    return VBUF_FFI_OK;
}

int main() {
    VbufRuntimeGraphHandle graph;
    const vbuf_ggml::PortableActivation input{ { 0.25f, 3.0f, 3.0f, -1.0f }, { 4, 1 } };
    const std::shared_ptr<const void> lease = std::make_shared<int>(1);
    const vbuf_ggml::PortableTensorResolver resolver = [](uint32_t id,
        vbuf_ggml::PortableResolvedTensor * output, std::string *) {
        if (id != 10 && id != 11 && id != 12) return false;
        output->ref.tensor_id = id;
        const bool mha = attention_kv_heads == 2;
        const bool query = id == 10;
        output->ref.view = {
            0,
            4,
            query ? attention_q_dimensions : mha ? attention_mha_dimensions : attention_kv_dimensions,
            reinterpret_cast<const uint8_t *>(query ? attention_q.data()
                : id == 11 ? (mha ? attention_mha_k.data() : attention_k.data())
                : (mha ? attention_mha_v.data() : attention_v.data())),
            (query ? attention_q.size() : mha ? attention_mha_k.size() : attention_k.size()) *
                sizeof(float),
        };
        return true;
    };
    vbuf_ggml::PortableRouterPrefixResult result;
    std::string error;
    assert(vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert((result.selection.ids == std::vector<uint32_t>{ 1, 2 }));
    assert(result.selection.scores.size() == 2);

    invalid_topk = true;
    result = {};
    error.clear();
    assert(!vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert(error == "portable TopK semantics are incomplete");

    operation_mode = 1;
    invalid_topk = false;
    result = {};
    error.clear();
    assert(vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert(result.selection.ids.empty());
    assert(result.normalized.empty());

    operation_mode = 2;
    result = {};
    error.clear();
    assert(vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));

    operation_mode = 3;
    result = {};
    error.clear();
    assert(!vbuf_ggml::execute_portable_graph(&graph, input, lease, resolver, &result, &error));
    assert(error == "attention backend operation requires a sequence/state contract");

    operation_mode = 4;
    attention_v2 = true;
    auto * state = vbuf_ggml::vbuf_portable_execution_state_create(7, 4);
    assert(state != nullptr);
    for (int repetition = 0; repetition < 10; ++repetition) {
        vbuf_ggml::vbuf_portable_execution_state_reset(state);
        result = {};
        error.clear();
        assert(vbuf_ggml::execute_portable_graph_with_state(&graph, input, lease, resolver,
            state, &result, &error));
        assert(vbuf_ggml::vbuf_portable_execution_state_length(state) == 2);
        assert(result.attention.scratch_peak_bytes == result.attention.score_bytes * 2);
        const std::vector<float> expected = reference_attention(attention_q, attention_k,
            attention_v, 2, 1, 2, 1.0f / std::sqrt(2.0f));
        assert(result.attention.output_bytes == expected.size() * sizeof(float));
        for (size_t index = 0; index < expected.size(); ++index)
            assert(std::fabs(result.attention_output[index] - expected[index]) < 1e-6f);
    }
    assert(vbuf_ggml::vbuf_portable_execution_state_bytes(state) ==
        (attention_k.size() + attention_v.size()) * sizeof(float));
    attention_kv_heads = 2;
    result = {};
    error.clear();
    assert(!vbuf_ggml::execute_portable_graph_with_state(&graph, input, lease, resolver,
        state, &result, &error));
    assert(error == "attention state geometry does not match operation");
    assert(vbuf_ggml::vbuf_portable_execution_state_length(state) == 2);
    attention_kv_heads = 1;
    auto * isolated_state = vbuf_ggml::vbuf_portable_execution_state_create(7, 4);
    assert(isolated_state != nullptr);
    result = {};
    error.clear();
    assert(vbuf_ggml::execute_portable_graph_with_state(&graph, input, lease, resolver,
        isolated_state, &result, &error));
    assert(vbuf_ggml::vbuf_portable_execution_state_length(state) == 2);
    assert(vbuf_ggml::vbuf_portable_execution_state_length(isolated_state) == 2);
    vbuf_ggml::vbuf_portable_execution_state_destroy(isolated_state);
    attention_kv_heads = 2;
    vbuf_ggml::vbuf_portable_execution_state_reset(state);
    result = {};
    error.clear();
    assert(vbuf_ggml::execute_portable_graph_with_state(&graph, input, lease, resolver,
        state, &result, &error));
    const std::vector<float> expected_mha = reference_attention(attention_q, attention_mha_k,
        attention_mha_v, 2, 2, 2, 1.0f / std::sqrt(2.0f));
    assert(result.attention.output_bytes == expected_mha.size() * sizeof(float));
    for (size_t index = 0; index < expected_mha.size(); ++index)
        assert(std::fabs(result.attention_output[index] - expected_mha[index]) < 1e-6f);
    vbuf_ggml::vbuf_portable_execution_state_destroy(state);
    return 0;
}
