#include "vbuf_model_architecture.h"

#include <cmath>
#include <limits>
#include <unordered_set>

namespace vbuf_ggml {
namespace {

bool fail(std::string * error, const char * message) {
    if (error != nullptr) *error = message;
    return false;
}

struct ExpectedTensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint8_t representation;
    uint8_t alternate_representation = UINT8_MAX;
};

bool expected_bytes(const ExpectedTensor & tensor, uint8_t representation, uint64_t * bytes) {
    uint64_t elements = 1;
    for (uint64_t dimension : tensor.dimensions) {
        if (dimension == 0 || elements > UINT64_MAX / dimension) return false;
        elements *= dimension;
    }
    if (representation == 0) {
        if (elements > UINT64_MAX / 4) return false;
        *bytes = elements * 4;
        return true;
    }

    uint64_t block_elements = 0, block_bytes = 0;
    if (representation == 6) {
        block_elements = 256;
        block_bytes = 144;
    } else if (representation == 14) {
        block_elements = 256;
        block_bytes = 210;
    } else {
        return false;
    }
    if (tensor.dimensions.empty() || tensor.dimensions.front() % block_elements != 0)
        return false;
    uint64_t rows = 1;
    for (size_t index = 1; index < tensor.dimensions.size(); ++index) {
        if (rows > UINT64_MAX / tensor.dimensions[index]) return false;
        rows *= tensor.dimensions[index];
    }
    const uint64_t blocks_per_row = tensor.dimensions.front() / block_elements;
    if (rows > UINT64_MAX / blocks_per_row) return false;
    const uint64_t blocks = rows * blocks_per_row;
    if (blocks > UINT64_MAX / block_bytes) return false;
    *bytes = blocks * block_bytes;
    return true;
}

void add(std::vector<ExpectedTensor> * tensors, const std::string & name,
    std::vector<uint64_t> dimensions, uint8_t representation,
    uint8_t alternate_representation = UINT8_MAX) {
    tensors->push_back({ name, std::move(dimensions), representation, alternate_representation });
}

std::vector<ExpectedTensor> expected_qwen3_tensors(const ModelMetadataDescriptor & model) {
    std::vector<ExpectedTensor> expected;
    const uint64_t hidden = model.embedding_length;
    const uint64_t vocab = model.vocabulary_size;
    const uint64_t q_width = model.embedding_length;
    const uint64_t kv_width = model.kv_head_count * model.key_head_dimension;
    const uint64_t value_width = model.kv_head_count * model.value_head_dimension;
    const uint64_t ffn = model.feed_forward_length;

    add(&expected, "token_embd.weight", { hidden, vocab }, 6);
    add(&expected, "output_norm.weight", { hidden }, 0);
    add(&expected, "output.weight", { hidden, vocab }, 14);
    for (uint64_t layer = 0; layer < model.layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add(&expected, prefix + "attn_norm.weight", { hidden }, 0);
        add(&expected, prefix + "attn_q.weight", { hidden, q_width }, 6);
        add(&expected, prefix + "attn_q_norm.weight", { model.key_head_dimension }, 0);
        add(&expected, prefix + "attn_k.weight", { hidden, kv_width }, 6);
        add(&expected, prefix + "attn_k_norm.weight", { model.key_head_dimension }, 0);
        add(&expected, prefix + "attn_v.weight", { hidden, value_width }, 14, 6);
        add(&expected, prefix + "attn_output.weight", { q_width, hidden }, 6);
        add(&expected, prefix + "ffn_norm.weight", { hidden }, 0);
        add(&expected, prefix + "ffn_gate.weight", { hidden, ffn }, 6);
        add(&expected, prefix + "ffn_up.weight", { hidden, ffn }, 6);
        add(&expected, prefix + "ffn_down.weight", { ffn, hidden }, 14, 6);
    }
    return expected;
}

} // namespace

ModelArchitecture parse_model_architecture(const std::string & name) {
    if (name == "deepseek2") return ModelArchitecture::DeepSeekV2;
    if (name == "qwen3") return ModelArchitecture::Qwen3;
    return ModelArchitecture::Unsupported;
}

const char * model_architecture_name(ModelArchitecture architecture) {
    switch (architecture) {
    case ModelArchitecture::DeepSeekV2: return "deepseek2";
    case ModelArchitecture::Qwen3: return "qwen3";
    case ModelArchitecture::Unsupported: return "unsupported";
    }
    return "unsupported";
}

bool validate_model_metadata(const ModelMetadataDescriptor & model, std::string * error) {
    if (model.architecture == ModelArchitecture::Unsupported)
        return fail(error, "model architecture is unsupported");
    if (model.source_name != model_architecture_name(model.architecture))
        return fail(error, "model architecture identity is inconsistent");
    if (model.context_length == 0 || model.embedding_length == 0 || model.layer_count == 0 ||
        model.head_count == 0 || model.kv_head_count == 0 || model.key_head_dimension == 0 ||
        model.value_head_dimension == 0 || model.feed_forward_length == 0 ||
        model.vocabulary_size == 0 || model.head_count % model.kv_head_count != 0 ||
        !std::isfinite(model.normalization_epsilon) || model.normalization_epsilon <= 0.0 ||
        !std::isfinite(model.rope_theta) || model.rope_theta <= 0.0 ||
        model.rope_dimension == 0 || model.rope_dimension > model.key_head_dimension ||
        model.rope_dimension % 2 != 0 ||
        model.kv_head_count > UINT64_MAX / model.key_head_dimension ||
        model.kv_head_count > UINT64_MAX / model.value_head_dimension)
        return fail(error, "model metadata has invalid dimensions or numeric attributes");
    if (model.architecture == ModelArchitecture::Qwen3 &&
        (model.embedding_length % model.head_count != 0 ||
         model.embedding_length / model.head_count != model.key_head_dimension ||
         model.key_head_dimension != model.value_head_dimension ||
         model.rope_dimension != model.key_head_dimension ||
         model.expert_count != 0 || model.expert_used_count != 0))
        return fail(error, "Qwen3 dense metadata is inconsistent with standard full-head attention");
    return true;
}

bool discover_qwen3_dense_tensors(const ModelMetadataDescriptor & model,
    const std::vector<ModelTensorMetadata> & tensors,
    Qwen3DenseTensorCatalog * catalog, std::string * error) {
    if (catalog == nullptr) return fail(error, "Qwen3 tensor catalog output is null");
    if (!validate_model_metadata(model, error)) return false;
    if (model.architecture != ModelArchitecture::Qwen3)
        return fail(error, "Qwen3 dense tensor discovery received a different architecture");
    if (model.layer_count > (std::numeric_limits<size_t>::max() - 3) / 11)
        return fail(error, "Qwen3 layer count exceeds catalog limits");
    const size_t required_count = static_cast<size_t>(model.layer_count) * 11 + 3;
    if (tensors.size() != required_count)
        return fail(error, "Qwen3 tensor count differs from the dense architecture contract");

    const std::vector<ExpectedTensor> expected = expected_qwen3_tensors(model);
    std::unordered_map<std::string, const ModelTensorMetadata *> by_name;
    by_name.reserve(tensors.size());
    for (const ModelTensorMetadata & tensor : tensors) {
        if (tensor.name.empty()) return fail(error, "Qwen3 tensor name is empty");
        if (!by_name.emplace(tensor.name, &tensor).second)
            return fail(error, "Qwen3 tensor name is duplicated");
    }

    Qwen3DenseTensorCatalog discovered;
    discovered.tensors.reserve(expected.size());
    for (const ExpectedTensor & descriptor : expected) {
        const auto found = by_name.find(descriptor.name);
        if (found == by_name.end()) return fail(error, "required Qwen3 tensor is missing");
        const ModelTensorMetadata & actual = *found->second;
        if (actual.dimensions != descriptor.dimensions ||
            (actual.representation != descriptor.representation &&
                actual.representation != descriptor.alternate_representation))
            return fail(error, "Qwen3 tensor shape or representation is invalid");
        uint64_t length = 0;
        if (!expected_bytes(descriptor, actual.representation, &length) || actual.payload_length != length ||
            actual.source_offset > UINT64_MAX - actual.payload_length)
            return fail(error, "Qwen3 tensor source range or payload length is invalid");
        discovered.tensors.emplace(actual.name, actual);
    }
    *catalog = std::move(discovered);
    return true;
}

} // namespace vbuf_ggml
