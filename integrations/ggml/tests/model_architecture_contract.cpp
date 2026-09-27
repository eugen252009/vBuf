#include "vbuf_model_architecture.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

vbuf_ggml::ModelMetadataDescriptor qwen3_metadata() {
    vbuf_ggml::ModelMetadataDescriptor model;
    model.architecture = vbuf_ggml::ModelArchitecture::Qwen3;
    model.source_name = "qwen3";
    model.context_length = 32768;
    model.embedding_length = 5120;
    model.layer_count = 40;
    model.head_count = 40;
    model.kv_head_count = 8;
    model.key_head_dimension = 128;
    model.value_head_dimension = 128;
    model.feed_forward_length = 17408;
    model.normalization_epsilon = 1e-6;
    model.rope_theta = 1000000.0;
    model.rope_dimension = 128;
    model.vocabulary_size = 151936;
    return model;
}

uint64_t payload_bytes(const vbuf_ggml::ModelTensorMetadata & tensor) {
    uint64_t elements = 1;
    for (uint64_t dimension : tensor.dimensions) elements *= dimension;
    if (tensor.representation == 0) return elements * 4;
    const uint64_t block_bytes = tensor.representation == 6 ? 144 : 210;
    return elements / 256 * block_bytes;
}

void add_tensor(std::vector<vbuf_ggml::ModelTensorMetadata> * tensors,
    const std::string & name, std::vector<uint64_t> dimensions, uint8_t representation,
    uint64_t * next_offset) {
    vbuf_ggml::ModelTensorMetadata tensor;
    tensor.name = name;
    tensor.dimensions = std::move(dimensions);
    tensor.representation = representation;
    tensor.payload_length = payload_bytes(tensor);
    tensor.source_offset = *next_offset;
    *next_offset += tensor.payload_length;
    tensors->push_back(std::move(tensor));
}

std::vector<vbuf_ggml::ModelTensorMetadata> qwen3_tensors() {
    const auto model = qwen3_metadata();
    std::vector<vbuf_ggml::ModelTensorMetadata> tensors;
    uint64_t offset = 0;
    add_tensor(&tensors, "token_embd.weight", {5120, 151936}, 6, &offset);
    add_tensor(&tensors, "output_norm.weight", {5120}, 0, &offset);
    add_tensor(&tensors, "output.weight", {5120, 151936}, 14, &offset);
    for (uint64_t layer = 0; layer < model.layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add_tensor(&tensors, prefix + "attn_k.weight", {5120, 1024}, 6, &offset);
        add_tensor(&tensors, prefix + "attn_k_norm.weight", {128}, 0, &offset);
        add_tensor(&tensors, prefix + "attn_norm.weight", {5120}, 0, &offset);
        add_tensor(&tensors, prefix + "attn_output.weight", {5120, 5120}, 6, &offset);
        add_tensor(&tensors, prefix + "attn_q.weight", {5120, 5120}, 6, &offset);
        add_tensor(&tensors, prefix + "attn_q_norm.weight", {128}, 0, &offset);
        const uint8_t mixed_quant = layer % 2 == 0 ? 14 : 6;
        add_tensor(&tensors, prefix + "attn_v.weight", {5120, 1024}, mixed_quant, &offset);
        add_tensor(&tensors, prefix + "ffn_down.weight", {17408, 5120}, mixed_quant, &offset);
        add_tensor(&tensors, prefix + "ffn_gate.weight", {5120, 17408}, 6, &offset);
        add_tensor(&tensors, prefix + "ffn_norm.weight", {5120}, 0, &offset);
        add_tensor(&tensors, prefix + "ffn_up.weight", {5120, 17408}, 6, &offset);
    }
    return tensors;
}

void require_error(const vbuf_ggml::ModelMetadataDescriptor & model,
    const std::vector<vbuf_ggml::ModelTensorMetadata> & tensors) {
    vbuf_ggml::Qwen3DenseTensorCatalog catalog;
    std::string error;
    assert(!vbuf_ggml::discover_qwen3_dense_tensors(model, tensors, &catalog, &error));
    assert(!error.empty());
}

} // namespace

int main() {
    using namespace vbuf_ggml;
    assert(parse_model_architecture("deepseek2") == ModelArchitecture::DeepSeekV2);
    assert(parse_model_architecture("qwen3") == ModelArchitecture::Qwen3);
    assert(parse_model_architecture("qwen3moe") == ModelArchitecture::Unsupported);
    assert(parse_model_architecture("unknown") == ModelArchitecture::Unsupported);

    ModelMetadataDescriptor model = qwen3_metadata();
    std::string error;
    assert(validate_model_metadata(model, &error));
    assert(error.empty());
    model.kv_head_count = 7;
    assert(!validate_model_metadata(model, &error));
    model = qwen3_metadata();
    model.expert_count = 8;
    assert(!validate_model_metadata(model, &error));

    model = qwen3_metadata();
    std::vector<ModelTensorMetadata> tensors = qwen3_tensors();
    assert(tensors.size() == 443);
    Qwen3DenseTensorCatalog catalog;
    assert(discover_qwen3_dense_tensors(model, tensors, &catalog, &error));
    assert(catalog.tensors.size() == 443);
    assert(catalog.tensors.at("blk.39.attn_v.weight").representation == 6);
    assert(catalog.tensors.at("output.weight").dimensions ==
        (std::vector<uint64_t>{5120, 151936}));

    auto missing = tensors;
    missing.erase(missing.begin() + 5);
    require_error(model, missing);

    auto wrong_name = tensors;
    wrong_name.back().name = "blk.39.ffn_extra.weight";
    require_error(model, wrong_name);

    auto wrong_shape = tensors;
    wrong_shape[3].dimensions[0] += 1;
    require_error(model, wrong_shape);

    auto wrong_representation = tensors;
    wrong_representation[0].representation = 14;
    require_error(model, wrong_representation);

    auto wrong_length = tensors;
    ++wrong_length[0].payload_length;
    require_error(model, wrong_length);

    auto duplicate = tensors;
    duplicate.back().name = duplicate.front().name;
    require_error(model, duplicate);

    auto invalid_range = tensors;
    invalid_range.front().source_offset = std::numeric_limits<uint64_t>::max() - 2;
    require_error(model, invalid_range);

    ModelMetadataDescriptor deepseek;
    deepseek.architecture = ModelArchitecture::DeepSeekV2;
    deepseek.source_name = "deepseek2";
    deepseek.context_length = 4096;
    deepseek.embedding_length = 2048;
    deepseek.layer_count = 27;
    deepseek.head_count = 16;
    deepseek.kv_head_count = 16;
    deepseek.key_head_dimension = 128;
    deepseek.value_head_dimension = 128;
    deepseek.feed_forward_length = 10944;
    deepseek.normalization_epsilon = 1e-6;
    deepseek.rope_theta = 10000.0;
    deepseek.rope_dimension = 64;
    deepseek.vocabulary_size = 102400;
    assert(validate_model_metadata(deepseek, &error));

    return 0;
}
