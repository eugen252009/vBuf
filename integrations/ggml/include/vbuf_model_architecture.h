#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace vbuf_ggml {

enum class ModelArchitecture : uint8_t {
    DeepSeekV2,
    Qwen3,
    Unsupported,
};

ModelArchitecture parse_model_architecture(const std::string & name);
const char * model_architecture_name(ModelArchitecture architecture);

struct ModelMetadataDescriptor {
    ModelArchitecture architecture = ModelArchitecture::Unsupported;
    std::string source_name;
    uint64_t context_length = 0;
    uint64_t embedding_length = 0;
    uint64_t layer_count = 0;
    uint64_t head_count = 0;
    uint64_t kv_head_count = 0;
    uint64_t key_head_dimension = 0;
    uint64_t value_head_dimension = 0;
    uint64_t feed_forward_length = 0;
    double normalization_epsilon = 0.0;
    double rope_theta = 0.0;
    uint32_t rope_dimension = 0;
    uint32_t vocabulary_size = 0;
    uint32_t expert_count = 0;
    uint32_t expert_used_count = 0;
};

bool validate_model_metadata(const ModelMetadataDescriptor & metadata,
    std::string * error = nullptr);

struct ModelTensorMetadata {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint8_t representation = 0;
    uint64_t source_offset = 0;
    uint64_t payload_length = 0;
};

struct Qwen3DenseTensorCatalog {
    std::unordered_map<std::string, ModelTensorMetadata> tensors;
};

// Strict inventory for the selected dense Qwen3 GGUF contract. It only
// validates already-discovered metadata; it performs no lookup or I/O.
bool discover_qwen3_dense_tensors(const ModelMetadataDescriptor & model,
    const std::vector<ModelTensorMetadata> & tensors,
    Qwen3DenseTensorCatalog * catalog, std::string * error = nullptr);

} // namespace vbuf_ggml
