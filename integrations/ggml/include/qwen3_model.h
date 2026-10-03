#pragma once

#include "ggml.h"
#include "vbuf_materializer.h"
#include "vbuf_model_architecture.h"
#include "vbuf_range_source.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

extern "C" {
struct VbufMlConsumerHandle;
struct VbufMlTensorView;
}

namespace vbuf_ggml {

struct Qwen3TensorView {
    const char * name = nullptr;
    uint64_t name_len = 0;
    uint8_t representation = 0;
    uint8_t rank = 0;
    const uint64_t * dimensions = nullptr;
    const uint8_t * payload = nullptr;
    uint64_t payload_len = 0;
};

struct Qwen3Tensor {
    uint64_t id = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    std::string name;
    Qwen3TensorView view{};
    ggml_tensor * ggml = nullptr;
    VbufTensorView generic{};

    PersistentTensorRef persistent() const {
        return { id, name, generic, offset };
    }
};

// Metadata/source owner shared by qualification and the canonical runtime.
// CUDA allocations and session-owned KV state are deliberately not part of it.
struct Qwen3Model {
    VbufMlConsumerHandle * handle = nullptr;
    const VbufMlTensorView * views = nullptr;
    uint64_t count = 0;
    uint64_t source_id = 0;
    uint32_t layer_count = 0;
    std::string artifact_identity;
    ModelMetadataDescriptor metadata;
    Qwen3DenseTensorCatalog catalog;
    std::array<uint64_t, 4> special_tokens{};
    std::array<bool, 4> special_token_present{};
    bool add_bos = false;
    std::unordered_map<std::string, Qwen3Tensor> tensors;
    std::shared_ptr<HttpRangeSource> source;
    std::unique_ptr<LocalVbufRangeMaterializer> materializer;

    Qwen3Model() = default;
    Qwen3Model(const Qwen3Model &) = delete;
    Qwen3Model & operator=(const Qwen3Model &) = delete;
    ~Qwen3Model();
};

inline constexpr const char * QWEN3_14B_Q4_K_M_SHA256 =
    "f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";

Qwen3Tensor & qwen3_tensor(Qwen3Model & model, const std::string & name);
const Qwen3Tensor & qwen3_tensor(const Qwen3Model & model, const std::string & name);
bool qwen3_artifact_identity_is_qualified(const std::string & source_sha256) noexcept;
bool is_qwen3_semantic_artifact(const std::string & semantic_artifact);
void open_qwen3_model(const std::string & semantic_artifact, const std::string & source_endpoint,
    Qwen3Model * model, bool require_exact_qualified_artifact = false);

} // namespace vbuf_ggml
