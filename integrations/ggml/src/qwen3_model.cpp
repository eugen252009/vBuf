#include "qwen3_model.h"

#include "ggml-backend.h"
#include "vbuf_ml_model_metadata_ffi.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
struct VbufMlTensorView {
    const char * name;
    uint64_t name_len;
    uint8_t representation;
    uint8_t rank;
    const uint64_t * dimensions;
    const uint8_t * payload;
    uint64_t payload_len;
};
void vbuf_ml_consumer_close(VbufMlConsumerHandle *);
VbufMlConsumerHandle * vbuf_ml_consumer_open_metadata(const char *);
uint32_t vbuf_ml_consumer_architecture(const VbufMlConsumerHandle *, char *, size_t);
uint32_t vbuf_ml_consumer_metadata(const VbufMlConsumerHandle *, VbufMlModelMetadataInfo *);
uint32_t vbuf_ml_consumer_tensor_views(const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
uint32_t vbuf_ml_consumer_tensor_source(const VbufMlConsumerHandle *, uint64_t, VbufMlTensorSourceInfo *);
uint32_t vbuf_ml_consumer_source_identity(const VbufMlConsumerHandle *, uint64_t, VbufMlSourceIdentityInfo *);
uint32_t vbuf_ml_consumer_special_token(const VbufMlConsumerHandle *, uint8_t, uint64_t *);
uint32_t vbuf_ml_consumer_add_bos(const VbufMlConsumerHandle *, bool *);
}

namespace vbuf_ggml {
namespace {

constexpr uint8_t descriptor_sentinel = 0;

std::string hex_sha(const uint8_t * value, size_t bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        result += digits[value[i] >> 4];
        result += digits[value[i] & 15];
    }
    return result;
}

ModelMetadataDescriptor describe(const char * architecture, const VbufMlModelMetadataInfo & info) {
    ModelMetadataDescriptor result{};
    result.architecture = parse_model_architecture(architecture);
    result.source_name = architecture;
    result.context_length = info.context_length;
    result.embedding_length = info.embedding_length;
    result.layer_count = info.layer_count;
    result.head_count = info.head_count;
    result.kv_head_count = info.kv_head_count;
    result.key_head_dimension = info.key_head_dimension;
    result.value_head_dimension = info.value_head_dimension;
    result.feed_forward_length = info.feed_forward_length;
    result.normalization_epsilon = info.normalization_epsilon;
    result.rope_theta = info.rope_theta;
    result.rope_dimension = info.rope_dimension;
    result.vocabulary_size = info.vocabulary_size;
    result.expert_count = info.expert_count;
    result.expert_used_count = info.expert_used_count;
    return result;
}

} // namespace

Qwen3Model::~Qwen3Model() {
    if (handle != nullptr) vbuf_ml_consumer_close(handle);
}

bool qwen3_artifact_identity_is_qualified(const std::string & source_sha256) noexcept {
    return source_sha256 == QWEN3_14B_Q4_K_M_SHA256;
}

bool qwen3_artifact_identity_is_experimental_8b(const std::string & source_sha256) noexcept {
    return source_sha256 == QWEN3_8B_VBUF_SOURCE_SHA256;
}

bool is_qwen3_semantic_artifact(const std::string & semantic_artifact) {
    VbufMlConsumerHandle * handle = vbuf_ml_consumer_open_metadata(semantic_artifact.c_str());
    if (handle == nullptr) return false;
    char architecture[128]{};
    const bool is_qwen3 = vbuf_ml_consumer_architecture(handle, architecture, sizeof(architecture)) == 0 &&
        std::string(architecture) == "qwen3";
    vbuf_ml_consumer_close(handle);
    return is_qwen3;
}

Qwen3Tensor & qwen3_tensor(Qwen3Model & model, const std::string & name) {
    const auto found = model.tensors.find(name);
    if (found == model.tensors.end()) throw std::runtime_error("missing Qwen3 tensor: " + name);
    return found->second;
}

const Qwen3Tensor & qwen3_tensor(const Qwen3Model & model, const std::string & name) {
    const auto found = model.tensors.find(name);
    if (found == model.tensors.end()) throw std::runtime_error("missing Qwen3 tensor: " + name);
    return found->second;
}

void configure_qwen3_model_source(Qwen3Model & model, const std::string & source_endpoint) {
    if (source_endpoint.empty()) throw std::invalid_argument("Qwen3 source endpoint must not be empty");
    if (model.source || model.materializer) {
        if (model.source_endpoint != source_endpoint)
            throw std::runtime_error("Qwen3 model runtime cannot change its payload source");
        return;
    }
    if (model.source_size == 0 || model.artifact_identity.rfind("sha256:", 0) != 0)
        throw std::runtime_error("Qwen3 model source identity was not admitted");
    model.source_endpoint = source_endpoint;
    model.source = std::make_shared<HttpRangeSource>(source_endpoint, "", model.source_size,
        model.artifact_identity.substr(7));
    model.materializer = std::make_unique<LocalVbufRangeMaterializer>(model.source);
}

void open_qwen3_model(const std::string & semantic_artifact, const std::string & source_endpoint,
    Qwen3Model * model, bool require_exact_qualified_artifact, bool allow_experimental_qwen3_8b) {
    if (model == nullptr) throw std::invalid_argument("Qwen3 model output is null");
    if (model->handle != nullptr) throw std::invalid_argument("Qwen3 model output is already initialized");
    model->handle = vbuf_ml_consumer_open_metadata(semantic_artifact.c_str());
    if (model->handle == nullptr)
        throw std::runtime_error("Qwen3 semantic bootstrap open failed");
    const uint32_t tensor_status = vbuf_ml_consumer_tensor_views(model->handle, &model->views, &model->count);
    if (tensor_status != 0)
        throw std::runtime_error("Qwen3 tensor metadata lookup failed with status " + std::to_string(tensor_status));

    char architecture[128]{};
    VbufMlModelMetadataInfo ffi{};
    if (vbuf_ml_consumer_architecture(model->handle, architecture, sizeof(architecture)) != 0 ||
        vbuf_ml_consumer_metadata(model->handle, &ffi) != 0 || std::string(architecture) != "qwen3")
        throw std::runtime_error("Qwen3 metadata lookup/architecture validation failed");
    model->metadata = describe(architecture, ffi);
    std::string error;
    if (!validate_model_metadata(model->metadata, &error))
        throw std::runtime_error("invalid Qwen3 model metadata: " + error);
    if (ffi.layer_count > UINT32_MAX) throw std::runtime_error("Qwen3 layer count exceeds runtime range");
    model->layer_count = static_cast<uint32_t>(ffi.layer_count);

    std::vector<ModelTensorMetadata> inventory;
    inventory.reserve(model->count);
    for (uint64_t i = 0; i < model->count; ++i) {
        VbufMlTensorSourceInfo binding{};
        if (vbuf_ml_consumer_tensor_source(model->handle, i, &binding) != 0 || binding.source_id == 0)
            throw std::runtime_error("invalid Qwen3 tensor source binding");
        if (model->source_id == 0) model->source_id = binding.source_id;
        if (model->source_id != binding.source_id)
            throw std::runtime_error("Qwen3 tensors span multiple external sources");
        const VbufMlTensorView & view = model->views[i];
        if (view.name == nullptr || view.name_len == 0 || view.dimensions == nullptr ||
            view.rank == 0 || view.rank > GGML_MAX_DIMS || binding.length == 0 ||
            binding.offset > UINT64_MAX - binding.length)
            throw std::runtime_error("invalid Qwen3 tensor geometry or source range");
        Qwen3Tensor tensor;
        tensor.id = i;
        tensor.offset = binding.offset;
        tensor.length = binding.length;
        tensor.view = { view.name, view.name_len, view.representation, view.rank,
            view.dimensions, view.payload, view.payload_len };
        tensor.name.assign(view.name, view.name_len);
        tensor.generic = { view.representation, view.rank, view.dimensions,
            &descriptor_sentinel, binding.length };
        ModelTensorMetadata item;
        item.name = tensor.name;
        item.dimensions.assign(view.dimensions, view.dimensions + view.rank);
        item.representation = view.representation;
        item.source_offset = binding.offset;
        item.payload_length = binding.length;
        inventory.push_back(std::move(item));
        if (!model->tensors.emplace(tensor.name, std::move(tensor)).second)
            throw std::runtime_error("duplicate Qwen3 tensor name");
    }
    if (!discover_qwen3_dense_tensors(model->metadata, inventory, &model->catalog, &error))
        throw std::runtime_error("Qwen3 dense tensor inventory rejected: " + error);

    VbufMlSourceIdentityInfo identity{};
    if (vbuf_ml_consumer_source_identity(model->handle, model->source_id, &identity) != 0 ||
        identity.hash_algorithm != 1 || identity.hash_len != 32 || identity.declared_size == 0)
        throw std::runtime_error("Qwen3 payload source identity is not qualified SHA-256");
    const std::string source_sha256 = hex_sha(identity.full_source_hash, identity.hash_len);
    model->source_size = identity.declared_size;
    model->artifact_identity = "sha256:" + source_sha256;
    if (require_exact_qualified_artifact && !qwen3_artifact_identity_is_qualified(source_sha256) &&
        !(allow_experimental_qwen3_8b && qwen3_artifact_identity_is_experimental_8b(source_sha256)))
        throw std::runtime_error("Qwen3 production supports only the qualified Qwen3-14B Q4_K_M artifact; source_sha256=" + source_sha256);

    for (uint8_t kind = 0; kind < model->special_tokens.size(); ++kind) {
        uint64_t value = 0;
        if (vbuf_ml_consumer_special_token(model->handle, kind, &value) == 0) {
            if (value >= ffi.vocabulary_size) throw std::runtime_error("Qwen3 special token is outside vocabulary");
            model->special_tokens[kind] = value;
            model->special_token_present[kind] = true;
        }
    }
    if (!model->special_token_present[1])
        throw std::runtime_error("Qwen3 tokenizer metadata is missing EOS token");
    if (vbuf_ml_consumer_add_bos(model->handle, &model->add_bos) != 0)
        throw std::runtime_error("Qwen3 tokenizer metadata is missing add_bos policy");
    if (model->add_bos && !model->special_token_present[0])
        throw std::runtime_error("Qwen3 tokenizer requires BOS but has no BOS token");

    if (!source_endpoint.empty()) configure_qwen3_model_source(*model, source_endpoint);

    const Qwen3Tensor & down = qwen3_tensor(*model, "blk.0.ffn_down.weight");
    TensorGeometry geometry{};
    if (derive_tensor_geometry(down.generic, &geometry) != AdapterError::None)
        throw std::runtime_error("invalid Qwen3 block-0 FFN-down geometry");
    const auto * traits = ggml_get_type_traits(geometry.type);
    uint64_t rows = 1;
    for (uint8_t i = 1; i < down.view.rank; ++i) rows *= down.view.dimensions[i];
    const size_t row_bytes = ggml_row_size(geometry.type, geometry.ne[0]);
    const uint64_t blocks = rows * static_cast<uint64_t>(geometry.ne[0] / traits->blck_size);
    std::printf("model=Qwen3 layers=%u hidden=%llu heads=%llu kv_heads=%llu head_dim=%llu vocabulary=%u "
        "tensors=%llu source_size=%llu source_sha256=%s tokenizer_eos=%llu add_bos=%s\n",
        model->layer_count, static_cast<unsigned long long>(ffi.embedding_length),
        static_cast<unsigned long long>(ffi.head_count), static_cast<unsigned long long>(ffi.kv_head_count),
        static_cast<unsigned long long>(ffi.key_head_dimension), ffi.vocabulary_size,
        static_cast<unsigned long long>(model->count), static_cast<unsigned long long>(identity.declared_size),
        model->artifact_identity.c_str(), static_cast<unsigned long long>(model->special_tokens[1]),
        model->add_bos ? "YES" : "NO");
    std::printf("ffn_down_tensor name=%s shape=[%lld,%lld] representation=%u ggml_type=%s payload_bytes=%llu "
        "row_bytes=%zu total_blocks=%llu source_offset=%llu source_length=%llu\n", down.name.c_str(),
        static_cast<long long>(geometry.ne[0]), static_cast<long long>(geometry.ne[1]), down.view.representation,
        ggml_type_name(geometry.type), static_cast<unsigned long long>(down.length), row_bytes,
        static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(down.offset),
        static_cast<unsigned long long>(down.length));
}

} // namespace vbuf_ggml
