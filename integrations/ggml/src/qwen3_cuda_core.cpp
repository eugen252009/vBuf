#include "qwen3_cuda_core.h"

#include "vbuf_region_executor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vbuf_ggml {
namespace {
constexpr uint32_t embedding_width = 5120;
constexpr uint32_t attention_heads = 40;
constexpr uint32_t kv_heads = 8;
constexpr uint32_t head_dimension = 128;
constexpr uint32_t qwen3_layers = 40;
constexpr const char * embedding_name = "token_embd.weight";

void require_tensor(const ggml_tensor * tensor, const char * name) {
    if (tensor == nullptr) throw std::invalid_argument(std::string("Qwen CUDA graph missing ") + name);
}

ggml_tensor * checked_mul_mat(ggml_context * context, ggml_tensor * left,
    ggml_tensor * right, const char * label) {
    require_tensor(left, "matmul left input");
    require_tensor(right, "matmul right input");
    if (left->ne[0] != right->ne[0] || left->ne[2] == 0 || left->ne[3] == 0 ||
        right->ne[2] % left->ne[2] != 0 || right->ne[3] % left->ne[3] != 0)
        throw std::runtime_error(std::string("Qwen CUDA graph invalid matmul at ") + label);
    return ggml_mul_mat(context, left, right);
}

DeviceResidencyKey residency_key(const Qwen3Model & model, const Qwen3Tensor & tensor,
    const std::string & backend) {
    DeviceResidencyKey key;
    key.artifact_identity = model.artifact_identity;
    key.tensor_id = tensor.id;
    key.source_offset = tensor.offset;
    key.payload_length = tensor.length;
    key.representation = tensor.view.representation;
    key.shape.assign(tensor.view.dimensions, tensor.view.dimensions + tensor.view.rank);
    key.backend = backend;
    key.device_id = 0;
    return key;
}

void inject(QwenCudaFailurePoint actual, QwenCudaFailurePoint target, const char * label) {
    if (actual == target) throw std::runtime_error(std::string("injected Qwen CUDA resource failure: ") + label);
}
} // namespace

Qwen3CudaLayerGraph qwen3_cuda_build_layer(ggml_context * context,
    const Qwen3CudaLayerWeights & w, ggml_tensor * hidden,
    ggml_tensor * position_ids, ggml_tensor * causal_mask, ggml_tensor * cache_rows,
    ggml_tensor * key_cache, ggml_tensor * value_cache, ggml_tensor * packed_value_scratch,
    uint32_t query_count, uint32_t capacity, const Qwen3CudaCaptureTensor & capture) {
    require_tensor(context == nullptr ? nullptr : hidden, "hidden state");
    require_tensor(position_ids, "position IDs");
    require_tensor(causal_mask, "causal mask");
    require_tensor(cache_rows, "cache row indices");
    require_tensor(key_cache, "key cache");
    require_tensor(value_cache, "value cache");
    require_tensor(packed_value_scratch, "packed V scratch");
    require_tensor(w.attn_norm, "attention norm weight");
    require_tensor(w.q, "query weight");
    require_tensor(w.k, "key weight");
    require_tensor(w.v, "value weight");
    require_tensor(w.q_norm, "query norm weight");
    require_tensor(w.k_norm, "key norm weight");
    require_tensor(w.attn_out, "attention output weight");
    require_tensor(w.ffn_norm, "FFN norm weight");
    require_tensor(w.gate, "FFN gate weight");
    require_tensor(w.up, "FFN up weight");
    require_tensor(w.down, "FFN down weight");
    if (query_count == 0 || query_count > capacity ||
        hidden->ne[0] != embedding_width || position_ids->ne[0] != query_count ||
        causal_mask->ne[0] != capacity || causal_mask->ne[1] != query_count ||
        cache_rows->ne[0] != static_cast<int64_t>(kv_heads) * query_count ||
        key_cache->ne[0] != head_dimension || key_cache->ne[1] != static_cast<int64_t>(kv_heads) * capacity ||
        value_cache->ne[0] != head_dimension || value_cache->ne[1] != static_cast<int64_t>(kv_heads) * capacity)
        throw std::invalid_argument("Qwen CUDA layer graph geometry mismatch");

    const auto emit = [&](const char * name, ggml_tensor * tensor) {
        if (capture) capture(name, tensor);
    };
    emit("layer_input", hidden);
    ggml_tensor * attention_norm = ggml_mul(context,
        ggml_rms_norm(context, hidden, 1e-6f), w.attn_norm);
    emit("attention_rmsnorm", attention_norm);
    ggml_tensor * q_linear = checked_mul_mat(context, w.q, attention_norm, "q projection");
    ggml_tensor * k_linear = checked_mul_mat(context, w.k, attention_norm, "k projection");
    ggml_tensor * v_linear = checked_mul_mat(context, w.v, attention_norm, "v projection");
    emit("q_projection", q_linear);
    emit("k_projection", k_linear);
    emit("v_projection", v_linear);
    ggml_tensor * q_heads = ggml_reshape_3d(context, q_linear, head_dimension, attention_heads, query_count);
    ggml_tensor * k_heads = ggml_reshape_3d(context, k_linear, head_dimension, kv_heads, query_count);
    ggml_tensor * v_heads = ggml_reshape_3d(context, v_linear, head_dimension, kv_heads, query_count);
    ggml_tensor * q_norm = ggml_mul(context, ggml_rms_norm(context, q_heads, 1e-6f), w.q_norm);
    ggml_tensor * k_norm = ggml_mul(context, ggml_rms_norm(context, k_heads, 1e-6f), w.k_norm);
    emit("q_norm", q_norm);
    emit("k_norm", k_norm);
    ggml_tensor * q_rope = ggml_rope_ext(context, q_norm, position_ids, nullptr,
        head_dimension, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_rope = ggml_rope_ext(context, k_norm, position_ids, nullptr,
        head_dimension, GGML_ROPE_TYPE_NEOX, 0, 1000000.0f,
        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_current = ggml_reshape_2d(context,
        ggml_cast(context, k_rope, GGML_TYPE_F16), head_dimension, kv_heads * query_count);
    ggml_tensor * v_current = ggml_reshape_2d(context,
        ggml_cast(context, v_heads, GGML_TYPE_F16), head_dimension, kv_heads * query_count);
    emit("q_rope", q_rope);
    emit("k_rope", k_rope);
    emit("k_cache_f16", k_current);
    emit("v_cache_f16", v_current);
    ggml_tensor * k_updated = ggml_set_rows(context, key_cache, k_current, cache_rows);
    ggml_tensor * v_updated = ggml_set_rows(context, value_cache, v_current, cache_rows);
    ggml_tensor * q_batched = ggml_permute(context, q_rope, 0, 2, 1, 3);
    ggml_tensor * k_reshaped = ggml_reshape_3d(context, k_updated, head_dimension, kv_heads, capacity);
    ggml_tensor * v_reshaped = ggml_reshape_3d(context, v_updated, head_dimension, kv_heads, capacity);
    ggml_tensor * k_batched = ggml_permute(context, k_reshaped, 0, 2, 1, 3);
    // Layers execute in dependency order; each consumes packed V before the
    // next layer overwrites this session-owned scratch tensor.
    ggml_tensor * v_batched = ggml_cpy(context,
        ggml_permute(context, v_reshaped, 1, 2, 0, 3), packed_value_scratch);
    emit("k_attention_input", ggml_cont(context, k_batched));
    emit("v_attention_input", ggml_cont(context, v_batched));
    ggml_tensor * scores = checked_mul_mat(context, k_batched, q_batched, "attention scores");
    ggml_tensor * probabilities = ggml_soft_max_ext(context, scores, causal_mask,
        1.0f / std::sqrt(static_cast<float>(head_dimension)), 0.0f);
    emit("attention_scores", scores);
    emit("attention_probabilities", probabilities);
    ggml_tensor * context_heads = checked_mul_mat(context, v_batched, probabilities, "attention values");
    ggml_tensor * context_layout = ggml_permute(context, context_heads, 0, 2, 1, 3);
    ggml_tensor * attention_context = ggml_cont_2d(context, context_layout, embedding_width, query_count);
    emit("attention_context", attention_context);
    ggml_tensor * projected = checked_mul_mat(context, w.attn_out, attention_context, "attention output");
    ggml_tensor * residual = ggml_add(context, projected, hidden);
    ggml_tensor * ffn_norm = ggml_mul(context, ggml_rms_norm(context, residual, 1e-6f), w.ffn_norm);
    emit("attention_projection", projected);
    emit("attention_residual", residual);
    emit("ffn_rmsnorm", ffn_norm);
    ggml_tensor * gate = checked_mul_mat(context, w.gate, ffn_norm, "FFN gate");
    ggml_tensor * up = checked_mul_mat(context, w.up, ffn_norm, "FFN up");
    ggml_tensor * swiglu = ggml_mul(context, ggml_silu(context, gate), up);
    ggml_tensor * down = checked_mul_mat(context, w.down, swiglu, "FFN down");
    emit("ffn_gate", gate);
    emit("ffn_up", up);
    emit("ffn_swiglu", swiglu);
    emit("ffn_down", down);
    ggml_tensor * output = ggml_add(context, down, residual);
    emit("block_output", output);
    return { output, scores, probabilities };
}

struct QwenCudaRuntimeState::Impl {
    Qwen3Model * model = nullptr; // borrowed admitted metadata/source owner
    QwenCudaRuntimeConfig config{};
    ggml_backend_t backend = nullptr;
    ggml_context * model_context = nullptr;
    std::shared_ptr<void> model_allocation;
    std::shared_ptr<TensorResidencyStore> residency;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::string backend_name;
    uint64_t resident_bytes = 0;
    uint64_t uploaded_bytes = 0;
    size_t uploaded_tensors = 0;

    ~Impl() {
        if (backend != nullptr) ggml_backend_synchronize(backend);
        if (residency) residency->clear_device();
        model_allocation.reset();
        if (backend != nullptr) ggml_backend_free(backend);
        if (model_context != nullptr) ggml_free(model_context);
        if (model != nullptr) for (auto & entry : model->tensors) entry.second.ggml = nullptr;
    }
};

struct QwenCudaSessionState::Impl {
    std::shared_ptr<QwenCudaRuntimeState> runtime;
    uint32_t capacity = 0;
    uint32_t current_length = 0;
    uint64_t reset_generation = 0;
    ggml_context * context = nullptr;
    ggml_context * graph_context = nullptr;
    std::vector<ggml_context *> auxiliary_contexts;
    ggml_backend_buffer_t allocation = nullptr;
    ggml_backend_buffer_t prefill_scratch = nullptr;
    ggml_backend_buffer_t decode_scratch = nullptr;
    std::vector<ggml_tensor *> keys;
    std::vector<ggml_tensor *> values;
    ggml_tensor * packed_value = nullptr;
    QwenCudaFailurePoint inject_failure = QwenCudaFailurePoint::None;

    ~Impl() {
        if (runtime && runtime->backend() != nullptr)
            ggml_backend_synchronize(runtime->backend());
        if (prefill_scratch != nullptr) ggml_backend_buffer_free(prefill_scratch);
        if (decode_scratch != nullptr) ggml_backend_buffer_free(decode_scratch);
        if (allocation != nullptr) ggml_backend_buffer_free(allocation);
        for (ggml_context * auxiliary : auxiliary_contexts) if (auxiliary != nullptr) ggml_free(auxiliary);
        if (graph_context != nullptr) ggml_free(graph_context);
        if (context != nullptr) ggml_free(context);
    }
};

QwenCudaRuntimeState::QwenCudaRuntimeState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QwenCudaRuntimeState::~QwenCudaRuntimeState() = default;

std::shared_ptr<QwenCudaRuntimeState> QwenCudaRuntimeState::create(
    Qwen3Model & model, QwenCudaRuntimeConfig config) {
    if (model.handle == nullptr || model.materializer == nullptr || model.layer_count != qwen3_layers ||
        model.catalog.tensors.size() != model.count ||
        !qwen3_artifact_identity_is_qualified(model.artifact_identity.substr(model.artifact_identity.find(':') + 1)))
        throw std::invalid_argument("Qwen CUDA runtime requires admitted exact Qwen3-14B Q4_K_M metadata and source");
    if (config.prefill_chunk_size != 32 || config.prefill_scratch_bytes == 0 || config.decode_scratch_bytes == 0)
        throw std::invalid_argument("Qwen CUDA runtime requires the qualified 32-token prefill configuration");

    struct TensorBindingGuard {
        Qwen3Model & model;
        bool committed = false;
        ~TensorBindingGuard() {
            if (!committed) for (auto & entry : model.tensors) entry.second.ggml = nullptr;
        }
    } binding_guard{model};
    auto impl = std::make_unique<Impl>();
    impl->model = &model;
    impl->config = config;
    impl->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (impl->backend == nullptr) throw std::runtime_error("Qwen CUDA backend initialization failed");
    impl->backend_name = ggml_backend_name(impl->backend);
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterBackend, "backend");

    ggml_init_params params{ 16 * 1024 * 1024, nullptr, true };
    impl->model_context = ggml_init(params);
    if (impl->model_context == nullptr) throw std::runtime_error("Qwen CUDA model context allocation failed");
    std::vector<Qwen3Tensor *> ordered(static_cast<size_t>(model.count), nullptr);
    for (auto & entry : model.tensors) {
        Qwen3Tensor & tensor = entry.second;
        if (tensor.id >= ordered.size() || ordered[tensor.id] != nullptr || tensor.ggml != nullptr)
            throw std::runtime_error("Qwen CUDA tensor catalog has an invalid or already-bound tensor");
        ordered[tensor.id] = &tensor;
    }
    for (size_t index = 0; index < ordered.size(); ++index) {
        Qwen3Tensor * tensor = ordered[index];
        if (tensor == nullptr) throw std::runtime_error("Qwen CUDA tensor catalog is not dense");
        TensorGeometry geometry{};
        if (derive_tensor_geometry(tensor->generic, &geometry) != AdapterError::None ||
            geometry.nbytes != tensor->length)
            throw std::runtime_error("Qwen CUDA tensor descriptor rejected: " + tensor->name);
        tensor->ggml = ggml_new_tensor(impl->model_context, geometry.type, geometry.rank, geometry.ne);
        if (tensor->ggml == nullptr) throw std::runtime_error("Qwen CUDA tensor creation failed: " + tensor->name);
        impl->tensors.emplace(tensor->name, tensor->ggml);
        impl->resident_bytes += tensor->length;
    }
    ggml_backend_buffer_t raw_allocation = ggml_backend_alloc_ctx_tensors(impl->model_context, impl->backend);
    if (raw_allocation == nullptr) throw std::runtime_error("Qwen CUDA resident model allocation failed");
    impl->model_allocation = std::shared_ptr<void>(raw_allocation,
        [](void * allocation) { ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(allocation)); });
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterModelAllocation, "model allocation");

    impl->residency = std::make_shared<TensorResidencyStore>(0,
        ResidencyReplacementPolicyKind::LRU, impl->resident_bytes);
    std::vector<MaterializedTensor> leases;
    leases.reserve(ordered.size());
    std::vector<uint32_t> requested;
    requested.reserve(ordered.size());
    try {
        for (uint32_t index = 0; index < ordered.size(); ++index) {
            Qwen3Tensor & tensor = *ordered[index];
            if (!model.materializer->request(index, tensor.persistent(), impl->resident_bytes))
                throw std::runtime_error("Qwen CUDA weight materialization request failed: " + tensor.name);
            requested.push_back(index);
            if (model.materializer->wait(index) != MaterializationState::Ready)
                throw std::runtime_error("Qwen CUDA weight materialization failed: " + tensor.name);
            auto lease = model.materializer->obtain_ready_tensor(index);
            if (!lease || lease->payload == nullptr || lease->payload_len != tensor.length)
                throw std::runtime_error("Qwen CUDA weight payload lease failed: " + tensor.name);
            ggml_backend_tensor_set_async(impl->backend, tensor.ggml, lease->payload, 0, tensor.length);
            impl->uploaded_bytes += tensor.length;
            ++impl->uploaded_tensors;
            leases.push_back(*lease);
        }
        ggml_backend_synchronize(impl->backend);
        for (uint32_t index : requested) model.materializer->release(index);
        requested.clear();
        leases.clear();
    } catch (...) {
        ggml_backend_synchronize(impl->backend);
        for (uint32_t index : requested) model.materializer->release(index);
        throw;
    }

    for (Qwen3Tensor * tensor : ordered) {
        if (!impl->residency->insert_device(residency_key(model, *tensor, impl->backend_name),
            { impl->model_allocation, tensor->ggml, tensor->length }))
            throw std::runtime_error("Qwen CUDA resident tensor registration failed: " + tensor->name);
    }
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterResidency, "residency");
    if (impl->tensors.find(embedding_name) == impl->tensors.end())
        throw std::runtime_error("Qwen CUDA resident embedding is missing");
    auto state = std::shared_ptr<QwenCudaRuntimeState>(new QwenCudaRuntimeState(std::move(impl)));
    binding_guard.committed = true;
    return state;
}

std::shared_ptr<QwenCudaRuntimeState> QwenCudaRuntimeState::create_for_testing(QwenCudaRuntimeConfig config) {
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    impl->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (impl->backend == nullptr) throw std::runtime_error("test Qwen backend initialization failed");
    impl->backend_name = ggml_backend_name(impl->backend);
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterBackend, "test backend");
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    impl->model_context = ggml_init(params);
    if (impl->model_context == nullptr) throw std::runtime_error("test Qwen model context allocation failed");
    ggml_tensor * embedding = ggml_new_tensor_2d(impl->model_context, GGML_TYPE_F32, 8, 4);
    if (embedding == nullptr) throw std::runtime_error("test Qwen embedding creation failed");
    impl->tensors.emplace(embedding_name, embedding);
    impl->resident_bytes = ggml_nbytes(embedding);
    ggml_backend_buffer_t raw = ggml_backend_alloc_ctx_tensors(impl->model_context, impl->backend);
    if (raw == nullptr) throw std::runtime_error("test Qwen model allocation failed");
    impl->model_allocation = std::shared_ptr<void>(raw,
        [](void * allocation) { ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(allocation)); });
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterModelAllocation, "test model allocation");
    impl->residency = std::make_shared<TensorResidencyStore>(0,
        ResidencyReplacementPolicyKind::LRU, impl->resident_bytes);
    if (!impl->residency->insert_device({"test-qwen", 0, 0, impl->resident_bytes, 0,
            {8, 4}, impl->backend_name, 0}, {impl->model_allocation, embedding, impl->resident_bytes}))
        throw std::runtime_error("test Qwen residency setup failed");
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterResidency, "test residency");
    return std::shared_ptr<QwenCudaRuntimeState>(new QwenCudaRuntimeState(std::move(impl)));
}

std::shared_ptr<QwenCudaSessionState> QwenCudaRuntimeState::create_session(
    uint32_t capacity, QwenCudaFailurePoint fail_at) {
    if (capacity == 0 || capacity > 4096) throw std::invalid_argument("Qwen CUDA session capacity must be in 1..4096");
    if (fail_at == QwenCudaFailurePoint::None) fail_at = impl_->config.inject_failure;
    auto state = std::make_unique<QwenCudaSessionState::Impl>();
    state->runtime = shared_from_this();
    state->capacity = capacity;
    state->inject_failure = fail_at;
    ggml_init_params params{ 8 * 1024 * 1024, nullptr, true };
    state->context = ggml_init(params);
    if (state->context == nullptr) throw std::runtime_error("Qwen CUDA session context allocation failed");
    state->keys.reserve(qwen3_layers);
    state->values.reserve(qwen3_layers);
    for (uint32_t layer = 0; layer < qwen3_layers; ++layer) {
        state->keys.push_back(ggml_new_tensor_2d(state->context, GGML_TYPE_F16,
            head_dimension, kv_heads * capacity));
        state->values.push_back(ggml_new_tensor_2d(state->context, GGML_TYPE_F16,
            head_dimension, kv_heads * capacity));
        if (state->keys.back() == nullptr || state->values.back() == nullptr)
            throw std::runtime_error("Qwen CUDA session KV tensor creation failed");
    }
    state->packed_value = ggml_new_tensor_3d(state->context, GGML_TYPE_F16,
        capacity, head_dimension, kv_heads);
    if (state->packed_value == nullptr) throw std::runtime_error("Qwen CUDA packed-V scratch creation failed");
    state->allocation = ggml_backend_alloc_ctx_tensors(state->context, impl_->backend);
    if (state->allocation == nullptr) throw std::runtime_error("Qwen CUDA session KV allocation failed");
    inject(fail_at, QwenCudaFailurePoint::SessionAfterKvAllocation, "session KV allocation");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(impl_->backend);
    state->prefill_scratch = ggml_backend_buft_alloc_buffer(buft, impl_->config.prefill_scratch_bytes);
    if (state->prefill_scratch == nullptr) throw std::runtime_error("Qwen CUDA prefill scratch allocation failed");
    inject(fail_at, QwenCudaFailurePoint::SessionAfterPrefillScratch, "prefill scratch");
    ggml_init_params graph_params{ 256 * 1024 * 1024, nullptr, true };
    state->graph_context = ggml_init(graph_params);
    if (state->graph_context == nullptr) throw std::runtime_error("Qwen CUDA session graph context allocation failed");
    return std::shared_ptr<QwenCudaSessionState>(new QwenCudaSessionState(std::move(state)));
}

// The CPU test fixture has no Qwen metadata, but uses the same backend owner.
// Production metadata is retained by the admitted Qwen3Model, whose lifetime
// contract is documented on QwenCudaRuntimeState::create.
ggml_backend_t QwenCudaRuntimeState::backend() const noexcept { return impl_->backend; }
ggml_backend_dev_t QwenCudaRuntimeState::device() const noexcept { return ggml_backend_get_device(impl_->backend); }
void QwenCudaRuntimeState::device_memory(size_t * free_bytes, size_t * total_bytes) const noexcept {
    if (impl_->backend != nullptr) ggml_backend_dev_memory(device(), free_bytes, total_bytes);
}
ggml_context * QwenCudaRuntimeState::model_context() const noexcept { return impl_->model_context; }
ggml_backend_buffer_t QwenCudaRuntimeState::model_allocation() const noexcept {
    return impl_->model_allocation ? static_cast<ggml_backend_buffer_t>(impl_->model_allocation.get()) : nullptr;
}
const std::string & QwenCudaRuntimeState::backend_name() const noexcept { return impl_->backend_name; }
const std::string & QwenCudaRuntimeState::artifact_identity() const noexcept {
    static const std::string fixture{"test-qwen"};
    return impl_->model != nullptr ? impl_->model->artifact_identity : fixture;
}
const ModelMetadataDescriptor * QwenCudaRuntimeState::metadata() const noexcept {
    return impl_->model != nullptr ? &impl_->model->metadata : nullptr;
}
ggml_tensor * QwenCudaRuntimeState::tensor(const std::string & name) const {
    const auto found = impl_->tensors.find(name);
    return found == impl_->tensors.end() ? nullptr : found->second;
}
ggml_tensor * QwenCudaRuntimeState::embedding() const noexcept {
    const auto found = impl_->tensors.find(embedding_name);
    return found == impl_->tensors.end() ? nullptr : found->second;
}
std::shared_ptr<TensorResidencyStore> QwenCudaRuntimeState::residency() const noexcept { return impl_->residency; }
uint64_t QwenCudaRuntimeState::resident_model_bytes() const noexcept { return impl_->resident_bytes; }
uint64_t QwenCudaRuntimeState::uploaded_payload_bytes() const noexcept { return impl_->uploaded_bytes; }
size_t QwenCudaRuntimeState::uploaded_tensor_count() const noexcept { return impl_->uploaded_tensors; }
size_t QwenCudaRuntimeState::resident_allocation_bytes() const noexcept {
    return impl_->model_allocation ? ggml_backend_buffer_get_size(
        static_cast<ggml_backend_buffer_t>(impl_->model_allocation.get())) : 0;
}
size_t QwenCudaRuntimeState::resident_tensor_count() const noexcept {
    return impl_->residency ? impl_->residency->device_resident_count() : 0;
}
uint32_t QwenCudaRuntimeState::prefill_chunk_size() const noexcept { return impl_->config.prefill_chunk_size; }

QwenCudaSessionState::QwenCudaSessionState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QwenCudaSessionState::~QwenCudaSessionState() = default;
void QwenCudaSessionState::reset() noexcept {
    impl_->current_length = 0;
    ++impl_->reset_generation;
}
void QwenCudaSessionState::commit_tokens(uint32_t count) {
    if (count > impl_->capacity - impl_->current_length)
        throw std::out_of_range("Qwen CUDA session token commit exceeds capacity");
    impl_->current_length += count;
}
uint32_t QwenCudaSessionState::current_length() const noexcept { return impl_->current_length; }
uint32_t QwenCudaSessionState::capacity() const noexcept { return impl_->capacity; }
uint64_t QwenCudaSessionState::reset_generation() const noexcept { return impl_->reset_generation; }
const std::shared_ptr<QwenCudaRuntimeState> & QwenCudaSessionState::runtime() const noexcept { return impl_->runtime; }
ggml_context * QwenCudaSessionState::context() const noexcept { return impl_->context; }
ggml_context * QwenCudaSessionState::graph_context() const noexcept { return impl_->graph_context; }
ggml_context * QwenCudaSessionState::create_auxiliary_context(size_t arena_bytes) {
    if (arena_bytes == 0) throw std::invalid_argument("Qwen auxiliary graph arena must be nonzero");
    ggml_init_params params{ arena_bytes, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("Qwen auxiliary graph context allocation failed");
    try { impl_->auxiliary_contexts.push_back(context); }
    catch (...) { ggml_free(context); throw; }
    return context;
}
ggml_backend_buffer_t QwenCudaSessionState::allocation() const noexcept { return impl_->allocation; }
ggml_backend_buffer_t QwenCudaSessionState::allocate_decode_scratch() {
    if (impl_->decode_scratch != nullptr) return impl_->decode_scratch;
    ggml_backend_t backend = impl_->runtime->backend();
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(impl_->graph_context, buft);
    const size_t bytes = std::max(required, impl_->runtime->impl_->config.decode_scratch_bytes);
    if (bytes == 0) throw std::runtime_error("Qwen CUDA decode graph has no scratch tensors");
    impl_->decode_scratch = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (impl_->decode_scratch == nullptr) throw std::runtime_error("Qwen CUDA decode scratch allocation failed");
    ggml_tallocr allocator = ggml_tallocr_new(impl_->decode_scratch);
    for (ggml_tensor * tensor = ggml_get_first_tensor(impl_->graph_context); tensor != nullptr;
        tensor = ggml_get_next_tensor(impl_->graph_context, tensor)) {
        if (tensor->buffer != nullptr) continue;
        ggml_status status = GGML_STATUS_SUCCESS;
        if (tensor->view_src != nullptr) status = ggml_backend_view_init(tensor);
        else if (tensor->data == nullptr) status = ggml_tallocr_alloc(&allocator, tensor);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("Qwen CUDA decode scratch binding failed at tensor ") + tensor->name);
    }
    inject(impl_->inject_failure, QwenCudaFailurePoint::SessionAfterDecodeScratch, "decode scratch");
    return impl_->decode_scratch;
}
size_t QwenCudaSessionState::allocation_bytes() const noexcept {
    return impl_->allocation == nullptr ? 0 : ggml_backend_buffer_get_size(impl_->allocation);
}
size_t QwenCudaSessionState::prefill_scratch_bytes() const noexcept {
    return impl_->prefill_scratch == nullptr ? 0 : ggml_backend_buffer_get_size(impl_->prefill_scratch);
}
size_t QwenCudaSessionState::decode_scratch_bytes() const noexcept {
    return impl_->decode_scratch == nullptr ? 0 : ggml_backend_buffer_get_size(impl_->decode_scratch);
}
ggml_backend_buffer_t QwenCudaSessionState::prefill_scratch() const noexcept { return impl_->prefill_scratch; }
ggml_backend_buffer_t QwenCudaSessionState::decode_scratch() const noexcept { return impl_->decode_scratch; }
ggml_tensor * QwenCudaSessionState::key_cache(uint32_t layer) const {
    if (layer >= impl_->keys.size()) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    return impl_->keys[layer];
}
ggml_tensor * QwenCudaSessionState::value_cache(uint32_t layer) const {
    if (layer >= impl_->values.size()) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    return impl_->values[layer];
}
ggml_tensor * QwenCudaSessionState::packed_value_scratch() const noexcept { return impl_->packed_value; }

} // namespace vbuf_ggml
