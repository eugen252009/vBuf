#include "qwen3_cuda_core.h"

#include "qwen3_execution_plan.h"
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
    const std::string & backend, uint32_t device_id) {
    DeviceResidencyKey key;
    key.artifact_identity = model.artifact_identity;
    key.tensor_id = tensor.id;
    key.source_offset = tensor.offset;
    key.payload_length = tensor.length;
    key.representation = tensor.view.representation;
    key.shape.assign(tensor.view.dimensions, tensor.view.dimensions + tensor.view.rank);
    key.backend = backend;
    key.device_id = device_id;
    return key;
}

void inject(QwenCudaFailurePoint actual, QwenCudaFailurePoint target, const char * label) {
    if (actual == target) throw std::runtime_error(std::string("injected Qwen CUDA resource failure: ") + label);
}
} // namespace

QwenCudaPlacement QwenCudaPlacement::single_device(uint32_t device_id) {
    QwenCudaPlacement result;
    result.block_device_ids.assign(qwen3_layers, device_id);
    result.embedding_device_id = device_id;
    result.output_norm_device_id = device_id;
    result.output_head_device_id = device_id;
    return result;
}

QwenCudaPlacement QwenCudaPlacement::contiguous_split(uint32_t early_device_id,
    uint32_t late_device_id, uint32_t early_block_count) {
    if (early_block_count == 0 || early_block_count >= qwen3_layers || early_device_id == late_device_id)
        throw std::invalid_argument("Qwen CUDA contiguous split requires two devices and a cut inside the layer stack");
    QwenCudaPlacement result;
    result.block_device_ids.reserve(qwen3_layers);
    for (uint32_t layer = 0; layer < qwen3_layers; ++layer)
        result.block_device_ids.push_back(layer < early_block_count ? early_device_id : late_device_id);
    result.embedding_device_id = early_device_id;
    result.output_norm_device_id = late_device_id;
    result.output_head_device_id = late_device_id;
    return result;
}

uint32_t QwenCudaPlacement::owner_for_tensor(const std::string & name) const {
    if (name == embedding_name) return embedding_device_id;
    if (name == "output_norm.weight") return output_norm_device_id;
    if (name == "output.weight") return output_head_device_id;
    if (name.rfind("blk.", 0) != 0) throw std::invalid_argument("Qwen CUDA placement does not recognize tensor: " + name);
    const size_t end = name.find('.', 4);
    if (end == 4 || end == std::string::npos || end + 1 >= name.size())
        throw std::invalid_argument("Qwen CUDA placement has malformed block tensor name: " + name);
    uint32_t layer = 0;
    for (size_t i = 4; i < end; ++i) {
        if (name[i] < '0' || name[i] > '9' || layer > (UINT32_MAX - 9) / 10)
            throw std::invalid_argument("Qwen CUDA placement has malformed block index: " + name);
        layer = layer * 10 + static_cast<uint32_t>(name[i] - '0');
    }
    if (layer >= block_device_ids.size())
        throw std::out_of_range("Qwen CUDA placement block index exceeds configured layer map: " + name);
    return block_device_ids[layer];
}

std::vector<uint32_t> QwenCudaPlacement::device_ids() const {
    std::vector<uint32_t> result{embedding_device_id, output_norm_device_id, output_head_device_id};
    result.insert(result.end(), block_device_ids.begin(), block_device_ids.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

void QwenCudaPlacement::validate(const Qwen3Model & model) const {
    if (model.layer_count != qwen3_layers || block_device_ids.size() != model.layer_count)
        throw std::invalid_argument("Qwen CUDA placement must map every admitted transformer block exactly once");
    const uint32_t early = block_device_ids.front();
    const auto cut_it = std::find_if(block_device_ids.begin(), block_device_ids.end(),
        [early](uint32_t id) { return id != early; });
    const size_t cut = static_cast<size_t>(cut_it - block_device_ids.begin());
    if (cut == block_device_ids.size()) {
        if (embedding_device_id != early || output_norm_device_id != early || output_head_device_id != early)
            throw std::invalid_argument("single-device Qwen placement globals must share the block owner");
    } else {
        const uint32_t late = *cut_it;
        if (cut == 0 || late == early || embedding_device_id != early ||
            output_norm_device_id != late || output_head_device_id != late)
            throw std::invalid_argument("Qwen CUDA split placement globals must follow early/late device ownership");
        for (size_t layer = cut; layer < block_device_ids.size(); ++layer)
            if (block_device_ids[layer] != late)
                throw std::invalid_argument("Qwen CUDA placement must have exactly one contiguous early/late cut");
    }
    std::unordered_map<std::string, bool> seen;
    for (const auto & item : model.tensors) {
        if (!seen.emplace(item.first, true).second) throw std::invalid_argument("duplicate Qwen tensor ownership: " + item.first);
        (void) owner_for_tensor(item.first);
    }
    if (seen.size() != static_cast<size_t>(model.layer_count) * 11 + 3)
        throw std::invalid_argument("Qwen CUDA placement tensor catalog is incomplete");
    for (const char * global : {embedding_name, "output_norm.weight", "output.weight"})
        if (seen.count(global) != 1) throw std::invalid_argument(std::string("Qwen CUDA placement missing global tensor: ") + global);
    for (uint32_t layer = 0; layer < model.layer_count; ++layer)
        for (const char * suffix : {"attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                "attn_q_norm.weight", "attn_k_norm.weight", "attn_output.weight", "ffn_norm.weight",
                "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
            const std::string name = "blk." + std::to_string(layer) + "." + suffix;
            if (seen.count(name) != 1) throw std::invalid_argument("Qwen CUDA placement missing block tensor: " + name);
        }
}


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
    struct DeviceRuntime {
        uint32_t device_id = 0;
        ggml_backend_dev_t device = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_context * model_context = nullptr;
        std::shared_ptr<void> model_allocation;
        std::unordered_map<std::string, ggml_tensor *> tensors;
        std::string backend_name;
        uint64_t resident_bytes = 0;
        uint64_t uploaded_bytes = 0;
        size_t uploaded_tensors = 0;

        ~DeviceRuntime() {
            if (backend != nullptr) ggml_backend_synchronize(backend);
            model_allocation.reset();
            if (model_context != nullptr) ggml_free(model_context);
            if (backend != nullptr) ggml_backend_free(backend);
        }
    };

    Qwen3Model * model = nullptr; // borrowed admitted metadata/source owner
    QwenCudaRuntimeConfig config{};
    std::vector<std::unique_ptr<DeviceRuntime>> devices;
    std::unordered_map<std::string, uint32_t> tensor_owners;
    std::shared_ptr<TensorResidencyStore> residency;
    QwenExecutionPlanOptimizer execution_optimizer;

    DeviceRuntime & by_id(uint32_t device_id) const {
        for (const auto & item : devices) if (item->device_id == device_id) return *item;
        throw std::out_of_range("Qwen CUDA device is not part of this placement");
    }

    ~Impl() {
        if (residency) residency->clear_device();
        devices.clear();
        if (model != nullptr) for (auto & entry : model->tensors) entry.second.ggml = nullptr;
    }
};

struct QwenCudaSessionState::Impl {
    struct DeviceSession {
        uint32_t device_id = 0;
        ggml_context * context = nullptr;
        ggml_context * graph_context = nullptr;
        std::vector<ggml_context *> auxiliary_contexts;
        ggml_backend_buffer_t allocation = nullptr;
        ggml_backend_buffer_t prefill_scratch = nullptr;
        ggml_backend_buffer_t decode_scratch = nullptr;
        std::vector<ggml_tensor *> keys;
        std::vector<ggml_tensor *> values;
        ggml_tensor * packed_value = nullptr;
    };

    std::shared_ptr<QwenCudaRuntimeState> runtime;
    uint32_t capacity = 0;
    uint32_t current_length = 0;
    uint64_t reset_generation = 0;
    std::vector<DeviceSession> devices;
    ggml_backend_buffer_t boundary_host_buffer = nullptr;
    size_t boundary_host_bytes = 0;
    std::vector<uint32_t> boundary_audit_end_positions;
    QwenCudaFailurePoint inject_failure = QwenCudaFailurePoint::None;

    DeviceSession & by_id(uint32_t id) {
        for (auto & device : devices) if (device.device_id == id) return device;
        throw std::out_of_range("Qwen session does not own the requested device");
    }
    const DeviceSession & by_id(uint32_t id) const {
        for (const auto & device : devices) if (device.device_id == id) return device;
        throw std::out_of_range("Qwen session does not own the requested device");
    }
    ~Impl() {
        for (auto & device : devices) ggml_backend_synchronize(runtime->backend(device.device_id));
        if (boundary_host_buffer != nullptr) ggml_backend_buffer_free(boundary_host_buffer);
        for (auto & device : devices) {
            if (device.prefill_scratch != nullptr) ggml_backend_buffer_free(device.prefill_scratch);
            if (device.decode_scratch != nullptr) ggml_backend_buffer_free(device.decode_scratch);
            if (device.allocation != nullptr) ggml_backend_buffer_free(device.allocation);
            for (ggml_context * auxiliary : device.auxiliary_contexts) if (auxiliary != nullptr) ggml_free(auxiliary);
            if (device.graph_context != nullptr) ggml_free(device.graph_context);
            if (device.context != nullptr) ggml_free(device.context);
        }
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
    const bool multi_device = config.placement.device_ids().size() == 2;
    if ((config.prefill_chunk_size != 32 && !(multi_device && config.prefill_chunk_size == 16)) ||
        config.prefill_scratch_bytes == 0 || config.decode_scratch_bytes == 0)
        throw std::invalid_argument("Qwen CUDA runtime supports 32-token prefill, or 16-token prefill for explicit two-device qualification");
    config.placement.validate(model);

    struct TensorBindingGuard {
        Qwen3Model & model;
        bool committed = false;
        ~TensorBindingGuard() {
            if (!committed) for (auto & entry : model.tensors) entry.second.ggml = nullptr;
        }
    } binding_guard{model};
    for (auto & entry : model.tensors)
        if (entry.second.ggml != nullptr) throw std::runtime_error("Qwen CUDA model tensor is already bound: " + entry.first);

    auto impl = std::make_unique<Impl>();
    impl->model = &model;
    impl->config = config;
    ggml_backend_reg_t cuda_registry = ggml_backend_reg_by_name("CUDA");
    if (cuda_registry == nullptr) throw std::runtime_error("Qwen CUDA backend registry is unavailable");
    const size_t cuda_devices = ggml_backend_reg_dev_count(cuda_registry);
    const std::vector<uint32_t> requested_devices = config.placement.device_ids();
    for (uint32_t id : requested_devices) {
        if (id >= cuda_devices) throw std::invalid_argument("Qwen CUDA placement refers to an unavailable CUDA device");
        auto device_runtime = std::make_unique<Impl::DeviceRuntime>();
        device_runtime->device_id = id;
        device_runtime->device = ggml_backend_reg_dev_get(cuda_registry, id);
        if (device_runtime->device == nullptr ||
            ggml_backend_dev_type(device_runtime->device) != GGML_BACKEND_DEVICE_TYPE_GPU)
            throw std::runtime_error("Qwen CUDA placement device is not a CUDA GPU");
        device_runtime->backend = ggml_backend_dev_init(device_runtime->device, nullptr);
        if (device_runtime->backend == nullptr)
            throw std::runtime_error("Qwen CUDA backend initialization failed for device " + std::to_string(id));
        device_runtime->backend_name = ggml_backend_name(device_runtime->backend);
        inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterBackend, "device backend");
        impl->devices.push_back(std::move(device_runtime));
    }

    std::vector<Qwen3Tensor *> ordered(static_cast<size_t>(model.count), nullptr);
    for (auto & entry : model.tensors) {
        Qwen3Tensor & tensor = entry.second;
        if (tensor.id >= ordered.size() || ordered[tensor.id] != nullptr || tensor.ggml != nullptr)
            throw std::runtime_error("Qwen CUDA tensor catalog has an invalid or already-bound tensor");
        ordered[tensor.id] = &tensor;
        impl->tensor_owners.emplace(tensor.name, config.placement.owner_for_tensor(tensor.name));
    }
    for (size_t index = 0; index < ordered.size(); ++index)
        if (ordered[index] == nullptr) throw std::runtime_error("Qwen CUDA tensor catalog is not dense");

    for (auto & device_runtime_ptr : impl->devices) {
        auto & device_runtime = *device_runtime_ptr;
        ggml_init_params params{ 16 * 1024 * 1024, nullptr, true };
        device_runtime.model_context = ggml_init(params);
        if (device_runtime.model_context == nullptr)
            throw std::runtime_error("Qwen CUDA model context allocation failed on device " + std::to_string(device_runtime.device_id));
        for (Qwen3Tensor * tensor : ordered) {
            if (impl->tensor_owners.at(tensor->name) != device_runtime.device_id) continue;
            TensorGeometry geometry{};
            if (derive_tensor_geometry(tensor->generic, &geometry) != AdapterError::None || geometry.nbytes != tensor->length)
                throw std::runtime_error("Qwen CUDA tensor descriptor rejected: " + tensor->name);
            ggml_tensor * resident = ggml_new_tensor(device_runtime.model_context, geometry.type, geometry.rank, geometry.ne);
            if (resident == nullptr) throw std::runtime_error("Qwen CUDA tensor creation failed: " + tensor->name);
            device_runtime.tensors.emplace(tensor->name, resident);
            device_runtime.resident_bytes += tensor->length;
        }
        if (device_runtime.tensors.empty()) throw std::runtime_error("Qwen CUDA placement assigned no model tensors to a device");
        ggml_backend_buffer_t raw_allocation = ggml_backend_alloc_ctx_tensors(device_runtime.model_context, device_runtime.backend);
        if (raw_allocation == nullptr)
            throw std::runtime_error("Qwen CUDA resident model allocation failed on device " + std::to_string(device_runtime.device_id));
        device_runtime.model_allocation = std::shared_ptr<void>(raw_allocation,
            [](void * allocation) { ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(allocation)); });
        inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterModelAllocation, "device model allocation");
    }

    uint64_t total_payload_bytes = 0;
    for (const Qwen3Tensor * tensor : ordered) total_payload_bytes += tensor->length;
    impl->residency = std::make_shared<TensorResidencyStore>(0,
        ResidencyReplacementPolicyKind::LRU, total_payload_bytes);
    std::vector<uint32_t> requested;
    requested.reserve(ordered.size());
    std::vector<MaterializedTensor> leases;
    leases.reserve(ordered.size());
    try {
        for (uint32_t index = 0; index < ordered.size(); ++index) {
            Qwen3Tensor & tensor = *ordered[index];
            const uint32_t owner = impl->tensor_owners.at(tensor.name);
            auto & device_runtime = impl->by_id(owner);
            if (!model.materializer->request(index, tensor.persistent(), total_payload_bytes))
                throw std::runtime_error("Qwen CUDA weight materialization request failed: " + tensor.name);
            requested.push_back(index);
            if (model.materializer->wait(index) != MaterializationState::Ready)
                throw std::runtime_error("Qwen CUDA weight materialization failed: " + tensor.name);
            auto lease = model.materializer->obtain_ready_tensor(index);
            if (!lease || lease->payload == nullptr || lease->payload_len != tensor.length)
                throw std::runtime_error("Qwen CUDA weight payload lease failed: " + tensor.name);
            ggml_backend_tensor_set_async(device_runtime.backend, device_runtime.tensors.at(tensor.name),
                lease->payload, 0, tensor.length);
            device_runtime.uploaded_bytes += tensor.length;
            ++device_runtime.uploaded_tensors;
            // Keep materialized pages leased until all device transfers complete.
            leases.push_back(*lease);
        }
        for (auto & device_runtime : impl->devices) ggml_backend_synchronize(device_runtime->backend);
        leases.clear();
        for (uint32_t index : requested) model.materializer->release(index);
        requested.clear();
    } catch (...) {
        for (auto & device_runtime : impl->devices) ggml_backend_synchronize(device_runtime->backend);
        leases.clear();
        for (uint32_t index : requested) model.materializer->release(index);
        throw;
    }

    for (Qwen3Tensor * tensor : ordered) {
        const uint32_t owner = impl->tensor_owners.at(tensor->name);
        auto & device_runtime = impl->by_id(owner);
        if (!impl->residency->insert_device(residency_key(model, *tensor,
                device_runtime.backend_name, owner),
            { device_runtime.model_allocation, device_runtime.tensors.at(tensor->name), tensor->length }))
            throw std::runtime_error("Qwen CUDA resident tensor registration failed: " + tensor->name);
    }
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterResidency, "residency");
    if (impl->tensor_owners.count(embedding_name) != 1)
        throw std::runtime_error("Qwen CUDA resident embedding is missing");
    if (impl->devices.size() == 1) {
        const uint32_t only = impl->devices.front()->device_id;
        for (Qwen3Tensor * tensor : ordered) tensor->ggml = impl->devices.front()->tensors.at(tensor->name);
        if (config.placement.embedding_device_id != only) throw std::logic_error("single-device embedding owner mismatch");
    }
    auto state = std::shared_ptr<QwenCudaRuntimeState>(new QwenCudaRuntimeState(std::move(impl)));
    binding_guard.committed = true;
    return state;
}

std::shared_ptr<QwenCudaRuntimeState> QwenCudaRuntimeState::create_for_testing(QwenCudaRuntimeConfig config) {
    auto impl = std::make_unique<Impl>();
    impl->config = config;
    auto device = std::make_unique<Impl::DeviceRuntime>();
    device->device_id = 0;
    device->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (device->backend == nullptr) throw std::runtime_error("test Qwen backend initialization failed");
    device->device = ggml_backend_get_device(device->backend);
    device->backend_name = ggml_backend_name(device->backend);
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterBackend, "test backend");
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    device->model_context = ggml_init(params);
    if (device->model_context == nullptr) throw std::runtime_error("test Qwen model context allocation failed");
    ggml_tensor * embedding = ggml_new_tensor_2d(device->model_context, GGML_TYPE_F32, 8, 4);
    if (embedding == nullptr) throw std::runtime_error("test Qwen embedding creation failed");
    device->tensors.emplace(embedding_name, embedding);
    device->resident_bytes = ggml_nbytes(embedding);
    device->uploaded_bytes = device->resident_bytes;
    device->uploaded_tensors = 1;
    ggml_backend_buffer_t raw = ggml_backend_alloc_ctx_tensors(device->model_context, device->backend);
    if (raw == nullptr) throw std::runtime_error("test Qwen model allocation failed");
    device->model_allocation = std::shared_ptr<void>(raw,
        [](void * allocation) { ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(allocation)); });
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterModelAllocation, "test model allocation");
    impl->devices.push_back(std::move(device));
    impl->tensor_owners.emplace(embedding_name, 0);
    impl->residency = std::make_shared<TensorResidencyStore>(0,
        ResidencyReplacementPolicyKind::LRU, impl->devices.front()->resident_bytes);
    if (!impl->residency->insert_device({"test-qwen", 0, 0, impl->devices.front()->resident_bytes, 0,
            {8, 4}, impl->devices.front()->backend_name, 0},
            {impl->devices.front()->model_allocation, embedding, impl->devices.front()->resident_bytes}))
        throw std::runtime_error("test Qwen residency setup failed");
    inject(config.inject_failure, QwenCudaFailurePoint::RuntimeAfterResidency, "test residency");
    return std::shared_ptr<QwenCudaRuntimeState>(new QwenCudaRuntimeState(std::move(impl)));
}

std::shared_ptr<QwenCudaSessionState> QwenCudaRuntimeState::create_session(
    uint32_t capacity, QwenCudaFailurePoint fail_at) {
    const uint32_t capacity_limit = impl_->config.allow_experimental_capacity ? 32768 : 4096;
    if (capacity == 0 || capacity > capacity_limit)
        throw std::invalid_argument("Qwen CUDA session capacity exceeds the configured qualification limit");
    if (fail_at == QwenCudaFailurePoint::None) fail_at = impl_->config.inject_failure;
    auto state = std::make_unique<QwenCudaSessionState::Impl>();
    state->runtime = shared_from_this();
    state->capacity = capacity;
    state->inject_failure = fail_at;
    const std::vector<uint32_t> device_ids = impl_->config.placement.device_ids();
    for (uint32_t id : device_ids) {
        state->devices.emplace_back();
        auto & local = state->devices.back();
        local.device_id = id;
        local.keys.assign(qwen3_layers, nullptr);
        local.values.assign(qwen3_layers, nullptr);
        ggml_init_params params{ 8 * 1024 * 1024, nullptr, true };
        local.context = ggml_init(params);
        if (local.context == nullptr) throw std::runtime_error("Qwen CUDA session context allocation failed");
        for (uint32_t layer = 0; layer < qwen3_layers; ++layer) {
            if (impl_->config.placement.block_device_ids[layer] != id) continue;
            local.keys[layer] = ggml_new_tensor_2d(local.context, GGML_TYPE_F16,
                head_dimension, kv_heads * capacity);
            local.values[layer] = ggml_new_tensor_2d(local.context, GGML_TYPE_F16,
                head_dimension, kv_heads * capacity);
            if (local.keys[layer] == nullptr || local.values[layer] == nullptr)
                throw std::runtime_error("Qwen CUDA session KV tensor creation failed");
        }
        local.packed_value = ggml_new_tensor_3d(local.context, GGML_TYPE_F16,
            capacity, head_dimension, kv_heads);
        if (local.packed_value == nullptr) throw std::runtime_error("Qwen CUDA packed-V scratch creation failed");
        local.allocation = ggml_backend_alloc_ctx_tensors(local.context, backend(id));
        if (local.allocation == nullptr)
            throw std::runtime_error("Qwen CUDA session KV allocation failed on device " + std::to_string(id));
    }
    inject(fail_at, QwenCudaFailurePoint::SessionAfterKvAllocation, "session KV allocation");

    const uint64_t score_bytes = static_cast<uint64_t>(capacity) * impl_->config.prefill_chunk_size *
        attention_heads * sizeof(float);
    if (capacity != 0 && score_bytes / capacity != static_cast<uint64_t>(impl_->config.prefill_chunk_size) * attention_heads * sizeof(float))
        throw std::overflow_error("Qwen prefill scratch geometry overflow");
    const uint64_t graph_overhead = impl_->config.allow_experimental_capacity ?
        (impl_->config.prefill_chunk_size == 16 ? 48ULL : 40ULL) * 1024 * 1024 : 8ULL * 1024 * 1024;
    const uint64_t experimental_minimum = score_bytes * 3 + graph_overhead;
    if (experimental_minimum > SIZE_MAX) throw std::overflow_error("Qwen prefill scratch exceeds host size range");
    const size_t prefill_bytes = std::max(impl_->config.prefill_scratch_bytes,
        static_cast<size_t>(experimental_minimum));
    for (auto & local : state->devices) {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend(local.device_id));
        local.prefill_scratch = ggml_backend_buft_alloc_buffer(buft, prefill_bytes);
        if (local.prefill_scratch == nullptr)
            throw std::runtime_error("Qwen CUDA prefill scratch allocation failed on device " + std::to_string(local.device_id));
    }
    inject(fail_at, QwenCudaFailurePoint::SessionAfterPrefillScratch, "prefill scratch");
    for (auto & local : state->devices) {
        ggml_init_params graph_params{ 256 * 1024 * 1024, nullptr, true };
        local.graph_context = ggml_init(graph_params);
        if (local.graph_context == nullptr)
            throw std::runtime_error("Qwen CUDA session graph context allocation failed on device " + std::to_string(local.device_id));
    }

    if (device_ids.size() > 1) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device(device_ids.front()), &props);
        if (!props.caps.host_buffer)
            throw std::runtime_error("Qwen CUDA boundary requires a pinned host buffer supported by the source device");
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(device(device_ids.front()));
        if (host_buft == nullptr) throw std::runtime_error("Qwen CUDA pinned host buffer type is unavailable");
        state->boundary_host_bytes = static_cast<size_t>(impl_->config.prefill_chunk_size) * embedding_width * sizeof(float);
        state->boundary_host_buffer = ggml_backend_buft_alloc_buffer(host_buft, state->boundary_host_bytes);
        if (state->boundary_host_buffer == nullptr || !ggml_backend_buffer_is_host(state->boundary_host_buffer))
            throw std::runtime_error("Qwen CUDA pinned boundary host allocation failed");
        const char * boundary_type = ggml_backend_buft_name(
            ggml_backend_buffer_get_type(state->boundary_host_buffer));
        if (boundary_type == nullptr || std::string(boundary_type).find("CUDA") == std::string::npos ||
            std::string(boundary_type).find("Host") == std::string::npos)
            throw std::runtime_error("Qwen CUDA pinned boundary allocation fell back to pageable host memory");
    }
    return std::shared_ptr<QwenCudaSessionState>(new QwenCudaSessionState(std::move(state)));
}

// The CPU test fixture has no Qwen metadata, but uses the same device owner.
// Production metadata is retained by the admitted Qwen3Model.
ggml_backend_t QwenCudaRuntimeState::backend() const noexcept {
    return impl_->devices.empty() ? nullptr : impl_->devices.front()->backend;
}
ggml_backend_t QwenCudaRuntimeState::backend(uint32_t device_id) const { return impl_->by_id(device_id).backend; }
ggml_backend_dev_t QwenCudaRuntimeState::device() const noexcept {
    return impl_->devices.empty() ? nullptr : impl_->devices.front()->device;
}
ggml_backend_dev_t QwenCudaRuntimeState::device(uint32_t device_id) const { return impl_->by_id(device_id).device; }
uint32_t QwenCudaRuntimeState::device_count() const noexcept { return static_cast<uint32_t>(impl_->devices.size()); }
uint32_t QwenCudaRuntimeState::owner_device(const std::string & name) const {
    const auto found = impl_->tensor_owners.find(name);
    if (found == impl_->tensor_owners.end()) throw std::out_of_range("Qwen runtime has no owner for tensor: " + name);
    return found->second;
}
const QwenCudaPlacement & QwenCudaRuntimeState::placement() const noexcept { return impl_->config.placement; }
void QwenCudaRuntimeState::device_memory(size_t * free_bytes, size_t * total_bytes) const noexcept {
    if (impl_->devices.empty()) { if (free_bytes) *free_bytes = 0; if (total_bytes) *total_bytes = 0; return; }
    ggml_backend_dev_memory(device(), free_bytes, total_bytes);
}
void QwenCudaRuntimeState::device_memory(uint32_t id, size_t * free_bytes, size_t * total_bytes) const {
    ggml_backend_dev_memory(device(id), free_bytes, total_bytes);
}
ggml_context * QwenCudaRuntimeState::model_context() const noexcept {
    return impl_->devices.empty() ? nullptr : impl_->devices.front()->model_context;
}
ggml_context * QwenCudaRuntimeState::model_context(uint32_t id) const { return impl_->by_id(id).model_context; }
ggml_backend_buffer_t QwenCudaRuntimeState::model_allocation() const noexcept {
    return impl_->devices.empty() || !impl_->devices.front()->model_allocation ? nullptr :
        static_cast<ggml_backend_buffer_t>(impl_->devices.front()->model_allocation.get());
}
const std::string & QwenCudaRuntimeState::backend_name() const noexcept {
    static const std::string empty;
    return impl_->devices.empty() ? empty : impl_->devices.front()->backend_name;
}
const std::string & QwenCudaRuntimeState::artifact_identity() const noexcept {
    static const std::string fixture{"test-qwen"};
    return impl_->model != nullptr ? impl_->model->artifact_identity : fixture;
}
const ModelMetadataDescriptor * QwenCudaRuntimeState::metadata() const noexcept {
    return impl_->model != nullptr ? &impl_->model->metadata : nullptr;
}
ggml_tensor * QwenCudaRuntimeState::tensor(const std::string & name) const {
    const auto owner = impl_->tensor_owners.find(name);
    if (owner == impl_->tensor_owners.end()) return nullptr;
    const auto & tensors = impl_->by_id(owner->second).tensors;
    const auto found = tensors.find(name);
    return found == tensors.end() ? nullptr : found->second;
}
ggml_tensor * QwenCudaRuntimeState::tensor(const std::string & name, uint32_t id) const {
    const auto owner = impl_->tensor_owners.find(name);
    if (owner == impl_->tensor_owners.end() || owner->second != id) return nullptr;
    const auto & tensors = impl_->by_id(id).tensors;
    const auto found = tensors.find(name);
    return found == tensors.end() ? nullptr : found->second;
}
ggml_tensor * QwenCudaRuntimeState::embedding() const noexcept { return tensor(embedding_name); }
std::shared_ptr<TensorResidencyStore> QwenCudaRuntimeState::residency() const noexcept { return impl_->residency; }
uint64_t QwenCudaRuntimeState::resident_model_bytes() const noexcept {
    uint64_t total = 0; for (const auto & device : impl_->devices) total += device->resident_bytes; return total;
}
uint64_t QwenCudaRuntimeState::uploaded_payload_bytes() const noexcept {
    uint64_t total = 0; for (const auto & device : impl_->devices) total += device->uploaded_bytes; return total;
}
size_t QwenCudaRuntimeState::uploaded_tensor_count() const noexcept {
    size_t total = 0; for (const auto & device : impl_->devices) total += device->uploaded_tensors; return total;
}
size_t QwenCudaRuntimeState::resident_allocation_bytes() const noexcept {
    size_t total = 0; for (const auto & device : impl_->devices)
        if (device->model_allocation) total += ggml_backend_buffer_get_size(static_cast<ggml_backend_buffer_t>(device->model_allocation.get()));
    return total;
}
size_t QwenCudaRuntimeState::resident_tensor_count() const noexcept {
    return impl_->residency ? impl_->residency->device_resident_count() : 0;
}
uint64_t QwenCudaRuntimeState::uploaded_payload_bytes(uint32_t id) const { return impl_->by_id(id).uploaded_bytes; }
size_t QwenCudaRuntimeState::uploaded_tensor_count(uint32_t id) const { return impl_->by_id(id).uploaded_tensors; }
size_t QwenCudaRuntimeState::resident_allocation_bytes(uint32_t id) const {
    const auto & owner = impl_->by_id(id);
    return owner.model_allocation ? ggml_backend_buffer_get_size(static_cast<ggml_backend_buffer_t>(owner.model_allocation.get())) : 0;
}
size_t QwenCudaRuntimeState::resident_tensor_count(uint32_t id) const {
    size_t count = 0; for (const auto & entry : impl_->tensor_owners) if (entry.second == id) ++count; return count;
}
uint32_t QwenCudaRuntimeState::prefill_chunk_size() const noexcept { return impl_->config.prefill_chunk_size; }
bool QwenCudaRuntimeState::experimental_capacity_enabled() const noexcept { return impl_->config.allow_experimental_capacity; }
QwenExecutionPlanOptimizer & QwenCudaRuntimeState::execution_optimizer() noexcept { return impl_->execution_optimizer; }

QwenCudaSessionState::QwenCudaSessionState(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QwenCudaSessionState::~QwenCudaSessionState() = default;
void QwenCudaSessionState::reset() noexcept {
    for (const auto & device : impl_->devices)
        ggml_backend_synchronize(impl_->runtime->backend(device.device_id));
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
ggml_context * QwenCudaSessionState::context() const noexcept { return impl_->devices.empty() ? nullptr : impl_->devices.front().context; }
ggml_context * QwenCudaSessionState::context(uint32_t id) const { return impl_->by_id(id).context; }
ggml_context * QwenCudaSessionState::graph_context() const noexcept { return impl_->devices.empty() ? nullptr : impl_->devices.front().graph_context; }
ggml_context * QwenCudaSessionState::graph_context(uint32_t id) const { return impl_->by_id(id).graph_context; }
ggml_context * QwenCudaSessionState::create_auxiliary_context(size_t bytes) {
    return create_auxiliary_context(impl_->devices.front().device_id, bytes);
}
ggml_context * QwenCudaSessionState::create_auxiliary_context(uint32_t id, size_t bytes) {
    if (bytes == 0) throw std::invalid_argument("Qwen auxiliary graph arena must be nonzero");
    auto & local = impl_->by_id(id);
    ggml_init_params params{ bytes, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("Qwen auxiliary graph context allocation failed");
    try { local.auxiliary_contexts.push_back(context); }
    catch (...) { ggml_free(context); throw; }
    return context;
}
ggml_backend_buffer_t QwenCudaSessionState::allocation() const noexcept { return impl_->devices.empty() ? nullptr : impl_->devices.front().allocation; }
ggml_backend_buffer_t QwenCudaSessionState::allocation(uint32_t id) const { return impl_->by_id(id).allocation; }
ggml_backend_buffer_t QwenCudaSessionState::allocate_decode_scratch() {
    return allocate_decode_scratch(impl_->devices.front().device_id);
}
ggml_backend_buffer_t QwenCudaSessionState::allocate_decode_scratch(uint32_t id) {
    auto & local = impl_->by_id(id);
    if (local.decode_scratch != nullptr) return local.decode_scratch;
    ggml_backend_t backend = impl_->runtime->backend(id);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    const size_t required = ggml_backend_alloc_ctx_tensors_from_buft_size(local.graph_context, buft);
    const size_t bytes = std::max(required, impl_->runtime->impl_->config.decode_scratch_bytes);
    if (bytes == 0) throw std::runtime_error("Qwen CUDA decode graph has no scratch tensors");
    local.decode_scratch = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (local.decode_scratch == nullptr) throw std::runtime_error("Qwen CUDA decode scratch allocation failed");
    ggml_tallocr allocator = ggml_tallocr_new(local.decode_scratch);
    for (ggml_tensor * tensor = ggml_get_first_tensor(local.graph_context); tensor != nullptr;
        tensor = ggml_get_next_tensor(local.graph_context, tensor)) {
        if (tensor->buffer != nullptr) continue;
        ggml_status status = GGML_STATUS_SUCCESS;
        if (tensor->view_src != nullptr) status = ggml_backend_view_init(tensor);
        else if (tensor->data == nullptr) status = ggml_tallocr_alloc(&allocator, tensor);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("Qwen CUDA decode scratch binding failed at tensor ") + tensor->name);
    }
    inject(impl_->inject_failure, QwenCudaFailurePoint::SessionAfterDecodeScratch, "decode scratch");
    return local.decode_scratch;
}
size_t QwenCudaSessionState::allocation_bytes() const noexcept { return allocation_bytes(impl_->devices.front().device_id); }
size_t QwenCudaSessionState::allocation_bytes(uint32_t id) const {
    ggml_backend_buffer_t allocation = impl_->by_id(id).allocation;
    return allocation == nullptr ? 0 : ggml_backend_buffer_get_size(allocation);
}
size_t QwenCudaSessionState::prefill_scratch_bytes() const noexcept { return prefill_scratch_bytes(impl_->devices.front().device_id); }
size_t QwenCudaSessionState::prefill_scratch_bytes(uint32_t id) const {
    ggml_backend_buffer_t buffer = impl_->by_id(id).prefill_scratch;
    return buffer == nullptr ? 0 : ggml_backend_buffer_get_size(buffer);
}
size_t QwenCudaSessionState::decode_scratch_bytes() const noexcept { return decode_scratch_bytes(impl_->devices.front().device_id); }
size_t QwenCudaSessionState::decode_scratch_bytes(uint32_t id) const {
    ggml_backend_buffer_t buffer = impl_->by_id(id).decode_scratch;
    return buffer == nullptr ? 0 : ggml_backend_buffer_get_size(buffer);
}
ggml_backend_buffer_t QwenCudaSessionState::prefill_scratch() const noexcept { return prefill_scratch(impl_->devices.front().device_id); }
ggml_backend_buffer_t QwenCudaSessionState::prefill_scratch(uint32_t id) const { return impl_->by_id(id).prefill_scratch; }
ggml_backend_buffer_t QwenCudaSessionState::decode_scratch() const noexcept { return decode_scratch(impl_->devices.front().device_id); }
ggml_backend_buffer_t QwenCudaSessionState::decode_scratch(uint32_t id) const { return impl_->by_id(id).decode_scratch; }
ggml_tensor * QwenCudaSessionState::key_cache(uint32_t layer) const {
    if (layer >= qwen3_layers) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    return key_cache(layer, impl_->runtime->placement().block_device_ids[layer]);
}
ggml_tensor * QwenCudaSessionState::key_cache(uint32_t layer, uint32_t id) const {
    if (layer >= qwen3_layers) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    if (impl_->runtime->placement().block_device_ids[layer] != id) return nullptr;
    return impl_->by_id(id).keys[layer];
}
ggml_tensor * QwenCudaSessionState::value_cache(uint32_t layer) const {
    if (layer >= qwen3_layers) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    return value_cache(layer, impl_->runtime->placement().block_device_ids[layer]);
}
ggml_tensor * QwenCudaSessionState::value_cache(uint32_t layer, uint32_t id) const {
    if (layer >= qwen3_layers) throw std::out_of_range("Qwen CUDA KV layer is out of range");
    if (impl_->runtime->placement().block_device_ids[layer] != id) return nullptr;
    return impl_->by_id(id).values[layer];
}
ggml_tensor * QwenCudaSessionState::packed_value_scratch() const noexcept {
    return impl_->devices.empty() ? nullptr : impl_->devices.front().packed_value;
}
ggml_tensor * QwenCudaSessionState::packed_value_scratch(uint32_t id) const { return impl_->by_id(id).packed_value; }
void * QwenCudaSessionState::boundary_host_data() const noexcept {
    return impl_->boundary_host_buffer == nullptr ? nullptr : ggml_backend_buffer_get_base(impl_->boundary_host_buffer);
}
size_t QwenCudaSessionState::boundary_host_bytes() const noexcept { return impl_->boundary_host_bytes; }
bool QwenCudaSessionState::boundary_host_is_pinned() const noexcept {
    if (impl_->boundary_host_buffer == nullptr || !ggml_backend_buffer_is_host(impl_->boundary_host_buffer)) return false;
    const char * type = ggml_backend_buft_name(ggml_backend_buffer_get_type(impl_->boundary_host_buffer));
    return type != nullptr && std::string(type).find("CUDA") != std::string::npos &&
        std::string(type).find("Host") != std::string::npos;
}
void QwenCudaSessionState::set_boundary_audit_end_positions(std::vector<uint32_t> positions) {
    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    impl_->boundary_audit_end_positions = std::move(positions);
}
bool QwenCudaSessionState::should_audit_boundary(uint32_t end_position) const noexcept {
    return impl_->boundary_audit_end_positions.empty() ||
        std::binary_search(impl_->boundary_audit_end_positions.begin(), impl_->boundary_audit_end_positions.end(), end_position);
}
void QwenCudaSessionState::set_execution_failure(QwenCudaFailurePoint point) {
    if (point != QwenCudaFailurePoint::None && point < QwenCudaFailurePoint::ExecutionBeforeBoundary)
        throw std::invalid_argument("session execution failure point must be an execution-stage value");
    impl_->inject_failure = point;
}
void QwenCudaSessionState::inject_execution_failure(QwenCudaFailurePoint point, const char * label) const {
    inject(impl_->inject_failure, point, label);
}

} // namespace vbuf_ggml
