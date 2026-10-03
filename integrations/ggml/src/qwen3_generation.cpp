#include "qwen3_generation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace vbuf_ggml {
namespace {
constexpr uint32_t embed_width = 5120;
constexpr uint32_t heads = 40;
constexpr uint32_t kv_heads = 8;
constexpr uint32_t head_dim = 128;
constexpr uint32_t layers = 40;
constexpr uint32_t prefill_chunk = 32;
constexpr uint32_t executable_capacity_limit = 1032;

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point start, Clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

DeviceResidencyKey make_key(const Qwen3Model & model, const Qwen3Tensor & tensor,
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

void bind_context_to_buffer(ggml_context * context, ggml_backend_buffer_t buffer) {
    ggml_tallocr allocator = ggml_tallocr_new(buffer);
    for (ggml_tensor * tensor = ggml_get_first_tensor(context); tensor != nullptr;
        tensor = ggml_get_next_tensor(context, tensor)) {
        if (tensor->buffer != nullptr) continue;
        ggml_status status = GGML_STATUS_SUCCESS;
        if (tensor->view_src != nullptr) status = ggml_backend_view_init(tensor);
        else if (tensor->data == nullptr) status = ggml_tallocr_alloc(&allocator, tensor);
        if (status != GGML_STATUS_SUCCESS)
            throw std::runtime_error(std::string("Qwen graph scratch binding failed at tensor ") + tensor->name);
    }
}

} // namespace

struct Qwen3GenerationExecutor::Impl {
    struct LayerGraph {
        ggml_context * context = nullptr; // owned by QwenCudaSessionState
        ggml_cgraph * graph = nullptr;
        ggml_tensor * scores = nullptr;
        ggml_tensor * probabilities = nullptr;
        ggml_tensor * logits = nullptr;
        ggml_tensor * argmax = nullptr;
        size_t allocation_bytes = 0;
    };

    Qwen3Model * model;
    std::shared_ptr<QwenCudaRuntimeState> runtime;
    std::shared_ptr<QwenCudaSessionState> session;
    ggml_backend_t backend;
    uint32_t capacity;
    uint64_t graph_allocation_vram_bytes = 0;
    ggml_tensor * embedding;
    std::vector<Qwen3CudaLayerWeights> weights;

    ggml_tensor * decode_tokens = nullptr;
    ggml_tensor * decode_positions = nullptr;
    ggml_tensor * decode_mask = nullptr;
    ggml_tensor * decode_rows = nullptr;
    ggml_tensor * decode_logits = nullptr;
    ggml_tensor * decode_argmax = nullptr;
    ggml_cgraph * decode_graph = nullptr;

    ggml_tensor * prefill_tokens = nullptr;
    ggml_tensor * prefill_positions = nullptr;
    ggml_tensor * prefill_mask = nullptr;
    ggml_tensor * prefill_rows = nullptr;
    ggml_tensor * prefill_hidden_ping = nullptr;
    ggml_tensor * prefill_hidden_pong = nullptr;
    std::vector<LayerGraph> prefill_layers;
    ggml_tensor * prefill_logits = nullptr;
    ggml_tensor * prefill_argmax = nullptr;
    ggml_cgraph * prefill_first_graph = nullptr;

    explicit Impl(Qwen3Model & admitted, std::shared_ptr<QwenCudaRuntimeState> runtime_state,
        std::shared_ptr<QwenCudaSessionState> session_state)
        : model(&admitted), runtime(std::move(runtime_state)), session(std::move(session_state)),
          backend(runtime->backend()), capacity(session->capacity()), embedding(runtime->embedding()) {
        if (model->layer_count != layers || runtime->metadata() == nullptr || embedding == nullptr ||
            session->runtime().get() != runtime.get())
            throw std::invalid_argument("canonical Qwen executor requires matching admitted runtime/session state");
        if (capacity == 0 || capacity > executable_capacity_limit)
            throw std::invalid_argument("Qwen executable session capacity must be in 1..1032; larger KV allocations are not prefill-qualified");
        if (runtime->prefill_chunk_size() != prefill_chunk)
            throw std::invalid_argument("canonical Qwen execution requires 32-token prefill chunks");
        const uint64_t score_bytes = static_cast<uint64_t>(capacity) * prefill_chunk * heads * sizeof(float);
        const uint64_t conservative_prefill_bytes = score_bytes * 3 + 8 * 1024 * 1024;
        if (conservative_prefill_bytes > session->prefill_scratch_bytes())
            throw std::invalid_argument("Qwen prefill capacity exceeds the allocated reusable layer scratch before graph construction");
        build_graphs();
        size_t free_bytes = 0, total_bytes = 0;
        runtime->device_memory(&free_bytes, &total_bytes);
        graph_allocation_vram_bytes = total_bytes >= free_bytes ? total_bytes - free_bytes : 0;
    }

    ggml_tensor * model_tensor(const std::string & name) const {
        ggml_tensor * result = runtime->tensor(name);
        if (result == nullptr) throw std::runtime_error("Qwen runtime tensor is missing: " + name);
        return result;
    }

    Qwen3CudaLayerWeights layer_weights(uint32_t layer) const {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        return {
            model_tensor(prefix + "attn_norm.weight"), model_tensor(prefix + "attn_q.weight"),
            model_tensor(prefix + "attn_k.weight"), model_tensor(prefix + "attn_v.weight"),
            model_tensor(prefix + "attn_q_norm.weight"), model_tensor(prefix + "attn_k_norm.weight"),
            model_tensor(prefix + "attn_output.weight"), model_tensor(prefix + "ffn_norm.weight"),
            model_tensor(prefix + "ffn_gate.weight"), model_tensor(prefix + "ffn_up.weight"),
            model_tensor(prefix + "ffn_down.weight") };
    }

    void build_graphs() {
        ggml_context * graph_context = session->graph_context();
        const auto output_norm = model_tensor("output_norm.weight");
        const auto output_weight = model_tensor("output.weight");
        weights.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) weights.push_back(layer_weights(layer));

        decode_tokens = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, 1);
        decode_positions = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, 1);
        decode_mask = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, capacity, 1);
        decode_rows = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, kv_heads);
        ggml_tensor * hidden = ggml_get_rows(graph_context, embedding, decode_tokens);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const auto result = qwen3_cuda_build_layer(graph_context, weights[layer], hidden,
                decode_positions, decode_mask, decode_rows,
                session->key_cache(layer), session->value_cache(layer),
                session->packed_value_scratch(), 1, capacity);
            hidden = result.hidden;
        }
        const ggml_tensor * norm_weight = output_norm;
        const ggml_tensor * lm_head = output_weight;
        ggml_tensor * normalized = ggml_mul(graph_context,
            ggml_rms_norm(graph_context, hidden, 1e-6f), const_cast<ggml_tensor *>(norm_weight));
        decode_logits = ggml_mul_mat(graph_context, const_cast<ggml_tensor *>(lm_head), normalized);
        decode_argmax = ggml_argmax(graph_context, decode_logits);
        decode_graph = ggml_new_graph_custom(graph_context, 4096, false);
        if (decode_graph == nullptr) throw std::runtime_error("Qwen decode graph allocation failed");
        ggml_build_forward_expand(decode_graph, decode_argmax);

        prefill_tokens = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, prefill_chunk);
        prefill_positions = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, prefill_chunk);
        prefill_mask = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, capacity, prefill_chunk);
        prefill_rows = ggml_new_tensor_1d(graph_context, GGML_TYPE_I32, kv_heads * prefill_chunk);
        prefill_hidden_ping = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, embed_width, prefill_chunk);
        prefill_hidden_pong = ggml_new_tensor_2d(graph_context, GGML_TYPE_F32, embed_width, prefill_chunk);

        prefill_layers.reserve(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            LayerGraph item;
            item.context = session->create_auxiliary_context(2 * 1024 * 1024);
            ggml_tensor * layer_hidden = layer == 0 ?
                ggml_get_rows(item.context, embedding, prefill_tokens) :
                ((layer & 1U) != 0 ? prefill_hidden_ping : prefill_hidden_pong);
            ggml_tensor * hidden_output = (layer & 1U) == 0 ? prefill_hidden_ping : prefill_hidden_pong;
            const auto result = qwen3_cuda_build_layer(item.context, weights[layer], layer_hidden,
                prefill_positions, prefill_mask, prefill_rows,
                session->key_cache(layer), session->value_cache(layer),
                session->packed_value_scratch(), prefill_chunk, capacity);
            item.scores = result.scores;
            item.probabilities = result.probabilities;
            ggml_tensor * committed = ggml_cpy(item.context, result.hidden, hidden_output);
            ggml_tensor * graph_output = committed;
            if (layer + 1 == layers) {
                ggml_tensor * normalized_layer = ggml_mul(item.context,
                    ggml_rms_norm(item.context, committed, 1e-6f), const_cast<ggml_tensor *>(norm_weight));
                item.logits = ggml_mul_mat(item.context, const_cast<ggml_tensor *>(lm_head), normalized_layer);
                item.argmax = ggml_argmax(item.context, item.logits);
                graph_output = item.argmax;
                prefill_logits = item.logits;
                prefill_argmax = item.argmax;
            }
            item.graph = ggml_new_graph_custom(item.context, 1024, false);
            if (item.graph == nullptr) throw std::runtime_error("Qwen per-layer prefill graph allocation failed");
            ggml_build_forward_expand(item.graph, graph_output);
            if (layer == 0) prefill_first_graph = item.graph;
            prefill_layers.push_back(item);
        }

        session->allocate_decode_scratch();
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        size_t required = 0;
        for (LayerGraph & layer : prefill_layers) {
            layer.allocation_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(layer.context, buft);
            required = std::max(required, layer.allocation_bytes);
        }
        if (required == 0 || required > session->prefill_scratch_bytes())
            throw std::invalid_argument("Qwen executable prefill scratch is undersized: required=" +
                std::to_string(required) + " allocated=" + std::to_string(session->prefill_scratch_bytes()) +
                " capacity=" + std::to_string(capacity));
        ggml_backend_buffer_t prefill_scratch = session->prefill_scratch();
        for (LayerGraph & layer : prefill_layers) bind_context_to_buffer(layer.context, prefill_scratch);
    }

    void set_tensor(ggml_tensor * tensor, const void * data, size_t offset, size_t bytes,
        Qwen3GenerationExecution * result) {
        ggml_backend_tensor_set_async(backend, tensor, data, offset, bytes);
        ++result->h2d_calls;
        result->h2d_bytes += bytes;
    }

    void get_tensor(ggml_tensor * tensor, void * data, size_t offset, size_t bytes,
        Qwen3GenerationExecution * result) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, bytes);
        ++result->d2h_calls;
        result->d2h_bytes += bytes;
    }

    std::vector<DeviceResidencyKey> acquire_weights() {
        const auto residency = runtime->residency();
        std::vector<DeviceResidencyKey> keys;
        keys.reserve(model->tensors.size());
        try {
            for (const auto & item : model->tensors) {
                const Qwen3Tensor & tensor = item.second;
                DeviceResidencyKey key = make_key(*model, tensor, runtime->backend_name());
                auto lease = residency->acquire_device(key);
                if (!lease || lease->backend_handle != runtime->tensor(tensor.name))
                    throw std::runtime_error("Qwen runtime residency lease failed: " + tensor.name);
                keys.push_back(std::move(key));
            }
        } catch (...) {
            for (const auto & key : keys) residency->release_device(key);
            throw;
        }
        return keys;
    }

    void release_weights(const std::vector<DeviceResidencyKey> & keys) noexcept {
        const auto residency = runtime->residency();
        for (const auto & key : keys) residency->release_device(key);
    }

    std::vector<float> run_decode(uint32_t position, uint32_t token,
        Qwen3GenerationExecution * result) {
        if (position != session->current_length() || position >= capacity)
            throw std::logic_error("Qwen decode position disagrees with session logical length/capacity");
        std::vector<float> mask(capacity);
        for (uint32_t key = 0; key < capacity; ++key) mask[key] = key <= position ? 0.0f : -INFINITY;
        std::vector<int32_t> row_ids(kv_heads);
        for (uint32_t head = 0; head < kv_heads; ++head)
            row_ids[head] = static_cast<int32_t>(position * kv_heads + head);
        const int32_t token_id = static_cast<int32_t>(token);
        const int32_t pos = static_cast<int32_t>(position);
        set_tensor(decode_tokens, &token_id, 0, sizeof(token_id), result);
        set_tensor(decode_positions, &pos, 0, sizeof(pos), result);
        set_tensor(decode_mask, mask.data(), 0, mask.size() * sizeof(float), result);
        set_tensor(decode_rows, row_ids.data(), 0, row_ids.size() * sizeof(int32_t), result);
        if (ggml_backend_graph_compute_async(backend, decode_graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("canonical Qwen decode graph execution failed");
        ggml_backend_synchronize(backend);
        std::vector<float> logits(static_cast<size_t>(decode_logits->ne[0]));
        get_tensor(decode_logits, logits.data(), 0, logits.size() * sizeof(float), result);
        ggml_backend_synchronize(backend);
        session->commit_tokens(1);
        return logits;
    }

    std::vector<float> run_prefill_chunk(uint32_t start, const std::vector<uint32_t> & tokens,
        Qwen3GenerationExecution * result) {
        if (start != session->current_length() || start + prefill_chunk > tokens.size() ||
            start + prefill_chunk > capacity)
            throw std::logic_error("Qwen prefill chunk disagrees with session logical length/capacity");
        std::vector<int32_t> token_ids(prefill_chunk), positions(prefill_chunk);
        std::vector<int32_t> row_ids(static_cast<size_t>(prefill_chunk) * kv_heads);
        std::vector<float> mask(static_cast<size_t>(capacity) * prefill_chunk);
        for (uint32_t q = 0; q < prefill_chunk; ++q) {
            const uint32_t position = start + q;
            token_ids[q] = static_cast<int32_t>(tokens[position]);
            positions[q] = static_cast<int32_t>(position);
            for (uint32_t head = 0; head < kv_heads; ++head)
                row_ids[static_cast<size_t>(q) * kv_heads + head] =
                    static_cast<int32_t>(position * kv_heads + head);
            for (uint32_t key = 0; key < capacity; ++key)
                mask[static_cast<size_t>(q) * capacity + key] = key <= position ? 0.0f : -INFINITY;
        }
        set_tensor(prefill_tokens, token_ids.data(), 0, token_ids.size() * sizeof(int32_t), result);
        set_tensor(prefill_positions, positions.data(), 0, positions.size() * sizeof(int32_t), result);
        set_tensor(prefill_mask, mask.data(), 0, mask.size() * sizeof(float), result);
        set_tensor(prefill_rows, row_ids.data(), 0, row_ids.size() * sizeof(int32_t), result);
        for (const LayerGraph & layer : prefill_layers)
            if (ggml_backend_graph_compute_async(backend, layer.graph) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("canonical Qwen prefill layer graph execution failed");
        ggml_backend_synchronize(backend);
        const size_t offset = static_cast<size_t>(prefill_chunk - 1) * sizeof(int32_t);
        int32_t argmax = -1;
        get_tensor(prefill_argmax, &argmax, offset, sizeof(argmax), result);
        ggml_backend_synchronize(backend);
        if (argmax < 0 || static_cast<uint32_t>(argmax) >= static_cast<uint32_t>(prefill_logits->ne[0]))
            throw std::runtime_error("canonical Qwen prefill returned an invalid argmax");
        session->commit_tokens(prefill_chunk);
        std::vector<float> logits(static_cast<size_t>(prefill_logits->ne[0]));
        get_tensor(prefill_logits, logits.data(), static_cast<size_t>(prefill_chunk - 1) * logits.size() * sizeof(float),
            logits.size() * sizeof(float), result);
        ggml_backend_synchronize(backend);
        return logits;
    }
};

Qwen3GenerationExecutor::Qwen3GenerationExecutor(Qwen3Model & model,
    std::shared_ptr<QwenCudaRuntimeState> runtime, std::shared_ptr<QwenCudaSessionState> session)
    : impl_(std::make_unique<Impl>(model, std::move(runtime), std::move(session))) {}
Qwen3GenerationExecutor::~Qwen3GenerationExecutor() = default;

Qwen3GenerationExecution Qwen3GenerationExecutor::run(const std::vector<uint32_t> & prompt,
    uint32_t max_new_tokens, std::optional<uint32_t> stop_token,
    const std::function<bool(uint32_t, uint32_t)> & on_token,
    const std::function<bool()> & should_cancel) {
    if (prompt.empty() || max_new_tokens == 0 ||
        static_cast<uint64_t>(prompt.size()) + max_new_tokens > impl_->capacity)
        throw std::invalid_argument("Qwen request is empty or exceeds its configured session capacity");
    const auto embedding = impl_->runtime->embedding();
    if (embedding == nullptr || *std::max_element(prompt.begin(), prompt.end()) >=
        static_cast<uint32_t>(embedding->ne[1]))
        throw std::invalid_argument("Qwen prompt token is outside the admitted vocabulary");

    Qwen3GenerationExecution result;
    result.peak_vram_bytes = impl_->graph_allocation_vram_bytes;
    impl_->session->reset();
    const std::vector<DeviceResidencyKey> leases = impl_->acquire_weights();
    struct LeaseRelease {
        Impl * impl;
        const std::vector<DeviceResidencyKey> * keys;
        ~LeaseRelease() {
            if (impl->backend != nullptr) ggml_backend_synchronize(impl->backend);
            impl->release_weights(*keys);
        }
    } release{impl_.get(), &leases};
    try {
        std::vector<float> logits;
        const auto prefill_start = Clock::now();
        size_t prompt_position = 0;
        while (prompt_position < prompt.size()) {
            if (should_cancel && should_cancel()) { result.cancelled = true; break; }
            const size_t remaining = prompt.size() - prompt_position;
            if (remaining >= prefill_chunk) {
                logits = impl_->run_prefill_chunk(static_cast<uint32_t>(prompt_position), prompt, &result);
                prompt_position += prefill_chunk;
            } else {
                logits = impl_->run_decode(static_cast<uint32_t>(prompt_position), prompt[prompt_position], &result);
                ++prompt_position;
            }
            ++result.completed_positions;
        }
        const auto prefill_end = Clock::now();
        result.prefill_ns = elapsed_ns(prefill_start, prefill_end);
        if (result.cancelled) return result;

        uint32_t next_token = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        const auto decode_start = Clock::now();
        for (uint32_t generated = 0; generated < max_new_tokens; ++generated) {
            if (should_cancel && should_cancel()) { result.cancelled = true; break; }
            if (stop_token && next_token == *stop_token) { result.completed = true; break; }
            result.tokens.push_back(next_token);
            if (on_token && !on_token(next_token, impl_->session->current_length() - 1)) {
                result.cancelled = true;
                break;
            }
            logits = impl_->run_decode(impl_->session->current_length(), next_token, &result);
            ++result.completed_positions;
            if (generated + 1 == max_new_tokens) {
                result.completed = true;
                break;
            }
            next_token = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        result.decode_ns = elapsed_ns(decode_start, Clock::now());
        result.final_logits = std::move(logits);
        size_t free_bytes = 0, total_bytes = 0;
        impl_->runtime->device_memory(&free_bytes, &total_bytes);
        result.post_run_free_vram_bytes = free_bytes;
        result.peak_vram_bytes = std::max(result.peak_vram_bytes,
            total_bytes >= free_bytes ? total_bytes - free_bytes : 0);
        return result;
    } catch (...) {
        impl_->session->reset();
        throw;
    }
}

} // namespace vbuf_ggml
