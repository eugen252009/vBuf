#include "qwen3_generation.h"
#include "qwen3_execution_plan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
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

QwenOptimizerDecision safe_select_plan(std::optional<QwenExecutionPlan> & plan,
    Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
    const std::shared_ptr<QwenCudaSessionState> & session, QwenExecutionPlanPath path,
    uint32_t rows, uint32_t context_length, QwenExecutionPhase phase,
    bool request_native_attention_av = false) noexcept {
    auto & optimizer = runtime->execution_optimizer();
    QwenOptimizerDecision fallback;
    fallback.mode = optimizer.mode();
    fallback.canonical_selected = true;
    if (fallback.mode == QwenOptimizerMode::Disabled) {
        fallback.fallback = QwenOptimizerFallback::Disabled;
        return fallback;
    }
    try {
        if (!plan) plan = build_qwen_execution_plan(model, *runtime, *session, path);
        const auto facts = collect_qwen_runtime_facts(*plan, *runtime, rows, context_length, phase);
        const std::string candidate_identity = request_native_attention_av ?
            qwen3_native_attention_av_candidate_identity(*plan, phase) : std::string{};
        return optimizer.select(*plan, facts, candidate_identity);
    } catch (...) {
        optimizer.record_optimizer_failure();
        fallback.fallback = QwenOptimizerFallback::InternalFailure;
        return fallback;
    }
}

void safe_record_plan(std::optional<QwenExecutionPlan> & plan,
    Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
    const std::shared_ptr<QwenCudaSessionState> & session, QwenExecutionPlanPath path,
    uint32_t rows, uint32_t context_length, QwenExecutionPhase phase,
    uint64_t execution_count, uint64_t elapsed) noexcept {
    auto & optimizer = runtime->execution_optimizer();
    if (optimizer.mode() != QwenOptimizerMode::Shadow || execution_count == 0) return;
    try {
        if (!plan) plan = build_qwen_execution_plan(model, *runtime, *session, path);
        const auto facts = collect_qwen_runtime_facts(*plan, *runtime, rows, context_length, phase);
        optimizer.record_execution(*plan, facts, execution_count, elapsed);
    } catch (...) { optimizer.record_optimizer_failure(); }
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
    std::optional<QwenExecutionPlan> optimizer_plan;
    uint64_t graph_allocation_vram_bytes = 0;
    bool capture_final_hidden = false;
    ggml_tensor * embedding;
    std::vector<Qwen3CudaLayerWeights> weights;

    ggml_tensor * decode_tokens = nullptr;
    ggml_tensor * decode_positions = nullptr;
    ggml_tensor * decode_mask = nullptr;
    ggml_tensor * decode_rows = nullptr;
    ggml_tensor * decode_logits = nullptr;
    ggml_tensor * decode_final_hidden = nullptr;
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
    ggml_tensor * prefill_final_hidden = nullptr;
    ggml_tensor * prefill_argmax = nullptr;
    ggml_cgraph * prefill_first_graph = nullptr;

    explicit Impl(Qwen3Model & admitted, std::shared_ptr<QwenCudaRuntimeState> runtime_state,
        std::shared_ptr<QwenCudaSessionState> session_state, bool capture_hidden)
        : model(&admitted), runtime(std::move(runtime_state)), session(std::move(session_state)),
          backend(runtime->backend()), capacity(session->capacity()), capture_final_hidden(capture_hidden),
          embedding(runtime->embedding()) {
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
        decode_final_hidden = hidden;
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
                prefill_final_hidden = committed;
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
                const uint32_t owner = runtime->owner_device(tensor.name);
                DeviceResidencyKey key = make_key(*model, tensor, ggml_backend_name(runtime->backend(owner)));
                key.device_id = owner;
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

    std::vector<float> final_hidden_values;

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
        if (capture_final_hidden) {
            final_hidden_values.resize(embed_width);
            get_tensor(decode_final_hidden, final_hidden_values.data(), 0, embed_width * sizeof(float), result);
            ggml_backend_synchronize(backend);
        }
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
        if (capture_final_hidden) {
            final_hidden_values.resize(embed_width);
            get_tensor(prefill_final_hidden, final_hidden_values.data(),
                static_cast<size_t>(prefill_chunk - 1) * embed_width * sizeof(float),
                embed_width * sizeof(float), result);
            ggml_backend_synchronize(backend);
        }
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
    std::shared_ptr<QwenCudaRuntimeState> runtime, std::shared_ptr<QwenCudaSessionState> session,
    bool capture_final_hidden)
    : impl_(std::make_unique<Impl>(model, std::move(runtime), std::move(session), capture_final_hidden)) {}
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
    const QwenExecutionPhase request_phase = prompt.size() >= prefill_chunk ?
        QwenExecutionPhase::Prefill : QwenExecutionPhase::Decode;
    const uint32_t request_rows = request_phase == QwenExecutionPhase::Prefill ? prefill_chunk : 1;
    (void) safe_select_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
        QwenExecutionPlanPath::SingleGpu, request_rows, static_cast<uint32_t>(prompt.size()), request_phase);
    impl_->session->reset();
    impl_->final_hidden_values.clear();
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
        const uint64_t prefill_operations = prompt.size() / prefill_chunk + prompt.size() % prefill_chunk;
        safe_record_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
            QwenExecutionPlanPath::SingleGpu, request_rows, impl_->session->current_length(), request_phase,
            prefill_operations, result.prefill_ns);
        if (result.cancelled) return result;

        uint32_t next_token = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        (void) safe_select_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
            QwenExecutionPlanPath::SingleGpu, 1, impl_->session->current_length(), QwenExecutionPhase::Decode);
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
        safe_record_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
            QwenExecutionPlanPath::SingleGpu, 1, impl_->session->current_length(), QwenExecutionPhase::Decode,
            result.tokens.size(), result.decode_ns);
        result.final_logits = std::move(logits);
        result.final_hidden = impl_->final_hidden_values;
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

struct Qwen3MultiDeviceGenerationExecutor::Impl {
    struct LayerGraph {
        ggml_context * context = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * hidden_input = nullptr;
        ggml_tensor * hidden_output = nullptr;
        ggml_tensor * logits = nullptr;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
        ggml_tensor * diagnostic_value = nullptr;
        ggml_tensor * diagnostic_probabilities = nullptr;
        ggml_tensor * diagnostic_positions = nullptr;
        ggml_tensor * diagnostic_canonical_output = nullptr;
        ggml_tensor * diagnostic_native_output = nullptr;
        ggml_tensor * diagnostic_scores = nullptr;
        ggml_tensor * diagnostic_query = nullptr;
        ggml_tensor * diagnostic_key = nullptr;
#endif
    };
    struct DeviceGraphs {
        uint32_t id = 0;
        ggml_tensor * decode_tokens = nullptr;
        ggml_tensor * decode_positions = nullptr;
        ggml_tensor * decode_mask = nullptr;
        ggml_tensor * decode_rows = nullptr;
        ggml_tensor * decode_ping = nullptr;
        ggml_tensor * decode_pong = nullptr;
        ggml_tensor * prefill_tokens = nullptr;
        ggml_tensor * prefill_positions = nullptr;
        ggml_tensor * prefill_mask = nullptr;
        ggml_tensor * prefill_rows = nullptr;
        ggml_tensor * prefill_ping = nullptr;
        ggml_tensor * prefill_pong = nullptr;
        std::vector<LayerGraph> decode_layers;
        std::vector<LayerGraph> prefill_layers;
        std::vector<LayerGraph> native_decode_layers;
        std::vector<LayerGraph> native_prefill_layers;
    };
    struct PreboundDecodeDispatch {
        ggml_backend_t backend = nullptr;
        ggml_cgraph * graph = nullptr;
    };

    Qwen3Model * model;
    std::shared_ptr<QwenCudaRuntimeState> runtime;
    std::shared_ptr<QwenCudaSessionState> session;
    uint32_t capacity;
    uint32_t prefill_chunk;
    std::optional<QwenExecutionPlan> optimizer_plan;
    uint32_t early_device;
    uint32_t late_device;
    uint32_t cut;
    bool capture_decode_profile = false;
    bool enable_native_attention_av_candidate = false;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
    bool capture_attention_av_boundary = false;
    bool diagnostic_av_side_branch_built = false;
    bool capture_layer_activations = false;
    uint32_t diagnostic_capture_position = UINT32_MAX;
    int32_t diagnostic_native_av_layer = -1;
    bool diagnostic_allow_native_prefill = true;
    bool diagnostic_allow_native_decode = true;
#endif
    bool native_av_graphs_ready = false;
    std::vector<uint32_t> native_av_device_ids;
    bool prebound_decode_dispatch_attempted = false;
    bool prebound_decode_dispatch_ready = false;
    std::vector<PreboundDecodeDispatch> prebound_decode_dispatch;
    ggml_tensor * final_decode_hidden = nullptr;
    ggml_tensor * final_decode_logits = nullptr;
    ggml_tensor * native_final_decode_hidden = nullptr;
    ggml_tensor * native_final_decode_logits = nullptr;
    ggml_tensor * final_prefill_hidden = nullptr;
    ggml_tensor * final_prefill_logits = nullptr;
    ggml_tensor * native_final_prefill_hidden = nullptr;
    ggml_tensor * native_final_prefill_logits = nullptr;
    std::vector<DeviceGraphs> devices;

    Impl(Qwen3Model & admitted, std::shared_ptr<QwenCudaRuntimeState> runtime_state,
        std::shared_ptr<QwenCudaSessionState> session_state, bool profile_decode,
        bool enable_native_av_candidate
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
        , bool capture_av_boundary
#endif
        )
        : model(&admitted), runtime(std::move(runtime_state)), session(std::move(session_state)),
          capture_decode_profile(profile_decode), enable_native_attention_av_candidate(enable_native_av_candidate)
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
          , capture_attention_av_boundary(capture_av_boundary),
          diagnostic_av_side_branch_built(capture_av_boundary),
#else
          ,
#endif
          capacity(session->capacity()), prefill_chunk(runtime->prefill_chunk_size()),
          early_device(runtime->placement().block_device_ids.front()),
          late_device(runtime->placement().block_device_ids.back()) {
        if (!runtime || !session || model->layer_count != layers || runtime->metadata() == nullptr ||
            session->runtime().get() != runtime.get() || runtime->device_count() != 2 || early_device == late_device)
            throw std::invalid_argument("multi-device Qwen executor requires an admitted two-device placement/runtime/session");
        const auto & placement = runtime->placement();
        cut = 0;
        while (cut < layers && placement.block_device_ids[cut] == early_device) ++cut;
        if (cut == 0 || cut == layers || placement.embedding_device_id != early_device ||
            placement.output_norm_device_id != late_device || placement.output_head_device_id != late_device)
            throw std::invalid_argument("multi-device Qwen executor requires a contiguous early/late split and late final head");
        for (uint32_t i = 0; i < layers; ++i)
            if (placement.block_device_ids[i] != (i < cut ? early_device : late_device))
                throw std::invalid_argument("multi-device Qwen placement is not contiguous");
        if (capacity == 0 || capacity > (runtime->experimental_capacity_enabled() ? 32768U : executable_capacity_limit))
            throw std::invalid_argument("Qwen multi-device capacity exceeds explicit qualification enablement");
        if (prefill_chunk != 16 && prefill_chunk != 32)
            throw std::invalid_argument("Qwen multi-device execution supports 16- or 32-row prefill chunks");
        if (session->boundary_host_data() == nullptr || !session->boundary_host_is_pinned() ||
            session->boundary_host_bytes() < static_cast<size_t>(prefill_chunk) * embed_width * sizeof(float))
            throw std::invalid_argument("Qwen multi-device execution has no correctly sized pinned boundary buffer");
        build_graphs();
    }

    DeviceGraphs & device(uint32_t id) {
        for (auto & item : devices) if (item.id == id) return item;
        throw std::logic_error("Qwen graph device missing from placement");
    }
    ggml_tensor * model_tensor(const std::string & name, uint32_t id) const {
        ggml_tensor * tensor = runtime->tensor(name, id);
        if (tensor == nullptr) throw std::runtime_error("Qwen placement tensor is missing on its owner: " + name);
        return tensor;
    }
    Qwen3CudaLayerWeights layer_weights(uint32_t layer, uint32_t id) const {
        const std::string p = "blk." + std::to_string(layer) + ".";
        return { model_tensor(p + "attn_norm.weight", id), model_tensor(p + "attn_q.weight", id),
            model_tensor(p + "attn_k.weight", id), model_tensor(p + "attn_v.weight", id),
            model_tensor(p + "attn_q_norm.weight", id), model_tensor(p + "attn_k_norm.weight", id),
            model_tensor(p + "attn_output.weight", id), model_tensor(p + "ffn_norm.weight", id),
            model_tensor(p + "ffn_gate.weight", id), model_tensor(p + "ffn_up.weight", id),
            model_tensor(p + "ffn_down.weight", id) };
    }

    void build_graphs() {
        const std::vector<uint32_t> ids = runtime->placement().device_ids();
        devices.reserve(ids.size());
        for (uint32_t id : ids) {
            DeviceGraphs d;
            d.id = id;
            ggml_context * ctx = session->graph_context(id);
            d.decode_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            d.decode_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            d.decode_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, capacity, 1);
            d.decode_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv_heads);
            d.decode_ping = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_width, 1);
            d.decode_pong = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_width, 1);
            d.prefill_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefill_chunk);
            d.prefill_positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, prefill_chunk);
            d.prefill_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, capacity, prefill_chunk);
            d.prefill_rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kv_heads * prefill_chunk);
            d.prefill_ping = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_width, prefill_chunk);
            d.prefill_pong = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, embed_width, prefill_chunk);
            if (!d.decode_tokens || !d.decode_positions || !d.decode_mask || !d.decode_rows || !d.decode_ping || !d.decode_pong ||
                !d.prefill_tokens || !d.prefill_positions || !d.prefill_mask || !d.prefill_rows || !d.prefill_ping || !d.prefill_pong)
                throw std::runtime_error("Qwen multi-device control tensor allocation failed");
            devices.push_back(std::move(d));
        }

        for (uint32_t layer = 0; layer < layers; ++layer) {
            const uint32_t owner = runtime->placement().block_device_ids[layer];
            auto & d = device(owner);
            for (bool prefill : {false, true}) {
                LayerGraph item;
                item.context = session->create_auxiliary_context(owner, 2 * 1024 * 1024);
                ggml_context * control = session->graph_context(owner);
                const bool last = layer + 1 == layers;
                ggml_tensor * tokens = prefill ? d.prefill_tokens : d.decode_tokens;
                ggml_tensor * positions = prefill ? d.prefill_positions : d.decode_positions;
                ggml_tensor * mask = prefill ? d.prefill_mask : d.decode_mask;
                ggml_tensor * rows = prefill ? d.prefill_rows : d.decode_rows;
                ggml_tensor * ping = prefill ? d.prefill_ping : d.decode_ping;
                ggml_tensor * pong = prefill ? d.prefill_pong : d.decode_pong;
                if (layer == 0) item.hidden_input = ggml_get_rows(item.context,
                    model_tensor("token_embd.weight", owner), tokens);
                else item.hidden_input = ((layer - 1) % 2 == 0) ? ping : pong;
                item.hidden_output = (layer % 2 == 0) ? ping : pong;
                const uint32_t qcount = prefill ? prefill_chunk : 1;
                const auto graph = qwen3_cuda_build_layer(item.context, layer_weights(layer, owner),
                    item.hidden_input, positions, mask, rows, session->key_cache(layer),
                    session->value_cache(layer), session->packed_value_scratch(owner), qcount, capacity);
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                if (capture_attention_av_boundary && layer < cut) {
                    item.diagnostic_value = graph.value_cache;
                    item.diagnostic_probabilities = graph.probabilities;
                    item.diagnostic_positions = positions;
                    item.diagnostic_canonical_output = graph.attention_av_output;
                    item.diagnostic_native_output = ggml_attention_av(item.context, graph.value_cache,
                        graph.probabilities, positions);
                    if (layer == 0) {
                        item.diagnostic_scores = graph.scores;
                        item.diagnostic_query = graph.query;
                        item.diagnostic_key = graph.key_cache;
                    }
                }
#endif
                ggml_tensor * committed = ggml_cpy(item.context, graph.hidden, item.hidden_output);
                ggml_tensor * root = committed;
                if (last) {
                    ggml_tensor * output_norm = model_tensor("output_norm.weight", late_device);
                    ggml_tensor * output_weight = model_tensor("output.weight", late_device);
                    ggml_tensor * normalized = ggml_mul(item.context,
                        ggml_rms_norm(item.context, graph.hidden, 1e-6f), output_norm);
                    item.logits = ggml_mul_mat(item.context, output_weight, normalized);
                    root = item.logits;
                    if (prefill) { final_prefill_hidden = item.hidden_output; final_prefill_logits = item.logits; }
                    else { final_decode_hidden = item.hidden_output; final_decode_logits = item.logits; }
                }
                item.graph = ggml_new_graph_custom(item.context, 1024, false);
                if (item.graph == nullptr) throw std::runtime_error("Qwen per-layer graph allocation failed");
                ggml_build_forward_expand(item.graph, root);
                if (last) ggml_build_forward_expand(item.graph, committed);
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                if (item.diagnostic_native_output != nullptr)
                    ggml_build_forward_expand(item.graph, item.diagnostic_native_output);
#endif
                (prefill ? d.prefill_layers : d.decode_layers).push_back(item);
            }
        }

        for (auto & d : devices) {
            session->allocate_decode_scratch(d.id);
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(runtime->backend(d.id));
            size_t decode_required = 0, prefill_required = 0;
            for (const auto & layer : d.decode_layers)
                decode_required = std::max(decode_required, ggml_backend_alloc_ctx_tensors_from_buft_size(layer.context, buft));
            for (const auto & layer : d.prefill_layers)
                prefill_required = std::max(prefill_required, ggml_backend_alloc_ctx_tensors_from_buft_size(layer.context, buft));
            if (decode_required > session->prefill_scratch_bytes(d.id) || prefill_required > session->prefill_scratch_bytes(d.id))
                throw std::invalid_argument("Qwen device-local layer scratch is undersized: device=" + std::to_string(d.id) +
                    " decode_required=" + std::to_string(decode_required) +
                    " prefill_required=" + std::to_string(prefill_required) +
                    " available=" + std::to_string(session->prefill_scratch_bytes(d.id)));
            // Auxiliary layer graphs must not alias the main graph-context controls/hidden buffers.
            // Decode and prefill reuse this larger scratch sequentially.
            for (const auto & layer : d.decode_layers) bind_context_to_buffer(layer.context, session->prefill_scratch(d.id));
            for (const auto & layer : d.prefill_layers) bind_context_to_buffer(layer.context, session->prefill_scratch(d.id));
        }
        if (enable_native_attention_av_candidate) build_native_attention_av_graphs();
    }

    bool native_av_supported_on_device(const QwenExecutionPlan & plan, uint32_t id) const {
        if (id != early_device || id >= plan.stable_device_sm_versions.size() ||
            !plan.stable_device_sm_versions[id]) return false;
        for (uint32_t rows : {1U, prefill_chunk}) {
            Qwen3AttentionAVFacts facts;
            facts.sm_version = *plan.stable_device_sm_versions[id];
            facts.capacity = capacity;
            facts.visible_context = capacity;
            facts.query_rows = rows;
            facts.kv_heads = kv_heads;
            facts.query_heads = heads;
            facts.head_dimension = head_dim;
            facts.value_type = GGML_TYPE_F16;
            facts.probability_type = GGML_TYPE_F32;
            facts.position_type = GGML_TYPE_I32;
            facts.canonical_position_major_v = true;
            if (!resolve_qwen3_attention_av(facts, true).native_supported) return false;
        }
        return true;
    }

    void build_native_attention_av_graphs() noexcept {
        try {
            const QwenExecutionPlan plan = build_qwen_execution_plan(
                *model, *runtime, *session, QwenExecutionPlanPath::MultiGpu);
            for (auto & d : devices) {
                if (!native_av_supported_on_device(plan, d.id)) continue;
                native_av_device_ids.push_back(d.id);
                const uint32_t first_layer = d.id == early_device ? 0 : cut;
                for (bool prefill : {false, true}) {
                    const auto & canonical = prefill ? d.prefill_layers : d.decode_layers;
                    auto & native = prefill ? d.native_prefill_layers : d.native_decode_layers;
                    native.reserve(canonical.size());
                    const uint32_t query_count = prefill ? prefill_chunk : 1;
                    for (size_t local = 0; local < canonical.size(); ++local) {
                        const uint32_t layer = first_layer + static_cast<uint32_t>(local);
                        const bool last = layer + 1 == layers;
                        LayerGraph item;
                        item.context = session->create_auxiliary_context(d.id, 2 * 1024 * 1024);
                        item.hidden_input = canonical[local].hidden_input;
                        item.hidden_output = canonical[local].hidden_output;
                        const auto graph = qwen3_cuda_build_layer(item.context, layer_weights(layer, d.id),
                            item.hidden_input, prefill ? d.prefill_positions : d.decode_positions,
                            prefill ? d.prefill_mask : d.decode_mask,
                            prefill ? d.prefill_rows : d.decode_rows, session->key_cache(layer, d.id),
                            session->value_cache(layer, d.id), session->packed_value_scratch(d.id),
                            query_count, capacity, {}, Qwen3AttentionAVPath::NativeLayout);
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                        item.diagnostic_value = graph.value_cache;
                        item.diagnostic_probabilities = graph.probabilities;
                        item.diagnostic_positions = prefill ? d.prefill_positions : d.decode_positions;
                        item.diagnostic_native_output = graph.attention_av_output;
                        item.diagnostic_scores = graph.scores;
                        item.diagnostic_query = graph.query;
                        item.diagnostic_key = graph.key_cache;
#endif
                        ggml_tensor * committed = ggml_cpy(item.context, graph.hidden, item.hidden_output);
                        ggml_tensor * root = committed;
                        if (last) {
                            ggml_tensor * output_norm = model_tensor("output_norm.weight", late_device);
                            ggml_tensor * output_weight = model_tensor("output.weight", late_device);
                            ggml_tensor * normalized = ggml_mul(item.context,
                                ggml_rms_norm(item.context, graph.hidden, 1e-6f), output_norm);
                            item.logits = ggml_mul_mat(item.context, output_weight, normalized);
                            root = item.logits;
                            if (prefill) {
                                native_final_prefill_hidden = item.hidden_output;
                                native_final_prefill_logits = item.logits;
                            } else {
                                native_final_decode_hidden = item.hidden_output;
                                native_final_decode_logits = item.logits;
                            }
                        }
                        item.graph = ggml_new_graph_custom(item.context, 1024, false);
                        if (item.graph == nullptr)
                            throw std::runtime_error("native AV Qwen layer graph allocation failed");
                        ggml_build_forward_expand(item.graph, root);
                        if (last) ggml_build_forward_expand(item.graph, committed);
                        native.push_back(item);
                    }
                    if (native.size() != canonical.size())
                        throw std::runtime_error("native AV graph set has incomplete layer coverage");
                    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(runtime->backend(d.id));
                    size_t required = 0;
                    for (const auto & item : native)
                        required = std::max(required, ggml_backend_alloc_ctx_tensors_from_buft_size(item.context, buft));
                    if (required > session->prefill_scratch_bytes(d.id))
                        throw std::runtime_error("native AV layer graph exceeds reusable device scratch");
                    for (const auto & item : native)
                        bind_context_to_buffer(item.context, session->prefill_scratch(d.id));
                }
            }
            native_av_graphs_ready = !native_av_device_ids.empty();
        } catch (...) {
            for (auto & d : devices) {
                d.native_decode_layers.clear();
                d.native_prefill_layers.clear();
            }
            native_av_device_ids.clear();
            native_final_decode_hidden = nullptr;
            native_final_decode_logits = nullptr;
            native_final_prefill_hidden = nullptr;
            native_final_prefill_logits = nullptr;
            native_av_graphs_ready = false;
            runtime->execution_optimizer().record_optimizer_failure();
        }
    }

    void build_prebound_decode_dispatch() noexcept {
        if (prebound_decode_dispatch_attempted) return;
        prebound_decode_dispatch_attempted = true;
        try {
            std::vector<PreboundDecodeDispatch> candidate;
            candidate.reserve(layers);
            for (uint32_t layer = 0; layer < layers; ++layer) {
                const uint32_t owner = runtime->placement().block_device_ids[layer];
                auto & graphs = device(owner).decode_layers;
                auto & graph = graphs[layer < cut ? layer : layer - cut];
                ggml_backend_t backend = runtime->backend(owner);
                if (backend == nullptr || graph.graph == nullptr)
                    throw std::runtime_error("prebound Qwen decode dispatch has a null backend/graph");
                candidate.push_back({backend, graph.graph});
            }
            if (candidate.size() != layers)
                throw std::runtime_error("prebound Qwen decode dispatch has incomplete layer coverage");
            prebound_decode_dispatch = std::move(candidate);
            prebound_decode_dispatch_ready = true;
        } catch (...) {
            prebound_decode_dispatch.clear();
            prebound_decode_dispatch_ready = false;
            runtime->execution_optimizer().record_optimizer_failure();
        }
    }

    std::vector<DeviceResidencyKey> acquire_weights() {
        auto store = runtime->residency();
        std::vector<DeviceResidencyKey> keys;
        keys.reserve(model->tensors.size());
        try {
            for (const auto & entry : model->tensors) {
                const auto & tensor = entry.second;
                const uint32_t owner = runtime->owner_device(tensor.name);
                DeviceResidencyKey key = make_key(*model, tensor, ggml_backend_name(runtime->backend(owner)));
                key.device_id = owner;
                auto lease = store->acquire_device(key);
                if (!lease || lease->backend_handle != runtime->tensor(tensor.name, owner))
                    throw std::runtime_error("Qwen device residency lease failed: " + tensor.name);
                keys.push_back(std::move(key));
            }
        } catch (...) {
            for (const auto & key : keys) store->release_device(key);
            throw;
        }
        return keys;
    }
    void release_weights(const std::vector<DeviceResidencyKey> & keys) noexcept {
        for (const auto & key : keys) runtime->residency()->release_device(key);
    }
    void set(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t bytes,
        Qwen3GenerationExecution * result) {
        ggml_backend_tensor_set_async(backend, tensor, data, 0, bytes);
        ++result->h2d_calls; result->h2d_bytes += bytes;
    }
    void get(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t bytes,
        Qwen3GenerationExecution * result, size_t offset = 0) {
        ggml_backend_tensor_get_async(backend, tensor, data, offset, bytes);
        ++result->d2h_calls; result->d2h_bytes += bytes;
    }

    void transfer_hidden(uint32_t position, uint32_t rows, bool prefill, Qwen3GenerationExecution * result,
        bool profile = false) {
        const auto descriptor_start = profile ? Clock::now() : Clock::time_point{};
        auto & src_device = device(early_device);
        auto & dst_device = device(late_device);
        const auto & source_layers = prefill ? src_device.prefill_layers : src_device.decode_layers;
        const auto & destination_layers = prefill ? dst_device.prefill_layers : dst_device.decode_layers;
        ggml_tensor * src = source_layers.back().hidden_output;
        // The late-device vector begins with global block `cut`; its input is the destination buffer.
        ggml_tensor * dst = destination_layers.front().hidden_input;
        const size_t bytes = static_cast<size_t>(rows) * embed_width * sizeof(float);
        if (bytes > session->boundary_host_bytes()) throw std::out_of_range("Qwen boundary transfer exceeds pinned buffer");
        session->inject_execution_failure(QwenCudaFailurePoint::ExecutionBeforeBoundary, "before boundary transfer");
        void * pinned = session->boundary_host_data();
        ggml_backend_t src_backend = runtime->backend(early_device);
        ggml_backend_t dst_backend = runtime->backend(late_device);
        if (profile) result->decode_profile.boundary_descriptor_cpu_ns += elapsed_ns(descriptor_start, Clock::now());
        const auto source_start = profile ? Clock::now() : Clock::time_point{};
        get(src_backend, src, pinned, bytes, result);
        ggml_backend_synchronize(src_backend);
        if (profile) result->decode_profile.boundary_source_wait_d2h_ns += elapsed_ns(source_start, Clock::now());
        session->inject_execution_failure(QwenCudaFailurePoint::ExecutionAfterBoundaryD2H, "after boundary D2H");
        const auto destination_start = profile ? Clock::now() : Clock::time_point{};
        set(dst_backend, dst, pinned, bytes, result);
        ggml_backend_synchronize(dst_backend);
        if (profile) result->decode_profile.boundary_destination_h2d_wait_ns += elapsed_ns(destination_start, Clock::now());
        session->inject_execution_failure(QwenCudaFailurePoint::ExecutionAfterBoundaryH2D, "after boundary H2D");
        if (session->should_audit_boundary(position + rows)) {
            // Qualification audit reads back selected received activations; this is not a second handoff.
            std::vector<uint8_t> received(bytes);
            ggml_backend_tensor_get_async(dst_backend, dst, received.data(), 0, bytes);
            ggml_backend_synchronize(dst_backend);
            ++result->boundary_audits;
            result->boundary_audit_bytes += bytes;
            if (std::memcmp(pinned, received.data(), bytes) != 0) {
                result->boundary_bytes_equal = false;
                throw std::runtime_error("Qwen staged boundary bytes changed across H2D");
            }
        }
        ++result->boundary_handoffs;
        result->boundary_bytes += bytes;
    }

    std::vector<float> run_step(uint32_t position, const uint32_t * tokens, uint32_t rows,
        bool prefill, Qwen3GenerationExecution * result, bool use_prebound_dispatch = false,
        bool actual_decode_step = false, bool use_native_attention_av = false) {
        if (position != session->current_length() || position + rows > capacity ||
            rows == 0 || (prefill && rows != prefill_chunk) || (!prefill && rows != 1))
            throw std::logic_error("Qwen multi-device step disagrees with global logical position/capacity");
        const bool profile = capture_decode_profile && actual_decode_step && rows == 1;
        const auto step_start = profile ? Clock::now() : Clock::time_point{};
        std::vector<int32_t> token_ids(rows), positions(rows), row_ids(static_cast<size_t>(rows) * kv_heads);
        std::vector<float> mask(static_cast<size_t>(capacity) * rows);
        for (uint32_t q = 0; q < rows; ++q) {
            token_ids[q] = static_cast<int32_t>(tokens[q]);
            positions[q] = static_cast<int32_t>(position + q);
            for (uint32_t h = 0; h < kv_heads; ++h)
                row_ids[static_cast<size_t>(q) * kv_heads + h] = static_cast<int32_t>((position + q) * kv_heads + h);
            for (uint32_t k = 0; k < capacity; ++k)
                mask[static_cast<size_t>(q) * capacity + k] = k <= position + q ? 0.0f : -INFINITY;
        }
        if (profile) result->decode_profile.control_prepare_ns += elapsed_ns(step_start, Clock::now());
        const uint64_t h2d_bytes_before = result->h2d_bytes;
        const uint64_t h2d_calls_before = result->h2d_calls;
        const auto enqueue_start = profile ? Clock::now() : Clock::time_point{};
        for (auto & d : devices) {
            ggml_backend_t backend = runtime->backend(d.id);
            set(backend, prefill ? d.prefill_tokens : d.decode_tokens, token_ids.data(), token_ids.size() * sizeof(int32_t), result);
            set(backend, prefill ? d.prefill_positions : d.decode_positions, positions.data(), positions.size() * sizeof(int32_t), result);
            set(backend, prefill ? d.prefill_mask : d.decode_mask, mask.data(), mask.size() * sizeof(float), result);
            set(backend, prefill ? d.prefill_rows : d.decode_rows, row_ids.data(), row_ids.size() * sizeof(int32_t), result);
        }
        if (profile) {
            result->decode_profile.control_enqueue_cpu_ns += elapsed_ns(enqueue_start, Clock::now());
            result->decode_profile.control_h2d_bytes += result->h2d_bytes - h2d_bytes_before;
            result->decode_profile.control_h2d_calls += result->h2d_calls - h2d_calls_before;
        }
        if (use_prebound_dispatch && (!prebound_decode_dispatch_ready || prefill || rows != 1))
            throw std::logic_error("prebound Qwen decode dispatch selected outside its guarded geometry");
        if (use_native_attention_av && !native_av_graphs_ready)
            throw std::logic_error("native AV candidate selected without a complete guarded graph set");
        const auto execute_layer = [&](uint32_t layer, ggml_backend_t layer_backend, ggml_cgraph * layer_graph,
                const LayerGraph * diagnostic_graph) {
            if (layer == cut) session->inject_execution_failure(QwenCudaFailurePoint::ExecutionBeforeLateBlock,
                "before late-device execution");
            if (layer + 1 == layers) session->inject_execution_failure(QwenCudaFailurePoint::ExecutionBeforeFinalBlock,
                "before final transformer block");
            const auto submit_start = profile ? Clock::now() : Clock::time_point{};
            const ggml_status status = ggml_backend_graph_compute_async(layer_backend, layer_graph);
            if (profile) result->decode_profile.graph_submit_cpu_ns += elapsed_ns(submit_start, Clock::now());
            if (status != GGML_STATUS_SUCCESS)
                throw std::runtime_error("Qwen multi-device layer graph execution failed at block " + std::to_string(layer));
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            if (capture_attention_av_boundary && layer < cut && diagnostic_graph != nullptr &&
                (diagnostic_capture_position == UINT32_MAX || diagnostic_capture_position == position)) {
                ggml_backend_synchronize(layer_backend);
                const LayerGraph & diagnostic = *diagnostic_graph;
                if (diagnostic.diagnostic_value == nullptr || diagnostic.diagnostic_probabilities == nullptr ||
                    diagnostic.diagnostic_positions == nullptr || diagnostic.diagnostic_native_output == nullptr)
                    throw std::logic_error("requested AV boundary diagnostic was not built");
                const ggml_tensor * canonical_output = diagnostic.diagnostic_canonical_output;
                const ggml_tensor * output_metadata = canonical_output != nullptr ?
                    canonical_output : diagnostic.diagnostic_native_output;
                Qwen3AttentionAVBoundaryCapture capture;
                capture.layer = layer;
                capture.device_id = runtime->placement().block_device_ids[layer];
                capture.prefill = prefill;
                capture.native_intervention = canonical_output == nullptr && !use_native_attention_av;
                capture.capacity = capacity;
                capture.query_rows = rows;
                const auto save_metadata = [](const ggml_tensor * tensor, auto & ne, auto & nb) {
                    if (tensor == nullptr) throw std::logic_error("null AV boundary diagnostic tensor");
                    for (size_t i = 0; i < 4; ++i) {
                        ne[i] = tensor->ne[i];
                        nb[i] = tensor->nb[i];
                    }
                };
                capture.value_type = diagnostic.diagnostic_value->type;
                capture.probability_type = diagnostic.diagnostic_probabilities->type;
                capture.position_type = diagnostic.diagnostic_positions->type;
                save_metadata(diagnostic.diagnostic_value, capture.value_ne, capture.value_nb);
                save_metadata(diagnostic.diagnostic_probabilities, capture.probability_ne, capture.probability_nb);
                save_metadata(output_metadata, capture.output_ne, capture.output_nb);
                if (layer == 0) {
                    capture.score_type = diagnostic.diagnostic_scores->type;
                    capture.query_type = diagnostic.diagnostic_query->type;
                    capture.key_type = diagnostic.diagnostic_key->type;
                    save_metadata(diagnostic.diagnostic_scores, capture.score_ne, capture.score_nb);
                    save_metadata(diagnostic.diagnostic_query, capture.query_ne, capture.query_nb);
                    save_metadata(diagnostic.diagnostic_key, capture.key_ne, capture.key_nb);
                }
                capture.value_bytes.resize(ggml_nbytes(diagnostic.diagnostic_value));
                capture.probabilities.resize(ggml_nbytes(diagnostic.diagnostic_probabilities) / sizeof(float));
                capture.positions.resize(rows);
                if (canonical_output != nullptr)
                    capture.canonical_output.resize(ggml_nbytes(canonical_output) / sizeof(float));
                capture.native_output.resize(ggml_nbytes(diagnostic.diagnostic_native_output) / sizeof(float));
                if (layer == 0) {
                    capture.scores.resize(ggml_nbytes(diagnostic.diagnostic_scores) / sizeof(float));
                    capture.query.resize(ggml_nbytes(diagnostic.diagnostic_query) / sizeof(float));
                    capture.key_bytes.resize(ggml_nbytes(diagnostic.diagnostic_key));
                }
                ggml_backend_tensor_get(diagnostic.diagnostic_value, capture.value_bytes.data(), 0,
                    capture.value_bytes.size());
                ggml_backend_tensor_get(diagnostic.diagnostic_probabilities, capture.probabilities.data(), 0,
                    capture.probabilities.size() * sizeof(float));
                ggml_backend_tensor_get(diagnostic.diagnostic_positions, capture.positions.data(), 0,
                    capture.positions.size() * sizeof(int32_t));
                if (canonical_output != nullptr)
                    ggml_backend_tensor_get(canonical_output, capture.canonical_output.data(), 0,
                        capture.canonical_output.size() * sizeof(float));
                ggml_backend_tensor_get(diagnostic.diagnostic_native_output, capture.native_output.data(), 0,
                    capture.native_output.size() * sizeof(float));
                if (layer == 0) {
                    ggml_backend_tensor_get(diagnostic.diagnostic_scores, capture.scores.data(), 0,
                        capture.scores.size() * sizeof(float));
                    ggml_backend_tensor_get(diagnostic.diagnostic_query, capture.query.data(), 0,
                        capture.query.size() * sizeof(float));
                    ggml_backend_tensor_get(diagnostic.diagnostic_key, capture.key_bytes.data(), 0,
                        capture.key_bytes.size());
                }
                result->attention_av_boundary_captures.push_back(std::move(capture));
            }
#endif
            if (layer + 1 == cut) {
                const auto boundary_start = profile ? Clock::now() : Clock::time_point{};
                transfer_hidden(position, rows, prefill, result, profile);
                if (profile) result->decode_profile.boundary_wait_transfer_ns += elapsed_ns(boundary_start, Clock::now());
            }
        };
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
        const auto capture_layer_activation = [&](uint32_t layer, ggml_backend_t layer_backend,
                ggml_tensor * hidden_output) {
            if (!capture_layer_activations || diagnostic_capture_position != position) return;
            ggml_backend_synchronize(layer_backend);
            Qwen3LayerActivationCapture capture;
            capture.layer = layer;
            capture.device_id = runtime->placement().block_device_ids[layer];
            capture.position = position;
            capture.rows = rows;
            capture.prefill = prefill;
            capture.hidden.resize(static_cast<size_t>(rows) * embed_width);
            const size_t bytes = capture.hidden.size() * sizeof(float);
            ggml_backend_tensor_get(hidden_output, capture.hidden.data(), 0, bytes);
            result->layer_activation_captures.push_back(std::move(capture));
        };
#endif
        if (use_prebound_dispatch) {
            ++result->specialized_decode_steps;
            for (uint32_t layer = 0; layer < layers; ++layer) {
                const auto dispatch_start = profile ? Clock::now() : Clock::time_point{};
                const auto & binding = prebound_decode_dispatch[layer];
                if (profile) result->decode_profile.candidate_dispatch_cpu_ns += elapsed_ns(dispatch_start, Clock::now());
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                const uint32_t owner = runtime->placement().block_device_ids[layer];
                const size_t local_index = layer < cut ? layer : layer - cut;
                const auto & canonical_graphs = prefill ? device(owner).prefill_layers : device(owner).decode_layers;
                execute_layer(layer, binding.backend, binding.graph, &canonical_graphs[local_index]);
                capture_layer_activation(layer, binding.backend, canonical_graphs[local_index].hidden_output);
#else
                execute_layer(layer, binding.backend, binding.graph, nullptr);
#endif
            }
        } else {
            for (uint32_t layer = 0; layer < layers; ++layer) {
                const auto resolve_start = profile ? Clock::now() : Clock::time_point{};
                const uint32_t owner = runtime->placement().block_device_ids[layer];
                auto & d = device(owner);
                const size_t local_index = layer < cut ? layer : layer - cut;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                const bool diagnostic_intervention = diagnostic_native_av_layer == static_cast<int32_t>(layer) &&
                    !prefill && diagnostic_capture_position == position;
                if (diagnostic_intervention && !native_av_graphs_ready)
                    throw std::logic_error("diagnostic AV intervention lacks native layer graphs");
                const bool native_on_device = diagnostic_intervention || (use_native_attention_av &&
                    std::find(native_av_device_ids.begin(), native_av_device_ids.end(), owner) != native_av_device_ids.end());
                if (diagnostic_intervention) ++result->diagnostic_native_av_interventions;
#else
                const bool native_on_device = use_native_attention_av &&
                    std::find(native_av_device_ids.begin(), native_av_device_ids.end(), owner) != native_av_device_ids.end();
#endif
                auto & graphs = prefill ? d.prefill_layers : d.decode_layers;
                auto & native_graphs = prefill ? d.native_prefill_layers : d.native_decode_layers;
                auto & graph = native_on_device ? native_graphs[local_index] : graphs[local_index];
                ggml_backend_t layer_backend = runtime->backend(owner);
                if (profile) result->decode_profile.layer_resolve_cpu_ns += elapsed_ns(resolve_start, Clock::now());
                execute_layer(layer, layer_backend, graph.graph, &graph);
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
                capture_layer_activation(layer, layer_backend, graph.hidden_output);
#endif
            }
        }
        const uint32_t final_device = late_device;
        ggml_backend_t backend = runtime->backend(final_device);
        const auto final_start = profile ? Clock::now() : Clock::time_point{};
        ggml_backend_synchronize(backend);
        if (profile) result->decode_profile.final_device_wait_ns += elapsed_ns(final_start, Clock::now());
        const auto output_start = profile ? Clock::now() : Clock::time_point{};
        ggml_tensor * logits_tensor = prefill ?
            (use_native_attention_av && native_final_prefill_logits != nullptr ? native_final_prefill_logits : final_prefill_logits) :
            (use_native_attention_av && native_final_decode_logits != nullptr ? native_final_decode_logits : final_decode_logits);
        ggml_tensor * hidden_tensor = prefill ?
            (use_native_attention_av && native_final_prefill_hidden != nullptr ? native_final_prefill_hidden : final_prefill_hidden) :
            (use_native_attention_av && native_final_decode_hidden != nullptr ? native_final_decode_hidden : final_decode_hidden);
        std::vector<float> logits(static_cast<size_t>(logits_tensor->ne[0]));
        const size_t last_row = prefill ? rows - 1 : 0;
        const size_t logits_offset = last_row * logits.size() * sizeof(float);
        get(backend, logits_tensor, logits.data(), logits.size() * sizeof(float), result, logits_offset);
        ggml_backend_synchronize(backend);
        result->final_hidden.resize(embed_width);
        const size_t hidden_offset = prefill ? static_cast<size_t>(rows - 1) * embed_width * sizeof(float) : 0;
        get(backend, hidden_tensor, result->final_hidden.data(), embed_width * sizeof(float), result, hidden_offset);
        ggml_backend_synchronize(backend);
        session->commit_tokens(rows);
        if (use_native_attention_av) {
            ++result->native_av_steps;
            result->native_av_layers += cut;
            result->packed_v_copy_bytes_avoided += static_cast<uint64_t>(capacity) * kv_heads *
                head_dim * sizeof(ggml_fp16_t) * cut;
        }
        if (profile) {
            result->decode_profile.final_output_readback_ns += elapsed_ns(output_start, Clock::now());
            result->decode_profile.final_sync_readback_ns += elapsed_ns(final_start, Clock::now());
            const uint64_t step_wall = elapsed_ns(step_start, Clock::now());
            result->decode_profile.step_wall_ns += step_wall;
            result->decode_profile.step_wall_samples_ns.push_back(step_wall);
            ++result->decode_profile.steps;
        }
        return logits;
    }
};

Qwen3MultiDeviceGenerationExecutor::Qwen3MultiDeviceGenerationExecutor(Qwen3Model & model,
    std::shared_ptr<QwenCudaRuntimeState> runtime, std::shared_ptr<QwenCudaSessionState> session,
    bool capture_decode_profile, bool enable_native_attention_av_candidate
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
    , bool capture_attention_av_boundary
#endif
    )
    : impl_(std::make_unique<Impl>(model, std::move(runtime), std::move(session),
        capture_decode_profile, enable_native_attention_av_candidate
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
        , capture_attention_av_boundary
#endif
        )) {}
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
void Qwen3MultiDeviceGenerationExecutor::configure_attention_av_diagnostic(uint32_t capture_position,
        bool capture_local_av, int32_t native_intervention_layer, bool allow_native_prefill,
        bool allow_native_decode) {
    if (capture_position > 32 || capture_position >= impl_->capacity)
        throw std::invalid_argument("diagnostic AV capture position is outside the qualified context window");
    if (capture_local_av && !impl_->diagnostic_av_side_branch_built)
        throw std::invalid_argument("local AV capture was not enabled when diagnostic graphs were built");
    if (native_intervention_layer < -1 || native_intervention_layer >= static_cast<int32_t>(impl_->cut))
        throw std::invalid_argument("diagnostic native AV intervention must target an early-device block");
    if (native_intervention_layer >= 0) {
        if (impl_->capacity > 512)
            throw std::invalid_argument("diagnostic native AV intervention is bounded to capacity 512");
        if (!impl_->native_av_graphs_ready) impl_->build_native_attention_av_graphs();
        const uint32_t owner = impl_->runtime->placement().block_device_ids[
            static_cast<uint32_t>(native_intervention_layer)];
        const auto & device = impl_->device(owner);
        if (!impl_->native_av_graphs_ready || device.native_decode_layers.size() != impl_->cut ||
            device.native_prefill_layers.size() != impl_->cut)
            throw std::runtime_error("native AV diagnostic graph set failed to initialize");
    }
    impl_->diagnostic_capture_position = capture_position;
    impl_->capture_attention_av_boundary = capture_local_av;
    impl_->capture_layer_activations = true;
    impl_->diagnostic_native_av_layer = native_intervention_layer;
    impl_->diagnostic_allow_native_prefill = allow_native_prefill;
    impl_->diagnostic_allow_native_decode = allow_native_decode;
}
#endif
Qwen3MultiDeviceGenerationExecutor::~Qwen3MultiDeviceGenerationExecutor() = default;

Qwen3GenerationExecution Qwen3MultiDeviceGenerationExecutor::run(const std::vector<uint32_t> & prompt,
    uint32_t max_new_tokens, std::optional<uint32_t> stop_token,
    const std::function<bool(uint32_t, uint32_t)> & on_token,
    const std::function<bool()> & should_cancel,
    const std::function<void(uint32_t, const std::vector<float> &)> & on_progress) {
    if (prompt.empty() || max_new_tokens == 0 || static_cast<uint64_t>(prompt.size()) + max_new_tokens > impl_->capacity)
        throw std::invalid_argument("Qwen multi-device request exceeds session capacity");
    auto embedding = impl_->runtime->embedding();
    if (embedding == nullptr || *std::max_element(prompt.begin(), prompt.end()) >= static_cast<uint32_t>(embedding->ne[1]))
        throw std::invalid_argument("Qwen prompt token is outside admitted vocabulary");
    Qwen3GenerationExecution result;
    if (impl_->capture_decode_profile) {
        result.decode_profile.step_wall_samples_ns.reserve(static_cast<size_t>(max_new_tokens) + prompt.size());
        result.decode_profile.outer_token_wall_samples_ns.reserve(max_new_tokens);
    }
    const QwenExecutionPhase request_phase = prompt.size() >= impl_->prefill_chunk ?
        QwenExecutionPhase::Prefill : QwenExecutionPhase::Decode;
    const uint32_t request_rows = request_phase == QwenExecutionPhase::Prefill ? impl_->prefill_chunk : 1;
    const auto initial_select_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
    (void) safe_select_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
        QwenExecutionPlanPath::MultiGpu, request_rows, static_cast<uint32_t>(prompt.size()), request_phase,
        impl_->enable_native_attention_av_candidate);
    if (impl_->capture_decode_profile)
        result.decode_profile.plan_select_ns += elapsed_ns(initial_select_start, Clock::now());
    for (const auto & d : impl_->devices) {
        size_t free = 0, total = 0; impl_->runtime->device_memory(d.id, &free, &total);
        result.peak_vram_bytes += total >= free ? total - free : 0;
    }
    impl_->session->reset();
    const auto leases = impl_->acquire_weights();
    struct Release {
        Impl * impl; const std::vector<DeviceResidencyKey> * keys;
        ~Release() { for (uint32_t id : impl->runtime->placement().device_ids()) ggml_backend_synchronize(impl->runtime->backend(id)); impl->release_weights(*keys); }
    } release{impl_.get(), &leases};
    try {
        std::vector<float> logits;
        const auto prefill_start = Clock::now();
        size_t offset = 0;
        while (offset < prompt.size()) {
            if (should_cancel && should_cancel()) { result.cancelled = true; break; }
            const size_t remaining = prompt.size() - offset;
            const bool batch_prefill = remaining >= impl_->prefill_chunk;
            const uint32_t step_rows = batch_prefill ? impl_->prefill_chunk : 1;
            const QwenExecutionPhase step_phase = batch_prefill ?
                QwenExecutionPhase::Prefill : QwenExecutionPhase::Decode;
            QwenOptimizerDecision av_decision;
            bool use_native_av = false;
            bool allow_native_av_prefill = impl_->enable_native_attention_av_candidate;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            if (step_phase == QwenExecutionPhase::Prefill)
                allow_native_av_prefill = allow_native_av_prefill && impl_->diagnostic_allow_native_prefill;
            else
                allow_native_av_prefill = allow_native_av_prefill && impl_->diagnostic_allow_native_decode;
#endif
            if (allow_native_av_prefill) {
                av_decision = safe_select_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime,
                    impl_->session, QwenExecutionPlanPath::MultiGpu, step_rows,
                    static_cast<uint32_t>(offset), step_phase, true);
                use_native_av = av_decision.candidate_selected &&
                    av_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV;
                if (use_native_av && (!impl_->native_av_graphs_ready ||
                    impl_->runtime->execution_optimizer().consume_candidate_execution_fault_for_testing())) {
                    (void) impl_->runtime->execution_optimizer().invalidate_candidate(av_decision.candidate_identity);
                    use_native_av = false;
                }
            }
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            const uint64_t native_layers_before = result.native_av_layers;
            const uint64_t interventions_before = result.diagnostic_native_av_interventions;
#endif
            try {
                logits = impl_->run_step(static_cast<uint32_t>(offset), prompt.data() + offset,
                    step_rows, batch_prefill, &result, false, false, use_native_av);
            } catch (...) {
                if (use_native_av)
                    (void) impl_->runtime->execution_optimizer().invalidate_candidate(av_decision.candidate_identity);
                throw;
            }
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            Qwen3GenerationStepDiagnostic step_diagnostic;
            step_diagnostic.position = static_cast<uint32_t>(offset + step_rows - 1);
            step_diagnostic.rows = step_rows;
            step_diagnostic.phase = step_phase;
            step_diagnostic.native_candidate_requested = allow_native_av_prefill;
            step_diagnostic.native_candidate_selected = av_decision.candidate_selected &&
                av_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV;
            step_diagnostic.native_av_executed = use_native_av;
            step_diagnostic.native_av_layer_graphs = static_cast<uint32_t>(result.native_av_layers - native_layers_before);
            step_diagnostic.diagnostic_native_av_interventions = static_cast<uint32_t>(
                result.diagnostic_native_av_interventions - interventions_before);
            result.attention_av_step_diagnostics.push_back(step_diagnostic);
            Qwen3GenerationOutputCapture output_capture;
            output_capture.position = step_diagnostic.position;
            output_capture.rows = step_rows;
            output_capture.phase = step_phase;
            output_capture.input_token = prompt[offset + step_rows - 1];
            output_capture.hidden = result.final_hidden;
            output_capture.logits = logits;
            result.attention_av_output_captures.push_back(std::move(output_capture));
#endif
            offset += step_rows;
            ++result.completed_positions;
            if (on_progress) on_progress(impl_->session->current_length(), result.final_hidden);
        }
        result.prefill_ns = elapsed_ns(prefill_start, Clock::now());
        const uint64_t prefill_operations = prompt.size() / impl_->prefill_chunk + prompt.size() % impl_->prefill_chunk;
        const auto prefill_record_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
        safe_record_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
            QwenExecutionPlanPath::MultiGpu, request_rows, impl_->session->current_length(), request_phase,
            prefill_operations, result.prefill_ns);
        if (impl_->capture_decode_profile)
            result.decode_profile.optimizer_record_ns += elapsed_ns(prefill_record_start, Clock::now());
        if (result.cancelled) return result;
        uint32_t next = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        const auto decode_start = Clock::now();
        for (uint32_t generated = 0; generated < max_new_tokens; ++generated) {
            if (should_cancel && should_cancel()) { result.cancelled = true; break; }
            if (stop_token && next == *stop_token) { result.completed = true; break; }
            const auto outer_step_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
            const uint32_t decode_position = impl_->session->current_length();
            const auto decode_select_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
            bool allow_native_av_decode = impl_->enable_native_attention_av_candidate;
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            allow_native_av_decode = allow_native_av_decode && impl_->diagnostic_allow_native_decode;
#endif
            QwenOptimizerDecision decode_decision = safe_select_plan(impl_->optimizer_plan, *impl_->model,
                impl_->runtime, impl_->session, QwenExecutionPlanPath::MultiGpu, 1, decode_position,
                QwenExecutionPhase::Decode, allow_native_av_decode);
            if (impl_->capture_decode_profile)
                result.decode_profile.plan_select_ns += elapsed_ns(decode_select_start, Clock::now());
            bool use_prebound_dispatch = decode_decision.candidate_selected &&
                decode_decision.strategy == QwenCandidateStrategy::PreboundDecodeDispatch;
            bool use_native_av = decode_decision.candidate_selected &&
                decode_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV;
            if (use_native_av && (!impl_->native_av_graphs_ready ||
                impl_->runtime->execution_optimizer().consume_candidate_execution_fault_for_testing())) {
                (void) impl_->runtime->execution_optimizer().invalidate_candidate(decode_decision.candidate_identity);
                use_native_av = false;
            }
            if (use_prebound_dispatch && !impl_->prebound_decode_dispatch_attempted) {
                const auto setup_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
                impl_->build_prebound_decode_dispatch();
                if (impl_->capture_decode_profile)
                    result.decode_profile.candidate_setup_ns += elapsed_ns(setup_start, Clock::now());
            }
            if (use_prebound_dispatch && (!impl_->prebound_decode_dispatch_ready ||
                impl_->runtime->execution_optimizer().consume_candidate_execution_fault_for_testing())) {
                (void) impl_->runtime->execution_optimizer().invalidate_candidate(decode_decision.candidate_identity);
                decode_decision = safe_select_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime,
                    impl_->session, QwenExecutionPlanPath::MultiGpu, 1, decode_position, QwenExecutionPhase::Decode);
                use_prebound_dispatch = false;
            }
            result.tokens.push_back(next);
            if (on_token && !on_token(next, decode_position)) { result.cancelled = true; break; }
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            const uint64_t native_layers_before = result.native_av_layers;
            const uint64_t interventions_before = result.diagnostic_native_av_interventions;
#endif
            try {
                logits = impl_->run_step(decode_position, &next, 1, false, &result,
                    use_prebound_dispatch, true, use_native_av);
            } catch (...) {
                if (use_prebound_dispatch || use_native_av)
                    (void) impl_->runtime->execution_optimizer().invalidate_candidate(decode_decision.candidate_identity);
                throw;
            }
#ifdef VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC
            Qwen3GenerationStepDiagnostic step_diagnostic;
            step_diagnostic.position = decode_position;
            step_diagnostic.rows = 1;
            step_diagnostic.phase = QwenExecutionPhase::Decode;
            step_diagnostic.native_candidate_requested = allow_native_av_decode;
            step_diagnostic.native_candidate_selected = decode_decision.candidate_selected &&
                decode_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV;
            step_diagnostic.native_av_executed = use_native_av;
            step_diagnostic.native_av_layer_graphs = static_cast<uint32_t>(result.native_av_layers - native_layers_before);
            step_diagnostic.diagnostic_native_av_interventions = static_cast<uint32_t>(
                result.diagnostic_native_av_interventions - interventions_before);
            result.attention_av_step_diagnostics.push_back(step_diagnostic);
            Qwen3GenerationOutputCapture output_capture;
            output_capture.position = decode_position;
            output_capture.rows = 1;
            output_capture.phase = QwenExecutionPhase::Decode;
            output_capture.input_token = next;
            output_capture.hidden = result.final_hidden;
            output_capture.logits = logits;
            result.attention_av_output_captures.push_back(std::move(output_capture));
#endif
            if (!use_prebound_dispatch && !use_native_av) ++result.canonical_decode_steps;
            if (impl_->capture_decode_profile)
                result.decode_profile.outer_token_wall_samples_ns.push_back(elapsed_ns(outer_step_start, Clock::now()));
            ++result.completed_positions;
            if (on_progress) on_progress(impl_->session->current_length(), result.final_hidden);
            if (generated + 1 == max_new_tokens) { result.completed = true; break; }
            next = static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        result.decode_ns = elapsed_ns(decode_start, Clock::now());
        const auto decode_record_start = impl_->capture_decode_profile ? Clock::now() : Clock::time_point{};
        safe_record_plan(impl_->optimizer_plan, *impl_->model, impl_->runtime, impl_->session,
            QwenExecutionPlanPath::MultiGpu, 1, impl_->session->current_length(), QwenExecutionPhase::Decode,
            result.tokens.size(), result.decode_ns);
        if (impl_->capture_decode_profile)
            result.decode_profile.optimizer_record_ns += elapsed_ns(decode_record_start, Clock::now());
        result.final_logits = std::move(logits);
        for (const auto & d : impl_->devices) {
            size_t free = 0, total = 0; impl_->runtime->device_memory(d.id, &free, &total);
            result.post_run_free_vram_bytes += free;
        }
        return result;
    } catch (...) {
        impl_->session->reset();
        throw;
    }
}

} // namespace vbuf_ggml
