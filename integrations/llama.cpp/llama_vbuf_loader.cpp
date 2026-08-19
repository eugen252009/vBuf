#include "llama_vbuf_loader.h"

#include "gguf.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "vbuf_ml_adapter.h"
#include "vbuf_direct_source.h"
#include "vbuf_remote_source.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using step21a_clock = std::chrono::steady_clock;
static double elapsed_us(step21a_clock::time_point start, step21a_clock::time_point end) { return std::chrono::duration<double, std::micro>(end - start).count(); }
static bool trace_enabled() { static const bool enabled = std::getenv("VBUF_STEP22A_TRACE") != nullptr; return enabled; }

struct VbufRuntime {
    struct PayloadBinding {
        const uint8_t * data = nullptr;
        uint64_t bytes = 0;
        uint64_t source_id = 0;
        uint64_t source_offset = 0;
        std::shared_ptr<const void> lease;
    };
    VbufMlConsumerHandle * handle = nullptr;
    std::unordered_map<std::string, PayloadBinding> payloads;
    std::map<std::string, double> phases;
    uint64_t ffi_calls = 0;
    uint64_t ffi_bytes = 0;
    uint64_t tensor_callbacks = 0;
    double payload_attachment_us = 0;
    bool trace = trace_enabled();

    void phase(const char * name, step21a_clock::time_point start, step21a_clock::time_point end) { if (trace) phases[name] += elapsed_us(start, end); }

    explicit VbufRuntime(const char * path) {
        auto start = step21a_clock::now();
        handle = vbuf_ml_consumer_open(path); ++ffi_calls;
        auto open_end = step21a_clock::now();
        phase("rust_consumer_open", start, open_end);
        if (!handle) throw std::runtime_error("vBuf consumer validation failed");
        uint64_t count = 0;
        auto inventory_start = step21a_clock::now();
        ++ffi_calls;
        if (vbuf_ml_consumer_tensor_count(handle, &count) != 0) throw std::runtime_error("vBuf tensor inventory failed");
        for (uint64_t i = 0; i < count; ++i) {
            VbufMlTensorInfo info{}; char name[4096]{};
            ++ffi_calls;
            if (vbuf_ml_consumer_tensor_info(handle, i, &info, name, sizeof(name)) != 0) throw std::runtime_error("vBuf tensor descriptor failed");
            ffi_bytes += std::strlen(name);
            VbufMlTensorSourceInfo source_info{};
            if (vbuf_ml_consumer_tensor_source(handle, i, &source_info) != 0) throw std::runtime_error("vBuf tensor source descriptor failed");
            payloads.emplace(name, PayloadBinding { info.payload, info.payload_len, source_info.source_id, source_info.offset, {} });
        }
        phase("cpp_tensor_inventory", inventory_start, step21a_clock::now());
    }

    ~VbufRuntime() { vbuf_ml_consumer_close(handle); }
    VbufRuntime(const VbufRuntime &) = delete;
};

static std::mutex g_vbuf_models_mutex;
static std::string g_remote_error;
static std::unordered_map<llama_model *, VbufRuntime *> g_vbuf_models;
static std::unordered_map<llama_model *, std::shared_ptr<llama_model_source>> g_direct_models;

static void set_model_metadata(gguf_context * meta, VbufRuntime & source) {
    auto model_start = step21a_clock::now();
    VbufMlModelMetadataInfo info{}; ++source.ffi_calls;
    if (vbuf_ml_consumer_metadata(source.handle, &info) != 0) throw std::runtime_error("vBuf model metadata failed");
    gguf_set_val_str(meta, "general.architecture", "qwen3");
    gguf_set_val_u32(meta, "qwen3.context_length", static_cast<uint32_t>(info.context_length));
    gguf_set_val_u32(meta, "qwen3.embedding_length", static_cast<uint32_t>(info.embedding_length));
    gguf_set_val_u32(meta, "qwen3.block_count", static_cast<uint32_t>(info.layer_count));
    gguf_set_val_u32(meta, "qwen3.feed_forward_length", static_cast<uint32_t>(info.feed_forward_length));
    gguf_set_val_u32(meta, "qwen3.attention.head_count", static_cast<uint32_t>(info.head_count));
    gguf_set_val_u32(meta, "qwen3.attention.head_count_kv", static_cast<uint32_t>(info.kv_head_count));
    gguf_set_val_u32(meta, "qwen3.attention.key_length", static_cast<uint32_t>(info.key_head_dimension));
    gguf_set_val_u32(meta, "qwen3.attention.value_length", static_cast<uint32_t>(info.value_head_dimension));
    gguf_set_val_f32(meta, "qwen3.attention.layer_norm_rms_epsilon", static_cast<float>(info.normalization_epsilon));
    gguf_set_val_f32(meta, "qwen3.rope.freq_base", static_cast<float>(info.rope_theta));
    gguf_set_val_u32(meta, "qwen3.rope.dimension_count", info.rope_dimension);

    source.phase("model_metadata_projection", model_start, step21a_clock::now());
    auto vocab_start = step21a_clock::now();
    uint64_t token_count = 0; ++source.ffi_calls;
    if (vbuf_ml_consumer_token_count(source.handle, &token_count) != 0) throw std::runtime_error("vBuf vocabulary count failed");
    std::vector<std::string> token_storage;
    std::vector<const char *> tokens;
    std::vector<int32_t> token_types;
    std::vector<float> scores;
    token_storage.reserve(token_count); tokens.reserve(token_count); token_types.reserve(token_count); scores.reserve(token_count);
    for (uint64_t i = 0; i < token_count; ++i) {
        std::vector<char> text(4097);
        ++source.ffi_calls; if (vbuf_ml_consumer_token_text(source.handle, i, text.data(), text.size()) != 0) throw std::runtime_error("vBuf token text failed");
        source.ffi_bytes += std::strlen(text.data()) + 1;
        token_storage.emplace_back(text.data());
        tokens.push_back(token_storage.back().c_str());
        int32_t token_type = 1;
        ++source.ffi_calls; if (vbuf_ml_consumer_token_type(source.handle, i, &token_type) != 0) throw std::runtime_error("vBuf token type failed");
        token_types.push_back(token_type);
        float score = 0.0f;
        ++source.ffi_calls; if (vbuf_ml_consumer_token_score(source.handle, i, &score) == 0) scores.push_back(score); else scores.push_back(0.0f);
    }
    gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_data(meta, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_val_str(meta, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "qwen2");
    source.phase("vocabulary_projection", vocab_start, step21a_clock::now());

    auto merge_start = step21a_clock::now();
    uint64_t merge_count = 0; ++source.ffi_calls;
    if (vbuf_ml_consumer_merge_count(source.handle, &merge_count) != 0) throw std::runtime_error("vBuf merge count failed");
    std::vector<std::string> merge_storage;
    std::vector<const char *> merges;
    merge_storage.reserve(merge_count); merges.reserve(merge_count);
    for (uint64_t i = 0; i < merge_count; ++i) {
        uint64_t left = 0, right = 0;
        ++source.ffi_calls; if (vbuf_ml_consumer_merge_pair(source.handle, i, &left, &right) != 0 || left >= token_count || right >= token_count) throw std::runtime_error("vBuf merge pair failed");
        std::vector<char> l(4097), r(4097);
        source.ffi_calls += 2; if (vbuf_ml_consumer_token_text(source.handle, left, l.data(), l.size()) != 0 || vbuf_ml_consumer_token_text(source.handle, right, r.data(), r.size()) != 0) throw std::runtime_error("vBuf merge token lookup failed");
        source.ffi_bytes += std::strlen(l.data()) + std::strlen(r.data()) + 1;
        merge_storage.emplace_back(std::string(l.data()) + " " + r.data());
        merges.push_back(merge_storage.back().c_str());
    }
    gguf_set_arr_str(meta, "tokenizer.ggml.merges", merges.data(), merges.size());
    source.phase("merge_projection", merge_start, step21a_clock::now());
    uint64_t special = 0; source.ffi_calls++;
    if (vbuf_ml_consumer_special_token(source.handle, 0, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.bos_token_id", static_cast<uint32_t>(special));
    ++source.ffi_calls; if (vbuf_ml_consumer_special_token(source.handle, 1, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", static_cast<uint32_t>(special));
    ++source.ffi_calls; if (vbuf_ml_consumer_special_token(source.handle, 3, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.padding_token_id", static_cast<uint32_t>(special));
    bool add_bos = false;
    ++source.ffi_calls; if (vbuf_ml_consumer_add_bos(source.handle, &add_bos) == 0) gguf_set_val_bool(meta, "tokenizer.ggml.add_bos_token", add_bos);
    std::vector<char> chat_template(1024 * 1024);
    ++source.ffi_calls; if (vbuf_ml_consumer_chat_template(source.handle, chat_template.data(), chat_template.size()) == 0) { source.ffi_bytes += std::strlen(chat_template.data()) + 1; gguf_set_val_str(meta, "tokenizer.chat_template", chat_template.data()); }
}

static ggml_type tensor_type(uint8_t representation) {
    switch (representation) { case 0: return GGML_TYPE_F32; case 1: return GGML_TYPE_BF16; case 2: return GGML_TYPE_Q8_0; case 3: return GGML_TYPE_Q4_0; case 4: return GGML_TYPE_Q2_K; case 5: return GGML_TYPE_IQ1_S; case 6: return GGML_TYPE_Q4_K; case 7: return GGML_TYPE_IQ4_NL; case 8: return GGML_TYPE_IQ4_XS; case 9: return GGML_TYPE_Q3_K; case 10: return GGML_TYPE_IQ2_XXS; case 11: return GGML_TYPE_IQ2_XS; case 12: return GGML_TYPE_IQ2_S; case 13: return GGML_TYPE_Q5_K; default: throw std::runtime_error("unknown vBuf representation"); }
}

static ggml_context_ptr add_tensor_metadata(gguf_context * meta, VbufRuntime & source) {
    auto start = step21a_clock::now();
    uint64_t count = 0;
    if (vbuf_ml_consumer_tensor_count(source.handle, &count) != 0) throw std::runtime_error("vBuf tensor count failed");
    ggml_init_params params{ ggml_tensor_overhead() * (count + 1), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("GGML metadata context allocation failed");
    ggml_context_ptr owner(ctx);
    for (uint64_t i = 0; i < count; ++i) {
        VbufMlTensorInfo info{}; char name[4096]{};
        ++source.ffi_calls;
        if (vbuf_ml_consumer_tensor_info(source.handle, i, &info, name, sizeof(name)) != 0 || info.rank > GGML_MAX_DIMS) throw std::runtime_error("vBuf tensor metadata failed");
        std::vector<int64_t> dims(info.rank);
        for (uint8_t d = 0; d < info.rank; ++d) dims[d] = static_cast<int64_t>(info.dimensions[d]);
        ggml_tensor * tensor = ggml_new_tensor(ctx, tensor_type(info.representation), info.rank, dims.data());
        if (!tensor) throw std::runtime_error("GGML tensor metadata allocation failed");
        ggml_set_name(tensor, name);
        gguf_add_tensor(meta, tensor);
    }
    source.phase("tensor_metadata_projection", start, step21a_clock::now());
    return owner;
}

static void set_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & source = *static_cast<VbufRuntime *>(userdata);
    auto callback_start = step21a_clock::now();
    ++source.tensor_callbacks;
    auto it = source.payloads.find(ggml_get_name(tensor));
    if (it == source.payloads.end()) throw std::runtime_error(std::string("vBuf payload is missing: ") + ggml_get_name(tensor));
    if (it->second.bytes != ggml_nbytes(tensor)) throw std::runtime_error(std::string("vBuf payload does not match GGML tensor: ") + ggml_get_name(tensor) + " expected=" + std::to_string(ggml_nbytes(tensor)) + " actual=" + std::to_string(it->second.bytes));
    tensor->data = const_cast<uint8_t *>(it->second.data);
    source.payload_attachment_us += elapsed_us(callback_start, step21a_clock::now());
}

} // namespace

extern "C" llama_model * llama_model_load_vbuf(const char * path, llama_model_params params) {
    try {
        auto source = std::make_unique<VbufRuntime>(path);
        auto metadata_start = step21a_clock::now();
        gguf_context_ptr meta(gguf_init_empty());
        if (!meta) throw std::runtime_error("GGUF metadata context allocation failed");
        set_model_metadata(meta.get(), *source);
        source->phase("gguf_compatible_metadata_synthesis", metadata_start, step21a_clock::now());
        auto tensor_context = add_tensor_metadata(meta.get(), *source);
        auto init_start = step21a_clock::now();
        llama_model * model = llama_model_init_from_user(meta.get(), set_tensor_data, source.get(), params);
        source->phase("llama_model_init_from_user", init_start, step21a_clock::now());
        if (!model) throw std::runtime_error("llama model construction failed");
        if (source->trace) {
            for (const auto & [name, duration] : source->phases) std::fprintf(stderr, "STEP22A phase=%s us=%.3f\n", name.c_str(), duration);
            std::fprintf(stderr, "STEP22A ffi_calls=%llu ffi_string_bytes=%llu tensor_callbacks=%llu payload_attachment_us=%.3f\n", (unsigned long long) source->ffi_calls, (unsigned long long) source->ffi_bytes, (unsigned long long) source->tensor_callbacks, source->payload_attachment_us);
        }
        {
            std::lock_guard lock(g_vbuf_models_mutex);
            g_vbuf_models.emplace(model, source.release());
        }
        return model;
    } catch (...) { return nullptr; }
}

extern "C" llama_model * llama_model_load_vbuf_direct(const char * path, llama_model_params params) {
    try {
        auto source = vbuf_llama::make_vbuf_direct_source(path);
        llama_model * model = llama_model_init_from_source(source, vbuf_llama::set_vbuf_direct_tensor_data, source.get(), params);
        if (!model) return nullptr;
        std::lock_guard lock(g_vbuf_models_mutex);
        g_direct_models.emplace(model, std::move(source));
        return model;
    } catch (...) { return nullptr; }
}

extern "C" llama_model * llama_model_load_vbuf_remote(const char * bootstrap_path, const char * endpoint, llama_model_params params) {
    g_remote_error.clear();
    try {
        auto source = vbuf_llama::make_vbuf_remote_source(bootstrap_path, endpoint);
        llama_model * model = llama_model_init_from_source(source, vbuf_llama::set_vbuf_remote_tensor_data, source.get(), params);
        if (!model) return nullptr;
        std::lock_guard lock(g_vbuf_models_mutex);
        g_direct_models.emplace(model, std::move(source));
        return model;
    } catch (const std::exception & error) {
        g_remote_error = error.what();
        std::fprintf(stderr, "REMOTE_LOAD_FAIL %s\n", error.what());
        return nullptr;
    } catch (...) {
        g_remote_error = "unknown";
        std::fprintf(stderr, "REMOTE_LOAD_FAIL unknown\n");
        return nullptr;
    }
}

extern "C" const char * llama_model_last_remote_error() { return g_remote_error.c_str(); }

extern "C" bool llama_model_probe_vbuf_remote(const char * bootstrap_path, const char * endpoint, vbuf_llama::VbufRemoteMetrics * metrics) {
    return metrics != nullptr && vbuf_llama::probe_vbuf_remote_source(bootstrap_path, endpoint, *metrics);
}

extern "C" bool llama_model_vbuf_remote_metrics(llama_model * model, vbuf_llama::VbufRemoteMetrics * metrics) {
    if (!metrics) return false;
    std::lock_guard lock(g_vbuf_models_mutex);
    const auto it = g_direct_models.find(model);
    if (it == g_direct_models.end()) return false;
    const auto remote = std::dynamic_pointer_cast<vbuf_llama::VbufRemoteSource>(it->second);
    return remote != nullptr && remote->metrics(*metrics);
}

extern "C" bool llama_model_step28_prepare_vbuf(llama_model * model, const char * variant, vbuf_llama::Step28PreparationResult * result) {
    if (!result || !variant) return false;
    std::lock_guard lock(g_vbuf_models_mutex);
    auto it = g_direct_models.find(model);
    return it != g_direct_models.end() && vbuf_llama::step28_prepare_vbuf_source(it->second.get(), variant, *result);
}

extern "C" void llama_model_free_vbuf_direct(llama_model * model) {
    std::shared_ptr<llama_model_source> source;
    {
        std::lock_guard lock(g_vbuf_models_mutex);
        auto it = g_direct_models.find(model);
        if (it != g_direct_models.end()) { source = std::move(it->second); g_direct_models.erase(it); }
    }
    llama_model_free(model);
}

extern "C" void llama_model_free_vbuf(llama_model * model) {
    VbufRuntime * source = nullptr;
    {
        std::lock_guard lock(g_vbuf_models_mutex);
        auto it = g_vbuf_models.find(model);
        if (it != g_vbuf_models.end()) { source = it->second; g_vbuf_models.erase(it); }
    }
    llama_model_free(model);
    delete source;
}
