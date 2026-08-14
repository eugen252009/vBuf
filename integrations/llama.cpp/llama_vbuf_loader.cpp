#include "llama_vbuf_loader.h"

#include "gguf.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "vbuf_ml_adapter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct VbufRuntime {
    VbufMlConsumerHandle * handle = nullptr;
    std::unordered_map<std::string, const uint8_t *> payloads;
    std::unordered_map<std::string, size_t> sizes;

    explicit VbufRuntime(const char * path) : handle(vbuf_ml_consumer_open(path)) {
        if (!handle) throw std::runtime_error("vBuf consumer validation failed");
        uint64_t count = 0;
        if (vbuf_ml_consumer_tensor_count(handle, &count) != 0) throw std::runtime_error("vBuf tensor inventory failed");
        for (uint64_t i = 0; i < count; ++i) {
            VbufMlTensorInfo info{};
            char name[4096]{};
            if (vbuf_ml_consumer_tensor_info(handle, i, &info, name, sizeof(name)) != 0) throw std::runtime_error("vBuf tensor descriptor failed");
            payloads.emplace(name, info.payload);
            sizes.emplace(name, static_cast<size_t>(info.payload_len));
        }
    }

    ~VbufRuntime() { vbuf_ml_consumer_close(handle); }
    VbufRuntime(const VbufRuntime &) = delete;
};

static std::mutex g_vbuf_models_mutex;
static std::unordered_map<llama_model *, VbufRuntime *> g_vbuf_models;

static void set_model_metadata(gguf_context * meta, VbufRuntime & source) {
    VbufMlModelMetadataInfo info{};
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

    uint64_t token_count = 0;
    if (vbuf_ml_consumer_token_count(source.handle, &token_count) != 0) throw std::runtime_error("vBuf vocabulary count failed");
    std::vector<std::string> token_storage;
    std::vector<const char *> tokens;
    std::vector<int32_t> token_types;
    std::vector<float> scores;
    token_storage.reserve(token_count); tokens.reserve(token_count); token_types.reserve(token_count); scores.reserve(token_count);
    for (uint64_t i = 0; i < token_count; ++i) {
        std::vector<char> text(4097);
        if (vbuf_ml_consumer_token_text(source.handle, i, text.data(), text.size()) != 0) throw std::runtime_error("vBuf token text failed");
        token_storage.emplace_back(text.data());
        tokens.push_back(token_storage.back().c_str());
        int32_t token_type = 1;
        if (vbuf_ml_consumer_token_type(source.handle, i, &token_type) != 0) throw std::runtime_error("vBuf token type failed");
        token_types.push_back(token_type);
        float score = 0.0f;
        if (vbuf_ml_consumer_token_score(source.handle, i, &score) == 0) scores.push_back(score); else scores.push_back(0.0f);
    }
    gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), token_types.size());
    gguf_set_arr_data(meta, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_val_str(meta, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "qwen2");

    uint64_t merge_count = 0;
    if (vbuf_ml_consumer_merge_count(source.handle, &merge_count) != 0) throw std::runtime_error("vBuf merge count failed");
    std::vector<std::string> merge_storage;
    std::vector<const char *> merges;
    merge_storage.reserve(merge_count); merges.reserve(merge_count);
    for (uint64_t i = 0; i < merge_count; ++i) {
        uint64_t left = 0, right = 0;
        if (vbuf_ml_consumer_merge_pair(source.handle, i, &left, &right) != 0 || left >= token_count || right >= token_count) throw std::runtime_error("vBuf merge pair failed");
        std::vector<char> l(4097), r(4097);
        if (vbuf_ml_consumer_token_text(source.handle, left, l.data(), l.size()) != 0 || vbuf_ml_consumer_token_text(source.handle, right, r.data(), r.size()) != 0) throw std::runtime_error("vBuf merge token lookup failed");
        merge_storage.emplace_back(std::string(l.data()) + " " + r.data());
        merges.push_back(merge_storage.back().c_str());
    }
    gguf_set_arr_str(meta, "tokenizer.ggml.merges", merges.data(), merges.size());
    uint64_t special = 0;
    if (vbuf_ml_consumer_special_token(source.handle, 0, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.bos_token_id", static_cast<uint32_t>(special));
    if (vbuf_ml_consumer_special_token(source.handle, 1, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", static_cast<uint32_t>(special));
    if (vbuf_ml_consumer_special_token(source.handle, 3, &special) == 0) gguf_set_val_u32(meta, "tokenizer.ggml.padding_token_id", static_cast<uint32_t>(special));
    bool add_bos = false;
    if (vbuf_ml_consumer_add_bos(source.handle, &add_bos) == 0) gguf_set_val_bool(meta, "tokenizer.ggml.add_bos_token", add_bos);
    std::vector<char> chat_template(1024 * 1024);
    if (vbuf_ml_consumer_chat_template(source.handle, chat_template.data(), chat_template.size()) == 0) gguf_set_val_str(meta, "tokenizer.chat_template", chat_template.data());
}

static ggml_type tensor_type(uint8_t representation) {
    switch (representation) { case 0: return GGML_TYPE_F32; case 1: return GGML_TYPE_BF16; case 2: return GGML_TYPE_Q8_0; default: throw std::runtime_error("unknown vBuf representation"); }
}

static ggml_context_ptr add_tensor_metadata(gguf_context * meta, VbufRuntime & source) {
    uint64_t count = 0;
    if (vbuf_ml_consumer_tensor_count(source.handle, &count) != 0) throw std::runtime_error("vBuf tensor count failed");
    ggml_init_params params{ ggml_tensor_overhead() * (count + 1), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("GGML metadata context allocation failed");
    ggml_context_ptr owner(ctx);
    for (uint64_t i = 0; i < count; ++i) {
        VbufMlTensorInfo info{}; char name[4096]{};
        if (vbuf_ml_consumer_tensor_info(source.handle, i, &info, name, sizeof(name)) != 0 || info.rank > GGML_MAX_DIMS) throw std::runtime_error("vBuf tensor metadata failed");
        std::vector<int64_t> dims(info.rank);
        for (uint8_t d = 0; d < info.rank; ++d) dims[d] = static_cast<int64_t>(info.dimensions[d]);
        ggml_tensor * tensor = ggml_new_tensor(ctx, tensor_type(info.representation), info.rank, dims.data());
        if (!tensor) throw std::runtime_error("GGML tensor metadata allocation failed");
        ggml_set_name(tensor, name);
        gguf_add_tensor(meta, tensor);
    }
    return owner;
}

static void set_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & source = *static_cast<VbufRuntime *>(userdata);
    auto it = source.payloads.find(ggml_get_name(tensor));
    if (it == source.payloads.end()) throw std::runtime_error(std::string("vBuf payload is missing: ") + ggml_get_name(tensor));
    if (source.sizes.at(it->first) != ggml_nbytes(tensor)) throw std::runtime_error(std::string("vBuf payload does not match GGML tensor: ") + ggml_get_name(tensor) + " expected=" + std::to_string(ggml_nbytes(tensor)) + " actual=" + std::to_string(source.sizes.at(it->first)));
    tensor->data = const_cast<uint8_t *>(it->second);
}

} // namespace

extern "C" llama_model * llama_model_load_vbuf(const char * path, llama_model_params params) {
    try {
        auto source = std::make_unique<VbufRuntime>(path);
        gguf_context_ptr meta(gguf_init_empty());
        if (!meta) throw std::runtime_error("GGUF metadata context allocation failed");
        set_model_metadata(meta.get(), *source);
        auto tensor_context = add_tensor_metadata(meta.get(), *source);
        llama_model * model = llama_model_init_from_user(meta.get(), set_tensor_data, source.get(), params);
        if (!model) throw std::runtime_error("llama model construction failed");
        {
            std::lock_guard lock(g_vbuf_models_mutex);
            g_vbuf_models.emplace(model, source.release());
        }
        return model;
    } catch (...) { return nullptr; }
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
