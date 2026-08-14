#include "vbuf_direct_source.h"

#include "vbuf_ml_adapter.h"

#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

extern "C" {
struct VbufMlTokenView { const char * text; uint64_t text_len; float score; int32_t token_type; };
struct VbufMlMergeView { uint64_t left; uint64_t right; };
struct VbufMlTensorView { const char * name; uint64_t name_len; uint8_t representation; uint8_t rank; const uint64_t * dimensions; const uint8_t * payload; uint64_t payload_len; };
VbufMlConsumerHandle * vbuf_ml_consumer_open(const char * path);
void vbuf_ml_consumer_close(VbufMlConsumerHandle * handle);
uint32_t vbuf_ml_consumer_metadata(const VbufMlConsumerHandle *, VbufMlModelMetadataInfo *);
uint32_t vbuf_ml_consumer_architecture(const VbufMlConsumerHandle *, char *, size_t);
uint32_t vbuf_ml_consumer_token_views(const VbufMlConsumerHandle *, const VbufMlTokenView **, uint64_t *);
uint32_t vbuf_ml_consumer_merge_views(const VbufMlConsumerHandle *, const VbufMlMergeView **, uint64_t *);
uint32_t vbuf_ml_consumer_tensor_views(const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
uint32_t vbuf_ml_consumer_special_token(const VbufMlConsumerHandle *, uint8_t, uint64_t *);
uint32_t vbuf_ml_consumer_add_bos(const VbufMlConsumerHandle *, bool *);
uint32_t vbuf_ml_consumer_chat_template(const VbufMlConsumerHandle *, char *, size_t);
}

namespace vbuf_llama {

class VbufDirectSource final : public llama_model_source {
public:
    explicit VbufDirectSource(const char * path) : handle_(vbuf_ml_consumer_open(path)) {
        if (!handle_) throw std::runtime_error("vBuf direct source open failed");
        char architecture[128]{};
        if (vbuf_ml_consumer_architecture(handle_, architecture, sizeof(architecture)) != 0) throw std::runtime_error("vBuf architecture lookup failed");
        architecture_ = architecture;
        if (vbuf_ml_consumer_metadata(handle_, &metadata_) != 0) throw std::runtime_error("vBuf metadata lookup failed");
        if (vbuf_ml_consumer_token_views(handle_, &tokens_, &token_count_) != 0 || vbuf_ml_consumer_merge_views(handle_, &merges_, &merge_count_) != 0 || vbuf_ml_consumer_tensor_views(handle_, &tensors_, &tensor_count_) != 0) throw std::runtime_error("vBuf bulk source view failed");
        for (uint8_t kind = 0; kind < 4; ++kind) vbuf_ml_consumer_special_token(handle_, kind, &special_[kind]);
        vbuf_ml_consumer_add_bos(handle_, &add_bos_);
        char chat[1024 * 1024]{};
        if (vbuf_ml_consumer_chat_template(handle_, chat, sizeof(chat)) == 0) chat_template_ = chat;
    }
    ~VbufDirectSource() override { vbuf_ml_consumer_close(handle_); }

    int metadata_count() const override { return 22; }
    bool get_string(const std::string & key, std::string & value) const override {
        if (key == "general.architecture") { value = architecture_; return true; }
        if (key == "tokenizer.ggml.model") { value = "gpt2"; return true; }
        if (key == "tokenizer.ggml.pre") { value = "qwen2"; return true; }
        if (key == "tokenizer.chat_template" && chat_template_) { value = *chat_template_; return true; }
        return false;
    }
    bool get_u32(const std::string & key, uint32_t & value) const override {
        const uint64_t * v = nullptr;
        if (key == "qwen3.context_length") v = &metadata_.context_length;
        else if (key == "qwen3.embedding_length") v = &metadata_.embedding_length;
        else if (key == "qwen3.block_count") v = &metadata_.layer_count;
        else if (key == "qwen3.feed_forward_length") v = &metadata_.feed_forward_length;
        else if (key == "qwen3.attention.head_count") v = &metadata_.head_count;
        else if (key == "qwen3.attention.head_count_kv") v = &metadata_.kv_head_count;
        else if (key == "qwen3.attention.key_length") v = &metadata_.key_head_dimension;
        else if (key == "qwen3.attention.value_length") v = &metadata_.value_head_dimension;
        else if (key == "tokenizer.ggml.bos_token_id") v = &special_[0];
        else if (key == "tokenizer.ggml.eos_token_id") v = &special_[1];
        else if (key == "tokenizer.ggml.padding_token_id") v = &special_[3];
        else return false;
        value = static_cast<uint32_t>(*v); return true;
    }
    bool get_i32(const std::string &, int32_t &) const override { return false; }
    bool get_u64(const std::string &, uint64_t &) const override { return false; }
    bool get_f32(const std::string & key, float & value) const override {
        if (key == "qwen3.attention.layer_norm_rms_epsilon") { value = static_cast<float>(metadata_.normalization_epsilon); return true; }
        if (key == "qwen3.rope.freq_base") { value = static_cast<float>(metadata_.rope_theta); return true; }
        return false;
    }
    bool get_bool(const std::string & key, bool & value) const override {
        if (key == "tokenizer.ggml.add_bos_token") { value = add_bos_; return true; }
        return false;
    }
    bool get_i32_array(const std::string &, std::vector<int32_t> &) const override { return false; }

    uint64_t tensor_count() const override { return tensor_count_; }
    bool tensor(uint64_t index, std::string & name, ggml_type & type, std::vector<int64_t> & dimensions, const uint8_t * & payload, uint64_t & payload_bytes) const override {
        if (index >= tensor_count_) return false;
        const auto & view = tensors_[index];
        name.assign(view.name, view.name_len); dimensions.assign(view.dimensions, view.dimensions + view.rank); payload = view.payload; payload_bytes = view.payload_len;
        switch (view.representation) { case 0: type = GGML_TYPE_F32; break; case 1: type = GGML_TYPE_BF16; break; case 2: type = GGML_TYPE_Q8_0; break; default: return false; }
        return true;
    }
    uint64_t token_count() const override { return token_count_; }
    bool token(uint64_t index, std::string & text, float & score, int32_t & type) const override {
        if (index >= token_count_) return false; const auto & view = tokens_[index]; text.assign(view.text, view.text_len); score = view.score; type = view.token_type; return true;
    }
    uint64_t merge_count() const override { return merge_count_; }
    bool merge(uint64_t index, uint64_t & left, uint64_t & right) const override {
        if (index >= merge_count_) return false; left = merges_[index].left; right = merges_[index].right; return true;
    }
private:
    VbufMlConsumerHandle * handle_ = nullptr;
    VbufMlModelMetadataInfo metadata_{};
    std::string architecture_;
    std::optional<std::string> chat_template_;
    const VbufMlTokenView * tokens_ = nullptr;
    const VbufMlMergeView * merges_ = nullptr;
    const VbufMlTensorView * tensors_ = nullptr;
    uint64_t token_count_ = 0, merge_count_ = 0, tensor_count_ = 0;
    uint64_t special_[4]{};
    bool add_bos_ = false;
};

std::shared_ptr<llama_model_source> make_vbuf_direct_source(const char * path) {
    return std::make_shared<VbufDirectSource>(path);
}

void set_vbuf_direct_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & source = *static_cast<llama_model_source *>(userdata);
    std::string wanted = ggml_get_name(tensor);
    if (wanted == "output.weight") wanted = "token_embd.weight";
    for (uint64_t i = 0; i < source.tensor_count(); ++i) {
        std::string name; ggml_type type; std::vector<int64_t> dimensions; const uint8_t * payload = nullptr; uint64_t bytes = 0;
        if (!source.tensor(i, name, type, dimensions, payload, bytes)) throw std::runtime_error("invalid direct tensor source descriptor");
        if (name == wanted) {
            if (type != tensor->type || bytes != ggml_nbytes(tensor)) throw std::runtime_error("direct tensor source metadata mismatch");
            tensor->data = const_cast<uint8_t *>(payload);
            return;
        }
    }
    throw std::runtime_error("direct tensor source payload is missing");
}

} // namespace vbuf_llama
