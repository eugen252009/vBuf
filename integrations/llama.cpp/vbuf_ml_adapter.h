#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

extern "C" {
struct VbufMlConsumerHandle;
struct VbufMlTensorInfo {
    uint8_t representation;
    uint8_t rank;
    uint64_t dimensions[16];
    const uint8_t * payload;
    uint64_t payload_len;
};
struct VbufMlModelMetadataInfo {
    uint64_t context_length;
    uint64_t embedding_length;
    uint64_t layer_count;
    uint64_t head_count;
    uint64_t kv_head_count;
    uint64_t key_head_dimension;
    uint64_t value_head_dimension;
    uint64_t feed_forward_length;
    double normalization_epsilon;
    double rope_theta;
};
VbufMlConsumerHandle * vbuf_ml_consumer_open(const char * path);
void vbuf_ml_consumer_close(VbufMlConsumerHandle * handle);
uint32_t vbuf_ml_consumer_tensor_count(const VbufMlConsumerHandle *, uint64_t * count);
uint32_t vbuf_ml_consumer_token_count(const VbufMlConsumerHandle *, uint64_t * count);
uint32_t vbuf_ml_consumer_token_type(const VbufMlConsumerHandle *, uint64_t index, int32_t * value);
uint32_t vbuf_ml_consumer_tensor_physical_range(const VbufMlConsumerHandle *, uint64_t index,
                                               uint64_t * offset, uint64_t * length);
uint32_t vbuf_ml_consumer_tensor_info(const VbufMlConsumerHandle *, uint64_t index,
                                      VbufMlTensorInfo *, char * name, size_t name_capacity);
uint32_t vbuf_ml_consumer_metadata(const VbufMlConsumerHandle *, VbufMlModelMetadataInfo *);
uint32_t vbuf_ml_consumer_token_text(const VbufMlConsumerHandle *, uint64_t index, char * buffer, size_t capacity);
uint32_t vbuf_ml_consumer_chat_template(const VbufMlConsumerHandle *, char * buffer, size_t capacity);
uint32_t vbuf_ml_consumer_token_score(const VbufMlConsumerHandle *, uint64_t index, float * value);
uint32_t vbuf_ml_consumer_special_token(const VbufMlConsumerHandle *, uint8_t kind, uint64_t * value);
uint32_t vbuf_ml_consumer_merge_count(const VbufMlConsumerHandle *, uint64_t * count);
uint32_t vbuf_ml_consumer_merge_pair(const VbufMlConsumerHandle *, uint64_t index, uint64_t * left, uint64_t * right);
uint32_t vbuf_ml_consumer_add_bos(const VbufMlConsumerHandle *, bool * value);
uint32_t vbuf_ml_consumer_runtime_indexes(const VbufMlConsumerHandle *);
uint32_t vbuf_ml_consumer_token_id(const VbufMlConsumerHandle *, const uint8_t *, size_t, uint32_t *);
uint32_t vbuf_ml_consumer_merge_rank(const VbufMlConsumerHandle *, uint64_t, uint64_t, uint32_t *);
}

namespace vbuf_llama {

struct TensorDescriptor {
    std::string name;
    std::vector<uint64_t> dimensions;
    ggml_type type;
    const uint8_t * payload;
    uint64_t payload_bytes;
};

// Descriptor-only first seam. It owns the Rust mapping handle, so payload
// pointers remain valid until this object is destroyed. It intentionally does
// not reach into llama_model_loader or construct a parallel runtime.
class VbufMlAdapter {
public:
    explicit VbufMlAdapter(const char * path);
    ~VbufMlAdapter();
    VbufMlAdapter(const VbufMlAdapter &) = delete;
    VbufMlAdapter & operator=(const VbufMlAdapter &) = delete;

    bool valid() const { return handle_ != nullptr; }
    uint64_t tensor_count() const;
    bool tensor(uint64_t index, TensorDescriptor & out) const;
    bool metadata(VbufMlModelMetadataInfo & out) const;
    bool token_text(uint64_t index, std::string & out) const;
    uint64_t merge_count() const;
    bool merge_pair(uint64_t index, uint64_t & left, uint64_t & right) const;
    bool add_bos(bool & out) const;
    bool build_runtime_indexes() const;
    bool token_id(const std::string & bytes, uint32_t & out) const;
    bool merge_rank(uint64_t left, uint64_t right, uint32_t & out) const;

private:
    VbufMlConsumerHandle * handle_ = nullptr;
};

} // namespace vbuf_llama
