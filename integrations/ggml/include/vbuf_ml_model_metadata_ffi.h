#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct VbufMlConsumerHandle;

struct VbufMlTensorSourceInfo {
    uint64_t source_id;
    uint64_t offset;
    uint64_t length;
};

struct VbufMlSourceIdentityInfo {
    uint64_t source_id;
    uint64_t declared_size;
    uint16_t hash_algorithm;
    uint16_t hash_len;
    uint8_t full_source_hash[32];
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
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint32_t expert_shared_count;
    uint32_t expert_feed_forward_length;
    uint32_t leading_dense_block_count;
    uint32_t kv_lora_rank;
    uint32_t rope_dimension;
    uint32_t vocabulary_size;
};

uint32_t vbuf_ml_consumer_architecture(
    const VbufMlConsumerHandle * handle, char * buffer, size_t capacity);
uint32_t vbuf_ml_consumer_metadata(
    const VbufMlConsumerHandle * handle, VbufMlModelMetadataInfo * info);
uint32_t vbuf_ml_consumer_tensor_source(
    const VbufMlConsumerHandle * handle, uint64_t index, VbufMlTensorSourceInfo * info);
uint32_t vbuf_ml_consumer_source_identity(
    const VbufMlConsumerHandle * handle, uint64_t source_id, VbufMlSourceIdentityInfo * info);

} // extern "C"
