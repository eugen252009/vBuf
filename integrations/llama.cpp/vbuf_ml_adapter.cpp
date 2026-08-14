#include "vbuf_ml_adapter.h"

namespace vbuf_llama {

VbufMlAdapter::VbufMlAdapter(const char * path) : handle_(vbuf_ml_consumer_open(path)) {}
VbufMlAdapter::~VbufMlAdapter() { vbuf_ml_consumer_close(handle_); }

uint64_t VbufMlAdapter::tensor_count() const {
    uint64_t count = 0;
    return handle_ && vbuf_ml_consumer_tensor_count(handle_, &count) == 0 ? count : 0;
}

bool VbufMlAdapter::tensor(uint64_t index, TensorDescriptor & out) const {
    if (!handle_) return false;
    VbufMlTensorInfo info{};
    char name[4096]{};
    if (vbuf_ml_consumer_tensor_info(handle_, index, &info, name, sizeof(name)) != 0) return false;
    out.name = name;
    out.dimensions.assign(info.dimensions, info.dimensions + info.rank);
    switch (info.representation) {
        case 0: out.type = GGML_TYPE_F32; break;
        case 1: out.type = GGML_TYPE_BF16; break;
        case 2: out.type = GGML_TYPE_Q8_0; break;
        default: return false;
    }
    out.payload = info.payload;
    out.payload_bytes = info.payload_len;
    return true;
}

bool VbufMlAdapter::metadata(VbufMlModelMetadataInfo & out) const {
    return handle_ && vbuf_ml_consumer_metadata(handle_, &out) == 0;
}

bool VbufMlAdapter::token_text(uint64_t index, std::string & out) const {
    if (!handle_) return false;
    char buffer[4097]{};
    if (vbuf_ml_consumer_token_text(handle_, index, buffer, sizeof(buffer)) != 0) return false;
    out = buffer;
    return true;
}

uint64_t VbufMlAdapter::merge_count() const {
    uint64_t count = 0;
    return handle_ && vbuf_ml_consumer_merge_count(handle_, &count) == 0 ? count : 0;
}

bool VbufMlAdapter::merge_pair(uint64_t index, uint64_t & left, uint64_t & right) const {
    return handle_ && vbuf_ml_consumer_merge_pair(handle_, index, &left, &right) == 0;
}

bool VbufMlAdapter::add_bos(bool & out) const {
    return handle_ && vbuf_ml_consumer_add_bos(handle_, &out) == 0;
}

bool VbufMlAdapter::build_runtime_indexes() const {
    return handle_ && vbuf_ml_consumer_runtime_indexes(handle_) == 0;
}

bool VbufMlAdapter::token_id(const std::string & bytes, uint32_t & out) const {
    return handle_ && vbuf_ml_consumer_token_id(handle_, reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size(), &out) == 0;
}

bool VbufMlAdapter::merge_rank(uint64_t left, uint64_t right, uint32_t & out) const {
    return handle_ && vbuf_ml_consumer_merge_rank(handle_, left, right, &out) == 0;
}

} // namespace vbuf_llama
