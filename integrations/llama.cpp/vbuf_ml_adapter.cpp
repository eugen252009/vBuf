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

} // namespace vbuf_llama
