#pragma once

#include "llama-model-source.h"
#include "vbuf_materializer.h"
#include "vbuf_ml_adapter.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace vbuf_llama {

struct VbufRemoteMetrics {
    uint64_t requests = 0;
    uint64_t bytes = 0;
    uint64_t unique_bytes = 0;
    uint64_t connections = 0;
    uint64_t min_request_ns = 0;
    uint64_t median_request_ns = 0;
    uint64_t max_request_ns = 0;
};

class VbufRemoteSource final : public llama_model_source {
public:
    VbufRemoteSource(const char * bootstrap_path, const char * endpoint);
    ~VbufRemoteSource() override = default;

    int metadata_count() const override;
    bool get_string(const std::string & key, std::string & value) const override;
    bool get_u32(const std::string & key, uint32_t & value) const override;
    bool get_i32(const std::string &, int32_t &) const override { return false; }
    bool get_u64(const std::string &, uint64_t &) const override { return false; }
    bool get_f32(const std::string & key, float & value) const override;
    bool get_bool(const std::string & key, bool & value) const override;
    bool get_i32_array(const std::string &, std::vector<int32_t> &) const override { return false; }

    uint64_t tensor_count() const override;
    bool tensor(uint64_t index, std::string & name, ggml_type & type,
        std::vector<int64_t> & dimensions, const uint8_t * & payload,
        uint64_t & payload_bytes) const override;
    uint64_t token_count() const override;
    bool token(uint64_t index, std::string & text, float & score, int32_t & type) const override;
    uint64_t merge_count() const override;
    bool merge(uint64_t index, uint64_t & left, uint64_t & right) const override;

    bool materialize_tensor(uint64_t index) const;
    bool metrics(VbufRemoteMetrics & out) const;

private:
    bool tensor_type(uint64_t index, ggml_type & type) const;
    bool tensor_representation(uint64_t index, uint8_t & representation) const;

    VbufMlAdapter adapter_;
    VbufMlModelMetadataInfo metadata_{};
    std::string architecture_;
    std::string endpoint_;
    std::vector<TensorDescriptor> tensors_;
    std::vector<vbuf_ggml::PersistentTensorRef> refs_;
    std::shared_ptr<vbuf_ggml::HttpRangeSource> http_source_;
    std::shared_ptr<vbuf_ggml::RangeSource> range_source_;
    mutable std::unique_ptr<vbuf_ggml::LocalVbufRangeMaterializer> materializer_;
    mutable std::unordered_map<uint64_t, vbuf_ggml::MaterializedTensor> materialized_;
    mutable std::mutex materialized_mutex_;
};

std::shared_ptr<VbufRemoteSource> make_vbuf_remote_source(
    const char * bootstrap_path, const char * endpoint);
bool probe_vbuf_remote_source(const char * bootstrap_path, const char * endpoint,
    VbufRemoteMetrics & metrics);
void set_vbuf_remote_tensor_data(ggml_tensor * tensor, void * userdata);

} // namespace vbuf_llama
