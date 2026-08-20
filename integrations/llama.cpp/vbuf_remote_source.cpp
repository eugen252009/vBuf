#include "vbuf_remote_source.h"
#include "../ggml/include/vbuf_d0_1_diagnostics.h"

#include <algorithm>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vbuf_llama {
namespace {

uint64_t timestamp_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint8_t representation(ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32: return 0;
    case GGML_TYPE_BF16: return 1;
    case GGML_TYPE_Q8_0: return 2;
    case GGML_TYPE_Q4_0: return 3;
    case GGML_TYPE_Q2_K: return 4;
    case GGML_TYPE_IQ1_S: return 5;
    case GGML_TYPE_Q4_K: return 6;
    case GGML_TYPE_IQ4_NL: return 7;
    case GGML_TYPE_IQ4_XS: return 8;
    case GGML_TYPE_Q3_K: return 9;
    case GGML_TYPE_IQ2_XXS: return 10;
    case GGML_TYPE_IQ2_XS: return 11;
    case GGML_TYPE_IQ2_S: return 12;
    case GGML_TYPE_Q5_K: return 13;
    default: throw std::runtime_error("unsupported remote vBuf tensor representation");
    }
}

} // namespace

VbufRemoteSource::VbufRemoteSource(const char * bootstrap_path, const char * endpoint)
    : adapter_(bootstrap_path, true), endpoint_(endpoint),
      http_source_(std::make_shared<vbuf_ggml::HttpRangeSource>(endpoint_)),
      range_source_(http_source_) {
    if (!adapter_.valid() || !adapter_.metadata(metadata_) || !adapter_.architecture(architecture_)) {
        throw std::runtime_error("remote vBuf bootstrap discovery failed");
    }
    const uint64_t count = adapter_.tensor_count();
    if (count == 0) throw std::runtime_error("remote vBuf bootstrap has no tensors");
    vbuf_d0_1_set_tensor_count(count);
    tensors_.reserve(count);
    refs_.reserve(count);
    for (uint64_t index = 0; index < count; ++index) {
        TensorDescriptor descriptor;
        if (!adapter_.tensor_metadata(index, descriptor) || descriptor.source_id != 1) {
            throw std::runtime_error("remote vBuf tensor binding is not SourceId 1");
        }
        tensors_.push_back(std::move(descriptor));
    }
    for (uint64_t index = 0; index < count; ++index) {
        uint8_t rep = 0;
        if (!tensor_representation(index, rep)) throw std::runtime_error("remote vBuf tensor representation failed");
        const TensorDescriptor & descriptor = tensors_[index];
        vbuf_ggml::VbufTensorView view{};
        view.representation = rep;
        view.rank = static_cast<uint8_t>(descriptor.dimensions.size());
        view.dimensions = descriptor.dimensions.data();
        view.payload_len = descriptor.payload_bytes;
        refs_.push_back({ index, descriptor.name, view, descriptor.source_offset });
    }
    materializer_ = std::make_unique<vbuf_ggml::LocalVbufRangeMaterializer>(range_source_);
}

int VbufRemoteSource::metadata_count() const { return 22; }

bool VbufRemoteSource::get_string(const std::string & key, std::string & value) const {
    if (key == "general.architecture") { value = architecture_; return true; }
    if (key == "tokenizer.ggml.model") { value = "gpt2"; return true; }
    if (key == "tokenizer.ggml.pre") { value = "qwen2"; return true; }
    if (key == "tokenizer.chat_template") return adapter_.chat_template(value);
    return false;
}

bool VbufRemoteSource::get_u32(const std::string & key, uint32_t & value) const {
    uint64_t v = 0;
    if (key == "qwen3.context_length") v = metadata_.context_length;
    else if (key == "qwen3.embedding_length") v = metadata_.embedding_length;
    else if (key == "qwen3.block_count") v = metadata_.layer_count;
    else if (key == "qwen3.feed_forward_length") v = metadata_.feed_forward_length;
    else if (key == "qwen3.attention.head_count") v = metadata_.head_count;
    else if (key == "qwen3.attention.head_count_kv") v = metadata_.kv_head_count;
    else if (key == "qwen3.attention.key_length") v = metadata_.key_head_dimension;
    else if (key == "qwen3.attention.value_length") v = metadata_.value_head_dimension;
    else if (key == "qwen3.rope.dimension_count") v = metadata_.rope_dimension;
    else if (key == "tokenizer.ggml.bos_token_id" || key == "tokenizer.ggml.eos_token_id" || key == "tokenizer.ggml.padding_token_id") {
        const uint8_t kind = key == "tokenizer.ggml.bos_token_id" ? 0 : key == "tokenizer.ggml.eos_token_id" ? 1 : 3;
        uint64_t special = 0;
        if (!adapter_.special_token(kind, special)) return false;
        value = static_cast<uint32_t>(special);
        return true;
    }
    else return false;
    value = static_cast<uint32_t>(v);
    return true;
}

bool VbufRemoteSource::get_f32(const std::string & key, float & value) const {
    if (key == "qwen3.attention.layer_norm_rms_epsilon") { value = static_cast<float>(metadata_.normalization_epsilon); return true; }
    if (key == "qwen3.rope.freq_base") { value = static_cast<float>(metadata_.rope_theta); return true; }
    return false;
}

bool VbufRemoteSource::get_bool(const std::string & key, bool & value) const {
    if (key != "tokenizer.ggml.add_bos_token") return false;
    if (!adapter_.add_bos(value)) return false;
    return true;
}

uint64_t VbufRemoteSource::tensor_count() const { return tensors_.size(); }

bool VbufRemoteSource::tensor_type(uint64_t index, ggml_type & type) const {
    if (index >= tensors_.size()) return false;
    type = tensors_[index].type;
    return true;
}

bool VbufRemoteSource::tensor_representation(uint64_t index, uint8_t & rep) const {
    ggml_type type;
    if (!tensor_type(index, type)) return false;
    rep = representation(type);
    return true;
}

bool VbufRemoteSource::tensor(uint64_t index, std::string & name, ggml_type & type,
    std::vector<int64_t> & dimensions, const uint8_t * & payload,
    uint64_t & payload_bytes) const {
    if (index >= tensors_.size() || !tensor_type(index, type)) return false;
    const TensorDescriptor & descriptor = tensors_[index];
    name = descriptor.name;
    dimensions.assign(descriptor.dimensions.begin(), descriptor.dimensions.end());
    payload_bytes = descriptor.payload_bytes;
    std::optional<vbuf_ggml::MaterializedTensor> materialized;
    {
        std::lock_guard<std::mutex> lock(materialized_mutex_);
        auto found = materialized_.find(index);
        if (found != materialized_.end()) materialized = found->second;
    }
    if (!materialized) {
        if (!materialize_tensor(index)) return false;
        std::lock_guard<std::mutex> lock(materialized_mutex_);
        auto found = materialized_.find(index);
        if (found != materialized_.end()) materialized = found->second;
    }
    payload = materialized ? materialized->view().payload : nullptr;
    return true;
}

bool VbufRemoteSource::materialize_tensor(uint64_t index) const {
    if (index >= refs_.size()) return false;
    {
        std::lock_guard<std::mutex> lock(materialized_mutex_);
        if (materialized_.find(index) != materialized_.end()) return true;
    }
    if (!materializer_->request(static_cast<uint32_t>(index), refs_[index], UINT64_MAX)) return false;
    if (materializer_->wait(static_cast<uint32_t>(index)) != vbuf_ggml::MaterializationState::Ready) return false;
    auto ready = materializer_->obtain_ready_tensor(static_cast<uint32_t>(index));
    if (!ready) return false;
    {
        std::lock_guard<std::mutex> lock(materialized_mutex_);
        materialized_.emplace(index, std::move(*ready));
        vbuf_d0_1_payload_ready(materialized_.size(), timestamp_ns());
        if (materialized_.size() == tensors_.size()) {
            vbuf_d0_1_final_payload_ready(timestamp_ns());
        }
    }
    return true;
}

uint64_t VbufRemoteSource::token_count() const { return adapter_.token_count(); }

bool VbufRemoteSource::token(uint64_t index, std::string & text, float & score, int32_t & type) const {
    if (!adapter_.token_text(index, text) || !adapter_.token_type(index, type)) return false;
    score = 0.0f;
    adapter_.token_score(index, score);
    return true;
}

uint64_t VbufRemoteSource::merge_count() const { return adapter_.merge_count(); }

bool VbufRemoteSource::merge(uint64_t index, uint64_t & left, uint64_t & right) const {
    return adapter_.merge_pair(index, left, right);
}

bool VbufRemoteSource::metrics(VbufRemoteMetrics & out) const {
    out = {};
    std::unordered_set<std::string> unique;
    std::unordered_set<std::string> connections;
    std::unordered_map<uint32_t, uint64_t> starts;
    std::vector<uint64_t> durations;
    for (const auto & event : materializer_->trace()) {
        if (event.event == "STATE" && event.state == vbuf_ggml::MaterializationState::InFlight) {
            starts[event.tensor_ref] = event.timestamp_ns;
            continue;
        }
        if (event.event != "STATE" || event.state != vbuf_ggml::MaterializationState::Ready) continue;
        ++out.requests;
        out.bytes += event.returned_bytes;
        unique.insert(std::to_string(event.requested_offset) + ":" + std::to_string(event.returned_bytes));
        if (!event.local_endpoint.empty()) connections.insert(event.local_endpoint);
        const auto start = starts.find(event.tensor_ref);
        if (start != starts.end() && event.timestamp_ns >= start->second) durations.push_back(event.timestamp_ns - start->second);
    }
    for (const auto & range : unique) {
        const size_t separator = range.find(':');
        out.unique_bytes += std::stoull(range.substr(separator + 1));
    }
    out.connections = connections.size();
    if (!durations.empty()) {
        std::sort(durations.begin(), durations.end());
        out.min_request_ns = durations.front();
        out.median_request_ns = durations[durations.size() / 2];
        out.max_request_ns = durations.back();
    }
    const vbuf_ggml::RangeSourceMetrics transport = http_source_->metrics();
    out.requests = transport.requests;
    out.bytes = transport.bytes;
    out.unique_bytes = transport.unique_bytes;
    out.connections = transport.connections;
    if (!durations.empty()) {
        std::sort(durations.begin(), durations.end());
        out.min_request_ns = durations.front();
        out.median_request_ns = durations[durations.size() / 2];
        out.max_request_ns = durations.back();
    }
    return true;
}

bool VbufRemoteSource::transport_control(bool large_range, VbufTransportControlResult & out) const {
    out = {};
    if (refs_.empty()) return false;

    uint64_t min_offset = UINT64_MAX;
    uint64_t max_end = 0;
    for (const auto & ref : refs_) {
        min_offset = std::min(min_offset, ref.source_offset);
        max_end = std::max(max_end, ref.source_offset + ref.view.payload_len);
        out.payload_bytes += ref.view.payload_len;
    }

    const uint64_t start_ns = timestamp_ns();
    if (large_range) {
        const uint64_t length = max_end - min_offset;
        std::vector<uint8_t> buffer(static_cast<size_t>(length));
        vbuf_ggml::RangeReadResult read{};
        if (!http_source_->read_range(min_offset, length, buffer.data(), &read)) return false;
        out.range_count = 1;
        out.requested_bytes = length;
    } else {
        uint64_t max_length = 0;
        for (const auto & ref : refs_) max_length = std::max(max_length, ref.view.payload_len);
        std::vector<uint8_t> buffer(static_cast<size_t>(max_length));
        for (const auto & ref : refs_) {
            vbuf_ggml::RangeReadResult read{};
            if (!http_source_->read_range(ref.source_offset, ref.view.payload_len, buffer.data(), &read)) return false;
            ++out.range_count;
            out.requested_bytes += ref.view.payload_len;
        }
    }
    out.transport_total_ns = timestamp_ns() - start_ns;
    out.overfetch_bytes = out.requested_bytes - out.payload_bytes;
    out.transport.requests = http_source_->metrics().requests;
    out.transport.bytes = http_source_->metrics().bytes;
    out.transport.unique_bytes = http_source_->metrics().unique_bytes;
    out.transport.connections = http_source_->metrics().connections;
    out.transport.min_request_ns = http_source_->metrics().min_request_ns;
    out.transport.median_request_ns = http_source_->metrics().median_request_ns;
    out.transport.max_request_ns = http_source_->metrics().max_request_ns;
    return true;
}

std::shared_ptr<VbufRemoteSource> make_vbuf_remote_source(
    const char * bootstrap_path, const char * endpoint) {
    return std::make_shared<VbufRemoteSource>(bootstrap_path, endpoint);
}

bool probe_vbuf_remote_source(const char * bootstrap_path, const char * endpoint,
    VbufRemoteMetrics & metrics) {
    auto source = make_vbuf_remote_source(bootstrap_path, endpoint);
    if (!source->materialize_tensor(0)) return false;
    return source->metrics(metrics);
}

bool transport_control_vbuf_remote_source(const char * bootstrap_path, const char * endpoint,
    bool large_range, VbufTransportControlResult & result) {
    auto source = make_vbuf_remote_source(bootstrap_path, endpoint);
    return source->transport_control(large_range, result);
}

void set_vbuf_remote_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & source = *static_cast<VbufRemoteSource *>(userdata);
    std::string wanted = ggml_get_name(tensor);
    const auto lookup_start = std::chrono::steady_clock::now();
    uint64_t lookup_iterations = 0;
    vbuf_d0_1_pointer_bind_begin();
    for (uint64_t index = 0; index < source.tensor_count(); ++index) {
        ++lookup_iterations;
        std::string name;
        ggml_type type;
        std::vector<int64_t> dimensions;
        const uint8_t * payload = nullptr;
        uint64_t bytes = 0;
        if (!source.tensor(index, name, type, dimensions, payload, bytes)) throw std::runtime_error("remote tensor descriptor failed");
        if (name == wanted || (wanted == "output.weight" && name == "token_embd.weight")) {
            if (!source.materialize_tensor(index)) throw std::runtime_error("remote tensor materialization failed");
            if (!source.tensor(index, name, type, dimensions, payload, bytes) || type != tensor->type || bytes != ggml_nbytes(tensor)) {
                throw std::runtime_error("remote tensor metadata mismatch");
            }
            const auto pointer_start = std::chrono::steady_clock::now();
            tensor->data = const_cast<uint8_t *>(payload);
            const auto pointer_end = std::chrono::steady_clock::now();
            const uint64_t lookup_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pointer_start - lookup_start).count());
            const uint64_t pointer_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(pointer_end - pointer_start).count());
            vbuf_d0_1_pointer_bind_complete(lookup_iterations, lookup_ns, pointer_ns);
            return;
        }
    }
    throw std::runtime_error("remote tensor payload is missing");
}

} // namespace vbuf_llama
