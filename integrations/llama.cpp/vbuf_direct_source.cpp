#include "vbuf_direct_source.h"

#include "vbuf_ml_adapter.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

extern "C" {
struct VbufMlTokenArrays { const uint8_t * text; uint64_t text_len; const uint8_t * offsets; uint64_t offset_count; const uint8_t * types; uint64_t type_bytes; const uint8_t * scores; uint64_t score_bytes; uint64_t token_count; };
struct VbufMlMergeArrays { const uint8_t * left; const uint8_t * right; uint64_t merge_count; };
struct VbufMlTensorView { const char * name; uint64_t name_len; uint8_t representation; uint8_t rank; const uint64_t * dimensions; const uint8_t * payload; uint64_t payload_len; };
VbufMlConsumerHandle * vbuf_ml_consumer_open(const char * path);
void vbuf_ml_consumer_close(VbufMlConsumerHandle * handle);
uint32_t vbuf_ml_consumer_metadata(const VbufMlConsumerHandle *, VbufMlModelMetadataInfo *);
uint32_t vbuf_ml_consumer_architecture(const VbufMlConsumerHandle *, char *, size_t);
uint32_t vbuf_ml_consumer_token_arrays(const VbufMlConsumerHandle *, VbufMlTokenArrays *);
uint32_t vbuf_ml_consumer_merge_arrays(const VbufMlConsumerHandle *, VbufMlMergeArrays *);
uint32_t vbuf_ml_consumer_tensor_views(const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
uint32_t vbuf_ml_consumer_tensor_physical_range(const VbufMlConsumerHandle *, uint64_t, uint64_t *, uint64_t *);
uint32_t vbuf_ml_consumer_special_token(const VbufMlConsumerHandle *, uint8_t, uint64_t *);
uint32_t vbuf_ml_consumer_add_bos(const VbufMlConsumerHandle *, bool *);
uint32_t vbuf_ml_consumer_chat_template(const VbufMlConsumerHandle *, char *, size_t);
}

namespace vbuf_llama {

static uint32_t load_le32(const uint8_t * p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
static uint64_t load_le64(const uint8_t * p) { uint64_t value = 0; for (unsigned i = 0; i < 8; ++i) value |= uint64_t(p[i]) << (8 * i); return value; }
static float load_le_f32(const uint8_t * p) { const uint32_t bits = load_le32(p); float value; std::memcpy(&value, &bits, sizeof(value)); return value; }

class VbufDirectSource final : public llama_model_source {
public:
    explicit VbufDirectSource(const char * path) : handle_(vbuf_ml_consumer_open(path)) {
        if (!handle_) throw std::runtime_error("vBuf direct source open failed");
        char architecture[128]{};
        if (vbuf_ml_consumer_architecture(handle_, architecture, sizeof(architecture)) != 0) throw std::runtime_error("vBuf architecture lookup failed");
        architecture_ = architecture;
        if (vbuf_ml_consumer_metadata(handle_, &metadata_) != 0) throw std::runtime_error("vBuf metadata lookup failed");
        if (vbuf_ml_consumer_token_arrays(handle_, &token_arrays_) != 0 || vbuf_ml_consumer_merge_arrays(handle_, &merge_arrays_) != 0 || vbuf_ml_consumer_tensor_views(handle_, &tensors_, &tensor_count_) != 0) throw std::runtime_error("vBuf borrowed source view failed");
        offsets_.resize(tensor_count_); lengths_.resize(tensor_count_);
        for (uint64_t i = 0; i < tensor_count_; ++i) if (vbuf_ml_consumer_tensor_physical_range(handle_, i, &offsets_[i], &lengths_[i]) != 0) throw std::runtime_error("vBuf physical tensor range failed");
        token_count_ = token_arrays_.token_count; merge_count_ = merge_arrays_.merge_count;
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
        if (index >= token_count_) return false;
        uint64_t start = load_le64(token_arrays_.offsets + index * 8); uint64_t end = load_le64(token_arrays_.offsets + (index + 1) * 8);
        if (end < start || end > token_arrays_.text_len) return false; text.assign(reinterpret_cast<const char *>(token_arrays_.text + start), end - start);
        type = 1; if (token_arrays_.type_bytes != 0) { const uint64_t width = token_arrays_.type_bytes / token_count_; uint64_t value = 0; if (width == 1) value = token_arrays_.types[index]; else if (width == 2) { value = uint64_t(token_arrays_.types[index * width]) | (uint64_t(token_arrays_.types[index * width + 1]) << 8); } else if (width == 4) value = load_le32(token_arrays_.types + index * width); else if (width == 8) value = load_le64(token_arrays_.types + index * width); else return false; type = static_cast<int32_t>(value); }
        score = 0.0f; if (token_arrays_.score_bytes != 0) { const uint64_t width = token_arrays_.score_bytes / token_count_; if (width == 4) score = load_le_f32(token_arrays_.scores + index * width); else if (width == 8) { uint64_t bits = load_le64(token_arrays_.scores + index * width); double v; std::memcpy(&v, &bits, sizeof(v)); score = static_cast<float>(v); } else return false; } return true;
    }
    uint64_t merge_count() const override { return merge_count_; }
    bool merge(uint64_t index, uint64_t & left, uint64_t & right) const override {
        if (index >= merge_count_) return false; uint32_t l = load_le32(merge_arrays_.left + index * 4); uint32_t r = load_le32(merge_arrays_.right + index * 4); left = l; right = r; return true;
    }
    bool prepare(const char * variant, Step28PreparationResult & result) const {
        struct Record { std::string name; uint64_t offset, length; const uint8_t * payload; int layer = -1; };
        struct WorkSpan { Step28PhysicalSpan report; const uint8_t * begin = nullptr; const uint8_t * end = nullptr; };
        std::vector<Record> records;
        records.reserve(tensor_count_);
        for (uint64_t i = 0; i < tensor_count_; ++i) {
            const auto & view = tensors_[i];
            Record record{std::string(view.name, view.name_len), offsets_[i], lengths_[i], view.payload};
            if (record.name.rfind("blk.", 0) == 0) {
                unsigned layer = 0; if (std::sscanf(record.name.c_str(), "blk.%u.", &layer) != 1) return false;
                record.layer = static_cast<int>(layer);
            }
            if (record.payload == nullptr || record.length == 0 || record.offset + record.length < record.offset) return false;
            records.push_back(std::move(record));
        }
        auto make_span = [](const std::vector<const Record *> & members, const char * role, uint64_t layer_id) {
            WorkSpan span; span.report.role = role; span.report.layer_id = layer_id;
            auto first = std::min_element(members.begin(), members.end(), [](auto a, auto b) { return a->offset < b->offset; });
            auto last = std::max_element(members.begin(), members.end(), [](auto a, auto b) { return a->offset + a->length < b->offset + b->length; });
            span.report.start_offset = (*first)->offset; span.report.end_offset = (*last)->offset + (*last)->length;
            span.report.span_bytes = span.report.end_offset - span.report.start_offset;
            span.begin = (*first)->payload; span.end = (*last)->payload + (*last)->length;
            for (const Record * record : members) span.report.useful_bytes += record->length;
            span.report.gap_bytes = span.report.span_bytes - span.report.useful_bytes;
            span.report.tensor_count = members.size();
            if (static_cast<uint64_t>(span.end - span.begin) != span.report.span_bytes) return WorkSpan{};
            return span;
        };
        std::vector<WorkSpan> layers;
        std::vector<WorkSpan> globals;
        for (uint64_t layer = 0; layer < metadata_.layer_count; ++layer) {
            std::vector<const Record *> members;
            for (const auto & record : records) if (record.layer == static_cast<int>(layer)) members.push_back(&record);
            if (members.empty()) return false;
            auto span = make_span(members, "layer", layer); if (span.begin == nullptr || span.end == nullptr) return false;
            layers.push_back(std::move(span));
        }
        for (const auto & record : records) { if (record.layer < 0) { auto span = make_span({&record}, "global", UINT64_MAX); if (span.begin == nullptr || span.end == nullptr) return false; globals.push_back(std::move(span)); } }
        std::sort(layers.begin(), layers.end(), [](const auto & a, const auto & b) { return a.report.start_offset < b.report.start_offset; });
        std::sort(globals.begin(), globals.end(), [](const auto & a, const auto & b) { return a.report.start_offset < b.report.start_offset; });
        if (layers.size() != metadata_.layer_count) return false;
        result.layer_count = layers.size(); result.global_span_count = globals.size(); result.span_count = layers.size() + globals.size();
        for (const auto & span : layers) { result.covered_bytes += span.report.span_bytes; result.useful_bytes += span.report.useful_bytes; result.gap_bytes += span.report.gap_bytes; result.spans.push_back(span.report); result.layers.push_back({span.report.layer_id, span.report.start_offset, span.report.end_offset, span.report.span_bytes, span.report.useful_bytes, span.report.gap_bytes, span.report.tensor_count, 0, 0, 0}); }
        for (const auto & span : globals) { result.covered_bytes += span.report.span_bytes; result.useful_bytes += span.report.useful_bytes; result.gap_bytes += span.report.gap_bytes; result.spans.push_back(span.report); }
        std::vector<WorkSpan> order;
        if (std::strcmp(variant, "baseline") == 0) return true;
        if (std::strcmp(variant, "sequential") == 0) {
            std::vector<const Record *> all; for (const auto & record : records) all.push_back(&record);
            order.push_back(make_span(all, "whole-payload", UINT64_MAX));
        } else if (std::strcmp(variant, "layer") == 0) {
            for (const auto & span : globals) if (span.report.end_offset < layers.front().report.start_offset) order.push_back(span);
            for (const auto & span : layers) order.push_back(span);
            for (const auto & span : globals) if (span.report.start_offset > layers.back().report.end_offset) order.push_back(span);
            for (const auto & span : globals) if (span.report.start_offset >= layers.front().report.start_offset && span.report.end_offset <= layers.back().report.end_offset) order.push_back(span);
        } else if (std::strcmp(variant, "advisory") == 0) {
            order = globals; order.insert(order.end(), layers.begin(), layers.end()); std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.report.start_offset < b.report.start_offset; });
        } else return false;
        const auto begin = std::chrono::steady_clock::now(); struct rusage before{}; getrusage(RUSAGE_SELF, &before);
        volatile uint8_t sink = 0;
        const long page_size = sysconf(_SC_PAGESIZE); result.page_size = static_cast<uint64_t>(page_size);
        for (const auto & span : order) {
            result.prepared_span_bytes += span.report.span_bytes; result.prepared_useful_bytes += span.report.useful_bytes;
            const auto layer_begin = std::chrono::steady_clock::now(); struct rusage layer_before{}; getrusage(RUSAGE_SELF, &layer_before);
            const uintptr_t begin_address = reinterpret_cast<uintptr_t>(span.begin) & ~static_cast<uintptr_t>(page_size - 1);
            const uintptr_t end_address = (reinterpret_cast<uintptr_t>(span.end) + page_size - 1) & ~static_cast<uintptr_t>(page_size - 1);
            if (std::strcmp(variant, "advisory") == 0) {
                if (madvise(reinterpret_cast<void *>(begin_address), end_address - begin_address, MADV_WILLNEED) != 0) return false;
                continue;
            }
            for (uintptr_t address = begin_address; address < end_address; address += static_cast<uintptr_t>(page_size)) { sink ^= *reinterpret_cast<volatile const uint8_t *>(address); ++result.pages_touched; ++result.touch_operations; }
            if (std::strcmp(variant, "layer") == 0 && span.report.layer_id != UINT64_MAX) {
                struct rusage layer_after{}; getrusage(RUSAGE_SELF, &layer_after);
                result.layers[span.report.layer_id].prepare_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - layer_begin).count();
                result.layers[span.report.layer_id].minor_faults = layer_after.ru_minflt - layer_before.ru_minflt; result.layers[span.report.layer_id].major_faults = layer_after.ru_majflt - layer_before.ru_majflt;
            }
        }
        std::atomic_signal_fence(std::memory_order_seq_cst); (void) sink;
        struct rusage after{}; getrusage(RUSAGE_SELF, &after);
        result.prepare_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
        result.minor_faults = after.ru_minflt - before.ru_minflt; result.major_faults = after.ru_majflt - before.ru_majflt;
        return true;
    }

private:
    VbufMlConsumerHandle * handle_ = nullptr;
    VbufMlModelMetadataInfo metadata_{};
    std::string architecture_;
    std::optional<std::string> chat_template_;
    VbufMlTokenArrays token_arrays_{};
    VbufMlMergeArrays merge_arrays_{};
    const VbufMlTensorView * tensors_ = nullptr;
    std::vector<uint64_t> offsets_, lengths_;
    uint64_t token_count_ = 0, merge_count_ = 0, tensor_count_ = 0;
    uint64_t special_[4]{};
    bool add_bos_ = false;
};

std::shared_ptr<llama_model_source> make_vbuf_direct_source(const char * path) {
    return std::make_shared<VbufDirectSource>(path);
}

bool step28_prepare_vbuf_source(llama_model_source * source, const char * variant, Step28PreparationResult & result) {
    auto * direct = dynamic_cast<VbufDirectSource *>(source);
    return direct != nullptr && direct->prepare(variant, result);
}

void set_vbuf_direct_tensor_data(ggml_tensor * tensor, void * userdata) {
    auto & source = *static_cast<llama_model_source *>(userdata);
    std::string wanted = ggml_get_name(tensor);
    if (wanted == "output.weight") {
        bool has_output = false;
        for (uint64_t i = 0; i < source.tensor_count(); ++i) {
            std::string name; ggml_type type; std::vector<int64_t> dimensions; const uint8_t * payload = nullptr; uint64_t bytes = 0;
            if (!source.tensor(i, name, type, dimensions, payload, bytes)) throw std::runtime_error("invalid direct tensor source descriptor");
            if (name == "output.weight") { has_output = true; break; }
        }
        if (!has_output) wanted = "token_embd.weight";
    }
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
