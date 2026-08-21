#define VBUF_ANDROID_DIRECT_RUNTIME
#define VBUF_POC22_LIBRARY_ONLY
#include "autoregressive_poc22.cpp"
#undef VBUF_POC22_LIBRARY_ONLY
#include "vbuf_prefill_batch.h"

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct VbufMlConsumerHandle;

struct VbufMlSourceIdentityInfo {
    uint64_t source_id;
    uint64_t declared_size;
    uint16_t hash_algorithm;
    uint16_t hash_len;
    uint8_t full_source_hash[32];
};

struct VbufMlTokenView {
    const char * text;
    uint64_t text_len;
    float score;
    int32_t token_type;
};

extern "C" uint32_t vbuf_ml_consumer_token_views(
    const VbufMlConsumerHandle *, const VbufMlTokenView **, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_merge_pair(
    const VbufMlConsumerHandle *, uint64_t, uint64_t *, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_merge_count(
    const VbufMlConsumerHandle *, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_special_token(
    const VbufMlConsumerHandle *, uint8_t, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_add_bos(
    const VbufMlConsumerHandle *, bool *);
extern "C" uint32_t vbuf_ml_consumer_source_identity(
    const VbufMlConsumerHandle *, uint64_t, VbufMlSourceIdentityInfo *);

namespace {

constexpr const char * TAG = "vbuf-android-direct";
#ifndef VBUF_RESIDENCY_BUDGET_BYTES
#define VBUF_RESIDENCY_BUDGET_BYTES 268435456
#endif
constexpr uint64_t RESIDENCY_BUDGET = static_cast<uint64_t>(VBUF_RESIDENCY_BUDGET_BYTES);
constexpr uint32_t MAX_CONTEXT = 128;
constexpr uint32_t MODEL_BLOCKS = 27;
#ifdef VBUF_ANDROID_QUALIFICATION
constexpr RuntimeMode ANDROID_RUNTIME_MODE = RuntimeMode::Qualification;
#else
constexpr RuntimeMode ANDROID_RUNTIME_MODE = RuntimeMode::NormalInference;
#endif

uint64_t android_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

const char * attribution_representation_name(uint8_t representation) {
    switch (representation) {
    case 0: return "F32";
    case 4: return "Q2_K";
    case 5: return "IQ1_S";
    case 7: return "IQ4_NL";
    case 10: return "IQ2_XXS";
    case 13: return "Q5_K";
    default: return "UNKNOWN";
    }
}

std::string attribution_dimensions(const std::array<uint64_t, 4> & dimensions, uint8_t rank) {
    std::string result;
    for (uint8_t index = 0; index < rank && index < dimensions.size(); ++index) {
        if (index != 0) result += "x";
        result += std::to_string(dimensions[index]);
    }
    return result;
}

std::string attribution_signature_line(const char * role,
    size_t index, const AttributionOperationSignature & signature) {
    const uint64_t m = signature.input_rank >= 2 ? signature.input_dims[1] : 0;
    const uint64_t n = signature.weight_rank >= 2 ? signature.weight_dims[1] : 0;
    const uint64_t k = signature.input_rank >= 1 ? signature.input_dims[0] : 0;
    return std::string("Attribution ") + role + " signature " + std::to_string(index) +
        ": count=" + std::to_string(signature.count) +
        " input_dims=" + attribution_dimensions(signature.input_dims, signature.input_rank) +
        " weight_dims=" + attribution_dimensions(signature.weight_dims, signature.weight_rank) +
        " output_dims=" + attribution_dimensions(signature.output_dims, signature.output_rank) +
        " M=" + std::to_string(m) + " N=" + std::to_string(n) + " K=" + std::to_string(k) +
        " weight_type=" + attribution_representation_name(signature.weight_representation) +
        " input_type=" + attribution_representation_name(signature.input_representation) +
        " output_type=" + attribution_representation_name(signature.output_representation) +
        " weight_representation=" + std::to_string(signature.weight_representation) +
        " weight_payload_bytes=" + std::to_string(signature.weight_payload_bytes) +
        " logical_macs=" + std::to_string(signature.logical_macs) +
        " logical_weight_bytes=" + std::to_string(signature.count * signature.weight_payload_bytes) + "\n";
}

uint64_t merge_key(uint32_t left, uint32_t right) {
    return (static_cast<uint64_t>(left) << 32) | right;
}

std::string utf8_codepoint(uint32_t codepoint) {
    std::string result;
    if (codepoint <= 0x7f) {
        result.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        result.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        result.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        result.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        result.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
    return result;
}

uint32_t read_codepoint(const std::string & value, size_t * position) {
    const uint8_t first = static_cast<uint8_t>(value[*position]);
    if (first < 0x80) {
        ++*position;
        return first;
    }
    if ((first & 0xe0) == 0xc0 && *position + 1 < value.size()) {
        const uint32_t result = ((first & 0x1f) << 6) |
            (static_cast<uint8_t>(value[*position + 1]) & 0x3f);
        *position += 2;
        return result;
    }
    if ((first & 0xf0) == 0xe0 && *position + 2 < value.size()) {
        const uint32_t result = ((first & 0x0f) << 12) |
            ((static_cast<uint8_t>(value[*position + 1]) & 0x3f) << 6) |
            (static_cast<uint8_t>(value[*position + 2]) & 0x3f);
        *position += 3;
        return result;
    }
    if ((first & 0xf8) == 0xf0 && *position + 3 < value.size()) {
        const uint32_t result = ((first & 0x07) << 18) |
            ((static_cast<uint8_t>(value[*position + 1]) & 0x3f) << 12) |
            ((static_cast<uint8_t>(value[*position + 2]) & 0x3f) << 6) |
            (static_cast<uint8_t>(value[*position + 3]) & 0x3f);
        *position += 4;
        return result;
    }
    ++*position;
    return first;
}

class ByteBpeTokenizer final {
public:
    explicit ByteBpeTokenizer(const VbufMlConsumerHandle * handle) {
        const VbufMlTokenView * views = nullptr;
        uint64_t count = 0;
        if (handle == nullptr || vbuf_ml_consumer_token_views(handle, &views, &count) != 0 ||
            views == nullptr || count > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("tokenizer metadata unavailable");
        }
        tokens_.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            tokens_.emplace_back(views[i].text, static_cast<size_t>(views[i].text_len));
            token_ids_[tokens_.back()] = static_cast<uint32_t>(i);
        }
        uint64_t merge_count = 0;
        if (vbuf_ml_consumer_merge_count(handle, &merge_count) != 0) {
            throw std::runtime_error("tokenizer merge metadata unavailable");
        }
        for (uint64_t i = 0; i < merge_count; ++i) {
            uint64_t left = 0, right = 0;
            if (vbuf_ml_consumer_merge_pair(handle, i, &left, &right) != 0) {
                throw std::runtime_error("tokenizer merge metadata invalid");
            }
            if (left > std::numeric_limits<uint32_t>::max() ||
                right > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("tokenizer merge ID exceeds ABI width");
            }
            merge_ranks_[merge_key(static_cast<uint32_t>(left), static_cast<uint32_t>(right))] =
                static_cast<uint32_t>(i / 2);
        }
        bool add_bos = false;
        if (vbuf_ml_consumer_add_bos(handle, &add_bos) == 0) add_bos_ = add_bos;
        uint64_t bos = 0;
        if (vbuf_ml_consumer_special_token(handle, 0, &bos) == 0 &&
            bos <= std::numeric_limits<uint32_t>::max()) {
            bos_ = static_cast<uint32_t>(bos);
        }
        uint64_t eos = 0;
        if (vbuf_ml_consumer_special_token(handle, 1, &eos) == 0 &&
            eos <= std::numeric_limits<uint32_t>::max()) {
            eos_ = static_cast<uint32_t>(eos);
        }
        initialize_byte_map();
    }

    std::vector<uint32_t> encode(const std::string & text) const {
        std::vector<uint32_t> result;
        if (add_bos_ && bos_.has_value()) result.push_back(*bos_);
        std::vector<std::string> symbols;
        symbols.reserve(text.size());
        for (unsigned char byte : text) symbols.push_back(byte_symbols_[byte]);
        while (symbols.size() > 1) {
            size_t best = symbols.size();
            uint32_t best_rank = std::numeric_limits<uint32_t>::max();
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto left = token_ids_.find(symbols[i]);
                const auto right = token_ids_.find(symbols[i + 1]);
                if (left == token_ids_.end() || right == token_ids_.end()) continue;
                const auto rank = merge_ranks_.find(merge_key(left->second, right->second));
                if (rank != merge_ranks_.end() && rank->second < best_rank) {
                    best = i;
                    best_rank = rank->second;
                }
            }
            if (best == symbols.size()) break;
            symbols[best] += symbols[best + 1];
            symbols.erase(symbols.begin() + static_cast<ptrdiff_t>(best + 1));
        }
        for (const std::string & symbol : symbols) {
            const auto found = token_ids_.find(symbol);
            if (found == token_ids_.end()) throw std::runtime_error("prompt has no tokenizer coverage");
            result.push_back(found->second);
        }
        return result;
    }

    std::string decode(uint32_t token) const {
        if (token >= tokens_.size()) return {};
        const std::string & piece = tokens_[token];
        std::string result;
        for (size_t position = 0; position < piece.size();) {
            const uint32_t codepoint = read_codepoint(piece, &position);
            const auto found = unicode_to_byte_.find(codepoint);
            if (found == unicode_to_byte_.end()) {
                result += utf8_codepoint(codepoint);
            } else {
                result.push_back(static_cast<char>(found->second));
            }
        }
        return result;
    }

    bool is_eos(uint32_t token) const {
        return eos_.has_value() && token == *eos_;
    }

private:
    void initialize_byte_map() {
        std::array<uint32_t, 256> byte_to_unicode{};
        std::array<bool, 256> direct{};
        for (uint32_t byte = 33; byte <= 126; ++byte) direct[byte] = true;
        for (uint32_t byte = 161; byte <= 172; ++byte) direct[byte] = true;
        for (uint32_t byte = 174; byte <= 255; ++byte) direct[byte] = true;
        for (uint32_t byte = 0; byte <= 255; ++byte) {
            if (direct[byte]) byte_to_unicode[byte] = byte;
        }
        uint32_t extra = 256;
        for (uint32_t byte = 0; byte <= 255; ++byte) {
            if (!direct[byte]) {
                byte_to_unicode[byte] = extra;
                unicode_to_byte_[extra++] = static_cast<uint8_t>(byte);
            }
        }
        for (uint32_t byte = 0; byte <= 255; ++byte) {
            byte_symbols_[byte] = utf8_codepoint(byte_to_unicode[byte]);
        }
    }

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, uint32_t> token_ids_;
    std::unordered_map<uint64_t, uint32_t> merge_ranks_;
    std::array<std::string, 256> byte_symbols_{};
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte_;
    std::optional<uint32_t> bos_;
    std::optional<uint32_t> eos_;
    bool add_bos_ = false;
};

struct StepResult {
    uint32_t next = 0;
    uint64_t elapsed_ns = 0;
};

struct DecodeTokenMetrics {
    uint64_t elapsed_ms = 0;
    uint64_t source_requests = 0;
    uint64_t source_bytes = 0;
    uint64_t residency_hits = 0;
    uint64_t residency_misses = 0;
    uint64_t evictions = 0;
    uint64_t reload_bytes = 0;
    uint64_t materializations = 0;
    uint64_t reacquisitions = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t actual_compute_ms = 0;
    RuntimeTiming attribution;
};

class DirectSession final {
public:
    DirectSession(const std::string & metadata_path, const std::string & endpoint,
        RuntimeMode mode = ANDROID_RUNTIME_MODE)
        : mode_(mode) {
        const uint64_t start = android_now_ns();
        load_metadata(metadata_path, &metadata_);
        VbufMlSourceIdentityInfo source_identity{};
        if (vbuf_ml_consumer_source_identity(metadata_.handle, 1, &source_identity) != 0) {
            throw std::runtime_error("qualified source identity unavailable");
        }
        tokenizer_ = std::make_unique<ByteBpeTokenizer>(metadata_.handle);
        for (uint32_t block = 0; block < MODEL_BLOCKS; ++block) {
            plans_.push_back(make_plan(metadata_, block, (block + 1) * 10000));
        }
        lease_ = model_lease(metadata_.handle);
        auto remote = std::make_shared<HttpRangeSource>(endpoint);
        ProgressiveSourceIdentity persistent_identity;
        persistent_identity.declared_size = source_identity.declared_size;
        persistent_identity.hash_algorithm = source_identity.hash_algorithm;
        std::copy(std::begin(source_identity.full_source_hash),
            std::end(source_identity.full_source_hash),
            persistent_identity.full_source_hash.begin());
        source_ = std::make_shared<ProgressiveRangeSource>(std::move(remote),
            metadata_path + ".payload", persistent_identity);
        residency_ = std::make_shared<TensorResidencyStore>(RESIDENCY_BUDGET,
            ResidencyReplacementPolicyKind::CostAware);
        backing_ = std::make_shared<LocalVbufRangeMaterializer>(source_);
        materializer_ = std::make_shared<ResidentTensorMaterializer>(backing_, residency_);
        embedding_ = lookup(metadata_, "token_embd.weight");
        output_norm_ = lookup(metadata_, "output_norm.weight");
        output_ = lookup(metadata_, "output.weight");
        reset_state();
        open_ms_ = (android_now_ns() - start) / 1000000;
    }

    void cancel() { cancelled_.store(true, std::memory_order_relaxed); }

    std::string generate(const std::string & prompt, uint32_t max_tokens) {
        if (prompt.empty()) throw std::runtime_error("prompt is empty");
        if (max_tokens == 0 || max_tokens > 64) throw std::runtime_error("invalid token limit");
        const uint64_t generation_start = android_now_ns();
        cancelled_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard lock(progress_mutex_);
            progress_text_.clear();
            progress_tokens_ = 0;
            progress_prompt_tokens_ = 0;
            progress_phase_ = "TOKENIZING";
            progress_active_ = true;
        }
        reset_state();
        loaded_residency_ids_.clear();
        prompt_char_count_ = prompt.size();
        decode_token_metrics_.clear();
        first_decode_token_ms_ = 0;
        decode_ms_total_ = 0;
        generated_text_chars_ = 0;
        ttft_ms_ = 0;
        decode_attribution_ = {};
        const std::vector<uint32_t> prompt_tokens = tokenizer_->encode(prompt);
        if (prompt_tokens.empty()) throw std::runtime_error("prompt produced no tokens");
        if (prompt_tokens.size() + max_tokens > MAX_CONTEXT)
            throw std::runtime_error("prompt exceeds bounded context");
        const vbuf_ggml::PromptBatch prompt_batch{ prompt_tokens, 0 };
        prompt_batch.validate();
        prompt_token_count_ = prompt_tokens.size();
        {
            std::lock_guard lock(progress_mutex_);
            progress_prompt_tokens_ = prompt_token_count_;
            progress_phase_ = "PREFILL";
        }
        std::fprintf(stderr, "BASELINE_PROMPT mode=%s token_count=%llu\n",
            mode_ == RuntimeMode::Qualification ? "QUALIFICATION" : "NORMAL_INFERENCE",
            static_cast<unsigned long long>(prompt_token_count_));
        RuntimeTiming timing;
        prefill_ms_ = 0;
        last_position_ms_ = 0;
        uint32_t input = prompt_tokens.front();
        uint32_t position = 0;
        const bool batched_prefill = mode_ == RuntimeMode::NormalInference;
        prefill_batch_size_ = batched_prefill ? prompt_batch.size() : 1;
        prefill_mode_ = batched_prefill ? "BATCHED" : "SERIAL";
        if (batched_prefill) {
            const size_t trace_begin = residency_->trace().size();
            const uint64_t prefill_start = android_now_ns();
            std::vector<Activation> embeddings;
            embeddings.reserve(prompt_batch.size());
            for (uint32_t prompt_token : prompt_batch.token_ids) {
                if (cancelled_.load(std::memory_order_relaxed)) throw std::runtime_error("cancelled");
                embeddings.push_back(run_embedding(embedding_, prompt_token, lease_, materializer_,
                    "android_prefill_embedding", mode_, &timing));
            }
            const uint64_t sequence_start = android_now_ns();
            const std::vector<Activation> sequence = run_sequence_batched(plans_, embeddings,
                &actual_k_, &actual_v_, lease_, materializer_, residency_, &timing);
            prefill_layer_sequence_ms_ = (android_now_ns() - sequence_start) / 1000000;
            if (sequence.empty()) throw std::runtime_error("batched prefill produced no output");
            const std::vector<float> logits = run_output_head(output_norm_, output_, sequence.back(),
                lease_, materializer_, "android_logits", 900000, mode_, &timing);
            input = greedy(logits);
            position = static_cast<uint32_t>(prompt_tokens.size());
            prefill_ms_ = (android_now_ns() - prefill_start) / 1000000;
            last_position_ms_ = prefill_ms_;
            const TraceDelta delta = trace_delta(plans_, residency_->trace(), trace_begin,
                residency_->trace().size(), &loaded_residency_ids_);
            reload_bytes_ += delta.reload_bytes;
            peak_resident_bytes_ = std::max(peak_resident_bytes_, residency_->resident_bytes());
            std::fprintf(stderr, "PREFILL mode=BATCHED prompt_tokens=%zu batch_size=%zu total_ms=%llu\n",
                prompt_batch.size(), prompt_batch.size(), static_cast<unsigned long long>(prefill_ms_));
            log_position_metrics(position - 1, "prefill_batch");
        } else {
            for (uint32_t prompt_token : prompt_tokens) {
                if (cancelled_.load(std::memory_order_relaxed)) throw std::runtime_error("cancelled");
                input = prompt_token;
                const uint64_t position_start = android_now_ns();
                const StepResult step = run_step(input, position++, &timing);
                last_position_ms_ = (android_now_ns() - position_start) / 1000000;
                prefill_ms_ += last_position_ms_;
                input = step.next;
            }
            prefill_layer_sequence_ms_ = timing.actual_attention_ns / 1000000 + timing.actual_ffn_ns / 1000000;
            std::fprintf(stderr, "PREFILL mode=SERIAL prompt_tokens=%zu batch_size=1 total_ms=%llu\n",
                prompt_tokens.size(), static_cast<unsigned long long>(prefill_ms_));
        }
        {
            std::lock_guard lock(progress_mutex_);
            progress_phase_ = "DECODE";
        }
        const uint64_t first_token_start = android_now_ns();
        std::string output;
        uint32_t generated = 0;
        for (; generated < max_tokens; ++generated) {
            if (cancelled_.load(std::memory_order_relaxed)) throw std::runtime_error("cancelled");
            const uint64_t position_start = android_now_ns();
            const RuntimeTiming timing_before = timing;
            const DecodeTokenMetrics counters_before = snapshot_counters();
            const StepResult step = run_step(input, position++, &timing);
            last_position_ms_ = (android_now_ns() - position_start) / 1000000;
            const DecodeTokenMetrics counters_after = snapshot_counters();
            DecodeTokenMetrics token_metrics;
            token_metrics.elapsed_ms = last_position_ms_;
            token_metrics.source_requests = counters_after.source_requests - counters_before.source_requests;
            token_metrics.source_bytes = counters_after.source_bytes - counters_before.source_bytes;
            token_metrics.residency_hits = counters_after.residency_hits - counters_before.residency_hits;
            token_metrics.residency_misses = counters_after.residency_misses - counters_before.residency_misses;
            token_metrics.evictions = counters_after.evictions - counters_before.evictions;
            token_metrics.reload_bytes = counters_after.reload_bytes - counters_before.reload_bytes;
            token_metrics.materializations = counters_after.materializations - counters_before.materializations;
            token_metrics.reacquisitions = counters_after.reacquisitions - counters_before.reacquisitions;
            token_metrics.peak_resident_bytes = counters_after.peak_resident_bytes;
            token_metrics.actual_compute_ms =
                (timing.actual_attention_ns - timing_before.actual_attention_ns +
                    timing.actual_ffn_ns - timing_before.actual_ffn_ns +
                    timing.actual_output_head_ns - timing_before.actual_output_head_ns) / 1000000;
            token_metrics.attribution = timing.delta(timing_before);
            decode_attribution_.accumulate(token_metrics.attribution);
            decode_token_metrics_.push_back(token_metrics);
            if (generated == 0) {
                first_decode_token_ms_ = last_position_ms_;
                ttft_ms_ = (android_now_ns() - first_token_start) / 1000000;
            }
            __android_log_print(ANDROID_LOG_INFO, TAG,
                "BASELINE mode=%s position=%u kind=generated total_ms=%llu actual_attention_ms=%llu "
                "reference_attention_ms=%s actual_ffn_ms=%llu reference_ffn_ms=%s output_head_ms=%llu "
                "reference_output_head_ms=%s",
                mode_ == RuntimeMode::Qualification ? "QUALIFICATION" : "NORMAL_INFERENCE",
                position - 1, static_cast<unsigned long long>(last_position_ms_),
                static_cast<unsigned long long>((timing.actual_attention_ns - timing_before.actual_attention_ns) / 1000000),
                mode_ == RuntimeMode::Qualification
                    ? std::to_string((timing.reference_attention_ns - timing_before.reference_attention_ns) / 1000000).c_str()
                    : "NOT_EXECUTED",
                static_cast<unsigned long long>((timing.actual_ffn_ns - timing_before.actual_ffn_ns) / 1000000),
                mode_ == RuntimeMode::Qualification
                    ? std::to_string((timing.reference_ffn_ns - timing_before.reference_ffn_ns) / 1000000).c_str()
                    : "NOT_EXECUTED",
                static_cast<unsigned long long>((timing.actual_output_head_ns - timing_before.actual_output_head_ns) / 1000000),
                mode_ == RuntimeMode::Qualification
                    ? std::to_string((timing.reference_output_head_ns - timing_before.reference_output_head_ns) / 1000000).c_str()
                    : "NOT_EXECUTED");
            std::fprintf(stderr, "BASELINE mode=%s position=%u kind=generated total_ms=%llu\n",
                mode_ == RuntimeMode::Qualification ? "QUALIFICATION" : "NORMAL_INFERENCE",
                position - 1, static_cast<unsigned long long>(last_position_ms_));
            log_position_metrics(position - 1, "generated");
            output += tokenizer_->decode(step.next);
            input = step.next;
            {
                std::lock_guard lock(progress_mutex_);
                progress_text_ = output;
                progress_tokens_ = generated + 1;
            }
            if (tokenizer_->is_eos(input)) break;
        }
        decode_ms_total_ = (android_now_ns() - first_token_start) / 1000000;
        generation_ms_ = (android_now_ns() - generation_start) / 1000000;
        timing_ = timing;
        generated_tokens_ = generated;
        generated_text_chars_ = output.size();
        {
            std::lock_guard lock(progress_mutex_);
            progress_phase_ = "COMPLETE";
            progress_active_ = false;
        }
        return output.empty() ? "<empty>" : output;
    }

    std::string progress() const {
        std::lock_guard lock(progress_mutex_);
        return std::string(progress_active_ ? "ACTIVE" : "IDLE") +
            " phase=" + progress_phase_ +
            " prompt_tokens=" + std::to_string(progress_prompt_tokens_) +
            " generated_tokens=" + std::to_string(progress_tokens_) +
            "\n" + progress_text_;
    }

    void finish_progress(const char * phase = "COMPLETE") {
        std::lock_guard lock(progress_mutex_);
        progress_phase_ = phase;
        progress_active_ = false;
    }

    void log_position_metrics(uint32_t position, const char * kind) const {
        const RangeSourceMetrics transport = source_->metrics();
        const ProgressiveRangeMetrics persistence = source_->progressive_metrics();
        uint64_t hits = 0, misses = 0, evictions = 0;
        for (const auto & event : residency_->trace()) {
            if (event.kind == ResidencyEventKind::Hit) ++hits;
            if (event.kind == ResidencyEventKind::Miss) ++misses;
            if (event.kind == ResidencyEventKind::Evict) ++evictions;
        }
        std::fprintf(stderr, "BASELINE_METRICS mode=%s position=%u kind=%s "
            "consumer_requests=%llu consumer_requested_bytes=%llu "
            "upstream_remote_requests=%llu upstream_remote_bytes=%llu "
            "acquisition_windows=%llu acquisition_window_bytes=%llu "
            "local_source_bytes=%llu local_chunk_hits=%llu remote_chunk_misses=%llu "
            "chunks_covered=%llu hits=%llu misses=%llu evictions=%llu "
            "reload_bytes=%llu materializations=%llu reacquisitions=%llu peak_resident=%llu\n",
            mode_ == RuntimeMode::Qualification ? "QUALIFICATION" : "NORMAL_INFERENCE",
            position, kind,
            static_cast<unsigned long long>(persistence.consumer_requests),
            static_cast<unsigned long long>(persistence.consumer_requested_bytes),
            static_cast<unsigned long long>(persistence.upstream_remote_requests),
            static_cast<unsigned long long>(persistence.upstream_remote_bytes),
            static_cast<unsigned long long>(persistence.acquisition_windows),
            static_cast<unsigned long long>(persistence.acquisition_window_bytes),
            static_cast<unsigned long long>(persistence.local_source_bytes),
            static_cast<unsigned long long>(persistence.local_chunk_hits),
            static_cast<unsigned long long>(persistence.remote_chunk_misses),
            static_cast<unsigned long long>(persistence.covered_chunks),
            static_cast<unsigned long long>(hits),
            static_cast<unsigned long long>(misses), static_cast<unsigned long long>(evictions),
            static_cast<unsigned long long>(reload_bytes_),
            static_cast<unsigned long long>(residency_->materialization_count()),
            static_cast<unsigned long long>(residency_->reacquisition_count()),
            static_cast<unsigned long long>(peak_resident_bytes_));
    }

    std::string metrics() const {
        const RangeSourceMetrics transport = source_->metrics();
        const ProgressiveRangeMetrics persistence = source_->progressive_metrics();
        uint64_t hits = 0, misses = 0, evictions = 0;
        for (const auto & event : residency_->trace()) {
            if (event.kind == ResidencyEventKind::Hit) ++hits;
            if (event.kind == ResidencyEventKind::Miss) ++misses;
            if (event.kind == ResidencyEventKind::Evict) ++evictions;
        }
        const double tokens_per_second = generation_ms_ == 0 ? 0.0 :
            1000.0 * static_cast<double>(generated_tokens_) / generation_ms_;
        const RuntimeTiming & attribution = decode_attribution_;
        const auto stage_build = [&](AttributionStage stage) {
            return attribution.stage_graph_build_ns[attribution_stage_index(stage)];
        };
        const auto stage_allocation = [&](AttributionStage stage) {
            return attribution.stage_graph_allocation_ns[attribution_stage_index(stage)];
        };
        const auto stage_result = [&](AttributionStage stage) {
            return attribution.stage_result_handling_ns[attribution_stage_index(stage)];
        };
        const auto stage_ready = [&](AttributionStage stage) {
            switch (stage) {
            case AttributionStage::Attention: return attribution.attention_ready_ns;
            case AttributionStage::Router: return attribution.router_ready_ns;
            case AttributionStage::DenseFfn: return attribution.dense_ffn_ready_ns;
            case AttributionStage::RoutedExpert: return attribution.routed_expert_ready_ns;
            case AttributionStage::SharedExpert: return attribution.shared_expert_ready_ns;
            case AttributionStage::OutputHead: return attribution.output_head_ready_ns;
            case AttributionStage::Embedding: return attribution.embedding_ready_ns;
            default: return uint64_t{0};
            }
        };
        const auto nonnegative_difference = [](uint64_t total, uint64_t parts) {
            return total > parts ? total - parts : uint64_t{0};
        };
        const uint64_t attention_measured = stage_ready(AttributionStage::Attention) +
            stage_build(AttributionStage::Attention) + stage_allocation(AttributionStage::Attention) +
            attribution.attention_backend_compute_ns + stage_result(AttributionStage::Attention);
        const uint64_t router_measured = stage_ready(AttributionStage::Router) +
            stage_build(AttributionStage::Router) + stage_allocation(AttributionStage::Router) +
            attribution.router_backend_compute_ns + attribution.router_selection_ns +
            stage_result(AttributionStage::Router);
        const uint64_t routed_measured = stage_ready(AttributionStage::RoutedExpert) +
            stage_build(AttributionStage::RoutedExpert) + stage_allocation(AttributionStage::RoutedExpert) +
            attribution.routed_expert_gate_up_compute_ns + attribution.routed_expert_activation_ns +
            attribution.routed_expert_down_compute_ns + attribution.routed_expert_other_compute_ns +
            attribution.routed_expert_accumulation_ns + stage_result(AttributionStage::RoutedExpert);
        const uint64_t shared_measured = stage_ready(AttributionStage::SharedExpert) +
            stage_build(AttributionStage::SharedExpert) + stage_allocation(AttributionStage::SharedExpert) +
            attribution.shared_expert_backend_compute_ns + stage_result(AttributionStage::SharedExpert);
        const uint64_t dense_measured = stage_ready(AttributionStage::DenseFfn) +
            stage_build(AttributionStage::DenseFfn) + stage_allocation(AttributionStage::DenseFfn) +
            attribution.dense_ffn_backend_compute_ns + stage_result(AttributionStage::DenseFfn);
        const uint64_t output_measured = stage_ready(AttributionStage::OutputHead) +
            stage_build(AttributionStage::OutputHead) + stage_allocation(AttributionStage::OutputHead) +
            attribution.output_head_backend_compute_ns + stage_result(AttributionStage::OutputHead);
        const uint64_t ffn_measured = router_measured + routed_measured + shared_measured + dense_measured;
        const uint64_t ffn_other = nonnegative_difference(attribution.actual_ffn_ns, ffn_measured);
        const uint64_t top_level_accounted = attribution.actual_embedding_ns +
            attribution.actual_attention_ns + attribution.actual_ffn_ns +
            attribution.actual_output_head_ns;
        const uint64_t decode_wall_ns = decode_ms_total_ * 1000000ULL;
        const uint64_t unaccounted = nonnegative_difference(decode_wall_ns, top_level_accounted);
        const double accounted_percent = decode_wall_ns == 0 ? 0.0 :
            100.0 * static_cast<double>(decode_wall_ns - unaccounted) / decode_wall_ns;
        size_t unique_routed_experts = 0;
        for (uint64_t count : attribution.routed_expert_id_counts)
            if (count != 0) ++unique_routed_experts;
        const auto sample_min_ns = [](const auto & samples, uint32_t count) {
            uint64_t value = 0;
            for (uint32_t index = 0; index < count; ++index)
                value = value == 0 ? samples[index] : std::min(value, samples[index]);
            return value;
        };
        const auto sample_max_ns = [](const auto & samples, uint32_t count) {
            uint64_t value = 0;
            for (uint32_t index = 0; index < count; ++index)
                value = std::max(value, samples[index]);
            return value;
        };
        const uint64_t gate_up_compute_min_ns = sample_min_ns(
            attribution.gate_up_compute_samples, attribution.gate_up_compute_sample_count);
        const uint64_t gate_up_compute_max_ns = sample_max_ns(
            attribution.gate_up_compute_samples, attribution.gate_up_compute_sample_count);
        const uint64_t down_compute_min_ns = sample_min_ns(
            attribution.down_compute_samples, attribution.down_compute_sample_count);
        const uint64_t down_compute_max_ns = sample_max_ns(
            attribution.down_compute_samples, attribution.down_compute_sample_count);
        const uint64_t gate_up_phase_total_ns = attribution.gate_up_ready_ns +
            attribution.gate_up_descriptor_setup_ns + attribution.gate_up_graph_build_ns +
            attribution.gate_up_backend_allocation_ns + attribution.routed_expert_gate_up_compute_ns +
            attribution.gate_up_result_handling_ns;
        const uint64_t down_phase_total_ns = attribution.down_ready_ns +
            attribution.down_descriptor_setup_ns + attribution.down_graph_build_ns +
            attribution.down_backend_allocation_ns + attribution.routed_expert_down_compute_ns +
            attribution.down_result_handling_ns;
        std::string signature_metrics;
        for (size_t index = 0; index < attribution.gate_up_signatures.size(); ++index) {
            if (attribution.gate_up_signatures[index].count != 0)
                signature_metrics += attribution_signature_line("gate/up", index,
                    attribution.gate_up_signatures[index]);
        }
        for (size_t index = 0; index < attribution.down_signatures.size(); ++index) {
            if (attribution.down_signatures[index].count != 0)
                signature_metrics += attribution_signature_line("down", index,
                    attribution.down_signatures[index]);
        }
        const std::string operation_metrics =
            "Routed MoE layers: " + std::to_string(attribution.routed_moe_layer_count) + "\n" +
            "Routed expert invocations: " + std::to_string(attribution.routed_expert_invocation_count) + "\n" +
            "Routed expert unique IDs: " + std::to_string(unique_routed_experts) + "\n" +
            "Gate/up logical invocations: " + std::to_string(attribution.gate_up_logical_invocations) + "\n" +
            "Gate/up backend submissions: " + std::to_string(attribution.gate_up_backend_submissions) + "\n" +
            "Gate matmul calls: " + std::to_string(attribution.gate_matmul_calls) + "\n" +
            "Up matmul calls: " + std::to_string(attribution.up_matmul_calls) + "\n" +
            "Gate matmul compute: " + std::to_string(attribution.gate_matmul_compute_ns / 1000000) + " ms\n" +
            "Up matmul compute: " + std::to_string(attribution.up_matmul_compute_ns / 1000000) + " ms\n" +
            "Down logical invocations: " + std::to_string(attribution.down_logical_invocations) + "\n" +
            "Down backend submissions: " + std::to_string(attribution.down_backend_submissions) + "\n" +
            "Indexed expert layers: " + std::to_string(timing_.indexed_expert_layer_count) + "\n" +
            "Indexed expert bank submissions: " + std::to_string(timing_.indexed_expert_bank_submissions) + "\n" +
            "Indexed expert logical slots: " + std::to_string(timing_.indexed_expert_logical_slots) + "\n" +
            "Indexed expert compute: " + std::to_string(timing_.indexed_expert_compute_ns / 1000000) + " ms\n" +
            "Indexed expert materialized bytes: " + std::to_string(timing_.indexed_expert_materialized_bytes) + "\n" +
            "Indexed expert repack bytes: " + std::to_string(timing_.indexed_expert_repack_bytes) + "\n" +
            "Gate/up phase total: " + std::to_string(gate_up_phase_total_ns / 1000000) + " ms\n" +
            "Gate/up ready: " + std::to_string(attribution.gate_up_ready_ns / 1000000) + " ms\n" +
            "Gate/up descriptor setup: " + std::to_string(attribution.gate_up_descriptor_setup_ns / 1000000) + " ms\n" +
            "Gate/up graph build: " + std::to_string(attribution.gate_up_graph_build_ns / 1000000) + " ms\n" +
            "Gate/up backend allocation: " + std::to_string(attribution.gate_up_backend_allocation_ns / 1000000) + " ms\n" +
            "Gate/up compute: " + std::to_string(attribution.routed_expert_gate_up_compute_ns / 1000000) + " ms\n" +
            "Gate/up compute samples: " + std::to_string(attribution.gate_up_compute_sample_count) + "\n" +
            "Gate/up compute avg: " + std::to_string(attribution.gate_up_compute_sample_count == 0 ? 0 :
                attribution.routed_expert_gate_up_compute_ns / attribution.gate_up_compute_sample_count / 1000000) + " ms\n" +
            "Gate/up compute min: " + std::to_string(gate_up_compute_min_ns / 1000000) + " ms\n" +
            "Gate/up compute max: " + std::to_string(gate_up_compute_max_ns / 1000000) + " ms\n" +
            "Gate/up result handling: " + std::to_string(attribution.gate_up_result_handling_ns / 1000000) + " ms\n" +
            "Down phase total: " + std::to_string(down_phase_total_ns / 1000000) + " ms\n" +
            "Down ready: " + std::to_string(attribution.down_ready_ns / 1000000) + " ms\n" +
            "Down descriptor setup: " + std::to_string(attribution.down_descriptor_setup_ns / 1000000) + " ms\n" +
            "Down graph build: " + std::to_string(attribution.down_graph_build_ns / 1000000) + " ms\n" +
            "Down backend allocation: " + std::to_string(attribution.down_backend_allocation_ns / 1000000) + " ms\n" +
            "Down compute: " + std::to_string(attribution.routed_expert_down_compute_ns / 1000000) + " ms\n" +
            "Down compute samples: " + std::to_string(attribution.down_compute_sample_count) + "\n" +
            "Down compute avg: " + std::to_string(attribution.down_compute_sample_count == 0 ? 0 :
                attribution.routed_expert_down_compute_ns / attribution.down_compute_sample_count / 1000000) + " ms\n" +
            "Down compute min: " + std::to_string(down_compute_min_ns / 1000000) + " ms\n" +
            "Down compute max: " + std::to_string(down_compute_max_ns / 1000000) + " ms\n" +
            "Down result handling: " + std::to_string(attribution.down_result_handling_ns / 1000000) + " ms\n" +
            "Descriptor setup total: " + std::to_string(attribution.descriptor_setup_ns / 1000000) + " ms\n";
        return "Model: DeepSeek-V2-Lite IQ2_XXS\n"
            "Residency budget bytes: " + std::to_string(RESIDENCY_BUDGET) + "\n"
            "Runtime mode: " + std::string(mode_ == RuntimeMode::Qualification
                ? "QUALIFICATION" : "NORMAL_INFERENCE") + "\n"
            "Prefill mode: " + prefill_mode_ + "\n"
            "Prefill batch size: " + std::to_string(prefill_batch_size_) + "\n"
            "Prompt chars: " + std::to_string(prompt_char_count_) + "\n"
            "Prompt tokens: " + std::to_string(prompt_token_count_) + "\n"
            "Model open: " + std::to_string(open_ms_) + " ms\n"
            "Prefill: " + std::to_string(prefill_ms_) + " ms\n"
            "Generated tokens: " + std::to_string(generated_tokens_) + "\n"
            "Generated text chars: " + std::to_string(generated_text_chars_) + "\n"
            "First decode token: " + std::to_string(first_decode_token_ms_) + " ms\n"
            "TTFT: " + std::to_string(ttft_ms_) + " ms\n"
            "Decode token 1: " + decode_token_line(0) + "\n"
            "Decode token 2: " + decode_token_line(1) + "\n"
            "Decode token 3: " + decode_token_line(2) + "\n"
            "Decode token 4: " + decode_token_line(3) + "\n"
            "Decode total: " + std::to_string(decode_ms_total_) + " ms\n"
            "Total generation: " + std::to_string(generation_ms_) + " ms\n"
            "Prefill layer sequence: " + std::to_string(prefill_layer_sequence_ms_) + " ms\n"
            "Last position: " + std::to_string(last_position_ms_) + " ms\n"
            "Actual attention: " + std::to_string(timing_.actual_attention_ns / 1000000) + " ms\n"
            "Reference attention: " + (mode_ == RuntimeMode::Qualification
                ? std::to_string(timing_.reference_attention_ns / 1000000) : "NOT_EXECUTED") + "\n"
            "Actual FFN: " + std::to_string(timing_.actual_ffn_ns / 1000000) + " ms\n"
            "Actual router/MoE: " + std::to_string(timing_.actual_router_moe_ns / 1000000) + " ms\n"
            "Reference FFN: " + (mode_ == RuntimeMode::Qualification
                ? std::to_string(timing_.reference_ffn_ns / 1000000) : "NOT_EXECUTED") + "\n"
            "Actual output head: " + std::to_string(timing_.actual_output_head_ns / 1000000) + " ms\n"
            "Reference output head: " + (mode_ == RuntimeMode::Qualification
                ? std::to_string(timing_.reference_output_head_ns / 1000000) : "NOT_EXECUTED") + "\n"
            "Materialization wait: " + std::to_string(attribution.attribution_ready_ns / 1000000) + " ms\n"
            "Attribution decode wall: " + std::to_string(decode_wall_ns / 1000000) + " ms\n"
            "Attribution accounted: " + std::to_string((decode_wall_ns - unaccounted) / 1000000) + " ms\n"
            "Attribution unaccounted: " + std::to_string(unaccounted / 1000000) + " ms\n"
            "Attribution accounted percent: " + std::to_string(accounted_percent) + "\n"
            "Attribution embedding total: " + std::to_string(attribution.actual_embedding_ns / 1000000) + " ms\n"
            "Attribution attention total: " + std::to_string(attribution.actual_attention_ns / 1000000) + " ms\n"
            "Attribution attention ready: " + std::to_string(attribution.attention_ready_ns / 1000000) + " ms\n"
            "Attribution attention compute: " + std::to_string(attribution.attention_backend_compute_ns / 1000000) + " ms\n"
            "Attribution FFN total: " + std::to_string(attribution.actual_ffn_ns / 1000000) + " ms\n"
            "Attribution router total: " + std::to_string(router_measured / 1000000) + " ms\n"
            "Attribution router ready: " + std::to_string(attribution.router_ready_ns / 1000000) + " ms\n"
            "Attribution router compute: " + std::to_string(attribution.router_backend_compute_ns / 1000000) + " ms\n"
            "Attribution router selection: " + std::to_string(attribution.router_selection_ns / 1000000) + " ms\n"
            "Attribution routed expert total: " + std::to_string(routed_measured / 1000000) + " ms\n"
            "Attribution routed expert ready: " + std::to_string(attribution.routed_expert_ready_ns / 1000000) + " ms\n"
            "Attribution routed gate up: " + std::to_string(attribution.routed_expert_gate_up_compute_ns / 1000000) + " ms\n"
            "Attribution routed activation: " + std::to_string(attribution.routed_expert_activation_ns / 1000000) + " ms\n"
            "Attribution routed down: " + std::to_string(attribution.routed_expert_down_compute_ns / 1000000) + " ms\n"
            "Attribution routed accumulation: " + std::to_string(attribution.routed_expert_accumulation_ns / 1000000) + " ms\n"
            "Attribution shared expert total: " + std::to_string(shared_measured / 1000000) + " ms\n"
            "Attribution shared expert ready: " + std::to_string(attribution.shared_expert_ready_ns / 1000000) + " ms\n"
            "Attribution shared expert compute: " + std::to_string(attribution.shared_expert_backend_compute_ns / 1000000) + " ms\n"
            "Attribution dense FFN total: " + std::to_string(dense_measured / 1000000) + " ms\n"
            "Attribution dense FFN ready: " + std::to_string(attribution.dense_ffn_ready_ns / 1000000) + " ms\n"
            "Attribution FFN other: " + std::to_string(ffn_other / 1000000) + " ms\n"
            "Attribution output head total: " + std::to_string(attribution.actual_output_head_ns / 1000000) + " ms\n"
            "Attribution output head ready: " + std::to_string(attribution.output_head_ready_ns / 1000000) + " ms\n"
            "Attribution output head compute: " + std::to_string(attribution.output_head_backend_compute_ns / 1000000) + " ms\n"
            "Attribution graph build: " + std::to_string(attribution.graph_build_ns / 1000000) + " ms\n"
            "Attribution graph allocation: " + std::to_string(attribution.graph_allocation_ns / 1000000) + " ms\n"
            "Attribution backend compute: " + std::to_string(attribution.backend_compute_ns / 1000000) + " ms\n"
            "Attribution result handling: " + std::to_string(attribution.result_handling_ns / 1000000) + " ms\n"
            + operation_metrics + signature_metrics +
            "Tokens/sec: " + std::to_string(tokens_per_second) + "\n"
            "HTTP requests: " + std::to_string(transport.requests) + "\n"
            "Returned bytes: " + std::to_string(transport.bytes) + "\n"
            "Consumer requests: " + std::to_string(persistence.consumer_requests) + "\n"
            "Consumer requested bytes: " + std::to_string(persistence.consumer_requested_bytes) + "\n"
            "Upstream remote requests: " + std::to_string(persistence.upstream_remote_requests) + "\n"
            "Upstream remote bytes: " + std::to_string(persistence.upstream_remote_bytes) + "\n"
            "Acquisition windows: " + std::to_string(persistence.acquisition_windows) + "\n"
            "Acquisition window bytes: " + std::to_string(persistence.acquisition_window_bytes) + "\n"
            "Local source bytes: " + std::to_string(persistence.local_source_bytes) + "\n"
            "Local chunk hits: " + std::to_string(persistence.local_chunk_hits) + "\n"
            "Remote chunk misses: " + std::to_string(persistence.remote_chunk_misses) + "\n"
            "Chunks covered: " + std::to_string(persistence.covered_chunks) + "\n"
            "Coverage bytes: " + std::to_string(persistence.coverage_bytes) + "\n"
            "Payload mirror: " + source_->mirror_path() + "\n"
            "Residency hits/misses/evictions: " + std::to_string(hits) + "/" +
            std::to_string(misses) + "/" + std::to_string(evictions) + "\n"
            "Residency materializations/reacquisitions: " +
                std::to_string(residency_->materialization_count()) + "/" +
                std::to_string(residency_->reacquisition_count()) + "\n"
            "Reload bytes: " + std::to_string(reload_bytes_) + "\n"
            "Peak resident bytes: " + std::to_string(peak_resident_bytes_) + "\n"
            "Peak active bytes: " + std::to_string(peak_active_bytes_) + "\n"
            "Consumer wait: not instrumented";
    }

    void close() {
        cancel();
        if (residency_) residency_->clear();
    }

private:
    DecodeTokenMetrics snapshot_counters() const {
        const RangeSourceMetrics transport = source_->metrics();
        uint64_t hits = 0, misses = 0, evictions = 0;
        for (const auto & event : residency_->trace()) {
            if (event.kind == ResidencyEventKind::Hit) ++hits;
            if (event.kind == ResidencyEventKind::Miss) ++misses;
            if (event.kind == ResidencyEventKind::Evict) ++evictions;
        }
        DecodeTokenMetrics result;
        result.source_requests = transport.requests;
        result.source_bytes = transport.bytes;
        result.residency_hits = hits;
        result.residency_misses = misses;
        result.evictions = evictions;
        result.reload_bytes = reload_bytes_;
        result.materializations = residency_->materialization_count();
        result.reacquisitions = residency_->reacquisition_count();
        result.peak_resident_bytes = peak_resident_bytes_;
        return result;
    }

    std::string decode_token_line(size_t index) const {
        if (index >= decode_token_metrics_.size()) return "UNMEASURED";
        const DecodeTokenMetrics & token = decode_token_metrics_[index];
        const RuntimeTiming & attribution = token.attribution;
        return std::to_string(token.elapsed_ms) + " ms requests=" +
            std::to_string(token.source_requests) + " bytes=" +
            std::to_string(token.source_bytes) + " hits=" +
            std::to_string(token.residency_hits) + " misses=" +
            std::to_string(token.residency_misses) + " evictions=" +
            std::to_string(token.evictions) + " reload_bytes=" +
            std::to_string(token.reload_bytes) + " materializations=" +
            std::to_string(token.materializations) + " reacquisitions=" +
            std::to_string(token.reacquisitions) + " peak_resident=" +
            std::to_string(token.peak_resident_bytes) + " actual_compute=" +
            std::to_string(token.actual_compute_ms) + " ms attribution_attention_ready=" +
            std::to_string(attribution.attention_ready_ns / 1000000) +
            " attribution_attention_compute=" +
            std::to_string(attribution.attention_backend_compute_ns / 1000000) +
            " attribution_router_compute=" +
            std::to_string(attribution.router_backend_compute_ns / 1000000) +
            " attribution_routed_gate_up=" +
            std::to_string(attribution.routed_expert_gate_up_compute_ns / 1000000) +
            " attribution_routed_activation=" +
            std::to_string(attribution.routed_expert_activation_ns / 1000000) +
            " attribution_routed_down=" +
            std::to_string(attribution.routed_expert_down_compute_ns / 1000000) +
            " attribution_routed_accumulation=" +
            std::to_string(attribution.routed_expert_accumulation_ns / 1000000) +
            " attribution_shared_compute=" +
            std::to_string(attribution.shared_expert_backend_compute_ns / 1000000) +
            " attribution_output_compute=" +
            std::to_string(attribution.output_head_backend_compute_ns / 1000000) +
            " attribution_graph_build=" +
            std::to_string(attribution.graph_build_ns / 1000000) +
            " attribution_graph_allocation=" +
            std::to_string(attribution.graph_allocation_ns / 1000000) +
            " attribution_result=" +
            std::to_string(attribution.result_handling_ns / 1000000) +
            " ms gate_up_invocations=" + std::to_string(attribution.gate_up_logical_invocations) +
            " gate_up_submissions=" + std::to_string(attribution.gate_up_backend_submissions) +
            " gate_up_compute=" + std::to_string(attribution.routed_expert_gate_up_compute_ns / 1000000) +
            " ms down_invocations=" + std::to_string(attribution.down_logical_invocations) +
            " down_submissions=" + std::to_string(attribution.down_backend_submissions) +
            " down_compute=" + std::to_string(attribution.routed_expert_down_compute_ns / 1000000) +
            " ms indexed_layers=" + std::to_string(attribution.indexed_expert_layer_count) +
            " indexed_slots=" + std::to_string(attribution.indexed_expert_logical_slots);
    }

    void reset_state() {
        actual_k_.clear();
        actual_v_.clear();
        reference_k_.clear();
        reference_v_.clear();
        for (size_t i = 0; i < plans_.size(); ++i) {
            actual_k_.emplace_back(16 * 192, MAX_CONTEXT);
            actual_v_.emplace_back(16 * 128, MAX_CONTEXT);
            reference_k_.emplace_back(16 * 192, MAX_CONTEXT);
            reference_v_.emplace_back(16 * 128, MAX_CONTEXT);
        }
    }

    StepResult run_step(uint32_t token, uint32_t position, RuntimeTiming * timing) {
        const size_t trace_begin = residency_->trace().size();
        const uint64_t start = android_now_ns();
        const Activation input = run_embedding(embedding_, token, lease_, materializer_,
            "android_embedding", mode_, timing);
        const SequenceRun sequence = run_sequence(plans_, input, position, &actual_k_, &actual_v_,
            &reference_k_, &reference_v_, lease_, materializer_, residency_, source_, "android_generation",
            false, nullptr, 2, mode_, timing);
        if (!sequence.ok) {
            const std::string detail = sequence.failure_detail.empty() ? "unknown failure" :
                sequence.failure_detail;
            throw std::runtime_error("direct runtime execution failed: " + detail);
        }
        const std::vector<float> logits = run_output_head(output_norm_, output_, sequence.output,
            lease_, materializer_, "android_logits", 900000 + position * 100, mode_, timing);
        const uint32_t next = greedy(logits);
        const TraceDelta delta = trace_delta(plans_, residency_->trace(), trace_begin,
            residency_->trace().size(), &loaded_residency_ids_);
        reload_bytes_ += delta.reload_bytes;
        peak_resident_bytes_ = std::max(peak_resident_bytes_, residency_->resident_bytes());
        peak_active_bytes_ = std::max(peak_active_bytes_, sequence.peak_active_persistent);
        return { next, android_now_ns() - start };
    }

    Metadata metadata_;
    std::unique_ptr<ByteBpeTokenizer> tokenizer_;
    std::vector<LayerPlan> plans_;
    std::shared_ptr<const void> lease_;
    std::shared_ptr<ProgressiveRangeSource> source_;
    std::shared_ptr<LocalVbufRangeMaterializer> backing_;
    std::shared_ptr<TensorResidencyStore> residency_;
    std::shared_ptr<ResidentTensorMaterializer> materializer_;
    Meta embedding_;
    Meta output_norm_;
    Meta output_;
    std::vector<RuntimeStateSlot> actual_k_, actual_v_, reference_k_, reference_v_;
    std::set<uint32_t> loaded_residency_ids_;
    std::atomic<bool> cancelled_ = false;
    mutable std::mutex progress_mutex_;
    std::string progress_text_;
    std::string progress_phase_ = "IDLE";
    uint64_t progress_tokens_ = 0;
    uint64_t progress_prompt_tokens_ = 0;
    bool progress_active_ = false;
    uint64_t open_ms_ = 0;
    uint64_t prompt_char_count_ = 0;
    uint64_t ttft_ms_ = 0;
    uint64_t generation_ms_ = 0;
    uint64_t generated_tokens_ = 0;
    uint64_t generated_text_chars_ = 0;
    uint64_t first_decode_token_ms_ = 0;
    uint64_t decode_ms_total_ = 0;
    std::vector<DecodeTokenMetrics> decode_token_metrics_;
    uint64_t reload_bytes_ = 0;
    uint64_t peak_resident_bytes_ = 0;
    uint64_t peak_active_bytes_ = 0;
    RuntimeMode mode_ = RuntimeMode::NormalInference;
    RuntimeTiming timing_;
    uint64_t prefill_ms_ = 0;
    uint64_t last_position_ms_ = 0;
    uint64_t prompt_token_count_ = 0;
    std::string prefill_mode_ = "SERIAL";
    uint64_t prefill_batch_size_ = 1;
    uint64_t prefill_layer_sequence_ms_ = 0;
    RuntimeTiming decode_attribution_;
};

std::mutex session_mutex;
std::unique_ptr<DirectSession> session;

jstring string_result(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_open(JNIEnv * env, jclass, jstring metadata, jstring endpoint,
    jboolean qualification) {
    std::lock_guard lock(session_mutex);
    if (session) return string_result(env, "OPEN_OK already_open");
    const char * raw_metadata = env->GetStringUTFChars(metadata, nullptr);
    const char * raw_endpoint = env->GetStringUTFChars(endpoint, nullptr);
    try {
        const RuntimeMode requested_mode = qualification == JNI_TRUE
            ? RuntimeMode::Qualification : RuntimeMode::NormalInference;
        session = std::make_unique<DirectSession>(raw_metadata, raw_endpoint, requested_mode);
        env->ReleaseStringUTFChars(metadata, raw_metadata);
        env->ReleaseStringUTFChars(endpoint, raw_endpoint);
        return string_result(env, "OPEN_OK DeepSeek-V2-Lite IQ2_XXS direct-runtime");
    } catch (const std::exception & error) {
        env->ReleaseStringUTFChars(metadata, raw_metadata);
        env->ReleaseStringUTFChars(endpoint, raw_endpoint);
        __android_log_print(ANDROID_LOG_ERROR, TAG, "open failed: %s", error.what());
        return string_result(env, std::string("OPEN_FAIL ") + error.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_generate(JNIEnv * env, jclass, jstring prompt, jint max_tokens) {
    DirectSession * active = nullptr;
    {
        std::lock_guard lock(session_mutex);
        active = session.get();
    }
    if (active == nullptr) return string_result(env, "GEN_FAIL model_not_open");
    const char * raw_prompt = env->GetStringUTFChars(prompt, nullptr);
    try {
        const std::string output = active->generate(raw_prompt, static_cast<uint32_t>(max_tokens));
        env->ReleaseStringUTFChars(prompt, raw_prompt);
        return string_result(env, output);
    } catch (const std::exception & error) {
        active->finish_progress("ERROR");
        env->ReleaseStringUTFChars(prompt, raw_prompt);
        __android_log_print(ANDROID_LOG_ERROR, TAG, "generation failed: %s", error.what());
        return string_result(env, std::string("GEN_FAIL ") + error.what());
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_metrics(JNIEnv * env, jclass) {
    std::lock_guard lock(session_mutex);
    return string_result(env, session ? session->metrics() : "Model: not open");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_eugen_vbufchat_NativeInference_progress(JNIEnv * env, jclass) {
    DirectSession * active = nullptr;
    {
        std::lock_guard lock(session_mutex);
        active = session.get();
    }
    return string_result(env, active ? active->progress() : "IDLE");
}

extern "C" JNIEXPORT void JNICALL
Java_com_eugen_vbufchat_NativeInference_cancel(JNIEnv *, jclass) {
    DirectSession * active = nullptr;
    {
        std::lock_guard lock(session_mutex);
        active = session.get();
    }
    if (active != nullptr) active->cancel();
}

extern "C" JNIEXPORT void JNICALL
Java_com_eugen_vbufchat_NativeInference_closeModel(JNIEnv *, jclass) {
    std::lock_guard lock(session_mutex);
    if (session) {
        session->close();
        session.reset();
    }
}
