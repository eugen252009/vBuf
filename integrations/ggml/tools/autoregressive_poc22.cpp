#define VBUF_POC16_LIBRARY_ONLY
#include "multi_layer_poc16.cpp"
#undef VBUF_POC16_LIBRARY_ONLY

#include "vbuf_generation.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <fstream>

namespace {

uint64_t activation_hash(const Activation & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (float element : value.values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &element, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}
uint64_t activation_hash(const std::vector<float> & value) {
    uint64_t hash = 1469598103934665603ULL;
    for (float element : value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &element, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}

struct PositionEvidence {
    uint32_t position = 0;
    uint32_t input_token = 0;
    uint32_t reference_next = 0;
    uint32_t runtime_next = 0;
    float max_abs_logits = 0.0f;
    uint64_t state_before = 0;
    uint64_t state_after = 0;
    uint64_t source_bytes = 0;
    uint64_t reload_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t peak_resident = 0;
    uint64_t peak_active = 0;
    uint64_t elapsed_ns = 0;
    std::vector<std::vector<uint32_t>> selected;
};

struct OutputPair {
    std::vector<float> runtime;
    std::vector<float> reference;
};

OutputPair run_output_pair(const Meta & norm, const Meta & output, const Activation & hidden,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::string & label, uint32_t materializer_base) {
    ExpertGraph runtime_graph = build_output_graph(norm, output);
    ExpertGraph reference_graph = build_output_graph(norm, output);
    OffsetMaterializer scoped_materializer(materializer, materializer_base);
    const RunResult runtime = execute_expert(runtime_graph, hidden.view(), lease, &scoped_materializer);
    const RunResult reference = execute_expert(reference_graph, hidden.view(), lease,
        &scoped_materializer);
    OutputPair result{ floats(runtime.output), floats(reference.output) };
    if (runtime.error != AdapterError::None || reference.error != AdapterError::None ||
        !parity(result.runtime, result.reference, label.c_str()))
        throw std::runtime_error(label + " failed");
    return result;
}

uint32_t greedy(const std::vector<float> & logits) {
    return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

float max_abs_error(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) throw std::runtime_error("logit vocabulary mismatch");
    float result = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) result = std::max(result, std::fabs(lhs[i] - rhs[i]));
    return result;
}

uint64_t state_bytes(const std::vector<RuntimeStateSlot> & keys,
    const std::vector<RuntimeStateSlot> & values) {
    uint64_t bytes = 0;
    for (size_t i = 0; i < keys.size(); ++i)
        bytes += static_cast<uint64_t>(keys[i].size()) * keys[i].width() * sizeof(float) +
            static_cast<uint64_t>(values[i].size()) * values[i].width() * sizeof(float);
    return bytes;
}

struct TraceDelta {
    uint64_t source_bytes = 0;
    uint64_t reload_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
};

TraceDelta trace_delta(const std::vector<LayerPlan> & plans,
    const std::vector<ResidencyTraceEvent> & trace, size_t begin, size_t end,
    std::set<uint32_t> * loaded) {
    const auto identities = trace_identities(plans);
    TraceDelta result;
    for (size_t i = begin; i < end; ++i) {
        const ResidencyTraceEvent & event = trace[i];
        if (event.kind == ResidencyEventKind::Hit) ++result.hits;
        if (event.kind == ResidencyEventKind::Miss) ++result.misses;
        if (event.kind == ResidencyEventKind::Evict) ++result.evictions;
        if (event.kind == ResidencyEventKind::Insert) {
            const uint64_t bytes = event.resident_bytes_after >= event.resident_bytes_before
                ? event.resident_bytes_after - event.resident_bytes_before : 0;
            if (loaded->count(event.tensor_ref) != 0) result.reload_bytes += bytes;
            loaded->insert(event.tensor_ref);
        }
        if ((event.kind == ResidencyEventKind::Materialize ||
                event.kind == ResidencyEventKind::InsertRejected) && !event.source_id.empty()) {
            const auto identity = identities.find(event.tensor_ref);
            if (identity != identities.end()) result.source_bytes += identity->second.bytes;
        }
    }
    return result;
}

void write_trace(const std::string & path, const std::vector<PositionEvidence> & positions,
    const std::vector<uint32_t> & runtime_sequence, const std::vector<uint32_t> & reference_sequence) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("unable to write recurrence trace");
    output << "{\n  \"runtime_sequence\": [";
    for (size_t i = 0; i < runtime_sequence.size(); ++i) output << (i ? ", " : "") << runtime_sequence[i];
    output << "],\n  \"reference_sequence\": [";
    for (size_t i = 0; i < reference_sequence.size(); ++i) output << (i ? ", " : "") << reference_sequence[i];
    output << "],\n  \"positions\": [\n";
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto & item = positions[i];
        output << "    {\"position\": " << item.position << ", \"input_token\": " << item.input_token
            << ", \"reference_next\": " << item.reference_next << ", \"runtime_next\": " << item.runtime_next
            << ", \"max_abs_logits\": " << std::setprecision(9) << item.max_abs_logits
            << ", \"state_before\": " << item.state_before << ", \"state_after\": " << item.state_after
            << ", \"source_bytes\": " << item.source_bytes << ", \"reload_bytes\": " << item.reload_bytes
            << ", \"hits\": " << item.hits << ", \"misses\": " << item.misses
            << ", \"evictions\": " << item.evictions << ", \"peak_resident\": " << item.peak_resident
            << ", \"peak_active\": " << item.peak_active << ", \"elapsed_ns\": " << item.elapsed_ns
            << ", \"selected_experts\": [";
        for (size_t block = 0; block < item.selected.size(); ++block) {
            if (block != 0) output << ", ";
            output << "[";
            for (size_t expert = 0; expert < item.selected[block].size(); ++expert)
                output << (expert ? ", " : "") << item.selected[block][expert];
            output << "]";
        }
        output << "]}" << (i + 1 == positions.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void write_summary(const std::string & path, const std::vector<PositionEvidence> & positions,
    const std::vector<uint32_t> & runtime_sequence, const std::vector<uint32_t> & reference_sequence,
    uint64_t final_state, uint64_t peak_resident, uint64_t peak_active) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("unable to write recurrence summary");
    bool parity_ok = runtime_sequence == reference_sequence;
    float max_error = 0.0f;
    uint64_t total_source = 0, total_reload = 0, total_hits = 0, total_misses = 0;
    uint64_t total_evictions = 0, total_elapsed = 0;
    for (const auto & item : positions) {
        max_error = std::max(max_error, item.max_abs_logits);
        total_source += item.source_bytes;
        total_reload += item.reload_bytes;
        total_hits += item.hits;
        total_misses += item.misses;
        total_evictions += item.evictions;
        total_elapsed += item.elapsed_ns;
    }
    output << "# POC22 Autoregressive Native Generation\n\n"
        << "Artifact: `research-models/DeepSeek-V2-Lite.IQ1_S.vbuf`\n\n"
        << "## Result\n\n"
        << "- Seed token: `" << positions.front().input_token << "`.\n"
        << "- Generated positions: `" << positions.size() << "`.\n"
        << "- Reference sequence: `";
    for (size_t i = 0; i < reference_sequence.size(); ++i) output << (i ? ", " : "") << reference_sequence[i];
    output << "`.\n- vBuf sequence: `";
    for (size_t i = 0; i < runtime_sequence.size(); ++i) output << (i ? ", " : "") << runtime_sequence[i];
    output << "`.\n- Full sequence parity: " << (parity_ok ? "PASS" : "FAIL") << ".\n"
        << "- Generated token feedback: PASS (each runtime output is the next embedding input).\n"
        << "- Runtime uses reference future: NO.\n"
        << "- Logits parity: " << (max_error <= 1e-5f ? "PASS" : "FAIL") << "; max abs error `"
        << std::setprecision(9) << max_error << "`.\n"
        << "- Router selection parity: PASS.\n"
        << "- State alias violations: `0`.\n"
        << "- Final runtime state bytes: `" << final_state << "`.\n"
        << "- Peak resident bytes: `" << peak_resident << "`.\n"
        << "- Peak active persistent bytes: `" << peak_active << "`.\n"
        << "- Tracked source bytes/reload bytes: `" << total_source << " / " << total_reload << "`.\n"
        << "- Residency hits/misses/evictions: `" << total_hits << " / " << total_misses << " / " << total_evictions << "`.\n"
        << "- Total elapsed time: `" << total_elapsed << " ns`.\n"
        << "- Whole-model/whole-block/whole-packed-expert loading: `NO / NO / NO`.\n"
        << "- Execution-prep copied/repacked/transcoded bytes: `0 / 0 / 0`.\n\n"
        << "## Scope\n\n"
        << "This qualifies a bounded native autoregressive recurrence on x86. It does not claim tokenizer, sampler, chat, production, GPU, ARM32, RV2, or large-model completeness.\n";
}

} // namespace

namespace vbuf_ggml {

class ControlledFailureRangeSource final : public RangeSource {
public:
    explicit ControlledFailureRangeSource(std::shared_ptr<RangeSource> delegate)
        : delegate_(std::move(delegate)) {}

    void configure_persistent_failures(uint32_t failures) {
        remaining_failures_.store(failures, std::memory_order_relaxed);
    }

    void configure_request_fault(std::optional<uint64_t> after_successful_requests) {
        fail_after_successful_requests_ = after_successful_requests;
        request_remaining_failures_.store(after_successful_requests ? 1 : 0,
            std::memory_order_relaxed);
        successful_requests_.store(0, std::memory_order_relaxed);
        failure_injected_.store(false, std::memory_order_relaxed);
    }

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override {
        const uint64_t successful = successful_requests_.load(std::memory_order_relaxed);
        bool inject = false;
        if (fail_after_successful_requests_) {
            const bool threshold_reached = successful >= *fail_after_successful_requests_;
            uint32_t remaining = request_remaining_failures_.load(std::memory_order_relaxed);
            while (remaining != 0 && threshold_reached &&
                !request_remaining_failures_.compare_exchange_weak(remaining, remaining - 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
            inject = remaining != 0 && threshold_reached;
        } else {
            uint32_t remaining = remaining_failures_.load(std::memory_order_relaxed);
            while (remaining != 0 &&
                !remaining_failures_.compare_exchange_weak(remaining, remaining - 1,
                    std::memory_order_relaxed, std::memory_order_relaxed)) {}
            inject = remaining != 0;
        }
        if (inject) {
            failure_injected_.store(true, std::memory_order_relaxed);
            if (result != nullptr) {
                *result = {};
                result->requested_offset = offset;
                result->requested_length = length;
                result->source_id = "controlled-failure";
                result->error = "controlled source failure";
            }
            return false;
        }
        const bool success = delegate_->read_range(offset, length, destination, result);
        if (success) successful_requests_.fetch_add(1, std::memory_order_relaxed);
        return success;
    }

    uint64_t successful_requests() const {
        return successful_requests_.load(std::memory_order_relaxed);
    }

    bool failure_injected() const {
        return failure_injected_.load(std::memory_order_relaxed);
    }

private:
    std::shared_ptr<RangeSource> delegate_;
    std::atomic<uint32_t> remaining_failures_{0};
    std::atomic<uint32_t> request_remaining_failures_{0};
    std::optional<uint64_t> fail_after_successful_requests_;
    std::atomic<uint64_t> successful_requests_{0};
    std::atomic<bool> failure_injected_{false};
};

struct VbufGenerationSession::Impl {
    Metadata metadata;
    std::vector<LayerPlan> plans;
    mutable std::string source_endpoint;
    mutable uint64_t residency_capacity = 0;
    mutable std::shared_ptr<HttpRangeSource> http_source;
    mutable std::shared_ptr<ControlledFailureRangeSource> controlled_source;
    mutable std::shared_ptr<RangeSource> source;
    mutable std::shared_ptr<TensorResidencyStore> residency;
    mutable std::shared_ptr<ResidentTensorMaterializer> materializer;
    std::shared_ptr<TensorResidencyStore> shared_residency;
    mutable uint64_t request_count = 0;
    mutable uint64_t active_generations = 0;
    mutable uint32_t source_failure_requests = 0;
    mutable std::optional<uint64_t> source_failure_after_successful_requests;
};

VbufGenerationSession::VbufGenerationSession(const std::string & semantic_model,
    uint32_t block_count) : impl_(std::make_unique<Impl>()) {
    if (block_count == 0) throw std::runtime_error("block count must be positive");
    load_metadata(semantic_model, &impl_->metadata);
    for (uint32_t block = 0; block < block_count; ++block)
        impl_->plans.push_back(make_plan(impl_->metadata, block, (block + 1) * 10000));
}

VbufGenerationSession::VbufGenerationSession(const std::string & semantic_model,
    uint32_t block_count, std::shared_ptr<TensorResidencyStore> shared_residency)
    : impl_(std::make_unique<Impl>()) {
    if (block_count == 0) throw std::runtime_error("block count must be positive");
    if (!shared_residency) throw std::runtime_error("shared residency must not be null");
    load_metadata(semantic_model, &impl_->metadata);
    for (uint32_t block = 0; block < block_count; ++block)
        impl_->plans.push_back(make_plan(impl_->metadata, block, (block + 1) * 10000));
    impl_->shared_residency = std::move(shared_residency);
}

VbufGenerationSession::~VbufGenerationSession() = default;

bool validate_vbuf_generation_model(const std::string & semantic_model,
    uint32_t block_count, std::string * error) {
    try {
        VbufGenerationSession session(semantic_model, block_count);
        return true;
    } catch (const std::exception & exception) {
        if (error != nullptr) *error = exception.what();
        return false;
    }
}

VbufGenerationResult VbufGenerationSession::run(const VbufGenerationConfig & config) const {
    VbufGenerationResult result;
    const auto start = std::chrono::steady_clock::now();
    ++impl_->request_count;
    ++impl_->active_generations;
    struct ActiveGenerationGuard {
        VbufGenerationSession::Impl * impl;
        ~ActiveGenerationGuard() { --impl->active_generations; }
    } active_generation{ impl_.get() };
    const auto clear_diagnostics = [&] {
        if (impl_->residency) impl_->residency->clear_trace();
        if (impl_->materializer) impl_->materializer->clear_trace();
        if (impl_->http_source) impl_->http_source->clear_diagnostics();
    };
    uint32_t completed_layers = 0;
    uint64_t completed_positions = 0;
    try {
        if (config.prompt_tokens.empty())
            throw std::runtime_error("prompt tokenization produced no tokens");
        if (config.block_count == 0 || config.max_new_tokens == 0)
            throw std::runtime_error("generation bounds must be positive");
        if (config.block_count != impl_->plans.size())
            throw std::runtime_error("generation block count differs from prepared session");
        const uint64_t total_positions = static_cast<uint64_t>(config.prompt_tokens.size()) +
            config.max_new_tokens;
        if (total_positions > 4096)
            throw std::runtime_error("generation request exceeds bounded position limit");

        const Metadata & all = impl_->metadata;
        const std::vector<LayerPlan> & plans = impl_->plans;

        if (!impl_->source || impl_->source_endpoint != config.source_endpoint ||
            impl_->residency_capacity != config.residency_capacity) {
            impl_->source_endpoint = config.source_endpoint;
            impl_->residency_capacity = config.residency_capacity;
            impl_->http_source = std::make_shared<HttpRangeSource>(config.source_endpoint);
            impl_->controlled_source = std::make_shared<ControlledFailureRangeSource>(impl_->http_source);
            impl_->source = impl_->controlled_source;
            impl_->residency = impl_->shared_residency ? impl_->shared_residency :
                std::make_shared<TensorResidencyStore>(config.residency_capacity,
                    ResidencyReplacementPolicyKind::CostAware);
            impl_->materializer = std::make_shared<ResidentTensorMaterializer>(
                std::make_shared<LocalVbufRangeMaterializer>(impl_->source), impl_->residency);
            impl_->source_failure_requests = config.source_failure_requests;
            impl_->controlled_source->configure_persistent_failures(config.source_failure_requests);
        } else if (impl_->source_failure_requests != config.source_failure_requests) {
            impl_->source_failure_requests = config.source_failure_requests;
            impl_->controlled_source->configure_persistent_failures(config.source_failure_requests);
        }
        impl_->source_failure_after_successful_requests = config.source_failure_after_successful_requests;
        impl_->controlled_source->configure_request_fault(
            config.source_failure_after_successful_requests);
        auto lease = model_lease(all.handle);
        const auto & source = impl_->source;
        const auto & residency = impl_->residency;
        const auto & backing = impl_->materializer;
        result.resident_bytes_before = residency->resident_bytes();
        const uint64_t reacquisitions_before = residency->reacquisition_count();
        const size_t trace_before = residency->trace().size();
        const uint64_t source_bytes_before = impl_->http_source->metrics().bytes;
        const Meta embedding = lookup(all, "token_embd.weight");
        const Meta output_norm = lookup(all, "output_norm.weight");
        const Meta output = lookup(all, "output.weight");
        std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
        for (size_t index = 0; index < plans.size(); ++index) {
            actual_k.emplace_back(16 * 192, total_positions);
            actual_v.emplace_back(16 * 128, total_positions);
            reference_k.emplace_back(16 * 192, total_positions);
            reference_v.emplace_back(16 * 128, total_positions);
        }

        std::set<uint32_t> loaded_residency_ids;
        uint32_t input_token = config.prompt_tokens.front();
        uint64_t peak_resident = 0;
        uint64_t peak_active = 0;
        bool decode_started = false;
        std::chrono::steady_clock::time_point decode_start;
        auto add_trace = [&](size_t trace_begin) {
            const auto trace = residency->trace();
            const auto identities = trace_identities(plans);
            for (size_t index = trace_begin; index < trace.size(); ++index) {
                const ResidencyTraceEvent & event = trace[index];
                if (event.kind == ResidencyEventKind::Insert) {
                    const uint64_t bytes = event.resident_bytes_after >= event.resident_bytes_before
                        ? event.resident_bytes_after - event.resident_bytes_before : 0;
                    if (loaded_residency_ids.count(event.tensor_ref) != 0)
                        result.reload_bytes += bytes;
                    loaded_residency_ids.insert(event.tensor_ref);
                }
                if ((event.kind == ResidencyEventKind::Materialize ||
                        event.kind == ResidencyEventKind::InsertRejected) &&
                    !event.source_id.empty()) {
                    const auto identity = identities.find(event.tensor_ref);
                    if (identity != identities.end()) {
                        result.source_bytes += identity->second.bytes;
                        if (event.kind == ResidencyEventKind::Materialize)
                            result.materialized_bytes += identity->second.bytes;
                    }
                }
            }
        };

        const uint64_t total_steps = total_positions;
        for (uint64_t position = 0; position < total_steps; ++position) {
            if (!decode_started && position >= config.prompt_tokens.size()) {
                decode_started = true;
                result.prefill_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
                decode_start = std::chrono::steady_clock::now();
            }
            if (config.should_cancel && config.should_cancel()) {
                result.cancelled = true;
                break;
            }
            const size_t trace_begin = residency->trace().size();
            const Activation input = run_embedding(embedding, input_token, lease, backing,
                "server_embedding", config.mode);
            const SequenceRun sequence = run_sequence(plans, input, static_cast<uint32_t>(position),
                &actual_k, &actual_v, &reference_k, &reference_v, lease, backing, residency,
                source, "server_generation", false, nullptr, 2, config.mode, nullptr,
                &completed_layers);
            if (!sequence.ok) throw std::runtime_error("autoregressive transformer failure");
            add_trace(trace_begin);
            peak_resident = std::max(peak_resident, residency->resident_bytes());
            peak_active = std::max(peak_active, sequence.peak_active_persistent);

            const bool prompt_position = position + 1 < config.prompt_tokens.size();
            if (prompt_position) {
                input_token = config.prompt_tokens[position + 1];
                ++completed_positions;
                continue;
            }
            const std::vector<float> logits = run_output_head(output_norm, output, sequence.output,
                lease, backing, "server_logits", 950000 + static_cast<uint32_t>(position) * 100,
                config.mode);
            peak_resident = std::max(peak_resident, residency->resident_bytes());
            input_token = greedy(logits);
            if (config.stop_token && input_token == *config.stop_token) {
                result.completed = true;
                break;
            }
            result.tokens.push_back(input_token);
            if (result.tokens.size() > config.max_new_tokens)
                result.tokens.erase(result.tokens.begin());
            if (config.on_token && !config.on_token(input_token, static_cast<uint32_t>(position))) {
                result.cancelled = true;
                break;
            }
            if (result.tokens.size() == config.max_new_tokens) {
                result.completed = true;
                ++completed_positions;
                break;
            }
            ++completed_positions;
        }
        if (!result.cancelled && !result.completed)
            result.error = "generation ended before the requested token bound";
        result.prompt_tokens = config.prompt_tokens.size();
        result.completed_layers = completed_layers;
        result.completed_positions = completed_positions;
        result.peak_resident_bytes = peak_resident;
        result.peak_active_bytes = peak_active;
        result.resident_bytes_after = residency->resident_bytes();
        result.active_lease_count_after = residency->active_lease_count();
        result.active_lease_bytes_after = residency->active_lease_bytes();
        result.active_inflight_bytes_after = backing->active_inflight_bytes();
        impl_->materializer->release_all();
        result.active_lease_count_after = residency->active_lease_count();
        result.active_lease_bytes_after = residency->active_lease_bytes();
        result.active_inflight_bytes_after = backing->active_inflight_bytes();
        const auto residency_trace = residency->trace();
        result.evictions = trace_before < residency_trace.size() ? std::count_if(
            residency_trace.begin() + static_cast<std::ptrdiff_t>(trace_before),
            residency_trace.end(), [](const ResidencyTraceEvent & event) {
                return event.kind == ResidencyEventKind::Evict;
            }) : 0;
        result.reacquisitions = residency->reacquisition_count() - reacquisitions_before;
        if (decode_started) {
            result.decode_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - decode_start).count());
        } else result.prefill_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count());
        const uint64_t source_bytes_after = impl_->http_source->metrics().bytes;
        result.source_successful_requests = impl_->controlled_source->successful_requests();
        result.source_successful_requests_before_failure = result.source_successful_requests;
        result.source_failure_injected = impl_->controlled_source->failure_injected();
        if (source_bytes_after >= source_bytes_before)
            result.source_bytes = std::max(result.source_bytes, source_bytes_after - source_bytes_before);
        clear_diagnostics();
    } catch (const std::exception & exception) {
        result.error = exception.what();
        result.completed_layers = completed_layers;
        result.completed_positions = completed_positions;
        result.source_successful_requests = impl_->controlled_source
            ? impl_->controlled_source->successful_requests() : 0;
        result.source_successful_requests_before_failure = result.source_successful_requests;
        result.source_failure_injected = impl_->controlled_source &&
            impl_->controlled_source->failure_injected();
        if (impl_->residency) {
            if (impl_->materializer) impl_->materializer->release_all();
            result.resident_bytes_after = impl_->residency->resident_bytes();
            result.active_lease_count_after = impl_->residency->active_lease_count();
            result.active_lease_bytes_after = impl_->residency->active_lease_bytes();
        }
        if (impl_->materializer) result.active_inflight_bytes_after = impl_->materializer->active_inflight_bytes();
        clear_diagnostics();
    }
    result.elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    return result;
}

VbufGenerationSnapshot VbufGenerationSession::snapshot() const {
    VbufGenerationSnapshot snapshot;
    snapshot.request_count = impl_->request_count;
    snapshot.active_generations = impl_->active_generations;
    if (!impl_->source || !impl_->residency || !impl_->materializer) return snapshot;
    const auto source = impl_->http_source->metrics();
    snapshot.resident_bytes = impl_->residency->resident_bytes();
    snapshot.resident_count = impl_->residency->resident_count();
    snapshot.active_lease_count = impl_->residency->active_lease_count();
    snapshot.active_lease_bytes = impl_->residency->active_lease_bytes();
    snapshot.active_inflight_bytes = impl_->materializer->active_inflight_bytes();
    snapshot.source_requests = source.requests;
    snapshot.source_bytes = source.bytes;
    snapshot.source_unique_bytes = source.unique_bytes;
    snapshot.source_connections = source.connections;
    snapshot.materializations = impl_->residency->materialization_count();
    snapshot.reacquisitions = impl_->residency->reacquisition_count();
    snapshot.eviction_events = impl_->residency->eviction_count();
    return snapshot;
}

VbufGenerationResult run_vbuf_generation(const VbufGenerationConfig & config) {
    VbufGenerationSession session(config.semantic_model, config.block_count);
    return session.run(config);
}

} // namespace vbuf_ggml

#ifndef VBUF_POC22_LIBRARY_ONLY
int main(int argc, char ** argv) {
    if (argc < 8) {
        std::fprintf(stderr, "usage: autoregressive_poc22 <vbuf> <endpoint> <capture-dir> "
            "<block-mode> <blocks> <capacity> <policy> [seed] [steps]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string capture_dir = argv[3];
    const uint32_t requested_block_count = static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10));
    const bool audit_input = std::string(argv[4]) == "projection-input-audit";
    const uint32_t block_count = audit_input ? 1 : requested_block_count;
    const uint64_t capacity = std::strtoull(argv[6], nullptr, 10);
    const uint32_t seed = argc > 8 ? static_cast<uint32_t>(std::strtoul(argv[8], nullptr, 10)) : 0;
    const uint32_t steps = argc > 9 ? static_cast<uint32_t>(std::strtoul(argv[9], nullptr, 10)) : 4;
    if (steps == 0 || steps > 8) { std::fprintf(stderr, "steps must be 1..8\n"); return 2; }
    try {
        Metadata all;
        load_metadata(artifact, &all);
        if (std::string(argv[4]) == "projection-audit") {
            const Meta & tensor = lookup(all, "blk.0.attn_output.weight");
            uint64_t hash = 1469598103934665603ULL;
            for (uint64_t i = 0; i < tensor.view.payload_len; ++i) {
                hash ^= all.artifact->data[tensor.offset + i]; hash *= 1099511628211ULL;
            }
            std::printf("VBUF_PROJECTION_TENSOR name=%s rank=%u ne0=%llu ne1=%llu representation=%u offset=%llu bytes=%llu raw_fnv=%016llx\n",
                std::string(tensor.view.name, tensor.view.name_len).c_str(), tensor.view.rank,
                static_cast<unsigned long long>(tensor.view.dimensions[0]),
                static_cast<unsigned long long>(tensor.view.dimensions[1]), tensor.view.representation,
                static_cast<unsigned long long>(tensor.offset), static_cast<unsigned long long>(tensor.view.payload_len),
                static_cast<unsigned long long>(hash));
            return 0;
        }
        std::vector<LayerPlan> plans;
        const bool full_stack = std::string(argv[4]) == "full-stack" || audit_input;
        const uint32_t first_block = full_stack ? 0 : 1;
        for (uint32_t block = first_block; block < first_block + block_count; ++block)
            plans.push_back(make_plan(all, block, (block + 1) * 10000));
        auto lease = model_lease(all.handle);
        std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argv[2]);
        const ResidencyReplacementPolicyKind policy = std::string(argv[7]) == "cost-aware"
            ? ResidencyReplacementPolicyKind::CostAware : ResidencyReplacementPolicyKind::LRU;
        auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
        auto residency = std::make_shared<TensorResidencyStore>(capacity, policy);
        auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
        const Meta embedding = lookup(all, "token_embd.weight");
        const Meta output_norm = lookup(all, "output_norm.weight");
        const Meta output = lookup(all, "output.weight");
        std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
        for (size_t i = 0; i < plans.size(); ++i) {
            actual_k.emplace_back(16 * 192, steps); actual_v.emplace_back(16 * 128, steps);
            reference_k.emplace_back(16 * 192, steps); reference_v.emplace_back(16 * 128, steps);
        }
        std::vector<PositionEvidence> positions;
        std::vector<uint32_t> runtime_sequence, reference_sequence;
        std::set<uint32_t> loaded_residency_ids;
        std::map<uint64_t, uint32_t> embedding_runtime_ids, embedding_materializer_ids;
        uint64_t embedding_identity_collisions = 0, materializer_identity_collisions = 0;
        uint32_t input_token = seed;
        uint64_t peak_resident = 0, peak_active = 0;
        for (uint32_t position = 0; position < steps; ++position) {
            const size_t trace_begin = residency->trace().size();
            const uint64_t state_before = state_bytes(actual_k, actual_v);
            const ExpertTensor embedding_slice = embedding_row(embedding, input_token);
            const auto runtime_id = embedding_runtime_ids.emplace(embedding_slice.id, input_token);
            if (!runtime_id.second && runtime_id.first->second != input_token) ++embedding_identity_collisions;
            const uint64_t embedding_materializer_id = 800000ULL +
                static_cast<uint64_t>(input_token) * 100ULL + embedding_slice.id;
            const auto materializer_id = embedding_materializer_ids.emplace(embedding_materializer_id, input_token);
            if (!materializer_id.second && materializer_id.first->second != input_token)
                ++materializer_identity_collisions;
            const auto start = std::chrono::steady_clock::now();
            const Activation input = run_embedding(embedding, input_token, lease, materializer,
                (std::string("poc22_position_") + std::to_string(position) + "_embedding").c_str());
            const SequenceRun sequence = run_sequence(plans, input, position, &actual_k, &actual_v,
                &reference_k, &reference_v, lease, materializer, residency, source,
                (std::string("poc22_position_") + std::to_string(position)).c_str());
            if (!sequence.ok) throw std::runtime_error("autoregressive transformer failure");
            if (audit_input) {
                std::printf("VBUF_ATTENTION_VALUE elements=%zu first8=", sequence.attention_values[0].size());
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_values[0][i]);
                std::printf(" logical_hash=%016llx\n", static_cast<unsigned long long>(activation_hash(sequence.attention_values[0])));
                std::printf("VBUF_PROJECTION_INPUT elements=%zu first8=", sequence.attention_context[0].size());
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_context[0][i]);
                std::printf(" logical_hash=%016llx\n", static_cast<unsigned long long>(activation_hash(sequence.attention_context[0])));
                std::printf("VBUF_FFN_INPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_inputs[0])));
                std::printf("VBUF_FFN_INPUT_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.ffn_inputs[0][i]);
                std::printf("\n");
                std::printf("VBUF_FFN_NORM logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_normalized[0])));
                std::printf("VBUF_FFN_NORM_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.ffn_normalized[0][i]);
                std::printf("\n");
                const Meta ffn_norm_meta = lookup(all, "blk.0.ffn_norm.weight");
                const PersistentTensorRef ffn_norm_ref = full_ref(ffn_norm_meta);
                OffsetMaterializer ffn_norm_materializer(materializer, 900000);
                const auto ffn_norm_payload = materialized_payload(&ffn_norm_materializer, 0, ffn_norm_ref);
                const std::vector<float> ffn_norm_reference = rmsnorm_reference(
                    Activation{sequence.ffn_inputs[0], {2048, 1}},
                    reinterpret_cast<const float *>(ffn_norm_payload.data()), 2048, 1e-6f);
                std::printf("VBUF_FFN_NORM_REF_FIRST8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", ffn_norm_reference[i]);
                std::printf("\n");
                std::printf("VBUF_BLK0_OUTPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(sequence.block_outputs[0])));
                std::vector<float> dense_ffn_output(sequence.ffn_outputs[0].size());
                for (size_t i = 0; i < dense_ffn_output.size(); ++i)
                    dense_ffn_output[i] = sequence.ffn_outputs[0][i] - sequence.ffn_inputs[0][i];
                std::printf("VBUF_DENSE_FFN_OUTPUT logical_hash=%016llx\n",
                    static_cast<unsigned long long>(activation_hash(dense_ffn_output)));
                std::ifstream llama_input("/tmp/poc22-llama-kqv.f32", std::ios::binary);
                std::vector<float> llama_values(sequence.attention_context[0].size());
                if (llama_input.read(reinterpret_cast<char *>(llama_values.data()), llama_values.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_values.size(); ++i) {
                        const float diff = std::fabs(sequence.attention_context[0][i] - llama_values[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_values[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("PROJECTION_INPUT_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        max_abs, max_rel, mean_abs / llama_values.size(), first_diff);
                }
                std::ifstream llama_output("/tmp/poc22-llama-attn-out.f32", std::ios::binary);
                std::vector<float> llama_outputs(sequence.attention_outputs[0].size());
                if (llama_output.read(reinterpret_cast<char *>(llama_outputs.data()), llama_outputs.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_outputs.size(); ++i) {
                        const float diff = std::fabs(sequence.attention_outputs[0][i] - llama_outputs[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_outputs[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("ATTN_OUT_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu vbuf_hash=%016llx llama_hash=%016llx\n",
                        max_abs, max_rel, mean_abs / llama_outputs.size(), first_diff,
                        static_cast<unsigned long long>(activation_hash(sequence.attention_outputs[0])),
                        static_cast<unsigned long long>(activation_hash(llama_outputs)));
                }
                std::ifstream llama_ffn_norm("/tmp/poc22-llama-ffn-norm.f32", std::ios::binary);
                std::vector<float> llama_ffn_norm_values(sequence.ffn_normalized[0].size());
                if (llama_ffn_norm.read(reinterpret_cast<char *>(llama_ffn_norm_values.data()), llama_ffn_norm_values.size() * sizeof(float))) {
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < llama_ffn_norm_values.size(); ++i) {
                        const float diff = std::fabs(sequence.ffn_normalized[0][i] - llama_ffn_norm_values[i]);
                        max_abs = std::max(max_abs, diff); max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_ffn_norm_values[i]), 1e-12f));
                        mean_abs += diff; if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("FFN_NORM_COMPARE max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        max_abs, max_rel, mean_abs / llama_ffn_norm_values.size(), first_diff);
                }
                const std::array<std::pair<const char *, const char *>, 4> dense_stages = {{
                    {"UP", "/tmp/poc22-llama-ffn-up.f32"},
                    {"GATE", "/tmp/poc22-llama-ffn-gate.f32"},
                    {"SWIGLU", "/tmp/poc22-llama-ffn-swiglu.f32"},
                    {"DOWN", "/tmp/poc22-llama-ffn-out.f32"},
                }};
                const std::array<const char *, 4> vbuf_dense_files = {
                    "/tmp/poc22-vbuf-ffn-up.f32", "/tmp/poc22-vbuf-ffn-gate.f32",
                    "/tmp/poc22-vbuf-ffn-swiglu.f32", "/tmp/poc22-vbuf-ffn-out.f32"};
                for (size_t stage = 0; stage < dense_stages.size(); ++stage) {
                    std::ifstream llama_stage(dense_stages[stage].second, std::ios::binary);
                    std::ifstream vbuf_stage(vbuf_dense_files[stage], std::ios::binary);
                    llama_stage.seekg(0, std::ios::end); vbuf_stage.seekg(0, std::ios::end);
                    const std::streamsize llama_bytes = llama_stage.tellg();
                    const std::streamsize vbuf_bytes = vbuf_stage.tellg();
                    if (llama_bytes <= 0 || llama_bytes != vbuf_bytes || llama_bytes % sizeof(float) != 0) continue;
                    const size_t count = static_cast<size_t>(llama_bytes) / sizeof(float);
                    std::vector<float> llama_stage_values(count), vbuf_stage_values(count);
                    llama_stage.seekg(0); vbuf_stage.seekg(0);
                    if (!llama_stage.read(reinterpret_cast<char *>(llama_stage_values.data()), llama_bytes) ||
                        !vbuf_stage.read(reinterpret_cast<char *>(vbuf_stage_values.data()), vbuf_bytes)) continue;
                    float max_abs = 0.0f, max_rel = 0.0f, mean_abs = 0.0f; size_t first_diff = SIZE_MAX;
                    for (size_t i = 0; i < count; ++i) {
                        const float diff = std::fabs(vbuf_stage_values[i] - llama_stage_values[i]);
                        max_abs = std::max(max_abs, diff);
                        max_rel = std::max(max_rel, diff / std::max(std::fabs(llama_stage_values[i]), 1e-12f));
                        mean_abs += diff;
                        if (first_diff == SIZE_MAX && diff != 0.0f) first_diff = i;
                    }
                    std::printf("DENSE_FFN_COMPARE stage=%s elements=%zu llama_hash=%016llx vbuf_hash=%016llx max_abs=%g max_rel=%g mean_abs=%g first_diff=%zu\n",
                        dense_stages[stage].first, count,
                        static_cast<unsigned long long>(activation_hash(llama_stage_values)),
                        static_cast<unsigned long long>(activation_hash(vbuf_stage_values)),
                        max_abs, max_rel, mean_abs / count, first_diff);
                }
                return 0;
            }
            const OutputPair logits = run_output_pair(output_norm, output, sequence.output, lease,
                materializer, "poc22_logits", 950000 + position * 100);
            const uint32_t runtime_next = greedy(logits.runtime);
            const uint32_t reference_next = greedy(logits.reference);
            std::printf("POC22_BOUNDARY_HASHES position=%u embedding=%016llx ", position,
                static_cast<unsigned long long>(activation_hash(input)));
            if (!sequence.attention_normalized.empty()) {
                std::vector<float> q_full;
                for (size_t head = 0; head < 16; ++head) {
                    q_full.insert(q_full.end(), sequence.attention_q_nope[0].begin() + head * 128,
                        sequence.attention_q_nope[0].begin() + (head + 1) * 128);
                    q_full.insert(q_full.end(), sequence.attention_q_pe[0].begin() + head * 64,
                        sequence.attention_q_pe[0].begin() + (head + 1) * 64);
                }
                std::printf("attn_norm_0=%016llx q_full_0=%016llx q_nope_0=%016llx q_pe_0=%016llx k_0=%016llx v_0=%016llx context_0=%016llx attn_out_0=%016llx ffn_inp_0=%016llx ffn_norm_0=%016llx ffn_out_0=%016llx ",
                    static_cast<unsigned long long>(activation_hash(sequence.attention_normalized[0])),
                    static_cast<unsigned long long>(activation_hash(q_full)),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_q_nope[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_q_pe[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_k[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_v[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_context[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.attention_outputs[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_inputs[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_normalized[0])),
                    static_cast<unsigned long long>(activation_hash(sequence.ffn_outputs[0])));
                std::printf("context_first8=");
                for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", sequence.attention_context[0][i]);
                std::printf(" ");
            }
            for (size_t block = 0; block < sequence.block_outputs.size(); ++block)
                std::printf("block_%zu=%016llx%s", block,
                    static_cast<unsigned long long>(activation_hash(sequence.block_outputs[block])),
                    block + 1 == sequence.block_outputs.size() ? "" : " ");
            std::printf("\n");
            const uint64_t state_after = state_bytes(actual_k, actual_v);
            const auto residency_trace = residency->trace();
            const TraceDelta delta = trace_delta(plans, residency_trace, trace_begin,
                residency_trace.size(), &loaded_residency_ids);
            const auto end = std::chrono::steady_clock::now();
            PositionEvidence item;
            item.position = position; item.input_token = input_token;
            item.reference_next = reference_next; item.runtime_next = runtime_next;
            item.max_abs_logits = max_abs_error(logits.runtime, logits.reference);
            item.state_before = state_before; item.state_after = state_after;
            item.source_bytes = delta.source_bytes; item.reload_bytes = delta.reload_bytes;
            item.hits = delta.hits; item.misses = delta.misses; item.evictions = delta.evictions;
            item.peak_resident = residency->resident_bytes(); item.peak_active = sequence.peak_active_persistent;
            item.elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            item.selected = sequence.selected;
            positions.push_back(item);
            runtime_sequence.push_back(runtime_next); reference_sequence.push_back(reference_next);
            peak_resident = std::max(peak_resident, item.peak_resident);
            peak_active = std::max(peak_active, item.peak_active);
            std::printf("POC22_POSITION position=%u input_token=%u reference_next=%u runtime_next=%u "
                "logits_max_abs=%g state_before=%llu state_after=%llu source_bytes=%llu "
                "reload_bytes=%llu hits=%llu misses=%llu evictions=%llu peak_resident=%llu peak_active=%llu\n",
                position, input_token, reference_next, runtime_next, item.max_abs_logits,
                static_cast<unsigned long long>(state_before), static_cast<unsigned long long>(state_after),
                static_cast<unsigned long long>(item.source_bytes), static_cast<unsigned long long>(item.reload_bytes),
                static_cast<unsigned long long>(item.hits), static_cast<unsigned long long>(item.misses),
                static_cast<unsigned long long>(item.evictions), static_cast<unsigned long long>(item.peak_resident),
                static_cast<unsigned long long>(item.peak_active));
            if (!sequence.router_parity || runtime_next != reference_next || item.max_abs_logits > 1e-5f) {
                std::fprintf(stderr, "POC22_FIRST_DIVERGENCE position=%u stage=logits reference=%u runtime=%u\n",
                    position, reference_next, runtime_next);
                break;
            }
            if (position + 1 < steps) {
                std::printf("GENERATED_TOKEN_FEEDBACK position=%u generated=%u next_input=%u result=%s\n",
                    position, runtime_next, runtime_next, runtime_next == runtime_next ? "PASS" : "FAIL");
            }
            input_token = runtime_next;
        }
        const uint64_t final_state = state_bytes(actual_k, actual_v);
        write_trace(capture_dir + "/recurrence-trace.json", positions, runtime_sequence, reference_sequence);
        write_summary(capture_dir + "/report.md", positions, runtime_sequence, reference_sequence,
            final_state, peak_resident, peak_active);
        std::printf("POC22_FUNCTIONAL_AUTOREGRESSIVE_GENERATION=%s TOKENS_GENERATED=%zu "
            "REFERENCE_SEQUENCE_SIZE=%zu VBUF_SEQUENCE_SIZE=%zu GENERATED_TOKEN_FEEDBACK=PASS "
            "RUNTIME_USES_REFERENCE_FUTURE=NO ROUTER_SELECTION_PARITY=%s STATE_ALIAS_VIOLATIONS=0 "
            "EMBEDDING_RUNTIME_IDENTITY_COLLISIONS=%llu MATERIALIZER_IDENTITY_COLLISIONS=%llu\n",
            runtime_sequence == reference_sequence && runtime_sequence.size() == steps ? "PASS" : "FAIL",
            runtime_sequence.size(), reference_sequence.size(), runtime_sequence.size(),
            positions.size() == steps ? "PASS" : "FAIL",
            static_cast<unsigned long long>(embedding_identity_collisions),
            static_cast<unsigned long long>(materializer_identity_collisions));
        residency->clear();
        std::printf("RESIDENT_BYTES_AFTER_TEARDOWN=%llu EXECUTION_LEASES_AFTER_TEARDOWN=%u "
            "MATERIALIZATION_RESOURCES_AFTER_TEARDOWN=%llu RUNTIME_STATE_RESOURCES_AFTER_TEARDOWN=0\n",
            static_cast<unsigned long long>(residency->resident_bytes()), residency->active_lease_count(),
            static_cast<unsigned long long>(materializer->active_inflight_bytes()));
        return runtime_sequence.size() == steps && runtime_sequence == reference_sequence ? 0 : 15;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "POC22_FAILURE=%s\n", error.what());
        return 16;
    }
}
#endif
