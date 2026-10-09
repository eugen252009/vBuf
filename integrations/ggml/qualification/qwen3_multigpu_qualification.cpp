#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml;
using namespace vbuf_ml::numerics;
namespace {
constexpr uint64_t expected_source_size = 9000232144ULL;
constexpr uint64_t expected_payload = 8995793920ULL;
constexpr uint64_t row_kv_bytes = 2048;
constexpr uint64_t boundary_bytes_per_position = 5120ULL * sizeof(float);

uint64_t token_hash(const std::vector<uint32_t> & tokens) {
    uint64_t hash = 14695981039346656037ULL;
    const auto update = [&](uint32_t value) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    };
    update(static_cast<uint32_t>(tokens.size()));
    for (uint32_t token : tokens) update(token);
    return hash;
}

std::vector<uint32_t> parse_tokens(const std::string & text) {
    std::vector<uint32_t> result;
    size_t start = 0;
    while (start < text.size()) {
        const size_t comma = text.find(',', start);
        const std::string part = text.substr(start, comma == std::string::npos ? comma : comma - start);
        if (part.empty()) throw std::invalid_argument("empty token ID");
        const unsigned long value = std::stoul(part);
        if (value > UINT32_MAX) throw std::out_of_range("token ID exceeds u32");
        result.push_back(static_cast<uint32_t>(value));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return result;
}

struct Metrics { double max_abs = 0, rel_rms = 0, cosine = 0; };
Metrics compare(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) throw std::runtime_error("comparison vector geometry mismatch");
    EvaluationContext context;
    context.output_shape = {a.size()};
    const auto measured = measure_tensor_pair(make_tensor_view(a, context.output_shape),
        make_tensor_view(b, context.output_shape), context);
    if (!*measured.values.at(Metric::FiniteOutputs).boolean_value)
        throw std::runtime_error("non-finite comparison value");
    Metrics result;
    result.max_abs = *measured.values.at(Metric::MaxAbsoluteError).numeric_value;
    result.rel_rms = measured.values.at(Metric::RelativeRmsError).numeric_value.value_or(
        std::numeric_limits<double>::quiet_NaN());
    result.cosine = measured.values.at(Metric::CosineSimilarity).numeric_value.value_or(
        std::numeric_limits<double>::quiet_NaN());
    return result;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

NumericalEvaluation evaluate_model_output(const std::string & operation,
        const std::vector<float> & reference_values, const std::vector<float> & candidate_values,
        const QwenExecutionPlan & plan, const QwenRuntimeFacts & facts,
        const std::vector<uint32_t> & reference_tokens, const std::vector<uint32_t> & candidate_tokens,
        const std::string & fixture_identity, const std::string & input_identity) {
    EvaluationContext context;
    context.operation = operation;
    context.output_name = operation;
    context.reference_kind = ReferenceKind::CanonicalExecution;
    context.reference_identity = "canonical-packed-v-v1:single-device-reference";
    context.model_identity = plan.model_identity;
    context.backend_family = plan.backend_family;
    context.implementation_identity = "qwen3-multigpu-generation";
    context.device_family = "CUDA";
    context.device_identities = plan.stable_device_identities;
    for (const auto & sm_version : plan.stable_device_sm_versions)
        if (sm_version) context.device_sm_versions.push_back(*sm_version);
    context.placement_identity = plan.placement_identity;
    context.phase = "decode";
    context.execution_topology = "prefill-then-single-decode";
    context.fixture_identity = fixture_identity;
    context.input_identity = input_identity;
    context.token_sequence_identity = input_identity;
    context.output_dtype = plan.activation_dtype;
    context.input_dtype = plan.kv_dtype;
    context.output_shape = {reference_values.size()};
    context.capacity = facts.capacity;
    context.context_length = facts.context_length;
    context.rows = facts.rows;
    context.logical_inputs_equivalent = true;
    context.reference_tokens = reference_tokens;
    context.candidate_tokens = candidate_tokens;
    const auto reference = make_tensor_view(reference_values, context.output_shape);
    const auto candidate = make_tensor_view(candidate_values, context.output_shape);
    const std::string contract_id = operation == "final_hidden" ?
        "qwen3.final_hidden.canonical_compatibility" : "qwen3.final_logits.canonical_compatibility";
    return evaluate_contract(contract_id, 1, &reference, &candidate, context);
}

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

std::vector<uint8_t> cache_row(QwenCudaRuntimeState & runtime, QwenCudaSessionState & session,
    uint32_t layer, bool key, uint32_t position) {
    ggml_tensor * tensor = key ? session.key_cache(layer) : session.value_cache(layer);
    std::vector<uint8_t> result(row_kv_bytes);
    const uint32_t owner = runtime.placement().block_device_ids[layer];
    (void) owner;
    ggml_backend_tensor_get(tensor, result.data(), static_cast<size_t>(position) * row_kv_bytes, result.size());
    return result;
}
bool finite_f16_rows(const std::vector<uint8_t> & data) {
    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        uint16_t h = 0; std::memcpy(&h, data.data() + i, sizeof(h));
        if ((h & 0x7c00U) == 0x7c00U) return false;
    }
    return true;
}

struct SavedOutput { Qwen3GenerationExecution execution; };

std::vector<uint32_t> prefixes{1, 2, 4, 8, 16, 32};

int run(const char * semantic, const char * source, const char * token_text, uint32_t capacity) {
    const std::vector<uint32_t> tokens = parse_tokens(token_text);
    if (tokens.size() < 32 || capacity < 33 || capacity > 1032)
        throw std::invalid_argument("initial multi-GPU qualification requires >=32 tokens and capacity 33..1032");
    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256,
        "wrong admitted semantic artifact");
    require(model.count == 443 && model.layer_count == 40 && model.source_size == expected_source_size,
        "admitted Qwen model geometry/source-size mismatch");

    std::vector<SavedOutput> reference;
    {
        auto runtime = QwenCudaRuntimeState::create(model);
        auto session = runtime->create_session(capacity);
        Qwen3GenerationExecutor executor(model, runtime, session, true);
        for (uint32_t prefix : prefixes) {
            const std::vector<uint32_t> prompt(tokens.begin(), tokens.begin() + prefix);
            SavedOutput item;
            item.execution = executor.run(prompt, 1);
            require(item.execution.completed && item.execution.tokens.size() == 1,
                "single-device canonical reference failed to complete");
            reference.push_back(std::move(item));
            std::printf("single_reference prefix=%u greedy=%u logits=%zu h2d=%llu d2h=%llu\n",
                prefix, reference.back().execution.tokens[0], reference.back().execution.final_logits.size(),
                static_cast<unsigned long long>(reference.back().execution.h2d_calls),
                static_cast<unsigned long long>(reference.back().execution.d2h_calls));
        }
    }

    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, 26);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "runtime did not own two CUDA backends");
    uint64_t combined = 0;
    for (uint32_t id : {0U, 1U}) {
        const uint64_t bytes = runtime->uploaded_payload_bytes(id);
        combined += bytes;
        std::printf("device_residency device=%u name=%s tensors=%zu payload_bytes=%llu allocation_bytes=%zu\n",
            id, ggml_backend_dev_name(runtime->device(id)), runtime->uploaded_tensor_count(id),
            static_cast<unsigned long long>(bytes), runtime->resident_allocation_bytes(id));
    }
    require(combined == expected_payload && runtime->uploaded_payload_bytes() == expected_payload,
        "per-device payload does not reconcile to admitted model bytes");
    require(runtime->uploaded_tensor_count() == model.count && runtime->resident_tensor_count() == model.count,
        "model tensor residency count is incomplete");
    for (const auto & entry : model.tensors) {
        const uint32_t owner = runtime->owner_device(entry.first);
        require(runtime->tensor(entry.first, owner) != nullptr, "owned model tensor missing on owner: " + entry.first);
        for (uint32_t other : {0U, 1U}) if (other != owner)
            require(runtime->tensor(entry.first, other) == nullptr, "model tensor duplicated across devices: " + entry.first);
    }

    auto session = runtime->create_session(capacity);
    for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
        const uint32_t owner = runtime->placement().block_device_ids[layer];
        ggml_tensor * key = session->key_cache(layer, owner);
        ggml_tensor * value = session->value_cache(layer, owner);
        require(key != nullptr && value != nullptr && key->type == GGML_TYPE_F16 && value->type == GGML_TYPE_F16 &&
            key->ne[0] == 128 && key->ne[1] == static_cast<int64_t>(8 * capacity) &&
            value->ne[0] == 128 && value->ne[1] == static_cast<int64_t>(8 * capacity),
            "device-local KV geometry/ownership mismatch");
        for (uint32_t other : {0U, 1U}) if (other != owner)
            require(session->key_cache(layer, other) == nullptr && session->value_cache(layer, other) == nullptr,
                "KV cache duplicated on non-owning device");
    }
    require(session->boundary_host_is_pinned(), "boundary staging is not pinned CUDA host memory");
    require(session->boundary_host_bytes() >= 32ULL * 5120 * sizeof(float), "boundary buffer is too small");
    std::printf("boundary pinned=yes bytes=%zu dtype=F32 hidden=5120\n", session->boundary_host_bytes());
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session);
    for (uint32_t id : {0U, 1U}) {
        size_t free = 0, total = 0; runtime->device_memory(id, &free, &total);
        std::printf("device_session device=%u kv_allocation_bytes=%zu prefill_scratch_bytes=%zu decode_scratch_bytes=%zu "
            "free_vram_bytes=%zu total_vram_bytes=%zu\n", id, session->allocation_bytes(id),
            session->prefill_scratch_bytes(id), session->decode_scratch_bytes(id), free, total);
    }

    for (size_t p = 0; p < prefixes.size(); ++p) {
        const uint32_t prefix = prefixes[p];
        const std::vector<uint32_t> prompt(tokens.begin(), tokens.begin() + prefix);
        std::vector<uint8_t> early_key, early_value, late_key, late_value;
        std::vector<uint8_t> early_key_first, early_value_first, late_key_first, late_value_first;
        bool callback_called = false;
        auto on_token = [&](uint32_t, uint32_t) {
            const uint32_t sample_position = prefix - 1;
            early_key = cache_row(*runtime, *session, 25, true, sample_position);
            early_value = cache_row(*runtime, *session, 25, false, sample_position);
            late_key = cache_row(*runtime, *session, 26, true, sample_position);
            late_value = cache_row(*runtime, *session, 26, false, sample_position);
            early_key_first = cache_row(*runtime, *session, 25, true, 0);
            early_value_first = cache_row(*runtime, *session, 25, false, 0);
            late_key_first = cache_row(*runtime, *session, 26, true, 0);
            late_value_first = cache_row(*runtime, *session, 26, false, 0);
            callback_called = true;
            return true;
        };
        Qwen3GenerationExecution disabled_reference;
        std::array<std::vector<uint8_t>, 8> disabled_kv;
        if (prefix == 32) {
            runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
            disabled_reference = executor.run(prompt, 1);
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t layer = i < 2 ? 25 : 26;
                const bool key = (i % 2) == 0;
                disabled_kv[i] = cache_row(*runtime, *session, layer, key, prefix - 1);
                disabled_kv[i + 4] = cache_row(*runtime, *session, layer, key, 0);
            }
            runtime->execution_optimizer().set_mode(QwenOptimizerMode::Shadow);
        }
        auto actual = executor.run(prompt, 1, std::nullopt, on_token);
        if (prefix == 32) {
            require(disabled_reference.tokens == actual.tokens && same(disabled_reference.final_logits, actual.final_logits) &&
                same(disabled_reference.final_hidden, actual.final_hidden),
                "optimizer-disabled and shadow executions differ in tokens/logits/hidden");
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t layer = i < 2 ? 25 : 26;
                const bool key = (i % 2) == 0;
                require(disabled_kv[i] == cache_row(*runtime, *session, layer, key, prefix - 1) &&
                    disabled_kv[i + 4] == cache_row(*runtime, *session, layer, key, 0),
                    "optimizer-disabled and shadow KV state differs");
            }
            std::puts("optimizer_mode_equivalence prefix=32 tensors=tokens+hidden+logits+KV bitwise=PASS");
        }
        require(actual.completed && actual.tokens.size() == 1 && callback_called,
            "two-device run failed to complete/capture KV");
        const uint64_t expected_handoffs = prefix == 32 ? 2 : static_cast<uint64_t>(prefix) + 1;
        const uint64_t expected_boundary_bytes = prefix == 32 ? 33ULL * boundary_bytes_per_position :
            expected_handoffs * boundary_bytes_per_position;
        require(actual.boundary_handoffs == expected_handoffs && actual.boundary_bytes == expected_boundary_bytes &&
            actual.boundary_bytes_equal, "boundary handoff count/bytes differ from expected staged path");
        const auto & expected = reference[p].execution;
        const Metrics hidden = compare(expected.final_hidden, actual.final_hidden);
        const Metrics logits = compare(expected.final_logits, actual.final_logits);
        require(actual.tokens[0] == expected.tokens[0], "greedy prompt token differs from canonical single-device runtime");
        require(static_cast<size_t>(std::max_element(actual.final_logits.begin(), actual.final_logits.end()) - actual.final_logits.begin()) ==
            static_cast<size_t>(std::max_element(expected.final_logits.begin(), expected.final_logits.end()) - expected.final_logits.begin()),
            "post-decode greedy token differs from canonical single-device runtime");
        const auto qualification_plan = build_qwen_execution_plan(model, *runtime, *session,
            QwenExecutionPlanPath::MultiGpu);
        const auto qualification_facts = collect_qwen_runtime_facts(qualification_plan, *runtime, 1,
            prefix, QwenExecutionPhase::Decode);
        const std::string fixture_identity = "qwen3-multigpu-prefix-" + std::to_string(prefix);
        const std::string input_identity = "prompt-token-hash=" + std::to_string(token_hash(prompt)) +
            ";prefix=" + std::to_string(prefix);
        const auto hidden_contract = evaluate_model_output("final_hidden", expected.final_hidden,
            actual.final_hidden, qualification_plan, qualification_facts, expected.tokens, actual.tokens,
            fixture_identity, input_identity);
        const auto logits_contract = evaluate_model_output("final_logits", expected.final_logits,
            actual.final_logits, qualification_plan, qualification_facts, expected.tokens, actual.tokens,
            fixture_identity, input_identity);
        require(hidden_contract.status == EvaluationStatus::Pass && logits_contract.status == EvaluationStatus::Pass &&
            *logits_contract.measured_metrics.values.at(Metric::Top1Equality).boolean_value &&
            *logits_contract.measured_metrics.values.at(Metric::TokenSequenceEquality).boolean_value,
            "cross-device output failed the scoped numerical contract or exact generated-token check");
        std::printf("numerical_contract_live_result %s\n", evaluation_json(hidden_contract).c_str());
        std::printf("numerical_contract_live_result %s\n", evaluation_json(logits_contract).c_str());
        require(actual.final_logits.size() == expected.final_logits.size(), "logit output geometry differs");
        require(std::all_of(actual.final_logits.begin(), actual.final_logits.end(), [](float v) { return std::isfinite(v); }),
            "multi-device logits are non-finite");
        require(finite_f16_rows(early_key) && finite_f16_rows(early_value) &&
            finite_f16_rows(late_key) && finite_f16_rows(late_value) &&
            finite_f16_rows(early_key_first) && finite_f16_rows(early_value_first) &&
            finite_f16_rows(late_key_first) && finite_f16_rows(late_value_first),
            "sampled device-local KV is non-finite");
        require(cache_row(*runtime, *session, 25, true, prefix - 1) == early_key &&
            cache_row(*runtime, *session, 25, false, prefix - 1) == early_value &&
            cache_row(*runtime, *session, 26, true, prefix - 1) == late_key &&
            cache_row(*runtime, *session, 26, false, prefix - 1) == late_value &&
            cache_row(*runtime, *session, 25, true, 0) == early_key_first &&
            cache_row(*runtime, *session, 25, false, 0) == early_value_first &&
            cache_row(*runtime, *session, 26, true, 0) == late_key_first &&
            cache_row(*runtime, *session, 26, false, 0) == late_value_first,
            "historical KV changed during incremental append");
        require(session->current_length() == prefix + 1, "global logical length did not commit exact completed positions");
        std::printf("two_device prefix=%u token=%u match=%s hidden_max_abs=%.9g hidden_rel_rms=%.9g hidden_cosine=%.12g "
            "logits_max_abs=%.9g logits_rel_rms=%.9g logits_cosine=%.12g boundary_handoffs=%llu "
            "boundary_bytes=%llu length=%u kv25=finite kv26=finite numeric_check=PASS vram_used_sum=%llu free_vram_sum=%llu\n",
            prefix, actual.tokens[0], actual.tokens[0] == expected.tokens[0] ? "YES" : "NO",
            hidden.max_abs, hidden.rel_rms, hidden.cosine, logits.max_abs, logits.rel_rms, logits.cosine,
            static_cast<unsigned long long>(actual.boundary_handoffs),
            static_cast<unsigned long long>(actual.boundary_bytes), session->current_length(),
            static_cast<unsigned long long>(actual.peak_vram_bytes),
            static_cast<unsigned long long>(actual.post_run_free_vram_bytes));
        auto replay = executor.run(prompt, 1);
        require(actual.tokens == replay.tokens && same(actual.final_logits, replay.final_logits) &&
            same(actual.final_hidden, replay.final_hidden), "two-device execution was not repeatable");
    }

    // Sequential session isolation A/B/A; both sessions share immutable weights, not KV or scratch.
    auto session_a = runtime->create_session(capacity);
    auto session_b = runtime->create_session(capacity);
    Qwen3MultiDeviceGenerationExecutor executor_a(model, runtime, session_a);
    Qwen3MultiDeviceGenerationExecutor executor_b(model, runtime, session_b);
    const std::vector<uint32_t> prompt_a(tokens.begin(), tokens.begin() + 4);
    const std::vector<uint32_t> prompt_b(tokens.begin() + 4, tokens.begin() + 8);
    const auto a1 = executor_a.run(prompt_a, 1);
    const auto a_key = cache_row(*runtime, *session_a, 25, true, 4);
    const auto a_value = cache_row(*runtime, *session_a, 26, false, 4);
    const auto b = executor_b.run(prompt_b, 1);
    require(session_a->current_length() == 5 && session_b->current_length() == 5 &&
        cache_row(*runtime, *session_a, 25, true, 4) == a_key &&
        cache_row(*runtime, *session_a, 26, false, 4) == a_value,
        "session B changed session A KV or logical progress");
    const auto a2 = executor_a.run(prompt_a, 1);
    require(a1.tokens == a2.tokens && same(a1.final_logits, a2.final_logits) &&
        same(a1.final_hidden, a2.final_hidden), "A/B/A session isolation/replay failed");
    require(!b.final_logits.empty(), "session B produced no output");
    std::printf("session_isolation=A/B/A PASS\n");

    for (const auto point : {QwenCudaFailurePoint::ExecutionBeforeBoundary,
             QwenCudaFailurePoint::ExecutionAfterBoundaryD2H,
             QwenCudaFailurePoint::ExecutionAfterBoundaryH2D,
             QwenCudaFailurePoint::ExecutionBeforeLateBlock,
             QwenCudaFailurePoint::ExecutionBeforeFinalBlock}) {
        auto failing_session = runtime->create_session(capacity, point);
        Qwen3MultiDeviceGenerationExecutor failing_executor(model, runtime, failing_session);
        bool failed = false;
        try { (void) failing_executor.run({tokens[0]}, 1); }
        catch (const std::exception &) { failed = true; }
        require(failed && failing_session->current_length() == 0,
            "injected partial multi-device failure committed logical progress");
        std::printf("failure_injection point=%u logical_length=%u PASS\n",
            static_cast<unsigned>(point), failing_session->current_length());
    }
    auto recovery_session = runtime->create_session(capacity);
    Qwen3MultiDeviceGenerationExecutor recovery_executor(model, runtime, recovery_session);
    const auto recovered = recovery_executor.run({tokens[0]}, 1);
    require(recovered.completed && recovered.tokens == reference[0].execution.tokens &&
        recovery_session->current_length() == 2, "fresh-session recovery after injected failures failed");
    const auto optimizer_state = runtime->execution_optimizer().snapshot();
    require(runtime->execution_optimizer().mode() == QwenOptimizerMode::Shadow &&
        optimizer_state.optimizer_errors == 0 && optimizer_state.last_decision.canonical_selected,
        "shadow optimizer failed or changed canonical selection during qualification");
    std::printf("optimizer_shadow mode=SHADOW cache_hits=%llu cache_misses=%llu guard_passes=%llu guard_failures=%llu "
        "errors=%llu observations=%llu canonical_selected=YES\n",
        static_cast<unsigned long long>(optimizer_state.cache_hits),
        static_cast<unsigned long long>(optimizer_state.cache_misses),
        static_cast<unsigned long long>(optimizer_state.guard_passes),
        static_cast<unsigned long long>(optimizer_state.guard_failures),
        static_cast<unsigned long long>(optimizer_state.optimizer_errors),
        static_cast<unsigned long long>(optimizer_state.observations));
    std::puts("failure_recovery=fresh-session PASS");
    std::puts("QWEN_MULTI_GPU_26_14_SMALL_CONTEXT=TESTED NOT_QUALIFIED_32K=NOT_TESTED production_capacity=1032_UNCHANGED");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("usage: qwen3_multigpu_qualification SEMANTIC.vbuf SOURCE_URL TOKEN_IDS_CSV CAPACITY");
        return run(argv[1], argv[2], argv[3], static_cast<uint32_t>(std::stoul(argv[4])));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3_multigpu_qualification=FAIL: %s\n", error.what());
        return 1;
    }
}
