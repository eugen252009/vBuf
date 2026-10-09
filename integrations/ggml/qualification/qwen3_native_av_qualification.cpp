#include "qwen3_attention_av.h"
#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml;
using namespace vbuf_ml::numerics;
namespace {
constexpr uint32_t prompt_rows = 32;
constexpr uint64_t expected_source_size = 9000232144ULL;
constexpr uint64_t bytes_per_v_copy = 8ULL * 128 * sizeof(ggml_fp16_t);
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

uint64_t token_hash(const std::vector<uint32_t> & tokens) {
    uint64_t hash = 14695981039346656037ULL;
    const auto update = [&](uint32_t value, uint64_t * current) {
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            *current ^= static_cast<uint8_t>(value >> shift);
            *current *= 1099511628211ULL;
        }
    };
    update(static_cast<uint32_t>(tokens.size()), &hash);
    for (uint32_t token : tokens) update(token, &hash);
    return hash;
}

std::vector<uint32_t> read_tokens(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open token IDs: " + path);
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string normalized;
    normalized.reserve(text.size());
    for (unsigned char c : text) normalized.push_back(c >= '0' && c <= '9' ? static_cast<char>(c) : ' ');
    std::istringstream input(normalized);
    std::vector<uint32_t> result;
    uint64_t token = 0;
    while (input >> token) {
        if (token > UINT32_MAX) throw std::out_of_range("token ID exceeds u32");
        result.push_back(static_cast<uint32_t>(token));
    }
    if (result.size() < prompt_rows) throw std::runtime_error("token fixture has fewer than 32 tokens");
    return result;
}

struct Metrics { double max_abs = 0.0, relative_rms = 0.0, cosine = 0.0; };
Metrics compare(const std::vector<float> & reference, const std::vector<float> & candidate) {
    require(reference.size() == candidate.size() && !reference.empty(), "activation comparison shape mismatch");
    EvaluationContext context;
    context.output_shape = {reference.size()};
    const auto measured = measure_tensor_pair(make_tensor_view(reference, context.output_shape),
        make_tensor_view(candidate, context.output_shape), context);
    require(*measured.values.at(Metric::FiniteOutputs).boolean_value,
        "activation comparison has non-finite values");
    Metrics result;
    result.max_abs = *measured.values.at(Metric::MaxAbsoluteError).numeric_value;
    result.relative_rms = measured.values.at(Metric::RelativeRmsError).numeric_value.value_or(
        std::numeric_limits<double>::quiet_NaN());
    result.cosine = measured.values.at(Metric::CosineSimilarity).numeric_value.value_or(
        std::numeric_limits<double>::quiet_NaN());
    return result;
}

NumericalEvaluation evaluate_model_output(const std::string & operation,
        const std::vector<float> & reference_values, const std::vector<float> & candidate_values,
        const QwenExecutionPlan & plan, uint32_t capacity, const std::vector<uint32_t> & prompt,
        const std::vector<uint32_t> & reference_tokens, const std::vector<uint32_t> & candidate_tokens) {
    EvaluationContext context;
    context.operation = operation;
    context.output_name = operation;
    context.reference_kind = ReferenceKind::CanonicalExecution;
    context.reference_identity = "canonical-packed-v-v1:native-av-qualification-control";
    context.model_identity = plan.model_identity;
    context.backend_family = plan.backend_family;
    context.implementation_identity = "qwen3-native-av-qualification-trial";
    context.device_family = "CUDA";
    context.device_identities = plan.stable_device_identities;
    for (const auto & sm : plan.stable_device_sm_versions) if (sm) context.device_sm_versions.push_back(*sm);
    context.placement_identity = plan.placement_identity;
    context.phase = "decode";
    context.execution_topology = "32-row-prefill-then-single-decode";
    context.fixture_identity = "native-av-qualification-prompt32";
    context.input_identity = "prompt-token-hash=" + std::to_string(token_hash(prompt)) +
        ";prompt-token-count=" + std::to_string(prompt.size());
    context.token_sequence_identity = "prompt-fnv64=" + std::to_string(token_hash(prompt)) +
        ";generated-fnv64=" + std::to_string(token_hash(reference_tokens));
    context.output_dtype = plan.activation_dtype;
    context.input_dtype = plan.kv_dtype;
    context.output_shape = {reference_values.size()};
    context.capacity = capacity;
    context.context_length = prompt_rows;
    context.rows = 1;
    context.logical_inputs_equivalent = true;
    context.reference_tokens = reference_tokens;
    context.candidate_tokens = candidate_tokens;
    const auto reference = make_tensor_view(reference_values, context.output_shape);
    const auto candidate = make_tensor_view(candidate_values, context.output_shape);
    const std::string contract_id = operation == "final_hidden" ?
        "qwen3.final_hidden.canonical_compatibility" : "qwen3.final_logits.canonical_compatibility";
    return evaluate_contract(contract_id, 1, &reference, &candidate, context);
}

bool bitwise_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

std::vector<uint8_t> read_kv_row(QwenCudaRuntimeState & runtime, QwenCudaSessionState & session,
        uint32_t layer, bool key, uint32_t position) {
    ggml_tensor * tensor = key ? session.key_cache(layer) : session.value_cache(layer);
    require(tensor != nullptr, "missing KV cache during native AV qualification");
    std::vector<uint8_t> data(static_cast<size_t>(bytes_per_v_copy));
    const uint32_t owner = runtime.placement().block_device_ids[layer];
    ggml_backend_synchronize(runtime.backend(owner));
    ggml_backend_tensor_get(tensor, data.data(),
        static_cast<size_t>(position) * bytes_per_v_copy, data.size());
    return data;
}

bool finite_f16(const std::vector<uint8_t> & bytes) {
    for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
        uint16_t h = 0;
        std::memcpy(&h, bytes.data() + i, sizeof(h));
        if ((h & 0x7c00U) == 0x7c00U) return false;
    }
    return true;
}

struct Snapshot {
    Qwen3GenerationExecution execution;
    uint64_t graph_setup_ns = 0;
    uint64_t wall_ns = 0;
    std::vector<std::vector<uint8_t>> sampled_kv;
};

Snapshot run_once(Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
        uint32_t capacity, const std::vector<uint32_t> & prompt, bool native) {
    Snapshot result;
    runtime->execution_optimizer().set_mode(native ? QwenOptimizerMode::Enabled : QwenOptimizerMode::Disabled);
    auto session = runtime->create_session(capacity);
    const auto setup_start = Clock::now();
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, true, native);
    const uint64_t setup_ns = elapsed_ns(setup_start, Clock::now());
    const auto run_start = Clock::now();
    result.execution = executor.run(prompt, 1);
    const uint64_t wall_ns = elapsed_ns(run_start, Clock::now());
    result.graph_setup_ns = setup_ns;
    result.wall_ns = wall_ns;
    std::printf("path=%s graph_setup_ns=%llu wall_ns=%llu prefill_ns=%llu decode_ns=%llu "
        "native_av_steps=%llu native_av_layers=%llu packed_v_copy_bytes_avoided=%llu "
        "boundary_handoffs=%llu current_length=%u completed=%s\n",
        native ? "native-av-trial" : "canonical-packed",
        static_cast<unsigned long long>(setup_ns), static_cast<unsigned long long>(wall_ns),
        static_cast<unsigned long long>(result.execution.prefill_ns),
        static_cast<unsigned long long>(result.execution.decode_ns),
        static_cast<unsigned long long>(result.execution.native_av_steps),
        static_cast<unsigned long long>(result.execution.native_av_layers),
        static_cast<unsigned long long>(result.execution.packed_v_copy_bytes_avoided),
        static_cast<unsigned long long>(result.execution.boundary_handoffs),
        session->current_length(), result.execution.completed ? "yes" : "no");
    require(result.execution.completed && session->current_length() == prompt_rows + 1,
        "Qwen run did not commit the expected prefill+decode positions");
    if (native) {
        require(result.execution.native_av_steps == 2 && result.execution.native_av_layers == 52,
            "native AV was not selected for both 32-row prefill and one-row decode on exactly 26 layers");
        const uint64_t expected_bytes = 2ULL * 26 * capacity * bytes_per_v_copy;
        require(result.execution.packed_v_copy_bytes_avoided == expected_bytes,
            "reported packed-V copy elimination does not match native graph coverage");
    } else {
        require(result.execution.native_av_steps == 0 && result.execution.packed_v_copy_bytes_avoided == 0,
            "canonical baseline unexpectedly selected native AV");
    }
    for (uint32_t layer : {0U, 25U, 26U, 39U}) {
        for (uint32_t position : {0U, 31U, 32U}) {
            result.sampled_kv.push_back(read_kv_row(*runtime, *session, layer, true, position));
            result.sampled_kv.push_back(read_kv_row(*runtime, *session, layer, false, position));
        }
    }
    for (const auto & row : result.sampled_kv) require(finite_f16(row), "sampled KV row is non-finite");
    return result;
}

void test_candidate_failure_recovery(Qwen3Model & model,
        const std::shared_ptr<QwenCudaRuntimeState> & runtime, uint32_t capacity,
        const std::vector<uint32_t> & prompt, const QwenExecutionPlan & plan) {
    auto & optimizer = runtime->execution_optimizer();
    const std::string prefill_id = qwen3_native_attention_av_candidate_identity(plan, QwenExecutionPhase::Prefill);
    auto failing_session = runtime->create_session(capacity, QwenCudaFailurePoint::ExecutionBeforeBoundary);
    Qwen3MultiDeviceGenerationExecutor failing_executor(model, runtime, failing_session, false, true);
    bool failed = false;
    try { (void) failing_executor.run(prompt, 1); }
    catch (const std::exception & error) {
        failed = true;
        std::printf("native_failure_injection error=%s\n", error.what());
    }
    require(failed && failing_session->current_length() == 0,
        "partial native AV failure did not leave logical session progress uncommitted");
    const auto invalidation = optimizer.snapshot();
    require(invalidation.invalidated_candidate_count >= 1,
        "native AV execution failure did not invalidate its candidate");
    std::printf("native_failure_recovery logical_length=%u invalidated_candidates=%zu PASS\n",
        failing_session->current_length(), invalidation.invalidated_candidate_count);

    QwenRuntimeFacts facts = collect_qwen_runtime_facts(plan, *runtime, prompt_rows, 0,
        QwenExecutionPhase::Prefill);
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    const auto invalidated = optimizer.select(plan, facts, prefill_id);
    require(invalidated.fallback == QwenOptimizerFallback::Invalidated && invalidated.canonical_selected,
        "invalidated native candidate remained selectable");
    optimizer.set_mode(QwenOptimizerMode::Disabled);
    auto recovery_session = runtime->create_session(capacity);
    Qwen3MultiDeviceGenerationExecutor recovery_executor(model, runtime, recovery_session, false, false);
    const auto recovered = recovery_executor.run(prompt, 1);
    require(recovered.completed && recovered.native_av_steps == 0 && recovery_session->current_length() == prompt_rows + 1,
        "fresh canonical session did not recover after the failed native request");
    std::puts("native_failure_recovery fresh_canonical_session=PASS");
}

int run(const std::string & semantic, const std::string & source, const std::string & token_file,
        uint32_t capacity) {
    const auto tokens = read_tokens(token_file);
    if (capacity < prompt_rows + 1 || capacity > 512)
        throw std::invalid_argument("native AV candidate is guarded to capacities 33..512 pending full-capacity numeric qualification");
    const std::vector<uint32_t> prompt(tokens.begin(), tokens.begin() + prompt_rows);

    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256 &&
        model.source_size == expected_source_size && model.layer_count == 40 && model.count == 443,
        "native AV qualification requires the exact admitted Qwen3-14B Q4_K_M artifact");

    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, 26);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "native AV qualification did not construct the 26/14 two-device runtime");
    const auto plan = build_qwen_execution_plan(model, *runtime,
        *runtime->create_session(capacity), QwenExecutionPlanPath::MultiGpu);
    auto prefill_candidate = make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Prefill);
    auto decode_candidate = make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Decode);

    std::printf("identity model=%s capacity=%u placement=%s devices=%s/%s\n",
        model.artifact_identity.c_str(), capacity, plan.placement_identity.c_str(),
        ggml_backend_dev_name(runtime->device(0)), ggml_backend_dev_name(runtime->device(1)));
    std::printf("av_contract cache=[dimension=128, position*8+head] physical=[position,kv_head,dimension] "
        "weights=[position,query,head] reduction=ascending-visible-position\n");

    auto & optimizer = runtime->execution_optimizer();
    optimizer.set_mode(QwenOptimizerMode::Disabled);
    Snapshot canonical = run_once(model, runtime, capacity, prompt, false);

    require(optimizer.register_candidate(prefill_candidate) && optimizer.register_candidate(decode_candidate),
        "could not register explicit native AV trial candidates");
    optimizer.set_unvalidated_trial_for_testing(true);
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    Snapshot candidate = run_once(model, runtime, capacity, prompt, true);
    const auto decision = optimizer.snapshot().last_decision;
    require(decision.candidate_selected && decision.qualification_trial &&
        decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV,
        "native AV execution was not selected through the explicit guarded qualification trial");

    const Metrics hidden = compare(canonical.execution.final_hidden, candidate.execution.final_hidden);
    const Metrics logits = compare(canonical.execution.final_logits, candidate.execution.final_logits);
    require(canonical.execution.tokens == candidate.execution.tokens,
        "canonical and native AV generated different token IDs");
    const auto hidden_contract = evaluate_model_output("final_hidden", canonical.execution.final_hidden,
        candidate.execution.final_hidden, plan, capacity, prompt, canonical.execution.tokens,
        candidate.execution.tokens);
    const auto logits_contract = evaluate_model_output("final_logits", canonical.execution.final_logits,
        candidate.execution.final_logits, plan, capacity, prompt, canonical.execution.tokens,
        candidate.execution.tokens);
    const bool numeric_gate_pass = hidden_contract.status == EvaluationStatus::Pass &&
        logits_contract.status == EvaluationStatus::Pass;
    std::printf("numerical_contract_live_result %s\n", evaluation_json(hidden_contract).c_str());
    std::printf("numerical_contract_live_result %s\n", evaluation_json(logits_contract).c_str());
    std::printf("native_trial_numeric_probe tokens_equal=%s hidden_max_abs=%.9g hidden_rel_rms=%.9g hidden_cosine=%.12g "
        "logits_max_abs=%.9g logits_rel_rms=%.9g logits_cosine=%.12g numeric_gate=%s qualification_trial=yes\n",
        canonical.execution.tokens == candidate.execution.tokens ? "yes" : "no",
        hidden.max_abs, hidden.relative_rms, hidden.cosine,
        logits.max_abs, logits.relative_rms, logits.cosine, numeric_gate_pass ? "PASS" : "FAIL");
    require(numeric_gate_pass,
        "canonical/native full-model activation parity exceeded the existing small-context numeric gate");
    size_t changed_kv_bytes = 0, compared_kv_bytes = 0;
    for (size_t i = 0; i < canonical.sampled_kv.size(); ++i) {
        require(canonical.sampled_kv[i].size() == candidate.sampled_kv[i].size(), "sampled KV size mismatch");
        compared_kv_bytes += canonical.sampled_kv[i].size();
        for (size_t j = 0; j < canonical.sampled_kv[i].size(); ++j)
            changed_kv_bytes += canonical.sampled_kv[i][j] != candidate.sampled_kv[i][j];
    }
    std::printf("real_qwen_compare tokens_equal=yes hidden_max_abs=%.9g hidden_rel_rms=%.9g hidden_cosine=%.12g "
        "logits_max_abs=%.9g logits_rel_rms=%.9g logits_cosine=%.12g sampled_kv_changed_bytes=%zu/%zu "
        "numeric_gate=PASS qualification_trial=yes candidate_status=Candidate_NOT_Valid\n",
        hidden.max_abs, hidden.relative_rms, hidden.cosine,
        logits.max_abs, logits.relative_rms, logits.cosine, changed_kv_bytes, compared_kv_bytes);

    std::vector<uint64_t> canonical_wall{canonical.wall_ns}, native_wall{candidate.wall_ns};
    std::vector<uint64_t> canonical_setup{canonical.graph_setup_ns}, native_setup{candidate.graph_setup_ns};
    for (uint32_t sample = 2; sample <= 3; ++sample) {
        Snapshot canonical_repeat;
        Snapshot native_repeat;
        if ((sample % 2) == 0) {
            native_repeat = run_once(model, runtime, capacity, prompt, true);
            canonical_repeat = run_once(model, runtime, capacity, prompt, false);
        } else {
            canonical_repeat = run_once(model, runtime, capacity, prompt, false);
            native_repeat = run_once(model, runtime, capacity, prompt, true);
        }
        require(canonical_repeat.execution.tokens == canonical.execution.tokens &&
            native_repeat.execution.tokens == candidate.execution.tokens,
            "repeated execution changed a path's generated token IDs");
        const Metrics repeated_hidden = compare(canonical_repeat.execution.final_hidden,
            native_repeat.execution.final_hidden);
        const Metrics repeated_logits = compare(canonical_repeat.execution.final_logits,
            native_repeat.execution.final_logits);
        const auto repeated_hidden_contract = evaluate_model_output("final_hidden",
            canonical_repeat.execution.final_hidden, native_repeat.execution.final_hidden, plan, capacity,
            prompt, canonical_repeat.execution.tokens, native_repeat.execution.tokens);
        const auto repeated_logits_contract = evaluate_model_output("final_logits",
            canonical_repeat.execution.final_logits, native_repeat.execution.final_logits, plan, capacity,
            prompt, canonical_repeat.execution.tokens, native_repeat.execution.tokens);
        require(repeated_hidden_contract.status == EvaluationStatus::Pass &&
            repeated_logits_contract.status == EvaluationStatus::Pass,
            "repeated canonical/native execution exceeded the centralized small-context numeric gate");
        canonical_wall.push_back(canonical_repeat.wall_ns);
        native_wall.push_back(native_repeat.wall_ns);
        canonical_setup.push_back(canonical_repeat.graph_setup_ns);
        native_setup.push_back(native_repeat.graph_setup_ns);
        std::printf("paired_sample=%u order=%s canonical_wall_ns=%llu native_wall_ns=%llu "
            "canonical_setup_ns=%llu native_setup_ns=%llu hidden_rel_rms=%.9g logits_rel_rms=%.9g parity=PASS\n",
            sample, (sample % 2) == 0 ? "native-first" : "canonical-first",
            static_cast<unsigned long long>(canonical_repeat.wall_ns),
            static_cast<unsigned long long>(native_repeat.wall_ns),
            static_cast<unsigned long long>(canonical_repeat.graph_setup_ns),
            static_cast<unsigned long long>(native_repeat.graph_setup_ns),
            repeated_hidden.relative_rms, repeated_logits.relative_rms);
    }
    const auto median = [](std::vector<uint64_t> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const uint64_t canonical_median = median(canonical_wall);
    const uint64_t native_median = median(native_wall);
    std::printf("timing_summary samples=3 scope=executor_run_including_prefill_decode setup_excluded "
        "canonical_wall_median_ns=%llu native_wall_median_ns=%llu ratio_canonical_over_native=%.6f "
        "canonical_graph_setup_median_ns=%llu native_graph_setup_median_ns=%llu "
        "performance_status=exploratory_small_capacity_not_production_qualification\n",
        static_cast<unsigned long long>(canonical_median), static_cast<unsigned long long>(native_median),
        static_cast<double>(canonical_median) / std::max<uint64_t>(native_median, 1),
        static_cast<unsigned long long>(median(canonical_setup)),
        static_cast<unsigned long long>(median(native_setup)));

    test_candidate_failure_recovery(model, runtime, capacity, prompt, plan);
    optimizer.set_unvalidated_trial_for_testing(false);
    optimizer.set_mode(QwenOptimizerMode::Shadow);
    std::puts("QWEN_NATIVE_LAYOUT_AV_SMALL_GATE=PASS model_capacity=PASS direct_tests=see_ctest "
        "real_activations=PASS session_failure=PASS candidate=EXPERIMENTAL_NOT_PRODUCTION_VALIDATED");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 5) throw std::invalid_argument(
            "usage: qwen3_native_av_qualification SEMANTIC.vbuf SOURCE_URL TOKEN_IDS_FILE CAPACITY");
        return run(argv[1], argv[2], argv[3], static_cast<uint32_t>(std::stoul(argv[4])));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3_native_av_qualification=FAIL: %s\n", error.what());
        return 1;
    }
}
