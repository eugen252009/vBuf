#include "qwen3_attention_av.h"
#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vbuf_ggml;
namespace {
constexpr uint64_t expected_source_size = 9000232144ULL;
constexpr uint32_t max_candidate_context = 32;
constexpr double max_relative_rms = 0.02;
constexpr double min_cosine = 0.9998;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<uint32_t> read_tokens(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open token fixture: " + path);
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
    return result;
}

struct Metrics {
    double max_abs = 0.0;
    double relative_rms = 0.0;
    double cosine = 0.0;
};

Metrics compare(const std::vector<float> & reference, const std::vector<float> & candidate) {
    require(reference.size() == candidate.size() && !reference.empty(), "activation comparison shape mismatch");
    double error2 = 0.0, reference2 = 0.0, candidate2 = 0.0, dot = 0.0;
    Metrics result;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "activation comparison has non-finite values");
        const double a = reference[i], b = candidate[i], delta = b - a;
        result.max_abs = std::max(result.max_abs, std::abs(delta));
        error2 += delta * delta;
        reference2 += a * a;
        candidate2 += b * b;
        dot += a * b;
    }
    result.relative_rms = std::sqrt(error2 / std::max(reference2, 1e-300));
    result.cosine = dot / std::max(std::sqrt(reference2 * candidate2), 1e-300);
    return result;
}

struct Progress {
    uint32_t position = 0;
    std::vector<float> hidden;
};

struct Run {
    Qwen3GenerationExecution execution;
    std::vector<Progress> progress;
    uint32_t current_length = 0;
};

Run execute(Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
        uint32_t capacity, const std::vector<uint32_t> & prompt, uint32_t decode_steps,
        bool native, const std::vector<uint32_t> * expected_tokens = nullptr,
        uint32_t * first_mismatch_position = nullptr) {
    Run result;
    auto session = runtime->create_session(capacity);
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, false, native);
    size_t token_index = 0;
    const auto on_token = [&](uint32_t token, uint32_t position) {
        if (expected_tokens != nullptr) {
            if (token_index >= expected_tokens->size() || token != expected_tokens->at(token_index)) {
                if (first_mismatch_position != nullptr) *first_mismatch_position = position;
                std::printf("sequence_token_mismatch position=%u candidate_token=%u canonical_token=%u\n",
                    position, token, token_index < expected_tokens->size() ? expected_tokens->at(token_index) : UINT32_MAX);
                return false;
            }
        }
        ++token_index;
        return true;
    };
    const auto on_progress = [&](uint32_t position, const std::vector<float> & hidden) {
        result.progress.push_back(Progress{position, hidden});
    };
    std::function<bool(uint32_t, uint32_t)> token_callback;
    if (expected_tokens != nullptr) token_callback = on_token;
    result.execution = executor.run(prompt, decode_steps, std::nullopt,
        token_callback, {}, on_progress);
    result.current_length = session->current_length();
    return result;
}

int run(const std::string & semantic, const std::string & source, const std::string & token_file,
        uint32_t prompt_rows, uint32_t decode_steps, uint32_t capacity, const std::string & csv_path) {
    const auto tokens = read_tokens(token_file);
    if (prompt_rows == 0 || prompt_rows >= 32 || prompt_rows > tokens.size() || decode_steps == 0 ||
        prompt_rows + decode_steps > max_candidate_context + 1 || capacity < prompt_rows + decode_steps || capacity > 512)
        throw std::invalid_argument("sequence must use a sub-32-token prompt, stay within decode context 32, and fit capacity <=512");
    const std::vector<uint32_t> prompt(tokens.begin(), tokens.begin() + prompt_rows);

    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256 &&
        model.source_size == expected_source_size && model.layer_count == 40 && model.count == 443,
        "sequence qualification requires the admitted Qwen3-14B Q4_K_M artifact");

    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, 26);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "sequence qualification requires the existing 26/14 two-device placement");
    const auto plan = [&] {
        auto plan_session = runtime->create_session(capacity);
        return build_qwen_execution_plan(model, *runtime, *plan_session, QwenExecutionPlanPath::MultiGpu);
    }();
    require(runtime->execution_optimizer().register_candidate(
        make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Prefill)) &&
        runtime->execution_optimizer().register_candidate(
        make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Decode)),
        "could not register guarded native AV qualification candidates");

    std::printf("sequence_qualification model=%s capacity=%u prompt_rows=%u decode_steps=%u "
        "last_native_decode_context=%u placement=%s devices=%s/%s\n",
        model.artifact_identity.c_str(), capacity, prompt_rows, decode_steps,
        prompt_rows + decode_steps - 1, plan.placement_identity.c_str(),
        ggml_backend_dev_name(runtime->device(0)), ggml_backend_dev_name(runtime->device(1)));

    auto & optimizer = runtime->execution_optimizer();
    optimizer.set_mode(QwenOptimizerMode::Disabled);
    const Run canonical = execute(model, runtime, capacity, prompt, decode_steps, false);
    require(canonical.execution.completed && canonical.execution.tokens.size() == decode_steps &&
        canonical.current_length == prompt_rows + decode_steps && canonical.execution.native_av_steps == 0,
        "canonical sequence did not complete without selecting native AV");

    optimizer.set_unvalidated_trial_for_testing(true);
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    uint32_t first_mismatch_position = UINT32_MAX;
    const Run candidate = execute(model, runtime, capacity, prompt, decode_steps, true,
        &canonical.execution.tokens, &first_mismatch_position);
    const auto final_decision = optimizer.snapshot().last_decision;
    require(final_decision.candidate_selected && final_decision.qualification_trial &&
        final_decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV,
        "sequence did not remain on the explicit unvalidated native AV qualification trial");
    const bool same_tokens = candidate.execution.tokens == canonical.execution.tokens;
    const bool complete_same_sequence = candidate.execution.completed && same_tokens &&
        candidate.current_length == canonical.current_length;
    const uint64_t expected_native_steps = prompt_rows + (same_tokens ? decode_steps :
        candidate.execution.tokens.empty() ? 0 : candidate.execution.tokens.size() - 1);

    std::filesystem::path csv(csv_path);
    if (csv.has_parent_path()) std::filesystem::create_directories(csv.parent_path());
    std::ofstream out(csv);
    if (!out) throw std::runtime_error("cannot create progress CSV: " + csv_path);
    out << "position,hidden_max_abs,hidden_relative_rms,hidden_cosine,token_sequence_equal\n";
    double worst_progress_rms = 0.0;
    uint32_t worst_progress_position = 0;
    size_t comparable_progress = 0;
    for (size_t i = 0; i < std::min(canonical.progress.size(), candidate.progress.size()); ++i) {
        require(canonical.progress[i].position == candidate.progress[i].position,
            "canonical and candidate progress positions diverged before token mismatch");
        const Metrics metric = compare(canonical.progress[i].hidden, candidate.progress[i].hidden);
        if (metric.relative_rms > worst_progress_rms) {
            worst_progress_rms = metric.relative_rms;
            worst_progress_position = canonical.progress[i].position;
        }
        ++comparable_progress;
        out << canonical.progress[i].position << ',' << metric.max_abs << ',' << metric.relative_rms << ','
            << metric.cosine << ',' << (same_tokens ? "yes" : "no") << '\n';
    }
    out.close();

    Metrics hidden{0.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    Metrics logits = hidden;
    bool numeric_gate = false;
    if (complete_same_sequence) {
        hidden = compare(canonical.execution.final_hidden, candidate.execution.final_hidden);
        logits = compare(canonical.execution.final_logits, candidate.execution.final_logits);
        numeric_gate = hidden.relative_rms <= 0.02 && hidden.cosine >= 0.9998 &&
            logits.relative_rms <= max_relative_rms && logits.cosine >= min_cosine;
    }
    const bool full_native_coverage = complete_same_sequence &&
        candidate.execution.native_av_steps == expected_native_steps &&
        candidate.execution.native_av_layers == expected_native_steps * 26;
    const std::string mismatch_position = first_mismatch_position == UINT32_MAX ?
        "none" : std::to_string(first_mismatch_position);
    std::printf("sequence_result completed=%s tokens_equal=%s first_mismatch_position=%s "
        "native_av_steps=%llu expected_native_av_steps=%llu native_av_layers=%llu "
        "packed_v_copy_bytes_avoided=%llu current_length=%u comparable_progress=%zu "
        "worst_progress_hidden_rel_rms=%.9g worst_progress_position=%u "
        "final_hidden_rel_rms=%.9g final_hidden_cosine=%.12g "
        "final_logits_rel_rms=%.9g final_logits_cosine=%.12g numeric_gate=%s full_native_coverage=%s "
        "candidate_status=Candidate_NOT_Valid progress_csv=%s\n",
        candidate.execution.completed ? "yes" : "no", same_tokens ? "yes" : "no",
        mismatch_position.c_str(),
        static_cast<unsigned long long>(candidate.execution.native_av_steps),
        static_cast<unsigned long long>(expected_native_steps),
        static_cast<unsigned long long>(candidate.execution.native_av_layers),
        static_cast<unsigned long long>(candidate.execution.packed_v_copy_bytes_avoided),
        candidate.current_length, comparable_progress, worst_progress_rms, worst_progress_position,
        hidden.relative_rms, hidden.cosine, logits.relative_rms, logits.cosine,
        numeric_gate ? "PASS" : "FAIL", full_native_coverage ? "PASS" : "FAIL", csv_path.c_str());

    return complete_same_sequence && numeric_gate && full_native_coverage ? 0 : 2;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s <semantic.vbuf> <source-url> <token-fixture> "
            "<prompt-rows> <decode-steps> <capacity> <progress.csv>\n", argv[0]);
        return 64;
    }
    try {
        return run(argv[1], argv[2], argv[3], static_cast<uint32_t>(std::stoul(argv[4])),
            static_cast<uint32_t>(std::stoul(argv[5])), static_cast<uint32_t>(std::stoul(argv[6])), argv[7]);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native AV sequence qualification failed: %s\n", error.what());
        return 1;
    }
}
