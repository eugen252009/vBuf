#include "qwen3_attention_av.h"
#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <set>
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

enum class RunMode { Both, PrefillOnly, DecodeOnly, Layer0Position8 };

struct Run {
    Qwen3GenerationExecution execution;
    std::vector<Progress> progress;
    uint32_t current_length = 0;
    bool session_reset_verified = false;
};

Run execute(Qwen3Model & model, const std::shared_ptr<QwenCudaRuntimeState> & runtime,
        uint32_t capacity, const std::vector<uint32_t> & prompt, uint32_t decode_steps,
        bool native, const std::vector<uint32_t> * expected_tokens = nullptr,
        uint32_t * first_mismatch_position = nullptr, RunMode mode = RunMode::Both,
        bool stop_after_prompt = false) {
    Run result;
    auto session = runtime->create_session(capacity);
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, false, native);
    const bool allow_native_prefill = mode == RunMode::Both || mode == RunMode::PrefillOnly;
    const bool allow_native_decode = mode == RunMode::Both || mode == RunMode::DecodeOnly;
    const uint32_t capture_position = mode == RunMode::Layer0Position8 ? 8 :
        std::min(max_candidate_context, capacity - 1);
    const int32_t intervention_layer = mode == RunMode::Layer0Position8 ? 0 : -1;
    executor.configure_attention_av_diagnostic(capture_position, false, intervention_layer,
        allow_native_prefill, allow_native_decode);
    bool prompt_finished = false;
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
        if (stop_after_prompt && position >= prompt.size()) prompt_finished = true;
    };
    std::function<bool(uint32_t, uint32_t)> token_callback;
    if (expected_tokens != nullptr) token_callback = on_token;
    const auto should_cancel = [&] { return stop_after_prompt && prompt_finished; };
    result.execution = executor.run(prompt, decode_steps, std::nullopt,
        token_callback, should_cancel, on_progress);
    result.current_length = session->current_length();
    session->reset();
    result.session_reset_verified = session->current_length() == 0;
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

struct Fixture {
    std::string id;
    std::string category;
    std::string description;
    std::string prompt_text;
    std::vector<uint32_t> prompt;
    uint32_t generated_tokens = 0;
    uint32_t capacity = 0;
    RunMode mode = RunMode::Both;
    std::string input_mode;
    uint32_t repeats = 0;
    std::string prior_result;
    bool common_replay = false;
    bool session_reuse = false;
};

std::vector<std::string> split(const std::string & text, char delimiter) {
    std::vector<std::string> fields;
    std::stringstream stream(text);
    std::string field;
    while (std::getline(stream, field, delimiter)) fields.push_back(field);
    if (!text.empty() && text.back() == delimiter) fields.emplace_back();
    return fields;
}

uint32_t parse_u32(const std::string & text, const std::string & label) {
    size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed);
    if (consumed != text.size() || value > UINT32_MAX)
        throw std::invalid_argument("invalid " + label + ": " + text);
    return static_cast<uint32_t>(value);
}

std::vector<uint32_t> parse_token_ids(const std::string & text) {
    std::vector<uint32_t> tokens;
    for (const auto & field : split(text, ',')) tokens.push_back(parse_u32(field, "prompt token ID"));
    if (tokens.empty()) throw std::invalid_argument("fixture prompt token list is empty");
    return tokens;
}

RunMode parse_mode(const std::string & text) {
    if (text == "BOTH") return RunMode::Both;
    if (text == "PREFILL_ONLY") return RunMode::PrefillOnly;
    if (text == "DECODE_ONLY") return RunMode::DecodeOnly;
    if (text == "LAYER0_POS8") return RunMode::Layer0Position8;
    throw std::invalid_argument("unknown native AV fixture mode: " + text);
}

const char * mode_name(RunMode mode) {
    switch (mode) {
        case RunMode::Both: return "BOTH";
        case RunMode::PrefillOnly: return "PREFILL_ONLY";
        case RunMode::DecodeOnly: return "DECODE_ONLY";
        case RunMode::Layer0Position8: return "LAYER0_POS8";
    }
    return "UNKNOWN";
}

std::vector<Fixture> read_manifest(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open fixture manifest: " + path);
    std::vector<Fixture> fixtures;
    std::string line;
    bool have_header = false;
    const std::string expected_header =
        "fixture_id|category|description|prompt_text|prompt_ids|generated_tokens|capacity|mode|input_mode|repeats|prior_result|common_replay|session_reuse";
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!have_header) {
            require(line == expected_header, "fixture manifest header does not match the versioned schema");
            have_header = true;
            continue;
        }
        const auto fields = split(line, '|');
        require(fields.size() == 13, "fixture manifest row must have 13 pipe-separated fields");
        Fixture fixture;
        fixture.id = fields[0];
        fixture.category = fields[1];
        fixture.description = fields[2];
        fixture.prompt_text = fields[3];
        fixture.prompt = parse_token_ids(fields[4]);
        fixture.generated_tokens = parse_u32(fields[5], "generated token count");
        fixture.capacity = parse_u32(fields[6], "capacity");
        fixture.mode = parse_mode(fields[7]);
        fixture.input_mode = fields[8];
        fixture.repeats = parse_u32(fields[9], "repeat count");
        fixture.prior_result = fields[10];
        fixture.common_replay = fields[11] == "yes";
        fixture.session_reuse = fields[12] == "yes";
        require(!fixture.id.empty() && fixture.repeats >= 2, "fixtures require an ID and at least two executions");
        require(fixture.capacity >= fixture.prompt.size() + fixture.generated_tokens &&
            fixture.capacity <= 512, "fixture capacity is outside the qualified <=512 range");
        require(fixture.generated_tokens > 0 && fixture.capacity <= 512,
            "fixture exceeds the guarded capacity or generation scope");
        if (fixture.input_mode == "COMMON_INPUT") {
            require(fixture.prompt.size() == 33 && fixture.generated_tokens == 1 &&
                fixture.capacity >= 34 && fixture.mode != RunMode::Layer0Position8,
                "common-input phase fixtures must provide 33 fixed tokens through position 32");
        } else if (fixture.input_mode == "PHASE_COMMON") {
            require(fixture.prompt.size() == 32 && fixture.generated_tokens == 1 &&
                fixture.capacity >= 34 && fixture.mode != RunMode::Layer0Position8,
                "phase-common fixtures must seed one generated token from a 32-token prompt");
        } else if (fixture.input_mode == "INTERVENTION") {
            require(fixture.prompt.size() == 8 && fixture.generated_tokens == 1 &&
                fixture.mode == RunMode::Layer0Position8,
                "layer-0 intervention fixtures must target decode position 8");
        } else {
            require(fixture.input_mode == "FREE" && fixture.prompt.size() <= 32 &&
                fixture.prompt.size() + fixture.generated_tokens <= max_candidate_context + 1 &&
                fixture.mode != RunMode::Layer0Position8,
                "free-running fixture exceeds the guarded position-32 scope or has an invalid mode");
        }
        fixtures.push_back(std::move(fixture));
    }
    require(have_header && !fixtures.empty(), "fixture manifest contains no fixtures");
    return fixtures;
}

std::string csv(const std::string & value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    escaped += '"';
    return escaped;
}

void csv_row(std::ostream & out, const std::vector<std::string> & fields) {
    for (size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) out << ',';
        out << csv(fields[i]);
    }
    out << '\n';
}

std::string join_tokens(const std::vector<uint32_t> & tokens) {
    std::ostringstream out;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) out << ';';
        out << tokens[i];
    }
    return out.str();
}

uint64_t fnv_byte(uint64_t hash, uint8_t value) {
    return (hash ^ value) * 1099511628211ULL;
}

uint64_t fnv_u32(uint64_t hash, uint32_t value) {
    for (uint32_t shift = 0; shift < 32; shift += 8) hash = fnv_byte(hash, static_cast<uint8_t>(value >> shift));
    return hash;
}

uint64_t token_hash(const std::vector<uint32_t> & tokens) {
    uint64_t hash = 14695981039346656037ULL;
    hash = fnv_u32(hash, static_cast<uint32_t>(tokens.size()));
    for (uint32_t token : tokens) hash = fnv_u32(hash, token);
    return hash;
}

bool same_capture_inputs(const Qwen3GenerationExecution & left, const Qwen3GenerationExecution & right) {
    if (left.attention_av_output_captures.size() != right.attention_av_output_captures.size()) return false;
    for (size_t i = 0; i < left.attention_av_output_captures.size(); ++i) {
        const auto & a = left.attention_av_output_captures[i];
        const auto & b = right.attention_av_output_captures[i];
        if (a.position != b.position || a.rows != b.rows || a.phase != b.phase || a.input_token != b.input_token)
            return false;
    }
    return true;
}

uint64_t capture_hash(const Qwen3GenerationExecution & execution) {
    uint64_t hash = 14695981039346656037ULL;
    hash = fnv_u32(hash, static_cast<uint32_t>(execution.attention_av_output_captures.size()));
    const auto add_floats = [&](const std::vector<float> & values) {
        hash = fnv_u32(hash, static_cast<uint32_t>(values.size()));
        for (float value : values) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            hash = fnv_u32(hash, bits);
        }
    };
    for (const auto & capture : execution.attention_av_output_captures) {
        hash = fnv_u32(hash, capture.position);
        hash = fnv_u32(hash, capture.rows);
        hash = fnv_u32(hash, static_cast<uint32_t>(capture.phase));
        hash = fnv_u32(hash, capture.input_token);
        add_floats(capture.hidden);
        add_floats(capture.logits);
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

std::string format_double(double value) {
    if (!std::isfinite(value)) return "nan";
    std::ostringstream out;
    out << std::setprecision(12) << value;
    return out.str();
}

uint32_t top1(const std::vector<float> & logits) {
    require(!logits.empty(), "captured logits are empty");
    return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

const char * phase_name(QwenExecutionPhase phase) {
    return phase == QwenExecutionPhase::Prefill ? "PREFILL" : "DECODE";
}

struct SelectionCounts {
    uint64_t requested = 0;
    uint64_t selected = 0;
    uint64_t executed = 0;
    uint64_t native_layers = 0;
    uint64_t canonical_layers = 0;
    uint64_t interventions = 0;
    uint64_t expected = 0;
    bool exact = true;
};

SelectionCounts selection_counts(const Run & run, uint32_t capacity, RunMode mode, bool native_candidate) {
    SelectionCounts counts;
    for (const auto & step : run.execution.attention_av_step_diagnostics) {
        const bool mode_allows = mode == RunMode::Both ||
            (mode == RunMode::PrefillOnly && step.phase == QwenExecutionPhase::Prefill) ||
            (mode == RunMode::DecodeOnly && step.phase == QwenExecutionPhase::Decode);
        const bool geometry_eligible = capacity <= 512 &&
            ((step.phase == QwenExecutionPhase::Prefill && step.rows == 32 && step.position == 31) ||
             (step.phase == QwenExecutionPhase::Decode && step.rows == 1 && step.position <= max_candidate_context));
        const bool expected = native_candidate && mode_allows && geometry_eligible;
        counts.expected += expected ? 1 : 0;
        counts.requested += step.native_candidate_requested ? 1 : 0;
        counts.selected += step.native_candidate_selected ? 1 : 0;
        counts.executed += step.native_av_executed ? 1 : 0;
        counts.native_layers += step.native_av_layer_graphs;
        counts.interventions += step.diagnostic_native_av_interventions;
        counts.canonical_layers += 40 - step.native_av_layer_graphs - step.diagnostic_native_av_interventions;
        if (step.native_candidate_requested != expected || step.native_candidate_selected != expected ||
            step.native_av_executed != expected) counts.exact = false;
    }
    counts.exact = counts.exact &&
        run.execution.attention_av_step_diagnostics.size() == run.execution.completed_positions &&
        counts.executed == run.execution.native_av_steps;
    return counts;
}

struct PairSummary {
    std::string fixture_id;
    std::string run_kind;
    std::string status;
    std::string detail;
    uint32_t repeat = 0;
    bool same_tokens = false;
    bool same_history = false;
    bool numeric_gate = false;
    bool eligibility = false;
    bool canonical_reset = false;
    bool candidate_reset = false;
    Metrics hidden{};
    Metrics logits{};
    bool numeric_comparable = false;
    uint32_t first_mismatch = UINT32_MAX;
    SelectionCounts counts;
    uint64_t canonical_token_hash = 0;
    uint64_t candidate_token_hash = 0;
    uint64_t canonical_capture_hash = 0;
    uint64_t candidate_capture_hash = 0;
    std::string canonical_tokens;
    std::string candidate_tokens;
    std::string input_tokens;
};

PairSummary compare_pair(std::ostream & positions_out, std::ostream & tokens_out,
        const Fixture & fixture, uint32_t repeat, const std::string & run_kind, RunMode mode,
        bool native_candidate, const std::vector<uint32_t> & input_tokens,
        const Run & canonical, const Run & candidate, bool free_running,
        bool expected_cancel, uint32_t first_mismatch) {
    PairSummary summary;
    summary.fixture_id = fixture.id;
    summary.run_kind = run_kind;
    summary.repeat = repeat;
    summary.first_mismatch = first_mismatch;
    summary.same_tokens = canonical.execution.tokens == candidate.execution.tokens;
    summary.same_history = free_running ? summary.same_tokens :
        input_tokens.size() == canonical.current_length && input_tokens.size() == candidate.current_length &&
        same_capture_inputs(canonical.execution, candidate.execution);
    summary.input_tokens = join_tokens(input_tokens);
    summary.canonical_tokens = join_tokens(canonical.execution.tokens);
    summary.candidate_tokens = join_tokens(candidate.execution.tokens);
    summary.canonical_token_hash = token_hash(canonical.execution.tokens);
    summary.candidate_token_hash = token_hash(candidate.execution.tokens);
    summary.canonical_capture_hash = capture_hash(canonical.execution);
    summary.candidate_capture_hash = capture_hash(candidate.execution);
    summary.canonical_reset = canonical.session_reset_verified;
    summary.candidate_reset = candidate.session_reset_verified;
    summary.counts = selection_counts(candidate, fixture.capacity, mode, native_candidate);
    summary.eligibility = summary.counts.exact && summary.counts.requested == summary.counts.expected &&
        summary.counts.selected == summary.counts.expected && summary.counts.executed == summary.counts.expected &&
        summary.counts.native_layers == summary.counts.executed * 26 &&
        (mode != RunMode::Layer0Position8 || summary.counts.interventions == 1);

    const bool canonical_state_ok = expected_cancel ?
        canonical.execution.cancelled && !canonical.execution.completed && canonical.current_length == input_tokens.size() :
        canonical.execution.completed;
    const bool candidate_state_ok = expected_cancel ?
        candidate.execution.cancelled && !candidate.execution.completed && candidate.current_length == input_tokens.size() :
        candidate.execution.completed;
    summary.numeric_comparable = summary.same_history && canonical_state_ok && candidate_state_ok &&
        canonical.current_length == candidate.current_length &&
        canonical.execution.final_hidden.size() == candidate.execution.final_hidden.size() &&
        canonical.execution.final_logits.size() == candidate.execution.final_logits.size();
    if (summary.numeric_comparable) {
        summary.hidden = compare(canonical.execution.final_hidden, candidate.execution.final_hidden);
        summary.logits = compare(canonical.execution.final_logits, candidate.execution.final_logits);
        summary.numeric_gate = summary.hidden.relative_rms <= 0.02 && summary.hidden.cosine >= 0.9998 &&
            summary.logits.relative_rms <= max_relative_rms && summary.logits.cosine >= min_cosine;
    }

    const size_t aligned = std::min(canonical.execution.attention_av_output_captures.size(),
        candidate.execution.attention_av_output_captures.size());
    for (size_t i = 0; i < aligned; ++i) {
        const auto & reference = canonical.execution.attention_av_output_captures[i];
        const auto & actual = candidate.execution.attention_av_output_captures[i];
        require(reference.position == actual.position && reference.rows == actual.rows &&
            reference.phase == actual.phase && reference.input_token == actual.input_token,
            "common execution inputs diverged before an output-capture boundary");
        const Metrics hidden = compare(reference.hidden, actual.hidden);
        const Metrics logits = compare(reference.logits, actual.logits);
        const auto & step = i < candidate.execution.attention_av_step_diagnostics.size() ?
            candidate.execution.attention_av_step_diagnostics[i] : Qwen3GenerationStepDiagnostic{};
        const uint64_t canonical_layers = 40 - step.native_av_layer_graphs -
            step.diagnostic_native_av_interventions;
        csv_row(positions_out, {fixture.id, std::to_string(repeat), run_kind,
            std::to_string(reference.position), std::to_string(reference.rows), phase_name(reference.phase),
            std::to_string(reference.input_token), std::to_string(top1(reference.logits)),
            std::to_string(top1(actual.logits)), top1(reference.logits) == top1(actual.logits) ? "yes" : "no",
            step.native_candidate_requested ? "yes" : "no", step.native_candidate_selected ? "yes" : "no",
            step.native_av_executed ? "yes" : "no", std::to_string(step.native_av_layer_graphs),
            std::to_string(step.diagnostic_native_av_interventions), std::to_string(canonical_layers),
            format_double(hidden.max_abs), format_double(hidden.relative_rms), format_double(hidden.cosine),
            format_double(logits.max_abs), format_double(logits.relative_rms), format_double(logits.cosine)});
    }
    csv_row(tokens_out, {fixture.id, std::to_string(repeat), run_kind, summary.input_tokens,
        summary.canonical_tokens, summary.candidate_tokens, hex64(summary.canonical_token_hash),
        hex64(summary.candidate_token_hash), free_running ? (summary.same_tokens ? "yes" : "no") : "common-input"});

    const bool token_ok = !free_running || summary.same_tokens;
    const bool state_ok = canonical_state_ok && candidate_state_ok && summary.canonical_reset &&
        summary.candidate_reset && summary.eligibility;
    if (fixture.prior_result == "FAIL" && state_ok && token_ok && summary.numeric_comparable && !summary.numeric_gate) {
        summary.status = "KNOWN_NEGATIVE_REPRODUCED";
        summary.detail = "prior failure repeated with the unchanged final hidden/logit gate";
    } else if (fixture.prior_result == "FAIL" && state_ok && token_ok && summary.numeric_gate) {
        summary.status = "PRIOR_NEGATIVE_NOT_REPRODUCED";
        summary.detail = "the previously recorded negative did not recur; no threshold was changed";
    } else if (!token_ok) {
        summary.status = "TOKEN_TRAJECTORY_DIVERGENCE";
        summary.detail = "candidate greedy token sequence differs from canonical";
    } else if (!state_ok) {
        summary.status = "EXECUTION_OR_GUARD_FAILURE";
        summary.detail = "completion, session reset, or exact native candidate selection/count failed";
    } else if (!summary.numeric_comparable) {
        summary.status = "NUMERIC_NOT_COMPARABLE";
        summary.detail = "same input history and final hidden/logits were not available on both paths";
    } else if (!summary.numeric_gate) {
        summary.status = "NUMERIC_GATE_FAIL";
        summary.detail = "existing final hidden/logit gate failed; no threshold relaxation";
    } else {
        summary.status = "PASS";
        summary.detail = "free-running or common-token comparison passed the existing final gate";
    }
    return summary;
}

void write_summary(std::ostream & out, const Fixture & fixture, const PairSummary & s,
        const Run & canonical, const Run & candidate) {
    csv_row(out, {s.fixture_id, std::to_string(s.repeat), s.run_kind, mode_name(fixture.mode),
        fixture.prior_result, s.status, canonical.execution.completed ? "yes" : "no",
        candidate.execution.completed ? "yes" : "no", canonical.execution.cancelled ? "yes" : "no",
        candidate.execution.cancelled ? "yes" : "no", s.same_history ? "yes" : "no",
        s.same_tokens ? "yes" : "no", s.first_mismatch == UINT32_MAX ? "none" : std::to_string(s.first_mismatch),
        std::to_string(canonical.current_length), std::to_string(candidate.current_length),
        s.canonical_reset ? "yes" : "no", s.candidate_reset ? "yes" : "no",
        std::to_string(s.counts.requested), std::to_string(s.counts.selected),
        std::to_string(s.counts.executed), std::to_string(s.counts.native_layers),
        std::to_string(s.counts.canonical_layers), std::to_string(s.counts.interventions),
        std::to_string(candidate.execution.packed_v_copy_bytes_avoided),
        s.numeric_comparable ? format_double(s.hidden.max_abs) : "NA",
        s.numeric_comparable ? format_double(s.hidden.relative_rms) : "NA",
        s.numeric_comparable ? format_double(s.hidden.cosine) : "NA",
        s.numeric_comparable ? format_double(s.logits.max_abs) : "NA",
        s.numeric_comparable ? format_double(s.logits.relative_rms) : "NA",
        s.numeric_comparable ? format_double(s.logits.cosine) : "NA",
        s.numeric_comparable ? (s.numeric_gate ? "PASS" : "FAIL") : "NA",
        s.eligibility ? "PASS" : "FAIL", hex64(s.canonical_token_hash), hex64(s.candidate_token_hash),
        hex64(s.canonical_capture_hash), hex64(s.candidate_capture_hash), s.input_tokens,
        s.canonical_tokens, s.candidate_tokens, s.detail});
}

bool run_fixture_is_expected_negative(const Fixture & fixture, const PairSummary & summary) {
    return fixture.prior_result == "FAIL" && summary.status == "KNOWN_NEGATIVE_REPRODUCED";
}

void write_repeatability(std::ostream & out, const Fixture & fixture,
        const std::vector<PairSummary> & summaries) {
    std::set<std::string> kinds;
    for (const auto & summary : summaries) kinds.insert(summary.run_kind);
    for (const auto & kind : kinds) {
        std::vector<const PairSummary *> rows;
        for (const auto & summary : summaries) if (summary.run_kind == kind) rows.push_back(&summary);
        const auto equal_hash = [&](bool candidate, bool captures) {
            if (rows.size() < 2) return false;
            const PairSummary * first = rows.front();
            const uint64_t hash = captures ? (candidate ? first->candidate_capture_hash : first->canonical_capture_hash) :
                (candidate ? first->candidate_token_hash : first->canonical_token_hash);
            for (const auto * row : rows) {
                const uint64_t current = captures ? (candidate ? row->candidate_capture_hash : row->canonical_capture_hash) :
                    (candidate ? row->candidate_token_hash : row->canonical_token_hash);
                if (current != hash) return false;
            }
            return true;
        };
        const bool canonical_tokens = equal_hash(false, false);
        const bool candidate_tokens = equal_hash(true, false);
        const bool canonical_captures = equal_hash(false, true);
        const bool candidate_captures = equal_hash(true, true);
        const bool all_stable = canonical_tokens && candidate_tokens && canonical_captures && candidate_captures;
        csv_row(out, {fixture.id, kind, std::to_string(rows.size()), canonical_tokens ? "yes" : "no",
            candidate_tokens ? "yes" : "no", canonical_captures ? "yes" : "no",
            candidate_captures ? "yes" : "no", all_stable ? "PASS" : "FAIL"});
    }
}

bool summaries_repeatable(const Fixture & fixture, const std::vector<PairSummary> & summaries) {
    std::set<std::string> kinds;
    for (const auto & summary : summaries) kinds.insert(summary.run_kind);
    for (const auto & kind : kinds) {
        std::vector<const PairSummary *> rows;
        for (const auto & summary : summaries) if (summary.run_kind == kind) rows.push_back(&summary);
        if (rows.size() != fixture.repeats) return false;
        const auto same = [&](auto select_hash) {
            const uint64_t first = select_hash(*rows.front());
            for (const auto * row : rows) if (select_hash(*row) != first) return false;
            return true;
        };
        if (!same([](const PairSummary & row) { return row.canonical_token_hash; }) ||
            !same([](const PairSummary & row) { return row.candidate_token_hash; }) ||
            !same([](const PairSummary & row) { return row.canonical_capture_hash; }) ||
            !same([](const PairSummary & row) { return row.candidate_capture_hash; })) return false;
    }
    return true;
}

void write_not_run(std::ostream & out, const Fixture & fixture, const std::string & reason) {
    csv_row(out, {fixture.id, "0", "NOT_RUN", mode_name(fixture.mode), fixture.prior_result,
        "NOT_RUN_AFTER_STOP", "NA", "NA", "NA", "NA", "NA", "NA", "none", "NA", "NA",
        "NA", "NA", "0", "0", "0", "0", "0", "0", "0", "NA", "NA", "NA", "NA",
        "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "", "", "", reason});
}

bool run_session_reuse_after_cancel(Qwen3Model & model,
        const std::shared_ptr<QwenCudaRuntimeState> & runtime, const Fixture & fixture,
        const PairSummary & fresh_candidate, std::ostream & out) {
    auto session = runtime->create_session(fixture.capacity);
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, false, true);
    executor.configure_attention_av_diagnostic(max_candidate_context, false, -1, true, true);
    bool prompt_finished = false;
    const auto on_progress = [&](uint32_t position, const std::vector<float> &) {
        if (position >= fixture.prompt.size()) prompt_finished = true;
    };
    const auto should_cancel = [&] { return prompt_finished; };
    const auto cancelled = executor.run(fixture.prompt, 1, std::nullopt, {}, should_cancel, on_progress);
    const uint32_t cancelled_length = session->current_length();
    const auto reused = executor.run(fixture.prompt, fixture.generated_tokens);
    const uint32_t reused_length = session->current_length();
    const bool cancelled_at_boundary = cancelled.cancelled && !cancelled.completed &&
        cancelled_length == fixture.prompt.size();
    const bool reentered_cleanly = reused.completed && reused_length == fixture.prompt.size() + fixture.generated_tokens;
    const bool tokens_equal = token_hash(reused.tokens) == fresh_candidate.candidate_token_hash;
    const bool captures_equal = capture_hash(reused) == fresh_candidate.candidate_capture_hash;
    const bool reset_after_reuse = (session->reset(), session->current_length() == 0);
    const bool pass = cancelled_at_boundary && reentered_cleanly && tokens_equal && captures_equal && reset_after_reuse;
    csv_row(out, {fixture.id, mode_name(RunMode::Both), std::to_string(cancelled_length),
        std::to_string(reused_length), cancelled_at_boundary ? "yes" : "no", reentered_cleanly ? "yes" : "no",
        reset_after_reuse ? "yes" : "no", tokens_equal ? "yes" : "no", captures_equal ? "yes" : "no",
        hex64(token_hash(reused.tokens)), hex64(capture_hash(reused)), pass ? "PASS" : "FAIL"});
    return pass;
}

int run_matrix(const std::string & semantic, const std::string & source,
        const std::string & manifest_path, const std::string & output_dir) {
    const std::vector<Fixture> fixtures = read_manifest(manifest_path);
    std::filesystem::create_directories(output_dir);
    std::ofstream summary_out(std::filesystem::path(output_dir) / "fixture_summary.csv");
    std::ofstream positions_out(std::filesystem::path(output_dir) / "per_position.csv");
    std::ofstream tokens_out(std::filesystem::path(output_dir) / "token_sequences.csv");
    std::ofstream repeats_out(std::filesystem::path(output_dir) / "repeatability.csv");
    std::ofstream reuse_out(std::filesystem::path(output_dir) / "session_reuse.csv");
    require(summary_out && positions_out && tokens_out && repeats_out && reuse_out,
        "could not create one or more matrix evidence files");
    summary_out << "fixture_id,repeat,run_kind,mode,prior_result,status,canonical_completed,candidate_completed,canonical_cancelled,candidate_cancelled,same_history,generated_tokens_equal,first_mismatch_position,canonical_length,candidate_length,canonical_session_reset,candidate_session_reset,candidate_requested_steps,candidate_selected_steps,candidate_native_steps,candidate_native_layer_graphs,candidate_canonical_layer_graphs,diagnostic_interventions,packed_v_copy_bytes_avoided,final_hidden_max_abs,final_hidden_relative_rms,final_hidden_cosine,final_logits_max_abs,final_logits_relative_rms,final_logits_cosine,numeric_gate,selection_guard,canonical_token_hash,candidate_token_hash,canonical_capture_hash,candidate_capture_hash,input_token_ids,canonical_generated_tokens,candidate_generated_tokens,detail\n";
    positions_out << "fixture_id,repeat,run_kind,position,rows,phase,input_token,canonical_next_top1,candidate_next_top1,next_top1_equal,candidate_requested,candidate_selected,native_av_executed,native_av_layer_graphs,diagnostic_interventions,canonical_av_layer_graphs,hidden_max_abs,hidden_relative_rms,hidden_cosine,logits_max_abs,logits_relative_rms,logits_cosine\n";
    tokens_out << "fixture_id,repeat,run_kind,input_token_ids,canonical_generated_tokens,candidate_generated_tokens,canonical_token_hash,candidate_token_hash,trajectory_equal\n";
    repeats_out << "fixture_id,run_kind,attempts,canonical_tokens_repeatable,candidate_tokens_repeatable,canonical_capture_repeatable,candidate_capture_repeatable,status\n";
    reuse_out << "fixture_id,mode,cancelled_length,reused_length,cancelled_at_prompt_boundary,reentered_cleanly,session_reset_verified,tokens_equal_fresh,captures_equal_fresh,token_hash,capture_hash,status\n";

    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256 &&
        model.source_size == expected_source_size && model.layer_count == 40 && model.count == 443,
        "matrix requires the admitted Qwen3-14B Q4_K_M artifact");
    QwenCudaRuntimeConfig config;
    config.placement = QwenCudaPlacement::contiguous_split(0, 1, 26);
    auto runtime = QwenCudaRuntimeState::create(model, config);
    require(runtime->device_count() == 2, "matrix requires the canonical 26/14 two-device placement");
    std::set<uint32_t> capacities;
    for (const auto & fixture : fixtures) capacities.insert(fixture.capacity);
    for (uint32_t capacity : capacities) {
        auto plan_session = runtime->create_session(capacity);
        const auto plan = build_qwen_execution_plan(model, *runtime, *plan_session, QwenExecutionPlanPath::MultiGpu);
        require(runtime->execution_optimizer().register_candidate(
            make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Prefill)) &&
            runtime->execution_optimizer().register_candidate(
            make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Decode)),
            "could not register guarded native AV candidates for capacity " + std::to_string(capacity));
        plan_session->reset();
    }
    runtime->execution_optimizer().set_unvalidated_trial_for_testing(true);
    std::printf("matrix_start model=%s fixtures=%zu capacities=", model.artifact_identity.c_str(), fixtures.size());
    for (uint32_t capacity : capacities) std::printf("%s%u", capacity == *capacities.begin() ? "" : ";", capacity);
    std::printf(" placement=multi:0x26,1x14; production_selection=disabled; candidate=unvalidated_trial\n");

    bool stop = false;
    size_t stop_index = fixtures.size();
    for (size_t fixture_index = 0; fixture_index < fixtures.size(); ++fixture_index) {
        const Fixture & fixture = fixtures[fixture_index];
        std::vector<PairSummary> fixture_summaries;
        std::vector<PairSummary> main_summaries;
        bool fixture_error = false;
        std::string error_text;
        std::printf("fixture_begin id=%s category=%s mode=%s prompt_rows=%zu generated=%u capacity=%u repeats=%u input_mode=%s\n",
            fixture.id.c_str(), fixture.category.c_str(), mode_name(fixture.mode), fixture.prompt.size(),
            fixture.generated_tokens, fixture.capacity, fixture.repeats, fixture.input_mode.c_str());
        try {
            for (uint32_t repeat = 1; repeat <= fixture.repeats; ++repeat) {
                Run canonical;
                Run candidate;
                std::vector<uint32_t> phase_input;
                std::string run_kind;
                bool free_running = false;
                bool expected_cancel = false;
                uint32_t mismatch = UINT32_MAX;
                RunMode candidate_mode = fixture.mode;
                bool native_candidate = fixture.mode != RunMode::Layer0Position8;

                if (fixture.input_mode == "PHASE_COMMON" || fixture.input_mode == "COMMON_INPUT") {
                    if (fixture.input_mode == "PHASE_COMMON") {
                        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
                        const Run seed = execute(model, runtime, fixture.capacity, fixture.prompt, 1, false);
                        require(seed.execution.completed && seed.execution.tokens.size() == 1 &&
                            seed.current_length == fixture.prompt.size() + 1,
                            "phase-common canonical seed generation failed");
                        phase_input = fixture.prompt;
                        phase_input.push_back(seed.execution.tokens.front());
                    } else {
                        phase_input = fixture.prompt;
                    }
                    run_kind = "PHASE_COMMON_TOKEN";
                    expected_cancel = true;
                    runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
                    canonical = execute(model, runtime, fixture.capacity, phase_input, 1, false,
                        nullptr, nullptr, RunMode::Both, true);
                    runtime->execution_optimizer().set_mode(QwenOptimizerMode::Enabled);
                    candidate = execute(model, runtime, fixture.capacity, phase_input, 1, native_candidate,
                        nullptr, nullptr, candidate_mode, true);
                } else {
                    phase_input = fixture.prompt;
                    run_kind = fixture.input_mode == "INTERVENTION" ? "LAYER0_INTERVENTION" : "FREE_RUNNING";
                    free_running = true;
                    runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
                    canonical = execute(model, runtime, fixture.capacity, fixture.prompt,
                        fixture.generated_tokens, false);
                    require(canonical.execution.completed && canonical.execution.tokens.size() == fixture.generated_tokens &&
                        canonical.current_length == fixture.prompt.size() + fixture.generated_tokens,
                        "canonical fixture execution did not complete");
                    if (fixture.input_mode == "INTERVENTION") {
                        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
                        candidate = execute(model, runtime, fixture.capacity, fixture.prompt,
                            fixture.generated_tokens, false, nullptr, nullptr, candidate_mode);
                    } else {
                        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Enabled);
                        candidate = execute(model, runtime, fixture.capacity, fixture.prompt,
                            fixture.generated_tokens, true, &canonical.execution.tokens, &mismatch, candidate_mode);
                    }
                }
                PairSummary main = compare_pair(positions_out, tokens_out, fixture, repeat, run_kind,
                    candidate_mode, native_candidate, phase_input, canonical, candidate,
                    free_running, expected_cancel, mismatch);
                write_summary(summary_out, fixture, main, canonical, candidate);
                fixture_summaries.push_back(main);
                main_summaries.push_back(main);
                std::printf("fixture_repeat id=%s repeat=%u run=%s status=%s candidate_steps=%llu selected=%llu interventions=%llu hidden_rms=%.9g logits_rms=%.9g token_equal=%s\n",
                    fixture.id.c_str(), repeat, run_kind.c_str(), main.status.c_str(),
                    static_cast<unsigned long long>(candidate.execution.native_av_steps),
                    static_cast<unsigned long long>(main.counts.selected),
                    static_cast<unsigned long long>(main.counts.interventions),
                    main.numeric_comparable ? main.hidden.relative_rms : std::numeric_limits<double>::quiet_NaN(),
                    main.numeric_comparable ? main.logits.relative_rms : std::numeric_limits<double>::quiet_NaN(),
                    main.same_tokens ? "yes" : "no");

                if (fixture.common_replay) {
                    require(fixture.input_mode == "FREE" && canonical.execution.tokens.size() == fixture.generated_tokens,
                        "common-token replay requires a complete canonical free-running sequence");
                    std::vector<uint32_t> common_input = fixture.prompt;
                    common_input.insert(common_input.end(), canonical.execution.tokens.begin(), canonical.execution.tokens.end());
                    runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
                    const Run common_canonical = execute(model, runtime, fixture.capacity, common_input, 1, false,
                        nullptr, nullptr, RunMode::Both, true);
                    runtime->execution_optimizer().set_mode(QwenOptimizerMode::Enabled);
                    const Run common_candidate = execute(model, runtime, fixture.capacity, common_input, 1, true,
                        nullptr, nullptr, RunMode::Both, true);
                    PairSummary common = compare_pair(positions_out, tokens_out, fixture, repeat,
                        "COMMON_TOKEN_REPLAY", RunMode::Both, true, common_input,
                        common_canonical, common_candidate, false, true, UINT32_MAX);
                    write_summary(summary_out, fixture, common, common_canonical, common_candidate);
                    fixture_summaries.push_back(common);
                    std::printf("common_replay id=%s repeat=%u status=%s common_tokens=%zu native_steps=%llu hidden_rms=%.9g logits_rms=%.9g\n",
                        fixture.id.c_str(), repeat, common.status.c_str(), common_input.size(),
                        static_cast<unsigned long long>(common_candidate.execution.native_av_steps),
                        common.numeric_comparable ? common.hidden.relative_rms : std::numeric_limits<double>::quiet_NaN(),
                        common.numeric_comparable ? common.logits.relative_rms : std::numeric_limits<double>::quiet_NaN());
                }
            }
        } catch (const std::exception & error) {
            fixture_error = true;
            error_text = error.what();
            std::fprintf(stderr, "fixture_error id=%s error=%s\n", fixture.id.c_str(), error_text.c_str());
        }
        write_repeatability(repeats_out, fixture, fixture_summaries);

        bool unexpected_failure = fixture_error;
        if (!fixture_error && !summaries_repeatable(fixture, fixture_summaries)) unexpected_failure = true;
        for (const auto & result : fixture_summaries) {
            const bool passed = result.status == "PASS" || result.status == "PRIOR_NEGATIVE_NOT_REPRODUCED";
            if (!passed && !run_fixture_is_expected_negative(fixture, result)) unexpected_failure = true;
        }
        if (!fixture_error && fixture_summaries.size() < fixture.repeats * (fixture.common_replay ? 2 : 1))
            unexpected_failure = true;

        if (fixture.session_reuse && !fixture_error && !main_summaries.empty()) {
            const bool reuse_pass = run_session_reuse_after_cancel(model, runtime, fixture,
                main_summaries.front(), reuse_out);
            if (!reuse_pass) unexpected_failure = true;
            std::printf("session_reuse id=%s status=%s\n", fixture.id.c_str(), reuse_pass ? "PASS" : "FAIL");
        }
        if (unexpected_failure) {
            stop = true;
            stop_index = fixture_index;
            std::printf("matrix_stop_after_fixture id=%s reason=%s; all remaining fixtures are not run\n",
                fixture.id.c_str(), fixture_error ? error_text.c_str() : "new failure or non-repeatable result");
            for (size_t remaining = fixture_index + 1; remaining < fixtures.size(); ++remaining)
                write_not_run(summary_out, fixtures[remaining], "qualification stopped after " + fixture.id);
            break;
        }
    }
    summary_out.close();
    positions_out.close();
    tokens_out.close();
    repeats_out.close();
    reuse_out.close();
    std::printf("matrix_result status=%s fixtures_run=%zu fixtures_total=%zu candidate_status=Candidate_NOT_Valid output=%s\n",
        stop ? "STOPPED_ON_FAILURE" : "COMPLETED", stop ? stop_index + 1 : fixtures.size(),
        fixtures.size(), output_dir.c_str());
    return stop ? 2 : 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 6 && std::string(argv[3]) == "--matrix")
            return run_matrix(argv[1], argv[2], argv[4], argv[5]);
        if (argc == 8)
            return run(argv[1], argv[2], argv[3], static_cast<uint32_t>(std::stoul(argv[4])),
                static_cast<uint32_t>(std::stoul(argv[5])), static_cast<uint32_t>(std::stoul(argv[6])), argv[7]);
        std::fprintf(stderr, "usage: %s <semantic.vbuf> <source-url> --matrix <fixtures.tsv> <output-dir>\n"
            "   or: %s <semantic.vbuf> <source-url> <token-fixture> <prompt-rows> <decode-steps> <capacity> <progress.csv>\n",
            argv[0], argv[0]);
        return 64;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "native AV sequence qualification failed: %s\n", error.what());
        return 1;
    }
}
