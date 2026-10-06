#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml;
namespace {
constexpr uint64_t expected_source_size = 9000232144ULL;
constexpr uint64_t expected_payload = 8995793920ULL;
constexpr uint64_t row_kv_bytes = 2048;
constexpr uint64_t boundary_bytes_per_position = 5120ULL * sizeof(float);
constexpr double max_relative_rms = 0.02;
constexpr double min_cosine = 0.9998;

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
    double err2 = 0, ref2 = 0, dot = 0;
    Metrics result;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) throw std::runtime_error("non-finite comparison value");
        const double d = static_cast<double>(b[i]) - a[i];
        result.max_abs = std::max(result.max_abs, std::abs(d));
        err2 += d * d; ref2 += static_cast<double>(a[i]) * a[i];
        dot += static_cast<double>(a[i]) * b[i];
    }
    result.rel_rms = std::sqrt(err2 / std::max(ref2, 1e-300));
    result.cosine = dot / std::max(std::sqrt(ref2) * std::sqrt(std::inner_product(
        b.begin(), b.end(), b.begin(), 0.0)), 1e-300);
    return result;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
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
        auto actual = executor.run(prompt, 1, std::nullopt, on_token);
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
        require(hidden.rel_rms <= max_relative_rms && hidden.cosine >= min_cosine &&
            logits.rel_rms <= max_relative_rms && logits.cosine >= min_cosine,
            "cross-device hidden/logit comparison exceeded the declared small-context qualification tolerance");
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
