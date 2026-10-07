#include "qwen3_execution_plan.h"
#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(VBUF_QWEN3_HAS_CUDA_PROFILER_API)
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

using namespace vbuf_ggml;
namespace {
constexpr uint64_t source_bytes = 9000232144ULL;
constexpr uint64_t tensor_payload_bytes = 8995793920ULL;
constexpr uint64_t kv_bytes_per_block_token = 4096;
constexpr uint64_t packed_v_bytes_per_token = 2048;
constexpr uint32_t hidden_width = 5120;
constexpr uint32_t kv_row_bytes = 2048;
using Clock = std::chrono::steady_clock;

struct Memory { size_t free = 0, total = 0; std::string name; };
std::vector<Memory> probe_gpus() {
    ggml_backend_reg_t registry = ggml_backend_reg_by_name("CUDA");
    if (!registry || ggml_backend_reg_dev_count(registry) < 2)
        throw std::runtime_error("two CUDA devices are required");
    std::vector<Memory> result;
    for (uint32_t id = 0; id < 2; ++id) {
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry, id);
        if (!device) throw std::runtime_error("CUDA device registry entry missing");
        Memory item; item.name = ggml_backend_dev_name(device);
        ggml_backend_dev_memory(device, &item.free, &item.total);
        result.push_back(std::move(item));
    }
    return result;
}
void print_memory(const char * label, const std::vector<Memory> & memory) {
    for (size_t i = 0; i < memory.size(); ++i)
        std::printf("memory stage=%s device=%zu name=%s free_bytes=%zu total_bytes=%zu observed_used_bytes=%zu\n",
            label, i, memory[i].name.c_str(), memory[i].free, memory[i].total,
            memory[i].total >= memory[i].free ? memory[i].total - memory[i].free : 0);
}
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}
struct Distribution { uint64_t minimum = 0, median = 0, p95 = 0; };
Distribution distribution(std::vector<uint64_t> samples) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    const uint64_t median = samples.size() % 2 != 0 ? samples[middle] :
        samples[middle - 1] / 2 + samples[middle] / 2 +
        ((samples[middle - 1] % 2 + samples[middle] % 2) / 2);
    const size_t p95_index = (95 * samples.size() + 99) / 100 - 1;
    return {samples.front(), median, samples[p95_index]};
}
void print_decode_profile(const char * label, const Qwen3GenerationExecution & execution) {
    const auto & p = execution.decode_profile;
    const auto step = distribution(p.step_wall_samples_ns);
    const auto outer = distribution(p.outer_token_wall_samples_ns);
    if (p.steps == 0) return;
    const uint64_t steps = p.steps;
    std::printf("decode_profile label=%s steps=%llu outer_min_median_p95_ns=%llu,%llu,%llu "
        "step_min_median_p95_ns=%llu,%llu,%llu plan_select_ns_per_request=%llu optimizer_record_ns_per_request=%llu "
        "control_prepare_ns_per_step=%llu control_enqueue_cpu_ns_per_step=%llu layer_resolve_cpu_ns_per_step=%llu "
        "boundary_descriptor_cpu_ns_per_step=%llu graph_submit_cpu_ns_per_step=%llu "
        "candidate_setup_ns_per_request=%llu candidate_dispatch_cpu_ns_per_step=%llu "
        "boundary_wait_transfer_ns_per_step=%llu "
        "boundary_source_wait_d2h_ns_per_step=%llu "
        "boundary_destination_h2d_wait_ns_per_step=%llu final_sync_readback_ns_per_step=%llu "
        "final_device_wait_ns_per_step=%llu final_output_readback_ns_per_step=%llu "
        "control_h2d_bytes_per_step=%llu control_h2d_calls_per_step=%llu "
        "canonical_decode_steps=%llu specialized_decode_steps=%llu\n", label,
        static_cast<unsigned long long>(steps), static_cast<unsigned long long>(outer.minimum),
        static_cast<unsigned long long>(outer.median), static_cast<unsigned long long>(outer.p95),
        static_cast<unsigned long long>(step.minimum), static_cast<unsigned long long>(step.median),
        static_cast<unsigned long long>(step.p95), static_cast<unsigned long long>(p.plan_select_ns),
        static_cast<unsigned long long>(p.optimizer_record_ns),
        static_cast<unsigned long long>(p.control_prepare_ns / steps),
        static_cast<unsigned long long>(p.control_enqueue_cpu_ns / steps),
        static_cast<unsigned long long>(p.layer_resolve_cpu_ns / steps),
        static_cast<unsigned long long>(p.boundary_descriptor_cpu_ns / steps),
        static_cast<unsigned long long>(p.graph_submit_cpu_ns / steps),
        static_cast<unsigned long long>(p.candidate_setup_ns),
        static_cast<unsigned long long>(p.candidate_dispatch_cpu_ns / steps),
        static_cast<unsigned long long>(p.boundary_wait_transfer_ns / steps),
        static_cast<unsigned long long>(p.boundary_source_wait_d2h_ns / steps),
        static_cast<unsigned long long>(p.boundary_destination_h2d_wait_ns / steps),
        static_cast<unsigned long long>(p.final_sync_readback_ns / steps),
        static_cast<unsigned long long>(p.final_device_wait_ns / steps),
        static_cast<unsigned long long>(p.final_output_readback_ns / steps),
        static_cast<unsigned long long>(p.control_h2d_bytes / steps),
        static_cast<unsigned long long>(p.control_h2d_calls / steps),
        static_cast<unsigned long long>(execution.canonical_decode_steps),
        static_cast<unsigned long long>(execution.specialized_decode_steps));
}
void print_pooled_distribution(const char * mode, const std::vector<uint64_t> & samples) {
    const auto result = distribution(samples);
    std::printf("decode_mode_pooled mode=%s samples=%zu min_ns=%llu median_ns=%llu p95_ns=%llu\n", mode,
        samples.size(), static_cast<unsigned long long>(result.minimum),
        static_cast<unsigned long long>(result.median), static_cast<unsigned long long>(result.p95));
}
std::vector<uint32_t> read_tokens(const std::string & path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open token ID file: " + path);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const size_t left = text.find('['), right = text.find(']', left == std::string::npos ? 0 : left);
    if (left != std::string::npos && right != std::string::npos) text = text.substr(left + 1, right - left - 1);
    for (char & c : text) if (c == ',' || c == '\n' || c == '\r' || c == '\t') c = ' ';
    std::istringstream in(text);
    std::vector<uint32_t> result;
    uint64_t token;
    while (in >> token) {
        if (token > UINT32_MAX) throw std::out_of_range("token ID exceeds u32");
        result.push_back(static_cast<uint32_t>(token));
    }
    if (result.empty()) throw std::runtime_error("token ID file contains no IDs");
    return result;
}
std::vector<uint32_t> repeat_tokens(const std::vector<uint32_t> & seed, uint32_t count) {
    std::vector<uint32_t> result(count);
    for (uint32_t i = 0; i < count; ++i) result[i] = seed[i % seed.size()];
    return result;
}
std::vector<uint8_t> kv_row(QwenCudaSessionState & session, uint32_t layer, bool key, uint32_t position) {
    ggml_tensor * tensor = key ? session.key_cache(layer) : session.value_cache(layer);
    std::vector<uint8_t> bytes(kv_row_bytes);
    ggml_backend_tensor_get(tensor, bytes.data(), static_cast<size_t>(position) * kv_row_bytes, bytes.size());
    return bytes;
}
bool finite_half(const std::vector<uint8_t> & bytes) {
    for (size_t i = 0; i + 1 < bytes.size(); i += 2) {
        uint16_t value = 0; std::memcpy(&value, bytes.data() + i, sizeof(value));
        if ((value & 0x7c00U) == 0x7c00U) return false;
    }
    return true;
}
bool finite_f32(const std::vector<float> & values) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}
bool bitwise_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
template <typename T> uint64_t fnv1a(const std::vector<T> & values) {
    uint64_t hash = 14695981039346656037ULL;
    const auto * bytes = reinterpret_cast<const uint8_t *>(values.data());
    for (size_t i = 0; i < values.size() * sizeof(T); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}
struct Sample {
    uint32_t layer, position;
    bool key;
    std::vector<uint8_t> bytes;
};
const std::vector<uint32_t> sampled_layers{0, 12, 25, 26, 32, 39};
std::vector<Sample> sample_kv(QwenCudaSessionState & session, uint32_t visible) {
    std::vector<Sample> result;
    if (visible == 0) return result;
    const std::vector<uint32_t> positions = visible == 1 ? std::vector<uint32_t>{0} :
        std::vector<uint32_t>{0, visible - 1};
    for (uint32_t layer : sampled_layers)
        for (uint32_t position : positions)
            for (bool key : {true, false}) {
                Sample item{layer, position, key, kv_row(session, layer, key, position)};
                require(finite_half(item.bytes), "non-finite F16 KV sample at layer " + std::to_string(layer));
                result.push_back(std::move(item));
            }
    return result;
}
void compare_samples(QwenCudaSessionState & session, const std::vector<Sample> & samples) {
    for (const auto & sample : samples)
        require(kv_row(session, sample.layer, sample.key, sample.position) == sample.bytes,
            "historical KV changed: layer=" + std::to_string(sample.layer) +
            " position=" + std::to_string(sample.position));
}
std::vector<Sample> sample_kv_window(QwenCudaSessionState & session, uint32_t first, uint32_t count) {
    std::vector<Sample> result;
    result.reserve(static_cast<size_t>(count) * sampled_layers.size() * 2);
    for (uint32_t position = first; position < first + count; ++position)
        for (uint32_t layer : sampled_layers)
            for (bool key : {true, false}) {
                Sample item{layer, position, key, kv_row(session, layer, key, position)};
                require(finite_half(item.bytes), "non-finite F16 KV sample in candidate comparison window");
                result.push_back(std::move(item));
            }
    return result;
}
bool bitwise_equal_samples(const std::vector<Sample> & a, const std::vector<Sample> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].layer != b[i].layer || a[i].position != b[i].position || a[i].key != b[i].key ||
            a[i].bytes != b[i].bytes) return false;
    return true;
}
uint64_t expected_prefill_scratch(uint32_t capacity, uint32_t chunk) {
    const uint64_t score_bytes = static_cast<uint64_t>(capacity) * chunk * 40 * sizeof(float);
    const uint64_t graph_overhead = (chunk == 16 ? 48ULL : 40ULL) * 1024 * 1024;
    return std::max<uint64_t>(64ULL * 1024 * 1024, score_bytes * 3 + graph_overhead);
}

struct GateConfig {
    uint32_t capacity;
    uint32_t prefix;
    uint32_t append;
    uint32_t chunk;
    const char * label;
    bool allocation_only = false;
    bool replay = false;
    bool exact_fit_overflow_test = false;
    std::vector<uint32_t> boundary_audit_ends;
};

Qwen3GenerationExecution run_gate(Qwen3Model & model,
    const std::shared_ptr<QwenCudaRuntimeState> & runtime,
    const std::vector<uint32_t> & seed, const GateConfig & gate,
    uint64_t * min_free_out = nullptr) {
    const auto free_before_session = probe_gpus();
    print_memory("before_session", free_before_session);
    auto session = runtime->create_session(gate.capacity);
    session->set_boundary_audit_end_positions(gate.boundary_audit_ends);
    for (uint32_t device : {0U, 1U}) {
        uint64_t kv_bytes = 0;
        uint32_t local_layers = 0;
        for (uint32_t layer = 0; layer < model.layer_count; ++layer) {
            if (runtime->placement().block_device_ids[layer] != device) {
                require(session->key_cache(layer, device) == nullptr && session->value_cache(layer, device) == nullptr,
                    "KV exists on a non-owning device");
                continue;
            }
            ++local_layers;
            auto * key = session->key_cache(layer, device);
            auto * value = session->value_cache(layer, device);
            require(key && value && key->type == GGML_TYPE_F16 && value->type == GGML_TYPE_F16 &&
                key->ne[0] == 128 && key->ne[1] == static_cast<int64_t>(8ULL * gate.capacity),
                "device-local KV tensor geometry mismatch");
            kv_bytes += ggml_nbytes(key) + ggml_nbytes(value);
        }
        const uint64_t packed_bytes = ggml_nbytes(session->packed_value_scratch(device));
        const uint64_t expected_kv = static_cast<uint64_t>(local_layers) * kv_bytes_per_block_token * gate.capacity;
        const uint64_t expected_packed = packed_v_bytes_per_token * gate.capacity;
        require(kv_bytes == expected_kv && packed_bytes == expected_packed,
            "local KV or packed-V bytes differ from exact capacity geometry");
        require(session->allocation_bytes(device) == expected_kv + expected_packed,
            "local KV allocation has unexplained padding/duplication");
        require(session->prefill_scratch_bytes(device) == expected_prefill_scratch(gate.capacity, gate.chunk),
            "prefill scratch does not match the capacity-qualified bound");
        std::printf("session_allocation label=%s device=%u owned_layers=%u kv_bytes=%llu packed_v_bytes=%llu "
            "kv_plus_packed_allocation_bytes=%zu prefill_scratch_bytes=%zu boundary_host_bytes=%zu\n",
            gate.label, device, local_layers, static_cast<unsigned long long>(kv_bytes),
            static_cast<unsigned long long>(packed_bytes), session->allocation_bytes(device),
            session->prefill_scratch_bytes(device), session->boundary_host_bytes());
    }
    require(session->boundary_host_is_pinned() && session->boundary_host_bytes() ==
        static_cast<size_t>(gate.chunk) * hidden_width * sizeof(float),
        "pinned boundary host allocation differs from configured F32 chunk geometry");
    const char * profile_env = std::getenv("VBUF_QWEN3_CAPTURE_DECODE_PROFILE");
    const bool capture_decode_profile = profile_env != nullptr && std::string(profile_env) == "1";
    std::optional<uint32_t> cuda_profile_context;
    if (const char * value = std::getenv("VBUF_QWEN3_CUDA_PROFILE_CONTEXT")) {
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        require(end != value && *end == '\0' && parsed <= UINT32_MAX,
            "invalid VBUF_QWEN3_CUDA_PROFILE_CONTEXT value");
        cuda_profile_context = static_cast<uint32_t>(parsed);
#if !defined(VBUF_QWEN3_HAS_CUDA_PROFILER_API)
        throw std::runtime_error("CUDA profiler API capture is unavailable in this build");
#endif
    }
    bool cuda_profile_range_started = false;
    bool cuda_profile_range_stopped = false;
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session, capture_decode_profile);
    for (uint32_t device : {0U, 1U}) {
        const auto * owned = runtime->tensor(device == 0 ? "blk.0.attn_q.weight" : "blk.26.attn_q.weight", device);
        require(owned && owned->buffer != nullptr, "model buffer missing from assigned device");
        const auto model_buffer = owned->buffer;
        require(model_buffer != session->allocation(device) && model_buffer != session->prefill_scratch(device) &&
            model_buffer != session->decode_scratch(device), "model allocation aliases session scratch/KV");
        require(session->allocation(device) != session->prefill_scratch(device) &&
            session->allocation(device) != session->decode_scratch(device) &&
            session->prefill_scratch(device) != session->decode_scratch(device),
            "session KV, prefill, and decode allocations alias");
        require(session->packed_value_scratch(device)->buffer == session->allocation(device),
            "packed-V is not owned by the device-local KV allocation");
        std::printf("buffer_ownership label=%s device=%u model_kv_prefill_decode_distinct=YES packed_v=kv_allocation "
            "boundary=separate_pinned_host rows=%u\n", gate.label, device, gate.chunk);
    }
    auto after_session_and_graphs = probe_gpus();
    print_memory("after_session_and_executor", after_session_and_graphs);
    std::vector<size_t> minimum_free(2);
    for (size_t i = 0; i < 2; ++i) minimum_free[i] = after_session_and_graphs[i].free;

    if (gate.allocation_only) {
        std::printf("capacity_gate label=%s capacity=%u visible_prefix=0 mode=allocation_only result=PASS\n",
            gate.label, gate.capacity);
        return {};
    }
    require(gate.prefix != 0 && gate.prefix + gate.append <= gate.capacity,
        "gate request prefix+append exceeds capacity");
    const auto prompt = repeat_tokens(seed, gate.prefix);
    size_t next_checkpoint = 0;
    std::vector<uint32_t> checkpoints{256, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32736};
    if (gate.prefix >= 256) checkpoints.push_back(gate.prefix);
    std::sort(checkpoints.begin(), checkpoints.end());
    checkpoints.erase(std::unique(checkpoints.begin(), checkpoints.end()), checkpoints.end());
    std::vector<Sample> historical_samples;
    std::vector<Sample> progressive_samples;
    bool captured_prompt_kv = false;
    auto on_token = [&](uint32_t, uint32_t decode_position) {
        if (!(capture_decode_profile && captured_prompt_kv)) {
            historical_samples = sample_kv(*session, gate.prefix);
            captured_prompt_kv = true;
        }
        if (cuda_profile_context && decode_position == *cuda_profile_context && !cuda_profile_range_started) {
#if defined(VBUF_QWEN3_HAS_CUDA_PROFILER_API)
            const cudaError_t status = cudaProfilerStart();
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("cudaProfilerStart failed: ") + cudaGetErrorString(status));
            cuda_profile_range_started = true;
#endif
        }
        return true;
    };
    auto on_progress = [&](uint32_t visible, const std::vector<float> & hidden) {
        if (cuda_profile_range_started && !cuda_profile_range_stopped && cuda_profile_context &&
            visible == *cuda_profile_context + 1) {
#if defined(VBUF_QWEN3_HAS_CUDA_PROFILER_API)
            const cudaError_t status = cudaProfilerStop();
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("cudaProfilerStop failed: ") + cudaGetErrorString(status));
            cuda_profile_range_stopped = true;
#endif
        }
        if (capture_decode_profile && visible > gate.prefix) return;
        const auto current_memory = probe_gpus();
        for (size_t i = 0; i < 2; ++i) minimum_free[i] = std::min(minimum_free[i], current_memory[i].free);
        while (next_checkpoint < checkpoints.size() && checkpoints[next_checkpoint] <= visible &&
            checkpoints[next_checkpoint] <= gate.prefix) {
            const uint32_t checkpoint = checkpoints[next_checkpoint];
            auto samples = sample_kv(*session, checkpoint);
            progressive_samples.insert(progressive_samples.end(), samples.begin(), samples.end());
            double sum_sq = 0.0;
            for (float value : hidden) sum_sq += static_cast<double>(value) * value;
            require(finite_f32(hidden), "non-finite final hidden at prefix checkpoint");
            std::printf("prefix_checkpoint label=%s capacity=%u visible=%u hidden_rms=%.9g kv_samples=%zu kv=finite "
                "free0=%zu free1=%zu\n", gate.label, gate.capacity, checkpoint,
                std::sqrt(sum_sq / hidden.size()), samples.size(), current_memory[0].free, current_memory[1].free);
            ++next_checkpoint;
        }
    };
    const char * qualify_env = std::getenv("VBUF_QWEN3_QUALIFY_PREBOUND_DISPATCH");
    const bool qualify_prebound = qualify_env != nullptr && std::string(qualify_env) == "1";
    const char * compare_modes_env = std::getenv("VBUF_QWEN3_COMPARE_DISABLED_SHADOW");
    const bool compare_modes = compare_modes_env != nullptr && std::string(compare_modes_env) == "1";
    require(!cuda_profile_context || (*cuda_profile_context == gate.prefix && gate.append == 1 &&
        !qualify_prebound && !compare_modes && !capture_decode_profile),
        "CUDA kernel capture requires one canonical decode at the profiled prefix without host profiling");
    require(!qualify_prebound || (gate.capacity == 32768 && gate.chunk == 32 && gate.append >= 32),
        "prebound dispatch qualification requires capacity=32768, chunk=32, and at least 32 decode steps");
    require(!compare_modes || (gate.capacity == 32768 && gate.chunk == 32 && gate.prefix <= 64 && gate.append >= 32),
        "DISABLED/SHADOW comparison is restricted to the short-prefix capacity-32768 gate");
    struct TimedExecution { Qwen3GenerationExecution value; uint64_t wall_ns = 0; };
    const auto execute_timed = [&]() {
        const auto start = Clock::now();
        auto value = executor.run(prompt, gate.append, std::nullopt, on_token, {}, on_progress);
        return TimedExecution{std::move(value), static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count())};
    };
    Qwen3GenerationExecution execution;
    uint64_t wall_ns = 0;
    std::vector<Sample> mode_reference_historical, mode_reference_progressive, mode_reference_appended;
    std::vector<Sample> validated_appended_samples;
    if (compare_modes) {
        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Disabled);
        auto disabled = execute_timed();
        require(disabled.value.completed && disabled.value.tokens.size() == gate.append && captured_prompt_kv,
            "DISABLED comparison request did not complete/capture pre-decode KV");
        mode_reference_historical = historical_samples;
        mode_reference_progressive = progressive_samples;
        mode_reference_appended = sample_kv_window(*session, gate.prefix, gate.append);
        if (capture_decode_profile) print_decode_profile("disabled", disabled.value);
        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Shadow);
        auto shadow = execute_timed();
        require(shadow.value.completed && shadow.value.tokens == disabled.value.tokens &&
            bitwise_equal(shadow.value.final_logits, disabled.value.final_logits) &&
            bitwise_equal(shadow.value.final_hidden, disabled.value.final_hidden),
            "DISABLED and SHADOW canonical executions differ bitwise");
        require(bitwise_equal_samples(mode_reference_appended, sample_kv_window(*session, gate.prefix, gate.append)),
            "DISABLED and SHADOW appended KV rows differ bitwise");
        compare_samples(*session, mode_reference_historical);
        compare_samples(*session, mode_reference_progressive);
        std::printf("optimizer_disabled_shadow_equivalence tokens=PASS hidden=BITWISE logits=BITWISE kv=BITWISE\n");
        if (capture_decode_profile) print_decode_profile("shadow", shadow.value);
        execution = std::move(shadow.value);
        wall_ns = shadow.wall_ns;
    } else {
        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Shadow);
        auto canonical = execute_timed();
        execution = std::move(canonical.value);
        wall_ns = canonical.wall_ns;
    }
    require(execution.completed && execution.tokens.size() == gate.append && captured_prompt_kv,
        "capacity gate generation did not finish or capture pre-decode KV");
    if (cuda_profile_context) {
        require(cuda_profile_range_started && cuda_profile_range_stopped,
            "CUDA profiler range did not enclose the requested decode token");
        std::printf("cuda_kernel_profile_range context=%u decode_steps=1 capture=PASS optimizer_mode=SHADOW\n",
            *cuda_profile_context);
    }
    if (qualify_prebound) {
        const Qwen3GenerationExecution canonical = execution;
        const std::vector<Sample> canonical_historical = historical_samples;
        const std::vector<Sample> canonical_progressive = progressive_samples;
        const std::vector<Sample> canonical_appended = sample_kv_window(*session, gate.prefix, gate.append);
        validated_appended_samples = canonical_appended;
        const auto observed = runtime->execution_optimizer().snapshot();
        require(observed.last_decision.candidate_found && observed.last_decision.candidate_identity.size() != 0,
            "guarded prebound candidate was not discovered in shadow baseline");
        runtime->execution_optimizer().set_unvalidated_trial_for_testing(true);
        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Enabled);
        auto trial = execute_timed();
        runtime->execution_optimizer().set_unvalidated_trial_for_testing(false);
        require(trial.value.completed && trial.value.specialized_decode_steps != 0 &&
            trial.value.tokens == canonical.tokens && bitwise_equal(trial.value.final_logits, canonical.final_logits) &&
            bitwise_equal(trial.value.final_hidden, canonical.final_hidden),
            "prebound candidate trial differed from canonical tokens/hidden/logits");
        const std::vector<Sample> trial_appended = sample_kv_window(*session, gate.prefix, gate.append);
        require(bitwise_equal_samples(canonical_appended, trial_appended),
            "prebound candidate trial changed selected appended KV rows");
        compare_samples(*session, canonical_historical);
        compare_samples(*session, canonical_progressive);
        require(runtime->execution_optimizer().mark_candidate_valid(observed.last_decision.candidate_identity,
            "exact token/hidden/logit/sampled-KV match against canonical decode"),
            "candidate failed hotness/equivalence validation transition");
        const auto validated_plan = build_qwen_execution_plan(model, *runtime, *session,
            QwenExecutionPlanPath::MultiGpu);
        const auto validated_facts = collect_qwen_runtime_facts(validated_plan, *runtime, 1,
            session->current_length(), QwenExecutionPhase::Decode);
        const auto validated_decision = runtime->execution_optimizer().select(validated_plan, validated_facts);
        require(validated_decision.candidate_validated &&
            (validated_decision.candidate_selected || validated_decision.fallback == QwenOptimizerFallback::GuardFailed),
            "Valid candidate did not remain fail-closed under post-validation facts");
        execution = std::move(trial.value);
        wall_ns = trial.wall_ns;
        const auto validated = runtime->execution_optimizer().snapshot();
        require(validated.valid_candidate_count == 1 && execution.specialized_decode_steps != 0,
            "validated candidate was not selected/executed");
        std::printf("prebound_dispatch_qualification candidate=%s status=VALID trial=PASS exact_tokens=PASS "
            "exact_hidden=PASS exact_logits=PASS sampled_kv_rows=PASS canonical_steps=%llu specialized_steps=%llu "
            "canonical_decode_ns=%llu candidate_trial_decode_ns=%llu\n", observed.last_decision.candidate_identity.c_str(),
            static_cast<unsigned long long>(canonical.canonical_decode_steps),
            static_cast<unsigned long long>(execution.specialized_decode_steps),
            static_cast<unsigned long long>(canonical.decode_ns), static_cast<unsigned long long>(execution.decode_ns));
        if (capture_decode_profile) print_decode_profile("enabled_trial", execution);
    }
    require(session->current_length() == gate.prefix + gate.append,
        "global logical position did not equal visible prefix plus appended tokens");
    require(finite_f32(execution.final_hidden) && finite_f32(execution.final_logits),
        "capacity gate produced non-finite hidden/logits");
    compare_samples(*session, historical_samples);
    compare_samples(*session, progressive_samples);
    for (uint32_t position = gate.prefix; position < gate.prefix + gate.append; ++position)
        for (uint32_t layer : sampled_layers)
            require(finite_half(kv_row(*session, layer, true, position)) &&
                finite_half(kv_row(*session, layer, false, position)),
                "appended device-local K/V row is non-finite");
    require(execution.boundary_bytes_equal, "one or more selected boundary audits failed");
    const auto optimizer_state = runtime->execution_optimizer().snapshot();
    const bool optimizer_enabled = runtime->execution_optimizer().mode() == QwenOptimizerMode::Enabled;
    if (qualify_prebound) {
        require(optimizer_enabled && optimizer_state.valid_candidate_count == 1 &&
            optimizer_state.last_decision.candidate_validated && execution.specialized_decode_steps != 0,
            "validated candidate was not retained/enabled after qualification");
    } else {
        require(runtime->execution_optimizer().mode() == QwenOptimizerMode::Shadow &&
            optimizer_state.last_decision.canonical_selected && execution.specialized_decode_steps == 0,
            "shadow mode did not preserve canonical selection");
    }
    if (gate.capacity == 32768 && gate.prefix + gate.append <= 32767)
        require(optimizer_state.last_decision.candidate_found && optimizer_state.last_decision.guards_passed,
            "guarded decode interval did not pass at the final in-range context");
    if (gate.capacity == 32768 && gate.prefix + gate.append >= 32768)
        require(optimizer_state.last_decision.candidate_found &&
            optimizer_state.last_decision.fallback == QwenOptimizerFallback::GuardFailed,
            "candidate did not fail closed at the upper context guard");
    std::printf("optimizer_selection label=%s mode=%s candidate_found=%s cache_hit=%s guards_passed=%s "
        "eligible=%s validated=%s candidate_selected=%s canonical_selected=%s fallback=%u "
        "cache_hits=%llu cache_misses=%llu guard_passes=%llu guard_failures=%llu errors=%llu observations=%llu "
        "valid_candidates=%zu invalidated_candidates=%zu\n", gate.label,
        optimizer_enabled ? "ENABLED" : "SHADOW",
        optimizer_state.last_decision.candidate_found ? "YES" : "NO",
        optimizer_state.last_decision.cache_hit ? "YES" : "NO",
        optimizer_state.last_decision.guards_passed ? "YES" : "NO",
        optimizer_state.last_decision.candidate_eligible ? "YES" : "NO",
        optimizer_state.last_decision.candidate_validated ? "YES" : "NO",
        optimizer_state.last_decision.candidate_selected ? "YES" : "NO",
        optimizer_state.last_decision.canonical_selected ? "YES" : "NO",
        static_cast<unsigned>(optimizer_state.last_decision.fallback),
        static_cast<unsigned long long>(optimizer_state.cache_hits),
        static_cast<unsigned long long>(optimizer_state.cache_misses),
        static_cast<unsigned long long>(optimizer_state.guard_passes),
        static_cast<unsigned long long>(optimizer_state.guard_failures),
        static_cast<unsigned long long>(optimizer_state.optimizer_errors),
        static_cast<unsigned long long>(optimizer_state.observations),
        optimizer_state.valid_candidate_count, optimizer_state.invalidated_candidate_count);
    if (gate.boundary_audit_ends.empty())
        require(execution.boundary_audits == execution.boundary_handoffs,
            "all-handoff audit mode missed a boundary transfer");
    else
        require(execution.boundary_audits == gate.boundary_audit_ends.size(),
            "selected long-prefix boundary audit count mismatch");
    std::printf("capacity_gate label=%s capacity=%u chunk=%u visible_prefix=%u appended=%u final_length=%u "
        "boundary_handoffs=%llu boundary_bytes=%llu boundary_audits=%llu audited_bytes=%llu "
        "prefill_ns=%llu decode_ns=%llu wall_ns=%llu decode_ns_per_token=%llu "
        "finite=YES historical_kv=preserved kv_append=PASS free_min0=%zu free_min1=%zu\n",
        gate.label, gate.capacity, gate.chunk, gate.prefix, gate.append, session->current_length(),
        static_cast<unsigned long long>(execution.boundary_handoffs),
        static_cast<unsigned long long>(execution.boundary_bytes),
        static_cast<unsigned long long>(execution.boundary_audits),
        static_cast<unsigned long long>(execution.boundary_audit_bytes),
        static_cast<unsigned long long>(execution.prefill_ns), static_cast<unsigned long long>(execution.decode_ns),
        static_cast<unsigned long long>(wall_ns), static_cast<unsigned long long>(execution.decode_ns / gate.append),
        minimum_free[0], minimum_free[1]);
    if (capture_decode_profile) print_decode_profile(gate.label, execution);
    std::printf("output_fingerprint label=%s chunk=%u token_fnv64=%016llx hidden_f32_fnv64=%016llx "
        "logits_f32_fnv64=%016llx generated=", gate.label, gate.chunk,
        static_cast<unsigned long long>(fnv1a(execution.tokens)),
        static_cast<unsigned long long>(fnv1a(execution.final_hidden)),
        static_cast<unsigned long long>(fnv1a(execution.final_logits)));
    for (size_t i = 0; i < execution.tokens.size(); ++i)
        std::printf("%s%u", i == 0 ? "" : ",", execution.tokens[i]);
    std::printf("\n");
    for (size_t i = 0; i < minimum_free.size(); ++i)
        if (min_free_out) min_free_out[i] = std::min<uint64_t>(min_free_out[i], minimum_free[i]);

    if (gate.replay) {
        const auto allocations_before = std::array<ggml_backend_buffer_t, 2>{session->allocation(0), session->allocation(1)};
        const auto replay = executor.run(prompt, gate.append);
        require(replay.completed && replay.tokens == execution.tokens && replay.final_logits == execution.final_logits &&
            replay.final_hidden == execution.final_hidden && session->current_length() == gate.prefix + gate.append,
            "same-session capacity reset/replay was not bitwise stable");
        require(session->allocation(0) == allocations_before[0] && session->allocation(1) == allocations_before[1],
            "reset/replay replaced or duplicated the device KV allocations");
        std::printf("capacity_replay label=%s capacity=%u bitwise=PASS allocation_reused=YES\n", gate.label, gate.capacity);
        if (capture_decode_profile) print_decode_profile("replay", replay);
    }

    const char * repeat_modes_env = std::getenv("VBUF_QWEN3_REPEAT_MODE_BENCH");
    const bool repeat_mode_bench = repeat_modes_env != nullptr && std::string(repeat_modes_env) == "1";
    require(!repeat_mode_bench || (qualify_prebound && compare_modes && capture_decode_profile && gate.prefix == 32),
        "repeated mode benchmark requires short-prefix candidate qualification and decode profiling");
    if (repeat_mode_bench) {
        std::array<std::vector<uint64_t>, 3> pooled_steps;
        const std::array<QwenOptimizerMode, 3> modes{
            QwenOptimizerMode::Disabled, QwenOptimizerMode::Shadow, QwenOptimizerMode::Enabled};
        const auto mode_name = [](QwenOptimizerMode value) {
            return value == QwenOptimizerMode::Disabled ? "DISABLED" :
                value == QwenOptimizerMode::Shadow ? "SHADOW" : "ENABLED";
        };
        for (uint32_t cycle = 0; cycle < 3; ++cycle) {
            for (uint32_t offset = 0; offset < modes.size(); ++offset) {
                const uint32_t index = (cycle + offset) % modes.size();
                runtime->execution_optimizer().set_mode(modes[index]);
                auto measured = execute_timed();
                require(measured.value.completed && measured.value.tokens == execution.tokens &&
                    bitwise_equal(measured.value.final_logits, execution.final_logits) &&
                    bitwise_equal(measured.value.final_hidden, execution.final_hidden) &&
                    bitwise_equal_samples(validated_appended_samples,
                        sample_kv_window(*session, gate.prefix, gate.append)),
                    std::string(mode_name(modes[index])) + " repeat changed qualified outputs/KV");
                compare_samples(*session, historical_samples);
                compare_samples(*session, progressive_samples);
                require((modes[index] == QwenOptimizerMode::Enabled &&
                        measured.value.specialized_decode_steps == gate.append) ||
                    (modes[index] != QwenOptimizerMode::Enabled &&
                        measured.value.canonical_decode_steps == gate.append),
                    std::string(mode_name(modes[index])) + " repeat selected an unexpected path");
                pooled_steps[index].insert(pooled_steps[index].end(),
                    measured.value.decode_profile.step_wall_samples_ns.begin(),
                    measured.value.decode_profile.step_wall_samples_ns.end());
                const std::string label = std::string(mode_name(modes[index])) + "_repeat_" + std::to_string(cycle + 1);
                print_decode_profile(label.c_str(), measured.value);
            }
        }
        print_pooled_distribution("DISABLED", pooled_steps[0]);
        print_pooled_distribution("SHADOW", pooled_steps[1]);
        print_pooled_distribution("ENABLED", pooled_steps[2]);
        runtime->execution_optimizer().set_mode(QwenOptimizerMode::Enabled);
    }

    if (gate.capacity == 2048 && gate.prefix >= 512) {
        session->set_execution_failure(QwenCudaFailurePoint::None);
        bool failure_armed = false;
        const auto arm_after_long_prefill = [&](uint32_t visible, const std::vector<float> &) {
            if (visible == gate.prefix && !failure_armed) {
                session->set_execution_failure(QwenCudaFailurePoint::ExecutionAfterBoundaryD2H);
                failure_armed = true;
            }
        };
        bool injected_failure = false;
        try { (void) executor.run(prompt, 1, std::nullopt, {}, {}, arm_after_long_prefill); }
        catch (const std::exception &) { injected_failure = true; }
        require(failure_armed && injected_failure && session->current_length() == 0 &&
            finite_half(kv_row(*session, 25, true, gate.prefix)) &&
            finite_half(kv_row(*session, 25, false, gate.prefix)),
            "nontrivial-context boundary failure did not reset logical progress after early KV write");
        session->set_execution_failure(QwenCudaFailurePoint::None);
        const auto recovered = executor.run(repeat_tokens(seed, 128), 1);
        require(recovered.completed && session->current_length() == 129,
            "same-session valid request did not recover after nontrivial failure");
        std::printf("large_context_failure label=%s prior_visible=%u point=after_early_KV_D2H current_length=0 "
            "early_KV=written recovery=same_session_PASS\n", gate.label, gate.prefix);
    }

    if (gate.exact_fit_overflow_test) {
        const uint32_t committed = session->current_length();
        bool rejected = false;
        try { (void) executor.run(repeat_tokens(seed, gate.capacity), 1); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected && session->current_length() == committed,
            "over-capacity request did not fail closed before changing logical state");
        const auto recovery = executor.run(repeat_tokens(seed, 128), 1);
        require(recovery.completed && session->current_length() == 129,
            "valid request did not recover after over-capacity rejection");
        std::printf("capacity_admission label=%s exact_fit=PASS overflow_rejected_before_execution=PASS recovery=PASS\n",
            gate.label);
    }
    if (qualify_prebound) {
        const auto prompt_a = repeat_tokens(seed, 31);
        const auto prompt_b = repeat_tokens(seed, 32);
        const auto a_first = executor.run(prompt_a, 2);
        const auto b_middle = executor.run(prompt_b, 2);
        const auto a_last = executor.run(prompt_a, 2);
        require(a_first.completed && a_last.completed && b_middle.completed &&
            a_first.canonical_decode_steps == 1 && a_first.specialized_decode_steps == 1 &&
            b_middle.canonical_decode_steps == 0 && b_middle.specialized_decode_steps == 2 &&
            a_last.tokens == a_first.tokens && bitwise_equal(a_last.final_logits, a_first.final_logits) &&
            bitwise_equal(a_last.final_hidden, a_first.final_hidden),
            "A/B/A reset isolation or lower-guard entry did not preserve dispatch/KV semantics");
        std::printf("prebound_dispatch_ab_a=PASS A(c31)=%llu-canonical+%llu-specialized "
            "B(c32)=%llu-canonical+%llu-specialized outputs=BITWISE\n",
            static_cast<unsigned long long>(a_first.canonical_decode_steps),
            static_cast<unsigned long long>(a_first.specialized_decode_steps),
            static_cast<unsigned long long>(b_middle.canonical_decode_steps),
            static_cast<unsigned long long>(b_middle.specialized_decode_steps));
        const char * failure_mode = std::getenv("VBUF_QWEN3_CANDIDATE_FAILURE_MODE");
        const bool graph_failure = failure_mode != nullptr && std::string(failure_mode) == "graph";
        Qwen3GenerationExecution recovered;
        if (graph_failure) {
            bool failure_armed = false;
            bool failure_thrown = false;
            const auto arm_candidate_failure = [&](uint32_t visible, const std::vector<float> &) {
                if (visible == prompt_b.size() && !failure_armed) {
                    session->set_execution_failure(QwenCudaFailurePoint::ExecutionBeforeFinalBlock);
                    failure_armed = true;
                }
            };
            try { (void) executor.run(prompt_b, 1, std::nullopt, {}, {}, arm_candidate_failure); }
            catch (const std::exception &) { failure_thrown = true; }
            require(failure_armed && failure_thrown && session->current_length() == 0,
                "candidate graph failure did not reset committed logical progress");
            session->set_execution_failure(QwenCudaFailurePoint::None);
            const auto after_fault = runtime->execution_optimizer().snapshot();
            require(after_fault.invalidated_candidate_count == 1,
                "candidate graph failure did not invalidate the optimized plan");
            recovered = executor.run(prompt_b, 1);
            require(recovered.completed && recovered.canonical_decode_steps == 1 &&
                recovered.specialized_decode_steps == 0,
                "request retry did not fall back to canonical after candidate graph failure");
            std::printf("prebound_dispatch_failure=PASS point=before_final_block_graph candidate=INVALIDATED "
                "failed_request=RESET retry=CANONICAL\n");
        } else {
            runtime->execution_optimizer().set_fault_for_testing(QwenOptimizerFault::CandidateExecution);
            recovered = executor.run(prompt_b, 1);
            const auto after_fault = runtime->execution_optimizer().snapshot();
            require(recovered.completed && recovered.canonical_decode_steps == 1 &&
                recovered.specialized_decode_steps == 0 &&
                after_fault.last_decision.fallback == QwenOptimizerFallback::Invalidated &&
                after_fault.invalidated_candidate_count == 1,
                "injected pre-execution candidate failure did not invalidate and fall back to canonical");
            std::printf("prebound_dispatch_failure=PASS point=before_first_decode_graph candidate=INVALIDATED "
                "current_request=CANONICAL\n");
        }
        const auto canonical_recovery = executor.run(prompt_b, 1);
        require(canonical_recovery.completed && canonical_recovery.canonical_decode_steps == 1 &&
            canonical_recovery.specialized_decode_steps == 0 && canonical_recovery.tokens == recovered.tokens,
            "canonical request did not recover after candidate invalidation");
        std::printf("prebound_dispatch_recovery=PASS canonical=YES\n");
    }
    return execution;
}

int run(const char * semantic, const char * source, const char * token_file,
    uint32_t capacity, uint32_t prefix, uint32_t append, uint32_t chunk, const std::string & mode,
    bool replay, bool exact_fit_overflow) {
    const auto seed = read_tokens(token_file);
    require(capacity == 2048 || capacity == 4096 || capacity == 8192 || capacity == 16384 || capacity == 32768,
        "capacity must be one rung of 2048/4096/8192/16384/32768");
    require(chunk == 16 || chunk == 32, "prefill chunk must be 16 or 32");
    require(mode == "run" || mode == "allocation-only", "mode must be run or allocation-only");
    require(mode != "allocation-only" || (prefix == 0 && append == 0),
        "allocation-only mode requires prefix=0 and append=0");
    if (mode == "run") require(prefix != 0 && append != 0 && prefix + append <= capacity,
        "run mode requires valid prefix and append within capacity");

    Qwen3Model model;
    open_qwen3_model(semantic, source, &model, true);
    require(model.artifact_identity == std::string("sha256:") + QWEN3_14B_Q4_K_M_SHA256 &&
        model.count == 443 && model.layer_count == 40 && model.source_size == source_bytes,
        "exact admitted Qwen3-14B Q4_K_M artifact is required");
    print_memory("before_runtime", probe_gpus());
    QwenCudaRuntimeConfig runtime_config;
    runtime_config.placement = QwenCudaPlacement::contiguous_split(0, 1, 26);
    runtime_config.prefill_chunk_size = chunk;
    // Qualification-only opt-in. VbufModelRuntime production admission remains capped at 1032.
    runtime_config.allow_experimental_capacity = true;
    auto runtime = QwenCudaRuntimeState::create(model, runtime_config);
    require(runtime->device_count() == 2 && runtime->uploaded_payload_bytes() == tensor_payload_bytes &&
        runtime->uploaded_tensor_count() == model.count, "device model residency is incomplete or duplicated");
    print_memory("after_runtime", probe_gpus());
    for (uint32_t device : {0U, 1U})
        std::printf("model_residency device=%u payload_bytes=%llu tensor_count=%zu allocation_bytes=%zu\n",
            device, static_cast<unsigned long long>(runtime->uploaded_payload_bytes(device)),
            runtime->uploaded_tensor_count(device), runtime->resident_allocation_bytes(device));

    GateConfig gate{capacity, prefix, append, chunk, mode.c_str(), mode == "allocation-only", replay,
        exact_fit_overflow, {}};
    const char * profile_env = std::getenv("VBUF_QWEN3_CAPTURE_DECODE_PROFILE");
    const bool capture_decode_profile = profile_env != nullptr && std::string(profile_env) == "1";
    const char * qualify_env = std::getenv("VBUF_QWEN3_QUALIFY_PREBOUND_DISPATCH");
    const bool qualify_prebound = qualify_env != nullptr && std::string(qualify_env) == "1";
    if (mode == "run" && capture_decode_profile && !qualify_prebound) {
        gate.boundary_audit_ends.push_back(prefix + append);
    } else if (mode == "run" && prefix >= 4096) {
        for (uint32_t checkpoint : {256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U, 24576U})
            if (checkpoint <= prefix) gate.boundary_audit_ends.push_back(checkpoint);
        if (prefix % 32 == 0) gate.boundary_audit_ends.push_back(prefix);
        for (uint32_t p = prefix + 1; p <= prefix + append; ++p) gate.boundary_audit_ends.push_back(p);
        std::sort(gate.boundary_audit_ends.begin(), gate.boundary_audit_ends.end());
        gate.boundary_audit_ends.erase(std::unique(gate.boundary_audit_ends.begin(),
            gate.boundary_audit_ends.end()), gate.boundary_audit_ends.end());
    }
    uint64_t min_free[2]{UINT64_MAX, UINT64_MAX};
    (void) run_gate(model, runtime, seed, gate, min_free);
    print_memory("after_session_destruction", probe_gpus());
    runtime.reset();
    print_memory("after_runtime_destruction", probe_gpus());
    if (mode == "run")
        std::printf("gate_result capacity=%u prefix=%u append=%u production_capacity=1032_UNCHANGED\n",
            capacity, prefix, append);
    else
        std::printf("gate_result capacity=%u allocation_only=PASS production_capacity=1032_UNCHANGED\n", capacity);
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 10 && argc != 11) throw std::invalid_argument(
            "usage: qwen3_multigpu_capacity_gate SEMANTIC SOURCE TOKEN_IDS_FILE CAPACITY PREFIX APPEND CHUNK(16|32) run|allocation-only REPLAY(0|1) [OVERFLOW(0|1)]");
        const bool overflow = argc > 10 && std::string(argv[10]) == "1";
        return run(argv[1], argv[2], argv[3], static_cast<uint32_t>(std::stoul(argv[4])),
            static_cast<uint32_t>(std::stoul(argv[5])), static_cast<uint32_t>(std::stoul(argv[6])),
            static_cast<uint32_t>(std::stoul(argv[7])), argv[8], std::string(argv[9]) == "1", overflow);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "qwen3_multigpu_capacity_gate=FAIL: %s\n", error.what());
        return 1;
    }
}
