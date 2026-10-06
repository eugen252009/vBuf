#include "qwen3_generation.h"
#include "qwen3_model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
    Qwen3MultiDeviceGenerationExecutor executor(model, runtime, session);
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
    auto on_token = [&](uint32_t, uint32_t) {
        historical_samples = sample_kv(*session, gate.prefix);
        captured_prompt_kv = true;
        return true;
    };
    auto on_progress = [&](uint32_t visible, const std::vector<float> & hidden) {
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
    const auto start = Clock::now();
    Qwen3GenerationExecution execution = executor.run(prompt, gate.append, std::nullopt,
        on_token, {}, on_progress);
    const uint64_t wall_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    require(execution.completed && execution.tokens.size() == gate.append && captured_prompt_kv,
        "capacity gate generation did not finish or capture pre-decode KV");
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
    if (mode == "run" && prefix >= 4096) {
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
