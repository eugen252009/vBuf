#include "qwen3_attention_av.h"
#include "qwen3_execution_plan.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vbuf_ggml;
namespace {
constexpr int64_t dim = 128;
constexpr int64_t kv_heads = 8;
constexpr int64_t query_heads = 40;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

struct InputData {
    int64_t capacity;
    int64_t rows;
    ggml_type value_type;
    std::vector<uint8_t> values;
    std::vector<float> weights;
    std::vector<int32_t> positions;
    std::vector<float> reference;
};

InputData make_input(int64_t capacity, int64_t rows, ggml_type value_type) {
    InputData input{capacity, rows, value_type};
    const size_t element_bytes = ggml_type_size(value_type);
    input.values.resize(static_cast<size_t>(dim * kv_heads * capacity) * element_bytes);
    input.weights.resize(static_cast<size_t>(capacity * rows * query_heads));
    input.positions.resize(static_cast<size_t>(rows));
    input.reference.resize(static_cast<size_t>(dim * rows * query_heads));

    for (int64_t p = 0; p < capacity; ++p) {
        for (int64_t h = 0; h < kv_heads; ++h) {
            for (int64_t d = 0; d < dim; ++d) {
                const float value = std::sin(static_cast<float>((p + 3) * 17 + h * 11 + d * 5) * 0.0017f) * 0.7f;
                const size_t index = static_cast<size_t>(d + dim * (p * kv_heads + h));
                if (value_type == GGML_TYPE_F16) {
                    const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
                    std::memcpy(input.values.data() + index * sizeof(encoded), &encoded, sizeof(encoded));
                } else {
                    std::memcpy(input.values.data() + index * sizeof(value), &value, sizeof(value));
                }
            }
        }
    }

    for (int64_t q = 0; q < rows; ++q) {
        const int64_t last = std::min<int64_t>(capacity - 1, (capacity > rows ? 3 : 0) + q * 2);
        input.positions[static_cast<size_t>(q)] = static_cast<int32_t>(last);
        for (int64_t h = 0; h < query_heads; ++h) {
            double sum = 0.0;
            for (int64_t p = 0; p <= last; ++p) {
                const float weight = 1.0f + static_cast<float>((p * 17 + q * 13 + h * 5) % 23) / 23.0f;
                input.weights[static_cast<size_t>(p + capacity * (q + rows * h))] = weight;
                sum += weight;
            }
            for (int64_t p = 0; p <= last; ++p)
                input.weights[static_cast<size_t>(p + capacity * (q + rows * h))] = static_cast<float>(
                    input.weights[static_cast<size_t>(p + capacity * (q + rows * h))] / sum);
            for (int64_t p = last + 1; p < capacity; ++p)
                input.weights[static_cast<size_t>(p + capacity * (q + rows * h))] = 0.0f;
        }
    }

    for (int64_t h = 0; h < query_heads; ++h) {
        const int64_t kv_h = h / (query_heads / kv_heads);
        for (int64_t q = 0; q < rows; ++q) {
            for (int64_t d = 0; d < dim; ++d) {
                double sum = 0.0;
                const int64_t last = input.positions[static_cast<size_t>(q)];
                for (int64_t p = 0; p <= last; ++p) {
                    const size_t v_index = static_cast<size_t>(d + dim * (p * kv_heads + kv_h));
                    float value;
                    if (value_type == GGML_TYPE_F16) {
                        ggml_fp16_t encoded;
                        std::memcpy(&encoded, input.values.data() + v_index * sizeof(encoded), sizeof(encoded));
                        value = ggml_fp16_to_fp32(encoded);
                    } else {
                        std::memcpy(&value, input.values.data() + v_index * sizeof(value), sizeof(value));
                    }
                    const float weight = input.weights[static_cast<size_t>(p + capacity * (q + rows * h))];
                    sum += static_cast<double>(weight) * value;
                }
                input.reference[static_cast<size_t>(d + dim * (q + rows * h))] = static_cast<float>(sum);
            }
        }
    }
    return input;
}

std::vector<float> execute(ggml_backend_t backend, const InputData & input,
    std::vector<float> * canonical_result, bool * supported) {
    ggml_init_params params{8 * 1024 * 1024, nullptr, true};
    ggml_context * context = ggml_init(params);
    require(context != nullptr, "ggml context allocation failed");
    ggml_tensor * value = ggml_new_tensor_2d(context, input.value_type, dim, input.capacity * kv_heads);
    ggml_tensor * weights = ggml_new_tensor_3d(context, GGML_TYPE_F32,
        input.capacity, input.rows, query_heads);
    ggml_tensor * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, input.rows);
    ggml_tensor * output = ggml_attention_av(context, value, weights, positions);
    ggml_tensor * v_reshaped = ggml_reshape_3d(context, value, dim, kv_heads, input.capacity);
    ggml_tensor * packed = ggml_new_tensor_3d(context, input.value_type, input.capacity, dim, kv_heads);
    ggml_tensor * copied = ggml_cpy(context, ggml_permute(context, v_reshaped, 1, 2, 0, 3), packed);
    ggml_tensor * canonical = ggml_mul_mat(context, copied, weights);
    if (!ggml_backend_dev_supports_op(ggml_backend_get_device(backend), output) ||
        !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), canonical)) {
        *supported = false;
        ggml_free(context);
        return {};
    }
    *supported = true;
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, canonical);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, "backend tensor allocation failed");
    ggml_backend_tensor_set(value, input.values.data(), 0, input.values.size());
    ggml_backend_tensor_set(weights, input.weights.data(), 0, input.weights.size() * sizeof(float));
    ggml_backend_tensor_set(positions, input.positions.data(), 0, input.positions.size() * sizeof(int32_t));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    require(status == GGML_STATUS_SUCCESS, "attention AV graph compute failed");

    std::vector<float> result(input.reference.size());
    canonical_result->resize(input.reference.size());
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_tensor_get(canonical, canonical_result->data(), 0, canonical_result->size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return result;
}

void check_result(const std::vector<float> & expected, const std::vector<float> & actual,
    const std::string & backend_name, int64_t capacity, int64_t rows, ggml_type value_type,
    double tolerance) {
    require(expected.size() == actual.size(), backend_name + " output size mismatch");
    double max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), backend_name + " produced non-finite output");
        max_abs = std::max(max_abs, std::abs(static_cast<double>(expected[i]) - actual[i]));
    }
    require(max_abs < tolerance, backend_name + " exceeded AV reference tolerance: " + std::to_string(max_abs));
    std::cout << "av_backend=" << backend_name << " capacity=" << capacity << " rows=" << rows
        << " value_type=" << ggml_type_name(value_type) << " max_abs=" << max_abs << " parity=PASS\n";
}

QwenExecutionPlan make_plan() {
    QwenExecutionPlanInput input;
    input.model_identity = "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
    input.backend_family = "GGML_CUDA";
    input.stable_device_identities = {"CUDA0@0000:04:00.0", "CUDA1@0000:07:00.0"};
    input.stable_device_sm_versions = {86, 75};
    input.block_device_ids.assign(26, 0);
    input.block_device_ids.insert(input.block_device_ids.end(), 14, 1);
    input.embedding_device_id = 0;
    input.final_norm_device_id = 1;
    input.output_head_device_id = 1;
    input.capacity = 512;
    input.prefill_chunk_size = 32;
    return make_qwen_execution_plan(input);
}

void test_capability_guards() {
    Qwen3AttentionAVFacts facts;
    facts.sm_version = 86;
    facts.capacity = 512;
    facts.visible_context = 32;
    facts.query_rows = 32;
    facts.kv_heads = 8;
    facts.query_heads = 40;
    facts.head_dimension = 128;
    facts.value_type = GGML_TYPE_F16;
    facts.probability_type = GGML_TYPE_F32;
    facts.position_type = GGML_TYPE_I32;
    facts.canonical_position_major_v = true;
    const auto selected = resolve_qwen3_attention_av(facts, true);
    require(selected.path == Qwen3AttentionAVPath::NativeLayout && selected.native_supported,
        "qualified SM86 F16 AV capability did not select native layout");

    facts.sm_version = 75;
    auto fallback = resolve_qwen3_attention_av(facts, true);
    require(fallback.path == Qwen3AttentionAVPath::PackedCanonical && !fallback.native_supported,
        "unsupported SM75 did not fail closed to packed canonical AV");
    facts.sm_version = 86;
    facts.value_type = GGML_TYPE_Q4_0;
    fallback = resolve_qwen3_attention_av(facts, true);
    require(fallback.path == Qwen3AttentionAVPath::PackedCanonical,
        "unsupported V type did not fail closed");
    facts.value_type = GGML_TYPE_F16;
    facts.canonical_position_major_v = false;
    fallback = resolve_qwen3_attention_av(facts, true);
    require(fallback.path == Qwen3AttentionAVPath::PackedCanonical,
        "noncanonical V layout did not fail closed");
    facts.canonical_position_major_v = true;
    facts.capacity = 1024;
    fallback = resolve_qwen3_attention_av(facts, true);
    require(fallback.path == Qwen3AttentionAVPath::PackedCanonical && !fallback.native_supported,
        "capacity above the measured numeric gate did not fail closed to canonical AV");
    facts.capacity = 512;
    facts.query_rows = 16;
    fallback = resolve_qwen3_attention_av(facts, true);
    require(fallback.path == Qwen3AttentionAVPath::PackedCanonical,
        "unsupported row count did not fail closed");
}

void test_optimizer_candidate_guards() {
    const auto plan = make_plan();
    auto unsupported_capacity_plan = plan;
    unsupported_capacity_plan.capacity = 1024;
    bool rejected_unsupported_capacity = false;
    try {
        (void) make_qwen3_native_attention_av_candidate(
            unsupported_capacity_plan, QwenExecutionPhase::Decode);
    } catch (const std::invalid_argument &) {
        rejected_unsupported_capacity = true;
    }
    require(rejected_unsupported_capacity,
        "native AV candidate factory accepted a capacity above its numeric qualification limit");
    const auto candidate = make_qwen3_native_attention_av_candidate(plan, QwenExecutionPhase::Decode);
    QwenExecutionPlanOptimizer optimizer;
    require(optimizer.register_candidate(candidate), "native AV candidate registration failed");
    auto facts = QwenRuntimeFacts{};
    facts.rows = 1;
    facts.context_length = 31;
    facts.capacity = plan.capacity;
    facts.prefill_chunk_size = plan.prefill_chunk_size;
    facts.phase = QwenExecutionPhase::Decode;
    facts.placement_identity = plan.placement_identity;
    facts.activation_dtype = plan.activation_dtype;
    facts.kv_dtype = plan.kv_dtype;
    facts.devices = {{0, plan.stable_device_identities[0], 86}, {1, plan.stable_device_identities[1], 75}};

    optimizer.set_mode(QwenOptimizerMode::Enabled);
    auto decision = optimizer.select(plan, facts, candidate.identity);
    require(decision.fallback == QwenOptimizerFallback::CandidateNotValidated &&
        !decision.candidate_selected && decision.canonical_selected,
        "unvalidated native candidate did not fail closed");
    optimizer.set_mode(QwenOptimizerMode::Shadow);
    optimizer.record_execution(plan, facts, QwenExecutionPlanOptimizer::min_candidate_hotness_observations, 1);
    require(optimizer.mark_candidate_valid(candidate.identity, "direct AV contract"),
        "could not mark directly qualified synthetic candidate valid");
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    decision = optimizer.select(plan, facts, candidate.identity);
    require(decision.candidate_selected && !decision.canonical_selected &&
        decision.strategy == QwenCandidateStrategy::NativeLayoutAttentionAV,
        "validated native AV candidate was not selected");
    facts.context_length = 33;
    decision = optimizer.select(plan, facts, candidate.identity);
    require(!decision.candidate_selected && decision.canonical_selected &&
        decision.fallback == QwenOptimizerFallback::GuardFailed,
        "native AV candidate escaped its measured visible-context guard");
    facts.context_length = 31;
    facts.devices[0].sm_version = 75;
    decision = optimizer.select(plan, facts, candidate.identity);
    require(!decision.candidate_selected && decision.canonical_selected &&
        decision.fallback == QwenOptimizerFallback::GuardFailed,
        "SM mismatch did not select canonical fallback");
    decision = optimizer.select(plan, facts);
    require(decision.canonical_selected && decision.strategy == QwenCandidateStrategy::PreboundDecodeDispatch,
        "implicit optimizer path did not remain independent of explicit native AV candidate");
}

bool test_backend(ggml_backend_t backend, const std::string & name) {
    for (const auto [capacity, rows] : {
            std::pair<int64_t, int64_t>{37, 1}, {64, 32}, {1032, 32}}) {
        for (ggml_type value_type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            const InputData input = make_input(capacity, rows, value_type);
            bool supported = false;
            std::vector<float> canonical;
            const std::vector<float> output = execute(backend, input, &canonical, &supported);
            if (!supported) return false;
            check_result(input.reference, output, name + "/native", capacity, rows, value_type, 3e-6);
            check_result(input.reference, canonical, name + "/packed", capacity, rows, value_type, 1e-3);
            double max_native_packed = 0.0;
            for (size_t i = 0; i < output.size(); ++i)
                max_native_packed = std::max(max_native_packed,
                    std::abs(static_cast<double>(output[i]) - canonical[i]));
            require(max_native_packed < 1e-3, name + " native-versus-packed AV exceeded direct parity tolerance");
            std::cout << "av_direct_native_packed_max_abs=" << max_native_packed << '\n';
        }
    }
    return true;
}
} // namespace

int main() {
    try {
        test_capability_guards();
        test_optimizer_candidate_guards();
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        require(cpu != nullptr, "CPU backend initialization failed");
        require(test_backend(cpu, "CPU"), "CPU backend does not support attention AV");
        ggml_backend_free(cpu);

        ggml_backend_reg_t cuda = ggml_backend_reg_by_name("CUDA");
        if (cuda == nullptr || ggml_backend_reg_dev_count(cuda) == 0) {
            std::cout << "CUDA direct AV numerical test skipped: no CUDA device\n";
            return 77;
        }
        bool ran_cuda = false;
        for (size_t i = 0; i < ggml_backend_reg_dev_count(cuda); ++i) {
            ggml_backend_dev_t device = ggml_backend_reg_dev_get(cuda, i);
            ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
            if (backend == nullptr) continue;
            if (test_backend(backend, ggml_backend_dev_name(device))) ran_cuda = true;
            ggml_backend_free(backend);
            if (ran_cuda) break;
        }
        require(ran_cuda, "no CUDA device completed direct native AV numerical tests");
        std::cout << "qwen3_native_attention_av_contract=PASS cpu+cuda capability=fail-closed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_native_attention_av_contract=FAIL: " << error.what() << '\n';
        return 1;
    }
}
