#include "vbuf_high_precision_reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml::reference;
using namespace vbuf_ml::numerics;
namespace fs = std::filesystem;

namespace {
constexpr size_t head_dimension = 128;
constexpr size_t query_heads = 40;
constexpr size_t kv_heads = 8;
constexpr size_t embedding_width = head_dimension * query_heads;
constexpr const char * model_identity =
    "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
constexpr const char * placement = "multi:0x26,1x14;emb=0;norm=1;head=1";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
std::vector<T> read_elements(std::ifstream & input, size_t count, const std::string & section) {
    if (count > static_cast<size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
        throw std::overflow_error("capture section is too large: " + section);
    std::vector<T> result(count);
    input.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw std::runtime_error("truncated capture section: " + section);
    return result;
}

std::map<std::string, std::string> read_metadata(const fs::path & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open capture metadata: " + path.string());
    std::map<std::string, std::string> result;
    std::string line;
    while (std::getline(input, line)) {
        const size_t equal = line.find('=');
        if (equal == std::string::npos) continue;
        result[line.substr(0, equal)] = line.substr(equal + 1);
    }
    return result;
}

size_t metadata_size(const std::map<std::string, std::string> & metadata, const std::string & key) {
    const auto found = metadata.find(key);
    if (found == metadata.end()) throw std::runtime_error("capture metadata is missing " + key);
    size_t consumed = 0;
    const size_t value = std::stoull(found->second, &consumed);
    if (consumed != found->second.size()) throw std::runtime_error("invalid capture metadata " + key);
    return value;
}

struct Capture {
    fs::path path;
    std::string phase;
    size_t rows = 0;
    size_t visible = 0;
    size_t layer = 0;
    size_t capacity = 0;
    std::vector<uint16_t> values_f16;
    std::vector<float> probabilities;
    std::vector<int32_t> positions;
    std::vector<float> canonical;
    std::vector<float> native;
    std::vector<float> oracle;
    std::vector<uint16_t> keys_f16;
    std::vector<float> query;
    std::vector<float> scores;
    std::vector<float> qk_oracle;
    std::vector<float> softmax_actual_scores;
    std::vector<float> softmax_oracle_scores;
};

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open binary fixture: " + path.string());
    input.seekg(0, std::ios::end);
    const auto length = input.tellg();
    if (length < 0) throw std::runtime_error("cannot size binary fixture: " + path.string());
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("cannot read binary fixture: " + path.string());
    return bytes;
}

template<typename T>
std::vector<T> read_raw_elements(const fs::path & path, size_t count) {
    const auto bytes = read_bytes(path);
    if (bytes.size() != count * sizeof(T))
        throw std::runtime_error("binary fixture has an unexpected byte count: " + path.string());
    std::vector<T> values(count);
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

Capture read_capture(const fs::path & directory, size_t capacity, const std::string & phase) {
    std::ostringstream stem_stream;
    stem_stream << "boundary-capacity-" << capacity << '-' << phase << "-layer-00";
    const std::string stem = stem_stream.str();
    const auto metadata = read_metadata(directory / (stem + ".meta"));
    require(metadata.at("format") == "qwen3-native-av-boundary-v1", "unsupported Qwen boundary capture format");
    require(metadata.at("phase") == phase && metadata_size(metadata, "capacity") == capacity &&
        metadata_size(metadata, "layer") == 0 && metadata_size(metadata, "query_heads") == query_heads &&
        metadata_size(metadata, "kv_heads") == kv_heads && metadata_size(metadata, "head_dim") == head_dimension,
        "capture metadata does not match the qualified Qwen3 layer-0 fixture geometry");
    require(metadata.at("value_type") == "F16" && metadata.at("probability_type") == "F32" &&
        metadata.at("qk_capture") == "yes", "capture is missing required F16/F32 QK sections");

    Capture capture;
    capture.path = directory / (stem + ".bin");
    capture.phase = phase;
    capture.rows = metadata_size(metadata, "query_rows");
    capture.visible = metadata_size(metadata, "visible_context");
    capture.layer = metadata_size(metadata, "layer");
    capture.capacity = capacity;
    require((phase == "prefill" && capture.rows == 32 && capture.visible == 32) ||
        (phase == "decode" && capture.rows == 1 && capture.visible == 33),
        "unexpected captured prefill/decode extent");
    std::ifstream input(capture.path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open boundary capture: " + capture.path.string());
    const size_t values = head_dimension * capture.visible * kv_heads;
    const size_t probabilities = capture.visible * capture.rows * query_heads;
    const size_t outputs = head_dimension * capture.rows * query_heads;
    const size_t scores = capture.visible * capture.rows * query_heads;
    capture.values_f16 = read_elements<uint16_t>(input, values, "V_F16");
    capture.probabilities = read_elements<float>(input, probabilities, "P_F32");
    capture.positions = read_elements<int32_t>(input, capture.rows, "positions_I32");
    capture.canonical = read_elements<float>(input, outputs, "canonical_F32");
    capture.native = read_elements<float>(input, outputs, "native_F32");
    capture.oracle = read_elements<float>(input, outputs, "oracle_F32");
    capture.keys_f16 = read_elements<uint16_t>(input, values, "K_F16");
    capture.query = read_elements<float>(input, head_dimension * query_heads * capture.rows, "Q_F32");
    capture.scores = read_elements<float>(input, scores, "scores_F32");
    capture.qk_oracle = read_elements<float>(input, scores, "qk_oracle_F32");
    capture.softmax_actual_scores = read_elements<float>(input, probabilities, "softmax_from_scores_F32");
    capture.softmax_oracle_scores = read_elements<float>(input, probabilities, "softmax_from_qk_oracle_F32");
    char trailing = 0;
    input.read(&trailing, 1);
    require(input.eof(), "capture contains unrecognized trailing payload");
    require(std::all_of(capture.positions.begin(), capture.positions.end(), [&](int32_t position) {
        return position >= 0 && static_cast<size_t>(position) < capture.visible;
    }), "capture contains an invalid causal position");
    return capture;
}

Tensor make_f32_tensor(const std::vector<float> & values, std::vector<size_t> shape) {
    Tensor result;
    result.type = ScalarType::F32;
    result.shape = std::move(shape);
    result.f32 = values;
    return result;
}

void require_bitwise_equal(const std::vector<float> & expected, const std::vector<float> & actual,
        const std::string & name) {
    require(expected.size() == actual.size(), name + " element count mismatch");
    if (expected.empty() || std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0)
        return;
    size_t first_difference = 0;
    while (first_difference < expected.size() &&
            std::memcmp(&expected[first_difference], &actual[first_difference], sizeof(float)) == 0)
        ++first_difference;
    std::ostringstream message;
    message << name << " differs bitwise at element " << first_difference;
    if (first_difference < expected.size())
        message << " (expected=" << std::setprecision(9) << expected[first_difference]
                << ", actual=" << actual[first_difference] << ')';
    throw std::runtime_error(message.str());
}

std::vector<float> convert_legacy_matrix(const std::vector<float> & source,
        size_t rows, size_t heads, size_t positions) {
    std::vector<float> result(rows * heads * positions);
    for (size_t q = 0; q < rows; ++q) for (size_t h = 0; h < heads; ++h)
        for (size_t p = 0; p < positions; ++p)
            result[(q * heads + h) * positions + p] = source[p + positions * (q + rows * h)];
    return result;
}

std::vector<size_t> causal_extents(const Capture & capture, size_t heads) {
    std::vector<size_t> result;
    result.reserve(capture.rows * heads);
    for (const int32_t position : capture.positions)
        for (size_t head = 0; head < heads; ++head) result.push_back(static_cast<size_t>(position) + 1);
    return result;
}

Tensor legacy_av_output(const Tensor & qhd) {
    require(qhd.shape.size() == 3 && qhd.shape[1] == query_heads && qhd.shape[2] == head_dimension,
        "reference AV result shape mismatch");
    const size_t rows = qhd.shape[0];
    Tensor result;
    result.type = ScalarType::F32;
    result.shape = {head_dimension, rows, query_heads};
    result.f32.resize(head_dimension * rows * query_heads);
    for (size_t q = 0; q < rows; ++q) for (size_t h = 0; h < query_heads; ++h)
        for (size_t d = 0; d < head_dimension; ++d)
            result.f32[d + head_dimension * (q + rows * h)] =
                qhd.f32[(q * query_heads + h) * head_dimension + d];
    return result;
}

EvaluationContext base_context(const Capture & capture, const std::string & operation,
        const std::string & implementation, const std::string & input_dtype,
        std::vector<uint64_t> output_shape) {
    EvaluationContext context;
    context.output_name = operation;
    context.candidate_identity = "diagnostic-only:" + implementation;
    context.qualification_run_identity = "historical-qwen3-capture-replay:v1:" +
        capture.path.parent_path().filename().string() + ":cap" +
        std::to_string(capture.capacity) + ":" + capture.phase;
    context.model_identity = model_identity;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = implementation;
    context.device_family = "CUDA";
    context.device_identities = {"CUDA0@0000:04:00.0"};
    context.device_sm_versions = {86};
    context.placement_identity = placement;
    context.phase = capture.phase;
    context.execution_topology = "native-av-side-branch";
    const std::string phase_fixture = capture.phase == "prefill" ? "prefill32" : "decode32";
    context.fixture_identity = "qwen3-native-av-" + phase_fixture + "-cap" +
        std::to_string(capture.capacity) + "-layer" + std::to_string(capture.layer);
    context.input_identity = "captured-qwen3-layer0-boundary:" + capture.path.parent_path().filename().string();
    context.output_dtype = "F32";
    context.input_dtype = input_dtype;
    context.output_shape = std::move(output_shape);
    context.capacity = static_cast<uint32_t>(capture.capacity);
    context.context_length = static_cast<uint32_t>(capture.visible);
    context.rows = static_cast<uint32_t>(capture.rows);
    context.logical_inputs_equivalent = true;
    return context;
}

ReferenceProvenance provenance(std::string accumulation, std::string input, std::string output,
        std::map<std::string, std::string> parameters) {
    ReferenceProvenance result;
    result.accumulation_precision = std::move(accumulation);
    result.input_representation = std::move(input);
    result.output_representation = std::move(output);
    result.operation_parameters = std::move(parameters);
    result.replayed_input_data = true;
    return result;
}

void require_diagnostic_only(const NumericalEvaluation & evaluation, const std::string & name) {
    require(evaluation.status == EvaluationStatus::NotTested &&
        evaluation.contract_status == ContractStatus::NeedsCalibration &&
        evaluation.replayed_metrics_only,
        name + " unexpectedly became authorizing evidence: " + evaluation_json(evaluation));
    std::cout << "qwen3_high_precision_reference " << evaluation_json(evaluation) << '\n';
}

void qualify_q4k_projection(const fs::path & weights_path, const fs::path & activation_path,
        const fs::path & candidate_path) {
    constexpr size_t input_dimension = 5120;
    constexpr size_t input_rows = 25;
    constexpr size_t output_channels = 64;
    const size_t weight_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, input_dimension);
    const auto packed_weights = read_bytes(weights_path);
    require(packed_weights.size() == weight_row_bytes * input_dimension,
        "Q4_K weight payload size does not match Qwen3 block-0 attention-Q matrix geometry");
    const size_t activation_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, input_dimension);
    const auto packed_activations = read_bytes(activation_path);
    require(packed_activations.size() == activation_row_bytes * input_rows,
        "Q8_K activation fixture size does not match the 25-row Qwen3 prefix");
    const auto packed_output = read_raw_elements<float>(candidate_path, input_rows * input_dimension);

    const Tensor decoded_weights = decode_ggml_rows(GGML_TYPE_Q4_K, packed_weights.data(),
        output_channels * weight_row_bytes, output_channels, input_dimension);
    const Tensor decoded_activations = decode_ggml_rows(GGML_TYPE_Q8_K, packed_activations.data(),
        packed_activations.size(), input_rows, input_dimension);
    const TensorView weight_transposed{decoded_weights.f32.data(), ScalarType::F32,
        {input_dimension, output_channels},
        {static_cast<ptrdiff_t>(sizeof(float)),
         static_cast<ptrdiff_t>(input_dimension * sizeof(float))}};
    const Tensor reference = matmul(view(decoded_activations), weight_transposed,
        Accumulation::F64, ScalarType::F32);
    std::vector<float> candidate_values(input_rows * output_channels);
    for (size_t row = 0; row < input_rows; ++row) for (size_t channel = 0; channel < output_channels; ++channel)
        candidate_values[row * output_channels + channel] = packed_output[row * input_dimension + channel];
    const Tensor candidate = make_f32_tensor(candidate_values, {input_rows, output_channels});

    EvaluationContext context;
    context.candidate_identity = "diagnostic-only:qwen3-q4k-cpu-q-projection";
    context.qualification_run_identity = "historical-qwen3-q4k-prefix25-projection-replay:v1";
    context.model_identity = model_identity;
    context.backend_family = "GGML_CPU";
    context.implementation_identity = "ggml-cpu-q4k-projection";
    context.device_family = "CPU";
    context.placement_identity = "single:cpu-reference";
    context.phase = "prefill";
    context.execution_topology = "single-block-q4k-q-projection";
    context.fixture_identity = "qwen3-q4k-prefix25-layer0-q-projection-first64-output-channels";
    context.input_identity = "Q4_K:blk.0.attn_q.weight;Q8_K:prefix25-q-input";
    context.input_dtype = "Q4_K+Q8_K";
    context.capacity = input_rows;
    context.context_length = input_rows;
    context.rows = input_rows;
    context.logical_inputs_equivalent = true;
    ReferenceProvenance q4k_provenance;
    q4k_provenance.accumulation_precision = "binary64-product-and-left-to-right-sum";
    q4k_provenance.input_representation = "Q4_K represented block-0 attn_q weights decoded to F32; Q8_K represented 25-row activation decoded to F32";
    q4k_provenance.output_representation = "F32-rounded-once";
    q4k_provenance.operation_parameters = {{"M", "25"}, {"N", "64"}, {"K", "5120"},
        {"weight_layout", "GGML [output_channel,input_feature], transposed for logical matmul"},
        {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"},
        {"candidate_scope", "first 64 of 5120 output channels"},
        {"quantization_error_attribution", "inputs are the same decoded Q4_K and Q8_K representations"}};
    q4k_provenance.replayed_input_data = true;
    const auto evaluation = evaluate_reference_comparison("vbuf.general_matmul.fp64.operation_accuracy", 1,
        reference, candidate, context, q4k_provenance);
    require_diagnostic_only(evaluation, "Q4_K Q8_K projection");
}

std::string qk_dispatch_family(const Capture & capture) {
    // Pinned GGML 2d191b5 dispatch predicates for CUDA0 (SM 8.6):
    // F16 K, F32 Q, ne0=128, ne2=8. MMVF handles one-row decode only
    // through capacity 512; MMF also requires capacity divisible by 32 and
    // at most 16 query columns, so 32-row prefill falls through to cuBLAS.
    if (capture.phase == "decode" && capture.capacity <= 512) return "MMVF";
    if (capture.phase == "decode" && capture.capacity % 32 == 0) return "MMF";
    return "cuBLAS";
}

std::string metric_number(const NumericalMetrics & metrics, Metric metric);

void emit_av_proposal_metrics(const Capture & capture, const std::string & candidate_identity,
        const NumericalMetrics & metrics) {
    std::cout << "{\"record_type\":\"diagnostic_scope_proposal_metrics\","
        << "\"contract_proposal_id\":\"qwen3.attention_av.fp64_operation_accuracy\","
        << "\"contract_status\":\"NEEDS_CALIBRATION\",\"evaluation_status\":\"NOT_TESTED\","
        << "\"candidate_identity\":\"diagnostic-only:" << candidate_identity << "\","
        << "\"reference_identity\":\"qwen3-av-boundary-fp64-oracle-v1\","
        << "\"model_identity\":\"" << model_identity << "\",\"capacity\":" << capture.capacity
        << ",\"rows\":" << capture.rows << ",\"context_length\":" << capture.visible
        << ",\"layer\":" << capture.layer << ",\"phase\":\"" << capture.phase << "\","
        << "\"relative_rms_error\":" << metric_number(metrics, Metric::RelativeRmsError)
        << ",\"max_absolute_error\":" << metric_number(metrics, Metric::MaxAbsoluteError)
        << ",\"cosine_similarity\":" << metric_number(metrics, Metric::CosineSimilarity)
        << ",\"production_authority\":false}\n";
}

void qualify_capture(const fs::path & directory, size_t capacity, const std::string & phase) {
    const Capture capture = read_capture(directory, capacity, phase);
    const size_t rows = capture.rows, visible = capture.visible;
    const auto decoded_keys = decode_ggml_rows(GGML_TYPE_F16, capture.keys_f16.data(),
        capture.keys_f16.size() * sizeof(uint16_t), visible * kv_heads, head_dimension);
    const auto decoded_values = decode_ggml_rows(GGML_TYPE_F16, capture.values_f16.data(),
        capture.values_f16.size() * sizeof(uint16_t), visible * kv_heads, head_dimension);
    const TensorView key_view{decoded_keys.f32.data(), ScalarType::F32,
        {visible, kv_heads, head_dimension},
        {static_cast<ptrdiff_t>(kv_heads * head_dimension * sizeof(float)),
         static_cast<ptrdiff_t>(head_dimension * sizeof(float)), static_cast<ptrdiff_t>(sizeof(float))}};
    const TensorView value_view{decoded_values.f32.data(), ScalarType::F32,
        {visible, kv_heads, head_dimension},
        {static_cast<ptrdiff_t>(kv_heads * head_dimension * sizeof(float)),
         static_cast<ptrdiff_t>(head_dimension * sizeof(float)), static_cast<ptrdiff_t>(sizeof(float))}};
    const TensorView query_view = contiguous_view(capture.query.data(), {rows, query_heads, head_dimension});
    const Tensor qk_reference_result = qk_scores(query_view, key_view, Accumulation::F64, ScalarType::F32);
    const Tensor qk_candidate = make_f32_tensor(
        convert_legacy_matrix(capture.scores, rows, query_heads, visible), {rows, query_heads, visible});
    const Tensor saved_qk_oracle = make_f32_tensor(
        convert_legacy_matrix(capture.qk_oracle, rows, query_heads, visible), {rows, query_heads, visible});
    require_bitwise_equal(saved_qk_oracle.f32, qk_reference_result.f32,
        "independent QK replay versus captured FP64 QK reference");
    const std::string qk_family = qk_dispatch_family(capture);
    auto qk_context = base_context(capture, "qk_matmul", "ggml-cuda-qk-" + qk_family, "F32+F16",
        {rows, query_heads, visible});
    auto qk_eval = evaluate_reference_comparison("vbuf.qk_matmul.fp64.operation_accuracy", 1,
        qk_reference_result, qk_candidate, qk_context,
        provenance("binary64-product-and-left-to-right-sum",
            "Q=F32 post-RoPE;K=F16 cache decoded by pinned GGML trait",
            "F32-rounded-once", {{"head_dimension", "128"}, {"query_heads", "40"},
                {"kv_heads", "8"}, {"gqa_mapping", "floor(query_head/5)"},
                {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"},
                {"attention_scale_applied", "false;applied by softmax"},
                {"dispatch_family", qk_family},
                {"dispatch_provenance", "derived from pinned GGML CUDA predicates and captured tensor geometry"},
                {"key_capacity", std::to_string(capture.capacity)}}));
    require_diagnostic_only(qk_eval, phase + " QK");
    std::vector<float> causal_qk_reference, causal_qk_candidate;
    for (size_t q = 0; q < rows; ++q) for (size_t h = 0; h < query_heads; ++h)
        for (size_t p = 0; p <= static_cast<size_t>(capture.positions[q]); ++p) {
            const size_t index = (q * query_heads + h) * visible + p;
            causal_qk_reference.push_back(qk_reference_result.f32[index]);
            causal_qk_candidate.push_back(qk_candidate.f32[index]);
        }
    EvaluationContext causal_qk_context;
    causal_qk_context.output_shape = {causal_qk_reference.size()};
    const auto causal_qk_metrics = measure_tensor_pair(
        make_tensor_view(causal_qk_reference, causal_qk_context.output_shape),
        make_tensor_view(causal_qk_candidate, causal_qk_context.output_shape), causal_qk_context);

    std::vector<float> score_rows = qk_candidate.f32;
    std::vector<float> probability_rows(rows * query_heads * visible);
    for (size_t q = 0; q < rows; ++q) for (size_t h = 0; h < query_heads; ++h)
        for (size_t p = 0; p < visible; ++p)
            probability_rows[(q * query_heads + h) * visible + p] =
                capture.probabilities[p + visible * (q + rows * h)];
    const std::vector<size_t> extents = causal_extents(capture, query_heads);
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(head_dimension));
    const auto high_precision_softmax = softmax(
        contiguous_view(score_rows.data(), {rows * query_heads, visible}),
        attention_scale, extents, nullptr, ScalarType::F32);
    const Tensor saved_softmax_from_scores = make_f32_tensor(
        convert_legacy_matrix(capture.softmax_actual_scores, rows, query_heads, visible),
        {rows * query_heads, visible});
    const auto high_precision_softmax_from_qk_oracle = softmax(
        contiguous_view(saved_qk_oracle.f32.data(), {rows * query_heads, visible}),
        attention_scale, extents, nullptr, ScalarType::F32);
    const Tensor saved_softmax_from_qk_oracle = make_f32_tensor(
        convert_legacy_matrix(capture.softmax_oracle_scores, rows, query_heads, visible),
        {rows * query_heads, visible});
    require_bitwise_equal(saved_softmax_from_scores.f32, high_precision_softmax.f32,
        "independent softmax replay from captured CUDA scores");
    require_bitwise_equal(saved_softmax_from_qk_oracle.f32, high_precision_softmax_from_qk_oracle.f32,
        "independent softmax replay from captured FP64 QK reference");
    std::cout << "capture_replay_integrity phase=" << phase
        << " layer=0 qk_oracle=BITWISE_MATCH softmax_from_scores=BITWISE_MATCH"
        << " softmax_from_fp64_qk=BITWISE_MATCH\n";
    const Tensor probability_candidate = make_f32_tensor(probability_rows, {rows * query_heads, visible});
    auto softmax_context = base_context(capture, "softmax", "ggml-cuda-softmax-f32", "F32",
        {rows * query_heads, visible});
    const auto invariants = probability_invariants(view(probability_candidate), extents);
    softmax_context.invariant_values.insert(invariants.begin(), invariants.end());
    auto softmax_eval = evaluate_reference_comparison("vbuf.softmax.fp64.operation_accuracy", 1,
        high_precision_softmax, probability_candidate, softmax_context,
        provenance("binary64-exp-and-normalization",
            "captured F32 QK scores; unmasked causal prefix only", "F32-rounded-once",
            {{"axis", "last"}, {"scale", "1/sqrt(128)"},
                {"logical_extents", "per-query-position+1"}, {"masked_output", "exact-zero"},
                {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"}}));
    require_diagnostic_only(softmax_eval, phase + " softmax");
    const auto oracle_qk_probability_metrics = measure_tensor_pair(
        numeric_view(high_precision_softmax_from_qk_oracle), numeric_view(probability_candidate), softmax_context);
    std::cout << "qwen3_causal_active_diagnostic phase=" << phase << " layer=0"
        << " qk_all_positions_relative_rms=" << metric_number(qk_eval.measured_metrics, Metric::RelativeRmsError)
        << " qk_causal_positions_relative_rms=" << metric_number(causal_qk_metrics, Metric::RelativeRmsError)
        << " cuda_probabilities_vs_fp64_softmax_of_cuda_qk_relative_rms="
        << metric_number(softmax_eval.measured_metrics, Metric::RelativeRmsError)
        << " cuda_probabilities_vs_fp64_softmax_of_fp64_qk_relative_rms="
        << metric_number(oracle_qk_probability_metrics, Metric::RelativeRmsError) << '\n';

    const Tensor probabilities_qhp = make_f32_tensor(probability_rows, {rows, query_heads, visible});
    const Tensor av_reference_qhd = attention_av(view(probabilities_qhp), value_view,
        [&] {
            std::vector<size_t> per_query;
            per_query.reserve(rows);
            for (int32_t position : capture.positions) per_query.push_back(static_cast<size_t>(position) + 1);
            return per_query;
        }(), Accumulation::F64, ScalarType::F32);
    const Tensor av_reference = legacy_av_output(av_reference_qhd);
    const Tensor saved_av_oracle = make_f32_tensor(capture.oracle,
        {head_dimension, rows, query_heads});
    require_bitwise_equal(saved_av_oracle.f32, av_reference.f32,
        "independent AV replay versus captured ascending-position FP64 oracle");
    const Tensor native_candidate = make_f32_tensor(capture.native,
        {head_dimension, rows, query_heads});
    auto av_context = base_context(capture, "attention_av", "ggml-native-layout-av", "F16",
        {head_dimension, rows, query_heads});
    const auto av_provenance = provenance("binary64-product-and-ascending-position-sum",
        "captured probabilities=F32;V=F16 cache decoded by pinned GGML trait",
        "F32-rounded-once", {{"query_heads", "40"}, {"kv_heads", "8"},
            {"head_dimension", "128"}, {"gqa_mapping", "floor(query_head/5)"},
            {"visible_extent", "per-query-position+1"},
            {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"}});
    if (capture.capacity == 512) {
        const auto av_eval = evaluate_reference_comparison(
            "qwen3.attention_av.fp64_model_boundary_observation", 1,
            av_reference, native_candidate, av_context, av_provenance);
        require_diagnostic_only(av_eval, phase + " native AV");
    }

    const Tensor canonical_candidate = make_f32_tensor(capture.canonical,
        {head_dimension, rows, query_heads});
    const auto canonical_metrics = measure_tensor_pair(numeric_view(av_reference),
        numeric_view(canonical_candidate), av_context);
    const auto native_metrics = measure_tensor_pair(numeric_view(av_reference),
        numeric_view(native_candidate), av_context);
    emit_av_proposal_metrics(capture, "ggml-cuda-canonical-packed-v-av", canonical_metrics);
    emit_av_proposal_metrics(capture, "ggml-native-layout-av", native_metrics);
    const auto relative = [](const NumericalMetrics & metrics) {
        const auto found = metrics.values.find(Metric::RelativeRmsError);
        if (found == metrics.values.end() || !found->second.numeric_value) return std::string("null");
        std::ostringstream value;
        value << std::setprecision(17) << *found->second.numeric_value;
        return value.str();
    };
    std::cout << "qwen3_reference_capture_check phase=" << phase << " layer=0 capacity=" << capture.capacity << " rows=" << rows
        << " visible=" << visible << " saved_fp64_oracle=bitwise-match"
        << " av_contract_scope=" << (capture.capacity == 512 ? "applicable" : "out-of-scope-diagnostic-only")
        << " canonical_relative_rms=" << relative(canonical_metrics)
        << " native_relative_rms=" << relative(native_metrics) << '\n';
}

EvaluationContext operator_capture_context(const std::string & operation, const std::string & implementation,
        const std::string & input_dtype, std::vector<uint64_t> output_shape) {
    EvaluationContext context;
    context.output_name = operation;
    context.candidate_identity = "diagnostic-only:" + implementation;
    context.qualification_run_identity = "historical-qwen3-operator-capture-replay:v1:layer21-position24";
    context.model_identity = model_identity;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = implementation;
    context.device_family = "CUDA";
    context.device_identities = {"CUDA0@0000:04:00.0"};
    context.device_sm_versions = {86};
    context.placement_identity = placement;
    context.phase = "prefill";
    context.execution_topology = "resident-operator-capture-replay";
    context.fixture_identity = "qwen3-resident-prefix32-layer21-position24";
    context.input_identity = "captured-qwen3-resident-layer21-position24";
    context.output_dtype = "F32";
    context.input_dtype = input_dtype;
    context.output_shape = std::move(output_shape);
    context.capacity = 32;
    context.context_length = 32;
    context.rows = 32;
    context.logical_inputs_equivalent = true;
    return context;
}

std::string metric_number(const NumericalMetrics & metrics, Metric metric) {
    const auto found = metrics.values.find(metric);
    if (found == metrics.values.end() || !found->second.numeric_value) return "null";
    std::ostringstream value;
    value << std::setprecision(17) << *found->second.numeric_value;
    return value.str();
}

void qualify_operator_capture(const fs::path & directory, const fs::path & rmsnorm_weight_path,
        const std::string & tensor_sha256, const std::string & gguf_sha256) {
    require(tensor_sha256.size() == 64 && gguf_sha256 ==
        "915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6",
        "RMSNorm weight provenance does not match the registered Qwen3 source GGUF");
    const auto metadata_path = directory / "resident-capture.meta";
    const auto metadata_bytes = read_bytes(metadata_path);
    const std::string metadata(metadata_bytes.begin(), metadata_bytes.end());
    for (const std::string required : {
            "layer=21 position=24 positions=32",
            "layer-21-layer_input type=f32 ne=5120,32,1,1 bytes=655360",
            "layer-21-attention_rmsnorm type=f32 ne=5120,32,1,1 bytes=655360",
            "layer-21-q_rope type=f32 ne=128,40,32,1 bytes=655360",
            "layer-21-k_cache_f16 type=f16 ne=128,8,32,1 bytes=65536",
            "layer-21-v_cache_f16 type=f16 ne=128,8,32,1 bytes=65536",
            "layer-21-attention_scores type=f32 ne=32,32,40,1 bytes=163840",
            "layer-21-attention_probabilities type=f32 ne=32,32,40,1 bytes=163840",
            "layer-21-attention_context type=f32 ne=5120,32,1,1 bytes=655360"})
        require(metadata.find(required) != std::string::npos,
            "operator capture metadata is missing expected geometry: " + required);

    constexpr size_t rows = 32;
    constexpr size_t visible = 32;
    const fs::path input_path = directory / "layer-21-layer_input.bin";
    const auto input = read_raw_elements<float>(input_path, rows * embedding_width);
    const auto rmsnorm_candidate_values = read_raw_elements<float>(directory / "layer-21-attention_rmsnorm.bin",
        rows * embedding_width);
    const auto rmsnorm_scale_values = read_raw_elements<float>(rmsnorm_weight_path, embedding_width);

    const auto q_values = read_raw_elements<float>(directory / "layer-21-q_rope.bin",
        rows * query_heads * head_dimension);
    const auto k_values = read_raw_elements<uint16_t>(directory / "layer-21-k_cache_f16.bin",
        visible * kv_heads * head_dimension);
    const auto score_values_legacy = read_raw_elements<float>(directory / "layer-21-attention_scores.bin",
        rows * query_heads * visible);
    const auto probability_values_legacy = read_raw_elements<float>(directory / "layer-21-attention_probabilities.bin",
        rows * query_heads * visible);
    const auto score_values = convert_legacy_matrix(score_values_legacy, rows, query_heads, visible);
    const auto probability_values = convert_legacy_matrix(probability_values_legacy, rows, query_heads, visible);
    const auto v_values = read_raw_elements<uint16_t>(directory / "layer-21-v_cache_f16.bin",
        visible * kv_heads * head_dimension);
    const auto av_candidate_values = read_raw_elements<float>(directory / "layer-21-attention_context.bin",
        rows * query_heads * head_dimension);

    const auto decoded_keys = decode_ggml_rows(GGML_TYPE_F16, k_values.data(),
        k_values.size() * sizeof(uint16_t), visible * kv_heads, head_dimension);
    const auto decoded_values = decode_ggml_rows(GGML_TYPE_F16, v_values.data(),
        v_values.size() * sizeof(uint16_t), visible * kv_heads, head_dimension);
    const TensorView query_view = contiguous_view(q_values.data(), {rows, query_heads, head_dimension});
    const TensorView key_view = contiguous_view(decoded_keys.f32.data(), {visible, kv_heads, head_dimension});
    const Tensor qk_reference = qk_scores(query_view, key_view, Accumulation::F64, ScalarType::F32);
    const Tensor qk_candidate = make_f32_tensor(score_values, {rows, query_heads, visible});
    auto qk_context = operator_capture_context("qk_matmul", "ggml-cuda-qk-cuBLAS", "Q=F32;K=F16",
        {rows, query_heads, visible});
    const auto qk_eval = evaluate_reference_comparison("vbuf.qk_matmul.fp64.operation_accuracy", 1,
        qk_reference, qk_candidate, qk_context,
        provenance("binary64-product-and-left-to-right-sum",
            "Q=F32 post-RoPE;K=F16 captured cache decoded by pinned GGML trait",
            "F32-rounded-once", {{"head_dimension", "128"}, {"query_heads", "40"},
                {"kv_heads", "8"}, {"gqa_mapping", "floor(query_head/5)"},
                {"physical_key_capacity", "32"}, {"dispatch_family", "cuBLAS (source-derived: query rows 32 exceed MMF 16-column limit)"},
                {"attention_scale_applied", "false;applied by softmax"},
                {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"}}));
    require_diagnostic_only(qk_eval, "layer-21 QK");

    std::vector<size_t> extents;
    extents.reserve(rows * query_heads);
    for (size_t row = 0; row < rows; ++row)
        for (size_t head = 0; head < query_heads; ++head) extents.push_back(row + 1);
    const Tensor softmax_reference = softmax(
        contiguous_view(score_values.data(), {rows * query_heads, visible}),
        1.0 / std::sqrt(static_cast<double>(head_dimension)), extents, nullptr, ScalarType::F32);
    const Tensor softmax_candidate = make_f32_tensor(probability_values, {rows * query_heads, visible});
    auto softmax_context = operator_capture_context("softmax", "ggml-cuda-softmax-f32", "F32",
        {rows * query_heads, visible});
    const auto softmax_invariants = probability_invariants(view(softmax_candidate), extents);
    softmax_context.invariant_values.insert(softmax_invariants.begin(), softmax_invariants.end());
    const auto softmax_eval = evaluate_reference_comparison("vbuf.softmax.fp64.operation_accuracy", 1,
        softmax_reference, softmax_candidate, softmax_context,
        provenance("binary64-exp-and-normalization", "captured F32 QK scores; causal prefix mask",
            "F32-rounded-once", {{"axis", "last"}, {"scale", "1/sqrt(128)"},
                {"logical_extents", "row+1 across 40 query heads"}, {"masked_output", "exact-zero"},
                {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"}}));
    require_diagnostic_only(softmax_eval, "layer-21 softmax");

    const Tensor rmsnorm_reference = rms_norm(contiguous_view(input.data(), {rows, embedding_width}),
        contiguous_view(rmsnorm_scale_values.data(), {embedding_width}), 1e-6, ScalarType::F32);
    const Tensor rmsnorm_candidate = make_f32_tensor(rmsnorm_candidate_values, {rows, embedding_width});
    auto rmsnorm_context = operator_capture_context("rmsnorm", "ggml-cuda-rmsnorm-f32-mul", "F32+F32",
        {rows, embedding_width});
    rmsnorm_context.invariant_values["epsilon_semantics_match"] = true;
    const auto rmsnorm_eval = evaluate_reference_comparison("vbuf.rmsnorm.fp64.operation_accuracy", 1,
        rmsnorm_reference, rmsnorm_candidate, rmsnorm_context,
        provenance("binary64-sum-of-squares-and-normalization",
            "captured F32 input; exact F32 blk.21.attn_norm.weight from source GGUF",
            "F32-rounded-once", {{"normalized_dimension", "5120"}, {"epsilon", "1e-6"},
                {"scale_tensor", "blk.21.attn_norm.weight:F32[5120]"},
                {"scale_tensor_sha256", tensor_sha256}, {"source_gguf_sha256", gguf_sha256},
                {"operator", "ggml_rms_norm then elementwise multiply"},
                {"ggml_commit", "2d191b5dee1a591c41ee8a653ce42bfcd9c8716d"}}));
    require_diagnostic_only(rmsnorm_eval, "layer-21 RMSNorm");

    std::vector<size_t> per_query_extents(rows, visible);
    for (size_t row = 0; row < rows; ++row) per_query_extents[row] = row + 1;
    const Tensor av_reference = attention_av(
        contiguous_view(probability_values.data(), {rows, query_heads, visible}),
        TensorView{decoded_values.f32.data(), ScalarType::F32,
            {visible, kv_heads, head_dimension},
            {static_cast<ptrdiff_t>(kv_heads * head_dimension * sizeof(float)),
             static_cast<ptrdiff_t>(head_dimension * sizeof(float)), static_cast<ptrdiff_t>(sizeof(float))}},
        per_query_extents, Accumulation::F64, ScalarType::F32);
    const Tensor av_candidate = make_f32_tensor(av_candidate_values, {rows, query_heads, head_dimension});
    auto av_context = operator_capture_context("attention_av", "ggml-cuda-canonical-packed-v-av", "F16",
        {rows, query_heads, head_dimension});
    const NumericalMetrics av_metrics = measure_tensor_pair(numeric_view(av_reference),
        numeric_view(av_candidate), av_context);
    std::cout << "{\"record_type\":\"diagnostic_scope_proposal_metrics\","
        << "\"contract_proposal_id\":\"qwen3.attention_av.fp64_operation_accuracy\","
        << "\"contract_status\":\"NEEDS_CALIBRATION\",\"evaluation_status\":\"NOT_TESTED\","
        << "\"candidate_identity\":\"diagnostic-only:ggml-cuda-canonical-packed-v-av\","
        << "\"reference_identity\":\"qwen3-av-boundary-fp64-oracle-v1\","
        << "\"model_identity\":\"" << model_identity << "\",\"capacity\":32,\"rows\":32,"
        << "\"layer\":21,\"phase\":\"prefill\",\"relative_rms_error\":"
        << metric_number(av_metrics, Metric::RelativeRmsError) << ",\"max_absolute_error\":"
        << metric_number(av_metrics, Metric::MaxAbsoluteError) << ",\"cosine_similarity\":"
        << metric_number(av_metrics, Metric::CosineSimilarity) << ",\"production_authority\":false}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "--operator-capture") {
            const fs::path directory(argv[2]);
            require(fs::exists(directory), "operator capture directory does not exist: " + directory.string());
            qualify_operator_capture(directory, argv[3], argv[4], argv[5]);
            std::cout << "vbuf_qwen3_reference_capture_qualification=PASS operator_capture="
                << directory.string() << " revision=" << implementation_revision()
                << " policy_version=" << project_policy_version() << " authorizing=false\n";
            return 0;
        }
        const bool q4k_projection = argc >= 3 && std::string(argv[2]) == "--q4k-qproj";
        require((!q4k_projection && (argc == 2 || argc == 3)) || (q4k_projection && argc == 6),
            "usage: vbuf_qwen3_reference_capture_qualification CAPTURE_DIRECTORY [CAPACITY] "
            "| CAPTURE_DIRECTORY --q4k-qproj Q4K_WEIGHT_BYTES Q8K_ACTIVATION_BYTES CPU_OUTPUT_F32 "
            "| --operator-capture DIRECTORY RMSNORM_WEIGHT_F32 RMSNORM_WEIGHT_SHA256 SOURCE_GGUF_SHA256");
        const fs::path directory = fs::path(argv[1]);
        require(fs::exists(directory), "capture directory does not exist: " + directory.string());
        const size_t capacity = !q4k_projection && argc == 3 ?
            static_cast<size_t>(std::stoull(argv[2])) : 512;
        qualify_capture(directory, capacity, "prefill");
        qualify_capture(directory, capacity, "decode");
        if (q4k_projection)
            qualify_q4k_projection(argv[3], argv[4], argv[5]);
        std::cout << "vbuf_qwen3_reference_capture_qualification=PASS source="
                  << directory.string() << " capacity=" << capacity
                  << " revision=" << implementation_revision()
                  << " policy_version=" << project_policy_version()
                  << " authorizing=false\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_qwen3_reference_capture_qualification=FAIL: " << error.what() << '\n';
        return 1;
    }
}
