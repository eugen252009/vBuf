#include "vbuf_high_precision_reference.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml::reference;
using namespace vbuf_ml::numerics;

namespace {
void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename F>
void must_throw(F && callback, const std::string & message) {
    bool threw = false;
    try { callback(); } catch (const std::exception &) { threw = true; }
    require(threw, message);
}

bool near(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

void test_strided_matmul_and_precision() {
    const float left_storage[] = {1, 2, 3, -9, 4, 5, 6, -9};
    const float right_storage[] = {7, 9, 11, 8, 10, 12};
    const TensorView left{left_storage, ScalarType::F32, {2, 3},
        {4 * static_cast<ptrdiff_t>(sizeof(float)), static_cast<ptrdiff_t>(sizeof(float))}};
    // The logical B matrix is transposed relative to its backing storage.
    const TensorView right{right_storage, ScalarType::F32, {3, 2},
        {static_cast<ptrdiff_t>(sizeof(float)), 3 * static_cast<ptrdiff_t>(sizeof(float))}};
    const Tensor result = matmul(left, right, Accumulation::F64, ScalarType::F64);
    require(result.shape == std::vector<size_t>({2, 2}) && result.type == ScalarType::F64,
        "strided matmul output metadata mismatch");
    require(result.at(0) == 58.0 && result.at(1) == 64.0 &&
        result.at(2) == 139.0 && result.at(3) == 154.0,
        "strided/transposed matmul values mismatch");

    const float cancellation_left[] = {1.0e8f, 1.0f, -1.0e8f};
    const float ones[] = {1.0f, 1.0f, 1.0f};
    const auto f64 = matmul(contiguous_view(cancellation_left, {1, 3}),
        contiguous_view(ones, {3, 1}), Accumulation::F64, ScalarType::F64);
    const auto f32 = matmul(contiguous_view(cancellation_left, {1, 3}),
        contiguous_view(ones, {3, 1}), Accumulation::F32, ScalarType::F32);
    require(f64.at(0) == 1.0 && f32.at(0) == 0.0,
        "explicit F32 and F64 accumulation modes were not distinguished");

    must_throw([&] { (void) matmul(contiguous_view(ones, {3, 1}),
        contiguous_view(ones, {2, 1})); }, "incompatible matmul dimensions were accepted");
    const size_t small_limit = 2;
    must_throw([&] { (void) matmul(contiguous_view(left_storage, {2, 3}),
        contiguous_view(right_storage, {3, 2}), Accumulation::F64, ScalarType::F32,
        Limits{small_limit}); }, "reference output element limit was ignored");
}

void test_qk_gqa_and_layout() {
    const float query[] = {
        1, 0,  0, 1,  1, 1,  -1, 1,
    };
    const float key[] = {
        2, 3,  5, 7,
        11, 13, 17, 19,
    };
    const Tensor result = qk_scores(contiguous_view(query, {1, 4, 2}),
        contiguous_view(key, {2, 2, 2}), Accumulation::F64, ScalarType::F32);
    const double expected[] = {2, 11, 3, 13, 12, 36, 2, 2};
    require(result.shape == std::vector<size_t>({1, 4, 2}), "QK output geometry mismatch");
    for (size_t i = 0; i < result.size(); ++i)
        require(result.at(i) == expected[i], "QK dot/GQA head mapping mismatch");

    must_throw([&] { (void) qk_scores(contiguous_view(query, {1, 3, 2}),
        contiguous_view(key, {2, 2, 2})); }, "non-integral GQA head ratio was accepted");
    const float bad_query[] = {std::numeric_limits<float>::infinity(), 0};
    must_throw([&] { (void) qk_scores(contiguous_view(bad_query, {1, 1, 2}),
        contiguous_view(key, {2, 2, 2})); }, "non-finite QK input was accepted");
}

void test_masked_stable_softmax() {
    const double scores[] = {1000.0, -2000.0, 999.0, 5.0,
        0.0, -1.0, 2.0, 4.0};
    const uint8_t mask_values[] = {0, 1, 0, 0, 0, 0, 1, 0};
    const MaskView mask{mask_values, {2, 4}, {4, 1}};
    const auto output = softmax(contiguous_view(scores, {2, 4}), 1.0, {3, 4}, &mask);
    require(output.at(0) > 0.7 && output.at(2) > 0.2 && output.at(1) == 0.0 && output.at(3) == 0.0,
        "masked stable softmax did not honor mask/logical extent");
    require(output.at(6) == 0.0, "explicitly masked second-row probability was not zero");
    const auto invariants = probability_invariants(view(output), {3, 4}, &mask);
    require(invariants.at("probabilities_in_unit_interval") &&
        invariants.at("probability_rows_normalized") && invariants.at("masked_positions_zero"),
        "softmax hard invariants did not pass on normalized masked rows");
    const float misnormalized[] = {0.5001f, 0.5f};
    const auto bad_invariants = probability_invariants(contiguous_view(misnormalized, {1, 2}), {});
    require(bad_invariants.at("probabilities_in_unit_interval") &&
        !bad_invariants.at("probability_rows_normalized"),
        "probability invariant check accepted a materially misnormalized row");

    const double invalid_masked[] = {0.0, std::numeric_limits<double>::quiet_NaN()};
    const uint8_t all_masked_second[] = {0, 1};
    const MaskView ignore_nan{all_masked_second, {1, 2}, {2, 1}};
    const auto ignored = softmax(contiguous_view(invalid_masked, {1, 2}), 1.0, {}, &ignore_nan);
    require(ignored.at(0) == 1.0 && ignored.at(1) == 0.0,
        "masked non-finite score affected the logical softmax row");
    must_throw([&] { (void) softmax(contiguous_view(scores, {2, 4}), 1.0, {0, 4}); },
        "empty logical softmax row was accepted");
    const uint8_t all_masked[] = {1, 1};
    const MaskView all_masked_view{all_masked, {1, 2}, {2, 1}};
    must_throw([&] { (void) softmax(contiguous_view(scores, {1, 2}), 1.0, {}, &all_masked_view); },
        "all-masked softmax row was assigned a distribution");
    const double infinite[] = {std::numeric_limits<double>::infinity(), 0.0};
    must_throw([&] { (void) softmax(contiguous_view(infinite, {1, 2})); },
        "unmasked infinite softmax score was accepted");
}

void test_gqa_av_layouts_and_visible_extent() {
    const float probabilities[] = {
        0.2f, 0.3f, 0.5f, 0.2f, 0.3f, 0.5f,
        0.2f, 0.3f, 0.5f, 0.2f, 0.3f, 0.5f,
    };
    // Logical [position,kv_head,dimension] representation.
    const float canonical[] = {
        1, 2,  3, 4,
        5, 6,  7, 8,
        9, 10, 11, 12,
    };
    // Same logical values stored [kv_head,position,dimension].
    const float native[] = {
        1, 2,  5, 6,  9, 10,
        3, 4,  7, 8,  11, 12,
    };
    const TensorView native_logical{native, ScalarType::F32, {3, 2, 2},
        {2 * static_cast<ptrdiff_t>(sizeof(float)),
         3 * 2 * static_cast<ptrdiff_t>(sizeof(float)),
         static_cast<ptrdiff_t>(sizeof(float))}};
    const auto p = contiguous_view(probabilities, {1, 4, 3});
    const auto packed_result = attention_av(p, contiguous_view(canonical, {3, 2, 2}), {3},
        Accumulation::F64, ScalarType::F64);
    const auto native_result = attention_av(p, native_logical, {3},
        Accumulation::F64, ScalarType::F64);
    require(packed_result.shape == std::vector<size_t>({1, 4, 2}), "AV output geometry mismatch");
    for (size_t i = 0; i < packed_result.size(); ++i)
        require(packed_result.at(i) == native_result.at(i), "AV changed under equivalent strided V layout");
    require(near(packed_result.at(0), 6.2, 1e-7) && near(packed_result.at(1), 7.2, 1e-7),
        "AV visible-extent weighted reduction mismatch: " + std::to_string(packed_result.at(0)) + "," +
            std::to_string(packed_result.at(1)));
    const auto short_result = attention_av(p, contiguous_view(canonical, {3, 2, 2}), {2},
        Accumulation::F64, ScalarType::F64);
    require(near(short_result.at(0), 1.7, 1e-7) && near(short_result.at(1), 2.2, 1e-7),
        "AV consumed values beyond the explicit visible extent");
}

void test_rms_norm_strides_epsilon_and_errors() {
    const double input_storage[] = {1, 2, 3, 99, 4, 5, 6, 99};
    const double scale_storage[] = {2.0, 1.0, 0.5};
    const TensorView input{input_storage, ScalarType::F64, {2, 3},
        {4 * static_cast<ptrdiff_t>(sizeof(double)), static_cast<ptrdiff_t>(sizeof(double))}};
    const TensorView reversed_scale{scale_storage + 2, ScalarType::F64, {3},
        {-static_cast<ptrdiff_t>(sizeof(double))}};
    const auto output = rms_norm(input, reversed_scale, 1e-6, ScalarType::F64);
    const double inv0 = 1.0 / std::sqrt((1.0 + 4.0 + 9.0) / 3.0 + 1e-6);
    require(near(output.at(0), inv0 * 0.5) && near(output.at(1), inv0 * 2.0) &&
        near(output.at(2), inv0 * 6.0), "strided RMSNorm/scale/epsilon semantics mismatch");
    must_throw([&] { (void) rms_norm(input, reversed_scale, 0.0); },
        "undefined zero-epsilon RMSNorm was accepted");
    const double nan_input[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const double scale[] = {1.0, 1.0};
    must_throw([&] { (void) rms_norm(contiguous_view(nan_input, {1, 2}),
        contiguous_view(scale, {2}), 1e-6); }, "non-finite RMSNorm input was accepted");
}

void test_q4k_decode_and_reference_evaluator() {
    constexpr size_t columns = 256;
    std::vector<float> source(columns);
    for (size_t i = 0; i < columns; ++i)
        source[i] = std::sin(static_cast<float>(i) * 0.071f) * 2.0f;
    std::vector<uint8_t> encoded(ggml_row_size(GGML_TYPE_Q4_K, columns));
    const size_t written = ggml_quantize_chunk(GGML_TYPE_Q4_K, source.data(), encoded.data(),
        0, 1, columns, nullptr);
    require(written == encoded.size(), "upstream Q4_K quantizer returned an unexpected payload size");
    const auto * traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    require(traits != nullptr && traits->to_float != nullptr, "upstream Q4_K decoder trait unavailable");
    std::vector<float> upstream(columns);
    traits->to_float(encoded.data(), upstream.data(), columns);
    const auto decoded = decode_ggml_rows(GGML_TYPE_Q4_K, encoded.data(), encoded.size(), 1, columns);
    require(decoded.shape == std::vector<size_t>({1, columns}), "Q4_K decoded shape mismatch");
    for (size_t i = 0; i < columns; ++i)
        require(decoded.f32[i] == upstream[i], "Q4_K decoder differs from the pinned GGML type trait");
    Tensor unquantized_source{ScalarType::F32, {1, columns}, source, {}};
    EvaluationContext reconstruction_context;
    reconstruction_context.candidate_identity = "synthetic-only:q4k-reconstruction-test";
    reconstruction_context.qualification_run_identity = "synthetic-q4k-reconstruction-run:v1";
    reconstruction_context.model_identity = "synthetic:q4k-known-source-vector";
    reconstruction_context.backend_family = "GGML_CPU";
    reconstruction_context.implementation_identity = "ggml-q4k-trait-dequantizer";
    reconstruction_context.device_family = "CPU";
    reconstruction_context.placement_identity = "single:synthetic-cpu";
    reconstruction_context.fixture_identity = "deterministic-q4k-source-vector-v1";
    reconstruction_context.input_identity = "source=f32-sine-vector;candidate=ggml-q4_k";
    reconstruction_context.input_dtype = "quantized-weight-format";
    reconstruction_context.capacity = 1;
    reconstruction_context.rows = 1;
    reconstruction_context.logical_inputs_equivalent = true;
    reconstruction_context.invariant_values["verified_source_artifact"] = true;
    ReferenceProvenance reconstruction_provenance;
    reconstruction_provenance.accumulation_precision = "not_applicable_elementwise_representation_reconstruction";
    reconstruction_provenance.input_representation = "verified F32 source vector versus GGML Q4_K row";
    reconstruction_provenance.output_representation = "F32 source and F32 decoded reconstruction";
    reconstruction_provenance.operation_parameters = {{"ggml_type", "Q4_K"},
        {"source_identity", "deterministic-q4k-source-vector-v1"}, {"elements", "256"}};
    const auto reconstruction = evaluate_reference_comparison(
        "vbuf.quantization.weight_reconstruction_accuracy", 1, unquantized_source, decoded,
        reconstruction_context, reconstruction_provenance);
    require(reconstruction.status == EvaluationStatus::NotTested &&
        reconstruction.contract_status == ContractStatus::NeedsCalibration &&
        reconstruction.measured_metrics.values.count(Metric::RelativeRmsError) == 1,
        "Q4_K reconstruction metrics did not remain diagnostic under the uncalibrated contract");
    const auto copied_f32 = decode_ggml_rows(GGML_TYPE_F32, source.data(),
        source.size() * sizeof(float), 1, columns);
    require(copied_f32.f32 == source, "F32 GGML representation decoder changed values");
    std::vector<uint8_t> q8k_encoded(ggml_row_size(GGML_TYPE_Q8_K, columns));
    quantize_row_q8_K_ref(source.data(), reinterpret_cast<block_q8_K *>(q8k_encoded.data()), columns);
    const auto decoded_q8k = decode_ggml_rows(GGML_TYPE_Q8_K, q8k_encoded.data(),
        q8k_encoded.size(), 1, columns);
    double q8k_max_error = 0.0;
    for (size_t i = 0; i < columns; ++i)
        q8k_max_error = std::max(q8k_max_error, std::abs(static_cast<double>(source[i]) - decoded_q8k.at(i)));
    require(q8k_max_error < 0.02, "pinned Q8_K fallback dequantizer exceeded this synthetic block's bound");

    std::vector<float> activation(columns, 0.125f);
    const TensorView transposed_weight{decoded.f32.data(), ScalarType::F32, {columns, 1},
        {static_cast<ptrdiff_t>(sizeof(float)), 0}};
    const auto product = matmul(contiguous_view(activation.data(), {1, columns}), transposed_weight,
        Accumulation::F64, ScalarType::F64);
    double independent_sum = 0.0;
    for (size_t i = 0; i < columns; ++i)
        independent_sum += static_cast<double>(activation[i]) * static_cast<double>(upstream[i]);
    require(product.shape == std::vector<size_t>({1, 1}) && product.at(0) == independent_sum,
        "Q4_K-dequantized matmul did not consume the exact represented weights");
    must_throw([&] { (void) decode_ggml_rows(GGML_TYPE_Q4_K, encoded.data(), encoded.size() - 1, 1, columns); },
        "truncated Q4_K payload was accepted");

    constexpr size_t kv_heads = 8, q_heads = 40, dimension = 128, positions = 37;
    std::vector<float> values(positions * kv_heads * dimension);
    std::vector<float> weights(q_heads * positions);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(static_cast<float>(i) * 0.0031f) * 0.4f;
    for (size_t h = 0; h < q_heads; ++h)
        for (size_t p = 0; p < positions; ++p) weights[h * positions + p] = 1.0f / positions;
    const auto oracle_result = attention_av(contiguous_view(weights.data(), {1, q_heads, positions}),
        contiguous_view(values.data(), {positions, kv_heads, dimension}), {positions},
        Accumulation::F64, ScalarType::F32);
    const auto candidate_result = attention_av(contiguous_view(weights.data(), {1, q_heads, positions}),
        contiguous_view(values.data(), {positions, kv_heads, dimension}), {positions},
        Accumulation::F32, ScalarType::F32);
    Tensor oracle{ScalarType::F32, {dimension, 1, q_heads}, {}, {}};
    Tensor candidate{ScalarType::F32, {dimension, 1, q_heads}, {}, {}};
    oracle.f32.resize(dimension * q_heads);
    candidate.f32.resize(dimension * q_heads);
    for (size_t head = 0; head < q_heads; ++head) for (size_t d = 0; d < dimension; ++d) {
        oracle.f32[d + dimension * head] = oracle_result.f32[head * dimension + d];
        candidate.f32[d + dimension * head] = candidate_result.f32[head * dimension + d];
    }
    EvaluationContext context;
    context.output_name = "av_output";
    context.candidate_identity = "synthetic-only:native-av-reference-test";
    context.qualification_run_identity = "synthetic-reference-contract-run:v1";
    context.model_identity = "synthetic:qwen3-gqa-40q-8kv-d128";
    context.backend_family = "GGML_CPU";
    context.implementation_identity = "ggml-native-layout-av";
    context.device_family = "CPU";
    context.placement_identity = "single:synthetic-cpu";
    context.phase = "decode";
    context.execution_topology = "synthetic-direct-operation-fixture";
    context.fixture_identity = "q4k-decoder-and-qwen-gqa-av-reference-smoke";
    context.input_identity = "deterministic-q4k-decoder-av-fixture-v1";
    context.input_dtype = "F32";
    context.output_shape = {dimension, 1, q_heads};
    context.capacity = positions;
    context.context_length = positions;
    context.rows = 1;
    context.logical_inputs_equivalent = true;
    ReferenceProvenance provenance;
    provenance.accumulation_precision = "binary64-product-and-left-to-right-sum";
    provenance.input_representation = "probabilities=F32;V=F32;GQA=40:8";
    provenance.output_representation = "F32-rounded-once";
    provenance.operation_parameters = {{"visible_positions", "37"}, {"head_dimension", "128"},
        {"kv_head_mapping", "floor(query_head/5)"}};
    const auto evaluation = evaluate_reference_comparison(
        "qwen3.attention_av.native_fp64_synthetic_accuracy", 1,
        oracle, candidate, context, provenance);
    require(evaluation.status == EvaluationStatus::Pass &&
        !evaluation.context.reference_implementation_revision.empty(),
        "independent FP64 AV evaluation did not pass the existing scoped synthetic contract");
    require(evaluation.context.reference_accumulation_precision == provenance.accumulation_precision &&
        evaluation.context.reference_operation_parameters.at("visible_positions") == "37",
        "reference provenance was not retained in live evaluation context");
    const std::string json = evaluation_json(evaluation);
    require(json.find("reference_implementation_revision") != std::string::npos &&
        json.find("binary64-product-and-left-to-right-sum") != std::string::npos &&
        json.find("kv_head_mapping") != std::string::npos &&
        json.find("synthetic-reference-contract-run:v1") != std::string::npos,
        "reference qualification JSON omitted algorithm/run provenance");
    auto mutated = evaluation;
    mutated.context.reference_input_representation = "forged-input-provenance";
    require(numerical_evidence_rejection_reason(mutated, RuntimeQualificationContext{}) ==
        EvidenceRejectionReason::InvalidEvidence,
        "reference provenance mutation did not invalidate live evaluation integrity");
    const auto replayed = mark_numerical_evaluation_replay_only(evaluation);
    require(replayed.replayed_metrics_only &&
        numerical_evidence_rejection_reason(replayed, RuntimeQualificationContext{}) == EvidenceRejectionReason::ReplayOnly,
        "captured/replayed tensor evaluation retained admission authority");

    provenance.input_representation = "GGML Q4_K decoded to F32; activation=F32";
    context.operation = "general_matmul";
    context.implementation_identity = "ggml-cpu-matmul";
    context.output_shape = {1, 1};
    const auto diagnostic = evaluate_reference_comparison(
        "vbuf.general_matmul.fp64.operation_accuracy", 1, product, product, context, provenance);
    require(diagnostic.status == EvaluationStatus::NotTested &&
        diagnostic.contract_status == ContractStatus::NeedsCalibration,
        "matmul reference unexpectedly acquired authorizing status without calibration");
}

} // namespace

int main() {
    try {
        test_strided_matmul_and_precision();
        test_qk_gqa_and_layout();
        test_masked_stable_softmax();
        test_gqa_av_layouts_and_visible_extent();
        test_rms_norm_strides_epsilon_and_errors();
        test_q4k_decode_and_reference_evaluator();
        std::cout << "vbuf_high_precision_reference_contract=PASS revision="
                  << implementation_revision() << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_high_precision_reference_contract=FAIL: " << error.what() << '\n';
        return 1;
    }
}
