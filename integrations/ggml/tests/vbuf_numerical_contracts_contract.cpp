#include "vbuf_numerical_contracts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ml::numerics;
namespace {
constexpr const char * model_sha = "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
constexpr const char * placement = "multi:0x26,1x14;emb=0;norm=1;head=1";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

bool near(double a, double b, double tolerance = 1e-12) {
    return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

EvaluationContext qwen_context(const std::string & operation, const std::string & implementation = "native-layout-av-v1") {
    EvaluationContext context;
    context.operation = operation;
    context.output_name = operation;
    context.reference_kind = ReferenceKind::CanonicalExecution;
    context.reference_identity = "canonical-packed-v-v1:fixture-identity-001";
    context.model_identity = model_sha;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = implementation;
    context.device_family = "CUDA";
    context.device_identities = {"CUDA0@0000:04:00.0", "CUDA1@0000:07:00.0"};
    context.device_sm_versions = {86, 75};
    context.placement_identity = placement;
    context.phase = "decode";
    context.execution_topology = "tokenwise-incremental";
    context.fixture_identity = "unit-fixture";
    context.input_identity = "token-history:001";
    context.output_dtype = "F32";
    context.input_dtype = "F16";
    context.capacity = 512;
    context.context_length = 33;
    context.rows = 1;
    context.logical_inputs_equivalent = true;
    context.reference_tokens = std::vector<uint32_t>{11, 12};
    context.candidate_tokens = std::vector<uint32_t>{11, 12};
    return context;
}

NumericalEvaluation evaluate_qwen_output(const std::string & operation,
        const std::vector<float> & reference_values, const std::vector<float> & candidate_values,
        EvaluationContext context) {
    context.output_shape = {reference_values.size()};
    const auto reference = make_tensor_view(reference_values, context.output_shape);
    const auto candidate = make_tensor_view(candidate_values, context.output_shape);
    const std::string contract_id = operation == "final_hidden" ?
        "qwen3.final_hidden.canonical_compatibility" : "qwen3.final_logits.canonical_compatibility";
    return evaluate_contract(contract_id, 1, &reference, &candidate, context);
}

void test_registry_and_policy_versions() {
    require(project_policy_id() == std::string("vbuf.numerical-contracts"), "wrong numerical policy identity");
    require(project_policy_version() == 2 && std::string(project_policy_digest()).size() == 64,
        "policy version/digest is missing");
    require(project_contracts().size() == 16, "generated policy contract version history changed unexpectedly");
    for (const auto & contract : project_contracts()) {
        require(find_contract(contract.contract_id, contract.version) != nullptr,
            "generated contract was not registered: " + contract.contract_id);
        if (contract.status == ContractStatus::NeedsCalibration)
            require(contract.criteria.empty(), "NEEDS_CALIBRATION contract unexpectedly has authorizing criteria");
    }
    const auto * historical_logits = find_contract("qwen3.final_logits.canonical_compatibility", 1);
    const auto * active_logits = find_contract("qwen3.final_logits.canonical_compatibility", 2);
    require(historical_logits != nullptr && active_logits != nullptr &&
        historical_logits->scope.shapes == std::vector<std::string>{"*"} &&
        active_logits->scope.shapes == std::vector<std::string>{"151936"},
        "final logits contract history/current shape scope is not explicit");
    require(find_contract("qwen3.final_logits.canonical_compatibility", 3) == nullptr,
        "unknown contract version resolved to the active contract");
}

void test_versioned_output_shape_scopes() {
    auto context = qwen_context("final_logits");
    std::vector<float> reference_values(151936, 1.0f);
    std::vector<float> candidate_values = reference_values;
    candidate_values.back() += 0.001f;
    context.output_shape = {151936};
    const auto reference = make_tensor_view(reference_values, context.output_shape);
    const auto candidate = make_tensor_view(candidate_values, context.output_shape);
    auto result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 2,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::Pass,
        "v2 final-logits contract rejected its exact declared output shape");

    const std::vector<float> short_reference{1.0f, 2.0f};
    const std::vector<float> short_candidate{1.0f, 2.0f};
    context.output_shape = {2};
    auto short_ref = make_tensor_view(short_reference, context.output_shape);
    auto short_cand = make_tensor_view(short_candidate, context.output_shape);
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 2,
        &short_ref, &short_cand, context);
    require(result.status == EvaluationStatus::NotApplicable,
        "v2 final-logits contract accepted a different output shape");
}

void test_units_and_criterion_boundaries() {
    require(normalize_tolerance(0.001, ToleranceUnit::Fraction) == 0.001,
        "fraction tolerance changed value");
    require(near(normalize_tolerance(0.1, ToleranceUnit::Percent), 0.001),
        "percentage normalization is incorrect");
    require(near(normalize_tolerance(1.0, ToleranceUnit::Permille), 0.001),
        "permille normalization is incorrect");
    require(normalize_tolerance(0.001, ToleranceUnit::Absolute) == 0.001,
        "absolute tolerance changed value");

    const MetricCriterion inclusive{Metric::RelativeRmsError, Comparison::LessEqual, 0.02,
        ToleranceUnit::Fraction, true};
    auto at_boundary = evaluate_metric_criterion(inclusive, {0.02, std::nullopt});
    require(at_boundary.available && at_boundary.passed && at_boundary.normalized_limit == 0.02,
        "inclusive threshold did not pass equality at the boundary");
    require(!evaluate_metric_criterion(inclusive, {std::nextafter(0.02, 1.0), std::nullopt}).passed,
        "inclusive threshold passed an above-boundary value");
    const MetricCriterion strict{Metric::MaxAbsoluteError, Comparison::LessThan, 1.0,
        ToleranceUnit::Absolute, true};
    require(!evaluate_metric_criterion(strict, {1.0, std::nullopt}).passed,
        "strict maximum threshold passed equality");
    const MetricCriterion minimum{Metric::CosineSimilarity, Comparison::GreaterEqual, 0.9998,
        ToleranceUnit::Fraction, true};
    require(evaluate_metric_criterion(minimum, {0.9998, std::nullopt}).passed,
        "inclusive minimum threshold failed equality");
    bool rejected_negative = false;
    try { (void) normalize_tolerance(-0.1, ToleranceUnit::Percent); }
    catch (const std::invalid_argument &) { rejected_negative = true; }
    require(rejected_negative, "negative tolerance was accepted");
}

void test_metric_definitions_and_edge_values() {
    EvaluationContext context;
    const std::vector<double> reference_values{1.0, 2.0, 3.0};
    const std::vector<double> candidate_values{2.0, 4.0, 6.0};
    const auto reference = make_tensor_view(reference_values, {3});
    const auto candidate = make_tensor_view(candidate_values, {3});
    const NumericalMetrics metrics = measure_tensor_pair(reference, candidate, context);
    require(near(*metrics.values.at(Metric::MaxAbsoluteError).numeric_value, 3.0), "max absolute error definition mismatch");
    require(near(*metrics.values.at(Metric::MeanAbsoluteError).numeric_value, 2.0), "mean absolute error definition mismatch");
    require(near(*metrics.values.at(Metric::RmsError).numeric_value, std::sqrt(14.0 / 3.0)), "RMS definition mismatch");
    require(near(*metrics.values.at(Metric::RelativeRmsError).numeric_value, 1.0), "relative RMS definition mismatch");
    require(near(*metrics.values.at(Metric::CosineSimilarity).numeric_value, 1.0), "cosine definition mismatch");
    require(!*metrics.values.at(Metric::BitwiseEquality).boolean_value, "different tensors reported bitwise equal");
    require(*metrics.values.at(Metric::Top1Equality).boolean_value, "top-1 equality was not measured");

    const std::vector<double> zero_reference{0.0, 0.0};
    const std::vector<double> zero_candidate{0.0, 0.0};
    const auto zero = measure_tensor_pair(make_tensor_view(zero_reference, {2}),
        make_tensor_view(zero_candidate, {2}), context);
    require(*zero.values.at(Metric::MaxAbsoluteError).numeric_value == 0.0 &&
        !zero.values.at(Metric::RelativeRmsError).numeric_value &&
        !zero.values.at(Metric::CosineSimilarity).numeric_value,
        "zero-norm reference behavior must be explicit and undefined for relative/cosine metrics");

    const std::vector<double> near_zero_reference{1e-300};
    const std::vector<double> near_zero_candidate{2e-300};
    const auto near_zero = measure_tensor_pair(make_tensor_view(near_zero_reference, {1}),
        make_tensor_view(near_zero_candidate, {1}), context);
    require(near(*near_zero.values.at(Metric::RelativeRmsError).numeric_value, 1.0),
        "near-zero reference norm was hidden by an implicit floor");

    const std::vector<double> signed_zero_a{0.0};
    const std::vector<double> signed_zero_b{-0.0};
    const auto signed_zero = measure_tensor_pair(make_tensor_view(signed_zero_a, {1}),
        make_tensor_view(signed_zero_b, {1}), context);
    require(*signed_zero.values.at(Metric::MaxAbsoluteError).numeric_value == 0.0 &&
        !*signed_zero.values.at(Metric::BitwiseEquality).boolean_value,
        "signed-zero numerical and bitwise semantics were conflated");

    const std::vector<double> subnormal{std::numeric_limits<double>::denorm_min()};
    const auto subnormal_result = measure_tensor_pair(make_tensor_view(subnormal, {1}),
        make_tensor_view(subnormal, {1}), context);
    require(*subnormal_result.values.at(Metric::BitwiseEquality).boolean_value &&
        *subnormal_result.values.at(Metric::RelativeRmsError).numeric_value == 0.0 &&
        *subnormal_result.values.at(Metric::CosineSimilarity).numeric_value == 1.0,
        "non-zero subnormal reference was treated as a zero norm");
}

void test_fail_closed_statuses() {
    const std::vector<float> reference_values{1.0f, 0.0f};
    const std::vector<float> passing_values{1.0f, 0.001f};
    auto context = qwen_context("final_logits");
    auto reference = make_tensor_view(reference_values, {2});
    auto candidate = make_tensor_view(passing_values, {2});
    auto result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::Pass, "in-scope acceptable Qwen output did not pass");
    require(result.criteria.size() == 2 && result.criteria[0].required && result.criteria[1].required,
        "multi-metric criteria were not retained individually");

    context.capacity = 64;
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::NotApplicable, "unsupported capacity did not report NOT_APPLICABLE");
    context.capacity = 512;
    context.logical_inputs_equivalent = false;
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::InvalidEvaluation,
        "non-equivalent inputs were not rejected as an invalid comparison");
    context.logical_inputs_equivalent = true;

    result = evaluate_contract("does.not.exist", 1, &reference, &candidate, context);
    require(result.status == EvaluationStatus::InvalidEvaluation, "unknown contract did not fail closed");
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 9,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::InvalidEvaluation, "unknown contract version did not fail closed");
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        nullptr, &candidate, context);
    require(result.status == EvaluationStatus::NotTested, "missing reference did not return NOT_TESTED");

    const std::vector<float> wrong_shape{1.0f};
    auto wrong = make_tensor_view(wrong_shape, {1});
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &wrong, context);
    require(result.status == EvaluationStatus::InvalidEvaluation, "shape mismatch did not fail closed");
    const std::vector<float> empty;
    auto empty_view = make_tensor_view(empty, {0});
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &empty_view, &empty_view, context);
    require(result.status == EvaluationStatus::InvalidEvaluation, "empty tensor was accepted");

    const std::vector<float> nonfinite{std::numeric_limits<float>::quiet_NaN(), 0.0f};
    auto nonfinite_view = make_tensor_view(nonfinite, {2});
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &nonfinite_view, context);
    require(result.status == EvaluationStatus::Fail &&
        !*result.measured_metrics.values.at(Metric::FiniteOutputs).boolean_value,
        "NaN did not fail the finite invariant");
    const std::vector<float> infinite{std::numeric_limits<float>::infinity(), 0.0f};
    auto infinite_view = make_tensor_view(infinite, {2});
    result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &infinite_view, context);
    require(result.status == EvaluationStatus::Fail, "Inf did not fail the finite invariant");

    EvaluationContext qk_context;
    qk_context.operation = "qk_matmul";
    qk_context.reference_kind = ReferenceKind::Fp64OperationOracle;
    qk_context.reference_identity = "declared-per-fixture-fp64-qk-oracle";
    qk_context.model_identity = "model-under-test";
    qk_context.backend_family = "GGML_CUDA";
    qk_context.implementation_identity = "mmvf";
    qk_context.device_family = "CUDA";
    qk_context.output_dtype = "F32";
    qk_context.input_dtype = "F32";
    qk_context.capacity = 1024;
    qk_context.context_length = 32;
    qk_context.rows = 32;
    qk_context.logical_inputs_equivalent = true;
    const std::vector<float> qk_reference{1.0f, 2.0f};
    const std::vector<float> qk_candidate{1.0001f, 2.0f};
    auto qr = make_tensor_view(qk_reference, {2});
    auto qc = make_tensor_view(qk_candidate, {2});
    result = evaluate_contract("vbuf.qk_matmul.fp64.operation_accuracy", 1, &qr, &qc, qk_context);
    require(result.status == EvaluationStatus::NotTested &&
        result.measured_metrics.values.count(Metric::RelativeRmsError) == 1,
        "NEEDS_CALIBRATION did not retain diagnostics without authorizing a pass");
}

void test_lifecycle_invariants_are_hard_and_unqualified() {
    EvaluationContext context;
    context.operation = "session_lifecycle";
    context.output_name = "session_state";
    context.reference_kind = ReferenceKind::InvariantReference;
    context.reference_identity = "vbuf-session-lifecycle-invariants-v1";
    context.model_identity = "test-model";
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = "test-session-runtime";
    context.device_family = "CUDA";
    context.placement_identity = "test-placement";
    context.capacity = 512;
    context.context_length = 32;
    context.rows = 1;
    context.invariant_values["session_reset_verified"] = true;
    context.invariant_values["logical_progress_atomic"] = true;
    context.invariant_values["failure_recovery_canonical"] = true;
    auto result = evaluate_invariants("vbuf.lifecycle.session_atomicity", 1, context);
    require(result.status == EvaluationStatus::NotTested &&
        result.contract_status == ContractStatus::NeedsCalibration,
        "unregistered lifecycle evidence was treated as an authorizing pass");
    context.invariant_values["failure_recovery_canonical"] = false;
    result = evaluate_invariants("vbuf.lifecycle.session_atomicity", 1, context);
    require(result.status == EvaluationStatus::Fail,
        "lifecycle failure recovery was not enforced as a hard invariant");
}

void test_recorded_metrics_cannot_authorize_runtime() {
    auto context = qwen_context("final_logits");
    context.output_shape = {2};
    context.candidate_identity = "candidate-A";
    context.qualification_run_identity = "unit-run-A";
    context.invariant_values["same_shape"] = true;
    context.invariant_values["finite_outputs"] = true;
    NumericalMetrics recorded;
    recorded.values[Metric::MaxAbsoluteError].numeric_value = 0.01;
    recorded.values[Metric::RelativeRmsError].numeric_value = 0.01;
    recorded.values[Metric::CosineSimilarity].numeric_value = 0.9999;
    recorded.values[Metric::FiniteOutputs].boolean_value = true;
    recorded.values[Metric::BitwiseEquality].boolean_value = false;
    recorded.values[Metric::Top1Equality].boolean_value = true;
    recorded.values[Metric::TokenSequenceEquality].boolean_value = true;
    auto evaluation = evaluate_recorded_metrics("qwen3.final_logits.canonical_compatibility", 1,
        context, recorded);
    require(evaluation.status == EvaluationStatus::Pass && evaluation.replayed_metrics_only,
        "in-scope historical metrics were not evaluated as a replay-only pass");
    RuntimeQualificationContext runtime;
    runtime.model_identity = context.model_identity;
    runtime.backend_family = context.backend_family;
    runtime.implementation_identity = context.implementation_identity;
    runtime.device_family = context.device_family;
    runtime.placement_identity = context.placement_identity;
    runtime.activation_dtype = context.output_dtype;
    runtime.kv_dtype = context.input_dtype;
    runtime.phase = context.phase;
    runtime.capacity = context.capacity;
    runtime.context_length = context.context_length;
    runtime.rows = context.rows;
    runtime.device_identities = context.device_identities;
    runtime.device_sm_versions = context.device_sm_versions;
    runtime.execution_topology = context.execution_topology;
    runtime.candidate_identity = context.candidate_identity;
    runtime.fixture_identity = context.fixture_identity;
    runtime.input_identity = context.input_identity;
    runtime.token_sequence_identity = context.token_sequence_identity;
    require(!numerical_evidence_matches_runtime(evaluation, runtime),
        "replayed metric-only evidence authorized an execution candidate");

    auto active = evaluate_qwen_output("final_logits", {1.0f, 0.0f}, {1.0f, 0.001f}, context);
    require(active.status == EvaluationStatus::Pass && numerical_evidence_matches_runtime(active, runtime),
        "live scoped active evidence did not match its runtime context");
    auto wrong_reference = active;
    wrong_reference.reference_identity = "unapproved-reference";
    require(!numerical_evidence_matches_runtime(wrong_reference, runtime),
        "active output evidence with a forged reference identity authorized a candidate");
    NumericalEvaluation default_constructed;
    default_constructed.contract_id = active.contract_id;
    default_constructed.contract_version = active.contract_version;
    default_constructed.policy_id = active.policy_id;
    default_constructed.policy_version = active.policy_version;
    default_constructed.policy_digest = active.policy_digest;
    default_constructed.status = EvaluationStatus::Pass;
    default_constructed.context = active.context;
    default_constructed.reference_shape = active.reference_shape;
    default_constructed.candidate_shape = active.candidate_shape;
    default_constructed.candidate_dtype = active.candidate_dtype;
    default_constructed.measured_metrics = active.measured_metrics;
    default_constructed.criteria = active.criteria;
    require(!numerical_evidence_matches_runtime(default_constructed, runtime),
        "caller-constructed PASS object acquired in-process qualification authority");
    auto candidate_b_runtime = runtime;
    candidate_b_runtime.candidate_identity = "candidate-B";
    require(!numerical_evidence_matches_runtime(active, candidate_b_runtime),
        "evidence for candidate A authorized candidate B with the same implementation family");
    auto wrong_implementation_runtime = runtime;
    wrong_implementation_runtime.implementation_identity = "another-candidate-build";
    require(numerical_evidence_rejection_reason(active, wrong_implementation_runtime) ==
            EvidenceRejectionReason::ImplementationMismatch,
        "implementation identity mismatch was not distinguished at the admission boundary");
    const std::vector<double> double_reference_values{1.0, 0.0};
    const std::vector<float> float_candidate_values{1.0f, 0.001f};
    const auto double_reference = make_tensor_view(double_reference_values, {2});
    const auto float_candidate = make_tensor_view(float_candidate_values, {2});
    const auto wrong_reference_dtype = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &double_reference, &float_candidate, context);
    require(wrong_reference_dtype.status == EvaluationStatus::Pass &&
        numerical_evidence_rejection_reason(wrong_reference_dtype, runtime) == EvidenceRejectionReason::DTypeMismatch,
        "canonical compatibility evidence with a non-runtime reference dtype was accepted");
    auto tampered_metrics = active;
    tampered_metrics.criteria.clear();
    require(!numerical_evidence_matches_runtime(tampered_metrics, runtime),
        "mutated or incomplete active evidence retained its qualification authority");
}

void test_accuracy_and_compatibility_are_independent() {
    std::vector<float> fp64_oracle(128 * 40, 0.0f);
    std::vector<float> native_av(128 * 40, 0.0f);
    fp64_oracle[0] = 0.75f;
    fp64_oracle[1] = -0.25f;
    fp64_oracle[2] = 1.0f;
    native_av[0] = 0.7500001f;
    native_av[1] = -0.25f;
    native_av[2] = 1.0f;
    EvaluationContext av_context;
    av_context.operation = "attention_av";
    av_context.output_name = "av_output";
    av_context.reference_kind = ReferenceKind::Fp64OperationOracle;
    av_context.reference_identity = "fp64-qwen3-gqa-av-oracle-v1";
    av_context.model_identity = "synthetic:qwen3-gqa-40q-8kv-d128";
    av_context.backend_family = "GGML_CUDA";
    av_context.implementation_identity = "ggml-native-layout-av";
    av_context.device_family = "CUDA";
    av_context.device_sm_versions = {86};
    av_context.placement_identity = "single:0x40;emb=0;norm=0;head=0";
    av_context.phase = "prefill";
    av_context.execution_topology = "32-row-prefill";
    av_context.fixture_identity = "synthetic-gqa-fixture";
    av_context.input_identity = "captured-p-v-and-positions-sha256:synthetic";
    av_context.output_dtype = "F32";
    av_context.input_dtype = "F16";
    av_context.output_shape = {128, 1, 40};
    av_context.capacity = 512;
    av_context.context_length = 32;
    av_context.rows = 1;
    av_context.logical_inputs_equivalent = true;
    auto oracle = make_tensor_view(fp64_oracle, av_context.output_shape);
    auto native = make_tensor_view(native_av, av_context.output_shape);
    const auto mathematical = evaluate_contract("qwen3.attention_av.fp64_operation_accuracy", 1,
        &oracle, &native, av_context);
    require(mathematical.status == EvaluationStatus::Pass,
        "native AV did not independently pass the scoped FP64 operation contract");

    const std::vector<float> canonical_logits{1.0f, 0.0f};
    const std::vector<float> candidate_logits{1.0f, 0.03f};
    auto compatibility_context = qwen_context("final_logits");
    compatibility_context.reference_tokens = std::vector<uint32_t>{11};
    compatibility_context.candidate_tokens = std::vector<uint32_t>{11};
    auto canonical = make_tensor_view(canonical_logits, {2});
    auto candidate = make_tensor_view(candidate_logits, {2});
    const auto compatibility = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &canonical, &candidate, compatibility_context);
    require(compatibility.status == EvaluationStatus::Fail &&
        *compatibility.measured_metrics.values.at(Metric::Top1Equality).boolean_value &&
        *compatibility.measured_metrics.values.at(Metric::TokenSequenceEquality).boolean_value,
        "model-level compatibility failure was overridden by top-1/token agreement");
    require(mathematical.status == EvaluationStatus::Pass && compatibility.status == EvaluationStatus::Fail,
        "operation mathematical accuracy and model-level compatibility were conflated");
}

void test_model_boundary_observation_scope() {
    std::vector<float> reference_values(128 * 32 * 40, 0.0f);
    std::vector<float> candidate_values(reference_values.size(), 0.0f);
    candidate_values[0] = 1e-7f;
    EvaluationContext context;
    context.operation = "attention_av";
    context.output_name = "native_av_output";
    context.reference_kind = ReferenceKind::Fp64OperationOracle;
    context.reference_identity = "qwen3-av-boundary-fp64-oracle-v1";
    context.model_identity = model_sha;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = "ggml-native-layout-av";
    context.device_family = "CUDA";
    context.device_identities = {"CUDA0@0000:04:00.0"};
    context.device_sm_versions = {86};
    context.placement_identity = placement;
    context.phase = "prefill";
    context.execution_topology = "native-av-side-branch";
    context.fixture_identity = "qwen3-native-av-prefill32-cap512-layer0";
    context.input_identity = "captured-input-v1";
    context.token_sequence_identity = "fnv64:0123456789abcdef";
    context.output_dtype = "F32";
    context.input_dtype = "F16";
    context.output_shape = {128, 32, 40};
    context.capacity = 512;
    context.context_length = 32;
    context.rows = 32;
    context.logical_inputs_equivalent = true;
    auto reference = make_tensor_view(reference_values, context.output_shape);
    auto candidate = make_tensor_view(candidate_values, context.output_shape);
    auto result = evaluate_contract("qwen3.attention_av.fp64_model_boundary_observation", 1,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::NotTested && !result.replayed_metrics_only &&
        result.contract_status == ContractStatus::NeedsCalibration &&
        result.measured_metrics.values.count(Metric::MaxAbsoluteError) == 1,
        "scoped actual-model AV metrics were not retained as non-authorizing measurements");

    context.fixture_identity = "another-prompt-or-layer";
    result = evaluate_contract("qwen3.attention_av.fp64_model_boundary_observation", 1,
        &reference, &candidate, context);
    require(result.status == EvaluationStatus::NotApplicable,
        "actual-model AV observation escaped its exact fixture scope");
}

void test_json_record_is_self_describing() {
    const std::vector<float> reference_values{1.0f, 0.0f};
    const std::vector<float> candidate_values{1.0f, 0.001f};
    auto context = qwen_context("final_logits");
    auto reference = make_tensor_view(reference_values, {2});
    auto candidate = make_tensor_view(candidate_values, {2});
    const auto result = evaluate_contract("qwen3.final_logits.canonical_compatibility", 1,
        &reference, &candidate, context);
    const std::string json = evaluation_json(result);
    for (const std::string & field : {"contract_id", "contract_version", "policy_sha256", "reference_identity",
             "model_identity", "fixture_identity", "input_identity", "token_sequence_identity", "output_dtype", "reference_shape",
             "candidate_shape", "candidate_identity", "qualification_run_identity", "capacity", "context_length",
             "device_sm_versions", "measured_metrics",
             "normalized_limit", "evaluation_status"})
        require(json.find("\"" + field + "\"") != std::string::npos,
            "qualification result JSON omits " + field);
}
} // namespace

int main() {
    try {
        test_registry_and_policy_versions();
        test_versioned_output_shape_scopes();
        test_units_and_criterion_boundaries();
        test_metric_definitions_and_edge_values();
        test_fail_closed_statuses();
        test_lifecycle_invariants_are_hard_and_unqualified();
        test_recorded_metrics_cannot_authorize_runtime();
        test_accuracy_and_compatibility_are_independent();
        test_model_boundary_observation_scope();
        test_json_record_is_self_describing();
        std::cout << "vbuf_numerical_contracts_contract=PASS policy=v2 metrics=9 scope=fail-closed "
            << "accuracy-vs-compatibility=separate tolerance-units=explicit lifecycle=typed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_numerical_contracts_contract=FAIL: " << error.what() << '\n';
        return 1;
    }
}
