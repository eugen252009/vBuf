#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace vbuf_ml::numerics {

enum class ContractCategory : uint8_t {
    OperationAccuracy,
    QuantizationAccuracy,
    NumericalCompatibility,
    ModelOutputAccuracy,
    ExecutionInvariants,
    LifecycleCorrectness,
};

enum class ReferenceKind : uint8_t {
    Fp64OperationOracle,
    Fp32OperationReference,
    CanonicalExecution,
    UnquantizedWeightReference,
    QuantizedWeightReference,
    InvariantReference,
};

enum class ContractStatus : uint8_t { Active, Provisional, NeedsCalibration, Deprecated };
enum class EvaluationStatus : uint8_t { Pass, Fail, NotApplicable, NotTested, InvalidEvaluation };
enum class EvidenceRejectionReason : uint8_t {
    None, MissingEvidence, InvalidEvidence, FailedQualification, ReplayOnly,
    PolicyMismatch, ContractUnknown, ContractVersionMismatch, ContractNotActive,
    CandidateIdentityMismatch, ReferenceMismatch, ModelArtifactMismatch, BackendMismatch, ImplementationMismatch,
    DeviceMismatch, PlacementMismatch, DTypeMismatch, ShapeMismatch, RuntimeFactsMismatch,
    ExecutionScopeMismatch, RequiredMetricMissing, RequiredMetricFailed, EvidenceInvalidated,
};
enum class DataType : uint8_t { F32, F64, Other };
enum class Metric : uint8_t {
    MaxAbsoluteError,
    MeanAbsoluteError,
    RmsError,
    RelativeRmsError,
    CosineSimilarity,
    FiniteOutputs,
    BitwiseEquality,
    Top1Equality,
    TokenSequenceEquality,
};
enum class ToleranceUnit : uint8_t { Absolute, Fraction, Percent, Permille };
enum class Comparison : uint8_t { LessThan, LessEqual, GreaterThan, GreaterEqual, Equal };

struct MetricCriterion {
    Metric metric = Metric::MaxAbsoluteError;
    Comparison comparison = Comparison::LessEqual;
    double value = 0.0;
    ToleranceUnit unit = ToleranceUnit::Absolute;
    bool required = true;
};

struct NumericRange {
    uint32_t minimum = 0;
    uint32_t maximum = UINT32_MAX;
};

struct ContractScope {
    std::string model_identity = "*";
    std::string backend_family = "*";
    std::string implementation_identity = "*";
    std::string device_family = "*";
    std::vector<uint32_t> device_sm_versions;
    bool any_device_sm_versions = true;
    std::string placement_identity = "*";
    std::string output_dtype = "*";
    std::vector<std::string> input_dtypes{"*"};
    std::vector<std::string> shapes{"*"};
    std::vector<std::string> phases{"*"};
    std::vector<std::string> execution_topologies{"*"};
    std::vector<std::string> fixture_identities{"*"};
    std::vector<std::string> input_identities{"*"};
    std::vector<std::string> token_sequence_identities{"*"};
    NumericRange capacity;
    NumericRange context_length;
    NumericRange rows{1, UINT32_MAX};
    bool requires_logical_inputs_equivalent = false;
    std::string empirical_scope;
};

struct NumericalContractRequirement {
    std::string contract_id;
    uint32_t version = 0;
};

struct NumericalContract {
    std::string contract_id;
    uint32_t version = 0;
    ContractCategory category = ContractCategory::OperationAccuracy;
    std::string operation;
    ReferenceKind reference_kind = ReferenceKind::InvariantReference;
    std::string reference_identity;
    std::string reference_semantics;
    std::vector<Metric> measurement_metrics;
    std::vector<MetricCriterion> criteria;
    std::vector<std::string> required_invariants;
    ContractScope scope;
    std::vector<std::string> qualification_tests;
    ContractStatus status = ContractStatus::NeedsCalibration;
    std::string threshold_provenance;
    std::string tolerance_profile;
};

struct NumericTensorView {
    const void * data = nullptr;
    size_t element_count = 0;
    std::vector<uint64_t> shape;
    DataType dtype = DataType::Other;
};

template<typename T>
NumericTensorView make_tensor_view(const std::vector<T> & values, std::vector<uint64_t> shape) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "numerical tensor views support float32 and float64 values");
    const DataType dtype = std::is_same_v<T, float> ? DataType::F32 : DataType::F64;
    return {values.data(), values.size(), std::move(shape), dtype};
}

struct EvaluationContext {
    std::string operation;
    std::string output_name;
    ReferenceKind reference_kind = ReferenceKind::InvariantReference;
    std::string reference_identity;
    std::string reference_implementation_revision;
    std::string reference_accumulation_precision;
    std::string reference_input_representation;
    std::string reference_output_representation;
    std::map<std::string, std::string> reference_operation_parameters;
    std::string candidate_identity;
    std::string qualification_run_identity;
    std::string model_identity;
    std::string backend_family;
    std::string implementation_identity;
    std::string device_family;
    std::vector<std::string> device_identities;
    std::vector<uint32_t> device_sm_versions;
    std::string placement_identity;
    std::string phase;
    std::string execution_topology;
    std::string fixture_identity;
    std::string input_identity;
    std::string token_sequence_identity;
    std::string output_dtype;
    std::string input_dtype;
    std::vector<uint64_t> output_shape;
    uint32_t capacity = 0;
    uint32_t context_length = 0;
    uint32_t rows = 0;
    bool logical_inputs_equivalent = false;
    std::optional<std::vector<uint32_t>> reference_tokens;
    std::optional<std::vector<uint32_t>> candidate_tokens;
    std::map<std::string, bool> invariant_values;
};

struct MetricObservation {
    std::optional<double> numeric_value;
    std::optional<bool> boolean_value;
};

struct NumericalMetrics {
    std::map<Metric, MetricObservation> values;
};

struct CriterionResult {
    Metric metric = Metric::MaxAbsoluteError;
    Comparison comparison = Comparison::LessEqual;
    bool required = true;
    std::optional<double> observed_value;
    std::optional<bool> observed_boolean;
    double declared_limit = 0.0;
    double normalized_limit = 0.0;
    ToleranceUnit declared_unit = ToleranceUnit::Absolute;
    bool passed = false;
    bool available = false;
};

class NumericalEvaluationAuthority;

struct NumericalEvaluation {
    std::string contract_id;
    uint32_t contract_version = 0;
    std::string policy_id;
    uint32_t policy_version = 0;
    std::string policy_digest;
    ContractCategory category = ContractCategory::OperationAccuracy;
    ContractStatus contract_status = ContractStatus::NeedsCalibration;
    EvaluationStatus status = EvaluationStatus::InvalidEvaluation;
    ReferenceKind reference_kind = ReferenceKind::InvariantReference;
    std::string reference_identity;
    EvaluationContext context;
    std::vector<uint64_t> reference_shape;
    std::vector<uint64_t> candidate_shape;
    std::string reference_dtype;
    std::string candidate_dtype;
    NumericalMetrics measured_metrics;
    std::vector<CriterionResult> criteria;
    std::vector<std::string> failure_reasons;
    bool replayed_metrics_only = false;

private:
    std::string authority_snapshot_;
    friend class NumericalEvaluationAuthority;
    friend NumericalEvaluation mark_numerical_evaluation_replay_only(NumericalEvaluation evaluation);
};

struct RuntimeQualificationContext {
    std::string candidate_identity;
    std::string model_identity;
    std::string backend_family;
    std::string implementation_identity;
    std::string device_family;
    std::string placement_identity;
    std::string activation_dtype;
    std::string kv_dtype;
    std::string phase;
    uint32_t capacity = 0;
    uint32_t context_length = 0;
    uint32_t rows = 0;
    std::vector<std::string> device_identities;
    std::vector<uint32_t> device_sm_versions;
    std::string execution_topology;
    std::string fixture_identity;
    std::string input_identity;
    std::string token_sequence_identity;
};

const std::vector<NumericalContract> & project_contracts();
const NumericalContract * find_contract(const std::string & contract_id, uint32_t version);
const char * project_policy_id() noexcept;
uint32_t project_policy_version() noexcept;
const char * project_policy_digest() noexcept;

double normalize_tolerance(double value, ToleranceUnit unit);
CriterionResult evaluate_metric_criterion(const MetricCriterion & criterion,
    const MetricObservation & observation);
std::string shape_identity(const std::vector<uint64_t> & shape);
NumericalMetrics measure_tensor_pair(const NumericTensorView & reference,
    const NumericTensorView & candidate,
    const EvaluationContext & context);
NumericalEvaluation evaluate_contract(const std::string & contract_id, uint32_t version,
    const NumericTensorView * reference, const NumericTensorView * candidate,
    const EvaluationContext & context);
NumericalEvaluation evaluate_invariants(const std::string & contract_id, uint32_t version,
    const EvaluationContext & context);
NumericalEvaluation evaluate_recorded_metrics(const std::string & contract_id, uint32_t version,
    const EvaluationContext & context, const NumericalMetrics & recorded_metrics);
NumericalEvaluation mark_numerical_evaluation_replay_only(NumericalEvaluation evaluation);
EvidenceRejectionReason numerical_evidence_rejection_reason(const NumericalEvaluation & evaluation,
    const RuntimeQualificationContext & runtime_context);
bool numerical_evidence_matches_runtime(const NumericalEvaluation & evaluation,
    const RuntimeQualificationContext & runtime_context);
const char * evidence_rejection_reason_name(EvidenceRejectionReason reason) noexcept;
std::string evaluation_status_name(EvaluationStatus status);
std::string contract_status_name(ContractStatus status);
std::string metric_name(Metric metric);
std::string evaluation_json(const NumericalEvaluation & evaluation);

} // namespace vbuf_ml::numerics
