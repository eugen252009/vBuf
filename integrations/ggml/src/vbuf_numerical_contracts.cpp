#include "vbuf_numerical_contracts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace vbuf_ml::numerics {

class NumericalEvaluationAuthority {
public:
    static void seal(NumericalEvaluation & evaluation) {
        evaluation.authority_snapshot_ = evaluation_json(evaluation);
    }

    static bool authentic(const NumericalEvaluation & evaluation) {
        return !evaluation.authority_snapshot_.empty() &&
            evaluation.authority_snapshot_ == evaluation_json(evaluation);
    }
};

std::vector<NumericalContract> generated_contracts();
const char * generated_policy_id() noexcept;
uint32_t generated_policy_version() noexcept;
const char * generated_policy_digest() noexcept;

namespace {

std::string dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::F32: return "F32";
        case DataType::F64: return "F64";
        default: return "OTHER";
    }
}

bool wildcard_match(const std::string & expected, const std::string & actual) {
    return expected == "*" || expected == actual;
}

bool list_match(const std::vector<std::string> & expected, const std::string & actual) {
    return std::find(expected.begin(), expected.end(), "*") != expected.end() ||
        std::find(expected.begin(), expected.end(), actual) != expected.end();
}

bool range_match(const NumericRange & range, uint32_t value) {
    return value >= range.minimum && value <= range.maximum;
}

bool valid_tensor_geometry(const NumericTensorView & tensor) {
    if (tensor.data == nullptr || tensor.element_count == 0 || tensor.shape.empty() ||
        tensor.dtype == DataType::Other) return false;
    size_t elements = 1;
    for (uint64_t dimension : tensor.shape) {
        if (dimension == 0 || dimension > std::numeric_limits<size_t>::max() / elements) return false;
        elements *= static_cast<size_t>(dimension);
    }
    return elements == tensor.element_count;
}

bool all_finite(const NumericTensorView & tensor) {
    if (tensor.data == nullptr || tensor.element_count == 0) return false;
    if (tensor.dtype == DataType::F32) {
        const auto * values = static_cast<const float *>(tensor.data);
        for (size_t i = 0; i < tensor.element_count; ++i) if (!std::isfinite(values[i])) return false;
        return true;
    }
    if (tensor.dtype == DataType::F64) {
        const auto * values = static_cast<const double *>(tensor.data);
        for (size_t i = 0; i < tensor.element_count; ++i) if (!std::isfinite(values[i])) return false;
        return true;
    }
    return false;
}

double value_at(const NumericTensorView & tensor, size_t index) {
    if (tensor.dtype == DataType::F32) return static_cast<const float *>(tensor.data)[index];
    if (tensor.dtype == DataType::F64) return static_cast<const double *>(tensor.data)[index];
    throw std::invalid_argument("numerical metrics support only F32 and F64 tensors");
}

struct ScaledSumSquares {
    double scale = 0.0;
    double sum_squares = 1.0;

    void add(double value) {
        const double magnitude = std::abs(value);
        if (magnitude == 0.0) return;
        if (scale < magnitude) {
            const double ratio = scale / magnitude;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = magnitude;
        } else {
            const double ratio = magnitude / scale;
            sum_squares += ratio * ratio;
        }
    }

    bool is_zero() const { return scale == 0.0; }
    double normalized_sum_squares() const { return is_zero() ? 0.0 : sum_squares; }
    double norm() const { return is_zero() ? 0.0 : scale * std::sqrt(sum_squares); }
};

std::optional<double> relative_norm(const ScaledSumSquares & numerator, const ScaledSumSquares & denominator) {
    if (denominator.is_zero()) return std::nullopt;
    if (numerator.is_zero()) return 0.0;
    return (numerator.scale / denominator.scale) *
        std::sqrt(numerator.sum_squares / denominator.sum_squares);
}

std::optional<double> cosine_similarity(const NumericTensorView & reference,
        const NumericTensorView & candidate) {
    double reference_scale = 0.0;
    double candidate_scale = 0.0;
    for (size_t i = 0; i < reference.element_count; ++i) {
        reference_scale = std::max(reference_scale, std::abs(value_at(reference, i)));
        candidate_scale = std::max(candidate_scale, std::abs(value_at(candidate, i)));
    }
    if (reference_scale == 0.0 || candidate_scale == 0.0) return std::nullopt;
    double dot = 0.0;
    double reference_square = 0.0;
    double candidate_square = 0.0;
    for (size_t i = 0; i < reference.element_count; ++i) {
        const double left = value_at(reference, i) / reference_scale;
        const double right = value_at(candidate, i) / candidate_scale;
        dot += left * right;
        reference_square += left * left;
        candidate_square += right * right;
    }
    const double result = dot / std::sqrt(reference_square * candidate_square);
    return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}

bool bitwise_equal(const NumericTensorView & reference, const NumericTensorView & candidate) {
    if (reference.dtype != candidate.dtype || reference.element_count != candidate.element_count ||
        reference.data == nullptr || candidate.data == nullptr) return false;
    const size_t element_size = reference.dtype == DataType::F32 ? sizeof(float) :
        reference.dtype == DataType::F64 ? sizeof(double) : 0;
    if (element_size == 0) return false;
    return std::memcmp(reference.data, candidate.data, reference.element_count * element_size) == 0;
}

size_t argmax(const NumericTensorView & tensor) {
    size_t best = 0;
    for (size_t i = 1; i < tensor.element_count; ++i)
        if (value_at(tensor, i) > value_at(tensor, best)) best = i;
    return best;
}

void set_numeric(NumericalMetrics & metrics, Metric metric, std::optional<double> value) {
    metrics.values[metric].numeric_value = value;
}

void set_boolean(NumericalMetrics & metrics, Metric metric, std::optional<bool> value) {
    metrics.values[metric].boolean_value = value;
}

bool invariant_value(const std::string & name, const NumericalContract & contract,
        const EvaluationContext & context, const NumericalMetrics & metrics, bool * available) {
    *available = true;
    if (name == "same_shape") {
        const auto supplied = context.invariant_values.find(name);
        if (supplied != context.invariant_values.end()) return supplied->second;
        const auto found = metrics.values.find(Metric::BitwiseEquality);
        if (found == metrics.values.end()) { *available = false; return false; }
        return true; // Shape equality is a hard precondition before live metrics are measured.
    }
    if (name == "finite_outputs") {
        const auto found = metrics.values.find(Metric::FiniteOutputs);
        if (found == metrics.values.end() || !found->second.boolean_value) {
            *available = false;
            return false;
        }
        return *found->second.boolean_value;
    }
    if (name == "logical_inputs_equivalent") return context.logical_inputs_equivalent;
    if (name == "probabilities_in_unit_interval") {
        const auto found = context.invariant_values.find(name);
        if (found == context.invariant_values.end()) { *available = false; return false; }
        return found->second;
    }
    const auto supplied = context.invariant_values.find(name);
    if (supplied == context.invariant_values.end()) { *available = false; return false; }
    (void) contract;
    return supplied->second;
}

std::string json_escape(const std::string & text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char value : text) {
        switch (value) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (value < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(value) << std::dec;
                else out << static_cast<char>(value);
        }
    }
    out << '"';
    return out.str();
}

std::string json_number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

bool contract_reference_matches(const NumericalContract & contract, const EvaluationContext & context) {
    if (contract.reference_kind != context.reference_kind || context.reference_identity.empty()) return false;
    if (!contract.reference_identity.empty() && contract.reference_identity.back() == ':')
        return context.reference_identity.rfind(contract.reference_identity, 0) == 0;
    return context.reference_identity == contract.reference_identity;
}

bool scope_matches(const NumericalContract & contract, const EvaluationContext & context,
        const NumericTensorView * reference, const NumericTensorView * candidate, std::string * reason) {
    const auto reject = [&](const std::string & message) { if (reason != nullptr) *reason = message; return false; };
    const auto & scope = contract.scope;
    if (context.operation != contract.operation) return reject("operation does not match contract");
    if (!contract_reference_matches(contract, context)) return reject("reference kind or identity does not match contract");
    if (!wildcard_match(scope.model_identity, context.model_identity)) return reject("model identity is outside contract scope");
    if (!wildcard_match(scope.backend_family, context.backend_family)) return reject("backend family is outside contract scope");
    if (!wildcard_match(scope.implementation_identity, context.implementation_identity)) return reject("implementation identity is outside contract scope");
    if (!wildcard_match(scope.device_family, context.device_family)) return reject("device family is outside contract scope");
    if (!scope.any_device_sm_versions && scope.device_sm_versions != context.device_sm_versions)
        return reject("device SM set is outside contract scope");
    if (!wildcard_match(scope.placement_identity, context.placement_identity)) return reject("placement is outside contract scope");
    if (!list_match(scope.phases, context.phase)) return reject("execution phase is outside contract scope");
    if (!list_match(scope.execution_topologies, context.execution_topology)) return reject("execution topology is outside contract scope");
    if (!list_match(scope.fixture_identities, context.fixture_identity)) return reject("fixture identity is outside contract scope");
    if (!list_match(scope.input_identities, context.input_identity)) return reject("input identity is outside contract scope");
    if (!list_match(scope.token_sequence_identities, context.token_sequence_identity))
        return reject("token sequence identity is outside contract scope");
    const std::string output_dtype = candidate != nullptr ? dtype_name(candidate->dtype) : context.output_dtype;
    const std::string input_dtype = context.input_dtype;
    if (!wildcard_match(scope.output_dtype, output_dtype)) return reject("output dtype is outside contract scope");
    if (!list_match(scope.input_dtypes, input_dtype)) return reject("input dtype is outside contract scope");
    if (!range_match(scope.capacity, context.capacity)) return reject("capacity is outside contract scope");
    if (!range_match(scope.context_length, context.context_length)) return reject("context length is outside contract scope");
    if (!range_match(scope.rows, context.rows)) return reject("row count is outside contract scope");
    if (reference != nullptr && candidate != nullptr) {
        if (reference->shape != candidate->shape) return reject("reference and candidate shapes differ");
        const std::string shape = shape_identity(candidate->shape);
        if (!list_match(scope.shapes, shape) &&
            std::find(scope.shapes.begin(), scope.shapes.end(), "same_as_reference") == scope.shapes.end())
            return reject("tensor shape is outside contract scope");
    } else if (candidate != nullptr) {
        const std::string shape = shape_identity(candidate->shape);
        if (!list_match(scope.shapes, shape) &&
            std::find(scope.shapes.begin(), scope.shapes.end(), "same_as_reference") == scope.shapes.end())
            return reject("tensor shape is outside contract scope");
    }
    return true;
}

const char * category_name(ContractCategory value) {
    switch (value) {
        case ContractCategory::OperationAccuracy: return "OPERATION_ACCURACY";
        case ContractCategory::QuantizationAccuracy: return "QUANTIZATION_ACCURACY";
        case ContractCategory::NumericalCompatibility: return "NUMERICAL_COMPATIBILITY";
        case ContractCategory::ModelOutputAccuracy: return "MODEL_OUTPUT_ACCURACY";
        case ContractCategory::ExecutionInvariants: return "EXECUTION_INVARIANTS";
        case ContractCategory::LifecycleCorrectness: return "LIFECYCLE_CORRECTNESS";
    }
    return "UNKNOWN";
}

const char * reference_kind_name(ReferenceKind value) {
    switch (value) {
        case ReferenceKind::Fp64OperationOracle: return "FP64_OPERATION_ORACLE";
        case ReferenceKind::Fp32OperationReference: return "FP32_OPERATION_REFERENCE";
        case ReferenceKind::CanonicalExecution: return "CANONICAL_EXECUTION";
        case ReferenceKind::UnquantizedWeightReference: return "UNQUANTIZED_WEIGHT_REFERENCE";
        case ReferenceKind::QuantizedWeightReference: return "QUANTIZED_WEIGHT_REFERENCE";
        case ReferenceKind::InvariantReference: return "INVARIANT_REFERENCE";
    }
    return "UNKNOWN";
}

const char * comparison_name(Comparison value) {
    switch (value) {
        case Comparison::LessThan: return "less_than";
        case Comparison::LessEqual: return "less_equal";
        case Comparison::GreaterThan: return "greater_than";
        case Comparison::GreaterEqual: return "greater_equal";
        case Comparison::Equal: return "equal";
    }
    return "unknown";
}

const char * tolerance_unit_name(ToleranceUnit value) {
    switch (value) {
        case ToleranceUnit::Absolute: return "absolute";
        case ToleranceUnit::Fraction: return "fraction";
        case ToleranceUnit::Percent: return "percent";
        case ToleranceUnit::Permille: return "permille";
    }
    return "unknown";
}

void append_tokens_json(std::ostringstream & out, const std::optional<std::vector<uint32_t>> & tokens) {
    if (!tokens) { out << "null"; return; }
    out << '[';
    for (size_t i = 0; i < tokens->size(); ++i) out << (i == 0 ? "" : ",") << (*tokens)[i];
    out << ']';
}

void append_metric_json(std::ostringstream & out, const NumericalMetrics & metrics) {
    out << '{';
    bool first = true;
    for (const auto & entry : metrics.values) {
        if (!first) out << ',';
        first = false;
        out << json_escape(metric_name(entry.first)) << ':';
        if (entry.second.numeric_value) out << json_number(*entry.second.numeric_value);
        else if (entry.second.boolean_value) out << (*entry.second.boolean_value ? "true" : "false");
        else out << "null";
    }
    out << '}';
}

} // namespace

const std::vector<NumericalContract> & project_contracts() {
    static const std::vector<NumericalContract> contracts = generated_contracts();
    return contracts;
}

const NumericalContract * find_contract(const std::string & contract_id, uint32_t version) {
    const auto & contracts = project_contracts();
    const auto found = std::find_if(contracts.begin(), contracts.end(), [&](const NumericalContract & contract) {
        return contract.contract_id == contract_id && contract.version == version;
    });
    return found == contracts.end() ? nullptr : &*found;
}

const char * project_policy_id() noexcept { return generated_policy_id(); }
uint32_t project_policy_version() noexcept { return generated_policy_version(); }
const char * project_policy_digest() noexcept { return generated_policy_digest(); }

std::string shape_identity(const std::vector<uint64_t> & shape) {
    std::ostringstream out;
    for (size_t i = 0; i < shape.size(); ++i) out << (i == 0 ? "" : "x") << shape[i];
    return out.str();
}

double normalize_tolerance(double value, ToleranceUnit unit) {
    if (!std::isfinite(value) || value < 0.0) throw std::invalid_argument("tolerance must be finite and non-negative");
    switch (unit) {
        case ToleranceUnit::Absolute:
        case ToleranceUnit::Fraction: return value;
        case ToleranceUnit::Percent: return value / 100.0;
        case ToleranceUnit::Permille: return value / 1000.0;
    }
    throw std::invalid_argument("unknown tolerance unit");
}

CriterionResult evaluate_metric_criterion(const MetricCriterion & criterion,
        const MetricObservation & observation) {
    CriterionResult result;
    result.metric = criterion.metric;
    result.comparison = criterion.comparison;
    result.required = criterion.required;
    result.declared_limit = criterion.value;
    result.declared_unit = criterion.unit;
    result.normalized_limit = normalize_tolerance(criterion.value, criterion.unit);
    result.observed_value = observation.numeric_value;
    result.observed_boolean = observation.boolean_value;
    if (observation.numeric_value) {
        result.available = std::isfinite(*observation.numeric_value);
        const double measured = *observation.numeric_value;
        switch (criterion.comparison) {
            case Comparison::LessThan: result.passed = measured < result.normalized_limit; break;
            case Comparison::LessEqual: result.passed = measured <= result.normalized_limit; break;
            case Comparison::GreaterThan: result.passed = measured > result.normalized_limit; break;
            case Comparison::GreaterEqual: result.passed = measured >= result.normalized_limit; break;
            case Comparison::Equal: result.passed = measured == result.normalized_limit; break;
        }
    }
    return result;
}

NumericalMetrics measure_tensor_pair(const NumericTensorView & reference,
        const NumericTensorView & candidate, const EvaluationContext & context) {
    if (!valid_tensor_geometry(reference) || !valid_tensor_geometry(candidate) ||
        reference.element_count != candidate.element_count || reference.shape != candidate.shape)
        throw std::invalid_argument("numerical metric inputs have invalid geometry or dtype");

    NumericalMetrics result;
    const bool finite = all_finite(reference) && all_finite(candidate);
    set_boolean(result, Metric::FiniteOutputs, finite);
    if (!finite) {
        set_numeric(result, Metric::MaxAbsoluteError, std::nullopt);
        set_numeric(result, Metric::MeanAbsoluteError, std::nullopt);
        set_numeric(result, Metric::RmsError, std::nullopt);
        set_numeric(result, Metric::RelativeRmsError, std::nullopt);
        set_numeric(result, Metric::CosineSimilarity, std::nullopt);
        set_boolean(result, Metric::BitwiseEquality, std::nullopt);
        set_boolean(result, Metric::Top1Equality, std::nullopt);
        if (context.reference_tokens && context.candidate_tokens)
            set_boolean(result, Metric::TokenSequenceEquality,
                *context.reference_tokens == *context.candidate_tokens);
        return result;
    }

    double max_abs = 0.0;
    ScaledSumSquares error_norm;
    ScaledSumSquares reference_norm;
    double absolute_error_scale = 0.0;
    for (size_t i = 0; i < reference.element_count; ++i) {
        const double left = value_at(reference, i);
        const double right = value_at(candidate, i);
        const double error = right - left;
        const double magnitude = std::abs(error);
        max_abs = std::max(max_abs, magnitude);
        error_norm.add(error);
        reference_norm.add(left);
        absolute_error_scale = std::max(absolute_error_scale, magnitude);
    }
    double normalized_abs_sum = 0.0;
    if (absolute_error_scale != 0.0) {
        for (size_t i = 0; i < reference.element_count; ++i)
            normalized_abs_sum += std::abs(value_at(candidate, i) - value_at(reference, i)) / absolute_error_scale;
    }
    const double mean_abs = absolute_error_scale == 0.0 ? 0.0 :
        absolute_error_scale * (normalized_abs_sum / static_cast<double>(reference.element_count));
    const double rms = error_norm.norm() / std::sqrt(static_cast<double>(reference.element_count));

    set_numeric(result, Metric::MaxAbsoluteError, max_abs);
    set_numeric(result, Metric::MeanAbsoluteError, mean_abs);
    set_numeric(result, Metric::RmsError, rms);
    set_numeric(result, Metric::RelativeRmsError, relative_norm(error_norm, reference_norm));
    set_numeric(result, Metric::CosineSimilarity, cosine_similarity(reference, candidate));
    set_boolean(result, Metric::BitwiseEquality, bitwise_equal(reference, candidate));
    set_boolean(result, Metric::Top1Equality, argmax(reference) == argmax(candidate));
    if (context.reference_tokens && context.candidate_tokens)
        set_boolean(result, Metric::TokenSequenceEquality,
            *context.reference_tokens == *context.candidate_tokens);
    return result;
}

NumericalEvaluation evaluate_contract(const std::string & contract_id, uint32_t version,
        const NumericTensorView * reference, const NumericTensorView * candidate,
        const EvaluationContext & context) {
    NumericalEvaluation result;
    result.contract_id = contract_id;
    result.contract_version = version;
    result.policy_id = project_policy_id();
    result.policy_version = project_policy_version();
    result.policy_digest = project_policy_digest();
    result.reference_identity = context.reference_identity;
    result.reference_kind = context.reference_kind;
    result.context = context;

    const auto & contracts = project_contracts();
    const auto by_id = std::find_if(contracts.begin(), contracts.end(), [&](const NumericalContract & item) {
        return item.contract_id == contract_id;
    });
    const NumericalContract * contract = find_contract(contract_id, version);
    if (by_id == contracts.end()) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("unknown contract id");
        return result;
    }
    if (contract == nullptr) {
        result.contract_version = by_id->version;
        result.contract_status = by_id->status;
        result.category = by_id->category;
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("contract version mismatch");
        return result;
    }
    result.category = contract->category;
    result.contract_status = contract->status;
    result.reference_kind = contract->reference_kind;
    if (reference != nullptr) {
        result.reference_shape = reference->shape;
        result.reference_dtype = dtype_name(reference->dtype);
    }
    if (candidate != nullptr) {
        result.candidate_shape = candidate->shape;
        result.candidate_dtype = dtype_name(candidate->dtype);
    }

    const bool needs_numeric_reference = std::any_of(contract->measurement_metrics.begin(),
        contract->measurement_metrics.end(), [](Metric metric) {
            return metric != Metric::FiniteOutputs && metric != Metric::BitwiseEquality;
        });
    if (needs_numeric_reference && reference == nullptr) {
        result.status = EvaluationStatus::NotTested;
        result.failure_reasons.push_back("required numerical reference is missing");
        return result;
    }
    if (candidate == nullptr && !contract->measurement_metrics.empty()) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("candidate tensor is missing");
        return result;
    }
    if (reference != nullptr && candidate != nullptr &&
        (!valid_tensor_geometry(*reference) || !valid_tensor_geometry(*candidate) ||
         reference->element_count != candidate->element_count || reference->shape != candidate->shape)) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("reference and candidate tensor shapes/counts are incompatible");
        return result;
    }
    if (candidate != nullptr && !valid_tensor_geometry(*candidate)) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("candidate tensor storage, count, or dtype is invalid");
        return result;
    }
    std::string scope_error;
    if (!scope_matches(*contract, context, reference, candidate, &scope_error)) {
        result.status = EvaluationStatus::NotApplicable;
        result.failure_reasons.push_back(scope_error);
        return result;
    }
    if (contract->scope.requires_logical_inputs_equivalent && !context.logical_inputs_equivalent) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("equivalent logical inputs were not established");
        return result;
    }

    if (reference != nullptr && candidate != nullptr) {
        try {
            result.measured_metrics = measure_tensor_pair(*reference, *candidate, context);
        } catch (const std::exception & error) {
            result.status = EvaluationStatus::InvalidEvaluation;
            result.failure_reasons.push_back(error.what());
            return result;
        }
    } else if (candidate != nullptr) {
        const bool finite = all_finite(*candidate);
        set_boolean(result.measured_metrics, Metric::FiniteOutputs, finite);
    }
    if (context.reference_tokens && context.candidate_tokens) {
        set_boolean(result.measured_metrics, Metric::TokenSequenceEquality,
            *context.reference_tokens == *context.candidate_tokens);
    }

    bool invariant_failed = false;
    bool invariant_unavailable = false;
    for (const auto & invariant : contract->required_invariants) {
        bool available = false;
        const bool passed = invariant_value(invariant, *contract, context, result.measured_metrics, &available);
        if (!available) {
            invariant_unavailable = true;
            result.failure_reasons.push_back("required invariant unavailable: " + invariant);
        } else if (!passed) {
            invariant_failed = true;
            result.failure_reasons.push_back("required invariant failed: " + invariant);
        }
    }
    if (invariant_failed) {
        result.status = EvaluationStatus::Fail;
        return result;
    }

    if (contract->status != ContractStatus::Active) {
        result.status = EvaluationStatus::NotTested;
        result.failure_reasons.push_back(contract->status == ContractStatus::NeedsCalibration ?
            "contract needs calibration; measured diagnostics do not authorize selection" :
            contract->status == ContractStatus::Provisional ?
                "contract is provisional; it cannot authorize selection" :
                "contract is deprecated and cannot authorize selection");
        return result;
    }
    if (invariant_unavailable) {
        result.status = EvaluationStatus::InvalidEvaluation;
        return result;
    }

    bool required_failed = false;
    bool required_unavailable = false;
    for (const auto & criterion : contract->criteria) {
        const auto found = result.measured_metrics.values.find(criterion.metric);
        const MetricObservation unavailable;
        const auto & observation = found == result.measured_metrics.values.end() ? unavailable : found->second;
        CriterionResult evaluated;
        try {
            evaluated = evaluate_metric_criterion(criterion, observation);
        } catch (const std::exception & error) {
            result.status = EvaluationStatus::InvalidEvaluation;
            result.failure_reasons.push_back(error.what());
            return result;
        }
        if (!evaluated.available && criterion.required) required_unavailable = true;
        if (evaluated.available && !evaluated.passed && criterion.required) required_failed = true;
        result.criteria.push_back(evaluated);
    }
    if (required_failed) {
        result.status = EvaluationStatus::Fail;
        result.failure_reasons.push_back("one or more mandatory numerical criteria failed (AND aggregation)");
    } else if (required_unavailable) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("one or more mandatory numerical criteria are unavailable");
    } else {
        result.status = EvaluationStatus::Pass;
    }
    if (result.status == EvaluationStatus::Pass && contract->status == ContractStatus::Active)
        NumericalEvaluationAuthority::seal(result);
    return result;
}

NumericalEvaluation evaluate_invariants(const std::string & contract_id, uint32_t version,
        const EvaluationContext & context) {
    return evaluate_contract(contract_id, version, nullptr, nullptr, context);
}

NumericalEvaluation evaluate_recorded_metrics(const std::string & contract_id, uint32_t version,
        const EvaluationContext & context, const NumericalMetrics & recorded_metrics) {
    NumericalEvaluation result;
    result.contract_id = contract_id;
    result.contract_version = version;
    result.policy_id = project_policy_id();
    result.policy_version = project_policy_version();
    result.policy_digest = project_policy_digest();
    result.reference_identity = context.reference_identity;
    result.context = context;
    result.reference_shape = context.output_shape;
    result.candidate_shape = context.output_shape;
    result.reference_dtype = context.output_dtype;
    result.candidate_dtype = context.output_dtype;
    result.measured_metrics = recorded_metrics;
    result.replayed_metrics_only = true;

    const auto & contracts = project_contracts();
    const auto by_id = std::find_if(contracts.begin(), contracts.end(), [&](const NumericalContract & item) {
        return item.contract_id == contract_id;
    });
    const NumericalContract * contract = find_contract(contract_id, version);
    if (by_id == contracts.end()) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("unknown contract id");
        return result;
    }
    if (contract == nullptr) {
        result.contract_version = by_id->version;
        result.contract_status = by_id->status;
        result.category = by_id->category;
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("contract version mismatch");
        return result;
    }
    result.category = contract->category;
    result.contract_status = contract->status;
    result.reference_kind = contract->reference_kind;
    if (!contract_reference_matches(*contract, context)) {
        result.status = EvaluationStatus::NotApplicable;
        result.failure_reasons.push_back("reference kind or identity does not match contract");
        return result;
    }
    std::string scope_error;
    if (!scope_matches(*contract, context, nullptr, nullptr, &scope_error)) {
        result.status = EvaluationStatus::NotApplicable;
        result.failure_reasons.push_back(scope_error);
        return result;
    }
    for (const auto & entry : recorded_metrics.values) {
        if (std::find(contract->measurement_metrics.begin(), contract->measurement_metrics.end(), entry.first) ==
            contract->measurement_metrics.end()) {
            result.status = EvaluationStatus::InvalidEvaluation;
            result.failure_reasons.push_back("record contains a metric not declared by the contract");
            return result;
        }
    }

    bool invariant_failed = false;
    bool invariant_unavailable = false;
    for (const auto & invariant : contract->required_invariants) {
        bool available = false;
        const bool passed = invariant_value(invariant, *contract, context, result.measured_metrics, &available);
        if (!available) {
            invariant_unavailable = true;
            result.failure_reasons.push_back("required invariant unavailable: " + invariant);
        } else if (!passed) {
            invariant_failed = true;
            result.failure_reasons.push_back("required invariant failed: " + invariant);
        }
    }
    if (invariant_failed) {
        result.status = EvaluationStatus::Fail;
        return result;
    }
    if (contract->status != ContractStatus::Active) {
        result.status = EvaluationStatus::NotTested;
        result.failure_reasons.push_back(contract->status == ContractStatus::NeedsCalibration ?
            "contract needs calibration; replayed metrics are diagnostic only" :
            contract->status == ContractStatus::Provisional ?
                "contract is provisional; replayed metrics are diagnostic only" :
                "contract is deprecated; replayed metrics are diagnostic only");
        return result;
    }
    if (invariant_unavailable) {
        result.status = EvaluationStatus::InvalidEvaluation;
        return result;
    }

    bool required_failed = false;
    bool required_unavailable = false;
    for (const auto & criterion : contract->criteria) {
        const auto found = result.measured_metrics.values.find(criterion.metric);
        const MetricObservation unavailable;
        const auto & observation = found == result.measured_metrics.values.end() ? unavailable : found->second;
        CriterionResult evaluated;
        try {
            evaluated = evaluate_metric_criterion(criterion, observation);
        } catch (const std::exception & error) {
            result.status = EvaluationStatus::InvalidEvaluation;
            result.failure_reasons.push_back(error.what());
            return result;
        }
        if (!evaluated.available && criterion.required) required_unavailable = true;
        if (evaluated.available && !evaluated.passed && criterion.required) required_failed = true;
        result.criteria.push_back(evaluated);
    }
    if (required_failed) {
        result.status = EvaluationStatus::Fail;
        result.failure_reasons.push_back("one or more replayed numerical criteria failed (AND aggregation)");
    } else if (required_unavailable) {
        result.status = EvaluationStatus::InvalidEvaluation;
        result.failure_reasons.push_back("one or more required replayed metrics are unavailable");
    } else {
        result.status = EvaluationStatus::Pass;
    }
    return result;
}

NumericalEvaluation mark_numerical_evaluation_replay_only(NumericalEvaluation evaluation) {
    evaluation.replayed_metrics_only = true;
    evaluation.authority_snapshot_.clear();
    return evaluation;
}

EvidenceRejectionReason numerical_evidence_rejection_reason(const NumericalEvaluation & evaluation,
        const RuntimeQualificationContext & runtime_context) {
    if (evaluation.replayed_metrics_only) return EvidenceRejectionReason::ReplayOnly;
    if (evaluation.status == EvaluationStatus::Fail) return EvidenceRejectionReason::FailedQualification;
    if (evaluation.status != EvaluationStatus::Pass || !NumericalEvaluationAuthority::authentic(evaluation))
        return EvidenceRejectionReason::InvalidEvidence;
    if (runtime_context.candidate_identity.empty() || evaluation.context.candidate_identity.empty())
        return EvidenceRejectionReason::CandidateIdentityMismatch;
    if (evaluation.context.qualification_run_identity.empty())
        return EvidenceRejectionReason::InvalidEvidence;
    if (evaluation.policy_id != project_policy_id() || evaluation.policy_version != project_policy_version() ||
        evaluation.policy_digest != project_policy_digest()) return EvidenceRejectionReason::PolicyMismatch;
    const NumericalContract * contract = find_contract(evaluation.contract_id, evaluation.contract_version);
    if (contract == nullptr) {
        return std::any_of(project_contracts().begin(), project_contracts().end(), [&](const NumericalContract & item) {
            return item.contract_id == evaluation.contract_id;
        }) ? EvidenceRejectionReason::ContractVersionMismatch : EvidenceRejectionReason::ContractUnknown;
    }
    if (contract->status != ContractStatus::Active || evaluation.contract_status != contract->status)
        return EvidenceRejectionReason::ContractNotActive;
    if (evaluation.category != contract->category || evaluation.reference_kind != contract->reference_kind)
        return EvidenceRejectionReason::InvalidEvidence;

    const auto & context = evaluation.context;
    if (context.candidate_identity != runtime_context.candidate_identity)
        return EvidenceRejectionReason::CandidateIdentityMismatch;
    if (context.reference_kind != contract->reference_kind ||
        evaluation.reference_identity != context.reference_identity ||
        !contract_reference_matches(*contract, context)) return EvidenceRejectionReason::ReferenceMismatch;
    if (context.model_identity.empty() || context.model_identity != runtime_context.model_identity)
        return EvidenceRejectionReason::ModelArtifactMismatch;
    if (context.backend_family != runtime_context.backend_family || context.device_family != runtime_context.device_family)
        return EvidenceRejectionReason::BackendMismatch;
    if (context.implementation_identity != runtime_context.implementation_identity)
        return EvidenceRejectionReason::ImplementationMismatch;
    if (context.device_identities != runtime_context.device_identities ||
        context.device_sm_versions != runtime_context.device_sm_versions)
        return EvidenceRejectionReason::DeviceMismatch;
    if (context.placement_identity != runtime_context.placement_identity)
        return EvidenceRejectionReason::PlacementMismatch;
    if (context.output_dtype != runtime_context.activation_dtype || context.input_dtype != runtime_context.kv_dtype ||
        evaluation.candidate_dtype != context.output_dtype ||
        (contract->reference_kind == ReferenceKind::CanonicalExecution &&
            evaluation.reference_dtype != context.output_dtype))
        return EvidenceRejectionReason::DTypeMismatch;
    if (evaluation.candidate_shape != context.output_shape || evaluation.reference_shape.empty() ||
        evaluation.reference_shape != evaluation.candidate_shape ||
        (evaluation.candidate_shape.empty() && !contract->scope.shapes.empty() &&
            std::find(contract->scope.shapes.begin(), contract->scope.shapes.end(), "*") == contract->scope.shapes.end()))
        return EvidenceRejectionReason::ShapeMismatch;
    if (!evaluation.candidate_shape.empty() &&
        !list_match(contract->scope.shapes, shape_identity(evaluation.candidate_shape)) &&
        std::find(contract->scope.shapes.begin(), contract->scope.shapes.end(), "same_as_reference") ==
            contract->scope.shapes.end()) return EvidenceRejectionReason::ShapeMismatch;
    if (context.phase != runtime_context.phase || context.capacity != runtime_context.capacity ||
        context.context_length != runtime_context.context_length || context.rows != runtime_context.rows)
        return EvidenceRejectionReason::RuntimeFactsMismatch;
    if (contract->scope.requires_logical_inputs_equivalent && !context.logical_inputs_equivalent)
        return EvidenceRejectionReason::ExecutionScopeMismatch;
    if (!list_match(contract->scope.execution_topologies, "*") &&
        context.execution_topology != runtime_context.execution_topology)
        return EvidenceRejectionReason::ExecutionScopeMismatch;
    if (!list_match(contract->scope.fixture_identities, "*") &&
        context.fixture_identity != runtime_context.fixture_identity)
        return EvidenceRejectionReason::ExecutionScopeMismatch;
    if (!list_match(contract->scope.input_identities, "*") && context.input_identity != runtime_context.input_identity)
        return EvidenceRejectionReason::ExecutionScopeMismatch;
    if (!list_match(contract->scope.token_sequence_identities, "*") &&
        context.token_sequence_identity != runtime_context.token_sequence_identity)
        return EvidenceRejectionReason::ExecutionScopeMismatch;
    std::string scope_error;
    if (!scope_matches(*contract, context, nullptr, nullptr, &scope_error))
        return scope_error.find("shape") != std::string::npos ? EvidenceRejectionReason::ShapeMismatch :
            EvidenceRejectionReason::ExecutionScopeMismatch;
    for (Metric metric : contract->measurement_metrics)
        if (evaluation.measured_metrics.values.count(metric) == 0)
            return EvidenceRejectionReason::RequiredMetricMissing;
    for (const auto & invariant : contract->required_invariants) {
        bool available = false;
        if (!invariant_value(invariant, *contract, context, evaluation.measured_metrics, &available) || !available)
            return EvidenceRejectionReason::InvalidEvidence;
    }
    if (evaluation.criteria.size() != contract->criteria.size()) return EvidenceRejectionReason::InvalidEvidence;
    for (size_t i = 0; i < contract->criteria.size(); ++i) {
        const auto & criterion = contract->criteria[i];
        const auto & recorded = evaluation.criteria[i];
        if (recorded.metric != criterion.metric || recorded.comparison != criterion.comparison ||
            recorded.required != criterion.required || recorded.declared_limit != criterion.value ||
            recorded.declared_unit != criterion.unit) return EvidenceRejectionReason::InvalidEvidence;
        const auto found = evaluation.measured_metrics.values.find(criterion.metric);
        const MetricObservation unavailable;
        const auto & observation = found == evaluation.measured_metrics.values.end() ? unavailable : found->second;
        try {
            const auto checked = evaluate_metric_criterion(criterion, observation);
            if (!checked.available && criterion.required) return EvidenceRejectionReason::RequiredMetricMissing;
            if (checked.available && !checked.passed && criterion.required)
                return EvidenceRejectionReason::RequiredMetricFailed;
            if (recorded.available != checked.available || recorded.passed != checked.passed ||
                recorded.observed_value != checked.observed_value ||
                recorded.observed_boolean != checked.observed_boolean ||
                recorded.normalized_limit != checked.normalized_limit)
                return EvidenceRejectionReason::InvalidEvidence;
        } catch (...) { return EvidenceRejectionReason::InvalidEvidence; }
    }
    return EvidenceRejectionReason::None;
}

bool numerical_evidence_matches_runtime(const NumericalEvaluation & evaluation,
        const RuntimeQualificationContext & runtime_context) {
    return numerical_evidence_rejection_reason(evaluation, runtime_context) == EvidenceRejectionReason::None;
}

const char * evidence_rejection_reason_name(EvidenceRejectionReason reason) noexcept {
    switch (reason) {
        case EvidenceRejectionReason::None: return "NONE";
        case EvidenceRejectionReason::MissingEvidence: return "MISSING_EVIDENCE";
        case EvidenceRejectionReason::InvalidEvidence: return "EVIDENCE_INVALID";
        case EvidenceRejectionReason::FailedQualification: return "REQUIRED_METRIC_FAIL";
        case EvidenceRejectionReason::ReplayOnly: return "REPLAY_ONLY";
        case EvidenceRejectionReason::PolicyMismatch: return "POLICY_MISMATCH";
        case EvidenceRejectionReason::ContractUnknown: return "UNKNOWN_CONTRACT";
        case EvidenceRejectionReason::ContractVersionMismatch: return "CONTRACT_VERSION_MISMATCH";
        case EvidenceRejectionReason::ContractNotActive: return "CONTRACT_NOT_ACTIVE";
        case EvidenceRejectionReason::CandidateIdentityMismatch: return "CANDIDATE_IDENTITY_MISMATCH";
        case EvidenceRejectionReason::ReferenceMismatch: return "REFERENCE_MISMATCH";
        case EvidenceRejectionReason::ModelArtifactMismatch: return "MODEL_ARTIFACT_MISMATCH";
        case EvidenceRejectionReason::BackendMismatch: return "BACKEND_SCOPE_MISMATCH";
        case EvidenceRejectionReason::ImplementationMismatch: return "IMPLEMENTATION_IDENTITY_MISMATCH";
        case EvidenceRejectionReason::DeviceMismatch: return "DEVICE_SCOPE_MISMATCH";
        case EvidenceRejectionReason::PlacementMismatch: return "PLACEMENT_SCOPE_MISMATCH";
        case EvidenceRejectionReason::DTypeMismatch: return "DTYPE_SCOPE_MISMATCH";
        case EvidenceRejectionReason::ShapeMismatch: return "SHAPE_SCOPE_MISMATCH";
        case EvidenceRejectionReason::RuntimeFactsMismatch: return "RUNTIME_FACTS_MISMATCH";
        case EvidenceRejectionReason::ExecutionScopeMismatch: return "EXECUTION_SCOPE_MISMATCH";
        case EvidenceRejectionReason::RequiredMetricMissing: return "REQUIRED_METRIC_MISSING";
        case EvidenceRejectionReason::RequiredMetricFailed: return "REQUIRED_METRIC_FAIL";
        case EvidenceRejectionReason::EvidenceInvalidated: return "EVIDENCE_INVALIDATED";
    }
    return "EVIDENCE_INVALID";
}

std::string evaluation_status_name(EvaluationStatus status) {
    switch (status) {
        case EvaluationStatus::Pass: return "PASS";
        case EvaluationStatus::Fail: return "FAIL";
        case EvaluationStatus::NotApplicable: return "NOT_APPLICABLE";
        case EvaluationStatus::NotTested: return "NOT_TESTED";
        case EvaluationStatus::InvalidEvaluation: return "INVALID_EVALUATION";
    }
    return "INVALID_EVALUATION";
}

std::string contract_status_name(ContractStatus status) {
    switch (status) {
        case ContractStatus::Active: return "ACTIVE";
        case ContractStatus::Provisional: return "PROVISIONAL";
        case ContractStatus::NeedsCalibration: return "NEEDS_CALIBRATION";
        case ContractStatus::Deprecated: return "DEPRECATED";
    }
    return "NEEDS_CALIBRATION";
}

std::string metric_name(Metric metric) {
    switch (metric) {
        case Metric::MaxAbsoluteError: return "max_absolute_error";
        case Metric::MeanAbsoluteError: return "mean_absolute_error";
        case Metric::RmsError: return "rms_error";
        case Metric::RelativeRmsError: return "relative_rms_error";
        case Metric::CosineSimilarity: return "cosine_similarity";
        case Metric::FiniteOutputs: return "finite_outputs";
        case Metric::BitwiseEquality: return "bitwise_equality";
        case Metric::Top1Equality: return "top1_equality";
        case Metric::TokenSequenceEquality: return "token_sequence_equality";
    }
    return "unknown";
}

std::string evaluation_json(const NumericalEvaluation & evaluation) {
    std::ostringstream out;
    out << std::setprecision(17) << '{'
        << "\"contract_id\":" << json_escape(evaluation.contract_id)
        << ",\"contract_version\":" << evaluation.contract_version
        << ",\"contract_status\":" << json_escape(contract_status_name(evaluation.contract_status))
        << ",\"category\":" << json_escape(category_name(evaluation.category))
        << ",\"evaluation_status\":" << json_escape(evaluation_status_name(evaluation.status))
        << ",\"policy_id\":" << json_escape(evaluation.policy_id)
        << ",\"policy_version\":" << evaluation.policy_version
        << ",\"policy_sha256\":" << json_escape(evaluation.policy_digest)
        << ",\"reference_kind\":" << json_escape(reference_kind_name(evaluation.reference_kind))
        << ",\"context_reference_kind\":" << json_escape(reference_kind_name(evaluation.context.reference_kind))
        << ",\"reference_identity\":" << json_escape(evaluation.reference_identity)
        << ",\"reference_implementation_revision\":" << json_escape(evaluation.context.reference_implementation_revision)
        << ",\"reference_accumulation_precision\":" << json_escape(evaluation.context.reference_accumulation_precision)
        << ",\"reference_input_representation\":" << json_escape(evaluation.context.reference_input_representation)
        << ",\"reference_output_representation\":" << json_escape(evaluation.context.reference_output_representation)
        << ",\"candidate_identity\":" << json_escape(evaluation.context.candidate_identity)
        << ",\"qualification_run_identity\":" << json_escape(evaluation.context.qualification_run_identity)
        << ",\"model_identity\":" << json_escape(evaluation.context.model_identity)
        << ",\"backend_family\":" << json_escape(evaluation.context.backend_family)
        << ",\"implementation_identity\":" << json_escape(evaluation.context.implementation_identity)
        << ",\"fixture_identity\":" << json_escape(evaluation.context.fixture_identity)
        << ",\"input_identity\":" << json_escape(evaluation.context.input_identity)
        << ",\"token_sequence_identity\":" << json_escape(evaluation.context.token_sequence_identity)
        << ",\"operation\":" << json_escape(evaluation.context.operation)
        << ",\"output_name\":" << json_escape(evaluation.context.output_name)
        << ",\"output_dtype\":" << json_escape(evaluation.context.output_dtype)
        << ",\"input_dtype\":" << json_escape(evaluation.context.input_dtype)
        << ",\"capacity\":" << evaluation.context.capacity
        << ",\"context_length\":" << evaluation.context.context_length
        << ",\"rows\":" << evaluation.context.rows
        << ",\"phase\":" << json_escape(evaluation.context.phase)
        << ",\"execution_topology\":" << json_escape(evaluation.context.execution_topology)
        << ",\"reference_dtype\":" << json_escape(evaluation.reference_dtype)
        << ",\"candidate_dtype\":" << json_escape(evaluation.candidate_dtype)
        << ",\"placement_identity\":" << json_escape(evaluation.context.placement_identity)
        << ",\"device_family\":" << json_escape(evaluation.context.device_family)
        << ",\"logical_inputs_equivalent\":" << (evaluation.context.logical_inputs_equivalent ? "true" : "false")
        << ",\"reference_operation_parameters\":";
    out << '{';
    bool first_parameter = true;
    for (const auto & parameter : evaluation.context.reference_operation_parameters) {
        if (!first_parameter) out << ',';
        first_parameter = false;
        out << json_escape(parameter.first) << ':' << json_escape(parameter.second);
    }
    out << "},\"reference_tokens\":";
    append_tokens_json(out, evaluation.context.reference_tokens);
    out << ",\"candidate_tokens\":";
    append_tokens_json(out, evaluation.context.candidate_tokens);
    out << ",\"invariant_values\":{";
    bool first_invariant = true;
    for (const auto & invariant : evaluation.context.invariant_values) {
        if (!first_invariant) out << ',';
        first_invariant = false;
        out << json_escape(invariant.first) << ':' << (invariant.second ? "true" : "false");
    }
    out << "},\"device_identities\":[";
    for (size_t i = 0; i < evaluation.context.device_identities.size(); ++i)
        out << (i == 0 ? "" : ",") << json_escape(evaluation.context.device_identities[i]);
    out << "],\"device_sm_versions\":[";
    for (size_t i = 0; i < evaluation.context.device_sm_versions.size(); ++i)
        out << (i == 0 ? "" : ",") << evaluation.context.device_sm_versions[i];
    out << "],\"reference_shape\":[";
    for (size_t i = 0; i < evaluation.reference_shape.size(); ++i)
        out << (i == 0 ? "" : ",") << evaluation.reference_shape[i];
    out << "],\"candidate_shape\":[";
    for (size_t i = 0; i < evaluation.candidate_shape.size(); ++i)
        out << (i == 0 ? "" : ",") << evaluation.candidate_shape[i];
    out << "],\"context_output_shape\":[";
    for (size_t i = 0; i < evaluation.context.output_shape.size(); ++i)
        out << (i == 0 ? "" : ",") << evaluation.context.output_shape[i];
    out << "],\"measured_metrics\":";
    append_metric_json(out, evaluation.measured_metrics);
    out << ",\"criteria\":[";
    for (size_t i = 0; i < evaluation.criteria.size(); ++i) {
        const auto & criterion = evaluation.criteria[i];
        if (i != 0) out << ',';
        out << "{\"metric\":" << json_escape(metric_name(criterion.metric))
            << ",\"observed\":";
        if (criterion.observed_value) out << json_number(*criterion.observed_value);
        else if (criterion.observed_boolean) out << (*criterion.observed_boolean ? "true" : "false");
        else out << "null";
        out << ",\"comparison\":" << json_escape(comparison_name(criterion.comparison))
            << ",\"declared_limit\":" << json_number(criterion.declared_limit)
            << ",\"normalized_limit\":" << json_number(criterion.normalized_limit)
            << ",\"declared_unit\":" << json_escape(tolerance_unit_name(criterion.declared_unit))
            << ",\"required\":" << (criterion.required ? "true" : "false")
            << ",\"available\":" << (criterion.available ? "true" : "false")
            << ",\"passed\":" << (criterion.passed ? "true" : "false") << '}';
    }
    out << "],\"failure_reasons\":[";
    for (size_t i = 0; i < evaluation.failure_reasons.size(); ++i)
        out << (i == 0 ? "" : ",") << json_escape(evaluation.failure_reasons[i]);
    out << "],\"replayed_metrics_only\":" << (evaluation.replayed_metrics_only ? "true" : "false") << '}';
    return out.str();
}

} // namespace vbuf_ml::numerics
