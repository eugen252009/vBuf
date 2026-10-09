#include "vbuf_numerical_contracts.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ml::numerics;
namespace {
constexpr const char * model_identity =
    "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
constexpr const char * placement_identity = "multi:0x26,1x14;emb=0;norm=1;head=1";

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::string> csv_fields(const std::string & line) {
    std::vector<std::string> result;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char value = line[i];
        if (value == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (value == ',' && !quoted) {
            result.push_back(field);
            field.clear();
        } else {
            field.push_back(value);
        }
    }
    require(!quoted, "unterminated CSV quoted field");
    result.push_back(field);
    return result;
}

std::map<std::string, size_t> csv_header(const std::string & line) {
    const auto fields = csv_fields(line);
    std::map<std::string, size_t> result;
    for (size_t i = 0; i < fields.size(); ++i) result.emplace(fields[i], i);
    return result;
}

const std::string & field(const std::vector<std::string> & row,
        const std::map<std::string, size_t> & header, const std::string & name) {
    const auto found = header.find(name);
    require(found != header.end() && found->second < row.size(), "CSV field missing: " + name);
    return row[found->second];
}

double number(const std::vector<std::string> & row,
        const std::map<std::string, size_t> & header, const std::string & name) {
    const double value = std::stod(field(row, header, name));
    require(std::isfinite(value), "non-finite value in historical evidence field " + name);
    return value;
}

bool yes(const std::string & value) {
    return value == "yes" || value == "true" || value == "1";
}

void record_metric(NumericalMetrics & metrics, Metric metric, double value) {
    metrics.values[metric].numeric_value = value;
}

void record_metric(NumericalMetrics & metrics, Metric metric, bool value) {
    metrics.values[metric].boolean_value = value;
}

std::vector<std::string> device_ids() {
    return {"CUDA0@0000:04:00.0", "CUDA1@0000:07:00.0"};
}

EvaluationContext topology_context(const std::vector<std::string> & row,
        const std::map<std::string, size_t> & header, const std::string & operation) {
    EvaluationContext context;
    const std::string comparison = field(row, header, "comparison");
    context.operation = operation;
    context.output_name = operation;
    context.reference_kind = ReferenceKind::CanonicalExecution;
    context.reference_identity = "canonical-packed-v-v1:" + comparison;
    context.model_identity = model_identity;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = comparison;
    context.device_family = "CUDA";
    context.device_identities = device_ids();
    context.device_sm_versions = {86, 75};
    context.placement_identity = placement_identity;
    context.phase = "decode";
    context.execution_topology = comparison;
    context.fixture_identity = field(row, header, "fixture_id") + ":repeat=" + field(row, header, "repeat");
    context.input_identity = "token_hash=" + field(row, header, "token_hash");
    context.token_sequence_identity = field(row, header, "token_hash");
    context.output_dtype = "F32";
    context.input_dtype = "F16";
    context.capacity = 512;
    context.context_length = static_cast<uint32_t>(std::stoul(field(row, header, "position")));
    context.rows = 1;
    context.logical_inputs_equivalent = true;
    context.invariant_values["same_shape"] = true;
    context.invariant_values["finite_outputs"] = true;
    return context;
}

NumericalEvaluation replay_output(const std::vector<std::string> & row,
        const std::map<std::string, size_t> & header, const std::string & operation) {
    const std::string prefix = operation == "final_hidden" ? "hidden_" : "logits_";
    NumericalMetrics metrics;
    record_metric(metrics, Metric::MaxAbsoluteError, number(row, header, prefix + "max_abs"));
    record_metric(metrics, Metric::RelativeRmsError, number(row, header, prefix + "relative_rms"));
    record_metric(metrics, Metric::CosineSimilarity, number(row, header, prefix + "cosine"));
    record_metric(metrics, Metric::FiniteOutputs, true);
    if (operation == "final_logits")
        record_metric(metrics, Metric::Top1Equality, yes(field(row, header, "top1_equal")));
    return evaluate_recorded_metrics(operation == "final_hidden" ?
        "qwen3.final_hidden.canonical_compatibility" : "qwen3.final_logits.canonical_compatibility",
        1, topology_context(row, header, operation), metrics);
}

void replay_topology(const std::string & path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot open historical topology evidence: " + path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "historical topology evidence is empty");
    const auto header = csv_header(line);
    size_t rows_seen = 0;
    size_t failed_logits = 0;
    size_t passed_logits = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto row = csv_fields(line);
        if (field(row, header, "fixture_id") != "baseline-existing-natural-repeat-p8-g25-cap512") continue;
        auto hidden = replay_output(row, header, "final_hidden");
        auto logits = replay_output(row, header, "final_logits");
        require(hidden.replayed_metrics_only && logits.replayed_metrics_only,
            "historical output metrics were not labeled replay-only");
        require(hidden.status == EvaluationStatus::Pass,
            "historical hidden result unexpectedly left the preserved compatibility gate: " + evaluation_json(hidden));
        if (field(row, header, "comparison") == "native_incremental_vs_native_prefill32") {
            require(logits.status == EvaluationStatus::Fail,
                "known native topology failure did not fail the centralized logits contract");
            ++failed_logits;
        } else if (field(row, header, "comparison") == "canonical_prefill32_vs_native_prefill32") {
            require(logits.status == EvaluationStatus::Fail,
                "known common-history prefill failure did not fail the centralized logits contract");
            ++failed_logits;
        } else {
            require(logits.status == EvaluationStatus::Pass,
                "known passing topology control failed the centralized logits contract");
            ++passed_logits;
        }
        std::cout << "numerical_contract_historical_replay " << evaluation_json(hidden) << '\n';
        std::cout << "numerical_contract_historical_replay " << evaluation_json(logits) << '\n';
        ++rows_seen;
    }
    require(rows_seen == 8 && failed_logits == 4 && passed_logits == 4,
        "historical topology repeat matrix was incomplete or changed expected pass/fail classification");
}

EvaluationContext av_context(const std::vector<std::string> & row,
        const std::map<std::string, size_t> & header) {
    const std::string phase = field(row, header, "phase");
    const std::string workload = field(row, header, "workload");
    const uint32_t layer = static_cast<uint32_t>(std::stoul(field(row, header, "layer")));
    EvaluationContext context;
    context.operation = "attention_av";
    context.output_name = "native_av_output";
    context.reference_kind = ReferenceKind::Fp64OperationOracle;
    context.reference_identity = "qwen3-av-boundary-fp64-oracle-v1";
    context.model_identity = model_identity;
    context.backend_family = "GGML_CUDA";
    context.implementation_identity = "ggml-native-layout-av";
    context.device_family = "CUDA";
    context.device_identities = {"CUDA0@0000:04:00.0"};
    context.device_sm_versions = {86};
    context.placement_identity = placement_identity;
    context.phase = phase;
    context.execution_topology = "native-av-side-branch";
    context.fixture_identity = "qwen3-native-av-" + phase + "32-cap512-layer" + std::to_string(layer);
    context.input_identity = "captured-qwen3-av-boundary:" + workload + ":layer=" + std::to_string(layer);
    context.token_sequence_identity = "historical-captured-qwen3-input";
    context.output_dtype = "F32";
    context.input_dtype = "F16";
    context.capacity = 512;
    context.context_length = static_cast<uint32_t>(std::stoul(field(row, header, "visible_context")));
    context.rows = static_cast<uint32_t>(std::stoul(field(row, header, "count")) == 163840 ? 32 : 1);
    context.output_shape = {128, context.rows, 40};
    context.logical_inputs_equivalent = true;
    context.invariant_values["same_shape"] = true;
    context.invariant_values["finite_outputs"] = true;
    return context;
}

void replay_av_capture(const std::string & path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot open captured AV evidence: " + path);
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "captured AV evidence is empty");
    const auto header = csv_header(line);
    size_t rows_seen = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto row = csv_fields(line);
        if (field(row, header, "scope") != "all" || field(row, header, "layer") != "0") continue;
        const std::string phase = field(row, header, "phase");
        const std::string workload = field(row, header, "workload");
        if (workload != "prefill32-cap512" && workload != "decode32-cap512") continue;
        NumericalMetrics metrics;
        record_metric(metrics, Metric::MaxAbsoluteError, number(row, header, "native_oracle_max_abs"));
        record_metric(metrics, Metric::RmsError, number(row, header, "native_oracle_rms"));
        record_metric(metrics, Metric::RelativeRmsError, number(row, header, "native_oracle_relative_rms"));
        record_metric(metrics, Metric::CosineSimilarity, number(row, header, "native_oracle_cosine"));
        record_metric(metrics, Metric::FiniteOutputs, true);
        auto evaluation = evaluate_recorded_metrics("qwen3.attention_av.fp64_model_boundary_observation",
            1, av_context(row, header), metrics);
        require(evaluation.status == EvaluationStatus::NotTested && evaluation.replayed_metrics_only &&
            evaluation.contract_status == ContractStatus::NeedsCalibration,
            "captured model-boundary oracle metrics were not preserved as non-authorizing observations");
        std::cout << "numerical_contract_historical_replay " << evaluation_json(evaluation) << '\n';
        ++rows_seen;
    }
    require(rows_seen == 2, "expected layer-0 prefill and decode AV-oracle observations");
}
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "expected topology-comparison CSV and captured AV metrics CSV");
        replay_topology(argv[1]);
        replay_av_capture(argv[2]);
        std::cout << "vbuf_numerical_contract_evidence_replay=PASS replayed_only=true\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "vbuf_numerical_contract_evidence_replay=FAIL: " << error.what() << '\n';
        return 1;
    }
}
