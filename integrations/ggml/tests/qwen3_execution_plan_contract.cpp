#include "qwen3_execution_plan.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace vbuf_ggml;
namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

QwenExecutionPlanInput multi_input() {
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
    input.capacity = 32768;
    input.prefill_chunk_size = 32;
    return input;
}

QwenRuntimeFacts decode_facts(const QwenExecutionPlan & plan, uint32_t context = 8192) {
    QwenRuntimeFacts facts;
    facts.rows = 1;
    facts.context_length = context;
    facts.capacity = plan.capacity;
    facts.prefill_chunk_size = plan.prefill_chunk_size;
    facts.phase = QwenExecutionPhase::Decode;
    facts.placement_identity = plan.placement_identity;
    facts.activation_dtype = plan.activation_dtype;
    facts.kv_dtype = plan.kv_dtype;
    for (uint32_t i = 0; i < plan.stable_device_identities.size(); ++i) {
        if (plan.stable_device_identities[i].empty()) continue;
        const auto sm = i < plan.stable_device_sm_versions.size() ? plan.stable_device_sm_versions[i] : std::nullopt;
        facts.devices.push_back({i, plan.stable_device_identities[i], sm});
    }
    return facts;
}

vbuf_ml::numerics::NumericalEvaluation make_output_evidence(const QwenExecutionPlan & plan,
        const QwenRuntimeFacts & facts, const std::string & implementation_identity,
        const std::string & operation) {
    using namespace vbuf_ml::numerics;
    const std::vector<float> reference_values{1.0f, 2.0f};
    const std::vector<float> candidate_values{1.0f, 2.0f};
    EvaluationContext context;
    context.operation = operation;
    context.output_name = operation;
    context.reference_kind = ReferenceKind::CanonicalExecution;
    context.reference_identity = "canonical-packed-v-v1:unit-fixture";
    context.model_identity = plan.model_identity;
    context.backend_family = plan.backend_family;
    context.implementation_identity = implementation_identity;
    context.device_family = "CUDA";
    context.placement_identity = plan.placement_identity;
    context.phase = "decode";
    context.execution_topology = "single-row-decode";
    context.fixture_identity = "qwen3-execution-plan-contract";
    context.input_identity = "same-input-fixture";
    context.output_dtype = facts.activation_dtype;
    context.input_dtype = facts.kv_dtype;
    context.output_shape = {2};
    context.capacity = facts.capacity;
    context.context_length = facts.context_length;
    context.rows = facts.rows;
    context.logical_inputs_equivalent = true;
    for (const auto & device : facts.devices) {
        context.device_identities.push_back(device.stable_identity);
        if (device.sm_version) context.device_sm_versions.push_back(*device.sm_version);
    }
    context.reference_tokens = std::vector<uint32_t>{7, 8};
    context.candidate_tokens = std::vector<uint32_t>{7, 8};
    const auto reference = make_tensor_view(reference_values, context.output_shape);
    const auto candidate = make_tensor_view(candidate_values, context.output_shape);
    const std::string contract = operation == "final_hidden" ?
        "qwen3.final_hidden.canonical_compatibility" : "qwen3.final_logits.canonical_compatibility";
    return evaluate_contract(contract, 1, &reference, &candidate, context);
}

QwenExecutionCandidate manual_candidate(const QwenExecutionPlan & plan, std::string suffix,
    QwenExecutionGuard guard) {
    QwenExecutionCandidate candidate;
    candidate.identity = plan.stable_identity + "|candidate=" + suffix;
    candidate.base_plan_identity = plan.stable_identity;
    candidate.execution_plan_identity = plan.stable_identity;
    candidate.strategy = QwenCandidateStrategy::GuardProbe;
    candidate.status = QwenCandidateStatus::Candidate;
    candidate.guards.push_back(std::move(guard));
    return candidate;
}

void test_plans() {
    auto multi = make_qwen_execution_plan(multi_input());
    require(multi.path == QwenExecutionPlanPath::MultiGpu, "26/14 plan was not classified as multi-GPU");
    require(multi.placement_identity == "multi:0x26,1x14;emb=0;norm=1;head=1",
        "26/14 placement identity is not canonical/stable");
    require(multi.stages.size() == 6, "26/14 plan stage count mismatch");
    require(multi.stages[0].kind == QwenExecutionStageKind::Embedding && multi.stages[0].device_id == 0,
        "embedding owner missing from plan");
    require(multi.stages[1].kind == QwenExecutionStageKind::BlockRange && multi.stages[1].first_block == 0 &&
        multi.stages[1].block_count == 26 && multi.stages[1].device_id == 0,
        "early block stage mismatch");
    require(multi.stages[2].kind == QwenExecutionStageKind::PinnedHostTransfer &&
        multi.stages[2].transfer_from == 0 && multi.stages[2].transfer_to == 1 &&
        multi.stages[2].pinned_host_staging, "staged boundary is not explicit in the plan");
    require(multi.stages[3].kind == QwenExecutionStageKind::BlockRange && multi.stages[3].first_block == 26 &&
        multi.stages[3].block_count == 14 && multi.stages[3].device_id == 1,
        "late block stage mismatch");
    require(multi.stages[4].kind == QwenExecutionStageKind::FinalNorm && multi.stages[4].device_id == 1 &&
        multi.stages[5].kind == QwenExecutionStageKind::OutputHead && multi.stages[5].device_id == 1,
        "final stages mismatch");
    const auto repeated = make_qwen_execution_plan(multi_input());
    require(repeated.stable_identity == multi.stable_identity && repeated.stable_id == multi.stable_id,
        "plan identity depends on process-local state");

    auto single_input = multi_input();
    single_input.backend_family = "GGML_CUDA";
    single_input.stable_device_identities.resize(1);
    single_input.stable_device_sm_versions.resize(1);
    single_input.block_device_ids.assign(40, 0);
    single_input.final_norm_device_id = 0;
    single_input.output_head_device_id = 0;
    auto single = make_qwen_execution_plan(single_input);
    require(single.path == QwenExecutionPlanPath::SingleGpu && single.stages.size() == 4,
        "single-GPU canonical plan was not represented");
    require(single.placement_identity.rfind("single:", 0) == 0,
        "single-GPU identity was incorrectly labeled multi-device");
    auto changed = multi_input();
    changed.prefill_chunk_size = 16;
    require(make_qwen_execution_plan(changed).stable_identity != multi.stable_identity,
        "chunk-size specialization is missing from the stable identity");
    changed = multi_input();
    changed.capacity = 16384;
    require(make_qwen_execution_plan(changed).stable_identity != multi.stable_identity,
        "capacity class is missing from the stable identity");
    changed = multi_input();
    changed.kv_dtype.clear();
    bool rejected = false;
    try { (void) make_qwen_execution_plan(changed); }
    catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "incomplete tensor dtype facts were accepted into a stable plan");
}

void test_candidate_is_bound_to_qualified_artifact_and_hardware() {
    const auto require_no_candidate = [](QwenExecutionPlanInput input, const char * message) {
        const auto plan = make_qwen_execution_plan(input);
        QwenExecutionPlanOptimizer optimizer;
        optimizer.set_mode(QwenOptimizerMode::Shadow);
        const auto decision = optimizer.select(plan, decode_facts(plan));
        require(!decision.candidate_found && decision.fallback == QwenOptimizerFallback::NoCandidate, message);
    };
    auto input = multi_input();
    input.model_identity = "sha256:other-model";
    require_no_candidate(input, "candidate was offered to a different semantic model");
    input = multi_input();
    input.backend_family = "GGML_CPU";
    require_no_candidate(input, "candidate was offered to a different backend");
    input = multi_input();
    input.stable_device_identities[0] = "CUDA0@0000:05:00.0";
    require_no_candidate(input, "candidate was offered to a different CUDA device identity");
    input = multi_input();
    input.stable_device_sm_versions[1] = 86;
    require_no_candidate(input, "candidate was offered to a different CUDA SM pair");
    input = multi_input();
    input.activation_dtype = "BF16";
    require_no_candidate(input, "candidate was offered to a different activation dtype");
    input = multi_input();
    input.kv_dtype = "F32";
    require_no_candidate(input, "candidate was offered to a different KV dtype");
    input = multi_input();
    input.capacity = 16384;
    require_no_candidate(input, "candidate was offered at a different capacity");
    input = multi_input();
    input.prefill_chunk_size = 16;
    require_no_candidate(input, "candidate was offered at a different chunk size");
    input = multi_input();
    input.block_device_ids[0] = 1;
    require_no_candidate(input, "candidate was offered at a different block placement");
}

void test_shadow_guards_cache_and_profiler() {
    const auto plan = make_qwen_execution_plan(multi_input());
    QwenExecutionPlanOptimizer optimizer;
    require(optimizer.mode() == QwenOptimizerMode::Shadow, "shadow mode is not the safe default");
    optimizer.set_mode(QwenOptimizerMode::Disabled);
    const auto disabled = optimizer.select(plan, decode_facts(plan));
    require(disabled.fallback == QwenOptimizerFallback::Disabled && disabled.canonical_selected,
        "disabled optimizer did not retain canonical selection");

    optimizer.set_mode(QwenOptimizerMode::Shadow);
    QwenRuntimeFacts prefill = decode_facts(plan);
    prefill.rows = 32;
    prefill.phase = QwenExecutionPhase::Prefill;
    auto decision = optimizer.select(plan, prefill);
    require(decision.candidate_found && !decision.cache_hit &&
        decision.fallback == QwenOptimizerFallback::GuardFailed && decision.canonical_selected,
        "prefill guard failure did not fall back to canonical");

    QwenRuntimeFacts wrong_phase = decode_facts(plan);
    wrong_phase.phase = QwenExecutionPhase::Prefill;
    const auto phase_mismatch = optimizer.select(plan, wrong_phase);
    require(phase_mismatch.fallback == QwenOptimizerFallback::GuardFailed &&
        phase_mismatch.canonical_selected, "phase guard failure did not fall back");

    const auto facts = decode_facts(plan);
    decision = optimizer.select(plan, facts);
    require(decision.candidate_found && decision.cache_hit && decision.guards_passed &&
        decision.candidate_eligible && decision.canonical_selected &&
        decision.fallback == QwenOptimizerFallback::None,
        "prebound decode candidate did not pass guards in shadow mode");
    const auto repeated = optimizer.select(plan, facts);
    require(repeated.cache_hit && repeated.candidate_eligible && repeated.canonical_selected,
        "shadow candidate cache hit did not retain canonical selection");
    const auto failed_context = optimizer.select(plan, decode_facts(plan, 31));
    require(failed_context.fallback == QwenOptimizerFallback::GuardFailed &&
        failed_context.canonical_selected, "lower context guard failure did not fall back");
    const auto context_32k = optimizer.select(plan, decode_facts(plan, 32767));
    require(context_32k.fallback == QwenOptimizerFallback::GuardFailed &&
        context_32k.canonical_selected, "upper context guard did not fail closed");

    optimizer.record_execution(plan, prefill, 256, 1000);
    optimizer.record_execution(plan, facts, 8, 2000);
    const std::string prebound_identity = decision.candidate_identity;
    require(!optimizer.mark_candidate_valid(prebound_identity, "premature validation"),
        "candidate became valid below the hotness threshold");
    optimizer.record_execution(plan, facts, 24, 3000);
    auto stats = optimizer.snapshot();
    require(stats.cache_size == 1 && stats.cache_hits >= 5 && stats.cache_misses == 1 &&
        stats.guard_passes == 2 && stats.guard_failures == 4,
        "shadow cache/guard counters mismatch");
    require(stats.observations == 288 && stats.profile_key_count == 2 && stats.profile_records.size() == 2,
        "bounded profiler did not record phase hotness");
    require(!optimizer.mark_candidate_valid(prebound_identity, "exact parity without numerical records"),
        "candidate became valid without numerical contract evidence");
    auto hidden_evidence = make_output_evidence(plan, facts,
        plan.stable_identity + "|strategy=prebound-decode-dispatch-v1", "final_hidden");
    auto logits_evidence = make_output_evidence(plan, facts,
        plan.stable_identity + "|strategy=prebound-decode-dispatch-v1", "final_logits");
    require(hidden_evidence.status == vbuf_ml::numerics::EvaluationStatus::Pass &&
        logits_evidence.status == vbuf_ml::numerics::EvaluationStatus::Pass,
        "test fixture failed to create active final-output contract records");
    require(optimizer.mark_candidate_valid(prebound_identity, "exact decode parity with active numerical contracts",
        {hidden_evidence, logits_evidence}),
        "candidate could not transition Candidate->Valid with hotness and both numerical records");
    optimizer.set_mode(QwenOptimizerMode::Enabled);
    const auto enabled = optimizer.select(plan, facts);
    require(enabled.candidate_found && enabled.candidate_validated && enabled.candidate_selected &&
        !enabled.canonical_selected && enabled.fallback == QwenOptimizerFallback::None,
        "ENABLED mode did not select the Valid guarded candidate");
    const auto unqualified_context = optimizer.select(plan, decode_facts(plan, 8193));
    require(unqualified_context.fallback == QwenOptimizerFallback::NumericalQualificationMissing &&
        unqualified_context.canonical_selected && !unqualified_context.candidate_eligible,
        "candidate numerical evidence was reused outside its exact qualified runtime context");
    optimizer.set_mode(QwenOptimizerMode::Shadow);

    auto unsupported = manual_candidate(plan, "prebound-decode-dispatch-v1",
        {QwenGuardField::TensorMin, QwenGuardOperator::Less, 1});
    require(optimizer.register_candidate(unsupported), "could not register test unsupported guard");
    const auto unsupported_result = optimizer.select(plan, facts);
    require(unsupported_result.fallback == QwenOptimizerFallback::UnsupportedGuard &&
        unsupported_result.canonical_selected, "unsupported fact did not fail closed");
    require(optimizer.invalidate_candidate(unsupported.identity), "candidate invalidation failed");
    const auto invalidated = optimizer.select(plan, facts);
    require(invalidated.fallback == QwenOptimizerFallback::Invalidated && invalidated.canonical_selected,
        "invalidated candidate remained selectable");

    auto range_candidate = manual_candidate(plan, "prebound-decode-dispatch-v1",
        {QwenGuardField::ContextLength, QwenGuardOperator::StrictlyBetween, 8191, 8193});
    require(optimizer.register_candidate(range_candidate), "could not register scalar range guard");
    require(optimizer.select(plan, facts).guards_passed,
        "strict lower/upper guard interval did not pass at its interior value");
    auto sm_candidate = manual_candidate(plan, "prebound-decode-dispatch-v1",
        {QwenGuardField::DeviceSmVersion, QwenGuardOperator::Equal, 75, 0, 1});
    require(optimizer.register_candidate(sm_candidate) && optimizer.select(plan, facts).guards_passed,
        "known device-SM guard did not evaluate deterministically");

    QwenExecutionPlanOptimizer bounded;
    for (size_t i = 0; i < QwenExecutionPlanOptimizer::max_cached_candidates + 4; ++i) {
        auto candidate = manual_candidate(plan, "bounded-" + std::to_string(i),
            {QwenGuardField::Rows, QwenGuardOperator::Equal, 1});
        require(bounded.register_candidate(std::move(candidate)), "bounded cache registration failed");
    }
    stats = bounded.snapshot();
    require(stats.cache_size == QwenExecutionPlanOptimizer::max_cached_candidates && stats.cache_evictions == 4,
        "plan cache is not bounded/deterministic");
}

void test_optimizer_faults_fail_closed() {
    const auto plan = make_qwen_execution_plan(multi_input());
    const auto facts = decode_facts(plan);
    for (QwenOptimizerFault fault : {QwenOptimizerFault::CandidateConstruction,
             QwenOptimizerFault::GuardEvaluation, QwenOptimizerFault::CacheLookup}) {
        QwenExecutionPlanOptimizer optimizer;
        optimizer.set_mode(QwenOptimizerMode::Shadow);
        optimizer.set_fault_for_testing(fault);
        const auto result = optimizer.select(plan, facts);
        require(result.fallback == QwenOptimizerFallback::InternalFailure && result.canonical_selected,
            "injected optimizer failure did not choose canonical fallback");
        require(optimizer.select(plan, facts).canonical_selected,
            "canonical selection did not recover after optimizer fault");
        require(optimizer.snapshot().optimizer_errors == 1, "optimizer fault was not counted");
    }

    QwenExecutionPlanOptimizer trial;
    trial.set_mode(QwenOptimizerMode::Enabled);
    const auto unvalidated = trial.select(plan, facts);
    require(unvalidated.fallback == QwenOptimizerFallback::CandidateNotValidated &&
        unvalidated.canonical_selected && !unvalidated.candidate_selected,
        "ENABLED mode selected an unvalidated candidate outside trial mode");
    trial.set_unvalidated_trial_for_testing(true);
    const auto trial_decision = trial.select(plan, facts);
    require(trial_decision.qualification_trial && trial_decision.candidate_selected &&
        !trial_decision.canonical_selected, "explicit qualification trial did not select the candidate");
    trial.set_unvalidated_trial_for_testing(false);
    trial.set_fault_for_testing(QwenOptimizerFault::CandidateExecution);
    require(trial.consume_candidate_execution_fault_for_testing() &&
        !trial.consume_candidate_execution_fault_for_testing(),
        "candidate execution fault was not one-shot");

    QwenExecutionPlanOptimizer forged;
    forged.set_mode(QwenOptimizerMode::Shadow);
    auto prevalidated_without_records = manual_candidate(plan, "forged-valid",
        {QwenGuardField::Rows, QwenGuardOperator::Equal, 1});
    prevalidated_without_records.status = QwenCandidateStatus::Valid;
    prevalidated_without_records.required_numerical_contracts = {
        "qwen3.final_hidden.canonical_compatibility",
        "qwen3.final_logits.canonical_compatibility",
    };
    require(forged.register_candidate(prevalidated_without_records),
        "could not register fail-closed numerical-qualification probe");
    const auto missing_evidence = forged.select(plan, facts, prevalidated_without_records.identity);
    require(missing_evidence.fallback == QwenOptimizerFallback::NumericalQualificationMissing &&
        missing_evidence.canonical_selected && !missing_evidence.candidate_eligible,
        "directly registered VALID candidate bypassed numerical qualification records");

    QwenExecutionPlanOptimizer unannotated;
    unannotated.set_mode(QwenOptimizerMode::Enabled);
    auto executable_without_requirements = manual_candidate(plan, "prebound-decode-dispatch-v1",
        {QwenGuardField::Rows, QwenGuardOperator::Equal, 1});
    executable_without_requirements.strategy = QwenCandidateStrategy::PreboundDecodeDispatch;
    executable_without_requirements.execution_plan_identity =
        plan.stable_identity + "|strategy=prebound-decode-dispatch-v1";
    executable_without_requirements.status = QwenCandidateStatus::Valid;
    executable_without_requirements.required_numerical_contracts.clear();
    require(unannotated.register_candidate(executable_without_requirements),
        "could not register unannotated executable candidate probe");
    const auto auto_required = unannotated.select(plan, facts, executable_without_requirements.identity);
    require(auto_required.fallback == QwenOptimizerFallback::NumericalQualificationMissing &&
        auto_required.canonical_selected,
        "executable candidate without declared contract IDs bypassed default model-output requirements");

    QwenExecutionPlanOptimizer profiler;
    profiler.set_mode(QwenOptimizerMode::Shadow);
    profiler.set_fault_for_testing(QwenOptimizerFault::ProfilerRecord);
    profiler.record_execution(plan, facts, 1, 50);
    require(profiler.snapshot().optimizer_errors == 1 && profiler.snapshot().observations == 0,
        "profiler failure escaped fail-closed accounting");
}
} // namespace

int main() {
    try {
        test_plans();
        test_candidate_is_bound_to_qualified_artifact_and_hardware();
        test_shadow_guards_cache_and_profiler();
        test_optimizer_faults_fail_closed();
        std::cout << "qwen3_execution_plan_contract=PASS canonical=single+26/14 shadow=canonical "
            "enabled=validated-prebound-dispatch guards=fail-closed lifecycle=PASS cache=bounded "
            "invalidation=PASS profiler=bounded optimizer-failures=PASS\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen3_execution_plan_contract=FAIL: " << error.what() << '\n';
        return 1;
    }
}
