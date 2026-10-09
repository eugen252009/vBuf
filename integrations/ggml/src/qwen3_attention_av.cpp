#include "qwen3_attention_av.h"

#include "qwen3_execution_plan.h"

#include <stdexcept>

namespace vbuf_ggml {
namespace {
constexpr uint32_t max_native_av_capacity = 512;
}

Qwen3AttentionAVResolution resolve_qwen3_attention_av(
        const Qwen3AttentionAVFacts & facts, bool request_native) noexcept {
    Qwen3AttentionAVResolution result;
    result.native_requested = request_native;
    if (!request_native) return result;
    if (facts.sm_version != 86) {
        result.reason = "native AV is qualified only for SM 8.6; use packed canonical AV";
        return result;
    }
    if (facts.value_type != GGML_TYPE_F16 || facts.probability_type != GGML_TYPE_F32 ||
        facts.position_type != GGML_TYPE_I32) {
        result.reason = "native AV requires F16 V, F32 probabilities, and I32 positions";
        return result;
    }
    if (facts.kv_heads != 8 || facts.query_heads != 40 || facts.head_dimension != 128 ||
        facts.query_heads % facts.kv_heads != 0) {
        result.reason = "native AV requires Qwen3-14B GQA geometry 40 query heads / 8 KV heads / 128 dimensions";
        return result;
    }
    if (facts.query_rows != 1 && facts.query_rows != 32) {
        result.reason = "native AV supports only one-row decode or 32-row prefill";
        return result;
    }
    if (facts.capacity == 0 || facts.capacity > max_native_av_capacity || facts.visible_context == 0 ||
        facts.visible_context > facts.capacity || facts.query_rows > facts.visible_context) {
        result.reason = facts.capacity > max_native_av_capacity ?
            "native AV is not numerically qualified above capacity 512; use packed canonical AV" :
            "native AV context/capacity is outside the checked range";
        return result;
    }
    if (!facts.canonical_position_major_v) {
        result.reason = "native AV source is not the canonical position-major V cache";
        return result;
    }
    result.path = Qwen3AttentionAVPath::NativeLayout;
    result.native_supported = true;
    result.reason = "native-layout AV capability matched";
    return result;
}

std::string qwen3_native_attention_av_candidate_identity(
        const QwenExecutionPlan & plan, QwenExecutionPhase phase) {
    return plan.stable_identity + "|candidate=native-layout-av-v1|capacity=" +
        std::to_string(plan.capacity) + "|phase=" +
        (phase == QwenExecutionPhase::Prefill ? "prefill" : "decode");
}

QwenExecutionCandidate make_qwen3_native_attention_av_candidate(
        const QwenExecutionPlan & plan, QwenExecutionPhase phase) {
    constexpr const char * qualified_model_identity =
        "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
    constexpr const char * expected_placement = "multi:0x26,1x14;emb=0;norm=1;head=1";
    if (plan.path != QwenExecutionPlanPath::MultiGpu || plan.model_identity != qualified_model_identity ||
        plan.backend_family != "GGML_CUDA" || plan.capacity == 0 || plan.capacity > max_native_av_capacity ||
        plan.prefill_chunk_size != 32 || plan.placement_identity != expected_placement ||
        plan.activation_dtype != "F32" || plan.kv_dtype != "F16" ||
        plan.stable_device_identities.size() < 2 || plan.stable_device_sm_versions.size() < 2 ||
        plan.stable_device_identities[0] != "CUDA0@0000:04:00.0" ||
        plan.stable_device_identities[1] != "CUDA1@0000:07:00.0" ||
        plan.stable_device_sm_versions[0] != 86 || plan.stable_device_sm_versions[1] != 75 ||
        (phase != QwenExecutionPhase::Decode && phase != QwenExecutionPhase::Prefill)) {
        throw std::invalid_argument("native AV candidate requires the qualified Qwen3 26/14 CUDA plan");
    }
    QwenExecutionCandidate candidate;
    candidate.identity = qwen3_native_attention_av_candidate_identity(plan, phase);
    candidate.base_plan_identity = plan.stable_identity;
    candidate.execution_plan_identity = plan.stable_identity +
        "|strategy=native-layout-av-v1|capacity=" + std::to_string(plan.capacity) +
        (phase == QwenExecutionPhase::Prefill ? "|phase=prefill" : "|phase=decode");
    candidate.strategy = QwenCandidateStrategy::NativeLayoutAttentionAV;
    candidate.status = QwenCandidateStatus::Candidate;
    candidate.validation_note = "awaiting canonical/native AV parity qualification";
    candidate.required_numerical_contracts = {
        "qwen3.final_hidden.canonical_compatibility",
        "qwen3.final_logits.canonical_compatibility",
    };
    const uint32_t rows = phase == QwenExecutionPhase::Prefill ? plan.prefill_chunk_size : 1;
    candidate.guards = {
        {QwenGuardField::Phase, QwenGuardOperator::Equal, static_cast<uint8_t>(phase)},
        {QwenGuardField::Rows, QwenGuardOperator::Equal, rows},
        {QwenGuardField::ContextLength,
            phase == QwenExecutionPhase::Prefill ? QwenGuardOperator::Equal : QwenGuardOperator::LessEqual,
            phase == QwenExecutionPhase::Prefill ? 0U : 32U},
        {QwenGuardField::Capacity, QwenGuardOperator::Equal, plan.capacity},
        {QwenGuardField::PrefillChunk, QwenGuardOperator::Equal, plan.prefill_chunk_size},
        {QwenGuardField::Placement, QwenGuardOperator::StringEqual, 0, 0, 0, plan.placement_identity},
        {QwenGuardField::ActivationDtype, QwenGuardOperator::StringEqual, 0, 0, 0, plan.activation_dtype},
        {QwenGuardField::KvDtype, QwenGuardOperator::StringEqual, 0, 0, 0, plan.kv_dtype},
        {QwenGuardField::DeviceIdentity, QwenGuardOperator::StringEqual, 0, 0, 0,
            plan.stable_device_identities[0]},
        {QwenGuardField::DeviceIdentity, QwenGuardOperator::StringEqual, 0, 0, 1,
            plan.stable_device_identities[1]},
        {QwenGuardField::DeviceSmVersion, QwenGuardOperator::Equal, 86, 0, 0},
        {QwenGuardField::DeviceSmVersion, QwenGuardOperator::Equal, 75, 0, 1},
    };
    return candidate;
}

} // namespace vbuf_ggml
