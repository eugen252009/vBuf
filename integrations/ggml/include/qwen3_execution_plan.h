#pragma once

#include "vbuf_numerical_contracts.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace vbuf_ggml {

class Qwen3Model;
class QwenCudaRuntimeState;
class QwenCudaSessionState;

enum class QwenExecutionPlanPath : uint8_t { SingleGpu, MultiGpu };
enum class QwenExecutionStageKind : uint8_t { Embedding, BlockRange, PinnedHostTransfer, FinalNorm, OutputHead };

struct QwenExecutionStage {
    QwenExecutionStageKind kind = QwenExecutionStageKind::Embedding;
    uint32_t device_id = 0;
    uint32_t first_block = 0;
    uint32_t block_count = 0;
    uint32_t transfer_from = 0;
    uint32_t transfer_to = 0;
    bool pinned_host_staging = false;
};

struct QwenExecutionPlanInput {
    std::string model_identity;
    std::string backend_family;
    std::vector<std::string> stable_device_identities;
    std::vector<std::optional<uint32_t>> stable_device_sm_versions;
    std::vector<uint32_t> block_device_ids;
    uint32_t embedding_device_id = 0;
    uint32_t final_norm_device_id = 0;
    uint32_t output_head_device_id = 0;
    uint32_t capacity = 0;
    uint32_t prefill_chunk_size = 32;
    std::string activation_dtype = "F32";
    std::string kv_dtype = "F16";
};

struct QwenExecutionPlan {
    QwenExecutionPlanPath path = QwenExecutionPlanPath::SingleGpu;
    std::string stable_id;
    std::string stable_identity;
    std::string model_identity;
    std::string backend_family;
    std::string placement_identity;
    std::string capacity_class;
    std::string activation_dtype;
    std::string kv_dtype;
    std::vector<std::string> stable_device_identities;
    std::vector<std::optional<uint32_t>> stable_device_sm_versions;
    std::vector<uint32_t> block_device_ids;
    std::vector<QwenExecutionStage> stages;
    uint32_t capacity = 0;
    uint32_t prefill_chunk_size = 32;
};

QwenExecutionPlan make_qwen_execution_plan(const QwenExecutionPlanInput & input);
QwenExecutionPlan build_qwen_execution_plan(const Qwen3Model & model,
    const QwenCudaRuntimeState & runtime, const QwenCudaSessionState & session,
    QwenExecutionPlanPath path);

enum class QwenExecutionPhase : uint8_t { Prefill, Decode };
struct QwenRuntimeDeviceFact {
    uint32_t logical_id = 0;
    std::string stable_identity;
    std::optional<uint32_t> sm_version;
};

struct QwenRuntimeFacts {
    uint32_t rows = 0;
    uint32_t context_length = 0;
    uint32_t capacity = 0;
    uint32_t prefill_chunk_size = 0;
    QwenExecutionPhase phase = QwenExecutionPhase::Prefill;
    std::string placement_identity;
    std::string activation_dtype;
    std::string kv_dtype;
    std::vector<QwenRuntimeDeviceFact> devices;
    std::optional<bool> all_finite;
    std::optional<double> tensor_min;
    std::optional<double> tensor_max;
};

QwenRuntimeFacts collect_qwen_runtime_facts(const QwenExecutionPlan & plan,
    const QwenCudaRuntimeState & runtime, uint32_t rows, uint32_t context_length,
    QwenExecutionPhase phase);

enum class QwenGuardField : uint8_t {
    Rows, ContextLength, Capacity, PrefillChunk, Phase, Placement, ActivationDtype, KvDtype,
    DeviceIdentity, DeviceSmVersion, TensorMin, TensorMax, AllFinite
};
enum class QwenGuardOperator : uint8_t {
    Equal, Less, LessEqual, Greater, GreaterEqual, StrictlyBetween, StringEqual, IsFinite
};

struct QwenExecutionGuard {
    QwenGuardField field = QwenGuardField::Rows;
    QwenGuardOperator op = QwenGuardOperator::Equal;
    uint64_t lower = 0;
    uint64_t upper = 0;
    uint32_t device_index = 0;
    std::string expected;
};

enum class QwenCandidateStatus : uint8_t { Candidate, Valid, Invalidated };
enum class QwenCandidateStrategy : uint8_t { PreboundDecodeDispatch, GuardProbe, NativeLayoutAttentionAV };
struct QwenExecutionCandidate {
    std::string identity;
    QwenCandidateStrategy strategy = QwenCandidateStrategy::PreboundDecodeDispatch;
    std::string base_plan_identity;
    std::string execution_plan_identity;
    std::vector<QwenExecutionGuard> guards;
    QwenCandidateStatus status = QwenCandidateStatus::Candidate;
    std::string validation_note;
    std::vector<vbuf_ml::numerics::NumericalContractRequirement> required_numerical_contracts;
    std::vector<vbuf_ml::numerics::NumericalEvaluation> numerical_qualifications;
};

enum class QwenOptimizerMode : uint8_t { Disabled, Shadow, Enabled };
enum class QwenOptimizerFault : uint8_t {
    None, CandidateConstruction, GuardEvaluation, CacheLookup, ProfilerRecord, CandidateExecution
};
enum class QwenOptimizerFallback : uint8_t {
    None, Disabled, NoCandidate, GuardFailed, UnsupportedGuard, Invalidated,
    CandidateMismatch, CandidateNotValidated, InternalFailure, NumericalQualificationMissing
};

struct QwenOptimizerDecision {
    QwenOptimizerMode mode = QwenOptimizerMode::Disabled;
    QwenOptimizerFallback fallback = QwenOptimizerFallback::Disabled;
    std::string canonical_plan_identity;
    std::string candidate_identity;
    bool candidate_found = false;
    bool cache_hit = false;
    bool guards_passed = false;
    bool candidate_eligible = false;
    bool candidate_validated = false;
    bool candidate_selected = false;
    bool qualification_trial = false;
    QwenCandidateStrategy strategy = QwenCandidateStrategy::PreboundDecodeDispatch;
    bool canonical_selected = true;
    vbuf_ml::numerics::EvidenceRejectionReason evidence_rejection =
        vbuf_ml::numerics::EvidenceRejectionReason::None;
};

struct QwenOptimizerProfileRecord {
    std::string plan_identity;
    QwenExecutionPhase phase = QwenExecutionPhase::Prefill;
    std::string rows_bucket;
    std::string context_bucket;
    uint64_t execution_count = 0;
    uint64_t cumulative_ns = 0;
    uint64_t last_seen_ordinal = 0;
};

struct QwenOptimizerSnapshot {
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t cache_evictions = 0;
    uint64_t guard_passes = 0;
    uint64_t guard_failures = 0;
    uint64_t optimizer_errors = 0;
    uint64_t observations = 0;
    size_t cache_size = 0;
    size_t candidate_count = 0;
    size_t valid_candidate_count = 0;
    size_t invalidated_candidate_count = 0;
    size_t profile_key_count = 0;
    QwenOptimizerDecision last_decision;
    std::vector<QwenOptimizerProfileRecord> profile_records;
};

// Shadow mode is observation-only and always selects canonical execution.
// Enabled may select only a guarded, explicitly Valid candidate.
class QwenExecutionPlanOptimizer final {
public:
    static constexpr size_t max_cached_candidates = 64;
    static constexpr size_t max_profile_keys = 128;
    static constexpr uint64_t min_candidate_hotness_observations = 32;

    QwenExecutionPlanOptimizer() = default;
    QwenExecutionPlanOptimizer(const QwenExecutionPlanOptimizer &) = delete;
    QwenExecutionPlanOptimizer & operator=(const QwenExecutionPlanOptimizer &) = delete;

    QwenOptimizerMode mode() const noexcept;
    void set_mode(QwenOptimizerMode mode) noexcept;
    QwenOptimizerDecision select(const QwenExecutionPlan & canonical,
        const QwenRuntimeFacts & facts, const std::string & requested_candidate_identity = {}) noexcept;
    void record_execution(const QwenExecutionPlan & canonical,
        const QwenRuntimeFacts & facts, uint64_t execution_count,
        uint64_t elapsed_ns) noexcept;
    void record_optimizer_failure() noexcept;
    bool register_candidate(QwenExecutionCandidate candidate) noexcept;
    bool register_candidate(QwenExecutionCandidate candidate, const QwenExecutionPlan & plan) noexcept;
    bool mark_candidate_valid(const std::string & identity, const std::string & validation_note,
        std::vector<vbuf_ml::numerics::NumericalEvaluation> numerical_qualifications,
        const QwenExecutionPlan & plan, const QwenRuntimeFacts & facts) noexcept;
    bool invalidate_candidate(const std::string & identity) noexcept;
    void set_unvalidated_trial_for_testing(bool enabled) noexcept;
    bool consume_candidate_execution_fault_for_testing() noexcept;
    void set_fault_for_testing(QwenOptimizerFault fault) noexcept;
    QwenOptimizerSnapshot snapshot() const;

private:
    bool store_candidate(QwenExecutionCandidate candidate) noexcept;
    mutable std::mutex mutex_;
    std::atomic<QwenOptimizerMode> mode_{QwenOptimizerMode::Shadow};
    QwenOptimizerFault test_fault_ = QwenOptimizerFault::None;
    bool allow_unvalidated_trial_for_testing_ = false;
    std::map<std::string, QwenExecutionCandidate> candidates_;
    std::map<std::string, QwenOptimizerProfileRecord> profiles_;
    uint64_t profile_ordinal_ = 0;
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t cache_evictions_ = 0;
    uint64_t guard_passes_ = 0;
    uint64_t guard_failures_ = 0;
    uint64_t optimizer_errors_ = 0;
    uint64_t observations_ = 0;
    QwenOptimizerDecision last_decision_;

    static std::optional<QwenExecutionCandidate> make_prebound_decode_candidate(const QwenExecutionPlan & plan);
    static std::string candidate_identity(const QwenExecutionPlan & plan);
    static bool evaluate_guard(const QwenExecutionGuard & guard, const QwenRuntimeFacts & facts,
        bool * supported);
};

} // namespace vbuf_ggml
