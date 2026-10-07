#include "qwen3_execution_plan.h"

#include "qwen3_cuda_core.h"
#include "qwen3_model.h"

#if defined(VBUF_QWEN3_HAS_CUDA_RUNTIME)
#include <cuda_runtime_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <utility>
#include <stdexcept>

namespace vbuf_ggml {
namespace {
constexpr uint32_t qwen_layer_count = 40;

uint64_t fnv1a(const std::string & value) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

void append_field(std::ostringstream & out, const std::string & name, const std::string & value) {
    out << name << value.size() << ':' << value << ';';
}

std::string capacity_class(uint32_t capacity) {
    if (capacity <= 2048) return "le-2048";
    if (capacity <= 4096) return "le-4096";
    if (capacity <= 8192) return "le-8192";
    if (capacity <= 16384) return "le-16384";
    if (capacity <= 32768) return "le-32768";
    return "gt-32768";
}

std::string placement_identity(const std::vector<uint32_t> & blocks, bool multi,
    uint32_t embedding, uint32_t final_norm, uint32_t output_head) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << (multi ? "multi:" : "single:");
    if (blocks.empty()) return out.str() + "empty";
    size_t begin = 0;
    while (begin < blocks.size()) {
        size_t end = begin + 1;
        while (end < blocks.size() && blocks[end] == blocks[begin]) ++end;
        if (begin != 0) out << ',';
        out << blocks[begin] << 'x' << (end - begin);
        begin = end;
    }
    out << ";emb=" << embedding << ";norm=" << final_norm << ";head=" << output_head;
    return out.str();
}

std::string rows_bucket(uint32_t rows) {
    if (rows <= 1) return "1";
    if (rows <= 16) return "2-16";
    if (rows <= 32) return "17-32";
    return "gt-32";
}

std::string context_bucket(uint32_t context) {
    if (context < 2048) return "lt-2k";
    if (context < 8192) return "2k-8k";
    if (context < 16384) return "8k-16k";
    if (context < 32768) return "16k-32k";
    return "ge-32k";
}

uint64_t saturated_add(uint64_t left, uint64_t right) {
    return right > std::numeric_limits<uint64_t>::max() - left ?
        std::numeric_limits<uint64_t>::max() : left + right;
}

std::optional<double> numeric_fact(QwenGuardField field, const QwenRuntimeFacts & facts) {
    switch (field) {
        case QwenGuardField::Rows: return facts.rows;
        case QwenGuardField::ContextLength: return facts.context_length;
        case QwenGuardField::Capacity: return facts.capacity;
        case QwenGuardField::PrefillChunk: return facts.prefill_chunk_size;
        case QwenGuardField::Phase: return static_cast<uint8_t>(facts.phase);
        case QwenGuardField::TensorMin: return facts.tensor_min;
        case QwenGuardField::TensorMax: return facts.tensor_max;
        default: return std::nullopt;
    }
}
} // namespace

QwenExecutionPlan make_qwen_execution_plan(const QwenExecutionPlanInput & input) {
    if (input.model_identity.empty() || input.backend_family.empty() || input.activation_dtype.empty() ||
        input.kv_dtype.empty() || input.capacity == 0 || input.prefill_chunk_size == 0 ||
        input.block_device_ids.size() != qwen_layer_count || input.stable_device_identities.empty() ||
        input.stable_device_sm_versions.size() > input.stable_device_identities.size())
        throw std::invalid_argument("canonical Qwen plan input is incomplete");
    if (input.embedding_device_id >= input.stable_device_identities.size() ||
        input.final_norm_device_id >= input.stable_device_identities.size() ||
        input.output_head_device_id >= input.stable_device_identities.size() ||
        input.stable_device_identities[input.embedding_device_id].empty() ||
        input.stable_device_identities[input.final_norm_device_id].empty() ||
        input.stable_device_identities[input.output_head_device_id].empty())
        throw std::invalid_argument("canonical Qwen plan references an unknown device");
    for (uint32_t owner : input.block_device_ids)
        if (owner >= input.stable_device_identities.size() || input.stable_device_identities[owner].empty())
            throw std::invalid_argument("canonical Qwen block references an unknown device");

    QwenExecutionPlan plan;
    plan.model_identity = input.model_identity;
    plan.backend_family = input.backend_family;
    plan.stable_device_identities = input.stable_device_identities;
    plan.stable_device_sm_versions = input.stable_device_sm_versions;
    if (plan.stable_device_sm_versions.size() < plan.stable_device_identities.size())
        plan.stable_device_sm_versions.resize(plan.stable_device_identities.size());
    plan.block_device_ids = input.block_device_ids;
    plan.capacity = input.capacity;
    plan.prefill_chunk_size = input.prefill_chunk_size;
    plan.capacity_class = capacity_class(input.capacity);
    plan.activation_dtype = input.activation_dtype;
    plan.kv_dtype = input.kv_dtype;
    std::set<uint32_t> owners(input.block_device_ids.begin(), input.block_device_ids.end());
    owners.insert(input.embedding_device_id);
    owners.insert(input.final_norm_device_id);
    owners.insert(input.output_head_device_id);
    plan.path = owners.size() == 1 ? QwenExecutionPlanPath::SingleGpu : QwenExecutionPlanPath::MultiGpu;
    const bool multi = plan.path == QwenExecutionPlanPath::MultiGpu;
    plan.placement_identity = placement_identity(input.block_device_ids, multi, input.embedding_device_id,
        input.final_norm_device_id, input.output_head_device_id);
    plan.stages.push_back({QwenExecutionStageKind::Embedding, input.embedding_device_id});
    if (input.embedding_device_id != input.block_device_ids.front())
        plan.stages.push_back({QwenExecutionStageKind::PinnedHostTransfer, 0, 0, 0,
            input.embedding_device_id, input.block_device_ids.front(), true});
    uint32_t block = 0;
    while (block < qwen_layer_count) {
        const uint32_t owner = input.block_device_ids[block];
        uint32_t end = block + 1;
        while (end < qwen_layer_count && input.block_device_ids[end] == owner) ++end;
        if (block != 0) {
            const uint32_t previous_owner = input.block_device_ids[block - 1];
            plan.stages.push_back({QwenExecutionStageKind::PinnedHostTransfer, 0, 0, 0,
                previous_owner, owner, true});
        }
        plan.stages.push_back({QwenExecutionStageKind::BlockRange, owner, block, end - block});
        block = end;
    }
    const uint32_t final_block_owner = input.block_device_ids.back();
    if (final_block_owner != input.final_norm_device_id)
        plan.stages.push_back({QwenExecutionStageKind::PinnedHostTransfer, 0, 0, 0,
            final_block_owner, input.final_norm_device_id, true});
    plan.stages.push_back({QwenExecutionStageKind::FinalNorm, input.final_norm_device_id});
    if (input.final_norm_device_id != input.output_head_device_id)
        plan.stages.push_back({QwenExecutionStageKind::PinnedHostTransfer, 0, 0, 0,
            input.final_norm_device_id, input.output_head_device_id, true});
    plan.stages.push_back({QwenExecutionStageKind::OutputHead, input.output_head_device_id});

    std::ostringstream descriptor;
    descriptor.imbue(std::locale::classic());
    descriptor << "qwen3-execution-plan-v1;";
    append_field(descriptor, "model=", input.model_identity);
    append_field(descriptor, "backend=", input.backend_family);
    append_field(descriptor, "placement=", plan.placement_identity);
    append_field(descriptor, "capacity-class=", plan.capacity_class);
    append_field(descriptor, "chunk=", std::to_string(input.prefill_chunk_size));
    append_field(descriptor, "activation=", input.activation_dtype);
    append_field(descriptor, "kv=", input.kv_dtype);
    for (size_t i = 0; i < input.stable_device_identities.size(); ++i) {
        append_field(descriptor, "device" + std::to_string(i) + '=', input.stable_device_identities[i]);
        append_field(descriptor, "sm" + std::to_string(i) + '=', plan.stable_device_sm_versions[i] ?
            std::to_string(*plan.stable_device_sm_versions[i]) : std::string("unknown"));
    }
    plan.stable_identity = descriptor.str();
    plan.stable_id = "qwen3-plan-" + hex64(fnv1a(plan.stable_identity));
    return plan;
}

QwenExecutionPlan build_qwen_execution_plan(const Qwen3Model & model,
    const QwenCudaRuntimeState & runtime, const QwenCudaSessionState & session,
    QwenExecutionPlanPath path) {
    QwenExecutionPlanInput input;
    input.model_identity = runtime.artifact_identity();
    input.backend_family = "GGML_CUDA";
    input.block_device_ids = runtime.placement().block_device_ids;
    input.embedding_device_id = runtime.placement().embedding_device_id;
    input.final_norm_device_id = runtime.placement().output_norm_device_id;
    input.output_head_device_id = runtime.placement().output_head_device_id;
    input.capacity = session.capacity();
    input.prefill_chunk_size = runtime.prefill_chunk_size();
    const auto device_ids = runtime.placement().device_ids();
    if (device_ids.empty()) throw std::invalid_argument("Qwen plan runtime has no devices");
    const size_t topology_size = static_cast<size_t>(*std::max_element(device_ids.begin(), device_ids.end())) + 1;
    input.stable_device_identities.resize(topology_size);
    input.stable_device_sm_versions.resize(topology_size);
    for (uint32_t id : device_ids) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(runtime.device(id), &props);
        const std::string name = props.name != nullptr ? props.name : ggml_backend_dev_name(runtime.device(id));
        const std::string physical = props.device_id != nullptr ? props.device_id : std::string("unknown-pci");
        if (id >= input.stable_device_identities.size())
            throw std::invalid_argument("Qwen plan device IDs are not a dense runtime index");
        input.stable_device_identities[id] = name + "@" + physical;
#if defined(VBUF_QWEN3_HAS_CUDA_RUNTIME)
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, static_cast<int>(id)) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, static_cast<int>(id)) == cudaSuccess)
            input.stable_device_sm_versions[id] = static_cast<uint32_t>(major * 10 + minor);
#endif
    }
    auto plan = make_qwen_execution_plan(input);
    if (path != plan.path || model.artifact_identity != input.model_identity)
        throw std::invalid_argument("requested canonical plan path/model does not match runtime ownership");
    return plan;
}

QwenRuntimeFacts collect_qwen_runtime_facts(const QwenExecutionPlan & plan,
    const QwenCudaRuntimeState & runtime, uint32_t rows, uint32_t context_length,
    QwenExecutionPhase phase) {
    QwenRuntimeFacts facts;
    facts.rows = rows;
    facts.context_length = context_length;
    facts.capacity = plan.capacity;
    facts.prefill_chunk_size = plan.prefill_chunk_size;
    facts.phase = phase;
    facts.placement_identity = plan.placement_identity;
    facts.activation_dtype = plan.activation_dtype;
    facts.kv_dtype = plan.kv_dtype;
    for (uint32_t id : runtime.placement().device_ids()) {
        if (id >= plan.stable_device_identities.size())
            throw std::invalid_argument("Qwen runtime facts reference a device outside the canonical plan");
        const auto sm = id < plan.stable_device_sm_versions.size() ? plan.stable_device_sm_versions[id] : std::nullopt;
        facts.devices.push_back({id, plan.stable_device_identities[id], sm});
    }
    return facts;
}

QwenOptimizerMode QwenExecutionPlanOptimizer::mode() const noexcept {
    return mode_.load(std::memory_order_relaxed);
}

void QwenExecutionPlanOptimizer::set_mode(QwenOptimizerMode value) noexcept {
    mode_.store(value, std::memory_order_relaxed);
}

std::string QwenExecutionPlanOptimizer::candidate_identity(const QwenExecutionPlan & plan) {
    return plan.stable_identity + "|candidate=prebound-decode-dispatch-v1";
}

std::optional<QwenExecutionCandidate> QwenExecutionPlanOptimizer::make_prebound_decode_candidate(
    const QwenExecutionPlan & plan) {
    constexpr const char * qualified_model_identity =
        "sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31";
    if (plan.path != QwenExecutionPlanPath::MultiGpu || plan.model_identity != qualified_model_identity ||
        plan.backend_family != "GGML_CUDA" || plan.capacity != 32768 || plan.prefill_chunk_size != 32 ||
        plan.placement_identity != "multi:0x26,1x14;emb=0;norm=1;head=1" ||
        plan.activation_dtype != "F32" || plan.kv_dtype != "F16" ||
        plan.stable_device_identities.size() != 2 || plan.stable_device_sm_versions.size() != 2 ||
        plan.stable_device_identities[0] != "CUDA0@0000:04:00.0" ||
        plan.stable_device_identities[1] != "CUDA1@0000:07:00.0" ||
        plan.stable_device_sm_versions[0] != 86 || plan.stable_device_sm_versions[1] != 75)
        return std::nullopt;
    QwenExecutionCandidate candidate;
    candidate.identity = candidate_identity(plan);
    candidate.base_plan_identity = plan.stable_identity;
    candidate.execution_plan_identity = plan.stable_identity + "|strategy=prebound-decode-dispatch-v1";
    candidate.strategy = QwenCandidateStrategy::PreboundDecodeDispatch;
    candidate.status = QwenCandidateStatus::Candidate;
    candidate.validation_note = "awaiting canonical/optimized exact decode comparison";
    candidate.guards = {
        {QwenGuardField::Phase, QwenGuardOperator::Equal, static_cast<uint8_t>(QwenExecutionPhase::Decode)},
        {QwenGuardField::Rows, QwenGuardOperator::Equal, 1},
        {QwenGuardField::ContextLength, QwenGuardOperator::GreaterEqual, 32},
        {QwenGuardField::ContextLength, QwenGuardOperator::Less, 32767},
        {QwenGuardField::Capacity, QwenGuardOperator::Equal, 32768},
        {QwenGuardField::PrefillChunk, QwenGuardOperator::Equal, 32},
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

bool QwenExecutionPlanOptimizer::evaluate_guard(const QwenExecutionGuard & guard,
    const QwenRuntimeFacts & facts, bool * supported) {
    if (supported != nullptr) *supported = true;
    if (guard.op == QwenGuardOperator::StringEqual) {
        const std::string * value = nullptr;
        switch (guard.field) {
            case QwenGuardField::Placement: value = &facts.placement_identity; break;
            case QwenGuardField::ActivationDtype: value = &facts.activation_dtype; break;
            case QwenGuardField::KvDtype: value = &facts.kv_dtype; break;
            case QwenGuardField::DeviceIdentity:
                if (guard.device_index < facts.devices.size()) value = &facts.devices[guard.device_index].stable_identity;
                break;
            default: break;
        }
        if (value == nullptr) {
            if (supported != nullptr) *supported = false;
            return false;
        }
        return *value == guard.expected;
    }
    if (guard.field == QwenGuardField::AllFinite && guard.op == QwenGuardOperator::IsFinite) {
        if (!facts.all_finite) {
            if (supported != nullptr) *supported = false;
            return false;
        }
        return *facts.all_finite == (guard.expected != "false");
    }
    if (guard.field == QwenGuardField::DeviceSmVersion) {
        if (guard.device_index >= facts.devices.size() || !facts.devices[guard.device_index].sm_version) {
            if (supported != nullptr) *supported = false;
            return false;
        }
        const double value = *facts.devices[guard.device_index].sm_version;
        switch (guard.op) {
            case QwenGuardOperator::Equal: return value == static_cast<double>(guard.lower);
            case QwenGuardOperator::Less: return value < static_cast<double>(guard.lower);
            case QwenGuardOperator::LessEqual: return value <= static_cast<double>(guard.lower);
            case QwenGuardOperator::Greater: return value > static_cast<double>(guard.lower);
            case QwenGuardOperator::GreaterEqual: return value >= static_cast<double>(guard.lower);
            case QwenGuardOperator::StrictlyBetween:
                return value > static_cast<double>(guard.lower) && value < static_cast<double>(guard.upper);
            default:
                if (supported != nullptr) *supported = false;
                return false;
        }
    }
    const auto value = numeric_fact(guard.field, facts);
    if (!value) {
        if (supported != nullptr) *supported = false;
        return false;
    }
    switch (guard.op) {
        case QwenGuardOperator::Equal: return *value == static_cast<double>(guard.lower);
        case QwenGuardOperator::Less: return *value < static_cast<double>(guard.lower);
        case QwenGuardOperator::LessEqual: return *value <= static_cast<double>(guard.lower);
        case QwenGuardOperator::Greater: return *value > static_cast<double>(guard.lower);
        case QwenGuardOperator::GreaterEqual: return *value >= static_cast<double>(guard.lower);
        case QwenGuardOperator::StrictlyBetween:
            return *value > static_cast<double>(guard.lower) && *value < static_cast<double>(guard.upper);
        default:
            if (supported != nullptr) *supported = false;
            return false;
    }
}

QwenOptimizerDecision QwenExecutionPlanOptimizer::select(const QwenExecutionPlan & canonical,
    const QwenRuntimeFacts & facts) noexcept {
    QwenOptimizerDecision decision;
    decision.mode = mode();
    try {
        decision.canonical_plan_identity = canonical.stable_identity;
    } catch (...) {
        decision.fallback = QwenOptimizerFallback::InternalFailure;
        try { std::lock_guard<std::mutex> lock(mutex_); ++optimizer_errors_; last_decision_ = decision; } catch (...) {}
        return decision;
    }
    if (decision.mode == QwenOptimizerMode::Disabled) {
        decision.fallback = QwenOptimizerFallback::Disabled;
        try { std::lock_guard<std::mutex> lock(mutex_); last_decision_ = decision; } catch (...) {}
        return decision;
    }
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (test_fault_ == QwenOptimizerFault::CacheLookup) {
            test_fault_ = QwenOptimizerFault::None;
            throw std::runtime_error("injected optimizer cache lookup failure");
        }
        const std::string key = candidate_identity(canonical);
        auto found = candidates_.find(key);
        QwenExecutionCandidate candidate;
        if (found != candidates_.end()) {
            ++cache_hits_;
            decision.cache_hit = true;
            candidate = found->second;
        } else {
            ++cache_misses_;
            if (test_fault_ == QwenOptimizerFault::CandidateConstruction) {
                test_fault_ = QwenOptimizerFault::None;
                throw std::runtime_error("injected optimizer candidate construction failure");
            }
            auto made = make_prebound_decode_candidate(canonical);
            if (!made) {
                decision.fallback = QwenOptimizerFallback::NoCandidate;
                last_decision_ = decision;
                return decision;
            }
            if (candidates_.size() >= max_cached_candidates) {
                candidates_.erase(candidates_.begin());
                ++cache_evictions_;
            }
            found = candidates_.emplace(made->identity, *made).first;
            candidate = found->second;
        }
        decision.candidate_found = true;
        decision.candidate_identity = candidate.identity;
        if (candidate.status == QwenCandidateStatus::Invalidated) {
            decision.fallback = QwenOptimizerFallback::Invalidated;
            last_decision_ = decision;
            return decision;
        }
        const bool strategy_identity_matches =
            (candidate.strategy == QwenCandidateStrategy::PreboundDecodeDispatch &&
                candidate.execution_plan_identity == canonical.stable_identity + "|strategy=prebound-decode-dispatch-v1") ||
            (candidate.strategy == QwenCandidateStrategy::GuardProbe &&
                candidate.execution_plan_identity == canonical.stable_identity);
        if ((candidate.status != QwenCandidateStatus::Candidate && candidate.status != QwenCandidateStatus::Valid) ||
            candidate.base_plan_identity != canonical.stable_identity || !strategy_identity_matches) {
            decision.fallback = QwenOptimizerFallback::CandidateMismatch;
            last_decision_ = decision;
            return decision;
        }
        decision.strategy = candidate.strategy;
        decision.candidate_validated = candidate.status == QwenCandidateStatus::Valid;
        if (test_fault_ == QwenOptimizerFault::GuardEvaluation) {
            test_fault_ = QwenOptimizerFault::None;
            throw std::runtime_error("injected optimizer guard evaluation failure");
        }
        for (const auto & guard : candidate.guards) {
            bool supported = false;
            const bool pass = evaluate_guard(guard, facts, &supported);
            if (!supported) {
                ++guard_failures_;
                decision.fallback = QwenOptimizerFallback::UnsupportedGuard;
                last_decision_ = decision;
                return decision;
            }
            if (!pass) {
                ++guard_failures_;
                decision.fallback = QwenOptimizerFallback::GuardFailed;
                last_decision_ = decision;
                return decision;
            }
        }
        ++guard_passes_;
        decision.guards_passed = true;
        decision.candidate_eligible = true;
        if (decision.mode == QwenOptimizerMode::Enabled) {
            if (candidate.strategy != QwenCandidateStrategy::PreboundDecodeDispatch) {
                decision.candidate_eligible = false;
                decision.fallback = QwenOptimizerFallback::CandidateMismatch;
                last_decision_ = decision;
                return decision;
            }
            if (candidate.status == QwenCandidateStatus::Valid || allow_unvalidated_trial_for_testing_) {
                decision.candidate_selected = true;
                decision.qualification_trial = candidate.status != QwenCandidateStatus::Valid;
                decision.canonical_selected = false;
            } else {
                decision.candidate_eligible = false;
                decision.fallback = QwenOptimizerFallback::CandidateNotValidated;
                last_decision_ = decision;
                return decision;
            }
        }
        decision.fallback = QwenOptimizerFallback::None;
        last_decision_ = decision;
        return decision;
    } catch (...) {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            ++optimizer_errors_;
            decision.fallback = QwenOptimizerFallback::InternalFailure;
            decision.canonical_selected = true;
            last_decision_ = decision;
        } catch (...) {
            decision.fallback = QwenOptimizerFallback::InternalFailure;
        }
        return decision;
    }
}

void QwenExecutionPlanOptimizer::record_optimizer_failure() noexcept {
    try { std::lock_guard<std::mutex> lock(mutex_); ++optimizer_errors_; } catch (...) {}
}

void QwenExecutionPlanOptimizer::record_execution(const QwenExecutionPlan & canonical,
    const QwenRuntimeFacts & facts, uint64_t execution_count, uint64_t elapsed_ns) noexcept {
    if (mode() != QwenOptimizerMode::Shadow || execution_count == 0) return;
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (test_fault_ == QwenOptimizerFault::ProfilerRecord) {
            test_fault_ = QwenOptimizerFault::None;
            throw std::runtime_error("injected optimizer profiler failure");
        }
        const std::string phase = facts.phase == QwenExecutionPhase::Prefill ? "prefill" : "decode";
        const std::string key = canonical.stable_identity + '|' + phase + '|' +
            rows_bucket(facts.rows) + '|' + context_bucket(facts.context_length);
        auto found = profiles_.find(key);
        if (found == profiles_.end()) {
            if (profiles_.size() >= max_profile_keys) {
                auto victim = std::min_element(profiles_.begin(), profiles_.end(),
                    [](const auto & a, const auto & b) {
                        if (a.second.last_seen_ordinal != b.second.last_seen_ordinal)
                            return a.second.last_seen_ordinal < b.second.last_seen_ordinal;
                        return a.first < b.first;
                    });
                if (victim != profiles_.end()) profiles_.erase(victim);
            }
            QwenOptimizerProfileRecord record;
            record.plan_identity = canonical.stable_identity;
            record.phase = facts.phase;
            record.rows_bucket = rows_bucket(facts.rows);
            record.context_bucket = context_bucket(facts.context_length);
            found = profiles_.emplace(key, std::move(record)).first;
        }
        found->second.execution_count = saturated_add(found->second.execution_count, execution_count);
        found->second.cumulative_ns = saturated_add(found->second.cumulative_ns, elapsed_ns);
        found->second.last_seen_ordinal = ++profile_ordinal_;
        observations_ = saturated_add(observations_, execution_count);
    } catch (...) {
        try { std::lock_guard<std::mutex> lock(mutex_); ++optimizer_errors_; } catch (...) {}
    }
}

bool QwenExecutionPlanOptimizer::register_candidate(QwenExecutionCandidate candidate) noexcept {
    try {
        if (candidate.identity.empty() || candidate.base_plan_identity.empty() ||
            candidate.execution_plan_identity.empty()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        auto existing = candidates_.find(candidate.identity);
        if (existing == candidates_.end() && candidates_.size() >= max_cached_candidates) {
            candidates_.erase(candidates_.begin());
            ++cache_evictions_;
        }
        candidates_[candidate.identity] = std::move(candidate);
        return true;
    } catch (...) { return false; }
}

bool QwenExecutionPlanOptimizer::mark_candidate_valid(const std::string & identity,
    const std::string & validation_note) noexcept {
    try {
        if (validation_note.empty()) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = candidates_.find(identity);
        if (found == candidates_.end() || found->second.status != QwenCandidateStatus::Candidate) return false;
        uint64_t hotness = 0;
        for (const auto & entry : profiles_) {
            const auto & profile = entry.second;
            if (profile.plan_identity == found->second.base_plan_identity &&
                profile.phase == QwenExecutionPhase::Decode)
                hotness = saturated_add(hotness, profile.execution_count);
        }
        if (hotness < min_candidate_hotness_observations) return false;
        found->second.status = QwenCandidateStatus::Valid;
        found->second.validation_note = validation_note;
        return true;
    } catch (...) { return false; }
}

bool QwenExecutionPlanOptimizer::invalidate_candidate(const std::string & identity) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = candidates_.find(identity);
        if (found == candidates_.end()) return false;
        found->second.status = QwenCandidateStatus::Invalidated;
        return true;
    } catch (...) { return false; }
}

void QwenExecutionPlanOptimizer::set_unvalidated_trial_for_testing(bool enabled) noexcept {
    try { std::lock_guard<std::mutex> lock(mutex_); allow_unvalidated_trial_for_testing_ = enabled; } catch (...) {}
}

bool QwenExecutionPlanOptimizer::consume_candidate_execution_fault_for_testing() noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        if (test_fault_ != QwenOptimizerFault::CandidateExecution) return false;
        test_fault_ = QwenOptimizerFault::None;
        return true;
    } catch (...) { return true; }
}

void QwenExecutionPlanOptimizer::set_fault_for_testing(QwenOptimizerFault fault) noexcept {
    try { std::lock_guard<std::mutex> lock(mutex_); test_fault_ = fault; } catch (...) {}
}

QwenOptimizerSnapshot QwenExecutionPlanOptimizer::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    QwenOptimizerSnapshot result;
    result.cache_hits = cache_hits_;
    result.cache_misses = cache_misses_;
    result.cache_evictions = cache_evictions_;
    result.guard_passes = guard_passes_;
    result.guard_failures = guard_failures_;
    result.optimizer_errors = optimizer_errors_;
    result.observations = observations_;
    result.cache_size = candidates_.size();
    result.candidate_count = candidates_.size();
    for (const auto & entry : candidates_) {
        if (entry.second.status == QwenCandidateStatus::Valid) ++result.valid_candidate_count;
        else if (entry.second.status == QwenCandidateStatus::Invalidated) ++result.invalidated_candidate_count;
    }
    result.profile_key_count = profiles_.size();
    result.last_decision = last_decision_;
    result.profile_records.reserve(profiles_.size());
    for (const auto & entry : profiles_) result.profile_records.push_back(entry.second);
    return result;
}

} // namespace vbuf_ggml
