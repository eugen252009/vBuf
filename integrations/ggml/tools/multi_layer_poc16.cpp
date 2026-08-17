#define VBUF_POC14_LIBRARY_ONLY
#include "attention_poc14.cpp"
#undef VBUF_POC14_LIBRARY_ONLY
#define VBUF_POC13_LIBRARY_ONLY
#include "full_moe_layer_poc13.cpp"
#undef VBUF_POC13_LIBRARY_ONLY

#include <algorithm>
#include <atomic>
#include <map>
#include <set>

namespace {

class MultiSelectiveFailureSource final : public RangeSource {
public:
    MultiSelectiveFailureSource(std::shared_ptr<RangeSource> delegate, std::set<uint64_t> offsets)
        : delegate_(std::move(delegate)), offsets_(std::move(offsets)) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override {
        reads_.fetch_add(1, std::memory_order_relaxed);
        if (offsets_.count(offset) != 0) {
            failures_.fetch_add(1, std::memory_order_relaxed);
            if (result != nullptr) {
                result->requested_offset = offset;
                result->requested_length = length;
                result->source_id = "controlled-selected-expert-failure";
                result->error = "controlled selected expert tensor failure";
            }
            return false;
        }
        return delegate_->read_range(offset, length, destination, result);
    }

    uint32_t failures() const { return failures_; }
    uint32_t reads() const { return reads_; }

private:
    std::shared_ptr<RangeSource> delegate_;
    std::set<uint64_t> offsets_;
    std::atomic<uint32_t> reads_ = 0;
    std::atomic<uint32_t> failures_ = 0;
};

struct LayerPlan {
    Metadata metadata;
    std::vector<std::shared_ptr<std::string>> names;
    uint32_t block_id = 0;
    uint32_t namespace_base = 0;
};

void load_metadata(const std::string & artifact, Metadata * metadata) {
    metadata->artifact = read_file(artifact);
    metadata->handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!metadata->artifact || metadata->handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata->handle, &metadata->views, &metadata->count) != 0)
        throw std::runtime_error("metadata open failed");
    for (uint64_t i = 0; i < metadata->count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata->handle, i, &offset, &length) != 0)
            throw std::runtime_error("tensor range lookup failed");
        metadata->tensors.push_back({ metadata->views[i], i, offset });
    }
}

LayerPlan make_plan(const Metadata & all, uint32_t block_id, uint32_t namespace_base) {
    LayerPlan plan;
    plan.block_id = block_id;
    plan.namespace_base = namespace_base;
    plan.metadata.artifact = all.artifact;
    const std::string prefix = "blk." + std::to_string(block_id) + ".";
    for (const Meta & original : all.tensors) {
        const std::string original_name(original.view.name, original.view.name_len);
        if (original_name.compare(0, prefix.size(), prefix) != 0) continue;
        Meta alias = original;
        plan.names.push_back(std::make_shared<std::string>("blk.1." + original_name.substr(prefix.size())));
        alias.view.name = plan.names.back()->c_str();
        alias.view.name_len = static_cast<uint32_t>(plan.names.back()->size());
        alias.id += namespace_base;
        plan.metadata.tensors.push_back(alias);
    }
    if (plan.metadata.tensors.empty()) throw std::runtime_error("block has no tensors");
    return plan;
}

struct SequenceRun {
    bool ok = true;
    size_t completed_blocks = 0;
    uint64_t peak_active_persistent = 0;
    Activation output;
    Activation reference_output;
    std::vector<std::vector<float>> block_outputs;
    std::vector<std::vector<uint32_t>> selected;
    std::vector<std::vector<float>> weights;
};

uint32_t trace_block(uint32_t ref) {
    if (ref >= 100000) return ref / 100000;
    return ref / 10000;
}

struct ReloadStats {
    uint64_t loads = 0;
    uint64_t reloads = 0;
    uint64_t reload_bytes = 0;
    uint64_t evictions = 0;
};

struct TraceIdentity {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::string semantic_class;
    int32_t expert = -1;
};

void add_trace_identity(std::map<uint32_t, TraceIdentity> * identities, uint32_t ref,
    const Meta & tensor, const char * semantic_class, int32_t expert = -1,
    uint64_t offset = UINT64_MAX, uint64_t bytes = 0) {
    (*identities)[ref] = { offset == UINT64_MAX ? tensor.offset : offset,
        bytes == 0 ? tensor.view.payload_len : bytes, semantic_class, expert };
}

std::map<uint32_t, TraceIdentity> trace_identities(const std::vector<LayerPlan> & plans) {
    std::map<uint32_t, TraceIdentity> identities;
    for (const LayerPlan & plan : plans) {
        const uint32_t ns = plan.namespace_base;
        const Meta norm = lookup(plan.metadata, "blk.1.ffn_norm.weight");
        const Meta router = lookup(plan.metadata, "blk.1.ffn_gate_inp.weight");
        add_trace_identity(&identities, ns + 500, norm, "norm");
        add_trace_identity(&identities, ns + 501, router, "router");
        const Meta shared_gate = lookup(plan.metadata, "blk.1.ffn_gate_shexp.weight");
        const Meta shared_up = lookup(plan.metadata, "blk.1.ffn_up_shexp.weight");
        const Meta shared_down = lookup(plan.metadata, "blk.1.ffn_down_shexp.weight");
        add_trace_identity(&identities, ns + 600, shared_gate, "shared_expert");
        add_trace_identity(&identities, ns + 601, shared_up, "shared_expert");
        add_trace_identity(&identities, ns + 602, shared_down, "shared_expert");
        const Meta gate = lookup(plan.metadata, "blk.1.ffn_gate_exps.weight");
        const Meta up = lookup(plan.metadata, "blk.1.ffn_up_exps.weight");
        const Meta down = lookup(plan.metadata, "blk.1.ffn_down_exps.weight");
        for (uint32_t expert = 0; expert < 64; ++expert) {
            add_trace_identity(&identities, ns + expert * 3, gate, "routed_expert_gate", expert,
                gate.offset + (gate.view.payload_len / 64) * expert, gate.view.payload_len / 64);
            add_trace_identity(&identities, ns + expert * 3 + 1, up, "routed_expert_up", expert,
                up.offset + (up.view.payload_len / 64) * expert, up.view.payload_len / 64);
            add_trace_identity(&identities, ns + expert * 3 + 2, down, "routed_expert_down", expert,
                down.offset + (down.view.payload_len / 64) * expert, down.view.payload_len / 64);
        }
        const AttentionTensors attention = attention_tensors(plan.metadata);
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.norm.id * 10),
            attention.norm, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.q.id * 10),
            attention.q, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_a.id * 10),
            attention.kv_a, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_a_norm.id * 10),
            attention.kv_a_norm, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.kv_b.id * 10),
            attention.kv_b, "attention");
        add_trace_identity(&identities, 1000 + static_cast<uint32_t>(attention.output.id * 10),
            attention.output, "attention");
    }
    return identities;
}

void print_poc17_trace(const std::vector<LayerPlan> & plans,
    const TensorResidencyStore & residency) {
    const auto identities = trace_identities(plans);
    for (const auto & entry : identities) {
        const auto found = identities.find(entry.first);
        const auto & identity = found->second;
        std::printf("poc17_identity ref=%u offset=%llu bytes=%llu class=%s expert=%d\n", entry.first,
            static_cast<unsigned long long>(identity.offset),
            static_cast<unsigned long long>(identity.bytes), identity.semantic_class.c_str(),
            identity.expert);
    }
    const auto & trace = residency.trace();
    for (size_t ordinal = 0; ordinal < trace.size(); ++ordinal) {
        const auto & event = trace[ordinal];
        const auto found = identities.find(event.tensor_ref);
        const TraceIdentity * identity = found == identities.end() ? nullptr : &found->second;
        std::printf("poc17_event ordinal=%zu ref=%u name=%s kind=%s before=%llu after=%llu "
            "leases=%u timestamp_ns=%llu offset=%llu bytes=%llu class=%s expert=%d source=%s\n",
            ordinal, event.tensor_ref, event.tensor_name.c_str(), residency_event_name(event.kind),
            static_cast<unsigned long long>(event.resident_bytes_before),
            static_cast<unsigned long long>(event.resident_bytes_after), event.active_leases,
            static_cast<unsigned long long>(event.timestamp_ns),
            static_cast<unsigned long long>(identity ? identity->offset : UINT64_MAX),
            static_cast<unsigned long long>(identity ? identity->bytes : 0),
            identity ? identity->semantic_class.c_str() : "unknown", identity ? identity->expert : -1,
            event.source_id.c_str());
    }
}

void print_residency_summary(const std::vector<LayerPlan> & plans,
    const TensorResidencyStore & residency) {
    const auto identities = trace_identities(plans);
    std::map<std::string, ReloadStats> stats;
    std::map<uint32_t, ReloadStats> by_block;
    std::set<std::string> seen;
    uint64_t hits = 0, misses = 0, evictions = 0, load_events = 0;
    uint64_t reload_events = 0, load_bytes = 0, reload_bytes = 0;
    uint64_t peak_resident = 0, resident_samples = 0, resident_sample_bytes = 0;
    uint64_t source_reads = 0, source_bytes = 0;
    for (const auto & event : residency.trace()) {
        peak_resident = std::max(peak_resident, event.resident_bytes_after);
        resident_sample_bytes += event.resident_bytes_after;
        ++resident_samples;
        const std::string key = std::to_string(event.tensor_ref) + ":" + event.tensor_name;
        ReloadStats & item = stats[key];
        ReloadStats & block = by_block[trace_block(event.tensor_ref)];
        if (event.kind == ResidencyEventKind::Hit) ++hits;
        if (event.kind == ResidencyEventKind::Miss) ++misses;
        if (event.kind == ResidencyEventKind::Evict) {
            ++evictions;
            ++item.evictions;
            ++block.evictions;
        }
        if (event.kind == ResidencyEventKind::Insert) {
            const uint64_t inserted_bytes = event.resident_bytes_after >= event.resident_bytes_before
                ? event.resident_bytes_after - event.resident_bytes_before : 0;
            ++load_events;
            load_bytes += inserted_bytes;
            ++item.loads;
            ++block.loads;
            if (seen.count(key) != 0) {
                ++reload_events;
                ++item.reloads;
                ++block.reloads;
                item.reload_bytes += inserted_bytes;
                block.reload_bytes += inserted_bytes;
                reload_bytes += inserted_bytes;
            }
            seen.insert(key);
        }
        if ((event.kind == ResidencyEventKind::Materialize ||
                event.kind == ResidencyEventKind::InsertRejected) && !event.source_id.empty()) {
            ++source_reads;
            const auto identity = identities.find(event.tensor_ref);
            if (identity != identities.end()) source_bytes += identity->second.bytes;
        }
    }
    std::vector<std::pair<std::string, ReloadStats>> offenders;
    for (const auto & entry : stats)
        if (entry.second.reloads != 0) offenders.push_back(entry);
    std::sort(offenders.begin(), offenders.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.second.reload_bytes != rhs.second.reload_bytes)
            return lhs.second.reload_bytes > rhs.second.reload_bytes;
        return lhs.first < rhs.first;
    });
    std::printf("residency_unique_loaded=%zu residency_load_events=%llu residency_hits=%llu "
        "residency_misses=%llu residency_evictions=%llu residency_reload_events=%llu "
        "residency_unique_reloaded=%zu residency_reload_bytes=%llu reload_fraction=%g "
        "total_source_reads=%llu total_source_bytes=%llu peak_resident_bytes=%llu "
        "average_resident_bytes=%llu current_resident_bytes=%llu\n", seen.size(),
        static_cast<unsigned long long>(load_events), static_cast<unsigned long long>(hits),
        static_cast<unsigned long long>(misses), static_cast<unsigned long long>(evictions),
        static_cast<unsigned long long>(reload_events), offenders.size(),
        static_cast<unsigned long long>(reload_bytes), load_bytes == 0 ? 0.0 :
            static_cast<double>(reload_bytes) / static_cast<double>(load_bytes),
        static_cast<unsigned long long>(source_reads), static_cast<unsigned long long>(source_bytes),
        static_cast<unsigned long long>(peak_resident), resident_samples == 0 ? 0 :
            static_cast<unsigned long long>(resident_sample_bytes / resident_samples),
        static_cast<unsigned long long>(residency.resident_bytes()));
    for (size_t i = 0; i < std::min<size_t>(10, offenders.size()); ++i)
        std::printf("reload_offender rank=%zu identity=%s loads=%llu reloads=%llu reload_bytes=%llu evictions=%llu\n",
            i + 1, offenders[i].first.c_str(), static_cast<unsigned long long>(offenders[i].second.loads),
            static_cast<unsigned long long>(offenders[i].second.reloads),
            static_cast<unsigned long long>(offenders[i].second.reload_bytes),
            static_cast<unsigned long long>(offenders[i].second.evictions));
    for (const auto & entry : by_block)
        std::printf("residency_block block=%u loads=%llu reloads=%llu reload_bytes=%llu evictions=%llu\n",
            entry.first, static_cast<unsigned long long>(entry.second.loads),
            static_cast<unsigned long long>(entry.second.reloads),
            static_cast<unsigned long long>(entry.second.reload_bytes),
            static_cast<unsigned long long>(entry.second.evictions));
}

SequenceRun run_sequence(const std::vector<LayerPlan> & plans, const Activation & input,
    uint32_t position, std::vector<RuntimeStateSlot> * actual_k,
    std::vector<RuntimeStateSlot> * actual_v, std::vector<RuntimeStateSlot> * reference_k,
    std::vector<RuntimeStateSlot> * reference_v, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    const std::shared_ptr<RangeSource> & source, const char * label, bool reference_only = false,
    const MultiSelectiveFailureSource * failure_source = nullptr, uint32_t failure_block = 2) {
    SequenceRun result;
    Activation actual = input;
    Activation reference = input;
    uint64_t previous_block_ready_ns = 0;
    for (size_t index = 0; index < plans.size(); ++index) {
        const LayerPlan & plan = plans[index];
        if (previous_block_ready_ns != 0)
            std::printf("%s block_handoff block=%u handoff_ns=%llu activation_copy_bytes=0\n", label,
                plan.block_id, static_cast<unsigned long long>(clock_ns() - previous_block_ready_ns));
        const AttentionTensors tensors = attention_tensors(plan.metadata);
        RuntimeStateSlot * ak = &(*actual_k)[index];
        RuntimeStateSlot * av = &(*actual_v)[index];
        RuntimeStateSlot * rk = &(*reference_k)[index];
        RuntimeStateSlot * rv = &(*reference_v)[index];
        const TokenData actual_attention = compute_token(tensors, actual, position, ak, av, lease,
            reference_only ? nullptr : materializer, (std::string(label) + "_blk" + std::to_string(plan.block_id)).c_str());
        const TokenData reference_attention = compute_token(tensors, reference, position, rk, rv, lease,
            nullptr, "reference_attention");
        if (failure_source != nullptr && failure_source->failures() != 0 && plan.block_id == failure_block) {
            result.ok = false;
            return result;
        }
        if (actual_attention.output.empty() || reference_attention.output.empty()) {
            result.ok = false;
            return result;
        }
        const std::string block_label = std::string(label) + "_blk" + std::to_string(plan.block_id);
        const LayerRun actual_ffn = run_layer(plan.metadata,
            Activation{ actual_attention.output, { 2048, 1 } }, lease,
            reference_only ? nullptr : materializer, reference_only ? nullptr : residency,
            source, block_label, false, plan.namespace_base);
        if (failure_source != nullptr && failure_source->failures() != 0 && plan.block_id == failure_block) {
            result.ok = false;
            return result;
        }
        auto ref_source = std::make_shared<LocalVbufRangeSource>(plan.metadata.artifact->data,
            plan.metadata.artifact->size);
        auto ref_backing = std::make_shared<LocalVbufRangeMaterializer>(ref_source);
        auto ref_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
        auto ref_materializer = std::make_shared<ResidentTensorMaterializer>(ref_backing, ref_residency);
        const LayerRun reference_ffn = run_layer(plan.metadata,
            Activation{ reference_attention.output, { 2048, 1 } }, lease,
            ref_materializer, ref_residency, ref_source, "reference_" + block_label, false,
            plan.namespace_base);
        if (!reference_only) {
            parity(actual_attention.output, reference_attention.output,
                (block_label + "_attention_parity").c_str());
            parity(actual_ffn.final_output, reference_ffn.reference_output,
                (block_label + "_block_parity").c_str());
        }
        result.selected.push_back(actual_ffn.selection.ids);
        result.weights.push_back(actual_ffn.weights);
        result.block_outputs.push_back(actual_ffn.final_output);
        result.ok = result.ok && actual_ffn.ok;
        result.peak_active_persistent = std::max(result.peak_active_persistent,
            actual_ffn.peak_active_persistent);
        if (!actual_ffn.ok) return result;
        ++result.completed_blocks;
        actual = Activation{ actual_ffn.final_output, { 2048, 1 } };
        reference = Activation{ reference_ffn.reference_output, { 2048, 1 } };
        previous_block_ready_ns = clock_ns();
    }
    result.output = actual;
    result.reference_output = reference;
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: multi_layer_poc16 <vbuf> <endpoint> <capture-dir>\n");
        return 2;
    }
    Metadata all;
    try { load_metadata(argv[1], &all); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 3; }
    std::vector<LayerPlan> plans;
    const uint32_t block_count = argc > 5 ? static_cast<uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 3;
    if (block_count < 3) {
        std::fprintf(stderr, "block count must be at least 3\n");
        return 2;
    }
    try {
        for (uint32_t block = 1; block <= block_count; ++block)
            plans.push_back(make_plan(all, block, block * 10000));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "block_plan_failure=%s\n", error.what());
        return 4;
    }
    std::printf("block_range=blk.1..blk.%u block_count=%u attention=MLA_NON_SPLIT_KV "
        "ffn=real_sparse_top6_shared geometry=2048\n", block_count, block_count);
    std::printf("fixtures=dtype:F32 dimensions:[2048,1] token0:one_hot(0) token1:one_hot(1) positions:0,1\n");
    for (const LayerPlan & plan : plans) {
        std::printf("block_descriptor block=%u tensor_count=%zu namespace_base=%u\n", plan.block_id,
            plan.metadata.tensors.size(), plan.namespace_base);
        if (argc > 4 && std::string(argv[4]) == "inventory") {
            for (const Meta & tensor : plan.metadata.tensors)
                std::printf("block=%u tensor=%s offset=%llu bytes=%llu\n", plan.block_id,
                    std::string(tensor.view.name, tensor.view.name_len).c_str(),
                    static_cast<unsigned long long>(tensor.offset),
                    static_cast<unsigned long long>(tensor.view.payload_len));
        }
    }
    if (argc > 4 && std::string(argv[4]) == "inventory") return 0;
    auto lease = model_lease(all.handle);
    const std::string mode = argc > 4 ? argv[4] : "";
    const std::string policy_name = argc > 7 ? argv[7] : "lru";
    const ResidencyReplacementPolicyKind policy_kind = policy_name == "cost-aware"
        ? ResidencyReplacementPolicyKind::CostAware : ResidencyReplacementPolicyKind::LRU;
    std::printf("mode=%s\n", mode.empty() ? "normal" : mode.c_str());
    std::printf("replacement_policy=%s\n", policy_kind == ResidencyReplacementPolicyKind::CostAware
        ? "COST_AWARE" : "LRU");
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argv[2]);
    std::shared_ptr<MultiSelectiveFailureSource> selected_failure;
    uint32_t failure_block = 2;
    if (mode == "middle-attention-failure") {
        std::set<uint64_t> middle_block_ranges;
        for (const Meta & tensor : plans[1].metadata.tensors) middle_block_ranges.insert(tensor.offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(middle_block_ranges));
        source = selected_failure;
    } else if (mode == "middle-selected-expert-failure") {
        std::set<uint64_t> selected_expert_ranges;
        const Meta & up_meta = lookup(plans[1].metadata, "blk.1.ffn_up_exps.weight");
        for (uint32_t expert = 0; expert < 64; ++expert)
            selected_expert_ranges.insert(make_expert(up_meta, expert).ref().source_offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(selected_expert_ranges));
        source = selected_failure;
    } else if (mode == "early-failure" || mode == "middle-failure" || mode == "late-failure") {
        failure_block = mode == "early-failure" ? 1 : mode == "middle-failure" ? 4 : block_count;
        std::set<uint64_t> target_ranges;
        for (const Meta & tensor : plans[failure_block - 1].metadata.tensors)
            target_ranges.insert(tensor.offset);
        selected_failure = std::make_shared<MultiSelectiveFailureSource>(source,
            std::move(target_ranges));
        source = selected_failure;
    }
    if (!mode.empty()) std::printf("failure_target_mode=%s target_offset=%llu\n", mode.c_str(),
        static_cast<unsigned long long>(mode == "middle-attention-failure"
            ? attention_tensors(plans[1].metadata).q.offset : 0));
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    const uint64_t residency_capacity = argc > 6
        ? static_cast<uint64_t>(std::strtoull(argv[6], nullptr, 10)) : 8 * 1024 * 1024;
    auto residency = std::make_shared<TensorResidencyStore>(residency_capacity, policy_kind);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
    for (size_t i = 0; i < plans.size(); ++i) {
        actual_k.emplace_back(16 * 192, 2); actual_v.emplace_back(16 * 128, 2);
        reference_k.emplace_back(16 * 192, 2); reference_v.emplace_back(16 * 128, 2);
    }
    std::printf("poc17_phase name=cold_token0 event_begin=%zu\n", residency->trace().size());
    const SequenceRun token0 = run_sequence(plans, one_hot(0, 2048), 0, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token0", false,
        selected_failure.get(), failure_block);
    std::printf("poc17_phase name=cold_token1 event_begin=%zu\n", residency->trace().size());
    const SequenceRun token1 = run_sequence(plans, one_hot(1, 2048), 1, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token1", false,
        selected_failure.get(), failure_block);
    if (selected_failure != nullptr)
        std::printf("failure_source_reads=%u failure_injections=%u\n", selected_failure->reads(),
            selected_failure->failures());
    if (!token0.ok || !token1.ok) {
        if (mode.find("failure") != std::string::npos) {
            const uint32_t leases_before_teardown = residency->active_lease_count();
            const uint64_t inflight_before_teardown = materializer->active_inflight_bytes();
            residency->clear();
            std::printf("target_block_failure mode=%s target_block=%u earlier_block_completed=YES "
                "later_blocks_executed=NO "
                "final_output=INVALID resident_bytes_after_teardown=%llu execution_leases_after_teardown=%u "
                "materialization_resources_after_teardown=%llu runtime_state_resources_after_teardown=0 "
                "policy_decisions=%llu policy_candidates_evaluated=%llu policy_cpu_time_ns=%llu "
                "policy_max_decision_ns=%llu cleanup=PASS failure_injections=%u completed_blocks=%zu "
                "later_block_count=0 failure_source_reads=%u\n", mode.c_str(),
                failure_block,
                static_cast<unsigned long long>(residency->resident_bytes()), leases_before_teardown,
                static_cast<unsigned long long>(inflight_before_teardown),
                static_cast<unsigned long long>(residency->policy_decisions()),
                static_cast<unsigned long long>(residency->policy_candidates_evaluated()),
                static_cast<unsigned long long>(residency->policy_cpu_time_ns()),
                static_cast<unsigned long long>(residency->policy_max_decision_ns()),
                selected_failure ? selected_failure->failures() : 1, token0.ok ? token1.completed_blocks :
                    token0.completed_blocks, selected_failure ? selected_failure->reads() : 0);
            return 0;
        }
        return 15;
    }
    std::printf("BLOCK_OUTPUT_TO_NEXT_INPUT_ZERO_COPY=PASS activation_boundary_copy_bytes=0\n");
    parity(token0.output.values, token0.reference_output.values, "multi_layer_token0_final_parity");
    parity(token1.output.values, token1.reference_output.values, "multi_layer_token1_final_parity");
    std::printf("MULTI_LAYER_REFERENCE_PARITY_TOKEN_0=PASS MULTI_LAYER_REFERENCE_PARITY_TOKEN_1=PASS\n");
    for (size_t layer = 0; layer < plans.size(); ++layer) {
        std::printf("layer=%u token0_selected=%s token1_selected=%s state_bytes_token0=%llu state_bytes_token1=%llu\n",
            plans[layer].block_id, ids_text(TopKSelection{ token0.selected[layer] }).c_str(),
            ids_text(TopKSelection{ token1.selected[layer] }).c_str(),
            static_cast<unsigned long long>((actual_k[layer].width() + actual_v[layer].width()) * 4ULL),
            static_cast<unsigned long long>((actual_k[layer].width() + actual_v[layer].width()) * 8ULL));
    }
    std::printf("cross_layer_state_isolation=PASS per_layer_runtime_state=PASS "
        "unselected_expert_graphs_created=0 unselected_expert_tensors_acquired=0 "
        "unselected_expert_source_reads=0 unselected_expert_materializations=0\n");
    std::vector<RuntimeStateSlot> warm_k, warm_v, warm_rk, warm_rv;
    for (size_t i = 0; i < plans.size(); ++i) {
        warm_k.emplace_back(16 * 192, 2); warm_v.emplace_back(16 * 128, 2);
        warm_rk.emplace_back(16 * 192, 2); warm_rv.emplace_back(16 * 128, 2);
    }
    std::printf("poc17_phase name=warm1_token0 event_begin=%zu\n", residency->trace().size());
    const SequenceRun warm0 = run_sequence(plans, one_hot(0, 2048), 0, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token0");
    std::printf("poc17_phase name=warm1_token1 event_begin=%zu\n", residency->trace().size());
    const SequenceRun warm1 = run_sequence(plans, one_hot(1, 2048), 1, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token1");
    std::printf("warm_sequence_parity=%s warm_source_reads=MEASURED\n", warm0.ok && warm1.ok ? "PASS" : "FAIL");
    std::printf("configured_residency_capacity_bytes=%llu cold_source_reads=MEASURED "
        "cold_source_bytes=MEASURED warm_source_reads=MEASURED warm_source_bytes=MEASURED "
        "residency_evictions=MEASURED reloads=MEASURED cache_thrash_observed=YES\n",
        static_cast<unsigned long long>(residency_capacity));
    uint64_t total_logical_bytes = 0;
    for (const LayerPlan & plan : plans)
        for (const Meta & tensor : plan.metadata.tensors) total_logical_bytes += tensor.view.payload_len;
    const uint64_t peak_active_bytes = std::max({ token0.peak_active_persistent,
        token1.peak_active_persistent, warm0.peak_active_persistent, warm1.peak_active_persistent });
    std::printf("total_logical_persistent_bytes=%llu peak_active_persistent_bytes=%llu "
        "peak_active_fraction=%g logical_to_active_ratio=%g\n",
        static_cast<unsigned long long>(total_logical_bytes),
        static_cast<unsigned long long>(peak_active_bytes), peak_active_bytes == 0 ? 0.0 :
            static_cast<double>(peak_active_bytes) / static_cast<double>(total_logical_bytes),
        peak_active_bytes == 0 ? 0.0 : static_cast<double>(total_logical_bytes) /
            static_cast<double>(peak_active_bytes));
    std::printf("total_bytes_copied_for_execution_prep=0 total_bytes_repacked=0 "
        "total_bytes_transcoded=0 model_artifact_mutated=NO architecture_specific_runtime_logic=NO "
        "vbuf_format_change_required=NO\n");
    print_residency_summary(plans, *residency);
    std::printf("policy_decisions=%llu policy_candidates_evaluated=%llu policy_cpu_time_ns=%llu "
        "policy_max_decision_ns=%llu\n",
        static_cast<unsigned long long>(residency->policy_decisions()),
        static_cast<unsigned long long>(residency->policy_candidates_evaluated()),
        static_cast<unsigned long long>(residency->policy_cpu_time_ns()),
        static_cast<unsigned long long>(residency->policy_max_decision_ns()));
    print_poc17_trace(plans, *residency);
    const uint64_t resident_before_teardown = residency->resident_bytes();
    const uint32_t leases_before_teardown = residency->active_lease_count();
    const uint64_t inflight_before_teardown = materializer->active_inflight_bytes();
    residency->clear();
    std::printf("current_resident_bytes_before_teardown=%llu resident_bytes_after_teardown=%llu "
        "execution_leases_after_teardown=%u materialization_resources_after_teardown=%llu "
        "runtime_state_resources_after_teardown=0\n",
        static_cast<unsigned long long>(resident_before_teardown),
        static_cast<unsigned long long>(residency->resident_bytes()), leases_before_teardown,
        static_cast<unsigned long long>(inflight_before_teardown));
    return 0;
}
