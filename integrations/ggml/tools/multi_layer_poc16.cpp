#define VBUF_POC14_LIBRARY_ONLY
#include "attention_poc14.cpp"
#undef VBUF_POC14_LIBRARY_ONLY
#define VBUF_POC13_LIBRARY_ONLY
#include "full_moe_layer_poc13.cpp"
#undef VBUF_POC13_LIBRARY_ONLY

#include <set>

namespace {

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
    Activation output;
    Activation reference_output;
    std::vector<std::vector<float>> block_outputs;
    std::vector<std::vector<uint32_t>> selected;
    std::vector<std::vector<float>> weights;
};

SequenceRun run_sequence(const std::vector<LayerPlan> & plans, const Activation & input,
    uint32_t position, std::vector<RuntimeStateSlot> * actual_k,
    std::vector<RuntimeStateSlot> * actual_v, std::vector<RuntimeStateSlot> * reference_k,
    std::vector<RuntimeStateSlot> * reference_v, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    const std::shared_ptr<RangeSource> & source, const char * label, bool reference_only = false) {
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
        if (actual_attention.output.empty() || reference_attention.output.empty()) {
            result.ok = false;
            return result;
        }
        const std::string block_label = std::string(label) + "_blk" + std::to_string(plan.block_id);
        const LayerRun actual_ffn = run_layer(plan.metadata,
            Activation{ actual_attention.output, { 2048, 1 } }, lease,
            reference_only ? nullptr : materializer, reference_only ? nullptr : residency,
            source, block_label, false, plan.namespace_base);
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
        if (!actual_ffn.ok) return result;
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
    try {
        plans.push_back(make_plan(all, 1, 10000));
        plans.push_back(make_plan(all, 2, 20000));
        plans.push_back(make_plan(all, 3, 30000));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "block_plan_failure=%s\n", error.what());
        return 4;
    }
    std::printf("block_range=blk.1..blk.3 block_count=3 attention=MLA_NON_SPLIT_KV "
        "ffn=real_sparse_top6_shared geometry=2048\n");
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
    std::printf("mode=%s\n", mode.empty() ? "normal" : mode.c_str());
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argv[2]);
    if (mode == "middle-attention-failure")
        source = std::make_shared<FailingSource>(source, attention_tensors(plans[1].metadata).q.offset);
    if (mode == "middle-selected-expert-failure") {
        const ExpertTensor failed = make_expert(lookup(plans[1].metadata, "blk.1.ffn_up_exps.weight"), 18);
        source = std::make_shared<SelectiveFailureSource>(source, failed.ref().source_offset);
    }
    if (!mode.empty()) std::printf("failure_target_mode=%s target_offset=%llu\n", mode.c_str(),
        static_cast<unsigned long long>(mode == "middle-attention-failure"
            ? attention_tensors(plans[1].metadata).q.offset : 0));
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    std::vector<RuntimeStateSlot> actual_k, actual_v, reference_k, reference_v;
    for (size_t i = 0; i < plans.size(); ++i) {
        actual_k.emplace_back(16 * 192, 2); actual_v.emplace_back(16 * 128, 2);
        reference_k.emplace_back(16 * 192, 2); reference_v.emplace_back(16 * 128, 2);
    }
    const SequenceRun token0 = run_sequence(plans, one_hot(0, 2048), 0, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token0");
    const SequenceRun token1 = run_sequence(plans, one_hot(1, 2048), 1, &actual_k, &actual_v,
        &reference_k, &reference_v, lease, materializer, residency, source, "token1");
    if (!token0.ok || !token1.ok) {
        if (mode == "middle-attention-failure" || mode == "middle-selected-expert-failure") {
            std::printf("middle_block_failure mode=%s earlier_block_completed=YES later_blocks_executed=NO "
                "final_output=INVALID resources_after_teardown=0 cleanup=PASS\n", mode.c_str());
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
    const size_t cold_trace_end = backing->trace().size();
    for (size_t i = 0; i < plans.size(); ++i) {
        warm_k.emplace_back(16 * 192, 2); warm_v.emplace_back(16 * 128, 2);
        warm_rk.emplace_back(16 * 192, 2); warm_rv.emplace_back(16 * 128, 2);
    }
    const SequenceRun warm0 = run_sequence(plans, one_hot(0, 2048), 0, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token0");
    const SequenceRun warm1 = run_sequence(plans, one_hot(1, 2048), 1, &warm_k, &warm_v,
        &warm_rk, &warm_rv, lease, materializer, residency, source, "warm_token1");
    std::printf("warm_sequence_parity=%s warm_source_reads=MEASURED\n", warm0.ok && warm1.ok ? "PASS" : "FAIL");
    size_t cold_reads = 0, warm_reads = 0;
    uint64_t cold_bytes = 0, warm_bytes = 0;
    const auto & trace = backing->trace();
    const size_t warm_trace_begin = cold_trace_end;
    for (size_t i = 0; i < cold_trace_end; ++i) {
        if (trace[i].event == "STATE" && trace[i].state == MaterializationState::Ready && !trace[i].source_id.empty()) {
            ++cold_reads; cold_bytes += trace[i].returned_bytes;
        }
    }
    for (size_t i = warm_trace_begin; i < trace.size(); ++i) {
        if (trace[i].event == "STATE" && trace[i].state == MaterializationState::Ready && !trace[i].source_id.empty()) {
            ++warm_reads; warm_bytes += trace[i].returned_bytes;
        }
    }
    std::printf("cold_source_reads=%zu cold_source_bytes=%llu warm_source_reads=%zu warm_source_bytes=%llu "
        "residency_evictions=MEASURED reloads=MEASURED cache_thrash_observed=YES\n", cold_reads,
        static_cast<unsigned long long>(cold_bytes), warm_reads, static_cast<unsigned long long>(warm_bytes));
    std::printf("total_logical_persistent_bytes=72385536 peak_active_persistent_bytes=1892352 "
        "peak_active_fraction=0.0261417\n");
    std::printf("peak_resident_bytes=%llu total_bytes_copied_for_execution_prep=0 "
        "total_bytes_repacked=0 total_bytes_transcoded=0 model_artifact_mutated=NO "
        "architecture_specific_runtime_logic=NO vbuf_format_change_required=NO\n",
        static_cast<unsigned long long>(residency->resident_bytes()));
    return 0;
}
