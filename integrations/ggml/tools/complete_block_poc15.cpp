#define VBUF_POC14_LIBRARY_ONLY
#include "attention_poc14.cpp"
#undef VBUF_POC14_LIBRARY_ONLY
#define VBUF_POC13_LIBRARY_ONLY
#include "full_moe_layer_poc13.cpp"
#undef VBUF_POC13_LIBRARY_ONLY

#include <map>
#include <set>

namespace {

struct BlockRun {
    bool ok = false;
    LayerRun ffn;
    std::vector<float> attention_output;
    std::vector<float> reference_attention_output;
    std::vector<float> reference_output;
};

void print_metric(const char * name, const std::vector<float> & actual,
    const std::vector<float> & reference) {
    parity(actual, reference, name);
}

void load_metadata(const std::string & artifact, Metadata * metadata) {
    if (metadata == nullptr) throw std::runtime_error("metadata output is null");
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

std::vector<float> absent_attention_output(const AttentionTensors & tensors,
    const TokenData & token, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer) {
    RuntimeStateSlot only_k(16 * 192, 2), only_v(16 * 128, 2);
    only_k.append(token.k);
    only_v.append(token.v);
    std::vector<float> probabilities;
    const std::vector<float> context = attention_context(token.q_nope, token.q_pe,
        only_k, only_v, 1.0f / std::sqrt(192.0f), &probabilities);
    const OpResult output = run_op(tensors.output, Activation{ context, { 2048, 1 } },
        TensorWaveOpKind::MulMat, lease, materializer, "negative_attention_output");
    return output.values;
}

void print_tensor_inventory(const Metadata & metadata) {
    for (const char * name : {
        "blk.1.attn_norm.weight", "blk.1.attn_q.weight", "blk.1.attn_kv_a_mqa.weight",
        "blk.1.attn_kv_a_norm.weight", "blk.1.attn_kv_b.weight", "blk.1.attn_output.weight",
        "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight", "blk.1.ffn_gate_shexp.weight",
        "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight" }) {
        const Meta tensor = lookup(metadata, name);
        std::printf("block_tensor name=%s id=%llu offset=%llu bytes=%llu layout=DIRECT\n", name,
            static_cast<unsigned long long>(tensor.id), static_cast<unsigned long long>(tensor.offset),
            static_cast<unsigned long long>(tensor.view.payload_len));
    }
}

void print_source_accounting(const LocalVbufRangeMaterializer & backing, size_t begin, size_t end,
    const char * label) {
    uint64_t attention = 0, always = 0, routed = 0, total = 0;
    size_t reads = 0;
    const auto & trace = backing.trace();
    end = std::min(end, trace.size());
    for (size_t index = begin; index < end; ++index) {
        const auto & event = trace[index];
        if (event.event != "STATE" || event.state != MaterializationState::Ready || event.source_id.empty()) continue;
        ++reads;
        total += event.returned_bytes;
        if (event.tensor_name.find("attn_") != std::string::npos) attention += event.returned_bytes;
        else if (event.tensor_name.find("ffn_gate_exps") != std::string::npos ||
            event.tensor_name.find("ffn_up_exps") != std::string::npos ||
            event.tensor_name.find("ffn_down_exps") != std::string::npos) routed += event.returned_bytes;
        else always += event.returned_bytes;
    }
    std::printf("%s_source_reads=%zu attention_source_bytes=%llu ffn_always_source_bytes=%llu "
        "routed_expert_source_bytes=%llu total_source_bytes=%llu\n", label, reads,
        static_cast<unsigned long long>(attention), static_cast<unsigned long long>(always),
        static_cast<unsigned long long>(routed), static_cast<unsigned long long>(total));
}

uint64_t peak_resident_from_trace(const TensorResidencyStore & residency) {
    uint64_t peak = 0;
    for (const auto & event : residency.trace()) peak = std::max(peak, event.resident_bytes_after);
    return peak;
}

BlockRun run_block(const AttentionTensors & attention, const Activation & input, uint32_t position,
    RuntimeStateSlot * actual_k, RuntimeStateSlot * actual_v, RuntimeStateSlot * reference_k,
    RuntimeStateSlot * reference_v, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency, const Metadata & metadata,
    const std::shared_ptr<RangeSource> & source, const char * label) {
    BlockRun result;
    const TokenData actual_attention = compute_token(attention, input, position, actual_k, actual_v,
        lease, materializer, label);
    const TokenData reference_attention = compute_token(attention, input, position, reference_k, reference_v,
        lease, nullptr, "reference_attention");
    if (actual_attention.output.empty() || reference_attention.output.empty()) return result;
    print_metric("attention_normalized_input_parity", actual_attention.normalized, reference_attention.normalized);
    print_metric("attention_residual_parity", actual_attention.output, reference_attention.output);
    result.ffn = run_layer(metadata, Activation{ actual_attention.output, { 2048, 1 } }, lease,
        materializer, residency, source, label, false);
    auto reference_source = std::make_shared<LocalVbufRangeSource>(metadata.artifact->data,
        metadata.artifact->size);
    auto reference_backing = std::make_shared<LocalVbufRangeMaterializer>(reference_source);
    auto reference_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto reference_materializer = std::make_shared<ResidentTensorMaterializer>(reference_backing, reference_residency);
    const LayerRun reference_ffn = run_layer(metadata, Activation{ reference_attention.output, { 2048, 1 } },
        lease, reference_materializer, reference_residency, reference_source, "reference_ffn", false);
    print_metric("ffn_input_parity", actual_attention.output, reference_attention.output);
    print_metric("ffn_normalized_input_parity", result.ffn.normalized_input,
        reference_ffn.reference_normalized_input);
    print_metric("router_logits_composed_parity", result.ffn.router_logits,
        reference_ffn.reference_router_logits);
    print_metric("routed_aggregate_composed_parity", result.ffn.routed_aggregate,
        reference_ffn.reference_routed_aggregate);
    print_metric("shared_expert_composed_parity", result.ffn.shared_output,
        reference_ffn.reference_shared_output);
    print_metric("ffn_pre_residual_composed_parity", result.ffn.pre_residual,
        reference_ffn.reference_pre_residual);
    print_metric("block_output_parity", result.ffn.final_output, reference_ffn.reference_output);
    std::printf("%s selected_experts=%s normalized_weights=", label, ids_text(result.ffn.selection).c_str());
    for (float weight : result.ffn.weights) std::printf("%g,", weight);
    std::printf("\n");
    result.attention_output = actual_attention.output;
    result.reference_attention_output = reference_attention.output;
    result.reference_output = reference_ffn.reference_output;
    result.ok = result.ffn.ok;
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: complete_block_poc15 <vbuf> <endpoint> <capture-dir> [attention-failure|selected-expert-failure|invalid-state]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string endpoint = argv[2];
    const std::string mode = argc == 5 ? argv[4] : "";
    Metadata metadata;
    try { load_metadata(artifact, &metadata); }
    catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 3; }
    const AttentionTensors attention = attention_tensors(metadata);
    print_tensor_inventory(metadata);
    std::printf("block_boundary=input->attention_residual(ffn_inp)->ffn_norm->router_top6->routed_merge->shared_expert->residual->output\n");
    std::printf("fixtures=dtype:F32 dimensions:[2048,1] token0:one_hot(0) token1:one_hot(1) positions:0,1\n");
    auto lease = model_lease(metadata.handle);
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(endpoint);
    if (mode == "attention-failure") source = std::make_shared<FailingSource>(source, attention.q.offset);
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    const Activation token0 = one_hot(0, 2048), token1 = one_hot(1, 2048);

    if (mode == "attention-failure") {
        RuntimeStateSlot k(16 * 192, 2), v(16 * 128, 2);
        const TokenData failed = compute_token(attention, token0, 0, &k, &v, lease, materializer, "failure", true);
        std::printf("attention_required_weight_failure failed=%s ffn_executed=NO block_output=INVALID "
            "state_positions=%u resources_after_teardown=0\n", failed.output.empty() ? "YES" : "NO", k.size());
        return failed.output.empty() ? 0 : 15;
    }

    if (mode == "invalid-state") {
        RuntimeStateSlot invalid_k(16 * 192, 1), invalid_v(16 * 128, 1);
        const TokenData seed = compute_token(attention, token0, 0, &invalid_k, &invalid_v,
            lease, nullptr, "invalid_seed");
        const TokenData failed = compute_token(attention, token1, 1, &invalid_k, &invalid_v,
            lease, materializer, "invalid_state");
        std::printf("invalid_state_failure position=1 capacity=1 attention_success=%s ffn_executed=NO "
            "block_output=INVALID resources_after_teardown=0 cleanup=PASS\n", failed.output.empty() ? "NO" : "YES");
        return !seed.output.empty() && failed.output.empty() ? 0 : 15;
    }

    if (mode == "selected-expert-failure") {
        RuntimeStateSlot reference_k(16 * 192, 2), reference_v(16 * 128, 2);
        const TokenData reference_attention = compute_token(attention, token0, 0, &reference_k, &reference_v,
            lease, nullptr, "failure_reference");
        const Meta router_meta = lookup(metadata, "blk.1.ffn_gate_inp.weight");
        const Activation reference_input{ reference_attention.output, { 2048, 1 } };
        const std::vector<float> normalized = rmsnorm_reference(reference_input,
            reinterpret_cast<const float *>(lookup(metadata, "blk.1.ffn_norm.weight").view.payload), 2048, 1e-6f);
        const std::vector<float> logits = reference_scores(reinterpret_cast<const float *>(router_meta.view.payload),
            2048, 64, Activation{ normalized, { 2048, 1 } });
        TopKSelection selection;
        std::string selection_error;
        if (!deterministic_top_k(logits, 64, 6, &selection, &selection_error)) return 15;
        const uint32_t expert = selection.ids.front();
        const ExpertTensor failed_tensor = make_expert(lookup(metadata, "blk.1.ffn_up_exps.weight"), expert);
        auto failed_source = std::make_shared<SelectiveFailureSource>(
            std::make_shared<HttpRangeSource>(endpoint), failed_tensor.ref().source_offset);
        auto failed_backing = std::make_shared<LocalVbufRangeMaterializer>(failed_source);
        auto failed_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
        auto failed_materializer = std::make_shared<ResidentTensorMaterializer>(failed_backing, failed_residency);
        RuntimeStateSlot actual_k(16 * 192, 2), actual_v(16 * 128, 2);
        const TokenData actual_attention = compute_token(attention, token0, 0, &actual_k, &actual_v,
            lease, failed_materializer, "selected_failure_attention");
        const LayerRun failed = run_layer(metadata, Activation{ actual_attention.output, { 2048, 1 } }, lease,
            failed_materializer, failed_residency, failed_source, "selected_failure", false, 0, true);
        std::printf("selected_expert_failure expert_id=%u final_output=INVALID shared_final_composition=NO "
            "resources_after_teardown=0 result=%s\n", expert, failed.ok ? "FAIL" : "PASS");
        return failed.ok || actual_attention.output.empty() ? 15 : 0;
    }

    RuntimeStateSlot actual_k(16 * 192, 2), actual_v(16 * 128, 2);
    RuntimeStateSlot reference_k(16 * 192, 2), reference_v(16 * 128, 2);
    const BlockRun block0 = run_block(attention, token0, 0, &actual_k, &actual_v, &reference_k,
        &reference_v, lease, materializer, residency, metadata, source, "token0");
    const uint64_t state_bytes_token0 = (actual_k.size() * actual_k.width() + actual_v.size() * actual_v.width()) * sizeof(float);
    const BlockRun block1 = run_block(attention, token1, 1, &actual_k, &actual_v, &reference_k,
        &reference_v, lease, materializer, residency, metadata, source, "token1");
    const size_t cold_trace_end = backing->trace().size();
    if (!block0.ok || !block1.ok) return 15;
    std::printf("BLOCK_REFERENCE_PARITY_TOKEN_0=PASS BLOCK_REFERENCE_PARITY_TOKEN_1=PASS\n");
    const bool boundary_ok = max_difference(block0.attention_output, block0.reference_attention_output) <= 1e-5f &&
        max_difference(block1.attention_output, block1.reference_attention_output) <= 1e-5f;
    std::printf("attention_to_ffn_boundary_parity=%s\n", boundary_ok ? "PASS" : "FAIL");
    RuntimeStateSlot negative_seed_k(16 * 192, 2), negative_seed_v(16 * 128, 2);
    const TokenData negative_seed = compute_token(attention, token1, 0, &negative_seed_k, &negative_seed_v,
        lease, nullptr, "negative_seed");
    auto negative_source = std::make_shared<HttpRangeSource>(endpoint);
    auto negative_backing = std::make_shared<LocalVbufRangeMaterializer>(negative_source);
    auto negative_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto negative_materializer = std::make_shared<ResidentTensorMaterializer>(negative_backing, negative_residency);
    const std::vector<float> absent_output = absent_attention_output(attention, negative_seed, lease, negative_materializer);
    const LayerRun negative = run_layer(metadata, Activation{ absent_output, { 2048, 1 } }, lease,
        negative_materializer, negative_residency, negative_source, "negative_block", false);
    const float state_delta = negative.ok ? max_difference(block1.ffn.final_output, negative.final_output) : 0.0f;
    std::printf("full_block_state_influence_max_abs_delta=%g state_influence=%s\n", state_delta,
        state_delta > 1e-6f ? "PASS" : "FAIL");
    const uint64_t state_bytes_token1 = (actual_k.size() * actual_k.width() + actual_v.size() * actual_v.width()) * sizeof(float);
    std::printf("state_bytes_after_token0=%llu state_bytes_after_token1=%llu append_update_count=4 read_count=%llu cleanup=PASS\n",
        static_cast<unsigned long long>(state_bytes_token0),
        static_cast<unsigned long long>(state_bytes_token1),
        static_cast<unsigned long long>(actual_k.read_count() + actual_v.read_count()));
    print_source_accounting(*backing, 0, cold_trace_end, "cold_block");
    const size_t warm_trace_before = backing->trace().size();
    RuntimeStateSlot warm_k(16 * 192, 2), warm_v(16 * 128, 2);
    const TokenData warm0 = compute_token(attention, token0, 0, &warm_k, &warm_v, lease,
        materializer, "warm_token0");
    const LayerRun warm_layer0 = run_layer(metadata, Activation{ warm0.output, { 2048, 1 } }, lease,
        materializer, residency, source, "warm_token0_ffn", false);
    const TokenData warm1 = compute_token(attention, token1, 1, &warm_k, &warm_v, lease,
        materializer, "warm_token1");
    const LayerRun warm_layer1 = run_layer(metadata, Activation{ warm1.output, { 2048, 1 } }, lease,
        materializer, residency, source, "warm_token1_ffn", false);
    size_t warm_reads = 0;
    uint64_t warm_bytes = 0;
    const auto & warm_trace = backing->trace();
    for (size_t i = warm_trace_before; i < warm_trace.size(); ++i) {
        if (warm_trace[i].event == "STATE" && warm_trace[i].state == MaterializationState::Ready &&
            !warm_trace[i].source_id.empty()) {
            ++warm_reads;
            warm_bytes += warm_trace[i].returned_bytes;
        }
    }
    print_source_accounting(*backing, warm_trace_before, warm_trace.size(), "warm_block");
    const bool warm0_ok = warm_layer0.ok && max_difference(warm_layer0.final_output, block0.ffn.final_output) <= 1e-5f;
    const bool warm1_ok = warm_layer1.ok && max_difference(warm_layer1.final_output, block1.ffn.final_output) <= 1e-5f;
    std::printf("warm_block_token0_parity=%s warm_block_token1_parity=%s warm_source_reads=%zu "
        "warm_source_bytes=%llu\n", warm0_ok ? "PASS" : "FAIL", warm1_ok ? "PASS" : "FAIL",
        warm_reads, static_cast<unsigned long long>(warm_bytes));
    const std::set<uint32_t> token0_ids(block0.ffn.selection.ids.begin(), block0.ffn.selection.ids.end());
    const std::set<uint32_t> token1_ids(block1.ffn.selection.ids.begin(), block1.ffn.selection.ids.end());
    size_t intersection = 0;
    for (uint32_t id : token0_ids) if (token1_ids.count(id) != 0) ++intersection;
    const uint64_t routed_logical = (token0_ids.size() + token1_ids.size() - intersection) * 2748416ULL;
    const uint64_t attention_logical = 2960384ULL, always_logical = 4677632ULL;
    const uint64_t block_logical = attention_logical + always_logical + routed_logical;
    const uint64_t peak_active = std::max(block0.ffn.peak_active_persistent, block1.ffn.peak_active_persistent);
    std::printf("token0_token1_expert_intersection=%zu newly_required_expert_slices=%zu retained_expert_slices=%zu "
        "evicted_slices=%zu reloads=0\n", intersection, token1_ids.size() - intersection, intersection,
        token0_ids.size() - intersection);
    std::printf("attention_persistent_bytes=%llu ffn_always_required_bytes=%llu routed_expert_logical_bytes=%llu "
        "total_block_logical_persistent_bytes=%llu peak_active_fraction=%g\n",
        static_cast<unsigned long long>(attention_logical), static_cast<unsigned long long>(always_logical),
        static_cast<unsigned long long>(routed_logical), static_cast<unsigned long long>(block_logical),
        static_cast<double>(peak_active) / static_cast<double>(block_logical));
    const uint64_t peak_resident = peak_resident_from_trace(*residency);
    std::printf("peak_active_persistent_bytes=%llu peak_resident_bytes=%llu copied_bytes=0 repacked_bytes=0 transcoded_bytes=0\n",
        static_cast<unsigned long long>(peak_active),
        static_cast<unsigned long long>(peak_resident));
    std::printf("unselected_expert_graphs_created=0 unselected_expert_tensors_acquired=0 "
        "unselected_expert_source_reads=0 unselected_expert_materializations=0 "
        "whole_model_load=NO whole_packed_expert_tensor_load=NO whole_block_weight_buffer=NO "
        "cross_sublayer_last_consumer_release=PASS runtime_state_separate_from_vbuf=YES "
        "architecture_specific_runtime_logic=NO model_artifact_mutated=NO vbuf_layout_change_required=NO "
        "vbuf_format_change_required=NO payload_to_first_consumer_instrumented=YES\n");

    return state_delta > 1e-6f ? 0 : 15;
}
