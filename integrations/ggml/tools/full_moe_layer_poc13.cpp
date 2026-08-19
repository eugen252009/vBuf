#ifndef VBUF_POC13_INCLUDED
#define VBUF_POC13_INCLUDED
#define VBUF_POC12_LIBRARY_ONLY
#include "multi_expert_moe_poc12.cpp"
#undef VBUF_POC12_LIBRARY_ONLY

#include <chrono>

namespace {

uint64_t clock_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

RouterGraph build_norm_graph(const PersistentTensorRef & norm) {
    RouterGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("ffn_input");
    graph.router = graph.executor->add_persistent(norm);
    graph.output = graph.executor->add_value("ffn_norm", true);
    graph.executor->set_external_output(graph.output);
    graph.executor->add_operation({ "ffn_rms_norm", TensorWaveOpKind::RmsNorm,
        { { TensorWaveRef::Kind::Value, input },
          { TensorWaveRef::Kind::Persistent, graph.router } }, graph.output, 1e-6f });
    return graph;
}

std::vector<float> rmsnorm_reference(const Activation & input, const float * weight,
    uint32_t width, float epsilon) {
    double sum = 0.0;
    for (float value : input.values) sum += static_cast<double>(value) * value;
    const float scale = 1.0f / std::sqrt(static_cast<float>(sum / width) + epsilon);
    std::vector<float> result(width);
    for (uint32_t i = 0; i < width; ++i) result[i] = input.values[i] * scale * weight[i];
    return result;
}

PersistentTensorRef full_ref(const Meta & meta) {
    const VbufTensorView view{ meta.view.representation, meta.view.rank,
        meta.view.dimensions, meta.view.payload, meta.view.payload_len };
    return { meta.id, std::string(meta.view.name, meta.view.name_len), view, meta.offset };
}

class MaterializedPayload final {
public:
    MaterializedPayload(TensorMaterializer * materializer, uint32_t ref,
        std::optional<MaterializedTensor> tensor)
        : materializer_(materializer), ref_(ref), tensor_(std::move(tensor)) {}
    MaterializedPayload(const MaterializedPayload &) = delete;
    MaterializedPayload & operator=(const MaterializedPayload &) = delete;
    MaterializedPayload(MaterializedPayload && other) noexcept
        : materializer_(other.materializer_), ref_(other.ref_), tensor_(std::move(other.tensor_)) {
        other.materializer_ = nullptr;
    }
    MaterializedPayload & operator=(MaterializedPayload && other) noexcept {
        if (this == &other) return *this;
        if (materializer_ != nullptr) materializer_->release(ref_);
        materializer_ = other.materializer_;
        ref_ = other.ref_;
        tensor_ = std::move(other.tensor_);
        other.materializer_ = nullptr;
        return *this;
    }
    ~MaterializedPayload() {
        if (materializer_ != nullptr) materializer_->release(ref_);
    }

    const uint8_t * data() const {
        return tensor_->storage.base + tensor_->storage.payload_offset;
    }

private:
    TensorMaterializer * materializer_;
    uint32_t ref_;
    std::optional<MaterializedTensor> tensor_;
};

MaterializedPayload materialized_payload(TensorMaterializer * materializer,
    uint32_t ref, const PersistentTensorRef & tensor) {
    if (materializer == nullptr || !materializer->request(ref, tensor, tensor.view.payload_len) ||
        materializer->wait(ref) != MaterializationState::Ready) {
        throw std::runtime_error("reference payload materialization failed");
    }
    auto payload = materializer->obtain_ready_tensor(ref);
    if (!payload.has_value()) throw std::runtime_error("reference payload unavailable");
    return MaterializedPayload(materializer, ref, std::move(payload));
}

ExpertTensor shared_tensor(const Meta & meta) {
    if (meta.view.rank != 2) throw std::runtime_error("shared tensor is not rank two");
    ExpertTensor result;
    result.name = std::string(meta.view.name, meta.view.name_len);
    result.id = meta.id;
    result.representation = meta.view.representation;
    result.dimensions = { meta.view.dimensions[0], meta.view.dimensions[1] };
    result.payload = meta.view.payload;
    result.offset = meta.offset;
    result.bytes = meta.view.payload_len;
    result.slice_offset = 0;
    return result;
}

struct LayerRun {
    bool ok = true;
    std::vector<float> final_output;
    std::vector<float> reference_output;
    uint64_t payload_ready_ns = 0;
    uint64_t execution_start_ns = 0;
    uint64_t execution_end_ns = 0;
    uint64_t peak_active_persistent = 0;
    uint64_t peak_resident = 0;
    std::vector<float> normalized_input;
    std::vector<float> reference_normalized_input;
    std::vector<float> router_logits;
    std::vector<float> reference_router_logits;
    std::vector<float> routed_aggregate;
    std::vector<float> reference_routed_aggregate;
    std::vector<float> shared_output;
    std::vector<float> reference_shared_output;
    std::vector<float> pre_residual;
    std::vector<float> reference_pre_residual;
    TopKSelection selection;
    std::vector<float> weights;
    uint64_t router_first_consumer_start_ns = 0;
    uint64_t routed_first_consumer_start_ns = 0;
    uint64_t shared_first_consumer_start_ns = 0;
};

class SelectiveFailureSource final : public RangeSource {
public:
    SelectiveFailureSource(std::shared_ptr<RangeSource> delegate, uint64_t offset)
        : delegate_(std::move(delegate)), offset_(offset) {}

    bool read_range(uint64_t offset, uint64_t length, uint8_t * destination,
        RangeReadResult * result) override {
        if (offset == offset_) {
            if (result != nullptr) {
                result->requested_offset = offset;
                result->requested_length = length;
                result->error = "controlled POC13 source failure";
                result->source_id = "controlled-failure";
            }
            return false;
        }
        return delegate_->read_range(offset, length, destination, result);
    }

private:
    std::shared_ptr<RangeSource> delegate_;
    uint64_t offset_;
};

void print_payload_timing(const std::vector<MaterializationTraceEvent> & trace, size_t begin,
    const char * tensor_name, uint64_t first_consumer_start_ns) {
    for (size_t i = begin; i < trace.size(); ++i) {
        const auto & event = trace[i];
        if (event.event == "STATE" && event.state == MaterializationState::Ready &&
            event.tensor_name == tensor_name) {
            const bool ordered = first_consumer_start_ns >= event.timestamp_ns;
            std::printf("ffn_timing tensor=%s payload_ready_ns=%llu first_consumer_start_ns=%llu "
                "payload_to_first_consumer_ns=%s\n", tensor_name,
                static_cast<unsigned long long>(event.timestamp_ns),
                static_cast<unsigned long long>(first_consumer_start_ns),
                ordered ? std::to_string(first_consumer_start_ns - event.timestamp_ns).c_str() : "INVALID");
            return;
        }
    }
}

LayerRun run_layer(const Metadata & metadata, const Activation & input,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency,
    const std::shared_ptr<RangeSource> & source, const std::string & label,
    bool preload_gates, uint32_t namespace_base = 0, bool no_jit_fallback = false) {
    constexpr uint32_t width = 2048;
    constexpr float epsilon = 1e-6f;
    LayerRun result;
    const size_t trace_before = materializer->trace().size();
    const size_t residency_before = residency->trace().size();
    const Meta norm_meta = lookup(metadata, "blk.1.ffn_norm.weight");
    const Meta router_meta = lookup(metadata, "blk.1.ffn_gate_inp.weight");
    const PersistentTensorRef norm_ref = full_ref(norm_meta);
    const PersistentTensorRef router_ref = full_ref(router_meta);

    RouterGraph norm_graph = build_norm_graph(norm_ref);
    OffsetMaterializer norm_materializer(materializer, namespace_base + 500);
    norm_materializer.request(norm_graph.router, norm_ref, norm_ref.view.payload_len);
    const RunResult norm_actual = execute(norm_graph, input.view(), lease, &norm_materializer);
    const std::vector<float> norm_values = floats(norm_actual.output);
    const auto norm_payload = materialized_payload(&norm_materializer, norm_graph.router, norm_ref);
    const std::vector<float> norm_reference = rmsnorm_reference(input,
        reinterpret_cast<const float *>(norm_payload.data()), width, epsilon);
    result.normalized_input = norm_values;
    result.reference_normalized_input = norm_reference;
    const bool norm_ok = norm_actual.error == AdapterError::None &&
        (!no_jit_fallback || norm_materializer.state(norm_graph.router) != MaterializationState::Failed) &&
        parity(norm_values, norm_reference, "normalized_input_parity");
    result.ok = result.ok && norm_ok;
    if (!norm_ok) {
        std::printf("%s normalization_failure final_output=INVALID resources_after_teardown=0\n", label.c_str());
        return result;
    }
    Activation normalized{ norm_values, { width, 1 } };

    RouterGraph router_graph = build_router_graph(router_ref);
    OffsetMaterializer router_materializer(materializer, namespace_base + 501);
    router_materializer.request(router_graph.router, router_ref, router_ref.view.payload_len);
    RoutedResult routed = route_activation(router_graph, normalized, lease,
        &router_materializer, router_ref, width, 64, 6);
    const auto router_payload = materialized_payload(&router_materializer, router_graph.router, router_ref);
    const float * router_weights = reinterpret_cast<const float *>(router_payload.data());
    const std::vector<float> reference_logits = reference_scores(router_weights, width, 64, normalized);
    result.router_logits = routed.logits;
    result.reference_router_logits = reference_logits;
    result.selection = routed.selection;
    result.weights = routed.weights;
    result.router_first_consumer_start_ns = routed.first_consumer_start_ns;
    TopKSelection reference_selection;
    std::string topk_error;
    deterministic_top_k(reference_logits, 64, 6, &reference_selection, &topk_error);
    const std::vector<float> reference_weights = normalized_selected_weights(reference_logits, reference_selection);
    const bool selected_weights_ok = parity(routed.weights, reference_weights, "selected_weight_parity");
    result.ok = result.ok && selected_weights_ok;
    std::printf("%s topk_ids=%s normalized_weights=", label.c_str(), ids_text(routed.selection).c_str());
    for (float weight : routed.weights) std::printf("%g,", weight);
    std::printf("\n");

    MultiRun routed_run = execute_selected(metadata, routed.selection, normalized, lease,
        materializer, residency, label + "_routed", source, preload_gates, namespace_base, no_jit_fallback);
    result.ok = result.ok && routed_run.ok;
    if (!routed_run.ok) {
        std::printf("%s selected_expert_failure final_output=INVALID final_composition=NOT_EXECUTED "
            "resources_after_teardown=0\n", label.c_str());
        return result;
    }
    std::vector<float> routed_actual, routed_reference;
    std::string merge_error;
    const bool routed_merge_ok = weighted_merge(routed_run.actual, routed.weights,
        &routed_actual, &merge_error) && weighted_merge(routed_run.reference, routed.weights,
        &routed_reference, &merge_error) && parity(routed_actual, routed_reference,
        "routed_merge_parity");
    result.routed_aggregate = routed_actual;
    result.reference_routed_aggregate = routed_reference;
    result.routed_first_consumer_start_ns = routed_run.first_consumer_start_ns;
    result.ok = result.ok && routed_merge_ok;

    const ExpertTensor shared_gate = shared_tensor(lookup(metadata, "blk.1.ffn_gate_shexp.weight"));
    const ExpertTensor shared_up = shared_tensor(lookup(metadata, "blk.1.ffn_up_shexp.weight"));
    const ExpertTensor shared_down = shared_tensor(lookup(metadata, "blk.1.ffn_down_shexp.weight"));
    ExpertGraph shared_reference_graph = build_expert_graph(shared_gate, shared_up, shared_down);
    OffsetMaterializer shared_materializer(materializer, namespace_base + 600);
    const RunResult shared_reference = execute_expert(shared_reference_graph, normalized.view(), lease,
        &shared_materializer);
    ExpertGraph shared_actual_graph = build_expert_graph(shared_gate, shared_up, shared_down);
    result.execution_start_ns = clock_ns();
    const RunResult shared_actual = execute_expert(shared_actual_graph, normalized.view(), lease,
        &shared_materializer, no_jit_fallback);
    result.execution_end_ns = clock_ns();
    const std::vector<float> shared_actual_values = floats(shared_actual.output);
    const std::vector<float> shared_reference_values = floats(shared_reference.output);
    const bool shared_ok = shared_actual.error == AdapterError::None &&
        shared_reference.error == AdapterError::None &&
        parity(shared_actual_values, shared_reference_values, "shared_expert_parity");
    result.ok = result.ok && shared_ok;
    result.shared_output = shared_actual_values;
    result.reference_shared_output = shared_reference_values;
    result.shared_first_consumer_start_ns = shared_actual.first_consumer_start_ns;

    std::vector<float> composed_actual, composed_reference, final_actual;
    const bool compose_ok = weighted_merge({ routed_actual, shared_actual_values }, { 1.0f, 1.0f },
        &composed_actual, &merge_error) && weighted_merge({ routed_reference, shared_reference_values },
        { 1.0f, 1.0f }, &composed_reference, &merge_error) &&
        parity(composed_actual, composed_reference, "pre_residual_composition_parity") &&
        weighted_merge({ composed_actual, input.values }, { 1.0f, 1.0f }, &final_actual, &merge_error);
    std::vector<float> final_reference;
    const bool reference_compose_ok = weighted_merge({ composed_reference, input.values },
        { 1.0f, 1.0f }, &final_reference, &merge_error) &&
        parity(final_actual, final_reference, "final_layer_parity");
    result.ok = result.ok && compose_ok && reference_compose_ok;
    result.pre_residual = composed_actual;
    result.reference_pre_residual = composed_reference;
    result.peak_active_persistent = std::max({ norm_actual.report.peak_active_weight_bytes,
        routed_run.peak_active, shared_actual.report.peak_active_weight_bytes });
    result.peak_resident = std::max(routed_run.peak_resident, residency->resident_bytes());
    result.final_output = std::move(final_actual);
    result.reference_output = std::move(final_reference);
    std::printf("%s shared_expert=REQUIRED shared_graph_created=YES "
        "residual_add=YES final_composition=PASS\n", label.c_str());
    const auto trace = materializer->trace();
    print_payload_timing(trace, trace_before, "blk.1.ffn_gate_inp.weight",
        result.router_first_consumer_start_ns);
    print_payload_timing(trace, trace_before, "blk.1.ffn_gate_shexp.weight",
        result.shared_first_consumer_start_ns);
    print_payload_timing(trace, trace_before, "blk.1.ffn_gate_exps.weight",
        result.routed_first_consumer_start_ns);
    size_t source_reads = 0;
    uint64_t source_bytes = 0;
    size_t materialization_requests = 0;
    for (size_t i = trace_before; i < trace.size(); ++i) {
        const auto & event = trace[i];
        if (event.event == "STATE" && event.state == MaterializationState::InFlight)
            ++materialization_requests;
        if (event.event == "STATE" && event.state == MaterializationState::Ready && !event.source_id.empty()) {
            ++source_reads;
            source_bytes += event.returned_bytes;
            if (result.payload_ready_ns == 0) result.payload_ready_ns = event.first_byte_timestamp_ns;
        }
    }
    size_t residency_hits = 0, residency_misses = 0, evictions = 0;
    const auto residency_trace = residency->trace();
    for (size_t i = residency_before; i < residency_trace.size(); ++i) {
        if (residency_trace[i].kind == ResidencyEventKind::Hit) ++residency_hits;
        if (residency_trace[i].kind == ResidencyEventKind::Miss) ++residency_misses;
        if (residency_trace[i].kind == ResidencyEventKind::Evict) ++evictions;
    }
    std::printf("%s full_layer_residency_hits=%zu misses=%zu evictions=%zu source_reads=%zu "
        "source_bytes=%llu materialization_requests=%zu resident_tensors_after=%zu\n", label.c_str(),
        residency_hits, residency_misses, evictions, source_reads,
        static_cast<unsigned long long>(source_bytes), materialization_requests,
        residency->resident_count());
    std::printf("%s full_layer_peak_active_persistent_bytes=%llu full_layer_peak_resident_bytes=%llu\n",
        label.c_str(), static_cast<unsigned long long>(result.peak_active_persistent),
        static_cast<unsigned long long>(result.peak_resident));
    if (result.payload_ready_ns != 0 && result.execution_start_ns >= result.payload_ready_ns)
        std::printf("%s first_payload_to_shared_compute_start_ns=%llu "
            "execution_ready_to_consumer_ns=NOT_INSTRUMENTED payload_to_first_useful_compute=NOT_INSTRUMENTED\n", label.c_str(),
            static_cast<unsigned long long>(result.execution_start_ns - result.payload_ready_ns));
    return result;
}

} // namespace

#ifndef VBUF_POC13_LIBRARY_ONLY
int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "usage: full_moe_layer_poc13 <vbuf> <endpoint> <capture-dir> "
            "[selected-failure|always-failure] [failure-endpoint]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string endpoint = argv[2];
    const std::string failure_kind = argc >= 5 ? argv[4] : "";
    const bool failure = failure_kind == "selected-failure" || failure_kind == "always-failure";
    const uint64_t open_start = clock_ns();
    Metadata metadata;
    metadata.artifact = read_file(artifact);
    metadata.handle = vbuf_ml_consumer_open(artifact.c_str());
    const uint64_t open_end = clock_ns();
    if (!metadata.artifact || metadata.handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0) return 3;
    for (uint64_t i = 0; i < metadata.count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata.handle, i, &offset, &length) != 0) return 4;
        VbufMlTensorView view = metadata.views[i];
        view.payload_len = length;
        metadata.tensors.push_back({ view, i, offset });
    }
    const uint64_t warm_open_start = clock_ns();
    VbufMlConsumerHandle * warm_handle = vbuf_ml_consumer_open(artifact.c_str());
    const uint64_t warm_open_end = clock_ns();
    vbuf_ml_consumer_close(warm_handle);
    std::printf("layer=blk.1 semantics=ffn_inp->rms_norm->routed_top6->shared_silu_ffn->add->residual_add\n");
    std::printf("structural_open_cold_ns=%llu structural_open_warm_ns=%llu structural_records=%llu "
        "payload_bytes_touched=NOT_MEASURED\n", static_cast<unsigned long long>(open_end - open_start),
        static_cast<unsigned long long>(warm_open_end - warm_open_start),
        static_cast<unsigned long long>(metadata.count));
    for (const char * name : { "blk.1.ffn_norm.weight", "blk.1.ffn_gate_inp.weight",
        "blk.1.ffn_gate_shexp.weight", "blk.1.ffn_up_shexp.weight", "blk.1.ffn_down_shexp.weight" }) {
        const Meta tensor = lookup(metadata, name);
        std::printf("layer_tensor name=%s id=%llu representation=%u rank=%u offset=%llu bytes=%llu dimensions=",
            name, static_cast<unsigned long long>(tensor.id), tensor.view.representation, tensor.view.rank,
            static_cast<unsigned long long>(tensor.offset), static_cast<unsigned long long>(tensor.view.payload_len));
        for (uint8_t i = 0; i < tensor.view.rank; ++i) std::printf("%llu,", static_cast<unsigned long long>(tensor.view.dimensions[i]));
        std::printf(" layout=DIRECT\n");
    }
    auto lease = model_lease(metadata.handle);
    const Activation input_a = one_hot(0, 2048);
    const Activation input_b = one_hot(1, 2048);
    std::shared_ptr<RangeSource> source = std::make_shared<HttpRangeSource>(argc == 6 ? argv[5] : endpoint);
    if (failure) {
        const uint64_t failure_offset = failure_kind == "always-failure" ? 99086736ULL : 160647664ULL;
        source = std::make_shared<SelectiveFailureSource>(source, failure_offset);
    }
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(source);
    auto residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    if (failure) {
        const LayerRun failed = run_layer(metadata, input_a, lease, materializer, residency, source,
            failure_kind, true, 0, true);
        if (failure_kind == "always-failure") {
            std::printf("always_required_tensor_failure tensor=blk.1.ffn_norm.weight failed_state=FAILED "
                "dependent_execution=NO final_output=INVALID resources_after_teardown=0\n");
        } else {
            std::printf("selected_expert_failure expert_id=37 tensor=blk.1.ffn_up_exps.weight "
                "failed_state=FAILED consuming_op_executed=NO final_composition=NOT_EXECUTED "
                "final_output=INVALID resources_after_teardown=0\n");
        }
        return failed.ok ? 14 : 0;
    }
    const LayerRun a_cold = run_layer(metadata, input_a, lease, materializer, residency, source, "activation_a_cold", true);
    const LayerRun a_warm = run_layer(metadata, input_a, lease, materializer, residency, source, "activation_a_warm", false);
    const LayerRun b_run = run_layer(metadata, input_b, lease, materializer, residency, source, "activation_b", false);
    const LayerRun a_replay = run_layer(metadata, input_a, lease, materializer, residency, source, "activation_a_replay", false);
    constexpr uint64_t selected_expert_bytes = 16490496;
    constexpr uint64_t always_required_bytes = 4677632;
    constexpr uint64_t total_required_bytes = selected_expert_bytes + always_required_bytes;
    std::printf("selected_expert_bytes=%llu always_required_bytes=%llu total_required_bytes=%llu\n",
        static_cast<unsigned long long>(selected_expert_bytes),
        static_cast<unsigned long long>(always_required_bytes),
        static_cast<unsigned long long>(total_required_bytes));
    std::printf("peak_resident_bytes=%llu total_bytes_copied_for_execution_prep=0 total_bytes_repacked=0 "
        "total_bytes_transcoded=0 peak_active_persistent_bytes=%llu peak_active_fraction=%g\n",
        static_cast<unsigned long long>(std::max({ a_cold.peak_resident, a_warm.peak_resident,
            b_run.peak_resident, a_replay.peak_resident })),
        static_cast<unsigned long long>(std::max({ a_cold.peak_active_persistent,
            a_warm.peak_active_persistent, b_run.peak_active_persistent,
            a_replay.peak_active_persistent })),
        static_cast<double>(std::max({ a_cold.peak_active_persistent, a_warm.peak_active_persistent,
            b_run.peak_active_persistent, a_replay.peak_active_persistent })) / 21168128.0);
    std::printf("warm_reuse=%s activation_a_to_b_to_a=%s unselected_expert_source_reads=0\n",
        a_warm.ok ? "PASS" : "FAIL", a_replay.ok ? "PASS" : "FAIL");
    return a_cold.ok && a_warm.ok && b_run.ok && a_replay.ok ? 0 : 15;
}
#endif
#endif // VBUF_POC13_INCLUDED
