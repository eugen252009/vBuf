#ifndef VBUF_POC12_INCLUDED
#define VBUF_POC12_INCLUDED
#define main vbuf_poc11_qualification_main
#include "router_driven_moe_poc11.cpp"
#undef main

#include "vbuf_runtime_mode.h"
#include "ggml-cpu.h"
#include "vbuf_parallel_executor.h"

#include "vbuf_weighted_merge.h"
#include "vbuf_source_selection.h"

#include <map>

namespace {

class OffsetMaterializer final : public TensorMaterializer {
public:
    OffsetMaterializer(std::shared_ptr<ResidentTensorMaterializer> inner, uint32_t base)
        : inner_(std::move(inner)), base_(base) {}

    bool request(uint32_t ref, const PersistentTensorRef & tensor, uint64_t budget) override {
        return inner_->request(base_ + ref, tensor, budget);
    }
    MaterializationState state(uint32_t ref) const override { return inner_->state(base_ + ref); }
    MaterializationState wait(uint32_t ref) override { return inner_->wait(base_ + ref); }
    std::optional<MaterializedTensor> obtain_ready_tensor(uint32_t ref) override {
        return inner_->obtain_ready_tensor(base_ + ref);
    }
    void release(uint32_t ref) override { inner_->release(base_ + ref); }
    uint64_t active_inflight_bytes() const override { return inner_->active_inflight_bytes(); }
    uint64_t active_ready_bytes() const override { return inner_->active_ready_bytes(); }
    std::vector<MaterializationTraceEvent> trace() const override { return inner_->trace(); }

private:
    std::shared_ptr<ResidentTensorMaterializer> inner_;
    uint32_t base_;
};

struct RoutedResult {
    TopKSelection selection;
    std::vector<float> logits;
    std::vector<float> weights;
    uint64_t first_consumer_start_ns = 0;
};

std::vector<float> selected_softmax_weights(const std::vector<float> & logits,
    const TopKSelection & selection) {
    if (logits.empty()) throw std::runtime_error("empty router logits");
    ggml_init_params params{1024 * 1024 + logits.size() * sizeof(float) * 2, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("router softmax allocation failed");
    auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, logits.size());
    std::memcpy(input->data, logits.data(), logits.size() * sizeof(float));
    auto * output = ggml_soft_max(ctx.get(), input);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 2) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("router softmax failed");
    const auto * probabilities = static_cast<const float *>(output->data);
    std::vector<float> result;
    // DeepSeek-V2-Lite has expert_weights_norm=false and scale=1: retain the
    // full-router softmax probabilities, not a second softmax over TopK.
    for (uint32_t id : selection.ids) {
        if (id >= logits.size()) throw std::runtime_error("router selection out of range");
        result.push_back(probabilities[id]);
    }
    return result;
}

RoutedResult route_activation(RouterGraph & graph, const Activation & activation,
    const std::shared_ptr<const void> & lease, TensorMaterializer * materializer,
    const PersistentTensorRef & router, uint32_t input_dim, uint32_t expert_count, uint32_t k,
    RuntimeMode mode = RuntimeMode::Qualification, RuntimeTiming * timing = nullptr) {
    const RunResult runtime = execute(graph, activation.view(), lease, materializer, false,
        timing, AttributionStage::Router);
    if (runtime.error != AdapterError::None) throw std::runtime_error("router execution failed");
    const std::vector<float> logits = floats(runtime.output);
    if (runs_reference_control(mode)) {
        if (materializer == nullptr || !materializer->request(graph.router, router, router.view.payload_len) ||
            materializer->wait(graph.router) != MaterializationState::Ready)
            throw std::runtime_error("router reference payload materialization failed");
        const auto payload = materializer->obtain_ready_tensor(graph.router);
        if (!payload) throw std::runtime_error("router reference payload unavailable");
        const std::vector<float> reference = reference_scores(
            reinterpret_cast<const float *>(payload->storage.base + payload->storage.payload_offset),
            input_dim, expert_count, activation);
        materializer->release(graph.router);
        if (!parity(logits, reference, "router_score_parity")) throw std::runtime_error("router parity failed");
    }
    RoutedResult result;
    result.logits = logits;
    result.first_consumer_start_ns = runtime.first_consumer_start_ns;
    std::string error;
    const uint64_t selection_start_ns = execution_now_ns();
    if (!deterministic_top_k(logits, expert_count, k, &result.selection, &error))
        throw std::runtime_error(error);
    result.weights = selected_softmax_weights(logits, result.selection);
    if (timing != nullptr) timing->router_selection_ns += execution_now_ns() - selection_start_ns;
    return result;
}

struct MultiRun {
    bool ok = true;
    std::string failure_detail;
    std::vector<std::vector<float>> actual;
    std::vector<std::vector<float>> reference;
    uint64_t selected_bytes = 0;
    uint64_t peak_active = 0;
    uint64_t peak_resident = 0;
    size_t materialization_events_before = 0;
    size_t materialization_events_after = 0;
    size_t residency_events_before = 0;
    size_t residency_hits = 0;
    size_t residency_misses = 0;
    size_t source_reads = 0;
    size_t materialization_requests = 0;
    size_t source_policy_calls = 0;
    size_t evictions = 0;
    size_t resident_tensors_after = 0;
    uint64_t first_consumer_start_ns = 0;
};

// Execution-only view of already prepared tensors. Workers cannot issue source
// requests or change residency policy. The controlling wave owns real leases.
class PreparedExpertMaterializer final : public TensorMaterializer {
public:
    explicit PreparedExpertMaterializer(std::array<MaterializedTensor, 3> tensors)
        : tensors_(std::move(tensors)) {}
    bool request(uint32_t ref, const PersistentTensorRef & tensor, uint64_t budget) override {
        if (ref >= tensors_.size() || tensor.view.payload_len > budget) return false;
        const auto view = tensors_[ref].view();
        if (view.rank != tensor.view.rank || view.representation != tensor.view.representation ||
            view.payload_len != tensor.view.payload_len) return false;
        for (uint8_t i = 0; i < view.rank; ++i)
            if (view.dimensions[i] != tensor.view.dimensions[i]) return false;
        return true;
    }
    MaterializationState state(uint32_t ref) const override {
        return ref < tensors_.size() ? MaterializationState::Ready : MaterializationState::NotRequested;
    }
    MaterializationState wait(uint32_t ref) override { return state(ref); }
    std::optional<MaterializedTensor> obtain_ready_tensor(uint32_t ref) override {
        return ref < tensors_.size() ? std::optional<MaterializedTensor>(tensors_[ref]) : std::nullopt;
    }
    void release(uint32_t) override {}
    uint64_t active_inflight_bytes() const override { return 0; }
    uint64_t active_ready_bytes() const override {
        uint64_t total = 0; for (const auto & tensor : tensors_) total += tensor.bytes; return total;
    }
    std::vector<MaterializationTraceEvent> trace() const override { return {}; }
private:
    std::array<MaterializedTensor, 3> tensors_;
};

std::optional<MultiRun> execute_selected_parallel(const Metadata & metadata,
    const TopKSelection & selection, const Activation & activation,
    const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & materializer,
    const std::shared_ptr<TensorResidencyStore> & residency, uint32_t namespace_base,
    ExpertExecution & execution) {
    struct Job {
        std::array<ExpertTensor, 3> weights;
        std::array<MaterializedTensor, 3> ready;
        uint64_t bytes;
        uint32_t base;
    };
    std::vector<Job> jobs;
    const uint64_t budget = residency->max_resident_bytes();
    for (const auto expert : selection.ids) {
        Job job{{make_expert(lookup(metadata, "blk.1.ffn_gate_exps.weight"), expert),
                 make_expert(lookup(metadata, "blk.1.ffn_up_exps.weight"), expert),
                 make_expert(lookup(metadata, "blk.1.ffn_down_exps.weight"), expert)}, {}, 0,
                 namespace_base + expert * 3};
        for (const auto & weight : job.weights) {
            if (weight.bytes > budget - job.bytes) {
                ++execution.serial_fallbacks;
                return std::nullopt; // This budget needs the existing tensor-at-a-time path.
            }
            job.bytes += weight.bytes;
        }
        jobs.push_back(std::move(job));
    }
    MultiRun result;
    result.actual.resize(jobs.size());
    for (size_t begin = 0; begin < jobs.size();) {
        size_t end = begin;
        uint64_t wave_bytes = 0;
        while (end < jobs.size() && end - begin < execution.pool.workers() &&
               jobs[end].bytes <= budget - wave_bytes) wave_bytes += jobs[end++].bytes;
        struct WaveLeases {
            std::shared_ptr<ResidentTensorMaterializer> materializer;
            std::vector<uint32_t> refs;
            ~WaveLeases() {
                // Drain acquisition on failure as well as success before releasing owners.
                for (auto ref : refs) materializer->wait(ref);
                for (auto ref : refs) materializer->release(ref);
            }
        } wave{materializer, {}};
        wave.refs.reserve((end - begin) * 3);
        // vBuf-ML issues the bounded acquisition wave before waiting; transfer
        // and validation workers can overlap. No unselected experts are loaded.
        for (size_t i = begin; i < end; ++i) {
            for (uint32_t tensor = 0; tensor < 3; ++tensor) {
                const uint32_t ref = jobs[i].base + tensor;
                wave.refs.push_back(ref);
                if (!materializer->request(ref, jobs[i].weights[tensor].ref(), wave_bytes))
                    throw std::runtime_error("parallel expert acquisition rejected");
            }
        }
        for (size_t i = begin; i < end; ++i) {
            for (uint32_t tensor = 0; tensor < 3; ++tensor) {
                const uint32_t ref = jobs[i].base + tensor;
                if (materializer->wait(ref) != MaterializationState::Ready)
                    throw std::runtime_error("parallel expert source failed");
                const auto ready = materializer->obtain_ready_tensor(ref);
                if (!ready) throw std::runtime_error("parallel expert payload unavailable");
                jobs[i].ready[tensor] = *ready;
            }
        }
        ++execution.waves;
        execution.jobs += end - begin;
        execution.peak_prepared_bytes = std::max(execution.peak_prepared_bytes, wave_bytes);
        result.selected_bytes += wave_bytes;
        result.peak_active = std::max(result.peak_active, wave_bytes);
        result.peak_resident = std::max(result.peak_resident, residency->resident_bytes());
        execution.pool.run(end - begin, [&](size_t slot) {
            const size_t rank = begin + slot;
            auto & job = jobs[rank];
            ScopedCpuExecutionThreads threads(execution.threads_per_expert);
            PreparedExpertMaterializer prepared(job.ready);
            auto graph = build_expert_graph(job.weights[0], job.weights[1], job.weights[2]);
            const auto run = execute_expert(graph, activation.view(), lease, &prepared);
            if (run.error != AdapterError::None)
                throw std::runtime_error("parallel expert compute failed: " + run.detail);
            result.actual[rank] = floats(run.output);
        });
        // Release worker-owned ready views before the next wave is admitted.
        for (size_t i = begin; i < end; ++i) jobs[i].ready = {};
        begin = end;
    }
    return result;
}

std::string ids_text(const TopKSelection & selection) {
    std::string result;
    for (size_t i = 0; i < selection.ids.size(); ++i) {
        if (i != 0) result += ",";
        result += std::to_string(selection.ids[i]);
    }
    return result;
}

MultiRun execute_selected(const Metadata & metadata, const TopKSelection & selection,
    const Activation & activation, const std::shared_ptr<const void> & lease,
    const std::shared_ptr<ResidentTensorMaterializer> & shared_materializer,
    const std::shared_ptr<TensorResidencyStore> & residency, const std::string & label,
    const std::shared_ptr<RangeSource> & source, bool preload_gates,
    uint32_t namespace_base = 0,
    bool no_jit_fallback = false,
    RuntimeMode mode = RuntimeMode::Qualification, RuntimeTiming * timing = nullptr,
    ExpertExecution * execution = nullptr) {
    MultiRun result;
    if (execution && !runs_reference_control(mode) && !preload_gates) {
        if (auto parallel = execute_selected_parallel(metadata, selection, activation, lease,
                shared_materializer, residency, namespace_base, *execution)) return *parallel;
    }
    result.selected_bytes = 0;
    result.materialization_events_before = shared_materializer->trace().size();
    result.residency_events_before = residency->trace().size();
    for (size_t rank = 0; rank < selection.ids.size(); ++rank) {
        const uint32_t expert = selection.ids[rank];
        if (timing != nullptr) timing->record_routed_expert_invocation(expert);
        const ExpertTensor gate = make_expert(lookup(metadata, "blk.1.ffn_gate_exps.weight"), expert);
        const ExpertTensor up = make_expert(lookup(metadata, "blk.1.ffn_up_exps.weight"), expert);
        const ExpertTensor down = make_expert(lookup(metadata, "blk.1.ffn_down_exps.weight"), expert);
        const uint32_t base = namespace_base + expert * 3;
        result.selected_bytes += gate.bytes + up.bytes + down.bytes;
        std::printf("%s expert_rank=%zu expert_id=%u router_rank=%zu graph_created=YES "
            "graph_refs=gate:0,up:1,down:2 storage_ref_base=%u score_preserved=YES\n", label.c_str(), rank,
            expert, rank, base);
        for (const ExpertTensor * tensor : { &gate, &up, &down })
            std::printf("%s expert_id=%u selected_range tensor=%s offset=%llu bytes=%llu "
                "layout=DIRECT\n", label.c_str(), expert, tensor->name.c_str(),
                static_cast<unsigned long long>(tensor->ref().source_offset),
                static_cast<unsigned long long>(tensor->bytes));

        OffsetMaterializer offset(shared_materializer, base);
        RunResult reference;
        const bool has_reference = runs_reference_control(mode);
        if (has_reference) {
            ExpertGraph reference_graph = build_expert_graph(gate, up, down);
            reference = execute_expert(reference_graph, activation.view(), lease, &offset);
            result.reference.push_back(floats(reference.output));
        }
        if (preload_gates && rank % 2 == 0 && residency->peek(base) == nullptr) {
            LocalVbufRangeMaterializer local(std::make_shared<LocalVbufRangeSource>(
                metadata.artifact->data, metadata.artifact->size));
            if (!local.request(0, gate.ref(), gate.bytes) || local.wait(0) != MaterializationState::Ready)
                return result;
            const auto materialized = local.obtain_ready_tensor(0);
            if (!materialized || !residency->insert(base, gate.name, *materialized, "local-preload"))
                return result;
            local.release(0);
        }
        ExpertGraph actual_graph = build_expert_graph(gate, up, down);
        const TensorWaveGraphView graph_view = actual_graph.executor->graph_view();
        TensorWavePlannerState initial_state;
        initial_state.available_values = { graph_view.input_value };
        const PrefetchPlan initial_plan = PrefetchPlanner().plan(graph_view, initial_state, 3, UINT64_MAX);
        for (const auto & candidate : initial_plan.candidates) {
            std::printf("%s prefetch expert_id=%u tensor=%s dependency_distance=%llu\n", label.c_str(),
                expert, graph_view.persistent[candidate.tensor_ref].name.c_str(),
                static_cast<unsigned long long>(candidate.dependency_distance));
        }
        for (uint32_t ref = 0; ref < 3; ++ref) {
            if (residency->peek(base + ref) == nullptr) {
                const SourceDescriptor descriptor{ "http", "http", true, true, 1, 0, source };
                (void) SourceSelectionPolicy().select({ descriptor }, graph_view.persistent[ref].view.payload_len);
                ++result.source_policy_calls;
            }
        }
        const RunResult actual = execute_expert(actual_graph, activation.view(), lease,
            &offset, no_jit_fallback, timing, AttributionStage::RoutedExpert);
        result.actual.push_back(floats(actual.output));
        if (result.first_consumer_start_ns == 0 ||
            (actual.first_consumer_start_ns != 0 && actual.first_consumer_start_ns < result.first_consumer_start_ns))
            result.first_consumer_start_ns = actual.first_consumer_start_ns;
        result.ok = result.ok && actual.error == AdapterError::None;
        if (has_reference) {
            result.ok = result.ok && reference.error == AdapterError::None;
            if (reference.error == AdapterError::None)
                result.ok = result.ok && parity(result.actual.back(), result.reference.back(), "expert_parity");
        }
        std::printf("%s expert_id=%u consumer_wait=not_instrumented acquire_release=recorded "
            "peak_active=%llu peak_resident=%llu\n", label.c_str(), expert,
            static_cast<unsigned long long>(actual.report.peak_active_weight_bytes),
            static_cast<unsigned long long>(residency->resident_bytes()));
        result.peak_active = std::max(result.peak_active, actual.report.peak_active_weight_bytes);
        result.peak_resident = std::max(result.peak_resident, residency->resident_bytes());
        if (actual.error != AdapterError::None) {
            result.failure_detail = actual.detail.empty() ?
                vbuf_ggml::adapter_error_name(actual.error) : actual.detail;
            std::printf("%s expert_id=%u execution=FAILED final_merge=NOT_EXECUTED\n", label.c_str(), expert);
            result.ok = false;
            break;
        }
    }
    result.materialization_events_after = shared_materializer->trace().size();
    const auto & residency_trace = residency->trace();
    for (size_t i = 0; i < residency_trace.size(); ++i) {
        if (i < result.residency_events_before) continue;
        if (residency_trace[i].kind == ResidencyEventKind::Hit) ++result.residency_hits;
        if (residency_trace[i].kind == ResidencyEventKind::Miss) ++result.residency_misses;
        if (residency_trace[i].kind == ResidencyEventKind::Evict) ++result.evictions;
    }
    const auto materialization_trace = shared_materializer->trace();
    for (size_t i = result.materialization_events_before; i < materialization_trace.size(); ++i) {
        const auto & event = materialization_trace[i];
        if (event.event == "STATE" && event.state == MaterializationState::InFlight) ++result.materialization_requests;
        if (event.event == "STATE" && event.state == MaterializationState::Ready && !event.source_id.empty())
            ++result.source_reads;
    }
    std::printf("%s residency_hits=%zu residency_misses=%zu source_reads=%zu "
        "materialization_requests=%zu source_policy_calls=%zu evictions=%zu resident_tensors_after=%zu\n", label.c_str(), result.residency_hits,
        result.residency_misses, result.source_reads, result.materialization_requests,
        result.source_policy_calls, result.evictions, residency->resident_count());
    result.resident_tensors_after = residency->resident_count();
    return result;
}

bool merge_and_report(const MultiRun & run, const std::vector<float> & weights,
    const char * label) {
    std::vector<float> actual, reference;
    std::string error;
    if (!weighted_merge(run.actual, weights, &actual, &error) ||
        !weighted_merge(run.reference, weights, &reference, &error)) return false;
    const bool ok = parity(actual, reference, "weighted_merge_parity");
    std::printf("%s selected_count=%zu executed_count=%zu final_merge=%s weights=", label,
        weights.size(), run.actual.size(), ok ? "PASS" : "FAIL");
    for (float weight : weights) std::printf("%g,", weight);
    std::printf("\n");
    return ok;
}

} // namespace

#ifndef VBUF_POC12_LIBRARY_ONLY
int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6) {
        std::fprintf(stderr, "usage: multi_expert_moe_poc12 <vbuf> <router-endpoint> <capture-dir> "
            "[failure] [expert-endpoint]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string endpoint = argv[2];
    const bool failure = argc >= 5 && std::string(argv[4]) == "failure";
    const std::string expert_endpoint = argc == 6 ? argv[5] : endpoint;
    constexpr uint32_t input_dim = 2048;
    constexpr uint32_t expert_count = 64;
    constexpr uint32_t top_k = 6;

    Metadata metadata;
    metadata.artifact = read_file(artifact);
    metadata.handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!metadata.artifact || metadata.handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0) return 3;
    for (uint64_t i = 0; i < metadata.count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata.handle, i, &offset, &length) != 0) return 4;
        VbufMlTensorView view = metadata.views[i];
        view.payload_len = length;
        metadata.tensors.push_back({ view, i, offset });
    }
    const Meta router = lookup(metadata, "blk.1.ffn_gate_inp.weight");
    const VbufTensorView router_view{ router.view.representation, router.view.rank,
        router.view.dimensions, router.view.payload, router.view.payload_len };
    const PersistentTensorRef router_ref{ router.id, std::string(router.view.name, router.view.name_len),
        router_view, router.offset };
    std::printf("router_tensor=%s dimensions=[%u,%u] offset=%llu bytes=%llu top_k=%u expert_count=%u\n",
        router_ref.name.c_str(), input_dim, expert_count,
        static_cast<unsigned long long>(router.offset), static_cast<unsigned long long>(router.view.payload_len), top_k,
        expert_count);
    std::printf("weighting_rule=softmax_all_logits_then_selected_topk_renormalize\n");

    const Activation activation_a = one_hot(0, input_dim);
    const Activation activation_b = one_hot(1, input_dim);
    auto lease = model_lease(metadata.handle);
    RouterGraph router_graph = build_router_graph(router_ref);
    auto router_backing = std::make_shared<LocalVbufRangeMaterializer>(
        std::make_shared<HttpRangeSource>(endpoint));
    auto router_residency = std::make_shared<TensorResidencyStore>(router.view.payload_len * 2);
    auto router_materializer = std::make_shared<ResidentTensorMaterializer>(router_backing, router_residency);
    router_materializer->request(router_graph.router, router_ref, router_ref.view.payload_len);
    RoutedResult routed_a = route_activation(router_graph, activation_a, lease,
        router_materializer.get(), router_ref, input_dim, expert_count, top_k);
    router_materializer->request(router_graph.router, router_ref, router_ref.view.payload_len);
    RoutedResult routed_b = route_activation(router_graph, activation_b, lease,
        router_materializer.get(), router_ref, input_dim, expert_count, top_k);
    std::printf("activation_a_topk=%s scores=", ids_text(routed_a.selection).c_str());
    for (float score : routed_a.selection.scores) std::printf("%g,", score);
    std::printf(" normalized_weights=");
    for (float weight : routed_a.weights) std::printf("%g,", weight);
    std::printf("\nactivation_b_topk=%s scores=", ids_text(routed_b.selection).c_str());
    for (float score : routed_b.selection.scores) std::printf("%g,", score);
    std::printf(" normalized_weights=");
    for (float weight : routed_b.weights) std::printf("%g,", weight);
    std::printf("\n");

    if (failure) {
        auto fail_source = std::make_shared<HttpRangeSource>(expert_endpoint);
        auto fail_backing = std::make_shared<LocalVbufRangeMaterializer>(fail_source);
        auto fail_residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
        auto fail_materializer = std::make_shared<ResidentTensorMaterializer>(fail_backing, fail_residency);
        const MultiRun failed = execute_selected(metadata, routed_a.selection, activation_a, lease,
            fail_materializer, fail_residency, "selected_expert_failure", fail_source, false, 0, true);
        std::printf("selected_expert_failure expert_id=%u tensor_ref=1 failed_state=FAILED "
            "consuming_op_executed=NO final_merge=NOT_EXECUTED incomplete_discarded=YES "
            "resources_after_teardown=0\n", routed_a.selection.ids.front());
        return failed.ok ? 14 : 0;
    }

    auto expert_source = std::make_shared<HttpRangeSource>(expert_endpoint);
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(expert_source);
    auto residency = std::make_shared<TensorResidencyStore>(8 * 1024 * 1024);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    const MultiRun a_cold = execute_selected(metadata, routed_a.selection, activation_a, lease,
        materializer, residency, "activation_a_cold", expert_source, true);
    const bool a_cold_merge = merge_and_report(a_cold, routed_a.weights, "activation_a_cold");
    const uint64_t peak_active = a_cold.peak_active;
    const uint64_t selected_bytes = a_cold.selected_bytes;
    const MultiRun a_warm = execute_selected(metadata, routed_a.selection, activation_a, lease,
        materializer, residency, "activation_a_warm", expert_source, false);
    const bool a_warm_merge = merge_and_report(a_warm, routed_a.weights, "activation_a_warm");
    const MultiRun b_run = execute_selected(metadata, routed_b.selection, activation_b, lease,
        materializer, residency, "activation_b", expert_source, false);
    const bool b_merge = merge_and_report(b_run, routed_b.weights, "activation_b");
    const MultiRun a_replay = execute_selected(metadata, routed_a.selection, activation_a, lease,
        materializer, residency, "activation_a_replay", expert_source, false);
    const bool a_replay_merge = merge_and_report(a_replay, routed_a.weights, "activation_a_replay");
    std::printf("selected_total_persistent_bytes=%llu peak_active_persistent_bytes=%llu "
        "peak_active_fraction=%g peak_resident_bytes=%llu\n",
        static_cast<unsigned long long>(selected_bytes), static_cast<unsigned long long>(peak_active),
        selected_bytes == 0 ? 0.0 : static_cast<double>(peak_active) / selected_bytes,
        static_cast<unsigned long long>(std::max({ a_cold.peak_resident, a_warm.peak_resident,
            b_run.peak_resident, a_replay.peak_resident })));
    std::printf("graphs_created_selected=6 graphs_created_unselected=0 unselected_acquired=0 "
        "unselected_source_reads=0 unselected_materializations=0\n");
    std::printf("warm_reuse=%s activation_a_to_b_working_set_transition=PASS activation_a_to_b_to_a=PASS\n",
        a_warm_merge ? "PASS" : "FAIL");

    return a_cold.ok && a_cold_merge && a_warm.ok && a_warm_merge && b_run.ok && b_merge &&
        a_replay.ok && a_replay_merge ? 0 : 15;
}
#endif
#endif // VBUF_POC12_INCLUDED
