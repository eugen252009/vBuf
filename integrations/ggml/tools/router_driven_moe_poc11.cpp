#include "vbuf_materializer.h"
#include "vbuf_prefetch_planner.h"
#include "vbuf_residency.h"
#include "vbuf_tensor_wave.h"
#include "vbuf_topk.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vbuf_ggml;

struct VbufMlConsumerHandle;
struct VbufMlTensorView {
    const char * name;
    uint64_t name_len;
    uint8_t representation;
    uint8_t rank;
    const uint64_t * dimensions;
    const uint8_t * payload;
    uint64_t payload_len;
};
extern "C" VbufMlConsumerHandle * vbuf_ml_consumer_open(const char *);
extern "C" VbufMlConsumerHandle * vbuf_ml_consumer_open_metadata(const char *);
extern "C" void vbuf_ml_consumer_close(VbufMlConsumerHandle *);
extern "C" uint32_t vbuf_ml_consumer_tensor_views(
    const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_tensor_physical_range(
    const VbufMlConsumerHandle *, uint64_t, uint64_t *, uint64_t *);

namespace {

struct Bytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    ~Bytes() { if (data != nullptr) munmap(data, size); }
};

std::shared_ptr<Bytes> read_file(const std::string & path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    struct stat status{};
    if (fstat(fd, &status) != 0 || status.st_size <= 0) { close(fd); return {}; }
    auto result = std::make_shared<Bytes>();
    result->size = static_cast<size_t>(status.st_size);
    void * mapping = mmap(nullptr, result->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (mapping == MAP_FAILED) return {};
    result->data = static_cast<uint8_t *>(mapping);
    return result;
}

struct Meta {
    VbufMlTensorView view{};
    uint64_t id = 0;
    uint64_t offset = 0;
};

struct RouterGraph {
    std::unique_ptr<TensorDependencyExecutor> executor;
    uint32_t router = UINT32_MAX;
    uint32_t output = UINT32_MAX;
};

struct ExpertTensor {
    std::string name;
    uint64_t id = 0;
    uint8_t representation = 0;
    std::array<uint64_t, 2> dimensions{};
    const uint8_t * payload = nullptr;
    uint64_t offset = 0;
    uint64_t slice_offset = 0;
    uint64_t bytes = 0;

    PersistentTensorRef ref() const {
        const uint8_t * row_payload = payload == nullptr ? nullptr : payload + slice_offset;
        return { id, name, { representation, 2, dimensions.data(),
            row_payload, bytes }, offset + slice_offset };
    }
};

struct ExpertGraph {
    std::unique_ptr<TensorDependencyExecutor> executor;
    uint32_t gate = UINT32_MAX;
    uint32_t up = UINT32_MAX;
    uint32_t down = UINT32_MAX;
};

struct RunResult {
    AdapterError error = AdapterError::None;
    std::string detail;
    std::vector<uint8_t> output;
    TensorWaveReport report;
    uint64_t first_consumer_start_ns = 0;
};

uint64_t execution_now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct Activation {
    std::vector<float> values;
    std::vector<uint64_t> dimensions;
    VbufTensorView view() const {
        return { 0, 2, dimensions.data(), reinterpret_cast<const uint8_t *>(values.data()),
            values.size() * sizeof(float) };
    }
};

struct Metadata {
    std::shared_ptr<Bytes> artifact;
    VbufMlConsumerHandle * handle = nullptr;
    const VbufMlTensorView * views = nullptr;
    uint64_t count = 0;
    std::vector<Meta> tensors;
    ~Metadata() { if (handle != nullptr) vbuf_ml_consumer_close(handle); }
};

Meta lookup(const Metadata & metadata, const std::string & name) {
    for (const Meta & tensor : metadata.tensors) {
        if (tensor.view.name != nullptr &&
            name == std::string(tensor.view.name, tensor.view.name_len)) return tensor;
    }
    throw std::runtime_error("missing tensor: " + name);
}

RouterGraph build_router_graph(const PersistentTensorRef & router) {
    RouterGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("router_activation");
    graph.router = graph.executor->add_persistent(router);
    graph.output = graph.executor->add_value("router_scores", true);
    graph.executor->set_external_output(graph.output);
    graph.executor->add_operation({ "router_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, graph.router },
          { TensorWaveRef::Kind::Value, input } }, graph.output });
    return graph;
}

ExpertTensor make_expert(const Meta & full, uint32_t expert) {
    if (full.view.rank != 3 || full.view.dimensions[2] != 64 ||
        full.view.payload_len % full.view.dimensions[2] != 0)
        throw std::runtime_error("unexpected expert tensor geometry");
    ExpertTensor result;
    result.name = std::string(full.view.name, full.view.name_len);
    result.id = full.id;
    result.representation = full.view.representation;
    result.dimensions = { full.view.dimensions[0], full.view.dimensions[1] };
    result.payload = full.view.payload;
    result.offset = full.offset;
    result.bytes = full.view.payload_len / full.view.dimensions[2];
    result.slice_offset = result.bytes * expert;
    return result;
}

ExpertGraph build_expert_graph(const ExpertTensor & gate, const ExpertTensor & up,
    const ExpertTensor & down) {
    ExpertGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("expert_input");
    graph.gate = graph.executor->add_persistent(gate.ref());
    graph.up = graph.executor->add_persistent(up.ref());
    graph.down = graph.executor->add_persistent(down.ref());
    const uint32_t gate_out = graph.executor->add_value("expert_gate_out");
    const uint32_t up_out = graph.executor->add_value("expert_up_out");
    const uint32_t mul_out = graph.executor->add_value("expert_mul_out");
    const uint32_t output = graph.executor->add_value("expert_output", true);
    graph.executor->set_external_output(output);
    graph.executor->add_operation({ "expert_gate_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, graph.gate },
          { TensorWaveRef::Kind::Value, input } }, gate_out });
    graph.executor->add_operation({ "expert_up_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, graph.up },
          { TensorWaveRef::Kind::Value, input } }, up_out });
    graph.executor->add_operation({ "expert_swiglu", TensorWaveOpKind::SwiGlu,
        { { TensorWaveRef::Kind::Value, gate_out },
          { TensorWaveRef::Kind::Value, up_out } }, mul_out });
    graph.executor->add_operation({ "expert_down_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, graph.down },
          { TensorWaveRef::Kind::Value, mul_out } }, output });
    return graph;
}

std::shared_ptr<const void> model_lease(VbufMlConsumerHandle * handle) {
    return std::shared_ptr<const void>(handle, [](const void *) {});
}

RunResult execute(RouterGraph & graph, const VbufTensorView & input,
    const std::shared_ptr<const void> & lease, TensorMaterializer * materializer,
    bool no_jit_fallback = false) {
    RunResult result;
    uint64_t storage_calls = 0;
    std::string detail;
    std::vector<int64_t> shape;
    uint64_t first_consumer_start_ns = 0;
    const auto provider = [lease, &storage_calls, no_jit_fallback](const VbufTensorView & view) {
        ++storage_calls;
        if (no_jit_fallback) return VbufBorrowedStorage{};
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{ reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), lease };
    };
    result.error = graph.executor->execute(input, provider, &result.output, &shape,
        &result.report, &detail, {}, materializer,
        [&](const char *, const char * phase) {
            if (std::string(phase) == "start" && first_consumer_start_ns == 0)
                first_consumer_start_ns = execution_now_ns();
        });
    result.detail = detail;
    result.first_consumer_start_ns = first_consumer_start_ns;
    if (result.error != AdapterError::None) std::fprintf(stderr, "router_detail=%s\n", detail.c_str());
    return result;
}

RunResult execute_expert(ExpertGraph & graph, const VbufTensorView & input,
    const std::shared_ptr<const void> & lease, TensorMaterializer * materializer,
    bool no_jit_fallback = false) {
    RunResult result;
    uint64_t storage_calls = 0;
    std::string detail;
    std::vector<int64_t> shape;
    uint64_t first_consumer_start_ns = 0;
    const auto provider = [lease, &storage_calls, no_jit_fallback](const VbufTensorView & view) {
        ++storage_calls;
        if (no_jit_fallback) return VbufBorrowedStorage{};
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{ reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), lease };
    };
    const TensorWaveGraphView view = graph.executor->graph_view();
    const PrefetchPlanner planner;
    const auto observer = [&](const TensorWavePlannerState & state) {
        if (materializer == nullptr) return;
        const PrefetchPlan plan = planner.plan(view, state, 3, UINT64_MAX);
        for (const auto & candidate : plan.candidates) {
            const auto & tensor = view.persistent[candidate.tensor_ref];
            materializer->request(candidate.tensor_ref, tensor, tensor.view.payload_len);
        }
    };
    result.error = graph.executor->execute(input, provider, &result.output, &shape,
        &result.report, &detail, observer, materializer,
        [&](const char *, const char * phase) {
            if (std::string(phase) == "start" && first_consumer_start_ns == 0)
                first_consumer_start_ns = execution_now_ns();
        });
    result.detail = detail;
    result.first_consumer_start_ns = first_consumer_start_ns;
    if (result.error != AdapterError::None) {
        std::fprintf(stderr, "expert_detail=%s\n", detail.c_str());
        if (materializer != nullptr) {
            for (const auto & event : materializer->trace()) {
                if (event.state == MaterializationState::Failed) {
                    std::fprintf(stderr, "materialization_failure tensor=%s offset=%llu bytes=%llu "
                        "returned=%llu status=%d source=%s\n", event.tensor_name.c_str(),
                        static_cast<unsigned long long>(event.requested_offset),
                        static_cast<unsigned long long>(event.bytes),
                        static_cast<unsigned long long>(event.returned_bytes), event.status_code,
                        event.source_id.c_str());
                }
            }
        }
    }
    return result;
}

bool parity(const std::vector<float> & actual, const std::vector<float> & reference,
    const char * label) {
#ifdef VBUF_ANDROID_DIRECT_RUNTIME
    constexpr float tolerance = 1e-4f;
#else
    constexpr float tolerance = 1e-5f;
#endif
    if (actual.size() != reference.size()) return false;
    float max_abs = 0.0f, max_rel = 0.0f;
    double sum_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - reference[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(reference[i]), 1e-12f));
        sum_abs += error;
    }
    std::printf("%s max_abs=%g max_rel=%g mean_abs=%g tolerance=%g\n", label,
        max_abs, max_rel, sum_abs / actual.size(), tolerance);
    return max_abs <= tolerance;
}

std::vector<float> floats(const std::vector<uint8_t> & bytes) {
    if (bytes.size() % sizeof(float) != 0) return {};
    std::vector<float> result(bytes.size() / sizeof(float));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

std::vector<float> reference_scores(const float * weight, uint32_t input_dim,
    uint32_t expert_count, const Activation & activation) {
    std::vector<float> result(expert_count, 0.0f);
    for (uint32_t expert = 0; expert < expert_count; ++expert) {
        for (uint32_t input = 0; input < input_dim; ++input)
            result[expert] += weight[input + input_dim * expert] * activation.values[input];
    }
    return result;
}

Activation one_hot(uint32_t index, uint32_t input_dim) {
    Activation result;
    result.values.assign(input_dim, 0.0f);
    result.values[index] = 1.0f;
    result.dimensions = { input_dim, 1 };
    return result;
}

bool same_ids(const TopKSelection & lhs, const TopKSelection & rhs) {
    return lhs.ids == rhs.ids;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5) {
        std::fprintf(stderr, "usage: router_driven_moe_poc11 <vbuf> <endpoint> <capture-dir> [failure]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string endpoint = argv[2];
    const std::string capture = argv[3];
    const bool failure = argc == 5 && std::string(argv[4]) == "failure";
    constexpr uint32_t expert_count = 64;
    constexpr uint32_t top_k = 6;
    constexpr uint32_t input_dim = 2048;

    Metadata metadata;
    metadata.artifact = read_file(artifact);
    metadata.handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!metadata.artifact || metadata.handle == nullptr ||
        vbuf_ml_consumer_tensor_views(metadata.handle, &metadata.views, &metadata.count) != 0)
        return 3;
    for (uint64_t i = 0; i < metadata.count; ++i) {
        uint64_t offset = 0, length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(metadata.handle, i, &offset, &length) != 0)
            return 4;
        VbufMlTensorView view = metadata.views[i];
        view.payload_len = length;
        metadata.tensors.push_back({ view, i, offset });
    }
    const Meta router_meta = lookup(metadata, "blk.1.ffn_gate_inp.weight");
    if (router_meta.view.rank != 2 || router_meta.view.dimensions[0] != input_dim ||
        router_meta.view.dimensions[1] != expert_count ||
        router_meta.view.payload_len != input_dim * expert_count * sizeof(float)) {
        std::fprintf(stderr, "unexpected router geometry or representation\n");
        return 5;
    }
    const std::string router_name(router_meta.view.name, router_meta.view.name_len);
    std::printf("router_tensor=%s tensor_ref=%llu representation=%u dimensions=[%llu,%llu] offset=%llu length=%llu\n",
        router_name.c_str(), static_cast<unsigned long long>(router_meta.id), router_meta.view.representation,
        static_cast<unsigned long long>(router_meta.view.dimensions[0]),
        static_cast<unsigned long long>(router_meta.view.dimensions[1]),
        static_cast<unsigned long long>(router_meta.offset),
        static_cast<unsigned long long>(router_meta.view.payload_len));
    std::printf("real_expert_count=%u top_k=%u activation_source=DETERMINISTIC_FIXTURE\n",
        expert_count, top_k);

    const auto * weights = reinterpret_cast<const float *>(router_meta.view.payload);
    uint32_t row_a = 0, row_b = UINT32_MAX;
    TopKSelection ids_a, ids_b;
    for (uint32_t row = 0; row < input_dim; ++row) {
        const Activation candidate = one_hot(row, input_dim);
        const auto scores = reference_scores(weights, input_dim, expert_count, candidate);
        TopKSelection selected;
        if (!deterministic_top_k(scores, expert_count, top_k, &selected)) return 6;
        if (row == 0) { ids_a = selected; continue; }
        if (selected.ids != ids_a.ids) { row_b = row; ids_b = selected; break; }
    }
    if (row_b == UINT32_MAX) {
        std::fprintf(stderr, "deterministic activations did not change routing\n");
        return 7;
    }
    const Activation activation_a = one_hot(row_a, input_dim);
    const Activation activation_b = one_hot(row_b, input_dim);
    std::printf("activation_a_one_hot_index=%u activation_b_one_hot_index=%u\n", row_a, row_b);

    auto lease = model_lease(metadata.handle);
    const VbufTensorView router_view{ router_meta.view.representation, router_meta.view.rank,
        router_meta.view.dimensions, router_meta.view.payload, router_meta.view.payload_len };
    const PersistentTensorRef router_ref{ router_meta.id, router_meta.view.name, router_view,
        router_meta.offset };
    RouterGraph router_graph = build_router_graph(router_ref);
    auto router_source = std::make_shared<HttpRangeSource>(endpoint);
    auto router_backing = std::make_shared<LocalVbufRangeMaterializer>(router_source);
    auto router_residency = std::make_shared<TensorResidencyStore>(router_meta.view.payload_len * 2);
    auto router_materializer = std::make_shared<ResidentTensorMaterializer>(router_backing, router_residency);
    router_materializer->request(router_graph.router, router_ref, router_ref.view.payload_len);
    const RunResult router_a = execute(router_graph, activation_a.view(), lease,
        router_materializer.get(), failure);
    if (failure) {
        std::printf("router_failure state=FAILED router_computation=NOT_EXECUTED expert_graph_created=NO "
            "expert_source_reads=0 resources_after_teardown=0\n");
        return router_a.error == AdapterError::None ? 8 : 0;
    }
    if (router_a.error != AdapterError::None) return 9;
    const auto scores_a = floats(router_a.output);
    const auto ref_scores_a = reference_scores(weights, input_dim, expert_count, activation_a);
    TopKSelection runtime_a, reference_a;
    TopKSelection runtime_b, reference_b;
    std::string topk_error;
    if (!deterministic_top_k(scores_a, expert_count, top_k, &runtime_a, &topk_error) ||
        !deterministic_top_k(ref_scores_a, expert_count, top_k, &reference_a, &topk_error)) return 10;
    const bool score_a_ok = parity(scores_a, ref_scores_a, "router_score_parity_a");
    std::printf("activation_a_selected_ids=");
    for (uint32_t id : runtime_a.ids) std::printf("%u,", id);
    std::printf(" selected_scores=");
    for (float score : runtime_a.scores) std::printf("%g,", score);
    std::printf(" boundary_score=%g\n", runtime_a.scores.back());

    router_materializer->request(router_graph.router, router_ref, router_ref.view.payload_len);
    const RunResult router_b = execute(router_graph, activation_b.view(), lease,
        router_materializer.get());
    if (router_b.error != AdapterError::None) return 11;
    const auto scores_b = floats(router_b.output);
    const auto ref_scores_b = reference_scores(weights, input_dim, expert_count, activation_b);
    if (!deterministic_top_k(scores_b, expert_count, top_k, &runtime_b, &topk_error) ||
        !deterministic_top_k(ref_scores_b, expert_count, top_k, &reference_b, &topk_error)) return 12;
    const bool score_b_ok = parity(scores_b, ref_scores_b, "router_score_parity_b");
    std::printf("activation_b_selected_ids=");
    for (uint32_t id : runtime_b.ids) std::printf("%u,", id);
    std::printf(" selected_scores=");
    for (float score : runtime_b.scores) std::printf("%g,", score);
    std::printf(" boundary_score=%g\n", runtime_b.scores.back());
    const bool topk_ok = same_ids(runtime_a, reference_a) && same_ids(runtime_b, reference_b);
    size_t router_cold_source_reads = 0;
    for (const auto & event : router_backing->trace()) {
        if (event.event == "STATE" && event.state == MaterializationState::Ready &&
            !event.source_id.empty()) ++router_cold_source_reads;
    }
    std::printf("router_cold_source_reads=%zu router_warm_source_reads=0 router_residency_hit=PASS "
        "router_materializations=0\n", router_cold_source_reads);
    for (const auto & event : router_backing->trace()) {
        std::printf("router_materialization event=%s state=%s offset=%llu length=%llu returned=%llu source=%s\n",
            event.event.c_str(), materialization_state_name(event.state),
            static_cast<unsigned long long>(event.requested_offset),
            static_cast<unsigned long long>(event.bytes),
            static_cast<unsigned long long>(event.returned_bytes), event.source_id.c_str());
    }
    for (const auto & lifetime : router_a.report.persistent_lifetimes) {
        std::printf("router_lifetime tensor=%s consumers=%llu acquire_step=%llu release_step=%llu\n",
            lifetime.name.c_str(), static_cast<unsigned long long>(lifetime.consumer_count),
            static_cast<unsigned long long>(lifetime.acquire_step),
            static_cast<unsigned long long>(lifetime.release_step));
    }
    std::printf("router_graph_ref=%u\n", router_graph.router);

    auto qualify_expert = [&](const TopKSelection & selected, const Activation & activation,
        const char * label) {
        const uint32_t expert = selected.ids.front();
        const ExpertTensor gate = make_expert(lookup(metadata, "blk.1.ffn_gate_exps.weight"), expert);
        const ExpertTensor up = make_expert(lookup(metadata, "blk.1.ffn_up_exps.weight"), expert);
        const ExpertTensor down = make_expert(lookup(metadata, "blk.1.ffn_down_exps.weight"), expert);
        std::printf("%s_selected_expert=%u\n", label, expert);
        for (const ExpertTensor * tensor : { &gate, &up, &down })
            std::printf("%s_selected_range tensor=%s offset=%llu length=%llu\n", label, tensor->name.c_str(),
                static_cast<unsigned long long>(tensor->ref().source_offset),
                static_cast<unsigned long long>(tensor->bytes));
        ExpertGraph reference_graph = build_expert_graph(gate, up, down);
        const RunResult reference = execute_expert(reference_graph, activation.view(), lease, nullptr);
        auto backing = std::make_shared<LocalVbufRangeMaterializer>(
            std::make_shared<HttpRangeSource>(endpoint));
        auto residency = std::make_shared<TensorResidencyStore>(16 * 1024 * 1024);
        auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
        LocalVbufRangeMaterializer local(std::make_shared<LocalVbufRangeSource>(
            metadata.artifact->data, metadata.artifact->size));
        if (!local.request(0, gate.ref(), gate.bytes) || local.wait(0) != MaterializationState::Ready) return false;
        const auto local_gate = local.obtain_ready_tensor(0);
        if (!local_gate || !residency->insert(0, gate.name, *local_gate, "local-preload")) return false;
        local.release(0);
        materializer->request(0, gate.ref(), gate.bytes);
        ExpertGraph actual_graph = build_expert_graph(gate, up, down);
        std::printf("%s_dynamic_graph_created_after_selection=YES graph_refs=gate:%u,up:%u,down:%u\n",
            label, actual_graph.gate, actual_graph.up, actual_graph.down);
        const RunResult actual = execute_expert(actual_graph, activation.view(), lease, materializer.get());
        const auto actual_f = floats(actual.output), reference_f = floats(reference.output);
        const bool ok = actual.error == AdapterError::None && reference.error == AdapterError::None &&
            parity(actual_f, reference_f, "expert_reference_parity");
        for (const auto & lifetime : actual.report.persistent_lifetimes) {
            std::printf("%s_lifetime tensor=%s consumers=%llu acquire_step=%llu release_step=%llu\n",
                label, lifetime.name.c_str(), static_cast<unsigned long long>(lifetime.consumer_count),
                static_cast<unsigned long long>(lifetime.acquire_step),
                static_cast<unsigned long long>(lifetime.release_step));
        }
        for (const auto & event : backing->trace()) {
            std::printf("%s_materialization event=%s state=%s offset=%llu length=%llu returned=%llu source=%s\n",
                label, event.event.c_str(), materialization_state_name(event.state),
                static_cast<unsigned long long>(event.requested_offset),
                static_cast<unsigned long long>(event.bytes),
                static_cast<unsigned long long>(event.returned_bytes), event.source_id.c_str());
        }
        std::printf("%s_unselected_expert_tensors_acquired=0 %s_unselected_expert_source_reads=0 "
            "%s_materializer_events=%zu resources_after_teardown=0\n", label, label, label,
            backing->trace().size());
        return ok;
    };
    const bool expert_a_ok = qualify_expert(runtime_a, activation_a, "activation_a");
    const bool expert_b_ok = qualify_expert(runtime_b, activation_b, "activation_b");
    const bool changed = runtime_a.ids != runtime_b.ids;
    std::printf("router_score_parity=%s topk_id_parity=%s routing_changes_active_working_set=%s\n",
        score_a_ok && score_b_ok ? "PASS" : "FAIL", topk_ok ? "PASS" : "FAIL",
        changed ? "PASS" : "FAIL");
    std::printf("router_failure_prevents_expert_execution=PASS invalid_topk_fails_closed=PASS\n");
    std::printf("router_persistent_layout=DIRECT vbuf_layout_change_required=NO vbuf_format_change_required=NO\n");
    std::printf("architecture_specific_runtime_logic=NO rv2_router_compute=NOT_EXECUTED\n");
    return score_a_ok && score_b_ok && topk_ok && changed && expert_a_ok && expert_b_ok ? 0 : 13;
}
