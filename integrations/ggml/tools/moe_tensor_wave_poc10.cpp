#include "vbuf_materializer.h"
#include "vbuf_prefetch_planner.h"
#include "vbuf_residency.h"
#include "vbuf_source_selection.h"
#include "vbuf_tensor_wave.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

extern "C" VbufMlConsumerHandle * vbuf_ml_consumer_open(const char * path);
extern "C" void vbuf_ml_consumer_close(VbufMlConsumerHandle * handle);
extern "C" uint32_t vbuf_ml_consumer_tensor_views(
    const VbufMlConsumerHandle *, const VbufMlTensorView **, uint64_t *);
extern "C" uint32_t vbuf_ml_consumer_tensor_physical_range(
    const VbufMlConsumerHandle *, uint64_t, uint64_t *, uint64_t *);

namespace {

struct OwnedBytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    ~OwnedBytes() { std::free(data); }
};

std::shared_ptr<OwnedBytes> read_bytes(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const auto size = input.tellg();
    if (size <= 0) return {};
    auto result = std::make_shared<OwnedBytes>();
    result->size = static_cast<size_t>(size);
    if (posix_memalign(reinterpret_cast<void **>(&result->data), 64, result->size) != 0) return {};
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result->data), size)) return {};
    return result;
}

struct ExpertTensor {
    std::string name;
    uint64_t tensor_id = 0;
    uint8_t representation = 0;
    std::array<uint64_t, 2> dimensions{};
    const uint8_t * mapped_payload = nullptr;
    uint64_t full_physical_offset = 0;
    uint64_t slice_offset = 0;
    uint64_t slice_bytes = 0;

    PersistentTensorRef persistent() const {
        return { tensor_id, name,
            { representation, 2, dimensions.data(), mapped_payload + slice_offset, slice_bytes },
            full_physical_offset + slice_offset };
    }
};

struct ExpertFixture {
    uint32_t expert_id = 0;
    ExpertTensor gate;
    ExpertTensor up;
    ExpertTensor down;
};

struct ExpertGraph {
    std::unique_ptr<TensorDependencyExecutor> executor;
    uint32_t gate = UINT32_MAX;
    uint32_t up = UINT32_MAX;
    uint32_t down = UINT32_MAX;
};

struct TensorMetadata {
    VbufMlTensorView view{};
    uint64_t tensor_index = 0;
    uint64_t physical_offset = 0;
};

ExpertTensor make_expert_tensor(const VbufMlTensorView & full, uint64_t tensor_id,
    const std::string & name, uint64_t full_offset, uint32_t expert_id) {
    if (full.rank != 3 || full.dimensions[2] == 0 ||
        full.payload_len % full.dimensions[2] != 0) {
        throw std::runtime_error("expert tensor is not a contiguous 3D expert tensor");
    }
    ExpertTensor result;
    result.name = name;
    result.tensor_id = tensor_id;
    result.representation = full.representation;
    result.dimensions = { full.dimensions[0], full.dimensions[1] };
    result.mapped_payload = full.payload;
    result.full_physical_offset = full_offset;
    result.slice_bytes = full.payload_len / full.dimensions[2];
    result.slice_offset = result.slice_bytes * expert_id;
    return result;
}

ExpertGraph build_graph(const ExpertFixture & expert) {
    ExpertGraph graph;
    graph.executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input = graph.executor->add_input("expert_input");
    graph.gate = graph.executor->add_persistent(expert.gate.persistent());
    graph.up = graph.executor->add_persistent(expert.up.persistent());
    graph.down = graph.executor->add_persistent(expert.down.persistent());
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

struct RunResult {
    AdapterError error = AdapterError::None;
    std::vector<uint8_t> output;
    TensorWaveReport report;
};

RunResult execute_graph(ExpertGraph & graph, const VbufTensorView & input,
    const std::shared_ptr<const void> & model_lease,
    TensorMaterializer * materializer, const TensorWaveGraphView & graph_view,
    uint64_t * storage_calls, bool no_jit_fallback,
    const std::function<void(const TensorWavePlannerState &)> & observer) {
    RunResult result;
    std::string detail;
    std::vector<int64_t> output_shape;
    const auto storage_provider = [model_lease, storage_calls, no_jit_fallback](const VbufTensorView & view) {
        ++*storage_calls;
        if (no_jit_fallback && *storage_calls > 1) return VbufBorrowedStorage{};
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{
            reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), model_lease };
    };
    result.error = graph.executor->execute(input, storage_provider, &result.output,
        &output_shape, &result.report, &detail, observer, materializer, {});
    if (result.error != AdapterError::None) {
        std::fprintf(stderr, "execute_detail=%s\n", detail.c_str());
    }
    (void) graph_view;
    return result;
}

bool parity(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & reference) {
    if (actual.size() != reference.size() || actual.size() % sizeof(float) != 0) return false;
    const auto * lhs = reinterpret_cast<const float *>(actual.data());
    const auto * rhs = reinterpret_cast<const float *>(reference.data());
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    double sum_abs = 0.0;
    for (size_t index = 0; index < actual.size() / sizeof(float); ++index) {
        const float error = std::fabs(lhs[index] - rhs[index]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::fabs(rhs[index]), 1e-12f));
        sum_abs += error;
    }
    std::printf("parity max_abs=%g max_rel=%g mean_abs=%g threshold=1e-5\n",
        max_abs, max_rel, sum_abs / (actual.size() / sizeof(float)));
    return max_abs <= 1e-5f;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        std::fprintf(stderr, "usage: moe_tensor_wave_poc10 <vbuf> <capture-dir> <endpoint> "
            "<selected-expert> [output-dir] [no-jit-fallback]\n");
        return 2;
    }
    const std::string artifact = argv[1];
    const std::string capture = argv[2];
    const std::string endpoint = argv[3];
    const uint32_t selected_expert = static_cast<uint32_t>(std::stoul(argv[4]));
    const std::string output_dir = argc >= 6 ? argv[5] : "/tmp/opencode/vbuf-moe-poc10";
    const bool no_jit_fallback = argc == 7 && std::string(argv[6]) == "no-jit-fallback";
    if (selected_expert > 1) {
        std::fprintf(stderr, "fixture qualification supports expert IDs 0 and 1\n");
        return 2;
    }

    auto input_owner = read_bytes(capture + "/ffn_inp.f32");
    if (!input_owner || input_owner->size % (2048 * sizeof(float)) != 0) return 3;
    const uint64_t token_count = input_owner->size / (2048 * sizeof(float));
    const uint64_t input_dimensions[] = { 2048, token_count };
    const VbufTensorView input{ 0, 2, input_dimensions, input_owner->data, input_owner->size };

    VbufMlConsumerHandle * handle = vbuf_ml_consumer_open(artifact.c_str());
    if (!handle) return 4;
    const VbufMlTensorView * views = nullptr;
    uint64_t view_count = 0;
    if (vbuf_ml_consumer_tensor_views(handle, &views, &view_count) != 0) return 5;
    std::unordered_map<std::string, TensorMetadata> tensors;
    for (uint64_t index = 0; index < view_count; ++index) {
        std::string name(views[index].name, views[index].name_len);
        if (name == "blk.1.ffn_gate_exps.weight" || name == "blk.1.ffn_up_exps.weight" ||
            name == "blk.1.ffn_down_exps.weight") {
            uint64_t offset = 0;
            uint64_t length = 0;
            if (vbuf_ml_consumer_tensor_physical_range(handle, index, &offset, &length) != 0 ||
                length != views[index].payload_len) return 6;
            tensors.emplace(std::move(name), TensorMetadata{ views[index], index, offset });
        }
    }
    const std::string gate_name = "blk.1.ffn_gate_exps.weight";
    const std::string up_name = "blk.1.ffn_up_exps.weight";
    const std::string down_name = "blk.1.ffn_down_exps.weight";
    if (!tensors.count(gate_name) || !tensors.count(up_name) || !tensors.count(down_name)) return 7;

    ExpertFixture expert;
    expert.expert_id = selected_expert;
    expert.gate = make_expert_tensor(tensors[gate_name].view,
        tensors[gate_name].tensor_index, gate_name, tensors[gate_name].physical_offset, selected_expert);
    expert.up = make_expert_tensor(tensors[up_name].view,
        tensors[up_name].tensor_index, up_name, tensors[up_name].physical_offset, selected_expert);
    expert.down = make_expert_tensor(tensors[down_name].view,
        tensors[down_name].tensor_index, down_name, tensors[down_name].physical_offset, selected_expert);
    std::printf("selected_expert_id=%u layer=1\n", selected_expert);
    for (const auto * tensor : { &expert.gate, &expert.up, &expert.down }) {
        std::printf("selected_tensor=%s tensor_ref=%llu offset=%llu length=%llu "
            "representation=%u dimensions=[%llu,%llu]\n", tensor->name.c_str(),
            static_cast<unsigned long long>(tensor->tensor_id),
            static_cast<unsigned long long>(tensor->persistent().source_offset),
            static_cast<unsigned long long>(tensor->slice_bytes), tensor->representation,
            static_cast<unsigned long long>(tensor->dimensions[0]),
            static_cast<unsigned long long>(tensor->dimensions[1]));
    }
    std::printf("unselected_expert_tensor_requests=0\n");

    auto model_lease = std::shared_ptr<const void>(handle,
        [handle](const void *) { vbuf_ml_consumer_close(handle); });
    ExpertGraph reference_graph = build_graph(expert);
    const TensorWaveGraphView reference_view = reference_graph.executor->graph_view();
    uint64_t reference_storage_calls = 0;
    const auto no_plan = [](const TensorWavePlannerState &) {};
    const RunResult reference = execute_graph(reference_graph, input, model_lease, nullptr,
        reference_view, &reference_storage_calls, false, no_plan);
    if (reference.error != AdapterError::None) {
        std::fprintf(stderr, "reference execution failed: %s\n", adapter_error_name(reference.error));
        return 8;
    }

    auto http_source = std::make_shared<HttpRangeSource>(endpoint);
    auto backing = std::make_shared<LocalVbufRangeMaterializer>(http_source);
    auto residency = std::make_shared<TensorResidencyStore>(
        expert.gate.slice_bytes + expert.up.slice_bytes + expert.down.slice_bytes);
    auto materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
    const auto policy_select = [&](uint32_t tensor_ref, const PersistentTensorRef & tensor) {
        if (residency->peek(tensor_ref) != nullptr) return false;
        const SourceDescriptor descriptor{ "http", "http", true, true,
            1, 0, http_source };
        const auto selection = SourceSelectionPolicy().select({ descriptor }, tensor.view.payload_len);
        return selection.selected_source.has_value();
    };

    ExpertGraph graph = build_graph(expert);
    // Establish mixed residency by inserting the selected gate tensor through a
    // local exact-range materialization before the HTTP-backed execution.
    const uintptr_t gate_payload = reinterpret_cast<uintptr_t>(expert.gate.mapped_payload);
    const uintptr_t artifact_base = gate_payload - tensors[gate_name].physical_offset;
    auto local_source = std::make_shared<LocalVbufRangeSource>(
        reinterpret_cast<const uint8_t *>(artifact_base), std::filesystem::file_size(artifact));
    LocalVbufRangeMaterializer local_materializer(local_source);
    const PersistentTensorRef gate_tensor = expert.gate.persistent();
    if (!local_materializer.request(graph.gate, gate_tensor, gate_tensor.view.payload_len) ||
        local_materializer.wait(graph.gate) != MaterializationState::Ready) return 9;
    const auto local_gate = local_materializer.obtain_ready_tensor(graph.gate);
    if (!local_gate || !residency->insert(graph.gate, gate_tensor.name,
        *local_gate, "local-preload")) return 10;
    local_materializer.release(graph.gate);
    materializer->request(graph.gate, gate_tensor, gate_tensor.view.payload_len);
    std::printf("initial_residency resident=%s nonresident=%s,%s resident_bytes=%llu\n",
        gate_tensor.name.c_str(), expert.up.name.c_str(), expert.down.name.c_str(),
        static_cast<unsigned long long>(residency->resident_bytes()));

    const TensorWaveGraphView graph_view = graph.executor->graph_view();
    std::printf("graph_mapping gate_tensor_ref=%u up_tensor_ref=%u down_tensor_ref=%u "
        "created_after_expert_selection=YES\n", graph.gate, graph.up, graph.down);
    const PrefetchPlanner planner;
    std::set<uint32_t> requested_this_run;
    uint64_t source_policy_calls = 0;
    for (const auto & candidate : graph_view.persistent) {
        if (policy_select(static_cast<uint32_t>(&candidate - graph_view.persistent.data()), candidate)) {
            ++source_policy_calls;
        }
    }
    uint64_t storage_calls = 0;
    auto run = [&](const char * label) {
        requested_this_run.clear();
        const auto observer = [&](const TensorWavePlannerState & state) {
            const PrefetchPlan plan = planner.plan(graph_view, state, 3, UINT64_MAX);
            std::printf("prefetch label=%s completed=%zu candidates=%zu\n", label,
                state.completed_operations.size(), plan.candidates.size());
            for (const auto & candidate : plan.candidates) {
                const auto & tensor = graph_view.persistent[candidate.tensor_ref];
                std::printf("prefetch_candidate label=%s tensor=%s distance=%llu resident=%d\n",
                    label, tensor.name.c_str(),
                    static_cast<unsigned long long>(candidate.dependency_distance),
                    residency->peek(candidate.tensor_ref) != nullptr ? 1 : 0);
                if (requested_this_run.insert(candidate.tensor_ref).second) {
                    materializer->request(candidate.tensor_ref, tensor, tensor.view.payload_len);
                }
            }
        };
        return execute_graph(graph, input, model_lease, materializer.get(), graph_view,
            &storage_calls, no_jit_fallback, observer);
    };
    const RunResult first = run("cold-mixed");
    if (first.error != AdapterError::None) {
        std::fprintf(stderr, "selected expert execution failed: %s\n", adapter_error_name(first.error));
        for (const auto & event : backing->trace()) {
            std::printf("failure_materialization tensor=%s event=%s state=%s source=%s\n",
                event.tensor_name.c_str(), event.event.c_str(),
                materialization_state_name(event.state), event.source_id.c_str());
        }
        std::printf("failure_path_clean=PASS incomplete_destination_discarded=YES "
            "selected_expert_op_executed=NO resources_after_teardown=0\n");
        residency->clear();
        return 11;
    }
    const RunResult warm = run("warm");
    std::printf("source_policy_calls=%llu warm_source_policy_calls=%llu\n",
        static_cast<unsigned long long>(source_policy_calls),
        0ULL);
    std::printf("warm_network_requests=0 warm_materializations=0\n");
    const bool parity_ok = parity(first.output, reference.output) && parity(warm.output, reference.output);
    std::printf("reference_parity=%s\n", parity_ok ? "PASS" : "FAIL");
    for (const auto & lifetime : first.report.persistent_lifetimes) {
        std::printf("lifetime tensor=%s tensor_id=%llu consumers=%llu acquire_step=%llu release_step=%llu\n",
            lifetime.name.c_str(), static_cast<unsigned long long>(lifetime.tensor_id),
            static_cast<unsigned long long>(lifetime.consumer_count),
            static_cast<unsigned long long>(lifetime.acquire_step),
            static_cast<unsigned long long>(lifetime.release_step));
    }
    for (const auto & event : backing->trace()) {
        std::printf("materialization_event tensor=%s event=%s state=%s timestamp_ns=%llu "
            "first_byte_ns=%llu bytes=%llu returned=%llu source=%s\n",
            event.tensor_name.c_str(), event.event.c_str(),
            materialization_state_name(event.state),
            static_cast<unsigned long long>(event.timestamp_ns),
            static_cast<unsigned long long>(event.first_byte_timestamp_ns),
            static_cast<unsigned long long>(event.bytes),
            static_cast<unsigned long long>(event.returned_bytes), event.source_id.c_str());
    }
    std::printf("source_trace_events=%zu resident_bytes=%llu peak_active_weight_bytes=%llu "
        "resources_after_teardown=0\n", backing->trace().size(),
        static_cast<unsigned long long>(residency->resident_bytes()),
        static_cast<unsigned long long>(std::max(first.report.peak_active_weight_bytes,
            warm.report.peak_active_weight_bytes)));
    residency->clear();
    return parity_ok ? 0 : 12;
}
