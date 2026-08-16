#include "vbuf_tensor_wave.h"
#include "vbuf_prefetch_planner.h"
#include "vbuf_materializer.h"
#include "vbuf_range_source.h"
#include "vbuf_source_selection.h"
#include "vbuf_residency.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using vbuf_ggml::AdapterError;
using vbuf_ggml::PersistentTensorRef;
using vbuf_ggml::TensorDependencyExecutor;
using vbuf_ggml::TensorWaveOp;
using vbuf_ggml::TensorWaveOpKind;
using vbuf_ggml::TensorWaveRef;
using vbuf_ggml::TensorWaveReport;
using vbuf_ggml::PrefetchPlan;
using vbuf_ggml::PrefetchPlanner;
using vbuf_ggml::LocalVbufRangeMaterializer;
using vbuf_ggml::SourceDescriptor;
using vbuf_ggml::SourceSelectionPolicy;
using vbuf_ggml::ResidentTensorMaterializer;
using vbuf_ggml::TensorResidencyStore;
using vbuf_ggml::TensorMaterializer;
using vbuf_ggml::VbufBorrowedStorage;
using vbuf_ggml::VbufTensorView;

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
    const VbufMlConsumerHandle * handle, const VbufMlTensorView ** views, uint64_t * count);
extern "C" uint32_t vbuf_ml_consumer_tensor_physical_range(
    const VbufMlConsumerHandle * handle, uint64_t index, uint64_t * offset, uint64_t * length);

namespace {

struct OwnedBytes {
    uint8_t * data = nullptr;
    size_t size = 0;
    ~OwnedBytes() { std::free(data); }
};

std::shared_ptr<OwnedBytes> read_aligned(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) return {};
    const std::streamsize size = input.tellg();
    if (size <= 0) return {};
    auto result = std::make_shared<OwnedBytes>();
    result->size = static_cast<size_t>(size);
    if (posix_memalign(reinterpret_cast<void **>(&result->data), 64, result->size) != 0) return {};
    input.seekg(0);
    if (!input.read(reinterpret_cast<char *>(result->data), size)) return {};
    return result;
}

uint64_t fnv1a(const std::vector<uint8_t> & bytes) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool compare_output(const char * label, const std::vector<uint8_t> & actual,
    const std::vector<uint8_t> & reference) {
    if (actual.size() != reference.size() || actual.size() % sizeof(float) != 0) return false;
    const auto * lhs = reinterpret_cast<const float *>(actual.data());
    const auto * rhs = reinterpret_cast<const float *>(reference.data());
    const size_t count = actual.size() / sizeof(float);
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        const float abs_error = std::fabs(lhs[index] - rhs[index]);
        sum_abs += abs_error;
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel,
            abs_error / std::max(std::fabs(rhs[index]), 1e-12f));
    }
    std::printf("parity=%s elements=%zu max_abs=%g max_rel=%g mean_abs=%g "
        "actual_hash=%016llx reference_hash=%016llx\n", label, count, max_abs, max_rel,
        static_cast<float>(sum_abs / count),
        static_cast<unsigned long long>(fnv1a(actual)),
        static_cast<unsigned long long>(fnv1a(reference)));
    return max_abs <= 2e-4f && (max_rel <= 2e-3f || max_abs <= 1e-5f);
}

void print_names(const char * key, const std::vector<std::string> & names) {
    std::printf(" %s=", key);
    for (size_t index = 0; index < names.size(); ++index) {
        if (index != 0) std::printf(",");
        std::printf("%s", names[index].c_str());
    }
}

void print_plan(const vbuf_ggml::TensorWaveGraphView & graph,
    const vbuf_ggml::TensorWavePlannerState & state, const PrefetchPlan & plan,
    uint64_t storage_calls_before, uint64_t storage_calls_after) {
    std::printf("plan_state completed=");
    for (size_t index = 0; index < state.completed_operations.size(); ++index) {
        if (index != 0) std::printf(",");
        std::printf("%s", graph.operations[state.completed_operations[index]].op_id.c_str());
    }
    std::printf(" active_weight_bytes_before_plan=%llu active_weight_bytes_after_plan=%llu "
        "active_lease_count_before_plan=%llu active_lease_count_after_plan=%llu "
        "storage_calls_before=%llu storage_calls_after=%llu byte_budget=%llu\n",
        static_cast<unsigned long long>(state.active_persistent_weight_bytes),
        static_cast<unsigned long long>(state.active_persistent_weight_bytes),
        static_cast<unsigned long long>(state.active_persistent_tensor_count),
        static_cast<unsigned long long>(state.active_persistent_tensor_count),
        static_cast<unsigned long long>(storage_calls_before),
        static_cast<unsigned long long>(storage_calls_after),
        static_cast<unsigned long long>(plan.byte_budget));
    for (const auto & candidate : plan.visible_candidates) {
        std::printf("visible_candidate tensor=%s required_by=%s distance=%llu bytes=%llu resident=%d\n",
            candidate.tensor_name.c_str(), candidate.required_by_op.c_str(),
            static_cast<unsigned long long>(candidate.dependency_distance),
            static_cast<unsigned long long>(candidate.byte_size),
            candidate.already_resident ? 1 : 0);
    }
    std::printf("selected_candidates=");
    for (size_t index = 0; index < plan.candidates.size(); ++index) {
        if (index != 0) std::printf(",");
        std::printf("%s", plan.candidates[index].tensor_name.c_str());
    }
    std::printf(" total_planned_bytes=%llu\n",
        static_cast<unsigned long long>(plan.total_planned_bytes));
}

uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

uint64_t rss_kib() {
    std::ifstream input("/proc/self/smaps_rollup");
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        uint64_t value = 0;
        fields >> key >> value;
        if (key == "Rss:") return value;
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 8) {
        std::fprintf(stderr, "usage: tensor_wave_poc3 <vbuf> <capture-dir> <output-dir> "
            "[baseline|local|remote|auto] [http-endpoint] "
            "[local-bytes-per-second remote-bytes-per-second]\n");
        return 2;
    }
    const std::string mode = argc >= 5 ? argv[4] : "baseline";
    const bool warm = mode.size() > 5 && mode.substr(mode.size() - 5) == "-warm";
    const std::string source_mode = warm ? mode.substr(0, mode.size() - 5) : mode;
    const bool prefetch_enabled = source_mode == "local" || source_mode == "remote" || source_mode == "auto";
    if ((source_mode == "remote" || source_mode == "auto") && argc < 6) {
        std::fprintf(stderr, "%s mode requires an HTTP endpoint\n", source_mode.c_str());
        return 2;
    }
    if (source_mode == "auto" && argc != 8) {
        std::fprintf(stderr, "auto mode requires local and remote throughput inputs\n");
        return 2;
    }
    auto input_owner = read_aligned((std::string(argv[2]) + "/ffn_inp.f32").c_str());
    auto swiglu_owner = read_aligned((std::string(argv[2]) + "/ffn_swiglu.f32").c_str());
    auto output_owner = read_aligned((std::string(argv[2]) + "/ffn_out.f32").c_str());
    if (!input_owner || !swiglu_owner || !output_owner ||
        input_owner->size % (2048 * sizeof(float)) != 0) return 3;

    const uint64_t token_count = input_owner->size / (2048 * sizeof(float));
    const uint64_t input_dimensions[] = { 2048, token_count };
    const VbufTensorView input{ 0, 2, input_dimensions, input_owner->data, input_owner->size };

    VbufMlConsumerHandle * handle = vbuf_ml_consumer_open(argv[1]);
    if (handle == nullptr) return 4;
    const VbufMlTensorView * views = nullptr;
    uint64_t view_count = 0;
    if (vbuf_ml_consumer_tensor_views(handle, &views, &view_count) != 0) return 5;
    std::unordered_map<std::string, VbufTensorView> named;
    std::unordered_map<std::string, uint64_t> ids;
    std::unordered_map<std::string, uint64_t> offsets;
    for (uint64_t index = 0; index < view_count; ++index) {
        const VbufMlTensorView & source = views[index];
        const std::string name(source.name, source.name_len);
        named.emplace(name, VbufTensorView{ source.representation, source.rank,
            source.dimensions, source.payload, source.payload_len });
        ids.emplace(name, index);
        uint64_t offset = 0;
        uint64_t length = 0;
        if (vbuf_ml_consumer_tensor_physical_range(handle, index, &offset, &length) != 0 ||
            length != source.payload_len) return 6;
        offsets.emplace(name, offset);
    }
    const char * names[] = {
        "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight" };
    for (const char * name : names) if (!named.count(name)) return 6;

    auto executor = std::make_unique<TensorDependencyExecutor>();
    const uint32_t input_value = executor->add_input("ffn_inp");
    const uint32_t norm = executor->add_persistent(
        { ids[names[0]], names[0], named[names[0]], offsets[names[0]] });
    const uint32_t gate = executor->add_persistent(
        { ids[names[1]], names[1], named[names[1]], offsets[names[1]] });
    const uint32_t up = executor->add_persistent(
        { ids[names[2]], names[2], named[names[2]], offsets[names[2]] });
    const uint32_t down = executor->add_persistent(
        { ids[names[3]], names[3], named[names[3]], offsets[names[3]] });
    const uint32_t norm_out = executor->add_value("norm_out");
    const uint32_t gate_out = executor->add_value("gate_out");
    const uint32_t up_out = executor->add_value("up_out");
    const uint32_t mul_out = executor->add_value("mul_out");
    const uint32_t ffn_out = executor->add_value("ffn_out", true);
    executor->set_external_output(ffn_out);
    executor->set_capture_value(mul_out);
    executor->add_operation({ "rms_norm", TensorWaveOpKind::RmsNorm,
        { { TensorWaveRef::Kind::Value, input_value },
          { TensorWaveRef::Kind::Persistent, norm } }, norm_out, 1.0e-6f });
    executor->add_operation({ "gate_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, gate },
          { TensorWaveRef::Kind::Value, norm_out } }, gate_out });
    executor->add_operation({ "up_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, up },
          { TensorWaveRef::Kind::Value, norm_out } }, up_out });
    executor->add_operation({ "swiglu", TensorWaveOpKind::SwiGlu,
        { { TensorWaveRef::Kind::Value, gate_out },
          { TensorWaveRef::Kind::Value, up_out } }, mul_out });
    executor->add_operation({ "down_matmul", TensorWaveOpKind::MulMat,
        { { TensorWaveRef::Kind::Persistent, down },
          { TensorWaveRef::Kind::Value, mul_out } }, ffn_out });

    std::printf("graph=ffn_inp->rms_norm->norm_out->{gate_matmul,up_matmul}->"
        "{gate_out,up_out}->swiglu->mul_out->down_matmul->ffn_out\n");
    std::printf("value_consumers=ffn_inp:1 norm_out:2 gate_out:1 up_out:1 mul_out:1 ffn_out:0\n");
    const vbuf_ggml::TensorWaveGraphView graph = executor->graph_view();
    const PrefetchPlanner planner;
    uint64_t total_weight_bytes = 0;
    for (const auto & tensor : graph.persistent) total_weight_bytes += tensor.view.payload_len;

    auto model_lease = std::shared_ptr<const void>(handle,
        [handle](const void *) { vbuf_ml_consumer_close(handle); });
    std::shared_ptr<vbuf_ggml::RangeSource> local_source;
    std::shared_ptr<vbuf_ggml::RangeSource> remote_source;
    if (source_mode == "local" || source_mode == "auto") {
        const uintptr_t payload = reinterpret_cast<uintptr_t>(named[names[3]].payload);
        const uintptr_t base = payload - offsets[names[3]];
        local_source = std::make_shared<vbuf_ggml::LocalVbufRangeSource>(
            reinterpret_cast<const uint8_t *>(base), std::filesystem::file_size(argv[1]));
    }
    if (source_mode == "remote" || source_mode == "auto") {
        remote_source = std::make_shared<vbuf_ggml::HttpRangeSource>(argv[5]);
    }
    std::vector<SourceDescriptor> source_descriptors;
    if (source_mode == "local" || source_mode == "auto") {
        source_descriptors.push_back({ "local", "file", true, true,
            source_mode == "auto" ? std::strtoull(argv[6], nullptr, 10) : 1, 0, local_source });
    }
    if (source_mode == "remote" || source_mode == "auto") {
        source_descriptors.push_back({ "remote", "http", true, true,
            source_mode == "auto" ? std::strtoull(argv[7], nullptr, 10) : 1, 0, remote_source });
    }
    std::shared_ptr<vbuf_ggml::RangeSource> range_source;
    std::shared_ptr<vbuf_ggml::RangeSource> fallback_source;
    std::string selected_source_id = "none";
    if (prefetch_enabled) {
        const auto selection = SourceSelectionPolicy().select(source_descriptors,
            named[names[3]].payload_len);
        if (!selection.selected_source) {
            std::fprintf(stderr, "source selection failed: %s\n", selection.reason.c_str());
            return 7;
        }
        for (const auto & estimate : selection.estimates) {
            std::printf("source_candidate=%s eligible=%s estimated_ns=%llu reason=%s\n",
                estimate.source_id.c_str(), estimate.eligible ? "yes" : "no",
                static_cast<unsigned long long>(estimate.estimated_time_ns), estimate.reason.c_str());
        }
        std::printf("source_selection=%s fallback=%s selection_ns=%llu reason=%s\n",
            source_descriptors[*selection.selected_source].id.c_str(),
            selection.fallback_source ? source_descriptors[*selection.fallback_source].id.c_str() : "none",
            static_cast<unsigned long long>(selection.selection_duration_ns), selection.reason.c_str());
        range_source = source_descriptors[*selection.selected_source].source;
        selected_source_id = source_descriptors[*selection.selected_source].id;
        if (selection.fallback_source) fallback_source =
            source_descriptors[*selection.fallback_source].source;
    }
    const uint64_t prefetch_budget = [&]() {
        uint64_t value = 0;
        for (const auto & tensor : graph.persistent) value = std::max(value, tensor.view.payload_len);
        return value;
    }();
    std::shared_ptr<TensorMaterializer> materializer;
    std::shared_ptr<TensorResidencyStore> residency;
    if (prefetch_enabled) {
        auto backing = std::make_shared<LocalVbufRangeMaterializer>(range_source, fallback_source);
        if (warm) {
            residency = std::make_shared<TensorResidencyStore>(prefetch_budget);
            materializer = std::make_shared<ResidentTensorMaterializer>(backing, residency);
        } else {
            materializer = std::move(backing);
        }
    }
    uint64_t storage_calls = 0;
    const auto storage_provider = [model_lease, &storage_calls](const VbufTensorView & view) {
        ++storage_calls;
        const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
        const uintptr_t base = address & ~static_cast<uintptr_t>(63);
        return VbufBorrowedStorage{
            reinterpret_cast<const uint8_t *>(base),
            static_cast<uint64_t>(address - base) + view.payload_len,
            static_cast<uint64_t>(address - base), model_lease };
    };

    std::vector<uint8_t> actual;
    std::vector<int64_t> actual_shape;
    TensorWaveReport report;
    AdapterError error;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> op_timings;
    bool prefetch_requested = false;
    const auto planning_observer = [&](const vbuf_ggml::TensorWavePlannerState & state) {
        const uint64_t before = storage_calls;
        const PrefetchPlan plan = planner.plan(graph, state, 3, UINT64_MAX);
        const uint64_t after = storage_calls;
        print_plan(graph, state, plan, before, after);
        if (prefetch_enabled && !prefetch_requested && materializer) {
            const vbuf_ggml::PrefetchCandidate * selected = nullptr;
            for (const auto & candidate : plan.candidates) {
                if (candidate.dependency_distance == 0) continue;
                if (selected == nullptr || candidate.dependency_distance > selected->dependency_distance) {
                    selected = &candidate;
                }
            }
            if (selected != nullptr) {
                const bool accepted = materializer->request(selected->tensor_ref,
                    graph.persistent[selected->tensor_ref], prefetch_budget);
        std::printf("prefetch_request tensor=%s offset=%llu length=%llu distance=%llu budget=%llu accepted=%d rss_kib=%llu\n",
                    selected->tensor_name.c_str(),
                    static_cast<unsigned long long>(graph.persistent[selected->tensor_ref].source_offset),
                    static_cast<unsigned long long>(selected->byte_size),
                    static_cast<unsigned long long>(selected->dependency_distance),
                    static_cast<unsigned long long>(prefetch_budget), accepted ? 1 : 0,
                    static_cast<unsigned long long>(rss_kib()));
                prefetch_requested = accepted;
            }
        }
    };
    const vbuf_ggml::TensorWavePlanningObserver active_planning_observer =
        prefetch_enabled ? planning_observer : vbuf_ggml::TensorWavePlanningObserver{};
    const auto execution_observer = [&](const char * op_id, const char * phase) {
        const uint64_t timestamp = now_ns();
        op_timings.push_back({ op_id, { timestamp, rss_kib() } });
        std::printf("op_timing op=%s phase=%s timestamp_ns=%llu rss_kib=%llu\n", op_id, phase,
            static_cast<unsigned long long>(timestamp),
            static_cast<unsigned long long>(op_timings.back().second.second));
    };
    const int run_count = warm ? 2 : 1;
    for (int run = 0; run < run_count; ++run) {
        actual.clear();
        actual_shape.clear();
        report = {};
        op_timings.clear();
        const uint64_t execute_start = now_ns();
        const uint64_t rss_before = rss_kib();
        error = executor->execute(input, storage_provider, &actual, &actual_shape, &report,
            nullptr, active_planning_observer, materializer.get(), execution_observer);
        const uint64_t execute_end = now_ns();
        const uint64_t rss_after_execute = rss_kib();
        std::printf("residency_run=%d run_mode=%s source=%s endpoint=%s execute_ms=%.3f "
            "rss_before_kib=%llu rss_after_execute_kib=%llu prefetch_budget_bytes=%llu\n",
            run + 1, mode.c_str(), prefetch_enabled ? selected_source_id.c_str() : "none",
            selected_source_id == "remote" ? argv[5] : "none",
            static_cast<double>(execute_end - execute_start) / 1000000.0,
            static_cast<unsigned long long>(rss_before),
            static_cast<unsigned long long>(rss_after_execute),
            static_cast<unsigned long long>(prefetch_budget));
        if (error != AdapterError::None) break;
    }
    executor.reset();
    if (error != AdapterError::None) {
        std::fprintf(stderr, "tensor wave execution failed: %s\n",
            vbuf_ggml::adapter_error_name(error));
        return 7;
    }

    if (materializer) {
        const auto events = materializer->trace();
        bool materialization_failed = false;
        for (const auto & event : events) {
            materialization_failed = materialization_failed ||
                event.state == vbuf_ggml::MaterializationState::Failed;
            std::printf("materialization_event tensor=%s offset=%llu event=%s state=%s timestamp_ns=%llu "
                "first_byte_ns=%llu bytes=%llu returned_bytes=%llu payload_hash=%016llx status=%d source=%s content_range=%s "
                "inflight_bytes=%llu ready_bytes=%llu rss_kib=%llu\n",
                event.tensor_name.c_str(),
                static_cast<unsigned long long>(event.requested_offset),
                event.event.c_str(),
                vbuf_ggml::materialization_state_name(event.state),
                static_cast<unsigned long long>(event.timestamp_ns),
                static_cast<unsigned long long>(event.first_byte_timestamp_ns),
                static_cast<unsigned long long>(event.bytes),
                static_cast<unsigned long long>(event.returned_bytes),
                static_cast<unsigned long long>(event.payload_hash), event.status_code,
                event.source_id.c_str(), event.content_range.c_str(),
                static_cast<unsigned long long>(event.active_inflight_bytes),
                static_cast<unsigned long long>(event.active_ready_bytes),
                static_cast<unsigned long long>(event.rss_kib));
        }
        std::printf("materializer_after_execute inflight_bytes=%llu ready_bytes=%llu\n",
            static_cast<unsigned long long>(materializer->active_inflight_bytes()),
            static_cast<unsigned long long>(materializer->active_ready_bytes()));
        if (source_mode == "remote" && materialization_failed) {
            std::printf("remote_prefetch_fallback=explicit_local_jit\n");
        }
    }
    if (residency) {
        for (const auto & event : residency->trace()) {
            std::printf("residency_event tensor_ref=%u tensor=%s event=%s resident_before=%llu "
                "resident_after=%llu active_leases=%u timestamp_ns=%llu source=%s\n",
                event.tensor_ref, event.tensor_name.c_str(),
                vbuf_ggml::residency_event_name(event.kind),
                static_cast<unsigned long long>(event.resident_bytes_before),
                static_cast<unsigned long long>(event.resident_bytes_after),
                event.active_leases, static_cast<unsigned long long>(event.timestamp_ns),
                event.source_id.c_str());
        }
        std::printf("residency_summary resident_count=%zu resident_bytes=%llu active_lease_count=%u "
            "active_lease_bytes=%llu\n", residency->resident_count(),
            static_cast<unsigned long long>(residency->resident_bytes()),
            residency->active_lease_count(),
            static_cast<unsigned long long>(residency->active_lease_bytes()));
        std::printf("partial_layer_residency resident=blk.0.ffn_down.weight "
            "nonresident=blk.0.ffn_norm.weight,blk.0.ffn_gate.weight,blk.0.ffn_up.weight\n");
        residency->clear();
        std::printf("residency_teardown resident_resources=%zu resident_bytes=%llu\n",
            residency->resident_count(), static_cast<unsigned long long>(residency->resident_bytes()));
    }

    for (size_t index = 0; index < report.trace.size(); ++index) {
        const auto & step = report.trace[index];
        std::printf("step=%zu op_id=%s op_kind=%s", index + 1, step.op_id.c_str(),
            vbuf_ggml::tensor_wave_op_name(step.op_kind));
        print_names("runnable_before", step.runnable_before);
        print_names("persistent_acquired", step.persistent_acquired);
        print_names("persistent_released", step.persistent_released);
        print_names("input_values_consumed", step.input_values_consumed);
        print_names("output_values_produced", step.output_values_produced);
        print_names("values_released", step.values_released);
        std::printf(" active_weight_bytes_during=%llu active_tensor_count_during=%llu "
            "active_weight_bytes_after=%llu active_tensor_count_after=%llu active_transient_bytes=%llu\n",
            static_cast<unsigned long long>(step.active_persistent_weight_bytes_during),
            static_cast<unsigned long long>(step.active_persistent_tensor_count_during),
            static_cast<unsigned long long>(step.active_persistent_weight_bytes),
            static_cast<unsigned long long>(step.active_persistent_tensor_count),
            static_cast<unsigned long long>(step.active_transient_bytes));
    }
    for (const auto & lifetime : report.persistent_lifetimes) {
        std::printf("lifetime tensor=%s consumers=%llu first=%s last=%s acquire_step=%llu "
            "release_step=%llu duration_steps=%llu bytes=%llu\n", lifetime.name.c_str(),
            static_cast<unsigned long long>(lifetime.consumer_count),
            lifetime.first_consumer.c_str(), lifetime.last_consumer.c_str(),
            static_cast<unsigned long long>(lifetime.acquire_step),
            static_cast<unsigned long long>(lifetime.release_step),
            static_cast<unsigned long long>(lifetime.release_step - lifetime.acquire_step + 1),
            static_cast<unsigned long long>(lifetime.bytes));
    }
    std::printf("total_model_bytes=%llu total_ffn_weight_bytes=%llu peak_active_weight_bytes=%llu "
        "sum_acquired_weight_bytes=%llu persistent_leases_after_graph=%d "
        "external_output_survives_graph_release=%d output_shape=[%lld,%lld]\n",
        static_cast<unsigned long long>(std::filesystem::file_size(argv[1])),
        static_cast<unsigned long long>(report.total_ffn_weight_bytes),
        static_cast<unsigned long long>(report.peak_active_weight_bytes),
        static_cast<unsigned long long>(report.sum_acquired_weight_bytes),
        report.all_persistent_released ? 0 : 1,
        report.external_output_preserved ? 1 : 0,
        static_cast<long long>(actual_shape[0]), static_cast<long long>(actual_shape[1]));

    const std::vector<uint8_t> swiglu_reference(
        swiglu_owner->data, swiglu_owner->data + swiglu_owner->size);
    const std::vector<uint8_t> output_reference(
        output_owner->data, output_owner->data + output_owner->size);
    const bool activation_parity = compare_output("activation", report.captured_value,
        swiglu_reference);
    const bool output_parity = compare_output("output", actual, output_reference);
    std::ofstream output_file(std::string(argv[3]) + "/ffn_out.f32", std::ios::binary);
    output_file.write(reinterpret_cast<const char *>(actual.data()),
        static_cast<std::streamsize>(actual.size()));
    return activation_parity && output_parity ? 0 : 8;
}
