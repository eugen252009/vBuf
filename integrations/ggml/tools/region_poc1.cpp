#include "vbuf_execution_region.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <memory>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

using vbuf_ggml::AdapterError;
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

struct MemorySnapshot {
    uint64_t rss_kib = 0;
    uint64_t anonymous_kib = 0;
    uint64_t private_clean_kib = 0;
    uint64_t shared_clean_kib = 0;
};

MemorySnapshot memory_snapshot() {
    std::ifstream input("/proc/self/smaps_rollup");
    std::string line;
    uint64_t value;
    MemorySnapshot result;
    while (std::getline(input, line)) {
        std::istringstream fields(line);
        std::string key;
        fields >> key >> value;
        if (fields.fail()) continue;
        if (key == "Rss:") result.rss_kib = value;
        if (key == "Anonymous:") result.anonymous_kib = value;
        if (key == "Private_Clean:") result.private_clean_kib = value;
        if (key == "Shared_Clean:") result.shared_clean_kib = value;
    }
    return result;
}

bool compare_output(const std::vector<uint8_t> & actual, const std::vector<uint8_t> & reference) {
    if (actual.size() != reference.size() || actual.size() % sizeof(float) != 0) return false;
    const auto * lhs = reinterpret_cast<const float *>(actual.data());
    const auto * rhs = reinterpret_cast<const float *>(reference.data());
    const size_t count = actual.size() / sizeof(float);
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float abs_error = std::fabs(lhs[i] - rhs[i]);
        const float rel_error = abs_error / std::max(std::fabs(rhs[i]), 1e-12f);
        sum_abs += abs_error;
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, rel_error);
    }
    std::printf("output_elements=%zu max_abs=%g max_rel=%g mean_abs=%g "
        "actual_hash=%016llx reference_hash=%016llx\n", count, max_abs, max_rel,
        static_cast<float>(sum_abs / count),
        static_cast<unsigned long long>(fnv1a(actual)),
        static_cast<unsigned long long>(fnv1a(reference)));
    return max_abs <= 2e-4f && (max_rel <= 2e-3f || max_abs <= 1e-5f);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: region_poc1 <vbuf> <capture-dir> <output-dir>\n");
        return 2;
    }
    auto input_owner = read_aligned((std::string(argv[2]) + "/ffn_inp.f32").c_str());
    auto swiglu_reference_owner = read_aligned(
        (std::string(argv[2]) + "/ffn_swiglu.f32").c_str());
    auto reference_owner = read_aligned((std::string(argv[2]) + "/ffn_out.f32").c_str());
    if (!input_owner || !swiglu_reference_owner || !reference_owner ||
        input_owner->size % (2048 * sizeof(float)) != 0) return 3;
    const uint64_t token_count = input_owner->size / (2048 * sizeof(float));
    const uint64_t input_dimensions[] = { 2048, token_count };
    const VbufTensorView input{
        0, 2, input_dimensions, input_owner->data, input_owner->size };

    VbufMlConsumerHandle * handle = vbuf_ml_consumer_open(argv[1]);
    if (handle == nullptr) return 4;
    const VbufMlTensorView * views = nullptr;
    uint64_t view_count = 0;
    if (vbuf_ml_consumer_tensor_views(handle, &views, &view_count) != 0) return 5;
    std::unordered_map<std::string, VbufTensorView> named;
    std::unordered_map<std::string, uint64_t> ids;
    for (uint64_t index = 0; index < view_count; ++index) {
        const VbufMlTensorView & source = views[index];
        named.emplace(std::string(source.name, source.name_len), VbufTensorView{
            source.representation, source.rank, source.dimensions,
            source.payload, source.payload_len });
        ids.emplace(std::string(source.name, source.name_len), index);
    }
    const char * required[] = {
        "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight" };
    for (const char * name : required) if (!named.count(name)) return 6;

    vbuf_ggml::ExecutionRegion region_a;
    const uint32_t input_slot_a = region_a.add_input();
    const uint32_t norm_slot = region_a.add_weight(ids[required[0]], named[required[0]]);
    const uint32_t gate_slot = region_a.add_weight(ids[required[1]], named[required[1]]);
    const uint32_t up_slot = region_a.add_weight(ids[required[2]], named[required[2]]);
    const uint32_t normalized_slot = region_a.add_temporary();
    const uint32_t gate_output_slot = region_a.add_temporary();
    const uint32_t up_output_slot = region_a.add_temporary();
    const uint32_t swiglu_slot = region_a.add_output();
    region_a.add_operation({ vbuf_ggml::RegionOpKind::RmsNorm,
        normalized_slot, input_slot_a, norm_slot, 1.0e-6f });
    region_a.add_operation({ vbuf_ggml::RegionOpKind::MulMat,
        gate_output_slot, gate_slot, normalized_slot, 0.0f });
    region_a.add_operation({ vbuf_ggml::RegionOpKind::MulMat,
        up_output_slot, up_slot, normalized_slot, 0.0f });
    region_a.add_operation({ vbuf_ggml::RegionOpKind::SwiGluSplit,
        swiglu_slot, gate_output_slot, up_output_slot, 0.0f });

    vbuf_ggml::ExecutionRegion region_b;
    const uint32_t input_slot_b = region_b.add_input();
    const uint32_t down_slot = region_b.add_weight(ids[required[3]], named[required[3]]);
    const uint32_t output_slot_b = region_b.add_output();
    region_b.add_operation({ vbuf_ggml::RegionOpKind::MulMat,
        output_slot_b, down_slot, input_slot_b, 0.0f });

    auto model_lease = std::shared_ptr<const void>(handle,
        [handle](const void *) { vbuf_ml_consumer_close(handle); });
    std::vector<uint8_t> activation;
    std::vector<uint8_t> actual;
    std::vector<int64_t> activation_shape;
    std::vector<int64_t> output_shape;
    vbuf_ggml::RegionExecutionReport report_a;
    vbuf_ggml::RegionExecutionReport report_b;
    const MemorySnapshot memory_before = memory_snapshot();
    AdapterError error = AdapterError::None;
    {
        const auto storage_provider = [model_lease](const VbufTensorView & view) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(view.payload);
            const uintptr_t base = address & ~static_cast<uintptr_t>(63);
            return VbufBorrowedStorage{
                reinterpret_cast<const uint8_t *>(base),
                static_cast<uint64_t>(address - base) + view.payload_len,
                static_cast<uint64_t>(address - base), model_lease };
        };
        const auto trace = [](const char * region_name) {
            return [region_name](const char * phase, uint64_t bytes, uint64_t leases) {
                std::printf("region=%s phase=%s active_weight_bytes=%llu active_weight_leases=%llu\n",
                    region_name, phase, static_cast<unsigned long long>(bytes),
                    static_cast<unsigned long long>(leases));
            };
        };
        error = region_a.execute(input, storage_provider,
            &activation, &activation_shape, &report_a, nullptr, trace("A"));
        if (error == AdapterError::None) {
            const uint64_t activation_dimensions[] = {
                static_cast<uint64_t>(activation_shape[0]),
                static_cast<uint64_t>(activation_shape[1]) };
            const VbufTensorView activation_view{
                0, 2, activation_dimensions, activation.data(), activation.size() };
            error = region_b.execute(activation_view, storage_provider,
                &actual, &output_shape, &report_b, nullptr, trace("B"));
        }
    }
    const MemorySnapshot memory_after_execute = memory_snapshot();
    if (error != AdapterError::None) {
        std::fprintf(stderr, "region execution failed: %s\n",
            vbuf_ggml::adapter_error_name(error));
        return 7;
    }
    std::vector<uint8_t> reference(reference_owner->data,
        reference_owner->data + reference_owner->size);
    const std::vector<uint8_t> swiglu_reference(
        swiglu_reference_owner->data,
        swiglu_reference_owner->data + swiglu_reference_owner->size);
    std::printf("activation_parity=");
    const bool activation_parity = compare_output(activation, swiglu_reference);
    std::printf("output_parity=");
    const bool output_parity = compare_output(actual, reference);
    const uint64_t mapped_source_bytes = std::filesystem::file_size(argv[1]);
    model_lease.reset();
    const MemorySnapshot memory_after_release = memory_snapshot();
    std::printf("requested_tensor_count=%zu acquired_tensor_count=%zu "
        "selected_weight_bytes=%llu acquired_weight_bytes=%llu graph_nodes=%llu "
        "mapped_source_bytes=%llu "
        "rss_before_kib=%llu rss_after_execute_kib=%llu rss_after_release_kib=%llu "
        "anonymous_before_kib=%llu anonymous_after_execute_kib=%llu anonymous_after_release_kib=%llu "
        "private_clean_before_kib=%llu shared_clean_before_kib=%llu "
        "output_shape=[%lld,%lld]\n",
        report_a.requested_tensor_ids.size() + report_b.requested_tensor_ids.size(),
        report_a.acquired_tensor_ids.size() + report_b.acquired_tensor_ids.size(),
        static_cast<unsigned long long>(report_a.selected_weight_bytes + report_b.selected_weight_bytes),
        static_cast<unsigned long long>(report_a.acquired_weight_bytes + report_b.acquired_weight_bytes),
        static_cast<unsigned long long>(report_a.graph_tensor_count + report_b.graph_tensor_count),
        static_cast<unsigned long long>(mapped_source_bytes),
        static_cast<unsigned long long>(memory_before.rss_kib),
        static_cast<unsigned long long>(memory_after_execute.rss_kib),
        static_cast<unsigned long long>(memory_after_release.rss_kib),
        static_cast<unsigned long long>(memory_before.anonymous_kib),
        static_cast<unsigned long long>(memory_after_execute.anonymous_kib),
        static_cast<unsigned long long>(memory_after_release.anonymous_kib),
        static_cast<unsigned long long>(memory_before.private_clean_kib),
        static_cast<unsigned long long>(memory_before.shared_clean_kib),
        static_cast<long long>(output_shape[0]), static_cast<long long>(output_shape[1]));
    std::ofstream output_file(std::string(argv[3]) + "/region_output.f32", std::ios::binary);
    output_file.write(reinterpret_cast<const char *>(actual.data()),
        static_cast<std::streamsize>(actual.size()));
    std::ofstream activation_file(std::string(argv[3]) + "/region_a_activation.f32", std::ios::binary);
    activation_file.write(reinterpret_cast<const char *>(activation.data()),
        static_cast<std::streamsize>(activation.size()));
    return activation_parity && output_parity ? 0 : 8;
}
