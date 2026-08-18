#include "vbuf_ml_adapter.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

struct AlignedLease {
    std::shared_ptr<void> owner;
    std::shared_ptr<const void> view;
    const uint8_t * bytes = nullptr;
};

AlignedLease read_materialized(uint64_t length) {
    void * raw = nullptr;
    if (posix_memalign(&raw, 64, static_cast<size_t>(length)) != 0) throw std::runtime_error("materialized allocation failed");
    std::shared_ptr<void> owner(raw, std::free);
    std::cin.read(static_cast<char *>(raw), static_cast<std::streamsize>(length));
    if (std::cin.gcount() != static_cast<std::streamsize>(length)) throw std::runtime_error("materialized span is truncated");
    return { owner, std::shared_ptr<const void>(owner, raw), static_cast<const uint8_t *>(raw) };
}

bool descriptor_equal(const vbuf_llama::TensorDescriptor & left, const vbuf_llama::TensorDescriptor & right) {
    return left.name == right.name && left.dimensions == right.dimensions && left.type == right.type && left.payload_bytes == right.payload_bytes;
}

uint64_t descriptor_hash(const ggml_tensor * tensor) {
    uint64_t hash = 1469598103934665603ULL;
    auto add = [&hash](uint64_t value) { for (unsigned shift = 0; shift < 64; shift += 8) { hash ^= (value >> shift) & 0xff; hash *= 1099511628211ULL; } };
    add(static_cast<uint64_t>(tensor->type));
    for (int index = 0; index < GGML_MAX_DIMS; ++index) { add(static_cast<uint64_t>(tensor->ne[index])); add(static_cast<uint64_t>(tensor->nb[index])); }
    add(ggml_nbytes(tensor));
    return hash;
}

float run_sum(const vbuf_llama::TensorDescriptor & descriptor) {
    ggml_init_params params{ 1024 * 1024, nullptr, true };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("ggml context allocation failed");
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) { ggml_free(context); throw std::runtime_error("ggml CPU backend unavailable"); }
    std::vector<int64_t> dimensions(descriptor.dimensions.begin(), descriptor.dimensions.end());
    ggml_tensor * tensor = ggml_new_tensor(context, descriptor.type, static_cast<int>(dimensions.size()), dimensions.data());
    if (tensor == nullptr || ggml_nbytes(tensor) != descriptor.payload_bytes) throw std::runtime_error("ggml tensor geometry mismatch");
    ggml_backend_buffer_t input_buffer = ggml_backend_cpu_buffer_from_ptr(const_cast<uint8_t *>(descriptor.payload), static_cast<size_t>(descriptor.payload_bytes));
    if (input_buffer == nullptr || ggml_backend_tensor_alloc(input_buffer, tensor, const_cast<uint8_t *>(descriptor.payload)) != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml input binding failed");
    ggml_tensor * output = ggml_sum(context, tensor);
    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (compute_buffer == nullptr || ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml bounded compute failed");
    float result = 0.0f;
    ggml_backend_tensor_get(output, &result, 0, sizeof(result));
    ggml_backend_buffer_free(compute_buffer);
    ggml_backend_buffer_free(input_buffer);
    ggml_backend_free(backend);
    ggml_free(context);
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <local-vbuf|none> <semantic-bootstrap> <ordinal> [compute]\n", argv[0]); return 2; }
    try {
        const std::string local_path = argv[1];
        const uint64_t ordinal = std::stoull(argv[3]);
        std::unique_ptr<vbuf_llama::VbufMlAdapter> local;
        if (local_path != "none") local = std::make_unique<vbuf_llama::VbufMlAdapter>(local_path.c_str());
        vbuf_llama::VbufMlAdapter external(argv[2], true);
        if (!external.valid()) throw std::runtime_error("metadata-only adapter open failed");
        vbuf_llama::TensorDescriptor external_metadata{};
        if (!external.tensor_metadata(ordinal, external_metadata)) throw std::runtime_error("external descriptor discovery failed");
        AlignedLease materialized = read_materialized(external_metadata.payload_bytes);
        vbuf_llama::TensorDescriptor external_tensor{};
        if (!external.tensor_materialized(ordinal, materialized.bytes, external_metadata.payload_bytes, materialized.view, external_tensor)) throw std::runtime_error("external materialized binding failed");
        if (external_tensor.source_id == 0 || external_tensor.source_offset == 0) throw std::runtime_error("external source provenance was lost");
        std::printf("external source_id=%llu offset=%llu bytes=%llu type=%d rank=%zu\n", (unsigned long long)external_tensor.source_id, (unsigned long long)external_tensor.source_offset, (unsigned long long)external_tensor.payload_bytes, static_cast<int>(external_tensor.type), external_tensor.dimensions.size());
        bool descriptor_pass = true;
        float local_result = 0.0f, external_result = 0.0f;
        if (local) {
            vbuf_llama::TensorDescriptor local_tensor{};
            if (!local->tensor(ordinal, local_tensor)) throw std::runtime_error("local descriptor discovery failed");
            descriptor_pass = descriptor_equal(local_tensor, external_tensor);
            std::printf("local source_id=%llu offset=%llu bytes=%llu type=%d rank=%zu\n", (unsigned long long)local_tensor.source_id, (unsigned long long)local_tensor.source_offset, (unsigned long long)local_tensor.payload_bytes, static_cast<int>(local_tensor.type), local_tensor.dimensions.size());
            ggml_init_params descriptor_params{ 1024 * 1024, nullptr, true };
            ggml_context * descriptor_context = ggml_init(descriptor_params);
            std::vector<int64_t> local_dimensions(local_tensor.dimensions.begin(), local_tensor.dimensions.end());
            std::vector<int64_t> external_dimensions(external_tensor.dimensions.begin(), external_tensor.dimensions.end());
            ggml_tensor * local_ggml = ggml_new_tensor(descriptor_context, local_tensor.type, static_cast<int>(local_dimensions.size()), local_dimensions.data());
            ggml_tensor * external_ggml = ggml_new_tensor(descriptor_context, external_tensor.type, static_cast<int>(external_dimensions.size()), external_dimensions.data());
            std::printf("local_ggml_descriptor_hash=%016llx external_ggml_descriptor_hash=%016llx\n", (unsigned long long)descriptor_hash(local_ggml), (unsigned long long)descriptor_hash(external_ggml));
            if (descriptor_hash(local_ggml) != descriptor_hash(external_ggml)) descriptor_pass = false;
            ggml_free(descriptor_context);
            if (argc > 4 && std::string(argv[4]) == "compute") { local_result = run_sum(local_tensor); external_result = run_sum(external_tensor); }
        } else {
            ggml_init_params descriptor_params{ 1024 * 1024, nullptr, true };
            ggml_context * descriptor_context = ggml_init(descriptor_params);
            std::vector<int64_t> dimensions(external_tensor.dimensions.begin(), external_tensor.dimensions.end());
            ggml_tensor * external_ggml = ggml_new_tensor(descriptor_context, external_tensor.type, static_cast<int>(dimensions.size()), dimensions.data());
            std::printf("external_ggml_descriptor_hash=%016llx\n", (unsigned long long)descriptor_hash(external_ggml));
            ggml_free(descriptor_context);
        }
        std::printf("descriptor_parity=%s\n", descriptor_pass ? "PASS" : "FAIL");
        if (argc > 4 && std::string(argv[4]) == "compute") std::printf("compute_local=%.9g compute_external=%.9g compute_parity=%s\n", local_result, external_result, local_result == external_result ? "PASS" : "FAIL");
        return descriptor_pass && (!(argc > 4 && std::string(argv[4]) == "compute") || local_result == external_result) ? 0 : 1;
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
