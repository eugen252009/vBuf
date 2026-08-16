#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "vbuf_region_executor.h"

namespace {

using vbuf_ggml::AdapterError;
using vbuf_ggml::BorrowedGgmlTensor;
using vbuf_ggml::VbufBorrowedStorage;
using vbuf_ggml::VbufTensorView;

struct AlignedBytes {
    alignas(64) std::array<uint8_t, 256> bytes{};
};

bool expect_error(AdapterError actual, AdapterError expected, const char * label) {
    if (actual != expected) {
        std::fprintf(stderr, "%s: got %s, expected %s\n", label,
            vbuf_ggml::adapter_error_name(actual),
            vbuf_ggml::adapter_error_name(expected));
        return false;
    }
    return true;
}

bool descriptor_cases() {
    struct Case {
        uint8_t representation;
        uint64_t block_elements;
        uint64_t block_bytes;
    } cases[] = {
        { 0, 1, 4 },
        { 5, 256, 50 },
        { 7, 32, 18 },
        { 10, 256, 66 },
        { 4, 256, 84 },
        { 13, 256, 176 },
    };
    for (const Case & item : cases) {
        const uint64_t dimensions[] = { item.block_elements, 2 };
        const uint64_t rows = dimensions[1];
        const uint64_t payload_len = rows * item.block_bytes;
        std::vector<uint8_t> payload(payload_len);
        VbufTensorView view{ item.representation, 2, dimensions,
            payload.data(), payload_len };
        vbuf_ggml::TensorGeometry geometry{};
        std::string detail;
        if (!expect_error(vbuf_ggml::derive_tensor_geometry(view, &geometry, &detail),
                AdapterError::None, "valid descriptor")) return false;
        if (geometry.nbytes != payload_len || geometry.ne[0] != static_cast<int64_t>(item.block_elements) ||
            geometry.ne[1] != 2 || geometry.nb[0] != item.block_bytes ||
            geometry.nb[1] != item.block_bytes) {
            std::fprintf(stderr, "descriptor geometry mismatch for representation %u: %s\n",
                item.representation, detail.c_str());
            return false;
        }
        AdapterError error = AdapterError::None;
        auto tensor = BorrowedGgmlTensor::create(view, &error, &detail);
        if (!tensor || error != AdapterError::None || tensor->geometry().nbytes != payload_len) {
            std::fprintf(stderr, "ggml descriptor construction failed for representation %u: %s\n",
                item.representation, detail.c_str());
            return false;
        }
    }
    return true;
}

bool malformed_cases() {
    const uint64_t dimensions[] = { 256, 1 };
    AlignedBytes storage;
    VbufTensorView view{ 4, 2, dimensions, storage.bytes.data(), 84 };
    vbuf_ggml::TensorGeometry geometry{};
    std::string detail;
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 99, 2, dimensions, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::UnsupportedRepresentation, "unknown representation")) return false;
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 2, dimensions, storage.bytes.data(), 83 },
            &geometry, &detail), AdapterError::PayloadSizeMismatch, "payload mismatch")) return false;
    const uint64_t bad_width[] = { 255, 1 };
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 2, bad_width, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::InvalidBlockGeometry, "block divisibility")) return false;
    const uint64_t zero[] = { 0, 1 };
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 2, zero, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::InvalidDimension, "zero dimension")) return false;
    const uint64_t overflow[] = { std::numeric_limits<uint64_t>::max(), 1 };
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 2, overflow, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::InvalidDimension, "dimension overflow")) return false;
    const uint64_t product_overflow[] = { 256, static_cast<uint64_t>(INT64_MAX), 2 };
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 3, product_overflow, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::ShapeOverflow, "shape product overflow")) return false;
    if (!expect_error(vbuf_ggml::derive_tensor_geometry(
            VbufTensorView{ 4, 0, dimensions, storage.bytes.data(), 84 },
            &geometry, &detail), AdapterError::InvalidRank, "zero rank")) return false;
    return true;
}

bool borrowed_f32_execution() {
    struct alignas(64) F32Storage { uint8_t prefix[64]{}; float values[2]; };
    auto owner = std::make_shared<F32Storage>();
    owner->values[0] = 3.0f;
    owner->values[1] = 4.0f;
    std::shared_ptr<const void> lease(owner, static_cast<const void *>(owner.get()));
    const uint64_t dimensions[] = { 2, 1 };
    VbufTensorView view{ 0, 2, dimensions,
        reinterpret_cast<const uint8_t *>(owner->values), sizeof(owner->values) };
    auto tensor = BorrowedGgmlTensor::create(view);
    const auto base = reinterpret_cast<const uint8_t *>(owner.get());
    const auto payload = reinterpret_cast<const uint8_t *>(owner->values);
    const VbufBorrowedStorage storage{
        base, sizeof(*owner), static_cast<uint64_t>(payload - base), lease };
    std::string bind_detail;
    const VbufBorrowedStorage out_of_range{
        base, static_cast<uint64_t>(payload - base + 4),
        static_cast<uint64_t>(payload - base), lease };
    if (!tensor || tensor->bind_cpu(out_of_range, &bind_detail) !=
        AdapterError::StorageRangeMismatch) {
        std::fprintf(stderr, "out-of-range storage was accepted\n");
        return false;
    }
    const AdapterError f32_bind = tensor
        ? tensor->bind_cpu(storage, &bind_detail) : AdapterError::InvalidArgument;
    if (!tensor || f32_bind != AdapterError::None) {
        std::fprintf(stderr, "F32 bind failed: %s %s\n",
            vbuf_ggml::adapter_error_name(f32_bind), bind_detail.c_str());
        return false;
    }
    ggml_tensor * output = ggml_scale(tensor->context(), tensor->tensor(), 2.0f);
    ggml_cgraph * graph = ggml_new_graph(tensor->context());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t compute = ggml_backend_alloc_ctx_tensors(
        tensor->context(), tensor->backend());
    if (compute == nullptr || ggml_backend_graph_compute(tensor->backend(), graph) != GGML_STATUS_SUCCESS) {
        if (compute != nullptr) ggml_backend_buffer_free(compute);
        std::fprintf(stderr, "F32 compute failed\n");
        return false;
    }
    float result[2]{};
    ggml_backend_tensor_get(output, result, 0, sizeof(result));
    const bool pass = tensor->bound_data() == reinterpret_cast<const uint8_t *>(owner->values) &&
        result[0] == 6.0f && result[1] == 8.0f && owner->values[0] == 3.0f;
    ggml_backend_buffer_free(compute);
    if (!pass) std::fprintf(stderr, "F32 output/pointer mismatch: %g %g %p %p\n",
        result[0], result[1], tensor->bound_data(), owner->values);
    return pass;
}

bool borrowed_q2_execution() {
    struct alignas(64) Q2Storage { uint8_t bytes[84]{}; };
    auto owner = std::make_shared<Q2Storage>();
    const uint64_t dimensions[] = { 256, 1 };
    VbufTensorView view{ 4, 2, dimensions, owner->bytes, sizeof(owner->bytes) };
    auto tensor = BorrowedGgmlTensor::create(view);
    std::shared_ptr<const void> lease(owner, static_cast<const void *>(owner.get()));
    std::string bind_detail;
    const AdapterError q2_bind = tensor
        ? tensor->bind_cpu_exact(lease, &bind_detail) : AdapterError::InvalidArgument;
    if (!tensor || q2_bind != AdapterError::None) {
        std::fprintf(stderr, "Q2 bind failed: %s %s\n",
            vbuf_ggml::adapter_error_name(q2_bind), bind_detail.c_str());
        return false;
    }
    ggml_tensor * input = ggml_new_tensor_1d(tensor->context(), GGML_TYPE_F32, 256);
    ggml_tensor * output = ggml_mul_mat(tensor->context(), tensor->tensor(), input);
    ggml_cgraph * graph = ggml_new_graph(tensor->context());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t compute = ggml_backend_alloc_ctx_tensors(
        tensor->context(), tensor->backend());
    if (compute == nullptr) {
        std::fprintf(stderr, "Q2 compute allocation failed\n");
        return false;
    }
    std::array<float, 256> values{};
    values.fill(1.0f);
    ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
    const bool computed = ggml_backend_graph_compute(tensor->backend(), graph) == GGML_STATUS_SUCCESS;
    float result = 1.0f;
    ggml_backend_tensor_get(output, &result, 0, sizeof(result));
    std::printf("Q2_K source=%p tensor_data=%p type=%d shape=[%lld,%lld] "
        "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] expected_bytes=%zu "
        "actual_bytes=%zu result=%g reference=0\n",
        static_cast<const void *>(owner->bytes), tensor->bound_data(),
        static_cast<int>(tensor->tensor()->type),
        static_cast<long long>(dimensions[0]), static_cast<long long>(dimensions[1]),
        static_cast<long long>(tensor->tensor()->ne[0]),
        static_cast<long long>(tensor->tensor()->ne[1]),
        static_cast<long long>(tensor->tensor()->ne[2]),
        static_cast<long long>(tensor->tensor()->ne[3]),
        tensor->tensor()->nb[0], tensor->tensor()->nb[1],
        tensor->tensor()->nb[2], tensor->tensor()->nb[3],
        sizeof(owner->bytes), ggml_nbytes(tensor->tensor()), result);
    const bool pass = computed && result == 0.0f && tensor->bound_data() == owner->bytes;
    ggml_backend_buffer_free(compute);
    if (!pass) std::fprintf(stderr, "Q2 output mismatch: computed=%d result=%g ptr=%p expected=%p\n",
        computed, result, tensor->bound_data(), owner->bytes);
    return pass;
}

} // namespace

int main() {
    const bool descriptors = descriptor_cases();
    const bool malformed = malformed_cases();
    const bool f32 = borrowed_f32_execution();
    const bool q2 = borrowed_q2_execution();
    const bool pass = descriptors && malformed && f32 && q2;
    std::printf("descriptors=%d malformed=%d f32=%d q2=%d\n",
        descriptors, malformed, f32, q2);
    std::printf("tensor_adapter_qualification=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
