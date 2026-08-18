#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "ggml-backend.h"
#include "ggml.h"

namespace vbuf_ggml {

// This is the smallest generic subset of the existing Rust VbufMlTensorView
// needed by the native boundary. It contains no model, graph, or backend data.
struct VbufTensorView {
    uint8_t representation;
    uint8_t rank;
    const uint64_t * dimensions;
    const uint8_t * payload;
    uint64_t payload_len;
};

// A runtime-only containing storage span. `base` must own or borrow a range
// that contains the tensor payload at `payload_offset`. The lease keeps the
// source mapping alive until the ggml wrapper and descriptor are destroyed.
struct VbufBorrowedStorage {
    const uint8_t * base;
    uint64_t size;
    uint64_t payload_offset;
    std::shared_ptr<const void> lease;
};

enum class AdapterError {
    None,
    InvalidArgument,
    UnsupportedRepresentation,
    InvalidRank,
    InvalidDimension,
    ShapeOverflow,
    InvalidBlockGeometry,
    PayloadSizeMismatch,
    ContextAllocationFailed,
    TensorConstructionFailed,
    BackendUnavailable,
    StorageRangeMismatch,
    StorageAlignment,
    TensorAlreadyBound,
    BackendAllocationFailed,
};

struct TensorGeometry {
    ggml_type type;
    uint8_t rank;
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    size_t nbytes;
    uint64_t block_elements;
    uint64_t block_bytes;
};

const char * adapter_error_name(AdapterError error);

// Derive and validate a transient ggml descriptor without binding storage.
AdapterError derive_tensor_geometry(
    const VbufTensorView & view,
    TensorGeometry * geometry,
    std::string * detail = nullptr);

class BorrowedGgmlTensor final {
public:
    static std::unique_ptr<BorrowedGgmlTensor> create(
        const VbufTensorView & view,
        AdapterError * error = nullptr,
        std::string * detail = nullptr);

    ~BorrowedGgmlTensor();

    BorrowedGgmlTensor(const BorrowedGgmlTensor &) = delete;
    BorrowedGgmlTensor & operator=(const BorrowedGgmlTensor &) = delete;

    AdapterError bind_cpu(
        const VbufBorrowedStorage & storage,
        std::string * detail = nullptr);

    AdapterError bind_cpu_repacked(
        const VbufBorrowedStorage & storage,
        std::string * detail = nullptr);

    AdapterError bind_cpu_exact(
        std::shared_ptr<const void> lease = {},
        std::string * detail = nullptr);

    const TensorGeometry & geometry() const { return geometry_; }
    ggml_context * context() const { return context_; }
    const ggml_tensor * tensor() const { return tensor_; }
    ggml_tensor * tensor() { return tensor_; }
    ggml_backend_t backend() const { return backend_; }
    ggml_backend_buffer_t buffer() const { return buffer_; }
    const uint8_t * bound_data() const;
    std::shared_ptr<const void> lease() const { return lease_; }

private:
    BorrowedGgmlTensor(
        TensorGeometry geometry,
        ggml_context * context,
        ggml_tensor * tensor,
        const uint8_t * payload);

    TensorGeometry geometry_{};
    ggml_context * context_ = nullptr;
    ggml_tensor * tensor_ = nullptr;
    const uint8_t * payload_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::shared_ptr<const void> lease_;
};

// Substrate capability query retained from the initial build boundary.
const char * backend_name();

} // namespace vbuf_ggml

extern "C" const char * vbuf_region_executor_backend_name(void);
