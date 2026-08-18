#include "vbuf_region_executor.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

extern ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace vbuf_ggml {
namespace {

struct RepresentationSpec {
    ggml_type type;
    uint64_t block_elements;
    uint64_t block_bytes;
};

bool checked_mul(uint64_t left, uint64_t right, uint64_t * result) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        return false;
    }
    *result = left * right;
    return true;
}

bool spec_for(uint8_t representation, RepresentationSpec * spec) {
    switch (representation) {
    case 0: *spec = { GGML_TYPE_F32, 1, 4 }; return true;
    case 5: *spec = { GGML_TYPE_IQ1_S, 256, 50 }; return true;
    case 7: *spec = { GGML_TYPE_IQ4_NL, 32, 18 }; return true;
    case 10: *spec = { GGML_TYPE_IQ2_XXS, 256, 66 }; return true;
    case 4: *spec = { GGML_TYPE_Q2_K, 256, 84 }; return true;
    case 13: *spec = { GGML_TYPE_Q5_K, 256, 176 }; return true;
    default: return false;
    }
}

void set_detail(std::string * detail, const char * message) {
    if (detail != nullptr) {
        *detail = message;
    }
}

uint64_t audit_hash(const uint8_t * data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

} // namespace

const char * adapter_error_name(AdapterError error) {
    switch (error) {
    case AdapterError::None: return "NONE";
    case AdapterError::InvalidArgument: return "INVALID_ARGUMENT";
    case AdapterError::UnsupportedRepresentation: return "UNSUPPORTED_REPRESENTATION";
    case AdapterError::InvalidRank: return "INVALID_RANK";
    case AdapterError::InvalidDimension: return "INVALID_DIMENSION";
    case AdapterError::ShapeOverflow: return "SHAPE_OVERFLOW";
    case AdapterError::InvalidBlockGeometry: return "INVALID_BLOCK_GEOMETRY";
    case AdapterError::PayloadSizeMismatch: return "PAYLOAD_SIZE_MISMATCH";
    case AdapterError::ContextAllocationFailed: return "CONTEXT_ALLOCATION_FAILED";
    case AdapterError::TensorConstructionFailed: return "TENSOR_CONSTRUCTION_FAILED";
    case AdapterError::BackendUnavailable: return "BACKEND_UNAVAILABLE";
    case AdapterError::StorageRangeMismatch: return "STORAGE_RANGE_MISMATCH";
    case AdapterError::StorageAlignment: return "STORAGE_ALIGNMENT";
    case AdapterError::TensorAlreadyBound: return "TENSOR_ALREADY_BOUND";
    case AdapterError::BackendAllocationFailed: return "BACKEND_ALLOCATION_FAILED";
    }
    return "UNKNOWN";
}

AdapterError derive_tensor_geometry(
    const VbufTensorView & view,
    TensorGeometry * geometry,
    std::string * detail) {
    if (geometry == nullptr || view.dimensions == nullptr || view.payload == nullptr) {
        set_detail(detail, "tensor view contains a null required pointer");
        return AdapterError::InvalidArgument;
    }
    if (view.rank == 0 || view.rank > GGML_MAX_DIMS) {
        set_detail(detail, "rank is outside the ggml descriptor range");
        return AdapterError::InvalidRank;
    }

    RepresentationSpec spec{};
    if (!spec_for(view.representation, &spec)) {
        set_detail(detail, "representation is not supported by this adapter");
        return AdapterError::UnsupportedRepresentation;
    }

    TensorGeometry result{};
    result.type = spec.type;
    result.rank = view.rank;
    result.block_elements = spec.block_elements;
    result.block_bytes = spec.block_bytes;
    uint64_t rows = 1;
    for (uint8_t index = 0; index < view.rank; ++index) {
        const uint64_t dimension = view.dimensions[index];
        if (dimension == 0 || dimension > static_cast<uint64_t>(INT64_MAX)) {
            set_detail(detail, "tensor dimensions must be non-zero signed int64 values");
            return AdapterError::InvalidDimension;
        }
        result.ne[index] = static_cast<int64_t>(dimension);
        if (index > 0 && !checked_mul(rows, dimension, &rows)) {
            set_detail(detail, "row count overflows uint64");
            return AdapterError::ShapeOverflow;
        }
    }
    for (uint8_t index = view.rank; index < GGML_MAX_DIMS; ++index) {
        result.ne[index] = 1;
    }

    const uint64_t row_width = view.dimensions[0];
    if (row_width % spec.block_elements != 0) {
        set_detail(detail, "innermost row is not divisible by representation block width");
        return AdapterError::InvalidBlockGeometry;
    }
    uint64_t blocks = 0;
    if (!checked_mul(rows, row_width / spec.block_elements, &blocks) ||
        !checked_mul(blocks, spec.block_bytes, &blocks)) {
        set_detail(detail, "payload size overflows uint64");
        return AdapterError::ShapeOverflow;
    }
    if (view.payload_len != blocks) {
        set_detail(detail, "payload length does not match representation and shape");
        return AdapterError::PayloadSizeMismatch;
    }

    result.nb[0] = static_cast<size_t>(spec.block_bytes);
    for (uint8_t index = 1; index < GGML_MAX_DIMS; ++index) {
        const uint64_t previous = result.nb[index - 1];
        const uint64_t previous_ne = static_cast<uint64_t>(result.ne[index - 1]);
        const uint64_t divisor = index == 1 ? spec.block_elements : 1;
        uint64_t next = previous_ne;
        if (divisor != 1) {
            next /= divisor;
        }
        if (!checked_mul(previous, next, &next) || next > SIZE_MAX) {
            set_detail(detail, "runtime stride overflows size_t");
            return AdapterError::ShapeOverflow;
        }
        result.nb[index] = static_cast<size_t>(next);
    }
    result.nbytes = static_cast<size_t>(blocks);
    *geometry = result;
    return AdapterError::None;
}

std::unique_ptr<BorrowedGgmlTensor> BorrowedGgmlTensor::create(
    const VbufTensorView & view,
    AdapterError * error,
    std::string * detail) {
    TensorGeometry geometry{};
    AdapterError result = derive_tensor_geometry(view, &geometry, detail);
    if (result != AdapterError::None) {
        if (error != nullptr) *error = result;
        return nullptr;
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        set_detail(detail, "ggml context allocation failed");
        if (error != nullptr) *error = AdapterError::ContextAllocationFailed;
        return nullptr;
    }
    ggml_tensor * tensor = ggml_new_tensor(context, geometry.type, geometry.rank, geometry.ne);
    if (tensor == nullptr || ggml_nbytes(tensor) != geometry.nbytes) {
        ggml_free(context);
        set_detail(detail, "ggml descriptor size differs from canonical payload size");
        if (error != nullptr) *error = AdapterError::TensorConstructionFailed;
        return nullptr;
    }
    for (int index = 0; index < GGML_MAX_DIMS; ++index) {
        if (tensor->nb[index] != geometry.nb[index]) {
            ggml_free(context);
            set_detail(detail, "ggml descriptor stride differs from canonical contiguous stride");
            if (error != nullptr) *error = AdapterError::TensorConstructionFailed;
            return nullptr;
        }
    }
    if (error != nullptr) *error = AdapterError::None;
    return std::unique_ptr<BorrowedGgmlTensor>(
        new BorrowedGgmlTensor(geometry, context, tensor, view.payload));
}

BorrowedGgmlTensor::BorrowedGgmlTensor(
    TensorGeometry geometry,
    ggml_context * context,
    ggml_tensor * tensor,
    const uint8_t * payload)
    : geometry_(geometry), context_(context), tensor_(tensor), payload_(payload) {}

BorrowedGgmlTensor::~BorrowedGgmlTensor() {
    if (buffer_ != nullptr) {
        ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
    }
    if (context_ != nullptr) {
        ggml_free(context_);
        context_ = nullptr;
        tensor_ = nullptr;
    }
    if (backend_ != nullptr) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    lease_.reset();
}

AdapterError BorrowedGgmlTensor::bind_cpu(
    const VbufBorrowedStorage & storage,
    std::string * detail) {
    if (buffer_ != nullptr) {
        set_detail(detail, "tensor is already bound");
        return AdapterError::TensorAlreadyBound;
    }
    if (storage.base == nullptr || storage.size == 0 ||
        storage.payload_offset > storage.size ||
        geometry_.nbytes > storage.size - storage.payload_offset ||
        storage.base + storage.payload_offset != payload_) {
        set_detail(detail, "storage span does not contain the complete tensor payload");
        return AdapterError::StorageRangeMismatch;
    }
    backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_ == nullptr) {
        set_detail(detail, "CPU backend initialization failed");
        return AdapterError::BackendUnavailable;
    }
    const size_t alignment = ggml_backend_get_alignment(backend_);
    if (reinterpret_cast<uintptr_t>(storage.base) % alignment != 0) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
        set_detail(detail, "containing storage base does not meet CPU backend alignment");
        return AdapterError::StorageAlignment;
    }
    buffer_ = ggml_backend_cpu_buffer_from_ptr(
        const_cast<uint8_t *>(storage.base), static_cast<size_t>(storage.size));
    if (buffer_ == nullptr || ggml_backend_tensor_alloc(
        buffer_, tensor_, const_cast<uint8_t *>(payload_)) != GGML_STATUS_SUCCESS) {
        if (buffer_ != nullptr) ggml_backend_buffer_free(buffer_);
        buffer_ = nullptr;
        ggml_backend_free(backend_);
        backend_ = nullptr;
        set_detail(detail, "CPU external buffer tensor allocation failed");
        return AdapterError::BackendAllocationFailed;
    }
    lease_ = storage.lease;
    return AdapterError::None;
}

AdapterError BorrowedGgmlTensor::bind_cpu_repacked(
    const VbufBorrowedStorage & storage,
    std::string * detail) {
    if (buffer_ != nullptr || storage.base == nullptr || storage.size == 0 ||
        storage.payload_offset > storage.size || geometry_.nbytes > storage.size - storage.payload_offset ||
        storage.base + storage.payload_offset != payload_) {
        set_detail(detail, "repacked tensor storage span mismatch");
        return AdapterError::StorageRangeMismatch;
    }
    backend_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend_ == nullptr) {
        set_detail(detail, "CPU backend initialization failed");
        return AdapterError::BackendUnavailable;
    }
    const ggml_backend_buffer_type_t repack_buft = ::ggml_backend_cpu_repack_buffer_type();
    if (repack_buft == nullptr) {
        ggml_backend_free(backend_); backend_ = nullptr;
        set_detail(detail, "CPU repack buffer type unavailable");
        return AdapterError::BackendUnavailable;
    }
    const size_t allocation_size = ggml_backend_buft_get_alloc_size(repack_buft, tensor_);
    if (std::getenv("VBUF_AUDIT_REPACKED_DOWN") != nullptr)
        std::fprintf(stderr, "REPACK_DESCRIPTOR raw_nbytes=%zu alloc_bytes=%zu ne0=%lld ne1=%lld nb0=%zu nb1=%zu\n",
            geometry_.nbytes, allocation_size, static_cast<long long>(tensor_->ne[0]),
            static_cast<long long>(tensor_->ne[1]), tensor_->nb[0], tensor_->nb[1]);
    if (std::getenv("VBUF_REPACK_V3_GATE_ONLY") != nullptr &&
        allocation_size != 2711642112ULL) {
        std::fprintf(stderr,
            "VBUF_REPACK_V3_FIRST_FAILED_GATE=REPACK_ALLOCATION_PARITY "
            "llama_alloc_bytes=2711642112 vbuf_alloc_bytes=%zu alignment=UNKNOWN\n",
            allocation_size);
        set_detail(detail, "vBuf intervention 3 stopped before set_tensor: allocation contract mismatch");
        ggml_backend_free(backend_);
        backend_ = nullptr;
        return AdapterError::BackendAllocationFailed;
    }
    buffer_ = ggml_backend_buft_alloc_buffer(repack_buft, allocation_size);
    if (buffer_ == nullptr) {
        ggml_backend_free(backend_); backend_ = nullptr;
        set_detail(detail, "CPU repack buffer allocation failed");
        return AdapterError::BackendAllocationFailed;
    }
    void * base = ggml_backend_buffer_get_base(buffer_);
    if (base == nullptr || ggml_backend_tensor_alloc(buffer_, tensor_, base) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer_); buffer_ = nullptr;
        ggml_backend_free(backend_); backend_ = nullptr;
        set_detail(detail, "CPU repack tensor initialization failed");
        return AdapterError::BackendAllocationFailed;
    }
    ggml_backend_tensor_set(tensor_, payload_, 0, geometry_.nbytes);
    if (std::getenv("VBUF_AUDIT_REPACKED_DOWN") != nullptr) {
        const uint8_t * data = static_cast<const uint8_t *>(tensor_->data);
        std::fprintf(stderr,
            "VBUF_REPACK_V3_STORAGE bytes=%zu hash=%016llx data_offset=0 data=%p base=%p\n",
            geometry_.nbytes,
            static_cast<unsigned long long>(audit_hash(data, geometry_.nbytes)),
            tensor_->data, base);
        std::fprintf(stderr, "REPACK_STORAGE_FIRST16=");
        for (size_t i = 0; i < 16; ++i) std::fprintf(stderr, "%s%02x", i ? ":" : "", data[i]);
        std::fprintf(stderr, "\n");
        static FILE * packed_dump = nullptr;
        if (packed_dump == nullptr) {
            packed_dump = std::fopen("/tmp/opencode/vbuf-packed.bin", "wb");
            if (packed_dump != nullptr) {
                std::fwrite(data, 1, geometry_.nbytes, packed_dump);
                std::fflush(packed_dump);
            }
        }
    }
    lease_ = storage.lease;
    return AdapterError::None;
}

AdapterError BorrowedGgmlTensor::bind_cpu_exact(
    std::shared_ptr<const void> lease,
    std::string * detail) {
    VbufBorrowedStorage storage{
        payload_,
        geometry_.nbytes,
        0,
        std::move(lease),
    };
    return bind_cpu(storage, detail);
}

const uint8_t * BorrowedGgmlTensor::bound_data() const {
    return tensor_ == nullptr ? nullptr : static_cast<const uint8_t *>(tensor_->data);
}

const char * backend_name() {
    static char name[64];
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) return "unavailable";
    std::snprintf(name, sizeof(name), "%s", ggml_backend_name(backend));
    ggml_backend_free(backend);
    return name;
}

} // namespace vbuf_ggml

extern "C" const char * vbuf_region_executor_backend_name(void) {
    return vbuf_ggml::backend_name();
}
