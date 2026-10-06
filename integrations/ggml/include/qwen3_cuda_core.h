#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "qwen3_model.h"
#include "vbuf_residency.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vbuf_ggml {

class QwenExecutionPlanOptimizer;

struct Qwen3CudaLayerWeights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * attn_out = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
};

struct Qwen3CudaLayerGraph {
    ggml_tensor * hidden = nullptr;
    ggml_tensor * scores = nullptr;
    ggml_tensor * probabilities = nullptr;
};

using Qwen3CudaCaptureTensor = std::function<void(const char *, ggml_tensor *)>;

// Shared graph math used by decode and bounded prefill. Callers supply and own
// the model weights, session cache, scratch tensors, and graph arena.
Qwen3CudaLayerGraph qwen3_cuda_build_layer(ggml_context * context,
    const Qwen3CudaLayerWeights & weights, ggml_tensor * hidden,
    ggml_tensor * position_ids, ggml_tensor * causal_mask, ggml_tensor * cache_rows,
    ggml_tensor * key_cache, ggml_tensor * value_cache, ggml_tensor * packed_value_scratch,
    uint32_t query_count, uint32_t capacity, const Qwen3CudaCaptureTensor & capture = {});

struct QwenCudaPlacement {
    std::vector<uint32_t> block_device_ids;
    uint32_t embedding_device_id = 0;
    uint32_t output_norm_device_id = 0;
    uint32_t output_head_device_id = 0;

    static QwenCudaPlacement single_device(uint32_t device_id = 0);
    static QwenCudaPlacement contiguous_split(uint32_t early_device_id,
        uint32_t late_device_id, uint32_t early_block_count);
    void validate(const Qwen3Model & model) const;
    uint32_t owner_for_tensor(const std::string & tensor_name) const;
    std::vector<uint32_t> device_ids() const;
};

enum class QwenCudaFailurePoint : uint8_t {
    None,
    RuntimeAfterBackend,
    RuntimeAfterModelAllocation,
    RuntimeAfterResidency,
    SessionAfterKvAllocation,
    SessionAfterPrefillScratch,
    SessionAfterDecodeScratch,
    ExecutionBeforeBoundary,
    ExecutionAfterBoundaryD2H,
    ExecutionAfterBoundaryH2D,
    ExecutionBeforeLateBlock,
    ExecutionBeforeFinalBlock,
};

struct QwenCudaRuntimeConfig {
    uint32_t prefill_chunk_size = 32;
    QwenCudaPlacement placement = QwenCudaPlacement::single_device();
    size_t prefill_scratch_bytes = 64 * 1024 * 1024;
    size_t decode_scratch_bytes = 32 * 1024 * 1024;
    bool allow_experimental_capacity = false;
    QwenCudaFailurePoint inject_failure = QwenCudaFailurePoint::None;
};

// Model-lifetime owner. A production instance borrows the already-admitted
// Qwen3Model; that model must outlive this state. Sessions retain this state
// with shared_ptr, so a runtime cannot be destroyed while a session is alive.
class QwenCudaRuntimeState final : public std::enable_shared_from_this<QwenCudaRuntimeState> {
public:
    static std::shared_ptr<QwenCudaRuntimeState> create(Qwen3Model & admitted_model,
        QwenCudaRuntimeConfig config = {});

    // Small CPU-backed fixture for lifecycle contracts; never used by runtime
    // dispatch or qualification. It exercises the same resource owners.
    static std::shared_ptr<QwenCudaRuntimeState> create_for_testing(
        QwenCudaRuntimeConfig config = {});

    ~QwenCudaRuntimeState();
    QwenCudaRuntimeState(const QwenCudaRuntimeState &) = delete;
    QwenCudaRuntimeState & operator=(const QwenCudaRuntimeState &) = delete;

    std::shared_ptr<class QwenCudaSessionState> create_session(uint32_t capacity,
        QwenCudaFailurePoint inject_failure = QwenCudaFailurePoint::None);

    ggml_backend_t backend() const noexcept;
    ggml_backend_t backend(uint32_t device_id) const;
    ggml_backend_dev_t device() const noexcept;
    ggml_backend_dev_t device(uint32_t device_id) const;
    uint32_t device_count() const noexcept;
    uint32_t owner_device(const std::string & tensor_name) const;
    const QwenCudaPlacement & placement() const noexcept;
    void device_memory(size_t * free_bytes, size_t * total_bytes) const noexcept;
    void device_memory(uint32_t device_id, size_t * free_bytes, size_t * total_bytes) const;
    ggml_context * model_context() const noexcept;
    ggml_context * model_context(uint32_t device_id) const;
    ggml_backend_buffer_t model_allocation() const noexcept;
    const std::string & backend_name() const noexcept;
    const std::string & artifact_identity() const noexcept;
    const ModelMetadataDescriptor * metadata() const noexcept;
    ggml_tensor * tensor(const std::string & name) const;
    ggml_tensor * tensor(const std::string & name, uint32_t device_id) const;
    ggml_tensor * embedding() const noexcept;
    std::shared_ptr<TensorResidencyStore> residency() const noexcept;
    uint64_t resident_model_bytes() const noexcept;
    uint64_t uploaded_payload_bytes() const noexcept;
    size_t uploaded_tensor_count() const noexcept;
    size_t resident_allocation_bytes() const noexcept;
    size_t resident_tensor_count() const noexcept;
    uint64_t uploaded_payload_bytes(uint32_t device_id) const;
    size_t uploaded_tensor_count(uint32_t device_id) const;
    size_t resident_allocation_bytes(uint32_t device_id) const;
    size_t resident_tensor_count(uint32_t device_id) const;
    uint32_t prefill_chunk_size() const noexcept;
    bool experimental_capacity_enabled() const noexcept;
    QwenExecutionPlanOptimizer & execution_optimizer() noexcept;

private:
    struct Impl;
    explicit QwenCudaRuntimeState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class QwenCudaSessionState;
};

// Session-lifetime owner. KV and execution scratch are private to each object;
// reset invalidates logical visibility without requiring a cache memset.
class QwenCudaSessionState final {
public:
    ~QwenCudaSessionState();
    QwenCudaSessionState(const QwenCudaSessionState &) = delete;
    QwenCudaSessionState & operator=(const QwenCudaSessionState &) = delete;

    void reset() noexcept;
    void commit_tokens(uint32_t count);
    uint32_t current_length() const noexcept;
    uint32_t capacity() const noexcept;
    uint64_t reset_generation() const noexcept;

    const std::shared_ptr<QwenCudaRuntimeState> & runtime() const noexcept;
    // context() owns persistent K/V and packed-V tensors; graph_context() is
    // session-owned storage for reusable decode graph tensors.
    ggml_context * context() const noexcept;
    ggml_context * context(uint32_t device_id) const;
    ggml_context * graph_context() const noexcept;
    ggml_context * graph_context(uint32_t device_id) const;
    ggml_context * create_auxiliary_context(size_t arena_bytes);
    ggml_context * create_auxiliary_context(uint32_t device_id, size_t arena_bytes);
    ggml_backend_buffer_t allocation() const noexcept;
    ggml_backend_buffer_t allocation(uint32_t device_id) const;
    ggml_backend_buffer_t allocate_decode_scratch();
    ggml_backend_buffer_t allocate_decode_scratch(uint32_t device_id);
    size_t allocation_bytes() const noexcept;
    size_t allocation_bytes(uint32_t device_id) const;
    size_t prefill_scratch_bytes() const noexcept;
    size_t prefill_scratch_bytes(uint32_t device_id) const;
    size_t decode_scratch_bytes() const noexcept;
    size_t decode_scratch_bytes(uint32_t device_id) const;
    ggml_backend_buffer_t prefill_scratch() const noexcept;
    ggml_backend_buffer_t prefill_scratch(uint32_t device_id) const;
    ggml_backend_buffer_t decode_scratch() const noexcept;
    ggml_backend_buffer_t decode_scratch(uint32_t device_id) const;
    ggml_tensor * key_cache(uint32_t layer) const;
    ggml_tensor * key_cache(uint32_t layer, uint32_t device_id) const;
    ggml_tensor * value_cache(uint32_t layer) const;
    ggml_tensor * value_cache(uint32_t layer, uint32_t device_id) const;
    ggml_tensor * packed_value_scratch() const noexcept;
    ggml_tensor * packed_value_scratch(uint32_t device_id) const;
    void * boundary_host_data() const noexcept;
    size_t boundary_host_bytes() const noexcept;
    bool boundary_host_is_pinned() const noexcept;
    // Empty means audit every handoff; otherwise audit only transfers ending at these positions.
    void set_boundary_audit_end_positions(std::vector<uint32_t> positions);
    bool should_audit_boundary(uint32_t end_position) const noexcept;
    void set_execution_failure(QwenCudaFailurePoint point);
    void inject_execution_failure(QwenCudaFailurePoint point, const char * label) const;

private:
    struct Impl;
    explicit QwenCudaSessionState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class QwenCudaRuntimeState;
};

} // namespace vbuf_ggml
