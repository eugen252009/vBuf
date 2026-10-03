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

namespace vbuf_ggml {

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

enum class QwenCudaFailurePoint : uint8_t {
    None,
    RuntimeAfterBackend,
    RuntimeAfterModelAllocation,
    RuntimeAfterResidency,
    SessionAfterKvAllocation,
    SessionAfterPrefillScratch,
    SessionAfterDecodeScratch,
};

struct QwenCudaRuntimeConfig {
    uint32_t prefill_chunk_size = 32;
    size_t prefill_scratch_bytes = 40'206'464;
    size_t decode_scratch_bytes = 32 * 1024 * 1024;
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
    ggml_backend_dev_t device() const noexcept;
    const std::string & backend_name() const noexcept;
    const std::string & artifact_identity() const noexcept;
    const ModelMetadataDescriptor * metadata() const noexcept;
    ggml_tensor * tensor(const std::string & name) const;
    ggml_tensor * embedding() const noexcept;
    std::shared_ptr<TensorResidencyStore> residency() const noexcept;
    uint64_t resident_model_bytes() const noexcept;
    size_t resident_tensor_count() const noexcept;
    uint32_t prefill_chunk_size() const noexcept;

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
    ggml_context * context() const noexcept;
    ggml_backend_buffer_t allocation() const noexcept;
    ggml_backend_buffer_t prefill_scratch() const noexcept;
    ggml_backend_buffer_t decode_scratch() const noexcept;
    ggml_tensor * key_cache(uint32_t layer) const;
    ggml_tensor * value_cache(uint32_t layer) const;
    ggml_tensor * packed_value_scratch() const noexcept;

private:
    struct Impl;
    explicit QwenCudaSessionState(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class QwenCudaRuntimeState;
};

} // namespace vbuf_ggml
