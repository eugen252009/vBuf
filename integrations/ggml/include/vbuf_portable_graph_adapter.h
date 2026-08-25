#pragma once

#include "vbuf_portable_graph_ffi.h"
#include "vbuf_topk.h"
#include "vbuf_tensor_wave.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vbuf_ggml {

struct PortableResolvedTensor {
    // The resolver must provide a validated ready payload. No source or range
    // lookup occurs in the generic backend adapter.
    PersistentTensorRef ref;
};

struct PortableActivation {
    std::vector<float> values;
    std::vector<uint64_t> dimensions;
};

using PortableTensorResolver = std::function<bool(uint32_t, PortableResolvedTensor *,
    std::string *)>;

struct PortableRouterPrefixResult {
    std::vector<float> normalized;
    std::vector<float> logits;
    std::vector<float> attention_output;
    TopKSelection selection;
    struct AttentionReport {
        uint64_t q_bytes = 0;
        uint64_t k_bytes = 0;
        uint64_t v_bytes = 0;
        uint64_t score_bytes = 0;
        uint64_t probability_bytes = 0;
        uint64_t output_bytes = 0;
        uint64_t state_bytes = 0;
        uint64_t scratch_peak_bytes = 0;
        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f;
    } attention;
};

struct VbufPortableExecutionState;

extern "C" VbufPortableExecutionState * vbuf_portable_execution_state_create(
    uint32_t binding_id, uint64_t capacity);
extern "C" void vbuf_portable_execution_state_reset(VbufPortableExecutionState * state);
extern "C" void vbuf_portable_execution_state_destroy(VbufPortableExecutionState * state);
extern "C" uint64_t vbuf_portable_execution_state_length(
    const VbufPortableExecutionState * state);
extern "C" uint64_t vbuf_portable_execution_state_bytes(
    const VbufPortableExecutionState * state);

// Generic graph operation kinds exported by the vBuf runtime ABI. Attention
// remains a separate contract because it requires explicit sequence/state
// geometry; unsupported kinds must fail closed in this adapter.
enum class PortableGraphOperationKind : uint32_t {
    RmsNorm = 1,
    MatMul = 2,
    Activation = 3,
    TopK = 4,
    IndexedMatMul = 5,
    Attention = 6,
    ResidualAdd = 7,
};

// Consumes only the Rust-owned lowered graph. The resolver is the already-
// resolved vBuf-ML TensorBinding/payload boundary; this function performs no
// source lookup, model detection, or tensor-name interpretation.
bool execute_portable_graph(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, PortableRouterPrefixResult * result,
    std::string * error = nullptr);

bool execute_portable_graph_with_state(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, VbufPortableExecutionState * state,
    PortableRouterPrefixResult * result, std::string * error = nullptr);

} // namespace vbuf_ggml
