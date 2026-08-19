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
    TopKSelection selection;
};

// Consumes only the Rust-owned lowered graph. The resolver is the already-
// resolved vBuf-ML TensorBinding/payload boundary; this function performs no
// source lookup, model detection, or tensor-name interpretation.
bool execute_portable_graph(const VbufRuntimeGraphHandle * graph,
    const PortableActivation & input, const std::shared_ptr<const void> & lease,
    const PortableTensorResolver & resolver, PortableRouterPrefixResult * result,
    std::string * error = nullptr);

} // namespace vbuf_ggml
