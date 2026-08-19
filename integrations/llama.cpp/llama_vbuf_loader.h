#pragma once

#include "llama.h"

#include <cstdint>

namespace vbuf_llama {
struct Step28PreparationResult;
struct VbufRemoteMetrics;
struct VbufTransportControlResult;
}

#ifdef __cplusplus
extern "C" {
#endif

// Loads a validated vBuf-ML Qwen3 model through llama_model_init_from_user.
// The returned model must be released with llama_model_free_vbuf so the
// mmap-backed consumer handle remains alive for the GGML tensor lifetime.
struct llama_model * llama_model_load_vbuf(const char * path, struct llama_model_params params);
void llama_model_free_vbuf(struct llama_model * model);

// Direct source-neutral path. The compatibility entry point above remains the
// Step-21 oracle and is intentionally unchanged.
struct llama_model * llama_model_load_vbuf_direct(const char * path, struct llama_model_params params);
struct llama_model * llama_model_load_vbuf_remote(const char * bootstrap_path, const char * endpoint, struct llama_model_params params);
bool llama_model_probe_vbuf_remote(const char * bootstrap_path, const char * endpoint, vbuf_llama::VbufRemoteMetrics * metrics);
bool llama_model_transport_control_vbuf_remote(const char * bootstrap_path, const char * endpoint, bool large_range, vbuf_llama::VbufTransportControlResult * result);
bool llama_model_vbuf_remote_metrics(struct llama_model * model, vbuf_llama::VbufRemoteMetrics * metrics);
const char * llama_model_last_remote_error();
void llama_model_free_vbuf_direct(struct llama_model * model);
bool llama_model_step28_prepare_vbuf(struct llama_model * model, const char * variant, vbuf_llama::Step28PreparationResult * result);

#ifdef __cplusplus
}
#endif
