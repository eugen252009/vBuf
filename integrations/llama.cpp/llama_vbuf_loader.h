#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loads a validated vBuf-ML Qwen3 model through llama_model_init_from_user.
// The returned model must be released with llama_model_free_vbuf so the
// mmap-backed consumer handle remains alive for the GGML tensor lifetime.
struct llama_model * llama_model_load_vbuf(const char * path, struct llama_model_params params);
void llama_model_free_vbuf(struct llama_model * model);

#ifdef __cplusplus
}
#endif
