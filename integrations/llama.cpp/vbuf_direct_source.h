#pragma once

#include "llama-model-source.h"

#include <memory>

namespace vbuf_llama {

std::shared_ptr<llama_model_source> make_vbuf_direct_source(const char * path);
void set_vbuf_direct_tensor_data(ggml_tensor * tensor, void * userdata);

} // namespace vbuf_llama
