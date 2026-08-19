#include "llama_vbuf_loader.h"
#include "vbuf_direct_source.h"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
std::mutex models_mutex;
std::unordered_map<llama_model *, std::shared_ptr<llama_model_source>> models;
}

extern "C" llama_model * llama_model_load_vbuf_direct(const char * path, llama_model_params params) {
    try {
        auto source = vbuf_llama::make_vbuf_direct_source(path);
        llama_model * model = llama_model_init_from_source(source, vbuf_llama::set_vbuf_direct_tensor_data, source.get(), params);
        if (!model) return nullptr;
        std::lock_guard lock(models_mutex);
        models.emplace(model, std::move(source));
        return model;
    } catch (...) {
        return nullptr;
    }
}
