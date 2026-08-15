#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include "llama.h"
#include "common.h"
#include "ggml.h"
#include "ggml-quants.h"

static std::vector<float> g_captured_activations;
static int g_captured_token_count = 0;
static std::string g_target_tensor_name = "blk.32.attn_k.weight";

static bool cb_eval_capture(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    if (!t) return true;
    
    bool matches = false;
    struct ggml_tensor * act_tensor = nullptr;
    
    for (int i = 0; i < 2; i++) {
        if (t->src[i] && t->src[i]->name) {
            std::string s_name(t->src[i]->name);
            if (s_name == g_target_tensor_name || s_name.find("blk.32.attn_k") != std::string::npos || s_name.find("blk.0.attn_k") != std::string::npos) {
                matches = true;
                act_tensor = t->src[1 - i];
                break;
            }
        }
    }
    
    if (!matches && t->name) {
        std::string t_name(t->name);
        if (t_name.find("blk.32.attn_k") != std::string::npos || t_name.find("blk.0.attn_k") != std::string::npos) {
            matches = true;
            act_tensor = t->src[1] ? t->src[1] : t->src[0];
        }
    }
    
    if (matches && act_tensor) {
        int64_t in_dim = act_tensor->ne[0];
        int64_t n_tokens = act_tensor->ne[1];
        
        if ((in_dim == 1024 || in_dim == 5120) && act_tensor->data && act_tensor->type == GGML_TYPE_F32) {
            const float * data = (const float *) act_tensor->data;
            size_t total_floats = in_dim * n_tokens;
            g_captured_activations.insert(g_captured_activations.end(), data, data + total_floats);
            g_captured_token_count += n_tokens;
            std::cout << "[CAPTURE SEAM] Captured " << n_tokens << " vectors of dim " << in_dim 
                      << " (Total captured: " << g_captured_token_count << ")" << std::endl;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> <output_activations.bin> [target_tensor_name]" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string out_path = argv[2];
    if (argc >= 4) {
        g_target_tensor_name = argv[3];
    }
    
    std::cout << "Target tensor name: " << g_target_tensor_name << std::endl;
    std::cout << "Loading model: " << model_path << std::endl;
    
    common_params params;
    params.model.path = model_path;
    params.n_gpu_layers = 0; // Force pure CPU memory allocation
    params.n_ctx = 2048;
    params.n_predict = 128;
    params.cb_eval = cb_eval_capture;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;
    
    common_init();
    llama_backend_init();
    
    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    
    if (!model || !ctx) {
        std::cerr << "Failed to initialize llama model or context!" << std::endl;
        return 1;
    }
    
    std::vector<std::string> prompts = {
        "The Contextual Correction Code (CCC) is a model-weight compression representation designed to reduce storage bandwidth.",
        "In artificial intelligence and deep neural networks, quantitative optimization of weight matrices is essential for low-latency inference on hardware platforms.",
        "DeepSeek and Qwen represent state-of-the-art open-weights large language models with mixture-of-experts architectures.",
        "Signal processing theory, Gersho's theorem, and companding quantizers provide fundamental mathematical bounds for scalar codebooks."
    };
    
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    
    for (size_t p_idx = 0; p_idx < prompts.size(); p_idx++) {
        std::vector<llama_token> tokens = common_tokenize(ctx, prompts[p_idx], add_bos, true);
        std::cout << "Prompt " << p_idx + 1 << ": token count = " << tokens.size() << std::endl;
        if (!tokens.empty()) {
            llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
        }
    }
    
    std::cout << "\nCaptured total " << g_captured_token_count << " activation vectors!" << std::endl;
    
    if (!g_captured_activations.empty()) {
        std::ofstream fout(out_path, std::ios::binary);
        uint64_t n_vecs = g_captured_token_count;
        uint64_t dim = g_captured_activations.size() / n_vecs;
        
        fout.write(reinterpret_cast<const char*>(&n_vecs), sizeof(n_vecs));
        fout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        fout.write(reinterpret_cast<const char*>(g_captured_activations.data()), g_captured_activations.size() * sizeof(float));
        fout.close();
        
        std::cout << "Saved binary activation dataset to " << out_path << " (" << n_vecs << " vectors x " << dim << " dim)" << std::endl;
    } else {
        std::cerr << "Warning: No activation vectors captured!" << std::endl;
    }
    
    llama_backend_free();
    return 0;
}
