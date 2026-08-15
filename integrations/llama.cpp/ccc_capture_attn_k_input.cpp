// Research-only capture of Qwen3 layer-0 attention-K matrix inputs.
#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct Capture {
    FILE * values = nullptr;
    FILE * inventory = nullptr;
    const std::vector<llama_token> * tokens = nullptr;
    int prompt = -1;
    int64_t vectors = 0;
    bool association_verified = true;
};

static bool capture_attn_norm_0(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & capture = *static_cast<Capture *>(user_data);
    const bool target = std::strcmp(ggml_get_name(tensor), "attn_norm-0") == 0;
    if (ask) return target;
    if (!target || tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 5120 || tensor->ne[1] <= 0 || !ggml_is_contiguous(tensor)) {
        capture.association_verified = false;
        return false;
    }
    const int64_t count = tensor->ne[0] * tensor->ne[1];
    if (!capture.tokens || size_t(tensor->ne[1]) != capture.tokens->size()) {
        capture.association_verified = false;
        return false;
    }
    std::vector<float> values(count);
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    if (std::fwrite(values.data(), sizeof(float), values.size(), capture.values) != values.size()) {
        capture.association_verified = false;
        return false;
    }
    for (int64_t token = 0; token < tensor->ne[1]; ++token) {
        std::fprintf(capture.inventory, "%lld,%d,%lld,%d,%s,%d,%s\n", (long long) capture.vectors++, capture.prompt,
            (long long) token, (*capture.tokens)[token], "attn_norm-0", 0, "blk.0.attn_k.weight");
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    std::vector<llama_token> tokens(256);
    int32_t count = llama_tokenize(vocab, text, std::strlen(text), tokens.data(), tokens.size(), true, false);
    if (count < 0) {
        tokens.resize(size_t(-count));
        count = llama_tokenize(vocab, text, std::strlen(text), tokens.data(), tokens.size(), true, false);
    }
    if (count < 0) return {};
    tokens.resize(count);
    return tokens;
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    const char * prompts[] = {
        "The old observatory recorded a clear winter sky while researchers checked every instrument before dawn.",
        "A compact numerical experiment should separate measurement, validation, and the final decision with care.",
        "During the afternoon, the engineer reviewed a matrix calculation and documented the assumptions precisely.",
        "Reliable systems use small interfaces, explicit bounds, and repeatable evidence rather than optimistic guesses.",
    };
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.check_tensors = false;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) return 3;
    FILE * values = std::fopen(argv[2], "wb");
    FILE * inventory = std::fopen(argv[3], "w");
    if (!values || !inventory) return 4;
    std::fprintf(inventory, "vector_id,prompt_id,token_position,token_id,graph_node,layer,tensor\n");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    Capture capture{values, inventory};
    for (int prompt = 0; prompt < 4; ++prompt) {
        auto tokens = tokenize(vocab, prompts[prompt]);
        if (tokens.empty()) return 5;
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 256;
        context_params.n_batch = 256;
        context_params.n_threads = 2;
        context_params.n_threads_batch = 2;
        context_params.cb_eval = capture_attn_norm_0;
        context_params.cb_eval_user_data = &capture;
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) return 6;
        capture.tokens = &tokens;
        capture.prompt = prompt;
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        const int status = llama_decode(context, batch);
        llama_free(context);
        if (status != 0 || !capture.association_verified) return 7;
    }
    std::fclose(values);
    std::fclose(inventory);
    std::printf("{\"vectors\":%lld,\"dimension\":5120,\"graph_node\":\"attn_norm-0\",\"layer\":0,\"tensor\":\"blk.0.attn_k.weight\",\"dtype\":\"F32\",\"prompts\":4}\n", (long long) capture.vectors);
    llama_model_free(model);
    llama_backend_free();
    return capture.vectors >= 64 ? 0 : 8;
}
