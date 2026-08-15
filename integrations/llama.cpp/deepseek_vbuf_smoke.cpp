#include "llama_vbuf_loader.h"

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = std::getenv("VBUF_SMOKE_GPU") != nullptr ? -1 : 0;
    // Keep the smoke path focused on source metadata/tensor projection when
    // VBUF_SMOKE_VOCAB_ONLY is set; full tensor loading is tested separately.
    model_params.vocab_only = std::getenv("VBUF_SMOKE_VOCAB_ONLY") != nullptr;
    model_params.no_alloc = std::getenv("VBUF_SMOKE_NO_ALLOC") != nullptr;
    std::fprintf(stderr, "deepseek-vbuf-smoke loading\n");
    llama_model * model = llama_model_load_vbuf_direct(argv[1], model_params);
    std::fprintf(stderr, "deepseek-vbuf-smoke model_loaded=%d\n", model != nullptr);
    if (!model) { llama_backend_free(); return 3; }
    if (model_params.vocab_only) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 0; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const char * prompt = "Hello world";
    std::vector<llama_token> tokens(64);
    int32_t count = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (count < 0) { tokens.resize(static_cast<size_t>(-count)); count = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), tokens.data(), -count, true, false); }
    if (count <= 0) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 4; }
    tokens.resize(static_cast<size_t>(count));
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 256;
    context_params.n_batch = 64;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 5; }
    std::vector<int8_t> logits(tokens.size(), 1);
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    batch.logits = logits.data();
    const int decode = llama_decode(context, batch);
    std::fprintf(stderr, "deepseek-vbuf-smoke tokens=%d decode=%d\n", count, decode);
    llama_free(context);
    llama_model_free_vbuf_direct(model);
    llama_backend_free();
    return decode == 0 ? 0 : 6;
}
