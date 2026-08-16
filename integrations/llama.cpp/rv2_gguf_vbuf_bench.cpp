#include "llama.h"
#include "llama_vbuf_loader.h"
#include "bench_trace.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy(const float * logits, int32_t n_vocab) {
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const bool is_vbuf = std::strcmp(argv[1], "vbuf") == 0;
    const char * path = argv[2];
    vbuf_bench::event("PROCESS_START", is_vbuf ? "vbuf" : "gguf");
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.no_alloc = false;
    vbuf_bench::event("MODEL_OPEN_BEGIN", is_vbuf ? "vbuf" : "gguf");
    llama_model * model = is_vbuf
        ? llama_model_load_vbuf_direct(path, model_params)
        : llama_model_load_from_file(path, model_params);
    vbuf_bench::event("MODEL_OPEN_END", model ? "success" : "failure");
    if (!model) { vbuf_bench::event("RUN_END", "model_failure"); llama_backend_free(); return 3; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens(64);
    const char * prompt = "Hello world";
    int32_t n_prompt = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()), true, false);
    if (n_prompt < 0) {
        prompt_tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)), prompt_tokens.data(), -n_prompt, true, false);
    }
    if (n_prompt <= 0) return 4;
    prompt_tokens.resize(static_cast<size_t>(n_prompt));

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 256;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    context_params.n_threads = 8;
    context_params.n_threads_batch = 8;
    vbuf_bench::event("CONTEXT_CREATE_BEGIN");
    llama_context * context = llama_init_from_model(model, context_params);
    vbuf_bench::event("CONTEXT_CREATE_END", context ? "success" : "failure");
    if (!context) return 5;
    vbuf_bench::event("EXECUTION_READY");

    std::vector<int8_t> logits(prompt_tokens.size(), 1);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()));
    batch.logits = logits.data();
    vbuf_bench::event("PROMPT_EVAL_BEGIN");
    const int prompt_rc = llama_decode(context, batch);
    vbuf_bench::event("PROMPT_EVAL_END", prompt_rc == 0 ? "success" : "failure");
    if (prompt_rc != 0) return 6;

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token token = greedy(llama_get_logits_ith(context, -1), n_vocab);
    for (int i = 0; i < 3; ++i) {
        batch = llama_batch_get_one(&token, 1);
        batch.logits = logits.data();
        if (i == 0) vbuf_bench::event("FIRST_DECODE_BEGIN");
        const int rc = llama_decode(context, batch);
        if (i == 0) {
            vbuf_bench::event("FIRST_DECODE_END", rc == 0 ? "success" : "failure");
            if (rc == 0) vbuf_bench::event("FIRST_TOKEN");
        }
        if (rc != 0) return 7;
        token = greedy(llama_get_logits_ith(context, -1), n_vocab);
    }
    std::fprintf(stderr, "BENCH_SUCCESS prompt_tokens=%d generated_tokens=3\n", n_prompt);
    vbuf_bench::event("RUN_END", "success");
    llama_free(context);
    if (is_vbuf) llama_model_free_vbuf_direct(model); else llama_model_free(model);
    llama_backend_free();
    return 0;
}
