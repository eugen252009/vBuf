#include "llama.h"
#include "llama_vbuf_loader.h"
#include "bench_trace.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

static int64_t elapsed_us(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

static llama_token greedy(const float * logits, int32_t n_vocab) {
    return static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
}

static bool env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) return 2;

    const char * path = argv[2];
    const int n_generate = 32;
    const bool trace_tokens = env_flag("VBUF_AB_TOKENS");
    vbuf_bench::event("PROCESS_START", "vbuf");
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.no_alloc = false;

    const auto process_start = Clock::now();
    vbuf_bench::event("MODEL_OPEN_BEGIN", "vbuf");
    llama_model * model = llama_model_load_vbuf_direct(path, model_params);
    vbuf_bench::event("MODEL_OPEN_END", model ? "success" : "failure");
    if (!model) {
        vbuf_bench::event("RUN_END", "model_failure");
        llama_backend_free();
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt_tokens(64);
    const char * prompt = "Hello world";
    int32_t n_prompt = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)),
                                      prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()), true, false);
    if (n_prompt < 0) {
        prompt_tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab, prompt, static_cast<int32_t>(std::strlen(prompt)),
                                  prompt_tokens.data(), -n_prompt, true, false);
    }
    if (n_prompt <= 0) return 4;
    prompt_tokens.resize(static_cast<size_t>(n_prompt));
    std::fprintf(stderr, "AB_CONFIG context=256 batch=64 ubatch=64 threads=8 prompt=Hello world prompt_tokens=%d generate=%d sampling=greedy seed=default\n", n_prompt, n_generate);

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
    const auto prompt_start = Clock::now();
    const int prompt_rc = llama_decode(context, batch);
    const int64_t prompt_us = elapsed_us(prompt_start);
    vbuf_bench::event("PROMPT_EVAL_END", prompt_rc == 0 ? "success" : "failure");
    if (prompt_rc != 0) return 6;

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_token token = greedy(llama_get_logits_ith(context, -1), n_vocab);
    std::vector<llama_token> generated;
    generated.reserve(n_generate);
    int64_t first_decode_us = 0;
    const auto decode_start = Clock::now();
    for (int i = 0; i < n_generate; ++i) {
        batch = llama_batch_get_one(&token, 1);
        batch.logits = logits.data();
        if (i == 0) vbuf_bench::event("FIRST_DECODE_BEGIN");
        const auto one_start = Clock::now();
        const int rc = llama_decode(context, batch);
        const int64_t one_us = elapsed_us(one_start);
        if (i == 0) {
            first_decode_us = one_us;
            vbuf_bench::event("FIRST_DECODE_END", rc == 0 ? "success" : "failure");
            if (rc == 0) vbuf_bench::event("FIRST_TOKEN");
        }
        if (rc != 0) return 7;
        generated.push_back(token);
        token = greedy(llama_get_logits_ith(context, -1), n_vocab);
    }
    const int64_t decode_us = elapsed_us(decode_start);
    std::fprintf(stderr, "AB_METRICS prompt_us=%lld prompt_tokens=%d prompt_tok_s=%.6f first_decode_us=%lld decode_us=%lld decode_tokens=%d decode_tok_s=%.6f total_us=%lld\n",
                 (long long) prompt_us, n_prompt, n_prompt * 1e6 / std::max<int64_t>(prompt_us, 1),
                 (long long) first_decode_us, (long long) decode_us, n_generate,
                 n_generate * 1e6 / std::max<int64_t>(decode_us, 1), (long long) elapsed_us(process_start));
    if (trace_tokens) {
        std::fprintf(stderr, "AB_TOKENS");
        for (llama_token value : generated) std::fprintf(stderr, " %d", value);
        std::fprintf(stderr, "\n");
    }
    std::fprintf(stderr, "BENCH_SUCCESS prompt_tokens=%d generated_tokens=%d\n", n_prompt, n_generate);
    vbuf_bench::event("RUN_END", "success");
    llama_free(context);
    llama_model_free_vbuf_direct(model);
    llama_backend_free();
    return 0;
}
