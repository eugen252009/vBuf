#include "llama_vbuf_loader.h"
#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}

struct Capture {
    std::vector<float> hidden;
    std::vector<float> k_output;
    std::vector<size_t> prompt_vectors;
    size_t pending_hidden = 0;
    size_t pending_k = 0;
    size_t hidden_callbacks = 0;
    size_t k_callbacks = 0;
    bool awaiting_k = false;
};

static bool capture_cb(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & c = *static_cast<Capture *>(userdata);
    const char * name = ggml_get_name(tensor);
    const bool hidden = std::strcmp(name, "attn_norm-32") == 0;
    const bool kout = std::strcmp(name, "Kcur-32") == 0 && c.awaiting_k;
    if (!hidden && !kout) return false;
    if (ask) {
        if (hidden) c.pending_hidden = ggml_nelements(tensor);
        if (kout) c.pending_k = ggml_nelements(tensor);
        return true;
    }
    if (tensor->type != GGML_TYPE_F32) throw std::runtime_error(std::string(name) + " is not F32");
    const size_t count = ggml_nelements(tensor);
    if (hidden) {
        if (count != c.pending_hidden || tensor->ne[0] != 5120) throw std::runtime_error("invalid attn_norm-32 shape");
        const size_t old = c.hidden.size(); c.hidden.resize(old + count);
        ggml_backend_tensor_get(tensor, c.hidden.data() + old, 0, count * sizeof(float));
        c.prompt_vectors.push_back(count / 5120); ++c.hidden_callbacks; c.pending_hidden = 0; c.awaiting_k = true;
    } else {
        if (count != c.pending_k || count % 1024 != 0) throw std::runtime_error("invalid Kcur-32 shape");
        const size_t old = c.k_output.size(); c.k_output.resize(old + count);
        ggml_backend_tensor_get(tensor, c.k_output.data() + old, 0, count * sizeof(float));
        ++c.k_callbacks; c.pending_k = 0; c.awaiting_k = false;
    }
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, true, true);
    if (n >= 0) throw std::runtime_error("token count query unexpectedly succeeded");
    std::vector<llama_token> tokens((size_t) -n);
    n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    if (n < 0) throw std::runtime_error("tokenization failed"); tokens.resize((size_t) n); return tokens;
}

static llama_token greedy(const llama_vocab * vocab, llama_context * ctx) {
    float * logits = llama_get_logits_ith(ctx, -1); if (!logits) return 0;
    llama_token best = 0; for (int i = 1; i < llama_vocab_n_tokens(vocab); ++i) if (logits[i] > logits[best]) best = i; return best;
}

static void write_f32(const char * path, const std::vector<float> & values) {
    std::ofstream out(path, std::ios::binary); if (!out) throw std::runtime_error("output open failed");
    out.write(reinterpret_cast<const char *>(values.data()), (std::streamsize) (values.size() * sizeof(float)));
    if (!out) throw std::runtime_error("output write failed");
}

int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    llama_log_set(quiet_log, nullptr); llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.check_tensors = false;
    llama_model * model = llama_model_load_vbuf_direct(argv[1], mp); if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::vector<std::string> prompts = {
        "Explain why deterministic validation matters in binary file formats. Compare structural validation, bounds checking, and semantic validation, then give a concrete example involving tensor storage and alignment.",
        "A researcher measures model loading, first-token latency, page faults, and resident compute. Describe a careful experiment that separates storage movement from computation and avoids misleading cache conclusions.",
        "Write a concise technical discussion of low-bit neural-network weight representations, including learned codebooks, geometric levels, sparse outliers, true byte accounting, and functional matrix-vector error.",
        "Consider a transformer attention layer with query, key, and value projections. Explain the exact input to the key projection, the role of normalization, and how one could verify a captured activation vector."
    };
    Capture capture;
    try {
        llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 8; cp.n_threads_batch = 8;
        llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) throw std::runtime_error("context init failed");
        for (const auto & prompt : prompts) {
            auto tokens = tokenize(vocab, prompt); if (tokens.size() < 32) { llama_free(ctx); throw std::runtime_error("prompt has fewer than 32 tokens"); } tokens.resize(32);
            std::vector<int8_t> logits(tokens.size(), 0); logits.back() = 1; llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); batch.logits = logits.data();
            if (llama_decode(ctx, batch) != 0) { llama_free(ctx); throw std::runtime_error("baseline decode failed"); }
            ggml_backend_sched_set_eval_callback(ctx->get_sched(), capture_cb, &capture);
            if (llama_decode(ctx, batch) != 0) { llama_free(ctx); throw std::runtime_error("capture decode failed"); }
            ggml_backend_sched_set_eval_callback(ctx->get_sched(), nullptr, nullptr);
        }
        ggml_backend_sched_set_eval_callback(ctx->get_sched(), nullptr, nullptr); llama_free(ctx);
        if (capture.hidden_callbacks != prompts.size() || capture.k_callbacks != prompts.size()) throw std::runtime_error("capture callback count mismatch hidden=" + std::to_string(capture.hidden_callbacks) + " k=" + std::to_string(capture.k_callbacks));
        if (capture.hidden.size()/5120 != capture.k_output.size()/1024 || capture.hidden.size()/5120 < 64) throw std::runtime_error("insufficient or mismatched vectors");
        write_f32(argv[2], capture.hidden); write_f32(argv[3], capture.k_output);
        std::printf("{\"capture_point\":\"attn_norm-32 output\",\"target_tensor\":\"blk.32.attn_k.weight\",\"k_output_point\":\"Kcur-32 before k_norm/RoPE\",\"prompts\":%zu,\"vectors\":%zu,\"input_dimension\":5120,\"output_dimension\":1024,\"dtype\":\"F32\",\"prompt_vectors\":[", prompts.size(), capture.hidden.size()/5120);
        for (size_t i = 0; i < capture.prompt_vectors.size(); ++i) std::printf("%s%zu", i ? "," : "", capture.prompt_vectors[i]);
        std::printf("]}\n");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what()); llama_model_free_vbuf_direct(model); llama_backend_free(); return 4;
    }
    llama_model_free_vbuf_direct(model); llama_backend_free(); return 0;
}
