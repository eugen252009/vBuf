#include "llama_vbuf_loader.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

enum class Path { GGUF, Compatibility, Direct };
static llama_model * load(const char * path, Path kind, bool vocab_only = false) {
    llama_model_params p = llama_model_default_params(); p.n_gpu_layers = 0; p.vocab_only = vocab_only; p.check_tensors = false;
    if (kind == Path::Direct) return llama_model_load_vbuf_direct(path, p);
    return kind == Path::Compatibility ? llama_model_load_vbuf(path, p) : llama_model_load_from_file(path, p);
}
static void free_model(llama_model * model, Path kind) { if (kind == Path::Direct) llama_model_free_vbuf_direct(model); else if (kind == Path::Compatibility) llama_model_free_vbuf(model); else llama_model_free(model); }
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> result(512); int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), false, false);
    if (n < 0) { result.resize(-n); n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), false, false); }
    if (n < 0) return {}; result.resize(n); return result;
}
static const char * label(Path p) { return p == Path::GGUF ? "gguf" : p == Path::Compatibility ? "compatibility" : "direct"; }
static bool logits(const char * path, Path kind, std::vector<float> & out) {
    llama_model * model = load(path, kind); if (!model) return false; const llama_vocab * vocab = llama_model_get_vocab(model); auto tokens = tokenize(vocab, "Hello world");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 2; cp.n_threads_batch = 2; llama_context * ctx = llama_init_from_model(model, cp); if (!ctx || tokens.empty()) return false;
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); std::vector<int8_t> flags(tokens.size(), 1); batch.logits = flags.data(); bool ok = llama_decode(ctx, batch) == 0;
    if (ok) { float * values = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1); out.assign(values, values + llama_vocab_n_tokens(vocab)); }
    llama_free(ctx); free_model(model, kind); return ok;
}
static bool generation(const char * path, Path kind, std::vector<llama_token> & out) {
    llama_model * model = load(path, kind); if (!model) return false; const llama_vocab * vocab = llama_model_get_vocab(model); auto tokens = tokenize(vocab, "Hello world");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 2; cp.n_threads_batch = 2; llama_context * ctx = llama_init_from_model(model, cp); if (!ctx || tokens.empty()) return false;
    std::vector<int8_t> flags(tokens.size(), 1); llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); batch.logits = flags.data(); if (llama_decode(ctx, batch) != 0) return false;
    for (int step = 0; step < 8; ++step) { float * values = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1); llama_token best = 0; for (int i = 1; i < llama_vocab_n_tokens(vocab); ++i) if (values[i] > values[best]) best = i; out.push_back(best); tokens.assign(1, best); int8_t flag = 1; batch = llama_batch_get_one(tokens.data(), 1); batch.logits = &flag; if (llama_decode(ctx, batch) != 0) return false; }
    llama_free(ctx); free_model(model, kind); return true;
}
static void print_tokens(const std::vector<llama_token> & values) { for (size_t i = 0; i < values.size(); ++i) std::printf("%s%d", i ? "," : "", values[i]); std::printf("\n"); }
int main(int argc, char ** argv) {
    if (argc != 4) return 2; llama_backend_init(); const char * gguf = argv[1], * vbuf = argv[2], * mode = argv[3]; const Path paths[] = {Path::GGUF, Path::Compatibility, Path::Direct}; const char * files[] = {gguf, vbuf, vbuf};
    if (!std::strcmp(mode, "metadata")) {
        const char * keys[] = {"general.architecture", "qwen3.context_length", "qwen3.embedding_length", "qwen3.block_count", "qwen3.attention.head_count", "qwen3.attention.head_count_kv", "qwen3.attention.key_length", "qwen3.attention.value_length", "qwen3.feed_forward_length", "qwen3.attention.layer_norm_rms_epsilon", "qwen3.rope.freq_base", "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.chat_template"};
        for (int p = 0; p < 3; ++p) { llama_model * model = load(files[p], paths[p], true); if (!model) return 1; std::printf("%s|", label(paths[p])); for (const char * key : keys) { char value[1024 * 1024]{}; if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) return 1; std::printf("%s=%s;", key, value); } std::printf("\n"); free_model(model, paths[p]); }
    } else if (!std::strcmp(mode, "tokens")) {
        const std::vector<std::string> corpus = {"", "Hello world", "  whitespace\n", "你好，世界", "<|im_start|>user\nHi<|im_end|>"};
        for (int p = 0; p < 3; ++p) { llama_model * model = load(files[p], paths[p], true); if (!model) return 1; const llama_vocab * vocab = llama_model_get_vocab(model); for (size_t i = 0; i < corpus.size(); ++i) { std::printf("%s|%zu|", label(paths[p]), i); print_tokens(tokenize(vocab, corpus[i])); } free_model(model, paths[p]); }
    } else if (!std::strcmp(mode, "logits")) {
        std::vector<std::vector<float>> all(3); for (int p = 0; p < 3; ++p) if (!logits(files[p], paths[p], all[p])) return 1; for (int p = 1; p < 3; ++p) { float max_diff = 0; for (size_t i = 0; i < all[0].size(); ++i) max_diff = std::max(max_diff, std::fabs(all[0][i] - all[p][i])); std::printf("gguf_vs_%s max_abs_diff=%.9g\n", label(paths[p]), max_diff); }
    } else if (!std::strcmp(mode, "generation")) {
        for (int p = 0; p < 3; ++p) { std::vector<llama_token> out; if (!generation(files[p], paths[p], out)) return 1; std::printf("%s=", label(paths[p])); print_tokens(out); }
    } else return 2;
    llama_backend_free(); return 0;
}
