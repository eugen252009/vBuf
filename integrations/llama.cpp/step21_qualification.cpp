#include "llama_vbuf_loader.h"
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_model * load(const char * path, bool vbuf, bool vocab_only) {
    llama_model_params p = llama_model_default_params(); p.n_gpu_layers = 0; p.vocab_only = vocab_only; p.check_tensors = false;
    return vbuf ? llama_model_load_vbuf(path, p) : llama_model_load_from_file(path, p);
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> result(512); int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), false, false);
    if (n < 0) { result.resize(-n); n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), result.data(), (int32_t) result.size(), false, false); }
    if (n < 0) return {}; result.resize(n); return result;
}
static void print_tokens(const std::vector<llama_token> & tokens) { for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i ? "," : "", tokens[i]); std::printf("\n"); }
static bool generate(const char * path, bool vbuf, std::vector<llama_token> & out) {
    llama_model * model = load(path, vbuf, false); if (!model) return false; const llama_vocab * vocab = llama_model_get_vocab(model); auto tokens = tokenize(vocab, "Hello world");
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 2; cp.n_threads_batch = 2; llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return false;
    std::vector<int8_t> flags(tokens.size(), 1); llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); batch.logits = flags.data(); if (llama_decode(ctx, batch) != 0) return false;
    for (int step = 0; step < 8; ++step) { float * logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1); int n_vocab = llama_vocab_n_tokens(vocab); llama_token best = 0; for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i; out.push_back(best); tokens.assign(1, best); int8_t output = 1; batch = llama_batch_get_one(tokens.data(), 1); batch.logits = &output; if (llama_decode(ctx, batch) != 0) return false; }
    llama_free(ctx); if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); return true;
}
static bool logits(const char * path, bool vbuf, std::vector<float> & out) {
    llama_model * model = load(path, vbuf, false); if (!model) return false; const llama_vocab * vocab = llama_model_get_vocab(model); auto tokens = tokenize(vocab, "Hello world"); if (tokens.empty()) return false;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 2; cp.n_threads_batch = 2; llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return false;
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); std::vector<int8_t> flags(tokens.size(), 1); batch.logits = flags.data(); bool ok = llama_decode(ctx, batch) == 0;
    if (ok) { int n = llama_vocab_n_tokens(vocab); float * values = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1); out.assign(values, values + n); }
    llama_free(ctx); if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); return ok;
}
static bool metadata(const char * path, bool vbuf, int pass) {
    llama_model * model = load(path, vbuf, true); if (!model) return false;
    const char * keys[] = {"general.architecture", "qwen3.context_length", "qwen3.embedding_length", "qwen3.block_count", "qwen3.attention.head_count", "qwen3.attention.head_count_kv", "qwen3.attention.key_length", "qwen3.attention.value_length", "qwen3.feed_forward_length", "qwen3.attention.layer_norm_rms_epsilon", "qwen3.rope.freq_base", "tokenizer.chat_template"};
    for (const char * key : keys) { char value[8192]{}; if (llama_model_meta_val_str(model, key, value, sizeof(value)) < 0) return false; if (!std::strcmp(key, "tokenizer.chat_template")) std::printf("%d|%s|len:%zu\n", pass, key, std::strlen(value)); else std::printf("%d|%s|%s\n", pass, key, value); }
    if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); return true;
}
int main(int argc, char ** argv) {
    if (argc < 4) return 2; llama_backend_init(); const char * gguf = argv[1], * vbuf = argv[2], * mode = argv[3];
    if (!std::strcmp(mode, "metadata")) { if (!metadata(gguf, false, 0) || !metadata(vbuf, true, 1)) return 1; }
    else if (!std::strcmp(mode, "tokens")) { for (int pass = 0; pass < 2; ++pass) { llama_model * model = load(pass ? vbuf : gguf, pass, true); if (!model) return 1; const llama_vocab * vocab = llama_model_get_vocab(model); int case_id = 0; for (const std::string & text : {"", "Hello world", "  whitespace\n", "你好，世界", "<|im_start|>user\nHi<|im_end|>"}) { auto t = tokenize(vocab, text); std::printf("%d|%d|", pass, case_id++); print_tokens(t); } if (pass) llama_model_free_vbuf(model); else llama_model_free(model); } }
    else if (!std::strcmp(mode, "generation")) { std::vector<llama_token> a, b; if (!generate(gguf, false, a) || !generate(vbuf, true, b)) return 1; std::printf("gguf="); print_tokens(a); std::printf("vbuf="); print_tokens(b); }
    else if (!std::strcmp(mode, "logits")) { std::vector<float> a, b; if (!logits(gguf, false, a) || !logits(vbuf, true, b) || a.size() != b.size()) return 1; float max_diff = 0, mean = 0; size_t max_i = 0; for (size_t i = 0; i < a.size(); ++i) { float d = std::fabs(a[i] - b[i]); mean += d; if (d > max_diff) { max_diff = d; max_i = i; } } mean /= a.size(); auto top = [](const std::vector<float> & x) { return (int) std::distance(x.begin(), std::max_element(x.begin(), x.end())); }; std::printf("vocab=%zu max_abs_diff=%.9g mean_abs_diff=%.9g max_index=%zu top1_gguf=%d top1_vbuf=%d\n", a.size(), max_diff, mean, max_i, top(a), top(b)); }
    else return 2; llama_backend_free(); return 0;
}
