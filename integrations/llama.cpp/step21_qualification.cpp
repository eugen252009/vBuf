#include "llama_vbuf_loader.h"
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <fstream>

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
static uint64_t logits_hash(const float * values, int n_vocab) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < n_vocab; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
    }
    return hash;
}
struct BoundaryCapture {
    std::map<std::string, std::vector<float>> values;
    std::map<std::string, size_t> pending;
    std::map<std::string, uint64_t> logical_hashes;
    std::map<std::string, std::array<int64_t, 4>> ne;
    std::map<std::string, std::array<size_t, 4>> nb;
};
static uint64_t hash_f32(const float * values, size_t count) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0; std::memcpy(&bits, values + i, sizeof(bits));
        hash ^= bits; hash *= 1099511628211ULL;
    }
    return hash;
}
static bool boundary_capture_cb(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & capture = *static_cast<BoundaryCapture *>(userdata);
    const char * raw_name = ggml_get_name(tensor);
    const std::string name = raw_name ? raw_name : "";
    const bool wanted = name == "tok_embd" || name == "tok_embd-0" || name == "inpL" || name == "attn_norm-0" ||
        name == "q-0" || name == "q_nope-0" || name == "q_pe-0" || name == "kv_cmpr-0" || name == "Kcur-0" ||
        name == "Vcur-0" || name == "Vcur_view-0" || name == "Vcur_cont-0" || name == "Qcur-0" ||
        name == "kqv_out-0" || name == "attn_out-0" || name == "ffn_inp-0" ||
        name == "ffn_norm-0" || name == "ffn_up-0" || name == "ffn_gate-0" ||
        name == "ffn_silu-0" || name == "ffn_swiglu-0" || name == "ffn_out-0" ||
        name == "l_out-0" || name == "l_out-1" || name == "l_out-26" ||
        name == "result_norm" || name == "result_output";
    if (!wanted) {
        if (ask && (name.find("embd") != std::string::npos || name.find("l_out") != std::string::npos ||
                name.find("result") != std::string::npos || name.find("attn_norm") != std::string::npos))
            std::fprintf(stderr, "BOUNDARY_CANDIDATE name=%s elements=%lld type=%d\n", name.c_str(),
                static_cast<long long>(ggml_nelements(tensor)), static_cast<int>(tensor->type));
        return false;
    }
    if (ask) { capture.pending[name] = ggml_nelements(tensor); return true; }
    if (tensor->type != GGML_TYPE_F32) return false;
    const size_t count = ggml_nelements(tensor);
    if (capture.pending[name] != count) return false;
    auto & values = capture.values[name]; values.resize(count);
    ggml_backend_tensor_get(tensor, values.data(), 0, count * sizeof(float));
    std::array<int64_t, 4> ne{}; std::array<size_t, 4> nb{};
    for (int i = 0; i < 4; ++i) { ne[i] = tensor->ne[i]; nb[i] = tensor->nb[i]; }
    capture.ne[name] = ne; capture.nb[name] = nb;
    if (name == "kqv_out-0" || name == "Vcur_cont-0") {
        std::vector<float> logical(count);
        size_t index = 0;
        for (int64_t i3 = 0; i3 < ne[3]; ++i3) for (int64_t i2 = 0; i2 < ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < ne[1]; ++i1) for (int64_t i0 = 0; i0 < ne[0]; ++i0) {
                ggml_backend_tensor_get(tensor, &logical[index++], i0 * nb[0] + i1 * nb[1] + i2 * nb[2] + i3 * nb[3], sizeof(float));
            }
        capture.logical_hashes[name] = hash_f32(logical.data(), logical.size());
        if (name == "kqv_out-0") std::ofstream("/tmp/poc22-llama-kqv.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(logical.data()), logical.size() * sizeof(float));
        if (name == "Vcur_cont-0") std::ofstream("/tmp/poc22-llama-vcur.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(logical.data()), logical.size() * sizeof(float));
        if (name == "attn_out-0") std::ofstream("/tmp/poc22-llama-attn-out.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(logical.data()), logical.size() * sizeof(float));
        if (name == "ffn_inp-0") std::ofstream("/tmp/poc22-llama-ffn-inp.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(logical.data()), logical.size() * sizeof(float));
        if (name == "ffn_norm-0") std::ofstream("/tmp/poc22-llama-ffn-norm.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(logical.data()), logical.size() * sizeof(float));
    }
    if (name == "ffn_inp-0") std::ofstream("/tmp/poc22-llama-ffn-inp.f32", std::ios::binary).write(
        reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_norm-0") std::ofstream("/tmp/poc22-llama-ffn-norm.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_up-0") std::ofstream("/tmp/poc22-llama-ffn-up.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_gate-0") std::ofstream("/tmp/poc22-llama-ffn-gate.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_silu-0") std::ofstream("/tmp/poc22-llama-ffn-silu.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_swiglu-0") std::ofstream("/tmp/poc22-llama-ffn-swiglu.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (name == "ffn_out-0") std::ofstream("/tmp/poc22-llama-ffn-out.f32", std::ios::binary).write(
            reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    return true;
}
static bool deepseek_boundary_trace(const char * path, llama_token seed) {
    llama_model * model = load(path, false, false); if (!model) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (seed < 0 || seed >= llama_vocab_n_tokens(vocab)) return false;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 64; cp.n_batch = 1; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    BoundaryCapture capture; cp.cb_eval = boundary_capture_cb; cp.cb_eval_user_data = &capture;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return false;
    int8_t logits_flag = 1;
    llama_batch batch = llama_batch_get_one(&seed, 1); batch.logits = &logits_flag;
    const int decode = llama_decode(ctx, batch);
    std::printf("{\"decode\":%d,\"boundaries\":{", decode);
    bool first = true;
    for (const auto & entry : capture.values) {
        if (!first) std::printf(","); first = false;
        std::printf("\"%s\":{\"elements\":%zu,\"hash\":\"%016llx\"", entry.first.c_str(),
            entry.second.size(), static_cast<unsigned long long>(logits_hash(entry.second.data(), entry.second.size())));
        const auto logical = capture.logical_hashes.find(entry.first);
        if (logical != capture.logical_hashes.end())
            std::printf(",\"logical_hash\":\"%016llx\"", static_cast<unsigned long long>(logical->second));
        const auto ne = capture.ne.find(entry.first); const auto nb = capture.nb.find(entry.first);
        if (ne != capture.ne.end()) std::printf(",\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu]",
            (long long) ne->second[0], (long long) ne->second[1], (long long) ne->second[2], (long long) ne->second[3],
            nb->second[0], nb->second[1], nb->second[2], nb->second[3]);
        if (entry.first == "kqv_out-0" || entry.first == "Vcur_cont-0") {
            std::fprintf(stderr, "LLAMA_PROJECTION_INPUT elements=%zu first8=", entry.second.size());
            for (size_t i = 0; i < 8; ++i) std::fprintf(stderr, "%s%.9g", i ? "," : "", entry.second[i]);
            std::fprintf(stderr, " logical_hash=%016llx\n", static_cast<unsigned long long>(capture.logical_hashes[entry.first]));
            std::printf(",\"first8\":[");
            for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", entry.second[i]);
            std::printf("]");
        }
        if (entry.first == "ffn_norm-0") {
            std::printf(",\"first8\":[");
            for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", entry.second[i]);
            std::printf("]");
        }
        if (entry.first == "ffn_up-0" || entry.first == "ffn_gate-0" || entry.first == "ffn_silu-0" ||
            entry.first == "ffn_swiglu-0" || entry.first == "ffn_out-0") {
            std::printf(",\"first8\":[");
            for (size_t i = 0; i < 8; ++i) std::printf("%s%.9g", i ? "," : "", entry.second[i]);
            std::printf("]");
        }
        std::printf("}");
    }
    std::printf("}}\n");
    llama_free(ctx); llama_model_free(model); return decode == 0;
}
static bool deepseek_generation(const char * path, bool vbuf, llama_token seed, int steps) {
    llama_model * model = load(path, vbuf, false); if (!model) return false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (seed < 0 || seed >= n_vocab || steps < 1 || steps > 4) return false;
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 64; cp.n_batch = 1; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return false;
    std::vector<llama_token> generated; generated.reserve(static_cast<size_t>(steps));
    llama_token input = seed;
    std::printf("{\"model\":\"%s\",\"format\":\"%s\",\"vocab_size\":%d,\"positions\":[",
        path, vbuf ? "vbuf" : "gguf", n_vocab);
    bool ok = true;
    for (int position = 0; position < steps; ++position) {
        int8_t logits_flag = 1;
        llama_batch batch = llama_batch_get_one(&input, 1); batch.logits = &logits_flag;
        if (llama_decode(ctx, batch) != 0) { ok = false; break; }
        float * values = llama_get_logits_ith(ctx, 0);
        if (!values) { ok = false; break; }
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) if (values[i] > values[best]) best = i;
        generated.push_back(best);
        std::printf("%s{\"position\":%d,\"input_token\":%d,\"argmax_token\":%d,\"logits_hash\":\"%016llx\"}",
            position ? "," : "", position, input, best,
            static_cast<unsigned long long>(logits_hash(values, n_vocab)));
        input = best;
    }
    std::printf("],\"generated\":["); print_tokens(generated); std::printf("]}\n");
    llama_free(ctx); llama_model_free(model); return ok && static_cast<int>(generated.size()) == steps;
}
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
    else if (!std::strcmp(mode, "deepseek-generation")) { const int seed = argc > 4 ? std::atoi(argv[4]) : 0; const int steps = argc > 5 ? std::atoi(argv[5]) : 4; if (!deepseek_generation(gguf, false, seed, steps)) return 1; }
    else if (!std::strcmp(mode, "deepseek-generation-vbuf")) { const int seed = argc > 4 ? std::atoi(argv[4]) : 0; const int steps = argc > 5 ? std::atoi(argv[5]) : 4; if (!deepseek_generation(vbuf, true, seed, steps)) return 1; }
    else if (!std::strcmp(mode, "deepseek-boundary-trace")) { const int seed = argc > 4 ? std::atoi(argv[4]) : 0; if (!deepseek_boundary_trace(gguf, seed)) return 1; }
    else if (!std::strcmp(mode, "logits")) { std::vector<float> a, b; if (!logits(gguf, false, a) || !logits(vbuf, true, b) || a.size() != b.size()) return 1; float max_diff = 0, mean = 0; size_t max_i = 0; for (size_t i = 0; i < a.size(); ++i) { float d = std::fabs(a[i] - b[i]); mean += d; if (d > max_diff) { max_diff = d; max_i = i; } } mean /= a.size(); auto top = [](const std::vector<float> & x) { return (int) std::distance(x.begin(), std::max_element(x.begin(), x.end())); }; std::printf("vocab=%zu max_abs_diff=%.9g mean_abs_diff=%.9g max_index=%zu top1_gguf=%d top1_vbuf=%d\n", a.size(), max_diff, mean, max_i, top(a), top(b)); }
    else return 2; llama_backend_free(); return 0;
}
