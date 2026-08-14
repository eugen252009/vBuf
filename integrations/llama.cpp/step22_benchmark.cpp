#include "llama_vbuf_loader.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using clock_type = std::chrono::steady_clock;
struct Counters { rusage usage{}; long rss_kb = 0; long vm_kb = 0; long pss_kb = -1; long long read_bytes = -1; };
static long read_kb(const char * key) { std::ifstream in("/proc/self/status"); std::string line; while (std::getline(in, line)) if (line.rfind(key, 0) == 0) { std::istringstream values(line.substr(std::strlen(key))); long value = -1; values >> value; return value; } return -1; }
static long long read_io(const char * key) { std::ifstream in("/proc/self/io"); std::string k; long long value; while (in >> k >> value) if (k == std::string(key) + ":") return value; return -1; }
static long read_pss() { std::ifstream in("/proc/self/smaps_rollup"); std::string k, value, unit; while (in >> k >> value >> unit) if (k == "Pss:") return std::stol(value); return -1; }
static Counters sample() { Counters c; getrusage(RUSAGE_SELF, &c.usage); c.rss_kb = read_kb("VmRSS:"); c.vm_kb = read_kb("VmSize:"); c.pss_kb = read_pss(); c.read_bytes = read_io("read_bytes"); return c; }
static double us(clock_type::time_point a, clock_type::time_point b) { return std::chrono::duration<double, std::micro>(b-a).count(); }
static void emit(const char * format, const char * phase, double duration, const Counters & before, const Counters & after, long file_bytes, int prompt_tokens = 0, int generated = 0) {
    const double prompt_tps = phase && !std::strcmp(phase, "prompt_eval") && duration > 0 ? prompt_tokens / (duration / 1e6) : 0;
    const double generation_tps = phase && !std::strcmp(phase, "generation") && duration > 0 ? generated / (duration / 1e6) : 0;
    std::printf("%s,%s,%.3f,%ld,%ld,%ld,%ld,%ld,%lld,%lld,%ld,%d,%.6f,%.6f\n", format, phase, duration, after.usage.ru_minflt-before.usage.ru_minflt, after.usage.ru_majflt-before.usage.ru_majflt, after.rss_kb, after.vm_kb, after.pss_kb, after.read_bytes, file_bytes, prompt_tokens, generated, prompt_tps, generation_tps);
}
static void quiet_log(enum ggml_log_level, const char *, void *) {}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) { std::vector<llama_token> tokens(2048); int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, false); if (n < 0) { tokens.resize(-n); n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, false); } if (n < 0) return {}; tokens.resize(n); return tokens; }
static llama_model * load_model(const std::string & path, bool vbuf, bool vocab_only) { llama_model_params p = llama_model_default_params(); p.n_gpu_layers = 0; p.vocab_only = vocab_only; p.check_tensors = false; return vbuf ? llama_model_load_vbuf(path.c_str(), p) : llama_model_load_from_file(path.c_str(), p); }
static llama_token greedy(const llama_vocab * vocab, float * logits) { llama_token best = 0; for (int i = 1; i < llama_vocab_n_tokens(vocab); ++i) if (logits[i] > logits[best]) best = i; return best; }
int main(int argc, char ** argv) {
    if (argc < 5) return 2;
    const char * format = argv[1]; const std::string path = argv[2]; const std::string prompt = argv[3]; const int threads = std::atoi(argv[4]); const bool vocab_only = argc >= 6 && !std::strcmp(argv[5], "vocab_only"); const bool vbuf = !std::strcmp(format, "vbuf");
    struct stat st{}; if (stat(path.c_str(), &st) != 0) return 3;
    std::printf("format,phase,duration_us,minor_faults,major_faults,rss_kb,vmsize_kb,pss_kb,read_bytes,file_bytes,prompt_tokens,generated_tokens,prompt_tokens_sec,generation_tokens_sec\n");
    llama_log_set(quiet_log, nullptr); llama_backend_init(); Counters process_start = sample(); auto open_start = clock_type::now(); llama_model * model = load_model(path, vbuf, vocab_only); auto open_end = clock_type::now(); Counters ready = sample(); if (!model) return 4;
    emit(format, vocab_only ? "metadata_control" : "model_ready", us(open_start, open_end), process_start, ready, st.st_size);
    if (vocab_only) { if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); llama_backend_free(); return 0; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tok_start = clock_type::now(); auto tokens = tokenize(vocab, prompt); auto tok_end = clock_type::now(); Counters tok_after = sample(); emit(format, "prompt_tokenization", us(tok_start, tok_end), ready, tok_after, st.st_size, (int) tokens.size());
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = threads; cp.n_threads_batch = threads; llama_context * ctx = llama_init_from_model(model, cp); if (!ctx || tokens.empty()) return 5;
    std::vector<int8_t> flags(tokens.size(), 1); llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size()); batch.logits = flags.data(); Counters prompt_before = sample(); auto prompt_start = clock_type::now(); if (llama_decode(ctx, batch) != 0) return 6; auto prompt_end = clock_type::now(); Counters prompt_after = sample(); emit(format, "prompt_eval", us(prompt_start, prompt_end), prompt_before, prompt_after, st.st_size, (int) tokens.size());
    const int generated_count = 8; std::vector<llama_token> generated; generated.reserve(generated_count);
    Counters first_before = sample(); auto first_start = clock_type::now();
    llama_token first = greedy(vocab, llama_get_logits_ith(ctx, (int32_t) tokens.size()-1)); generated.push_back(first); tokens.assign(1, first); int8_t first_output = 1; batch = llama_batch_get_one(tokens.data(), 1); batch.logits = &first_output; if (llama_decode(ctx, batch) != 0) return 7;
    auto first_end = clock_type::now(); Counters first_after = sample(); emit(format, "first_token", us(first_start, first_end), first_before, first_after, st.st_size, 0, 1);
    Counters generation_before = sample(); auto generation_start = clock_type::now();
    for (int i = 1; i < generated_count; ++i) { llama_token next = greedy(vocab, llama_get_logits_ith(ctx, 0)); generated.push_back(next); tokens.assign(1, next); int8_t output = 1; batch = llama_batch_get_one(tokens.data(), 1); batch.logits = &output; if (llama_decode(ctx, batch) != 0) return 7; }
    auto generation_end = clock_type::now(); Counters gen_after = sample(); emit(format, "generation", us(generation_start, generation_end), generation_before, gen_after, st.st_size, 0, generated_count - 1);
    std::fprintf(stderr, "generated:"); for (llama_token token : generated) std::fprintf(stderr, "%s%d", token == generated.front() ? "" : ",", token); std::fprintf(stderr, "\n");
    llama_free(ctx); if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); llama_backend_free(); return 0;
}
