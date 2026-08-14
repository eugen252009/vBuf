#include "llama_vbuf_loader.h"
#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}
using Clock = std::chrono::steady_clock;

struct Segment {
    std::string marker;
    int layer_id = -1;
    double start_ms = 0;
    double complete_ms = 0;
    double elapsed_ms = 0;
    uint64_t order = 0;
};

struct Trace {
    Clock::time_point process_start;
    Clock::time_point segment_start;
    std::string pending;
    uint64_t order = 0;
    std::vector<Segment> segments;
};

static bool marker_for(const char * name) {
    return std::strcmp(name, "embd") == 0 || std::strncmp(name, "l_out-", 6) == 0 ||
           std::strcmp(name, "result_norm") == 0 || std::strcmp(name, "result_output") == 0;
}

static bool eval_callback(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & trace = *static_cast<Trace *>(userdata);
    const char * name = ggml_get_name(tensor);
    if (!ask) {
        if (trace.pending != name) return false;
        const auto now = Clock::now();
        int layer = -1;
        if (std::strncmp(name, "l_out-", 6) == 0) std::sscanf(name + 6, "%d", &layer);
        trace.segments.push_back({name, layer,
            std::chrono::duration<double, std::milli>(trace.segment_start - trace.process_start).count(),
            std::chrono::duration<double, std::milli>(now - trace.process_start).count(),
            std::chrono::duration<double, std::milli>(now - trace.segment_start).count(), trace.order++});
        trace.pending.clear();
        return true;
    }
    if (!marker_for(name)) return false;
    trace.pending = name;
    trace.segment_start = Clock::now();
    return true;
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    std::vector<llama_token> result(512);
    int32_t count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), result.data(), (int32_t) result.size(), false, false);
    if (count < 0) { result.resize((size_t) -count); count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), result.data(), (int32_t) result.size(), false, false); }
    if (count < 0) return {};
    result.resize((size_t) count); return result;
}

static llama_token greedy(const llama_vocab * vocab, llama_context * context, int32_t index) {
    float * values = llama_get_logits_ith(context, index);
    if (!values) return 0;
    llama_token best = 0; const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 1; i < n_vocab; ++i) if (values[i] > values[best]) best = i;
    return best;
}

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[1], "vbuf") != 0) return 2;
    llama_log_set(quiet_log, nullptr); llama_backend_init();
    const auto process_start = Clock::now();
    llama_model_params params = llama_model_default_params(); params.n_gpu_layers = 0; params.check_tensors = false;
    llama_model * model = llama_model_load_vbuf_direct(argv[2], params);
    if (!model) { llama_backend_free(); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto prompt = tokenize(vocab, "Hello world");
    if (prompt.empty()) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 4; }
    llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * context = llama_init_from_model(model, cp);
    if (!context) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 5; }
    std::vector<int8_t> logits(prompt.size(), 1); llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size()); batch.logits = logits.data();
    if (llama_decode(context, batch) != 0) { llama_free(context); llama_model_free_vbuf_direct(model); llama_backend_free(); return 6; }
    std::vector<llama_token> generated; generated.reserve(8); std::vector<llama_token> one(1);
    for (int step = 0; step < 8; ++step) {
        llama_token best = greedy(vocab, context, 0); generated.push_back(best); one[0] = best; int8_t flag = 1; llama_batch next = llama_batch_get_one(one.data(), 1); next.logits = &flag;
        if (llama_decode(context, next) != 0) break;
    }
    std::vector<int8_t> second_logits(prompt.size(), 1); llama_batch second = llama_batch_get_one(prompt.data(), (int32_t) prompt.size()); second.logits = second_logits.data();
    const auto baseline_start = Clock::now();
    if (llama_decode(context, second) != 0) { llama_free(context); llama_model_free_vbuf_direct(model); llama_backend_free(); return 7; }
    const double resident_baseline_ms = std::chrono::duration<double, std::milli>(Clock::now() - baseline_start).count();
    Trace trace{process_start}; ggml_backend_sched_set_eval_callback(context->get_sched(), eval_callback, &trace);
    const auto traced_start = Clock::now();
    if (llama_decode(context, second) != 0) { llama_free(context); llama_model_free_vbuf_direct(model); llama_backend_free(); return 8; }
    const double traced_total_ms = std::chrono::duration<double, std::milli>(Clock::now() - traced_start).count();
    ggml_backend_sched_set_eval_callback(context->get_sched(), nullptr, nullptr);

    std::printf("{\"format\":\"vBuf\",\"artifact\":\"%s\",\"resident_second_eval_ms\":%.6f,\"traced_eval_ms\":%.6f,\"generated_tokens\":[", argv[2], resident_baseline_ms, traced_total_ms);
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("],\"segments\":[");
    for (size_t i = 0; i < trace.segments.size(); ++i) { const auto & s = trace.segments[i]; std::printf("%s{\"marker\":\"%s\",\"layer_id\":%d,\"start_ms\":%.6f,\"complete_ms\":%.6f,\"elapsed_ms\":%.6f,\"order\":%llu}", i ? "," : "", s.marker.c_str(), s.layer_id, s.start_ms, s.complete_ms, s.elapsed_ms, (unsigned long long) s.order); }
    std::printf("]}\n");
    llama_free(context); llama_model_free_vbuf_direct(model); llama_backend_free();
    return generated.size() == 8 ? 0 : 9;
}
