#include "llama_vbuf_loader.h"
#include "llama.h"
#include "vbuf_direct_source.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static void quiet_log(enum ggml_log_level, const char *, void *) {}
using Clock = std::chrono::steady_clock;

struct Snapshot {
    long minor_faults = 0;
    long major_faults = 0;
    long user_us = 0;
    long sys_us = 0;
    long rss_kb = -1;
    long long read_bytes = -1;
    long long rchar = -1;
    long long syscr = -1;
};

static Snapshot snapshot() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    Snapshot result{usage.ru_minflt, usage.ru_majflt, usage.ru_utime.tv_sec * 1000000L + usage.ru_utime.tv_usec, usage.ru_stime.tv_sec * 1000000L + usage.ru_stime.tv_usec};
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") { status >> result.rss_kb; break; }
        std::string ignored;
        std::getline(status, ignored);
    }
    std::ifstream io("/proc/self/io");
    while (io >> key) {
        long long value = -1; io >> value;
        if (key == "read_bytes:") result.read_bytes = value;
        else if (key == "rchar:") result.rchar = value;
        else if (key == "syscr:") result.syscr = value;
    }
    return result;
}

static double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    std::vector<llama_token> result(512);
    int32_t count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), result.data(), (int32_t) result.size(), false, false);
    if (count < 0) {
        result.resize((size_t) -count);
        count = llama_tokenize(vocab, text, (int32_t) std::strlen(text), result.data(), (int32_t) result.size(), false, false);
    }
    if (count < 0) return {};
    result.resize((size_t) count);
    return result;
}

static void print_snapshot(const char * phase, double ms, const Snapshot & value) {
    std::printf("\"%s\":{\"ms\":%.3f,\"minor_faults\":%ld,\"major_faults\":%ld,\"user_us\":%ld,\"sys_us\":%ld,\"rss_kb\":%ld,\"read_bytes\":%lld,\"rchar\":%lld,\"syscr\":%lld}", phase, ms, value.minor_faults, value.major_faults, value.user_us, value.sys_us, value.rss_kb, value.read_bytes, value.rchar, value.syscr);
}

int main(int argc, char ** argv) {
    if ((argc != 3 && argc != 4) || (std::strcmp(argv[1], "gguf") != 0 && std::strcmp(argv[1], "vbuf") != 0)) return 2;
    const bool vbuf = std::strcmp(argv[1], "vbuf") == 0;
    const bool step28_mode = argc == 4;
    const char * preparation_variant = step28_mode ? argv[3] : "baseline";
    if (!vbuf && std::strcmp(preparation_variant, "baseline") != 0) return 2;
    llama_log_set(quiet_log, nullptr);
    llama_backend_init();
    const auto process_start = Clock::now();
    const Snapshot process_snapshot = snapshot();
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) return 3;
    const Snapshot source_open_snapshot = snapshot();
    const double source_open_ms = elapsed_ms(process_start);
    close(fd);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.check_tensors = false;
    const auto construction_begin = Clock::now();
    llama_model * model = vbuf ? llama_model_load_vbuf_direct(argv[2], model_params) : llama_model_load_from_file(argv[2], model_params);
    const double model_ready_ms = elapsed_ms(process_start);
    const double model_construction_ms = std::chrono::duration<double, std::milli>(Clock::now() - construction_begin).count();
    const Snapshot model_ready_snapshot = snapshot();
    if (!model) { llama_backend_free(); return 4; }

    vbuf_llama::Step28PreparationResult preparation{};
    const Snapshot preparation_begin_snapshot = snapshot();
    if (step28_mode && vbuf && !llama_model_step28_prepare_vbuf(model, preparation_variant, &preparation)) { llama_model_free_vbuf_direct(model); llama_backend_free(); return 4; }
    const double preparation_complete_ms = elapsed_ms(process_start);
    const Snapshot preparation_complete_snapshot = snapshot();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> prompt = tokenize(vocab, "Hello world");
    if (prompt.empty()) { vbuf ? llama_model_free_vbuf_direct(model) : llama_model_free(model); llama_backend_free(); return 5; }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 512;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) { vbuf ? llama_model_free_vbuf_direct(model) : llama_model_free(model); llama_backend_free(); return 6; }

    std::vector<int8_t> logits(prompt.size(), 1);
    llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    batch.logits = logits.data();
    const Snapshot first_eval_begin_snapshot = snapshot();
    const double first_eval_begin_ms = elapsed_ms(process_start);
    const auto eval_begin = Clock::now();
    const int eval_rc = llama_decode(context, batch);
    const double first_eval_ms = elapsed_ms(process_start);
    const double first_eval_duration_ms = std::chrono::duration<double, std::milli>(Clock::now() - eval_begin).count();
    const Snapshot first_eval_snapshot = snapshot();
    if (eval_rc != 0) { llama_free(context); vbuf ? llama_model_free_vbuf_direct(model) : llama_model_free(model); llama_backend_free(); return 7; }
    float * first_values = llama_get_logits_ith(context, (int32_t) prompt.size() - 1);
    llama_token first_token = 0;
    if (first_values != nullptr) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 1; i < n_vocab; ++i) if (first_values[i] > first_values[first_token]) first_token = i;
    }

    std::vector<llama_token> generated;
    generated.reserve(8);
    std::vector<llama_token> one(1);
    for (int step = 0; step < 8; ++step) {
        float * values = step == 0 ? first_values : llama_get_logits_ith(context, 0);
        if (values == nullptr) break;
        llama_token best = 0;
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 1; i < n_vocab; ++i) if (values[i] > values[best]) best = i;
        generated.push_back(best);
        one[0] = best;
        int8_t flag = 1;
        llama_batch next = llama_batch_get_one(one.data(), 1);
        next.logits = &flag;
        if (llama_decode(context, next) != 0) break;
    }
    const double generation_ms = elapsed_ms(process_start);
    const Snapshot generation_snapshot = snapshot();

    // Reuse the model, but use a fresh context so the second evaluation is the
    // same prompt rather than an append to the generation sequence.
    llama_context * second_context = llama_init_from_model(model, context_params);
    if (!second_context) { llama_free(context); vbuf ? llama_model_free_vbuf_direct(model) : llama_model_free(model); llama_backend_free(); return 8; }
    std::vector<int8_t> second_logits(prompt.size(), 1);
    llama_batch second_batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
    second_batch.logits = second_logits.data();
    const auto second_begin = Clock::now();
    const int second_rc = llama_decode(second_context, second_batch);
    const double second_eval_ms = std::chrono::duration<double, std::milli>(Clock::now() - second_begin).count();
    const Snapshot second_eval_snapshot = snapshot();
    llama_free(second_context);
    if (second_rc != 0) { llama_free(context); vbuf ? llama_model_free_vbuf_direct(model) : llama_model_free(model); llama_backend_free(); return 8; }
    const double first_token_ms = first_eval_ms;

    std::printf("{\"format\":\"%s\",\"path\":\"%s\",\"preparation_variant\":\"%s\",\"prompt_tokens\":%zu,\"first_token\":%d,\"generated_tokens\":[", argv[1], argv[2], preparation_variant, prompt.size(), first_token);
    for (size_t i = 0; i < generated.size(); ++i) std::printf("%s%d", i ? "," : "", generated[i]);
    std::printf("],\"model_construction_ms\":%.3f,\"preparation_ms\":%.3f,\"first_eval_duration_ms\":%.3f,\"second_eval_ms\":%.3f,\"ttfuc_ms\":%.3f,\"ttft_ms\":%.3f,\"generation_complete_ms\":%.3f,\"preparation\":{\"span_count\":%llu,\"global_span_count\":%llu,\"covered_bytes\":%llu,\"useful_bytes\":%llu,\"gap_bytes\":%llu,\"prepared_span_bytes\":%llu,\"prepared_useful_bytes\":%llu,\"page_size\":%llu,\"pages_touched\":%llu,\"touch_operations\":%llu,\"minor_faults\":%ld,\"major_faults\":%ld,\"layers\":[", model_construction_ms, preparation.prepare_ms, first_eval_duration_ms, second_eval_ms, first_eval_ms, first_token_ms, generation_ms, (unsigned long long) preparation.span_count, (unsigned long long) preparation.global_span_count, (unsigned long long) preparation.covered_bytes, (unsigned long long) preparation.useful_bytes, (unsigned long long) preparation.gap_bytes, (unsigned long long) preparation.prepared_span_bytes, (unsigned long long) preparation.prepared_useful_bytes, (unsigned long long) preparation.page_size, (unsigned long long) preparation.pages_touched, (unsigned long long) preparation.touch_operations, preparation.minor_faults, preparation.major_faults);
    for (size_t i = 0; i < preparation.layers.size(); ++i) { const auto & layer = preparation.layers[i]; std::printf("%s{\"layer_id\":%llu,\"start_offset\":%llu,\"end_offset\":%llu,\"span_bytes\":%llu,\"useful_bytes\":%llu,\"gap_bytes\":%llu,\"tensor_count\":%llu,\"prepare_ms\":%.6f,\"minor_faults\":%ld,\"major_faults\":%ld}", i ? "," : "", (unsigned long long) layer.layer_id, (unsigned long long) layer.start_offset, (unsigned long long) layer.end_offset, (unsigned long long) layer.span_bytes, (unsigned long long) layer.useful_bytes, (unsigned long long) layer.gap_bytes, (unsigned long long) layer.tensor_count, layer.prepare_ms, layer.minor_faults, layer.major_faults); }
    std::printf("],\"spans\":[");
    for (size_t i = 0; i < preparation.spans.size(); ++i) { const auto & span = preparation.spans[i]; std::printf("%s{\"role\":\"%s\",\"layer_id\":%llu,\"start_offset\":%llu,\"end_offset\":%llu,\"span_bytes\":%llu,\"useful_bytes\":%llu,\"gap_bytes\":%llu,\"tensor_count\":%llu}", i ? "," : "", span.role.c_str(), (unsigned long long) span.layer_id, (unsigned long long) span.start_offset, (unsigned long long) span.end_offset, (unsigned long long) span.span_bytes, (unsigned long long) span.useful_bytes, (unsigned long long) span.gap_bytes, (unsigned long long) span.tensor_count); }
    std::printf("]},\"phases\":{");
    print_snapshot("PROCESS_START", 0.0, process_snapshot); std::printf(",");
    print_snapshot("SOURCE_OPEN", source_open_ms, source_open_snapshot); std::printf(",");
    print_snapshot("MODEL_READY", model_ready_ms, model_ready_snapshot); std::printf(",");
    print_snapshot("PREPARATION_BEGIN", model_ready_ms, preparation_begin_snapshot); std::printf(",");
    print_snapshot("PREPARATION_COMPLETE", preparation_complete_ms, preparation_complete_snapshot); std::printf(",");
    print_snapshot("FIRST_EVAL_BEGIN", first_eval_begin_ms, first_eval_begin_snapshot); std::printf(",");
    print_snapshot("FIRST_EVAL_COMPLETE", first_eval_ms, first_eval_snapshot); std::printf(",");
    print_snapshot("FIRST_TOKEN", first_token_ms, first_eval_snapshot); std::printf(",");
    print_snapshot("GENERATION_COMPLETE", generation_ms, generation_snapshot); std::printf(",");
    print_snapshot("SECOND_EVAL_COMPLETE", generation_ms + second_eval_ms, second_eval_snapshot);
    std::printf("}}\n");

    llama_free(context);
    if (vbuf) llama_model_free_vbuf_direct(model); else llama_model_free(model);
    llama_backend_free();
    return generated.size() == 8 ? 0 : 9;
}
