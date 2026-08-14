#include "llama_vbuf_loader.h"
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstring>
static void quiet_log(enum ggml_log_level, const char *, void *) {}
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const bool vbuf = std::strcmp(argv[1], "vbuf") == 0;
    llama_log_set(quiet_log, nullptr); llama_backend_init();
    llama_model_params params = llama_model_default_params(); params.n_gpu_layers = 0; params.check_tensors = false;
    auto start = std::chrono::steady_clock::now();
    llama_model * model = vbuf ? llama_model_load_vbuf(argv[2], params) : llama_model_load_from_file(argv[2], params);
    auto end = std::chrono::steady_clock::now();
    if (!model) return 1;
    std::printf("{\"format\":\"%s\",\"phase\":\"model_ready\",\"duration_us\":%.3f}\n", argv[1], std::chrono::duration<double, std::micro>(end-start).count());
    if (vbuf) llama_model_free_vbuf(model); else llama_model_free(model); llama_backend_free(); return 0;
}
