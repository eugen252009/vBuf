#include "gguf.h"
#include "ggml-cpp.h"
#include <chrono>
#include <cstdio>
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    auto start = std::chrono::steady_clock::now();
    ggml_context * ctx = nullptr;
    gguf_init_params params{true, &ctx};
    gguf_context_ptr meta(gguf_init_from_file(argv[1], params));
    if (!meta) return 1;
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end-start).count();
    std::printf("{\"phase\":\"gguf_format_control\",\"duration_us\":%.3f,\"kv\":%d,\"tensors\":%lld}\n", us, gguf_get_n_kv(meta.get()), (long long) gguf_get_n_tensors(meta.get()));
    return 0;
}
