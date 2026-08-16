#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

using dot_fn = void (*)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
using repack_fn = void (*)(int, float *, size_t, const void *, const void *, int, int);

static void * resolve(const char * name) {
    void * symbol = dlsym(RTLD_NEXT, name);
    if (!symbol) {
        std::fprintf(stderr, "AB_PRELOAD_RESOLVE_FAILED %s: %s\n", name, dlerror());
        std::abort();
    }
    return symbol;
}

extern "C" void ggml_vec_dot_iq4_nl_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static dot_fn next = reinterpret_cast<dot_fn>(resolve("ggml_vec_dot_iq4_nl_q8_0"));
    static bool logged = false;
    if (!logged) { std::fprintf(stderr, "AB_KERNEL ordinary_iq4_nl_q8_0\n"); logged = true; }
    next(n, s, bs, vx, bx, vy, by, nrc);
}

extern "C" void ggml_vec_dot_q2_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    static dot_fn next = reinterpret_cast<dot_fn>(resolve("ggml_vec_dot_q2_K_q8_K"));
    static bool logged = false;
    if (!logged) { std::fprintf(stderr, "AB_KERNEL ordinary_q2_K_q8_K\n"); logged = true; }
    next(n, s, bs, vx, bx, vy, by, nrc);
}

extern "C" void ggml_gemv_iq4_nl_16x1_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    static repack_fn next = reinterpret_cast<repack_fn>(resolve("ggml_gemv_iq4_nl_16x1_q8_0"));
    static bool logged = false;
    if (!logged) { std::fprintf(stderr, "AB_KERNEL iq4_nl_16x1\n"); logged = true; }
    next(n, s, bs, vx, vy, nr, nc);
}

extern "C" void ggml_gemv_q2_K_16x1_q8_K(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    static repack_fn next = reinterpret_cast<repack_fn>(resolve("ggml_gemv_q2_K_16x1_q8_K"));
    static bool logged = false;
    if (!logged) { std::fprintf(stderr, "AB_KERNEL q2_K_16x1\n"); logged = true; }
    next(n, s, bs, vx, vy, nr, nc);
}
