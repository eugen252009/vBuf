#include <cstdio>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "vbuf_region_executor.h"

int main() {
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "CPU backend initialization failed\n");
        return 1;
    }

    struct ggml_init_params params = { 1 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        ggml_backend_free(backend);
        return 1;
    }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * sum = ggml_add(ctx, a, b);
    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const float values[1] = { 2.0f };
    ggml_backend_tensor_set(a, values, 0, sizeof(values));
    ggml_backend_tensor_set(b, values, 0, sizeof(values));
    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    float result = 0.0f;
    ggml_backend_tensor_get(sum, &result, 0, sizeof(result));

    const bool pass = status == GGML_STATUS_SUCCESS && result == 4.0f;
    std::printf("backend=%s result=%g\n", vbuf_region_executor_backend_name(), result);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return pass ? 0 : 1;
}
