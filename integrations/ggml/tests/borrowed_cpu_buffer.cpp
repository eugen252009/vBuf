#include <array>
#include <cstdio>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

int main() {
    alignas(64) std::array<float, 1> source = { 3.0f };
    ggml_backend_t backend = ggml_backend_init_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        return 1;
    }

    struct ggml_init_params params = { 1 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * input = ctx != nullptr
        ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1) : nullptr;
    struct ggml_tensor * output = input != nullptr
        ? ggml_scale(ctx, input, 2.0f) : nullptr;
    struct ggml_cgraph * graph = output != nullptr
        ? ggml_new_graph(ctx) : nullptr;
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t borrowed = ggml_backend_cpu_buffer_from_ptr(
        source.data(), sizeof(source));
    if (ctx == nullptr || input == nullptr || output == nullptr ||
        graph == nullptr || borrowed == nullptr) {
        if (borrowed != nullptr) {
            ggml_backend_buffer_free(borrowed);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        ggml_backend_free(backend);
        return 1;
    }

    // This is the capability under test: bind the graph input to the external
    // source range, while the output remains backend-allocated.
    if (ggml_backend_tensor_alloc(borrowed, input, source.data()) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(borrowed);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_buffer_t compute = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (compute == nullptr) {
        ggml_backend_buffer_free(borrowed);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    float result = 0.0f;
    ggml_backend_tensor_get(output, &result, 0, sizeof(result));
    const bool source_unchanged = source[0] == 3.0f;
    const bool pass = status == GGML_STATUS_SUCCESS && result == 6.0f && source_unchanged;
    std::printf("borrowed_source=%g result=%g\n", source[0], result);

    ggml_backend_buffer_free(compute);
    ggml_backend_buffer_free(borrowed);
    const bool lifetime_preserved = source[0] == 3.0f;
    ggml_free(ctx);
    ggml_backend_free(backend);
    return pass && lifetime_preserved ? 0 : 1;
}
