#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

constexpr int64_t kExperts = 64;
constexpr int64_t kSelected = 6;
constexpr std::array<int32_t, kSelected> kExpertIds = { 3, 11, 17, 29, 41, 53 };

struct Case {
    const char * name;
    ggml_type weight_type;
    int64_t input_width;
    int64_t output_width;
    size_t source_offset;
    size_t payload_bytes;
};

struct MappedFile {
    int fd = -1;
    size_t size = 0;
    void * address = MAP_FAILED;

    bool open(const char * path) {
        fd = ::open(path, O_RDONLY);
        if (fd < 0) {
            return false;
        }

        struct stat info {};
        if (fstat(fd, &info) != 0 || info.st_size <= 0) {
            return false;
        }
        size = static_cast<size_t>(info.st_size);
        address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        return address != MAP_FAILED;
    }

    ~MappedFile() {
        if (address != MAP_FAILED) {
            munmap(address, size);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    const void * pointer(size_t offset, size_t bytes) const {
        if (offset > size || bytes > size - offset) {
            return nullptr;
        }
        return static_cast<const uint8_t *>(address) + offset;
    }
};

struct GraphRun {
    ggml_cgraph * graph = nullptr;
    std::vector<ggml_tensor *> outputs;
    ggml_tensor * grouped = nullptr;
};

bool check_status(ggml_status status, const char * operation) {
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed with status %d\n", operation, static_cast<int>(status));
        return false;
    }
    return true;
}

GraphRun build_graph(ggml_context * ctx, ggml_tensor * experts, ggml_tensor * input,
    ggml_tensor * ids, bool grouped) {
    GraphRun run;
    run.graph = ggml_new_graph(ctx);
    if (grouped) {
        run.grouped = ggml_mul_mat_id(ctx, experts, input, ids);
        if (run.grouped != nullptr) {
            ggml_build_forward_expand(run.graph, run.grouped);
        }
    } else {
        run.outputs.reserve(kSelected);
        for (int64_t rank = 0; rank < kSelected; ++rank) {
            const size_t offset = static_cast<size_t>(kExpertIds[rank]) * experts->nb[2];
            ggml_tensor * expert = ggml_view_2d(
                ctx, experts, experts->ne[0], experts->ne[1], experts->nb[1], offset);
            ggml_tensor * output = ggml_mul_mat(ctx, expert, input);
            run.outputs.push_back(output);
            ggml_build_forward_expand(run.graph, output);
        }
    }
    return run;
}

double benchmark(ggml_backend_t backend, ggml_cgraph * graph, int iterations) {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (!check_status(ggml_backend_graph_compute(backend, graph), "graph compute")) {
            return -1.0;
        }
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(clock::now() - start).count();
    return elapsed / iterations;
}

bool run_case(const Case & test_case, const MappedFile & file, ggml_backend_t backend,
    int iterations) {
    constexpr size_t context_size = 32 * 1024 * 1024;
    ggml_init_params params { context_size, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "%s: ggml context initialization failed\n", test_case.name);
        return false;
    }

    ggml_tensor * experts = ggml_new_tensor_3d(
        ctx, test_case.weight_type, test_case.input_width, test_case.output_width, kExperts);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, test_case.input_width, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kSelected, 1);
    GraphRun separate = build_graph(ctx, experts, input, ids, false);
    GraphRun grouped = build_graph(ctx, experts, input, ids, true);
    const void * source = file.pointer(test_case.source_offset, test_case.payload_bytes);
    ggml_backend_buffer_t weights = source == nullptr
        ? nullptr : ggml_backend_cpu_buffer_from_ptr(const_cast<void *>(source), test_case.payload_bytes);

    bool valid = experts != nullptr && input != nullptr && ids != nullptr &&
        separate.graph != nullptr && grouped.graph != nullptr && grouped.grouped != nullptr &&
        separate.outputs.size() == kSelected && weights != nullptr;
    if (valid) {
        valid = ggml_backend_tensor_alloc(weights, experts, const_cast<void *>(source)) == GGML_STATUS_SUCCESS;
    }

    ggml_backend_buffer_t compute = valid ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    valid = valid && compute != nullptr;
    if (valid) {
        std::vector<float> values(test_case.input_width);
        for (int64_t index = 0; index < test_case.input_width; ++index) {
            values[index] = std::sin(static_cast<float>(index) * 0.013f) * 0.5f +
                std::cos(static_cast<float>(index) * 0.007f) * 0.25f;
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        ggml_backend_tensor_set(ids, kExpertIds.data(), 0, sizeof(kExpertIds));

        constexpr int warmup = 2;
        for (int iteration = 0; valid && iteration < warmup; ++iteration) {
            valid = check_status(ggml_backend_graph_compute(backend, separate.graph), "separate warmup") &&
                check_status(ggml_backend_graph_compute(backend, grouped.graph), "grouped warmup");
        }
    }

    double separate_ms = valid ? benchmark(backend, separate.graph, iterations) : -1.0;
    double grouped_ms = valid ? benchmark(backend, grouped.graph, iterations) : -1.0;
    valid = valid && separate_ms >= 0.0 && grouped_ms >= 0.0;

    std::vector<float> separate_values(kSelected * test_case.output_width);
    std::vector<float> grouped_values(kSelected * test_case.output_width);
    if (valid) {
        valid = check_status(ggml_backend_graph_compute(backend, separate.graph), "separate result") &&
            check_status(ggml_backend_graph_compute(backend, grouped.graph), "grouped result");
    }
    if (valid) {
        for (int64_t rank = 0; rank < kSelected; ++rank) {
            ggml_backend_tensor_get(separate.outputs[rank],
                separate_values.data() + rank * test_case.output_width, 0,
                test_case.output_width * sizeof(float));
        }
        ggml_backend_tensor_get(grouped.grouped, grouped_values.data(), 0,
            grouped_values.size() * sizeof(float));
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    bool finite = true;
    if (valid) {
        for (size_t index = 0; index < separate_values.size(); ++index) {
            const float a = separate_values[index];
            const float b = grouped_values[index];
            finite = finite && std::isfinite(a) && std::isfinite(b);
            const float absolute = std::fabs(a - b);
            max_abs = std::max(max_abs, absolute);
            max_rel = std::max(max_rel, absolute / std::max(1.0f, std::fabs(a)));
        }
    }

    const double speedup = grouped_ms > 0.0 ? separate_ms / grouped_ms : 0.0;
    std::printf(
        "CASE=%s INPUT_WIDTH=%lld OUTPUT_WIDTH=%lld EXPERTS=%lld SELECTED=%lld "
        "WEIGHT_TYPE=%s PAYLOAD_BYTES=%zu GROUPED_OUTPUT_BYTES=%zu "
        "SEPARATE_AVG_MS=%.3f GROUPED_AVG_MS=%.3f GROUPED_SPEEDUP=%.3fx "
        "MAX_ABS_DIFF=%.9g MAX_REL_DIFF=%.9g PARITY=%s\n",
        test_case.name, static_cast<long long>(test_case.input_width),
        static_cast<long long>(test_case.output_width), static_cast<long long>(kExperts),
        static_cast<long long>(kSelected), ggml_type_name(test_case.weight_type),
        test_case.payload_bytes, grouped_values.size() * sizeof(float), separate_ms, grouped_ms,
        speedup, max_abs, max_rel, valid && finite && max_abs <= 1e-4f ? "PASS" : "FAIL");

    if (compute != nullptr) {
        ggml_backend_buffer_free(compute);
    }
    if (weights != nullptr) {
        ggml_backend_buffer_free(weights);
    }
    ggml_free(ctx);
    return valid && finite && max_abs <= 1e-4f;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s <DeepSeek-V2-Lite.IQ2_XXS.gguf> [iterations]\n", argv[0]);
        return 2;
    }

    const int iterations = argc == 3 ? std::atoi(argv[2]) : 10;
    if (iterations <= 0) {
        std::fprintf(stderr, "iterations must be positive\n");
        return 2;
    }

    MappedFile file;
    if (!file.open(argv[1])) {
        std::fprintf(stderr, "failed to map source file: %s\n", argv[1]);
        return 1;
    }

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "CPU backend initialization failed\n");
        return 1;
    }
    int threads = 1;
    if (const char * value = std::getenv("VBUF_MOE_THREADS")) {
        threads = std::max(1, std::atoi(value));
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
    std::printf("SOURCE=%s ITERATIONS=%d THREADS=%d\n", argv[1], iterations, threads);

    const std::array<Case, 2> cases = {{
        { "gate_up", GGML_TYPE_IQ2_XXS, 2048, 1408, 348536352, 47579136 },
        { "down", GGML_TYPE_IQ4_NL, 1408, 2048, 244727328, 103809024 },
    }};

    bool pass = true;
    for (const Case & test_case : cases) {
        pass = run_case(test_case, file, backend, iterations) && pass;
    }
    ggml_backend_free(backend);
    return pass ? 0 : 1;
}
