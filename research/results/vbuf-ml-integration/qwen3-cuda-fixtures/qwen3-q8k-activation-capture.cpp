#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("open failed: ") + path);
    in.seekg(0, std::ios::end);
    const auto n = in.tellg();
    in.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(data.data()), n);
    if (!in) throw std::runtime_error(std::string("read failed: ") + path);
    return data;
}

struct LocalBlockQ8K {
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(LocalBlockQ8K) == 292);

static void write_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error("write failed: " + path);
}

int main(int argc, char ** argv) try {
    if (argc != 3) throw std::runtime_error("usage: helper INPUT_F32 OUTPUT_PREFIX");
    constexpr int64_t width = 5120;
    constexpr int64_t positions = 25;
    const auto input = read_bytes(argv[1]);
    if (input.size() != static_cast<size_t>(width * positions * sizeof(float)))
        throw std::runtime_error("input geometry mismatch");
    const auto q4 = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q4 || q4->vec_dot_type != GGML_TYPE_Q8_K)
        throw std::runtime_error("unexpected CPU Q4_K activation trait");
    const auto q8_cpu = ggml_get_type_traits_cpu(q4->vec_dot_type);
    if (!q8_cpu || !q8_cpu->from_float)
        throw std::runtime_error("CPU Q8_K activation quantizer unavailable");
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_K, width);
    std::vector<uint8_t> packed(row_bytes * positions);
    std::vector<float> effective(width * positions);
    for (int64_t p = 0; p < positions; ++p) {
        const auto * x = reinterpret_cast<const float *>(input.data()) + p * width;
        void * dst = packed.data() + p * row_bytes;
        q8_cpu->from_float(x, dst, width);
        const auto * blocks = static_cast<const LocalBlockQ8K *>(dst);
        for (int64_t b = 0; b < width / 256; ++b) {
            for (int i = 0; i < 256; ++i)
                effective[p * width + b * 256 + i] = blocks[b].d * blocks[b].qs[i];
        }
    }
    const std::string prefix(argv[2]);
    write_bytes(prefix + ".q8_k.bin", packed.data(), packed.size());
    write_bytes(prefix + ".effective.f32", effective.data(), effective.size() * sizeof(float));
    std::printf("q4_k_cpu_vec_dot_type=%s row_bytes=%zu positions=%lld input_width=%lld output_bytes=%zu\n",
        ggml_type_name(q4->vec_dot_type), row_bytes, (long long)positions,
        (long long)width, packed.size());
    return 0;
} catch (const std::exception & e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
}
