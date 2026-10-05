#include "ggml.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint8_t> read_bytes(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("open failed: ") + path);
    in.seekg(0, std::ios::end); const auto n = in.tellg(); in.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(n));
    in.read(reinterpret_cast<char *>(data.data()), n);
    if (!in) throw std::runtime_error("read failed");
    return data;
}
int main(int argc, char ** argv) try {
    if (argc != 3) throw std::runtime_error("usage: helper Q4K_BYTES OUTPUT_F32");
    constexpr int64_t n = 5120LL * 5120;
    const auto packed = read_bytes(argv[1]);
    const auto traits = ggml_get_type_traits(GGML_TYPE_Q4_K);
    if (!traits || !traits->to_float || packed.size() != ggml_row_size(GGML_TYPE_Q4_K, 5120) * 5120)
        throw std::runtime_error("Q4_K decoder/geometry unavailable");
    std::vector<float> decoded(n);
    traits->to_float(packed.data(), decoded.data(), n);
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char *>(decoded.data()), decoded.size() * sizeof(float));
    if (!out) throw std::runtime_error("write failed");
    std::printf("ggml_q4_k_decode_elements=%lld output_bytes=%zu\n", (long long)n, decoded.size() * sizeof(float));
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
