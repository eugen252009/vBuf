// Research-only canonical GGML quantize/dequantize roundtrip helper.
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

struct Format {
    const char * name;
    ggml_type type;
    int values;
    size_t bytes;
    void (*quantize)(const float *, void *, int64_t);
    void (*dequantize)(const void *, float *, int64_t);
    const char * quantizer;
    const char * metadata;
};

template<typename Block>
static void quantize(const float * input, void * output, int64_t count) {
    if constexpr (std::is_same_v<Block, block_q4_0>) quantize_row_q4_0_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_q4_K>) quantize_row_q4_K_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_q3_K>) quantize_row_q3_K_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_iq3_xxs>) quantize_row_iq3_xxs_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_iq3_s>) quantize_row_iq3_s_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_iq4_nl>) quantize_row_iq4_nl_ref(input, static_cast<Block *>(output), count);
    else if constexpr (std::is_same_v<Block, block_iq4_xs>) quantize_row_iq4_xs_ref(input, static_cast<Block *>(output), count);
}

template<typename Block>
static void dequantize(const void * input, float * output, int64_t count) {
    if constexpr (std::is_same_v<Block, block_q4_0>) dequantize_row_q4_0(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_q4_K>) dequantize_row_q4_K(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_q3_K>) dequantize_row_q3_K(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_iq3_xxs>) dequantize_row_iq3_xxs(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_iq3_s>) dequantize_row_iq3_s(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_iq4_nl>) dequantize_row_iq4_nl(static_cast<const Block *>(input), output, count);
    else if constexpr (std::is_same_v<Block, block_iq4_xs>) dequantize_row_iq4_xs(static_cast<const Block *>(input), output, count);
}

#define FORMAT(name, type, enum_type, values, quantizer, metadata) { name, enum_type, values, sizeof(type), quantize<type>, dequantize<type>, quantizer, metadata }

static const Format FORMATS[] = {
    FORMAT("Q4_0", block_q4_0, GGML_TYPE_Q4_0, QK4_0, "quantize_row_q4_0_ref", "fp16 scale + packed nibbles"),
    FORMAT("Q4_K", block_q4_K, GGML_TYPE_Q4_K, QK_K, "quantize_row_q4_K_ref", "two fp16 super-scales + 6-bit scale/min metadata + packed nibbles"),
    FORMAT("Q3_K", block_q3_K, GGML_TYPE_Q3_K, QK_K, "quantize_row_q3_K_ref", "fp16 super-scale + 6-bit scales + low/high quant bits"),
    FORMAT("IQ3_XXS", block_iq3_xxs, GGML_TYPE_IQ3_XXS, QK_K, "quantize_row_iq3_xxs_ref", "unweighted reference IQ quantizer; no importance matrix"),
    FORMAT("IQ3_S", block_iq3_s, GGML_TYPE_IQ3_S, QK_K, "quantize_row_iq3_s_ref", "unweighted reference IQ quantizer; no importance matrix"),
    FORMAT("IQ4_NL", block_iq4_nl, GGML_TYPE_IQ4_NL, QK4_NL, "quantize_row_iq4_nl_ref", "unweighted reference IQ quantizer; fp16 scale + nonlinear packed nibbles"),
    FORMAT("IQ4_XS", block_iq4_xs, GGML_TYPE_IQ4_XS, QK_K, "quantize_row_iq4_xs_ref", "unweighted reference IQ quantizer; no importance matrix"),
};

static const Format * get_format(const char * name) {
    for (const auto & format : FORMATS) if (std::strcmp(format.name, name) == 0) return &format;
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc != 6) return 2;
    const Format * format = get_format(argv[1]);
    const int rows = std::atoi(argv[2]);
    const int columns = std::atoi(argv[3]);
    if (!format || rows <= 0 || columns <= 0 || columns % format->values != 0) return 3;
    ggml_quantize_init(format->type);
    const size_t input_bytes = size_t(rows) * columns * sizeof(float);
    const size_t row_bytes = size_t(columns / format->values) * format->bytes;
    std::vector<float> input(size_t(rows) * columns);
    std::vector<uint8_t> packed(size_t(rows) * row_bytes);
    std::vector<uint8_t> repeat(size_t(rows) * row_bytes);
    std::vector<float> output(size_t(rows) * columns);
    FILE * source = std::fopen(argv[4], "rb");
    if (!source || std::fread(input.data(), 1, input_bytes, source) != input_bytes) return 4;
    std::fclose(source);
    for (int row = 0; row < rows; ++row) {
        const float * values = input.data() + size_t(row) * columns;
        void * block = packed.data() + size_t(row) * row_bytes;
        format->quantize(values, block, columns);
        format->dequantize(block, output.data() + size_t(row) * columns, columns);
        format->quantize(values, repeat.data() + size_t(row) * row_bytes, columns);
    }
    FILE * destination = std::fopen(argv[5], "wb");
    if (!destination || std::fwrite(output.data(), sizeof(float), output.size(), destination) != output.size()) return 5;
    std::fclose(destination);
    std::printf("{\"type\":\"%s\",\"values_per_block\":%d,\"bytes_per_block\":%zu,\"rows\":%d,\"columns\":%d,\"payload_bytes\":%zu,\"deterministic\":%s,\"quantizer\":\"%s\",\"metadata\":\"%s\"}\n", format->name, format->values, format->bytes, rows, columns, packed.size(), std::memcmp(packed.data(), repeat.data(), packed.size()) == 0 ? "true" : "false", format->quantizer, format->metadata);
    return 0;
}
