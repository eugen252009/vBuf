// Research-only reproduction of the parallel hard-gate canonical entry points.
#include "ggml.h"
#include "ggml-quants.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void write(const char * path, const void * p, size_t n) {
    std::ofstream out(path, std::ios::binary); if (!out) throw std::runtime_error("output open failed");
    out.write(static_cast<const char *>(p), (std::streamsize) n); if (!out) throw std::runtime_error("output write failed");
}
int main(int argc, char ** argv) {
    if (argc != 6) return 2;
    const std::string format=argv[1]; const size_t elements=std::stoull(argv[5]);
    std::vector<float> src(elements); std::ifstream in(argv[2],std::ios::binary); if (!in) return 3;
    in.read(reinterpret_cast<char *>(src.data()),(std::streamsize)(src.size()*sizeof(float))); if ((size_t)in.gcount()!=src.size()*sizeof(float)) return 4;
    ggml_type type; size_t block_bytes;
    if (format=="Q2_K") { type=GGML_TYPE_Q2_K; block_bytes=84; }
    else if (format=="Q3_K") { type=GGML_TYPE_Q3_K; block_bytes=110; }
    else if (format=="Q4_K") { type=GGML_TYPE_Q4_K; block_bytes=144; }
    else return 5;
    if (elements%256) return 6; const int64_t blocks=(int64_t)elements/256; std::vector<uint8_t> q((size_t)blocks*block_bytes); size_t written=0;
    if (type==GGML_TYPE_Q2_K) written=quantize_q2_K(src.data(),q.data(),blocks,256,nullptr);
    if (type==GGML_TYPE_Q3_K) written=quantize_q3_K(src.data(),q.data(),blocks,256,nullptr);
    if (type==GGML_TYPE_Q4_K) written=quantize_q4_K(src.data(),q.data(),blocks,256,nullptr);
    if (written!=q.size()) return 7; std::vector<float> restored(elements);
    for (int64_t i=0;i<blocks;++i) {
        if (type==GGML_TYPE_Q2_K) dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(q.data())+i,restored.data()+i*256,256);
        if (type==GGML_TYPE_Q3_K) dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(q.data())+i,restored.data()+i*256,256);
        if (type==GGML_TYPE_Q4_K) dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(q.data())+i,restored.data()+i*256,256);
    }
    write(argv[3],q.data(),q.size()); write(argv[4],restored.data(),restored.size()*sizeof(float)); return 0;
}
