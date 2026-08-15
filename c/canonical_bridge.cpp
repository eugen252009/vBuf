#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>
#include "ggml.h"
#include "ggml-quants.h"

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <format: Q2_K|Q3_K|Q4_0|Q4_K> <input_f32.bin> <output_recon_f32.bin>" << std::endl;
        return 1;
    }
    
    std::string fmt = argv[1];
    std::string in_path = argv[2];
    std::string out_path = argv[3];
    
    // Read input float matrix
    std::ifstream fin(in_path, std::ios::binary);
    uint64_t n_elems = 0;
    fin.read(reinterpret_cast<char*>(&n_elems), sizeof(n_elems));
    
    std::vector<float> src(n_elems);
    fin.read(reinterpret_cast<char*>(src.data()), n_elems * sizeof(float));
    fin.close();
    
    std::cout << "[CANONICAL BRIDGE] Read " << n_elems << " weights for format " << fmt << std::endl;
    
    ggml_quantize_init(GGML_TYPE_Q8_0);
    
    std::vector<float> recon(n_elems);
    size_t total_bytes = 0;
    
    if (fmt == "Q2_K") {
        size_t block_size = 256;
        size_t block_bytes = 84;
        size_t n_blocks = n_elems / block_size;
        total_bytes = n_blocks * block_bytes;
        
        std::vector<uint8_t> buf(total_bytes);
        quantize_q2_K(src.data(), buf.data(), n_blocks, block_size, nullptr);
        
        for (size_t i = 0; i < n_blocks; i++) {
            const block_q2_K * b = (const block_q2_K *)(buf.data() + i * block_bytes);
            dequantize_row_q2_K(b, recon.data() + i * block_size, block_size);
        }
    } else if (fmt == "Q3_K") {
        size_t block_size = 256;
        size_t block_bytes = 110;
        size_t n_blocks = n_elems / block_size;
        total_bytes = n_blocks * block_bytes;
        
        std::vector<uint8_t> buf(total_bytes);
        quantize_q3_K(src.data(), buf.data(), n_blocks, block_size, nullptr);
        
        for (size_t i = 0; i < n_blocks; i++) {
            const block_q3_K * b = (const block_q3_K *)(buf.data() + i * block_bytes);
            dequantize_row_q3_K(b, recon.data() + i * block_size, block_size);
        }
    } else if (fmt == "Q4_0") {
        size_t block_size = 32;
        size_t block_bytes = 18;
        size_t n_blocks = n_elems / block_size;
        total_bytes = n_blocks * block_bytes;
        
        std::vector<uint8_t> buf(total_bytes);
        quantize_q4_0(src.data(), buf.data(), n_blocks, block_size, nullptr);
        
        for (size_t i = 0; i < n_blocks; i++) {
            const block_q4_0 * b = (const block_q4_0 *)(buf.data() + i * block_bytes);
            dequantize_row_q4_0(b, recon.data() + i * block_size, block_size);
        }
    } else if (fmt == "Q4_K") {
        size_t block_size = 256;
        size_t block_bytes = 144;
        size_t n_blocks = n_elems / block_size;
        total_bytes = n_blocks * block_bytes;
        
        std::vector<uint8_t> buf(total_bytes);
        quantize_q4_K(src.data(), buf.data(), n_blocks, block_size, nullptr);
        
        for (size_t i = 0; i < n_blocks; i++) {
            const block_q4_K * b = (const block_q4_K *)(buf.data() + i * block_bytes);
            dequantize_row_q4_K(b, recon.data() + i * block_size, block_size);
        }
    } else {
        std::cerr << "Unknown canonical format: " << fmt << std::endl;
        return 1;
    }
    
    double bpw = (double)total_bytes * 8.0 / (double)n_elems;
    std::cout << "[CANONICAL BRIDGE] Format " << fmt << ": total_bytes=" << total_bytes << " -> True bpw=" << bpw << std::endl;
    
    // Write reconstructed floats out
    std::ofstream fout(out_path, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(&n_elems), sizeof(n_elems));
    fout.write(reinterpret_cast<const char*>(recon.data()), n_elems * sizeof(float));
    fout.close();
    
    return 0;
}
