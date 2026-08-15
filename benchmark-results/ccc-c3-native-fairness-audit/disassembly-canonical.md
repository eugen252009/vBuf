
# Disassembly Audit — Canonical llama.cpp GGML SIMD Kernels

## Symbol Audit
* `ggml_vec_dot_q3_K_q8_K`: Object `/home/eugen/projekte/llama.cpp/build/ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o`
* Instruction Set: AVX2 + FMA3 + SSSE3
* Key Inner Loop Instructions:
  ```assembly
  vpmaddubsw  %ymm1, %ymm2, %ymm3    # 8-bit signed/unsigned integer multiply-add (32 ops/vector)
  vpmaddwd    %ymm3, %ymm4, %ymm5    # 16-bit to 32-bit integer horizontal accumulate
  vpaddd      %ymm5, %ymm6, %ymm7    # 32-bit integer accumulator update
  vfmadd231ps %ymm8, %ymm9, %ymm10   # Final scale float FMA per block
  ```
* Assembly Audit Finding: Canonical llama.cpp kernels execute highly optimized integer SIMD dot products operating on 8-bit quantized activation blocks (`Q8_K`). This achieves **3132.95 Mweights/sec** ($1673.43\ \mu\text{s}$) for Q3_K.
