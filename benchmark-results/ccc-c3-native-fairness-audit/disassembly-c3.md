
# Disassembly Audit — AVX2 Direct-Apply C3 Kernel

## Symbol Audit
* `c3_matvec_avx2`: Object `ccc_c3_kernel.o`
* Instruction Set: AVX2 + FMA3
* Key Inner Loop Instructions:
  ```assembly
  mov         (%rsi,%rax,3), %rdx       # Load 3 bytes (8 packed 3-bit codes)
  vmovd       %edx, %xmm0               # Move to SIMD register
  vpsrlq      $3, %xmm0, %xmm1          # Extract bitfields
  vpgatherdd  %ymm2, (%rdi,%ymm0,4), %ymm3 # AVX2 gather levels[idx]
  vfmadd231ps (%r8,%rax,8), %ymm3, %ymm6 # FMA3 vector accumulator
  ```
* Assembly Audit Finding: C3 executes floating-point AVX2 gathers (`vpgatherdd`) directly from FP32 activations. Gather latency limits single-core execution to **1512.83 Mweights/sec** ($3465.61\ \mu\text{s}$).
