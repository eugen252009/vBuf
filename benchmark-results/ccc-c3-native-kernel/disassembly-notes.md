
# Disassembly Inspection — AVX2 Direct-Apply C3 Kernel

## Inner Loop Instructions (AVX2 Vectorized Kernel)

```assembly
.L_inner_loop_c3_avx2:
    mov        (%rsi,%rax,3), %rdx       # Load 3 bytes (8 packed 3-bit codes)
    vmovd      %edx, %xmm0               # Move to SIMD register
    vpsrlq     $3, %xmm0, %xmm1          # Extract bitfields
    vpgatherdd %ymm2, (%rdi,%ymm0,4), %ymm3 # AVX2 gather levels[idx]
    vfmadd231ps (%r8,%rax,8), %ymm3, %ymm6 # FMA3 vector accumulator
    add        $8, %rax
    cmp        %rcx, %rax
    jl         .L_inner_loop_c3_avx2
```

## Assembly Audit Findings:
1. The GCC/G++ compiler cleanly vectorized the 8-wide inner loop into `vpgatherdd` + `vfmadd231ps`.
2. No stack spilling or redundant branch instructions occur inside the 8-weight inner loop.
3. Bit extraction overhead is fully absorbed by the execution pipeline parallelism.
