# Orange Pi RV2 Environment

Captured 2026-08-16 from the native qualification host.

```text
Linux orangepirv2 6.6.63-ky #1.0.0 SMP PREEMPT Wed Mar 12 09:04:00 CST 2025 riscv64 riscv64 riscv64 GNU/Linux
riscv64
gcc-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0
g++-14 (Ubuntu 14.2.0-4ubuntu2~24.04.1) 14.2.0
cmake version 3.28.3
```

CPU:

```text
Model name: Ky(R) X1
Uarch: ky,x60
CPU(s): 8
CPU max MHz: 1600.0000
```

RVV and related ISA capabilities reported by `/proc/cpuinfo`:

```text
rv64imafdcv_zicbom_zicboz_zicntr_zicond_zicsr_zifencei_zihintpause_zihpm_zfh_zfhmin_zca_zcd_zba_zbb_zbc_zbs_zkt_zve32f_zve32x_zve64d_zve64f_zve64x_zvfh_zvfhmin_zvkt_sscofpmf_sstc_svinval_svnapot_svpbmt
```

Memory:

```text
total 7.7Gi; available 7.1Gi; swap 0B
```

The default `gcc`/`g++` symlinks were 13.3.0. The native qualification
explicitly used installed `gcc-14`/`g++-14` 14.2.0 because GCC 13 could not
compile the pinned ggml RVV FP16 intrinsic path. No fallback non-RVV build was
used.
