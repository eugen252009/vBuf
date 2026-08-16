# Orange Pi RV2 CPU_REPACK A/B Qualification

## Scope

This is a warm-cache A/B measurement on the same Orange Pi RV2, using the same
DeepSeek-V2-Lite.IQ1_S.vBuf artifact, pinned patched source, compiler, model
parameters, and CPU/RVV configuration as the qualified baseline. The previous
GGUF/vBuf benchmark evidence was not modified.

Evidence is in `research/results/cpu-repack-rv2-ab/`.

## Variants

### REPACK_ENABLED

- Existing `~/llama-vbuf-rvv-build`.
- `GGML_CPU_REPACK=ON`.
- 28 repacked tensors.
- RVV `iq4_nl_16x1` and `q2_K_16x1` paths.

### REPACK_DISABLED

- Separate build directory: `~/llama-vbuf-rvv-ab-off`.
- Same source directory: `~/llama-vbuf-pinned`.
- Same GCC/G++ 14, RVV, OpenMP, native CPU flags.
- Only CMake option changed:

```text
GGML_CPU_REPACK=OFF
```

The exact OFF shared libraries were selected at runtime with:

```text
LD_LIBRARY_PATH=/home/eugen/llama-vbuf-rvv-ab-off/bin:/home/eugen/projekte/vBuf/rust/target/release
```

The original `~/llama.cpp`, qualified `~/llama-vbuf-rvv-build`, vBuf library,
model, and committed benchmark directory were not modified.

## Fixed Parameters

```text
context:       256
batch:         64
ubatch:        64
threads:       8
batch threads: 8
prompt:        Hello world
prompt tokens: 3
generation:    32 tokens
sampling:      greedy
seed:          default, with no random sampling used
cache regime:  WARM_CACHE
```

Run order was balanced:

```text
A, B, B, A, A, B
ON-00, OFF-00, OFF-01, ON-01, ON-02, OFF-02
```

CPU frequency was 1.6 GHz in all samples. Maximum observed temperature was
63 C for ON and 65 C for OFF. No swap was configured or used. The sampler
recorded RSS, available memory, faults, I/O, process CPU, temperature, and
frequency for every run.

## Correctness

Both variants passed the correctness gate:

- model load succeeded;
- context creation succeeded;
- graph reservation succeeded with 1710 nodes and 1 split;
- prompt evaluation succeeded;
- all 32 decode calls returned zero;
- process exited zero;
- no crash, invalid access, or OOM occurred;
- generated token IDs were exactly identical.

The common generated sequence was:

```text
0 304 487 76 1062 0 304 487 313 803 5418 327 245 1477 11 285
304 487 2494 185 40 487 313 803 5418 327 245 1477 11 285 304 487
```

Dispatch qualification used a separate low-overhead `LD_PRELOAD` wrapper,
preserved in `integrations/llama.cpp/rv2_repack_ab_dispatch_preload.cpp`.
It confirmed:

```text
ON:  iq4_nl_16x1, q2_K_16x1
OFF: ordinary_iq4_nl_q8_0, ordinary_q2_K_q8_K
```

The final timed runs did not use the preload wrapper.

## Results

Times are milliseconds, RSS and available memory are GiB. Each cell is
minimum / median / maximum across three runs. The primary comparison is the
median.

| Metric | REPACK ON | REPACK OFF | OFF - ON median |
|---|---:|---:|---:|
| Structure ready | 2156.923 / 2162.207 / 2200.519 | 2166.300 / 2197.219 / 2221.270 | +35.012 ms |
| Payload end | 50364.396 / 68392.801 / 73394.188 | 50835.630 / 57906.993 / 59249.032 | -10485.808 ms |
| Execution ready | 50762.636 / 68718.511 / 73767.821 | 51154.678 / 58211.220 / 59584.987 | -10507.291 ms |
| First token | 52719.237 / 70604.725 / 75359.973 | 53103.447 / 59766.592 / 61112.386 | -10838.133 ms |
| Total startup/run | 65272.279 / 83235.914 / 87907.214 | 65198.792 / 71837.721 / 73455.560 | -11398.193 ms |
| Prompt evaluation | 1195.405 / 1487.760 / 1548.076 | 1143.463 / 1169.131 / 1464.463 | -318.629 ms |
| Prompt tok/s | 1.937889 / 2.016454 / 2.509610 | 2.048532 / 2.566008 / 2.623609 | +0.549554 tok/s |
| First decode | 395.740 / 397.454 / 407.469 | 382.959 / 385.170 / 483.304 | -12.284 ms |
| Steady decode tok/s | 2.456128 / 2.469034 / 2.472386 | 2.514515 / 2.543995 / 2.568983 | +0.074961 tok/s |
| Peak RSS | 7.317 / 7.317 / 7.319 | 7.317 / 7.317 / 7.319 | -0.002 GiB |
| RSS at execution ready | 7.269 / 7.301 / 7.319 | 7.299 / 7.319 / 7.319 | +0.015 GiB |
| Steady RSS | 7.239 / 7.242 / 7.292 | 7.245 / 7.252 / 7.262 | +0.005 GiB |
| Minimum available RAM | 2.521 / 2.524 / 2.527 | 2.517 / 2.524 / 2.529 | +0.003 GiB |
| Major faults | 22514 / 32152 / 33333 | 24306 / 28652 / 29573 | -3500 |

## Repack Accounting

| Metric | REPACK ON | REPACK OFF |
|---|---:|---:|
| Repack tensors | 28 / 28 / 28 | 0 / 0 / 0 |
| Source bytes | 2713534464 | 0 |
| Destination bytes | 2713534464 | 0 |
| Repack wall time | 39252.213 / 40956.498 / 41476.068 ms | 0 / 0 / 0 ms |

The OFF build did not disable RVV. It removed the CPU_REPACK buffer type from
selection, causing the original tensors to use ordinary CPU storage and the
ordinary RVV quantized dot-product kernels.

## Memory Cost

The expected 2.5 GiB RSS saving did not occur in this llama/ggml composition.
The allocations were rearranged, not removed:

| Buffer | REPACK ON | REPACK OFF |
|---|---:|---:|
| CPU model buffer | 2171.14 MiB | 4758.96 MiB |
| CPU_REPACK model buffer | 2587.83 MiB | absent |
| CPU output buffer | 0.39 MiB | 0.39 MiB |
| CPU KV buffer | 67.50 MiB | 67.50 MiB |
| CPU compute buffer | 30.03 MiB | 30.03 MiB |

The ON model buffers total 4758.97 MiB, essentially identical to the OFF CPU
model buffer of 4758.96 MiB. The direct vBuf mapping remains live in both
variants, so the process RSS remains about 7.32 GiB in both. In this current
loader, disabling repack means a larger ordinary CPU destination, not direct
zero-copy execution.

The measured memory conclusion is therefore:

```text
CPU_REPACK storage saved:       2587.83 MiB as a named buffer
total model storage saved:      approximately 0 MiB
peak RSS saved:                 approximately 0.002 GiB, within run noise
steady RSS saved:               none; OFF was about 0.005 GiB higher median
```

## What Is Gained and Paid

For this exact RV2 setup and current runtime composition:

### Disabling CPU_REPACK gains

- 10.507 seconds lower median execution-ready time.
- 10.838 seconds lower median first-token time.
- 11.398 seconds lower median total run time for the 32-token workload.
- 3.04% higher median steady decode throughput.
- 0.074961 tok/s higher median decode rate.
- Correct direct execution with identical deterministic output.

### Disabling CPU_REPACK pays

- No measured RSS or total model-storage reduction.
- The ordinary CPU model buffer grows by 2587.82 MiB.
- Prompt throughput is noisy because the prompt contains only three tokens;
  the three-run median happened to be higher OFF, but this is not a strong
  prompt-throughput claim.
- The direct path still copies the original payloads into a backend-owned CPU
  buffer, so it does not yet realize vBuf-native zero-copy execution.

### CPU_REPACK itself costs

- 28 tensor transformations over 2.527 GiB.
- Median measured repack wall time: 40.956 seconds.
- Net median startup penalty relative to OFF: 10.507 seconds, because OFF also
  materializes the original payloads into its larger CPU buffer.

## Break-Even

There is no positive break-even point in this measurement.

REPACK ON is slower to execution readiness and slower in steady decode:

```text
ON startup penalty:       approximately 10.507 s
ON decode advantage:      none; OFF is 3.04% faster
```

Since the supposedly optimized path is not faster after startup, it cannot
recover its startup cost at any token count in this A/B result. The measured
ordering is approximately:

```text
OFF total time =  startup_off + tokens / 2.543995 tok/s
ON total time  =  startup_on  + tokens / 2.469034 tok/s
```

Both the intercept and slope favor OFF. Memory pressure also does not provide
an ON advantage: the two variants have effectively identical RSS and available
RAM.

This result is specific to the measured DeepSeek model, RV2, pinned source,
RVV implementation, and runtime parameters. It is not a general claim about
all RVV CPUs or models.

## Architectural Classification

**NOT_WORTH_USING** for this measured Orange Pi RV2 setup and this model/runtime
composition.

This classification is a workload-specific optimization result only. It does
not change the architectural correctness result:

```text
DirectExecution is valid.
RepackExecution remains optional.
Default does not become Core.
```

The measured result also strengthens the separation between the canonical path
and the optimization wrapper. CPU_REPACK adds eager work without reducing
residency in the current loader, and it does not improve measured decode
throughput here.

## Follow-Up Hypotheses

- A true vBuf-native direct execution path could avoid both the CPU copy and
  CPU_REPACK destination, but that was not implemented or measured here.
- The large payload-materialization variance is dominated by demand paging and
  memory pressure; a future experiment can isolate that separately.
- Other RVV VLENs, kernels, models, or longer workloads must be measured before
  generalizing this result.
