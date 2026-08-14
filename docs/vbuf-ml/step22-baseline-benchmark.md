# Step 22: neutral CPU baseline qualification

Status: **baseline recorded; no optimization introduced**.

## Frozen checkpoint

```text
Tag:    vbuf-ml-0.1-consumer-parity
Commit: 73a1f36661482037573a789ab90e15039a782ad9
llama:  4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
Patch:  patches/llama.cpp/0001-user-metadata-tensor-source.patch
```

The benchmark verifies the tag, pinned checkout, patch application, artifact
hashes, and Step-21 PASS evidence before running.

## Harness and phases

`integrations/llama.cpp/step22_benchmark.cpp` measures in-process monotonic
phase timestamps and Linux process counters. `scripts/qualify_step22.py`
compiles the same CPU adapter/runtime and runs balanced GGUF/vBuf orderings.

Measured phases:

```text
metadata_control   vocab-only model construction
model_ready        full model construction
prompt_tokenization
prompt_eval
 generation        eight greedy tokens
```

Primary prompt:

```text
benchmarks/vbuf-ml/step22/prompts.json:primary
```

Settings:

```text
CPU backend, n_gpu_layers=0
n_threads=2, n_threads_batch=2
n_ctx=512, n_batch=512
```

Warm runs: 3 per format/artifact. Uncached runs: 2 per format/artifact.
Metadata-control runs currently have one sample per condition and are
therefore diagnostic rather than statistically strong.

## Cache and environment limitations

The host did not provide a privileged `drop_caches` method. The benchmark uses
`POSIX_FADV_DONTNEED` per file and labels these runs an **uncached
approximation**, not cold-cache proof. No CPU affinity was forced. The full
environment, hashes, settings, raw samples, and filesystem caveat are recorded
under `benchmark-results/vbuf-ml-step22/`.

## Median baseline observations

Relative delta is `(vBuf - GGUF) / GGUF`; negative means lower latency.

| Artifact / condition | Phase | GGUF median | vBuf median | Relative delta |
|---|---|---:|---:|---:|
| BF16 warm | model ready | 248.9 ms | 407.7 ms | +63.8% |
| BF16 uncached | model ready | 1,070.1 ms | 486.8 ms | -54.5% |
| BF16 warm | metadata control | 159.8 ms | 418.9 ms | +162.2% |
| BF16 warm | prompt eval | 286.4 ms | 318.0 ms | +11.0% |
| BF16 warm | generation | 285.3 ms | 274.8 ms | -3.7% |
| Q8_0 warm | model ready | 193.2 ms | 413.8 ms | +114.2% |
| Q8_0 uncached | model ready | 543.5 ms | 473.9 ms | -12.8% |
| Q8_0 warm | metadata control | 157.2 ms | 415.8 ms | +164.4% |
| Q8_0 warm | prompt eval | 223.5 ms | 237.6 ms | +6.3% |
| Q8_0 warm | generation | 164.4 ms | 161.1 ms | -2.0% |

These are observations from the recorded configuration, not general
performance claims. The steady-state control behaves as expected: generation
is effectively equal, while source-specific preparation and some first-touch
behavior differ materially.

## Resource observations

The raw process snapshots include minor/major faults, RSS, PSS, virtual size,
and `/proc/self/io` read bytes. The vBuf path has direct mmap-backed payloads;
its runtime-ready RSS/PSS is substantially lower before prompt execution in
these runs, while prompt evaluation faults/pages in the mapped payload. The
GGUF and vBuf loader paths show different fault/read patterns and must not be
collapsed into file-size claims.

No tensor payload copy, dequantization, requantization, Q8_0 repack, or tensor
transformation was added for benchmarking.

## Result classification

```text
Correctness gate: PASS (Step 21)
BF16 baseline: RECORDED
Q8_0 baseline: RECORDED
Steady-state control: approximately equal
Cold-cache claim: NOT ESTABLISHED; uncached approximation only
Performance claim: NONE beyond this disclosed configuration
GPU: DEFERRED
```

Raw data and generated summaries:

```text
benchmark-results/vbuf-ml-step22/raw/samples.csv
benchmark-results/vbuf-ml-step22/phase-summary.csv
benchmark-results/vbuf-ml-step22/summary.csv
benchmark-results/vbuf-ml-step22/resource-summary.csv
benchmark-results/vbuf-ml-step22/environment.json
```

No Step-23 optimization was started. The measured metadata/control overhead,
model-ready variance, and vBuf first-touch/page-fault behavior are hypotheses
for later investigation only.
