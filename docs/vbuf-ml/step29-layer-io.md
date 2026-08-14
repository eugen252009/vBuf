# Step 29 — layer compute boundaries and bulk I/O preparation

Status: **complete; host-relative qualification only**.

This step measures the producer/consumer envelope on one host. It does not
establish universal vBuf bandwidth, universal layer timing, or universal wave
feasibility.

```text
format capability != host execution policy
```

The artifact exposes exact physical spans and semantic layer IDs. The host
determines storage rate, RAM behavior, compute rate, concurrency, and useful
lookahead.

## Execution seam

The pinned llama.cpp/GGML build already exposes the internal
`llama_context::get_sched()` seam. The benchmark uses the existing
`ggml_backend_sched_set_eval_callback` evaluation callback. It observes named
terminal graph nodes:

```text
embd
l_out-0 ... l_out-63
result_norm
result_output
```

The Qwen3 graph construction assigns these names through its existing graph
callback. `l_out-N` is the terminal node of semantic transformer layer N.
The scheduler callback is diagnostic-only and forces observable graph segments;
it does not change tensor semantics, graph construction, kernels, or execution
order. The callback instrumentation overhead is separated by reporting both
resident ordinary SECOND_EVAL and traced evaluation time.

## Hardware profile

The measured host profile is recorded in `environment.json` with a stable
benchmark-local ID and no hostname, username, serial number, or home path.
Observed values include:

```text
CPU: AMD Ryzen 7 5800X, 8 physical / 16 logical CPUs
RAM: 65,727,312 kB total; 55,834,616 kB available at capture
page size: 4096 bytes
kernel: 6.12.101+deb13-amd64
filesystem: ext4 mount on /dev/nvme0n1p1
NVMe model: Intenso NVME
GPU/PCI/block topology: recorded where discoverable
memory channel/DIMM topology: unknown
```

Hardware metadata is descriptive only. No bandwidth is inferred from device
marketing specifications.

## Layer compute timing

Ten warm/resident 32B samples were collected. The traced callback observed 64
layer segments per sample.

Aggregate transformer-layer compute:

```text
min:    1,046.0 ms
median: 1,060.6 ms
mean:   1,071.1 ms
p95:    1,101.6 ms
max:    1,145.4 ms
```

Non-layer traced compute/dispatch overhead:

```text
median: ≈27.1 ms
```

Resident ordinary SECOND_EVAL was approximately 1.08–1.09 seconds in the
same runs. The traced callback is a diagnostic boundary instrument and may
change scheduling granularity; ordinary resident evaluation remains the
steady-state reference.

Per-layer compute timing is in `layer-compute.csv` and distribution summaries
are in `summary.json`.

## Bulk I/O mechanisms

Tested mechanisms:

```text
A  mmap page-touch control
B  synchronous bounded pread
C  bounded concurrent pread using worker threads
D  whole-span bounded-scratch pread control
```

No io_uring dependency was added. The bounded pread implementation reuses
per-worker scratch buffers and never allocates a full-model duplicate.

The bulk benchmark validates file bounds, handles short reads explicitly, and
fails samples on read errors.

## Read-width results

Uncached-approximation 32B examples:

| Width | QD | Effective GB/s | Elapsed |
|---:|---:|---:|---:|
| 4 MiB | 1 | 1.296 | 26.87 s |
| 4 MiB | 4 | 1.376 | 25.3 s |
| 4 MiB | 16 | 1.334 | 26.09 s |
| 16 MiB | 1 | 1.296 | 26.87 s |
| 16 MiB | 4 | **1.409** | 24.7 s |
| 16 MiB | 16 | 1.269 | 27.44 s |
| 64 MiB | 4 | 1.265 | 27.52 s |
| whole-span | 1 | 1.143 | 30.44 s |

All tested widths, requests, successful bytes, short reads, errors, CPU
accounting, and buffer reservations are recorded in
`io-configuration-comparison.csv`.

## Queue/concurrency results

The best measured uncached 32B configuration was:

```text
synchronous pread
chunk width: 16 MiB
queue depth: 4
buffer reservation: 64 MiB
throughput: ≈1.409 GB/s
```

Higher queue depth was not universally beneficial. QD16 reached approximately
1.269 GB/s at the best width tested.

## Page-touch control

The separate bulk benchmark measured uncached page-touch preparation at
approximately:

```text
1.232 GB/s
```

This is close to, but below, the Step-28 approximately 1.36 GB/s result. The
variation is expected from separate-process mapping/cache conditions. The best
bulk pread result was approximately 14% faster than this Step-29 control.

## Whole-model sequential control

The whole-span bounded-scratch control measured approximately 1.143 GB/s. It
uses one bounded scratch buffer per semantic span rather than allocating a
35 GB buffer.

Layer/file order remains effectively physical file order for this artifact.

## Layer preparation timing

For the best uncached 32B pread configuration:

```text
median layer preparation: ≈359 ms
p95:                     ≈382 ms
max:                     ≈489.6 ms
```

The exact per-layer values, offsets, chunk widths, requests, and effective
throughput are in `layer-preparation.csv`. Global spans are separately recorded
in `global-preparation.csv`.

## Global span preparation

The three global spans remain separate:

```text
token_embd.weight
output.weight
output_norm.weight
```

They are not attributed to transformer layers.

## Producer/consumer comparison

Host-local aggregate values:

```text
median layer compute:      ≈16.15 ms
median layer preparation:  ≈359 ms
median prepare/compute:    ≈22.2×
median supply ratio:       ≈0.045
```

The worst measured producer/consumer mismatch was layer 37 at approximately
31.4× preparation/compute.

## Required storage bandwidth

These values mean:

> bandwidth required on this measured host to prepare a layer within its
> measured compute window.

They do not mean bandwidth required by vBuf universally.

```text
minimum: ≈28.9 GB/s
median:  ≈32.1 GB/s
mean:    ≈31.9 GB/s
p95:     ≈33.2 GB/s
maximum: ≈33.7 GB/s
```

The best measured producer rate was approximately 1.409 GB/s, far below these
host-local requirements.

## Lookahead simulation

An idealized deterministic producer/consumer simulation was run using measured
host-local preparation and compute values. It assumes serial compute, a serial
producer, no interference penalty, and no readiness before preparation
completion. It is not evidence of actual overlap.

No tested initial prefix/window combination avoided modeled stalls. Examples:

```text
prefix 8, window 16:
  modeled stalls: 56
  modeled stall duration: ≈19.6 s
  resident weight budget: ≈12.4 GB

prefix 32, window 16:
  modeled stalls: 32
  modeled stall duration: ≈11.1 s
  resident weight budget: ≈24.9 GB
```

These are host-local simulation results, not portable vBuf requirements.

## Dense streaming feasibility

Host-relative classification:

```text
C — producer slower than consumer on this host
```

This does not establish that dense streaming is universally infeasible. It
means this measured host/storage configuration cannot feed the measured dense
Qwen3-32B layer compute rate using the tested producer path.

## Startup implication

Step 28's explicit preparation can improve startup/first-use relative to
reactive faults. Step 29 shows that bulk pread improves preparation throughput
modestly over page touching, but still pays substantial startup time.

## Steady-state implication

Resident layer compute is approximately 16 ms per layer on this host. The
measured preparation path is approximately 22× slower per layer. Startup
preparation evidence must not be interpreted as a steady-state dense token
throughput improvement.

Future MoE selective-expert behavior is a separate question and is not
qualified here.

## Small-model control

The same bulk machinery was exercised on Qwen3-0.6B. The small artifact did not
require special handling, and bounded buffer reservations remained controlled.
Small-model results are controls, not optimization targets.

## Cache / uncached approximation

Warm/cache-sensitive and uncached-approximation results are stored separately.
The uncached path uses `POSIX_FADV_DONTNEED` whole-file advisory eviction. It is
not guaranteed cold storage.

## Faults

The bulk benchmark records minor and major faults before/after preparation.
Explicit pread into scratch buffers measures storage movement independently; it
does not itself make the llama mmap mapping resident.

## RSS / buffer memory

The best pread configuration reserved 64 MiB of scratch buffers. Whole-span
control reserved at most one largest semantic span, not the full model.
RSS and buffer reservation are recorded per sample.

## I/O counters

`read_bytes`, `rchar`, and `syscr` are recorded. `read_bytes` is kernel
accounting and is not treated as exact NAND traffic.

## CPU / system time

The best 32B uncached pread configuration used approximately 10 seconds of
system time during the 24–25 second preparation interval. This is host-local
page/cache/filesystem behavior, not a portable format property.

A 512 MiB memcpy control is recorded separately in `memory-bandwidth.json` and
is not used as a substitute for model or storage measurements.

## Correctness

```text
artifact hashes: PASS
tensor repack/reorder: none
payload duplication: 0
generation parity: PASS
```

Step-26 remains the exact tensor/logit authority:

```text
logit max_abs_diff = 0
```

## Portability result

**YES.** The qualified artifacts and LayerSpanPlan remain independent of RAM,
CPU, GPU, NVMe, RAID, and host configuration. Hardware changes execution
policy, not artifact validity.

## Runtime policy implication

A future runtime would need host-local inputs including:

```text
validated LayerSpanPlan
available RAM/residency budget
measured or selected storage preparation rate
resident layer compute rate
bounded concurrency capability
cache/residency state
```

Possible policy decisions remain runtime-local and must not be persisted into
the vBuf artifact.

## Portable format conclusion

vBuf provides the physical and semantic information needed for host-specific
residency policy decisions:

```text
known physical spans
semantic layer mapping
exact byte ranges
near-zero span amplification
early structural readiness
```

It does not prescribe one wave policy or one universal bandwidth threshold.

## Decisions

Bulk I/O decision:

```text
YES — on this host, bounded bulk pread modestly outperformed page-touch
preparation and provides explicit successful-read accounting.
```

Wave Track-B decision:

```text
NO / NOT YET — the producer is much slower than resident dense layer compute
on this host, and the idealized bounded-window simulation still stalls.
```

A future positive decision would require another host/configuration or a
separately scoped execution experiment. No wave runtime was implemented.

## Constraints preserved

```text
wire changes: 0
BaseShift changes: 0
Nano changes: 0
Nested-vBuf changes: 0
production LayerView/RuntimeChunk APIs: 0
scheduler architecture: 0
llama/GGML kernels: 0
GPU offload: 0
```

Evidence:

```text
benchmark-results/vbuf-ml-step29-layer-io/
```
