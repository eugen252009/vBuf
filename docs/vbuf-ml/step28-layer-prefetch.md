# Step 28 — layer-span prefetch and wave-readiness qualification

Status: **complete; preparation qualified, wave execution not implemented**.

Step 28 tests whether the existing BaseShift 3 / BaseStep 8 physical layout can
prepare vBuf payload pages before first evaluation without changing model
semantics or the canonical wire format.

## Scope and variants

The direct vBuf loader was instrumented with a runtime-local, nonpersistent plan.
The four variants were:

```text
A  fault-driven baseline
B  whole-payload sequential prefault
C  ordered global/layer-span prefault
D  madvise(MADV_WILLNEED) advisory prefetch
```

Explicit prefault uses one volatile read per system page. Advisory prefetch is
reported separately and is not treated as completed residency.

Each variant ran 10 warm samples and 3 uncached-approximation samples for both
Qwen3-0.6B and Qwen3-32B vBuf artifacts. The uncached approximation retained
Step-27's `POSIX_FADV_DONTNEED` methodology and is not a true cold-cache test.

## Layer-span plan

The 32B plan was derived from validated tensor descriptors and physical tensor
ranges, not from Nano or wire metadata:

```text
layer spans: 64
physical span per layer: 1
global spans: 3
complete spans: 67
```

Global spans cover the embedding, output, and output-norm regions. They are
included in preparation ordering; no global weight region is silently omitted.

Plan totals:

```text
useful tensor bytes: 34,811,744,256
covered span bytes:  34,811,752,960
gap bytes:                 8,704
coverage amplification: 1.0000002500
```

The plan independently checks bounds, tensor coverage, physical offsets, and
pointer/span exactness. Full plan evidence is in:

```text
layer-span-plan.csv
global-span-plan.csv
```

## Primary uncached-approximation results

Times are milliseconds; medians over three samples:

| Model | Variant | MODEL_READY | PREPARATION | FIRST_EVAL | TTFUC | TTFT | SECOND_EVAL |
|---|---|---:|---:|---:|---:|---:|---:|
| 32B | A fault-driven | 322 | 0 | 35,663 | 35,999 | 35,999 | 1,072 |
| 32B | B sequential prefault | 318 | 25,543 | 1,066 | 26,950 | 26,950 | 1,042 |
| 32B | C layer-span prefault | 323 | 25,565 | 1,053 | 26,963 | 26,963 | 1,044 |
| 32B | D advisory prefetch | 321 | 1 | 35,732 | 36,077 | 36,077 | 1,048 |

The exact Step-27 control was approximately 36.145 s TTFUC. Controlled
sequential and layer prefault reduce uncached total first-use latency by about
9 seconds, while advisory prefetch does not.

Warm runs were intentionally retained as a separate cache-sensitive result.
Because the host page cache was already resident during the warm sequence, the
32B fault-driven warm baseline was approximately 1.899 s TTFUC and explicit
prefault cost approximately 1.64 s. Warm results therefore do not represent
Step-27's reactive first-touch condition; they are retained in the raw evidence.

## Fault attribution

For uncached 32B medians:

```text
A fault-driven:
  preparation major faults: 0
  FIRST_EVAL major faults: 42,116

B sequential prefault:
  preparation major faults: 142
  FIRST_EVAL major faults: 0

C layer-span prefault:
  preparation major faults: 142
  FIRST_EVAL major faults: 0

D advisory prefetch:
  preparation major faults: 0
  FIRST_EVAL major faults: 43,389
```

Explicit prefault moves the fault storm out of FIRST_EVAL. `WILLNEED` does not
produce that result in this qualification.

## Residency and I/O

For uncached 32B, explicit prefault reaches approximately full model RSS before
FIRST_EVAL, while the baseline remains small at MODEL_READY and grows during
first evaluation. Detailed RSS, `read_bytes`, `rchar`, `syscr`, and CPU phase
boundaries are in the corresponding CSV evidence.

The explicit preparation path pays a large system/page-touch cost before first
compute. Kernel `read_bytes` is retained as accounting evidence and is not
called exact physical storage traffic.

## Preparation bandwidth

Representative uncached 32B effective bandwidth:

```text
B sequential: ≈1.36 GB/s
C layer-span: ≈1.36 GB/s
```

This is covered span bytes divided by wall preparation time. It is not a raw
NVMe bandwidth measurement.

## Per-layer distribution

Layer-span preparation was recorded for every layer. For 32B uncached samples,
median individual layer preparation was approximately `379 ms`, with a p95 of
approximately `398 ms`. Warm samples were approximately `24 ms` median per
layer. The raw per-layer distribution is in `layer-preparation-raw.csv`.

No per-layer llama compute timing exists in this experiment. Therefore compute
and preparation overlap cannot be classified beyond **unknown**.

## Sequential versus layer order

The two physical strategies are effectively equivalent:

```text
B prepared span bytes: 34,816,197,376-ish complete payload coverage
C prepared span bytes: same complete layer/global coverage
B uncached TTFUC: 26.950 s
C uncached TTFUC: 26.963 s
```

The current layout is already layer-major with one contiguous span per layer.
Semantic layer ordering does not materially improve over ordered sequential
physical touching. This is evidence for category F.

## Advisory result

`MADV_WILLNEED` returned quickly but left the first-evaluation fault storm in
place:

```text
D preparation: ≈1 ms
D FIRST_EVAL: ≈35.732 s
D TTFUC: ≈36.077 s
```

Advisory prefetch was not effective as a completed residency mechanism here.

## Small-model control

Uncached 0.6B medians:

```text
A TTFUC: ≈809 ms
B TTFUC: ≈811 ms
C TTFUC: ≈813 ms
D TTFUC: ≈804 ms
```

Explicit preparation has no practical value for the small artifact and can be
slightly negative. This confirms that the issue is specifically large-model
physical readiness.

## Correctness

All variants preserved deterministic generation across both model sizes.
Step-26 remains the exact tensor/logit authority:

```text
707/707 tensor byte parity: PASS
logit max_abs_diff: 0
preparation generation parity: PASS
payload duplication: 0
```

Preparation is read-only and does not alter tensor bytes or inference
semantics.

## Attribution classification

The evidence supports multiple categories:

```text
A — controlled explicit preparation substantially improves uncached total TTFUC
B — preparation shifts faults from FIRST_EVAL into PREPARATION
F — sequential and layer-order preparation are effectively equivalent
D — advisory prefetch is not effective as completed residency
G — warm results are cache-sensitive and must not be mixed with uncached results
```

## Wave-processing prerequisite

Partial support only:

```text
exact/stable layer spans: yes
cheap runtime-local derivation: yes
fault storm controllable: yes
per-layer preparation measurable: yes
per-layer compute timing: unavailable
executed compute/I/O overlap: no
```

Therefore the Track-B decision is **NO/NOT YET**. The result justifies preserving
the preparation evidence, but not implementing a new asynchronous wave runtime.
A future experiment would require an explicit llama layer execution seam and
measured per-layer compute timing; it must not hack graph execution.

## Constraints preserved

```text
wire changes: 0
BaseShift changes: 0
Nano changes: 0
Nested-vBuf changes: 0
LayerView production API: 0
RuntimeChunk production architecture: 0
llama graph/kernel changes: 0
GPU offload: 0
```

Evidence is under:

```text
benchmark-results/vbuf-ml-step28-layer-prefetch/
```
