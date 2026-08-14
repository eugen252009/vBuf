# Step 27 — controlled loader scaling and first-use attribution

Status: **complete as an attribution benchmark; no optimization implemented**.

Step 27 compares freshly collected CPU-only controls for:

```text
Qwen3-0.6B-Q8_0 × GGUF/vBuf
Qwen3-32B-Q8_0 × GGUF/vBuf
```

The same pinned llama.cpp build, prompt, context, threads, batch size, and
deterministic argmax generation were used for all cases.

## Configuration

```text
llama.cpp: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
backend: CPU
n_gpu_layers: 0
threads: 2
batch: 512
context: 512
prompt: Hello world
prompt tokens: 2
short generation: 8 tokens
warm samples: 10 per case
uncached approximation: 3 per case
```

The uncached approximation uses whole-file `POSIX_FADV_DONTNEED` advisory
eviction. It is not a guaranteed cold-cache experiment.

Run order is deterministic and alternates GGUF/vBuf within each model-size
round. GPU was not involved.

## Artifact provenance

All four qualified artifacts were hash-verified before execution. Provenance
is recorded in:

```text
benchmark-results/vbuf-ml-step27-loader-scaling/artifact-provenance.json
```

## Warm median results

Times are milliseconds. `TTFUC` is process start through completion of the
first prompt forward pass. `TTFT` is the first generated token boundary.

| Model | Format | MODEL_READY | FIRST_EVAL | TTFUC | TTFT | SECOND_EVAL |
|---|---|---:|---:|---:|---:|---:|
| 0.6B | GGUF | 557.4 | 33.4 | 611.5 | 611.5 | 32.4 |
| 0.6B | vBuf | 217.9 | 505.2 | 801.0 | 801.0 | 32.3 |
| 32B | GGUF | 20,002.8 | 1,754.2 | 21,607.7 | 21,607.7 | 1,070.6 |
| 32B | vBuf | 261.6 | 35,840.3 | 36,144.8 | 36,144.8 | 1,073.8 |

Dispersion, raw samples, CPU time, faults, I/O counters, memory, and
uncached-approximation results are in the evidence directory.

## The Step-26 readiness delta

The approximate Step-26 `18.486 s` GGUF versus `0.461 s` vBuf control is not
reproduced numerically under this fresh harness, but its qualitative result is
strongly reproduced:

```text
fresh 32B GGUF MODEL_READY median: ≈20.003 s
fresh 32B vBuf MODEL_READY median: ≈0.262 s
```

The warm readiness delta is approximately `19.741 s`, or `98.7%` relative to
GGUF.

## First-use attribution

The readiness advantage does **not** survive first useful compute:

```text
32B GGUF TTFUC: ≈21.608 s
32B vBuf TTFUC: ≈36.145 s
```

The vBuf path shifts substantial weight access into first evaluation. The
warm median evidence shows:

```text
32B GGUF major faults at MODEL_READY: 0
32B vBuf major faults at MODEL_READY: 347
32B vBuf major faults by FIRST_EVAL: 45,953

32B GGUF RSS at MODEL_READY: ≈33.9 GB
32B vBuf RSS at MODEL_READY: ≈84 MB
32B vBuf RSS after FIRST_EVAL: ≈33.3 GB
```

The corresponding warm read-byte counters also move from model construction
to first evaluation for vBuf. `read_bytes` is kernel accounting and is not
claimed to equal physical storage reads.

## Repeated evaluation

After the first generation, a fresh context was created without rebuilding the
model and the same prompt was evaluated again:

```text
32B GGUF SECOND_EVAL: ≈1.071 s
32B vBuf SECOND_EVAL: ≈1.074 s
```

This supports common steady-state execution after residency has been
established.

## Scaling observations

Descriptive warm large/small ratios:

```text
GGUF MODEL_READY: 35.9×
GGUF FIRST_EVAL: 52.6×
GGUF TTFUC: 35.3×
vBuf MODEL_READY: 1.2×
vBuf FIRST_EVAL: 70.9×
vBuf TTFUC: 45.1×
```

These are not linear-scaling claims. They are measurements from this fixed
CPU configuration and cache methodology.

## Correctness

All warm samples preserved identical deterministic generation per model and
format. Step 26's exact tensor parity and zero logit-difference qualification
remain the correctness authority; Step 27 did not mutate artifacts or repeat
full logit comparison for every sample.

## Attribution conclusion

Classification: **C — mixture of genuine readiness/construction reduction and
deferred first-touch work**.

The current direct vBuf path establishes a very small resident semantic/runtime
state before `MODEL_READY`, while GGUF construction materializes/touches much
more weight state before that boundary. vBuf therefore has a real readiness
advantage, but first-use attribution shows that much of the payload/residency
cost is deferred to first evaluation. Once resident, second evaluation times
are effectively equal.

This benchmark does not establish a Step-27 scaling performance claim and does
not justify prefetch, residency, Nano, LayerView, RuntimeChunk, or other
runtime optimization work.

## Limitations

```text
true global cold-cache state was not established
peak RSS/PSS instrumentation was unavailable
swap was nonzero on the host
physical disk-read attribution is unavailable
GPU was not tested
```

The swap state and variability, especially on vBuf first-use runs, are retained
in raw evidence and limit strong I/O causal claims.

## Evidence

```text
benchmark-results/vbuf-ml-step27-loader-scaling/
```

No v0.6 wire, vBuf-ML wire, placement, Nano, or production runtime architecture
changes were made.
