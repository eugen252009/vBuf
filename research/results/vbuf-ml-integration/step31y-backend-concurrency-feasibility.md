# Step 31Y Backend Concurrency Feasibility Qualification

Status: **X86_64 ISOLATED-SESSION OVERLAP QUALIFIED; SHARED-SESSION AND
SHARED-RESIDENT-WEIGHT CONCURRENCY NOT QUALIFIED; PRODUCTION SERIAL POLICY
PRESERVED**.

Target: Linux x86_64 workstation. Orange Pi, Pixel, Android, ADB, APK,
external hardware, and RISC-V execution were not required or used.

The measurement used available GGML checkout
`a97123e497968f3440264c0464a7adc7c999c027`. The canonical pinned GGML
revision remains unavailable. This step does not alter the canonical pin.

The production `vbuf_compat_server` remains unchanged in policy:

```text
MAX_ACTIVE_GENERATIONS = 1
```

No server slots, batching, interleaving, priority scheduling, dynamic thread
tuning, backend changes, KV reuse, residency redesign, or format changes were
introduced.

## Qualification Boundary

The probe created two independent `VbufGenerationSession` objects. Each session
owned its own lazy source, HTTP connection, materializer, residency store,
request counters, active-generation counters, leases, KV state, and backend
execution contexts. The two sessions were warmed separately and then run with
the same A/B workload either serially or behind a synchronized start gate.

This is a private-session feasibility experiment. It does not test concurrent
calls on one session. One session is not safe for concurrent calls because its
request counters, diagnostics, source-fault configuration, residency maps,
materializer wrapper maps, and active-generation state are mutable and not
generally synchronized.

The probe also does not test shared resident payloads. The two sessions have
separate residency and materializer ownership, so equal resident-byte results
do not establish safe concurrent readers of one resident store. Shared resident
weights remain **NOT_QUALIFIED**.

## Architecture Audit

| Question | Classification | Basis |
|---|---|---|
| request semantic independence | `YES, DERIVED` | prompt, output, activation, and KV vectors are request-local |
| same-session concurrent execution | `NOT_SAFE` | mutable session/source/residency/materializer state is not generally synchronized |
| separate-session concurrent execution | `EMPIRICALLY PASS, NOT FORMALLY PROVEN` | three synchronized pairs completed with deterministic token parity and no runtime errors |
| same GGML context concurrent execution | `NOT_SAFE` | current execution creates and owns contexts locally rather than exposing a shared-context contract |
| separate GGML contexts | `HOST-OBSERVED PASS` | isolated sessions overlapped backend work without observed failure |
| shared immutable resident payload readers | `NOT_QUALIFIED` | current session API does not inject or synchronize one shared residency/materializer |
| concurrent throughput benefit | `MEASURED YES` | aggregate pair makespan speedup was `1.751x` |
| production scheduling recommendation | `NO-GO FOR NOW` | memory and thread costs are material, and shared-runtime ownership is unresolved |

The empirical pass is not a general thread-safety proof. No ThreadSanitizer
build was available for the complete GGML/vBuf-ML stack in this qualification,
and the test intentionally avoided shared mutable runtime objects.

## Measurements

The workload used two blocks, `NormalInference`, prompts `Say hi` and
`Count to one`, eight generated tokens, one warmup per private session, and
three measured A/B pairs in each fresh process. The prompt token hashes were
`232a1eb1cb807d1b` and `b0b030c4617b9bde`. All serial and concurrent runs
returned the same per-prompt token hashes:

```text
Say hi       -> 9b2aafabc894a70e
Count to one -> 1c9544a3bd9109be
```

| Metric | Serial A/B | Concurrent A/B | Classification |
|---|---:|---:|---|
| mean pair makespan | `8.798 s` | `5.025 s` | concurrent `1.751x` faster |
| total three-pair makespan | `26.393 s` | `15.076 s` | `42.9%` lower concurrent time |
| mean individual runtime | `4.399 s` | `4.958 s` | concurrent latency `1.127x` higher |
| total measured runtime overlap | `0 ms` | `14.675 s` | real backend overlap observed |
| peak RSS | `704288 KiB` | `943524 KiB` | `+239236 KiB` concurrent |
| peak PSS | `694345 KiB` | `933581 KiB` | `+239236 KiB` concurrent |
| peak process threads | `5` | `11` | `+6` concurrent |

Each private session ended with zero active leases, zero active inflight bytes,
zero active generations, and deterministic source/residency accounting. The
per-session resident-byte results were unchanged between serial and concurrent
runs, approximately `239-249 MiB` per session. Each session used one source
connection; source request and byte totals matched between the paired modes.

The combined derived KV estimate for the two measured prompts was
`3031040` bytes. This is a lower-bound accounting estimate for the current
two-block runtime-state shape, not an exclusive backend workspace measurement.
Activation, allocator, graph, and backend workspace peaks remain
`UNINSTRUMENTED`.

The concurrent run therefore improves aggregate throughput on this host, but
individual request latency worsens and process memory/thread pressure rises
substantially. The result is evidence for a possible future isolated-runtime
scheduler experiment, not authorization to enable production concurrency.

## Reproduction

The research-only native probe is
`integrations/ggml/tools/vbuf_step31y_backend_concurrency.cpp`, built as
`vbuf_step31y_backend_concurrency`. The coordinator and invariant checks are
in `integrations/ggml/tests/vbuf_step31y_backend_concurrency_test.py`.

The raw qualification evidence is:

```text
/tmp/opencode/vbuf-step31y-backend-concurrency-final.json
```

The probe reports overlap, output-token parity, per-session source/residency/
lease counters, process RSS/PSS, and thread high-water marks. It is not a
production server path and is not registered as a routine CTest because it
requires the real semantic model and HTTP range source.
