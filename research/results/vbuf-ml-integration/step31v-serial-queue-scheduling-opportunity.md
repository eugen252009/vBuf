# Step 31V SERIAL_QUEUE Scheduling Opportunity Audit

Status: **HOST-CONTRACT-AUDITED; X86_64_QUEUE BEHAVIOR MEASURED; CONTROL-PLANE
SEPARATION JUSTIFIED; INFERENCE CONCURRENCY NOT JUSTIFIED YET**.

Target: Linux x86_64 workstation. Orange Pi, Pixel, Android, ADB, APK, and
physical RISC-V execution were not required or used.

The qualification used the available ggml checkout at
`a97123e497968f3440264c0464a7adc7c999c027`. The canonical pinned ggml
revision remains unavailable. This audit does not change the vBuf runtime
contract or claim pinned-backend qualification.

Evidence classes used below are `CODE_AUDITED`, `HOST_CONTRACT_QUALIFIED`,
`X86_64_PHYSICALLY_MEASURED`, `DERIVED`, `HYPOTHESIS`, `UNINSTRUMENTED`, and
`NOT_IMPLEMENTED`.

## Serialization Boundary

The server's main loop calls `accept()`, then calls `handle_request()` directly
on the accepted socket, then closes the socket before accepting the next one.
There is no application queue and no request worker thread. The `handle_request`
call covers request reading, protocol validation, prompt parsing, tokenization,
generation, response construction, response/SSE writes, request cleanup, and
socket close.

```text
accept A
  read HTTP A
  parse/validate A
  build prompt A
  tokenize A
  VbufGenerationSession::run(A)
    source readiness / materialization
    prefill
    decode steps
    backend graph work
  serialize response or write SSE A
  release request leases and cleanup A
  close A
accept B
  ...
```

The B connection can reach the kernel listen backlog while A is active, but B
is not accepted or parsed by the application until A returns. `listen()` uses
backlog `8`; this is an operating-system pending-connection bound, not a
vBuf-ML queue contract. Behavior beyond that bound is OS/network dependent.

| Phase | Serialized by current main loop |
|---|---|
| HTTP parse/read | YES |
| prompt construction | YES |
| tokenization | YES |
| source readiness | YES |
| materialization | YES at request-dispatch level |
| prefill | YES |
| decode | YES |
| backend compute | YES |
| response serialization | YES |
| SSE writes | YES |
| request cleanup | YES |
| `/health` | YES, blocked behind active inference |
| `/v1/models` | YES, blocked behind active inference |

The new bounded timeline fields are `accepted_ns`, `dispatch_start_ns`,
`runtime_start_ns`, `first_token_ns`, `runtime_end_ns`, and request end timing.
Control endpoints emit bounded `vbuf_control_request` records. The first
generated token is currently included in the final prompt-position work;
`decode_ns` measures subsequent autoregressive positions and is zero for a
one-token request. This is an actual runtime boundary, not invented phase
precision.

## Shared State Audit

| State | Classification | Evidence and consequence |
|---|---|---|
| `ServerConfig` | IMMUTABLE_SHARED | Initialized before serving; no request mutation. |
| `VbufTokenizer` tables | IMMUTABLE_SHARED | Built at startup and read by requests. |
| session metadata and layer plans | IMMUTABLE_SHARED | Loaded once and not changed by generation. |
| `HttpRangeSource` socket | SERIALIZED_SHARED | `socket_mutex_` protects the persistent HTTP socket and metrics. |
| source fault configuration | SERIALIZED_SHARED | Request configuration mutates the shared wrapper. |
| `TensorResidencyStore` | SERIALIZED_SHARED | Lease/entry maps and counters have no mutex. |
| `ResidentTensorMaterializer` | SERIALIZED_SHARED | `requests_` and `known_tensors_` have no mutex. |
| local materializer workers | THREAD_SAFE_SHARED INTERNALLY | Worker state is mutex-protected, but the wrapper and shared residency are not concurrent-qualified. |
| generation counters | SERIALIZED_SHARED | Session request and active-generation counters are non-atomic. |
| actual/reference KV state | REQUEST_LOCAL | `run()` allocates separate K/V vectors for every request. |
| activations, logits, outputs | REQUEST_LOCAL | Local vectors and call-owned execution state. |
| leases | REQUEST_LOCAL SEMANTICS, SERIALIZED_SHARED ACCOUNTING | Lease ownership is request-scoped, but shared accounting maps are not concurrent-safe. |
| cancellation callback | REQUEST_LOCAL | Captures one request's disconnect state; non-stream queued disconnect is not observed before runtime entry. |
| HTTP stream state | REQUEST_LOCAL | File descriptor, output, and stream counters belong to the request handler. |

Any concurrent use of one `VbufGenerationSession` is therefore blocked by
shared mutable runtime state, independent of the semantic independence of the
K/V vectors.

## Request Independence

`PREFILL_SEMANTICALLY_PARALLELIZABLE = YES` is a derived result: request input,
K/V slots, activations, output, and token callbacks are local to `run()`, and
resident model payloads are logically immutable while leased. This is not a
claim that the current shared materializer/residency implementation can support
it.

`DECODE_SEMANTICALLY_INTERLEAVABLE = YES` is also derived for independent
requests because each request owns its K/V state and token/output state. The
current generation call is monolithic and not resumable at a scheduler boundary,
so no interleaving is implemented.

`PREFILL_DECODE_OVERLAP_SEMANTICALLY_SAFE = YES` across independent requests,
subject to separate request state and immutable weight reads. The current
shared residency/materializer accounting makes implementation unsafe without a
runtime ownership change. Same-request prefill/decode ordering remains
semantic and is not relaxed.

## Backend And Thread Audit

The current routed-expert path creates a local `ggml_context`, local CPU
backend, local buffers, and local graph for each indexed expert-bank
invocation. The available ggml CPU backend defaults to
`GGML_DEFAULT_N_THREADS = 4`; with no supplied threadpool it creates a
disposable threadpool for graph computation. `GGML_OPENMP=OFF` in the audited
build. ggml CPU initialization uses its own critical section, but this does not
establish a general concurrent graph-execution contract for the whole vBuf
runtime.

| Question | Classification |
|---|---|
| same backend context concurrently | NOT_SAFE |
| separate backend contexts concurrently | UNPROVEN |
| two complete current vBuf sessions concurrently | NOT_SAFE due shared session state |
| backend graph execution in current server | SERIALIZED by server boundary |

Host CPU inventory was 16 logical CPUs, 8 physical cores, one socket. One ggml
graph uses four CPU backend threads, plus the caller and temporary materializer
worker activity. Two independent graph executions would structurally create at
least eight backend workers plus callers and materialization workers. This is
not a measured concurrent run because the shared runtime is not safe; the
thread-oversubscription risk is **LIKELY**, especially when source/materializer
workers overlap.

## Residency And Memory

The measured warm shared resident model working set was `231130368` bytes. The
residency store prevents eviction of an entry with an active lease, but its
entry and lease maps are not synchronized. Therefore
`SHARED_RESIDENT_WEIGHTS_SAFE = SERIAL-ONLY; CONCURRENT ACCESS UNKNOWN`.

The session allocates four K/V slots per layer: actual K/V and reference K/V.
For two layers, the payload estimate is:

```text
4 * (16*192 + 16*128) floats * 4 bytes * positions
```

| Workload | Positions | Per-request K/V payload estimate |
|---|---:|---:|
| one token | `11` | `901120` bytes |
| eight tokens | `18` | `1474560` bytes |

Derived two-request K/V estimates are `1802240` bytes for short+short,
`2375680` bytes for long+short, and `2949120` bytes for long+long. Adding the
measured shared resident working set gives derived lower bounds of
`232932608`, `233506048`, and `234079488` bytes respectively. Activation,
temporary vector, allocator, and ggml backend workspace peaks are not exposed
as exclusive per-request measurements; `CONCURRENT_PEAK_ESTIMATE` including
those components is **UNINSTRUMENTED**. Parallel execution has a material
memory risk even though the derived K/V component is comparatively small.

## Queue Measurements

The harness warmed each fresh server with one request, then launched A and B
nearly simultaneously. The workload used two blocks, `NormalInference`, the
same deterministic prompt, and the available ggml checkout. Times below are
x86_64 host measurements from one qualification run; they are not production
capacity claims.

| Pair | A queue wait | B queue wait | A service | B service | Makespan |
|---|---:|---:|---:|---:|---:|
| short + short, one token | `1.374 ms` | `2929.339 ms` | `2948.405 ms` | `2958.697 ms` | `5909.868 ms` |
| long + short, eight + one tokens | `2.049 ms` | `9227.428 ms` | `9247.027 ms` | `3661.515 ms` | `12912.157 ms` |
| long + long, eight + eight tokens | `0.334 ms` | `9749.852 ms` | `9769.450 ms` | `10650.617 ms` | `20422.058 ms` |

The streaming short+short control also showed serialized B dispatch: B queue
wait was `3966.711 ms` and the pair makespan was `8856.567 ms`. `LONG A +
SHORT B` proves head-of-line blocking as scheduling latency, not compute
inefficiency: B's useful runtime did not begin until A completed.

Measured serial throughput for these bounded pairs was approximately `0.335
requests/s` for short+short, `0.149 requests/s` for long+short, and `0.102
requests/s` for long+long. These values include this host's current runtime and
source behavior and are not generalized serving capacity.

## Control Plane And Queue Lifecycle

While a long inference was active, `/health` and `/v1/models` both returned
HTTP `200` only after the inference completed. The result is
`HEALTH_DURING_GENERATION = BLOCKED_BY_SERIAL_DISPATCH` and
`MODELS_DURING_GENERATION = BLOCKED_BY_SERIAL_DISPATCH`.

The lifecycle harness used one persistent server process per scenario:

| Scenario | Result |
|---|---|
| queued request disconnected before dispatch | **GAP**; request still entered runtime after A |
| active streaming request cancelled with follower queued | PASS; follower completed |
| active fault-injected request with follower queued | PASS; follower completed |
| four queued followers | FIFO execution order |

The queued-cancellation gap is caused by the current architecture: B's full
HTTP request can remain in the kernel backlog, then be read and dispatched after
A without a request-local cancellation token being established. The server
does not currently promise cancellation of a request before runtime entry.

## Scheduling Strategies

| Strategy | Semantic feasibility | Backend feasibility | Expected effect | Complexity/risk |
|---|---|---|---|---|
| parallel request execution | POSSIBLE with isolated sessions and synchronized shared services | UNPROVEN for separate contexts; current shared runtime NOT_SAFE | Throughput MAY improve; B tail latency MAY improve | High memory, thread, source, lease, and ownership risk |
| cooperative interleaving | DERIVED SEMANTICALLY SAFE for independent K/V state | Sequential backend execution is feasible in principle | Likely improves short-request tail latency only after a preemption boundary | Requires resumable generation and queue/cancellation redesign |
| continuous/cross-request batching | PARTIAL representation exists for qualification prefill only | Current generation path does not provide decode row batching or backend qualification | Utilization benefit UNKNOWN | High attention-mask, KV routing, TopK, reduction-order, and response-routing risk |

No strategy is implemented. No slots, continuous batching, request
interleaving, thread tuning, or parallel inference were added.

## Decision

`SCHEDULING_OPPORTUNITY_CLASSIFICATION = CONTROL_PLANE_SEPARATION_JUSTIFIED`.

The evidence is sufficient to show that inference SERIAL_QUEUE has material
head-of-line cost, but the current vBuf runtime is not safe for concurrent use
of its shared session/residency/materializer state. The smallest separately
authorized experiment is to let immutable `/health` and `/v1/models` handling
progress independently while keeping inference serialized. It directly fixes a
measured operability block without copying llama-server slots or making an
unqualified inference-concurrency claim. Inference scheduling remains
`MORE_MEASUREMENT_REQUIRED` after that control-plane experiment, with
cooperative interleaving a possible later candidate if a resumable runtime
boundary is designed and qualified.

## Reproduction

The bounded qualification harness is
`integrations/ggml/tests/vbuf_step31v_serial_queue_test.py`. It emits the raw
JSON evidence used for this report and covers the pair measurements, streaming
control, health/models blocking, FIFO followers, queued cancellation, active
cancellation with follower, and fault follower. The raw local evidence was
`/tmp/opencode/vbuf-step31v-serial-queue-final.json` and is not a repository
artifact.
