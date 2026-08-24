# Step 31W Control-Plane / Inference-Plane Separation

Status: **X86_64 IMPLEMENTED, PHYSICALLY QUALIFIED, COMMITTED**.

Target: Linux x86_64 workstation. Orange Pi, Pixel, Android, ADB, APK, and
physical RISC-V execution were not required or used.

The qualification used available ggml checkout
`a97123e497968f3440264c0464a7adc7c999c027`. The canonical pinned ggml
revision remains unavailable. This change does not modify the canonical pin,
GGML execution, vBuf format, vBuf-ML materialization, residency, or KV
semantics.

## Control-Plane Dependency Audit

`GET /health` returns a fixed JSON body. It reads no runtime object, session,
tokenizer, source, materializer, residency, KV, or backend state.

`GET /v1/models` constructs the compatibility response from
`ServerRuntime::config.model_alias`. `ServerConfig` is initialized before the
listener starts and is not mutated by requests. The model catalog is therefore
immutable for the server lifetime.

`OPTIONS` also reads no inference state. Generation endpoints remain the only
paths that read or mutate the tokenizer request path, shared generation
session, source/materializer state, residency state, leases, KV, or GGML
execution state.

| State | Classification after Step 31W |
|---|---|
| `ServerConfig`, including model alias | `IMMUTABLE_SHARED` |
| tokenizer tables and startup model metadata | `IMMUTABLE_SHARED` |
| `/health` response | `READ_ONLY_STATIC` |
| `/v1/models` response identity | `READ_ONLY_IMMUTABLE` |
| generation session, source, residency, leases, KV, backend | `INFERENCE_OWNER_ONLY` |
| request body, socket, output, cancellation state | `REQUEST_LOCAL` |
| diagnostic counters | `ATOMIC` |

## Dispatch Structure

The listener now accepts connections and dispatches each accepted connection to
one tracked worker. The worker owns the request buffer, response socket, SSE
state, cancellation state, and diagnostics for that connection. Completed
workers are joined and reaped by the listener. At most eight connection workers
exist. If that bound is saturated, the newly accepted connection receives a
bounded `503 server_busy` response; no unbounded detached thread creation is
allowed.

```text
listener
  accept connection in accepted order
  -> tracked bounded connection worker
       -> /health or /v1/models: immutable read-only response
       -> generation: fair admission gate -> one VbufGenerationSession::run
```

The inference admission gate records accepted-connection order, lets control
requests bypass it, and grants at most one generation lease. This preserves
FIFO generation order without introducing an application generation queue.
The kernel listen backlog remains `8`; the pre-existing queued-request
cancellation gap therefore remains documented and intentionally unfixed.

Shutdown closes all tracked worker sockets before joining workers. A shutdown
request reaches the existing generation cancellation callback, while blocked
HTTP reads are interrupted by socket shutdown.

## Physical Qualification

Available-checkout x86_64 evidence:

| Scenario | Result |
|---|---|
| health during eight-token generation | `PASS`, `0.712 ms` |
| models during eight-token generation | `PASS`, `0.739 ms` |
| 20 health requests during generation | `PASS`, min `0.571 ms`, p50 `1.579 ms`, max `5.557 ms` |
| 20 models requests during generation | `PASS`, min `0.655 ms`, p50 `1.375 ms`, max `1.614 ms` |
| control traffic before generation completion | `PASS` |
| generation output with and without control traffic | `PASS`, normalized content/finish/usage parity |
| SSE plus health/models | `PASS`, valid chunks and `[DONE]` |
| active cancellation, control request, recovery generation | `PASS` |
| source failure, control requests, same-process recovery | `PASS` |
| two generation runtime intervals | `PASS`, no overlap, max active generations `1` |
| faulted generation with waiting follower | `PASS` |
| clean shutdown during active generation | `PASS` |

The previous Step 31V behavior made health and models wait for the complete
active generation. Step 31W does not claim an inference speedup. It changes
operability latency from full-generation blocking to sub-millisecond local
control responses in the measured run.

The matched Step 31U-style overhead rerun also remained small:

```text
direct median: 118.347 ms
server median: 119.192 ms
overhead:       0.845 ms / 0.714%
classification: SMALL
```

These measurements use the available checkout and are not a pinned-GGML
qualification claim.

## Preserved Invariants

The following were not implemented or changed:

- parallel inference;
- cooperative decode interleaving;
- continuous batching;
- request slots;
- multiple concurrent GGML contexts;
- backend thread tuning;
- residency semantics;
- KV semantics;
- application inference queue;
- queued-request cancellation behavior.

The existing Step 31U/31V lifecycle, HTTP, Python OpenAI client, source-fault,
active-cancellation, FIFO, and serial-generation qualifications were rerun or
covered by the Step 31W harness. The remaining next experiment requires
separate authorization.

Evidence harness: `integrations/ggml/tests/vbuf_step31w_control_plane_test.py`.
