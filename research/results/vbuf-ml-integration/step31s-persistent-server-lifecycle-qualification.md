# Step 31S Persistent vBuf Server Lifecycle Qualification

Status: **HOST-CONTRACT-QUALIFIED; CONTROLLED-SOURCE-FAILURE-RECOVERY-QUALIFIED; BOUNDED CACHE GROWTH OBSERVED**.

Target: Linux x86_64 workstation. No Android, ADB, Orange Pi, or physical
RISC-V evidence is implied.

The qualification used one `vbuf_compat_server` process for the persistent
request sequence. The server continued to own the native vBuf-ML runtime,
tokenizer, source, materialization, residency, and GGML execution boundary.
llama-server was used only for client/wire regression and was not used by the
vBuf server as a runtime, proxy, or loader.

## Lifetime Ownership Map

```text
server process
└─ ServerRuntime
   ├─ immutable ServerConfig
   ├─ VbufTokenizer
   │  └─ vBuf metadata consumer handle and tokenizer metadata
   └─ VbufGenerationSession
      ├─ semantic Metadata and immutable LayerPlan values
      ├─ persistent HttpRangeSource
      ├─ persistent TensorResidencyStore
      └─ persistent ResidentTensorMaterializer

HTTP request
└─ request body, parsed messages, prompt tokens, request ID, output stream,
   cancellation/disconnect state, response DTO, request timings

generation call
└─ model lease, actual/reference KV vectors, activation/logit temporaries,
   materializer request leases, generation callback state
```

Persistent state is model structure, tokenizer metadata, source connection,
bounded payload residency, and immutable execution plans. Per-request state is
prompt text/tokens, KV vectors, generation counters, output, cancellation,
temporary leases, and diagnostics. `VbufGenerationSession::run()` allocates KV
vectors locally for every call. The session's active-generation counter returns
to zero through an RAII guard even when generation reports an error.

The server remains intentionally serialized. The listen backlog queues a
second client while one request is executing; it does not permit concurrent
mutation of the session's source/residency/materializer state.

## Steady-State Invariants

- `REQUEST_STATE_DOES_NOT_LEAK`: PASS for the qualified sequence.
- `KV_STATE_DOES_NOT_CROSS_REQUESTS`: PASS through deterministic A/B/A parity.
- `CANCELLED_REQUEST_RELEASES_STATE`: PASS; cancellation log has zero leases,
  inflight bytes, active generations, streams, and cancellation handles after
  the request.
- `COMPLETED_REQUEST_RELEASES_STATE`: PASS for every logged request.
- `RESIDENCY_MAY_PERSIST`: intentional and bounded by `268435456` bytes.
- `SOURCE_CACHE_MAY_PERSIST`: intentional; warm requests can reuse resident
  model payloads and the persistent HTTP source connection.
- `MODEL_METADATA_MAY_PERSIST`: intentional and immutable after startup.
- `TEMPORARY_LEASES_RETURN_TO_ZERO`: PASS.
- `ACTIVE_GENERATION_COUNT_RETURNS_TO_ZERO`: PASS.
- `SERVER_REMAINS_HEALTHY_AFTER_FAILURE`: PASS for malformed JSON, unknown
  model, unsupported option, and invalid token bound.
- `SERVER_REMAINS_HEALTHY_AFTER_CANCELLATION`: PASS.

Step 31S found and fixed a lifecycle memory issue in the diagnostic path:
persistent residency traces, materializer event history, and HTTP range timing
and unique-range histories were unbounded. Scalar counters remain persistent,
while per-request diagnostic histories are trimmed after each generation.
This does not clear model payload residency or request-independent counters.

## Persistent Request Corpus

One server process handled 20 sequential bounded requests in this deterministic
corpus:

```text
A = "Say hi"
B = "Count to one"
C = "Name a color"
sequence = A, B, A, C, B, A, repeated to 20 requests
```

All 20 requests returned HTTP `200`, one generated token, and non-empty output.
The deterministic A outputs were all `行车`:

```text
A0 = 行车
A1 = 行车
A0_A1_PARITY = PASS
```

The same process then handled malformed/failed requests, OpenAI client
requests, repeated streaming, a real disconnect, a cancellation follow-up,
health during generation, two simultaneous clients, and a five-request steady
tail. No server restart occurred between these operations.

## Streaming And Cancellation

Three raw HTTP SSE requests were parsed by the qualification harness. Each had
one request ID, valid `data:` records, one terminal `[DONE]`, and no content
crossed into the next stream. One additional streaming request passed through
the Python OpenAI client.

The real cancellation test used a socket client, consumed two SSE data chunks,
and then closed the connection. The server recorded `cancelled=yes`; the next
request completed normally. The cancellation sequence was:

```text
stream request -> client disconnect -> cancelled generation
                  -> normal follow-up -> normal subsequent requests
```

Health during a long generation returned HTTP `200`, but only after the active
generation completed. This is the documented `SERIAL_QUEUE` limitation:
health is accepted by the listen backlog but dispatch is blocked by the
single-threaded request loop. Step 31S does not redesign that policy.

## Failure And Concurrency Recovery

The same process received and recovered from:

| Failure | Result | Follow-up generation |
|---|---|---|
| missing/invalid JSON body | `400` | PASS |
| unknown model | `400` | PASS |
| unsupported `temperature` | `400` | PASS |
| `max_tokens` above bound | `400` | PASS |

The follow-up qualification added one explicit source failure before the cold
request. The same process returned `500` with the controlled materialization
failure, then recovered on the next request with output `行车`. The captured
server log contained `source=controlled-failure`, and the recovery request
returned HTTP `200` with zero active leases, inflight bytes, generations,
streams, and cancellation handles.

The injection is qualification-only and request-count based. The default is
zero, so normal serving behavior is unchanged; `--source-failure-requests 1`
fails one source read and then permits subsequent reads. This qualifies
controlled source-failure recovery, not arbitrary allocator, transport, or
backend faults.

Two simultaneous client threads both completed correctly. The policy is
`SERIAL_QUEUE`; no parallel generation state mutation was observed or allowed.

## Cold And Warm Residency Behavior

Configuration:

```text
blocks: 2
residency capacity: 268435456 bytes (256 MiB)
max new tokens: 8
runtime mode: NormalInference
```

Representative first cold request:

- source bytes fetched: `255493376`
- materialized bytes: `83569664`
- resident bytes after: `255493376`
- peak vBuf residency: `255493376`
- prefill: `1346.896403 ms`

The steady tail reached three successive requests with:

- source bytes fetched: `0`
- materialized bytes: `0`
- resident bytes after: `257858816`
- peak vBuf residency: `257858816`

Warm source/residency reuse is therefore proven for the resident working set.
Other prompts can reacquire ranges when the bounded residency policy evicts
them; this is expected cache behavior, not request-state leakage. Across the
qualification log, peak residency remained below the configured `268435456`
byte cap and all post-request active lease/inflight counters were zero.

## Follow-Up Forty-Request Soak

The follow-up used the same process and configuration with
`--source-failure-requests 1`, then ran a 40-request A/B/C corpus before the
existing streaming, cancellation, concurrency, and five-request steady-tail
checks. The faulted request, recovery request, and all 40 corpus requests
completed without a server restart. The resulting evidence was:

| Field | Result |
|---|---:|
| corpus requests | `40` |
| controlled source failures | `1` |
| failure response | HTTP `500` |
| immediate recovery | HTTP `200`, output `行车` |
| total logged request operations | `60` |
| peak vBuf residency | `268190976` bytes |
| configured residency cap | `268435456` bytes |
| RSS start / warm-up / tail | `36624 / 304244 / 319148 KiB` |
| final steady-tail source bytes | `0` |
| final steady-tail resident bytes | `257858816` |

The final three steady-tail requests fetched zero source bytes. The first two
tail requests still reacquired ranges while the bounded working set converged;
this is expected residency behavior, not request-state leakage.

## Memory Behavior

The harness sampled `/proc/<pid>/status` RSS:

| Sample | RSS |
|---|---:|
| process start | `36704 KiB` |
| after initial warm-up | `300548 KiB` |
| after five-request steady tail | `324368 KiB` |

Classification: **BOUNDED_CACHE_GROWTH**.

The initial increase includes expected model/payload residency and allocator
warm-up. After diagnostic-history trimming, the five-request tail showed only a
small additional allocator/cache high-water increase while vBuf residency
remained bounded. This is not evidence of a monotonic request-history leak.
A much longer soak and allocator-specific profiling remain useful future work;
they are not required to claim an unbounded leak from this result.

## Full 26-Layer Normal Inference

One persistent server process was started with `--blocks 26`, handled one
canonical request, and shut down cleanly:

- mode: `NormalInference`
- HTTP status: `200`
- output: `**,`
- prompt tokens: `10`
- generated tokens: `1`
- source bytes: `6327692864`
- materialized bytes: `5998118912`
- peak vBuf residency: `268412928` bytes
- runtime elapsed: `101521.902818 ms`
- server total request: `101522.312509 ms`
- external HTTP wall time: `101522.664 ms`
- prefill: `101505.706211 ms`
- decode: `0 ms`
- response serialization: `0.018220 ms`
- clean process exit: return code `0`

The source traffic exceeds the `5,639,819,878` byte payload because the
bounded policy reacquires ranges under pressure. The model was not resident in
full at once.

## Timing Attribution

The server now records request parsing, prompt parsing, tokenization, response
serialization, prefill, decode, total runtime, and total request timing.
Source materialization has no trustworthy exclusive boundary in the current
execution seam and is explicitly recorded as `UNINSTRUMENTED` rather than
invented.

For the full 26-layer request:

| Field | Value |
|---|---:|
| HTTP parse | `0.028610 ms` |
| prompt build | `0.001900 ms` |
| tokenize | `0.069050 ms` |
| materialization | `UNINSTRUMENTED` |
| prefill | `101505.706211 ms` |
| decode | `0 ms` |
| response serialization | `0.018220 ms` |
| runtime elapsed | `101521.902818 ms` |
| total request | `101522.312509 ms` |

A matched direct-runtime control using the same model, prompt, mode, generation
count, and backend/thread settings was not available. Direct/server overhead is
therefore `NOT_MEASURED`; the small HTTP attribution values above must not be
reported as a direct-runtime overhead comparison.

## Startup, Shutdown, And Client Regression

- Health-ready time from process launch: approximately `410 ms`.
- Semantic model/runtime-ready boundary: `UNINSTRUMENTED`; construction occurs
  before the listening line and has no separate timestamp.
- Clean shutdown with no active request: PASS, return code `0`, explicit clean
  stop log.
- Shutdown during active generation: PASS; `SIGTERM` caused the generation to
  cancel, returned zero active leases/generation/inflight counters, and exited
  with return code `0`.
- Startup and full-layer teardown: no leaks observed; post-request counters were
  zero before shutdown.
- Python OpenAI client against vBuf: model listing, three chat requests, and
  streaming PASS.
- Python OpenAI client against local llama-server: model listing, three chat
  requests, and four stream chunks PASS.
- Android remote protocol audit: `YES`; base URL, model alias, chat JSON, SSE,
  and socket/client cancellation are sufficient. No Android code was changed.

## Evidence Classification

- 20-request lifecycle corpus: `X86_64_PHYSICALLY_MEASURED` and
  `HOST_CONTRACT_QUALIFIED`.
- A/B/A parity, stream parsing, cancellation, recovery, and concurrency:
  `HOST_CONTRACT_QUALIFIED`.
- RSS classification: `X86_64_PHYSICALLY_MEASURED`, classified
  `BOUNDED_CACHE_GROWTH`.
- Full 26-layer request: `FULL_26_LAYER_MEASURED` and
  `X86_64_PHYSICALLY_MEASURED`.
- Direct runtime/server overhead: `NOT_QUALIFIED` / `NOT_MEASURED`.
- Controlled source-failure injection and recovery:
  `X86_64_PHYSICALLY_MEASURED` and `HOST_CONTRACT_QUALIFIED`.
- Arbitrary runtime/backend/allocator failure recovery: `NOT_QUALIFIED`.
- Android/RISC-V physical execution: `NOT_QUALIFIED` for this host step.

## Verification

- Rust workspace tests: PASS.
- Full native build: PASS.
- Native CTest: `23/23 PASS`.
- Portable graph neutrality: `FORBIDDEN_LEAKAGE_COUNT=0`.
- Existing HTTP integration: PASS.
- Persistent sequential lifecycle harness: PASS, 20 baseline corpus requests.
- Follow-up faulted lifecycle harness: PASS, 40 corpus requests plus one
  controlled source failure and same-process recovery.
- KV/request isolation: PASS.
- SSE stream isolation: PASS.
- Real socket disconnect/cancellation: PASS.
- Failure recovery: PASS for four protocol failures.
- Concurrency policy: PASS, serialized queue.
- Full 26-layer server request: PASS.
- Python OpenAI regression against llama-server and vBuf: PASS.
- `git diff --check`: PASS.

Qualification artifacts are in `/tmp/opencode` or `/tmp`; no model files,
payload caches, generated binaries, or large logs are repository artifacts.

## Next Experiment

Qualify a matched direct-runtime control for server overhead attribution, and
separately expand source-failure coverage to a deterministic mid-request
failure matrix. Neither should change the serialized serving policy.
