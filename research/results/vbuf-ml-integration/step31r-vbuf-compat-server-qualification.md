# Step 31R vBuf llama-server-Compatible HTTP Qualification

Status: **FUNCTIONAL; BOUNDED AND FULL-LAYER HOST PATHS QUALIFIED**.

The server presents a small llama-server/OpenAI-compatible HTTP surface while
inference remains owned by the native vBuf-ML/TensorWave runtime. The local
llama-server was used only as a wire-protocol oracle. It was not invoked by the
vBuf server, used as a proxy, or used as a model loader.

## Protocol Audit

The locally running llama-server was queried at `http://127.0.0.1:8080` with
model alias `Qwen3.6-35B-A3B`. Representative observed behavior:

| Endpoint / case | Client requires | Implement | Observed wire behavior / scope |
|---|---:|---:|---|
| `GET /health` | Yes | Yes | `200`, JSON status; vBuf adds runtime readiness |
| `GET /v1/models` | Yes | Yes | `200`, `object=list`, public model IDs |
| `POST /v1/chat/completions` non-stream | Yes | Yes | JSON `chat.completion`, choices/message/usage |
| `POST /v1/chat/completions` stream | Yes | Yes | `text/event-stream`, `data:` records, `[DONE]` |
| `POST /v1/completions` | Existing local OpenAI client/API | Yes | JSON `text_completion` subset |
| Invalid JSON | Error handling | Yes | Local oracle returned `500`; vBuf returns a deliberate `400` JSON client error |
| Unknown model | Error handling | Yes | `400` explicit model-not-found error |
| Unsupported generation option | Error handling | Yes | `400`, never silently ignored |
| Tools, embeddings, multimodal, reranking | No | No | Explicitly out of scope |

The oracle stream used blank-line-separated SSE records and a terminal
`data: [DONE]`. The vBuf server emits the same framing and uses HTTP chunked
transfer with hexadecimal chunk sizes.

## Server Configuration

The new native target is `vbuf_compat_server`. It accepts:

```text
--semantic-model PATH
--source-url URL
--model-alias ID
--host HOST
--port PORT
--blocks N
--capacity BYTES
--max-new-tokens N
--runtime-mode normal|qualification
```

The default host is localhost. Non-local binding prints an unauthenticated
exposure warning. The tested bounded configuration was:

```text
semantic model: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf
source: http://127.0.0.1:18124/models/DeepSeek-V2-Lite.IQ2_XXS.vbuf
alias: vbuf-deepseek-bounded
blocks: 2
capacity: 268435456
max new tokens: 4
runtime mode: NormalInference
```

Semantic metadata, tokenizer metadata, tensor plans, the HTTP range source,
and the residency/materialization session are prepared by the server runtime.
The generation session reuses validated plans, the source connection, and
bounded residency across serialized requests. KV state is newly allocated for
every request.

## Real HTTP Qualification

The installed Python `openai` client was used without protocol-code changes.

- `openai` version: `2.53.0`.
- Same client against local llama-server: chat and streaming protocol PASS.
- Same client against `vbuf_compat_server`: model listing, non-stream chat, and
  streaming chat PASS.
- Client protocol changes required: `0`.
- The HTTP integration script passed:
  `integrations/ggml/tests/vbuf_compat_server_http_test.py`.
- The script covered health, model listing, chat, completions, invalid JSON,
  unknown model, unsupported options, SSE framing, repeated request isolation,
  and client disconnect recovery.

The bounded vBuf request returned actual generated text (`行车`) from the
current runtime. No mocked, reference-only, llama-server, or external-LLM
response was used.

## Runtime And Residency Results

### Bounded Two-Block Path

- Request: one user message, one generated token.
- Prompt tokens: `10`.
- Generated tokens: `1`.
- Runtime mode: `NormalInference`.
- Runtime/reference path in normal serving: reference path not executed.
- Example first cold request: `1.337 s`, requested source bytes `255493376`,
  materialized bytes `83569664`, peak residency `111305984` bytes.
- Subsequent requests reused source/residency state; observed source bytes were
  `0` for warm repeated requests.
- The server HTTP test observed cancellation recovery after a client disconnect.
- Teardown/resource leaks: none observed.

Each request log records a generated request ID, model alias, prompt and
generated token counts, requested source bytes, materialized bytes, peak
residency, elapsed nanoseconds, finish reason, and cancellation state.

### Full 26-Layer Path

- Same HTTP contract and vBuf server binary, `--blocks 26`.
- Request: one user message, one generated token.
- HTTP result: `200` with actual generated text (`**,`).
- Runtime mode: `NormalInference`.
- Prompt tokens: `10`.
- Generated tokens: `1`.
- Requested source bytes: `6327692864`.
- Peak residency: `268412928` bytes.
- Total request elapsed: `101027354284 ns` (`101.027 s`).
- Full-model server path: `PASS` for this bounded one-token request.

The full-layer source-byte total is requested range traffic and can exceed the
5.64 GB canonical payload because ranges are reacquired/overfetched under the
current bounded residency plan. It is not a claim that the full model was
resident at once.

## Ownership And Safety

- HTTP/OpenAI DTOs in TensorRef/materialization/TensorWave: `NO`.
- llama-server used as vBuf runtime: `NO`.
- llama-server used as proxy: `NO`.
- llama.cpp model loader used by vBuf server: `NO`.
- GGML source acquisition owned by backend: `NO`; source remains vBuf range
  materialization.
- GGML residency ownership reintroduced: `NO`; the server uses the existing
  vBuf residency/materialization path.
- Normal server mode executes `NormalInference` only.
- Qualification mode remains an explicit debug configuration and is not the
  normal HTTP default.

Concurrency is intentionally serialized: one active HTTP request/generation
at a time. Additional TCP clients wait in the listen backlog. Each request gets
new KV state and prompt tokens; no prior conversation is implicitly reused.
The weight residency/source session may be reused, but request KV and generated
token state are isolated.

Client disconnect is observed at the live SSE send boundary and stops the
generation loop at the next safe token boundary. The focused HTTP test confirms
the server remains healthy after disconnect.

## Supported Scope

Supported:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` with `model`, string `messages`, `stream`, and
  bounded `max_tokens` / `max_completion_tokens`
- `POST /v1/completions` with `model`, string `prompt`, and bounded `max_tokens`
- vBuf tokenizer metadata and the current metadata chat-template subset
- live SSE streaming
- external HTTP RangeSource and bounded materialization/residency

Explicitly unsupported and rejected:

- `temperature`, `top_p`, `stop`, `seed`
- tools and tool choice
- response formats
- embeddings, multimodal inputs, reranking, parallel slots, and speculative
  decoding

## Verification

- Rust workspace tests: PASS.
- Native build: PASS.
- Native CTest: `23/23 PASS`.
- Portable graph neutrality: `FORBIDDEN_LEAKAGE_COUNT=0`.
- Server HTTP integration: PASS.
- Real bounded non-stream generation: PASS.
- Real bounded live stream generation: PASS.
- Existing Python OpenAI client compatibility: PASS against llama-server and
  vBuf server.
- Cancellation/recovery test: PASS.
- `git diff --check`: PASS.

No model artifacts, payload caches, generated binaries, or server logs are
repository artifacts from this qualification.
