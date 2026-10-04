# OpenAI Text Content-Part and Pi Direct Compatibility Qualification

**Result: PASS** for the exact admitted Qwen3-14B Q4_K_M artifact, local range source, RTX 3060 CUDA runtime, and server-enforced 1,032-token executable capacity. Pi 1.0.2 reached vBuf directly, consumed text SSE, received a native tool call over SSE, executed one local synthetic tool, and completed the `role=tool` continuation. No Pi client shim or server-side tool execution was used.

## Baseline and scope

- Starting HEAD: `592fa23c27f8e7d21a0424608f5cfca14d47a15a` (`feat(server): stream Qwen tool calls over OpenAI SSE`).
- Protected server commits `22bb88a`, `d03593a`, and `592fa23` were not amended or rewritten.
- Pi version: **1.0.2**; provider: `vbuf-local`; model alias: `Qwen_Qwen3-14B-Q4_K_M`.
- Payload: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf`, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Semantic metadata file: `Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`, SHA-256 `cc20b816a4cfdd192d4870b853354c51dd1b1402b83d01fac1e8ea98f2a9fea7`. The supplied `f409...` digest is the **payload** digest, not the semantic metadata file digest.
- Executable capacity remains **1,032 tokens**. Same-runtime concurrent Qwen inference remains unsupported/not qualified.
- CUDA device: RTX 3060. No unrelated GPU process was terminated.

The earlier Pi failure was reproduced before the fix: its user content was a text-part array and vBuf returned HTTP 400, `only string content (or null assistant content with tool calls) is supported`; inference was not reached. Pi's installed package was inspected and not changed.

## Content contract and normalization

The OpenAI DTO boundary in `integrations/ggml/src/vbuf_agent_protocol.cpp` normalizes content before any template or tokenizer work:

- Strings are returned unchanged.
- `system`, `user`, and `assistant` messages accept arrays consisting only of `{ "type": "text", "text": <string> }` parts.
- Multiple text parts concatenate directly, with **no inserted separator**: `foo` + `bar` becomes `foobar`.
- Unknown/non-text part types, including `image_url`, `input_audio`, `file`, and `video`, are rejected explicitly. Mixed text and unsupported parts fail the whole request.
- Missing/non-string `type`, missing/non-string `text`, invalid content shapes, and extra text-part fields fail closed.
- `role=tool` retains its existing string-only result contract. Assistant `content: null` with `tool_calls` is unchanged.
- Normalized strings are stored in the existing `ChatMessage.content`; Qwen/model code has no content-part awareness. Token capacity is computed from these normalized messages once; raw content-part JSON is not tokenized.

The deterministic prompt serialization was recovered from decoded canonical prompt token IDs:

```text
<|im_start|>user
Reply with exactly: PI_VBUF_OK<|im_end|>
<|im_start|>assistant
```

## String versus text-part equivalence

Control prompt: `Reply with exactly: PI_VBUF_OK`; generation limit 256.

| Evidence | String | One text part | Match |
|---|---:|---:|---|
| Normalized content | `Reply with exactly: PI_VBUF_OK` | `Reply with exactly: PI_VBUF_OK` | Yes |
| Serialized Qwen prompt | As above | As above | Yes |
| Prompt tokens | 16 | 16 | Yes |
| Prompt token hash | `7adc78db81abccc1` | `7adc78db81abccc1` | Yes |
| Generated token count | 89 | 89 | Yes |
| Generated token hash | `702ad565e261e0f1` | `702ad565e261e0f1` | Yes |
| Final text | Identical | Identical | Yes |

Prompt token IDs:

```text
151644,872,198,20841,448,6896,25,22578,2334,43831,8375,151645,198,151644,77091,198
```

Generated token IDs:

```text
151667,198,32313,11,279,1196,6801,752,311,9851,448,6896,330,1893,2334,43831,8375,3263,6771,752,1779,279,11221,1549,13,2379,1053,311,9851,448,429,4734,914,13,2308,1184,369,894,4960,1467,476,40841,13,4599,1281,2704,279,2033,374,6896,330,1893,2334,43831,8375,3263,358,3278,7683,429,1052,525,902,13580,966,476,5107,5766,13,97593,11,429,594,30339,13,358,3278,3624,429,438,279,2033,624,151668,271,1893,2334,43831,8375
```

Final model text (including its generated Qwen think markers):

```text
<think>
Okay, the user wants me to reply with exactly "PI_VBUF_OK". Let me check the instructions again. They said to reply with that exact string. No need for any extra text or explanations. Just make sure the response is exactly "PI_VBUF_OK". I'll confirm that there are no typos or additional characters. Alright, that's straightforward. I'll send that as the response.
</think>

PI_VBUF_OK
```

A two-part `foo`/`bar` request and the equivalent string `foobar` also had identical serialized prompts, prompt IDs, generated IDs, and responses.

## Unsupported content and capacity

- `image_url`, unknown type, `input_audio`, `file`, `video`, mixed text plus image, malformed text parts, and a content-part array used for a tool result were rejected by protocol contract tests.
- HTTP image, unknown-type, and mixed-part requests returned 400 before inference; no inference record was added. A valid text request immediately afterward succeeded.
- Near capacity, an array containing `"word " * 1000` used **1,009 prompt tokens** plus an 8-token generation allowance (1,017/1,032). Its string equivalent matched token-for-token and in output usage.
- An array containing `"word " * 1200` plus 8 requested output tokens returned HTTP 400 for executable context-capacity overflow. A valid request after overflow succeeded.
- `/v1/completions` was not changed. The existing exact-artifact text HTTP qualification, including its completion string/SSE checks, passed.

## Text SSE and existing tool regression

- Content-parts and string chat requests both passed `stream=true`; reconstructed SSE text, terminal finish, prompt IDs, serialized prompt, and generated IDs matched.
- Existing native non-stream tool HTTP and native tool SSE qualifications both passed, including assistant `tool_calls`, `role=tool` continuation, and stream reconstruction.
- Server-side tool execution remains **NO**.

## Pi direct qualification

Isolated Pi configuration selected only `vbuf-local`, used offline mode, a sanitized environment without external-provider credentials or proxies, and a `/tmp` working directory. The first text smoke had no built-in tools, extensions, context files, or session. The tool smoke loaded exactly one explicit synthetic extension; its only declared tool was `lookup_magic_number`. No shell, filesystem, or Git tool was exposed. Pi's installed package was not changed and no Pi-specific shim was used.

Pi's normal OpenAI Completions path streamed SSE. For the text smoke, Pi returned `PI_VBUF_OK` after Qwen inference. The 95 streamed text deltas reconstructed exactly to Pi's final assistant text; the terminal stop reason was `stop`, response ID `chatcmpl-vbuf-2`, and the corresponding vBuf log recorded 49 prompt tokens, 95 generated tokens, `finish_reason=stop`, and `error=none`. There was no hanging or duplicated output. Pi reports zero token usage because this endpoint does not emit a streaming usage frame; compatibility configuration therefore sets `supportsUsageInStreaming=false`.

### Pi context-window caveat

With `contextWindow: 1032` explicitly declared, Pi 1.0.2's local `clampMaxTokensToContext()` reserves a hard-coded 4,096 tokens and reduced the outgoing `max_tokens` to 1. That request reached vBuf and ran inference, but stopped after `<think>` with `finish_reason=length`; it was not counted as a successful text response.

For the successful direct retest, the isolated `models.json` omitted `contextWindow`; Pi 1.0.2 then used its built-in 128K default and sent `max_tokens=256`. This changes only Pi's temporary client-side planning metadata, **not** the vBuf runtime. The server still enforces the actual 1,032-token capacity and rejects overflow; its near/over-capacity checks above passed. This Pi metadata discrepancy is a qualification caveat, not a claim that vBuf executes 128K context.

No external provider fallback was configured or observed. Pi recorded `provider=vbuf-local`, `model=Qwen_Qwen3-14B-Q4_K_M`, and `willRetry=false` on normal completion.

## Pi tool SSE and synthetic roundtrip

Pi received the native streamed tool call:

```text
id: call_chatcmpl-vbuf-3_0
index: 0
function: lookup_magic_number
arguments: {"key":"amber"}
finish_reason: tool_calls
```

Pi executed the sole local synthetic tool, returning `42`. It sent a `role=tool` result with the same `tool_call_id`; the vBuf continuation prompt contained `<tool_response>\n42\n</tool_response>`. Qwen returned final assistant text reporting 42 with `stop`. vBuf did not execute the tool. The Pi JSON event stream reported `willRetry=false`.

## Verification

- Rust workspace: `cargo test --manifest-path rust/Cargo.toml --workspace` — PASS.
- Full CUDA-enabled build: `cmake --build /tmp/vbuf-qwen-sse-build -j4` — PASS.
- Configured CTest: **43/43 passed**, including production-session, text HTTP, non-stream tools, SSE tools, and `vbuf_qwen3_openai_content_parts_qualification`.
- Content normalization / unsupported-part contract tests — PASS.
- String/array prompt, token-ID, output, text-SSE, multipart, and capacity equality qualification — PASS.
- Pi direct text/SSE and synthetic tool roundtrip — PASS, with the client context-window caveat above.
- `/v1/completions` regression — PASS via the existing text HTTP qualification.
- `python3 -m py_compile` on the new qualification script — PASS.
- `git diff --check` — PASS before final staging.
- `ccc index` — PASS (1,758 files listed; 1 added, 9 reprocessed, 0 errors).
- `ccc search` for message normalization and fail-closed content handling — PASS.

No server shim, Pi source change, server-side tool execution, or real coding-agent dogfood was used.
