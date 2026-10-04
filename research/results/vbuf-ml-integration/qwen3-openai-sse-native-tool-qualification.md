# Qwen3 OpenAI SSE Native Tool-Call Qualification

**Result: PASS** for the exact admitted Qwen3-14B Q4_K_M artifact, local HTTP/range source, RTX 3060 CUDA path, and 1,032-token executable context. This qualifies OpenAI-compatible streaming of native Qwen tool calls; it does not implement or execute tools on the server.

## Admitted artifact and runtime

- Payload: `Qwen_Qwen3-14B-Q4_K_M.vbuf`
- SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`
- Semantic metadata: `Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`
- Payload size: 9,000,232,144 bytes
- Source: local range server at `127.0.0.1:18946`; its HTTP ETag matched the required SHA-256.
- CUDA device: NVIDIA GeForce RTX 3060, selected as CUDA device 0. Before the live runs it had 12,029 MiB free; the 2080 SUPER was not selected.
- Executable context capacity: 1,032 tokens. The existing production session test also passed its 1,024-prompt + 8-generated-token boundary workload.
- Same-runtime concurrent Qwen inference remains **UNSUPPORTED / NOT QUALIFIED**.

## Implementation strategy

The server buffers the complete generated native response for streamed tool requests. It reuses `parse_qwen3_native_tool_output`, including request-tool membership and JSON-argument validation, then serializes the resulting `AssistantOutput` into SSE records. It sends no SSE headers or data until generation and native-call validation complete. There is no new token-by-token native parser. Tools are returned to the client; no tool is run by the server.

Each tool-call delta carries the full ID, type, index, and function name once. Argument JSON is emitted in UTF-8-safe fragments. The protocol reconstruction test concatenates fragments in order and validates the reconstructed call.

## Live streamed tool result

Request: `POST /v1/chat/completions`, `stream=true`, with the deterministic prompt requiring exactly one `lookup_magic_number` call for `amber`.

- HTTP status: 200
- Content-Type: `text/event-stream`
- SSE data records: 5 total (four JSON chunks and `[DONE]`)
- Choice index: 0
- Tool-call index: 0
- Tool-call ID: `call_chatcmpl-vbuf-3_0` (present once and stable through reconstruction)
- Type: `function`
- Function: `lookup_magic_number`
- Argument fragments: 1
- Reconstructed arguments: `{"key":"amber"}`; valid JSON
- Finish reason: `tool_calls`
- `[DONE]`: present

The qualification client reconstructed index, ID, type, function, arguments, and finish reason from the SSE records. All JSON data frames had the expected chunk object and choice index. The stream's HTTP response ID remained constant across records.

## Controls and continuation

- **Direct canonical execution:** matched the streamed HTTP prompt token IDs and generated token IDs. The direct native call selected `lookup_magic_number` with `{"key":"amber"}`.
- **Non-stream HTTP control:** returned the same function, arguments, and `tool_calls` finish reason. Its request-local call ID differed from the streamed request's ID, as expected.
- **Tool-result continuation:** the qualification client sent the reconstructed assistant `tool_calls` and matching `tool_call_id`, followed by a synthetic `role=tool` result of `42`. The existing non-stream continuation path returned final assistant content `\n\n42`, finish reason `stop`, and no further tool calls.
- **Text SSE:** streamed requests with `tools` omitted and `tools: []` both returned valid content deltas, a terminal `stop`/`length` reason, and `[DONE]`; their reconstructed text and finish reason matched. The existing text-only HTTP qualification also passed its text SSE and `/v1/completions` checks.

## Failure, capacity, isolation, and FIFO checks

- A client reset during native generation, before the complete call was available, produced a cancelled request without starting the SSE response. Active leases, generations, streams, and cancellations returned to zero; a subsequent valid request succeeded.
- A deterministic SSE writer failure after two records left only the initial tool-call metadata on the wire: no argument fragment, `finish_reason: tool_calls`, or `[DONE]` was fabricated. The request was cancelled and cleaned up; a subsequent valid request succeeded.
- Protocol contract tests rejected invalid JSON, an unknown tool, an unterminated call, malformed argument types, and other malformed native blocks. Invalid outputs did not produce serializable tool-call responses. These are synthetic parser/protocol tests; the live model fixture generated a valid call.
- An oversized streamed tool prompt plus generation budget was rejected with HTTP 400 before SSE headers/data and before a model request was recorded. A subsequent valid request succeeded.
- Request sequence covered text, non-stream tool, text SSE, and repeated tool SSE. Request records showed no retained runtime leases/generations; the repeated tool stream had a fresh response and call ID and reconstructed the same function/arguments.
- Two overlapping request pairs (streamed tool + text; streamed tool + non-stream tool) had non-overlapping runtime intervals and positive admission waits. This verifies FIFO serialization, not concurrent inference.
- Named `tool_choice` remains unsupported and is rejected before SSE begins.

## Build and tests

- `vbuf_compat_server`, `vbuf_direct_control`, and `vbuf_agent_protocol_contract` built successfully with the CUDA-enabled configuration.
- Full CUDA-enabled build completed successfully.
- Configured CTest suite: **42 registered, 42 executed, 42 passed**. This configuration included the opt-in Qwen production-session, text HTTP, native-tool HTTP, and SSE-tool qualifications. The baseline had 41 tests; the one additional registration is `vbuf_qwen3_native_tools_sse_qualification`.
- The full suite included the exact-artifact 1,032-token production session qualification, text HTTP/SSE and `/v1/completions`, non-streaming tools and continuation, and the new live SSE qualification.
- `vbuf_agent_protocol_contract` passed under AddressSanitizer and UndefinedBehaviorSanitizer; Python syntax checks and `git diff --check` passed.
- `cargo test --manifest-path rust/Cargo.toml --workspace` passed. No Rust code was changed for this feature.
- `ccc search` and `ccc index` both timed out; indexing was stopped by its command timeout. This did not block qualification because the implementation and tests were reviewed directly and the full configured build/test suite passed.

The qualification harness is `integrations/ggml/tests/qwen3_compat_server_native_tools_sse_qualification.py`.
