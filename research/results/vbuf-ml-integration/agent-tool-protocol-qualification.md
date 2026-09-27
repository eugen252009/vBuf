# AGENT TOOL PROTOCOL QUALIFICATION

**Overall: model-independent protocol implemented and qualified with deterministic fake outputs. No real model or Pi tool-use qualification is claimed.**

Implementation: `integrations/ggml/include/vbuf_agent_protocol.h` and
`integrations/ggml/src/vbuf_agent_protocol.cpp`. The component is independent of
Pi, GGML execution, vBuf payload acquisition, and model-family syntax. The
compatibility server invokes it to validate Chat Completions requests. Valid
tool requests/history receive an explicit `unsupported_feature` response until
a model-native adapter is qualified. The server never executes tools.

The preceding implementation audit is in
[`agent-harness-tool-calling-audit.md`](agent-harness-tool-calling-audit.md).

## ARCHITECTURE

```text
OpenAI request -> strict protocol parser/state validator -> normalized DTOs
    -> ModelConversationAdapter -> AssistantOutput validation
    -> OpenAI JSON or SSE serialization

Test adapter: deterministic semantic fixtures only
Production model adapters: not implemented
Tool execution: Pi / external harness only
```

No Pi-specific dependency or model syntax was added to the runtime. The test
adapter does not interpret fake textual tool syntax.

## IMPLEMENTED

- Strict JSON parser with duplicate-key rejection, valid UTF-8/escape handling,
  configurable nesting depth, trailing-data rejection, and typed protocol errors.
- Request DTO parsing for `messages`, `tools`, `tool_choice`, token limits, and
  streaming flag. Unsupported request/message fields fail explicitly. Optional
  function `strict` must be boolean and is retained.
- Conversation/outstanding-call validation, output-versus-request validation,
  OpenAI non-stream response serialization, and SSE encode/reconstruction.
- Model-independent adapter interface and deterministic fixture-backed adapter.
- Compatibility server validates chat requests and explicitly rejects otherwise
  valid tool requests/history because its configured model has no qualified
  native adapter.

## INTERNAL REPRESENTATION

- **ToolDefinition:** function name, description, retained JSON Schema parameters.
- **ToolChoice:** `Auto`, `None`, or `ForcedFunction(name)`.
- **ToolCall:** ID, function name, JSON-encoded arguments.
- **ToolResult:** call ID and result content; wire messages are represented by
  `ChatMessage` with role `Tool`.
- **ChatMessage:** `System`, `User`, `Assistant`, or `Tool`; nullable-capable
  assistant content, tool-call list, tool result ID/content.
- **AssistantOutput:** optional text and zero or more structured calls.

## OPENAI CONTRACT

- **tools:** function definitions; duplicate names rejected. Parameter schemas
  must be objects and are retained structurally; no full JSON Schema validation.
- **tool_choice:** absent/`auto`, `none`, or forced named function. Forced names
  must be declared. Unsupported/malformed choices fail explicitly.
- **assistant.tool_calls:** IDs, `type:function`, function name, JSON-string
  arguments; arguments must parse as a JSON object.
- **tool role:** string result content retained.
- **tool_call_id:** required and matched to exactly one outstanding call.
- **finish_reason:** normal text `stop` or `length`; structured calls
  `tool_calls`.
- **streaming:** chunk deltas carry stable indices and reconstructible ID/name/
  argument fragments, then finish reason and `[DONE]`.

## CONVERSATION VALIDATION

- **unknown call IDs:** rejected.
- **duplicate call IDs/results:** rejected, including duplicate IDs across history.
- **outstanding calls:** any number up to the configured cap; results may arrive
  in any order. All must be resolved before another non-tool message or end.
- **invalid ordering:** orphan tool result, duplicate result, premature assistant
  continuation, or missing result rejected.
- **multi-call behavior:** multiple calls in one assistant message and matching
  out-of-order result messages are represented and tested.
- Calls returned by an adapter must be declared in request tools. `none` forbids
  calls; forced choice must return exactly one call for the selected name.

## FAKE BACKEND QUALIFICATION

The deterministic fake adapter returns queued semantic outputs; fixture harness
code supplies harmless tool results and resubmits the conversation. It performs
no host filesystem or shell operations.

- **normal text/no-tool:** tested.
- **single tool / tool-result continuation:** tested with fixture `add(37,5)`
  result `42` carried in the next conversation request.
- **sequential tools:** tested across `add`, returned tool result, `echo`, second
  tool result, and final assistant continuation.
- **multiple calls:** two outstanding calls and both results, including
  out-of-order result messages, tested.
- **multi-turn:** continuation followed by another user message tested.
- **bounds/malformed fields:** request bytes, JSON depth, schema and tool-result
  limits, duplicate keys, invalid UTF-8, and unknown message fields tested.
- **malformed output:** invalid fixture arguments and undeclared/choice-violating
  calls fail as internal adapter errors.

## STREAMING QUALIFICATION

- **text:** text SSE deltas reconstruct correctly.
- **tool-call fragments:** IDs, names, and JSON arguments are emitted in
  fragments and reconstructed by index.
- **multiple calls:** tested.
- **reconstruction parity:** streaming reconstruction matches the structured
  assistant output and non-stream serialized fields for text, multiple calls,
  and mixed text/call output.
- **finish reason / DONE:** `tool_calls`, `stop`, or `length` followed by
  exactly one `[DONE]`; final-fragment-with-finish-chunk reconstruction, missing/
  duplicate terminators, and records after termination tested.

## REGRESSION

- **protocol contract:** `integrations/ggml/tests/run_agent_protocol_contract.sh` — PASS.
- **sanitizers:** AddressSanitizer + UndefinedBehaviorSanitizer protocol test — PASS.
- **server syntax:** `g++ -std=c++17 -fsyntax-only` with available GGML headers —
  PASS (two existing `-Wsubobject-linkage` warnings in the included PoC22 source).
- **HTTP chat/completions/SSE/protocol regressions:** PASS; see the
  [DeepSeek agent-backend qualification](deepseek-agent-backend-qualification.md)
  for the exact command, runtime scope, and expanded cases.
- **CMake server/protocol-contract targets:** PASS against the pinned GGML
  revision and release `libvbuf_ml.so`; a full all-target build was not run.
- **`git diff --check`:** PASS.

## BOUNDS

Defaults: **4 MiB** request body; JSON depth 64; 64 tools; 64-byte function
names; 64 KiB descriptions; 256 KiB parameter schemas; 32 calls per assistant
turn; 128-byte call IDs; 64 KiB arguments per call; 1 MiB per tool result; 1 MiB
assistant content; 20,000 SSE records. Existing server request bound is
unchanged. The server's **4096 prompt-token** limit remains; tool-enabled input
is rejected before prompt rendering/tokenization, so tool schema/result token
footprint has not been measured. No limit was raised.

## NOT IMPLEMENTED

- DeepSeek native adapter; GLM native adapter.
- Production model-native prompt rendering or native tool-call parsing/stops.
- Real-model tool calls, Pi end-to-end qualification, and real tool execution.
- Full JSON Schema validation; multimodal message content; unsupported OpenAI
  generation options remain rejected.

## FAILED / BLOCKED

- The initial HTTP/build blocker is resolved; the bounded server and expanded
  HTTP regression suite passed. Full all-target CMake qualification was not run.
- Real model-native tool use remains blocked by the selected DeepSeek-V2-Lite
  model/template contract. Pi qualification was not attempted.

## NEXT REQUIRED STEP

The exact DeepSeek-V2-Lite artifact and its GGUF/upstream tokenizer template
were inspected in [DeepSeek agent-backend qualification](deepseek-agent-backend-qualification.md).
That checkpoint has no tool schema/call/result representation in its template,
so no adapter was implemented and no native tool-call prompt was attempted.
Select an exact tool-capable DeepSeek checkpoint with an inspectable native
contract before resuming adapter work. Pi remains gated on physical model
call-and-continuation qualification.
