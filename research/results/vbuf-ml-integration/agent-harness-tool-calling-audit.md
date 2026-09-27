# Agent-Harness Tool-Calling Audit

Status: **AUDIT COMPLETE; TOOL CALLING NOT IMPLEMENTED OR QUALIFIED**
Scope: repository server and installed Pi OpenAI-compatible client behavior.

This is a protocol/runtime audit, not an agent-capability qualification. No model
requests were run and no tool call was fabricated. Existing unrelated working-tree
changes were present during this audit and were left untouched.

## Architecture Finding

The smallest compatible architecture remains:

```text
Pi
  -> standard OpenAI Chat Completions request/response
  -> vbuf_compat_server protocol + model-native adapter
  -> vBuf-owned vBufGenerationSession
  -> model
```

Tool calls belong above inference/runtime acquisition: the HTTP protocol layer
must parse and validate OpenAI tool contracts and conversation state; a
model-specific adapter must render definitions/history in the model's native
format and recognize native tool-call output. The vBuf residency/materializer
and GGML execution layers should remain unaware of Pi and OpenAI tool DTOs.

## Phase 1 Audit

### Request parsing and conversation reconstruction

`integrations/ggml/tools/vbuf_compat_server.cpp` currently has a handwritten
request parser (`top_level_key`, `parse_messages`) rather than a JSON DOM/parser.
It parses message `role` and string `content` only. Accepted roles are `system`,
`user`, and `assistant`. It does not represent `tool_calls`, tool-call IDs,
function names/arguments, `tool` role results, or assistant messages with null
content. It is not a validated protocol state machine for outstanding calls.

The body is bounded at 4 MiB and the rendered prompt is bounded at 4096 tokens
including requested output. Prompt tokens are reconstructed from all accepted
messages for every request. `VbufGenerationSession::run()` allocates KV state
per generation call; requests do not carry retained KV across HTTP calls. Thus
multi-step tool continuation would currently be full conversation resubmission
and fresh per-request KV, not retained cross-request KV.

### Template and model-native boundary

The server loads the persisted tokenizer chat-template bytes. Its local renderer
is a deliberately narrow interpreter: one `for message in messages` loop, role
branches, a small expression set (`message['content']`, BOS/EOS, string
literals), and a generation-prompt branch. It does not render tool definitions,
assistant tool calls, tool results, `tools`, or `tool_choice` template variables.

The tokenizer profile stores the chat template as opaque UTF-8 and explicitly
does not define template execution. Therefore protocol support cannot safely
invent a universal textual tool syntax. Tool-call rendering and recognition
must be model/template-specific and fail closed when that capability is absent.
The existing DeepSeek HTTP qualification proves normal chat only, not model
native function-call formatting. The GLM sidecar records no chat template and
has no HTTP server integration.

### Generation, stops, and finish reasons

Generation receives prompt tokens and the model EOS as its stop token. The
server decodes generated tokens directly into assistant text. There is no
structured output channel, grammar constraint, model-native tool-call parser,
or alternate stop condition for a tool-call terminator. Non-stream responses
always serialize `message.content` and use `stop` or `length`; SSE emits only
`delta.content`, then `stop` or `length`, followed by `[DONE]`. No `tool_calls`
finish reason or structured call object is emitted.

### Pi OpenAI-compatible client expectations

The installed Pi `openai-completions` implementation constructs ordinary
OpenAI Chat Completions requests. When tools are active it sends
`tools: [{type:"function", function:{name,description,parameters}}]` and may
send `tool_choice`. Its transcript conversion sends assistant calls as
`assistant.tool_calls` with IDs, function names, and JSON-serialized arguments;
results are `role:"tool"` messages with matching `tool_call_id`. Pi maps
`finish_reason:"tool_calls"` to a tool-use assistant stop.

For streaming, Pi accumulates `choices[0].delta.tool_calls` by index/ID. Call
IDs, names, and function argument JSON may be spread across chunks; the final
finish reason must identify tool use. A content string that merely looks like
JSON is not equivalent. This protocol expectation was inspected in the
installed Pi OpenAI-compatible client; no live Pi-to-vBuf request was made.

### Existing tests and qualification

`vbuf_compat_server_http_test.py` and lifecycle tests qualify normal text chat,
completions, text-only SSE framing, bounded requests, selected errors,
recovery, and cancellation. Tests currently assert that `temperature` is
rejected; the server also rejects `tools`, `tool_choice`, `stop`, `seed`,
`top_p`, `response_format`, and `stream_options`. Existing real-model tests do
not cover tool protocols, model-native function calls, tool-result continuation,
or Pi tool execution.

No server/model-independent tool protocol module or deterministic fake
inference backend was found. The Bloom registry contains no model-inference
capability; this task is properly implemented in the repository's native
server/runtime integration rather than by invoking Bloom.

## Minimum Safe Contract To Implement

Before enabling tools on a model, require all of these boundaries:

1. Strict JSON parsing and bounded structural validation for tool definitions,
   choice values, assistant tool calls, and tool-result messages.
2. A conversation validator that matches every tool result to exactly one
   preceding outstanding call ID, rejects duplicates/orphans/missing results,
   and rejects unsupported content shapes and outstanding-call sequences.
3. A model-native adapter contract for rendering tool schemas/history and
   parsing tool-call output. No generic prose/regex extraction.
4. Structured non-stream OpenAI response with `finish_reason:"tool_calls"`;
   streamed deltas with stable call index/ID and incrementally reconstructible
   function name/argument fragments; `[DONE]` only after the terminal chunk.
5. A documented supported subset of `tool_choice` (`auto`, `none`, forced
   function) and explicit rejection of all other options until tested.
6. A model-independent protocol/state-machine suite using deterministic model
   output fixtures, separate from real-model and live-Pi qualification.

The HTTP/tool-call layer must never execute tools. Pi (or another harness) owns
tool execution and returns results to the model. The server should only
validate and transport structured call requests.

## Qualification Status

| Requested case | Status | Reason |
|---|---|---|
| Basic chat / existing text SSE | Previously qualified | Step 31R/31S and HTTP tests; separate from tool qualification |
| No-tool agent decision | Not tested | Requires tools-aware request and model response evaluation |
| Single/sequential/multiple calls | Unsupported | No structured call representation or native model adapter |
| Tool result continuation / multi-turn | Unsupported | `tool` role and assistant calls are rejected; each generation is independent |
| Unknown tool / invalid arguments / tool failure | Not tested | The server must not execute tools; harness behavior needs agent-loop tests |
| `tool_choice` auto/none/forced | Unsupported | All `tool_choice` requests are rejected |
| Streamed tool-call fragments | Unsupported | SSE serializes only content deltas |
| Cancellation of tool-call continuation | Not tested | Generation cancellation is qualified, not tool lifecycle cancellation |
| Bounded residency during tool continuation | Not tested | Existing text request bound does not prove schema/result continuation behavior |
| Pi protocol integration | Not tested | No Pi configuration or live harness run |
| Real DeepSeek native tool calling | Blocked | Current chat-template renderer and generation path lack tool semantics |
| GLM via HTTP/API | Blocked | No template in the sidecar and no GLM HTTP server/runtime adapter |

## Recommended Next Step

Build a **model-independent tool protocol layer and deterministic test harness**
first, without changing the generation/runtime API or claiming model support.
Use a maintained strict JSON parser already acceptable to this project (or add a
small isolated parser dependency after dependency review); replace request
substring parsing for the tool-capable route. Define typed OpenAI DTOs and the
conversation/outstanding-call state machine, with explicit limits for schema
bytes/depth/count, calls per assistant turn, argument bytes, and tool-result
bytes.

Then define a `ModelToolAdapter` capability boundary. Initially, the production
DeepSeek adapter should report unsupported until its exact stored chat template,
required special tokens, native tool-call syntax, stop markers, and actual
runtime output have been inspected and qualified. A deterministic backend may
exercise protocol serialization/state transitions, but cannot stand in for
real-model tool generation or Pi end-to-end qualification.

Do not retrofit tool semantics into vBuf storage, materialization, residency,
GGML loader code, or a Pi-specific provider. Once the model adapter is qualified,
run the actual harness loop with a disposable Pi workspace and harmless tools.
