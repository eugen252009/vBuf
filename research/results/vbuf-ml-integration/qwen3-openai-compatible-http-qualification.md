# Qwen3 OpenAI-Compatible HTTP Qualification

## Result

**PASS for text-only `/v1/chat/completions` and `/v1/completions` on the exact admitted Qwen3-14B Q4_K_M artifact and qualified CUDA device.** The compatibility server now retains one canonical `VbufModelRuntime` for its lifetime and creates a fresh `VbufGenerationSession` per inference request. The server's FIFO inference gate serialized overlapping HTTP inference requests; same-runtime concurrent Qwen execution remains **UNSUPPORTED / NOT QUALIFIED**.

No HTTP service beyond this existing compatibility server, native tools, Pi, dogfood, or llama.cpp comparison was started.

## Exact artifact and setup

- Payload SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`
- Payload: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf`
- Semantic bootstrap: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`
- CUDA qualification device: RTX 3060 12 GB
- Source: local HTTP range server at `127.0.0.1:18946`
- Canonical production implementation: `integrations/ggml/tools/vbuf_compat_server.cpp`, using the `VbufModelRuntime` / `VbufGenerationSession` ownership path. Qwen execution remains restricted by the existing exact-artifact admission checks; unsupported Qwen variants do not become executable through the server.

## HTTP coverage

The opt-in CTest `vbuf_qwen3_openai_text_http_qualification` passed. Its script is `integrations/ggml/tests/qwen3_compat_server_http_qualification.py`.

- `/health` and `/v1/models` returned the configured ready model.
- Chat and completion non-streaming responses had the expected OpenAI response objects, choices, usage/finish fields, and assistant text.
- For a matched two-token `Say hi` prompt, direct canonical-control and HTTP prompt-token hashes and generated-token hashes matched. Returned chat/completion text also matched byte-for-byte.
  - Chat generated-token hash: `77bac70773d10562`
  - Completion generated-token hash: `97a88800c2f13af1`
- Chat and completion SSE returned `text/event-stream`, valid endpoint-specific frames, terminal finish reasons, `[DONE]`, and text equal to corresponding non-streaming requests. UTF-8 token bytes are buffered across token boundaries before SSE serialization; invalid trailing UTF-8 is replaced rather than emitted as malformed JSON.
- Multi-turn history was encoded from the request's `messages`: a request with history and the same final question without history had different prompt-token hashes. Requests do not share conversational/KV state.
- A/B/A/B/A request repetition returned repeatable per-prompt output, consistent with per-request session isolation.
- A synchronized two-request HTTP overlap reached `queued_generations=2`; the later request waited about 723 ms at the FIFO gate. Both completed. This qualifies serialized HTTP admission only, not concurrent Qwen inference.
- A client-disconnected SSE request recorded `cancelled=yes`; health and a subsequent generation succeeded.
- Rejected inputs included malformed JSON, wrong model, empty messages, unsupported tool-history role, zero token limit, unsupported generation options/fields, non-empty tools/tool choice, and prompt-plus-generation capacity overflow. Empty tools with `tool_choice: none` remained accepted. A valid request succeeded after these failures.

## Lifetime and resource observations

The server completed **49 HTTP inference entries**: 25 non-streaming requests and 11 streaming requests (the remainder included direct API checks, multi-turn, two serialized overlaps, and one cancelled stream). The telemetry recorded one server model-runtime creation, 443 resident model tensors and `8,995,793,920` model-upload bytes throughout the request series. No repeated model-runtime construction or model upload was observed. Every completed request reported zero active generations and zero active streams/cancellations after cleanup.

- Peak VRAM reported across requests: `9,593,880,576`–`9,595,977,728` bytes.
- Post-run free VRAM: `3,026,190,336`–`3,028,287,488` bytes (2 MiB observed range).
- The per-session H2D/D2H diagnostics are the existing GGML backend tensor set/get counters, not a trace of all CUDA API transfers.

## Scope and remaining limits

This qualifies text-only operation through the existing server for the single exact artifact, local range source, pinned GGML/CUDA build, and tested device. It does not qualify arbitrary Qwen variants, alternate tokenizers/templates, remote/non-local sources, another device/platform, sanitizer behavior, OOM or session-construction failures, long-context behavior beyond the existing 1032-token capacity, or same-runtime concurrent inference. Capacity overflow is rejected before execution. Non-empty tools and tool calls remain unsupported; no native tool-call adapter was added. No external llama.cpp parity or `1e-5` comparison was performed.

## Verification

- `cargo test --manifest-path rust/vbuf-ml/Cargo.toml`: passed (all crate tests).
- Full CUDA-enabled CMake build in `/tmp/vbuf-qwen3-phaseab-build`: passed.
- Full existing CTest suite: **39/39 passed**.
- Opt-in `vbuf_qwen3_openai_text_http_qualification`: passed (1/1, exact artifact).
- `git diff --check`: passed during implementation review.
