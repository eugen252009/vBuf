# DEEPSEEK AGENT-BACKEND QUALIFICATION

**Outcome: Phase A is qualified over HTTP on the existing bounded server path. Phase B is blocked by the exact model artifact/template: no native tool protocol is represented, so no DeepSeek adapter or real-model tool call was implemented or attempted. Pi qualification was not attempted.**

## BUILD / HTTP ENVIRONMENT

- **GGML revision:** `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d` (`ggml` 0.20.0), separately fetched into `${GGML_SOURCE_DIR}`; verified by CMake and `git rev-parse`.
- **vBuf-ML revision:** repository `73d38721841ec7ff0221bb4350c4c0d81f7af496`; Rust sources were clean at that commit when built. The surrounding repository worktree had unrelated pre-existing changes; these were preserved.
- **Compiler:** GNU C/C++ 14.2.0; `rustc` 1.95.0; Cargo 1.95.0; CMake 3.31.6; Ninja 1.12.1.
- **Build:** `cargo build --release --manifest-path rust/Cargo.toml -p vbuf-ml`; configured using the repository's `integrations/ggml/CMakeLists.txt`, `Release`, Ninja, `GGML_NATIVE=OFF`, pinned `VBUF_GGML_SOURCE_DIR`, `VBUF_ENABLE_CUDA=OFF`, `VBUF_BUILD_PROBES=ON`, and the just-built `VBUF_ML_LIBRARY`. Built `vbuf_compat_server` and `vbuf_agent_protocol_contract`.
- **Compiler flags / backend:** Release C++ flags `-O3 -DNDEBUG`; native CPU backend selected x86 AVX2/FMA/BMI2/SSE4.2. CUDA, OpenMP, BLAS, and CPU repacking were off.
- **Linked libraries:** `libggml.so.0.20.0`, `libggml-cpu.so.0.20.0`, `libggml-base.so.0.20.0`, and `rust/target/release/libvbuf_ml.so`; the vBuf region executor and agent protocol are static libraries. No llama.cpp runtime or model loader is linked.
- **HTTP test command:** `python3 integrations/ggml/tests/vbuf_compat_server_http_test.py --url http://127.0.0.1:18080 --model vbuf-deepseek-bounded` — **PASS**, `VBUF_COMPAT_SERVER_HTTP_TEST=PASS`.
- **Test server configuration:** `--blocks 2 --capacity 268435456 --max-new-tokens 4`, CPU. This is the established bounded two-block HTTP path, not full 27-layer model qualification.
- **Test artifact:** `legraphista/DeepSeek-V2-Lite-IMat-GGUF`, revision `3048fc1df365e992c92a055324e8fd872e5763b9`; GGUF SHA-256 `3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c`; converted payload SHA-256 `2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47`. Model/build artifacts and logs are outside the worktree under `${QUALIFICATION_CACHE_DIR}/`.
- **Build warnings:** the two pre-existing PoC22 `-Wsubobject-linkage` warnings remain; they did not block the build.
- **Path privacy:** local artifact/cache paths in this report are normalized to `${GGML_SOURCE_DIR}`, `${MODEL_ARTIFACT_DIR}`, `${QUALIFICATION_BUILD_DIR}`, and `${QUALIFICATION_CACHE_DIR}`; artifact hashes and test results are unchanged.

## HTTP PROTOCOL

- **text chat:** **QUALIFIED OVER HTTP**; real inference through the bounded two-block vBuf runtime returned a nonempty response.
- **completions:** **QUALIFIED OVER HTTP**; `/v1/completions` returned `text_completion` with `finish_reason=length` at the one-token request bound.
- **SSE:** **QUALIFIED OVER HTTP** for ordinary text; deltas reconstructed to exactly the non-stream response and terminal `finish_reason` matched, followed by `[DONE]`.
- **empty tools:** **QUALIFIED OVER HTTP**; `tools:[]` with `tool_choice:none` continues through ordinary text generation.
- **nonempty tools without adapter:** **QUALIFIED OVER HTTP** as an explicit `unsupported_feature` response; no native adapter is claimed.
- **tool_choice validation:** **QUALIFIED OVER HTTP** for an undeclared forced name; request is rejected before inference with the protocol error.
- **tool message validation:** **QUALIFIED OVER HTTP** for an orphan result; returns `invalid_conversation_state`.
- **assistant.tool_calls validation:** **QUALIFIED OVER HTTP** for malformed JSON arguments; rejected as an invalid request before inference. Valid calls are structurally covered by the model-independent contract suite, but cannot be returned by this model path.
- **finish reasons:** `length` was observed for bounded chat/completion and text SSE. Structured `tool_calls` output serialization/finish handling is covered by the protocol contract tests, not by an HTTP model response.
- **stream reconstruction:** ordinary text SSE matched non-stream text. Fragmented structured tool-call SSE reconstruction is **QUALIFIED WITH FAKE BACKEND** only; the HTTP server cannot produce native tool calls for this artifact.
- **bounds:** the HTTP layer rejected a declared body of 4 MiB + 1 byte with `400 HTTP request body too large`. Existing 4096-token prompt-plus-generation bound remains in code; this test did not drive the prompt to that boundary.
- **malformed requests:** invalid JSON, invalid escape, invalid conversation, malformed call arguments, unknown model, unsupported generation parameter, and oversized body returned client errors; the test suite passed.

## DEEPSEEK NATIVE CONTRACT

- **model/checkpoint:** quantized `DeepSeek-V2-Lite.IQ2_XXS.gguf` from the artifact above. Its repository README identifies `base_model: deepseek-ai/DeepSeek-V2-Lite`, not a tool-trained checkpoint. GGUF metadata identifies `general.name=DeepSeek-V2-Lite`, `general.architecture=deepseek2`, `deepseek2.block_count=27`, `deepseek2.expert_count=64`, and `deepseek2.expert_used_count=6`.
- **upstream metadata evidence:** `deepseek-ai/DeepSeek-V2-Lite` revision `604d5664dddd88a0433dbae533b7fe9472482de0`; config architecture `DeepseekV2ForCausalLM`, model type `deepseek_v2`, vocabulary 102400. The quantized artifact's README points to the base model. The separate `DeepSeek-V2-Lite-Chat` repository was inspected as a comparison, not substituted for the runtime checkpoint; its tokenizer template is likewise text-role-only.
- **tokenizer:** GGUF `tokenizer.ggml.model=gpt2`, pre-tokenizer `deepseek-llm`, vocab 102400, BOS ID 100000 (`<｜begin▁of▁sentence｜>`), EOS ID 100001 (`<｜end▁of▁sentence｜>`), add BOS true, add EOS false. GGUF token-type inspection found only BOS and EOS as actual special tokens; there are no tool-call special token IDs. Other occurrences of words such as “tool”, “call”, “function”, or “json” are ordinary vocabulary entries, not protocol markers.
- **chat template:** exact 459-byte template stored in the GGUF and copied from the upstream tokenizer config:

  ```jinja
  {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}
  ```

- **tool schema representation:** none. The template does not reference `tools`, `tool_choice`, function definitions, or schemas.
- **tool-call representation / function names / arguments:** none is specified by this checkpoint's metadata/template. No native marker or output grammar is evidenced. No regex/prose/JSON convention was inferred or tested.
- **tool result representation:** none. The template has no `role=tool` branch; roles other than `user`, `assistant`, and `system` render no content.
- **stop/termination semantics:** normal assistant messages append EOS `<｜end▁of▁sentence｜>`. There is no tool-call termination marker/contract in the selected artifact.
- **multiple calls:** **UNSUPPORTED** by the artifact evidence; no native encoding or lifecycle exists in the template.
- **existing llama.cpp/GGML support:** the local llama.cpp checkout is revision `a97123e497968f3440264c0464a7adc7c999c027`. `src/llama-chat.cpp` classifies this exact template as `DEEPSEEK_2`; its formatter emits only system, user, and assistant text and appends EOS to assistant text. `common/chat.cpp` has specialized parsers for other explicit tool formats, but none selected by this DeepSeek-V2 template. This reference behavior corroborates the artifact; it is not treated as the vBuf runtime architecture.
- **existing vBuf template engine:** it can render the artifact's ordinary text template used by the passing HTTP tests. The server's message representation/template path does not encode tool schemas, assistant structured calls, or tool-result continuation. It cannot correctly supply a native tool interaction for this template.

**Contract decision:** tool-definition insertion, `auto`, forced function selection, native calls/arguments, tool results, multiple calls, and tool-call stops are **BLOCKED BY MODEL / BLOCKED BY TEMPLATE** for the selected artifact. There is no evidence sufficient to implement an adapter. Adding invented prompt conventions would violate the task constraints.

## ADAPTER

- **architecture:** unchanged model-independent normalized protocol above inference; no DeepSeek-specific parser/template code was added.
- **implemented:** no native DeepSeek adapter.
- **unsupported:** all native tool request/output/continuation semantics for this checkpoint; normal text remains available. The HTTP layer rejects nonempty tools explicitly.
- **error handling:** HTTP/OpenAI parsing and conversation validation reject malformed requests before inference. The lack of a native tool contract is reported as `unsupported_feature`, not mislabeled as model output or runtime failure.

## REAL-MODEL QUALIFICATION

- **no-tool response:** ordinary text was physically exercised through HTTP on a two-block path (Phase A); this is not full-model qualification.
- **single tool call:** **BLOCKED BY MODEL / NOT TESTED**. No unsupported prompt convention was attempted.
- **tool-result continuation:** **BLOCKED BY TEMPLATE / NOT TESTED**; tool role is not represented.
- **external fixture result:** **NOT TESTED**; no model tool output to drive a harness.
- **tool failure:** **NOT TESTED**; no tool lifecycle.
- **multi-turn:** existing ordinary HTTP repeat/cancellation behavior was exercised. A structured tool multi-turn cycle is **NOT TESTED**.
- **multiple calls:** **UNSUPPORTED / NOT TESTED** for this checkpoint.
- **malformed native output:** **NOT TESTED**; no native output parser exists. The protocol validator rejects malformed synthetic outputs in its deterministic contract tests.

## STREAMING

- **supported:** normal text SSE is **QUALIFIED OVER HTTP**. Model-native tool streaming is **UNSUPPORTED / BLOCKED BY TEMPLATE**.
- **tool fragments:** structured IDs/names/argument fragments are **QUALIFIED WITH FAKE BACKEND** at the model-independent protocol layer only; no native DeepSeek fragments exist in evidence.
- **reconstruction parity:** text parity is **QUALIFIED OVER HTTP**; tool-call parity is **QUALIFIED WITH FAKE BACKEND**, not real model.
- **finish_reason:** text `length` passes over HTTP; structured `tool_calls` serialization is contract-tested with fake semantic output.
- **DONE:** text stream ends with `[DONE]`; checked by HTTP regression.

## RUNTIME

- **retained KV:** current `VbufGenerationSession::run` allocates request-local KV state; the HTTP/runtime session retains metadata, range source, and residency, not KV across requests. A tool-result continuation would therefore rebuild/re-tokenize the complete conversation and prefill it. This is code-inspected; tool continuation is blocked and was not measured.
- **materialization:** HTTP run used the existing two-block bounded path. Observed peak resident payload bytes stayed below the configured 268,435,456-byte residency cap; no materialization policy or bounds were changed.
- **residency:** repeated requests reused the persistent session's residency/source machinery; logs showed zero active leases/inflight bytes after completion. This does not qualify full-model residency or an agent turn.
- **cancellation:** HTTP client-disconnect test passed and server recovered; cancellation log returned with zero active leases/inflight bytes.
- **stale-state check:** repeated identical HTTP request produced identical text; KV is request-local by implementation. No tool-turn stale-state test was possible.

## BOUNDS

- **request bytes:** 4 MiB maximum; oversized HTTP body rejected, no limit raised.
- **prompt tokens without tools:** observed `Say hi` prompt was 10 tokens in the test log; other tested prompt forms used 3–10 tokens.
- **tool-schema tokens:** **NOT APPLICABLE / BLOCKED**; no schema representation exists in the selected template.
- **full agent turn tokens:** **NOT MEASURED / BLOCKED**; no tool cycle.
- **remaining context budget:** server bound remains `prompt_tokens + max_tokens <= 4096`; no schema/agent-turn allocation can be stated without inventing an input representation.
- **materialization/residency:** unchanged. Configured residency cap was 256 MiB; observed peak was below the cap.

## REGRESSION

- **contract tests:** `integrations/ggml/tests/run_agent_protocol_contract.sh` — PASS; CTest `vbuf_agent_protocol_contract` — PASS.
- **HTTP tests:** `integrations/ggml/tests/vbuf_compat_server_http_test.py` — PASS with real vBuf runtime inference on its configured two-block path.
- **ASan/UBSan:** standalone protocol contract test — PASS.
- **build:** Rust `vbuf-ml` release build and pinned-GGML CMake server/protocol-contract targets — PASS.
- **syntax/static:** server syntax-only compilation — PASS with the two existing PoC22 linkage warnings; Python HTTP test `py_compile` — PASS.
- **`git diff --check`:** PASS after this report/test update.
- **full all-target CMake build / all CTest probes:** NOT RUN; only the server and agent-protocol contract targets needed for the HTTP gate were built.
- **full 27-layer real-model tool behavior:** NOT TESTED; no native tool contract.

## QUALIFICATION STATUS

- **OpenAI tool protocol:** **IMPLEMENTED / TESTED**; deterministic protocol contract suite passes.
- **HTTP tool protocol:** **QUALIFIED OVER HTTP** for strict validation and explicit unsupported responses; no HTTP model-generated tool call exists.
- **DeepSeek native adapter:** **UNSUPPORTED** for this exact DeepSeek-V2-Lite artifact; blocked by model/template evidence.
- **real DeepSeek tool call:** **BLOCKED / NOT TESTED**.
- **tool continuation:** **BLOCKED / NOT TESTED**.
- **streaming:** ordinary text **QUALIFIED OVER HTTP**; structured fake protocol **QUALIFIED WITH FAKE BACKEND**; native tool streaming **UNSUPPORTED**.
- **Pi readiness:** **NO**. Required real tool call and continuation gates are unmet. Pi end-to-end qualification was not attempted.

## FAILED / BLOCKED

- The previous “required GGML and libvBuf-ML unavailable” block is resolved for the HTTP gate: the pinned source, Rust shared library, model artifact, converted payload, and bounded test server were restored outside the worktree; HTTP regressions passed.
- No tool-call request was sent to the real model. This is intentional, not a failed model inference: the selected checkpoint has no artifact-backed tool schema, call, result, or termination contract.
- Real-model agent qualification, token-budget measurement for schemas/tool turns, and native tool SSE are blocked by the selected model/template, not by the generic HTTP parser or SSE protocol layer.
- HTTP test construction initially expected a different wording for forced-tool validation and attempted to send a complete oversized payload after the server had already rejected its `Content-Length`; test expectations/transport were corrected. Product behavior was correct; the final suite passes.

## NEXT REQUIRED STEP

Select an exact DeepSeek checkpoint that is explicitly tool-trained and has a bundled, inspectable tool-capable chat template/native output contract (or provide such an artifact). Audit its tokenizer/template and actual output/continuation behavior before implementing anything. Re-run Phase A against that selected artifact/runtime path, then implement and physically qualify only the evidenced adapter. Do not substitute a DeepSeek-V3/V4 template or other model-family syntax for this V2-Lite artifact, and do not start Pi qualification until the real-model tool-call and continuation gate passes.
