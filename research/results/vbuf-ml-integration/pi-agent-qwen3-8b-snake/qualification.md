# Qwen3-8B vBuf → Pi C Snake Agent Qualification

**Overall: NOT QUALIFIED as an autonomous coding agent.** The exact Qwen3-8B artifact loaded and generated through the canonical vBuf runtime; OpenAI-compatible text/SSE, structured native tool calls, and Pi's real tool execution all worked. However, Pi did not produce a working Snake implementation: its strict build failed, its self-test reported a failure while exiting zero, and its repair attempts did not complete. Do not describe this as a passing coding-agent qualification.

## Scope and provenance

- Starting vBuf revision: `d672c7c` (`qualify Qwen3 CUDA numerical kernels`), branch `vbuf-ml`.
- Model: official `Qwen/Qwen3-8B-GGUF`, pinned repository revision `7c41481f57cb95916b40956ab2f0b139b296d974`, Q4_K_M.
- Original GGUF SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; size 5,027,783,488 bytes.
- Converted vBuf payload source: `Qwen3-8B-Q4_K_M.vbuf`, SHA-256 `cc85fa7afd90808484485de0e0a09e88ff5b98b58d1fb69c7083f64916417cb5`; size 5,026,262,216 bytes. The semantic sidecar records this vBuf source identity, not the original GGUF digest.
- Semantic sidecar SHA-256: `99f2892dbe58d427605457729a6edcfa037d67f8ec1ff7f325953345b5d11212`.
- Metadata observed: 36 layers, hidden 4096, 32 query heads, 8 KV heads, head dimension 128, FFN 12288, vocabulary 151936, model metadata context 40960. The 399 tensors represent 5,021,827,072 payload bytes.
- Raw inputs, conversion manifest/parity evidence, Pi transcripts, HTTP responses, server/range logs, project output, and build/test results are under this directory and `raw/`.

## Runtime admission and inference

A separate opt-in path was added for **only** the exact pinned vBuf source hash plus the expected Qwen3-8B shape. Default model admission remains the qualified Qwen3-14B artifact; a post-build launch without `--experimental-qwen3-8b` exited 1 with `Qwen3 production supports only the qualified Qwen3-14B Q4_K_M artifact` and the rejected 8B source hash (preserved under `raw/default-admission-rejection.*`). The opt-in also requires all 36 layers and explicitly enables experimental context capacity. It does not change default production capacity 1032, prefill chunk 32, optimizer SHADOW, execution thresholds, canonical packed-V attention, or native-AV experimental status.

The runtime loaded all 399 tensors from the local range source on an RTX 3060 12 GiB, using the canonical vBuf materialization/residency path and single-device CUDA execution. Server startup reached readiness in about 16 seconds. One measured request reported 5,021,827,072 uploaded bytes, peak device use 7,904,690,176 bytes, and 4,717,477,888 bytes free afterward. These are single-run observations, not a performance qualification.

The HTTP path was bound to `127.0.0.1:18088`; the isolated range source used `127.0.0.1:18318`. The server executable capacity was 8192 tokens, below the model metadata's 40960. Pi's model definition reports the actual 8192-token service capacity. No external service on port 8080 was used.

## HTTP, SSE, and Pi tool execution

- Plain HTTP chat inference returned `PI_QWEN3_8B_OK`; Pi reported `provider=vbuf-local`, `model=qwen3-8b`, and `stopReason=stop`. After the final build, a fresh server reload and direct inference also returned `FINAL_RUNTIME_RETEST_OK`.
- A native Qwen tool request returned a valid OpenAI `tool_calls` object with `lookup_magic_number` and `{"key":"amber"}`. The streamed endpoint emitted incremental function-name/argument SSE frames, `finish_reason=tool_calls`, and `[DONE]`.
- Pi 1.1.0 used its built-in `openai-completions` SSE provider and received a Qwen-generated `bash` tool call. Pi executed `printf PI_QWEN8B_REAL_TOOL_OK` locally; the recorded tool result had exit code 0, and Pi then returned that output. This is genuine Pi tool execution, not server-side tool execution or a client shim.
- Pi compatibility metadata correctly disables unsupported `store` and streaming-usage fields. An initial request without `supportsStore=false` was rejected before inference; the corrected isolated model configuration is preserved in `raw/pi/models.json`. A separate diagnostic request including `temperature` was rejected as unsupported; the final direct inference omitted it, and Pi's tested configuration also worked without it.

## Snake coding-agent attempt: failed

Pi worked in `/tmp/pi-qwen3-8b-snake-cli`, outside both source worktrees, with only read/write/edit/bash tools and no extensions, MCP, skills, project context, or online provider fallback. It read the task, wrote C and Makefile files, and invoked `make test` through its real Bash tool.

The resulting implementation does not pass qualification:

1. The required strict `make test` fails to compile because `usleep` is undeclared under the requested C11/POSIX build flags.
2. Compiling diagnostically with `-D_DEFAULT_SOURCE` let the test body run, but it printed `Test 3 failed: Growth not correct` and still exited **0**. The test is not fail-closed. Food seeding uses `srand(time(NULL))`, contrary to the deterministic contract.
3. Code inspection found further failures: the demo sleeps and is not stable; the food is not rendered; immediate reverse input becomes a self-collision/game-over rather than being ignored; growth does not preserve/extend the tail correctly; and the tests do not establish the required mechanics.
4. Pi attempted a compile repair, but its exact-text edit did not match and it never reran the build successfully. Other repair attempts ended in malformed/truncated Qwen reasoning/tool output or server context-capacity rejection. One first attempt tried `sudo apt-get` after assuming ncurses was available; sudo was rejected without a terminal/authentication, no package was installed, and subsequent runs used a sudo-blocking PATH guard.

The independent compiler/test logs and the unmodified generated project are preserved. We did not fix the generated project manually and count that as agent success.

## Verification and boundaries

- CUDA-enabled full build of `/tmp/vbuf-qwen-native-matrix-build`: **PASS**.
- CTest with existing configuration `VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS=OFF`: **45/45 PASS** after the runtime changes.
- Qwen3-8B exact-identity opt-in server load, short text inference, native HTTP tool-call JSON, SSE tool-call framing, Pi streamed text, and Pi bash-tool roundtrip: **PASS** as observed above.
- Autonomous Snake source generation, strict compilation, its tests, and independent game-rule verification: **FAIL**. Therefore full Pi coding-agent readiness is **not qualified**.
- The external GGML checkout's pre-existing modification and protected stashes were not changed; no unrelated GPU workload was stopped; no push was performed.

The task-owned implementation change provides a bounded experimental bring-up, not a promoted production model path. Raw evidence preserves failed attempts as well as successful protocol/runtime observations.
