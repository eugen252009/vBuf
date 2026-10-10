# Qwen3-8B second-turn truncation and tool-cycle follow-up

**Status:** the tool-protocol failure at the generation-limit boundary is fixed and tested. The underlying Qwen3-8B raw-generation divergence remains unresolved; autonomous coding-agent qualification remains unsuccessful. Qwen3-8B stays experimental and identity-gated.

## Finding

The captured second-turn request is equivalent at the vBuf-to-tokenizer boundary to the matched llama.cpp request: rendered prompt SHA-256 `39c32c329cd2683ed946e1903282ffe3afb5f54c4179de436cd15dcda018e99c`, 916 token IDs, token-ID SHA-256 `feaf88943eafee31328b6ed0c9a3fc7db5ceb82b393f8a6bde5d9d5637be2236`. The vBuf request had 11,372 available context slots but requested a 6,144-token completion cap.

Before the change, vBuf generated all 6,144 tokens, began a native `<tool_call>`, emitted no `</tool_call>`, then the parser returned HTTP 500 (`unterminated tool_call block`). The raw text is 12,206 bytes (SHA-256 `94f95871b33442fd7be1c5797530d55d8606217419ca073439dd1091d6250852`); it contains repeated `d` initializers, no `main` definition, and is not a complete tool call. The 8,192-token diagnostic replay shared the full 6,144-token sequence as a prefix and also exhausted its cap without closing the call. Raising the cap is not a demonstrated fix.

The matched llama.cpp request used the same 916 prompt token IDs and completed a `write` tool call after 2,975 generated tokens. Its generated file still failed the prior strict task test; this is comparison evidence, not a successful agent qualification.

**Boundary diagnosis:** prompt rendering, tokenization, and SSE serialization are not the source of this failure. Raw generation diverges first; the parser then incorrectly classified a budget-truncated native call as an internal error. The patch fixes that protocol classification and preserves fail-closed behavior. It does not fix the generation divergence.

## Change and safety behavior

`parse_qwen3_native_tool_output` now accepts whether generation actually reached its token budget. The server supplies that state from the generated-token count. At the limit, an unfinished reasoning block, partial native marker, or unterminated tool-call block produces an empty assistant result with `finish_reason=length`; all tool calls are discarded. This prevents execution or forwarding of partial tool arguments. Forced `tool_choice` also accepts this explicit length result instead of turning safe truncation into another internal error.

The default parser path remains strict. Malformed complete blocks and incomplete blocks not marked as limit-exhausted are still rejected. Fully completed calls remain valid at the cap. Regression coverage includes partial open/close markers, a complete call followed by a truncated call (all calls are discarded), malformed JSON, forced tool choice, and non-streaming/SSE reconstruction.

No model admission, sampling, numerical threshold, production execution path, AV behavior, or persistent format changed.

## Verification

- `vbuf_agent_protocol_contract`: CMake/CTest **1/1 passed**; the standalone protocol contract also passed. The full `vbuf_compat_server` CMake target built successfully; its binary is byte-identical to the one used in the live trials.
- Actual second-turn request with a 12-token cap through the patched server: HTTP 200, valid SSE terminal `finish_reason=length` and `[DONE]`, empty content, no `tool_calls`.
- Three independent real Pi sessions with fresh workspaces and no retained session state: **3/3 passed**. Each completed four sequential tool calls (`read`, `write`, `bash`, `read`), five model turns, and five HTTP exchanges. Each verified exact `cycle-result.txt` bytes `ORCHID-42` (9 bytes; SHA-256 `4e9264d77d3feb7149552d79463a930d0329492d1ac07bf66b7ea3be68130f3d`) and ended with `PASS ORCHID-42`. This is a protocol/workflow regression fixture, not coding-agent qualification.
- One bounded rerun of the original Snake task used the same 6,144-token request limit. It completed in 448.3 seconds; the first `read` call succeeded, then the second model completion exhausted 6,144 tokens. The patched server returned HTTP 200/SSE `finish_reason=length`, not HTTP 500. The second request took 424.9 seconds, reached 6,144 generated tokens, and had the same prompt-token hash and generated-token-ID hash (`a4d9e93c7cb6658c66a36fb79d0fa523329a6794e0daae05db3a90ee9e08a4ed`) as the earlier raw replay. Pi made no second tool call and generated no source file; the task therefore **failed**. This confirms the protocol fix while leaving the model-generation failure open.
- `git diff --check` and artifact identity checks passed. Runtime binary used in the Pi runs: SHA-256 `b779f2bbe0a1efcb8df02fc0fb43eb7e523ee1bde7fc43fd7816648e5472ee32`.

## Identity and scope

All Qwen3-8B runs used the previously gated artifacts: source GGUF SHA-256 `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`, vBuf payload `cc85fa7afd90808484485de0e0a09e88ff5b98b58d1fb69c7083f64916417cb5`, and semantic sidecar `99f2892dbe58d427605457729a6edcfa037d67f8ec1ff7f325953345b5d11212`. The sandbox exposed only the RTX 3060 to the backend; the unrelated RTX 2080 SUPER workload was left untouched.

## Evidence

- `raw/ctx12-6144-vbuf-01/`: original second-turn request, prompt, token IDs, raw generation, parser HTTP 500, and capture summary.
- `raw/ctx12-8192-vbuf-01/`: longer-cap diagnostic replay and identical-prefix evidence.
- `raw/ctx12-6144-llama-01/`: matched llama.cpp request, rendered prompt, tokenization, compressed full SSE response, and generated source.
- `raw/fixed-server-limit-12-vbuf-01/`: live patched-server SSE truncation regression.
- `raw/pi-sessions/`: three independent Pi runs, compressed agent/HTTP/server logs, manifests, generated files, and aggregate results.
- `raw/snake-after-protocol-fix/`: bounded real Pi Snake follow-up, request/response logs, token IDs, result, and `snake-followup-summary.json`.
- `raw/artifact-sha256sums.txt`: checksum inventory for the report, fixtures, harness, and raw evidence.
- `raw/compressed-raw-uncompressed-sha256sums.txt`: SHA-256 checks for the decompressed SSE and source-prefix captures, preserving their exact original whitespace.
- `raw/harness/`: reproducible session runner and the sandbox harness inputs. The shared harness implementation is reused from the earlier A/B qualification; the new runner overrides only the fixture/output/runtime paths.

The 14B native-AV candidate remains `Candidate_NOT_Valid`; these agent-protocol results do not change or backfill its stopped numerical matrix.

## Follow-up fixed-prefix diagnosis

The subsequent token/logit investigation is documented in [generation-divergence-diagnosis-20261010.md](generation-divergence-diagnosis-20261010.md). It locates the period-3 suffix at generated token 1,288 and the first greedy fixed-prefix mismatch against pinned llama.cpp at token 60, a near-tie. The follow-up found no specific implementation defect and made no production runtime changes; Qwen3-8B remains experimental and unqualified.
