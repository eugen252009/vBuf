# DEEPSEEK TWO-BLOCK HTTP REGRESSION INVESTIGATION

Date: 2026-09-26

## Outcome

**The numerical/runtime regression was not caused by the dirty source changes. The failing reproduction served the wrong external payload file.** The semantic vBuf-ML sidecar is bound to the converted `.vbuf` payload, but the failing command served the original `.gguf` at the same URL. The sidecar's tensor source offsets and representation descriptors are for the converted payload layout. Reading those offsets from the GGUF yielded invalid embedding/router bytes, which propagated non-finite values until Release GGML code reported `non-finite router score`.

Local cache/build paths in reproduction commands are normalized to `${MODEL_ARTIFACT_DIR}` and `${QUALIFICATION_BUILD_DIR}`; hashes, arguments, and measured outcomes are preserved.

No production-code fix was needed or made. Using the source artifact identified by the sidecar restores the bounded HTTP path. The failing configuration remains reproducible when the wrong source is deliberately served; no numeric values are clamped, replaced, skipped, or otherwise masked.

## Change Map (Worktree at Investigation Start)

| Category | Modified/untracked paths | Relevance to routed-layer failure |
|---|---|---|
| Qwen3 discovery/catalog | `integrations/ggml/include/vbuf_ml_model_metadata_ffi.h`, `include/vbuf_model_architecture.h`, `src/vbuf_model_architecture.cpp`, `tests/model_architecture_contract.cpp`; metadata/config/catalog additions in `tools/router_driven_moe_poc11.cpp`, `tools/multi_layer_poc16.cpp`, and the explicit architecture guard in `tools/autoregressive_poc22.cpp`; related CMake entries | Could affect metadata loading/dispatch, not tensor payload bytes or DeepSeek graph math. DeepSeek metadata passed. The Qwen3 guard is bypassed for `deepseek2`. Not causal in the matched-source comparison. |
| DeepSeek runtime | `tools/attention_poc14.cpp`, new `tools/deepseek_v2_lite_semantics.h`, `tests/deepseek_semantics_contract.cpp`; small session instrumentation/config additions in `tools/autoregressive_poc22.cpp` | Attention/RoPE changes can affect hidden states; inspected carefully. With correct source, the direct path is finite and passes. |
| FFN | `tools/multi_layer_poc16.cpp`: residual is now supplied to routed FFN input (including the changed block-0/batched behavior) | Plausible for routed activation values, but failure begins in token embedding before block 0 attention/FFN. Not causal here. |
| MoE/router | `tools/multi_expert_moe_poc12.cpp`, `tools/full_moe_layer_poc13.cpp`, batched MoE changes in `tools/multi_layer_poc16.cpp`; `include/vbuf_parallel_executor.h`, `src/vbuf_parallel_executor.cpp` | Router softmax/selected weights and optional selected-expert scheduling changed. The reported error precedes TopK/weight normalization; default worker/thread settings are serial. Not causal in the matched-source comparison. |
| Quantization | No new GGML quant type mapping was part of the DeepSeek changes. Existing GGUF/vBuf representation and tensor binding code was not changed to fix this failure. | Crucial to diagnosis: the sidecar described an F32 router at a source offset in the `.vbuf` payload. That offset was read from the distinct `.gguf` file and interpreted as the sidecar's F32 payload. |
| Tensor binding/materialization | `include/vbuf_materializer.h`, `src/vbuf_materializer.cpp`, `include/vbuf_residency.h`, `src/vbuf_residency.cpp`, and tracing/CPU-thread changes in `src/vbuf_tensor_wave.cpp` | Plausibly affect buffer lifetime or execution. Payload bytes and expected identity were directly distinguished; correct `.vbuf` data passed. No binding/lifetime defect was found. |
| GGML/backend | `attention_poc14.cpp` uses GGML attention operations; `src/vbuf_tensor_wave.cpp` has optional CPU-thread selection; CMake includes parallel executor | Plausible for numerical execution. Clean-HEAD and dirty-worktree builds both pass with the correct payload and fail with the wrong payload, separating this from the cause. |
| HTTP/server | `tools/vbuf_compat_server.cpp`, `tests/vbuf_compat_server_http_test.py`, `include/vbuf_generation.h`, agent protocol library/tests | Chat/protocol fields and diagnostics changed; the failing request is ordinary text (`Say hi`, one requested token). Server URL override selected the wrong file; protocol code did not affect inference values. |
| Tests | New protocol, model-architecture, parallel-executor, runtime-trace, materializer-lifetime, and DeepSeek-semantics tests; expanded HTTP test; CMake test registration | Test code did not create tensor values. The HTTP script itself did not select the source; server startup configuration did. |
| Unrelated | `AGENTS.md`, Android CMake update, agent-protocol files/reports, runtime-footprint script, readiness/research reports | No path into the two-block inference computation. `integrations/ggml/CMakeLists.txt` contains both relevant build entries and unrelated test/library registrations. |

## Exact Inputs and Environment

- **Checkpoint:** DeepSeek-V2-Lite `IQ2_XXS`; GGUF SHA-256 `3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c`.
- **Semantic sidecar:** `DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf`; SHA-256 `4c6316c8d1941d8c6e108c978425c03ca9500013a402d5c9a89a9c4f7e4da233`.
- **Authoritative payload required by that sidecar:** `DeepSeek-V2-Lite.IQ2_XXS.vbuf`; SHA-256 `2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47`, size `5,639,819,878` bytes. The persisted source profile contains this hash, declared size, and a `.vbuf` locator.
- **GGUF size:** `5,640,619,552` bytes; it is a different physical layout and has a different hash. Similar file size does not make its offsets compatible.
- **GGML:** `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- **vBuf base revision:** `73d38721841ec7ff0221bb4350c4c0d81f7af496`, with the dirty changes listed above. Rust vBuf-ML sources are clean at that revision.
- **Compiler/build:** GNU C/C++ 14.2.0, CMake Release, `-O3 -DNDEBUG`; x86 CPU backend with AVX2/FMA/BMI2/SSE4.2; CUDA, OpenMP, BLAS, and CPU repacking disabled. The separate known-good worktree used the same pinned GGML, compiler, and Release/backend settings.
- **Bounded runtime:** `--blocks 2 --capacity 268435456 --max-new-tokens 4`, CPU, default `expert_workers=1`, `expert_threads=1`.
- **HTTP request:** `POST /v1/chat/completions`, JSON `{"model":"vbuf-deepseek-bounded","messages":[{"role":"user","content":"Say hi"}],"max_tokens":1}`. Prompt token count 10; token hash `232a1eb1cb807d1b`.

## Reproduction and Baseline Comparison

### Wrong source (the failing setup)

The server was started with the semantic sidecar above, while its overridden `--source-url` served `DeepSeek-V2-Lite.IQ2_XXS.gguf`. The HTTP regression failed deterministically on both first/fresh and repeated/cached requests:

```text
HTTP 500: non-finite router score
completed_layers=1
completed_positions=0
generated_tokens=0
```

The same failure reproduced in a separate clean Git worktree at the base revision, with the same artifact, GGML, compiler, request, and bounds. This rules against the Qwen3/FFN/MoE dirty-code delta as a necessary cause. With Release `NDEBUG`, NaNs continued into the later router check; this was not a valid model inference.

### First non-finite and source validation

A temporary environment-gated diagnostic (removed after collection) established:

- **First invalid operation:** `embedding_row_dequantize`, before block 0 attention. The layer input was already non-finite.
- **Embedding binding:** `token_embd.weight`, row tensor ID `1100376`, source offset `70,182,064`, representation 4, row shape `[2048,1]`, payload length 672 bytes. Those offsets/descriptors belong to the converted `.vbuf` payload, not the GGUF.
- With the wrong `.gguf` source, the embedding output had its first non-finite at index 1280, finite-value min/max `-2,180,158.5 / 377,040`, and absolute max `2,180,158.5`. All subsequent attention/FFN/router activations were contaminated.
- **Router binding:** `blk.1.ffn_gate_inp.weight`, tensor ID `20019`, expected source offset `103,092,640`, representation F32 (0), shape `[2048,64]`, payload 524,288 bytes. The wrong file's bytes, interpreted as this F32 tensor, were non-finite (first invalid index 90; finite-value range approximately `-3.40058936e38..3.40031068e38`).
- With the correct `.vbuf` source, the router weight was all-finite (`min=-0.7421875`, `max=0.56640625`, `abs_max=0.7421875`, mean `-7.22531588e-5`). The first routed hidden input was F32 `[2048,1]`, all finite (`min=-1.6915592`, `max=1.23180664`, `abs_max=1.6915592`). The first router matmul logits were all finite (`min=-2.39841795`, `max=2.91887832`, `abs_max=2.91887832`).

Therefore the router check was a downstream symptom. The first transition was **finite source identity/layout → wrong source range bytes → non-finite embedding row output**. No router arithmetic change is justified.

### Correct source and known-good comparison

- Current dirty worktree + sidecar-matched `.vbuf` source: full HTTP regression **PASS twice**.
- Separate clean worktree at `73d3872` + same sidecar-matched `.vbuf` source: exact two-block `Say hi` request **PASS** (HTTP 200, one token).
- Prior saved successful server log has the same bounded alias/configuration and successful two-block executions; the sidecar persisted source identity confirms its payload is `.vbuf`.
- The earlier failure is reproduced by changing only the served source file from `.vbuf` to `.gguf`; it is not reproduced with the authoritative source. No source-code hunk had to be reverted or patched.

## Sanitizers

A separate Debug server build used `-fsanitize=address,undefined -fno-omit-frame-pointer` with halt-on-error settings.

- Correct `.vbuf` source, exact bounded request: **HTTP PASS**, no ASan/UBSan/LeakSanitizer reports.
- Wrong `.gguf` source: GGML's Debug assertion stopped earlier at `ggml_compute_forward_rms_norm_f32`, `Assertion 'scale > 0.0f' failed`, after invalid embedding values. No ASan/UBSan report was emitted. This assertion is a useful fail-fast diagnostic for invalid input, not a fix or a substitute for the Release numerical test.

## Required Regression Gates

- Pinned GGML Release build, all targets: **PASS**.
- CTest: **30/30 PASS**, including model-architecture catalog and DeepSeek semantics contracts.
- DeepSeek two-block correct-source HTTP regression suite (chat, completions, text SSE, protocol validation, oversized request): **PASS twice**.
- DeepSeek source/metadata discovery: exercised during server startup and both successful HTTP runs.
- Qwen3 discovery/catalog fixture: **PASS** in `vbuf_model_architecture_contract` (443 entries, shape/representation/payload checks). Explicit Qwen3 pre-DeepSeek rejection remains code-path inspection; no Qwen3 sidecar was available to invoke it end to end.
- ASan/UBSan valid-source HTTP request: **PASS**, no reports.
- `git diff --check`: **PASS** after the report update.

## Resolution / Follow-up

The reproducible command must serve the exact payload selected by the sidecar:

```bash
python3 scripts/range_server.py \
  --file ${MODEL_ARTIFACT_DIR}/DeepSeek-V2-Lite.IQ2_XXS.vbuf \
  --port 18090 --log

${QUALIFICATION_BUILD_DIR}/vbuf_compat_server \
  --semantic-model ${MODEL_ARTIFACT_DIR}/DeepSeek-V2-Lite.IQ2_XXS.semantic.vbuf \
  --source-url http://127.0.0.1:18090 \
  --blocks 2 --capacity 268435456 --max-new-tokens 4 \
  --port 18080 --model-alias vbuf-deepseek-bounded
```

Do not point this sidecar at the source GGUF. No Qwen3 Q4_K/Q6_K, graph, inference, or tool-call work was done in this investigation. No files were reset, cleaned, or committed; the temporary diagnostic edits were removed.
