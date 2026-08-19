# Phase D2.1: Higher-Quant DeepSeekV2 PoC22 Baseline

Date: 2026-08-19

## 1. Objective

Establish the current direct PoC22 baseline with a higher-quantization fixture
from the same DeepSeek-V2-Lite model lineage as the existing IQ1_S fixture.
No scheduler, residency, transport, kernel, tokenizer, persistent-format, or
execution-graph optimization was introduced.

## 2. Why Qwen Was Removed From This Baseline

The D2 Android failure used Qwen3 semantic metadata while PoC22's current
execution graph is DeepSeek-V2-Lite-specific. It failed at `output.weight`
lookup before payload acquisition. Qwen is therefore excluded from this
baseline so that model-family and execution semantics remain constant.

## 3. Existing IQ1_S Fixture

Repository fixture:

```text
research-models/DeepSeek-V2-Lite.IQ1_S.gguf
research-models/DeepSeek-V2-Lite.IQ1_S.vbuf
```

Identity and provenance:

- Family: DeepSeek-V2-Lite, `deepseek2` GGUF architecture.
- Parameter count: 15,706,484,224 BF16 parameters in the upstream model.
- MoE configuration: 27 blocks, 64 experts, 6 active experts per token,
  shared experts.
- HF repository: `legraphista/DeepSeek-V2-Lite-IMat-GGUF`.
- HF revision: `3048fc1df365e992c92a055324e8fd872e5763b9`.
- Existing quantization: IQ1_S.
- Existing GGUF SHA-256: `9d3bc4a5bc25b7acb8bc31436745bd8cfaf94509fd1322bb36ab155b0daf1616`.
- Existing vBuf SHA-256: `780a55b77d2730705a93622338d9747149624d72e558e26176868210fafcc`.
- GGUF bytes: `4,994,131,488`.
- vBuf bytes: `4,993,331,814`.
- Tensor count: `377`.
- Tensor payload bytes: `4,990,135,296`.

The existing GGUF hash matches the HF repository IQ1_S LFS identity. The
existing PoC22 x86 fixture qualifies the direct graph with four-token
autoregressive generation, router parity, logits parity, and zero-resource
teardown.

## 4. Higher-Quant Candidate Selection

Selected candidate: `IQ2_XXS` from the same repository and revision.

IQ2_XXS is the smallest meaningful higher-quant step available in the exact
same DeepSeek-V2-Lite quantization family. It is explicitly recognized by the
current manifest builder, converter, vBuf representation table, and GGML
execution lowering. No new quantization implementation was needed.

## 5. Hugging Face Provenance

- Repository: `legraphista/DeepSeek-V2-Lite-IMat-GGUF`.
- Revision: `3048fc1df365e992c92a055324e8fd872e5763b9`.
- File: `DeepSeek-V2-Lite.IQ2_XXS.gguf`.
- HF file size: `5,640,619,552` bytes.
- HF LFS SHA-256: `3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c`.
- Local downloaded SHA-256: identical to the HF LFS SHA-256.
- Source model: `deepseek-ai/DeepSeek-V2-Lite`.
- Model license reference: `https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL`.
- Quantization provenance: llama.cpp imatrix repository artifact; IQ2_XXS is
  listed as available and uses the repository's imatrix quantization family.

The model file was downloaded outside the repository under
`/tmp/opencode/deepseek-v2-lite-imat`. No model artifact is tracked or staged.

## 6. Quantization Compatibility

The current unmodified manifest/import compatibility comparison produced:

- Architecture: identical `deepseek2`.
- Tensor count: identical `377`.
- Tensor names, shapes, semantic roles, layers, and placement: identical.
- Model metadata plan: identical.
- Tokenizer conversion plan: identical.
- MoE plan: identical.
- IQ1_S types: 212 IQ1_S tensors, 27 IQ2_XXS tensors, 27 IQ4_NL tensors,
  2 Q2_K tensors, 1 Q5_K tensor, and 108 F32 tensors.
- IQ2_XXS types: 239 IQ2_XXS tensors, 27 IQ4_NL tensors, 2 Q2_K tensors,
  1 Q5_K tensor, and 108 F32 tensors.

This is the same execution graph with a weight representation change only.

## 7. vBuf-ML Conversion

The existing `build_step18_manifest.py` and
`convert_gguf_to_vbuf_ml.py` pipeline was used without source or runtime code
changes. The generated target was independently reopened and validated by the
converter.

Higher-quant output:

- vBuf bytes: `5,639,819,878`.
- vBuf SHA-256: `2ef0cdde67154ecc68bd82558007a4ea9da36262c4405007cb83306f68456e47`.
- Semantic bootstrap bytes: `3,206,424`.
- Semantic bootstrap SHA-256: `639ac345136de7f3d36c8fea15a8bf7fca70d3d518915a3371a9bd9cc02df910`.
- Tensor count: `377`.
- Tensor payload bytes: `5,636,623,360`.
- Source count: `1` authoritative external source plus SELF metadata source.
- Physical source range count: `1` merged range.
- Tokenizer payload: preserved by the existing semantic bootstrap generator.

The semantic bootstrap contains no tensor payload bytes. Its source profile
binds tensor ranges to the higher-quant vBuf source.

## 8. x86 PoC22 Qualification

Both fixtures used the same current direct binary, local HTTP range source,
256 MiB cost-aware residency capacity, full 27-block stack, seed `0`, and four
generated positions.

IQ1_S:

- Runtime sequence: `94761, 94761, 86711, 86711`.
- Reference sequence: identical.
- Logits maximum absolute error: `0`.
- Router parity: PASS.
- Generated-token feedback: PASS.
- Peak resident bytes: `268,120,064`.
- Peak active persistent bytes: `12,607,488`.
- Tracked source/reload bytes: `2,225,197,056 / 1,391,077,024`.
- Residency hits/misses/evictions: `6,691 / 5,457 / 2,856`.
- Coarse process wall time to four positions: `31,610 ms`.

IQ2_XXS:

- Runtime sequence: `76681, 76681, 76681, 76681`.
- Reference sequence: identical.
- Logits maximum absolute error: `0`.
- Router parity: PASS.
- Generated-token feedback: PASS.
- Peak resident bytes: `268,435,456`.
- Peak active persistent bytes: `12,607,488`.
- Tracked source/reload bytes: `2,621,661,184 / 1,645,312,320`.
- Residency hits/misses/evictions: `6,675 / 5,473 / 2,862`.
- Coarse process wall time to four positions: `32,979 ms`.

The different selected token sequence is expected from changing quantization;
each run is internally parity-qualified against its independent native
reference graph. The x86 run does not claim tokenizer or external llama logits
parity.

## 9. IQ1_S vs Higher-Quant Comparison

| Metric | Existing IQ1_S | Higher IQ2_XXS |
| --- | ---: | ---: |
| Source GGUF bytes | 4,994,131,488 | 5,640,619,552 |
| vBuf bytes | 4,993,331,814 | 5,639,819,878 |
| Tensor payload bytes | 4,990,135,296 | 5,636,623,360 |
| Tensor count | 377 | 377 |
| Model open | NOT_INSTRUMENTED | NOT_INSTRUMENTED |
| TTFT | NOT_INSTRUMENTED | NOT_INSTRUMENTED |
| 4-token completion | 31,610 ms coarse wall | 32,979 ms coarse wall |
| Generation tok/s | 0.1265 coarse wall | 0.1213 coarse wall |
| Peak resident bytes | 268,120,064 | 268,435,456 |
| Post-load RSS | NOT_SAMPLED | NOT_SAMPLED |
| Quantization | IQ1_S | IQ2_XXS |

The higher quant increases source and payload bytes by approximately 12.95%.
Under the unchanged 256 MiB residency cap, peak resident bytes are effectively
the cap in both runs. The coarse four-position wall time increased by about
4.33% in this local-HTTP run.

## 10. Android ARM64 Qualification

The IQ2_XXS semantic bootstrap was copied to the Pixel; the 5.64 GB full model
was not copied. The direct ARM64 PoC22 executable was launched through ADB with
the existing `HttpRangeSource` over an ADB-reversed local HTTP endpoint.

Android reached:

- ARM64 native startup: PASS.
- DeepSeek semantic bootstrap discovery: PASS.
- Tensor graph lookup: PASS through embedding setup.
- Remote source connection: PASS.
- Remote payload materialization: observed before failure.
- GGML generation: NOT_YET_QUALIFIED.

The run stopped with:

```text
expert_detail=tensor view contains a null required pointer
POC22_FAILURE=poc22_position_0_embedding failed
```

The existing external semantic bootstrap intentionally has null inline tensor
payload pointers. PoC22's actual materializer can acquire the external range,
but its independent reference graph still directly consumes the inline payload
pointer. This is a correctness/qualification seam in the existing PoC22
driver, not a model-family mismatch and not a scheduler result.

## 11. Remote/Source Shape

The failed Android run reached remote acquisition before the reference-side
null-pointer check:

- Observed physical HTTP range requests: `2,934`.
- Observed requested/returned bytes: `3,271,514,752`.
- Observed transport overfetch: `0` for returned HTTP ranges.
- Connection count: `1` keep-alive `HttpRangeSource` instance observed.

These are partial-run metrics, not a successful model-open or generation plan.
They must not be interpreted as the complete IQ2_XXS request plan.

## 12. Residency and Memory

The x86 baseline uses a fixed 256 MiB residency capacity and reaches
`268,435,456` bytes for IQ2_XXS. The direct driver reports no execution-prep
copy, repack, or transcode bytes. The Android run did not reach a normal
teardown or generation boundary; RSS/PSS was not sampled.

## 13. Generation

Bounded four-position generation passed on x86 for IQ2_XXS with internal
router/logit parity and zero teardown resources. Android generation did not
run because the reference graph failed during the first embedding operation.

## 14. Unoptimized Baseline Cost Model

With the current direct runtime and no new optimization, the higher quant:

- increases payload volume by 646,488,064 bytes over IQ1_S;
- increases coarse x86 four-position wall time by 1,369 ms;
- reaches the same fixed 256 MiB residency ceiling;
- increases tracked source and reload traffic;
- preserves direct zero-copy execution preparation;
- remains bounded by serialized materialization and eviction/reload behavior.

On Android, the first limiting factor is earlier: the external-source reference
path cannot validate against null inline payload pointers after materialization.

## 15. Known Inefficiencies

- The current x86 path uses serialized per-range acquisition through the simple
  materializer.
- The 256 MiB cap causes substantial reload traffic for both quantizations.
- PoC22 does not expose separate model-open, TTFT, hash, memcpy, or RSS phase
  timers.
- The Android external-bootstrap reference graph assumes inline payload bytes.
- The Android partial run materialized many ranges before correctness stopped.

## 16. What Was Deliberately Not Optimized

No global load plan, range batching, request pipelining, concurrency, caching,
lazy residency, buffer tuning, kernel change, quantization kernel, tokenizer
change, chat rewrite, persistent-format change, or portable ExecutionGraph work
was performed.

## 17. Baseline for Future Scheduler Work

The useful baseline is the x86 IQ1_S versus IQ2_XXS pair under identical PoC22
configuration. It establishes that the existing DeepSeek direct graph scales
to a materially larger supported payload without changing semantic graph logic.
The Android result is not yet a scheduler baseline because correctness stops in
the external-source reference path before generation.

## 18. Next Experiment

Exactly one next experiment is recommended:

```text
Smallest ARM64 runtime-correctness fix: make the PoC22 external-source
qualification/reference path consume a validated materialized embedding and
subsequent tensor view, then rerun the same IQ2_XXS four-token Android test.
```

Do not optimize the scheduler until this correctness seam is closed and the
Android direct runtime produces a complete generation result.

## Validation

```text
HF_MODEL_IDENTIFIED: PASS
HF_REVISION_PINNED: PASS
DOWNLOAD_COMPLETE: PASS
DOWNLOAD_SHA256_VERIFIED: PASS
SAME_DEEPSEEK_EXECUTION_SEMANTICS: PASS
HIGHER_QUANT_ALREADY_SUPPORTED: PASS
VBUF_IMPORT: PASS
VBUF_VALIDATION: PASS
SEMANTIC_BOOTSTRAP: PASS
TENSOR_GRAPH_ACCEPTED: PASS x86; PARTIAL Android
POC22_X86_BUILD: PASS
POC22_X86_GENERATION: PASS for IQ1_S and IQ2_XXS
ROUTER_PARITY: PASS x86
LOGITS_PARITY: PASS x86
TOKEN_SEQUENCE_PARITY: PASS within each fixture
ANDROID_ARM64_BUILD: PASS
ANDROID_DIRECT_RUNTIME_START: PASS
ANDROID_MODEL_OPEN: PARTIAL; external source reached, reference pointer failed
ANDROID_GENERATION: NOT_YET_QUALIFIED
ANDROID_TEARDOWN: NOT_YET_QUALIFIED after early failure
RUST_TESTS: PASS from D2 qualification
POC22_TESTS: PASS x86 direct recurrence
JSON_VALIDATION: PASS (181 JSON files)
CCC_INDEX: PASS
GIT_DIFF_CHECK: PASS
```

No downloaded model or generated vBuf artifact is in the repository.
