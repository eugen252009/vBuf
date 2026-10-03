# Qwen3 Canonical Runtime / Session Integration — Current Status

**Checkpoint scope:** the focused admission commit contains the shared Qwen model-admission module, its qualification-runner adoption and admission contract. The canonical-runtime binding, runtime-admission contract, and generic runtime/session ownership changes remain in the dirty worktree and are not part of that checkpoint. They are preserved for the next integration step.

**Disposition: BLOCKED / INCOMPLETE.** This iteration shared Qwen3 model admission and exact-artifact metadata with the canonical model-runtime, but it did not extract the qualified CUDA model math or bind CUDA weights/KV/workspaces to runtime/session lifetimes. Canonical Qwen dispatch remains disabled. No qualification-runner shortcut was added.

## BASELINE

- Artifact payload: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf`
- Semantic bootstrap: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`
- SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31` (**verified**)
- Backend/device: qualification runner GGML CUDA, RTX 3060 12 GB (CUDA 12.4, SM 8.6); a second RTX 2080 SUPER was present but not used.

## GENERIC ARCHITECTURE

- `VbufModelRuntime`: prior generic lifecycle **QUALIFIED**; this iteration attaches shared Qwen validated metadata/catalog to its lifetime. CUDA model resources are **NOT IMPLEMENTED**.
- `VbufGenerationSession`: prior generic lifecycle **QUALIFIED**; Qwen-specific mutable state is **NOT IMPLEMENTED**.
- Concurrent same-runtime execution: **UNSUPPORTED / NOT QUALIFIED** (unchanged).

## PREVIOUS QWEN QUALIFICATION

- Resident CUDA, persistent KV, 32-token bounded prefill, per-layer scratch reuse and memory scaling remain qualified for the qualification-only implementation as recorded in `qwen3-resident-cuda-performance-and-production-text-qualification.md`.
- Historical 1024+8 qualification-only run: context 1032; approximately 3.20 GB free VRAM after chunks; sampled KV checks passed.
- This iteration reran the qualification runner at capacity 40 with a 32-token prompt plus 8 appends, exact artifact, per-layer scratch reuse and sampled KV-integrity audit. It passed: fixed layer scratch buffer, absolute position/mask/cache-row checks, sampled historical K/V immutability, and bit-identical repeated outputs. The runner logged 3.43 GB free after setup and returned to baseline free VRAM after teardown. This is a qualification-runner regression, **not production-session evidence**.

## KNOWN NUMERICAL STATUS (PRESERVED)

- Incremental/full CUDA: **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**.
- Chunked/full prefill: **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**.
- CPU/CUDA prefix 25: **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**.
- External llama.cpp strict `1e-5` gate: **FAIL**.
- No CPU/CUDA or llama.cpp parity claim is made.

## PHASE A — SHARED QWEN CORE

### Extracted module

- `integrations/ggml/include/qwen3_model.h`
- `integrations/ggml/src/qwen3_model.cpp`

These now share semantic-bootstrap opening, architecture/metadata validation, dense tensor-inventory admission, source SHA-256 identity, tokenizer BOS/EOS and `add_bos` metadata, and source/materializer descriptor setup between qualification and canonical model-runtime. The qualification tool calls this module rather than retaining a second implementation of those admission operations.

### Model operation implementation and qualification-only remainder

| Operation / concern | Current owner/status |
|---|---|
| RMSNorm, Q/K/V, RoPE, GQA attention, O projection/residual, FFN/SwiGLU, final norm, LM head | **Qualification-only** in `qwen3_block_qualification.cpp`; not extracted |
| KV allocation/access/append/read | **Qualification-only**; not extracted |
| Bounded 32-token prefill and per-layer scratch reuse | **Qualification-only**; not extracted |
| Single-token decode and logits | **Qualification-only**; not extracted |
| Memory tracing, checkpoint capture, CPU/CUDA comparisons, fixture/oracle work, diagnostic readbacks, CUDA audit hooks, report formatting | **Qualification-only** |

### Shared operations / duplicate model math

- Shared RMSNorm, Q/K/V, RoPE, attention, FFN, KV, prefill and decode: **NONE YET**.
- Duplicate implementations remaining: qualification code still has separate full-batch/decode and per-layer prefill graph construction, plus the older block qualification path. Production has no Qwen model-math implementation, so there is not yet a production-vs-qualification duplicate; instead, production is missing the implementation entirely.
- Result: the central shared-core extraction requirement is **BLOCKED / NOT COMPLETE**. The qualification runner still owns its private Qwen runtime math.

## PHASE B — QWEN MODEL RUNTIME

### Model admission

For the exact semantic bootstrap, shared admission validated architecture `qwen3`, 40 layers, hidden 5120, 40 attention heads, 8 KV heads, head dimension 128, FFN 17408, vocabulary 151936, 443 tensor entries, dense tensor shapes and Q4_K/Q6_K representation contract, source SHA-256, BOS/EOS IDs, and `add_bos=NO`. The artifact has separate embedding/output tensors with matching vocabulary geometry. Exact production identity admission rejects any other source SHA.

Unsupported Qwen variants are rejected by the exact-identity admission gate when using canonical Qwen metadata setup. Synthetic alternate-variant end-to-end admission fixtures were **NOT TESTED**.

### Backend and residency

- CUDA backend creation/destruction through `VbufModelRuntime`: **NOT IMPLEMENTED**.
- Runtime-owned resident weights and embedding: **NOT IMPLEMENTED**.
- Sharing backend/weights across sessions and one-time model upload: **NOT IMPLEMENTED / NOT TESTED**.
- Runtime teardown of Qwen device resources: **NOT IMPLEMENTED / NOT TESTED**.

The qualification runner still creates/uploads its own weights and destroys them after its test operation. Its warm decode transfer claims remain prior qualification evidence, not a result from the new canonical runtime.

## PHASE C — QWEN SESSION STATE

- Qwen KV owner/capacity/isolation: **NOT IMPLEMENTED**.
- Logical length: generic session length exists, but Qwen never advances it; Qwen processing is **NOT IMPLEMENTED**.
- Per-layer prefill scratch and reusable decode scratch in session lifetime: **NOT IMPLEMENTED**.
- Qwen reset, KV visibility, capacity overflow, session teardown: **NOT TESTED**.
- Two-session mutable-state isolation: **NOT TESTED** for Qwen.

## PHASE D — STATEFUL OPERATIONS

- Qwen prefill/decode/append through canonical `VbufGenerationSession`: **NOT IMPLEMENTED**.
- Qualification chunk size remains 32; production use is **NOT IMPLEMENTED**.
- Position boundaries and production capacity enforcement: **NOT TESTED**.

## PHASE E — DISPATCH

- Exact Qwen configuration enabled: **NO**.
- Current explicit unsupported-Qwen error: **PRESERVED**.
- Canonical model-runtime admission and whole-request wrapper were tested to preserve that fail-closed behavior.
- Qualification runner invoked by production: **NO**.

## PHASE F — QUALIFICATION VS PRODUCTION

- Exact artifact/capacity/chunk fixture: runner regression used capacity 40, 32-token prefill, 8 appends.
- Qualification prefill/replay/KV: **PASS** for the runner regression above.
- Production prefill, first token, 8/16-token trajectories, KV progression and sampled-cache comparison: **NOT TESTED** because the production Qwen core does not exist.
- Production-only divergence: **NOT MEASURABLE**.

## PHASE G — SESSION ISOLATION

- One shared Qwen runtime, two isolated Qwen sessions, interleaving, reset/destroy isolation: **NOT IMPLEMENTED / NOT TESTED**.
- Concurrent execution: **UNSUPPORTED / NOT QUALIFIED**.

## PHASE H — SHARED RESIDENCY

- Runtime-only / runtime+session A / runtime+sessions A+B VRAM measurements: **NOT TESTED**.
- Duplicate full weights or embedding per session: **NOT APPLICABLE YET**; canonical runtime has no Qwen CUDA residency.

## PHASE I — PRODUCTION RUNS

| Run | Status |
|---|---|
| 128 + 32 | **NOT TESTED** in canonical production session |
| 512 + 8 | **NOT TESTED** in canonical production session |
| 1024 + 8 | **NOT TESTED** in canonical production session; prior qualification-only capacity-1032 run remains valid only for that runner |

Production VRAM, decode rate, and H2D/D2H counters: **NOT TESTED**.

## PHASE J — RESET / LIFECYCLE

- Qwen reset reproducibility: **NOT TESTED**.
- Three production session cycles and VRAM trend: **NOT TESTED**.
- Qwen runtime teardown: **NOT TESTED**.
- Generic lifecycle remains previously qualified; it does not qualify Qwen device resources.

## PHASE K — `run()` COMPATIBILITY

- `runtime.run(Qwen request)`: intentionally returns the unchanged explicit unsupported-Qwen error.
- Temporary Qwen session/core path: **NO**, because no Qwen core exists.
- Qualification runner invoked: **NO**.
- Independent runtime recreation: generic Qwen metadata is attached to the supplied model runtime; CUDA model resources do not yet exist.

## PHASE L — REGRESSIONS

- Memory: old 409,728 B/token pathology was **not measured in production**. Qualification runner regression still uses the shared-scratch path and did not exhibit a change; production classification remains **NOT TESTED**.
- Transfers: production weights/embedding/historical-KV H2D and D2H, and inter-layer host staging: **NOT TESTED**.
- Performance: production @64/@128/@512: **NOT TESTED**; no optimization attempted.
- CPU Qwen: **NOT RERUN** in this iteration; prior CPU qualification unchanged.
- Q4_K/Q6_K focused controls: **NOT TESTED**.
- DeepSeek: lifecycle CTest regression and semantics contract passed; scope is the existing focused one-block/session regression, not full-model qualification.

## TESTS

- CUDA-enabled full build: **PASS** (pinned GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`).
- Full CTest: **38/38 PASS**, including opt-in exact-artifact admission and runtime fail-closed contracts; DeepSeek artifact-dependent lifecycle test used a local range source.
- Exact-artifact admission contract: **PASS**; identity/dimensions/catalog/tokenizer checks.
- Exact-artifact canonical runtime admission/fail-closed dispatch: **PASS**.
- Qwen qualification-runner regression, 32+8/capacity 40: **PASS** as detailed above.
- Qwen CPU contracts: relevant existing query-group/KV/execution-policy contracts passed; full real-artifact CPU inference **NOT RERUN**.
- Qwen residency/workspace/scratch/KV/prefill unit contracts: prior query-group/KV contracts and qualification-runner regression passed; no new production device-state contracts exist.
- Production session/reset/capacity/isolation/run: **NOT TESTED**.
- ASan/UBSan focused admission, runtime rejection and generic lifecycle tests: **PASS** (3/3). A zero-byte `memcpy` UBSan issue in the included generic DeepSeek helper was fixed by skipping the copy for empty buffers.
- CUDA qualification regression: **PASS**; production CUDA session: **NOT TESTED**.
- `git diff --check`: **PASS**.
- `ccc index`: **PASS** (see final run).

## REPOSITORY

- New shared metadata/admission module: `integrations/ggml/include/qwen3_model.h`, `integrations/ggml/src/qwen3_model.cpp`.
- Canonical model-runtime metadata binding and preserved rejection: `integrations/ggml/tools/autoregressive_poc22.cpp` (working-tree change; not included in the focused checkpoint).
- Qualification runner now uses shared Qwen model admission: `integrations/ggml/tools/qwen3_block_qualification.cpp`; the pre-existing qualification math remains there.
- Admission contracts: `integrations/ggml/tests/qwen3_model_admission_contract.cpp` is in the focused checkpoint; `integrations/ggml/tests/qwen3_runtime_admission_contract.cpp` remains an unstaged working-tree change.
- Build/test integration: `integrations/ggml/CMakeLists.txt`.
- Sanitizer fix: `integrations/ggml/tools/router_driven_moe_poc11.cpp` (empty-byte conversion guard).
- Existing dirty work, including generic lifecycle, runtime binding, residency and prior qualification evidence, was preserved outside the focused checkpoint.
- `stash@{0}`: untouched.
- `stash@{1}`: untouched.

## FINAL STATUS

- Shared Qwen model metadata/admission: **IMPLEMENTED / TESTED**.
- Shared Qwen CUDA runtime core: **BLOCKED / NOT IMPLEMENTED**.
- Qwen `VbufModelRuntime` CUDA weights/backend/residency: **NOT IMPLEMENTED**.
- Qwen `VbufGenerationSession` GPU KV/scratch/state: **NOT IMPLEMENTED**.
- Canonical production Qwen: **UNSUPPORTED**; production remains disabled.
- Production memory scaling/session isolation/context 1024/`run()`: **NOT TESTED**.
- Same-runtime concurrent sessions: **UNSUPPORTED / NOT QUALIFIED**.
- HTTP, tools, Pi: **NOT TESTED**.

## BLOCKED / NEXT INTERNAL GATE

The mandatory next step is still Phase A: extract the incremental resident CUDA graph/execution implementation into a production-neutral shared core with model-scope weights/backend/residency and session-scope independent KV, per-layer prefill scratch and decode workspace. The qualification runner must call that same core. Only then can dispatch be enabled and the production qualification matrix run. No protocol or agent integration gate was entered.
