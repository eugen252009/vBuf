# Qwen3 Canonical Runtime / Session Integration — Historical Checkpoint

**Checkpoint scope:** the focused admission commit contains the shared Qwen model-admission module, its qualification-runner adoption and admission contract. The canonical-runtime binding, runtime-admission contract, and generic runtime/session ownership changes remain in the dirty worktree and are not part of that checkpoint. They are preserved for the next integration step.

**Historical status:** this report records the state before the qualification-runner ownership migration and canonical production binding. Its later-phase `NOT IMPLEMENTED` and `DISABLED` statements are superseded by [`qwen3-vbuf-generation-session-production-qualification.md`](qwen3-vbuf-generation-session-production-qualification.md). Current qualification is limited to the exact admitted artifact, tested CUDA path, and session capacities through 1032; concurrency and later integrations remain unqualified.

## BASELINE

- Artifact payload: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf`
- Semantic bootstrap: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`
- SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31` (**verified**)
- Backend/device: qualification runner GGML CUDA, RTX 3060 12 GB (CUDA 12.4, SM 8.6); a second RTX 2080 SUPER was present but not used.

## GENERIC ARCHITECTURE

- `VbufModelRuntime`: prior generic lifecycle **QUALIFIED**; this iteration attaches shared Qwen validated metadata/catalog to its lifetime. CUDA model resources are **NOT IMPLEMENTED**.
- `VbufGenerationSession`: prior generic lifecycle **QUALIFIED**; Qwen-specific mutable state exists as a separate unbound `QwenCudaSessionState`; canonical binding is **NOT IMPLEMENTED**.
- Concurrent same-runtime execution: **UNSUPPORTED / NOT QUALIFIED** (unchanged).

## PREVIOUS QWEN QUALIFICATION

- Resident CUDA, persistent KV, 32-token bounded prefill, per-layer scratch reuse and memory scaling remain qualified for the qualification-only implementation as recorded in `qwen3-resident-cuda-performance-and-production-text-qualification.md`.
- Historical 1024+8 qualification-only run: context 1032; approximately 3.20 GB free VRAM after chunks; sampled KV checks passed.
- After extracting the shared layer graph, the qualification runner was rerun on the exact artifact at capacity 40 with a 32-token prompt plus 8 appends, chunk size 32, per-layer scratch reuse, and sampled KV integrity. It passed: fixed layer scratch buffer, absolute position/mask/cache-row checks, sampled historical K/V immutability, nonzero sampled new rows, and repeatable generated tokens across three runs. The timed warm decode was about 33.3 tok/s on the RTX 3060; this was a qualification-runner regression, not production-session evidence. The performance-mode run did not compare full logits bit-for-bit.

## KNOWN NUMERICAL STATUS (PRESERVED)

- Incremental/full CUDA: **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**.
- Chunked/full prefill: **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**.
- CPU/CUDA prefix 25: **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**.
- External llama.cpp strict `1e-5` gate: **FAIL**.
- No CPU/CUDA or llama.cpp parity claim is made.

## PHASE A — SHARED QWEN CORE

### Extracted modules

- `integrations/ggml/include/qwen3_model.h`
- `integrations/ggml/src/qwen3_model.cpp`
- `integrations/ggml/include/qwen3_cuda_core.h`
- `integrations/ggml/src/qwen3_cuda_core.cpp`

The committed model module shares semantic-bootstrap opening, architecture/metadata validation, dense tensor-inventory admission, source SHA-256 identity, tokenizer BOS/EOS and `add_bos` metadata, and source/materializer descriptor setup. The CUDA graph helper extracts the transformer-block graph (RMSNorm, Q/K/V, RoPE, GQA attention, output/residual, FFN/SwiGLU, and cache-row update/read). Separate `QwenCudaRuntimeState` and `QwenCudaSessionState` ownership objects now exist, but the qualification runner still uses its legacy locally owned resources and production does not call these objects.

### Model operation implementation and qualification-only remainder

| Operation / concern | Current owner/status |
|---|---|
| RMSNorm, Q/K/V, RoPE, GQA attention, O projection/residual, FFN/SwiGLU | Block graph construction shared through `qwen3_cuda_build_layer`; complete runtime execution remains qualification-only |
| Final norm, LM head, logits/argmax | **Qualification-only** in `qwen3_block_qualification.cpp` |
| KV allocation, logical positions, capacity, lifecycle | Ownership exists in `QwenCudaSessionState`; qualification execution remains runner-owned; row updates/attention reads are in the shared block graph helper |
| Bounded 32-token prefill orchestration and per-layer scratch reuse | **Qualification-only**; layer graph construction uses the shared helper |
| Single-token decode orchestration and logits | **Qualification-only**; layer graph construction uses the shared helper |
| Memory tracing, checkpoint capture, CPU/CUDA comparisons, fixture/oracle work, diagnostic readbacks, CUDA audit hooks, report formatting | **Qualification-only** |

### Shared operations / duplicate model math

- Shared block graph construction: RMSNorm, Q/K/V, RoPE, GQA attention, O projection/residual, FFN/SwiGLU, and KV row update/read are implemented by `qwen3_cuda_build_layer` and called by the qualification runner's incremental decode and layered chunked-prefill paths.
- Ownership now exists outside the runner for backend/model weights/device residency and per-session KV/scratch/logical length. Graph/session orchestration, use of those resources, chunk scheduling, decode loop, and logits/argmax orchestration remain qualification-only.
- Duplicate implementations remaining: the independent full-batch resident CUDA comparator and CPU/oracle graph paths still build related model math separately. They remain diagnostic/reference controls, not production dispatch. Production currently has no Qwen execution implementation.
- Result: shared layer-graph construction and Qwen CUDA ownership objects are **IMPLEMENTED**; the ownership lifecycle is **TESTED** on synthetic CPU resources. Runner migration, real-CUDA resource smoke, and canonical production integration remain **BLOCKED / NOT COMPLETE**.

## PHASE B — QWEN MODEL RUNTIME

### Model admission

For the exact semantic bootstrap, shared admission validated architecture `qwen3`, 40 layers, hidden 5120, 40 attention heads, 8 KV heads, head dimension 128, FFN 17408, vocabulary 151936, 443 tensor entries, dense tensor shapes and Q4_K/Q6_K representation contract, source SHA-256, BOS/EOS IDs, and `add_bos=NO`. The artifact has separate embedding/output tensors with matching vocabulary geometry. Exact production identity admission rejects any other source SHA.

Unsupported Qwen variants are rejected by the exact-identity admission gate when using canonical Qwen metadata setup. Synthetic alternate-variant end-to-end admission fixtures were **NOT TESTED**.

### Backend and residency

- `QwenCudaRuntimeState` creates/owns the GGML GPU backend, all admitted model tensors including embedding, their shared device backing, and residency records. Exact-artifact construction on CUDA0 passed: 443 tensors / 8,995,793,920 resident payload bytes.
- Sequential-session sharing is exercised with the synthetic CPU fixture; real-GPU repeated-session upload behavior is **NOT TESTED**.
- Runtime/session resource creation and teardown passed an exact-artifact CUDA smoke at capacity 40; no graph inference or transfer-counter audit was run. Partial-construction cleanup is lifecycle-tested on the CPU fixture.

The qualification runner still creates/uploads its own weights and destroys them after its test operation; it has not migrated to `QwenCudaRuntimeState`. Prior warm decode transfer claims remain qualification-only evidence, not a result from the new ownership objects.

## PHASE C — QWEN SESSION STATE

- `QwenCudaSessionState` allocates per-session K/V, packed-V, prefill scratch, and decode scratch; runner graph binding/use is **NOT IMPLEMENTED**.
- Logical `current_length`, bounded commit, and logical reset are **IMPLEMENTED / TESTED** with the synthetic CPU fixture.
- Distinct session tensor/scratch ownership, session teardown, and partial-create cleanup are **TESTED** with the synthetic CPU fixture.
- Capacity-40 real CUDA KV/scratch allocation and teardown smoke: **PASS**; KV visibility/reset through executed graphs is **NOT TESTED**.

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

- Two session objects against one runtime, distinct KV/scratch/logical state, reset and destruction: **TESTED** on synthetic CPU resources; GPU interleaving: **NOT TESTED**.
- Concurrent execution: **UNSUPPORTED / NOT QUALIFIED**.

## PHASE H — SHARED RESIDENCY

- Runtime-only / runtime+session A / runtime+sessions A+B GPU memory measurements: **NOT TESTED**.
- Synthetic CPU contracts confirm model residency belongs only to the runtime; actual CUDA allocation sharing is **NOT TESTED** and the qualification runner has not migrated.

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
- Temporary Qwen production execution path: **NO**; ownership objects are not attached to canonical runtime/session execution.
- Qualification runner invoked: **NO**.
- Independent runtime recreation: generic Qwen metadata is attached to the supplied model runtime; Qwen CUDA ownership objects remain separate and unbound.

## PHASE L — REGRESSIONS

- Memory: old 409,728 B/token pathology was **not measured in production**. Qualification runner regression still uses the shared-scratch path and did not exhibit a change; production classification remains **NOT TESTED**.
- Transfers: production weights/embedding/historical-KV H2D and D2H, and inter-layer host staging: **NOT TESTED**.
- Performance: production @64/@128/@512: **NOT TESTED**; no optimization attempted.
- CPU Qwen: **NOT RERUN** in this iteration; prior CPU qualification unchanged.
- Q4_K/Q6_K focused controls: **NOT TESTED**.
- DeepSeek: lifecycle CTest regression and semantics contract passed; scope is the existing focused one-block/session regression, not full-model qualification.

## TESTS

- CUDA-enabled full build: **PASS** (pinned GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`).
- Full CTest: **39/39 PASS**, including the new Qwen CUDA ownership lifecycle contract and opt-in exact-artifact admission/runtime fail-closed contracts; DeepSeek artifact-dependent lifecycle test used a local range source.
- Exact-artifact admission contract: **PASS**; identity/dimensions/catalog/tokenizer checks.
- Exact-artifact canonical runtime admission/fail-closed dispatch: **PASS**.
- Qwen qualification-runner regression, 32+8/capacity 40: **PASS** as detailed above.
- Qwen CPU contracts: relevant existing query-group/KV/execution-policy contracts passed; full real-artifact CPU inference **NOT RERUN**.
- Qwen CUDA ownership lifecycle contract: **PASS** on CPU-backed synthetic resources; exact-artifact CUDA backend/residency/embedding/KV/scratch construction and teardown smoke: **PASS** (inference not run).
- Focused ownership ASan/UBSan contract: **PASS** on CPU-backed synthetic resources.
- Production session/reset/capacity/isolation/run: **NOT TESTED**.
- ASan/UBSan focused admission, runtime rejection and generic lifecycle tests: **PASS** (3/3). A zero-byte `memcpy` UBSan issue in the included generic DeepSeek helper was fixed by skipping the copy for empty buffers.
- CUDA qualification regression: **PASS**; production CUDA session: **NOT TESTED**.
- `git diff --check`: **PASS**.
- `ccc index`: **PASS** (see final run).

## REPOSITORY

- Shared metadata/admission module: `integrations/ggml/include/qwen3_model.h`, `integrations/ggml/src/qwen3_model.cpp`.
- Uncommitted shared CUDA layer-graph helper: `integrations/ggml/include/qwen3_cuda_core.h`, `integrations/ggml/src/qwen3_cuda_core.cpp`; `qwen3_block_qualification.cpp` calls it for incremental decode and per-layer prefill.
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
- Shared Qwen CUDA layer-graph helper: **IMPLEMENTED / TESTED** for the qualification runner's decode and bounded prefill graph construction.
- Qwen-specific CUDA ownership objects: **IMPLEMENTED / TESTED** on a CPU-backed lifecycle fixture.
- Exact-artifact CUDA runtime/session resource smoke: **PASS**; inference through the new objects: **NOT TESTED**.
- Qwen `VbufModelRuntime` binding: **NOT IMPLEMENTED**.
- Qwen `VbufGenerationSession` binding: **NOT IMPLEMENTED**.
- Canonical production Qwen: **UNSUPPORTED**; production remains disabled.
- Production memory scaling/session isolation/context 1024/`run()`: **NOT TESTED**.
- Same-runtime concurrent sessions: **UNSUPPORTED / NOT QUALIFIED**.
- HTTP, tools, Pi: **NOT TESTED**.

## BLOCKED / NEXT INTERNAL GATE

The mandatory next step is to migrate the qualification runner's backend/residency/embedding/KV/scratch resources to `QwenCudaRuntimeState` and `QwenCudaSessionState`, bind its existing shared layer graphs to those allocations, and qualify real CUDA construction/teardown. Canonical runtime/session binding and production dispatch remain later gates. No protocol or agent integration gate was entered.
