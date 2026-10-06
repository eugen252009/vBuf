# Qwen3 CUDA execution-plan shadow foundation

Status: **Stage 2 foundation implemented and qualified in shadow/no-op mode.** This adds plan identities, fail-closed candidate guards, bounded profiling/cache state, and runtime observation above the frozen canonical 26/14 path. It does not execute an alternate plan or change Qwen math, placement, chunking, or capacity policy.

## Design boundary

- Stable plan identity includes the semantic artifact identity, backend family, exact contiguous block placement, embedding/norm/head owners, capacity class, prefill chunk, activation/KV dtypes, and stable device identity/SM facts. The plan also represents the pinned-host boundary explicitly.
- Shadow mode is the default. Disabled mode remains selectable for qualification. Both modes retain canonical selection; candidate eligibility is observation only.
- The registered 26/14 candidate is a semantic no-op naming the same canonical plan. It is guarded to one-row decode, contexts `[8192, 16384)`, capacity 32,768, chunk 32, exact 26/14 placement, dtypes, and device identities.
- The 64-entry candidate cache and 128-key profiler are bounded. Runtime facts and hotness are grouped by plan, phase, row bucket, and context bucket. Invalidation and optimizer failure accounting are explicit; unexpected optimizer failures fail closed to canonical selection.
- No fusion, JIT, alternate kernels, dynamic placement, 25/15 split, range-based math, or candidate execution was introduced. The canonical single-GPU default and production capacity 1,032 remain unchanged.

## Qualification results

### Contract and build tests

`vbuf_qwen3_execution_plan_contract` passed. It checks stable single/26/14 plan identities and stage layout; shadow default and disabled behavior; phase/row/context/capacity guards; 32K guard rejection; candidate eligibility; cache hits/misses and bounded eviction; invalidation; bounded profiles; unsupported facts; and injected candidate/cache/guard/profiler failures falling back safely.

CUDA-enabled qualification targets built against the pinned ggml commit, and the execution-plan contract also built and passed with CUDA disabled. `vbuf_qwen3_cuda_ownership_contract` and `vbuf_qwen3_model_admission_contract` passed under CTest. Direct `vbuf_qwen3_runtime_admission_contract` passed the missing-source fail-closed check.

### Real small-context shadow/disabled comparison

The Qwen3-14B 26/14 qualification ran with synthetic token IDs `1..32`, capacity 40, and prefixes 1, 2, 4, 8, 16, and 32. Greedy tokens matched the existing canonical single-device reference across all six prefixes and the declared test-specific hidden/logit tolerances passed. For the 32-token prefix, optimizer-disabled and shadow runs were **bitwise identical** for generated tokens, final hidden, logits, and sampled early/late-device KV rows.

The real runtime reported shadow mode, `canonical_selected=YES`, zero optimizer errors, 37 guard failures, and no guard passes because this small-capacity qualification intentionally does not satisfy the 32,768-capacity candidate guard. This is expected fail-closed behavior, not a numerical or execution failure.

### 32K guard scenario

The capacity gate ran the natural-repeat input at capacity 32,768 with prefix 32,736 plus 32 decode steps. The shadow observer found the cached candidate, rejected it at the 32K context guard (`eligible=NO`, `fallback=GuardFailed`), and retained canonical selection. The request reached logical length 32,768; 1,055 boundary handoffs and all 41 selected byte audits passed. Historical KV checks, finite output/KV checks, exact-fit admission, overflow rejection before execution, and recovery passed.

This run measured minimum free VRAM of 2,692,874,240 bytes on the RTX 3060 and 1,161,035,776 bytes on the RTX 2080 SUPER. The observed variation is retained as run-specific evidence; this is not a new capacity recommendation or performance comparison.

## Evidence

- `raw/execution-plan-contract.log` — deterministic optimizer/plan contract result.
- `raw/small-context-shadow-vs-disabled.log` — CUDA single-/multi-device small-context qualification and bitwise shadow-disabled comparison.
- `raw/capacity-32768-shadow-guard.log` — exact-capacity 32K execution and fail-closed shadow guard.
- The exact natural-repeat token input remains in `../qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv` (SHA-256 `23891eeb02382f8c8d517ccd0d039bbd3e3c72f6ef428ed13946798279a6ad44`).

Log SHA-256 values:

```text
execution-plan-contract.log         86831c29ffcfad23067808aeec7477f06f5f44554c4090c18b8f04db91684b7a
small-context-shadow-vs-disabled.log 3ffb1d5709f3917b89bc6c908ea6c2edd85ef586363e2a8a3bf16f783b054b0d
capacity-32768-shadow-guard.log      6a427e1c121fecfd83baa31db85ea235edf1b96c2fd8b73756a41ba551f9e9ec
```

## Boundary and next work

This is optimizer infrastructure qualification only. No speedup is claimed; shadow-observer overhead was not measured as a comparable benchmark. The 8K-context candidate guard pass is covered by the contract test, while the real 32K run confirms rejection. Before any behavior-changing specialization, add end-to-end qualification for the specific alternative, prove optimizer-disabled/shadow/candidate numerical equivalence where applicable, and preserve canonical fallback and the frozen production policy.
