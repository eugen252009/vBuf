# Qwen first guarded specialization qualification

**Outcome: numerically qualified; performance rejected.** The one behavior-changing decode candidate was implemented and exercised under explicit `ENABLED`/qualification control. It was bitwise equivalent to canonical execution, passed the guarded 32K mixed-plan run and lifecycle/failure tests, but did not show a repeatable end-to-end speedup. It is **not enabled by default**; ordinary runtime mode remains `SHADOW`, and a newly discovered candidate starts `Candidate`, not `Valid`.

## Baseline and scope

- Base HEAD: `f9ec242` — Qwen execution-plan shadow foundation.
- Model semantic SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Canonical placement: RTX 3060 embedding + blocks 0–25; RTX 2080 SUPER blocks 26–39 + final RMSNorm/head; pinned-host boundary.
- Production capacity remains **1,032**, single-GPU remains default, chunk remains **32**. No 25/15 placement or math/kernel changes.

### The one candidate

`prebound-decode-dispatch-v1` reuses a per-executor ordered table of the existing decode layer graph slots and their runtime-owned backend handles. The enabled loop still submits the same 40 layer graphs in the same order, performs the same control updates and pinned-host handoff, and uses the same session KV tensors. It removes repeated placement/device/backend/vector resolution from the per-layer loop; it does not fuse or alter any Qwen operation.

The candidate in the optimizer contains plan identity, strategy, lifecycle state, and guards—**no session/KV/graph pointers**. The pointer table is held only by the generation executor that owns a `shared_ptr` to its session/runtime, so it cannot be reused by another executor/session. Shadow and Disabled always execute canonical code. Enabled selects only a `Valid` candidate. The explicit unvalidated trial switch is a qualification test hook; the production default never uses it.

Guards are conjunctive and fail closed: 26/14 multi-GPU plan identity; Qwen3-14B model identity; decode phase; one row; context `[32,32767)`; capacity 32,768; chunk 32; F32 activations/F16 KV; exact 26/14 placement; both plan-bound device identities; and SM 86/75. A candidate must accrue at least 32 decode observations and receive an explicit validation note before `Candidate -> Valid`. It is still per-runtime and starts unvalidated on every new runtime.

The implementation is in `integrations/ggml/include/qwen3_execution_plan.h`, `src/qwen3_execution_plan.cpp`, and `src/qwen3_generation.cpp`; the prebound pointer table is built lazily on the first validated candidate selection, so Disabled and Shadow do not allocate it. The opt-in exact-artifact qualification path is in `qualification/qwen3_multigpu_capacity_gate.cpp`.

## Correctness and lifecycle

- **DISABLED vs SHADOW:** 32 generated tokens; tokens, final hidden, final logits, historical KV, and sampled appended KV were bitwise equal.
- **Canonical vs candidate trial:** 32 generated tokens; tokens, final hidden/logits, and sampled appended KV rows across six layers and all generated positions were bitwise equal. The candidate was marked Valid only after this comparison and the hotness threshold.
- **Valid ENABLED replay:** bitwise output replay passed; allocations were reused.
- **Guard transitions:** at context 31 the first generated row used canonical; after reaching context 32 the next row used prebound dispatch. At context 32 both generated rows used prebound dispatch. A/B/A repeated the context-31 request around the context-32 request and reproduced tokens/logits/hidden bitwise.
- **Upper transition:** the full-capacity run used the candidate through context 32,766 and returned to canonical at context 32,767.
- **Failure/deoptimization:** a pre-dispatch fault invalidated the candidate and used canonical for the current request. A separate injected failure before the final block graph invalidated the candidate and reset logical length to zero; the failed request aborted, and the retry used canonical successfully. Physical KV writes from the failed attempt are not claimed to be rolled back.
- **Construction/guard/lifecycle contracts:** the plan contract covers candidate-construction and guard faults, cache behavior, Candidate/Valid/Invalidated transitions, hotness threshold, unvalidated-Enabled rejection, bounded cache/profiler, and fail-closed accounting. The real gate exercises dispatch failure, invalidation, reset/recovery, and A/B/A.

## 32K mixed-plan qualification

The opt-in exact-capacity run used prefix 32,735 plus 33 generated positions and reached **32,768**. It compared a canonical SHADOW execution with the prebound candidate trial in the same runtime/session. Result:

- Candidate execution: **32 specialized + 1 canonical** decode steps; prefill/tail work remained canonical.
- Candidate/canonical tokens, final hidden/logits, and selected appended KV rows: exact match.
- Boundary handoffs: 1,086; selected boundary audits: **41**; audited bytes: **5,918,720**; all passed.
- Final length: 32,768; production capacity remained unchanged.
- At the upper guard, the final row fell back to canonical.

The raw records are `raw/prebound-mixed-32768-first-run.log` and `raw/prebound-mixed-32768.log` (the latter is the full-capacity run after lazy setup). The first run's older profile mixed in 31 single-row prompt-tail steps; the latest profile was corrected to count only the 33 generated decode steps. Both paired `decode_ns` timers span generated decode only. These remain observer-distorted single A/B pairs, not stable benchmarks.

## Performance decision

The pre-specialization host profile showed about 2.5–3.0 µs per token for repeated layer/device/backend/graph resolution against roughly 375 ms/token at 32K graph capacity. The specialized dispatch table reduced that measured lookup to about 0.9–1.0 µs/token: approximately **1.5–1.8 µs saved**, under 0.001% of end-to-end step latency. Enabled mode rechecks guards each token to catch interval transitions; cached selection measured about 6.3 µs/token, and Shadow pays essentially the same selection cost. Against Shadow, the net expected CPU saving is still only the ~1.5–1.8 µs layer-resolution delta. Disabled avoids most selector work (~0.75 µs/token), so Enabled guard/selection cost is larger than the dispatch saving relative to that bare path.

The final short-prefix comparison used three rotated Disabled/Shadow/Enabled cycles (96 generated-decode samples per mode), with bitwise tokens/logits/hidden/KV checks on every repeated run:

| Mode | Step median | Step p95 | Dispatch work |
|---|---:|---:|---|
| DISABLED | 378.835 ms | 384.245 ms | canonical |
| SHADOW | 373.146 ms | 376.323 ms | canonical |
| ENABLED | 372.953 ms | 379.554 ms | prebound |

Enabled was only 0.052% faster than Shadow at the pooled median and 0.86% slower at p95. The earlier 32-sample sessions varied materially in both directions, reinforcing that the apparent median deltas are not repeatable. In the final run, per-request plan-selection time was about 0.20 ms (roughly 6.3 µs/token), while replacing layer resolution saved about 1.6–1.8 µs/token; first-use pointer-table setup was below 1 µs and is lazy, so Disabled/Shadow pay no table-construction cost. The dispatch saving is not enough to offset per-token revalidation and establish an end-to-end benefit.

Two mixed-32K A/B runs measured 12,299,244,510 ns canonical vs 12,443,264,637 ns candidate (+1.17%) in the first run, then 12,632,946,881 ns canonical vs 12,613,829,570 ns candidate (-0.15%) in the final lazy-setup run. The canonical runs themselves differed by 2.7%, larger than either candidate delta. These runs occurred while the desktop RTX 2080 SUPER had variable background utilization (about 13–44%); they are observer-distorted and not a stable benchmark. The latest full-capacity run observed about 1,222 MiB minimum free on the RTX 2080 SUPER versus about 1,487 MiB in the frozen baseline qualification; placement/capacity/allocation policy did not change. Across pooled short-prefix and full-capacity results, **no repeatable speedup is demonstrated.** The candidate is therefore **performance-rejected** and must not be activated as a production/default candidate.

Raw mode comparison and lifecycle evidence: `raw/prebound-optimizer-modes-small-context.log` (includes the 96-sample rotated comparison) and `raw/prebound-graph-failure-recovery.log`. Earlier setup profiles at contexts 32, 1K, 8K, 16K, 24K, and near 32K remain in the other `raw/context-*-profile.log` files; those preceded the prebound implementation and are opportunity analysis, not a candidate benchmark.

## Verification

- CUDA build: execution-plan contract, CUDA ownership, model admission, runtime admission, multi-GPU capacity gate, and multi-GPU qualification targets built.
- Repeated mode experiment: three rotated cycles, 96 decode samples per mode; all repeated outputs and sampled KV remained bitwise equal to the qualified reference.
- Strict semantic-model/backend/device/SM/dtype/placement eligibility was tested with negative plan-contract cases, then requalified on the actual artifact and two GPUs (`raw/prebound-strict-identity-guard.log`).
- CUDA CTests: `vbuf_qwen3_execution_plan_contract`, `vbuf_qwen3_cuda_ownership_contract`, and `vbuf_qwen3_model_admission_contract` passed.
- Direct plan contract passed; direct runtime-admission contract confirmed missing-source fail-closed behavior.
- CPU build and execution-plan contract passed.
- `git diff --check` passed.

## Deferred follow-ups

No fusion, value/range specialization, alternate kernel, JIT, or placement change was implemented. Review-only fusion candidates remain: attention RMSNorm/weight multiply into Q/K/V projection; attention projection plus residual add; and FFN gate/up, SiLU×up, and down projection. Each risks changed rounding/reduction behavior or temporary pressure and requires its own parity/performance gate.

Current guards support scalar comparisons and strict intervals. Future facts may include `all_finite` and tile min/max, but must remain tensor/row/tile/head/block granularity, never per-element branching. The 25/15 split remains unselected.

## Final state

- Candidate numerical status: **VALIDATED** for exact output/selected-KV parity.
- Candidate performance status: **REJECTED / NOT FOR DEFAULT USE**.
- Default optimizer mode: **SHADOW**; canonical selection remains authoritative.
- Candidate selected by default: **NO**.
- Speedup claimed: **NO**.
- Commit/push: **none**. The branch changes remain uncommitted pending the performance-rejection disposition; protected stashes and unrelated state were not touched.
