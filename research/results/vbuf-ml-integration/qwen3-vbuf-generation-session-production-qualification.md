# Qwen3 VbufGenerationSession Production Qualification

**Status: EXACT-ARTIFACT CANONICAL CUDA EXECUTION QUALIFIED; production dispatch enabled only for the admitted Qwen3-14B Q4_K_M artifact.** This report supersedes the earlier blocked audit below. Qualification ran through `VbufModelRuntime` / `VbufGenerationSession`, not by invoking the qualification executable as a production shortcut. HTTP service integration, native tools, Pi, dogfood, and llama.cpp work were not started.

## Scope and exact artifact

- Payload: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf`.
- Semantic bootstrap: `/home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf`.
- Payload SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31` (verified immediately before the CUDA run).
- GGML: pinned commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.
- Device: RTX 3060 12 GB, CUDA 12.4, SM 8.6; CUDA device 0 only.
- Source: local HTTP range server over that exact payload.
- Admission remains exact-identity/40-layer only; other Qwen variants remain rejected. Whole-request `VbufModelRuntime::run()` is retained.

## Implementation

`QwenCudaRuntimeState` is attached to `VbufModelRuntime` and owns the backend, admitted model tensors, and device residency. `QwenCudaSessionState` is attached to each `VbufGenerationSession` and owns session KV, graph/storage contexts, scratch, capacity, and logical length. `Qwen3GenerationExecutor` drives the shared layer builder for bounded 32-token prefill and single-position decode, returns final logits, and records GGML backend tensor-transfer counters. Session reset clears logical visibility without discarding runtime weights.

The capacity-40 prefill scratch allocation (40,206,464 bytes) failed closed at capacity 160: the graph requested 41,926,784 bytes. Session prefill scratch was raised to 64 MiB. This supported the tested capacities through 1032. The current executable capacity limit is 1032; larger inference capacities remain unsupported and must not be inferred from allocation-only probes.

## Exact-shape production vs qualification-runner comparison

The migrated qualification runner's exported final logits for the exact 32-token prompt plus 8 generated tokens (`run-1-logits-prefix-40.f32`) were compared with final logits captured from canonical production. Both vectors contained 151,936 float32 values:

- Generated token sequence: `220,16,13,15,13,15,198,262` in both paths.
- Final logits: **bit-identical**, max absolute difference `0.0`, mean absolute difference `0.0`.
- Production reset/replay: token sequence and all final-logit bytes were also identical.

This is a same-artifact, same-shape CUDA execution comparison, not CPU/CUDA or llama.cpp parity. Existing numerical classifications remain unchanged: incremental/full CUDA **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**; chunked/full prefill **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**; CPU/CUDA prefix 25 **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**; external llama.cpp strict `1e-5` gate **FAIL**.

## Production workload results

All listed requests completed with finite final logits and the expected final session length. Model residency remained 443 tensors / 8,995,793,920 bytes through all requests; no model reupload was observed.

| Request | Prefill | Decode | Decode rate | Peak VRAM | Free VRAM after run | Session H2D | Session D2H |
|---|---:|---:|---:|---:|---:|---:|---:|
| 32 + 8 | 179.5 ms | 268.5 ms | 29.80 tok/s | 9,254,141,952 B | 3,368,026,112 B | 36 calls / 8,000 B | 10 calls / 5,469,700 B |
| 128 + 32 | 228.6 ms | 999.4 ms | 32.02 tok/s | 9,403,039,744 B | 3,219,128,320 B | 144 / 108,800 B | 40 / 21,878,800 B |
| 512 + 8 | 1,000.9 ms | 266.8 ms | 29.99 tok/s | 9,528,868,864 B | 3,093,299,200 B | 96 / 1,102,400 B | 40 / 14,585,920 B |
| 1024 + 8 | 2,125.4 ms | 279.9 ms | 28.58 tok/s | 9,705,029,632 B | 2,917,138,432 B | 160 / 4,301,376 B | 72 / 24,309,888 B |

Values are from the production qualification run; timing is one local CUDA run and is not a general performance guarantee. Transfer counts/bytes cover GGML backend tensor set/get calls, not a lower-level CUDA API trace. The simultaneous lifetime of two session objects was exercised sequentially (not concurrent graph execution); separate output/KV logical lifetimes and reset/destruction behavior were checked. Same-runtime concurrent inference remains **UNSUPPORTED / NOT QUALIFIED**.

## Lifecycle and compatibility checks

- 32+8 generated sequence matched the migrated runner; production reset/replay was bit-identical in tokens and logits.
- Two sessions shared one model runtime and retained independent logical lengths. Resetting/destroying session B did not change session A. No concurrent inference was performed.
- `VbufModelRuntime::run()` matched session execution for tokens and final logits and reused the same 443 resident tensors without reupload.
- An over-capacity request failed closed with logical session length zero; a capacity-1033 inference request also failed closed. Capacities through 1032 passed the 128+32, 512+8, and 1024+8 workload checks.
- Runtime residency/upload counters remained stable after creation; session transfer counters and VRAM snapshots are reported above.
- Missing-source admission continues to fail closed. Partial-construction ownership contracts and generic lifecycle tests remain in the test suite.

## Verification

- CUDA-enabled full build: **PASS**.
- Full CTest: **39/39 PASS**.
- Manual exact-artifact canonical CUDA qualification: **PASS** for 32+8 parity, reset/replay, sequential two-session isolation, `run()` compatibility, 128+32, 512+8, and 1024+8.
- `git diff --check`: **PASS**.
- `ccc search` timed out; a successful post-change `ccc index` has not been recorded in this turn.
- True concurrent session inference, capacity above 1032, sanitizer execution of CUDA graphs, non-local/remote source qualification, cross-device/platform qualification, CPU numerical parity, and llama.cpp comparison were **NOT TESTED**.

## Repository / safety notes

The worktree contains unrelated pre-existing and overlapping edits alongside this implementation. No blanket staging, stash operation, reset, or destructive Git operation was used. The three protected commits remain untouched; no commit was made and nothing was pushed. The protected stashes were not accessed.

## Final disposition

- Canonical Qwen3 execution: **ENABLED only for the exact admitted Qwen3-14B Q4_K_M identity**, on the tested CUDA path and supported session capacities through 1032.
- Other Qwen variants, capacities above 1032, and same-runtime concurrent inference: **REJECTED or UNSUPPORTED / NOT QUALIFIED**.
- HTTP, tools, Pi, dogfood, and llama.cpp benchmarking: **NOT STARTED**, as requested.
