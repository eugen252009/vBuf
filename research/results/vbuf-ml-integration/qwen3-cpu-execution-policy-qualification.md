# Qwen3 Isolated CPU Execution Policy Qualification

**Status:** correctness qualified for the tested isolated CPU qualification path; narrowly performance qualified for the 32-position workload at a four-thread budget. This is not a production-runtime qualification.

## Scope and boundary

This change adds a vBuf-owned execution policy only to `integrations/ggml/tools/qwen3_block_qualification.cpp`. It does not enable Qwen3 in `VbufGenerationSession`, change the generic vBuf format, change the strict llama.cpp comparison tolerance, or change tensor/source ownership. Qwen3 remains disabled in production. The earlier external Q4_K_M-vs-llama.cpp `1e-5` parity failures remain failures; this work did not waive or rerun that gate.

The policy uses the existing `BoundedExecutor` (`integrations/ggml/include/vbuf_parallel_executor.h` and `src/vbuf_parallel_executor.cpp`) rather than introducing another worker pool. GGML CPU thread counts are set on the independent projection contexts/backends; no global GGML thread-count state is changed. A scoped thread budget is passed into each projection task.

Before this change, Q/K/V and FFN gate/up matmuls were nodes in a synchronous GGML graph. Their backend kernels used a configured thread count, but the vBuf qualification path did not overlap those independent projection branches. The revised path computes the shared normalization first, runs independent projections under the vBuf scheduler, then copies their results into the downstream graph. Attention, residual/SwiGLU ordering, and downstream execution remain ordered as before.

## Policy and CLI

- Default: parallel branch scheduling, total CPU budget **4**.
- `--serial`: serial graph tasks and one kernel thread.
- `--execution serial --threads N`: serial graph tasks with an N-thread GGML kernel budget.
- `--execution parallel --threads N` or `--threads N`: parallel branch scheduling under total budget N.
- `VBUF_QWEN_THREADS` remains supported when there is no CLI thread override; explicit CLI settings take precedence. Accepted budgets are 1–128.

Q/K/V and gate/up budgets are deterministically partitioned. For example, parallel budget 4 assigns Q/K/V `2,1,1` and gate/up `2,2`; budget 8 assigns Q/K/V `4,2,2` and gate/up `4,4`. With parallel budget 1, batches are necessarily serial. Assigned active kernel budgets are checked against the global budget before dispatch. The bounded executor drains a batch before proceeding, and exceptions propagate through the existing executor behavior.

Per-run metrics include execution mode, total budget, scheduled task count, peak simultaneous executor workers/tasks, active assigned kernel-thread budget, and a histogram of per-task budgets. `oversubscription=NO_BY_ASSIGNED_BUDGET` means the scheduler's assigned kernel-thread sums stayed within the configured budget; OS-wide/native helper thread counts were not independently sampled.

## Correctness evidence

The real Qwen3-14B Q4_K_M artifact was run through the isolated 40-layer canonical full-sequence path with the same 32-token teacher-forced sequence. It ran in internal-only qualification mode; this isolates execution-policy comparison from the already-failing external llama.cpp numerical gate.

Binary exports were compared across serial one-thread, serial eight-thread, and parallel eight-thread executions:

- Serial one-thread and parallel eight-thread were bit-identical for **85 F32 exports**: final hidden state, final normalization, logits, all 40 layer outputs, and 42 intermediate checkpoints (layers 0, 20, and 39).
- Both compared paths had identical top-1 outputs at all positions and identical full-sequence F16 K/V rows: **2,560 rows** (40 layers × 32 positions × K/V).
- Serial eight-thread and parallel eight-thread were also bit-identical for final hidden state, normalization, logits, all 40 layer outputs, and all 2,560 K/V rows. This comparison holds the kernel-thread budget constant while changing graph scheduling.
- The existing append-only `Qwen3KvCache` incremental-replay/full-cache check passed for a one-token real-artifact run under serial one-thread and parallel eight-thread policies. Its incremental K/V export was bit-identical between the two policies. Direct persistent-cache execution was not extended to a 32-token run in this change.

The comparisons establish bit identity for these inputs/configurations, not universal determinism for every CPU, GGML build, or model. No tolerance or assertion was relaxed.

## CPU and performance measurements

Measurements used the pinned GGML checkout `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, CPU AVX2+FMA, the real Qwen3-14B Q4_K_M artifact, and the local loopback vBuf range source. Full qualification timings include the existing materialization/validation and model work, not just matmul time. Repeated measurements reused the loaded model/process and are warm-process comparisons; they are not transport-ceiling measurements.

For 32 teacher-forced positions through all 40 layers, observed warm-process totals were:

| Policy | Warm full-pass timings | Mean |
|---|---:|---:|
| Serial graph, 1 kernel thread | 52.2 s (single warm measurement) | — |
| Serial graph, 2 kernel threads | 44.6 s | — |
| Parallel graph, total 2 threads | 44.3 s | — |
| Serial graph, 4 kernel threads | 39.606, 41.223, 40.369, 40.019 s | 40.304 s |
| Parallel graph, total 4 threads | 39.425, 40.395, 39.862, 39.427 s | 39.777 s |
| Serial graph, 8 kernel threads | 56.7 s | — |
| Parallel graph, total 8 threads | 57.6 s | — |

At budget 4, parallel graph scheduling was faster in each of the four warm pair-index comparisons, with a mean difference of about **0.53 s (1.3%)** against serial graph scheduling at the same total kernel budget. This is a modest, workload- and machine-specific improvement, not a broad speedup claim. The 2-thread difference was about 0.6%; the 8-thread parallel result was about 1.4% slower than serial at that budget. Increasing total threads is not monotonically beneficial.

For smaller full-model sequence lengths, serial-graph/4-thread versus parallel/4-thread results were approximately 30.85/30.11 s at 1 position, 32.49/31.97 s at 8 positions, and 34.84/34.83 s at 16 positions. The one-position serial/one-thread warm run took 33.48 s. These timings reinforce that the branch-level contribution is small and varies with workload; most observed improvement over one-thread serial execution comes from giving GGML kernels more CPU budget, not from graph overlap alone.

At parallel budget 4 the scheduler reported at most three simultaneous branch tasks/workers and no assigned-budget excess; task groups used Q/K/V budgets `2,1,1` and gate/up `2,2`. The measured full-pass time and exact-output comparisons establish a limited correctness/performance result for this CPU path only.

## Verification

- Pinned-GGML release build of the Qwen qualification executable and execution-policy contract: passed.
- Full pinned-GGML CTest suite: **34/34 passed**, including executor, Qwen cache/query-group/softmax/policy, and DeepSeek semantics contracts.
- Focused ASan/UBSan CTest contracts: **5/5 passed** (`parallel_executor`, Qwen policy, KV cache, softmax extent, query groups).
- ASan/UBSan real-artifact run: one position through all 40 layers, parallel budget 4, internal-only mode: `canonical_full_reference_only=PASS`; no AddressSanitizer or UBSan diagnostic. Leak detection was disabled for this run.
- `git diff --check`: passed.

The full sanitizer CTest suite was not run. The earlier recorded strict external numerical gate, task-quality suite, production session/HTTP/tool path, and Android build retain their existing qualification status; this report does not change them.
