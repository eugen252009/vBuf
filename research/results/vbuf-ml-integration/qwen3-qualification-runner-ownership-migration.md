# Qwen3 Qualification Runner Ownership Migration

**Status: qualification-only migration; production Qwen remains `UNSUPPORTED / DISABLED`.**
Canonical `VbufModelRuntime` / `VbufGenerationSession` binding is not implemented.

## Scope

The resident full-batch and incremental/generated CUDA qualification paths now use
`QwenCudaRuntimeState` for the backend, admitted model tensor bindings, model
allocation, and residency, and `QwenCudaSessionState` for graph storage, KV,
packed-V, scratch allocations, and logical context length. The runner no longer
creates its own CUDA backend or `TensorResidencyStore` in these paths, stages
model weights a second time, or zeroes KV buffers for reset. The existing
qualification graph and diagnostic/oracle code remains qualification-only; this
change did not add Qwen production dispatch or another production model-math
path.

Sessions retain their runtime. Reset clears logical length while preserving the
physical buffers. Decode/prefill commits advance session length with checked
capacity bounds. The qualification runner still does not claim same-runtime
concurrent inference support.

## Exact-artifact CUDA results

Artifact: `Qwen_Qwen3-14B-Q4_K_M.vbuf`, SHA-256
`f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
The semantic bootstrap SHA-256 was
`cc20b816a4cfdd192d4870b853354c51dd1b1402b83d01fac1e8ea98f2a9fea7`.
Runs used CUDA0, an RTX 3060 12 GB, with the pinned local GGML checkout and a
local range server. No alternate checkpoint was used.

- **32+8 incremental generation:** 32-token chunked prefill followed by 8
  incremental decode positions, three complete replays. All generated token
  sequences matched exactly (`220,16,13,15,13,15,198,262`); the non-performance
  run also passed bit-identical replay checks. Logical length reached 40 in each
  replay.
- **KV integrity:** cache prefixes 32, 33, and 40 were audited at layers 0, 20,
  29, and 39. Newly written K/V rows were nonzero; prior rows remained bytewise
  unchanged; row ranges and logical positions matched. Reset did not memset KV.
- **Runtime/session lifecycle:** with the same real-artifact runtime, a second
  live session had distinct K/V storage; after destroying the first session, a
  sequential session allocated and reset successfully. Runtime model upload
  counters remained unchanged. Neither the simultaneous nor sequential probe
  ran a model graph; this establishes real-CUDA allocation/reset isolation, not
  per-session inference or concurrent execution.
- **Model transfers:** runtime registered 443 model tensor uploads totaling
  8,995,793,920 payload bytes once. Across the three incremental replays the
  runner counted 108 session H2D tensor-set calls / 24,000 bytes and 177 D2H
  tensor-get calls / 22,676,064 bytes. D2H includes explicit KV-integrity and
  qualification readbacks, not just generated-token reads. No post-creation
  weight tensor-set calls or reset cache memsets occurred. These are
  GGML-backend tensor API counters; the optional lower-level CUDA API snapshot
  reported `available=NO`, so no independent DMA/API-trace audit is claimed.
- **Performance-mode replay:** warmup plus two timed replays measured mean
  decode 30.185 ms/token (33.129 tokens/s); token identity only was checked in
  that performance-mode run. The separate non-performance replay established
  bit-identical outputs.
- **Full-batch path:** all 40 layers, 32 positions, three CUDA prefill runs
  completed with finite final logits. The two timed runs averaged 54.185 ms.
  This run was performance-only, not a CPU numerical-parity qualification.
- **Allocation-only capacity probe:** at capacity 4096, the real model runtime
  and session allocations succeeded: model 8,995,793,920 bytes; session KV
  allocation 679,477,248 bytes (including packed-V); decode graph scratch
  747,796,480 bytes; prefill scratch 40,206,464 bytes. **No inference was run**
  at this capacity. With chunked prefill enabled at this capacity, construction
  failed closed because the current fixed prefill scratch was smaller than the
  capacity-dependent layer plan. Long-context chunked-prefill execution remains
  unqualified.
- **Teardown:** after each real CUDA run, runtime residency entries/bytes
  returned to zero; measured free VRAM after teardown was 12,486,443,008 bytes.

## Verification

- CUDA-enabled build of `vbuf_qwen3_block_qualification` and
  `vbuf_qwen3_cuda_ownership_contract`: passed.
- CTest: **39/39 passed**.
- Focused ASan/UBSan ownership contract with leak detection: passed.
- `git diff --check`: passed.

## Qualification boundaries preserved

No new parity classification is asserted by this migration. Preserve the
recorded statuses:

- Incremental/full CUDA: **EXPECTED SHAPE-DEPENDENT CUDA NUMERICAL DIVERGENCE**.
- Chunked/full prefill: **EXPECTED CUDA KERNEL-DISPATCH-DEPENDENT PREFILL DIVERGENCE**.
- CPU/CUDA prefix 25: **UNRESOLVED CONTEXT-SENSITIVE ACCUMULATED DIVERGENCE**.
- External llama.cpp `1e-5` gate: **FAIL**.
- Production Qwen and canonical Qwen sessions: **UNSUPPORTED / DISABLED**.

The generated-token replay is not CPU parity, external-oracle parity, HTTP,
tools, Pi, dogfood, or llama.cpp qualification. The 4096-capacity run is an
allocation probe only. Same-runtime concurrent inference remains
**UNSUPPORTED / NOT QUALIFIED**.

## Worktree and commit

This repository already had unrelated dirty work before this migration,
including edits in `integrations/ggml/CMakeLists.txt`, residency code,
`integrations/ggml/tools/qwen3_block_qualification.cpp`, other tools, tests, and
research artifacts. Because the qualification-runner edits overlap that dirty
file, the migration was left uncommitted rather than stage or commit unrelated
work. Existing commits were not amended; no stashes were changed and nothing
was pushed.
