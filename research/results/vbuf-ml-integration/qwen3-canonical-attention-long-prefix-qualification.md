# QWEN3 CANONICAL ATTENTION LONG-PREFIX QUALIFICATION

**Current result:** the real-artifact qualification now covers canonical full-sequence attention, fresh incremental replay, and persistent KV through 32 positions on two deterministic sequences. All internal checkpoints, final logits, and K/V bytes were bit-identical. The isolated persistent greedy path also matched llama.cpp's seed-42 token sequence through 32 tokens. The fixed `1e-5` external logit criterion still fails at prefixes 8, 16, and 32; production Qwen3 remains disabled.

The sections below the follow-on record preserve the earlier prefix-8 status as historical evidence. The current follow-on results in the next section supersede any earlier “not tested” status for 9–32. No production path or tolerance changed.

See also: [canonical softmax compute-extent qualification](qwen3-persistent-kv-qualification.md).

## FOLLOW-ON: REAL PREFIXES 1–32

### Implementation and test scope

- Added canonical per-query extent grouping for the isolated Qwen3 full-sequence graph. Consecutive queries are grouped by `round_up(query_position + 1, 8)`; group outputs return to original query order. Future keys and synthetic padding are masked to `-inf`; logical KV visibility, RoPE positions, and cache offsets remain unrounded.
- Extended the reference dumper to 32 greedy prefixes and compact selected checkpoints. Added a 32-position real-artifact mode comparing full-sequence execution, a fresh independent incremental byte-history replay, and the append-only `Qwen3KvCache` path.
- Every prefix compared 18 float checkpoints per layer in all three modes (720 checkpoint comparisons across 40 layers), plus exact K/V storage bytes, cache readback, final hidden state, and final logits. The first mismatch fails immediately.
- Qualification was CPU-only, pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, AVX2+FMA, softmax granularity 8. No generic vBuf format or shared vBuf-ML runtime behavior changed.

### Deterministic sequences and internal equivalence

| Sequence | Source / length | Full vs fresh replay vs persistent KV | Greedy comparison |
|---|---|---|---|
| A | llama.cpp greedy, seed token `0`, 32 tokens | **BIT-IDENTICAL** at every prefix 1–32 | vBuf next-token choice first differs from the supplied llama.cpp continuation at prefix 9; this is behavioral divergence, not cache divergence |
| B | persistent-vBuf greedy, seed token `42`, 32 tokens | **BIT-IDENTICAL** at every prefix 1–32 | Generated token IDs match the independent llama.cpp seed-42 sequence at every one of 32 positions |
| B, fixed-reference boundary run | llama.cpp seed-42 sequence, 17 tokens | **BIT-IDENTICAL** at every prefix 1–17 | No greedy-token divergence through prefix 17; the real 8→9 and 16→17 transitions were checked under ASan/UBSan |

Sequence A tokens: `0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0`.

Sequence B tokens: `42,25,220,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16,13,16`.

For A, internal equivalence remains exact even after its greedy continuation stops matching llama.cpp: at prefix 9 vBuf predicts `220` while the llama.cpp continuation supplies `262`. For B, the independently generated persistent-vBuf path matches the llama.cpp seed-42 tokens through position 32. This distinction does not relax the external numerical-parity gate.

### Real extent boundaries (G=8)

Prefixes **7, 8, 9, 15, 16, 17, 23, 24, 25, 31, and 32** passed full/replay/persistent checks. Specifically, the real transitions **8→9**, **16→17**, and **24→25** passed with compute extents changing 8→16, 16→24, and 24→32. Exact K/V bytes, absolute-position RoPE values, causal visibility, attention probabilities through the compute extent, and final logits matched across modes. At prefix 32, persistent cache size was `5,242,880` bytes, exactly `32 × 163,840` bytes/token.

### External llama.cpp logits (unchanged 1e-5 criterion)

The selected-prefix comparisons use compact per-prefix llama.cpp reference logits and the existing fixed absolute tolerance. Token/top-k agreement is reported separately and does not override the numerical gate.

| Sequence | Prefix | Relative RMS | Max absolute | Cosine | Top-1 | Top-10 overlap |
|---|---:|---:|---:|---:|---|---:|
| A | 8 | 0.04896881 | 0.55225754 | 0.99892335 | match (`198`) | 10/10 |
| A | 16 | 0.05843522 | 0.55858135 | 0.99834485 | match (`13`) | 9/10 |
| A | 32 | 0.05192218 | 0.49960613 | 0.99865816 | match (`220`) | 9/10 |
| B | 8 | 0.02658902 | 0.64777470 | 0.99982843 | match (`13`) | 9/10 |
| B | 16 | 0.02952494 | 0.47412038 | 0.99956552 | match (`13`) | 10/10 |
| B | 32 | 0.04891015 | 0.45035768 | 0.99880379 | match (`13`) | 10/10 |

All six comparisons **FAIL** the unchanged `1e-5` maximum-absolute-error criterion. These are real external numerical differences despite matching top-1 tokens at these selected prefixes. No tolerance waiver is proposed.

### Resource measurements

- Release full-sequence plus final-head elapsed time: sequence A, **57.708 s**; sequence B, **58.622 s**.
- Incremental 40-layer token latency (timer wraps one token across all layers; replay and persistent measured separately):
  - A: independent replay mean **29.409 s** (28.503–31.511 s); persistent KV mean **29.493 s** (28.586–31.214 s).
  - B: independent replay mean **30.160 s** (28.515–32.027 s); persistent KV mean **30.330 s** (28.620–32.176 s).
  - These are qualification-tool CPU timings, not production serving latency; they include per-layer graph setup and execution and exclude process/model-open time.
- Peak process RSS, measured with child `ru_maxrss`: A **2,005,356 KiB**; B **2,051,676 KiB**.
- Largest observed GGML context tensor backing buffers on the 32-token full path: first stage **471,011,072 bytes**, FFN stage **205,398,048 bytes**, final head **659,565,568 bytes**. These are per-context allocations and include weight tensors plus intermediates; they are **not** an activation-only or summed concurrent graph-memory measurement.
- Persistent KV itself: **5,242,880 bytes** at 32 positions.

### Reset, recreation, and sanitizers

- Sequence A and B each passed same-cache reset followed by clean one-token replay, and cache destruction/recreation followed by clean replay; state was isolated and the logits were bit-identical. This tests the isolated qualification cache, **not** production `VbufGenerationSession`.
- Real sequence B through prefix 17 passed ASan/UBSan with leak detection and halt-on-error enabled, including both extent transitions 8→9 and 16→17, persistent/replay equivalence, reset, and recreation. No sanitizer diagnostics were reported. The full 32-token real run was not sanitizer-instrumented.
- Model-free ASan/UBSan contracts passed 3/3: KV cache through 32, softmax extents 1–64, and query grouping across 8/9, 16/17, 24/25, and 32.
- Full pinned-GGML CTest passed **33/33**; `git diff --check` passed.

### Current gate/status

- Real internal full/replay/persistent equivalence through 32 on A and B: **QUALIFIED** for these artifacts, CPU backend, and tested sequences.
- Real extent transitions 8→9, 16→17, 24→25: **QUALIFIED**; 8→9 and 16→17 also have real-artifact ASan/UBSan coverage.
- Persistent greedy generation through 32: **QUALIFIED** internally for seed 42 and token-matched to llama.cpp for that run.
- External strict logit parity: **FAILS** at prefixes 8, 16, and 32 for both sequences. Token parity at selected prefixes is not a substitute.
- Third sequence, other backends/platforms, production session integration, production Qwen3 enablement, HTTP text, native tools, and Pi: **NOT TESTED / DISABLED**.
- Keep production Qwen3 disabled pending project decision on the observed external numerical divergence and remaining qualification gates.

Evidence logs: `/tmp/qwen3-vbuf-long-A32.log`, `/tmp/qwen3-vbuf-long-B17.log`, `/tmp/qwen3-vbuf-long-B17-asan.log`, `/tmp/qwen3-vbuf-generated-seed42-32.log`, `/tmp/qwen3-vbuf-full-reference-A.log`, `/tmp/qwen3-vbuf-full-reference-B.log`, `/tmp/qwen3-vbuf-memory-A32.log`, `/tmp/qwen3-llama-prefix1-32/generation.meta`, `/tmp/qwen3-llama-seed42-prefix1-32/generation.meta`.

## HISTORICAL PREFIX-8 BASELINE AND EARLIER STATUS

The remainder preserves the original prefix-8 report snapshot. Any “NOT TESTED” or “BLOCKED” statements below that conflict with the follow-on section above are historical and superseded; the model identity, synthetic extent results, and strict external numerical-gate findings remain relevant.

### MODEL

- Model: Qwen3-14B Q4_K_M
- GGUF SHA-256: `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`
- Semantic vBuf payload SHA-256: `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`
- Backend: pinned GGML CPU `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, AVX2+FMA build
- Canonical compute granularity: 8

## CANONICAL EXTENT CONTRACT

- Logical extent: number of causally visible KV positions; unchanged by padding.
- Compute extent: checked `round_up(logical_extent, 8)` for this qualified CPU kernel.
- Padding rule: synthetic score slots are `-inf`; padded probability must be exactly zero. The isolated graph uses zero K/V storage for those slots and masks them to `-inf`.
- Causal semantics: future real keys remain masked; token/RoPE positions, cache length, GQA mapping, and historical cache offsets do not use the rounded extent.
- Backend dependency: extent 8 is qualified only for the pinned GGML CPU AVX2+FMA softmax branch. No CUDA, SVE/RVV, or other backend qualification is implied.

## PREFIX 1–32 QUALIFICATION

The existing real-artifact run used sequence `0,25,220,16,13,15,13,15` and compared canonical vBuf full sequence, fresh independent incremental replay, and persistent KV at every prefix from 1 through 8. All recorded checks were **BIT-IDENTICAL**; no first divergence was reported. Prefixes 9–32 are **NOT TESTED**.

| Prefix | Canonical full | Independent replay | Persistent KV | Result |
|---:|---|---|---|---|
| 1–8 | run | run | run | **BIT-IDENTICAL** at recorded checkpoints |
| 9–32 | blocked by current harness | not run | not run | **BLOCKED / NOT TESTED** |

The qualification binary currently requires `POSITIONS(1..8)` and `run_persistent_kv` rejects sequences longer than eight. Reference dumps exist only for `prefix-0` through `prefix-7`. Additionally, full-sequence canonical execution rejects `positions > granularity`. Thus no real-artifact values are reported for prefixes 9–32, and there is no claimed first divergence beyond prefix 8.

### BOUNDARY SUMMARY (G=8)

| Prefix | Compute extent | Padding | Real full/replay/persistent result |
|---:|---:|---:|---|
| 7 | 8 | 1 | **BIT-IDENTICAL**, existing run |
| 8 | 8 | 0 | **BIT-IDENTICAL**, existing run |
| 9 | 16 | 7 | **NOT TESTED** |
| 15 | 16 | 1 | **NOT TESTED** |
| 16 | 16 | 0 | **NOT TESTED** |
| 17 | 24 | 7 | **NOT TESTED** |
| 23 | 24 | 1 | **NOT TESTED** |
| 24 | 24 | 0 | **NOT TESTED** |
| 25 | 32 | 7 | **NOT TESTED** |
| 31 | 32 | 1 | **NOT TESTED** |
| 32 | 32 | 0 | **NOT TESTED** |

The 7/8 observations are model evidence; all later rows are extent arithmetic only, not real-model boundary qualification.

## CANONICAL EXTENT TRANSITION INVARIANTS

- **Synthetic:** checked rounding, padding count, input-score preservation, `-inf` initialization, zero padded probability, and FP64 logical-softmax comparison passed at every extent 1–64 for all three score distributions.
- **Real prefixes 1–8:** causal visibility and K/V indexing remained unchanged; stored K/V matched independent replay and full-sequence cache bytes exactly. RoPE absolute-position handling was checked in that run.
- **Real transitions 8→9, 16→17, and 24→25:** **NOT TESTED**. No claim that the later real-model transitions have been qualified.
- The model-free cache contract now appends, checks offsets/bytes at positions 7/8, 15/16, 23/24, and 31, then resets. This does not substitute for real-model RoPE/session testing.

## EXECUTION-MODE EQUIVALENCE

- Canonical full sequence vs independent incremental replay: **BIT-IDENTICAL**, prefixes 1–8 only.
- Independent incremental replay vs persistent KV: **BIT-IDENTICAL**, prefixes 1–8 only; pre-storage and readback K/V exact.
- Canonical full sequence vs persistent KV: **BIT-IDENTICAL**, prefixes 1–8 only; all cache bytes matched.
- Prefixes beyond 8: **NOT TESTED**. No equivalence status may be inferred from the synthetic sweep or cache contract.

## MULTI-SEQUENCE QUALIFICATION

- Sequence A: `0,25,220,16,13,15,13,15` — existing eight-token real-artifact path **QUALIFIED** for internal equivalence and token parity.
- Sequence B: **NOT TESTED**.
- Sequence C: **NOT TESTED**.
- First divergence: none for A through prefix 8; B/C and A after prefix 8 have no result.

The three synthetic score distributions are deliberately not described as token sequences: they do not exercise model tokenization, layer history, or generated contexts.

## GREEDY GENERATION

- 8 consumed tokens: sequence A passed full-recompute/persistent/replay token parity through the tested prefix; at prefix 8 each selected next token `198`.
- 16 tokens: **NOT TESTED**.
- 32 tokens: **NOT TESTED**.
- Stop-on-divergence beyond 8: **NOT TESTED**; no longer generated run was attempted.

## FULL-SEQUENCE GROUPING

- Current limitation: one `run_positions` full graph has a single rectangular `[query,key]` softmax extent. The canonical full path rejects `positions > G`; the CLI and persistent qualification also cap sequences at 8.
- Minimal design: retain the full-sequence Q/K/V projections and absolute positions; partition query rows into consecutive groups by `round_up(query_position+1,G)`. For each group, view/slice its Q rows, use the first `compute_extent` historical/current K/V rows (zero-fill only if the final extent exceeds available sequence keys), build a group-local causal mask with `-inf` for future and padding, and run group-local score/softmax/context. Copy group contexts back to their original token offsets and concatenate in original query order before the shared attention output projection and FFN. Diagnostics must similarly restore query order while preserving each row's logical length and compute extent.
- Required correctness conditions: each grouped query must see exactly its logical prefix; group boundaries must not reset RoPE or cache offsets; output order must be stable; per-row diagnostics cannot pretend all rows used the maximum batch extent.
- Implemented: **NO**. This is needed for full-vs-incremental comparison at prefixes beyond 8, but the current qualification graph and diagnostics are built around one common extent. A grouped refactor was not made because it is a substantial change to the isolated attention graph without longer-prefix references or a boundary qualification harness to validate it safely.
- Remaining restriction: full-sequence comparisons are available only for one extent group (prefixes ≤8 on this backend). Do not remove the guard without grouped-query tests.

## SYNTHETIC EXTENT SWEEP

- Range: logical extents 1–64, GGML CPU AVX2+FMA, three deterministic distributions (mixed bounded values, alternating high/low values, and one peaked value with negative background).
- Boundaries: every `G-1/G/G+1` transition through 64, including 7/8/9, 15/16/17, 23/24/25, 31/32/33, 39/40/41, 47/48/49, 55/56/57, and 63/64 (65 is outside the sweep).
- Result: **PASS** for extent arithmetic, padding, `-inf`, zero padded probability, and logical-softmax semantics.
- Across 192 synthetic rows, unpadded-vs-canonical max probability difference was `5.96046448e-8` (maximum 3 ULP); weighted-context proxy max difference `5.17621327e-8`; canonical-vs-FP64 max difference `5.96046448e-8`.
- This is a synthetic contract result, not a real-artifact prefix result.

## LLAMA.CPP REFERENCE

Only prefix 8 is available in the current saved reference run.

- Prefix 8 hidden/block-output comparison: layer-39 block-output relative RMS `0.02228591`, cosine `0.99975343` (this is the final block output, before final output normalization).
- Prefix 8 logits: relative RMS `0.04896881`, max absolute difference `0.55225754`, cosine `0.99892323`; fixed `1e-5` numerical criterion **FAILS**.
- Prefix 8 top-1: **TOKEN PARITY QUALIFIED** for this path; both choose `198`.
- Prefix 8 top-5 overlap: **NOT RECORDED**. Existing diagnostic records top-10 overlap `10/10`.
- Prefix 8 top-1/top-2 margin: llama.cpp `1.17023087`, vBuf `1.19040966`.
- Prefixes 16 and 32: **NOT TESTED**.
- Classification: external strict numerical parity is **NUMERICALLY DIVERGENT**; the observed top-token behavior at this one prefix is **TOKEN-PARITY QUALIFIED**. It does not establish multi-sequence behavioral parity.

## RESOURCE USAGE

Qwen3 F16 K/V geometry is 40 layers × 8 KV heads × 128 dimensions × 2 bytes × K/V:

- KV bytes prefix 8: `1,310,720` (measured in existing run).
- KV bytes prefix 16: `2,621,440` (**DERIVED**, not allocated in real qualification).
- KV bytes prefix 32: `5,242,880` (**DERIVED**, not allocated in real qualification).
- Peak host RAM: **NOT TESTED**.
- Temporary graph memory: **NOT TESTED**.
- Incremental token latency: **NOT TESTED**. The previous 27m28s wall time covers the eight-prefix diagnostic run and is not a per-token latency measurement.

## SESSION SAFETY

- Reset/clean replay after prefix 8: **PASS** in the existing isolated run (`bytes_after_reset=0`, clean replay matched).
- Model-free KV cache reset after a 32-position append: **PASS** in the extended contract test.
- Real 16/32-token destroy/recreate and cross-session isolation: **NOT TESTED**.

## SANITIZERS

- ASan/UBSan model-free KV cache contract through 32 positions: **PASS**.
- ASan/UBSan synthetic softmax contract, extents 1–64 × 3 distributions: **PASS**.
- Real-artifact prefix-8 canonical run: **PASS**, no sanitizer report (existing run).
- Real boundary immediately above 8 (prefix 9) and above 16 (prefix 17): **NOT TESTED**; the current real harness does not permit them.

## PRODUCTION-GATE ANALYSIS

- vBuf internal consistency: **QUALIFIED** only for sequence A through eight tokens. Exact replay/persistent equality is strong evidence for that tested context, not for longer history or other sequences.
- Persistent-KV confidence: **QUALIFIED** for artifact/backend/sequence A through prefix 8. Cache layout and reset are additionally model-free tested through position 31.
- Extent-boundary confidence: model evidence covers the 7→8 transition. Synthetic deterministic rows cover later boundaries through 64; real model evidence at 8→9 and later is absent.
- External numerical parity: **NUMERICALLY DIVERGENT** at prefix 8 under the unchanged `1e-5` criterion.
- External behavioral parity: **TOKEN-PARITY QUALIFIED** for sequence A only through the tested eight-token run; top-10 overlap was 10/10 at prefix 8. Multi-sequence and longer behavioral parity are **NOT TESTED**.
- Recommendation: keep the strict external numerical result visible and **do not waive or change the tolerance in this task**. Internal replay/persistent equality does not resolve external numerical divergence. Do not advance to production text integration until the grouped full path, longer prefixes, additional sequences, and a project decision on the numerical-vs-behavioral production gate are complete.

## DEEPSEEK REGRESSION

- Shared runtime/attention/session code changed: **NO**. This follow-on changes only model-free qualification tests and this report; the Qwen3 isolated runtime and production paths were not changed.
- DeepSeek rerun: **NOT RUN / NOT REQUIRED** for this isolated-only change. The full CTest suite's model-free DeepSeek semantics contract passed; that is not a real-model regression qualification.

## TESTS

- Pinned GGML build: **PASS** for the Qwen3 qualification executable and updated contracts.
- CTest: **32/32 PASS**.
- Canonical extent contract: **PASS**, 1–64 × 3 synthetic distributions.
- KV cache contract: **PASS**, 32-position append/offset/readback/reset checks.
- New real-artifact qualification beyond prefix 8: **BLOCKED / NOT RUN**.
- Focused ASan/UBSan contracts: **2/2 PASS**.
- Real prefix-8 ASan/UBSan: **PASS** (existing evidence).
- `git diff --check` plus checks for new untracked qualification files: **PASS**.

## STATUS

- Prefixes 1–8: **QUALIFIED**, internal paths **BIT-IDENTICAL**.
- Prefixes 1–16: **BLOCKED**; only prefixes 1–8 have real-artifact results.
- Prefixes 1–32: **BLOCKED**; only prefixes 1–8 have real-artifact results.
- Multiple real sequences: **NOT TESTED**.
- Synthetic extent boundaries through 64: **QUALIFIED** as a model-free contract.
- Real later extent boundaries: **NOT TESTED**.
- Persistent KV: **QUALIFIED** only through eight tokens on sequence A; model-free cache contract through 32.
- Production Qwen3: **DISABLED**.

## FAILED / BLOCKED

1. Numerical parity with the pinned llama.cpp reference still fails `1e-5` at prefix 8; no tolerance change was made.
2. Full-sequence canonical grouping across extent groups is not implemented.
3. Real prefix/reference artifacts stop at eight positions; qualification CLI/cache flow also caps at eight.
4. Sequences B/C, generated 16/32-token paths, later real boundaries, long-session isolation, peak RAM, graph memory, token latency, and prefix-16/32 external metrics are not tested.
5. Top-5 overlap was not recorded even at prefix 8.

## NEXT GATE

1. Extend/reference-generate deterministic sequences and per-prefix llama checkpoints, while keeping the existing 1e-5 diagnostic unchanged.
2. Implement and unit-test query grouping in the isolated full-sequence attention graph, restoring original query order and per-row compute extents; retain rejection for unsupported cases.
3. Extend the isolated cache/persistent harness beyond eight positions and add a single-stream independent replay so longer runs do not redundantly replay every prior prefix.
4. Run real A/B/C comparison at 9/15/16/17 and then through 32, add two fixed sequences, and stop generated comparisons at the first token divergence.
5. Measure process peak RAM, graph allocation, and incremental token latency; run longer reset/isolation and boundary ASan/UBSan.
6. Reassess the external numerical-parity production gate only after this evidence. Keep production Qwen3, HTTP, tools, and Pi disabled meanwhile.

Logs/evidence: `/tmp/qwen3-canonical-extent-prefix1-8-allfull.log`, `/tmp/qwen3-canonical-extent-prefix8-asan-ubsan.log`, `/tmp/vbuf-softmax-extent-1-64.log`.
