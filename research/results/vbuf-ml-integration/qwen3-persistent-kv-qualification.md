# QWEN3 CANONICAL SOFTMAX COMPUTE-EXTENT QUALIFICATION

**Disposition:** the canonical extent policy is **IMPLEMENTED and TESTED** in the isolated Qwen3 qualification path for the pinned GGML CPU backend. The eight-token full-sequence, independent incremental-replay, and persistent-KV paths are **BIT-IDENTICAL** at recorded checkpoints. Production Qwen3 remains **BLOCKED** pending longer-prefix, multi-sequence, other backend, and production-session qualification. No tolerance or production behavior changed.

Artifact: Qwen3-14B Q4_K_M GGUF SHA-256 `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`; semantic-vBuf payload SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.

## KERNEL ANALYSIS

- **Backend:** GGML CPU, AMD Ryzen 7 5800X; pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, built `-O3 -march=native`, CPU single-variant build. The compiled/runtime path reports AVX2+FMA; AVX-512 is not active.
- **GGML softmax implementation:** `ggml_compute_forward_soft_max_f32` in `ggml-cpu/ops.cpp` calls `ggml_vec_soft_max_f32(ne00, ...)` for each row. The max scan is scalar. On this build, `ggml_vec_soft_max_f32` uses an AVX2 loop guarded by `i + 7 < n`, then a scalar tail.
- **Discovered granularity:** **8 F32 values for this AVX2+FMA path**. This is a kernel-specific extent policy, not a universal GGML/backend guarantee.
- **Extent 7:** no vector iteration; seven `expf` scalar-tail values are accumulated into `ggml_float` (double) in order.
- **Extent 8:** one 8-lane AVX2 iteration; `ggml_v_expf` is vectorized and the sum uses a horizontal reduction tree. Extent 9 is one vector plus a scalar tail; extent 16 is two vectors.
- The harmless masked eighth item leaves the maximum unchanged, but changes how the first seven exponentials and normalization sum are evaluated. This accounts for the observed 3-ULP case. Runtime capability selection in the isolated tool maps these fixed-width paths to 16/8/4 for AVX-512/AVX2+FMA/SSE-or-NEON and uses 1 for scalar fallback. Scalable SVE/RVV paths are rejected/skipped until their runtime vector-length behavior is analyzed. Only the AVX2+FMA path is qualified here; other CPU paths and backends are **NOT TESTED**.

## DESIGN AND IMPLEMENTATION

- `logical_extent` is the number of keys causally visible to a query. It does not change cache length, positions, masks, or GQA mapping.
- `SoftmaxComputeExtent::make(logical, granularity)` computes checked `ceil(logical/granularity)*granularity`; it enforces nonzero inputs, `compute >= logical`, and divisibility by granularity. `padded_scores()` preserves real scores and initializes synthetic slots to `-inf`.
- In the isolated GGML graph, padding K/V values are zero-filled solely to construct the wider score/context tensors. The causal mask is zero only for visible keys and `-inf` for future and padded slots. Thus padded probabilities are zero; padded KV is never made visible.
- **Files:** `integrations/ggml/include/softmax_compute_extent.h`; `integrations/ggml/tools/qwen3_block_qualification.cpp`; `integrations/ggml/tests/softmax_compute_extent_contract.cpp`; `integrations/ggml/CMakeLists.txt`; this report.
- The extent arithmetic helper is backend-neutral; its granularity is supplied by the isolated GGML CPU policy. Padding is applied only in the isolated Qwen3 qualification path. No production attention/runtime path or other model path was changed. No `7 → 8` special case was added.
- A full-sequence batch is currently supported canonically only up to one granularity group (`positions <= granularity`). The isolated path rejects larger full-sequence batches rather than using a batch-wide extent that would be wrong for early query positions. Per-query/grouped full-sequence execution beyond that limit remains **BLOCKED**.

## KNOWN 7→8 CONTROL

- **Location:** layer 0, position 6. Seven visible raw F32 scores are bit-identical; all 280 values (40 heads × 7 keys) have FNV-1a64 `5ac244501a2d288c` on both inputs.
- Head-0 visible scores (hex-float): `[0x1.f80a46p+4, 0x1.0fc3c8p+5, 0x1.d88e2ep+4, 0x1.f52f7cp+4, 0x1.1db4e4p+5, 0x1.2a97dap+5, 0x1.0bba84p+5]`. The full path's eighth unmasked score is `0x1.e8e3cep+4`; the causal/control mask sets it to `-inf`.
- **Unpadded control:** extent 7 versus extent 8 differs by max abs `5.96046448e-8`, max 3 ULP in the real captured row. Query-row count alone does not change output.
- **Canonical control:** `7 real + -inf`, extent 8 matches the full-sequence extent-8 probabilities **BIT-IDENTICALLY** (max abs/RMS/ULP all zero). Actual full and canonical incremental layer-0 probabilities also match exactly.
- At prefix 8, canonical attention context and block outputs for position 6 match exactly; the layer-13 block output is bit-identical (max abs zero). K/V bytes remain exact.
- The padding is diagnostic execution behavior only; no future key is exposed and no model semantic extent is changed.

## EXTENT SWEEP AND BOUNDARIES

A model-free GGML CPU contract test exercised logical extents **1–32**, deterministic score rows, unpadded versus canonical results, FP64 logical-softmax reference, weighted context proxy, probability-zero padding, byte geometry, and extent-validation errors. It passed.

| Logical extents | Compute extents at G=8 | Result |
|---|---|---|
| 1–8 | 8 | tested |
| 9–16 | 16 | tested |
| 17–24 | 24 | tested |
| 25–32 | 32 | tested |

Across the 32 cases, unpadded-versus-canonical probability max error was at most `5.96046448e-8` (max 3 ULP); a deterministic single-row weighted-context proxy differed by at most `5.17621327e-8`. Canonical-versus-FP64 max error was at most `5.96046448e-8`. Padded probabilities were exactly zero. These are synthetic kernel tests, not real-model qualification at those prefix lengths.

Boundary cases `G-1/G/G+1`, `2G-1/2G/2G+1`, and `3G-1/3G/3G+1` (7/8/9, 15/16/17, 23/24/25) all passed the same semantic checks. The synthetic suite also covers every other logical extent from 1 through 32.

## CAUSAL SEMANTICS

For the real eight-position full-sequence batch, each query's visible logical extent is 1 through 8; all therefore have canonical compute extent 8 on this CPU. Incremental position `p` uses logical extent `p+1`, rounded independently. Positions 0–7 were checked. Future real tensor keys and synthetic padding are both masked to `-inf`; no probability is assigned to padding. The original vBuf/llama reference comparison remains separate and still exhibits its documented numerical drift.

## EXECUTION-MODE PARITY AND PREFIX QUALIFICATION

The real-artifact run compared:

- **A.** saved llama.cpp reference checkpoints (reference/oracle only);
- **B.** vBuf full-sequence execution with canonical extent;
- **C.** independent incremental replay, restarting each target prefix from empty ephemeral byte-vector history;
- **D.** streaming persistent `Qwen3KvCache`.

For sequence `0,25,220,16,13,15,13,15`, prefixes **1–8** all passed. `B == C == D` was **BIT-IDENTICAL** at every checked per-layer attention probability/context/projection/block output, F16 K/V, final hidden state, and logits. All recorded boundary positions had no first divergence. Independent replay versus persistent KV had zero difference at all recorded floating-point checkpoints; K/V before storage and persistent readback were exact. Full-sequence K/V bytes matched the persistent cache. For Qwen3's 40 layers × 8 KV heads × 128 dimensions × F16 K/V, the cache is `163,840 bytes/token`, or `1,310,720 bytes` at prefix 8 (excluding workspace). The extent-8 softmax control, real context, and block output checks passed.

Full-vBuf versus saved llama.cpp final logits remain an independent, known reference difference: prefix-8 relative RMS `0.04896881`, max abs `0.552257538` (fails `1e-5`). Canonical padding resolves the execution-mode difference in the tested prefix; it does **not** establish llama.cpp logit parity. Prefix-8 greedy next token is `198` on all tested paths.

- **Prefixes 1–8:** real artifact, all three vBuf paths compared; PASS / BIT-IDENTICAL.
- **Prefixes 1–16 and 1–32:** real artifact **NOT TESTED**. Synthetic kernel extents cover 1–32. The qualification harness deliberately blocks canonical full batches beyond one granularity group pending per-query/grouped execution.
- **Sequences 2 and 3:** **NOT TESTED**. Only the fixed sequence above was used.

## NUMERICAL RESULT

- Synthetic unpadded vs canonical softmax: max error `5.96046448e-8`, max 3 ULP; canonical vs FP64: max `5.96046448e-8`.
- Real canonical full vs independent incremental vs persistent KV: probability, context, block, hidden, and logit differences recorded as zero at the tested checkpoints; exact K/V.
- Original unpadded full-vs-single-row control: `5.96e-8`, 3 ULP; padded extent-8 control: bit-identical.
- Full-vBuf vs llama.cpp logits remain separate at relative RMS `0.04896881`; token parity is not used as a numerical pass criterion.

## PERFORMANCE

The model-free 1-thread softmax microtest (100 graph executions per shape; allocation excluded from timing) measured mean `0.389 µs` unpadded and `0.375 µs` padded across extents 1–32; medians were `0.373` and `0.371 µs`. Per-extent values were noisy (observed ranges `0.355–0.649 µs` unpadded and `0.353–0.441 µs` padded), so no performance benefit is claimed. Score/mask/probability storage grows from `3×logical×4` to `3×compute×4` bytes per row; padding overhead is bounded by the selected granularity minus one. The 27m28s real eight-prefix qualification run includes full-prefix replay and extensive diagnostics and is not a comparable unpadded/padded performance measurement.

## DEEPSEEK REGRESSION

- Shared production attention/runtime changed: **NO**. Only the isolated Qwen3 qualification tool, generic extent helper, contract test, and CMake registration changed.
- DeepSeek regression: **NOT RUN / NOT REQUIRED** for this isolated-only change.
- HTTP, chat, completions, SSE: **NOT TESTED**.

## TESTS

- Qualification executable and softmax contract built against pinned GGML `2d191b5d`.
- Full GGML CTest suite: **32/32 PASS**.
- ASan/UBSan cache and softmax extent contracts: **2/2 PASS**.
- ASan/UBSan real prefix-8 canonical qualification: exit 0; no sanitizer report; A/B/C/D results passed.
- Real artifact prefixes 1–8: **PASS**; final persistent cache bytes equal full canonical cache.
- `git diff --check`: **PASS**.

## STATUS

- Canonical extent arithmetic: **IMPLEMENTED / TESTED**.
- CPU AVX2+FMA extent 8: **QUALIFIED for the pinned build/path**.
- Softmax extent sweep 1–32: **TESTED** on synthetic deterministic rows.
- Known 7→8 control: **BIT-IDENTICAL** after canonical padding.
- Full-sequence/incremental/persistent parity, prefix 1–8: **BIT-IDENTICAL**.
- Persistent KV: **QUALIFIED for this artifact/backend/sequence through eight tokens**; no defect demonstrated.
- Longer real prefixes, more sequences, CUDA/other backends: **BLOCKED / NOT TESTED**.
- Production Qwen3: **BLOCKED / DISABLED**.

## FAILED / BLOCKED AND NEXT GATE

1. Implement per-query or query-grouped canonical extents for full-sequence batches spanning more than one granularity group; retain the guard against batch-width-dependent semantics.
2. Generate qualified reference data and test real prefixes through at least 16 (preferably 32), crossing extent boundaries; test at least two additional deterministic sequences.
3. Qualify other backends independently. Do not reuse CPU granularity for CUDA or other architectures without evidence.
4. Keep the full-vBuf versus llama.cpp numerical drift as a separate oracle comparison; do not widen tolerance. Run production-session lifecycle/resource/performance and ordinary generation gates only after these investigation gates pass.
5. Keep Qwen3 disabled until the above gates are satisfied. No production behavior, tolerance, HTTP text, tools, or Pi integration was changed or enabled.

Logs: `/tmp/qwen3-canonical-extent-prefix1-8-allfull.log`, `/tmp/qwen3-canonical-extent-prefix8-asan-ubsan.log`, `/tmp/vbuf-softmax-extent-contract.log`.
