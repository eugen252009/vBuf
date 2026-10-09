# Native-layout AV divergence investigation checkpoint

**Checkpoint source:** `65375bf14bd66762a8a6cc1b345636b239076b01` (`qwen3: add bounded native AV sequence matrix`), parent `3f927758feede3e05ff7df2a86e890e07b214752`. This file records the exact starting state before further numerical investigation. The matrix implementation and evidence in that commit are preserved; no result below is a production qualification.

## Fixed setup

- Model: Qwen3-14B Q4_K_M; semantic vBuf SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- GGML checkout used for runs: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`; its pre-existing `src/ggml-cuda/ggml-cuda.cu` modification is outside the repository and must remain untouched.
- Placement: RTX 3060/SM86 owns embedding and blocks 0–25; RTX 2080 SUPER/SM75 owns blocks 26–39 and final norm/head.
- Candidate remains `Candidate_NOT_Valid`; canonical packed-V AV is authoritative. Candidate guard remains capacity <=512, exact initial 32-row prefill or one-row decode with pre-decode context <=32. The existing final gate remains hidden/logit relative RMS <=0.02 and cosine >=0.9998.
- Production defaults remain capacity 1032, prefill chunk 32, single-GPU default, optimizer `SHADOW`; nothing in this checkpoint enables production candidate selection.

## Reproduced evidence at checkpoint

| Path | Same logical history? | Final hidden relative RMS | Final logits relative RMS / cosine | Outcome |
|---|---|---:|---:|---|
| 8-token prompt + 25 generated, incremental one-row path | Baseline | `0.00957307` | `0.0170682` / `0.999854` | Pass; 25/25 generated IDs match; 33/33 candidate AV steps; repeated twice |
| Same 33 IDs supplied as prompt: prefill32 + decode at position 32, native in both phases | Yes; token hash `4d25767f9dce13f5` | `0.0138161` | `0.0236490` / `0.999728` | Fail; repeated twice |
| Same fixed 33-ID history: native prefill + canonical decode | Yes | `0.0133640` | `0.0225540` / `0.999754` | Fail; repeated twice |
| Same fixed 33-ID history: canonical prefill + native decode | Yes | `0.00892133` | `0.0146598` / `0.999893` | Pass; repeated twice |
| 8-token prompt + one generated token, full native candidate | Greedy token unchanged | `0.00982415` | `0.0301166` / `0.999546` | Fail; repeated twice |

At the 32-row prefill endpoint, position 31 with 32 rows, logits relative RMS is `0.0244612` / cosine `0.999703`; the subsequent native decode in the both-native replay ends at `0.0236490`. Earlier same-input native AV versus FP64-oracle evidence is <=`1.02e-7` relative RMS. That supports the tested AV arithmetic/indexing but does not establish full-model numerical compatibility.

The 20-row fixture matrix was intentionally stopped after the common-token failure. Nineteen planned fixtures remain `NOT_RUN_AFTER_STOP`. The exact data is retained under `raw/autoregressive-fixture-matrix-20261009/`, `raw/common-token-phase-isolation-20261009/`, and `raw/session-reuse-lifecycle-20261009/`. The session cancellation/re-entry test matched fresh-session hashes and reset cleanly, while its numerical 8+1 comparison still failed.

## Build/test state recorded at checkpoint

The matrix target built at `/tmp/vbuf-qwen-native-matrix-build`. The focused `vbuf_qwen3_execution_plan_contract` and `vbuf_qwen3_native_attention_av_contract` were run and passed. A 39/39 CTest result exists from earlier qualification; it is historical, not a fresh run for the next investigation.

## Investigation plan after this checkpoint

1. Re-run and compare incremental 8+25, common-token 32-row replay, and 8+1 with explicit input token/KV/session/phase records. Verify the measurement boundary is the same before interpreting the 8+1 result as the successful run's prefix.
2. Record effective prefill/decode graph shapes and QK, attention/softmax extents, AV path, KV population/progress, and final-logit position for each path.
3. Use bounded captures at embedding, Q/K, scores/probabilities, AV input/output, attention output, residual, FFN, block hidden, final hidden, and logits. Find the first material difference against same-capacity canonical execution.
4. Separate upstream input/QK/KV differences, local AV arithmetic, and downstream propagation. Use a single-layer intervention only at the first implicated layer and preserve the independent FP64 oracle.
5. Change no AV arithmetic, threshold, guard, production default, or canonical fallback without a demonstrated defect. If reduction-order/phase behavior is the only supported explanation, report it as numerical incompatibility or unresolved behavior rather than forcing canonical rounding.
6. Re-run focused qualification and relevant tests, retain both pass/fail evidence, update the research report, and commit the bounded investigation separately. Do not push.
