# Step 31P Indexed Expert-Bank Qualification

Status: **physical A/B completed; numerical reference parity remains separate**.

## Scope

The Android adapter now has an opt-in `vbufIndexedExpert` build property. When
enabled, batched prompt prefill can execute the three full expert banks through
`ggml_mul_mat_id`. Autoregressive decode intentionally remains on the selected
rank-2 expert path because one-row decode does not amortize full-bank
materialization.

## Verified

- Android `arm64-v8a` builds succeeded with `vbufIndexedExpert=true` and
  `vbufQualification=false`.
- Android builds also succeeded with `vbufIndexedExpert=true` and
  `vbufQualification=true`.
- The default Android build with `vbufIndexedExpert=false` succeeded.
- Native CTest: 21/21 passed.
- Rust workspace tests passed.
- The generated semantic DeepSeek artifact passed conversion and independent
  validation, and the Android model-open path reached `OPEN_OK` on Pixel 7 Pro.
- Build-time qualification is now passed through `BuildConfig` to the JNI open
  call; the UI no longer hard-codes normal mode.
- Pixel 7 Pro indexed run completed with the canonical prompt, 7-token batched
  prefill, 4-token decode, and `----` output.
- Matched fallback run completed with the same prompt/token counts and `----`
  output.

## Measured Limitation

The first indexed implementation also selected full-bank execution for serial
decode. That path was not viable on the device: each MoE layer materializes
approximately 199 MB of gate/up/down banks, compared with approximately 19 MB
for six selected rank-2 slices. A multi-token run did not complete before the
device backgrounded. This is evidence against serial indexed execution, not a
failure of the rank-2 fallback.

The implementation was narrowed afterward so the indexed path is used only by
batched prefill. The corrected paired run measured:

| Build | Prefill | Total generation | Indexed layers | Materialized bank bytes | Repack bytes | Peak resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| indexed prefill | 65,007 ms | 154,179 ms | 26 | 5,173,149,696 | 0 | 268,099,584 |
| rank-2 fallback | 67,314 ms | 167,051 ms | 0 | 4,745,501,920 | 0 | 267,452,416 |

The indexed run recorded 78 bank submissions and 1,092 logical slots. The
fallback recorded 624 routed expert invocations, 1,248 gate/up submissions, and
624 down submissions. The total-generation difference must not be attributed
to indexed execution alone: both runs use rank-2 decode fallback, and their
residency/cache state differs. The result proves a physical A/B smoke path and
canonical output parity, not logits-level numerical reference parity.

## Remaining Gate

Step 31P still needs a reference-mode or host-backed numerical check for exact
TopK IDs, weights, and per-token outputs before it can be called full numerical
qualification. The Android A/B evidence above is sufficient to preserve the
physical smoke result and the materialization tradeoff.
