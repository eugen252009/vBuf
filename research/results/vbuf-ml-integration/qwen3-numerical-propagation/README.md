# Qwen3-14B Native AV and Capacity-Dependent Numerical Propagation

**Recommendation:** keep native-layout AV experimental and keep canonical packed-V AV authoritative. The original 8-token plus 25-generated-token free-running fixture reproduces and passes twice, but a common-token replay of that exact 33-token history using the 32-row-prefill/decode path fails the unchanged logits gate. Targeted phase isolation attributes the failure to native prefill; native decode alone passes on that same history. A separate 8-token plus one-generated-token full-candidate run also fails the final logits gate, reproducibly. The 20-fixture matrix therefore stopped after its first new failure; most planned fixtures were not run. This evidence does not validate or promote the candidate. The capacity-512 guard, 32-row prefill restriction, decode-through-position-32 limit, numerical thresholds, and production defaults remain unchanged.

## Scope and setup

These diagnostics used the admitted Qwen3-14B Q4_K_M artifact (`sha256:f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`, 9,000,232,144 bytes), pinned GGML commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`, and the existing 26/14 split: blocks 0–25 on the RTX 3060 (SM 8.6), blocks 26–39 plus normalization/head on the RTX 2080 SUPER (SM 7.5). Native AV was eligible only on the early SM 8.6 blocks. The initial boundary/propagation sweeps used the same 32-token fixture; the later sequence matrix used explicitly recorded offline-tokenized prompts. Comparisons retain the existing final hidden/logit gate: relative RMS ≤0.02 and cosine ≥0.9998 for both outputs. No thresholds were relaxed.

The runner is a diagnostic-only target. Its per-phase candidate suppression controls are compiled only under `VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC`; they isolate prefill and decode trials and do not change production selection or the candidate guard.

## Findings by concern

### 1. Mathematical correctness of AV

The native AV result uses the same captured probability weights and full-capacity F16 V-cache as the canonical path, and is compared with an independent FP64 AV oracle. Qwen3 maps 40 query heads to 8 KV heads (five query heads per KV head). Across the captured 26 early-device blocks, 32-row prefill, and decode positions 1, 8, 16, and 32, native-vs-oracle relative RMS is at most `1.0138e-7`. The largest canonical-vs-oracle and canonical-vs-native per-layer relative RMS in these samples is about `3.251e-4`—consistent with reduction-order differences between the packed matrix operation and the direct native reduction, not a layout or GQA-indexing defect.

This is strong sampled evidence for the AV arithmetic, not a formal proof for every model, device, or context. The direct CPU/CUDA contract tests are also covered by the passing test suite below.

### 2. Numerical compatibility and propagation through 40 blocks

A one-layer intervention replaces AV at one selected early block while keeping all other blocks canonical. For all 20 combinations (layers 0, 8, 16, 24, 25 × decode positions 1, 8, 16, 32):

- The AV V/probability inputs are bitwise equal to the canonical replay.
- The native result is bitwise equal to the same-input native side branch.
- Blocks preceding the intervention remain bitwise equal to baseline.
- The fed token and greedy top-1 remain equal.
- The A/B/A canonical replay after the context-32, layer-8 intervention is bitwise equal.

Nineteen of 20 final-output comparisons pass the unchanged numeric gate. The single failure is layer 0 at decode position 8: final logits relative RMS `0.0201839`, cosine `0.999796` (hidden relative RMS `0.00711116`; top-1 unchanged). This is a narrow threshold failure, not permission to adjust the gate. At position 32, errors introduced by single-layer interventions propagate through the remaining blocks but attenuate in the final hidden state; the selected interventions end with hidden relative RMS between about `0.00497` and `0.00624`.

The all-eligible-layer decode-only candidate at position 32 also passes: 26 native AV blocks, 14 canonical blocks, hidden relative RMS `0.00652323` / cosine `0.999979`, logits relative RMS `0.0156915` / cosine `0.999880`, with token and top-1 unchanged. Layerwise relative RMS peaks at block 13 (`0.0116569`) and falls to `0.00652323` by block 39.

The phase-isolated 32-row native prefill trial is less compatible: final hidden relative RMS `0.0140981` / cosine `0.999905`, but logits relative RMS `0.0243665` / cosine `0.999703`, so it fails the existing final-output gate (top-1 unchanged). The complete guarded 32-row native-prefill plus position-32 native-decode trial passes: 52 native AV block-steps across the two phases, hidden relative RMS `0.0128155` / cosine `0.999927`, logits relative RMS `0.0179616` / cosine `0.999847`, and token/top-1 unchanged. Its blockwise deviation peaks at `0.02263` around block 19 before ending below the final-output threshold. Thus the passing combined endpoint does not imply per-phase or intermediate-block equivalence; the prefill-only failure remains material qualification evidence.

A separate autoregressive run exercised the candidate on all 33 one-row steps from an 8-token prompt through decode context 32 (8 prompt-ingestion steps plus 25 generated-token steps). All generated token IDs matched canonical; all 33 steps selected native AV on the 26 eligible blocks (858 native block-steps). Final hidden relative RMS was `0.00957307` / cosine `0.999958`; final logits relative RMS was `0.0170682` / cosine `0.999854`, passing the unchanged final-output gate. The largest hidden relative RMS along the matched sequence was `0.0220137` at position 11 (cosine `0.999791`); this is a recorded intermediate observation, not a separately defined final-output gate. This broadens evidence within the existing context guard, but it is one prompt fixture, not broad prompt or production-context qualification.

### 2a. Bounded multi-fixture matrix and stop-on-failure follow-up

The new versioned manifest contains 20 planned fixtures: offline-tokenized natural-language prompts of lengths 1, 2, 4, 8, 16, 24, and 31; repeated, alternating, and permuted valid token-ID patterns explicitly labeled synthetic; tokenized code-like and JSON-like prompts; capacities 64/256/512; the 32-row prefill boundary; isolated prefill/decode modes; and a layer-0/position-8 intervention. Every planned fixture requests two runs. Token IDs are embedded in the manifest, not regenerated at run time. Text prompts were tokenized locally with `llama-tokenize --offline --no-bos` against the local Qwen3-14B GGUF using llama.cpp commit `a97123e497968f3440264c0464a7adc7c999c027`; no network tokenizer/model access was used.

Per the stop rule, broad expansion stopped on the first new failure. Only the original 8+25 fixture ran from this 20-row manifest; its remaining 19 fixtures are explicitly marked `NOT_RUN_AFTER_STOP`. The full free-running path reproduced twice with identical canonical/candidate token hash `9ac608097b99ad85`, identical numeric-capture hashes per path, 33/33 native candidate steps, and the prior final metrics above. Its token IDs and every step’s hidden/logit metrics are now retained in the matrix CSVs.

The same canonical 33-token history was then supplied to both paths as fixed input (common-token replay). This changes segmentation to a 32-row prefill followed by a one-row decode at position 32; it does not compare different token histories. Both candidate steps were selected and executed on all 26 eligible layers, and the greedy top-1 remained equal at each captured output. Nevertheless, final hidden relative RMS was `0.0138161`, while final logits relative RMS was `0.0236490` / cosine `0.999728`, failing the unchanged gate. At the 32-row-prefill endpoint (position 31, 32 rows), logits relative RMS was already `0.0244612` / cosine `0.999703`; after the native decode at position 32 it was `0.0236490`. Both repetitions were bitwise repeatable at the token and numeric-capture hash level. This is a phase-segmentation/common-token numerical failure despite the original free-running pass.

A focused phase-isolation replay then used those exact same 33 input IDs, cancelling only after the final prompt token had been consumed:

| Phase path | Native candidate steps | Final hidden relative RMS | Final logits relative RMS / cosine | Result |
|---|---:|---:|---:|---|
| Native 32-row prefill, canonical decode at context 32 | 1 | `0.0133640` | `0.0225540` / `0.999754` | **Fail** |
| Canonical 32-row prefill, native decode at context 32 | 1 | `0.00892133` | `0.0146598` / `0.999893` | Pass |

The prefill-only and decode-only outcomes repeated twice with identical captures. The prefill checkpoint itself retained logits relative RMS `0.0244612`; the final canonical decode did not bring the result inside the gate. This agrees with, but does not replace, the earlier native-prefill negative evidence. Native decode is numerically compatible on this fixed history in isolation; it does not compensate for native-prefill error.

A separate short full-candidate check used the baseline 8-token prompt and one generated token. Its generated token matched canonical, and all nine one-row steps were selected at the 26 eligible blocks, but final logits relative RMS was `0.0301166` / cosine `0.999546` (hidden relative RMS `0.00982415`). The failure repeated with identical token and capture hashes. Thus same-token output parity and candidate eligibility do not guarantee the final numeric gate at every short trajectory.

The targeted session lifecycle smoke cancelled at the prompt boundary after length 8, then re-entered the same executor/session for the 8+1 run. The reused session ended at length 9, matched the fresh-session token and numeric-capture hashes exactly, and reset to length zero. This lifecycle check passed even though the 8+1 numerical gate failed; it is not numerical qualification.

**Observed matrix coverage is limited to capacity 512.** Although plans for 64 and 256 were registered from the manifest, the stop occurred before those model fixtures ran. No code-like, JSON-like, repeated-ID, alternating-ID, permuted-ID, or varied natural-language fixture in the broad manifest was executed. The direct native-AV contract still passes its synthetic capacity/type/row tests, and the execution-plan contract remains the candidate-guard oracle; those are not full-model capacity qualification.

Raw evidence is under `raw/autoregressive-fixture-matrix-20261009/matrix/`, `raw/common-token-phase-isolation-20261009/`, and `raw/session-reuse-lifecycle-20261009/`. The 20-row manifest is `qwen3-native-av-fixture-matrix-v1.tsv`; focused follow-up manifests are adjacent. The matrix summary preserves planned-but-unrun rows, per-position top-1 and numerical comparisons, native selection/execution counts, canonical AV layer-graph counts, token IDs/hashes, numerical-capture hashes, and repeatability results.

### 3. Capacity-dependent QK dispatch is upstream of AV

With canonical candidate selection disabled, capacity 512, 1024, and 1032 runs use the same prompt and fed decode token. At layer zero, active Q and K are bitwise identical across capacities, but decode QK scores change *before* softmax:

| Capacity comparison | Score relative RMS | Max absolute score delta | Probability relative RMS | Independent-softmax relative RMS |
|---|---:|---:|---:|---:|
| 512 → 1024 | `9.21567e-5` | `0.027832` | `2.55858e-4` | `1.25706e-7` |
| 512 → 1032 | `5.07646e-4` | `0.231689` | `0.00209968` | `1.38521e-7` |

The high-capacity QK score errors against the independent FP64 QK oracle are `9.48322e-5` and `5.13236e-4`, respectively. This confirms that softmax accurately processes the scores it receives; it does not explain the preceding score divergence.

Pinned-source dispatch conditions classify this one-row F16-K-by-F32-Q decode as MMVF through capacity 512, MMF at 1024, and cuBLAS at 1032. This classification is derived from the pinned GGML source and tensor geometry—not an Nsight kernel trace. No generic GGML or canonical QK behavior was changed.

The canonical hidden-state drift propagates through all 40 blocks but is not monotonic: relative RMS first reaches `0.00224051` / `0.00318904` after block 0 for capacities 1024 / 1032, peaks at block 13 (`0.0120159` / `0.0119964`), and ends at block 39 at `0.00570707` / `0.00657684`. Final-logit relative RMS is `0.0113341` / `0.0140255`; greedy top-1 is unchanged. This capacity effect exists with native candidate selection disabled and is upstream of AV.

### 4. Lifecycle and state correctness

The earlier intervention runner checks unchanged upstream block outputs and exact AV inputs, replays canonical execution A/B/A, and then runs canonical execution again after native-candidate trials with optimizer mode restored to `SHADOW`; both canonical replay checks were bitwise equal. The new sequence runner records actual requested/selected/executed candidate steps and phase, resets each isolated session, and verifies cancellation followed by same-executor/session re-entry against a fresh-session capture. The re-entry check passed. These checks do not qualify long generation, extended-context residency, varied prompts, or recovery under arbitrary backend failure.

### 5. Structural copy elimination vs. performance

The executor’s `packed_v_copy_bytes_avoided` counter records `capacity × 8 KV heads × 128 dimensions × 2 F16 bytes × 26 eligible blocks` for a native phase. At capacity 512 this is **27,262,976 bytes (26 MiB) per phase**; the combined prefill-plus-decode trial reports **54,525,952 bytes (52 MiB)** across two phases. This is a structural byte count for the full-capacity packed-V representation avoided, not a measured reduction in DRAM traffic or a timing result.

The 33-step sequence trial reports `899,678,208` logical avoided-copy bytes (33 × 27,262,976), also structural accounting only. Neither experiment measured candidate latency or established a speedup. Keep performance claims separate from these counters; earlier exploratory timing also did not qualify a speedup.

## Qualification recommendation

- Preserve canonical packed-V AV as authoritative and leave the native candidate unvalidated/experimental.
- Keep the existing capacity ≤512, exact 32-row prefill, decode-through-position-32, and SM 8.6 eligibility guard unchanged. Do not extend it to 1024/1032 or other devices.
- Do not modify canonical QK dispatch based on this prototype; the capacity effect is upstream and needs separate, broader GGML qualification.
- Do not alter the numeric thresholds or claim AV arithmetic is defective. Same-input AV agrees with the FP64 oracle; the unresolved concern is numerical compatibility across phases and propagation.
- Do not claim a speedup from the avoided-copy counter.

The reproduced common-token prefill-path failure, the repeated short 8+1 full-candidate gate failure, the prior isolated-prefill and layer-0 failures, the single-prompt free-running pass, and the lack of performance evidence keep native AV unqualified. The matrix stopped after its first new failure; do not interpret the planned but unrun fixtures as passes or broaden the guard. The earlier bounded free-running endpoint remains a valid pass, but it does not establish phase-independent numerical compatibility or production readiness.

## Reproduction and raw evidence

Build and run from the repository root. Start the local range server with the admitted model, then run the diagnostic target:

```bash
python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf \
  --port 18784 > /tmp/qwen-propagation-range.log 2>&1 &
range_server_pid=$!
trap 'kill "$range_server_pid" 2>/dev/null || true' EXIT
sleep 1

cmake --build /tmp/vbuf-native-av-build \
  --target vbuf_qwen3_native_av_boundary_diagnostic -j8

output_dir="research/results/vbuf-ml-integration/qwen3-numerical-propagation/raw/propagation-repro-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 /tmp/vbuf-native-av-build/vbuf_qwen3_native_av_boundary_diagnostic \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18784 \
  research/results/vbuf-ml-integration/qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv \
  "$output_dir" --propagation
```

### Bounded autoregressive sequence reproduction

The following follow-up holds the candidate inside its existing guard: an 8-token prompt is processed one row at a time, then 25 output tokens are generated, so the final candidate decode uses pre-decode context 32. It does not widen the context or capacity guard and is not a timing benchmark.

```bash
cmake -S integrations/ggml -B /tmp/vbuf-native-av-sequence-build \
  -DVBUF_GGML_SOURCE_DIR=/home/eugen/.cache/vbuf-agent-qualification/ggml \
  -DVBUF_ML_LIBRARY=/home/eugen/projekte/vBuf/rust/target/release/libvbuf_ml.so \
  -DVBUF_ENABLE_CUDA=ON -DVBUF_BUILD_PROBES=ON -DVBUF_BUILD_COMPAT_SERVER=OFF
cmake --build /tmp/vbuf-native-av-sequence-build \
  --target vbuf_qwen3_native_av_sequence_qualification -j8

python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf \
  --port 18784 > /tmp/qwen-native-av-sequence-range.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
sleep 1
set -o pipefail
out_dir="research/results/vbuf-ml-integration/qwen3-numerical-propagation/raw/sequence-repro-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out_dir"
CUDA_VISIBLE_DEVICES=0,1 /tmp/vbuf-native-av-sequence-build/vbuf_qwen3_native_av_sequence_qualification \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18784 \
  research/results/vbuf-ml-integration/qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv \
  8 25 512 "$out_dir/progress.csv" 2>&1 | tee "$out_dir/run.log"
```

### Bounded deterministic fixture matrix

The matrix runner remains a qualification-only diagnostic target. It opens a fresh session per compared path, records per-position output/candidate-selection captures, repeats each fixture twice, and stops after a new failure. Its status `STOPPED_ON_FAILURE` (exit code 2) is an expected research result, not a successful matrix completion. The checked-in manifest contains 20 planned fixtures; unrun rows are retained as `NOT_RUN_AFTER_STOP`.

```bash
cmake --build /tmp/vbuf-qwen-native-matrix-build \
  --target vbuf_qwen3_native_av_sequence_qualification -j8
python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf \
  --port 18786 > /tmp/qwen-native-av-matrix-range.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
sleep 1
out_dir="research/results/vbuf-ml-integration/qwen3-numerical-propagation/raw/matrix-repro-$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0,1 /tmp/vbuf-qwen-native-matrix-build/vbuf_qwen3_native_av_sequence_qualification \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18786 --matrix \
  research/results/vbuf-ml-integration/qwen3-numerical-propagation/qwen3-native-av-fixture-matrix-v1.tsv \
  "$out_dir"
```

The phase-isolation manifest uses a fixed common-token history to compare native-prefill/canonical-decode and canonical-prefill/native-decode. The session-reuse manifest performs the cancellation/re-entry check. Their raw outputs are retained separately and do not alter the broad matrix stop point.

The range server was stopped after each run. The propagation stdout/stderr log and manifest are retained at `raw/propagation-20261008-final/run.log` and `raw/propagation-20261008-final/manifest.txt`. Its summary CSVs include `capacity_layer_hidden.csv`, `capacity_qk.csv`, `local_av_metrics.csv`, `single_layer_summary.csv`, `single_layer_interventions.csv`, `all_native_prefill_layers.csv`, `all_native_decode_layers.csv`, and `native_candidate_summary.csv`; binary AV boundary and intervention captures are under that run's `raw/` subdirectory. Two matched-token sequence runs with identical results are retained under `raw/sequence-context32-20261009/` and `raw/sequence-context32-repeat-20261009/`, each with per-position hidden metrics in `progress.csv` and the run summary in `run.log`.

The earlier full diagnostic build succeeded and all 39 CTests passed. The expanded sequence/matrix target built in `/tmp/vbuf-qwen-native-matrix-build`; the focused `vbuf_qwen3_execution_plan_contract` and `vbuf_qwen3_native_attention_av_contract` tests passed (2/2), including fail-closed capability checks. The original 33-position free-running trial remains a pass, while the new common-token and short-run failures remain explicit `Candidate_NOT_Valid` evidence. Production defaults, optimizer default `SHADOW`, the native guard, and canonical execution authority were not changed.
