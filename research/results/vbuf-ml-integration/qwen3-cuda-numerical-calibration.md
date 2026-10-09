# Qwen3-14B CUDA tensor-operation calibration

**Disposition: measured diagnostics only; no new tolerance or admission authority.**
The canonical packed-V attention path remains authoritative. Native-layout AV
remains experimental and unpromoted. The versioned Numerical Contract policy,
its active thresholds, production defaults, fail-closed guards, and execution
admission behavior were not changed.

## Identity and evidence boundary

- Canonical model: Qwen3-14B Q4_K_M vBuf, SHA-256
  `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Source GGUF used only to recover the captured layer-21 RMSNorm scale:
  SHA-256 `915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6`.
- GGML: `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`; placement remains
  `multi:0x26,1x14;emb=0;norm=1;head=1`.
- Devices: RTX 3060 / SM 8.6 and RTX 2080 SUPER / SM 7.5.
- Prompt evidence: one 32-token natural-repeat prefill plus one decode at
  visible context 33. The separate layer-21 operator capture covers a 32-row
  prefix. No full-model execution was started for this calibration.

The Python runner verifies the model SHA in the original run logs, hashes every
capture and metadata file, confirms the active Q and K tensors are byte-identical
across the 256/512/1024/1032 capacity sweeps and their repeated sweep, and records
all source and output hashes. The full capture payloads remain at their original
paths; only the small QK replay outputs are stored with the new run. Captures are
historical evidence, and all evaluation records remain replay-only.

## QK: observed MMVF, MMF and cuBLAS dispatch

The C++ runner replays captured F32 post-RoPE Q and F16 K through the pinned
GGML CUDA `ggml_mul_mat` at physical capacities 512, 1024 and 1032, for both
prefill and decode, on both devices. Temporary branch instrumentation was
applied only to the generated GGML build copy, recorded in the raw directory,
and restored before the tool exited. It did not modify the external GGML
checkout or repository production code.

| Phase / shape | Capacity | Observed dispatch | Relative RMS vs independent FP64 QK |
|---|---:|---|---:|
| Decode, 1 query row | 512 | MMVF | `1.33941574e-4` |
| Decode, 1 query row | 1024 | MMF | `9.48332056e-5` |
| Decode, 1 query row | 1032 | cuBLAS | `5.13229118e-4` |
| Prefill, 32 query rows | 512 / 1024 / 1032 | cuBLAS | `5.16968543e-4` |
| Layer 21 prefill, 32 rows | 32 | cuBLAS | `4.69145392e-4` |

Prefill does not select MMF even when capacity is divisible by 32: the pinned
`ggml_cuda_should_use_mmf` predicate declines non-`mul_mat_id` batches with more
than 16 query columns. For decode, 512 selects MMVF; 1024 declines MMVF and
satisfies MMF row divisibility; 1032 declines MMVF and fails MMF divisibility,
falling through to cuBLAS. The branch routes were observed on both tested GPUs,
not inferred solely from the numeric error values.

Across all 12 controlled replays (three capacities × two phases × two devices),
the active CUDA QK output was bitwise identical to the matching saved full-model
capture. This confirms the operation-only replay uses the same active Q/K and
reproduces the captured score calculation, including the capacity-dependent
route. The CUDA candidates also had identical observed metrics on the two
machines for these inputs. This is not a general cross-GPU equivalence claim.

The relative RMS values are not tolerance recommendations. They are one captured
sequence/layer with controlled physical geometry variants; repeated sweeps and
device replay do not add independent prompts or activation samples. In
particular, the higher cuBLAS error at capacity 1032 is an observed result, not a
basis to change a threshold or dispatch rule.

## Other operations with suitable captures

The independent reference uses binary64 accumulation/products and rounds the
operation result once to F32. Captured GGML layouts are converted to the logical
row/head/position representation before comparison.

| Operation and capture | Candidate vs FP64 relative RMS | Additional result |
|---|---:|---|
| CUDA softmax, layer 0 prefill, capacities 256/512/1024/1032 | `1.11530376e-7` | Probability range, row normalization and masked-zero invariants hold |
| CUDA softmax, layer 0 decode, capacities 256/512 | `1.27747012e-7` | Same invariants hold |
| CUDA softmax, layer 0 decode, capacity 1024 | `1.25706156e-7` | Same invariants hold |
| CUDA softmax, layer 0 decode, capacity 1032 | `1.38520828e-7` | Same invariants hold |
| CUDA softmax, layer 21 prefill | `1.21182081e-7` | Same invariants hold |
| CUDA RMSNorm + scale, layer 21, 32 × 5120 | `6.17027205e-8` | max absolute error `1.90734863e-6`; epsilon `1e-6` and exact F32 scale identity recorded |
| Canonical packed-V AV, layer 0 prefill | `2.63737050e-4` | Across tested capacities; same logical inputs |
| Native-layout AV, layer 0 prefill | `4.23411729e-8` | Diagnostic side branch only |
| Canonical packed-V AV, layer 0 decode | `2.36189347e-4`–`2.45790612e-4` | Capacity-dependent captured scores/probabilities |
| Native-layout AV, layer 0 decode | `5.02574707e-8`–`5.82098111e-8` | Diagnostic side branch only |
| Canonical packed-V AV, layer 21 prefill | `3.05187672e-4` | Captured CUDA operation output |

The RMSNorm scale was recovered as `blk.21.attn_norm.weight:F32[5120]` from the
registered source GGUF and bound by SHA-256
`dd37e4a627d56a2f28c6d513b1a46f24a892e69f0065cc42957460ef2a41f59d`. RMSNorm
compares the captured output of `ggml_rms_norm` followed by elementwise scale
multiplication against the independent operation formula.

These AV accuracy observations do **not** establish canonical/native AV
compatibility, end-to-end parity, or native-AV qualification. Existing
prefill/topology and capacity failures remain; canonical packed-V remains the
production authority. The general CUDA matmul and quantization reconstruction
contracts are not calibrated here: the available Q4_K/Q8_K slice is not a
matched CUDA kernel capture, and no verified unquantized source tensor was used.

## Contract proposal and disposition

[`contracts-proposal-v1.json`](qwen3-cuda-numerical-calibration/contracts-proposal-v1.json)
proposes only candidate-/operation-specific evidence fields for QK, softmax,
RMSNorm and AV. It is explicitly `PROPOSED_NOT_ACTIVE`; every threshold is
`null`. Existing QK, softmax and RMSNorm contracts remain `NEEDS_CALIBRATION`.
The AV proposal distinguishes canonical packed-V from experimental native AV
and keeps mathematical accuracy separate from canonical compatibility. No
threshold, status, contract scope, policy version, admission requirement, or
production selection was activated or relaxed.

The machine-readable evaluation records, input/output hashes, exact dispatch
logs, branch-trace patch, and replay-vs-full-capture comparisons are under
[`raw/calibration-20261009/`](qwen3-cuda-numerical-calibration/raw/calibration-20261009/).
The run contains 50 non-authorizing records; the numerical-contract results are
`NOT_TESTED` / `NEEDS_CALIBRATION` and `replayed_metrics_only=true`.

## Reproduction

Build the opt-in qualification executables with CUDA and unvalidated trials OFF,
then run the manifest-bound harness. It requires the saved capture paths and the
registered source GGUF for the RMSNorm scale; it does not load or run the full
model.

```bash
cmake -S integrations/ggml -B /tmp/vbuf-qwen-native-matrix-build \
  -DVBUF_ENABLE_CUDA=ON \
  -DVBUF_GGML_SOURCE_DIR=/home/eugen/.cache/vbuf-agent-qualification/ggml \
  -DVBUF_GGML_COMMIT=2d191b5dee1a591c41ee8a653ce42bfcd9c8716d \
  -DVBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS=OFF
cmake --build /tmp/vbuf-qwen-native-matrix-build \
  --target vbuf_qwen3_reference_capture_qualification vbuf_qwen3_cuda_qk_calibration -j2
python3 integrations/ggml/qualification/qwen3_numerical_calibration.py \
  --build-dir /tmp/vbuf-qwen-native-matrix-build \
  --output-dir research/results/vbuf-ml-integration/qwen3-cuda-numerical-calibration/raw/calibration-20261009
```

The harness also supports `--validate-only`, which verifies source identity,
capture geometry, model/GGUF hashes and repeated Q/K identity without launching
CUDA replays.

## Verification

The complete CTest suite passed **45/45** with
`VBUF_ENABLE_UNVALIDATED_QUALIFICATION_TRIALS=OFF`, passed **45/45** with it
explicitly ON, and passed **45/45** again after restoring OFF. The final build
configuration is OFF. No push was performed.
