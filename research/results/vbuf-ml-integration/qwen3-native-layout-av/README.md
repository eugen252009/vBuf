# Qwen3 native-layout attention AV prototype

**Status: experimental; not production-qualified.** Canonical packed-V AV remains
authoritative. The native path is fail-closed to the exact Qwen3-14B CUDA
26/14 plan, SM 8.6, F16 V, F32 probabilities, I32 positions, 40 query heads,
8 KV heads, 128 dimensions, initial 32-row prefill or one-row decode with
pre-decode context at most 32, and capacities up to 512. Production defaults
and the optimizer's `SHADOW` default are unchanged.

## Operation contract

The prototype reads V directly in canonical cache order
`[dimension, position * 8 + kv_head]`, with the physical grouping
`[position, kv_head, dimension]`. For query head `h`, the associated KV head is
`floor(h / 5)`. It reduces F32 attention weights over positions `0..=last_visible`
and returns `[dimension, query, query_head]`. Unsupported capability/plan facts
select the existing packed canonical path.

The patched GGML API, CPU implementation, CUDA dispatch, and qualification source
are pinned to GGML commit
`2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`. The external GGML checkout's
pre-existing tracing modification was not changed.

## Evidence

Model identity for full-model trials was SHA-256
`f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
The real-model fixture used a 32-token natural-repeat prefill plus one decode
step, GPUs RTX 3060 (SM 8.6) / RTX 2080 SUPER (SM 7.5), and the 26/14 layer
placement. Raw logs are in `raw/`.

| Capacity | Real-model numeric gate | Result |
|---:|---|---|
| 64 | Pass; logits relative RMS 0.017962, cosine 0.999847 | Three paired executor-run samples; no material repeatable speed difference |
| 256 | Pass; same measured activation metrics as capacity 64 | Three paired samples; executor-run median ratio canonical/native 1.0046 |
| 512 | Pass; same measured activation metrics as capacity 64 | Three paired samples; final guarded-run executor median ratio 1.0102 |
| 1024 | **Fail**; logits relative RMS 0.020468, cosine 0.999804 | Tokens equal; exceeds the existing `0.02` / `0.9998` gate |
| 1032 | **Fail**; logits relative RMS 0.029135, cosine 0.999602 | Tokens equal; exceeds the existing gate |

At capacity 64, the three-sample executor-run medians were 84.7585 ms
canonical versus 84.6418 ms native (ratio 1.0014). Graph-setup medians were
1.4767 ms versus 2.3783 ms, respectively. The first canonical sample was a cold
191 ms observation; the median is reported without treating that sample as a
steady-state measurement. At capacities 256/512, graph setup also cost more on
the native path. These are small-context exploratory timings, not production
performance qualification or a speedup claim.

Direct CPU/CUDA synthetic tests cover F16/F32 V, 40:8 GQA, one-row and
32-row cases, and capacities 37, 64, and 1032. Native output agrees with the
ascending-position mathematical reference; packed CUDA AV differs by up to
about `9.52e-4` in this fixture. The real-model capacity-64 failure-injection
test also confirmed that a pre-boundary execution failure invalidates the
candidate, commits no session progress, and permits a fresh canonical session
to recover.

The 1024/1032 full-model measurements were explicit diagnostic trials before
the capacity guard was tightened. They remain the recorded native-candidate
failures; the current qualification executable rejects capacities above 512.
The candidate remains bounded to capacity 512 and the measured context window
(initial prefill plus decode through position 32). Direct-op support at 1032 is
not evidence of full-model correctness.

## Capacity-dependent divergence investigation

The follow-up diagnostic reproduced the real-model AV boundary at capacities
256, 512, 1024, and 1032 using the same 32-token prompt and 26/14 placement.
For each run, candidate selection stayed disabled. A diagnostic-only native AV
side branch consumed the *same* V cache, attention probabilities, and positions
as canonical `GGML_OP_MUL_MAT`; its output was checked against an independent
ascending-position FP64-accumulation AV oracle. The capture records prefill and
the decode at visible context 33 for all 26 SM 8.6 layers, plus layer-zero Q/K,
scores, probabilities, strides, and layouts. The separate
`vbuf_qwen3_cuda_core_av_diagnostic` target compiles this instrumentation behind
`VBUF_QWEN3_AV_BOUNDARY_DIAGNOSTIC`; the ordinary runtime library and default
executor API do not contain or enable it. A capacity-512 run verified the
side-branch instrumentation left canonical final hidden and logits bitwise
unchanged. Two complete sweeps produced bitwise-identical saved boundary
captures.

Results localize the capacity dependence to **decode layer-zero QK scores, not
native AV indexing or softmax**:

- Active layer-zero query and K values were bitwise identical across capacities.
  Prefill V/probability inputs and both AV outputs were also identical across
  capacities for the same logical context.
- At decode position 32, capacity 512 and 1024 had different layer-zero QK
  scores despite identical active Q/K: relative RMS `9.2157e-5`, max absolute
  difference `0.027832`. Capacity 512 versus 1032 differed by relative RMS
  `5.0765e-4`, max absolute `0.231689`.
- The pinned GGML CUDA dispatcher explains the switch (derived from the exact
  tensor geometry and pinned source; not an Nsight kernel trace): for this F16
  K-by-F32-Q decode matmul with one query row, capacities through 512 select
  MMVF; capacity 1024 disables MMVF and satisfies MMF's 32-row divisibility;
  capacity 1032 disables MMVF and fails MMF's 32-row divisibility, falling back
  to cuBLAS. The relevant conditions are `mmvf.cu::ggml_cuda_should_use_mmvf`,
  `mmf.cu::ggml_cuda_should_use_mmf`, and their order in `ggml-cuda.cu::ggml_cuda_mul_mat`.
- The measured probabilities agree with an independent stable softmax computed
  from the captured actual scores to relative RMS `1.26e-7` at capacity 1024
  and `1.39e-7` at 1032. The high-capacity score differences therefore arise
  before softmax. Against the independent FP64 QK oracle, score relative RMS was
  `9.48e-5` at 1024 and `5.13e-4` at 1032.
- On identical captured AV inputs, native AV stayed within at most
  `1.02e-7` relative RMS of its FP64 AV oracle. Canonical packed-V matmul
  differed from that oracle by up to `3.251e-4`; canonical-versus-native AV
  relative RMS was also at most `3.251e-4` in the capture sweep. This is a
  reduction-order difference, present at all tested capacities, not a layout
  or GQA-indexing error.
- Canonical full-model output itself changed with capacity: versus capacity
  512, hidden/logit relative RMS was `0.005707 / 0.011334` at 1024 and
  `0.006577 / 0.014025` at 1032 (greedy token unchanged). Together with the
  layer-zero score differences, this shows canonical numerical execution is
  capacity-dependent even with candidate selection disabled. The retained
  full-model native-candidate runs still
  fail the canonical parity gate at 1024 (`0.020468` logits relative RMS) and
  1032 (`0.029135`); those failures are reproducible in the original
  pre-guard logs, while the current candidate guard remains closed above 512.

**Correction decision:** no AV arithmetic/indexing change is justified. The
native AV output agrees with the independent mathematical oracle; changing it
to imitate a capacity-dependent canonical reduction would make the operation
less mathematically direct and would not fix the upstream QK kernel switch.
Changing generic GGML or canonical Qwen execution is outside this prototype's
scope. Preserve canonical authority, the capacity-512 candidate guard,
all fallbacks, and production defaults. The high-capacity numerical risk is
resolved enough to explain the observed drift, but not to qualify native AV at
production capacity 1,032.

Raw diagnostic logs and compact binary captures are under `raw/av-boundary-qk/`
and `raw/av-boundary-qk-repeat/`; the earlier prefill-only and initial
context-33 captures are retained in the sibling `raw/av-boundary*` directories.
The original high-capacity full-model failures remain in
`raw/cap1024-natural32-diagnostic.log` and
`raw/cap1032-natural32-diagnostic.log`.

`packed_v_copy_bytes_avoided` counts graph-level V-copy payload bytes, not
measured memory traffic or bandwidth. At capacity 1032 the old diagnostic trial
reported 109,903,872 logical copy bytes avoided, but failed numeric
qualification.

## Multi-fixture sequence qualification update

The original 8-token plus 25-generated-token free-running baseline reproduced
twice with identical token IDs and passed the existing final gate. A common-token
replay of the same 33-token history, processed as a 32-row prefill plus decode
at context 32, failed the logits gate (`0.0236490` relative RMS). Phase-isolated
replay showed native prefill alone failed (`0.0225540` final logits relative
RMS), while native decode alone passed (`0.0146598`) on that exact fixed input.
A separate 8+1 full-candidate sequence also failed (`0.0301166` logits relative
RMS) with its generated token unchanged. Each result repeated deterministically.

The planned 20-fixture matrix stopped after the first new common-token failure;
only the baseline fixture ran from that matrix. Its other prompts, token patterns,
and capacities were not tested. The targeted session-reuse smoke confirmed
cancel-at-prompt-boundary followed by same-executor re-entry returns the same
captures as a fresh session, but this does not resolve the numerical failure.
No guard, threshold, production setting, or canonical fallback was changed, and
no speedup is claimed. Detailed per-position evidence and the exact manifests are
in [`../qwen3-numerical-propagation/README.md`](../qwen3-numerical-propagation/README.md).

## Reproduction

Configure/build with the pinned patched GGML preparation and local vBuf-ML
library, then run the direct contract and narrow lifecycle tests:

```bash
cmake -S integrations/ggml -B /tmp/vbuf-native-av-build \
  -DVBUF_GGML_SOURCE_DIR=/home/eugen/.cache/vbuf-agent-qualification/ggml \
  -DVBUF_ML_LIBRARY=/home/eugen/projekte/vBuf/rust/target/release/libvbuf_ml.so \
  -DVBUF_ENABLE_CUDA=ON -DVBUF_BUILD_PROBES=ON -DVBUF_BUILD_COMPAT_SERVER=OFF \
  -DVBUF_BUILD_REAL_GATE2B=OFF
cmake --build /tmp/vbuf-native-av-build \
  --target vbuf_qwen3_native_attention_av_contract vbuf_qwen3_native_av_qualification \
    vbuf_qwen3_native_av_boundary_diagnostic -j8
ctest --test-dir /tmp/vbuf-native-av-build \
  -R 'vbuf_qwen3_native_attention_av_contract|vbuf_qwen3_execution_plan_contract|vbuf_qwen3_cuda_ownership_contract' \
  --output-on-failure
```

The real-model runner requires the admitted semantic artifact, local range
server, and natural-repeat token fixture used for the raw runs. The capacity-512
trial can be replayed with:

```bash
python3 scripts/range_server.py \
  --file /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.vbuf \
  --port 18784
CUDA_VISIBLE_DEVICES=0,1 /tmp/vbuf-native-av-build/vbuf_qwen3_native_av_qualification \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18784 \
  research/results/vbuf-ml-integration/qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv \
  512
```

The diagnostic-only same-input sweep (does not select the native candidate) is:

```bash
CUDA_VISIBLE_DEVICES=0,1 /tmp/vbuf-native-av-build/vbuf_qwen3_native_av_boundary_diagnostic \
  /home/eugen/.cache/vbuf-agent-qualification/qwen3-14b-full/Qwen_Qwen3-14B-Q4_K_M.semantic.vbuf \
  http://127.0.0.1:18784 \
  research/results/vbuf-ml-integration/qwen3-multigpu-26-14-capacity-qualification/raw/natural-repeat-token-ids.csv \
  /tmp/qwen3-av-boundary-output 256 512 1024 1032
```

The follow-up all-40-block propagation, isolated intervention, and candidate lifecycle results are documented in [`../qwen3-numerical-propagation/README.md`](../qwen3-numerical-propagation/README.md); its authoritative raw output is under `../qwen3-numerical-propagation/raw/propagation-20261008-final/`.

Keep qualification mode explicit; do not mark this candidate valid or enable it
for production based on the smaller-capacity passes.
