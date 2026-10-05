# Qwen3 Sequence-A Prefix-25 Outlier and CUDA Probe

Status: **Phase A found no structural/runtime correctness defect in the investigated CPU path. The bounded CUDA block failed cross-backend numerical parity. Both the Q6_K V-projection control and the restored Q4_K Q-projection case are now classified as expected backend-specific activation-quantization differences, not vBuf payload/decoder defects. GPU depth qualification remains blocked pending exact CUDA residency/transfer accounting.** This is an uncommitted investigation record. It does not enable production Qwen3 or change the external llama.cpp `1e-5` gate.

## Phase A — prefix-25 classification

Sequence A was replayed with the recorded 25-token teacher-forced input:

```text
0,25,220,16,13,15,13,15,198,262,549,0,220,16,13,15,13,15,198,262,549,0,220,16,13
```

The vBuf Q4 source identity was SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`. BF16 reference artifact revision/hash remain those in the higher-precision report (`bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2`, SHA-256 `9677a58b0fa8da7771a4d8cc8080208ce02a32ab21305ded10a3798766132d3a`). The llama.cpp Q4/BF16 comparison used GGML revision `a97123e497968f3440264c0464a7adc7c999c027`; the vBuf qualification build uses pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`.

The outlier is an **isolated, depth-sensitive numerical divergence**, not a demonstrated structural/runtime defect:

- Final-hidden vBuf-vs-BF16 relative RMS was `0.0613` at prefix 24, `0.1367` at prefix 25, and `0.0698` at prefix 26. At prefix 25 llama Q4-vs-BF16 was `0.0520`. Both Q4 paths retain BF16 top-5/top-10 overlap of 5/5 and 10/10 and choose token 15; this is not numerical parity.
- Across layers at prefix 25, vBuf-vs-BF16 relative RMS grows from `0.2203` at layer 20 to `0.2835` at 21, `0.5305` at 23, and `6.3076` at 39. Llama Q4-vs-BF16 at those layers was `0.2117`, `0.2359`, `0.2969`, and `2.4007`. This is progressive divergence with amplification around layers 20–24, rather than a uniquely bad layer-21 output.
- A Q4 layer-21 reference-input control using the captured llama-Q4 layer-20 input reproduced the llama-Q4 layer-21 output with RMS difference about `1.9e-7` (relative about `2.2e-7`). On the same Q4 input, the Q/K/V stages matched exactly; the first measurable attention-score differences were small (RMS about `5.2e-6`, relative about `6.4e-8`), with small probability/context/residual differences. The vBuf-own-input control reproduced the full-path intermediates; prefix-25 serial and parallel exports were byte-identical across 2,123 files.
- The prefix-25 causal extent transition was checked: query position 24 uses logical length 25 and compute extent 32. Its physical padded score/probability entries were zero; probabilities over visible keys summed to `0.99999982..1.00000012`. Earlier rows retain causal masking. No extent/padding defect was established.

**Classification:** a real and still-unexplained numerical-sensitivity outlier between the Q4 execution lineages. The same-input controls, causal/padding checks, and serial/parallel identity rule out the main suspected local operator, extent, and scheduler defects in the vBuf CPU path. They do not establish the precise source of accumulated drift or make BF16 ground truth. This was sufficient to proceed to a bounded GPU diagnostic, not to declare Qwen numerically qualified.

## Phase B — bounded CUDA diagnostic

### Infrastructure and resource boundary

- CUDA 12.4.131, host compiler g++ 13.3.0, driver 550.163.01; RTX 3060 (SM 8.6, 12,288 MiB) and RTX 2080 SUPER (SM 7.5, 8,192 MiB) are visible.
- The existing generic Rust CUDA path is F32-oriented and does not provide Q4_K/Q6_K matrix execution. The pinned GGML CUDA backend was therefore built out-of-tree with the existing `VBUF_ENABLE_CUDA=ON` option. Build succeeded for pinned GGML `2d191b5`; Q4_K and Q6_K CUDA kernels were compiled for SM 8.6 and 7.5.
- At the probe, GPU 0 had 980 MiB free and GPU 1 had 877 MiB free. The Q4 artifact is 9,000,232,144 bytes, so full-model GPU preload was neither assumed nor attempted. Existing GPU processes were left untouched. After the CUDA process exited, GPU 0's system-reported usage matched its pre-run baseline; GPU 1 usage fluctuated between observations. Exact per-process peak device residency and H2D/range accounting were **not instrumented**, so no peak-memory or transfer qualification is claimed.
- The CUDA run was an out-of-tree diagnostic variant of `qwen3_block_qualification`, selecting the GGML CUDA backend while keeping vBuf-ML metadata/source resolution and range materialization. It did not use `llama_model_loader`, change tracked production code, or route acquisition policy through GGML. This is not production backend selection and does not qualify the canonical Rust generic CUDA runtime for Qwen.

### Real graph exercised and result

A complete layer-0 Qwen3 block at 25 positions ran on CUDA from a fixed Q4 reference embedding input (SHA-256 `e303824822361dd11d86b2886f486f11ff5a2ea1f4fcb10d503504fc5da06459`). The CPU and CUDA runs read the same cropped input bytes. vBuf supplied the materialized weights. The block exercised real quantized matrix operations, including the Q6_K FFN-down tensor (`[17408,5120]`, 73,113,600 bytes); its CUDA readback hash matched the materialized payload (`fnv1a64=a4c3145d3cafb9c3`). A separate layer-21 diagnostic exercised a Q4_K FFN-down tensor, but is not counted as a parity-qualified gate. The probe was in internal-only mode: a zero softmax checkpoint was supplied solely to satisfy a reference-only padded-context diagnostic; it was not a forward-path input or a CPU/CUDA expected output. CPU/CUDA metrics below were computed independently from the separately exported forward outputs.

All block outputs were finite, but CPU-vs-CUDA comparison on the same input **failed**. Selected results (RMS error / CPU-relative RMS) were:

| Stage | Max absolute difference | RMS difference | Relative RMS |
|---|---:|---:|---:|
| Q projection (first measured divergent stage) | 0.00684524 | 0.00109411 | 1.608% |
| K projection | 0.0115743 | 0.00132789 | 1.322% |
| V projection | 0.00166902 | 0.000332489 | 1.096% |
| Attention scores | 2.57440 | 0.480209 | 0.716% |
| Attention context | 0.00879531 | 0.000286875 | 1.184% |
| Q6_K FFN down | 0.0737758 | 0.00212083 | 1.396% |
| Block output | 0.0870199 | 0.00244170 | 1.060% |

These are not within the existing `1e-5` absolute comparison criterion. This local CPU/CUDA failure is a new qualification result, **not** a change to or waiver of the separate strict external llama.cpp gate, which remains **FAIL**. No top-1-only or token-agreement result is substituted for numerical parity.

### Follow-up — fixed-input Q6_K activation-path control

To distinguish weight/input identity from backend arithmetic, block-0 V projection was replayed as a single direct GGML `MUL_MAT` with a fixed activation override and the captured vBuf-materialized Q6_K payload. The override file is SHA-256 `d1cc5ba95059d9ea136b6967e1a57df56d3a276f9873f9e463e4bfc71d6883df`; both CPU and CUDA runs name this same override path. The Q6_K weight payload is byte-identical in both runs (SHA-256 `513873cbe0d55faaf0e38ea382849d3e4f45b661fb10caf810b9f1639f6270ac`). All 5,242,880 weight values had already been decoded independently and matched pinned GGML decoding bit-for-bit.

The matched-identity Q6_K control strongly localizes the CPU/CUDA output delta to their activation quantization paths:

- Pinned GGML CPU selects Q8_K for Q6_K matmul; Q8_K quantizes each 256-value block with one F32 scale. Its actual Q8_K quantization/dequantization on the fixed F32 input has relative activation RMS error `2.029%`.
- On the RTX 3060, pinned GGML source dispatch selects MMQ for this 25-column case. Q6_K MMQ uses its Q8_1 D4 input layout: one F32 scale per 32 values (four scales within each 128-value MMQ block). A host reproduction of the pinned CUDA quantizer algorithm on the same bytes gives relative activation RMS error `0.926%`; the internal temporary CUDA quantization buffer itself was not instrumented or read back.
- These distinct reconstructions differ at 109,856 of 128,000 activation elements. The CUDA reconstruction is closer to the original F32 input; this is consistent with CUDA being closer to the F64 projection oracle, but alone would not prove that either matmul uses the expected representation.
- The stronger control is the output comparison against independent F64 matrix products of the exact decoded Q6_K weights and each backend's own reconstructed quantized activation. CPU output vs Q8_K oracle relative RMS is `1.45e-7`; CUDA output vs Q8_1-MMQ oracle is `3.35e-5`. Cross-comparison is `1.096e-2`. Against the unquantized F32-input F64 projection, CPU is `1.116%` and CUDA is `0.511%`.

This rules out a Q6_K payload mismatch or a vBuf-specific matmul implementation in this control: the direct GGML primitive outputs track their respective backend activation representations. It identifies expected CPU-vs-CUDA quantization-path divergence as the cause of the measured Q6_K V-projection mismatch, rather than a demonstrated CUDA defect. The control did **not at that stage** test the first divergent Q projection: `blk.0.attn_q.weight` is Q4_K, whose CPU Q8_K and CUDA Q4_K-MMQ DS4 paths have different details. The later Phase C follow-up restores and classifies that exact 1.608% case with Q4_K-specific controls. No depth progression followed.

Temporary activation buffers and the reproduction are under `/tmp/qwen3-q6v-activation`; no tracked production code changed.

### Phase C — restored Q4_K fixture and activation-path diagnosis

The exact Q4_K_M vBuf artifact was restored from the persistent local qualification cache and its SHA-256 verified as `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`. Sequence-A's exact first 25 tokens, the shared `attn_norm` F32 projection input, Q4_K payload, and CPU/CUDA Q outputs are persisted outside `/tmp` under `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/fixture/`. A Git-tracked fixture manifest records the artifact/tensor/input/output identities, geometry, baseline metrics, and rerun instructions at `research/results/vbuf-ml-integration/qwen3-cuda-fixtures/prefix25-q4k-qproj-manifest.json`.

The original approximately 1.608% CPU/CUDA mismatch was reproduced with the same exact F32 Q input for both backends. The `blk.0.attn_q.weight` payload is Q4_K, shape `[5120,5120]`, 14,745,600 bytes at vBuf byte offset `442028624`; its SHA-256 is `3c4dd39531acff6eb56af4dfb0e39fed24ec3a71e6efe958a96c944596845bf9`. A direct read of that byte range from the verified 9,000,232,144-byte vBuf artifact, CPU materialization, CPU backend readback, CUDA upload source, and CUDA backend readback all produced the same payload hash. Each backend used the shared F32 `[5120,25]` input (SHA-256 `d1cc5ba95059d9ea136b6967e1a57df56d3a276f9873f9e463e4bfc71d6883df`). The reproducible capture script replays CPU and CUDA and verifies the input, weight, and output hashes.

The matched-input Q-projection result is max absolute error `0.0068452358`, RMS error `0.0010941095`, relative RMS `1.608385%` using CPU RMS as denominator (`1.607145%` using CUDA RMS), cosine `0.99987105`, and norm ratio CPU/CUDA `0.99922925`. CPU output hash `1e40f747...b623851d` also matches the llama.cpp Q4 `Qcur` checkpoint byte-for-byte. This reproduces the historical baseline; it is not a numerical parity pass.

#### Q4_K decoder and activation-path controls

- A separately implemented packed Q4_K decoder expanded all 26,214,400 weights and matched the pinned GGML Q4_K `to_float` output **bit-for-bit** (zero mismatching elements). This validates the packed weight identity and excludes a Q4_K decode discrepancy in the oracle.
- On the CPU path, the pinned Q4_K CPU type trait reports Q8_K as its activation `vec_dot_type`. Reconstructing Q8_K activations from the shared input and taking an FP64 product with the independently decoded weights matches the actual CPU projection at relative RMS `8.45e-8` (max absolute `6.07e-8`). This is strong evidence that the real CPU projection used the expected Q8_K activation representation.
- On the RTX 3060 (SM 8.6), the diagnostic actually ran with GGML backend `CUDA0`, one thread, pinned GGML `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`. For this 25-column F32-input operation, the pinned CUDA dispatcher rejects the quantized Q4_K path from MMVF/MMF, the 25-column case exceeds MMVQ's small-batch path, and Q4_K is supported by MMQ; on this Turing-MMA-capable device with the required shared memory, dispatch selects MMQ. Q4_K maps to Q8_1 DS4 (`mmq.cuh`); the source-defined DS4 quantizer (`quantize.cu`) groups 32 activations, rounds to int8, stores an FP16 scale, and stores a separate FP16 sum of the original pre-quantized F32 values. The Q4_K affine-minimum term consumes that partial sum (`vecdotq.cuh`); therefore a simple vector `q * scale` is not the CUDA kernel's complete effective activation representation.
- Oracle A (FP64 dot of independently decoded Q4_K weights and the original common F32 input) differs from CPU output by `0.4675%` relative RMS and CUDA output by `1.5422%`. This is a comparison, not a claim that the unquantized product is ground truth.
- Oracle B (FP64 dot using CPU Q8_K effective activations) matches the CPU output at `8.45e-8` relative RMS. A naive CUDA Q8_1 `q * FP16-scale` vector oracle differs from CUDA output by `1.5329%` relative RMS because it omits the DS4 auxiliary sums.
- Oracle C models the pinned CUDA Q4_K MMQ affine algebra: the int8 dot term uses the FP16 activation scale, while the Q4_K minimum correction uses the separate FP16 partial sum. It matches the actual CUDA output at relative RMS `0.01587%` (max absolute `9.36e-5`, cosine `0.9999999874`). The custom CUDA-effective oracle is about 96 times closer to actual CUDA output than the naive `q * scale` vector oracle.

**Classification: EXPECTED BACKEND-SPECIFIC ACTIVATION-QUANTIZATION DIFFERENCE.** The original 1.608% result is explained by the CPU Q8_K all-quantized activation path versus CUDA Q8_1 DS4's scaled-int8 dot plus separately summed affine correction. Exact input and Q4_K weight identity, bit-identical independent/pinned weight decoding, CPU Q8_K oracle agreement, and close CUDA DS4-MMQ oracle agreement exclude a vBuf range/payload mismatch and a Q4_K decoder mismatch for this case. The cross-backend outputs remain numerically different; no token agreement or relaxed tolerance is used to call them parity. The internal CUDA quantization temporary was reconstructed from the pinned kernel and validated by the output oracle; it was not directly copied back from device memory. The CUDA kernel-family selection is supported by the pinned deterministic dispatch conditions and oracle agreement, not a separately instrumented kernel-name log.

Replay and analysis scripts are in `research/results/vbuf-ml-integration/qwen3-cuda-fixtures/`: `reproduce-prefix25-q4k.sh` reproduces and hash-checks the matched CPU/CUDA baseline; `run-prefix25-q4k-oracles.sh` recreates the CPU Q8_K and pinned-GGML Q4_K decodes, independently decodes all weights, reconstructs CUDA DS4 activation metadata, and writes the oracle results to the persistent fixture directory. The numeric summary is also preserved in `prefix25-q4k-activation-oracle-results.json`. The CUDA capture remains a bounded one-block diagnostic; no production behavior changed.

### Stop decision

**Do not progress to two or more CUDA layers, full-sequence GPU execution, generation, or GPU performance claims yet.** Both the Q6_K V-projection and the first Q4_K Q-projection CPU/CUDA deltas now have bounded backend-quantization explanations, but exact CUDA residency and transfer accounting are still absent and the strict cross-backend numerical gate still fails. The Q4_K fixture is reproducible and classified; this does not qualify deeper CUDA execution. Keep production Qwen3 disabled and the strict external `1e-5` threshold unchanged.

## Evidence locations

The Q4_K model, shared input, packed weights, CPU/CUDA outputs, reconstructed activations, and oracle outputs are persisted under `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/fixture/`; the CUDA build/source copy is likewise outside `/tmp`. Earlier temporary builds and logs remain under `/tmp/vbuf-qwen3-cuda-build` and `/tmp/qwen3-prefix25-layer{0,21}-*`; no production behavior changed. The new manifest and diagnostic scripts are uncommitted investigation artifacts. Both Git stashes remain untouched (`stash@{0}` Android CMake hunk; `stash@{1}` pre-existing stash).
