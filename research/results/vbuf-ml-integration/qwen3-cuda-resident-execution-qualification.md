# Qwen3 CUDA Resident Execution Foundation Qualification

## Scope and decision

This qualifies a **diagnostic, qualification-only** CUDA resident execution path for Qwen3. It is not the production `VbufGenerationSession`, does not enable Qwen3 in production, and does not change the strict external llama.cpp `1e-5` numerical gate. HTTP/tools/Pi work was not performed.

The prior host-staging audit was not a suitable basis for GPU depth progression. A bounded vBuf-owned device residency tier was added to `TensorResidencyStore`, and the qualification graph now materializes model payloads through the vBuf materializer, uploads weights once, records each allocation under stable artifact/tensor/source/representation/shape/backend/device identity, and holds residency leases during execution. GGML allocates graph tensors on CUDA; token embedding lookup and all dependent block activations execute in one device graph. The model-facing production path remains unchanged.

The one-block repeated-run gate passed, so the required two-block device chain was then run. **No depth beyond two blocks was attempted.**

## Artifact and controls

- Artifact: Qwen3-14B-Q4_K_M, SHA-256 `f409ec946faf59cb338647c47efc2f28e9bdec7e8bd33acacd0cd6a36f2eaa31`.
- Source: local HTTP range server serving the vBuf artifact; source identity and payloads were validated by the existing qualification harness.
- Device: NVIDIA RTX 3060, CUDA device 0, compute capability 8.6, 12,037 MiB reported VRAM.
- Input: 25 token IDs, reused unchanged across baseline and resident runs.
- Baselines: separately captured legacy CUDA block outputs from the same artifact and input. Output hashes:
  - Block 0: `ad8689980e3c308675557c490665bb0092563a6da91bb37c4dfa5dd5d32700bf`
  - Blocks 0–1: `2a05016de7b463454b61d6691db809bb4a5f383bda30b380a474542797c1a3a8`

## Results

### One block, three resident executions

- Resident tensor payload: **647,742,464 bytes** across 12 weights, including the **437,575,680-byte embedding table**.
- GGML CUDA graph buffer: **663,254,272 bytes**, of which 15,511,808 bytes are non-weight graph/work tensors.
- Initialization: 15 H2D calls totaling **647,745,164 bytes**: 647,742,464 weight bytes plus 2,700 bytes for token IDs, position IDs, and causal mask.
- Each of three executions: **zero H2D calls/bytes**; one 512,000-byte final block-output readback. The first execution also read back 512,000 bytes of selected embedding rows for the CPU decode control.
- CUDA-side embedding gather matched CPU decoding exactly (max absolute difference 0).
- Each final block output matched the legacy CUDA baseline exactly (max absolute difference 0; bit-identical), and all three resident outputs were bit-identical to each other.

### Two-block device chain, three resident executions

- Resident tensor payload: **857,909,248 bytes** across 23 weights, including the embedding table.
- GGML CUDA graph buffer: **888,418,048 bytes**, of which 30,508,800 bytes are non-weight graph/work tensors.
- Initialization: 26 H2D calls totaling **857,911,948 bytes**: 857,909,248 weight bytes plus the same 2,700 bytes of controls.
- Each execution: **zero H2D calls/bytes**; one 512,000-byte final output readback. As above, the first execution additionally read back selected embedding rows for the control.
- Block 0's activations feed block 1 entirely inside the CUDA graph; no intermediate activation D2H/H2D staging occurs.
- Each final block-1 output matched the legacy CUDA two-block baseline exactly (max absolute difference 0; bit-identical); all three resident outputs were bit-identical.

For both runs, audit snapshots show zero H2D traffic between run boundaries and no resident device allocation changes during execution. The sole GGML backend buffer was freed at teardown; the residency store reported zero entries and bytes. After process exit, `nvidia-smi` reported device 0 at 12 MiB used / 12,027 MiB free, its observed idle state. The in-process free-memory query remained about 27 MiB below its pre-run value until CUDA context teardown; this is recorded rather than treated as a leak or hidden.

## Verification

- Built `vbuf_qwen3_block_qualification` against the pinned GGML CUDA build.
- Passed `vbuf_residency_contract`, `vbuf_shared_residency_concurrency_contract`, and the new `vbuf_device_residency_contract` (3/3).
- `git diff --check` passed.
- Raw execution logs and CUDA API/window audit files are retained outside the repository under:
  `~/.cache/vbuf-agent-qualification/qwen3-cuda-q4k/cuda-transfer-audit/resident-one/` and `resident-two/`.
  Legacy control captures are under `cuda-transfer-audit/legacy-actual/`.

## Qualification boundary and remaining blockers

This establishes a useful **one- and two-block residency foundation**, not GPU depth qualification or a production runtime. The result covers one 25-token prefill input and a diagnostic graph. In this probe each semantic tensor entry retains the same monolithic GGML graph-buffer owner; the store's per-key byte accounting tracks resident weight payload, while the larger backend-buffer/workspace size is reported separately. This does not establish independently allocatable or independently evictable production weights. GPU-resident persistent KV, incremental replay, model-wide residency planning, memory-pressure/eviction during inference, deeper graphs, logits, and generation remain untested. A 40-layer model cannot be assumed to fit based on these bounded runs.

The cross-backend CPU/CUDA numerical divergence previously measured for Qwen3 Q4_K/Q6_K remains classified as backend activation-quantization behavior. Exact comparison here is against the same legacy CUDA path, not an external llama.cpp `1e-5` parity pass. The external numerical gate remains **FAIL**; production Qwen3 remains disabled. The earlier Qwen CUDA residency audit and activation-quantization analysis remain applicable evidence.
