# Step 32F: Progressive Real Multi-Layer Execution

Date: 2026-08-25

## Result

`PASS_8_LAYERS`

The generic portable Rust runner executed eight consecutive real GLM
transformer blocks, layers `23..30`, using the actual output of each layer as
the next layer input. Every layer independently resolved its semantic catalog,
recomputed routing, acquired only selected experts, and released its F32
weight cache before advancing.

The run stopped at layer 30. It did not preload or dequantize the full model,
compute logits, generate a token, execute decode, or use a device backend.

## Qualified Artifacts

- Branch: `vbuf-ml`
- Starting commit: `5f31d6d`
- Model repository: `zai-org/GLM-4.5-Air-FP8`
- Model revision: `f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc`
- Model artifact: `.step32c/glm-4.5-air-fp8.vbuf`
- Model bytes: `112563538898`
- Model SHA-256: `15b4f0d3b7d72754c0e233bb787e72b1d6cdab655aec9ecab36dac8b654c0f14`
- Semantic sidecar: `.step32c/glm-4.5-air-fp8.semantic.vbuf`
- Sidecar bytes: `14306259`
- Sidecar SHA-256: `817630bd5579ab603c870e6041c75267d89c90e87b45b0f5e5b844d88840bbea`
- Real model redownloaded: `NO`
- Real model payload rewritten: `NO`
- Second model-sized copy created: `NO`
- Fresh sidecar reopen: `PASS`
- Payload size match: `PASS`

The payload was mapped read-only and the sidecar source profile resolved the
materialized ranges to source ID `1`. No Hugging Face, Safetensors, or config
file was accessed during execution.

## Layer Window

- Start layer: `23`
- Two-layer window: `23..24`
- Four-layer window: `23..26`
- Eight-layer window: `23..30`
- Layer count in the persisted architecture: `46` metadata layers plus the
  preserved physical predictor-layer index space.
- Layer types in the selected window: all normal MoE transformer blocks.
- MoE layers in the eight-layer window: `8`.
- Dense layers in the eight-layer window: `0`.
- Expert count: `128`.
- Top-K: `8`.
- Input batch: `1`.
- Input sequence length: `4`.
- Input hidden size: `4096`.
- Initial input hash:
  `5eb4f1bfdd34f9a76ea0189484d387cf1a047c27d2c49d3f2783211bb1ec604b`.

The layer catalog is generic and derives TensorId bindings from persisted MoE
roles and validated ordinal relationships. It does not perform source-name
runtime lookups. The only runner is `run_progressive`; no
`execute_glm_layer_23`-style layer-specific runner exists.

## Handoff And State

- Generic layer runner implemented: `YES`.
- Model-specific layer runner count: `0`.
- Activation handoff: `PASS`.
- Synthetic activation reinjection after the initial input: `0`.
- Activation strategy: layer-scoped value arena; the current output is handed
  directly as the next loop input and all previous layer values are released
  at the layer boundary.
- Activation buffer reuse/bounded release: `PASS`.
- Layer state addressing: opaque `StateId(1000 + layer_id)`.
- Layer 23 state: `StateId(1023)`.
- Layer 24 state: `StateId(1024)`.
- Layer 25 state: `StateId(1025)`.
- Layer 26 state: `StateId(1026)`.
- Layer 27 state: `StateId(1027)`.
- Layer 28 state: `StateId(1028)`.
- Layer 29 state: `StateId(1029)`.
- Layer 30 state: `StateId(1030)`.
- Cross-layer KV read count: `0`.
- Cross-layer KV isolation: `PASS`.
- Active execution states after run: `0`.
- Active layer states after run: `0`.
- Active transient leases after each layer: `0`.
- Active transient leases after run: `0`.

Each layer started prefill with empty state and sequence length four. Its state
was distinct from every other layer and reset before request teardown.

## Numerical Qualification

An independent bounded NumPy reference decoded the same persisted FP8/BF16
ranges and independently computed each layer and router decision. It did not
consume production intermediate values.

Multi-layer comparison policy:

- Maximum absolute error allowed: `2.5e-04`.
- Maximum relative error allowed: `3.0e-05`.
- Router IDs must match exactly at every layer.

Results:

| Gate | Execution | Reference | Max absolute | Max relative | Router IDs |
|---|---|---|---:|---:|---|
| 2 layers, 23..24 | PASS | PASS | `9.91821289e-05` | `1.65508209e-05` | exact |
| 4 layers, 23..26 | PASS | PASS | `1.20162964e-04` | `2.13032981e-05` | exact |
| 8 layers, 23..30 | PASS | PASS | `2.02178955e-04` | `2.13032981e-05` | exact |

The first layer retained the Step 32E-B behavior. Every later layer's input
checkpoint matched the preceding production output checkpoint within the
same reference comparison policy.

Activation output hashes:

```text
layer23 8410f5784b4d249a5698a484778eabe79370000cde8810391b5fb2005ae9ef56
layer24 727596f326fde68695dca2c834d4367c036031ed64f45d829bb3a592a614e29f
layer25 46c2abd35ca0c5fffdb29a7bab616746d12b28fda868741b7cdb295e36746077
layer26 d2c159e4fe85a5fe5d87b001c652c69c70fa9559ae72bd208fdadb6c7c774a10
layer27 308c2fe356669eb363dada5135e4f0e797a85a96433d3752548063f30fb0bb6b
layer28 384016bb8ecd2ba301caab8ffd17dd9ecdc392a70de864744cbc50161606d47a
layer29 414a5a335e8cc6766c21975b8a8f6db584387bb5c3a2ef2fa75440312ca5c2da
layer30 5196ba3e626c34b56af2d8949f9186a2a5884cac91c8fbdf65833af6fb7b557b
```

## Routing And Acquisition

Unique selected routed expert identities per layer were:

```text
layer23=9, layer24=10, layer25=12, layer26=11,
layer27=12, layer28=9, layer29=9, layer30=8
```

- Total unique selected expert identities, 2 layers: `19`.
- Total unique selected expert identities, 4 layers: `42`.
- Total unique selected expert identities, 8 layers: `80`.
- Total token-level expert choices, 2 layers: `64`.
- Total token-level expert choices, 4 layers: `128`.
- Total token-level expert choices, 8 layers: `256`.
- Router decisions, 2 layers: `8`.
- Router decisions, 4 layers: `16`.
- Router decisions, 8 layers: `32`.
- Unselected expert count touched at every gate: `0`.
- Unselected expert bytes touched at every gate: `0`.
- Expert overfetch bytes at every gate: `0`.

Per-layer required and resolved TensorId counts were identical:

```text
layer23 75/75, layer24 81/81, layer25 93/93, layer26 87/87,
layer27 93/93, layer28 75/75, layer29 75/75, layer30 69/69
```

Missing tensor count was `0` for every layer. The physical acquisition trace
therefore follows router selection plus required norms, attention, router, and
shared-expert tensors without touching unselected routed experts.

## Residency And Scaling

The runner evicts the layer's converted F32 tensor cache after producing the
layer output. The persistent FP8 payload remains a read-only mmap; no copied
FP8 resident tensor cache is created. File-backed mmap/page-cache RSS is
reported separately from converted F32 residency.

| Gate | Cumulative source bytes | Unique model bytes | Model fraction | Peak F32 bytes | Peak activation bytes | Peak KV bytes | Peak internal working set |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 layers | `584351744` | `584351744` | `0.5191%` | `1199661568` | `3489792` | `65536` | `1203184128` |
| 4 layers | `1238020096` | `1238020096` | `1.0998%` | `1338073600` | `3751936` | `131072` | `1341858304` |
| 8 layers | `2406723584` | `2406723584` | `2.1381%` | `1338073600` | `3751936` | `262144` | `1341858304` |

Peak source range requested by one layer was `335498752` bytes. Copied FP8
resident bytes were `0`; the full payload's virtual mapping is not counted as
materialized model residency. Converted F32 bytes after each layer and after
the full run were `0`.

- Working-set factor, 4 vs 2: `1.1153x`.
- Working-set factor, 8 vs 4: `1.0000x`.
- Working-set bounded: `YES`.
- Converted F32 memory grows linearly with depth: `NO`.
- Activation memory grows linearly with depth: `NO`.
- Model-weight residency grows linearly with depth: `NO`.

RSS, including file-backed mmap/page-cache effects, was:

| Gate | Before KiB | Peak KiB | After release KiB |
|---|---:|---:|---:|
| 2 layers | `43780` | `1827792` | `623688` |
| 4 layers | `43864` | `2536036` | `1260316` |
| 8 layers | `44068` | `3477052` | `2404084` |

RSS peak growth was `725241856` bytes from 2 to 4 layers and `963600384`
bytes from 4 to 8 layers. This is observer-distorted by file-backed payload
pages and is not treated as converted model memory; internal cache and
activation counters are the boundedness authority for this qualification.

## Repeatability And Isolation

- 2-layer repeat count: `3`.
- Repeated 2-layer routing parity: exact.
- Repeated 2-layer output parity: exact output hashes.
- Repeated 2-layer internal memory growth: `0` bytes.
- Cross-run isolation: `PASS`.
- Final 8-layer recheck wall-clock total: `251536.338 ms`.

The default input produced final hash
`727596f326fde68695dca2c834d4367c036031ed64f45d829bb3a592a614e29f` before
and after a sequential run with a different deterministic input variant. The
different variant produced a distinct final hash
`22bee4a24b3211aea0655504effc754fca031619be8d7721a90872032ffac436` and did
not alter the default run's routing or output.

## Verification

- `cargo test --manifest-path rust/Cargo.toml --workspace`: PASS.
- `cargo test --manifest-path rust/Cargo.toml -p vbuf-ml`: PASS.
- `cargo fmt --manifest-path rust/Cargo.toml --all -- --check`: PASS.
- Native portable graph adapter contract: PASS.
- Step 32E-B layer-23 regression: included in every 2/4/8 run and reference
  comparison; PASS.
- 2-layer progressive test: PASS.
- 4-layer progressive test: PASS.
- 8-layer progressive test: PASS.
- Activation handoff: PASS.
- Cross-layer state isolation: PASS, zero prior-state reads.
- Routing parity: PASS at all 32 layer/token decisions.
- Selected-expert acquisition: PASS, zero unselected expert touches.
- Converted-weight boundedness: PASS.
- Lease accumulation: PASS, zero active transient leases.
- State cleanup: PASS.
- Repeatability: PASS, three 2-layer runs.
- Neutrality scan: PASS, `FORBIDDEN_LEAKAGE_COUNT=0`.
- `git diff --check`: PASS.
- ccc index: refreshed after the Step 32F implementation.

The production command was:

```text
cargo run --manifest-path rust/Cargo.toml -q -p vbuf-runtime --bin vbuf-runtime-step32f-progressive -- \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.semantic.vbuf \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32f-8-final.checkpoints \
  /tmp/opencode/step32f-8-final.manifest \
  23 8
```

The independent reference control was:

```text
python /tmp/opencode/step32f_reference.py \
  /tmp/opencode/step32f-8-final.manifest \
  /home/eugen/projekte/vBuf/.step32c/glm-4.5-air-fp8.vbuf \
  /tmp/opencode/step32f-8-final.checkpoints
```

## Boundary And Next Gap

The qualified claim is limited to eight consecutive real transformer layers
with real activation handoff, exact independent routing parity, selected-only
expert acquisition, bounded converted-weight residency, isolated layer-local
prefill state, and clean teardown.

The single smallest remaining gap toward final logits is the generic
full-stack layer catalog and output-head contract: extending the same proven
layer loop beyond the bounded 23..30 window while adding the remaining physical
layer classes and final normalization/output projection semantics. No such
extension is implemented in this step.
