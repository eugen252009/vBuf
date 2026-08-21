# Step 31Q Host TensorWave Runtime Qualification

Status: **HOST_QUALIFIED; RISC-V PHYSICAL QUALIFICATION NOT_QUALIFIED**.

This report records the fast x86_64 qualification requested while the Orange Pi
SSH target was unavailable. It must not be read as riscv64 evidence.

## Target And Workflow

- Branch: `vbuf-ml`
- Starting head: `54ca9b1`
- Host architecture: `x86_64`
- Host compiler: GNU 14.2.0
- GGML source: pinned commit `2d191b5dee1a591c41ee8a653ce42bfcd9c8716d`
- Runtime source: local Rust workspace libraries
- Payload source: HTTP range server at `http://127.0.0.1:18124`
- Android/ADB: not used
- Orange Pi/RISC-V: not reached; `ssh pi` resolved to `192.168.188.42` but
  port 22 refused connections

The host build was configured with:

```text
VBUF_GGML_SOURCE_DIR=/tmp/opencode/vbuf-step31l-native-build/_deps/ggml_source-src
VBUF_ML_LIBRARY=rust/target/debug/libvbuf_ml.so
VBUF_RUNTIME_LIBRARY=rust/target/debug/libvbuf_runtime.so
VBUF_BUILD_PROBES=ON
VBUF_BUILD_MOE_GROUPING_MICROQUALIFICATION=ON
VBUF_BUILD_REAL_GATE2B=ON
```

## Host Results

### Contract And Neutrality

- Native CTest: `23/23 PASS`.
- Rust workspace: `PASS`.
- Portable graph neutrality: `FORBIDDEN_LEAKAGE_COUNT=0`.
- Indexed lowering contract: `PASS`.
- TensorWave dependency, readiness, lease, and residency contract: `PASS`.
- No TensorRef, persistent format, or residency policy change was made.

### Real Router Through Portable TensorWave/GGML

Executable: `vbuf_gate2b_real_deepseek`.

- Semantic bootstrap discovery: `PASS`.
- Real quantized router payload: `PASS`.
- HTTP range source: `PASS`.
- TensorBinding payload crossed the FFI: `YES`.
- Lease retained during compute: `YES`.
- RMSNorm parity: exact, max absolute error `0`.
- Router logits parity: max absolute error `2.26498e-6`.
- TopK index parity: `PASS`.
- TopK value parity: exact, max error `0`.
- Selected IDs: `37,21,31,54,6,33`.

### Real Selected Expert Through Current TensorWave Runtime

Executable: `vbuf_moe_tensor_wave_poc10`, selected layer 1 expert 0.

- Real gate bank: IQ2_XXS, rank-2 selected view, `743424` bytes.
- Real up bank: IQ2_XXS, rank-2 selected view, `743424` bytes.
- Real down bank: IQ4_NL, rank-2 selected view, `1622016` bytes.
- Unselected expert requests: `0`.
- Current TensorWave graph: `PASS`.
- Cold external HTTP materialization parity: exact.
- Warm residency parity: exact.
- Source trace: `10` events.
- Cold source reads: `18`.
- Residency after teardown: `0` resources.
- Reference parity: `PASS`.

### Bounded Full Token Path

Executable: `vbuf_autoregressive_poc22`, one token, two-block full-stack path.

- External semantic bootstrap: used.
- External HTTP payload source: used.
- Full token path attempted: `YES`.
- Full token path completed: `YES`, one generated token.
- Runtime/reference next-token ID: `47572 / 47572`.
- Runtime/reference logits parity: exact.
- Router selection parity: `PASS`.
- Generated token feedback: `PASS`.
- State alias violations: `0`.
- Materializer identity collisions: `0`.
- Source bytes: `27610112`.
- Source reads: `23`.
- Residency hits/misses/evictions: `238 / 69 / 0`.
- Peak resident bytes: `199529120`.
- Peak active bytes: `12607488`.
- Resources after teardown: `0`.

### Indexed Expert Backend Seam

Executable: `vbuf_moe_grouping_microqualification`, two threads, one iteration.

- Real IQ2_XXS gate/up payload: `PASS`.
- Real IQ4_NL down payload: `PASS`.
- `ggml_mul_mat_id` available on host: `YES`.
- Grouped-vs-rank-2 gate/up parity: exact.
- Grouped-vs-rank-2 down parity: exact.
- Gate/up timing: `0.601 ms` grouped versus `0.602 ms` rank-2.
- Down timing: `0.513 ms` grouped versus `0.546 ms` rank-2.
- Weight repack bytes: `0`.

This qualifies raw host GGML indexed execution over real quantized payloads. It
does not claim that the current generic TensorWave operation enum supports
`MUL_MAT_ID`; the current production runtime remains rank-2 canonical execution
with indexed lowering/provenance available as a backend seam.

## Model And Working Set

- Model: DeepSeek-V2-Lite IQ2_XXS.
- Full payload size: `5639819878` bytes.
- Semantic bootstrap: `3206431` bytes.
- Full payload present on the host for regression fixtures, but the qualified
  runtime path consumed payload through the HTTP range source.
- The one-token full-stack path peaked at `199529120` resident bytes, about
  `3.54%` of the full payload size.
- This demonstrates that persistent model size and resident working set remain
  separate on the host path.

## RISC-V Boundary

The requested physical target was not qualified. `ssh pi` resolved to
`192.168.188.42`, which responded to ping but refused TCP port 22. The earlier
`ssh alpine` alias was an unrelated ARMv7 Allwinner host and was not used for
this qualification. No RISC-V compiler, deployment, or physical runtime
evidence is claimed here.

## Evidence Classification

| Evidence | Classification |
|---|---|
| Host CTest and Rust | HOST_QUALIFIED |
| Real router and selected expert runtime | HOST_QUALIFIED |
| Real raw indexed GGML path | PHYSICALLY_MEASURED_X86_64 |
| Full external one-token path | PHYSICALLY_MEASURED_X86_64 |
| Orange Pi RV2 TensorWave runtime | NOT_QUALIFIED |
| Direct LAN RISC-V source traffic | NOT_QUALIFIED |
| Android build regression | NOT_RUN |

No model artifacts, payload caches, generated binaries, or host build outputs
are repository artifacts from this qualification.
