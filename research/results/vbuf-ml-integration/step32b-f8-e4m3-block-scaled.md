# Step 32B: Persistent F8_E4M3 Block-Scaled Storage

## Scope

This step adds the vBuf-ML representation and materialization contract for
Safetensors `F8_E4M3` weights. FP8 bytes remain the persistent payload. The
existing F32 execution representation is the portable fallback; no native
FP8 kernel, GGML format change, v0.6 wire change, remote model download, or
production concurrency change is introduced.

## Contract

- `TensorRepresentation::F8_E4M3` is one opaque byte per logical element.
- Safetensors `F8_E4M3` is the only accepted FP8 source dtype in this step.
  `F8_E5M2` and unknown dtypes remain unsupported.
- FP8 weights are 2-D matrices. Their scale tensor is resolved semantically as
  `<weight_name>_scale_inv`; it must be an F32 matrix.
- `weight_block_size` must be declared by `config.json`, either at the
  top-level or under `quantization_config`. No implicit block geometry is
  selected.
- The optional vBuf-ML `QuantizationMetadata` region persists the weight/scale
  association and block geometry. It is not a generic v0.6 region contract.
- The portable materializer decodes E4M3FN bytes and emits F32 values for the
  selected tensor only. It does not create a model-sized conversion cache.

## Decoder

The decoder uses explicit E4M3FN sign, exponent, and mantissa handling. All 256
byte patterns are compared against an independently written reference in the
unit test. Zero, subnormal, signed, maximum finite (`0x7e` = `448`), and NaN
encodings are covered. Block-scaled materialization covers row-local scales
and partial edge geometry.

## Offline Qualification

The synthetic Safetensors fixture contains an `F8_E4M3` `[2, 2]` weight and a
`[2, 1]` F32 scale tensor with configured block size `[1, 2]`. The test proves:

- exact FP8 source bytes are written to the planned final payload range;
- source/scale relationships are persisted and reopened through the canonical
  bootstrap, tensor directory, and quantization metadata readers;
- the F32 materialization result is `[1, 2, -2, 0]`;
- missing/invalid quantization geometry fails closed;
- representation payload length and opaque block semantics are checked.

## Qualification Boundary

This is an offline vBuf-ML storage, provenance, decoder, importer, and F32
materialization qualification. A real public FP8 model, production residency
trace for converted execution buffers, and GGML/backend MatMul integration are
not claimed here. Those require a separately selected model and an explicit
backend adapter path that consumes vBuf-ML materialized F32 tensors.

## Verification Record

```text
cargo check --manifest-path rust/Cargo.toml --workspace: PASS
cargo test --manifest-path rust/Cargo.toml -p vbuf-ml: PASS
F8 E4M3 all-256-pattern reference test: PASS
FP8 Safetensors exact-byte/reopen fixture: PASS
FP8 representation contract tests: PASS
git diff --check: PASS
```
