# Step 17: upstream representation qualification

Status: **qualified; converter still deferred**.

## Pinned external reference

```text
repository: https://github.com/ggml-org/llama.cpp.git
commit: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
commit date: 2026-08-14T15:14:19+03:00
```

The research checkout is external to the repository and is not a production
dependency. The qualification script verifies the exact commit and required
source symbols before running.

## Source provenance

Recorded in `benchmark-results/vbuf-ml-step17/upstream-provenance.json`:

| Fact | Pinned location |
|---|---|
| F32/BF16/Q8_0 numeric IDs | `ggml/include/ggml.h`, `enum ggml_type` |
| BF16 storage type | `ggml/include/ggml.h`, `ggml_bf16_t` |
| BF16 conversion | `ggml/src/ggml-impl.h`, `ggml_compute_bf16_to_fp32`, `ggml_compute_fp32_to_bf16` |
| Q8_0 block | `ggml/src/ggml-common.h`, `QK8_0`, `block_q8_0` |
| Q8_0 reference quantizer | `ggml/src/ggml-quants.c`, `quantize_row_q8_0_ref` |
| Q8_0 dequantizer | `ggml/src/ggml-quants.c`, `dequantize_row_q8_0` |
| type traits | `ggml/src/ggml.c`, `type_traits[GGML_TYPE_Q8_0/BF16]` |
| row sizing | `ggml/src/ggml.c`, `ggml_blck_size`, `ggml_type_size`, `ggml_row_size` |
| Qwen3 output fallback | `src/models/qwen3.cpp`, `load_arch_tensors` |
| duplicated embedding behavior | `src/llama-model-loader.cpp`, `create_tensor` |

## Numeric identity

The pinned revision confirms:

```text
GGML_TYPE_F32  = 0
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30
```

These agree with both local GGUF artifacts.

## BF16 semantics

The pinned source defines:

```c
struct { uint16_t bits; } ggml_bf16_t;
```

Decode is equivalent to:

```text
float_bits = uint32(bits) << 16
```

The contract uses explicit little-endian bytes and does not serialize a native
C/C++ struct or Rust ABI representation.

Known values validated by tests:

```text
0x0000 → 0.0
0x3f80 → 1.0
0xbf80 → -1.0
```

## Q8_0 semantics

The pinned block is:

```text
uint16_t d
int8_t qs[32]
```

with a static asserted size of 34 bytes. The reference quantizer requires
`k % 32 == 0`; the dequantizer uses the same block size.

The vBuf-ML contract validates row width and exact payload size but does not
inspect or decode each block at model-open time.

## Real-model parity

The qualification scanned all 621 tensor descriptors:

```text
Q8_0 artifact: 310 tensors
BF16 artifact: 311 tensors
```

Every F32, BF16, and Q8_0 tensor matched the checked contract formula. The
machine-readable result is:

```text
benchmark-results/vbuf-ml-step17/representation-parity.csv
benchmark-results/vbuf-ml-step17/q8_0-shape-parity.csv
benchmark-results/vbuf-ml-step17/bf16-parity.csv
benchmark-results/vbuf-ml-step17/f32-parity.csv
```

No payload bytes were copied or transformed.

## Output/tied-weight behavior

Pinned `src/models/qwen3.cpp` creates `output.weight` as optional. If absent,
it requests `token_embd.weight` with `TENSOR_DUPLICATED`. The loader explicitly
maps the duplicated embedding request to the output tensor role and reuses the
existing tensor allocation when possible.

Therefore the Q8_0 artifact's missing `output.weight` is an intentional
consumer-supported tied-output form, not evidence that the output projection
should be silently dropped. The future converter must preserve this semantic
relationship without duplicating the 165 MiB Q8_0 embedding payload.

The current TensorDirectory has no alias/reference relation. Step 17 therefore
qualifies the behavior but does not add an alias wire field.

## Model metadata gaps

Already represented or derivable:

- architecture;
- context length;
- embedding length;
- layer count;
- attention head count;
- vocabulary size through tokenizer metadata.

Present in the GGUF and relevant to a future consumer but not currently required
by ModelMetadata:

- KV head count;
- key/value head dimensions;
- feed-forward length;
- RMS normalization epsilon;
- RoPE base.

Feed-forward length, normalization epsilon, and RoPE base already have optional
profile keys. KV head count and head dimensions remain architecture-specific
metadata gaps to resolve before conversion if the selected consumer requires
them explicitly.

Tokenizer algorithm semantics remain outside this step.

## Byte preservation and placement

The future conversion invariant is:

```text
GGUF payload bytes
→ unchanged opaque canonical payload
→ vBuf-ML representation contract
→ pinned GGML consumer
```

Step 16 placement changes only tensor order. It must not reorder bytes inside a
payload.

## Alignment and laziness

No additional correctness alignment is required by the pinned F32, BF16, or
Q8_0 type contracts. Existing canonical alignment remains authoritative.

Validation uses only shape, row rules, block geometry, and payload length. It
does not hash, decode, or scan quantized payload contents.

## Remaining boundary

The representation seam is now sufficient for a byte-preserving converter, but
alias semantics for tied output weights and the final Qwen3 metadata requirement
matrix must be resolved at the converter/consumer boundary.
