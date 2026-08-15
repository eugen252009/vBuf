# vBuf-ML tensor representations

## Step 17 qualification result

The first real compatibility corpus and pinned external reference are now
available:

- `F32` → generic `CanonicalPrimitive` (profile ID `0`);
- `BF16` → exact opaque-byte profile contract (profile ID `1`);
- `GGML Q8_0` → exact opaque-byte profile contract (profile ID `2`);
- `GGML Q4_0` → exact opaque-byte profile contract (profile ID `3`);
- `GGML Q2_K` → exact opaque-byte profile contract (profile ID `4`);
- `GGML IQ1_S` → exact opaque-byte profile contract (profile ID `5`).
- `GGML Q4_K` → exact opaque-byte profile contract (profile ID `6`);
- `GGML IQ4_NL` → exact opaque-byte profile contract (profile ID `7`);
- `GGML IQ4_XS` → exact opaque-byte profile contract (profile ID `8`).
- `GGML Q3_K` → exact opaque-byte profile contract (profile ID `9`);
- `GGML IQ2_XXS` → exact opaque-byte profile contract (profile ID `10`);
- `GGML IQ2_XS` → exact opaque-byte profile contract (profile ID `11`);
- `GGML IQ2_S` → exact opaque-byte profile contract (profile ID `12`).

`Q4_K_S` and `Q4_K_M` are quantization/model aliases used by llama.cpp
packaging. Their tensor payloads are still the per-tensor `GGML_TYPE_Q4_K`
encoding and therefore map to profile ID `6`.
The same rule applies to `Q2_K_S/M` and `Q3_K_S/M`, which map to their
canonical per-tensor `Q2_K` and `Q3_K` encodings.

These IDs are profile-local. They are not copies of external GGML enum values.
The external mapping is explicit:

| vBuf-ML ID | Representation | GGML ID | Storage |
|---:|---|---:|---|
| 0 | `CanonicalPrimitive` | F32 = 0 when used for F32 | canonical primitive descriptor |
| 1 | `BF16` | 30 | opaque canonical bytes |
| 2 | `GGML_Q8_0` | 8 | opaque canonical bytes |
| 3 | `GGML_Q4_0` | 2 | opaque canonical bytes |
| 4 | `GGML_Q2_K` | 10 | opaque canonical bytes |
| 5 | `GGML_IQ1_S` | 19 | opaque canonical bytes |
| 6 | `GGML_Q4_K` | 12 | opaque canonical bytes |
| 7 | `GGML_IQ4_NL` | 20 | opaque canonical bytes |
| 8 | `GGML_IQ4_XS` | 23 | opaque canonical bytes |
| 9 | `GGML_Q3_K` | 11 | opaque canonical bytes |
| 10 | `GGML_IQ2_XXS` | 16 | opaque canonical bytes |
| 11 | `GGML_IQ2_XS` | 17 | opaque canonical bytes |
| 12 | `GGML_IQ2_S` | 22 | opaque canonical bytes |

The contracts were derived from llama.cpp commit
`4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`.

## F32

F32 remains `CanonicalPrimitive`. Its descriptor provides float semantics,
32-bit width, logical count, little-endian primitive bytes, and payload length.
No duplicate ML F32 type is needed.

The consumer mapping is:

```text
CanonicalPrimitive(float, 32 bits) → GGML_TYPE_F32
```

## BF16

BF16 is not treated as ordinary unsigned 16-bit data.

The contract is:

```text
logical elements: N
payload bytes: 2 × N
storage: opaque canonical bytes
stored scalar: IEEE binary32 high 16 bits
byte order: little-endian uint16 representation
correctness alignment: 1; canonical BaseStep/alignment remains authoritative
```

Pinned source defines `ggml_bf16_t` as a struct containing `uint16_t bits` and
converts to binary32 by shifting `bits << 16`. Conversion helpers use explicit
bit operations; vBuf-ML does not rely on native BF16 ABI serialization.

The consumer mapping is:

```text
BF16 → GGML_TYPE_BF16
```

## Q8_0

Q8_0 is an opaque payload contract. The canonical vBuf layer does not interpret
its scale or quantized values.

Pinned source establishes:

```text
GGML_TYPE_Q8_0 = 8
QK8_0 = 32 logical elements per block
block_q8_0 = uint16 scale + 32 int8 q values
physical block size = 34 bytes
```

The quantizer and dequantizer operate on complete 32-element blocks. For a
shape whose innermost dimension is `row_width`:

```text
row_width % 32 == 0
rows = product(shape[1..])
blocks_per_row = row_width / 32
payload_bytes = rows × blocks_per_row × 34
```

The consumer mapping is:

```text
GGML_Q8_0 → GGML_TYPE_Q8_0
```

No dequantization, requantization, repacking, or byte reordering is part of the
vBuf-ML contract.

## Canonical storage and validation

Packed representations require a canonical opaque byte-array descriptor whose
byte count equals the exact contract payload size. Shape, row divisibility,
block count, and checked arithmetic are validated without scanning payload
contents.

This preserves lazy loading:

```text
TensorDirectory
→ representation and range validation
→ ready
```

Q8_0, Q4_0, Q2_K, and IQ1_S values and scales are not decoded at model-open
time.

## Alignment and endianness

The pinned source supplies type/block geometry, not a new vBuf-ML correctness
alignment. The selected contracts therefore require no alignment beyond the
existing canonical arrangement. Consumer-specific SIMD alignment is separate
and non-normative.

F32 scalar bytes, BF16 `bits`, and quantized scale fields are represented as
little-endian canonical bytes. Quantized integer payload bytes are byte-order
independent where the pinned GGML layout defines them as byte arrays.

## Placement separation

Step 16's `LAYER_MAJOR_ROLE_ORDER` changes tensor-to-tensor file order only.
It never changes bytes inside an F32, BF16, Q8_0, Q4_0, Q2_K, or IQ1_S tensor payload.

## Unsupported representations

Q4_0, Q2_K, and IQ1_S are supported by the current profile. Other Q4/Q2/IQ
variants, Q5, and unqualified GGML types remain unsupported. Unknown profile IDs
fail closed.
