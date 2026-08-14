# vBuf-ML tensor representations

## Step 17 qualification result

The first real compatibility corpus and pinned external reference are now
available:

- `F32` → generic `CanonicalPrimitive` (profile ID `0`);
- `BF16` → exact opaque-byte profile contract (profile ID `1`);
- `GGML Q8_0` → exact opaque-byte profile contract (profile ID `2`).

These IDs are profile-local. They are not copies of external GGML enum values.
The external mapping is explicit:

| vBuf-ML ID | Representation | GGML ID | Storage |
|---:|---|---:|---|
| 0 | `CanonicalPrimitive` | F32 = 0 when used for F32 | canonical primitive descriptor |
| 1 | `BF16` | 30 | opaque canonical bytes |
| 2 | `GGML_Q8_0` | 8 | opaque canonical bytes |

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

Q8_0 values and scales are not decoded at model-open time.

## Alignment and endianness

The pinned source supplies type/block geometry, not a new vBuf-ML correctness
alignment. The selected contracts therefore require no alignment beyond the
existing canonical arrangement. Consumer-specific SIMD alignment is separate
and non-normative.

F32 scalar bytes, BF16 `bits`, and Q8_0 scale fields are represented as
little-endian canonical bytes. Q8_0 `int8` q-values are byte-order independent.

## Placement separation

Step 16's `LAYER_MAJOR_ROLE_ORDER` changes tensor-to-tensor file order only.
It never changes bytes inside an F32, BF16, or Q8_0 tensor payload.

## Unsupported representations

Q4, Q5, K-quants, IQ types, and other GGML types remain unsupported. Unknown
profile IDs fail closed.
