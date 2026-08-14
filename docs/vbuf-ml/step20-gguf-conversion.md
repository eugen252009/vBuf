# Step 20: deterministic manifest-driven GGUF conversion

Status: **converted and independently validated** for Q8_0 and BF16.

## Architecture

```text
GGUF source
  → verified read-only Python inspector
  → validated Step-18/19B ConversionManifest
  → compact execution plan
  → Rust vBuf-ML writer
  → existing LayoutPlan
  → canonical v0.6 writer
  → reopen through production readers
```

The Python entry point is:

```text
scripts/convert_gguf_to_vbuf_ml.py
```

The canonical writer is the Rust binary:

```text
rust/vbuf-ml/src/bin/vbuf-ml-convert.rs
```

The execution plan contains only already-qualified manifest decisions plus
source checked ranges. It is not a second semantic policy engine.

## Source verification

Before writing, the converter verifies:

- manifest source SHA-256 and size;
- profile `vbuf-ml-0.1`;
- pinned consumer revision;
- `READY_FOR_CONVERSION` status;
- `LAYER_MAJOR_ROLE_ORDER` policy;
- every tensor name, source range, representation, and target occurrence;
- qualified GPT2-BPE/Qwen2 tokenizer plan.

A mismatch fails before final-target promotion.

## Target layout

Control blocks are emitted first, followed by tokenizer payloads and tensors.
Tensor payloads use the target order already present in the manifest:

```text
GlobalPre
Layer 0 … Layer 27
GlobalPost
```

TensorDirectory remains name-sorted and references tensor Key-ID `0x0200` with
manifest target occurrences. Physical order and semantic directory order remain
separate.

The converter uses `LayoutPlan` and `vbuf-core::writer::VBufV06Writer`; it does
not hand-build v0.6 bytes.

## Byte preservation

Every tensor is copied directly from its checked GGUF source range. F32 uses the
canonical Float32 block descriptor; BF16 and Q8_0 remain opaque bytes. No
numeric decoding, byte swapping, dequantization, requantization, or repacking
occurs.

All 621 tensors match source and target SHA-256 payload digests:

```text
Q8_0:  310 / 310
BF16:  311 / 311
```

Evidence is in `tensor-payload-parity.csv` under the artifact-specific Step-20
evidence directories.

## Model metadata

The writer encodes the manifest's portable Step-19A values, including KV-head
count and independent key/value head dimensions. It does not derive or invent
metadata during conversion.

## Tokenizer

The writer emits `Gpt2BpeQwen2`:

```text
TokenTextBytes / TokenOffsets
TokenTypes
MergeLeftIds / MergeRightIds
GPT2_BPE identity
QWEN2 identity
add_bos
BOS/EOS/PAD IDs
exact ChatTemplate bytes
```

Merge arrays are taken from the manifest's resolved ordinal pairs. The writer
does not reinterpret merge strings.

## Tied output behavior

Q8_0 has no source `output.weight`; no target duplicate is created.

BF16 has explicit `token_embd.weight` and `output.weight`; both are emitted as
distinct target tensors and both payloads independently match their sources.

## Streaming and memory

Tensor payloads are mmap-backed source slices passed to the canonical writer.
The converter does not create a whole-model payload buffer or tensor-sized copy.
The compact execution plan contains control data and manifest ranges; target
writing proceeds in bounded semantic blocks.

The current canonical writer API accepts borrowed payload slices. This is
sufficient for the qualified artifacts without introducing a new streaming wire
writer.

## Target results

| Artifact | Source size | Target size | Tensor payload bytes | Result |
|---|---:|---:|---:|---|
| Q8_0 | 639,446,688 | 637,925,504 | 633,495,552 | `CONVERTED_AND_VALIDATED` |
| BF16 | 1,509,347,552 | 1,507,825,912 | 1,503,395,840 | `CONVERTED_AND_VALIDATED` |

The target-size difference reflects GGUF metadata/layout versus vBuf-ML control,
canonical headers, padding, and the selected target structure. No inference
performance claim is made.

## Determinism

Repeated Q8_0 conversion produced identical SHA-256 hashes:

```text
2982cedd0ddc12d762d12ff3426bcc105b2cee1c675ca13bbb5ca4c3cce9a998
```

No timestamps, random identifiers, or absolute paths enter target files.

## Validation

After writing, the target is closed and reopened through:

```text
canonical v0.6 parser
Bootstrap discovery
ModelMetadata parser
TensorDirectory parser
TokenizerMetadata parser
```

Independent evidence is under:

```text
benchmark-results/vbuf-ml-step20/
```

Integrity is currently disabled (`none`), matching the manifest configuration.

## Limitations

This step proves storage and semantic parity only. It does not prove llama.cpp
consumer loading, tokenizer output parity, or inference compatibility.
