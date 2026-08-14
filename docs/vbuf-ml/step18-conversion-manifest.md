# Step 18: read-only GGUF inspector and conversion manifest

Status: **manifest-qualified; target writer intentionally not implemented**.

## Boundary

Step 18 stops at:

```text
GGUF source
→ verified descriptors
→ pinned Qwen3/llama.cpp interpretation
→ deterministic ConversionManifest
→ validation
→ stop
```

The inspector never writes a vBuf file, copies payloads into a target, or
materializes tensor bytes. GGUF offsets are retained only as source read
coordinates.

The implementation is `scripts/build_step18_manifest.py`. It reuses the
Step-16 parser and Step-17 representation formulas rather than introducing a
second GGUF interpretation path.

## Source verification

Both known local artifacts are hash-verified before manifest construction. The
manifest records repository-relative source identity, size, GGUF version,
architecture, metadata count, tensor count, and SHA-256.

The pinned consumer is llama.cpp commit:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

## Manifest model

Each manifest contains:

```text
source_identity
consumer_revision
profile_version
control_plan
model_metadata_plan
tokenizer_plan
tensor_directory_plan
tensor_plans
shared_reference_plan
placement_plan
source_read_plan
accounting
validation
```

A tensor plan records:

```text
source name/ordinal/shape/type
source-only offset and payload length
semantic role and layer
target name/representation/storage
Key-ID and occurrence
target order
payload action
```

No target offset or target length is emitted as portable semantic identity.

## Payload actions

The qualified actions are:

- `COPY_BYTES`: the future writer reads the exact source payload range and
  writes the unchanged bytes under the mapped target representation;
- `SHARED_REFERENCE`: used only for the Q8_0 logical output fallback note;
- `DERIVED`: used for control-region plans and manifest metadata;
- `REJECT`: reserved for future unsupported source entries.

All source tensor entries in the current corpus are `COPY_BYTES`.

## Representation mapping

```text
GGML_TYPE_F32  → CanonicalPrimitive
GGML_TYPE_BF16 → BF16
GGML_TYPE_Q8_0 → GGML_Q8_0
```

Step-17 exact geometry is rechecked while creating every tensor plan.

## Target identity and placement

The first conversion-level policy uses:

```text
tensor payload Key-ID = 0x0200
occurrence = deterministic target placement order
```

Control plans reserve:

```text
Bootstrap          = 0x0201
ModelMetadata     = 0x0202
TokenizerMetadata = 0x0203
IntegrityMetadata = 0x0204
```

These are manifest planning conventions, not new generic vBuf semantics.

Target placement is:

```text
LAYER_MAJOR_ROLE_ORDER
```

The TensorDirectory plan is independently name-sorted for its existing wire
rule. Placement occurrence and directory name order are therefore explicit and
deterministic.

## Q8_0 tied output

The Q8_0 artifact has no `output.weight`. The pinned Qwen3 loader makes
`output.weight` optional and requests `token_embd.weight` with
`TENSOR_DUPLICATED` when absent.

The Q8_0 manifest therefore:

```text
copies token_embd.weight once
omits a second output target tensor
records a SHARED_REFERENCE fallback note
```

No portable alias field was invented.

## BF16 output

The BF16 artifact contains an explicit `output.weight`. It receives an
independent `COPY_BYTES` plan and target entry. It is not collapsed into the
token embedding merely because the Q8_0 artifact uses fallback behavior.

The manifest records hashes for the special embedding/output payloads as
read-only evidence. In this artifact the two hashes are equal, but the source
contains distinct tensor descriptors and the pinned consumer explicitly loads
`output.weight`; the manifest therefore retains two independent `COPY_BYTES`
plans rather than inferring a portable alias.

## Tensor accounting

| Artifact | Source tensors | COPY_BYTES | SHARED_REFERENCE note | DERIVED | Unaccounted |
|---|---:|---:|---:|---:|---:|
| Q8_0 | 310 | 310 | 1 | 3 | 0 |
| BF16 | 311 | 311 | 0 | 3 | 0 |

Every source tensor is accounted for exactly once.

## Model metadata matrix

Directly represented or derivable:

- architecture;
- context length;
- embedding length;
- layer count;
- attention head count;
- feed-forward length;
- normalization epsilon;
- RoPE base;
- vocabulary size.

Step 19A resolves the previous target gaps by adding optional profile-local
ModelMetadata keys for:

- KV head count;
- key head dimension;
- value head dimension.

The manifests now mark all model metadata requirements as directly represented.
The separate Step 19A evidence is in `benchmark-results/vbuf-ml-step19a/`.

## Tokenizer matrix

Both artifacts contain:

```text
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre = qwen2
tokenizer.ggml.tokens
tokenizer.ggml.token_type
tokenizer.ggml.merges
tokenizer.ggml.bos_token_id
tokenizer.ggml.eos_token_id
tokenizer.ggml.padding_token_id
tokenizer.ggml.add_bos_token
tokenizer.chat_template
```

The current VocabularyOnly profile can represent vocabulary, token types,
scores where present, and special IDs. It cannot represent the GPT-2/BPE model,
pre-tokenizer, merges, or chat template semantics required for a
consumer-complete llama.cpp path.

Both manifests are consequently blocked only by the tokenizer gap.

## Read behavior and streaming readiness

The future writer can stream source payloads in target order using positioned
reads:

```text
manifest tensor plan
→ source GGUF offset/length
→ optional payload hash
→ target writer
```

No full model payload buffer is required. The current source descriptor union is
one contiguous physical extent per artifact, but target-order traversal remains
separate from source physical order.

## Readiness classification

Both artifacts are:

```text
BLOCKED_BY_TOKENIZER_GAP
```

They are not called conversion-ready merely because representation geometry is
complete.

## Evidence

```text
benchmark-results/vbuf-ml-step18/
```

contains deterministic JSON manifests, metadata/tokenizer matrices, accounting
summaries, and qualification configuration. Running the manifest builder twice
produces identical serialized JSON hashes.

No model payloads are emitted.
