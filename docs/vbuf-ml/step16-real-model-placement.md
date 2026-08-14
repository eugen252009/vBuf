# Step 16: real-model tensor placement qualification

Status: **qualified from local evidence; no conversion performed**.

The qualification uses only these repository-relative research artifacts:

| Artifact | Size | SHA-256 |
|---|---:|---|
| `research-models/Qwen3-0.6B-Q8_0.gguf` | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| `research-models/Qwen3-0.6B-BF16.gguf` | 1,509,347,552 | `65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e` |

Both are GGUF v3 files with 32-byte alignment. They were parsed by the
research-only `scripts/qualify_step16.py`; neither binary is tracked.

## Artifact metadata

| Artifact | `general.architecture` | `general.name` | KV count | tensor count | data start | layers | embedding | heads / KV heads | vocabulary |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q8_0 | `qwen3` | `Qwen3 0.6B Instruct` | 28 | 310 | 5,951,136 | 28 | 1,024 | 16 / 8 | 151,936 |
| BF16 | `qwen3` | `Qwen3-0.6B` | 36 | 311 | 5,951,712 | 28 | 1,024 | 16 / 8 | 151,936 |

The Q8_0 artifact has `general.file_type=7`; BF16 has
`general.file_type=32`.

## Inventory and representation

The inventories are structurally equivalent for 310 names and shapes. BF16 has
one additional name, `output.weight`; Q8_0 has no corresponding descriptor.
This is recorded as an artifact/export difference, not silently treated as a
tied-weight fact. No explicit tie metadata was found that proves the
relationship from these files alone.

| Artifact | GGML type | tensors | payload bytes | payload share |
|---|---|---:|---:|---:|
| Q8_0 | F32 | 113 | 262,144 | 0.0174% |
| Q8_0 | Q8_0 (ID 8) | 197 | 633,233,408 | 99.9586% |
| BF16 | F32 | 113 | 262,144 | 0.0174% |
| BF16 | BF16 (ID 30) | 198 | 1,503,133,696 | 99.9826% |

All tensor payload sizes were resolved by checked type/shape geometry. No
payload bytes were read or repacked by the simulator.

## Derived roles

Actual names were classified as follows:

- `token_embd.weight`: `GlobalPre / TokenEmbedding`;
- `blk.N.*`: `Layer(N)` with the suffix retained as the runtime-local role;
- `output_norm.weight`: `GlobalPost / FinalNorm`;
- `output.weight`: `GlobalPost / OutputProjection`;
- anything unmatched remains `Unknown`.

No layer or role fields were added to TensorDirectory.

## Existing physical order

Q8_0 is already physically layer-monotonic after an initial
`output_norm.weight` followed by `token_embd.weight`. Its 308 block tensors
are contiguous by layer.

BF16 begins with `output.weight`, then `token_embd.weight`, and its block order
follows lexicographic names: after `blk.1`, `blk.10` precedes `blk.2`. The
individual layer tensors are contiguous, but multi-layer prefixes are
physically scattered. `output_norm.weight` is at the end.

All tensor payloads in both files are physically contiguous with zero measured
gaps between consecutive payloads. The 32-byte alignment therefore adds zero
padding to the existing GGUF layouts. Simulated alternatives also add zero
padding because every payload size is alignment-compatible.

## Candidates

The simulator evaluates:

1. `GGUF_SOURCE_ORDER` — mandatory baseline, preserving actual offsets;
2. `NAME_ORDER` — lexicographic tensor names;
3. `LAYER_MAJOR` — global-pre, layers in numeric order, global-post;
4. `LAYER_MAJOR_ROLE_ORDER` — the same with deterministic role ordering;
5. `PREFIX_FRIENDLY` — coarse pre-layer globals, numeric layers, post-layer
   globals.

For these artifacts, `LAYER_MAJOR_ROLE_ORDER` and `PREFIX_FRIENDLY` produce the
same order. This is not an artificial distinction.

All candidates preserve the exact payload sizes, representations and logical
names. Only order and 32-byte placement are simulated.

## Locality and prefix results

Every individual layer is one exact-adjacent extent in every candidate. The
important difference is multi-layer and sequential-prefix locality.

For Q8_0, source order is already effectively layer-major:

- layer-0 prefix span: 182,031,360 bytes;
- `LAYER_MAJOR_ROLE_ORDER`: 182,027,264 bytes;
- complete 28-layer prefix span: 633,495,552 vs 633,491,456 bytes.

This difference is only the 4,096-byte placement of the final norm.

For BF16:

| Prefix | GGUF source span | layer-major-role span | Reduction |
|---|---:|---:|---:|
| global/input only | 622,329,856 | 311,164,928 | 49.999% |
| layer 0 | 653,796,352 | 342,631,424 | 47.594% |
| layers 0–3 | 1,314,592,768 | 437,030,912 | 66.761% |
| layers 0–7 | 1,440,458,752 | 562,896,896 | 60.913% |
| layers 0–13 | 1,503,391,744 | 751,695,872 | 50.000% |
| layers 0–27 | 1,503,395,840 | 1,192,226,816 | 20.677% |

Selected-range bytes are unchanged between placements because the same tensor
set is selected. Source BF16 has up to three exact-read extents for early
multi-layer windows; layer-major candidates reduce these to one. Q8_0 source
already has one exact-adjacent extent for these selections.

The complete prefix-readiness curves are in:

```text
benchmark-results/vbuf-ml-step16/*-prefix-readiness.csv
```

Each row distinguishes exact selected-read bytes/ranges from sequential prefix
span. `*-layer-locality.csv` and `partial-loading-comparison.csv` contain the
per-layer and representative-window measurements.

## Sequential, random-range and mmap interpretations

**Sequential stream:** BF16 benefits materially from moving the output tensor
after the layers and numerically ordering layers. Early layer readiness no
longer requires traversing the 311 MiB output tensor or later lexicographic
layers. Q8_0 source order is already suitable.

**Random range reads:** individual-layer reads are already one extent in both
artifacts. Layer-major ordering materially improves BF16 multi-layer windows by
reducing exact-read range count; it has no meaningful Q8_0 benefit.

**mmap:** the result is structural page-span evidence only. A mapping is not a
page-touch claim. No cold-cache or inference performance claim was made.

## First-use milestones

The qualification models a coarse sequence only:

```text
token embedding / required pre-layer globals
→ complete layer 0
→ complete layers 0..k
→ final norm/output tensors
```

It does not claim that an actual runtime can execute at any milestone. The
consumer graph and tied-weight behavior remain later qualification work.

## LayerIndex qualification

A non-serialized runtime-local index was derived from actual names:

```text
layer number → source tensor ordinals
```

Results:

| Artifact | layers | construction | approximate memory | lookup sample |
|---|---:|---:|---:|---:|
| Q8_0 | 28 | ~0.20 ms | 15,728 bytes | ~53 ns |
| BF16 | 28 | ~0.20 ms | 15,728 bytes | ~54 ns |

This supports a runtime-local LayerIndex as useful and cheap for this model,
but does not justify serializing it or adding layer semantics to the wire.

## Placement decision

Recommended writer policy: **`LAYER_MAJOR_ROLE_ORDER`**, with the coarse
prefix treatment represented by `PREFIX_FRIENDLY` where the writer needs to
state that policy explicitly.

Rationale:

- it materially improves BF16 sequential prefix span and multi-layer range
  count;
- it is deterministic and representation-independent;
- it preserves Q8_0's already-good source locality, with only negligible
  movement of global post-layer tensors;
- it creates natural cumulative boundaries after every transformer layer;
- it requires no repacking, quantization knowledge, or backend behavior.

This is a writer recommendation, not a decoder requirement. `GGUF_SOURCE_ORDER`
remains valid and is already sufficient for Q8_0.

## Evidence still required

The files establish GGML type IDs, shapes, sizes and placement, but not the
complete packed-byte semantics required for a representation contract. Before
conversion or typed GGML payload access:

1. pin an exact `ggml-org/llama.cpp` or GGML commit;
2. record the exact type tables, block-size definitions and relevant loader/
kernels used for F32, BF16 and Q8_0;
3. qualify output/tied-weight behavior against that pinned consumer.

No moving branch or memory-based Q8_0 definition is accepted as normative.

## Evidence files

Generated by:

```text
python3 scripts/qualify_step16.py --self-test
python3 scripts/qualify_step16.py
```

The output directory is `benchmark-results/vbuf-ml-step16/` and contains:

- artifact summaries and metadata-key inventory;
- complete source tensor inventories;
- deterministic placement lists for every candidate;
- representation and size distributions;
- largest-tensor tables;
- per-layer locality;
- prefix-readiness curves;
- partial-loading comparisons;
- natural layer boundaries;
- runtime-local LayerIndex measurements;
- parser configuration and repository provenance.

The research models are never copied into the evidence directory.
