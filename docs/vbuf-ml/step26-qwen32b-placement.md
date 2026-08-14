# Step 26 — Qwen3-32B scaling and BaseShift placement qualification

Status: **semantic and placement qualification complete; final artifact blocked by available disk space**.

No physical runtime, Nano loading, LayerView, RuntimeChunk, prefetch,
residency, GPU, or streaming work was implemented.

## Source artifact

```text
research-models/Qwen3-32B-Q8_0.gguf
size: 34,817,718,912 bytes
SHA-256: 2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169
GGUF version: 3
```

Semantic inventory:

```text
architecture: qwen3
dense: yes
MoE: no
metadata keys: 28
tensors: 707
layers: 64
embedding dimension: 5120
attention heads: 64
KV heads: 8
key/value head dimensions: 128 / 128
FFN dimension: 25600
context length: 40960
vocabulary: 151,936
merges: 151,387
GGML types: 450 Q8_0, 257 F32
GGUF alignment: 32
```

The artifact is dense Qwen3, not Qwen3-MoE.

## Existing parser result

The existing GGUF parser completed successfully against the 32B artifact.
The semantic conversion path was extended through a generic Step-26 harness
rather than the 0.6B-specific expected-artifact table.

Verified:

```text
64 layer IDs: 0..63
all tensor names classified
all tensor types supported: Q8_0/F32
Q8_0 block geometry valid
Qwen3 metadata complete
GPT2/Qwen2 tokenizer metadata present
```

## 0.6B-specific assumptions found

The existing generic Rust writer/converter did not contain fixed 0.6B tensor
counts or dimensions.

The following research/tooling assumptions were identified and bypassed or
generalized:

```text
Step-18 qualification tables hardcode 0.6B artifact names/hashes
Step-20/22/23 harnesses hardcode 0.6B controls
Python conversion validation required BaseShift=3
Rust conversion requests hardcoded payload alignment=8
```

Step 26 uses a generic Qwen3 manifest path and accepts all legal BaseShift
values. Rust conversion requests now use the selected BaseStep as the minimum
payload alignment. No Qwen3 semantic contract was widened.

## Semantic profile

The existing vBuf-ML profile remains valid without semantic changes.

Verified:

```text
model metadata: PASS
64 layers: PASS
707 tensor roles: PASS
Q8_0/F32 representation inventory: PASS
151,936-token vocabulary: PASS
151,387 numeric merges: PASS
special token IDs: PASS
BOS behavior metadata: PASS
chat template metadata: PASS
```

`output.weight` is present as a distinct Q8_0 tensor in the 32B source and is
planned as an independent payload. The direct llama source now uses it when
present and only uses the historical tied-output fallback when it is absent.

## PlacementPlanner architecture

Added the research qualification harness:

```text
scripts/qualify_step26_qwen32b.py
```

It constructs one logical conversion manifest and one ordered request list,
then simulates all six legal BaseShift candidates without reading tensor
payloads for each candidate.

The placement model matches the canonical writer rules:

```text
24-byte v0.6 header
→ data-region BaseStep alignment
→ block-start BaseStep alignment
→ 8-byte or extended 16-byte block header
→ payload BaseStep alignment
→ exact payload length
```

The Rust `LayoutPlan` now records:

```text
block start
payload start/end
block padding
inner payload padding
final size
payload/header/padding totals
```

The writer checks emitted positions and final size against the computed plan.
Representative plan/write offset and size equality is covered by layout tests.

## Single-pass fan-out

The 32B logical manifest and source metadata are constructed once.
All candidates operate on descriptor/geometry state only:

```text
one logical request sequence
→ BS8 / BS16 / BS32 / BS64 / BS128 / BS256 geometry states
```

No six payload copies are created.

## Candidate evaluation

| BaseShift | BaseStep | Final size | Padding | Nano bytes | Layer spans | Coalesced spans |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 8 | 34,816,197,376 | 44 | 544,003,084 | 64 | 10 |
| 4 | 16 | 34,816,199,792 | 2,460 | 272,001,561 | 64 | 26 |
| 5 | 32 | 34,816,211,904 | 14,572 | 136,000,828 | 64 | 26 |
| 6 | 64 | 34,816,236,096 | 38,764 | 68,000,461 | 64 | 26 |
| 7 | 128 | 34,816,284,544 | 87,212 | 34,000,278 | 64 | 27 |
| 8 | 256 | 34,816,381,696 | 184,364 | 17,000,187 | 64 | 27 |

Common geometry:

```text
block count: 734
physical set-bit count: 734
tensor blocks: 707
layer count: 64
average spans/layer: 1
maximum spans/layer: 1
internal layer gap bytes: 0
bytes amplification: approximately 1.00000027–1.00000538
```

The layer-major placement has one physical span per layer for every candidate.

## Hypothetical Nano comparison

Nano remains noncanonical and is not persisted.

The mathematical hypothetical bitmap sizes are:

```text
BS8:   544.0 MB
BS16:  272.0 MB
BS32:  136.0 MB
BS64:   68.0 MB
BS128:  34.0 MB
BS256:  17.0 MB
```

This is only physical-bootstrap accounting. It does not justify Nano in normal
semantic loading.

## Pareto analysis

The candidate metrics trade off two independent objectives:

```text
BaseShift 3 minimizes file size and padding.
BaseShift 8 minimizes hypothetical Nano storage.
```

No candidate dominates all others across:

```text
final size
padding
hypothetical Nano bytes
layer spans
bytes amplification
```

The non-dominated set contains the practical endpoints. The deterministic
selection tie-break is:

```text
Pareto filter
→ lowest final size
→ lowest padding
→ smallest BaseStep
```

## Selected BaseShift

```text
selected BaseShift: 3
selected BaseStep: 8 bytes
```

Rationale:

```text
smallest final artifact
lowest padding
same one-span-per-layer geometry
lowest bytes amplification
Nano is not a wire/runtime feature in this step
```

This is an artifact-qualified choice, not a universal vBuf policy.

## Conversion tax

Measured on the local environment:

```text
logical parse/manifest construction: approximately 19.3 s
all-six-candidate planning: approximately 5.7 ms
candidate selection: negligible
```

The manifest/parser time includes source metadata parsing and source hashing.
Candidate fan-out itself does not reread or duplicate model payloads.

## Planner memory

```text
candidate count: 6
payload duplication during planning: 0
state proportional to descriptor/geometry counts
```

The planner does not create six model files or six payload buffers.

## Historical disk-capacity blocker

The initial Step-26 run stopped before emission because only
`25,510,162,432` bytes were available. The source was retained and no partial
target was created. This historical evidence remains in the earlier
`final-artifact.json` state recorded by the resource-limited qualification.

The planner was then corrected to include the two tokenizer identity payloads
that the production writer emits. This was deterministic accounting repair,
not a placement-policy change: BaseShift remained 3 and the selected offsets
now match the writer exactly.

## Final artifact

```text
path: research-models/Qwen3-32B-Q8_0.vbuf
actual size: 34,816,197,376 bytes
SHA-256: 84597064d5b3530572959345286368b17e891e640b5958bba7f0b67980cd119d
BaseShift: 3
BaseStep: 8 bytes
```

The final artifact is the only emitted 32B candidate. No six-way artifact
fan-out or second payload copy was created.

## Planned versus actual

```text
planned size: 34,816,197,376 bytes
actual size:  34,816,197,376 bytes
difference:   0 bytes
planned offsets: 707/707 tensor payload offsets match
writer checks: PASS
```

## Exactness and readers

```text
tensor inventory: 707/707 PASS
source/destination payload hashes: 707/707 PASS
repack/requantize/dequantize/reorder: 0
canonical checked reader: PASS
Bootstrap: PASS
ModelMetadata: PASS
TensorDirectory: PASS
TokenizerMetadata: PASS
Q8_0 geometry and payload ranges: PASS
BorrowedModelView: PASS
```

Tokenizer parity passed for vocabulary `151,936`, merges `151,387`, token
bytes, merge pairs, special IDs, BOS behavior, and chat template.

## Runtime indexes

The Step-25 reusable indexes were exercised independently:

```text
Borrowed TokenIndex: 151,936 entries, build ≈7.09 ms
Packed MergeRankIndex: 151,387 entries, build ≈4.17 ms
canonical token/merge key bytes copied: 0
```

The normal direct llama source does not inject duplicate Rust indexes.

## llama qualification

Pinned llama.cpp:

```text
4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
```

CPU-only direct-source construction passed:

```text
architecture and metadata: PASS
tensor dimensions/types: PASS
payload pointer validity: PASS
MODEL_READY: PASS
```

The direct source was corrected to preserve a distinct `output.weight` when
present; the tied-output fallback remains only for artifacts without that
tensor.

Controls:

```text
GGUF MODEL_READY: ≈18.486 s
vBuf MODEL_READY: ≈0.461 s
```

The direct path matched the GGUF control:

```text
logit max_abs_diff: 0
generation parity: PASS
output: 504,13027,0,863,198,198,2,19143
```

These are qualification controls only; no Step-27 scaling claim is made.

## Resource qualification

```text
free before emission: 237,715,451,904 bytes
free after qualification: 202,829,688,832 bytes
RAM available at capture: ≈61.1 GB
swap: 25.8 GB total, 6.5 GB used
GPU: not involved
peak RSS/page faults/bytes read: unavailable
```

## 0.6B regression

Existing Step-22/23/24/25 evidence remains present and unchanged. Workspace,
layout, Step-22/22A/23, and Step-26 planner tests pass. No existing 0.6B
artifact was overwritten.

## Format and architecture guards

```text
v0.6 wire changes: 0
vBuf-ML semantic wire changes: 0
Nano persistence/changes: 0
placement policy changes: 0
LayerView/RuntimeChunk/prefetch/residency/GPU/MoE work: 0
```

## Step-27 readiness

Step-26 now provides the large-model correctness-qualified control artifact
and raw controls for source open, canonical validation, BorrowedModelView,
runtime indexes, llama construction, MODEL_READY, logit parity, and generation.
A controlled small-versus-large benchmark remains Step 27 work; no scaling
claim is made here.

## Final status

```text
STEP 26: COMPLETE
```

Evidence is under:

```text
benchmark-results/vbuf-ml-step26-qwen32b-placement/
```
