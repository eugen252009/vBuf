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

`output.weight` is present as a distinct F32 tensor in the 32B source and is
planned as an independent payload. No tied-output fallback is used.

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
| 3 | 8 | 34,816,197,344 | 30 | 544,003,084 | 64 | 8 |
| 4 | 16 | 34,816,199,728 | 2,414 | 272,001,561 | 64 | 24 |
| 5 | 32 | 34,816,211,776 | 14,462 | 136,000,828 | 64 | 24 |
| 6 | 64 | 34,816,235,840 | 38,526 | 68,000,461 | 64 | 24 |
| 7 | 128 | 34,816,284,032 | 86,718 | 34,000,278 | 64 | 25 |
| 8 | 256 | 34,816,380,672 | 183,358 | 17,000,186 | 64 | 25 |

Common geometry:

```text
block count: 732
physical set-bit count: 732
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

## Final artifact status

The selected planned final size is:

```text
34,816,197,344 bytes
```

Available filesystem space at conversion qualification time:

```text
25,510,162,432 bytes
```

The source was retained and no partial target was created. Final artifact
emission was blocked before writing to avoid consuming the remaining disk space.

Evidence:

```text
benchmark-results/vbuf-ml-step26-qwen32b-placement/final-artifact.json
status = BLOCKED_RESOURCE
```

Therefore the following are not yet available:

```text
final vBuf SHA-256
planned-vs-actual 32B file size
planned-vs-actual 32B offsets
32B tensor payload parity against emitted vBuf
32B checked-reader qualification
32B llama structural load
```

The Rust writer now performs planned-versus-emitted position checks, and the
small layout test suite verifies those invariants, but they could not be run
against a completed 32B file in this environment.

## Tensor-byte parity

Source tensor geometry was validated for all 707 tensors.
No conversion was emitted, so source-to-final payload hash parity is pending
resource availability.

The conversion plan remains `COPY_BYTES` only:

```text
requantization: 0
dequantization: 0
repacking: 0
tensor merging: 0
```

## llama qualification

Not run for the 32B vBuf because no final 32B vBuf artifact was emitted.
The source GGUF was semantically qualified. Existing 0.6B llama correctness
and direct-view qualification remain unchanged.

## 0.6B regression

Existing Step-22/23/24/25 evidence remains present and unchanged.
Workspace tests, layout tests, Step-22/22A/23 tests, and Step-26 planner tests
pass.

No existing 0.6B artifact was overwritten.

## Format changes

```text
v0.6 wire changes: 0
vBuf-ML semantic wire changes: 0
BaseShift legality changes: 0
Nano wire changes: 0
```

The only writer change is explicit plan geometry accounting and validation;
wire semantics remain unchanged.

## Step-27 benchmark readiness

The Step-26 evidence schema is ready to compare arbitrary qualified artifacts
with:

```text
source/vBuf sizes
model identity
layer/tensor/token/merge counts
BaseShift/BaseStep
view/index timing
common runtime timing
model-ready timing
future page-touch/I/O fields when measurable
```

No small-vs-large loader performance claim is made here.

## Blocker and next action

The sole blocking issue is local storage capacity:

```text
required final artifact: ~34.8 GB
available filesystem space: ~25.5 GB
```

Provide a filesystem with at least the source plus one additional approximately
35 GB target capacity, then run the selected-plan conversion and the pending
checked-reader/tensor-parity qualification. Do not change the selected
BaseShift or introduce physical-runtime work to work around storage.
