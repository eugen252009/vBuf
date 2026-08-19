# Model Import Boundary Audit

## Scope

This audit asks what must be imported to execute model families through the
vBuf-owned runtime, what the current profile contains, what can be derived at
runtime, and what is redundant or incorrectly coupled to a pinned consumer.

The target families are:

- Dense Transformer models
- Transformer MoE models
- Hybrid attention/SSM MoE models such as Qwen3.6
- Future tokenizer and architecture families

## Conclusion

The current profile imports **model storage** well for selected GGUF-derived
models. It does not yet import a portable description of **model behavior,
state, and execution scheduling**.

The correct split is:

```text
vBuf import
    model identity, tensor facts, ranges, tokenizer data, graph recipe, state schema
        |
        v
generic runtime
    graph scheduling, range acquisition, residency, eviction, ggml execution
        |
        v
architecture lowering
    Dense Transformer, DeepSeek, Qwen MoE, Qwen hybrid SSM/attention, ...
```

The current graph abstraction is a good scheduler seed, but it is not yet an
executable interchange representation.

## Classification

| Area | Present | Derivable | Missing | Redundant or coupled |
| --- | --- | --- | --- | --- |
| Tensor identity | Name, shape, representation, canonical occurrence | Strides and byte count | Alias/tie semantics, storage class | Backend type IDs should remain adapter-local |
| Tensor payload | Exact source range and payload length | Quantization geometry | Sparse/view/strided layouts | Backend buffers and repacking state |
| Transformer metadata | Architecture, context, embedding, layers, heads, FFN, epsilon, RoPE, KV dimensions | Vocabulary size and some defaults | Attention variant, masks, windows, per-layer schedule | Global fields assume Transformer semantics |
| Tokenizer | GPT2 BPE/Qwen2 vocabulary, merges, special IDs, template | Runtime lookup indexes | SentencePiece, Unigram, WordPiece, byte fallback, normalization pipeline | Runtime indexes should not be serialized |
| MoE catalog | Expert count, active count, layers, shared count, entries | Layer membership from names | Router semantics, top-k policy, weighting, capacity/drop behavior, roles | Architecture detection by string |
| Hybrid SSM | SSM tensor presence in Qwen3.6 | Coarse layer grouping | Transition equations, convolution, recurrent state, schedule | `SsmScan` enum without attributes is only a label |
| Execution graph | Values, persistent tensors, operation kinds, first-use order | Total persistent bytes | Shapes, dtypes, axes, attributes, masks, dynamic dependencies, state | `source_offset` alone is insufficient for multi-source inputs |
| Residency | Range reads, LRU budget, acquire/release primitives | Working-set size from graph | Atomic graph transactions and leases tied to operation completion | Layer parsing from `blk.N.` names |

## Current Information

### Canonical and profile structure

**Present:** `vbuf-ml` validates canonical blocks, bootstrap roles, checked
ranges, tensor names, dimensions, representations, and exact payload sizes.
The source profile can identify persistent ranges and external sources.

**Derivable:** contiguous strides, quantization block geometry, layer grouping
for models following `blk.N.*`, tensor byte counts, tokenizer lookup indexes,
and source read plans.

**Missing:** a graph recipe role, state schema role, per-layer schedule, tensor
storage class, alias relation, and a typed architecture descriptor.

**Redundant:** ggml type IDs, backend placement, repack buffers, device handles,
and scheduler state should not be persisted in the format. They belong to the
runtime adapter.

Relevant implementation:

- `rust/vbuf-ml/src/bootstrap.rs`
- `rust/vbuf-ml/src/metadata.rs`
- `rust/vbuf-ml/src/tensor_directory.rs`
- `rust/vbuf-ml/src/source.rs`

### Model metadata

The current metadata keys cover the selected Qwen3 and DeepSeek consumers:

```text
architecture
context length
embedding length
layer count
attention heads and KV heads
key/value dimensions
feed-forward length
normalization epsilon
RoPE theta
```

This is insufficient for arbitrary families because it assumes global
attention-oriented values. Qwen3.6 exposes additional facts such as:

```text
qwen35moe.ssm.*
qwen35moe.full_attention_interval
qwen35moe.nextn_predict_layers
qwen35moe.expert_shared_feed_forward_length
qwen35moe.rope.dimension_sections
```

Those values are present in the source GGUF, but the current model metadata
profile does not preserve them as a structured execution contract.

Classification:

- **Present:** selected scalar Transformer metadata.
- **Derivable:** vocabulary size, some dimensions, layer count from tensor names.
- **Missing:** per-layer block type, attention schedule, SSM parameters/state,
  masks, positional variants, and execution attributes.
- **Redundant:** backend-specific tensor allocation and repacking information.

Relevant implementation:

- `rust/vbuf-ml/src/metadata.rs`
- `scripts/build_step18_manifest.py`

### Tensor descriptors

Current tensor descriptors preserve:

```text
name
dimensions
representation
canonical key/occurrence
source ID and range
```

This is sufficient to locate immutable contiguous weight bytes. It does not
say whether a descriptor is a weight, activation, mutable state, KV cache,
temporary, alias, or tied parameter.

Qwen3.6 demonstrates why names alone are insufficient. Its 753 tensors include
both attention tensors and SSM tensors such as:

```text
ssm_a
ssm_alpha
ssm_beta
ssm_conv1d
ssm_dt
ssm_norm
ssm_out
```

The importer can store these tensors, but the runtime cannot infer their
equations or lifetime from names alone.

### Tokenizers

The current tokenizer profile is deliberately GPT2 BPE/Qwen2-oriented. It
stores vocabulary text, offsets, token types, scores, merges, special IDs, BOS
behavior, and chat-template bytes.

This is **present for the qualified Qwen/DeepSeek fixtures**, but not a generic
tokenizer import. Missing families include:

- SentencePiece and Unigram models
- WordPiece
- Byte fallback and arbitrary byte tokens
- Normalizers and pre-tokenizer pipelines
- Decoder behavior and added-token rules
- Rich special-token policies

The current UTF-8/no-NUL vocabulary constraint may also reject valid byte-level
tokenizer entries.

Relevant implementation:

- `rust/vbuf-ml/src/tokenizer.rs`
- `rust/vbuf-ml/src/runtime_tokenizer.rs`
- `scripts/build_step18_manifest.py`

### MoE information

The current MoE directory stores expert counts, active counts, layer counts,
shared-expert count, layer/expert indexes, role numbers, and child names.

This is not enough to execute arbitrary MoE models. The following must be
imported or explicitly lowered:

- Router input and output dimensions
- Top-k and grouped-routing policy
- Router bias and normalization
- Expert score normalization
- Capacity and token-drop policy
- Shared, routed, gate, up, down, and auxiliary roles
- Per-layer expert availability
- Dynamic selected-expert dependencies

The numeric `role: u16` field currently has no portable semantic enum.

There is also a serious conversion limitation: the current Rust converter emits
synthetic nested MoE marker streams rather than child entries backed by the
actual expert tensor groups. The tensor payloads themselves remain in the main
directory, but the MoE child catalog is not yet an executable expert storage
description.

Relevant implementation:

- `rust/vbuf-ml/src/moe.rs`
- `rust/vbuf-ml/src/qwen_moe.rs`
- `rust/vbuf-ml/src/deepseek_moe.rs`
- `rust/vbuf-ml/src/bin/vbuf-ml-convert.rs`

## Execution Graph Requirements

The new graph view currently contains:

```text
PersistentTensor
ValueId
OperationKind
Operation inputs and one output
first-use tensor acquisition order
```

That is enough to demonstrate that dense and MoE graphs can share a scheduler
and residency view. It is not enough to execute them.

Each operation needs typed attributes. The minimum generic operation contract
should include:

| Requirement | Why it is needed |
| --- | --- |
| Input/output shapes | Validate graph construction and allocate values |
| Input/output representations | Select ggml kernels and views |
| Axes and permutations | Attention, reshape, transpose, concat, split |
| Numeric parameters | Epsilon, scale, RoPE, activation, clipping |
| Dynamic indices | Top-k, gather, scatter, expert dispatch |
| Masks and sequence positions | Causal, sliding-window, recurrent boundaries |
| State references | KV cache, SSM state, positions, sequence metadata |
| Multiple outputs | Router scores/indices, attention outputs, state updates |
| Lifetime/lease requirements | Release only after backend completion |
| Source and range identity | Acquire from local, HTTP, or other source |

The operation vocabulary also needs generic data movement and shape operators:

```text
reshape/view/permute
concat/split/repeat
softmax
top-k/argsort
gather/scatter/combine
indexed matmul
state read/write
```

`SsmScan` and `ExpertDispatch` are useful operation names, but they require
attributes and state contracts to be executable.

## Model-Family Matrix

| Family | Storage import | Graph import | State import | Current status |
| --- | --- | --- | --- | --- |
| Dense Qwen3 | Qualified selected formats | External/pinned consumer graph | KV is runtime-owned | Works only through selected consumers |
| Generic Dense Transformer | Often possible if tensor formats fit | Missing generic lowering | Missing generic state schema | Storage partial, execution not generic |
| DeepSeek2 MoE | Selected fixture supported | POC22 graph exists outside Rust runtime | Native POC state exists | Strong qualification prototype |
| Generic MoE | Partial catalog only | Missing routing contract | Missing dynamic expert/state contract | Not portable |
| Qwen3.6 35B-A3B | Storage vBuf converted and validated | Missing Qwen3.6 lowering | Missing SSM/hybrid state | Not executable yet |
| Phi-4-mini-instruct | Pinned config/index/header audit | Importer-owned metadata prototype | Generic KV schema prototype | P0 survives with generic alias/view/position refinements; no execution |
| Arbitrary tokenizer | No | N/A | N/A | GPT2/Qwen2 only |

## Priority Plan

### P0: Required for correctness

1. Add a versioned graph recipe region or an equivalent importer-owned graph
   descriptor.
2. Extend graph operations with typed attributes, shapes, multiple outputs, and
   state references.
3. Add a persistent state schema for KV, recurrent/SSM state, positions, masks,
   and sequence metadata.
4. Replace synthetic MoE child markers with actual tensor-group references and
   typed roles.
5. Make layer acquisition graph-driven rather than based on `blk.N.` naming.

### P1: Required for broad model coverage

1. Add architecture importer registration instead of architecture string tests.
2. Add tokenizer kinds and versioned tokenizer pipeline descriptors.
3. Add generic data movement and dynamic routing operations.
4. Add alias/tied-weight relations.
5. Add atomic layer acquisition preflight so a layer larger than the budget
   fails before partially inserting tensors.

### P2: Robustness and optimization

1. Add tensor storage classes and mutability declarations.
2. Add external source IDs and lease policies directly to runtime graph inputs.
3. Add per-operation backend capability requirements.
4. Avoid materializing owned tokenizer snapshots when borrowed views suffice.
5. Replace content-based control-index lookup in the converter with explicit
   control identities.

## Final Assessment

The current vBuf profile is a strong **validated weight/range container** and
the POC22 runtime proves that a model can execute with bounded tensor residency.
The missing layer is a portable **behavior and state import contract**.

The abstract graph should therefore become the central imported/lowered view,
while the scheduler, range source, residency policy, and ggml bridge remain
architecture-neutral.

## 1. Objective

The objective is a portable vBuf-ML semantic model that can be imported from a
vendor/model artifact and executed by a neutral runtime. The runtime must not
need to identify Qwen, Llama, Gemma, Mistral, DeepSeek, or another vendor in
order to choose model behavior.

This audit does not change vBuf v0.6. The v0.6 specification is authoritative
for canonical blocks, checked ranges, physical geometry, and generic extension
rules. Execution semantics belong in vBuf-ML or an importer-owned sidecar, not
in vBuf Core.

`AGENTS.md` was requested for this audit but is absent from the repository tree.
No additional repository-local AGENTS instructions were available.

## 2. Current Semantic Boundary

The current import pipeline is:

```text
GGUF metadata/tensor records
    -> family-specific manifest/converter
    -> canonical v0.6 blocks
    -> vBuf-ML metadata/tokenizer/tensor/MoE views
    -> direct consumer or PoC runtime
```

The current views preserve model storage facts and a limited tokenizer/profile
contract. They do not preserve an executable behavior/state contract. The
PoC22 lineage proves that bounded execution can be built from generic graph,
materializer, and residency concepts, but its graph lowering remains outside
the Rust runtime and is DeepSeek-shaped.

Authoritative evidence:

- `spec/spec_0.6.md:8-14,205-224` keeps vBuf Core generic and canonical.
- `research/vbuf-native-region-runtime-audit.md:877-947` defines the minimum
  generic operation and state concepts for the demonstrated region.
- `research/vbuf-native-region-runtime-audit.md:1011-1060` defines dynamic
  MoE selection as generic runtime resource planning.
- `research/vbuf-native-region-runtime-audit.md:1070-1116` separates the
  optional execution profile from vBuf Core and backend state.

## 3. Required Four-Way Classification

Every candidate field is classified using exactly one category:

```text
PERSISTENT_SEMANTIC_TRUTH
DERIVABLE_RUNTIME_INFORMATION
BACKEND_SPECIFIC_LOWERING_STATE
TRANSIENT_EXECUTION_STATE
```

| Candidate information | Classification | Persist? | Reason |
| --- | --- | ---: | --- |
| Tensor role/binding | PERSISTENT_SEMANTIC_TRUTH | Yes | A name and shape do not identify meaning portably. |
| Tensor shape | PERSISTENT_SEMANTIC_TRUTH | Yes | Required to reconstruct values and validate graph bindings. |
| Tensor representation contract | PERSISTENT_SEMANTIC_TRUTH | Yes | Payload interpretation and exact byte validation depend on it. |
| Tensor source range | PERSISTENT_SEMANTIC_TRUTH | Yes | Required to acquire immutable model bytes. |
| Tensor strides | DERIVABLE_RUNTIME_INFORMATION | No | Derive contiguous strides from shape and representation where valid. |
| Tensor byte size | DERIVABLE_RUNTIME_INFORMATION | No | Derive from representation and shape; validate against payload length. |
| Tensor alias/tie relation | PERSISTENT_SEMANTIC_TRUTH | Yes | Prevents semantic duplication and preserves tied weights. |
| Layer/expert physical locality | DERIVABLE_RUNTIME_INFORMATION | Optional hint | Derive from graph bindings and ranges; useful for acquisition only. |
| Execution dependencies | PERSISTENT_SEMANTIC_TRUTH | Yes or trusted importer sidecar | Required to reconstruct behavior and lifetime. |
| Operation shapes and semantic attributes | PERSISTENT_SEMANTIC_TRUTH | Yes or trusted importer sidecar | Backend cannot derive model equations from raw tensors alone. |
| State schema | PERSISTENT_SEMANTIC_TRUTH | Yes or trusted importer sidecar | Required to instantiate KV, recurrent, and sequence state. |
| Current-token expert dispatch | TRANSIENT_EXECUTION_STATE | No | Derived from router outputs for the current input. |
| Runtime execution schedule | DERIVABLE_RUNTIME_INFORMATION | No | Derive from dependencies, state, and backend policy. |
| Source read plan | DERIVABLE_RUNTIME_INFORMATION | No | Derive from required TensorRefs and source registry. |
| Residency cache contents | TRANSIENT_EXECUTION_STATE | No | Runtime working set, not model meaning. |
| HTTP request schedule | BACKEND_SPECIFIC_LOWERING_STATE | No | Transport policy must not become model semantics. |
| Backend buffer handle | BACKEND_SPECIFIC_LOWERING_STATE | No | Process/device ownership is not portable model meaning. |
| Device placement | BACKEND_SPECIFIC_LOWERING_STATE | No | Reconstructed from runtime/backend capabilities. |
| GGML type ID | BACKEND_SPECIFIC_LOWERING_STATE | No | Map portable representation to a backend type at lowering time. |
| Repacked tensor bytes | BACKEND_SPECIFIC_LOWERING_STATE | No | Optional backend cache; canonical bytes remain authoritative. |
| Tokenizer lookup indexes | DERIVABLE_RUNTIME_INFORMATION | No | Build from persisted tokenizer semantic data. |
| Chat conversation history | TRANSIENT_EXECUTION_STATE | No | Application/session state, not immutable model semantics. |

## 4. Tensor Roles and Aliases

The current tensor descriptor has name, shape, representation, occurrence, and
range. That is not enough for a neutral runtime because tensor names are source
conventions. Qwen3.6 and DeepSeek expose roles that cannot safely be inferred
by a backend from names alone.

The portable representation needs a binding relation with at least:

```text
semantic binding ID
role/path
TensorRef or alias target
shape/representation constraints
optional layer/region scope
```

The role should be a typed semantic path or importer-defined binding, not a
fixed vendor enum. For example, `attention.query.weight` is a portable role;
`blk.12.attn_q.weight` is source naming.

Aliases are persistent semantic truth. Tied embedding/output weights, shared
expert weights, and exact shared tensor storage cannot be safely rediscovered
from equal bytes alone. Backend views and slices remain derivable/lowering
state unless they change semantic identity.

## 5. Layer and Program Semantics

A plain static DAG is insufficient as the complete model representation.
The minimum portable model program is a typed dataflow graph plus explicit
state and dynamic alternatives:

```text
PortableProgram
    immutable TensorRefs and semantic bindings
    typed operation nodes and edges
    region boundaries
    StateRefs and state transitions
    dynamic selector -> alternative bindings
```

The current `rust/vbuf-runtime/src/graph.rs` is only a seed. It has operation
names and inputs/outputs, but no shapes, dtypes, axes, attributes, multiple
outputs, masks, state references, or dynamic alternatives.

The graph should represent behavior compositionally. A layer annotation may be
an import-time hint, but the runtime should execute an `ExecutionRegion` made
of node and TensorRef dependencies. This supports embedding/output regions,
expert subgraphs, and regions that cross a semantic layer boundary.

Required common operations are supported by repository evidence in
`research/vbuf-native-region-runtime-audit.md:902-918`:

```text
MatMul / IndexedMatMul
Add / Mul / Div
Norm
Activation
Reshape / View / Permute / Concat / Repeat
RoPE
Attention or decomposed attention
Softmax
TopK / Argsort
Gather / GetRows
Scatter / SetRows / Combine
StateRead / StateWrite
```

`Attention`, `ExpertDispatch`, and `SsmScan` may remain semantic composite
operations only when their attributes, state effects, and lowering contract
are explicit. An opaque family-specific node would hide required acquisition
and state dependencies and is therefore not acceptable as the default.

## 6. State Semantics

State is not optional for modern autoregressive execution. Immutable model
weights and transient execution buffers must be separated from persistent
execution state.

The minimum state descriptor is:

```text
StateRef {
    id
    representation
    shape/schema
    lifetime: step | sequence | session | region-chain
    scope: global | stream | layer-indexed | region-indexed
    access: read | write | read-write
    initialization
    reset behavior
}
```

State transitions carry a descriptor ID, slice/index expression, and graph
dependency. Contents are never serialized as immutable model data.

| State kind | Persist schema? | Runtime contents? | Required semantics |
| --- | ---: | ---: | --- |
| KV cache | Yes | No | Layer/head axes, position growth, read/write slices, reset and sequence scope |
| SSM recurrent state | Yes | No | State shape, transition, update order, initial state, per-sequence scope |
| Convolution state | Yes | No | Kernel/window shape, rolling update, initialization, reset |
| Position/token state | Yes | No | Position source, increment, prefill/decode behavior, mask interaction |
| Router result | No | No | Transient selector output; feeds dynamic alternatives |
| Conversation history | No | No | Application/session data, outside immutable model semantics |

## 7. Dynamic Routing

MoE is not just a static graph because selected experts depend on runtime
values. The semantic program must express two phases:

```text
router graph -> selector values
selector values -> dynamic TensorRef alternatives -> expert graph
```

The portable concept is a dynamic alternative set, not a DeepSeek/Qwen expert
class:

```text
DynamicTensorSet {
    selector value
    selector value -> TensorRef/range set
    consumer nodes
}
```

The importer persists router operation semantics, expert role bindings, and
selection policy. The runtime derives current-token IDs, selected alternatives,
acquisition, and release. Capacity/drop behavior is persisted only when it is
architecturally observable; cache policy and dispatch tables are transient.

## 8. Masks and Position Semantics

`causal = true` is not a sufficient portable mask contract. The semantic model
must describe mask construction and position transforms, including where
applicable:

```text
causal/prefix behavior
sliding or local windows
global/local attention schedules
position-dependent visibility
RoPE dimensions and sections
frequency/base scaling
attention sinks or other explicit mask rules
```

These are persistent semantic attributes. The concrete mask tensor, cache
layout, and backend implementation are derived runtime state.

## 9. Tokenizer Pipeline

Tokenizer data is not the same as a chat template. The portable tokenizer
semantic must distinguish:

```text
normalization
pre-tokenization
token model: BPE, Unigram, WordPiece, or other
byte fallback
added/special token recognition
BOS/EOS policy
decoder/byte reconstruction
```

The current `Gpt2BpeQwen2` profile is a qualified narrow pipeline. Its merge
lookup indexes are derivable. A chat template is optional application/profile
behavior and must not be confused with the immutable execution graph.

The current UTF-8/no-NUL text-pool rule is not sufficient for arbitrary byte
fallback tokenizers. A future tokenizer profile must preserve raw token bytes
when required and describe decoding separately.

## 10. Model-Family Capability Matrix

Only capabilities supported by repository evidence are marked. Others are
`NOT_YET_AUDITED`, not inferred from brand or common external knowledge.

| Semantic capability | Qwen3/Qwen3.6 evidence | Llama | Gemma | Mistral/Mixtral | DeepSeek evidence | Portable primitive |
| --- | --- | --- | --- | --- | --- | --- |
| Embedding | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | `Embedding`/`GetRows` |
| RMSNorm | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | `Norm(kind=RMS)` |
| LayerNorm | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | `Norm(kind=Layer)` |
| Linear projection | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | `MatMul` |
| RoPE | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | `RoPE` with position input |
| GQA/MQA | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | attention shape attributes |
| Sliding/local window | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | mask/visibility semantics |
| Dense gated MLP | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | composed activation/matmul |
| MoE | PRESENT | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | router/alternatives/dispatch |
| Dynamic routing | PRESENT in PoC22 | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT in PoC22 | dynamic selector set |
| Shared experts | PRESENT in selected artifacts | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | PRESENT | ordinary graph branch |
| SSM | PRESENT in Qwen3.6 tensor inventory | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | `SsmScan` plus state attributes |
| Recurrent state | Qwen3.6 semantics NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | native state in PoC22 is KV-focused | `StateSchema` |
| Convolutional recurrent state | PRESENT as Qwen3.6 tensor evidence only | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | stateful convolution primitive |
| KV state | Runtime evidence present | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | Present in native audit | `StateRead/Write` |
| Weight aliases | Qwen output fallback evidence | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | alias relation |
| Special mask semantics | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | NOT_YET_AUDITED | explicit mask program |

The strongest materially different local evidence is Qwen3.6 hybrid
attention/SSM/MoE and DeepSeek2 MoE/MLA. Llama, Gemma, and Mistral/Mixtral are
not sufficiently audited in this repository to claim capability coverage.

## 11. Common, Optional, Stateful, and Routing Primitives

### Common primitives

```text
TensorRef and semantic binding
Value and shape/view operations
MatMul and elementwise arithmetic
Norm
Activation
Residual/Add/Mul
Dependency and region boundaries
```

### Optional model primitives

```text
RoPE and other position transforms
Attention
Softmax
Gated MLP composition
SsmScan
Convolution
IndexedMatMul
```

### Stateful primitives

```text
StateRead
StateWrite
position/sequence state
KV append/read
SSM transition
rolling convolution state
```

### Routing/control primitives

```text
TopK/Argsort
Gather/Scatter/Combine
DynamicTensorSet
conditional region dependency
iteration over token/sequence steps
```

## 12. Persistent Versus Derived Model

The smallest justified conceptual model is:

```text
PortableModel
|
|-- immutable semantic metadata
|-- tensors and semantic bindings
|-- tokenizer pipeline semantics
|-- typed executable program
|-- state schemas and transitions
|-- routing/conditional alternatives
`-- aliases and shared-storage relations
        |
        v
neutral runtime
|-- derives execution schedule
|-- derives source/range acquisition
|-- owns residency and leases
|-- creates transient values/state contents
`-- lowers operations to a backend
        |
        v
backend implementation
```

The program may be serialized as an optional versioned vBuf-ML execution
profile or supplied by a trusted importer sidecar. It must not be added to the
vBuf v0.6 base contract. If it is not persisted, the importer is a deployment
dependency and must be available before the neutral runtime can execute.

## 13. Importer and Runtime Responsibilities

### Importer responsibility

- Read vendor/model metadata and tensor naming conventions.
- Validate source-specific tensor shapes and representation contracts.
- Assign portable semantic tensor bindings and aliases.
- Lower model-family behavior into typed graph/state/routing semantics.
- Select tokenizer pipeline semantics.
- Reject unsupported or ambiguous source behavior.

### Neutral runtime responsibility

- Validate the portable semantic profile.
- Execute graph dependencies and state transitions.
- Derive acquisition, source reads, residency, and eviction.
- Evaluate dynamic routing and acquire selected alternatives.
- Instantiate transient state and backend values.
- Lower portable operations without model-family switches.

### Backend responsibility

- Map representations to native types.
- Allocate buffers and choose device placement.
- Fuse or decompose operations.
- Select kernels and synchronization mechanisms.
- Maintain backend caches and implementation-specific scheduling.

## 14. PoC22 Relationship

PoC22 can be the first lowering target, not the semantic authority:

```text
DeepSeek source
    -> DeepSeek import validator/lowerer
    -> portable ops, TensorRefs, StateRefs, dynamic alternatives
    -> PoC22/ggml lowering
    -> bounded vBuf range/residency runtime
```

The existing PoC22 evidence proves the value of generic dynamic acquisition and
bounded residency, but it should not define the portable model schema. Its
architecture-specific graph construction remains an importer/lowering
implementation to be replaced or isolated.

## 15. Neutrality Acceptance Criteria

The portable representation is neutral only when all of the following hold:

```text
NO vendor/model switch in the execution runtime
NO tensor-name interpretation during execution
NO GGML type IDs in persisted semantics
NO backend buffer layout persisted
NO device placement persisted as model truth
NO runtime lookup indexes persisted unnecessarily
NO HTTP request schedule persisted
Dynamic expert acquisition uses selector values and TensorRefs
KV/SSM state is described through generic StateRefs
Qwen and DeepSeek lower into the same scheduler/resource interfaces
Backend lowering can fuse/decompose without changing model semantics
```

## 16. P0 / P1 / P2

### P0

Define the minimal **typed program plus state schema** needed to represent the
current Qwen3.6 hybrid artifact: tensor bindings, per-layer operation
composition, SSM/attention distinction, state descriptors, and dynamic expert
alternatives. Keep it as an importer-owned/sidecar semantic profile until its
wire representation is separately specified.

### P1

- Generalize tokenizer pipeline semantics beyond GPT2/Qwen2.
- Add generic MoE roles, routing policies, aliases, and masks.
- Add graph-driven region acquisition and atomic budget preflight.
- Audit a materially different dense family and a separately documented MoE
  family before standardizing additional primitives.

### P2

- Optional graph optimization hints and locality annotations.
- Capability matrices for backend lowering.
- Broader tokenizer byte/decoder behavior.
- Distributed region and activation boundaries.

## 17. Open Questions

- Should the typed program be a persisted optional vBuf-ML profile or a signed
  importer sidecar first?
- What is the smallest portable attribute encoding for arbitrary axes,
  dimensions, and slice/index expressions?
- Which Qwen3.6 SSM transition semantics are required for exact recurrence,
  rather than merely storing its weight tensors?
- Which mask and position behaviors are common enough for primitives versus
  represented as importer-defined composition?
- How should graph versioning and compatibility be validated independently of
  the v0.6 base version?

## 18. Recommended Next Implementation Step

Implement **one importer-owned typed program and state schema for the current
Qwen3.6 hybrid artifact**, without adding it to vBuf Core or implementing
kernels. It should be sufficient to express one attention layer, one SSM layer,
one MoE routing region, and their state transitions using generic TensorRefs,
StateRefs, operation attributes, and dynamic alternatives.

## 19. P0.1 Phi Falsification Evidence

The materially different `microsoft/Phi-4-mini-instruct` family was audited at
Hugging Face revision `cfbefacb99257ffa30c83adab238a50856ac3083` without
downloading or executing weights. Its pinned configuration and safetensors
headers establish dense 24/8 GQA, fused QKV and gate/up projections, partial
LongRoPE, per-layer KV state, and tied input/output embeddings.

The Qwen sidecar vocabulary could express dense attention and MLP by
composition, but the audit exposed four generic gaps: semantic storage views
for fused tensors, alias/shared-storage relations, structured position
transform attributes, and structured KV lifecycle fields. No Phi-specific
portable operation or runtime branch was required. SSM, MoE, and dynamic
alternatives are optional and must not be forced into dense regions.

Evidence and the metadata-only prototype are recorded in
`research/results/vbuf-import-boundary-audit/p0-1-phi-abstraction-falsification.md`
and `phi4-mini-typed-program.json`. The result is
`ABSTRACTION_SURVIVES_WITH_GENERIC_REFINEMENTS`; one more model-family audit is
recommended before PoC22 lowering.
