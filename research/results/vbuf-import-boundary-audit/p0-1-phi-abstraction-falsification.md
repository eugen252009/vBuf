# P0.1 Phi-4-mini Abstraction Falsification

## 1. Objective

Attack the Qwen3.6 importer-owned typed program/state sidecar with a materially
different dense Transformer, without executing Phi, changing vBuf v0.6, or
lowering into PoC22/GGML.

## 2. Why Phi Was Selected

Phi-4-mini-instruct is a dense decoder-only Transformer rather than Qwen3.6's
hybrid attention/SSM MoE. It stresses fused projection storage, dense gated
MLP composition, GQA, partial rotary position semantics, LongRoPE, KV state,
and tied input/output embeddings.

## 3. Exact Phi Artifact and Revision

- Repository: `microsoft/Phi-4-mini-instruct`
- Model/config/tokenizer revision: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Model revision is the verified Hugging Face `main` HEAD at audit time.
- No model payload was downloaded. Configuration, tokenizer files/metadata,
  model index, and safetensors headers were inspected over HTTP.
- Model index reports 3,836,021,760 BF16 parameters across two safetensors
  shards; the model card describes the model as 3.8B parameters.

## 4. Verified Phi Architecture Semantics

`config.json` declares `Phi3ForCausalLM`, 32 layers, hidden size 3072, 24
attention heads, 8 KV heads, head dimension 128, intermediate size 8192,
SILU, RMSNorm epsilon `1e-5`, vocabulary 200064, and 131072 maximum context.
It declares original context 4096, `partial_rotary_factor=0.75`, therefore 96
rotary dimensions, `rope_theta=10000`, LongRoPE short/long factors, and a
262144 sliding-window field. `tie_word_embeddings=true` is confirmed by the
config; the model index has `model.embed_tokens.weight` but no `lm_head.weight`.

The pinned `modeling_phi3.py` confirms QKV concatenation as Q (3072 output
features), K (1024), V (1024), KV repetition from 8 to 24 heads, rotary
application to only the rotary prefix, causal attention, KV cache update, and
the gated MLP `SILU(gate) * up` followed by `down`.

Layer-0 safetensors headers confirm BF16 shapes:

| Tensor | Shape | Shard |
| --- | --- | --- |
| `model.embed_tokens.weight` | `200064x3072` | 1 |
| `model.layers.0.self_attn.qkv_proj.weight` | `5120x3072` | 1 |
| `model.layers.0.self_attn.o_proj.weight` | `3072x3072` | 1 |
| `model.layers.0.mlp.gate_up_proj.weight` | `16384x3072` | 1 |
| `model.layers.0.mlp.down_proj.weight` | `3072x8192` | 1 |

## 5. Existing Qwen P0 Representation

The actual P0 implementation is a Python importer producing JSON dictionaries
with `tensor_bindings`, `program`, `state_schema`, and `portable_boundary`.
Bindings have semantic names, TensorDirectory semantic keys, shapes, a coarse
representation contract, and source provenance. Programs have string kinds,
inputs, outputs, weights, and attributes. State entries have refs, kinds,
shapes, and transition strings. The Qwen builder hard-requires
`general.architecture == qwen35moe`, emits attention plus SSM plus MoE, and its
validator requires a `moe.layer0` dynamic alternative.

There are no formal Rust portable program/state types yet; Rust `graph.rs` is a
separate scheduler seed with one output per operation and no typed attributes.

## 6. Compatibility Attempt Without Changes

| Phi semantic requirement | Existing P0 representation | Fits unchanged? | Problem |
| --- | --- | ---: | --- |
| Dense attention | `program.kind=attention` | Yes | Attributes need extension for position details |
| GQA 24/8 | `heads`, `kv_heads` | Yes | Geometry is representable |
| Fused QKV | one binding per semantic role | No | Three roles share one storage tensor and need slices |
| Partial rotary | `position_source` only | No | Rotary dimension and prefix semantics absent |
| LongRoPE | no position transform schema | No | Scaling vectors and switch condition absent |
| KV state | generic-looking `kv_cache` entry | Partly | Key/value axes, scope, access, reset, append/read are underspecified |
| Dense gated MLP | no Qwen MLP program | By composition | Needs generic view/split and composed operations |
| Fused gate/up | one binding per role | No | Gate/up are slices of one physical tensor |
| Tied LM head | no alias relation | No | Cannot identify two semantic roles sharing storage |
| Optional SSM/MoE | Qwen builder/validator | No | Builder requires Qwen components; neutral descriptor need not |
| RMSNorm | `attn_norm` naming/value | Partly | Role vocabulary is importer-shaped, not formal |
| GPT2 BPE tokenizer | current profile | Partly | GPT2 mechanics fit; Phi special/added token policy needs audit |

## 7. Qwen Assumptions Exposed

| Assumption | Classification |
| --- | --- |
| Qwen builder requires `qwen35moe` | ACCIDENTAL_QWEN_COUPLING in importer |
| Qwen validator requires MoE alternatives | ACCIDENTAL_QWEN_COUPLING in validator |
| SSM and MoE are present in the representative program | QWEN_ARTIFACT_VALUE, not a portable constraint |
| One semantic binding identifies one physical tensor | ACCIDENTAL_QWEN_COUPLING exposed by fused Phi tensors |
| No alias/shared storage relation | REAL_GENERIC_MISSING_SEMANTIC |
| `heads`, `kv_heads`, and `head_dimension` are generic | REAL_GENERIC_CONSTRAINT |
| Attention position is only a source reference | ACCIDENTAL_QWEN_COUPLING / missing generic position semantics |
| Current expert top-k fields are optional routing attributes | REAL_GENERIC optional capability, not required for dense Phi |
| Layer numbers in IDs are importer scope, not runtime name parsing | QWEN_ARTIFACT_VALUE; acceptable importer provenance |

No vendor switch, Phi operation kind, or GGML semantic authority was added.

## 8. Tensor Binding Comparison

Qwen bindings map separate source tensors to semantic roles. Phi requires one
physical `qkv_proj` TensorRef to expose query/key/value semantic views and one
physical `gate_up_proj` TensorRef to expose gate/up views. This is a generic
`TensorView`/slice relation, not a Phi-specific role.

The Phi prototype also maps `lm_head.weight` to the embedding storage through a
generic `same_storage` alias. It does not duplicate tied bytes.

## 9. Attention Comparison

Both models fit one generic attention operation with Q/K/V projections, output
projection, causal masking, head geometry, and KV state. Qwen uses 16/2/256 in
the audited region; Phi uses 24/8/128. GQA is therefore a geometry attribute,
not a family operation.

## 10. Position/RoPE Semantics

Qwen P0 only records `position_source` and `causal`. Phi proves that a generic
position-transform attribute is required: 96 of 128 dimensions rotate, with
LongRoPE short/long factors selected around original context 4096. Frequency
tables remain derived runtime state; theta, rotary dimensions, factor vectors,
and switch semantics are persistent semantic truth.

The configured 262144 window is persisted as a mask attribute. Within Phi's
131072 maximum context it does not impose a smaller effective window, but it is
not silently discarded because it is source configuration truth.

## 11. KV State

Phi requires per-layer key and value state, sequence scope, read/write access,
position-indexed append, range reads, and sequence reset. The existing Qwen
state vocabulary has `kv_cache`, but not enough structured fields to validate
these semantics. The prototype uses generic `kv_key` and `kv_value` StateRefs;
cache contents and backend cache layout remain excluded.

Prefill/decode is not made a separate persistent operation: the same append/read
state transition is sufficient, while schedule and cache implementation remain
runtime-derived.

## 12. Alias / Weight Tying

Phi config and model index establish tied embeddings: `tie_word_embeddings` is
true and no independent LM-head tensor is indexed. The prototype records one
generic alias relation from `lm_head.weight` to `embedding.token` with
`same_storage`. This is persistent semantic truth; backend views are not stored.

## 13. Dense MLP

Phi's MLP is expressible without a new primitive: RMSNorm, MatMul, split, SILU,
Mul, MatMul, and residual add. The fused gate/up tensor is handled by generic
storage views. No `PhiMLP` or vendor-specific operation was introduced.

## 14. Tokenizer Findings

The pinned tokenizer config identifies `GPT2Tokenizer`, `vocab.json`, and
`merges.txt`, with vocabulary size 200064. BOS and EOS insertion are false;
`<|endoftext|>` is also pad/unk at ID 199999. Added special tokens include
assistant/user/system/tool markers, and the chat template is application
profile behavior rather than execution-program semantics.

The current Qwen GPT2-BPE direction is mechanically close, but the current
profile does not fully model added-token matching/normalization/decoder rules.
This is a `PORTABLE_TOKENIZER_SEMANTIC_GAP`, not required to block the program
P0.1 prototype.

## 15. Generic Representation Changes

The prototype adds only generic descriptor fields:

- `view`: axis/start/length mapping semantic bindings to shared storage.
- `aliases`: semantic role to shared storage relation.
- `position_schema`: rotary dimensions, scaling factors, context switch, and
  transform identity.
- Structured KV state fields for scope, lifetime, access, reset, and append/read.
- `composed_dense_mlp` composition, using existing generic operation vocabulary.

These fields are sidecar-level research descriptors, not vBuf v0.6 changes.

## 16. Rejected Phi-Specific Designs

- `PhiAttention`, `PhiLongRoPE`, `PhiKVCache`, and `PhiMLP` were not added.
- No model-family switch was added to the runtime.
- No tensor-name parsing was added to execution; names occur only in importer
  provenance.
- No GGML or safetensors type/offset is used as portable semantic identity.
- No dummy SSM or MoE nodes are emitted for the dense layer.

## 17. Qwen vs Phi Capability Matrix

| Semantic concept | Qwen3.6 P0 | Phi-4-mini | Same portable primitive? |
| --- | --- | --- | ---: |
| Attention | 16/2/256 | 24/8/128 | Yes |
| GQA/MQA | GQA | GQA | Yes |
| Position transform | source position only in P0 | partial LongRoPE | After generic refinement |
| KV state | coarse KV entry | per-layer K/V append/read | After state refinement |
| SSM state | present | not applicable | Optional StateRef |
| MoE routing | dynamic alternatives | not applicable | Optional dynamic set |
| Dense MLP | not represented in Qwen sample | gated SILU composition | Generic composition |
| Dynamic alternatives | present | not applicable | Optional |
| Tensor aliasing | absent in P0 | required | After generic alias relation |
| Weight tying | not represented | embedding/LM-head tied | Generic alias relation |
| Tokenizer pipeline | GPT2/Qwen2 profile | GPT2 BPE plus added tokens | Partial shared profile |

## 18. Abstraction Churn

Counts are for the portable descriptor vocabulary, excluding importer scripts
and source-specific evidence:

```text
PORTABLE_TYPES_BEFORE: 4 descriptor categories (binding/program/state/alternatives), 0 formal types
PORTABLE_TYPES_AFTER: 4 descriptor categories, with generic view/alias/position/state attributes
NEW_GENERIC_TYPES: 0
MODIFIED_GENERIC_TYPES: 0 formal types; descriptor binding/state refinements only
QWEN_SPECIFIC_TYPES_REMOVED: 0 formal types; Qwen builder/validator coupling isolated, not generalized
PHI_SPECIFIC_TYPES_ADDED: 0
NEW_GENERIC_VALUES: 4 (same_storage, storage view, rotary_position_transform, kv_key/kv_value)
NEW_GENERIC_ATTRIBUTES: 13 grouped fields (view, alias relation, rotary parameters, KV lifecycle fields)
NEW_GENERIC_PRIMITIVES: 0
TYPE_REFINEMENTS: 1 (semantic binding may reference shared storage with a view)
VALIDATOR_REFINEMENTS: 1 (view, alias, position, and optional-region validation in Phi prototype)
QWEN_COUPLING_REMOVAL: 1 importer/validator requirement identified, not changed in Qwen builder
```

The practical churn is small and semantic. It did not create a Phi parallel
type family.

## 19. Neutrality Assessment

`ABSTRACTION_SURVIVES_WITH_GENERIC_REFINEMENTS`. Phi exposed real missing
cross-model dimensions: aliases, storage views, richer position transforms, and
structured KV lifecycle. It did not require a Phi-specific operation or runtime
branch. The Qwen builder remains intentionally Qwen-specific importer code;
that is acceptable, but it must not be mistaken for the portable schema.

## 20. Lowering Readiness

`ONE_MORE_MODEL_FAMILY_AUDIT_FIRST`. The semantic vocabulary survived Phi, but
the current prototype is still JSON-sidecar research code, the Rust graph seed
does not carry the new attributes, and tokenizer semantics remain narrow. A
PoC22 lowering experiment now would test an implementation gap rather than
portable maturity.

## 21. Recommended Next Step

Audit **Gemma 3** next, specifically its multimodal/text-only boundary,
sliding-window/global attention schedule, and tokenizer/added-token semantics.
Do not execute it. After that audit, define a shared typed sidecar validator
before any PoC22 lowering.

## Validation Boundary

The generated prototype is
`research/results/vbuf-import-boundary-audit/phi4-mini-typed-program.json`.
The builder self-tests fused storage views, tied aliases, optional SSM/MoE
absence, and partial rotary attributes without loading weights.
