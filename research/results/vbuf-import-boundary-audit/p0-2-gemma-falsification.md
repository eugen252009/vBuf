# P0.2 Gemma 3 Final Semantic Falsification

## 1. Objective

Use a text-capable Gemma 3 checkpoint as the final descriptive stress case
before execution lowering. No Gemma weights were downloaded or executed.

## 2. Artifact And Evidence

- Model: `google/gemma-3-270m-it`
- Official Hub revision: `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`
- Official config file manifest OID: `48bf8ed00bfb682ff0d4713914e9934c85caae4f`
- Architecture: `Gemma3ForCausalLM`, `gemma3_text`
- Text-only selected path: `Gemma3ForCausalLM`; Gemma multimodal wrappers are
  outside this selected program
- Parameters: 268,098,176 BF16
- Layers/hidden/intermediate: 18 / 640 / 2048
- Attention: 4 query heads, 1 KV head, head dimension 256
- Context: 32768; local window 512
- Attention pattern: layers 0-4 local, layer 5 global, layers 6-10 local,
  layer 11 global, layers 12-16 local, layer 17 global
- Position bases: local 10000, global 1000000
- Tokenizer: SentencePiece `tokenizer.model`, vocabulary 262144
- Weight tying: enabled; LM head shares embedding storage

The official repository is manually gated, so direct config/tokenizer payload
access returned HTTP 401. The official API/file manifest established the pinned
repository, revision, architecture, parameter count, and file identity. Exact
text configuration values were cross-checked against the public
`unsloth/gemma-3-270m-it` derivative and the pinned public Transformers Gemma3
implementation. This access limitation is recorded rather than presented as a
direct official payload read.

## 3. Verified Semantics

The public Gemma3 implementation confirms RMSNorm with `1 + weight`
parameterization, query/key head normalization, gated GELU MLP, causal
attention, local/full layer selection, distinct local/global RoPE bases, and
hybrid KV cache behavior. Attention softcapping and final-logit softcapping are
null for this checkpoint. The multimodal implementation has a separate vision
tower/projector and image-token mask path; neither is required by the selected
text-only model path.

## 4. Existing Representation Compatibility

| Gemma requirement | Existing Qwen/Phi representation | Fits unchanged? | Gap |
| --- | --- | ---: | --- |
| Dense attention | generic `attention` operation | Yes | Per-layer scope must be populated |
| GQA/MQA 4/1 | head geometry attributes | Yes | None |
| Local/global pattern | attention `mask_window` and position schema | Partly | Need explicit per-layer scope/pattern |
| Causal mask | `causal` attribute | Yes | None |
| Local window 512 | generic window attribute | Yes | None |
| Local/global RoPE bases | position transform schema | Yes | Two importer-assigned transform refs |
| Query/key norms | generic Norm composition | Yes | None |
| Gemma `1 + weight` norm | generic norm operation | Partly | Add `parameterization` attribute |
| Gated GELU MLP | composed MLP | Yes | None |
| Hybrid KV state | generic key/value StateRefs | Yes | State variants carry scope/window |
| Shared embedding/LM head | generic alias relation | Yes | None |
| SentencePiece tokenizer | tokenizer semantic concept | No | Current qualified profile is GPT2/Qwen2-only |
| Text/multimodal boundary | importer/profile boundary | Yes | Vision stays outside text program |

## 5. Classification

- Per-layer local/global attention scope: `NEW_GENERIC_ATTRIBUTE`.
- Norm `one_plus_weight` parameterization: `NEW_GENERIC_ATTRIBUTE`.
- Local/full state window variant: `NEW_GENERIC_VALUE` using existing state
  lifecycle fields.
- SentencePiece tokenizer: `MODEL_BOUNDARY_REFINEMENT`; required for a full
  tokenizer import, not for the execution-program gate.
- Vision adapter: `SUPPORTED_AS_IS` as an external multimodal/application
  boundary; not imported into the selected text program.
- No new primitive and no model-specific operation were required.

## 6. Gate 1

`GEMMA_FITS_WITH_GENERIC_REFINEMENTS`.

The pattern is stored as imported semantic truth (`attention_pattern`) rather
than derived by a runtime Gemma formula. The backend derives mask buffers and
frequency tables. Qwen and Phi remain valid because the new attributes are
optional and their existing descriptors do not require a layer pattern.

Generated prototype: `gemma3-typed-program.json`.
