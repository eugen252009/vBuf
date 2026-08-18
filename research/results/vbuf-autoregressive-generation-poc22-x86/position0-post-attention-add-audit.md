# Position-0 Post-Attention ADD Audit

## Scope

This audit compares only the two operands and the output of block-0
`ggml_add(cur, inpSA)` under the pinned F32-KV diagnostic. No FFN parity or
later boundary is qualified.

```text
REFERENCE_COMMIT: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
DIAGNOSTIC_MODE: F32_KV
KV_PRECISION: F32
HASH_ALGORITHM: FNV-1a
HASH_DOMAIN: raw IEEE-754 F32 bit patterns in element order
ELEMENT_TYPE: F32
```

## Result

```text
LLAMA_CUR_HASH: c415092f31bd2d54
VBUF_CUR_HASH: c415092f31bd2d54
CUR_HASH_PARITY: PASS

LLAMA_INPSA_HASH: 69c5d6519b190f83
VBUF_INPSA_HASH: 69c5d6519b190f83
INPSA_HASH_PARITY: PASS

LLAMA_ADD_OUTPUT_HASH: fd017832a837be21
VBUF_ADD_OUTPUT_HASH: fd017832a837be21
ADD_OUTPUT_HASH_PARITY: PASS
```

The exact F32-KV llama capture used temporary aliases at the ADD site:
`attn_out-0` for `cur` and `Vcur-0` for `inpSA`. The source expression remains
`ggml_add(cur, inpSA)` at `src/models/deepseek2.cpp:648-650`.

The previously recorded `3d607311e32ecf27` value came from the default F16
trace. It is superseded mixed-mode evidence and is not the F32-KV
`ffn_inp-0` value.

## Operand Contract

Both paths use contiguous F32 vectors with:

```text
ne:     [2048, 1, 1, 1]
nb:     [4, 8192, 8192, 8192]
nbytes: 8192
```

Llama uses CPU `GGML_OP_ADD` with the non-quantized F32 binary add path.
vBuf's corresponding bounded path is the F32 loop in
`integrations/ggml/tools/multi_layer_poc16.cpp:505-508`, where
`actual_attention.output[i] + actual.values[i]` is stored into the new
contiguous output vector. The descriptor and execution contracts match for
this operation.

The pinned callback does not expose stable model-buffer-relative data
offsets; both tensors are CPU-host F32 data and their pointer-relative offset
is therefore recorded as `NOT_EXPOSED`, not inferred.

## Classification

```text
LLAMA_ADD_OPERANDS_CAPTURED: YES
VBUF_ADD_OPERANDS_CAPTURED: YES
CUR_DESCRIPTOR_PARITY: PASS
INPSA_DESCRIPTOR_PARITY: PASS
HASH_DOMAIN_PARITY: PASS
ADD_DESCRIPTOR_CONTRACT_PARITY: PASS
ADD_EXECUTION_PATH_PARITY: PASS
ADD_DST_PARITY: PASS
PRE_EXECUTION_ADD_GATE: PASS
ADD_VS_LLAMA_MAX_ABS: 0
ADD_VS_LLAMA_MAX_REL: 0
ADD_VS_LLAMA_MEAN_ABS: 0
ADD_FIRST_DIFF_INDEX: NONE
POST_ATTENTION_ADD_DIVERGENCE: NO
PREVIOUS_3D607_EVIDENCE: SUPERSEDED
PREVIOUS_3D607_MODE: DEFAULT_F16_KV
QUALIFIED_COMPARISON_MODE: F32_KV
MIXED_MODE_COMPARISON_VALID: NO
DIVERGENCE_4_CONFIRMED: NO
DIVERGENCE_4_STATUS: SUPERSEDED_MIXED_MODE_EVIDENCE
DIVERGENCE_4_CAUSE: CROSS_CONFIGURATION_EVIDENCE_MIXUP
SCALAR_REFERENCE_USED: NO
DIVERGENCE_4_ROOT_CAUSE_CLASS: CROSS_CONFIGURATION_EVIDENCE_MIXUP
DIVERGENCE_4_CAUSALITY_CONFIRMED: NO
FIRST_TRUE_EXTERNAL_DIVERGENCE_BOUNDARY: NONE_YET_AT_ADD
PREVIOUS_VBUF_RESIDUAL_FIX_REVERTED: NO
DOWNSTREAM_FFN_PARITY_CLAIMS: NO
```

The ADD is not the source of a divergence. The prior vBuf residual fix remains
valid. The smallest next experiment, if needed, is a new boundary capture
using the same complete provenance tuple; no downstream parity should be run
without that configuration discipline.
