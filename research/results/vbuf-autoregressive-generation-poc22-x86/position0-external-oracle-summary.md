# Position-0 External Oracle Summary

## Qualified Scope

```text
TOKEN: 0
POSITION: 0
PINNED_LLAMA_COMMIT: 4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c
DIAGNOSTIC_MODE: F32_KV
KV_PRECISION: F32
HASH_ALGORITHM: FNV-1a
HASH_DOMAIN: raw IEEE-754 F32 bit patterns in element order
ELEMENT_TYPE: F32
```

The bounded qualification is authoritative through the post-attention ADD:

```text
LLAMA_CUR_HASH: c415092f31bd2d54
VBUF_CUR_HASH: c415092f31bd2d54
CUR_HASH_PARITY: PASS

LLAMA_INPSA_HASH: 69c5d6519b190f83
VBUF_INPSA_HASH: 69c5d6519b190f83
INPSA_HASH_PARITY: PASS

LLAMA_FFN_INP_F32_HASH: fd017832a837be21
VBUF_FFN_INP_F32_HASH: fd017832a837be21
POSITION0_POST_ATTENTION_ADD_PARITY: PASS
POSITION0_FFN_INPUT_PARITY: PASS
ADD_MAX_ABS: 0
ADD_MAX_REL: 0
ADD_MEAN_ABS: 0
ADD_FIRST_DIFF_INDEX: NONE
```

No downstream FFN, logits, or later-position parity is claimed by this
bounded ADD audit.

## Mixed-Mode Correction

The earlier chronology was:

1. A default llama trace, using the default F16-KV policy, produced
   `3d607311e32ecf27` for its captured `ffn_inp-0`.
2. F32-KV was selected as the diagnostic parity mode because of the proven
   attention precision policy.
3. The default-F16 `ffn_inp` hash was accidentally carried into the F32-KV
   comparison against vBuf's `fd017832a837be21`.
4. Direct F32-KV operand instrumentation captured matching `cur`, matching
   `inpSA`, and bitwise matching ADD output.
5. The qualified F32-KV llama `ffn_inp-0` is `fd017832a837be21`.

The Default-F16 behavior remains a legitimate llama execution policy. The
invalid result was the cross-configuration comparison, not a llama, ggml,
vBuf ADD, or F32-KV defect.

```text
PREVIOUS_3D607_EVIDENCE: SUPERSEDED
PREVIOUS_3D607_MODE: DEFAULT_F16_KV
QUALIFIED_COMPARISON_MODE: F32_KV
MIXED_MODE_COMPARISON_VALID: NO
MIXED_MODE_EVIDENCE_CORRECTED: YES
DIVERGENCE_4_CONFIRMED: NO
DIVERGENCE_4_STATUS: SUPERSEDED_MIXED_MODE_EVIDENCE
DIVERGENCE_4_CAUSE: CROSS_CONFIGURATION_EVIDENCE_MIXUP
```

Boundary hashes are comparable only when model/artifact identity, reference
commit, diagnostic mode, KV precision, relevant backend path, tensor semantic
boundary, element type, hash algorithm, and hash domain all match. A hash
without configuration provenance must not be promoted to oracle evidence.

## Established History

```text
DIVERGENCE_1: ATTENTION_PRECISION_POLICY
DIVERGENCE_1_CAUSALITY_CONFIRMED: YES

DIVERGENCE_2: VBUF_MODEL_PLAN_DATAFLOW
DIVERGENCE_2_CAUSALITY_CONFIRMED: YES
PREVIOUS_VBUF_RESIDUAL_FIX_REVERTED: NO

DIVERGENCE_3: RAW_IQ4_NL_VS_CPU_REPACK_KERNEL_PATH
DIVERGENCE_3_CAUSALITY_CONFIRMED: YES
```

The only confirmed vBuf-native defect in this investigation remains the
already-fixed model-plan/dataflow issue from Divergence #2. The CPU_REPACK
finding remains an execution-representation/backend-path distinction, not a
persistent vBuf-format defect.

```text
PERSISTENT_VBUF_FORMAT_DEFECT_FOUND: NO
VBUF_SOURCE_OR_RANGE_DEFECT_FOUND: NO
VBUF_RESIDENCY_DEFECT_FOUND: NO
VBUF_MATERIALIZATION_DEFECT_FOUND: NO
GENERIC_RUNTIME_ARCHITECTURE_CHANGE_REQUIRED: NO
POSITION0_EXTERNAL_ORACLE: PASS_THROUGH_POST_ATTENTION_ADD_DOWNSTREAM_UNQUALIFIED
POSITION0_LOGITS_PARITY: NOT_REQUALIFIED
POSITION0_NEXT_TOKEN_PARITY: NOT_REQUALIFIED
RUNTIME_CODE_CHANGED: NO
VBUF_FORMAT_CHANGED: NO
LLAMA_PIN_CHANGED: NO
MODEL_IS_WORKING_TAG_PRESERVED: YES
COMMIT_CREATED: NO
PUSH_PERFORMED: NO
```
