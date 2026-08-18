# Position-0 External Oracle Qualification

## Scope

This is a bounded external-oracle qualification for token `0` at position `0`.
It compares the vBuf path with pinned llama.cpp and stops at the first new
external divergence. It does not change runtime behavior, apply fixes, or
qualify later boundaries.

## Inputs

- Pinned llama.cpp commit: `4c1a0af40d88c7fbb3b15c85bf2e8016d1d5b64c`
- Diagnostic mode: `F32_KV`
- Token: `0`
- Position: `0`
- Hashes: FNV-1a over raw IEEE-754 F32 bit patterns in tensor order

## Qualified Convergence

| Boundary | Result |
|---|---|
| Token-0 embedding | PASS |
| Attention Q/K/V path | PASS |
| Attention context | PASS |
| Attention output | PASS, `c415092f31bd2d54` |

The F32-KV diagnostic correction restored attention-output parity. The
qualification converges from embedding through attention output.

## Historical Mixed-Mode Record

The previous `ffn_inp-0` divergence record is superseded for the F32-KV
qualification. The value `3d607311e32ecf27` came from the default F16-KV
trace (`DEFAULT_F16_KV_TRACE`) and was incorrectly compared with the F32-KV
vBuf value. It remains historical evidence, but is
`SUPERSEDED_FOR_F32_KV_ORACLE_COMPARISON`.

A strict F32-KV operand audit captured the exact ADD inputs and reproduced the
ADD output.

| Boundary | Pinned llama F32-KV | vBuf corrected path |
|---|---:|---:|
| attention output (`cur`) | `c415092f31bd2d54` | `c415092f31bd2d54` |
| pre-attention residual (`inpSA`) | `69c5d6519b190f83` | `69c5d6519b190f83` |
| ADD output (`ffn_inp-0`) | `fd017832a837be21` | `fd017832a837be21` |

The ADD output is bitwise equal. No later position-0 boundary is qualified;
this audit stops immediately after the ADD.

Pinned-source graph auditing proves that `ffn_inp-0` is the result of the
post-attention residual `ggml_add(cur, inpSA)` and is passed directly to the
block-0 FFN RMSNorm. The hook is attached immediately after construction.
Thus the graph semantic identity is confirmed, but there is no F32-KV
divergence at this boundary.

Status:

```text
POSITION0_EXTERNAL_ORACLE_STAGE: POST_ATTENTION_ADD
FIRST_EXTERNAL_DIVERGENCE: NONE_YET_AT_ADD
FIRST_EXTERNAL_DIVERGENCE_CAUSE: NONE_AT_POST_ATTENTION_ADD
POSITION0_EXTERNAL_ORACLE: PASS_THROUGH_POST_ATTENTION_ADD_DOWNSTREAM_UNQUALIFIED
COMPARISON_STOPPED_AT_FIRST_DIVERGENCE: NOT_APPLICABLE
LATER_BOUNDARIES_QUALIFIED: NO
FFN_INP_IS_DIRECT_FFN_RMSNORM_OPERAND: YES
ACTUAL_FFN_RMSNORM_INPUT_HASH: fd017832a837be21
POST_ATTENTION_RESIDUAL_HASH: fd017832a837be21
NEW_DIVERGENCE_CONFIRMED: NO
```

## Relationship To Divergence #2

The earlier vBuf model-plan audit proved that the dense FFN must consume the
post-attention residual rather than raw attention output. That correction
produced the vBuf boundary `fd017832a837be21` and fixed the internal
vBuf/reference dataflow bug. The fix remains valid and is not reverted.

The earlier independent trace exposed `3d607311e32ecf27` because it used the
default F16 path. The F32-KV trace exposes the actual `ffn_inp-0` as
`fd017832a837be21`, equal to vBuf. No implementation fix is applied here.

The F32-KV operand and ADD parity now establish pinned-llama semantic parity
at this boundary.

## Current Divergence Hierarchy

No new Divergence #4 is confirmed by this ADD audit:

```text
DIVERGENCE_1: ATTENTION_PRECISION_POLICY
CAUSALITY_CONFIRMED: YES

DIVERGENCE_2: VBUF_MODEL_PLAN_DATAFLOW
CAUSALITY_CONFIRMED: YES

DIVERGENCE_3: RAW_IQ4_NL_VS_CPU_REPACK_KERNEL_PATH
CAUSALITY_CONFIRMED: YES

POSITION0_EXTERNAL_ORACLE: PASS_THROUGH_POST_ATTENTION_ADD_DOWNSTREAM_UNQUALIFIED
FFN_INPUT_BOUNDARY_SEMANTICS: CONFIRMED_ACTUAL_FFN_RMSNORM_OPERAND
FFN_INPUT_BOUNDARY_ROOT_CAUSE_CLASS: NONE_AT_POST_ATTENTION_ADD
PREVIOUS_FFN_INP_0_HASH_CONTEXT: DEFAULT_F16_TRACE_ONLY
PREVIOUS_3D607_EVIDENCE: SUPERSEDED
PREVIOUS_3D607_MODE: DEFAULT_F16_KV
QUALIFIED_COMPARISON_MODE: F32_KV
MIXED_MODE_COMPARISON_VALID: NO
MIXED_MODE_EVIDENCE_CORRECTED: YES
F32_KV_ADD_OUTPUT_PARITY: PASS
```

## Next Smallest Audit

The next smallest research step, if needed, is to qualify the next boundary
with the same complete provenance tuple. Do not reuse any Default-F16 hash in
that comparison.
