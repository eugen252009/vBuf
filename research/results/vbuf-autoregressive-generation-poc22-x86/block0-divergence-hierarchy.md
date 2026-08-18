# Block-0 Divergence Hierarchy

## Divergence 1

`ATTENTION_PRECISION_POLICY`

The canonical pinned llama path uses F16 KV tensors. The temporary F32-KV
diagnostic restores attention context and `attn_out` parity. This remains a
diagnostic-only oracle configuration.

## Divergence 2

`MODEL_PLAN_DATAFLOW`

Before the fix, block 0 passed raw attention output into the dense FFN. After
the fix, the actual RMSNorm consumer receives the post-attention residual
`attention_output + residual`, hash `fd017832a837be21`.

RMSNorm parity is restored exactly without changing RMSNorm, weights,
materialization, or generic runtime code. Causality is confirmed.

Historical POC21/complete-block qualification used the same attention-output
boundary for its runtime and reference dense FFN inputs, so native-vs-native
parity did not expose this missing residual composition.

## Divergence 3

The first failed dense-FFN boundary is `DOWN_PROJECTION`:

```text
llama: 22960c20497db497
vBuf:  a5e4a18f6771bf94

## Intervention #3

The strict vBuf CPU_REPACK reproduction was gate-only and did not execute the
specialized kernel. The first strict contract failure is buffer selection:
vBuf's diagnostic path directly forces `CPU_REPACK` for the named down
operation rather than reproducing llama's complete `GGML_OP_MUL_MAT` selection
predicates. The first runtime allocation check also fails: vBuf requests
`12607488` bytes, while pinned llama requested `2711642112` bytes. Therefore
`set_tensor`, packed-storage comparison, Q8_0 preparation, workspace comparison,
and kernel execution were not reached by the corrected gate.

`INTERVENTION_3_VALID: NO`. This is a backend buffer-contract result, not
evidence for or against the tiny raw-vs-llama numerical difference.

## Allocation-Scope Audit

The previous 2.7-GB-versus-12.6-MB allocation comparison is superseded as an
`INVALID_CROSS_SCOPE_COMPARISON`. Pinned llama's 2,711,642,112-byte request is
one shared `CPU_REPACK` model-load buffer containing 27 tensors: block 0's
`ffn_down.weight` followed by 26 `ffn_down_exps.weight` tensors. The block-0
tensor starts at offset zero and occupies exactly 12,607,488 bytes. The total
is the exact sum of all 27 tensor spans, with no observed inter-tensor padding.

No vBuf allocation, `set_tensor`, or specialized kernel execution was
performed during this scope audit.

## Tensor-Local Intervention #3 Attempt 2

The strict tensor-local contract gate passed without reproducing the shared
2.7-GB model buffer. A standalone 12,607,488-byte CPU_REPACK buffer reproduced
the llama packed hash `3d8b9543543e6e57`, the Q8_0 operand hash
`d2fd48314bc644ab`, the two-thread `11756`-byte CPU work plan, and the
`n=10944,nr=1,nc=256,bs=2048` geometry. The specialized kernel returned finite
values.

The earlier apparent mismatch was a hash-domain mistake. `822669b42e16cde2`
is the vBuf byte-wise FNV audit hash, while `22960c20497db497` is the
element-wise float-bit hash used by the llama evidence. Recomputing the vBuf
output with the llama hash domain gives `22960c20497db497`; the full output and
the first 256-output tile are bitwise identical.

Therefore:

```text
TENSOR_LOCAL_BACKEND_EXECUTION_MATERIALIZATION_VALID: YES
SHARED_LLAMA_REPACK_BUFFER_REQUIRED: NO
INTERVENTION_3_ATTEMPT_2_VALID: YES
DIVERGENCE_3_CAUSALITY_CONFIRMED: YES
```

The kernel-entry tuple and AVX2 LUT GEMV variant also match. CPU_REPACK
materialization is therefore causally responsible for replacing the raw-vBuf
down path with llama-compatible output; the shared allocation policy is not
required.
max_abs: 7.45058e-08
mean_abs: 6.11762e-09
```

Up, gate, and SwiGLU are bitwise equal. Pinned llama logs show the dense down
weight is repacked as `iq4_nl_8x8`; vBuf uses the raw IQ4_NL descriptor. The
opt-in vBuf repack experiment reached the `iq4_nl_8x8` trait but produced NaN,
so the intervention was invalid and causality remains unproven. No fix has
been applied.
