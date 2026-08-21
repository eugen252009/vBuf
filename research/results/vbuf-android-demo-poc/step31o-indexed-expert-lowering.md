# Step 31O Indexed Expert Lowering

Date: 2026-08-21
Branch: `vbuf-ml`
Status: **HOST-QUALIFIED LOWERING SEAM; NO PRODUCTION RUNTIME CHANGE**

## Scope

Step 31O implements the smallest backend-neutral seam needed to preserve routed
expert bank/member provenance while allowing either rank-2 fallback execution or
an opt-in indexed-bank backend path. It does not add a grouped operation to the
TensorWave executor, change TensorRef, change persistence or residency, or add
an Android/full-model execution path.

The implementation is:

```text
PersistentTensorRef rank-3 bank
    -> ordered bank/member provenance
    -> rank-2 member views for fallback
    -> full-bank view for an indexed-capable backend
```

The lowerer preserves the bank tensor ID and name, source offsets, expert ID,
TopK rank, selection scores, and merge weights. It never copies or repacks
payload bytes. A null payload is accepted for metadata-only planning; the
existing materializer remains responsible for resolving and leasing bytes.

## Implementation Boundary

- `integrations/ggml/include/vbuf_indexed_expert.h`
- `integrations/ggml/src/vbuf_indexed_expert.cpp`
- `integrations/ggml/tests/indexed_expert_lowering_contract.cpp`
- `integrations/ggml/tests/moe_grouping_microqualification.cpp`

The generic capability is `Rank2Only` or `IndexedBank`; it does not mention
GGML. The opt-in host probe uses the `IndexedBank` plan to call raw
`ggml_mul_mat_id`. The existing TensorWave graph, materializer, lease, and
residency contracts remain unchanged.

## Contract Checks

The native contract test verifies:

- rank-3 bank to rank-2 member geometry and fixed-stride offsets;
- exact TopK rank and expert-ID ordering;
- score and merge-weight preservation;
- rank-2 fallback and indexed-bank capability paths;
- metadata-only plans with no payload pointer;
- invalid expert IDs, non-finite weights, and source-offset overflow fail closed;
- source payload bytes remain unchanged.

Existing TensorWave and residency contracts continue to qualify borrowed
payload lifetime and eviction protection. The new lowerer owns no lease and does
not extend payload lifetime; execution must continue to hold the existing
materializer or model lease around the selected backend operation.

## Host Qualification

Artifact:

```text
SOURCE: /tmp/opencode/deepseek-v2-lite-imat/DeepSeek-V2-Lite.IQ2_XXS.gguf
SOURCE_SHA256: 3b7da33584bebf89afcdbdd2e7a8e3e47e11092971559371f13b510f475e3c0c
EXPERT_COUNT: 64
SELECTED_IDS: [3, 11, 17, 29, 41, 53]
```

The opt-in probe compared six rank-2 `ggml_mul_mat` views produced by the
`Rank2Only` plan with one raw `ggml_mul_mat_id` using the `IndexedBank` plan.
The actual IQ2_XXS gate/up and IQ4_NL down payloads were mapped read-only and
borrowed without repacking. At ten iterations and one CPU thread:

| Case | Separate ms | Grouped ms | Speedup | Max abs diff | Parity |
|---|---:|---:|---:|---:|---|
| Gate/up IQ2_XXS | 1.238 | 1.129 | 1.097x | 0 | PASS |
| Down IQ4_NL | 0.948 | 0.794 | 1.194x | 0 | PASS |

This qualifies the lowering seam, raw host capability, and output parity. It
does not establish a stable optimization, Android support, full-model
scheduling behavior, or grouped materialization/residency economics.

## Verification

```text
vbuf_indexed_expert_lowering_contract: PASS
selected TensorWave/residency/TopK contracts: 4/4 PASS
real-payload host qualification: both cases PARITY=PASS
Android qualification: NOT PERFORMED
```
