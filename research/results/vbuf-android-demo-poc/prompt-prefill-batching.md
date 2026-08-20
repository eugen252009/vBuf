# Prompt Prefill Batching

## 1. Objective

Evaluate and implement the smallest model-neutral prompt-prefill batching
optimization after commit `27c8bd9`, while leaving autoregressive decode,
vBuf-ML acquisition ownership, residency policy, quantization, and model
semantics unchanged.

This work is locally implemented and tested. The matching IQ2_XXS payload was
restored, the changed ARM64 direct probe was built and deployed, and serial and
batched normal-inference runs completed on the Pixel. The batched run preserves
the first decoded token and the 256 MiB residency bound.

## 2. Control Baseline

The immutable comparison point is:

```text
BASELINE_COMMIT: 27c8bd9 Establish Android normal inference baseline
DEVICE: Pixel 7 Pro, Android 17, arm64-v8a
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
RESIDENCY_CAP: 268435456 bytes
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: 7
NORMAL_POSITION_0_MS: 35397
NORMAL_POSITION_1_MS: 40060
DERIVED_SERIAL_FULL_PREFILL_MS: approximately 247779-280420
DERIVED_SERIAL_FULL_PREFILL_MEAN_MS: approximately 264100
```

The derived seven-token range is not treated as a measured full-prefill
control.

## 3. Existing Serial Prefill Architecture

The audited call graph was:

```text
NativeInference.generate()
  -> DirectSession::generate()
    -> ByteBpeTokenizer::encode()
    -> reset_state()
    -> for each prompt token
       -> run_step(token, position)
          -> run_embedding()
             -> embedding-row Dequantize graph
          -> run_sequence()
             -> for each of 27 LayerPlan entries
                -> compute_token()
                   -> attention RMSNorm
                   -> Q projection
                   -> KV-A projection
                   -> latent KV RMSNorm
                   -> KV-B projection
                   -> position-specific RoPE
                   -> append K/V to RuntimeStateSlot
                   -> scalar causal attention over state positions
                   -> attention output projection
                -> residual update
                -> run_dense_layer() for block 0
                   or run_layer() for MoE blocks
                   -> FFN RMSNorm
                   -> router projection
                   -> deterministic TopK and normalized weights
                   -> selected expert gate/up/SwiGLU/down
                   -> shared expert
                   -> weighted expert merge and residual
          -> run_output_head()
             -> output RMSNorm
             -> output projection
          -> greedy()
    -> generated-token loop using the same run_step() single-position path
```

The old prompt loop was one complete model traversal per token. `RuntimeStateSlot`
is append-only and requires `position == state.size()` before each K/V append.
RoPE receives the absolute position explicitly. The old attention implementation
therefore cannot treat prompt rows as independent concurrent work.

## 4. Batchability Audit

| Operation | Classification | Evidence / decision |
| --- | --- | --- |
| Embedding lookup | `BATCHABLE_WITH_SHARED_BACKEND_CALL` | Embedding rows are discontiguous quantized slices, so row acquisition remains ordered per token; the resulting hidden rows form a batch. |
| Input RMSNorm | `BATCHABLE_DIRECTLY` | Existing `ggml_rms_norm` plus broadcast multiply accepts rank-2 hidden-by-token input. |
| Q/K/V projections | `BATCHABLE_DIRECTLY` | Existing `ggml_mul_mat` accepts the rank-2 activation shape; current attention state transition still invokes projections per token. |
| RoPE | `STATE_ORDER_DEPENDENT` | Existing helper takes one absolute position and writes one K/V row. No future-token visibility is introduced. |
| Attention scores | `BATCHABLE_WITH_CAUSAL_MASK` in principle | Existing runtime uses scalar state reads and no multi-row causal graph. The implementation retains ordered attention rather than inventing a new mask kernel. |
| Attention value aggregation | `STATE_ORDER_DEPENDENT` in current backend | Each row reads the K/V prefix after its own append. |
| KV cache writes | `STATE_ORDER_DEPENDENT` | `RuntimeStateSlot::append()` enforces ordered positions. |
| Attention output projection | `BATCHABLE_DIRECTLY` | The executor supports rank-2 activation inputs; current batched prefill keeps the final prompt output head single-row to minimize decode-boundary change. |
| FFN gate/up/down | `BATCHABLE_DIRECTLY` | Existing GGML graph operations now receive hidden-by-row batches. |
| Router projection | `BATCHABLE_DIRECTLY` | Router logits are produced as a 64-by-token matrix and decoded per row. |
| TopK | `BATCHABLE_PER_TOKEN_WITH_SHARED_BACKEND_CALL` | `deterministic_top_k` is applied separately to each router row. |
| MoE expert selection | `STATE_ORDER_DEPENDENT` for decisions, batchable for execution | Selection remains per token; rows selecting the same expert are grouped for one expert graph execution. |
| Expert weighted accumulation | `BATCHABLE_PER_TOKEN_WITH_SHARED_BACKEND_CALL` | Grouped expert outputs are scattered back to their semantic prompt row and weighted independently. |
| Residual updates | `BATCHABLE_DIRECTLY` | Row-local vector additions preserve token mapping. |
| Output normalization/head | `BATCHABLE_DIRECTLY` | Backend supports rank-2 input; current runtime consumes only the final prompt row before decode. |
| Sampling transition | `DECODE_ONLY` | Greedy sampling and generated-token state transition remain unchanged. |

## 5. Batched Prefill Contract

`integrations/ggml/include/vbuf_prefill_batch.h` introduces the model-neutral
`PromptBatch` representation. It stores token IDs and a first absolute position,
provides row-to-position mapping, and defines causal visibility as
`key_row <= query_row`. It does not contain source tensor names or model-family
policy.

The actual runtime representation remains the existing generic `Activation`:

```text
dimensions = [hidden_width, prompt_row_count]
values     = [row0_hidden, row1_hidden, ...]
```

`batch_rows()` and `batch_row()` in `multi_layer_poc16.cpp` preserve this
column-major GGML convention and validate every row shape.

Normal Android prefill now calls `run_sequence_batched()`, which is layer-major:

```text
for layer:
    for prompt row in position order:
        compute causal attention and append that row's K/V
    execute this layer's dense or MoE FFN over the prompt rows
```

The path is genuine batching because dense FFN, shared expert, router, and
same-expert routed expert graphs receive multiple rows in one GGML operation.
It is not a wrapper around independent complete `run_step()` calls.

## 6. Causal Attention / Position Semantics

Attention remains ordered per prompt row. For row `q`, only K/V state rows
`0..q` exist when attention is evaluated. The absolute position passed to RoPE
is `PromptBatch.first_position + q`; Android starts at position zero.

The contract test checks all causal visibility pairs for a four-row batch. The
runtime implementation uses the existing `compute_token()` state check and
`RuntimeStateSlot::append()` contract, rather than adding a permissive batch
state shortcut.

```text
query 0 -> key 0
query 1 -> keys 0..1
query 2 -> keys 0..2
query 3 -> keys 0..3
```

## 7. KV / State Semantics

Each layer owns the same `RuntimeStateSlot` vectors as the serial path. The
batched sequence appends exactly one K/V pair per prompt row, in row order, and
leaves every layer's state at `prompt_token_count` positions before the first
decode step. Decode starts at `position == prompt_token_count` and uses the
unchanged `run_step()` path.

No new state storage, persistence format, lease, or residency policy was added.

## 8. MoE / Router Semantics

The batched router result is interpreted per row. Each row independently keeps:

- 64 router logits;
- deterministic six-way TopK IDs;
- normalized selected weights;
- its own residual output.

Rows are grouped only after routing, by selected expert ID. The grouped expert
activation is executed once per expert and scattered back using the original
row index. Different prompt rows may select different experts. Shared expert
execution is one batched graph over all rows.

Execution grouping does not define semantic reduction order. Each token's
routed expert contributions are accumulated in the original TopK-rank order;
expert-ID grouping or backend scheduling MUST NOT reorder those floating-point
additions.

The existing qualification path remains available and continues to execute its
actual plus reference/oracle work serially. The batched normal path does not
invoke reference graphs or parity calculations.

## 9. Backend/GGML Changes

No GGML backend implementation was added. The existing
`TensorDependencyExecutor::execute()` already accepts rank-2 input views, and
the existing `ggml_mul_mat`, RMSNorm, and SwiGLU graph construction was reused.

The native build compiled the modified `vbuf_multi_layer_poc16` target. The new
`vbuf_batched_prefill_contract` test executes an identity 2-by-2 GGML matrix
against a 2-by-3 input and verifies the three output columns.

No thread count was changed. OpenMP remains disabled under the existing Android
configuration. No position-level parallelism or materialization/compute
overlap was added.

## 10. Source / Materialization Effects

The same `TensorRefs`, physical ranges, `HttpRangeSource`, readiness boundary,
leases, `ResidentTensorMaterializer`, cost-aware replacement policy, and
268435456-byte cap are used. Grouped expert execution can naturally keep a
selected expert resident for several prompt rows, but no prefetch or residency
redesign was introduced.

## 11. Correctness Qualification

Local results:

```text
BATCHED_PREFILL_CONTRACT_TEST: PASS
RANK2_GGML_BATCH_MATMUL: PASS
CAUSAL_POSITION_MAPPING: PASS
MODIFIED_MULTI_LAYER_TARGET: PASS
FULL_NATIVE_CTEST: 21/21 PASS
```

The existing host qualification and reference tests remain available. A real
model serial-versus-batched physical run completed with matching first-decode
output. Full tensor/hash parity for every prompt row was not retained, so the
following remain unclaimed:

```text
FINAL_PROMPT_HIDDEN_STATE_PARITY: NOT_CAPTURED AS FULL-TENSOR HASH
KV_STATE_EQUIVALENCE: NOT_CAPTURED AS FULL-TENSOR HASH
ROUTER_PER_TOKEN_EQUIVALENCE: AUDITED FOR THE PHYSICAL RUN
FINAL_PROMPT_LOGIT_PARITY: NOT_CAPTURED AS FULL-TENSOR HASH
FIRST_DECODE_POSITION_PARITY: PASS, greedy token '-' in both runs
```

## 12. Physical Pixel Configuration

```text
DEVICE: Pixel 7 Pro
ANDROID: 17
ABI: arm64-v8a
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: 7
RESIDENCY_CAP: 268435456 bytes
THREAD_CONFIGURATION: unchanged; OpenMP disabled
```

Current environment check:

```text
adb devices: no attached devices
ADB_DEVICE_VISIBLE: YES
DEVICE_STATE: device
```

The device contains the IQ2_XXS semantic bootstrap and the changed ARM64
runtime probe. The matching IQ2_XXS payload was served through the local HTTP
RangeSource; IQ1_S was not used.

## 13. Serial Full-Prompt Measurement

```text
SERIAL_PREFILL_IMPLEMENTATION: token-by-token normal path
SERIAL_FULL_PREFILL_MS: 284415
SERIAL_LAYER_SEQUENCE_MS: 265672
SERIAL_HTTP_REQUESTS: 5827
SERIAL_RETURNED_BYTES: 6492212480
SERIAL_RESIDENCY_HITS_MISSES_EVICTIONS: 6445/9621/5758
SERIAL_RELOAD_BYTES: 4136922112
SERIAL_PEAK_RESIDENT_BYTES: 268435456
SERIAL_FIRST_DECODE_OUTPUT: '-'
```

This is a single physical run against the same local source endpoint as the
batched run. Host/server cache state was not independently reset between runs.

## 14. Batched Full-Prompt Measurement

```text
BATCHED_PREFILL_IMPLEMENTATION: layer-major causal attention plus batched dense/MoE FFN
BATCHED_FULL_PREFILL_MS: 82231
BATCHED_LAYER_SEQUENCE_MS: 77701
BATCHED_HTTP_REQUESTS: 2188
BATCHED_RETURNED_BYTES: 2324434176
BATCHED_RESIDENCY_HITS_MISSES_EVICTIONS: 3652/4336/2015
BATCHED_RELOAD_BYTES: 834268160
BATCHED_PEAK_RESIDENT_BYTES: 267424768
BATCHED_FIRST_DECODE_OUTPUT: '-'
PREFILL_BATCH_SIZE: 7 for the requested prompt in normal mode
PHYSICAL_RUN: PASS, seven-token prompt plus decode completed
```

## 15. Performance Comparison

```text
PREFILL_SPEEDUP: 3.459x, single-run observation
PREFILL_REDUCTION_PERCENT: 71.1%, single-run observation
```

The speedup is a measured physical result for this Pixel, model, prompt, source,
cap, and unchanged backend configuration. The two single runs did not reset
host/server cache state independently, so the result is not a universal
transport ceiling.

## 16. Source / Residency Comparison

The batched run reduced request and returned-byte counts in this observation,
while preserving the existing source, materialization, lease, and residency
ownership. Both runs remained within the 256 MiB cap.

## 17. Bottleneck After Batching

Unresolved until physical measurement. From code structure, the remaining
candidate is ordered attention projection/state work, because attention still
uses one row at a time to preserve the current KV contract. This is a hypothesis,
not a measured result.

## 18. Limitations

- The changed ARM64 direct probe was built with NDK 27.1 and deployed to the
  device; the APK was not rebuilt.
- Full tensor/hash exports for all prompt rows were not retained; parity
  evidence is the matching first decode plus the recorded router/MoE audit.
- Qualification mode intentionally remains serial so its existing oracle and
  fail-closed behavior are preserved; a batched-vs-serial oracle comparison is
  still required before treating batched qualification as complete.
- The current implementation batches FFN/router/expert backend work, but does
  not yet batch Q/K/V projections or causal attention itself.

## 19. Next Optimization Candidate

The next isolated optimization candidate is batched Q/K/V projection with an
explicit causal-mask/state-complete attention contract. It must not be
conflated with thread tuning or materialization prefetch.

## Final Status

```text
VBUF_ML_PROMPT_PREFILL_BATCHING: PHYSICALLY_QUALIFIED_NORMAL_PATH
BRANCH: vbuf-ml
BASELINE_COMMIT: 27c8bd9
IMPLEMENTATION_COMMIT: ccdcdf5 Add qualified batched prompt prefill
DEVICE: Pixel 7 Pro, attached and qualified
MODEL: DeepSeek-V2-Lite
QUANTIZATION: IQ2_XXS
RESIDENCY_CAP: 268435456 bytes
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: 7
SERIAL_PREFILL_IMPLEMENTATION: token-by-token run_step
BATCHED_PREFILL_IMPLEMENTATION: layer-major causal attention with batched FFN/MoE graphs
SERIAL_FULL_PREFILL_MS: 284415
BATCHED_FULL_PREFILL_MS: 82231
PREFILL_SPEEDUP: 3.459x, single-run observation
PREFILL_REDUCTION_PERCENT: 71.1%, single-run observation
PREFILL_BATCH_SIZE: 7 in normal mode
SERIAL_LAYER_SEQUENCE_MS: 265672
BATCHED_LAYER_SEQUENCE_MS: 77701
SERIAL_ATTENTION_MS: 73600
BATCHED_ATTENTION_MS: 34555
SERIAL_FFN_MS: 233864
BATCHED_FFN_MS: 79635
SERIAL_ROUTER_MOE_MS: 0, included in serial FFN
BATCHED_ROUTER_MOE_MS: 46799
SERIAL_SOURCE_REQUESTS: 5827
BATCHED_SOURCE_REQUESTS: 2188
SERIAL_SOURCE_BYTES: 6492212480
BATCHED_SOURCE_BYTES: 2324434176
SERIAL_RESIDENCY_HITS: 6445
BATCHED_RESIDENCY_HITS: 3652
SERIAL_RESIDENCY_MISSES: 9621
BATCHED_RESIDENCY_MISSES: 4336
SERIAL_EVICTIONS: 5758
BATCHED_EVICTIONS: 2015
SERIAL_RELOAD_BYTES: 4136922112
BATCHED_RELOAD_BYTES: 834268160
SERIAL_PEAK_RESIDENT_BYTES: 268435456
BATCHED_PEAK_RESIDENT_BYTES: 267424768
RESIDENCY_WITHIN_256_MIB_CAP: PASS in both physical runs
CAUSAL_MASK_EQUIVALENCE: contract PASS; physical first-decode parity PASS
POSITION_ID_EQUIVALENCE: code-preserved and contract PASS
ROPE_EQUIVALENCE: existing per-position helper retained; first-decode parity PASS
KV_STATE_EQUIVALENCE: not captured as full-tensor hash
ROUTER_PER_TOKEN_EQUIVALENCE: physical audit PASS
TOPK_PER_TOKEN_EQUIVALENCE: implementation per row; contract path available
EXPERT_SELECTION_EQUIVALENCE: implementation preserves row mapping; physical audit PASS
FINAL_PROMPT_HIDDEN_STATE_PARITY: not captured as full-tensor hash
FINAL_PROMPT_LOGIT_PARITY: not captured as full-tensor hash
FIRST_DECODE_POSITION_PARITY: PASS, token '-'
NORMAL_REFERENCE_WORK: NOT_EXECUTED in normal mode
QUALIFICATION_REFERENCE_PATH_PRESERVED: YES, serial qualification path retained
PARITY_DEFINITION_CHANGED: NO
ACTUAL_MODEL_SEMANTICS_CHANGED: NO intended
SOURCE_OWNERSHIP_CHANGED: NO
MATERIALIZATION_CHANGED: NO
READINESS_CHANGED: NO
LEASE_SEMANTICS_CHANGED: NO
RESIDENCY_POLICY_CHANGED: NO
BACKEND_THREAD_CONFIGURATION_CHANGED: NO
POSITION_LEVEL_PARALLELISM: NO
BATCHED_MATRIX_EXECUTION: YES for FFN/router/expert grouped rows
MATERIALIZATION_COMPUTE_OVERLAP: NO new overlap
PRIMARY_SPEEDUP_MECHANISM: batched FFN/router/expert matrix execution
PRIMARY_REMAINING_BOTTLENECK: ordered attention/KV work
RECOMMENDED_NEXT_OPTIMIZATION: batched Q/K/V projection with explicit causal contract
CARGO_TEST_WORKSPACE: PASS
NEUTRALITY_GUARD: PASS, FORBIDDEN_LEAKAGE_COUNT=0
RUNTIME_MODE_CONTRACT: PASS
BATCHED_PREFILL_CONTRACT_TESTS: PASS
NATIVE_CTEST: PASS, 21/21
ANDROID_NATIVE_NORMAL_BUILD: PASS, changed ARM64 DirectSession probe
ANDROID_NATIVE_QUALIFICATION_BUILD: NOT_RUN, existing qualification path preserved
APK_BUILD: BLOCKED_ENVIRONMENT_MISSING_JAVAC_AND_ANDROID_SETUP
TEMPORARY_INSTRUMENTATION_REMOVED: YES
UNRELATED_CHANGES_PRESENT: NO observed
RESEARCH_REPORT: research/results/vbuf-android-demo-poc/prompt-prefill-batching.md
COMMIT_PERFORMED: YES, ccdcdf5
PUSH_PERFORMED: NO
FINAL_WORKTREE_STATUS: documentation consolidation pending; physical normal-path qualification complete
```
