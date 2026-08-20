# Step 31F First-Decode Failure Attribution

Date: 2026-08-20
Starting commit: `9c1f7f0084a0094018884dae8f2d82e145d5d3ca`
Status: **DIAGNOSTICALLY QUALIFIED; ROOT CAUSE UNRESOLVED**

## Objective

Surface the concrete failure behind the Step 31E larger-cap first-decode
failure without changing the residency policy, source behavior, backend
selection, or inference semantics.

## Code Audit

The failure path was:

```text
TensorDependencyExecutor::execute()
    -> RunResult
    -> run_dense_layer/run_layer
    -> SequenceRun
    -> DirectSession::run_step()
    -> JNI GEN_FAIL
```

Before this step, executor detail was local to `execute_expert()` and the
Android boundary reported only `direct runtime execution failed`. The
diagnostic change now:

- prefixes adapter failures with the executing tensor-wave operation;
- preserves `RunResult.detail` through dense and MoE layer helpers;
- preserves the detail through the sequence and Android JNI boundary;
- reports final tensor-wave completion invariant values;
- reports the failing input name, rank, dimensions, and payload size for
  descriptor-construction failures.

No allocation, replacement, materialization, source, or tensor interpretation
behavior was changed.

## Fixed Physical Reproduction

```text
Device: Pixel 7 Pro, Android 17, arm64-v8a
Model: DeepSeek-V2-Lite IQ2_XXS
Residency cap: 536870912 bytes (512 MiB)
Runtime mode: NORMAL_INFERENCE
Prompt: Explain the purpose of bounded generation
Prompt tokens: 7
Prefill: batched
Decode: first token attempt
Remote requests/bytes: 0/0
```

The run reproduced the Step 31E 512 MiB control locality and residency
numbers:

```text
Prefill: 67419 ms
Consumer requests: 1490
Consumer requested bytes: 1565833472
Local source bytes: 1565833472
Residency evictions: 1035
Materializations/reacquisitions: 1490/308
Peak resident bytes: 536003584
```

The first decode failed with:

```text
GEN_FAIL direct runtime execution failed: expert_down_matmul: \
blk.1.ffn_down_shexp.weight: tensor dimensions must be non-zero signed int64 \
values rank=2 dims=12970367413557264240,0 payload_bytes=1486848
```

The failure occurs in `block 7` while constructing the `expert_down_matmul`
input descriptor for `blk.1.ffn_down_shexp.weight`. The descriptor has an
invalid zero dimension and an implausibly large first dimension. The source
was fully local and no Android process kill or explicit OOM was observed.

## Interpretation

The result is **descriptor-corruption/lifetime evidence, root cause
unresolved**. It is not evidence that the 512 MiB cap is an OOM threshold, and
it is not sufficient to attribute the failure to a GGML allocation failure.
The malformed `VbufTensorView` geometry should be investigated next at the
metadata alias, tensor-view lifetime, and residency/materialization ownership
boundaries. No workaround or optimization is authorized by this result.

## Verification

- Android `assembleDebug` passed with the local GGML source override and the
  512 MiB budget property.
- The fixed physical run reproduced the failure and surfaced the descriptor.
- The focused host `vbuf_tensor_wave_dependency_contract` passed in the
  offline host build.
- The host multi-layer target compiled through its C++ object; linking the
  executable against the Android `arm64-v8a` Rust library was intentionally not
  possible on the x86_64 host.

## Scope Boundary

```text
RESIDENCY_POLICY_CHANGED: NO
RESIDENCY_ALGORITHM_CHANGED: NO
SOURCE_BEHAVIOR_CHANGED: NO
MATERIALIZATION_BEHAVIOR_CHANGED: NO
BACKEND_CHANGED: NO
INFERENCE_SEMANTICS_CHANGED: NO
DIAGNOSTIC_PROPAGATION_ADDED: YES
ROOT_CAUSE_FIXED: NO
```
