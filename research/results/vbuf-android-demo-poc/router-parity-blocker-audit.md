# Android Router-Parity Blocker Audit

Date: 2026-08-19
Branch: `vbuf-ml`
HEAD: `c006130`

## 1. Objective

Determine why the direct Android demo's arbitrary-prompt path reports
`router parity failed`, while the previously qualified direct runtime and the
real Gate 2B selected router-prefix pass.

This audit made no production-code changes and did not alter parity behavior.

## 2. Known-Good Controls

The D2.3 Android qualification remains a valid control. It passed external
materialization, readiness, bounded residency, GGML execution, four-token
generation, router parity, logits parity, token parity, and teardown.

Gate 2B remains a valid standalone control. It used real DeepSeek IQ2_XXS
tensors and a deterministic one-hot `2048 x 1` activation. RMSNorm was exact,
router logits were exact, and TopK indices and values passed. The generic path
read no model family, source names, or `blk.n` assumptions.

The controls do not exercise the same activation history as arbitrary prompt
prefill.

## 3. Failure Reproduction

The retained Phase A hardware report records:

```text
DEVICE: Pixel 7 Pro, Android 17, arm64-v8a
MODEL: DeepSeek-V2-Lite IQ2_XXS
PROMPT: Explain the purpose of bounded generation
HTTP_REQUESTS_AT_FAILURE: 2802
RETURNED_BYTES_AT_FAILURE: 2716374336
ERROR: router parity failed
```

The current audit build opens the model and reaches `Ready · direct vBuf-ML
runtime`. The ADB UI interaction could not activate Generate reliably after
prompt entry, so a fresh numerical reproduction was not obtained. The saved
Phase A run contains no token IDs, layer/position, vector values, hashes, or
TopK diagnostics.

An audit-only JNI receiver then invoked the existing `NativeInference.open()`
and `NativeInference.generate()` path directly. `OPEN_OK` was observed, but
the run did not reach the router gate during the bounded observation windows.
After approximately 32 minutes, the process remained alive with stable worker
thread activity, while the HTTP range-server log size stayed unchanged at
`8,785,205` bytes across a five-second sample. No parity, generation, or
crash marker appeared. The live run was not cancelled or restarted.

```text
CURRENT_PHASE: unknown native execution/materialization wait
ELAPSED_TIME: approximately 32 minutes
PROCESS: alive, worker threads active
TRANSPORT_PROGRESS: none over five seconds
RUNTIME_LOG_PROGRESS: none
LIVE_RUN_RESULT: stalled before router diagnostics
```

Later controlled runtime-mode timing superseded the interpretation of that live
observation as an unresolved native/materialization stall. The same direct
runtime was shown to advance through repeated complete serial positions: the
historical qualification-heavy position measured approximately `109.7 s`,
while the controlled qualification and normal baselines measured `73.588 s`
and `35.397 s` respectively. The later evidence explains the apparent
~32-minute run as repeated serial prompt-position execution, not a single
silent wait. The numerical router-divergence audit remains incomplete because
the failing run still has no vector diagnostics.

```text
FAILURE_REPRODUCED: NO, exact numerical reproduction unavailable
FAILURE_POSITION: NOT RECORDED
FAILURE_LAYER: NOT RECORDED
FAILURE_STAGE: router parity gate, from retained report only
```

The app launch itself did not produce a crash. The observed task closures in
this audit were caused by ADB back-key interaction, not an Android exception
or native fatal signal.

## 4. Exact Parity Gate

```text
PARITY_GATE_FILE: integrations/ggml/tools/multi_expert_moe_poc12.cpp
PARITY_GATE_FUNCTION: route_activation()
PARITY_GATE_CONDITION: !parity(logits, reference, "router_score_parity")
PARITY_GATE_TOLERANCE: 1e-4 on Android direct build; 1e-5 otherwise
PARITY_GATE_REFERENCE_SOURCE: reference_scores() scalar column-major dot product
PARITY_GATE_RUNTIME_SOURCE: GGML TensorDependencyExecutor MulMat output
```

The abort occurs at `multi_expert_moe_poc12.cpp:75`, after GGML router
execution and before TopK selection or expert execution.

## 5. Compared Values

The compared values are:

```text
runtime:   floats(runtime.output), produced by GGML router_matmul
reference: reference_scores(materialized_router_payload, activation)
shape:     64 router logits
```

This is not a comparison against a future token, a synthetic selected-expert
fixture, or a separate model loader.

The comparator prints max absolute, max relative, mean absolute error, and
tolerance, but the Android JNI failure converts the exception to the string
`GEN_FAIL router parity failed`. Those printed diagnostics were not retained
for the failing run.

```text
REFERENCE_VECTOR_COUNT: NOT RECORDED FOR FAILING RUN
RUNTIME_VECTOR_COUNT: NOT RECORDED FOR FAILING RUN
MAX_ABSOLUTE_ERROR: NOT RECORDED
MAX_RELATIVE_ERROR: NOT RECORDED
MAX_ERROR_INDEX: NOT RECORDED
REFERENCE_TOPK: NOT REACHED
RUNTIME_TOPK: NOT REACHED
TOPK_IDENTICAL: NOT APPLICABLE
```

## 6. Prompt/Tokenizer Path

The Android app uses `ByteBpeTokenizer` in
`integrations/android-vbuf-chat/app/src/main/cpp/vbuf_android_direct.cpp`.
It reads token pieces, merge pairs, BOS/EOS metadata from the semantic
consumer, applies byte-BPE merges, and feeds the resulting IDs to
`DirectSession::generate()`.

```text
TOKENIZER_IMPLEMENTATION: metadata-backed byte-BPE tokenizer
PROMPT_FORMAT: raw completion text
CHAT_TEMPLATE_APPLIED: NO
BOS_POLICY: consumer add_bos metadata, if present
EOS_POLICY: consumer special EOS metadata, if present; stops generation
PROMPT_TOKEN_IDS_MATCH_EXPECTATION: NOT RECORDED
```

The arbitrary prompt can legitimately produce a different valid token
sequence from the D2.3 seed-token fixture. That difference alone cannot make
the router scalar comparison fail because both sides of `route_activation()`
receive the same `Activation` object for the current call.

## 7. Position/State Path

`DirectSession::generate()` resets all actual and reference KV vectors before
encoding the prompt. It runs every prompt token at positions `0..N-1`, feeds
the returned token into the next step, and increments position once per step.

`run_sequence()` keeps separate actual and reference KV vectors and calls
`compute_token()` with the same position on each side. It updates each side
from its own preceding activation.

```text
POSITION_MATCH: structurally YES; failing-run values NOT RECORDED
STATE_INITIALIZATION_MATCH: structurally YES; both reset before prompt
STATE_RESET_CORRECT: YES for a fresh DirectSession generation
STALE_STATE_OBSERVED: NO evidence
FIRST_DIVERGENCE_PHASE: NOT RECORDED
FIRST_DIVERGENCE_PROMPT_TOKEN_INDEX: NOT RECORDED
FIRST_DIVERGENCE_GENERATED_TOKEN_INDEX: NOT RECORDED
```

The code-level route gate is reached during both prompt prefill and
autoregressive steps. The retained report does not identify which invocation
failed.

## 8. Router Input Identity

At `route_activation()` the runtime GGML graph and scalar reference are called
with the same `Activation & activation` argument. The reference does not
reconstruct or fetch a second hidden state.

```text
REFERENCE_ROUTER_INPUT_COUNT: same activation, expected 2048
RUNTIME_ROUTER_INPUT_COUNT: same activation, expected 2048
REFERENCE_ROUTER_INPUT_HASH: NOT INSTRUMENTED
RUNTIME_ROUTER_INPUT_HASH: NOT INSTRUMENTED
ROUTER_INPUT_IDENTICAL: YES by call-graph aliasing; numerical fingerprint NOT RECORDED
```

This rules out a reference/runtime prompt-input mismatch at the gate itself,
but does not prove that the hidden state is the intended model state. That
earlier-state question requires a successful diagnostic run.

## 9. RMSNorm Comparison

`run_layer()` first computes RMSNorm with epsilon `1e-6`, then uses the runtime
normalized vector as the router activation. Its separate scalar RMSNorm check
uses the same input and materialized norm payload.

```text
INPUT_SHAPE: 2048 x 1
WEIGHT_IDENTITY: same PersistentTensorRef within run_layer
EPSILON: 1e-6
RMSNORM_INPUT_IDENTICAL: same input passed to both calculations by call graph
RMSNORM_OUTPUT_PARITY: NOT RECORDED AT FAILING INVOCATION
```

Gate 2B proves this operation and epsilon are valid for the selected real
fixture; it does not prove the arbitrary hidden input was generated correctly.

## 10. Router MatMul Comparison

The runtime operation is `router_matmul`, with the persistent router tensor as
the GGML RHS operand. The scalar reference computes
`weight[input + input_dim * expert] * activation[input]`.

Gate 2B proves the orientation and tensor geometry for the real IQ2_XXS router
prefix. The Android gate's missing diagnostic values prevent distinguishing
ordinary accumulation drift from a larger backend error for the arbitrary
hidden state.

```text
MATMUL_INPUT_SHAPE: 2048 x 1
ROUTER_OUTPUT_SHAPE: 64
ROUTER_LOGITS_PARITY: NOT DETERMINABLE NUMERICALLY FROM RETAINED RUN
```

## 11. TopK Comparison

TopK is called only after router logit parity succeeds. Therefore the retained
failure occurred before TopK and cannot be classified as a TopK semantic
failure.

```text
TOPK_PARITY: NOT REACHED AT RETAINED FAILURE
```

## 12. Tensor/Payload Identity

The router tensor is resolved from the semantic metadata as
`blk.1.ffn_gate_inp.weight` through the layer plan alias and is passed as one
`PersistentTensorRef`. Both GGML execution and the scalar reference request
the same router ref and materializer scope. The reference payload is obtained
from the same materializer after runtime execution.

```text
ROUTER_WEIGHT_IDENTITY_MATCH: YES by shared PersistentTensorRef
ROUTER_WEIGHT_PAYLOAD_MATCH: YES by shared materialized payload path
NORM_WEIGHT_IDENTITY_MATCH: YES by shared run_layer norm ref
```

No retained evidence shows a failed range, zero-byte request, payload hash
mismatch, lease failure, or residency violation. D2.3 specifically qualified
the readiness and lease path.

## 13. Reference-Control Audit

The reference at the failing gate is not a historical fixed token or a
hard-coded router vector. It is a scalar calculation over the current
activation and current materialized router payload. It is therefore general
with respect to arbitrary prompt activations at this boundary.

The older Gate 2B harness does contain a fixture-specific one-hot activation,
but that is its qualification input, not a hidden assumption inside the
generic router comparator.

```text
REFERENCE_CONTROL_GENERAL_FOR_ARBITRARY_PROMPTS: YES at router logit boundary
FIXTURE_ASSUMPTION_LEAK: Gate 2B input fixture only; not shown at failing gate
```

## 14. Gate 2B vs Android Delta

| Boundary | Gate 2B | Android arbitrary prompt |
|---|---|---|
| Model tensors | Real IQ2_XXS router/norm tensors | Same semantic model lineage and refs |
| Router input | Deterministic one-hot 2048x1 | Hidden state after prompt/state execution |
| Position | Fixture-independent selected slice | Prompt position and autoregressive state |
| KV/state | No full prompt history | Separate actual/reference KV histories |
| MatMul check | Exact, max error 0 | Failure value not retained |
| TopK | Reached and passed | Not reached at failure |
| Transport/materialization | Real ready payloads | Real ranges and bounded residency observed |

The only proven execution-context delta is the router activation/history. The
first numerical divergence inside that context is not captured.

## 15. First Divergent Boundary

The earliest boundary that can be established from the retained evidence is:

```text
FIRST_DIVERGENT_BOUNDARY: router runtime logits vs scalar router logits
```

This is a gate location, not proof that the router operation is the root
cause. The required earlier fingerprints for prompt IDs, hidden state,
RMSNorm, tensor payload, position, and state were not recorded by c006130.

## 16. Primary Blocker

```text
PRIMARY_BLOCKER: UNKNOWN
```

The audit cannot responsibly choose between `ROUTER_PARITY_TOLERANCE_TOO_STRICT`
and `MATMUL_EXECUTION_MISMATCH` without the missing bounded vector diagnostics.
The gate is comparing the correct two mathematical quantities, and the call
graph proves the same current activation and materialized router payload are
used. That makes a prompt tokenization mismatch, reference future-token bug,
TopK bug, and payload-readiness bug unsupported as primary explanations.

## 17. Secondary Findings

- The Android-only `1e-4` tolerance in `router_driven_moe_poc11.cpp` is a
  qualification-affecting workaround, not an explanation; the retained report
  says it did not produce a successful run.
- `parity()` does not record the failing index, values, NaN/Inf counts, or TopK
  margin. This is the direct observability gap blocking classification.
- The Android JNI error path discards the comparator diagnostics and returns
  only `GEN_FAIL router parity failed`.
- The current app's retained run proves substantial real computation and
  external materialization, but not the first divergent hidden-state boundary.
- No crash was reproduced in the audit; the observed ADB task closures were
  input-script back actions.

## 18. Fix Ownership

```text
FIX_OWNER: QUALIFICATION_HARNESS / PARITY_CHECK instrumentation first
```

No runtime, tokenizer, state manager, tensor resolver, backend, or model
semantic fix is justified yet.

## 19. Minimal Recommended Fix

Add bounded diagnostic telemetry at the existing gate, without changing its
condition:

1. Record position, layer, phase, token ID, and 64-element vector counts.
2. Record stable hashes and first eight values of the router input and RMSNorm
   output for both named paths.
3. Record max absolute/relative error, mean error, max-error index and values,
   NaN/Inf counts, and both TopK results when logits are available.
4. Record semantic TensorId, source offset/length, representation, and a
   bounded payload checksum for norm/router weights.

Then rerun one short deterministic prompt and classify the first boundary. Do
not relax the gate or change runtime semantics before that result.

## 20. Regression Plan

After the evidence-derived fix, require:

- Gate 2B real DeepSeek router-prefix parity.
- Native CTest 19/19.
- Rust workspace tests and neutrality guard.
- Android arbitrary prompt: first token, four-token generation, router parity,
  logit parity, token parity, real text, and teardown.

Phase B, D3, and Gate 2C remain unauthorized.

## 21. Required Status

```text
VBUF_ML_ANDROID_ROUTER_PARITY_BLOCKER_AUDIT_COMPLETE: NO, numerical root cause unresolved

BRANCH: vbuf-ml
HEAD: c006130

FAILURE_REPRODUCED: historical router-parity failure only; fresh JNI run stalled before gate
PROMPT: Explain the purpose of bounded generation
PROMPT_TOKEN_COUNT: NOT RECORDED

FAILURE_PHASE: NOT RECORDED
FAILURE_POSITION: NOT RECORDED
FAILURE_LAYER: NOT RECORDED
FAILURE_STAGE: router parity gate

PARITY_GATE_FILE: integrations/ggml/tools/multi_expert_moe_poc12.cpp
PARITY_GATE_FUNCTION: route_activation()
PARITY_GATE_REFERENCE: scalar reference_scores()
PARITY_GATE_RUNTIME_VALUE: GGML router_matmul logits
PARITY_GATE_TOLERANCE: 1e-4 Android direct build

REFERENCE_VECTOR_COUNT: NOT RECORDED
RUNTIME_VECTOR_COUNT: NOT RECORDED

MAX_ABSOLUTE_ERROR: NOT RECORDED
MAX_RELATIVE_ERROR: NOT RECORDED
MAX_ERROR_INDEX: NOT RECORDED

REFERENCE_TOPK: NOT REACHED
RUNTIME_TOPK: NOT REACHED
TOPK_IDENTICAL: NOT APPLICABLE

PROMPT_TOKENIZATION_MATCH: NOT RECORDED
POSITION_MATCH: STRUCTURALLY YES, RUN VALUE NOT RECORDED
STATE_INITIALIZATION_MATCH: YES by code path
STATE_RESET_CORRECT: YES for fresh generation

ROUTER_INPUT_IDENTICAL: YES by shared call argument, fingerprint not recorded
RMSNORM_INPUT_IDENTICAL: YES by shared call argument
RMSNORM_OUTPUT_PARITY: NOT RECORDED
ROUTER_WEIGHT_IDENTITY_MATCH: YES by shared PersistentTensorRef
ROUTER_WEIGHT_PAYLOAD_MATCH: YES by shared materialized payload path
ROUTER_LOGITS_PARITY: FAIL at retained gate, error not recorded
TOPK_PARITY: NOT REACHED

FIRST_DIVERGENT_BOUNDARY: router runtime logits vs scalar router logits

GATE2B_STANDALONE_CONTROL: PASS exact
GATE2B_VS_ANDROID_KEY_DIFFERENCE: deterministic one-hot input vs prompt-derived hidden state/history

REFERENCE_CONTROL_GENERAL_FOR_ARBITRARY_PROMPTS: YES at router boundary

PRIMARY_BLOCKER: UNRESOLVED; no numerical vectors captured
SECONDARY_FINDINGS: historical fresh JNI run had no transport progress; later controlled timing explained the apparent stall as repeated serial positions

FIX_OWNER: QUALIFICATION_HARNESS / PARITY_CHECK instrumentation
MINIMAL_FIX: diagnose the pre-gate stall, then capture bounded gate/input/state/tensor diagnostics

PARITY_GATE_SHOULD_BE_RELAXED: NO evidence
RUNTIME_SEMANTICS_SHOULD_CHANGE: NO evidence
MODEL_SEMANTICS_SHOULD_CHANGE: NO

PHASE_A_CURRENT_RESULT: ANDROID_DEMO_GENERATION_BLOCKED
PHASE_B_READY: NO
D3_READY_FOR_DEMO_PATH: NO

PRODUCTION_CODE_CHANGED: NO
TEMPORARY_INSTRUMENTATION_ADDED: YES, audit-only receiver and bounded parity log
TEMPORARY_INSTRUMENTATION_REMOVED: YES

CARGO_TEST_WORKSPACE: PASS previously; not rerun for report-only audit
NATIVE_GGML_BUILD: PASS Android APK build
CTEST: PASS 19/19 previously; not rerun for report-only audit
NEUTRALITY_GUARD: PASS previously

RESEARCH_REPORT: research/results/vbuf-android-demo-poc/router-parity-blocker-audit.md

COMMIT_PERFORMED: NO
PUSH_PERFORMED: NO
FINAL_WORKTREE_STATUS: report uncommitted; temporary instrumentation removed; no production changes
```
